// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated per-run evaluator compatibility proxy.
//!
//! The proxy accepts exactly one typed host-operation DTO over a local
//! Unix-domain HTTP endpoint. It never accepts an upstream URL, HTTP method,
//! forwarding headers, credentials, or raw SSE. Accepted work crosses a
//! bounded channel into [`super::workload::EvaluationWorkload`], where pipe and
//! proxy requests share route preparation, fair admission, the operation
//! ledger, transport retries, cancellation, usage accounting, and reporting.
//! The HTTP response contains only normalized [`HostOperationEvent`] values.
//!
//! This replaces the buffered, TCP-listening agentic callback behavior ported
//! from `src/aiperf/accuracy/model_broker.py` and
//! `crates/aiperf/src/agentic_gateway.rs`: neither source had case-scoped
//! grants, process-subtree authentication, a shared evaluator ledger, or true
//! incremental normalized SSE.

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::task::{Context as TaskContext, Poll};

use aiperf_accuracy::{
    CanonicalJson, EvaluationCaseId, HostOperationEvent, HostOperationRequest, HostResponseMode,
    ScopedProxyBinding, ScopedProxyGrant, validate_no_secret_control_value,
};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::header::{ACCEPT, AUTHORIZATION, CONNECTION, CONTENT_LENGTH, CONTENT_TYPE, HOST};
use axum::http::{HeaderMap, HeaderName, StatusCode};
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response, Sse};
use axum::routing::post;
use axum::{Router, serve};
use futures::stream;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};
use tokio::task::JoinHandle;

use super::host::CompatibilityProxyIngress;

const PROXY_PATH: &str = "/v1/operations";
const OPENAI_CHAT_PATH: &str = "/v1/chat/completions";
const GRANT_HEADER: HeaderName = HeaderName::from_static("x-aiperf-proxy-grant");
const CASE_HEADER: HeaderName = HeaderName::from_static("x-aiperf-case-id");
const MAX_PROXY_OPERATIONS: u64 = 10_000_000;
const MAX_PROXY_CONNECTIONS: u64 = 256;
const MAX_PROXY_CONCURRENCY: u64 = MAX_PROXY_CONNECTIONS - 1;
const MAX_PROXY_BODY_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PROXY_AGGREGATE_BYTES: u64 = 16 * 1024 * 1024 * 1024;
const MAX_PROXY_STREAM_EVENTS: u64 = 10_000_000;
const MAX_PROXY_LIFETIME_MS: u64 = 24 * 60 * 60 * 1_000;

/// Registered local compatibility dialect over the shared typed host boundary.
pub trait CompatibilityProxyDialect: Send + Sync {
    /// Fixed local path owned by this adapter, never an upstream path.
    fn local_path(&self) -> &'static str;

    /// Lower a strict compatibility request into the canonical host operation.
    fn lower(
        &self,
        runtime: &ProxyGrantRuntime,
        headers: &HeaderMap,
        body: &[u8],
    ) -> std::result::Result<HostOperationRequest, ProxyRejection>;

    /// Project a normalized terminal event into dialect JSON.
    fn project_terminal(
        &self,
        event: HostOperationEvent,
    ) -> std::result::Result<CanonicalJson, ProxyRejection>;

    /// Project one normalized event into dialect-local SSE JSON.
    fn project_stream(
        &self,
        event: HostOperationEvent,
    ) -> std::result::Result<CanonicalJson, ProxyRejection>;
}

/// OpenAI-compatible chat-completions adapter with no caller model/route authority.
#[derive(Debug)]
pub struct OpenAiChatCompatibilityDialect {
    service_id: aiperf_accuracy::LogicalServiceId,
    purpose: aiperf_accuracy::OperationPurpose,
    semantic_operation_id: aiperf_accuracy::SemanticOperationId,
}

impl OpenAiChatCompatibilityDialect {
    fn from_grant(grant: &ScopedProxyGrant) -> Option<Self> {
        let [service_id] = grant.service_ids.as_slice() else {
            return None;
        };
        let [purpose] = grant.purposes.as_slice() else {
            return None;
        };
        let [semantic_operation_id] = grant.semantic_operation_ids.as_slice() else {
            return None;
        };
        (semantic_operation_id.as_str() == "model.generate").then(|| Self {
            service_id: service_id.clone(),
            purpose: purpose.clone(),
            semantic_operation_id: semantic_operation_id.clone(),
        })
    }
}

/// Safe rejection returned to the local compatibility client.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProxyRejection {
    /// Bearer capability or grant identity did not match.
    Unauthorized,
    /// The connecting process was outside the attested evaluator subtree.
    ProcessScope,
    /// An HTTP header or typed request field exceeded the proxy contract.
    InvalidRequest,
    /// Logical service, semantic operation, purpose, or case was not granted.
    GrantScope,
    /// A monotonic grant lifetime or bounded resource credit was exhausted.
    GrantExhausted,
    /// Operation, logical-call, or idempotency identity was already consumed.
    Duplicate,
    /// The shared evaluation host rejected the operation before an effect.
    Admission,
    /// The evaluation host is no longer accepting proxy work.
    Unavailable,
}

impl ProxyRejection {
    fn status(self) -> StatusCode {
        match self {
            Self::Unauthorized | Self::ProcessScope => StatusCode::UNAUTHORIZED,
            Self::InvalidRequest => StatusCode::BAD_REQUEST,
            Self::GrantScope => StatusCode::FORBIDDEN,
            Self::GrantExhausted => StatusCode::TOO_MANY_REQUESTS,
            Self::Duplicate => StatusCode::CONFLICT,
            Self::Admission => StatusCode::UNPROCESSABLE_ENTITY,
            Self::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    fn kind(self) -> &'static str {
        match self {
            Self::Unauthorized => "unauthorized",
            Self::ProcessScope => "process_scope",
            Self::InvalidRequest => "invalid_request",
            Self::GrantScope => "grant_scope",
            Self::GrantExhausted => "grant_exhausted",
            Self::Duplicate => "duplicate_identity",
            Self::Admission => "admission_rejected",
            Self::Unavailable => "proxy_unavailable",
        }
    }

    fn response(self) -> Response {
        let value = CanonicalJson::new(json!({
            "error": {
                "kind": self.kind(),
                "message": "Rust evaluator compatibility proxy rejected the request",
            }
        }))
        .expect("static proxy rejection is canonical JSON");
        (
            self.status(),
            [(CONTENT_TYPE, "application/json")],
            value.to_bytes(),
        )
            .into_response()
    }
}

#[derive(Debug)]
struct BoundedUnixListener {
    listener: UnixListener,
    permits: Arc<Semaphore>,
}

impl BoundedUnixListener {
    fn new(listener: UnixListener, permits: Arc<Semaphore>) -> Self {
        Self { listener, permits }
    }
}

#[derive(Debug)]
struct BoundedUnixStream {
    stream: UnixStream,
    _permit: OwnedSemaphorePermit,
}

impl AsyncRead for BoundedUnixStream {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(context, buffer)
    }
}

impl AsyncWrite for BoundedUnixStream {
    fn poll_write(
        mut self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(context, buffer)
    }

    fn poll_flush(mut self: Pin<&mut Self>, context: &mut TaskContext<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(context)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(context)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        context: &mut TaskContext<'_>,
        buffers: &[io::IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write_vectored(context, buffers)
    }
}

impl axum::serve::Listener for BoundedUnixListener {
    type Io = BoundedUnixStream;
    type Addr = tokio::net::unix::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let permit = self
                .permits
                .clone()
                .acquire_owned()
                .await
                .expect("proxy connection semaphore is never closed");
            match self.listener.accept().await {
                Ok((stream, address)) => {
                    return (
                        BoundedUnixStream {
                            stream,
                            _permit: permit,
                        },
                        address,
                    );
                }
                Err(_) => {
                    drop(permit);
                    tokio::task::yield_now().await;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.listener.local_addr()
    }
}

/// Kernel-authenticated identity of one local Unix-socket peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProxyPeerIdentity {
    /// Kernel peer process ID when supported by the platform.
    pub pid: Option<u32>,
    /// Kernel peer effective user ID.
    pub uid: u32,
    /// Kernel peer effective group ID.
    pub gid: u32,
}

impl axum::extract::connect_info::Connected<serve::IncomingStream<'_, BoundedUnixListener>>
    for ProxyPeerIdentity
{
    fn connect_info(stream: serve::IncomingStream<'_, BoundedUnixListener>) -> Self {
        match stream.io().stream.peer_cred() {
            Ok(credentials) => Self {
                pid: credentials.pid().and_then(|pid| u32::try_from(pid).ok()),
                uid: credentials.uid(),
                gid: credentials.gid(),
            },
            Err(_) => Self {
                pid: None,
                uid: u32::MAX,
                gid: u32::MAX,
            },
        }
    }
}

/// Replaceable platform verifier for an evaluator process-subtree grant.
pub trait ProxyProcessScopeAuthorizer: Send + Sync {
    /// Verify kernel peer identity against the Rust-minted process scope.
    fn authorize(
        &self,
        process_scope_sha256: &str,
        peer: ProxyPeerIdentity,
    ) -> std::result::Result<(), ProxyRejection>;
}

#[derive(Debug, Clone, Copy)]
struct LinuxProcessRoot {
    pid: u32,
    uid: u32,
    start_time_ticks: u64,
}

/// Linux `/proc` + `SO_PEERCRED` evaluator-subtree authorizer.
///
/// The proxy socket may be bound before the worker is spawned. The launcher
/// calls [`bind_root`](Self::bind_root) exactly once after the attested child
/// PID exists; requests fail closed before that binding. PID reuse is rejected
/// by pinning the root process start-time field from `/proc/<pid>/stat`.
#[derive(Debug)]
pub struct LinuxProcessSubtreeAuthorizer {
    process_scope_sha256: String,
    root: RwLock<Option<LinuxProcessRoot>>,
}

impl LinuxProcessSubtreeAuthorizer {
    /// Build an unbound authorizer for one pre-minted process-scope digest.
    pub fn new(process_scope_sha256: impl Into<String>) -> Result<Self> {
        let process_scope_sha256 = process_scope_sha256.into();
        ensure!(
            aiperf_accuracy::is_sha256(&process_scope_sha256),
            "proxy process scope must be a lowercase SHA-256 digest"
        );
        Ok(Self {
            process_scope_sha256,
            root: RwLock::new(None),
        })
    }

    /// Bind the sole attested evaluator worker root process.
    pub fn bind_root(&self, pid: u32) -> Result<()> {
        let (parent, start_time_ticks) = linux_process_stat(pid)?;
        let _ = parent;
        let uid = linux_process_uid(pid)?;
        let mut root = self
            .root
            .write()
            .map_err(|_| anyhow!("proxy process-scope lock was poisoned"))?;
        ensure!(
            root.is_none(),
            "proxy evaluator process root was bound twice"
        );
        *root = Some(LinuxProcessRoot {
            pid,
            uid,
            start_time_ticks,
        });
        Ok(())
    }
}

impl ProxyProcessScopeAuthorizer for LinuxProcessSubtreeAuthorizer {
    fn authorize(
        &self,
        process_scope_sha256: &str,
        peer: ProxyPeerIdentity,
    ) -> std::result::Result<(), ProxyRejection> {
        if !constant_time_eq(
            self.process_scope_sha256.as_bytes(),
            process_scope_sha256.as_bytes(),
        ) {
            return Err(ProxyRejection::ProcessScope);
        }
        let root = self
            .root
            .read()
            .map_err(|_| ProxyRejection::ProcessScope)?
            .ok_or(ProxyRejection::ProcessScope)?;
        let peer_pid = peer.pid.ok_or(ProxyRejection::ProcessScope)?;
        if peer.uid != root.uid {
            return Err(ProxyRejection::ProcessScope);
        }
        let (_, current_start_time) =
            linux_process_stat(root.pid).map_err(|_| ProxyRejection::ProcessScope)?;
        if current_start_time != root.start_time_ticks {
            return Err(ProxyRejection::ProcessScope);
        }
        let mut current = peer_pid;
        let mut visited = BTreeSet::new();
        for _ in 0..256 {
            if current == root.pid {
                return Ok(());
            }
            if current <= 1 || !visited.insert(current) {
                break;
            }
            let (parent, _) =
                linux_process_stat(current).map_err(|_| ProxyRejection::ProcessScope)?;
            current = parent;
        }
        Err(ProxyRejection::ProcessScope)
    }
}

impl aiperf_accuracy::EvaluatorProcessRootBinder for LinuxProcessSubtreeAuthorizer {
    fn bind_attested_root(
        &self,
        root_pid: u32,
    ) -> std::result::Result<(), aiperf_accuracy::ProviderRegistryError> {
        self.bind_root(root_pid).map_err(|_| {
            aiperf_accuracy::ProviderRegistryError::InvalidLaunch(
                "failed to bind the attested evaluator process subtree".to_string(),
            )
        })
    }
}

fn linux_process_stat(pid: u32) -> Result<(u32, u64)> {
    let text = fs::read_to_string(format!("/proc/{pid}/stat"))
        .with_context(|| format!("reading evaluator process stat for PID {pid}"))?;
    let close = text
        .rfind(')')
        .ok_or_else(|| anyhow!("evaluator process stat omitted command terminator"))?;
    let fields = text[close + 1..].split_whitespace().collect::<Vec<_>>();
    ensure!(
        fields.len() > 19,
        "evaluator process stat omitted ancestry/start-time fields"
    );
    Ok((fields[1].parse()?, fields[19].parse()?))
}

fn linux_process_uid(pid: u32) -> Result<u32> {
    let text = fs::read_to_string(format!("/proc/{pid}/status"))
        .with_context(|| format!("reading evaluator process status for PID {pid}"))?;
    let line = text
        .lines()
        .find(|line| line.starts_with("Uid:"))
        .ok_or_else(|| anyhow!("evaluator process status omitted UID"))?;
    line.split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow!("evaluator process UID was empty"))?
        .parse()
        .context("parsing evaluator process UID")
}

#[derive(Debug, Default)]
struct ProxyGrantUsage {
    accepted_operations: u64,
    active_operations: u64,
    request_bytes: u64,
    pending_request_bytes: u64,
    response_bytes: u64,
    stream_events: u64,
    pending_reservations: BTreeMap<u64, u64>,
    operation_ids: BTreeSet<String>,
    logical_call_ids: BTreeSet<String>,
    idempotency_keys: BTreeSet<String>,
    active_disconnects: BTreeMap<String, Arc<AtomicBool>>,
}

#[derive(Debug)]
/// Shared grant/case/budget authority behind every registered proxy dialect.
pub struct ProxyGrantRuntime {
    grant: ScopedProxyGrant,
    started_ns: i64,
    revoked: AtomicBool,
    case_scope: RwLock<BTreeMap<EvaluationCaseId, aiperf_accuracy::EvaluationUnitId>>,
    retired_cases: RwLock<BTreeSet<EvaluationCaseId>>,
    next_proxy_ordinal: AtomicU64,
    next_pending_ordinal: AtomicU64,
    usage: Mutex<ProxyGrantUsage>,
}

impl CompatibilityProxyDialect for OpenAiChatCompatibilityDialect {
    fn local_path(&self) -> &'static str {
        OPENAI_CHAT_PATH
    }

    fn lower(
        &self,
        runtime: &ProxyGrantRuntime,
        headers: &HeaderMap,
        body: &[u8],
    ) -> std::result::Result<HostOperationRequest, ProxyRejection> {
        let case_id = headers
            .get(&CASE_HEADER)
            .and_then(|value| value.to_str().ok())
            .ok_or(ProxyRejection::GrantScope)
            .and_then(|value| {
                EvaluationCaseId::new(value.to_string()).map_err(|_| ProxyRejection::GrantScope)
            })?;
        let unit_id = runtime.case_unit(&case_id)?;
        let body = CanonicalJson::from_slice(body, Default::default())
            .map_err(|_| ProxyRejection::InvalidRequest)?;
        let object = body
            .value()
            .as_object()
            .ok_or(ProxyRejection::InvalidRequest)?;
        const ALLOWED: &[&str] = &[
            "messages",
            "max_tokens",
            "temperature",
            "top_p",
            "stop",
            "tools",
            "tool_choice",
            "response_format",
            "stream",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "parallel_tool_calls",
        ];
        if object
            .keys()
            .any(|field| !ALLOWED.contains(&field.as_str()))
        {
            return Err(ProxyRejection::InvalidRequest);
        }
        let messages = object
            .get("messages")
            .filter(|value| {
                value
                    .as_array()
                    .is_some_and(|messages| !messages.is_empty())
            })
            .cloned()
            .ok_or(ProxyRejection::InvalidRequest)?;
        let max_tokens = object
            .get("max_tokens")
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or(ProxyRejection::InvalidRequest)?;
        let stream = object
            .get("stream")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if object
            .get("stream")
            .is_some_and(|value| !value.is_boolean())
        {
            return Err(ProxyRejection::InvalidRequest);
        }
        let mut generation = serde_json::Map::from_iter([(
            "max_tokens".to_string(),
            serde_json::Value::from(max_tokens),
        )]);
        for field in ["temperature", "top_p", "stop"] {
            if let Some(value) = object.get(field) {
                generation.insert(field.to_string(), value.clone());
            }
        }
        let mut payload = serde_json::Map::from_iter([
            ("messages".to_string(), messages),
            (
                "generation".to_string(),
                serde_json::Value::Object(generation),
            ),
        ]);
        for field in ["tools", "tool_choice", "response_format"] {
            if let Some(value) = object.get(field) {
                payload.insert(field.to_string(), value.clone());
            }
        }
        let mut parameters = serde_json::Map::new();
        for field in [
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "parallel_tool_calls",
        ] {
            if let Some(value) = object.get(field) {
                parameters.insert(field.to_string(), value.clone());
            }
        }
        if !parameters.is_empty() {
            payload.insert(
                "parameters".to_string(),
                serde_json::Value::Object(parameters),
            );
        }

        let ordinal = runtime
            .next_proxy_ordinal
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| ProxyRejection::GrantExhausted)?;
        let identity = aiperf_accuracy::sha256_hex(
            format!(
                "aiperf-proxy-openai-chat-v1\0{}\0{}\0{ordinal}",
                runtime.grant.grant_id, case_id
            )
            .as_bytes(),
        );
        let operation_id =
            aiperf_accuracy::HostOperationId::new(format!("proxy-operation-{identity}"))
                .map_err(|_| ProxyRejection::InvalidRequest)?;
        let semantic_attempt_id =
            aiperf_accuracy::SemanticAttemptId::new(format!("proxy-semantic-{identity}"))
                .map_err(|_| ProxyRejection::InvalidRequest)?;
        let logical_call_id = aiperf_accuracy::LogicalCallId::new(format!("proxy-call-{identity}"))
            .map_err(|_| ProxyRejection::InvalidRequest)?;
        let payload = CanonicalJson::new(serde_json::Value::Object(payload))
            .map_err(|_| ProxyRejection::InvalidRequest)?;
        validate_no_secret_control_value(&payload).map_err(|_| ProxyRejection::InvalidRequest)?;
        Ok(HostOperationRequest {
            operation_id,
            context: aiperf_accuracy::HostCallContext {
                session_id: runtime.grant.session_id.clone(),
                unit_id,
                case_id,
                semantic_attempt_id,
                logical_call_id,
            },
            service_id: self.service_id.clone(),
            purpose: self.purpose.clone(),
            semantic_operation_id: self.semantic_operation_id.clone(),
            payload,
            restricted_payload: None,
            response_mode: if stream {
                HostResponseMode::Streaming
            } else {
                HostResponseMode::Terminal
            },
            deadline_ms: None,
            idempotency_key: format!("proxy-idempotency-{identity}"),
        })
    }

    fn project_terminal(
        &self,
        event: HostOperationEvent,
    ) -> std::result::Result<CanonicalJson, ProxyRejection> {
        let HostOperationEvent::Terminal { terminal } = event else {
            return Err(ProxyRejection::Unavailable);
        };
        match terminal.disposition {
            aiperf_accuracy::HostOperationDisposition::Completed => {
                terminal.result.ok_or(ProxyRejection::Unavailable)
            }
            _ => CanonicalJson::new(json!({
                "error": {
                    "message": "Rust host operation did not complete",
                    "type": "aiperf_evaluator_infrastructure_error"
                }
            }))
            .map_err(|_| ProxyRejection::Unavailable),
        }
    }

    fn project_stream(
        &self,
        event: HostOperationEvent,
    ) -> std::result::Result<CanonicalJson, ProxyRejection> {
        match event {
            HostOperationEvent::StreamDelta { delta, .. } => {
                let object = delta
                    .value()
                    .as_object()
                    .ok_or(ProxyRejection::Unavailable)?;
                let index = object
                    .get("choice_index")
                    .cloned()
                    .ok_or(ProxyRejection::Unavailable)?;
                let delta = object
                    .get("delta")
                    .cloned()
                    .ok_or(ProxyRejection::Unavailable)?;
                CanonicalJson::new(json!({
                    "choices": [{"delta": delta, "finish_reason": null, "index": index}]
                }))
                .map_err(|_| ProxyRejection::Unavailable)
            }
            HostOperationEvent::Usage { usage, .. } => CanonicalJson::new(json!({
                "choices": [],
                "usage": usage,
            }))
            .map_err(|_| ProxyRejection::Unavailable),
            HostOperationEvent::Terminal { terminal } => match terminal.disposition {
                aiperf_accuracy::HostOperationDisposition::Completed => {
                    let result = terminal.result.ok_or(ProxyRejection::Unavailable)?;
                    let object = result
                        .value()
                        .as_object()
                        .ok_or(ProxyRejection::Unavailable)?;
                    let choices = object
                        .get("choices")
                        .and_then(serde_json::Value::as_array)
                        .ok_or(ProxyRejection::Unavailable)?
                        .iter()
                        .enumerate()
                        .map(|(index, choice)| {
                            let choice = choice
                                .as_object()
                                .ok_or(ProxyRejection::Unavailable)?;
                            Ok(json!({
                                "delta": {},
                                "finish_reason": choice.get("finish_reason").cloned().unwrap_or(serde_json::Value::Null),
                                "index": index,
                            }))
                        })
                        .collect::<std::result::Result<Vec<_>, ProxyRejection>>()?;
                    CanonicalJson::new(json!({
                        "choices": choices,
                        "usage": object.get("usage").cloned().unwrap_or(serde_json::Value::Null),
                    }))
                    .map_err(|_| ProxyRejection::Unavailable)
                }
                _ => CanonicalJson::new(json!({
                    "error": {
                        "message": "Rust host operation did not complete",
                        "type": "aiperf_evaluator_infrastructure_error"
                    }
                }))
                .map_err(|_| ProxyRejection::Unavailable),
            },
            HostOperationEvent::CancellationAcknowledged { .. } => Err(ProxyRejection::Unavailable),
        }
    }
}

impl ProxyGrantRuntime {
    fn new(grant: ScopedProxyGrant, started_ns: i64) -> Self {
        Self {
            grant,
            started_ns,
            revoked: AtomicBool::new(false),
            case_scope: RwLock::new(BTreeMap::new()),
            retired_cases: RwLock::new(BTreeSet::new()),
            next_proxy_ordinal: AtomicU64::new(0),
            next_pending_ordinal: AtomicU64::new(0),
            usage: Mutex::new(ProxyGrantUsage::default()),
        }
    }

    fn reserve_pending(
        self: &Arc<Self>,
        request_bytes: u64,
    ) -> std::result::Result<ProxyPendingReservation, ProxyRejection> {
        if self.revoked.load(Ordering::Acquire) {
            return Err(ProxyRejection::Unavailable);
        }
        if request_bytes == 0 || request_bytes > MAX_PROXY_BODY_BYTES {
            return Err(ProxyRejection::InvalidRequest);
        }
        let mut usage = self.usage.lock().map_err(|_| ProxyRejection::Unavailable)?;
        if self.revoked.load(Ordering::Acquire) {
            return Err(ProxyRejection::Unavailable);
        }
        let pending_operations = u64::try_from(usage.pending_reservations.len())
            .map_err(|_| ProxyRejection::GrantExhausted)?;
        let reserved_operations = usage
            .accepted_operations
            .checked_add(pending_operations)
            .ok_or(ProxyRejection::GrantExhausted)?;
        let concurrent_operations = usage
            .active_operations
            .checked_add(pending_operations)
            .ok_or(ProxyRejection::GrantExhausted)?;
        let reserved_request_bytes = usage
            .request_bytes
            .checked_add(usage.pending_request_bytes)
            .and_then(|bytes| bytes.checked_add(request_bytes))
            .ok_or(ProxyRejection::GrantExhausted)?;
        if reserved_operations >= self.grant.max_operations
            || concurrent_operations >= self.grant.max_concurrent_operations
            || reserved_request_bytes > self.grant.max_request_bytes
        {
            return Err(ProxyRejection::GrantExhausted);
        }
        let reservation_id = self
            .next_pending_ordinal
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                value.checked_add(1)
            })
            .map_err(|_| ProxyRejection::GrantExhausted)?;
        usage.pending_request_bytes = usage
            .pending_request_bytes
            .checked_add(request_bytes)
            .ok_or(ProxyRejection::GrantExhausted)?;
        if usage
            .pending_reservations
            .insert(reservation_id, request_bytes)
            .is_some()
        {
            usage.pending_request_bytes = usage.pending_request_bytes.saturating_sub(request_bytes);
            return Err(ProxyRejection::Unavailable);
        }
        Ok(ProxyPendingReservation {
            reservation_id,
            request_bytes,
            runtime: Arc::clone(self),
            phase: AtomicU8::new(PENDING_RESERVED),
        })
    }

    fn accept_pending(
        self: &Arc<Self>,
        pending: &ProxyPendingReservation,
        request: &HostOperationRequest,
        now_ns: i64,
        disconnect: Arc<AtomicBool>,
        events: mpsc::Sender<HostOperationEvent>,
    ) -> std::result::Result<ProxyOperationResponder, ProxyRejection> {
        if self.revoked.load(Ordering::Acquire) {
            return Err(ProxyRejection::Unavailable);
        }
        self.validate_scope(request)?;
        let lifetime_ns = i64::try_from(self.grant.expires_after_ms)
            .ok()
            .and_then(|milliseconds| milliseconds.checked_mul(1_000_000))
            .ok_or(ProxyRejection::GrantExhausted)?;
        if now_ns.saturating_sub(self.started_ns) > lifetime_ns {
            return Err(ProxyRejection::GrantExhausted);
        }
        if !Arc::ptr_eq(self, &pending.runtime)
            || pending.phase.load(Ordering::Acquire) != PENDING_RESERVED
        {
            return Err(ProxyRejection::Unavailable);
        }
        let mut usage = self.usage.lock().map_err(|_| ProxyRejection::Unavailable)?;
        if self.revoked.load(Ordering::Acquire) {
            return Err(ProxyRejection::Unavailable);
        }
        let Some(reserved_bytes) = usage
            .pending_reservations
            .get(&pending.reservation_id)
            .copied()
        else {
            return Err(ProxyRejection::Unavailable);
        };
        if reserved_bytes != pending.request_bytes {
            return Err(ProxyRejection::Unavailable);
        }
        let operation_id = request.operation_id.to_string();
        let logical_call_id = request.context.logical_call_id.to_string();
        if usage.operation_ids.contains(&operation_id)
            || usage.logical_call_ids.contains(&logical_call_id)
            || usage.idempotency_keys.contains(&request.idempotency_key)
        {
            return Err(ProxyRejection::Duplicate);
        }
        let accepted_operations = usage
            .accepted_operations
            .checked_add(1)
            .ok_or(ProxyRejection::GrantExhausted)?;
        let active_operations = usage
            .active_operations
            .checked_add(1)
            .ok_or(ProxyRejection::GrantExhausted)?;
        let request_bytes = usage
            .request_bytes
            .checked_add(pending.request_bytes)
            .ok_or(ProxyRejection::GrantExhausted)?;
        usage.pending_reservations.remove(&pending.reservation_id);
        usage.pending_request_bytes = usage
            .pending_request_bytes
            .saturating_sub(pending.request_bytes);
        usage.accepted_operations = accepted_operations;
        usage.active_operations = active_operations;
        usage.request_bytes = request_bytes;
        usage.operation_ids.insert(operation_id.clone());
        usage.logical_call_ids.insert(logical_call_id.clone());
        usage
            .idempotency_keys
            .insert(request.idempotency_key.clone());
        usage
            .active_disconnects
            .insert(operation_id.clone(), disconnect.clone());
        pending.phase.store(PENDING_TRANSFERRED, Ordering::Release);
        drop(usage);
        Ok(ProxyOperationResponder {
            operation_id,
            logical_call_id,
            idempotency_key: request.idempotency_key.clone(),
            request_bytes: pending.request_bytes,
            events,
            disconnect,
            runtime: Arc::clone(self),
            phase: Arc::new(AtomicU8::new(RESPONDER_RESERVED)),
        })
    }

    fn release_pending(&self, pending: &ProxyPendingReservation) {
        if pending
            .phase
            .compare_exchange(
                PENDING_RESERVED,
                PENDING_RELEASED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        if let Ok(mut usage) = self.usage.lock()
            && let Some(bytes) = usage.pending_reservations.remove(&pending.reservation_id)
        {
            usage.pending_request_bytes = usage.pending_request_bytes.saturating_sub(bytes);
        }
    }

    fn validate_scope(
        &self,
        request: &HostOperationRequest,
    ) -> std::result::Result<(), ProxyRejection> {
        if request.context.session_id != self.grant.session_id
            || !self
                .grant
                .service_ids
                .iter()
                .any(|allowed| allowed == &request.service_id)
            || !self
                .grant
                .semantic_operation_ids
                .iter()
                .any(|allowed| allowed == &request.semantic_operation_id)
            || !self
                .grant
                .purposes
                .iter()
                .any(|allowed| allowed == &request.purpose)
        {
            return Err(ProxyRejection::GrantScope);
        }
        let case_scope = self
            .case_scope
            .read()
            .map_err(|_| ProxyRejection::Unavailable)?;
        if !case_scope.contains_key(&request.context.case_id) {
            return Err(ProxyRejection::GrantScope);
        }
        Ok(())
    }

    fn activate_unit_cases(
        &self,
        unit_id: aiperf_accuracy::EvaluationUnitId,
        cases: impl IntoIterator<Item = EvaluationCaseId>,
    ) -> std::result::Result<(), ProxyRejection> {
        if self.revoked.load(Ordering::Acquire) {
            return Err(ProxyRejection::Unavailable);
        }
        let cases = cases.into_iter().collect::<BTreeSet<_>>();
        if cases.is_empty() {
            return Err(ProxyRejection::GrantScope);
        }
        let mut scope = self
            .case_scope
            .write()
            .map_err(|_| ProxyRejection::Unavailable)?;
        let retired = self
            .retired_cases
            .read()
            .map_err(|_| ProxyRejection::Unavailable)?;
        if cases.iter().any(|case_id| retired.contains(case_id)) {
            return Err(ProxyRejection::GrantScope);
        }
        if cases.iter().any(|case_id| {
            scope
                .get(case_id)
                .is_some_and(|existing| existing != &unit_id)
        }) {
            return Err(ProxyRejection::GrantScope);
        }
        for case_id in cases {
            scope.insert(case_id, unit_id.clone());
        }
        Ok(())
    }

    fn deactivate_case(
        &self,
        case_id: &EvaluationCaseId,
    ) -> std::result::Result<(), ProxyRejection> {
        let removed = self
            .case_scope
            .write()
            .map_err(|_| ProxyRejection::Unavailable)?
            .remove(case_id);
        if removed.is_none() {
            return Err(ProxyRejection::GrantScope);
        }
        self.retired_cases
            .write()
            .map_err(|_| ProxyRejection::Unavailable)?
            .insert(case_id.clone());
        Ok(())
    }

    fn case_unit(
        &self,
        case_id: &EvaluationCaseId,
    ) -> std::result::Result<aiperf_accuracy::EvaluationUnitId, ProxyRejection> {
        self.case_scope
            .read()
            .map_err(|_| ProxyRejection::Unavailable)?
            .get(case_id)
            .cloned()
            .ok_or(ProxyRejection::GrantScope)
    }

    fn reject_reservation(&self, responder: &ProxyOperationResponder) {
        if responder
            .phase
            .compare_exchange(
                RESPONDER_RESERVED,
                RESPONDER_FINISHED,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_err()
        {
            return;
        }
        if let Ok(mut usage) = self.usage.lock() {
            usage.accepted_operations = usage.accepted_operations.saturating_sub(1);
            usage.active_operations = usage.active_operations.saturating_sub(1);
            usage.operation_ids.remove(&responder.operation_id);
            usage.logical_call_ids.remove(&responder.logical_call_id);
            usage.idempotency_keys.remove(&responder.idempotency_key);
            usage.request_bytes = usage.request_bytes.saturating_sub(responder.request_bytes);
            usage.active_disconnects.remove(&responder.operation_id);
        }
    }

    fn finish(&self, responder: &ProxyOperationResponder) {
        if responder.phase.swap(RESPONDER_FINISHED, Ordering::AcqRel) == RESPONDER_FINISHED {
            return;
        }
        if let Ok(mut usage) = self.usage.lock() {
            usage.active_operations = usage.active_operations.saturating_sub(1);
            usage.active_disconnects.remove(&responder.operation_id);
        }
    }

    fn record_event(&self, event: &HostOperationEvent) -> std::result::Result<(), ProxyRejection> {
        let encoded = CanonicalJson::new(
            serde_json::to_value(event).map_err(|_| ProxyRejection::Unavailable)?,
        )
        .map_err(|_| ProxyRejection::Unavailable)?
        .to_bytes();
        let event_bytes =
            u64::try_from(encoded.len()).map_err(|_| ProxyRejection::GrantExhausted)?;
        let is_stream = matches!(event, HostOperationEvent::StreamDelta { .. });
        let mut usage = self.usage.lock().map_err(|_| ProxyRejection::Unavailable)?;
        let response_bytes = usage
            .response_bytes
            .checked_add(event_bytes)
            .ok_or(ProxyRejection::GrantExhausted)?;
        let stream_events = usage
            .stream_events
            .checked_add(u64::from(is_stream))
            .ok_or(ProxyRejection::GrantExhausted)?;
        if response_bytes > self.grant.max_response_bytes
            || stream_events > self.grant.max_stream_events
        {
            return Err(ProxyRejection::GrantExhausted);
        }
        usage.response_bytes = response_bytes;
        usage.stream_events = stream_events;
        Ok(())
    }

    fn disconnected_operation_ids(&self) -> Vec<String> {
        self.usage
            .lock()
            .map(|usage| {
                usage
                    .active_disconnects
                    .iter()
                    .filter(|(_, disconnected)| disconnected.load(Ordering::Acquire))
                    .map(|(operation_id, _)| operation_id.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn revoke(&self) {
        self.revoked.store(true, Ordering::Release);
        if let Ok(mut usage) = self.usage.lock() {
            for disconnected in usage.active_disconnects.values() {
                disconnected.store(true, Ordering::Release);
            }
            usage.active_operations = 0;
            usage.active_disconnects.clear();
            usage.pending_reservations.clear();
            usage.pending_request_bytes = 0;
        }
    }
}

const PENDING_RESERVED: u8 = 0;
const PENDING_TRANSFERRED: u8 = 1;
const PENDING_RELEASED: u8 = 2;

/// RAII grant reservation acquired before any request-body allocation.
pub struct ProxyPendingReservation {
    reservation_id: u64,
    request_bytes: u64,
    runtime: Arc<ProxyGrantRuntime>,
    phase: AtomicU8,
}

impl std::fmt::Debug for ProxyPendingReservation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProxyPendingReservation")
            .field("reservation_id", &self.reservation_id)
            .field("request_bytes", &self.request_bytes)
            .finish_non_exhaustive()
    }
}

impl Drop for ProxyPendingReservation {
    fn drop(&mut self) {
        self.runtime.release_pending(self);
    }
}

const RESPONDER_RESERVED: u8 = 0;
const RESPONDER_ACTIVE: u8 = 1;
const RESPONDER_FINISHED: u8 = 2;

/// Normalized response handle retained by a queued or active proxy operation.
#[derive(Clone)]
pub struct ProxyOperationResponder {
    operation_id: String,
    logical_call_id: String,
    idempotency_key: String,
    request_bytes: u64,
    events: mpsc::Sender<HostOperationEvent>,
    disconnect: Arc<AtomicBool>,
    runtime: Arc<ProxyGrantRuntime>,
    phase: Arc<AtomicU8>,
}

impl std::fmt::Debug for ProxyOperationResponder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProxyOperationResponder")
            .field("operation_id", &self.operation_id)
            .field("disconnected", &self.is_disconnected())
            .finish_non_exhaustive()
    }
}

impl ProxyOperationResponder {
    /// Commit a reserved grant operation after shared host admission succeeds.
    pub fn activate(&self) -> std::result::Result<(), ProxyRejection> {
        self.phase
            .compare_exchange(
                RESPONDER_RESERVED,
                RESPONDER_ACTIVE,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(|_| ProxyRejection::Unavailable)
    }

    /// Roll back a grant operation rejected by shared host admission.
    pub fn reject(&self) {
        self.runtime.reject_reservation(self);
    }

    /// Whether the local HTTP/SSE caller disconnected before terminal.
    pub fn is_disconnected(&self) -> bool {
        self.disconnect.load(Ordering::Acquire)
    }

    /// Publish one normalized event without allowing an unbounded slow-client queue.
    pub fn publish(&self, event: HostOperationEvent) -> std::result::Result<(), ProxyRejection> {
        if self.phase.load(Ordering::Acquire) != RESPONDER_ACTIVE || self.is_disconnected() {
            return Err(ProxyRejection::Unavailable);
        }
        self.runtime.record_event(&event)?;
        let terminal = event.is_terminal();
        self.events
            .try_send(event)
            .map_err(|_| ProxyRejection::GrantExhausted)?;
        if terminal {
            self.runtime.finish(self);
        }
        Ok(())
    }

    /// Release active grant credit when the client cannot receive a terminal.
    pub fn complete_without_delivery(&self) {
        self.runtime.finish(self);
    }
}

/// One authenticated proxy request waiting for shared workload admission.
pub struct ProxyOperationSubmission {
    request: HostOperationRequest,
    reservation: ProxyPendingReservation,
    events: mpsc::Sender<HostOperationEvent>,
    disconnect: Arc<AtomicBool>,
    admission: oneshot::Sender<std::result::Result<(), ProxyRejection>>,
}

impl std::fmt::Debug for ProxyOperationSubmission {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProxyOperationSubmission")
            .field("operation_id", &self.request.operation_id)
            .field("request_bytes", &self.reservation.request_bytes)
            .finish_non_exhaustive()
    }
}

impl ProxyOperationSubmission {
    /// Borrow the strict typed operation for shared admission.
    pub fn request(&self) -> &HostOperationRequest {
        &self.request
    }

    /// Whether the HTTP caller still waits for shared workload admission.
    pub fn is_connected(&self) -> bool {
        !self.disconnect.load(Ordering::Acquire) && !self.admission.is_closed()
    }

    /// Consume the submission after returning an admission result to HTTP.
    pub fn resolve(self, result: std::result::Result<(), ProxyRejection>) -> bool {
        let delivered = self.admission.send(result).is_ok();
        if !delivered {
            self.disconnect.store(true, Ordering::Release);
        }
        delivered
    }
}

/// Bounded proxy ingress consumed only by the local evaluation workload.
pub struct ProxyOperationReceiver {
    receiver: mpsc::Receiver<ProxyOperationSubmission>,
    runtime: Arc<ProxyGrantRuntime>,
}

impl std::fmt::Debug for ProxyOperationReceiver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProxyOperationReceiver")
            .field("grant_id", &self.runtime.grant.grant_id)
            .finish_non_exhaustive()
    }
}

impl ProxyOperationReceiver {
    /// Receive one already-authenticated local request without blocking.
    pub fn try_recv(
        &mut self,
    ) -> std::result::Result<ProxyOperationSubmission, mpsc::error::TryRecvError> {
        self.receiver.try_recv()
    }

    /// Whether no authenticated HTTP request awaits workload admission.
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Reserve grant credits using the workload's injected Clock timestamp.
    pub fn authorize(
        &self,
        submission: &ProxyOperationSubmission,
        now_ns: i64,
    ) -> std::result::Result<ProxyOperationResponder, ProxyRejection> {
        self.runtime.accept_pending(
            &submission.reservation,
            &submission.request,
            now_ns,
            Arc::clone(&submission.disconnect),
            submission.events.clone(),
        )
    }

    /// Extend the effective grant with exact Rust-registered case occurrences.
    ///
    /// This authority is intentionally Rust-only and absent from the serialized
    /// worker binding. The prelaunch grant starts with no case authority; the
    /// workload calls this only after provider bind/occurrence registration.
    pub fn activate_case_scope(
        &self,
        unit_id: aiperf_accuracy::EvaluationUnitId,
        cases: impl IntoIterator<Item = EvaluationCaseId>,
    ) -> std::result::Result<(), ProxyRejection> {
        self.runtime.activate_unit_cases(unit_id, cases)
    }

    /// Permanently revoke one exact case after provider-semantic terminal.
    pub fn deactivate_case_scope(
        &self,
        case_id: &EvaluationCaseId,
    ) -> std::result::Result<(), ProxyRejection> {
        self.runtime.deactivate_case(case_id)
    }

    /// Proxy operations whose local client disconnected before terminal.
    pub fn disconnected_operation_ids(&self) -> Vec<String> {
        self.runtime.disconnected_operation_ids()
    }

    /// Revoke the grant and cancel every outstanding proxy operation.
    pub fn revoke(&self) {
        self.runtime.revoke();
    }
}

#[derive(Clone)]
struct ProxyServerState {
    grant: ScopedProxyGrant,
    submissions: mpsc::Sender<ProxyOperationSubmission>,
    runtime: Arc<ProxyGrantRuntime>,
    process_scope: Arc<dyn ProxyProcessScopeAuthorizer>,
    event_capacity: usize,
    openai_chat: Option<Arc<dyn CompatibilityProxyDialect>>,
    shutdown: watch::Receiver<bool>,
}

/// Running Unix-domain evaluator compatibility proxy.
pub struct EvaluatorCompatibilityProxy {
    local_locator: String,
    socket_path: PathBuf,
    runtime: Arc<ProxyGrantRuntime>,
    connection_permits: Arc<Semaphore>,
    connection_capacity: usize,
    shutdown: watch::Sender<bool>,
    task: tokio::sync::Mutex<Option<JoinHandle<std::io::Result<()>>>>,
}

impl std::fmt::Debug for EvaluatorCompatibilityProxy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EvaluatorCompatibilityProxy")
            .field("local_locator", &self.local_locator)
            .field("connection_capacity", &self.connection_capacity)
            .field(
                "available_connection_permits",
                &self.connection_permits.available_permits(),
            )
            .finish_non_exhaustive()
    }
}

impl Drop for EvaluatorCompatibilityProxy {
    fn drop(&mut self) {
        self.runtime.revoke();
        let _ = self.shutdown.send(true);
        if let Some(task) = self.task.get_mut().take() {
            task.abort();
        }
        let _ = fs::remove_file(&self.socket_path);
    }
}

impl EvaluatorCompatibilityProxy {
    /// Verify that a bounded receiver came from this exact server/grant runtime.
    pub fn owns_receiver(&self, receiver: &ProxyOperationReceiver) -> bool {
        Arc::ptr_eq(&self.runtime, &receiver.runtime)
    }

    #[cfg(test)]
    fn available_connection_permits(&self) -> usize {
        self.connection_permits.available_permits()
    }
}

#[async_trait(?Send)]
impl CompatibilityProxyIngress for EvaluatorCompatibilityProxy {
    fn local_locator(&self) -> &str {
        &self.local_locator
    }

    async fn shutdown(&self) -> Result<()> {
        self.runtime.revoke();
        let _ = self.shutdown.send(true);
        if let Some(task) = self.task.lock().await.take() {
            task.await
                .context("joining evaluator compatibility proxy")??;
        }
        match fs::remove_file(&self.socket_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error).context("removing evaluator proxy socket"),
        }
        Ok(())
    }
}

/// Bind a per-run Unix HTTP/SSE proxy and its bounded workload ingress.
pub async fn start_evaluator_compatibility_proxy(
    binding: ScopedProxyBinding,
    grant_started_ns: i64,
    process_scope: Arc<dyn ProxyProcessScopeAuthorizer>,
) -> Result<(EvaluatorCompatibilityProxy, ProxyOperationReceiver)> {
    validate_proxy_binding(&binding)?;
    let socket_path = binding.host_socket_path.clone();
    ensure!(
        !socket_path.exists(),
        "evaluator proxy socket path already exists"
    );
    let parent = socket_path
        .parent()
        .ok_or_else(|| anyhow!("evaluator proxy socket had no parent directory"))?;
    ensure!(parent.is_dir(), "evaluator proxy socket parent is absent");
    let listener = UnixListener::bind(&socket_path).context("binding evaluator proxy socket")?;
    fs::set_permissions(&socket_path, fs::Permissions::from_mode(0o600))
        .context("restricting evaluator proxy socket permissions")?;

    let capacity = usize::try_from(binding.grant.max_concurrent_operations)
        .context("proxy concurrency exceeds usize")?;
    let connection_capacity = usize::try_from(
        binding
            .grant
            .max_concurrent_operations
            .checked_add(1)
            .ok_or_else(|| anyhow!("proxy connection capacity overflow"))?,
    )
    .context("proxy connection capacity exceeds usize")?;
    let connection_permits = Arc::new(Semaphore::new(connection_capacity));
    let listener = BoundedUnixListener::new(listener, Arc::clone(&connection_permits));
    let event_capacity = usize::try_from(binding.grant.max_stream_events.clamp(2, 1024))
        .context("proxy stream-event capacity exceeds usize")?;
    let (sender, receiver) = mpsc::channel(capacity);
    let runtime = Arc::new(ProxyGrantRuntime::new(
        binding.grant.clone(),
        grant_started_ns,
    ));
    let openai_chat = OpenAiChatCompatibilityDialect::from_grant(&binding.grant)
        .map(|dialect| Arc::new(dialect) as Arc<dyn CompatibilityProxyDialect>);
    let (shutdown, shutdown_rx) = watch::channel(false);
    let state = ProxyServerState {
        grant: binding.grant,
        submissions: sender,
        runtime: Arc::clone(&runtime),
        process_scope,
        event_capacity,
        openai_chat,
        shutdown: shutdown_rx.clone(),
    };
    let router = Router::new()
        .route(PROXY_PATH, post(proxy_operation))
        .route(OPENAI_CHAT_PATH, post(openai_chat_operation))
        .with_state(state);
    let mut shutdown_rx = shutdown_rx;
    let task = tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<ProxyPeerIdentity>(),
        )
        .with_graceful_shutdown(async move {
            while !*shutdown_rx.borrow() {
                if shutdown_rx.changed().await.is_err() {
                    break;
                }
            }
        })
        .await
    });
    Ok((
        EvaluatorCompatibilityProxy {
            local_locator: binding.local_locator,
            socket_path,
            runtime: Arc::clone(&runtime),
            connection_permits,
            connection_capacity,
            shutdown,
            task: tokio::sync::Mutex::new(Some(task)),
        },
        ProxyOperationReceiver { receiver, runtime },
    ))
}

fn validate_proxy_binding(binding: &ScopedProxyBinding) -> Result<()> {
    binding
        .validate()
        .map_err(|error| anyhow!(error.to_string()))?;
    validate_socket_path(&binding.host_socket_path)?;
    let grant = &binding.grant;
    ensure!(
        grant.max_operations <= MAX_PROXY_OPERATIONS
            && grant.max_concurrent_operations <= MAX_PROXY_CONCURRENCY
            && grant.max_request_bytes <= MAX_PROXY_AGGREGATE_BYTES
            && grant.max_response_bytes <= MAX_PROXY_AGGREGATE_BYTES
            && grant.max_stream_events <= MAX_PROXY_STREAM_EVENTS
            && grant.expires_after_ms <= MAX_PROXY_LIFETIME_MS,
        "evaluator proxy grant exceeded hard runtime ceilings"
    );
    Ok(())
}

fn validate_socket_path(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute(),
        "evaluator proxy socket path is not absolute"
    );
    ensure!(
        !path.as_os_str().is_empty()
            && !path.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::ParentDir | std::path::Component::CurDir
                )
            }),
        "evaluator proxy socket path was not normalized"
    );
    Ok(())
}

async fn proxy_operation(
    State(state): State<ProxyServerState>,
    ConnectInfo(peer): ConnectInfo<ProxyPeerIdentity>,
    request: Request,
) -> Response {
    let (_headers, body, reservation) =
        match collect_authenticated_body(&state, peer, request, &[]).await {
            Ok(parts) => parts,
            Err(rejection) => return rejection.response(),
        };
    let request = match parse_typed_request(&state, &body) {
        Ok(request) => request,
        Err(rejection) => return rejection.response(),
    };
    submit_proxy_request(state, request, reservation, ProxyResponseProjection::Typed).await
}

async fn openai_chat_operation(
    State(state): State<ProxyServerState>,
    ConnectInfo(peer): ConnectInfo<ProxyPeerIdentity>,
    request: Request,
) -> Response {
    let (headers, body, reservation) =
        match collect_authenticated_body(&state, peer, request, &[&CASE_HEADER]).await {
            Ok(parts) => parts,
            Err(rejection) => return rejection.response(),
        };
    let Some(dialect) = state.openai_chat.clone() else {
        return ProxyRejection::GrantScope.response();
    };
    let request = match dialect.lower(&state.runtime, &headers, &body) {
        Ok(request) => request,
        Err(rejection) => return rejection.response(),
    };
    if request.validate().is_err() || state.runtime.validate_scope(&request).is_err() {
        return ProxyRejection::InvalidRequest.response();
    }
    submit_proxy_request(
        state,
        request,
        reservation,
        ProxyResponseProjection::Dialect(dialect),
    )
    .await
}

#[derive(Clone)]
enum ProxyResponseProjection {
    Typed,
    Dialect(Arc<dyn CompatibilityProxyDialect>),
}

async fn submit_proxy_request(
    state: ProxyServerState,
    request: HostOperationRequest,
    reservation: ProxyPendingReservation,
    projection: ProxyResponseProjection,
) -> Response {
    let response_mode = request.response_mode;
    let (events, event_receiver) = mpsc::channel(state.event_capacity);
    let disconnect = Arc::new(AtomicBool::new(false));
    let (admission, admitted) = oneshot::channel();
    let submission = ProxyOperationSubmission {
        request,
        reservation,
        events,
        disconnect: Arc::clone(&disconnect),
        admission,
    };
    let mut guard = ProxyDisconnectGuard {
        disconnected: disconnect,
        terminal: false,
    };
    if let Err(rejection) = enqueue_proxy_submission(&state.submissions, submission) {
        guard.complete();
        return rejection.response();
    }
    let mut shutdown = state.shutdown.clone();
    let admission = tokio::select! {
        result = admitted => result.map_err(|_| ProxyRejection::Unavailable),
        () = wait_for_shutdown(&mut shutdown) => Err(ProxyRejection::Unavailable),
    };
    match admission {
        Ok(Ok(())) => {}
        Ok(Err(rejection)) => {
            guard.complete();
            return rejection.response();
        }
        Err(rejection) => return rejection.response(),
    }
    match response_mode {
        HostResponseMode::Terminal => {
            terminal_response(event_receiver, guard, projection, state.shutdown).await
        }
        HostResponseMode::Streaming => {
            streaming_response(event_receiver, guard, projection, state.shutdown)
        }
    }
}

fn enqueue_proxy_submission(
    sender: &mpsc::Sender<ProxyOperationSubmission>,
    submission: ProxyOperationSubmission,
) -> std::result::Result<(), ProxyRejection> {
    sender.try_send(submission).map_err(|error| match error {
        mpsc::error::TrySendError::Full(_) => ProxyRejection::GrantExhausted,
        mpsc::error::TrySendError::Closed(_) => ProxyRejection::Unavailable,
    })
}

async fn collect_authenticated_body(
    state: &ProxyServerState,
    peer: ProxyPeerIdentity,
    request: Request,
    extra_headers: &[&HeaderName],
) -> std::result::Result<(HeaderMap, bytes::Bytes, ProxyPendingReservation), ProxyRejection> {
    let (parts, body) = request.into_parts();
    let request_bytes = authenticate_proxy_peer(state, peer, &parts.headers, extra_headers)?;
    let reservation = state.runtime.reserve_pending(request_bytes)?;
    let request_bytes =
        usize::try_from(request_bytes).map_err(|_| ProxyRejection::InvalidRequest)?;
    let mut shutdown = state.shutdown.clone();
    let body = tokio::select! {
        result = axum::body::to_bytes(body, request_bytes) => {
            result.map_err(|_| ProxyRejection::InvalidRequest)?
        }
        () = wait_for_shutdown(&mut shutdown) => return Err(ProxyRejection::Unavailable),
    };
    if body.len() != request_bytes {
        return Err(ProxyRejection::InvalidRequest);
    }
    Ok((parts.headers, body, reservation))
}

fn parse_typed_request(
    state: &ProxyServerState,
    body: &[u8],
) -> std::result::Result<HostOperationRequest, ProxyRejection> {
    let mut deserializer = serde_json::Deserializer::from_slice(body);
    let request = HostOperationRequest::deserialize(&mut deserializer)
        .map_err(|_| ProxyRejection::InvalidRequest)?;
    deserializer
        .end()
        .map_err(|_| ProxyRejection::InvalidRequest)?;
    request
        .validate()
        .map_err(|_| ProxyRejection::InvalidRequest)?;
    validate_no_secret_control_value(&request.payload)
        .map_err(|_| ProxyRejection::InvalidRequest)?;
    state.runtime.validate_scope(&request)?;
    Ok(request)
}

fn authenticate_proxy_peer(
    state: &ProxyServerState,
    peer: ProxyPeerIdentity,
    headers: &HeaderMap,
    extra_headers: &[&HeaderName],
) -> std::result::Result<u64, ProxyRejection> {
    let request_bytes = validate_proxy_headers(headers, &state.grant, extra_headers)?;
    state
        .process_scope
        .authorize(&state.grant.process_scope_sha256, peer)?;
    Ok(request_bytes)
}

fn validate_proxy_headers(
    headers: &HeaderMap,
    grant: &ScopedProxyGrant,
    extra_headers: &[&HeaderName],
) -> std::result::Result<u64, ProxyRejection> {
    for name in headers.keys() {
        if !matches!(
            name,
            &HOST | &CONTENT_TYPE | &CONTENT_LENGTH | &ACCEPT | &AUTHORIZATION | &CONNECTION
        ) && name != GRANT_HEADER
            && !extra_headers.contains(&name)
        {
            return Err(ProxyRejection::InvalidRequest);
        }
    }
    let content_type = headers
        .get(CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .ok_or(ProxyRejection::InvalidRequest)?;
    if !content_type
        .split(';')
        .next()
        .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
    {
        return Err(ProxyRejection::InvalidRequest);
    }
    let grant_id = headers
        .get(&GRANT_HEADER)
        .and_then(|value| value.to_str().ok())
        .ok_or(ProxyRejection::Unauthorized)?;
    let authorization = headers
        .get(AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .ok_or(ProxyRejection::Unauthorized)?;
    if !constant_time_eq(grant_id.as_bytes(), grant.grant_id.as_bytes())
        || !constant_time_eq(
            authorization.as_bytes(),
            grant.secret.expose_secret().as_bytes(),
        )
    {
        return Err(ProxyRejection::Unauthorized);
    }
    let mut lengths = headers.get_all(CONTENT_LENGTH).iter();
    let content_length = lengths
        .next()
        .and_then(|value| value.to_str().ok())
        .filter(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0 && *value <= MAX_PROXY_BODY_BYTES)
        .ok_or(ProxyRejection::InvalidRequest)?;
    if lengths.next().is_some() {
        return Err(ProxyRejection::InvalidRequest);
    }
    Ok(content_length)
}

async fn wait_for_shutdown(shutdown: &mut watch::Receiver<bool>) {
    while !*shutdown.borrow() {
        if shutdown.changed().await.is_err() {
            break;
        }
    }
}

fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    let maximum = left.len().max(right.len());
    for index in 0..maximum {
        difference |= usize::from(
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0),
        );
    }
    difference == 0
}

struct ProxyDisconnectGuard {
    disconnected: Arc<AtomicBool>,
    terminal: bool,
}

impl ProxyDisconnectGuard {
    fn complete(&mut self) {
        self.terminal = true;
    }
}

impl Drop for ProxyDisconnectGuard {
    fn drop(&mut self) {
        if !self.terminal {
            self.disconnected.store(true, Ordering::Release);
        }
    }
}

async fn terminal_response(
    mut receiver: mpsc::Receiver<HostOperationEvent>,
    mut guard: ProxyDisconnectGuard,
    projection: ProxyResponseProjection,
    mut shutdown: watch::Receiver<bool>,
) -> Response {
    loop {
        let event = tokio::select! {
            event = receiver.recv() => {
                let Some(event) = event else {
                    return ProxyRejection::Unavailable.response();
                };
                event
            }
            () = wait_for_shutdown(&mut shutdown) => {
                return ProxyRejection::Unavailable.response();
            }
        };
        if matches!(event, HostOperationEvent::StreamDelta { .. }) {
            return ProxyRejection::InvalidRequest.response();
        }
        if event.is_terminal() {
            guard.complete();
            let value = match projection.clone() {
                ProxyResponseProjection::Typed => CanonicalJson::new(
                    serde_json::to_value(event).expect("host operation event serializes"),
                )
                .map_err(|_| ProxyRejection::Unavailable),
                ProxyResponseProjection::Dialect(dialect) => dialect.project_terminal(event),
            };
            let value = match value {
                Ok(value) => value,
                Err(_) => return ProxyRejection::Unavailable.response(),
            };
            return (
                StatusCode::OK,
                [(CONTENT_TYPE, "application/json")],
                value.to_bytes(),
            )
                .into_response();
        }
    }
}

fn streaming_response(
    receiver: mpsc::Receiver<HostOperationEvent>,
    guard: ProxyDisconnectGuard,
    projection: ProxyResponseProjection,
    shutdown: watch::Receiver<bool>,
) -> Response {
    let output = stream::unfold(
        (receiver, guard, projection, shutdown),
        |(mut receiver, mut guard, projection, mut shutdown)| async move {
            let event = tokio::select! {
                event = receiver.recv() => event?,
                () = wait_for_shutdown(&mut shutdown) => return None,
            };
            let event_name = match &projection {
                ProxyResponseProjection::Typed => match &event {
                    HostOperationEvent::StreamDelta { .. } => "stream_delta",
                    HostOperationEvent::Usage { .. } => "usage",
                    HostOperationEvent::Terminal { .. } => "terminal",
                    HostOperationEvent::CancellationAcknowledged { .. } => {
                        "cancellation_acknowledged"
                    }
                },
                ProxyResponseProjection::Dialect(_) => "message",
            };
            if event.is_terminal() {
                guard.complete();
            }
            let data = match &projection {
                ProxyResponseProjection::Typed => CanonicalJson::new(
                    serde_json::to_value(event).expect("host operation event serializes"),
                )
                .ok()?,
                ProxyResponseProjection::Dialect(dialect) => dialect.project_stream(event).ok()?,
            }
            .to_bytes();
            let event = Event::default()
                .event(event_name)
                .data(String::from_utf8(data).expect("canonical JSON is UTF-8"));
            Some((
                Ok::<_, Infallible>(event),
                (receiver, guard, projection, shutdown),
            ))
        },
    );
    Sse::new(output).into_response()
}

#[cfg(test)]
mod tests {
    use std::process::{Command, Stdio};
    use std::time::Duration;

    use aiperf_accuracy::{
        EvaluationSessionId, HostCallContext, HostOperationDisposition, HostOperationId,
        HostOperationTerminal, HostOperationUsage, LogicalCallId, LogicalServiceId,
        OperationPurpose, ScopedProxySecret, SemanticAttemptId, SemanticOperationId,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;

    use super::*;
    use crate::evaluation::ledger::{HostTerminalClass, OperationLedger, OperationRegistration};

    fn grant(scope: &str) -> ScopedProxyGrant {
        ScopedProxyGrant {
            grant_id: "grant-1".into(),
            session_id: EvaluationSessionId::new("session-1").unwrap(),
            secret: ScopedProxySecret::new("s".repeat(48)).unwrap(),
            service_ids: vec![LogicalServiceId::new("primary").unwrap()],
            semantic_operation_ids: vec![SemanticOperationId::new("model.generate").unwrap()],
            purposes: vec![OperationPurpose::new("primary").unwrap()],
            process_scope_sha256: scope.into(),
            max_operations: 8,
            max_concurrent_operations: 2,
            max_request_bytes: 64 * 1024,
            max_response_bytes: 64 * 1024,
            max_stream_events: 16,
            expires_after_ms: 10_000,
        }
    }

    fn proxy_request_head(
        binding: &ScopedProxyBinding,
        path: &str,
        content_length: usize,
        extra_headers: &str,
    ) -> String {
        format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nAuthorization: Bearer {}\r\nx-aiperf-proxy-grant: {}\r\n{extra_headers}Content-Length: {content_length}\r\nConnection: close\r\n\r\n",
            binding.grant.secret.expose_secret(),
            binding.grant.grant_id,
        )
    }

    async fn raw_proxy_request(socket_path: PathBuf, head: String, body: Vec<u8>) -> String {
        let mut stream = UnixStream::connect(socket_path).await.unwrap();
        stream.write_all(head.as_bytes()).await.unwrap();
        stream.write_all(&body).await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    fn request(id: &str) -> HostOperationRequest {
        HostOperationRequest {
            operation_id: HostOperationId::new(id).unwrap(),
            context: HostCallContext {
                session_id: EvaluationSessionId::new("session-1").unwrap(),
                unit_id: aiperf_accuracy::EvaluationUnitId::new("unit-1").unwrap(),
                case_id: EvaluationCaseId::new("case-1").unwrap(),
                semantic_attempt_id: SemanticAttemptId::new("semantic-1").unwrap(),
                logical_call_id: LogicalCallId::new(format!("call-{id}")).unwrap(),
            },
            service_id: LogicalServiceId::new("primary").unwrap(),
            purpose: OperationPurpose::new("primary").unwrap(),
            semantic_operation_id: SemanticOperationId::new("model.generate").unwrap(),
            payload: CanonicalJson::new(json!({
                "messages": [{"role":"user","content":"hello"}],
                "generation": {"max_tokens": 4}
            }))
            .unwrap(),
            restricted_payload: None,
            response_mode: HostResponseMode::Streaming,
            deadline_ms: None,
            idempotency_key: format!("idempotency-{id}"),
        }
    }

    fn registration(request: &HostOperationRequest) -> OperationRegistration {
        OperationRegistration {
            operation_id: request.operation_id.to_string(),
            unit_id: request.context.unit_id.to_string(),
            case_id: request.context.case_id.to_string(),
            semantic_attempt_id: request.context.semantic_attempt_id.to_string(),
            logical_call_id: request.context.logical_call_id.to_string(),
            idempotency_key: request.idempotency_key.clone(),
            service_id: request.service_id.to_string(),
            semantic_operation_id: request.semantic_operation_id.to_string(),
            replay_safe_after_output: false,
        }
    }

    fn reserve_operation(
        runtime: &Arc<ProxyGrantRuntime>,
        request: &HostOperationRequest,
        request_bytes: u64,
        now_ns: i64,
        disconnect: Arc<AtomicBool>,
        events: mpsc::Sender<HostOperationEvent>,
    ) -> std::result::Result<ProxyOperationResponder, ProxyRejection> {
        let pending = runtime.reserve_pending(request_bytes)?;
        runtime.accept_pending(&pending, request, now_ns, disconnect, events)
    }

    fn pending_usage(runtime: &ProxyGrantRuntime) -> (usize, u64, u64, u64, u64) {
        let usage = runtime.usage.lock().unwrap();
        (
            usage.pending_reservations.len(),
            usage.pending_request_bytes,
            usage.accepted_operations,
            usage.active_operations,
            usage.request_bytes,
        )
    }

    #[test]
    fn pre_body_budget_bounds_concurrency_and_aggregate_bytes_with_raii_rollback() {
        let mut limits = grant(&"7".repeat(64));
        limits.max_concurrent_operations = 1;
        limits.max_request_bytes = 100;
        let runtime = Arc::new(ProxyGrantRuntime::new(limits, 0));

        let first = runtime.reserve_pending(60).unwrap();
        assert_eq!(pending_usage(&runtime), (1, 60, 0, 0, 0));
        assert!(matches!(
            runtime.reserve_pending(1),
            Err(ProxyRejection::GrantExhausted)
        ));
        drop(first);
        assert_eq!(pending_usage(&runtime), (0, 0, 0, 0, 0));

        let mut limits = grant(&"8".repeat(64));
        limits.max_concurrent_operations = 2;
        limits.max_request_bytes = 100;
        let runtime = Arc::new(ProxyGrantRuntime::new(limits, 0));
        let first = runtime.reserve_pending(60).unwrap();
        assert!(matches!(
            runtime.reserve_pending(41),
            Err(ProxyRejection::GrantExhausted)
        ));
        let second = runtime.reserve_pending(40).unwrap();
        assert_eq!(pending_usage(&runtime), (2, 100, 0, 0, 0));
        drop((first, second));
        assert_eq!(pending_usage(&runtime), (0, 0, 0, 0, 0));
    }

    #[test]
    fn request_head_requires_one_exact_content_length_and_rejects_chunking() {
        let grant = grant(&"6".repeat(64));
        let mut headers = HeaderMap::new();
        headers.insert(HOST, "localhost".parse().unwrap());
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(
            AUTHORIZATION,
            format!("Bearer {}", grant.secret.expose_secret())
                .parse()
                .unwrap(),
        );
        headers.insert(&GRANT_HEADER, grant.grant_id.parse().unwrap());
        assert_eq!(
            validate_proxy_headers(&headers, &grant, &[]),
            Err(ProxyRejection::InvalidRequest)
        );

        headers.insert(CONTENT_LENGTH, "19".parse().unwrap());
        assert_eq!(validate_proxy_headers(&headers, &grant, &[]), Ok(19));
        headers.insert("transfer-encoding", "chunked".parse().unwrap());
        assert_eq!(
            validate_proxy_headers(&headers, &grant, &[]),
            Err(ProxyRejection::InvalidRequest)
        );
        headers.remove("transfer-encoding");
        headers.insert(CONTENT_LENGTH, "19, 19".parse().unwrap());
        assert_eq!(
            validate_proxy_headers(&headers, &grant, &[]),
            Err(ProxyRejection::InvalidRequest)
        );
    }

    #[test]
    fn one_operation_grant_accepts_exactly_its_first_reserved_operation() {
        let mut limits = grant(&"5".repeat(64));
        limits.max_operations = 1;
        limits.max_concurrent_operations = 1;
        let runtime = Arc::new(ProxyGrantRuntime::new(limits, 0));
        let request = request("only-operation");
        runtime
            .activate_unit_cases(
                request.context.unit_id.clone(),
                [request.context.case_id.clone()],
            )
            .unwrap();
        let (events, _event_rx) = mpsc::channel(2);
        let responder = reserve_operation(
            &runtime,
            &request,
            1,
            0,
            Arc::new(AtomicBool::new(false)),
            events,
        )
        .unwrap();
        responder.activate().unwrap();
        assert_eq!(pending_usage(&runtime), (0, 0, 1, 1, 1));
        assert!(matches!(
            runtime.reserve_pending(1),
            Err(ProxyRejection::GrantExhausted)
        ));
        responder.complete_without_delivery();
        assert_eq!(pending_usage(&runtime), (0, 0, 1, 0, 1));
    }

    #[test]
    fn staged_case_scope_and_reservation_rollback_are_fail_closed() {
        let runtime = Arc::new(ProxyGrantRuntime::new(grant(&"a".repeat(64)), 100));
        let request = request("operation-1");
        let disconnect = Arc::new(AtomicBool::new(false));
        let (events, _receiver) = mpsc::channel(4);

        assert!(matches!(
            reserve_operation(
                &runtime,
                &request,
                128,
                100,
                Arc::clone(&disconnect),
                events.clone(),
            ),
            Err(ProxyRejection::GrantScope)
        ));
        runtime
            .activate_unit_cases(
                request.context.unit_id.clone(),
                [request.context.case_id.clone()],
            )
            .unwrap();
        let reserved = reserve_operation(
            &runtime,
            &request,
            128,
            100,
            Arc::clone(&disconnect),
            events.clone(),
        )
        .unwrap();
        reserved.reject();
        assert_eq!(pending_usage(&runtime), (0, 0, 0, 0, 0));
        let accepted =
            reserve_operation(&runtime, &request, 128, 100, disconnect, events.clone()).unwrap();
        accepted.activate().unwrap();
        assert_eq!(pending_usage(&runtime), (0, 0, 1, 1, 128));
        assert!(matches!(
            reserve_operation(
                &runtime,
                &request,
                128,
                100,
                Arc::new(AtomicBool::new(false)),
                events,
            ),
            Err(ProxyRejection::Duplicate)
        ));
        accepted.complete_without_delivery();
        assert_eq!(pending_usage(&runtime), (0, 0, 1, 0, 128));
        runtime.deactivate_case(&request.context.case_id).unwrap();
        assert!(matches!(
            runtime.activate_unit_cases(
                request.context.unit_id.clone(),
                [request.context.case_id.clone()],
            ),
            Err(ProxyRejection::GrantScope)
        ));
        let (events, _receiver) = mpsc::channel(4);
        assert!(matches!(
            reserve_operation(
                &runtime,
                &request,
                128,
                100,
                Arc::new(AtomicBool::new(false)),
                events,
            ),
            Err(ProxyRejection::GrantScope)
        ));
    }

    #[test]
    fn grant_rejects_wrong_purpose_and_clock_expiry() {
        let runtime = Arc::new(ProxyGrantRuntime::new(grant(&"b".repeat(64)), 10));
        let mut request = request("operation-1");
        runtime
            .activate_unit_cases(
                request.context.unit_id.clone(),
                [request.context.case_id.clone()],
            )
            .unwrap();
        request.purpose = OperationPurpose::new("judge").unwrap();
        let (events, _receiver) = mpsc::channel(4);
        assert!(matches!(
            reserve_operation(
                &runtime,
                &request,
                1,
                10,
                Arc::new(AtomicBool::new(false)),
                events.clone(),
            ),
            Err(ProxyRejection::GrantScope)
        ));
        request.purpose = OperationPurpose::new("primary").unwrap();
        request.context.session_id = EvaluationSessionId::new("session-2").unwrap();
        assert!(matches!(
            reserve_operation(
                &runtime,
                &request,
                1,
                10,
                Arc::new(AtomicBool::new(false)),
                events.clone(),
            ),
            Err(ProxyRejection::GrantScope)
        ));
        request.context.session_id = EvaluationSessionId::new("session-1").unwrap();
        assert!(matches!(
            reserve_operation(
                &runtime,
                &request,
                1,
                10_000_000_011,
                Arc::new(AtomicBool::new(false)),
                events,
            ),
            Err(ProxyRejection::GrantExhausted)
        ));
    }

    #[test]
    fn openai_chat_dialect_injects_fixed_route_and_correlation_and_rejects_model() {
        let runtime = ProxyGrantRuntime::new(grant(&"f".repeat(64)), 0);
        runtime
            .activate_unit_cases(
                aiperf_accuracy::EvaluationUnitId::new("unit-1").unwrap(),
                [EvaluationCaseId::new("case-1").unwrap()],
            )
            .unwrap();
        let dialect = OpenAiChatCompatibilityDialect::from_grant(&runtime.grant).unwrap();
        let mut headers = HeaderMap::new();
        headers.insert(&CASE_HEADER, "case-1".parse().unwrap());
        let lowered = dialect
            .lower(
                &runtime,
                &headers,
                br#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":4,"stream":true}"#,
            )
            .unwrap();
        assert_eq!(lowered.context.session_id.as_str(), "session-1");
        assert_eq!(lowered.context.unit_id.as_str(), "unit-1");
        assert_eq!(lowered.context.case_id.as_str(), "case-1");
        assert_eq!(lowered.service_id.as_str(), "primary");
        assert_eq!(lowered.purpose.as_str(), "primary");
        assert_eq!(lowered.semantic_operation_id.as_str(), "model.generate");
        assert_eq!(lowered.response_mode, HostResponseMode::Streaming);
        assert!(
            lowered
                .payload
                .value()
                .as_object()
                .is_some_and(|object| !object.contains_key("model"))
        );
        assert!(matches!(
            dialect.lower(
                &runtime,
                &headers,
                br#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":4,"model":"caller-route"}"#,
            ),
            Err(ProxyRejection::InvalidRequest)
        ));
    }

    #[test]
    fn linux_scope_accepts_root_and_rejects_sibling_and_unknown_pid() {
        let scope = "c".repeat(64);
        let authorizer = LinuxProcessSubtreeAuthorizer::new(scope.clone()).unwrap();
        let mut root = Command::new("sleep")
            .arg("5")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut sibling = Command::new("sleep")
            .arg("5")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        authorizer.bind_root(root.id()).unwrap();
        let root_uid = linux_process_uid(root.id()).unwrap();
        assert_eq!(
            authorizer.authorize(
                &scope,
                ProxyPeerIdentity {
                    pid: Some(root.id()),
                    uid: root_uid,
                    gid: 0,
                },
            ),
            Ok(())
        );
        assert_eq!(
            authorizer.authorize(
                &scope,
                ProxyPeerIdentity {
                    pid: Some(sibling.id()),
                    uid: linux_process_uid(sibling.id()).unwrap(),
                    gid: 0,
                },
            ),
            Err(ProxyRejection::ProcessScope)
        );
        assert_eq!(
            authorizer.authorize(
                &scope,
                ProxyPeerIdentity {
                    pid: Some(u32::MAX),
                    uid: root_uid,
                    gid: 0,
                },
            ),
            Err(ProxyRejection::ProcessScope)
        );
        let _ = root.kill();
        let _ = root.wait();
        let _ = sibling.kill();
        let _ = sibling.wait();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn accepted_connection_cap_bounds_slowloris_and_shutdown_releases_permits() {
        let scope = "0".repeat(64);
        let socket_path = PathBuf::from(format!(
            "/tmp/aiperf-evaluator-proxy-slowloris-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let mut scoped_grant = grant(&scope);
        scoped_grant.max_concurrent_operations = 1;
        let binding = ScopedProxyBinding {
            local_locator: "unix:///run/aiperf/evaluator-proxy.sock".into(),
            host_socket_path: socket_path.clone(),
            grant: scoped_grant,
        };
        let authorizer = Arc::new(LinuxProcessSubtreeAuthorizer::new(scope).unwrap());
        authorizer.bind_root(std::process::id()).unwrap();
        let (proxy, _ingress) = start_evaluator_compatibility_proxy(binding, 0, authorizer)
            .await
            .unwrap();
        assert_eq!(proxy.connection_capacity, 2);

        let first = UnixStream::connect(&socket_path).await.unwrap();
        let second = UnixStream::connect(&socket_path).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while proxy.available_connection_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("proxy did not account accepted slowloris connections");
        let third = UnixStream::connect(&socket_path).await.unwrap();
        for _ in 0..128 {
            tokio::task::yield_now().await;
        }
        assert_eq!(proxy.available_connection_permits(), 0);

        tokio::time::timeout(Duration::from_secs(2), proxy.shutdown())
            .await
            .expect("proxy shutdown did not wake incomplete connections")
            .unwrap();
        assert_eq!(proxy.available_connection_permits(), 2);
        drop((first, second, third));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn undrained_concurrency_rejects_immediately_and_shutdown_wakes_admission() {
        let scope = "3".repeat(64);
        let socket_path = PathBuf::from(format!(
            "/tmp/aiperf-evaluator-proxy-bounds-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let mut scoped_grant = grant(&scope);
        scoped_grant.max_concurrent_operations = 1;
        let binding = ScopedProxyBinding {
            local_locator: "unix:///run/aiperf/evaluator-proxy.sock".into(),
            host_socket_path: socket_path.clone(),
            grant: scoped_grant,
        };
        let authorizer = Arc::new(LinuxProcessSubtreeAuthorizer::new(scope).unwrap());
        authorizer.bind_root(std::process::id()).unwrap();
        let (proxy, ingress) = start_evaluator_compatibility_proxy(binding.clone(), 0, authorizer)
            .await
            .unwrap();
        ingress
            .activate_case_scope(
                aiperf_accuracy::EvaluationUnitId::new("unit-1").unwrap(),
                [EvaluationCaseId::new("case-1").unwrap()],
            )
            .unwrap();

        let first_body = CanonicalJson::new(serde_json::to_value(request("pending-1")).unwrap())
            .unwrap()
            .to_bytes();
        let first_head = proxy_request_head(&binding, PROXY_PATH, first_body.len(), "");
        let first_client = tokio::spawn(raw_proxy_request(
            socket_path.clone(),
            first_head,
            first_body,
        ));
        tokio::time::timeout(Duration::from_secs(2), async {
            while ingress.is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("first proxy request never reached bounded admission");
        assert_eq!(pending_usage(&ingress.runtime).0, 1);

        let second_body = CanonicalJson::new(serde_json::to_value(request("pending-2")).unwrap())
            .unwrap()
            .to_bytes();
        let second_head = proxy_request_head(&binding, PROXY_PATH, second_body.len(), "");
        let second_response = tokio::time::timeout(
            Duration::from_secs(2),
            raw_proxy_request(socket_path.clone(), second_head, second_body),
        )
        .await
        .expect("second request waited behind an undrained operation");
        assert!(
            second_response.starts_with("HTTP/1.1 429 Too Many Requests"),
            "{second_response}"
        );

        tokio::time::timeout(Duration::from_secs(2), proxy.shutdown())
            .await
            .expect("shutdown did not wake pending admission")
            .unwrap();
        let first_response = tokio::time::timeout(Duration::from_secs(2), first_client)
            .await
            .expect("pending client did not wake at shutdown")
            .unwrap();
        assert!(
            first_response.starts_with("HTTP/1.1 503 Service Unavailable"),
            "{first_response}"
        );
        assert_eq!(pending_usage(&ingress.runtime), (0, 0, 0, 0, 0));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn aggregate_bodies_malformed_json_and_disconnect_roll_back_pre_admission_credits() {
        let scope = "4".repeat(64);
        let socket_path = PathBuf::from(format!(
            "/tmp/aiperf-evaluator-proxy-body-bounds-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let first_body = CanonicalJson::new(serde_json::to_value(request("body-1")).unwrap())
            .unwrap()
            .to_bytes();
        let mut scoped_grant = grant(&scope);
        scoped_grant.max_request_bytes =
            u64::try_from(first_body.len() + first_body.len() / 2).unwrap();
        let binding = ScopedProxyBinding {
            local_locator: "unix:///run/aiperf/evaluator-proxy.sock".into(),
            host_socket_path: socket_path.clone(),
            grant: scoped_grant,
        };
        let authorizer = Arc::new(LinuxProcessSubtreeAuthorizer::new(scope).unwrap());
        authorizer.bind_root(std::process::id()).unwrap();
        let (proxy, ingress) = start_evaluator_compatibility_proxy(binding.clone(), 0, authorizer)
            .await
            .unwrap();

        let mut first_stream = UnixStream::connect(&socket_path).await.unwrap();
        let first_head = proxy_request_head(&binding, PROXY_PATH, first_body.len(), "");
        first_stream.write_all(first_head.as_bytes()).await.unwrap();
        first_stream
            .write_all(&first_body[..first_body.len() / 2])
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while pending_usage(&ingress.runtime).0 != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("partial first body never acquired its aggregate reservation");

        let second_body = CanonicalJson::new(serde_json::to_value(request("body-2")).unwrap())
            .unwrap()
            .to_bytes();
        let second_head = proxy_request_head(&binding, PROXY_PATH, second_body.len(), "");
        let second_response = tokio::time::timeout(
            Duration::from_secs(2),
            raw_proxy_request(socket_path.clone(), second_head, second_body),
        )
        .await
        .expect("aggregate-overflow body waited behind the first extractor");
        assert!(
            second_response.starts_with("HTTP/1.1 429 Too Many Requests"),
            "{second_response}"
        );

        drop(first_stream);
        tokio::time::timeout(Duration::from_secs(2), async {
            while pending_usage(&ingress.runtime).0 != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("disconnected partial body retained pending credits");
        assert_eq!(pending_usage(&ingress.runtime), (0, 0, 0, 0, 0));

        let malformed = b"{".to_vec();
        let malformed_head = proxy_request_head(&binding, PROXY_PATH, malformed.len(), "");
        let malformed_response =
            raw_proxy_request(socket_path.clone(), malformed_head, malformed).await;
        assert!(
            malformed_response.starts_with("HTTP/1.1 400 Bad Request"),
            "{malformed_response}"
        );
        assert_eq!(pending_usage(&ingress.runtime), (0, 0, 0, 0, 0));
        proxy.shutdown().await.unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unix_http_sse_is_authenticated_normalized_and_uses_the_shared_ledger() {
        let scope = "d".repeat(64);
        let socket_path = PathBuf::from(format!(
            "/tmp/aiperf-evaluator-proxy-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let binding = ScopedProxyBinding {
            local_locator: "unix:///run/aiperf/evaluator-proxy.sock".into(),
            host_socket_path: socket_path.clone(),
            grant: grant(&scope),
        };
        let authorizer = Arc::new(LinuxProcessSubtreeAuthorizer::new(scope).unwrap());
        authorizer.bind_root(std::process::id()).unwrap();
        let (proxy, mut ingress) =
            start_evaluator_compatibility_proxy(binding.clone(), 0, authorizer)
                .await
                .unwrap();
        ingress
            .activate_case_scope(
                aiperf_accuracy::EvaluationUnitId::new("unit-1").unwrap(),
                [EvaluationCaseId::new("case-1").unwrap()],
            )
            .unwrap();

        let operation = request("operation-1");
        let body = CanonicalJson::new(serde_json::to_value(&operation).unwrap())
            .unwrap()
            .to_bytes();
        let client = async {
            let mut stream = UnixStream::connect(&socket_path).await.unwrap();
            let request = format!(
                "POST {PROXY_PATH} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nAuthorization: Bearer {}\r\nx-aiperf-proxy-grant: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                binding.grant.secret.expose_secret(),
                binding.grant.grant_id,
                body.len(),
            );
            stream.write_all(request.as_bytes()).await.unwrap();
            stream.write_all(&body).await.unwrap();
            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.unwrap();
            String::from_utf8(response).unwrap()
        };
        let authority = async {
            let submission = loop {
                match ingress.try_recv() {
                    Ok(submission) => break submission,
                    Err(mpsc::error::TryRecvError::Empty) => tokio::task::yield_now().await,
                    Err(error) => panic!("proxy ingress closed: {error}"),
                }
            };
            let request = submission.request().clone();
            let responder = ingress.authorize(&submission, 0).unwrap();
            let mut ledger = OperationLedger::default();
            ledger.register(registration(&request)).unwrap();
            ledger.admit(request.operation_id.as_str()).unwrap();
            ledger
                .start_attempt(request.operation_id.as_str(), "attempt-1".into())
                .unwrap();
            responder.activate().unwrap();
            let _ = submission.resolve(Ok(()));
            responder
                .publish(HostOperationEvent::StreamDelta {
                    operation_id: request.operation_id.clone(),
                    stream_sequence: 0,
                    delta: CanonicalJson::new(json!({
                        "choice_index": 0,
                        "delta": {"role":"assistant","content":"hello"}
                    }))
                    .unwrap(),
                })
                .unwrap();
            ledger
                .observe_output(request.operation_id.as_str(), "attempt-1")
                .unwrap();
            ledger
                .finish_attempt(
                    request.operation_id.as_str(),
                    "attempt-1",
                    HostTerminalClass::Completed,
                )
                .unwrap();
            ledger
                .finish_operation(request.operation_id.as_str(), HostTerminalClass::Completed)
                .unwrap();
            responder
                .publish(HostOperationEvent::Terminal {
                    terminal: HostOperationTerminal {
                        operation_id: request.operation_id,
                        semantic_attempt_id: request.context.semantic_attempt_id,
                        disposition: HostOperationDisposition::Completed,
                        result: Some(
                            CanonicalJson::new(json!({
                                "choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop","stop_reason":"stop"}],
                                "usage":{"prompt_tokens":1,"completion_tokens":1}
                            }))
                            .unwrap(),
                        ),
                        error: None,
                        usage: HostOperationUsage {
                            prompt_tokens: Some(1),
                            completion_tokens: Some(1),
                            reasoning_tokens: None,
                            cached_tokens: None,
                        },
                        observed_output: true,
                    },
                })
                .unwrap();
            ledger.validate_drained().unwrap();
        };
        let (response, ()) = tokio::join!(client, authority);
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains("event: stream_delta"));
        assert!(response.contains("event: terminal"));
        assert!(response.contains("\"kind\":\"stream_delta\""));
        assert!(!response.contains("https://"));
        assert!(!response.contains("api_key"));
        assert!(!response.contains("data: [DONE]"));
        proxy.shutdown().await.unwrap();
        assert!(!socket_path.exists());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn proxy_disconnect_revokes_the_same_operation_and_drop_removes_socket() {
        let scope = "1".repeat(64);
        let socket_path = PathBuf::from(format!(
            "/tmp/aiperf-evaluator-proxy-disconnect-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let binding = ScopedProxyBinding {
            local_locator: "unix:///run/aiperf/evaluator-proxy.sock".into(),
            host_socket_path: socket_path.clone(),
            grant: grant(&scope),
        };
        let authorizer = Arc::new(LinuxProcessSubtreeAuthorizer::new(scope).unwrap());
        authorizer.bind_root(std::process::id()).unwrap();
        let (proxy, mut ingress) =
            start_evaluator_compatibility_proxy(binding.clone(), 0, authorizer)
                .await
                .unwrap();
        ingress
            .activate_case_scope(
                aiperf_accuracy::EvaluationUnitId::new("unit-1").unwrap(),
                [EvaluationCaseId::new("case-1").unwrap()],
            )
            .unwrap();
        let operation = request("disconnect-operation");
        let body = CanonicalJson::new(serde_json::to_value(&operation).unwrap())
            .unwrap()
            .to_bytes();
        let client = async {
            let mut stream = UnixStream::connect(&socket_path).await.unwrap();
            let request = format!(
                "POST {PROXY_PATH} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nAuthorization: Bearer {}\r\nx-aiperf-proxy-grant: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                binding.grant.secret.expose_secret(),
                binding.grant.grant_id,
                body.len(),
            );
            stream.write_all(request.as_bytes()).await.unwrap();
            stream.write_all(&body).await.unwrap();
            let mut headers = vec![0_u8; 1024];
            let read = stream.read(&mut headers).await.unwrap();
            assert!(String::from_utf8_lossy(&headers[..read]).contains("200 OK"));
        };
        let authority = async {
            let submission = loop {
                match ingress.try_recv() {
                    Ok(submission) => break submission,
                    Err(mpsc::error::TryRecvError::Empty) => tokio::task::yield_now().await,
                    Err(error) => panic!("proxy ingress closed: {error}"),
                }
            };
            let responder = ingress.authorize(&submission, 0).unwrap();
            responder.activate().unwrap();
            let _ = submission.resolve(Ok(()));
            for _ in 0..100_000 {
                if responder.is_disconnected() {
                    responder.complete_without_delivery();
                    return;
                }
                tokio::task::yield_now().await;
            }
            panic!("local proxy disconnect did not cancel its operation");
        };
        tokio::join!(client, authority);
        assert!(ingress.disconnected_operation_ids().is_empty());
        drop(proxy);
        assert!(!socket_path.exists());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unix_openai_chat_route_injects_authority_and_returns_compatible_json() {
        let scope = "2".repeat(64);
        let socket_path = PathBuf::from(format!(
            "/tmp/aiperf-evaluator-proxy-openai-{}.sock",
            uuid::Uuid::new_v4()
        ));
        let binding = ScopedProxyBinding {
            local_locator: "unix:///run/aiperf/evaluator-proxy.sock".into(),
            host_socket_path: socket_path.clone(),
            grant: grant(&scope),
        };
        let authorizer = Arc::new(LinuxProcessSubtreeAuthorizer::new(scope).unwrap());
        authorizer.bind_root(std::process::id()).unwrap();
        let (proxy, mut ingress) =
            start_evaluator_compatibility_proxy(binding.clone(), 0, authorizer)
                .await
                .unwrap();
        ingress
            .activate_case_scope(
                aiperf_accuracy::EvaluationUnitId::new("unit-1").unwrap(),
                [EvaluationCaseId::new("case-1").unwrap()],
            )
            .unwrap();
        let body = br#"{"messages":[{"role":"user","content":"hello"}],"max_tokens":4}"#;
        let client = async {
            let mut stream = UnixStream::connect(&socket_path).await.unwrap();
            let request = format!(
                "POST {OPENAI_CHAT_PATH} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nAccept: application/json\r\nAuthorization: Bearer {}\r\nx-aiperf-proxy-grant: {}\r\nx-aiperf-case-id: case-1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                binding.grant.secret.expose_secret(),
                binding.grant.grant_id,
                body.len(),
            );
            stream.write_all(request.as_bytes()).await.unwrap();
            stream.write_all(body).await.unwrap();
            let mut response = Vec::new();
            stream.read_to_end(&mut response).await.unwrap();
            String::from_utf8(response).unwrap()
        };
        let authority = async {
            let submission = loop {
                match ingress.try_recv() {
                    Ok(submission) => break submission,
                    Err(mpsc::error::TryRecvError::Empty) => tokio::task::yield_now().await,
                    Err(error) => panic!("proxy ingress closed: {error}"),
                }
            };
            let request = submission.request().clone();
            assert_eq!(request.context.session_id.as_str(), "session-1");
            assert_eq!(request.context.unit_id.as_str(), "unit-1");
            assert_eq!(request.service_id.as_str(), "primary");
            assert_eq!(request.semantic_operation_id.as_str(), "model.generate");
            assert!(!request.payload.value().to_string().contains("model"));
            let responder = ingress.authorize(&submission, 0).unwrap();
            responder.activate().unwrap();
            let _ = submission.resolve(Ok(()));
            responder
                .publish(HostOperationEvent::Terminal {
                    terminal: HostOperationTerminal {
                        operation_id: request.operation_id,
                        semantic_attempt_id: request.context.semantic_attempt_id,
                        disposition: HostOperationDisposition::Completed,
                        result: Some(
                            CanonicalJson::new(json!({
                                "choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop","stop_reason":"stop"}],
                                "usage":{"prompt_tokens":1,"completion_tokens":1}
                            }))
                            .unwrap(),
                        ),
                        error: None,
                        usage: HostOperationUsage::default(),
                        observed_output: false,
                    },
                })
                .unwrap();
        };
        let (response, ()) = tokio::join!(client, authority);
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains("\"choices\""));
        assert!(response.contains("\"message\""));
        assert!(!response.contains("\"kind\":\"terminal\""));
        assert!(!response.contains("proxy-operation-"));
        proxy.shutdown().await.unwrap();
    }

    #[test]
    fn unauthorized_response_never_echoes_secret() {
        let secret = "never-echo-this-private-capability";
        let response = ProxyRejection::Unauthorized.response();
        let debug = format!("{response:?}");
        assert!(!debug.contains(secret));
        assert!(!format!("{:?}", grant(&"e".repeat(64)).secret).contains(&"s".repeat(48)));
    }

    #[test]
    fn ingress_rejects_before_waiting_when_its_bounded_queue_is_full() {
        let (sender, mut receiver) = mpsc::channel(1);
        let runtime = Arc::new(ProxyGrantRuntime::new(grant(&"9".repeat(64)), 0));
        let submission = |id: &str| {
            let (events, _events_rx) = mpsc::channel(1);
            let (admission, _admission_rx) = oneshot::channel();
            ProxyOperationSubmission {
                request: request(id),
                reservation: runtime.reserve_pending(1).unwrap(),
                events,
                disconnect: Arc::new(AtomicBool::new(false)),
                admission,
            }
        };

        enqueue_proxy_submission(&sender, submission("queued")).unwrap();
        assert_eq!(
            enqueue_proxy_submission(&sender, submission("overflow")),
            Err(ProxyRejection::GrantExhausted)
        );
        assert_eq!(
            receiver.try_recv().unwrap().request().operation_id.as_str(),
            "queued"
        );
        drop(receiver);
        assert_eq!(
            enqueue_proxy_submission(&sender, submission("closed")),
            Err(ProxyRejection::Unavailable)
        );
        assert_eq!(pending_usage(&runtime), (0, 0, 0, 0, 0));

        let (events, _events_rx) = mpsc::channel(1);
        let (admission, admission_rx) = oneshot::channel();
        let disconnected = ProxyOperationSubmission {
            request: request("disconnected"),
            reservation: runtime.reserve_pending(1).unwrap(),
            events,
            disconnect: Arc::new(AtomicBool::new(false)),
            admission,
        };
        drop(admission_rx);
        assert!(!disconnected.is_connected());
        drop(disconnected);
        assert_eq!(pending_usage(&runtime), (0, 0, 0, 0, 0));
    }
}
