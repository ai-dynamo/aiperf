// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online HTTP dispatch over the `aiperf-transport-http` (hyper) client.
//!
//! [`TransportSink`] implements `loadgen_core`'s [`RequestSink`] using the
//! Rust-native `aiperf-transport-http` client (hyper + the `aiperf-clock` `Clock`). It
//! is single-threaded (`!Send`, `Rc`-based) and driven on a `LocalSet`;
//! admit/token times are stamped from the same clock origin the run loop uses for
//! arrival, so all events share one timeline.
//!
//! Per-request cancellation and endpoint resolution consume the scalars ported
//! from `src/aiperf/credit/issuer.py:197-238` and preserve the full-send timer
//! invariant from `src/aiperf/timing/request_cancellation.py:53-82`.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use uuid::Uuid;

use aiperf_clock::Clock;
use aiperf_core::chat::chat_request_body;
use aiperf_core::sse::ChatChunk;
use aiperf_dataset::EndpointResolver;
use aiperf_endpoints::{Endpoint, EndpointConfig, EndpointId, EndpointKey, PreparedEndpointTable};
use aiperf_metrics::{HttpTrace, InferenceDimensions};
use aiperf_transport_http::config::ClientConfig;
use aiperf_transport_http::models::{
    ConnectionReuseStrategy, ErrorDetails, ErrorKind, HttpVersion, RequestConfig, RequestRecord,
    Response, SseMessage,
};
use aiperf_transport_http::transport::http_transport::HttpTransport;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    Dispatchable, ObservedTokenKind, ObservedUsage, RequestObserver, RequestSink,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::multiturn::TurnToSend;
use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome, TurnDispatcher};

mod endpoint_dispatch;

/// Return true only for an SSE message that the current OpenAI-chat parser
/// would record as a token. This mirrors the Python worker callback at
/// `src/aiperf/workers/worker.py:474-487`: role-only, usage-only, finish-only,
/// malformed, and `[DONE]` messages do not release prefill capacity.
fn is_meaningful_chat_token(message: &SseMessage) -> bool {
    let Some(data) = message.data() else {
        return false;
    };
    serde_json::from_str::<ChatChunk>(data).is_ok_and(|chunk| !chunk.delta_text().is_empty())
}

/// A slim online HTTP request carrying prompt text. This is the load
/// generator's own request type; implementing [`Dispatchable`] is all the
/// dispatch seam requires.
#[derive(Clone)]
pub struct HttpRequest {
    /// Stable per-request identifier used to correlate observer events.
    pub uuid: Uuid,
    /// Prompt length in tokens, for measurement accounting.
    pub input_length: usize,
    /// Maximum number of output tokens to request.
    pub max_output_tokens: usize,
    /// Prompt text placed on the wire.
    pub prompt_text: Option<String>,
    /// Optional prebuilt JSON request body. Accuracy benchmarks use this to
    /// preserve benchmark-specific messages, sampling settings, and stop strings;
    /// normal synthetic requests leave it absent and use the shared chat builder.
    pub request_body: Option<Value>,
    /// Optional already-serialized request body. Unified dataset materializers
    /// use this byte-exact fast path; it is mutually exclusive with
    /// [`request_body`](Self::request_body).
    pub request_body_bytes: Option<Bytes>,
    /// Per-request HTTP headers supplied by the dataset/endpoint seam.
    pub headers: BTreeMap<String, String>,
    /// Per-request URL query parameters supplied by the dataset/endpoint seam.
    pub parameters: BTreeMap<String, String>,
    /// Endpoint path selected by the request formatter. Absolute URLs are also
    /// accepted for a turn-specific target.
    pub endpoint_path: Option<String>,
    /// Whether the response uses server-sent events.
    pub streaming: bool,
    /// Optional session correlation id forwarded to the transport.
    pub x_correlation_id: Option<String>,
    /// Whether this request is the final turn for its correlated session.
    pub is_final_turn: bool,
    /// Fixed cancellation delay armed at transport send-complete.
    pub cancel_after_ns: Option<i64>,
    /// Effective endpoint index for this request.
    pub url_index: Option<u32>,
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpRequest")
            .field("uuid", &self.uuid)
            .field("input_length", &self.input_length)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("has_prompt_text", &self.prompt_text.is_some())
            .field("has_request_body", &self.request_body.is_some())
            .field(
                "request_body_bytes_len",
                &self.request_body_bytes.as_ref().map(Bytes::len),
            )
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("parameters", &self.parameters)
            .field("endpoint_path", &self.endpoint_path)
            .field("streaming", &self.streaming)
            .field("x_correlation_id", &self.x_correlation_id)
            .field("is_final_turn", &self.is_final_turn)
            .field("cancel_after_ns", &self.cancel_after_ns)
            .field("url_index", &self.url_index)
            .finish()
    }
}

/// Version of the trusted execution-command wire.
pub const HTTP_EXECUTION_COMMAND_VERSION: u32 = 1;

/// Data-only HTTP request carried across an execution-placement boundary.
///
/// `bytes::Bytes` is projected to `Vec<u8>` so this DTO has no transport- or
/// allocator-specific representation. Header values may contain credentials;
/// the DTO is therefore for an authenticated runner-to-worker channel, never
/// for reports or logs.
#[derive(Clone, Serialize, Deserialize)]
pub struct HttpRequestWire {
    /// Stable request identifier.
    pub uuid: Uuid,
    /// Accounted input-token length.
    pub input_length: usize,
    /// Requested output-token limit.
    pub max_output_tokens: usize,
    /// Optional synthetic prompt text.
    pub prompt_text: Option<String>,
    /// Optional decoded request body.
    pub request_body: Option<Value>,
    /// Optional byte-exact request body.
    pub request_body_bytes: Option<Vec<u8>>,
    /// Per-request headers, including credentials when the endpoint requires them.
    pub headers: BTreeMap<String, String>,
    /// Per-request URL query parameters.
    pub parameters: BTreeMap<String, String>,
    /// Endpoint path or absolute URL override.
    pub endpoint_path: Option<String>,
    /// Whether the response is streamed.
    pub streaming: bool,
    /// Optional session correlation identifier.
    pub x_correlation_id: Option<String>,
    /// Whether this is the final correlated turn.
    pub is_final_turn: bool,
    /// Post-send cancellation delay.
    pub cancel_after_ns: Option<i64>,
    /// Effective endpoint index.
    pub url_index: Option<u32>,
}

impl fmt::Debug for HttpRequestWire {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("HttpRequestWire")
            .field("uuid", &self.uuid)
            .field("input_length", &self.input_length)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("has_prompt_text", &self.prompt_text.is_some())
            .field("has_request_body", &self.request_body.is_some())
            .field(
                "request_body_bytes_len",
                &self.request_body_bytes.as_ref().map(Vec::len),
            )
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field(
                "parameter_names",
                &self.parameters.keys().collect::<Vec<_>>(),
            )
            .field("endpoint_path", &self.endpoint_path)
            .field("streaming", &self.streaming)
            .field("has_correlation_id", &self.x_correlation_id.is_some())
            .field("is_final_turn", &self.is_final_turn)
            .field("cancel_after_ns", &self.cancel_after_ns)
            .field("url_index", &self.url_index)
            .finish()
    }
}

impl From<HttpRequest> for HttpRequestWire {
    fn from(request: HttpRequest) -> Self {
        Self {
            uuid: request.uuid,
            input_length: request.input_length,
            max_output_tokens: request.max_output_tokens,
            prompt_text: request.prompt_text,
            request_body: request.request_body,
            request_body_bytes: request.request_body_bytes.map(|bytes| bytes.to_vec()),
            headers: request.headers,
            parameters: request.parameters,
            endpoint_path: request.endpoint_path,
            streaming: request.streaming,
            x_correlation_id: request.x_correlation_id,
            is_final_turn: request.is_final_turn,
            cancel_after_ns: request.cancel_after_ns,
            url_index: request.url_index,
        }
    }
}

impl From<HttpRequestWire> for HttpRequest {
    fn from(request: HttpRequestWire) -> Self {
        Self {
            uuid: request.uuid,
            input_length: request.input_length,
            max_output_tokens: request.max_output_tokens,
            prompt_text: request.prompt_text,
            request_body: request.request_body,
            request_body_bytes: request.request_body_bytes.map(Bytes::from),
            headers: request.headers,
            parameters: request.parameters,
            endpoint_path: request.endpoint_path,
            streaming: request.streaming,
            x_correlation_id: request.x_correlation_id,
            is_final_turn: request.is_final_turn,
            cancel_after_ns: request.cancel_after_ns,
            url_index: request.url_index,
        }
    }
}

/// Generated response returned by the response-capturing dispatch path.
#[derive(Debug, Clone)]
pub struct HttpDispatchResult {
    /// Clock timestamp when transport dispatch started.
    pub start_ns: i64,
    /// Clock timestamp when the request reached terminal.
    pub end_ns: i64,
    /// HTTP status code, when response headers were received.
    pub status: Option<u16>,
    /// Request terminal classification.
    pub terminal: ReplayTerminalStatus,
    /// Generated reasoning/content text in stream order.
    pub response_text: String,
    /// Endpoint-normalized assistant and terminal response metadata.
    pub model_response: ModelResponseMetadata,
    /// Authoritative prompt-token count, when the server emitted usage.
    pub prompt_tokens: Option<u32>,
    /// Authoritative completion-token count, when the server emitted usage.
    pub completion_tokens: Option<u32>,
    /// Fine-grained request trace converted into native metric facts.
    pub http: HttpTrace,
}

struct HttpCollectedDispatch {
    result: HttpDispatchResult,
    request_payload: Bytes,
    record: RequestRecord,
}

/// HTTP-specific terminal result retained by raw-artifact consumers.
///
/// Policy-neutral workloads continue to consume [`TurnDispatchOutcome`]. The
/// native subprocess runner calls the concrete collection method only when it
/// must preserve HTTP wire facts; alternate backends do not inherit an HTTP
/// dependency through the shared [`TurnDispatcher`] seam.
#[derive(Clone, Debug)]
pub struct HttpTurnDispatchResult {
    /// Backend-neutral result consumed by scheduling and record processors.
    pub outcome: TurnDispatchOutcome,
    /// Canonical JSON payload before transport-specific body preparation.
    pub request_payload: Bytes,
    /// Exact HTTP transport record.
    pub record: RequestRecord,
}

/// Owned execution command handed from the single logical dispatcher to an
/// injected HTTP execution backend.
///
/// The scheduling-only [`TurnToSend`] retains an `Rc` session backend so that
/// continuations can be materialized locally. This projection deliberately
/// removes that scheduler state: every remaining field is owned and `Send`, and
/// the endpoint's stable wire identity lives in
/// [`EndpointConfig::endpoint_type`]. A cross-process backend can transmit the
/// data fields and re-resolve the stateless endpoint adapter on the far side;
/// local backends retain the already-resolved adapter allocation.
#[derive(Clone)]
pub struct PreparedHttpTurn {
    /// Transport-ready request fields.
    pub request: HttpRequest,
    /// Worker-resolved endpoint binding selected during preparation.
    pub endpoint: PreparedHttpEndpoint,
    /// Whether the request came from the endpoint-aware dataset seam.
    pub endpoint_aware: bool,
}

impl fmt::Debug for PreparedHttpTurn {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedHttpTurn")
            .field("request", &self.request)
            .field("endpoint", &self.endpoint)
            .field("endpoint_aware", &self.endpoint_aware)
            .finish()
    }
}

/// Copyable open endpoint identity carried from the coordinator to a worker.
///
/// Every worker prepares profiles in the same deterministic order. The dense
/// key selects the hot-path binding, while the canonical ID detects a mismatched
/// remote registry before any request is sent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedEndpointReference {
    /// Worker-local dense table key.
    pub key: EndpointKey,
    /// Canonical open endpoint identity expected at that key.
    pub endpoint_id: EndpointId,
}

/// Endpoint selection retained by one scheduler-free HTTP command.
#[derive(Clone)]
pub enum PreparedHttpEndpoint {
    /// Protocol-v1 compatibility adapter.
    Legacy {
        /// Stateless legacy endpoint implementation.
        endpoint: Arc<dyn Endpoint>,
        /// Closed compatibility configuration.
        config: Box<EndpointConfig>,
    },
    /// Protocol-v2 worker-local prepared binding.
    Prepared(PreparedEndpointReference),
}

impl fmt::Debug for PreparedHttpEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Legacy { endpoint, config } => formatter
                .debug_struct("LegacyEndpoint")
                .field("endpoint", &endpoint.metadata().endpoint_type)
                .field("config", config)
                .finish(),
            Self::Prepared(reference) => formatter
                .debug_tuple("PreparedEndpoint")
                .field(reference)
                .finish(),
        }
    }
}

/// Versioned command sent from the one dispatcher to a placement backend.
///
/// This DTO deliberately excludes the `Arc<dyn Endpoint>` implementation.
/// Workers re-resolve the stable endpoint identity through their frozen
/// registry, so native thread pools and future ZMQ/RPC workers execute the same
/// adapter contract. Endpoint headers and API keys are carried separately
/// because ordinary [`EndpointConfig`] serialization intentionally redacts
/// them from artifacts. Consequently this command belongs only on a trusted,
/// authenticated execution channel and its [`Debug`] implementation never
/// prints secret values.
#[derive(Clone, Serialize, Deserialize)]
pub struct PreparedHttpTurnWire {
    /// Execution-command schema version.
    pub version: u32,
    /// Transport-ready request data.
    pub request: HttpRequestWire,
    /// Stable endpoint selection. Prepared bindings carry an open ID and dense
    /// key; legacy commands retain their compatibility configuration.
    pub endpoint: PreparedHttpEndpointWire,
    /// Endpoint-level headers omitted by artifact-safe config serialization.
    pub endpoint_headers: BTreeMap<String, String>,
    /// Endpoint API key omitted by artifact-safe config serialization.
    pub endpoint_api_key: Option<String>,
    /// Whether endpoint-aware dataset materialization produced this request.
    pub endpoint_aware: bool,
}

impl fmt::Debug for PreparedHttpTurnWire {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedHttpTurnWire")
            .field("version", &self.version)
            .field("request", &self.request)
            .field("endpoint", &self.endpoint)
            .field(
                "endpoint_header_names",
                &self.endpoint_headers.keys().collect::<Vec<_>>(),
            )
            .field("has_endpoint_api_key", &self.endpoint_api_key.is_some())
            .field("endpoint_aware", &self.endpoint_aware)
            .finish()
    }
}

/// Data-only endpoint selection for an authenticated execution channel.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PreparedHttpEndpointWire {
    /// Protocol-v1 closed compatibility configuration.
    Legacy(Box<EndpointConfig>),
    /// Protocol-v2 open endpoint identity and worker-local dense key.
    Prepared(PreparedEndpointReference),
}

impl PreparedHttpTurn {
    /// Remove scheduler-local session state and build one owned HTTP command.
    pub fn from_turn(turn: TurnToSend, model: &str) -> Self {
        let is_final_turn = turn.is_final_turn();
        let endpoint_aware = turn.request_body.is_some();
        let mut endpoint_config = turn.endpoint_config;
        endpoint_config.streaming = turn.streaming;
        let request_body = if turn.request_body.is_none() {
            let messages = turn
                .messages
                .iter()
                .map(|message| (message.role.as_str(), message.content.as_str()))
                .collect::<Vec<_>>();
            Some(chat_request_body(model, &messages, turn.max_output_tokens))
        } else {
            None
        };
        Self {
            request: HttpRequest {
                uuid: turn.uuid,
                input_length: turn.input_length,
                max_output_tokens: turn.max_output_tokens,
                prompt_text: None,
                request_body,
                request_body_bytes: turn.request_body,
                headers: turn.request_headers,
                parameters: turn.request_parameters,
                endpoint_path: turn.endpoint_path,
                streaming: turn.streaming,
                x_correlation_id: Some(turn.request_correlation_id),
                is_final_turn,
                cancel_after_ns: turn.cancel_after_ns,
                url_index: turn.url_index,
            },
            endpoint: PreparedHttpEndpoint::Legacy {
                endpoint: turn.endpoint,
                config: Box::new(endpoint_config),
            },
            endpoint_aware,
        }
    }

    /// Project this local command into the stable data-only execution wire.
    pub fn into_wire(self) -> PreparedHttpTurnWire {
        let (endpoint, endpoint_headers, endpoint_api_key) = match self.endpoint {
            PreparedHttpEndpoint::Legacy {
                endpoint: _,
                config,
            } => {
                let headers = config.headers.clone();
                let api_key = config.api_key.clone();
                (PreparedHttpEndpointWire::Legacy(config), headers, api_key)
            }
            PreparedHttpEndpoint::Prepared(reference) => (
                PreparedHttpEndpointWire::Prepared(reference),
                BTreeMap::new(),
                None,
            ),
        };
        PreparedHttpTurnWire {
            version: HTTP_EXECUTION_COMMAND_VERSION,
            request: self.request.into(),
            endpoint,
            endpoint_headers,
            endpoint_api_key,
            endpoint_aware: self.endpoint_aware,
        }
    }
}

impl PreparedHttpTurnWire {
    /// Rehydrate a received command through the worker's frozen endpoint registry.
    pub fn into_prepared(
        self,
        endpoint_resolver: &dyn EndpointResolver,
    ) -> Result<PreparedHttpTurn> {
        anyhow::ensure!(
            self.version == HTTP_EXECUTION_COMMAND_VERSION,
            "HTTP execution command version {} is unsupported; expected {}",
            self.version,
            HTTP_EXECUTION_COMMAND_VERSION,
        );
        let endpoint = match self.endpoint {
            PreparedHttpEndpointWire::Legacy(mut config) => {
                config.headers = self.endpoint_headers;
                config.api_key = self.endpoint_api_key;
                let endpoint = endpoint_resolver.resolve_type(config.endpoint_type)?;
                PreparedHttpEndpoint::Legacy { endpoint, config }
            }
            PreparedHttpEndpointWire::Prepared(reference) => {
                anyhow::ensure!(
                    self.endpoint_headers.is_empty() && self.endpoint_api_key.is_none(),
                    "prepared endpoint command must not duplicate profile credentials"
                );
                PreparedHttpEndpoint::Prepared(reference)
            }
        };
        Ok(PreparedHttpTurn {
            request: self.request.into(),
            endpoint,
            endpoint_aware: self.endpoint_aware,
        })
    }
}

/// Pluggable execution placement behind the one logical turn dispatcher.
///
/// Implementations may execute on the caller's reactor, a thread-per-core
/// local pool, or a remote transport such as ZMQ. Scheduling, phase policy,
/// admission, adaptive control, and record capture remain above this seam and
/// therefore do not change when execution placement changes.
#[async_trait(?Send)]
pub trait HttpTurnExecutionBackend {
    /// Set the shared run origin after backend startup and before dispatch.
    fn set_run_origin(&self, start_ns: i64) -> Result<()>;

    /// Resolve labels using the same endpoint/model selection as execution.
    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions;

    /// Execute one prepared request and replay its observations into the local
    /// dispatcher observer. `on_first_token` must be delivered promptly because
    /// it releases prefill admission before terminal completion.
    async fn execute_turn(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult>;

    /// Drain backend-owned execution resources after all dispatched turns have
    /// reached terminal. In-process direct execution owns no extra resources;
    /// thread pools and remote clients override this lifecycle hook.
    fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Construction policy for one online HTTP sink.
///
/// The client config owns Clock-enforced transport deadlines and protocol
/// selection; reuse and affinity remain per-request policies applied when a
/// materialized turn becomes a [`RequestConfig`].
#[derive(Clone, Debug, Default)]
pub struct TransportSinkConfig {
    /// Low-level HTTP client policy.
    pub client: ClientConfig,
    /// Connection pooling/lease strategy.
    pub connection_reuse: ConnectionReuseStrategy,
    /// Optional replacement for the default `X-Correlation-ID` header.
    pub session_header: Option<String>,
}

/// Response-capturing request-dispatch seam used by the shared paced issuer.
///
/// The online implementation is [`TransportSink`]; the optional in-process
/// simulator implements the same contract, so pacing, admission, adaptive
/// control, observers, and report construction do not branch on a backend.
#[async_trait(?Send)]
pub trait HttpRequestDispatcher: RequestSink<HttpRequest> {
    /// Resolve report dimensions using the same endpoint selection as dispatch.
    fn inference_dimensions(&self, _request: &HttpRequest) -> InferenceDimensions {
        InferenceDimensions::default()
    }

    /// Dispatch one request, retain its terminal response facts, and invoke
    /// `on_first_token` exactly once with TTFT in nanoseconds.
    async fn dispatch_collect(
        &self,
        req: HttpRequest,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpDispatchResult>;
}

impl Dispatchable for HttpRequest {
    fn uuid(&self) -> Uuid {
        self.uuid
    }
    fn input_length(&self) -> usize {
        self.input_length
    }
    fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }
}

/// Live OpenAI-chat sink over [`aiperf_transport_http`]. Shares the caller's clock and
/// origin (`start_ns`) so admit/token timestamps sit on the same timeline as the
/// run loop's arrival events.
pub struct TransportSink {
    transport: HttpTransport,
    clock: Rc<dyn Clock>,
    urls: Vec<String>,
    base_urls: Vec<String>,
    model: String,
    start_ns: Cell<i64>,
    connection_reuse: ConnectionReuseStrategy,
    prepared_endpoints: Option<Rc<PreparedEndpointTable>>,
}

impl TransportSink {
    /// Build a sink targeting `base_url` for `model`. When `http2` is set,
    /// cleartext HTTP/2 (h2c prior-knowledge) multiplexes many streams over one
    /// connection.
    pub fn new(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        base_url: &str,
        model: impl Into<String>,
        http2: bool,
    ) -> Self {
        Self::new_multi(clock, start_ns, &[base_url.to_string()], model, http2)
            .expect("a single base URL is a non-empty endpoint list")
    }

    /// Build a sink targeting an ordered, non-empty endpoint list. Request
    /// indices are resolved only at dispatch, so the issuer can pin a session's
    /// turn-0 selection across its continuation turns.
    pub fn new_multi(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        base_urls: &[String],
        model: impl Into<String>,
        http2: bool,
    ) -> Result<Self> {
        let client = ClientConfig {
            http_version: if http2 {
                HttpVersion::Http2PriorKnowledge
            } else {
                HttpVersion::Auto
            },
            ..ClientConfig::default()
        };
        Self::new_multi_configured(
            clock,
            start_ns,
            base_urls,
            model,
            TransportSinkConfig {
                client,
                ..TransportSinkConfig::default()
            },
        )
    }

    /// Build a sink with explicit deadline, reuse, protocol, and session-header
    /// policy supplied by a resolved benchmark configuration.
    pub fn new_multi_configured(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        base_urls: &[String],
        model: impl Into<String>,
        config: TransportSinkConfig,
    ) -> Result<Self> {
        if base_urls.is_empty() {
            anyhow::bail!("at least one base URL is required");
        }
        if base_urls.len() > u32::MAX as usize {
            anyhow::bail!("base URL count exceeds the u32 request-index representation");
        }
        if config
            .session_header
            .as_ref()
            .is_some_and(|header| header.trim().is_empty())
        {
            anyhow::bail!("session header must be non-empty when configured");
        }
        let mut transport = HttpTransport::new(clock.clone(), config.client);
        if let Some(header) = config.session_header {
            transport = transport.with_session_header(header);
        }
        let base_urls = base_urls
            .iter()
            .map(|base_url| base_url.trim_end_matches('/').to_string())
            .collect::<Vec<_>>();
        let urls = base_urls
            .iter()
            .map(|base_url| format!("{base_url}/v1/chat/completions"))
            .collect();
        Ok(Self {
            transport,
            clock,
            urls,
            base_urls,
            model: model.into(),
            start_ns: Cell::new(start_ns),
            connection_reuse: config.connection_reuse,
            prepared_endpoints: None,
        })
    }

    /// Install worker-local prepared endpoint bindings.
    ///
    /// Execution commands carry only dense keys and canonical IDs; endpoint
    /// configuration and credentials remain in this worker-owned table.
    pub fn with_prepared_endpoints(mut self, endpoints: Rc<PreparedEndpointTable>) -> Self {
        self.prepared_endpoints = Some(endpoints);
        self
    }

    fn ms(&self, ns: i64) -> f64 {
        (ns - self.start_ns.get()) as f64 / 1_000_000.0
    }

    /// Set the benchmark origin after execution resources have finished
    /// starting. This keeps backend startup outside phase timing.
    pub fn set_run_origin(&self, start_ns: i64) {
        self.start_ns.set(start_ns);
    }

    fn selected_url(&self, url_index: Option<u32>, endpoint_path: Option<&str>) -> Result<String> {
        let selected_index = url_index.unwrap_or(0) as usize;
        let selected_url = self.urls.get(selected_index).ok_or_else(|| {
            anyhow::anyhow!(
                "URL index {selected_index} is out of range for {} configured endpoints",
                self.urls.len()
            )
        })?;
        match endpoint_path {
            None => Ok(selected_url.clone()),
            Some(path) if path.starts_with('/') => Ok(format!(
                "{}{}",
                self.base_urls
                    .get(selected_index)
                    .expect("base/default URL vectors have equal length"),
                path
            )),
            Some(url) if url::Url::parse(url).is_ok() => Ok(url.to_string()),
            Some(value) => {
                anyhow::bail!("dataset endpoint target {value:?} must be an absolute path or URL")
            }
        }
    }

    /// Dispatch `req`, invoking `on_first_token` once when the transport observes
    /// TTFT. Request-rate scheduling uses this to release prefill capacity before
    /// the full stream reaches terminal.
    pub async fn dispatch_with_hooks(
        &self,
        req: HttpRequest,
        obs: &dyn RequestObserver,
        on_first_token: impl FnMut(i64),
    ) -> Result<()> {
        self.dispatch_collect_with_hooks(req, obs, on_first_token)
            .await
            .map(|_| ())
    }

    /// Dispatch and retain generated text plus authoritative usage for consumers
    /// such as the external accuracy response collector. Measurement events are identical to
    /// [`dispatch_with_hooks`](Self::dispatch_with_hooks).
    pub async fn dispatch_collect_with_hooks(
        &self,
        req: HttpRequest,
        obs: &dyn RequestObserver,
        on_first_token: impl FnMut(i64),
    ) -> Result<HttpDispatchResult> {
        self.dispatch_collect_record_with_hooks(req, obs, on_first_token)
            .await
            .map(|collected| collected.result)
    }

    async fn dispatch_collect_record_with_hooks(
        &self,
        req: HttpRequest,
        obs: &dyn RequestObserver,
        mut on_first_token: impl FnMut(i64),
    ) -> Result<HttpCollectedDispatch> {
        let HttpRequest {
            uuid,
            max_output_tokens,
            prompt_text,
            request_body,
            request_body_bytes,
            headers,
            parameters,
            endpoint_path,
            streaming,
            x_correlation_id,
            is_final_turn,
            cancel_after_ns,
            url_index,
            ..
        } = req;
        // No scheduler admission on the HTTP path; admit == dispatch time.
        let admit_ms = self.ms(self.clock.now_ns());
        obs.on_admit(uuid, admit_ms, 0);

        anyhow::ensure!(
            request_body.is_none() || request_body_bytes.is_none(),
            "an HTTP request cannot supply both JSON and serialized bodies"
        );
        let body = match request_body_bytes {
            Some(body) => body,
            None => {
                let payload = request_body.unwrap_or_else(|| {
                    let prompt = prompt_text.unwrap_or_default();
                    chat_request_body(&self.model, &[("user", prompt.as_str())], max_output_tokens)
                });
                Bytes::from(serde_json::to_vec(&payload)?)
            }
        };
        let request_payload = body.clone();

        let selected_url = self.selected_url(url_index, endpoint_path.as_deref())?;
        let mut cfg = RequestConfig::new(selected_url);
        cfg.headers = headers;
        cfg.params = parameters;
        cfg.correlation_id = x_correlation_id;
        cfg.request_id = Some(uuid.to_string());
        cfg.is_final_turn = is_final_turn;
        cfg.cancel_after_ns = cancel_after_ns;
        cfg.reuse = self.connection_reuse;
        let first_token_released = Cell::new(false);
        let rec = self
            .transport
            .send_request_bytes_with_first_token_filter(
                &cfg,
                body,
                streaming,
                |ttft_ns, message| {
                    if !is_meaningful_chat_token(message) {
                        return false;
                    }
                    if !first_token_released.replace(true) {
                        on_first_token(ttft_ns);
                    }
                    true
                },
            )
            .await;

        // Parse the collected SSE messages into per-token arrival times, stamped
        // from the transport clock (real inter-token timing).
        let mut done = false;
        let mut response_text = String::new();
        let mut model_response = ModelResponseMetadata::default();
        let mut prompt_tokens = None;
        let mut completion_tokens = None;
        for resp in &rec.responses {
            match resp {
                Response::Sse(msg) => {
                    if msg.is_done() {
                        done = true;
                        continue;
                    }
                    let Some(data) = msg.data() else { continue };
                    let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) else {
                        continue;
                    };
                    absorb_chat_chunk_metadata(&chunk, &mut model_response);
                    if let Some(usage) = &chunk.usage {
                        prompt_tokens = Some(usage.prompt_tokens);
                        completion_tokens = Some(usage.completion_tokens);
                        model_response.cached_prompt_tokens = usage.cached_tokens().map(u64::from);
                    }
                    let delta = chunk.delta_text();
                    if !delta.is_empty() {
                        response_text.push_str(&delta);
                        let kind = if chunk.has_output_delta() {
                            ObservedTokenKind::Output
                        } else {
                            ObservedTokenKind::Reasoning
                        };
                        obs.on_classified_token(uuid, self.ms(msg.perf_ns), kind);
                    }
                }
                Response::Text(response) => {
                    done = true;
                    if let Some(value) = response.json() {
                        let parsed = parse_non_streaming_response(&value);
                        response_text.push_str(&parsed.0);
                        prompt_tokens = parsed.1;
                        completion_tokens = parsed.2;
                        absorb_wire_response_metadata(&value, &mut model_response);
                        absorb_non_streaming_content(&value, &mut model_response);
                        if !response_text.is_empty() {
                            if !first_token_released.replace(true) {
                                on_first_token(response.perf_ns.saturating_sub(rec.start_ns));
                            }
                            obs.on_classified_token(
                                uuid,
                                self.ms(response.perf_ns),
                                ObservedTokenKind::Output,
                            );
                        }
                    }
                }
            }
        }

        let terminal = match rec.error.as_ref().map(|error| error.kind) {
            Some(ErrorKind::Cancelled) => ReplayTerminalStatus::Canceled,
            Some(_) => ReplayTerminalStatus::Failed,
            None if done && rec.status == Some(200) => ReplayTerminalStatus::Completed,
            None => ReplayTerminalStatus::Failed,
        };
        absorb_transport_error(
            rec.error.as_ref(),
            terminal,
            rec.status,
            &mut model_response,
        );
        obs.on_usage(
            uuid,
            ObservedUsage {
                prompt_tokens: prompt_tokens.map(|value| value as usize),
                completion_tokens: completion_tokens.map(|value| value as usize),
                ..ObservedUsage::default()
            },
        );
        obs.on_terminal(uuid, terminal);
        let http = {
            let mut http = rec
                .trace
                .as_ref()
                .map_or_else(HttpTrace::default, |trace| HttpTrace {
                    blocked_ns: trace.blocked(),
                    dns_lookup_ns: trace.dns_lookup(),
                    connecting_ns: trace.connecting(),
                    sending_ns: trace.sending(),
                    waiting_ns: trace.waiting(),
                    receiving_ns: trace.receiving(),
                    duration_ns: trace.duration(),
                    connection_reused: Some(trace.connection_reused_ns.is_some()),
                    data_sent_bytes: Some(trace.request_bytes_total),
                    data_received_bytes: Some(trace.response_bytes_total),
                    chunks_sent: Some(u64::from(trace.request_chunks_count)),
                    chunks_received: Some(u64::from(trace.response_chunks_count)),
                    ..HttpTrace::default()
                });
            http.stream_setup_ns = rec
                .recv_start_ns
                .map(|receive_start| receive_start.saturating_sub(rec.start_ns));
            http
        };
        let result = HttpDispatchResult {
            start_ns: rec.start_ns,
            end_ns: rec.end_ns.unwrap_or_else(|| self.clock.now_ns()),
            status: rec.status,
            terminal,
            response_text,
            model_response,
            prompt_tokens,
            completion_tokens,
            http,
        };
        Ok(HttpCollectedDispatch {
            result,
            request_payload,
            record: rec,
        })
    }
}

fn parse_non_streaming_response(value: &Value) -> (String, Option<u32>, Option<u32>) {
    let text = value
        .pointer("/choices/0/message/reasoning_content")
        .or_else(|| value.pointer("/choices/0/message/content"))
        .or_else(|| value.pointer("/choices/0/text"))
        .or_else(|| value.get("output_text"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            let output = value
                .get("output")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .flat_map(|item| {
                    item.get("content")
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                })
                .filter_map(|content| content.get("text").and_then(Value::as_str))
                .fold(String::new(), |mut output, text| {
                    output.push_str(text);
                    output
                });
            (!output.is_empty()).then_some(output)
        })
        .unwrap_or_default();
    let usage = value.get("usage");
    let prompt = usage
        .and_then(|usage| {
            usage
                .get("prompt_tokens")
                .or_else(|| usage.get("input_tokens"))
        })
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let completion = usage
        .and_then(|usage| {
            usage
                .get("completion_tokens")
                .or_else(|| usage.get("output_tokens"))
        })
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    (text, prompt, completion)
}

fn absorb_chat_chunk_metadata(chunk: &ChatChunk, metadata: &mut ModelResponseMetadata) {
    if let Some(response_id) = chunk.id.as_ref().filter(|value| !value.is_empty()) {
        metadata.response_id = Some(response_id.clone());
    }
    for choice in &chunk.choices {
        if let Some(content) = &choice.delta.content {
            append_optional_text(&mut metadata.content, content);
        }
        if let Some(reasoning) = &choice.delta.reasoning_content {
            metadata.content.get_or_insert_with(String::new);
            append_optional_text(&mut metadata.reasoning, reasoning);
        }
        if let Some(finish_reason) = choice
            .finish_reason
            .as_ref()
            .filter(|value| !value.is_empty())
        {
            metadata.finish_reason = Some(normalize_finish_reason(finish_reason));
        }
    }
}

fn absorb_non_streaming_content(value: &Value, metadata: &mut ModelResponseMetadata) {
    let reasoning = value
        .pointer("/choices/0/message/reasoning_content")
        .or_else(|| value.pointer("/choices/0/message/reasoning"))
        .and_then(Value::as_str);
    if let Some(reasoning) = reasoning {
        metadata.content.get_or_insert_with(String::new);
        append_optional_text(&mut metadata.reasoning, reasoning);
    }
    let content = value
        .pointer("/choices/0/message/content")
        .or_else(|| value.pointer("/choices/0/text"))
        .or_else(|| value.get("output_text"))
        .and_then(Value::as_str);
    if let Some(content) = content {
        append_optional_text(&mut metadata.content, content);
    }
}

pub(super) fn absorb_wire_response_metadata(value: &Value, metadata: &mut ModelResponseMetadata) {
    if let Some(response_id) = value
        .get("id")
        .or_else(|| value.pointer("/response/id"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        metadata.response_id = Some(response_id.to_string());
    }
    if let Some(finish_reason) = value
        .pointer("/choices/0/finish_reason")
        .or_else(|| value.pointer("/response/incomplete_details/reason"))
        .or_else(|| value.pointer("/incomplete_details/reason"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        metadata.finish_reason = Some(normalize_finish_reason(finish_reason));
    }
    let usage = value
        .get("usage")
        .or_else(|| value.pointer("/response/usage"));
    metadata.cached_prompt_tokens = usage
        .and_then(|usage| {
            usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .or_else(|| usage.pointer("/input_tokens_details/cached_tokens"))
                .or_else(|| usage.get("cache_read_input_tokens"))
        })
        .and_then(Value::as_u64)
        .or(metadata.cached_prompt_tokens);
}

pub(super) fn absorb_transport_error(
    error: Option<&ErrorDetails>,
    terminal: ReplayTerminalStatus,
    status: Option<u16>,
    metadata: &mut ModelResponseMetadata,
) {
    if terminal == ReplayTerminalStatus::Completed {
        metadata.error_kind = None;
        metadata.error_message = None;
        return;
    }
    if let Some(error) = error {
        metadata.error_kind = Some(error_kind_name(error.kind).to_string());
        metadata.error_message = Some(error.message.clone());
        return;
    }
    metadata.error_kind = Some(
        match terminal {
            ReplayTerminalStatus::Canceled => "cancelled",
            ReplayTerminalStatus::Rejected => "dispatch_rejected",
            ReplayTerminalStatus::Failed if status.is_some() => "http_error",
            ReplayTerminalStatus::Failed => "incomplete_response",
            ReplayTerminalStatus::Completed => unreachable!("handled above"),
        }
        .to_string(),
    );
    metadata.error_message = Some(status.map_or_else(
        || format!("request reached terminal status {terminal:?}"),
        |status| format!("request reached terminal HTTP status {status}"),
    ));
}

fn append_optional_text(target: &mut Option<String>, text: &str) {
    target.get_or_insert_with(String::new).push_str(text);
}

fn normalize_finish_reason(value: &str) -> String {
    match value {
        "max_output_tokens" | "max_tokens" => "length".to_string(),
        value => value.to_string(),
    }
}

fn error_kind_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Http => "http_error",
        ErrorKind::Sse => "sse_error",
        ErrorKind::Cancelled => "cancelled",
        ErrorKind::Connect => "connect_error",
        ErrorKind::Timeout => "timeout",
        ErrorKind::Other => "transport_error",
    }
}

#[async_trait(?Send)]
impl RequestSink<HttpRequest> for TransportSink {
    async fn dispatch(&self, req: HttpRequest, obs: &dyn RequestObserver) -> Result<()> {
        self.dispatch_with_hooks(req, obs, |_ttft_ns| {}).await
    }
}

#[async_trait(?Send)]
impl HttpRequestDispatcher for TransportSink {
    fn inference_dimensions(&self, request: &HttpRequest) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: self
                .selected_url(request.url_index, request.endpoint_path.as_deref())
                .ok(),
            model: Some(self.model.clone()),
        }
    }

    async fn dispatch_collect(
        &self,
        req: HttpRequest,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpDispatchResult> {
        self.dispatch_collect_with_hooks(req, observer, on_first_token)
            .await
    }
}

#[async_trait(?Send)]
impl TurnDispatcher for TransportSink {
    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: self
                .selected_url(turn.url_index, turn.endpoint_path.as_deref())
                .ok(),
            model: turn
                .effective_model
                .clone()
                .or_else(|| Some(self.model.clone())),
        }
    }

    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        Ok(self
            .dispatch_turn_collect_record(turn, observer, on_first_token)
            .await?
            .outcome)
    }
}

#[async_trait(?Send)]
impl HttpTurnExecutionBackend for TransportSink {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        TransportSink::set_run_origin(self, start_ns);
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <Self as TurnDispatcher>::inference_dimensions(self, turn)
    }

    async fn execute_turn(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        self.dispatch_prepared_turn_collect_record(turn, observer, on_first_token)
            .await
    }
}

impl TransportSink {
    /// Dispatch one scheduled turn while retaining the exact HTTP exchange.
    ///
    /// This is the raw-artifact counterpart to [`TurnDispatcher::dispatch_turn`]
    /// and intentionally remains a concrete HTTP method. Scheduling code sees
    /// only the backend-neutral outcome, while the subprocess runner can retain
    /// transport facts without rebuilding endpoint formatting or parsing.
    pub async fn dispatch_turn_collect_record(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        let turn = PreparedHttpTurn::from_turn(turn, &self.model);
        self.dispatch_prepared_turn_collect_record(turn, observer, on_first_token)
            .await
    }

    /// Execute an owned scheduler-free HTTP command and retain the exact wire
    /// exchange. Execution-placement adapters use this method on their local
    /// worker reactor while the ordinary direct path calls it in place.
    pub async fn dispatch_prepared_turn_collect_record(
        &self,
        turn: PreparedHttpTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpTurnDispatchResult> {
        let PreparedHttpTurn {
            request,
            endpoint,
            endpoint_aware,
        } = turn;
        let collected = if endpoint_aware {
            match endpoint {
                PreparedHttpEndpoint::Legacy { endpoint, config } => {
                    self.dispatch_endpoint_collect_record_with_hooks(
                        request,
                        endpoint.as_ref(),
                        &config,
                        observer,
                        on_first_token,
                    )
                    .await?
                }
                PreparedHttpEndpoint::Prepared(reference) => {
                    let table = self.prepared_endpoints.as_ref().ok_or_else(|| {
                        anyhow::anyhow!(
                            "HTTP worker received prepared endpoint key {} without a prepared table",
                            reference.key.index()
                        )
                    })?;
                    let endpoint = table.get(reference.key)?;
                    anyhow::ensure!(
                        endpoint.descriptor().id == reference.endpoint_id.as_str(),
                        "prepared endpoint key {} resolved to {:?}, expected {:?}",
                        reference.key.index(),
                        endpoint.descriptor().id,
                        reference.endpoint_id.as_str()
                    );
                    self.dispatch_prepared_endpoint_collect_record_with_hooks(
                        request,
                        endpoint,
                        observer,
                        on_first_token,
                    )
                    .await?
                }
            }
        } else {
            self.dispatch_collect_record_with_hooks(request, observer, on_first_token)
                .await?
        };
        let HttpCollectedDispatch {
            result,
            request_payload,
            record,
        } = collected;
        let HttpDispatchResult {
            start_ns,
            end_ns,
            terminal,
            response_text,
            model_response,
            prompt_tokens,
            completion_tokens,
            http,
            ..
        } = result;
        Ok(HttpTurnDispatchResult {
            outcome: TurnDispatchOutcome {
                start_ns,
                end_ns,
                terminal,
                response_text,
                model_response,
                prompt_tokens: prompt_tokens.map(u64::from),
                completion_tokens: completion_tokens.map(u64::from),
                http,
            },
            request_payload,
            record,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use aiperf_clock::RealClock;
    use aiperf_dataset::BuiltinEndpointResolver;
    use aiperf_endpoints::EndpointType;

    #[derive(Default)]
    struct RecordingObserver {
        tokens: std::sync::Mutex<Vec<f64>>,
        usage: std::sync::Mutex<Vec<ObservedUsage>>,
    }

    impl RequestObserver for RecordingObserver {
        fn on_arrival(
            &self,
            _uuid: Uuid,
            _arrival_ms: f64,
            _input_length: usize,
            _requested_output_length: usize,
        ) {
        }

        fn on_admit(&self, _uuid: Uuid, _admit_ms: f64, _reused_input_tokens: usize) {}

        fn on_token(&self, _uuid: Uuid, at_ms: f64) {
            self.tokens.lock().unwrap().push(at_ms);
        }

        fn on_usage(&self, _uuid: Uuid, usage: ObservedUsage) {
            self.usage.lock().unwrap().push(usage);
        }

        fn on_terminal(&self, _uuid: Uuid, _status: ReplayTerminalStatus) {}
    }

    #[test]
    fn first_token_filter_skips_non_content_sse_messages() {
        let role = SseMessage::parse(r#"data: {"choices":[{"delta":{"role":"assistant"}}]}"#, 1);
        let usage = SseMessage::parse(r#"data: {"choices":[],"usage":{"completion_tokens":1}}"#, 2);
        let content = SseMessage::parse(r#"data: {"choices":[{"delta":{"content":"hello"}}]}"#, 3);
        let reasoning = SseMessage::parse(
            r#"data: {"choices":[{"delta":{"reasoning_content":"think"}}]}"#,
            4,
        );
        assert!(!is_meaningful_chat_token(&role));
        assert!(!is_meaningful_chat_token(&usage));
        assert!(is_meaningful_chat_token(&content));
        assert!(is_meaningful_chat_token(&reasoning));
    }

    #[test]
    fn prepared_turn_is_send_between_reactor_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<PreparedHttpTurn>();
    }

    #[test]
    fn prepared_turn_wire_round_trips_and_redacts_debug_output() {
        let request_secret = "request-secret";
        let endpoint_secret = "endpoint-secret";
        let mut endpoint_config = EndpointConfig {
            endpoint_type: EndpointType::Messages,
            streaming: true,
            api_key: Some(endpoint_secret.into()),
            ..EndpointConfig::default()
        };
        endpoint_config
            .headers
            .insert("anthropic-beta".into(), "secret-beta".into());
        let prepared = PreparedHttpTurn {
            request: HttpRequest {
                uuid: Uuid::from_u128(42),
                input_length: 7,
                max_output_tokens: 11,
                prompt_text: None,
                request_body: None,
                request_body_bytes: Some(Bytes::from_static(br#"{"messages":[]}"#)),
                headers: BTreeMap::from([("x-api-key".into(), request_secret.into())]),
                parameters: BTreeMap::from([("version".into(), "1".into())]),
                endpoint_path: Some("/v1/messages".into()),
                streaming: true,
                x_correlation_id: Some("session-1".into()),
                is_final_turn: true,
                cancel_after_ns: Some(9),
                url_index: Some(2),
            },
            endpoint: PreparedHttpEndpoint::Legacy {
                endpoint: BuiltinEndpointResolver::default()
                    .resolve_type(EndpointType::Messages)
                    .unwrap(),
                config: Box::new(endpoint_config),
            },
            endpoint_aware: true,
        };

        let wire = prepared.into_wire();
        let debug = format!("{wire:?}");
        assert!(!debug.contains(request_secret));
        assert!(!debug.contains(endpoint_secret));
        assert!(!debug.contains("secret-beta"));

        let encoded = serde_json::to_vec(&wire).unwrap();
        let decoded: PreparedHttpTurnWire = serde_json::from_slice(&encoded).unwrap();
        let rehydrated = decoded
            .into_prepared(&BuiltinEndpointResolver::default())
            .unwrap();
        assert_eq!(rehydrated.request.uuid, Uuid::from_u128(42));
        assert_eq!(
            rehydrated.request.request_body_bytes.as_deref(),
            Some(br#"{"messages":[]}"#.as_slice())
        );
        assert_eq!(rehydrated.request.headers["x-api-key"], request_secret);
        let PreparedHttpEndpoint::Legacy { endpoint, config } = rehydrated.endpoint else {
            panic!("legacy wire must rehydrate a legacy endpoint")
        };
        assert_eq!(config.api_key.as_deref(), Some(endpoint_secret));
        assert_eq!(config.headers["anthropic-beta"], "secret-beta");
        assert_eq!(endpoint.metadata().endpoint_type, EndpointType::Messages);
        assert!(rehydrated.endpoint_aware);
    }

    #[test]
    fn prepared_turn_wire_rejects_unknown_versions() {
        let wire = PreparedHttpTurnWire {
            version: HTTP_EXECUTION_COMMAND_VERSION + 1,
            request: HttpRequestWire {
                uuid: Uuid::nil(),
                input_length: 1,
                max_output_tokens: 1,
                prompt_text: None,
                request_body: None,
                request_body_bytes: None,
                headers: BTreeMap::new(),
                parameters: BTreeMap::new(),
                endpoint_path: None,
                streaming: true,
                x_correlation_id: None,
                is_final_turn: true,
                cancel_after_ns: None,
                url_index: None,
            },
            endpoint: PreparedHttpEndpointWire::Legacy(Box::new(EndpointConfig::default())),
            endpoint_headers: BTreeMap::new(),
            endpoint_api_key: None,
            endpoint_aware: false,
        };
        let error = wire
            .into_prepared(&BuiltinEndpointResolver::default())
            .unwrap_err();
        assert!(error.to_string().contains("version"));
    }

    #[test]
    fn prepared_turn_wire_preserves_open_endpoint_identity_and_dense_key() {
        let reference = PreparedEndpointReference {
            key: EndpointKey::from_index(7),
            endpoint_id: EndpointId::new("chat").unwrap(),
        };
        let wire = PreparedHttpTurnWire {
            version: HTTP_EXECUTION_COMMAND_VERSION,
            request: HttpRequestWire {
                uuid: Uuid::nil(),
                input_length: 1,
                max_output_tokens: 1,
                prompt_text: None,
                request_body: None,
                request_body_bytes: None,
                headers: BTreeMap::new(),
                parameters: BTreeMap::new(),
                endpoint_path: None,
                streaming: true,
                x_correlation_id: None,
                is_final_turn: true,
                cancel_after_ns: None,
                url_index: None,
            },
            endpoint: PreparedHttpEndpointWire::Prepared(reference),
            endpoint_headers: BTreeMap::new(),
            endpoint_api_key: None,
            endpoint_aware: true,
        };
        let encoded = serde_json::to_vec(&wire).unwrap();
        let decoded: PreparedHttpTurnWire = serde_json::from_slice(&encoded).unwrap();
        let prepared = decoded
            .into_prepared(&BuiltinEndpointResolver::default())
            .unwrap();
        let PreparedHttpEndpoint::Prepared(reference) = prepared.endpoint else {
            panic!("prepared endpoint wire must not resolve through a legacy adapter")
        };
        assert_eq!(reference.key.index(), 7);
        assert_eq!(reference.endpoint_id.as_str(), "chat");
    }

    #[tokio::test]
    async fn transport_retries_first_token_filter_past_role_only_chunk() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let clock = RealClock::new();
                let transport = HttpTransport::new(clock, ClientConfig::default());
                let cfg = RequestConfig::new(format!(
                    "{}/v1/chat/completions",
                    base.trim_end_matches('/')
                ));
                let attempts = Cell::new(0);
                let record = transport
                    .send_request_with_first_token_filter(
                        &cfg,
                        chat_request_body("m", &[("user", "hello")], 1),
                        true,
                        |_ttft_ns, message| {
                            attempts.set(attempts.get() + 1);
                            is_meaningful_chat_token(message)
                        },
                    )
                    .await;
                assert!(!record.has_error(), "unexpected error: {:?}", record.error);
                assert_eq!(
                    attempts.get(),
                    2,
                    "role-only chunk must not release prefill"
                );
            })
            .await;
    }

    #[tokio::test]
    async fn dispatch_invokes_first_token_hook_once() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let clock = RealClock::new();
                let sink = TransportSink::new(clock.clone(), clock.now_ns(), &base, "m", false);
                let hook_calls = Rc::new(Cell::new(0));
                let hook_calls_for_hook = hook_calls.clone();
                let req = HttpRequest {
                    uuid: Uuid::new_v4(),
                    input_length: 4,
                    max_output_tokens: 2,
                    prompt_text: Some("hello world".to_string()),
                    request_body: None,
                    request_body_bytes: None,
                    headers: BTreeMap::new(),
                    parameters: BTreeMap::new(),
                    endpoint_path: None,
                    streaming: true,
                    x_correlation_id: None,
                    is_final_turn: true,
                    cancel_after_ns: None,
                    url_index: None,
                };

                let observer = RecordingObserver::default();
                let first_token_ns = Rc::new(Cell::new(None));
                let first_token_ns_for_hook = first_token_ns.clone();

                let result = sink.dispatch_collect_with_hooks(req, &observer, |ttft_ns| {
                    hook_calls_for_hook.set(hook_calls_for_hook.get() + 1);
                    first_token_ns_for_hook.set(Some(ttft_ns));
                })
                .await
                .unwrap();

                assert_eq!(hook_calls.get(), 1);
                assert_eq!(
                    observer.usage.lock().unwrap().as_slice(),
                    &[ObservedUsage {
                        prompt_tokens: result.prompt_tokens.map(|value| value as usize),
                        completion_tokens: result.completion_tokens.map(|value| value as usize),
                        ..ObservedUsage::default()
                    }]
                );
                let first_observed_token_ms = observer.tokens.lock().unwrap()[0];
                let first_hook_ms = first_token_ns.get().unwrap() as f64 / 1_000_000.0;
                let dispatch_start_ms = sink.ms(result.start_ns);
                assert!(
                    ((first_observed_token_ms - dispatch_start_ms) - first_hook_ms).abs()
                        < 0.1,
                    "hook TTFT {first_hook_ms:.6}ms must match first observed token at {:.6}ms from dispatch",
                    first_observed_token_ms - dispatch_start_ms,
                );
            })
            .await;
    }
}
