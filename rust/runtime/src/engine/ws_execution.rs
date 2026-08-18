// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local native WebSocket execution for Responses and Realtime endpoints.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::future::poll_fn;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use url::Url;

use crate::body_plan::{PreparedWsMessageRole, PreparedWsOpcode, PreparedWsOperation, RequestBody};
use crate::clock::Clock;
use crate::config::model::transport::{WebSocketFallback, WebSocketTransportConfig};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{
    ObservedTokenKind, ObservedTransportRoute, RequestObserver, TransportFallbackReason,
    TransportRoute,
};
use crate::engine::protocol::HopRouting;
use crate::engine::protocol_v2::AuthoredRunSpecV2;
use crate::engine::registry::{NativeTransportExecution, RunContext};
use crate::engine::turn_execution::{
    ExecutionBackendConfig, ExecutionSinkBuilder, PreparedEndpointTableFactory,
    RequestExecutorFactory, WorkerSink, build_native,
};
use crate::metrics::NativeMetricsObserver;
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RequestTrace};
use crate::multiturn::TurnToSend;
use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome, TurnResponseObserver};
use crate::transport::core::{
    ConnectionReuseStrategy, DispatchResult, MeasuredContext, MeasuredOutcome,
    PreparedEndpointBinding, PreparedTurn, RequestExecutor, RequestRecord, Response, TextResponse,
};
use crate::transport::measure::{self, WorkerMeasurement};
use crate::transport::ws::connector::{ConnectFailure, WebSocket, connect};
use crate::transport::ws::dialect::{
    ResponsesEvent, TurnOperationState, classify_realtime_event, classify_responses_event,
    full_history_retry, with_previous_response_id,
};
use crate::transport::ws::driver::{
    ApplicationQueueLimits, DriverEvent, DriverTiming, FallbackReason, SocketOperationDriver,
    WsDriverError,
};

const SOCKET_ROTATION_NS: i64 = 55 * 60 * 1_000_000_000;
const CLOSE_HANDSHAKE_NS: i64 = 1_000_000_000;

struct CachedSocket {
    affinity_key: Option<String>,
    continuation_id: Option<String>,
    url: String,
    headers: BTreeMap<String, String>,
    connected_ns: i64,
    retained_ns: i64,
    socket: WebSocket,
}

struct CheckedOutSocket {
    socket: WebSocket,
    connected_ns: i64,
    continuation_id: Option<String>,
}

struct AffinityGateCleanup<'a> {
    gates: &'a RefCell<BTreeMap<String, Rc<Semaphore>>>,
    affinity_key: &'a str,
    gate: Rc<Semaphore>,
}

impl Drop for AffinityGateCleanup<'_> {
    fn drop(&mut self) {
        let mut gates = self.gates.borrow_mut();
        let is_current = gates
            .get(self.affinity_key)
            .is_some_and(|current| Rc::ptr_eq(current, &self.gate));
        // Map + dispatch local + cleanup guard. Any waiter owns another handle.
        if is_current && Rc::strong_count(&self.gate) == 3 {
            gates.remove(self.affinity_key);
        }
    }
}

/// Native WebSocket execution factory with immutable per-run policy.
#[derive(Clone, Debug)]
pub struct WebSocketExecutionFactory {
    config: WebSocketTransportConfig,
}

/// Registry execution binding for native WebSocket endpoint dialects.
#[derive(Clone, Debug)]
pub struct WebSocketNativeExecution {
    config: WebSocketTransportConfig,
}

impl WebSocketNativeExecution {
    /// Construct one binding from the validated registered config.
    pub fn new(config: WebSocketTransportConfig) -> Self {
        Self { config }
    }
}

impl NativeTransportExecution for WebSocketNativeExecution {
    fn executor_factory(&self) -> Arc<dyn RequestExecutorFactory> {
        Arc::new(WebSocketExecutionFactory {
            config: self.config.clone(),
        })
    }

    fn request_materializer(&self) -> Arc<dyn crate::dataset::RequestMaterializer> {
        Arc::new(crate::dataset::WsRequestMaterializer)
    }

    fn readiness_enabled(&self) -> bool {
        false
    }

    fn build_graph_dispatcher(
        &self,
        _clock: Rc<dyn Clock>,
        _run_origin_ns: i64,
        _urls: &[String],
        _model: &str,
        _transport_config: crate::transport::http::TransportSinkConfig,
        _endpoints: Rc<crate::endpoints::PreparedEndpointTable>,
        _capture_raw: bool,
    ) -> Result<Rc<dyn crate::transport::core::Dispatcher>> {
        anyhow::bail!("websocket transport does not support graph execution")
    }

    fn graph_transport_label(&self) -> &'static str {
        "websocket"
    }

    fn validate_run(&self, run: &AuthoredRunSpecV2, context: &RunContext) -> Result<()> {
        ensure!(
            !run.artifacts.trace,
            "websocket execution does not support HTTP trace artifacts"
        );
        ensure!(
            run.sidecars.content_server.is_none()
                && run.sidecars.gpu_telemetry.is_none()
                && run.sidecars.network_latency.is_none()
                && run.sidecars.server_metrics.is_none()
                && run.sidecars.live_streaming.is_none(),
            "websocket execution has no registered sidecar adapter"
        );
        for (profile_id, profile) in context.endpoint_profiles() {
            ensure!(
                matches!(profile.endpoint_id.as_str(), "responses" | "realtime"),
                "websocket endpoint profile {profile_id:?} requires endpoint type responses or realtime"
            );
            ensure!(
                self.config.fallback == WebSocketFallback::Disabled
                    || profile.endpoint_id.as_str() == "responses",
                "websocket HTTP/SSE fallback is supported only by the responses endpoint"
            );
            ensure!(
                !profile.config.urls.is_empty(),
                "websocket endpoint profile {profile_id:?} has no URL"
            );
            for url in &profile.config.urls {
                let parsed = Url::parse(url).with_context(|| {
                    format!("parsing websocket endpoint profile {profile_id:?} URL")
                })?;
                ensure!(
                    matches!(parsed.scheme(), "ws" | "wss"),
                    "websocket endpoint profile {profile_id:?} requires ws:// or wss:// URLs, got {url:?}"
                );
            }
            ensure!(
                profile.client.uds_path.is_none(),
                "websocket endpoint profile {profile_id:?} configures unsupported UDS transport"
            );
            ensure!(
                profile.config.wait_for_model_timeout <= 0.0,
                "websocket endpoint profile {profile_id:?} enables unsupported readiness retries"
            );
        }
        Ok(())
    }

    fn run_metadata(&self) -> BTreeMap<String, String> {
        BTreeMap::from([("transport".to_owned(), "websocket".to_owned())])
    }
}

impl RequestExecutorFactory for WebSocketExecutionFactory {
    fn build(&self, config: ExecutionBackendConfig) -> Result<Rc<dyn RequestExecutor>> {
        ensure!(
            config.prepared_endpoints.is_some(),
            "native websocket execution requires prepared endpoints"
        );
        let workers = config.workers;
        let clock = config.coordinator_clock.clone();
        let anchor = config.real_clock_anchor;
        let hop_routing = HopRouting::Sticky;
        let labels = config.worker_labels.clone();
        build_native(
            WebSocketSinkBuilder::from_config(&config, self.config.clone())?,
            workers,
            clock,
            anchor,
            hop_routing,
            labels,
        )
    }
}

struct WebSocketSinkBuilder {
    base_urls: Vec<String>,
    model: String,
    transport: crate::transport::http::TransportSinkConfig,
    endpoints: Arc<dyn PreparedEndpointTableFactory>,
    config: WebSocketTransportConfig,
    capture_raw: bool,
}

impl WebSocketSinkBuilder {
    fn from_config(
        config: &ExecutionBackendConfig,
        websocket: WebSocketTransportConfig,
    ) -> Result<Self> {
        Ok(Self {
            base_urls: config.base_urls.clone(),
            model: config.model.clone(),
            transport: config.transport.clone(),
            endpoints: config
                .prepared_endpoints
                .clone()
                .ok_or_else(|| anyhow!("native websocket execution requires prepared endpoints"))?,
            config: websocket,
            capture_raw: config.raw_enabled,
        })
    }
}

impl ExecutionSinkBuilder for WebSocketSinkBuilder {
    type Sink = WebSocketTransportSink;

    fn label(&self) -> &'static str {
        "websocket"
    }

    fn build_sink(&self, clock: Rc<dyn Clock>, _worker_id: usize) -> Result<Self::Sink> {
        WebSocketTransportSink::new(
            clock,
            self.base_urls.clone(),
            self.model.clone(),
            self.transport.clone(),
            self.config.clone(),
            self.endpoints.prepare_worker()?,
            self.capture_raw,
        )
    }
}

/// Worker-local pool. A semaphore bounds concurrent operations; a checked-out
/// socket is absent from `idle` and therefore carries exactly one operation.
struct WebSocketTransportSink {
    clock: Rc<dyn Clock>,
    base_urls: Vec<String>,
    model: String,
    transport: crate::transport::http::TransportSinkConfig,
    config: WebSocketTransportConfig,
    endpoints: Rc<crate::endpoints::PreparedEndpointTable>,
    idle: RefCell<Vec<CachedSocket>>,
    affinity_gates: RefCell<BTreeMap<String, Rc<Semaphore>>>,
    slots: Semaphore,
    fallback: Option<crate::transport::http::TransportSink>,
    start_ns: Cell<i64>,
    measurement: WorkerMeasurement,
    capture_raw: bool,
}

impl WebSocketTransportSink {
    fn new(
        clock: Rc<dyn Clock>,
        base_urls: Vec<String>,
        model: String,
        transport: crate::transport::http::TransportSinkConfig,
        config: WebSocketTransportConfig,
        endpoints: crate::endpoints::PreparedEndpointTable,
        capture_raw: bool,
    ) -> Result<Self> {
        ensure!(!base_urls.is_empty(), "websocket execution requires a URL");
        ensure!(
            transport.client.max_connections_per_origin > 0,
            "websocket connection limit must be positive"
        );
        let endpoints = Rc::new(endpoints);
        let fallback = if config.fallback == WebSocketFallback::HttpSse {
            let fallback_urls = base_urls
                .iter()
                .map(|url| websocket_fallback_url(url))
                .collect::<Result<Vec<_>>>()?;
            Some(
                crate::transport::http::TransportSink::new_multi_configured(
                    clock.clone(),
                    0,
                    &fallback_urls,
                    model.clone(),
                    transport.clone(),
                )?
                .with_prepared_endpoints(endpoints.clone()),
            )
        } else {
            None
        };
        let slot_count = transport.client.max_connections_per_origin;
        Ok(Self {
            clock,
            base_urls,
            model,
            transport,
            config,
            endpoints,
            idle: RefCell::new(Vec::with_capacity(slot_count)),
            affinity_gates: RefCell::new(BTreeMap::new()),
            slots: Semaphore::new(slot_count),
            fallback,
            start_ns: Cell::new(0),
            measurement: WorkerMeasurement::default(),
            capture_raw,
        })
    }

    fn selected_url(
        &self,
        index: Option<u32>,
        endpoint_path: Option<&str>,
        parameters: &BTreeMap<String, String>,
    ) -> Result<Url> {
        let selected_index = index.unwrap_or(0) as usize;
        let base = self.base_urls.get(selected_index).ok_or_else(|| {
            anyhow!(
                "websocket URL index {selected_index} is outside {} configured URLs",
                self.base_urls.len()
            )
        })?;
        let mut selected = match endpoint_path {
            None => Url::parse(base),
            Some(path) if path.starts_with('/') => {
                let mut selected = Url::parse(base)?;
                let base_path = selected.path().trim_end_matches('/');
                let path = path.replace("{model_name}", &self.model);
                let joined = crate::transport::http::transport::url::dedup_path_overlap(
                    base_path,
                    path.trim_start_matches('/'),
                );
                selected.set_path(&joined);
                Ok(selected)
            }
            Some(url) if Url::parse(url).is_ok() => Url::parse(url),
            Some(value) => {
                anyhow::bail!("websocket endpoint target {value:?} must be an absolute path or URL")
            }
        }
        .with_context(|| format!("parsing websocket endpoint URL for base {base:?}"))?;
        ensure!(
            matches!(selected.scheme(), "ws" | "wss"),
            "websocket endpoint URL requires ws:// or wss://, got {selected:?}"
        );
        if !parameters.is_empty() {
            let mut merged = selected
                .query_pairs()
                .map(|(key, value)| (key.into_owned(), value.into_owned()))
                .collect::<BTreeMap<_, _>>();
            merged.extend(parameters.clone());
            selected.query_pairs_mut().clear().extend_pairs(merged);
        }
        Ok(selected)
    }

    fn websocket_config(&self) -> WebSocketConfig {
        WebSocketConfig::default()
            .max_message_size(Some(self.config.max_message_bytes))
            .max_frame_size(Some(self.config.max_frame_bytes))
            .write_buffer_size(
                self.config
                    .max_queued_bytes
                    .saturating_sub(1)
                    .min(128 * 1024),
            )
            .max_write_buffer_size(self.config.max_queued_bytes)
    }

    fn take_cached(
        &self,
        affinity_key: &str,
        url: &str,
        headers: &BTreeMap<String, String>,
        now_ns: i64,
    ) -> (Option<CachedSocket>, Vec<CachedSocket>) {
        let mut idle = self.idle.borrow_mut();
        let keepalive_ns = self.transport.client.keepalive_ns;
        let mut expired = Vec::new();
        let mut index = 0;
        while index < idle.len() {
            let is_expired = now_ns.saturating_sub(idle[index].connected_ns) >= SOCKET_ROTATION_NS
                || keepalive_ns
                    .is_some_and(|limit| now_ns.saturating_sub(idle[index].retained_ns) >= limit);
            if is_expired {
                expired.push(idle.swap_remove(index));
            } else {
                index += 1;
            }
        }
        let cached = if self.transport.connection_reuse == ConnectionReuseStrategy::Never {
            None
        } else {
            idle.iter()
                .position(|cached| {
                    cached.affinity_key.as_deref() == Some(affinity_key)
                        && cached.url == url
                        && cached.headers == *headers
                })
                .or_else(|| {
                    idle.iter().position(|cached| {
                        cached.affinity_key.is_none()
                            && cached.url == url
                            && cached.headers == *headers
                    })
                })
                .map(|index| idle.swap_remove(index))
        };
        if cached.is_none() && !idle.is_empty() {
            let least_recent = idle
                .iter()
                .enumerate()
                .min_by_key(|(_, cached)| cached.retained_ns)
                .map(|(index, _)| index)
                .unwrap_or(0);
            expired.push(idle.swap_remove(least_recent));
        }
        (cached, expired)
    }

    async fn checkout(
        &self,
        affinity_key: &str,
        url: &Url,
        headers: &BTreeMap<String, String>,
        deadline_ns: Option<i64>,
    ) -> Result<CheckedOutSocket, ConnectFailure> {
        let now_ns = self.clock.now_ns();
        let (cached, expired) = self.take_cached(affinity_key, url.as_str(), headers, now_ns);
        for cached in expired {
            close_socket(cached.socket, self.clock.clone(), deadline_ns).await;
        }
        if let Some(cached) = cached {
            return Ok(CheckedOutSocket {
                socket: cached.socket,
                connected_ns: cached.connected_ns,
                continuation_id: cached.continuation_id,
            });
        }
        let socket = connect(
            url,
            headers,
            &self.transport.client,
            self.websocket_config(),
            self.clock.clone(),
            deadline_ns,
        )
        .await?;
        Ok(CheckedOutSocket {
            socket,
            connected_ns: self.clock.now_ns(),
            continuation_id: None,
        })
    }

    fn affinity_gate(&self, affinity_key: &str) -> Rc<Semaphore> {
        self.affinity_gates
            .borrow_mut()
            .entry(affinity_key.to_owned())
            .or_insert_with(|| Rc::new(Semaphore::new(1)))
            .clone()
    }

    fn validate_endpoint(
        &self,
        turn: &PreparedTurn,
    ) -> Result<&dyn crate::endpoints::PreparedEndpoint> {
        let PreparedEndpointBinding::Prepared(reference) = &turn.endpoint;
        let endpoint = self.endpoints.get(reference.key)?;
        ensure!(
            endpoint.descriptor().id == reference.endpoint_id.as_str(),
            "prepared endpoint key {} resolved to {:?}, expected {:?}",
            reference.key.index(),
            endpoint.descriptor().id,
            reference.endpoint_id.as_str()
        );
        ensure!(
            matches!(endpoint.descriptor().id, "responses" | "realtime"),
            "websocket execution requires a websocket-capable endpoint, got {:?}",
            endpoint.descriptor().id
        );
        Ok(endpoint)
    }

    async fn dispatch_inner(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        let endpoint = self.validate_endpoint(&turn)?;
        let endpoint_id = endpoint.descriptor().id;
        let prepared_operation = match turn.request.body.as_ref() {
            Some(RequestBody::WebSocket(operation)) => operation.clone(),
            _ => anyhow::bail!("websocket execution requires a prepared websocket operation"),
        };
        let fallback_turn = (endpoint_id == "responses")
            .then(|| prepared_operation.http_sse_fallback_body())
            .flatten()
            .map(|body| {
                let mut fallback = turn.clone();
                fallback.request.body = Some(RequestBody::wire(body.clone()));
                fallback
            });
        let affinity_key = turn.runtime_session_id.clone();
        let request = turn.request;
        let url = self.selected_url(
            request.url_index,
            request.endpoint_path.as_deref(),
            &request.parameters,
        )?;
        let start_ns = self.clock.now_ns();
        observer.on_transport_route(
            request.uuid,
            ObservedTransportRoute {
                actual_route: TransportRoute::Websocket,
                fallback_reason: None,
            },
        );
        let deadline_ns = self
            .transport
            .client
            .total_timeout_ns
            .and_then(|timeout_ns| start_ns.checked_add(timeout_ns));
        let affinity_gate = self.affinity_gate(&affinity_key);
        let _affinity =
            acquire_before_deadline(affinity_gate.as_ref(), self.clock.clone(), deadline_ns)
                .await?;
        let _gate_cleanup = AffinityGateCleanup {
            gates: &self.affinity_gates,
            affinity_key: &affinity_key,
            gate: affinity_gate.clone(),
        };
        let _slot = acquire_before_deadline(&self.slots, self.clock.clone(), deadline_ns).await?;
        let mut record = RequestRecord::started(start_ns);
        record.request_headers = request.headers.clone();
        let checked_out = match self
            .checkout(&affinity_key, &url, &request.headers, deadline_ns)
            .await
        {
            Ok(value) => value,
            Err(error)
                if self.config.fallback == WebSocketFallback::HttpSse
                    && fallback_turn.is_some()
                    && error.fallback_reason().is_some() =>
            {
                let reason = error
                    .fallback_reason()
                    .ok_or_else(|| anyhow!("fallback failure lost its stable reason"))?;
                observer.on_transport_route(
                    request.uuid,
                    ObservedTransportRoute {
                        actual_route: TransportRoute::HttpSse,
                        fallback_reason: Some(transport_fallback_reason(reason)),
                    },
                );
                tracing::debug!(
                    component = "websocket",
                    fallback_reason = reason.as_str(),
                    error = %error,
                    "using declared HTTP/SSE fallback"
                );
                let fallback = self
                    .fallback
                    .as_ref()
                    .ok_or_else(|| anyhow!("websocket fallback sink was not prepared"))?;
                let turn =
                    fallback_turn.ok_or_else(|| anyhow!("fallback body was not prepared"))?;
                return dispatch_fallback_before_deadline(
                    fallback,
                    turn,
                    observer,
                    on_first_token,
                    self.clock.clone(),
                    deadline_ns,
                )
                .await;
            }
            Err(error) => return Err(error.into()),
        };
        let operation = if endpoint_id == "responses" {
            checked_out
                .continuation_id
                .as_deref()
                .map(|response_id| {
                    with_previous_response_id(&prepared_operation, response_id).ok_or_else(|| {
                        anyhow!(
                            "websocket continuation could not be injected into the prepared operation"
                        )
                    })
                })
                .transpose()?
                .unwrap_or_else(|| prepared_operation.as_ref().clone())
        } else {
            prepared_operation.as_ref().clone()
        };
        let attempt = self
            .run_attempt(
                checked_out.socket,
                endpoint,
                &operation,
                &request,
                &mut record,
                observer,
                on_first_token,
                responses,
                start_ns,
                deadline_ns,
                checked_out.connected_ns,
            )
            .await;
        let completed = match attempt {
            Ok(completed) => completed,
            Err(failure) if endpoint_id == "responses" && failure.can_retry => {
                let replay = full_history_retry(&prepared_operation).ok_or_else(|| {
                    anyhow!("websocket operation cannot rebuild a self-contained retry")
                })?;
                tracing::debug!(component = "websocket", error = %failure.error, "retrying websocket operation before visible output");
                let (socket, retry_connected_ns) = self
                    .checkout_fresh(&url, &request.headers, deadline_ns)
                    .await?;
                self.run_attempt(
                    socket,
                    endpoint,
                    &replay,
                    &request,
                    &mut record,
                    observer,
                    on_first_token,
                    responses,
                    start_ns,
                    deadline_ns,
                    retry_connected_ns,
                )
                .await
                .map_err(|failure| failure.error)?
            }
            Err(failure) => return Err(failure.error),
        };
        let end_ns = self.clock.now_ns();
        if endpoint_id == "responses" {
            observer
                .on_round_trip_metrics(request.uuid, completed.state.finish(completed.terminal));
        }
        observer.on_terminal(request.uuid, completed.terminal);
        let reusable = completed.terminal == ReplayTerminalStatus::Completed
            && self.transport.connection_reuse != ConnectionReuseStrategy::Never
            && (endpoint_id == "responses" || !request.is_final_turn);
        if reusable {
            self.idle.borrow_mut().push(CachedSocket {
                affinity_key: (!request.is_final_turn).then(|| affinity_key.clone()),
                continuation_id: (endpoint_id == "responses" && !request.is_final_turn)
                    .then(|| completed.response_id.clone())
                    .flatten(),
                url: url.to_string(),
                headers: request.headers.clone(),
                connected_ns: completed.connected_ns,
                retained_ns: end_ns,
                socket: completed.socket,
            });
        } else {
            close_socket(completed.socket, self.clock.clone(), deadline_ns).await;
        }
        record.end_ns = Some(end_ns);
        record.reusable_connection = reusable;
        Ok(DispatchResult {
            outcome: TurnDispatchOutcome {
                start_ns,
                end_ns,
                terminal: completed.terminal,
                response_text: completed.output.clone(),
                model_response: ModelResponseMetadata {
                    content: (!completed.output.is_empty()).then_some(completed.output),
                    response_id: completed.response_id,
                    error_kind: (completed.terminal != ReplayTerminalStatus::Completed)
                        .then_some("incomplete".to_owned()),
                    error_message: (completed.terminal != ReplayTerminalStatus::Completed)
                        .then_some("Responses operation ended incomplete".to_owned()),
                    ..ModelResponseMetadata::default()
                },
                prompt_tokens: None,
                completion_tokens: None,
                http: RequestTrace {
                    duration_ns: end_ns.checked_sub(start_ns),
                    data_received_bytes: Some(completed.response_bytes as u64),
                    ..RequestTrace::default()
                },
            },
            request_payload: if self.capture_raw {
                capture_operation(&operation)?
            } else {
                Bytes::new()
            },
            record,
        })
    }

    async fn checkout_fresh(
        &self,
        url: &Url,
        headers: &BTreeMap<String, String>,
        deadline_ns: Option<i64>,
    ) -> Result<(WebSocket, i64)> {
        let socket = connect(
            url,
            headers,
            &self.transport.client,
            self.websocket_config(),
            self.clock.clone(),
            deadline_ns,
        )
        .await?;
        Ok((socket, self.clock.now_ns()))
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_attempt(
        &self,
        socket: WebSocket,
        endpoint: &dyn crate::endpoints::PreparedEndpoint,
        operation: &PreparedWsOperation,
        request: &crate::transport::core::Request,
        record: &mut RequestRecord,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
        start_ns: i64,
        deadline_ns: Option<i64>,
        connected_ns: i64,
    ) -> std::result::Result<CompletedAttempt, AttemptFailure> {
        let endpoint_id = endpoint.descriptor().id;
        let timing = DriverTiming {
            deadline_ns,
            // Checkout rotates aged idle sockets; an active operation is never
            // aborted merely because the connection crosses the age boundary.
            rotation_deadline_ns: i64::MAX,
            ping_interval_ns: seconds_to_ns(self.config.ping_interval_seconds)
                .map_err(AttemptFailure::terminal)?,
            stream_idle_timeout_ns: seconds_to_ns(self.config.stream_idle_timeout_seconds)
                .map_err(AttemptFailure::terminal)?,
            cancel_after_ns: request.cancel_after_ns,
        };
        let mut driver = SocketOperationDriver::start(
            socket,
            self.clock.clone(),
            operation,
            ApplicationQueueLimits::new(
                self.config.max_queued_commands,
                self.config.max_queued_bytes,
            )
            .with_max_frame_bytes(self.config.max_frame_bytes),
            timing,
            self.config.max_response_bytes,
        )
        .map_err(|error| AttemptFailure::terminal(error.into()))?;
        let mut state = TurnOperationState::default();
        let mut output = String::new();
        let mut response_id = None;
        let mut response_bytes = 0_usize;
        let mut has_first_token = false;
        let mut terminal = ReplayTerminalStatus::Failed;
        loop {
            let event = driver
                .next()
                .await
                .map_err(|error| AttemptFailure::from_driver(error, &state))?;
            match event {
                DriverEvent::Flushed { role, timestamp_ns }
                    if role == PreparedWsMessageRole::MeasuredInput =>
                {
                    state.on_send(timestamp_ns);
                }
                DriverEvent::Flushed { .. } => {}
                DriverEvent::Application {
                    payload,
                    is_text,
                    timestamp_ns,
                } => {
                    response_bytes =
                        response_bytes.checked_add(payload.len()).ok_or_else(|| {
                            AttemptFailure {
                                error: anyhow!("websocket response byte count overflowed"),
                                can_retry: false,
                            }
                        })?;
                    record.recv_start_ns.get_or_insert(timestamp_ns);
                    let parsed_json = if is_text && responses.is_some() {
                        Some(
                            serde_json::from_slice(&payload).map_err(|error| AttemptFailure {
                                error: anyhow!(
                                    "websocket application message was not JSON: {error}"
                                ),
                                can_retry: false,
                            })?,
                        )
                    } else {
                        None
                    };
                    if self.capture_raw && is_text {
                        let text =
                            std::str::from_utf8(&payload).map_err(|error| AttemptFailure {
                                error: anyhow!("websocket text frame was not UTF-8: {error}"),
                                can_retry: false,
                            })?;
                        record.responses.push(Response::Text(TextResponse {
                            perf_ns: timestamp_ns,
                            body: payload.clone(),
                            text: text.to_owned(),
                            content_type: Some("application/json".to_owned()),
                        }));
                    }
                    if let Some(responses) = responses {
                        let perf_ns =
                            u64::try_from(timestamp_ns).map_err(|error| AttemptFailure {
                                error: anyhow!(
                                    "websocket response timestamp was negative: {error}"
                                ),
                                can_retry: false,
                            })?;
                        let server_response = crate::endpoints::ServerResponse {
                            perf_ns,
                            json: parsed_json,
                            raw: None,
                        };
                        if let Some(parsed) =
                            endpoint.parse_response(&server_response).map_err(|error| {
                                AttemptFailure {
                                    error: error.into(),
                                    can_retry: false,
                                }
                            })?
                        {
                            poll_fn(|context| responses.poll_ready(context))
                                .await
                                .map_err(AttemptFailure::terminal)?;
                            responses
                                .start_send(parsed)
                                .map_err(AttemptFailure::terminal)?;
                        }
                    }
                    let event = (if endpoint_id == "responses" {
                        classify_responses_event(&payload, is_text)
                    } else {
                        classify_realtime_event(&payload, is_text)
                    })
                    .map_err(|error| AttemptFailure {
                        error,
                        can_retry: false,
                    })?;
                    if let Some(content) = state.content_for_observation(&event, timestamp_ns) {
                        let text =
                            std::str::from_utf8(&content).map_err(|error| AttemptFailure {
                                error: anyhow!(
                                    "websocket classified content was not UTF-8: {error}"
                                ),
                                can_retry: false,
                            })?;
                        output.push_str(text);
                        observer.on_classified_token(
                            request.uuid,
                            timestamp_ns.saturating_sub(self.start_ns.get()) as f64 / 1_000_000.0,
                            ObservedTokenKind::Output,
                        );
                        if !has_first_token {
                            has_first_token = true;
                            on_first_token(timestamp_ns.saturating_sub(start_ns));
                        }
                    }
                    match &event {
                        ResponsesEvent::Content(_) => {}
                        ResponsesEvent::Reasoning => observer.on_classified_token(
                            request.uuid,
                            timestamp_ns.saturating_sub(self.start_ns.get()) as f64 / 1_000_000.0,
                            ObservedTokenKind::Reasoning,
                        ),
                        ResponsesEvent::Audio => {}
                        ResponsesEvent::Usage(usage) => observer.on_usage(request.uuid, *usage),
                        ResponsesEvent::Terminal {
                            response_id: id,
                            usage,
                            status,
                            ..
                        } => {
                            observer.on_usage(request.uuid, *usage);
                            response_id = id.clone();
                            terminal = *status;
                        }
                        ResponsesEvent::RetriableContinuationRejection => {
                            return Err(AttemptFailure {
                                error: anyhow!("websocket continuation identity was rejected"),
                                can_retry: state.can_retry(),
                            });
                        }
                        ResponsesEvent::Ignored => {}
                    }
                    if state.on_event(&event, timestamp_ns) {
                        break;
                    }
                }
            }
        }
        let (socket, final_events) = driver
            .finish()
            .await
            .map_err(|error| AttemptFailure::from_driver(error, &state))?;
        for event in final_events {
            if let DriverEvent::Flushed {
                role: PreparedWsMessageRole::MeasuredInput,
                timestamp_ns,
            } = event
            {
                state.on_send(timestamp_ns);
            }
        }
        Ok(CompletedAttempt {
            socket,
            connected_ns,
            state,
            terminal,
            output,
            response_id,
            response_bytes,
        })
    }

    fn measurement_observer(&self) -> Result<Rc<NativeMetricsObserver>> {
        self.measurement.observer()
    }
}

struct CompletedAttempt {
    socket: WebSocket,
    connected_ns: i64,
    state: TurnOperationState,
    terminal: ReplayTerminalStatus,
    output: String,
    response_id: Option<String>,
    response_bytes: usize,
}

struct AttemptFailure {
    error: anyhow::Error,
    can_retry: bool,
}

impl AttemptFailure {
    fn terminal(error: anyhow::Error) -> Self {
        Self {
            error,
            can_retry: false,
        }
    }

    fn from_driver(error: WsDriverError, state: &TurnOperationState) -> Self {
        let is_connection_failure = matches!(
            error,
            WsDriverError::Write(_)
                | WsDriverError::Read(_)
                | WsDriverError::PeerClosed
                | WsDriverError::StreamIdleTimeout
                | WsDriverError::ConnectionRotation
        );
        Self {
            error: error.into(),
            can_retry: is_connection_failure && state.can_retry(),
        }
    }
}

async fn dispatch_fallback_before_deadline(
    fallback: &crate::transport::http::TransportSink,
    turn: PreparedTurn,
    observer: &dyn RequestObserver,
    on_first_token: &dyn Fn(i64),
    clock: Rc<dyn Clock>,
    deadline_ns: Option<i64>,
) -> Result<DispatchResult> {
    let dispatch = fallback.dispatch_collect(turn, observer, on_first_token);
    let Some(deadline_ns) = deadline_ns else {
        return dispatch.await;
    };
    let now_ns = clock.now_ns();
    ensure!(
        now_ns < deadline_ns,
        "websocket operation reached its deadline before HTTP/SSE fallback"
    );
    tokio::select! {
        result = dispatch => result,
        () = clock.sleep(deadline_ns.saturating_sub(now_ns)) => {
            anyhow::bail!("websocket operation reached its deadline during HTTP/SSE fallback")
        }
    }
}

async fn close_socket(
    mut socket: WebSocket,
    clock: Rc<dyn Clock>,
    operation_deadline_ns: Option<i64>,
) {
    let now_ns = clock.now_ns();
    let deadline_ns = operation_deadline_ns
        .unwrap_or_else(|| now_ns.saturating_add(CLOSE_HANDSHAKE_NS))
        .min(now_ns.saturating_add(CLOSE_HANDSHAKE_NS));
    tokio::select! {
        result = socket.close(None) => {
            if let Err(error) = result {
                tracing::trace!(component = "websocket", error = %error, "websocket close handshake did not complete");
            }
        }
        () = clock.sleep(deadline_ns.saturating_sub(now_ns)) => {
            tracing::trace!(component = "websocket", "websocket close handshake reached its deadline");
        }
    }
}

async fn acquire_before_deadline<'a>(
    semaphore: &'a Semaphore,
    clock: Rc<dyn Clock>,
    deadline_ns: Option<i64>,
) -> Result<tokio::sync::SemaphorePermit<'a>> {
    let acquire = semaphore.acquire();
    let Some(deadline_ns) = deadline_ns else {
        return acquire.await.map_err(|_| anyhow!("websocket pool closed"));
    };
    let now_ns = clock.now_ns();
    ensure!(
        now_ns < deadline_ns,
        "websocket operation reached its deadline"
    );
    tokio::select! {
        permit = acquire => permit.map_err(|_| anyhow!("websocket pool closed")),
        () = clock.sleep(deadline_ns.saturating_sub(now_ns)) => {
            anyhow::bail!("websocket operation reached its deadline waiting for a connection")
        }
    }
}

fn seconds_to_ns(seconds: f64) -> Result<i64> {
    let nanoseconds = seconds * 1_000_000_000.0;
    ensure!(
        nanoseconds.is_finite() && nanoseconds > 0.0 && nanoseconds <= i64::MAX as f64,
        "websocket duration is outside the native Clock range"
    );
    Ok(nanoseconds.round() as i64)
}

fn capture_operation(operation: &PreparedWsOperation) -> Result<Bytes> {
    let messages = operation
        .messages()
        .iter()
        .map(|message| {
            let opcode = match message.opcode() {
                PreparedWsOpcode::Text => "text",
                PreparedWsOpcode::Binary => "binary",
            };
            let role = match message.role() {
                PreparedWsMessageRole::MeasuredInput => "measured_input",
                PreparedWsMessageRole::Control => "control",
                PreparedWsMessageRole::TerminalAck => "terminal_ack",
            };
            let payload = match message.opcode() {
                PreparedWsOpcode::Text => Value::String(
                    std::str::from_utf8(message.payload())
                        .context("capturing websocket text operation")?
                        .to_owned(),
                ),
                PreparedWsOpcode::Binary => Value::String(STANDARD.encode(message.payload())),
            };
            Ok(serde_json::json!({
                "opcode": opcode,
                "role": role,
                "payload": payload,
            }))
        })
        .collect::<Result<Vec<Value>>>()?;
    Ok(Bytes::from(serde_json::to_vec(
        &serde_json::json!({"transport":"websocket","messages":messages}),
    )?))
}

fn websocket_fallback_url(url: &str) -> Result<String> {
    let mut url = Url::parse(url).with_context(|| format!("parsing fallback URL {url:?}"))?;
    match url.scheme() {
        "ws" => url
            .set_scheme("http")
            .map_err(|_| anyhow!("cannot map ws URL to HTTP"))?,
        "wss" => url
            .set_scheme("https")
            .map_err(|_| anyhow!("cannot map wss URL to HTTPS"))?,
        scheme => anyhow::bail!("fallback requires ws:// or wss://, got {scheme}://"),
    }
    Ok(url.to_string())
}

const fn transport_fallback_reason(reason: FallbackReason) -> TransportFallbackReason {
    match reason {
        FallbackReason::NetworkConnect => TransportFallbackReason::NetworkConnect,
        FallbackReason::UnsupportedUpgrade => TransportFallbackReason::UnsupportedUpgrade,
    }
}

#[async_trait(?Send)]
impl WorkerSink for WebSocketTransportSink {
    fn set_run_origin(&self, origin_ns: i64) {
        self.start_ns.set(origin_ns);
        if let Some(fallback) = &self.fallback {
            fallback.set_run_origin(origin_ns);
        }
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: self
                .selected_url(
                    turn.url_index,
                    turn.endpoint_path.as_deref(),
                    &turn.request_parameters,
                )
                .ok()
                .map(|url| url.to_string()),
            model: turn
                .effective_model
                .clone()
                .or_else(|| Some(self.model.clone())),
        }
    }

    fn supports_response_streaming(&self) -> bool {
        true
    }

    async fn dispatch_measured(
        &self,
        observer: &NativeMetricsObserver,
        turn: PreparedTurn,
        context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        let uuid = turn.request.uuid;
        measure::measure_dispatch(
            observer,
            self.clock.as_ref(),
            uuid,
            context,
            self.dispatch_inner(turn, observer, on_first_token, responses),
        )
        .await
    }
}

#[async_trait(?Send)]
impl RequestExecutor for WebSocketTransportSink {
    fn set_run_origin(&self, origin_ns: i64) -> Result<()> {
        <Self as WorkerSink>::set_run_origin(self, origin_ns);
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <Self as WorkerSink>::inference_dimensions(self, turn)
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        self.measurement
            .configure(self.clock.clone(), config, origin_ns);
        Ok(())
    }

    async fn execute_measured(
        &self,
        turn: PreparedTurn,
        context: MeasuredContext,
        on_first_token: &dyn Fn(i64),
    ) -> Result<MeasuredOutcome> {
        let observer = self.measurement_observer()?;
        let uuid = turn.request.uuid;
        let result = self
            .dispatch_measured(&observer, turn, &context, on_first_token, None)
            .await?;
        Ok(MeasuredOutcome {
            live_record: measure::live_record(&observer, uuid, &context),
            result,
        })
    }

    fn drain_records(
        &self,
        end_ns: i64,
    ) -> Result<Vec<(uuid::Uuid, crate::metrics_core::RecordIngest)>> {
        Ok(self.measurement.drain(end_ns))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body_plan::PreparedWsMessage;

    #[test]
    fn fallback_url_preserves_authority_and_path() {
        assert_eq!(
            websocket_fallback_url("wss://example.test/v1/responses?x=1").expect("URL maps"),
            "https://example.test/v1/responses?x=1"
        );
    }

    #[test]
    fn rotation_happens_before_the_sixty_minute_service_limit() {
        assert!(SOCKET_ROTATION_NS < 60 * 60 * 1_000_000_000);
        assert_eq!(SOCKET_ROTATION_NS, 55 * 60 * 1_000_000_000);
    }

    #[test]
    fn driver_fallback_reasons_map_to_closed_transport_facts() {
        assert_eq!(
            transport_fallback_reason(FallbackReason::NetworkConnect),
            TransportFallbackReason::NetworkConnect
        );
        assert_eq!(
            transport_fallback_reason(FallbackReason::NetworkConnect).as_str(),
            "network_connect"
        );
        assert_eq!(
            transport_fallback_reason(FallbackReason::UnsupportedUpgrade),
            TransportFallbackReason::UnsupportedUpgrade
        );
        assert_eq!(
            transport_fallback_reason(FallbackReason::UnsupportedUpgrade).as_str(),
            "unsupported_upgrade"
        );
    }

    #[test]
    fn raw_operation_capture_preserves_every_message_and_role() {
        let operation = PreparedWsOperation::new(
            [
                PreparedWsMessage::text(
                    Bytes::from_static(br#"{"type":"conversation.item.create"}"#),
                    PreparedWsMessageRole::MeasuredInput,
                ),
                PreparedWsMessage::text(
                    Bytes::from_static(br#"{"type":"response.create"}"#),
                    PreparedWsMessageRole::Control,
                ),
            ],
            None,
        );
        let captured: Value = serde_json::from_slice(
            &capture_operation(&operation).expect("operation capture serializes"),
        )
        .expect("operation capture is JSON");
        assert_eq!(captured["transport"], "websocket");
        assert_eq!(captured["messages"].as_array().map(Vec::len), Some(2));
        assert_eq!(captured["messages"][0]["role"], "measured_input");
        assert_eq!(captured["messages"][1]["role"], "control");
    }
}
