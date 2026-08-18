// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local native WebSocket execution for the Responses endpoint.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use futures::SinkExt;
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
use url::Url;

use crate::body_plan::{PreparedWsMessageRole, PreparedWsOperation, RequestBody};
use crate::clock::Clock;
use crate::config::model::transport::{WebSocketFallback, WebSocketTransportConfig};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{ObservedTokenKind, RequestObserver};
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
    PreparedEndpointBinding, PreparedTurn, RequestExecutor, RequestRecord,
};
use crate::transport::measure::{self, WorkerMeasurement};
use crate::transport::ws::connector::{ConnectFailure, WebSocket, connect};
use crate::transport::ws::dialect::{
    ResponsesEvent, TurnOperationState, classify_responses_event, full_history_retry,
};
use crate::transport::ws::driver::{
    ApplicationQueueLimits, DriverEvent, DriverTiming, FallbackReason, SocketOperationDriver,
    WsDriverError,
};

const SOCKET_ROTATION_NS: i64 = 55 * 60 * 1_000_000_000;
const CLOSE_HANDSHAKE_NS: i64 = 1_000_000_000;

struct CachedSocket {
    url: String,
    headers: BTreeMap<String, String>,
    connected_ns: i64,
    retained_ns: i64,
    socket: WebSocket,
}

/// Native WebSocket execution factory with immutable per-run policy.
#[derive(Clone, Debug)]
pub struct WebSocketExecutionFactory {
    config: WebSocketTransportConfig,
}

/// Registry execution binding for Responses over WebSocket.
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
            run.artifacts.raw_path.is_none() && !run.artifacts.trace,
            "websocket execution does not support raw exchange or HTTP trace artifacts"
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
                profile.endpoint_id.as_str() == "responses",
                "websocket endpoint profile {profile_id:?} requires endpoint type responses"
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
        ensure!(
            !config.raw_enabled,
            "native websocket execution does not support raw exchange artifacts"
        );
        let workers = config.workers;
        let clock = config.coordinator_clock.clone();
        let anchor = config.real_clock_anchor;
        let hop_routing = config.hop_routing;
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
    slots: Semaphore,
    fallback: Option<crate::transport::http::TransportSink>,
    start_ns: Cell<i64>,
    measurement: WorkerMeasurement,
}

impl WebSocketTransportSink {
    fn new(
        clock: Rc<dyn Clock>,
        base_urls: Vec<String>,
        model: String,
        transport: crate::transport::http::TransportSinkConfig,
        config: WebSocketTransportConfig,
        endpoints: crate::endpoints::PreparedEndpointTable,
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
            slots: Semaphore::new(slot_count),
            fallback,
            start_ns: Cell::new(0),
            measurement: WorkerMeasurement::default(),
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
                .position(|cached| cached.url == url && cached.headers == *headers)
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
        url: &Url,
        headers: &BTreeMap<String, String>,
        deadline_ns: Option<i64>,
    ) -> Result<(WebSocket, i64), ConnectFailure> {
        let now_ns = self.clock.now_ns();
        let (cached, expired) = self.take_cached(url.as_str(), headers, now_ns);
        for cached in expired {
            close_socket(cached.socket, self.clock.clone(), deadline_ns).await;
        }
        if let Some(cached) = cached {
            return Ok((cached.socket, cached.connected_ns));
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
        Ok((socket, self.clock.now_ns()))
    }

    fn validate_endpoint(&self, turn: &PreparedTurn) -> Result<()> {
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
            endpoint.descriptor().id == "responses",
            "websocket execution requires the responses endpoint, got {:?}",
            endpoint.descriptor().id
        );
        Ok(())
    }

    async fn dispatch_inner(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        self.validate_endpoint(&turn)?;
        let operation = match turn.request.body.as_ref() {
            Some(RequestBody::WebSocket(operation)) => operation.clone(),
            _ => anyhow::bail!("websocket execution requires a prepared websocket operation"),
        };
        let fallback_turn = operation.http_sse_fallback_body().map(|body| {
            let mut fallback = turn.clone();
            fallback.request.body = Some(RequestBody::wire(body.clone()));
            fallback
        });
        let request = turn.request;
        let url = self.selected_url(
            request.url_index,
            request.endpoint_path.as_deref(),
            &request.parameters,
        )?;
        let start_ns = self.clock.now_ns();
        let deadline_ns = self
            .transport
            .client
            .total_timeout_ns
            .and_then(|timeout_ns| start_ns.checked_add(timeout_ns));
        let _slot = acquire_before_deadline(&self.slots, self.clock.clone(), deadline_ns).await?;
        let mut record = RequestRecord::started(start_ns);
        record.request_headers = request.headers.clone();
        let (socket, connected_ns) = match self.checkout(&url, &request.headers, deadline_ns).await
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
                tracing::debug!(
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
        let attempt = self
            .run_attempt(
                socket,
                &operation,
                &request,
                observer,
                on_first_token,
                start_ns,
                deadline_ns,
                connected_ns,
            )
            .await;
        let completed = match attempt {
            Ok(completed) => completed,
            Err(failure) if failure.can_retry => {
                let replay = full_history_retry(&operation).ok_or_else(|| {
                    anyhow!("websocket operation cannot rebuild a self-contained retry")
                })?;
                tracing::debug!(error = %failure.error, "retrying websocket operation before visible output");
                let (socket, retry_connected_ns) = self
                    .checkout_fresh(&url, &request.headers, deadline_ns)
                    .await?;
                self.run_attempt(
                    socket,
                    &replay,
                    &request,
                    observer,
                    on_first_token,
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
        observer.on_round_trip_metrics(request.uuid, completed.state.finish());
        observer.on_terminal(request.uuid, ReplayTerminalStatus::Completed);
        let reusable = self.transport.connection_reuse != ConnectionReuseStrategy::Never;
        if reusable {
            self.idle.borrow_mut().push(CachedSocket {
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
                terminal: ReplayTerminalStatus::Completed,
                response_text: completed.output.clone(),
                model_response: ModelResponseMetadata {
                    content: (!completed.output.is_empty()).then_some(completed.output),
                    response_id: completed.response_id,
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
            request_payload: bytes::Bytes::new(),
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
        operation: &PreparedWsOperation,
        request: &crate::transport::core::Request,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        start_ns: i64,
        deadline_ns: Option<i64>,
        connected_ns: i64,
    ) -> std::result::Result<CompletedAttempt, AttemptFailure> {
        let timing = DriverTiming {
            deadline_ns,
            ping_interval_ns: seconds_to_ns(self.config.ping_interval_seconds)
                .map_err(AttemptFailure::terminal)?,
            stream_idle_timeout_ns: seconds_to_ns(self.config.stream_idle_timeout_seconds)
                .map_err(AttemptFailure::terminal)?,
        };
        let mut driver = SocketOperationDriver::start(
            socket,
            self.clock.clone(),
            operation,
            ApplicationQueueLimits::new(
                self.config.max_queued_commands,
                self.config.max_queued_bytes,
            ),
            timing,
            self.config.max_response_bytes,
        )
        .map_err(|error| AttemptFailure::terminal(error.into()))?;
        let mut state = TurnOperationState::default();
        let mut output = String::new();
        let mut response_id = None;
        let mut response_bytes = 0_usize;
        let mut has_first_token = false;
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
                    let event = classify_responses_event(&payload, is_text).map_err(|error| {
                        AttemptFailure {
                            error,
                            can_retry: false,
                        }
                    })?;
                    match &event {
                        ResponsesEvent::Content(content) => {
                            let text =
                                std::str::from_utf8(content).map_err(|error| AttemptFailure {
                                    error: anyhow!("Responses content was not UTF-8: {error}"),
                                    can_retry: false,
                                })?;
                            output.push_str(text);
                            observer.on_classified_token(
                                request.uuid,
                                timestamp_ns.saturating_sub(self.start_ns.get()) as f64
                                    / 1_000_000.0,
                                ObservedTokenKind::Output,
                            );
                            if !has_first_token {
                                has_first_token = true;
                                on_first_token(timestamp_ns.saturating_sub(start_ns));
                            }
                        }
                        ResponsesEvent::Reasoning => observer.on_classified_token(
                            request.uuid,
                            timestamp_ns.saturating_sub(self.start_ns.get()) as f64 / 1_000_000.0,
                            ObservedTokenKind::Reasoning,
                        ),
                        ResponsesEvent::Usage(usage) => observer.on_usage(request.uuid, *usage),
                        ResponsesEvent::Terminal {
                            response_id: id,
                            usage,
                        } => {
                            observer.on_usage(request.uuid, *usage);
                            response_id = id.clone();
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
                tracing::trace!(error = %error, "websocket close handshake did not complete");
            }
        }
        () = clock.sleep(deadline_ns.saturating_sub(now_ns)) => {
            tracing::trace!("websocket close handshake reached its deadline");
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

    async fn dispatch_measured(
        &self,
        observer: &NativeMetricsObserver,
        turn: PreparedTurn,
        context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
        _responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        let uuid = turn.request.uuid;
        measure::measure_dispatch(
            observer,
            self.clock.as_ref(),
            uuid,
            context,
            self.dispatch_inner(turn, observer, on_first_token),
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
}
