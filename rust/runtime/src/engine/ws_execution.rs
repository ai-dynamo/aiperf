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
use bytes::Bytes;
use futures::{SinkExt, StreamExt};
use tokio::sync::Semaphore;
use tokio_tungstenite::tungstenite::{Message, protocol::WebSocketConfig};
use url::Url;

use crate::body_plan::{PreparedWsMessageRole, PreparedWsOperation, RequestBody};
use crate::clock::Clock;
use crate::config::model::transport::{
    WEBSOCKET_WRITER_RESERVE_BYTES, WebSocketFallback, WebSocketTransportConfig,
};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{
    ObservedTokenKind, ObservedTransportRoute, RequestObserver, TransportFallbackReason,
    TransportRoute,
};
use crate::endpoints::{WebSocketCapabilities, WebSocketConnectionModel, WebSocketDialect};
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
    ConnectionReuseStrategy, DispatchResult, ErrorDetails, ErrorKind, MeasuredContext,
    MeasuredOutcome, PreparedEndpointBinding, PreparedTurn, RequestExecutor, RequestRecord,
    Response, TextResponse,
};
use crate::transport::measure::{self, WorkerMeasurement};
use crate::transport::ws::connector::{ConnectFailure, WebSocket, connect};
use crate::transport::ws::dialect::{
    EventDisposition, OperationCorrelation, ResponsesEvent, TurnOperationState,
    classify_realtime_event, classify_responses_event, correlate_operation, full_history_retry,
    with_previous_response_id,
};
use crate::transport::ws::driver::{
    ApplicationQueueLimits, DriverEvent, DriverTiming, FallbackReason, SocketOperationDriver,
    WsDriverError,
};

const SOCKET_ROTATION_NS: i64 = 55 * 60 * 1_000_000_000;
const SOCKET_SERVICE_LIMIT_NS: i64 = 60 * 60 * 1_000_000_000;
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
    has_affinity_state: bool,
    is_reused: bool,
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
            let capabilities = context
                .product_registry()
                .endpoints()
                .resolve_factory(&profile.endpoint_id)?
                .websocket_capabilities()
                .ok_or_else(|| {
                    anyhow!(
                        "websocket endpoint profile {profile_id:?} selected endpoint {:?} without websocket capabilities",
                        profile.endpoint_id
                    )
                })?;
            ensure!(
                matches!(
                    capabilities.connection_model,
                    WebSocketConnectionModel::TurnSerialized | WebSocketConnectionModel::Duplex
                ),
                "websocket endpoint profile {profile_id:?} selected an unsupported connection model"
            );
            ensure!(
                self.config.fallback == WebSocketFallback::Disabled
                    || capabilities.supports_http_sse_fallback,
                "websocket endpoint profile {profile_id:?} does not register an HTTP/SSE fallback"
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
    max_write_buffer_size: usize,
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
        let config = config.validate()?;
        let max_write_buffer_size = config
            .max_queued_bytes
            .checked_add(WEBSOCKET_WRITER_RESERVE_BYTES)
            .ok_or_else(|| anyhow!("websocket writer capacity overflowed after validation"))?;
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
            max_write_buffer_size,
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
            .max_write_buffer_size(self.max_write_buffer_size)
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
        if cached.is_none()
            && socket_pool_requires_eviction(idle.len(), self.slots.available_permits())
        {
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
            let has_affinity_state = cached.affinity_key.as_deref() == Some(affinity_key);
            return Ok(CheckedOutSocket {
                socket: cached.socket,
                connected_ns: cached.connected_ns,
                continuation_id: cached.continuation_id,
                has_affinity_state,
                is_reused: true,
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
            has_affinity_state: false,
            is_reused: false,
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
    ) -> Result<(
        &dyn crate::endpoints::PreparedEndpoint,
        WebSocketCapabilities,
    )> {
        let PreparedEndpointBinding::Prepared(reference) = &turn.endpoint;
        let endpoint = self.endpoints.get(reference.key)?;
        ensure!(
            endpoint.descriptor().id == reference.endpoint_id.as_str(),
            "prepared endpoint key {} resolved to {:?}, expected {:?}",
            reference.key.index(),
            endpoint.descriptor().id,
            reference.endpoint_id.as_str()
        );
        let capabilities = endpoint.websocket_capabilities().ok_or_else(|| {
            anyhow!(
                "websocket execution requires registered capabilities, got endpoint {:?}",
                endpoint.descriptor().id
            )
        })?;
        Ok((endpoint, capabilities))
    }

    async fn dispatch_inner(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        let (endpoint, capabilities) = self.validate_endpoint(&turn)?;
        let prepared_operation = match turn.request.body.as_ref() {
            Some(RequestBody::WebSocket(operation)) => operation.clone(),
            _ => anyhow::bail!("websocket execution requires a prepared websocket operation"),
        };
        let fallback_turn = capabilities
            .supports_http_sse_fallback
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
        let _gate_cleanup = AffinityGateCleanup {
            gates: &self.affinity_gates,
            affinity_key: &affinity_key,
            gate: affinity_gate.clone(),
        };
        let _affinity =
            acquire_before_deadline(affinity_gate.as_ref(), self.clock.clone(), deadline_ns)
                .await?;
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
                let turn = fallback_turn
                    .clone()
                    .ok_or_else(|| anyhow!("fallback body was not prepared"))?;
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
            Err(error) => {
                return self.failed_dispatch_result(
                    &request,
                    &prepared_operation,
                    record,
                    observer,
                    start_ns,
                    AttemptFailure::connect(error.into()),
                );
            }
        };
        record.status = Some(101);
        if capabilities.dialect == WebSocketDialect::Realtime
            && prepared_operation.requires_affinity_state()
            && !checked_out.has_affinity_state
        {
            close_socket(checked_out.socket, self.clock.clone(), deadline_ns).await;
            return self.failed_dispatch_result(
                &request,
                &prepared_operation,
                record,
                observer,
                start_ns,
                AttemptFailure::protocol(anyhow!(
                    "Realtime continuation lost its affinity-owned socket"
                )),
            );
        }
        let operation = match capabilities.dialect {
            WebSocketDialect::Responses => checked_out
                .continuation_id
                .as_deref()
                .filter(|_| checked_out.has_affinity_state)
                .map(|response_id| {
                    with_previous_response_id(&prepared_operation, response_id).ok_or_else(|| {
                        anyhow!(
                            "websocket continuation could not be injected into the prepared operation"
                        )
                    })
                })
                .transpose()?
                .unwrap_or_else(|| prepared_operation.as_ref().clone()),
            WebSocketDialect::Realtime => prepared_operation.as_ref().clone(),
        };
        ensure!(
            operation
                .messages()
                .iter()
                .all(|message| message.opcode() == capabilities.application_opcode),
            "websocket endpoint {:?} prepared an unregistered application opcode",
            endpoint.descriptor().id
        );
        let correlated = match correlate_operation(
            &operation,
            capabilities.dialect,
            &format!("{}:0", request.uuid),
        ) {
            Ok(correlated) => correlated,
            Err(error) => {
                close_socket(checked_out.socket, self.clock.clone(), deadline_ns).await;
                return self.failed_dispatch_result(
                    &request,
                    &operation,
                    record,
                    observer,
                    start_ns,
                    AttemptFailure::protocol(error),
                );
            }
        };
        let attempt = self
            .run_attempt(
                checked_out.socket,
                endpoint,
                correlated.operation(),
                correlated.correlation(),
                checked_out.is_reused,
                &request,
                &mut record,
                observer,
                on_first_token,
                responses,
                capabilities.dialect,
                start_ns,
                deadline_ns,
                checked_out.connected_ns,
            )
            .await;
        let (completed, executed_operation) = match attempt {
            Ok(completed) => (completed, correlated.operation().clone()),
            Err(failure) if capabilities.supports_full_history_replay && failure.can_retry => {
                let replay = full_history_retry(&prepared_operation).ok_or_else(|| {
                    anyhow!("websocket operation cannot rebuild a self-contained retry")
                })?;
                let correlated_replay = match correlate_operation(
                    &replay,
                    capabilities.dialect,
                    &format!("{}:1", request.uuid),
                ) {
                    Ok(correlated) => correlated,
                    Err(error) => {
                        return self.failed_dispatch_result(
                            &request,
                            &replay,
                            record,
                            observer,
                            start_ns,
                            AttemptFailure::protocol(error),
                        );
                    }
                };
                tracing::debug!(component = "websocket", error = %failure.error, "retrying websocket operation before visible output");
                // Raw capture follows the winning-attempt policy: an abandoned
                // replay-safe attempt contributes neither receive timing nor
                // application messages to the retried exchange.
                reset_record_for_retry(&mut record);
                let (socket, retry_connected_ns) = match self
                    .checkout_fresh(&url, &request.headers, deadline_ns)
                    .await
                {
                    Ok(fresh) => fresh,
                    Err(error)
                        if self.config.fallback == WebSocketFallback::HttpSse
                            && fallback_turn.is_some()
                            && error.fallback_reason().is_some() =>
                    {
                        let reason = error.fallback_reason().ok_or_else(|| {
                            anyhow!("retry fallback failure lost its stable reason")
                        })?;
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
                            "using declared HTTP/SSE fallback after a replay-safe socket failure"
                        );
                        let fallback = self
                            .fallback
                            .as_ref()
                            .ok_or_else(|| anyhow!("websocket fallback sink was not prepared"))?;
                        let turn = fallback_turn
                            .clone()
                            .ok_or_else(|| anyhow!("fallback body was not prepared"))?;
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
                    Err(error) => {
                        return self.failed_dispatch_result(
                            &request,
                            correlated_replay.operation(),
                            record,
                            observer,
                            start_ns,
                            AttemptFailure::connect(error.into()),
                        );
                    }
                };
                record.status = Some(101);
                let retry_attempt = self
                    .run_attempt(
                        socket,
                        endpoint,
                        correlated_replay.operation(),
                        correlated_replay.correlation(),
                        false,
                        &request,
                        &mut record,
                        observer,
                        on_first_token,
                        responses,
                        capabilities.dialect,
                        start_ns,
                        deadline_ns,
                        retry_connected_ns,
                    )
                    .await;
                let completed = match retry_attempt {
                    Ok(completed) => completed,
                    Err(failure) => {
                        return self.failed_dispatch_result(
                            &request,
                            correlated_replay.operation(),
                            record,
                            observer,
                            start_ns,
                            failure,
                        );
                    }
                };
                (completed, correlated_replay.operation().clone())
            }
            Err(failure) => {
                return self.failed_dispatch_result(
                    &request,
                    correlated.operation(),
                    record,
                    observer,
                    start_ns,
                    failure,
                );
            }
        };
        let end_ns = self.clock.now_ns();
        if capabilities.supports_round_trip_metrics {
            observer
                .on_round_trip_metrics(request.uuid, completed.state.finish(completed.terminal));
        }
        observer.on_terminal(request.uuid, completed.terminal);
        let reusable = completed.terminal == ReplayTerminalStatus::Completed
            && completed.state.has_verified_correlation()
            && self.transport.connection_reuse != ConnectionReuseStrategy::Never
            && (capabilities.connection_model == WebSocketConnectionModel::TurnSerialized
                || !request.is_final_turn);
        if reusable {
            self.idle.borrow_mut().push(CachedSocket {
                affinity_key: (capabilities.has_affinity_state && !request.is_final_turn)
                    .then(|| affinity_key.clone()),
                continuation_id: (capabilities.dialect == WebSocketDialect::Responses
                    && !request.is_final_turn)
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
        if completed.terminal != ReplayTerminalStatus::Completed {
            let message = format!(
                "websocket {:?} operation ended {:?}",
                capabilities.dialect, completed.terminal
            );
            record.error = Some(if completed.terminal == ReplayTerminalStatus::Canceled {
                record.cancellation_ns = cancellation_timestamp(completed.terminal, end_ns);
                ErrorDetails::cancelled(message)
            } else {
                ErrorDetails {
                    kind: ErrorKind::Protocol,
                    code: None,
                    message,
                }
            });
        }
        Ok(DispatchResult {
            outcome: TurnDispatchOutcome {
                start_ns,
                end_ns,
                terminal: completed.terminal,
                response_text: completed.output.clone(),
                model_response: ModelResponseMetadata {
                    content: (!completed.output.is_empty()).then_some(completed.output),
                    response_id: completed.response_id,
                    error_kind: record
                        .error
                        .as_ref()
                        .map(|error| websocket_error_kind(error.kind).to_owned()),
                    error_message: record.error.as_ref().map(|error| error.message.clone()),
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
                executed_operation.to_artifact_bytes()?
            } else {
                Bytes::new()
            },
            record,
        })
    }

    fn failed_dispatch_result(
        &self,
        request: &crate::transport::core::Request,
        operation: &PreparedWsOperation,
        mut record: RequestRecord,
        observer: &dyn RequestObserver,
        start_ns: i64,
        failure: AttemptFailure,
    ) -> Result<DispatchResult> {
        let end_ns = self.clock.now_ns();
        let error = failure.into_details();
        let terminal = failure_terminal(error.kind);
        if error.kind == ErrorKind::Cancelled {
            record.cancellation_ns = Some(end_ns);
        }
        record.end_ns = Some(end_ns);
        record.error = Some(error.clone());
        record.reusable_connection = false;
        observer.on_terminal(request.uuid, terminal);
        Ok(DispatchResult {
            outcome: TurnDispatchOutcome {
                start_ns,
                end_ns,
                terminal,
                response_text: String::new(),
                model_response: ModelResponseMetadata {
                    error_kind: Some(websocket_error_kind(error.kind).to_owned()),
                    error_message: Some(error.message),
                    ..ModelResponseMetadata::default()
                },
                prompt_tokens: None,
                completion_tokens: None,
                http: RequestTrace {
                    duration_ns: end_ns.checked_sub(start_ns),
                    ..RequestTrace::default()
                },
            },
            request_payload: if self.capture_raw {
                operation.to_artifact_bytes()?
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
    ) -> std::result::Result<(WebSocket, i64), ConnectFailure> {
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
        correlation: &OperationCorrelation,
        is_reused_socket: bool,
        request: &crate::transport::core::Request,
        record: &mut RequestRecord,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
        dialect: WebSocketDialect,
        start_ns: i64,
        deadline_ns: Option<i64>,
        connected_ns: i64,
    ) -> std::result::Result<CompletedAttempt, AttemptFailure> {
        let timing = DriverTiming {
            deadline_ns,
            rotation_deadline_ns: connected_ns.saturating_add(SOCKET_SERVICE_LIMIT_NS),
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
        let mut state = TurnOperationState::new(correlation, is_reused_socket);
        let mut output = String::new();
        let mut response_id = None;
        let mut response_bytes = 0_usize;
        let mut has_first_token = false;
        let mut terminal = ReplayTerminalStatus::Failed;
        let mut has_terminal = false;
        let mut pending_response = None;
        loop {
            if pending_response.is_none() && has_terminal {
                break;
            }
            let Some(event) = next_while_delivering_response(
                &mut driver,
                &mut pending_response,
                responses,
                &state,
            )
            .await?
            else {
                continue;
            };
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
                    if has_terminal {
                        return Err(AttemptFailure::protocol(anyhow!(
                            "websocket application event arrived after terminal"
                        )));
                    }
                    let event = (match dialect {
                        WebSocketDialect::Responses => classify_responses_event(&payload, is_text),
                        WebSocketDialect::Realtime => classify_realtime_event(&payload, is_text),
                    })
                    .map_err(AttemptFailure::protocol)?;
                    let disposition = state
                        .on_correlated_event(&event, timestamp_ns)
                        .map_err(AttemptFailure::protocol)?;
                    let is_terminal = match disposition {
                        EventDisposition::Attributed { is_terminal } => is_terminal,
                        EventDisposition::AttributedError => {
                            return Err(AttemptFailure::retryable_protocol(
                                anyhow!(
                                    "{}",
                                    event
                                        .error_message()
                                        .unwrap_or("correlated websocket application error")
                                ),
                                state.can_retry(),
                            ));
                        }
                        EventDisposition::Unattributed => continue,
                        EventDisposition::UnsafeUnattributedError => {
                            return Err(AttemptFailure::retryable_protocol(
                                anyhow!(
                                    "websocket application error could not be correlated to the current operation"
                                ),
                                state.can_retry(),
                            ));
                        }
                    };
                    response_bytes =
                        response_bytes.checked_add(payload.len()).ok_or_else(|| {
                            AttemptFailure::protocol(anyhow!(
                                "websocket response byte count overflowed"
                            ))
                        })?;
                    record.recv_start_ns.get_or_insert(timestamp_ns);
                    let parsed_json = if is_text && responses.is_some() {
                        Some(serde_json::from_slice(&payload).map_err(|error| {
                            AttemptFailure::protocol(anyhow!(
                                "websocket application message was not JSON: {error}"
                            ))
                        })?)
                    } else {
                        None
                    };
                    if self.capture_raw && is_text {
                        let text = std::str::from_utf8(&payload).map_err(|error| {
                            AttemptFailure::protocol(anyhow!(
                                "websocket text frame was not UTF-8: {error}"
                            ))
                        })?;
                        record.responses.push(Response::Text(TextResponse {
                            perf_ns: timestamp_ns,
                            body: payload.clone(),
                            text: text.to_owned(),
                            content_type: Some("application/json".to_owned()),
                        }));
                    }
                    if responses.is_some() {
                        let perf_ns = u64::try_from(timestamp_ns).map_err(|error| {
                            AttemptFailure::protocol(anyhow!(
                                "websocket response timestamp was negative: {error}"
                            ))
                        })?;
                        let server_response = crate::endpoints::ServerResponse {
                            perf_ns,
                            json: parsed_json,
                            raw: None,
                        };
                        if let Some(parsed) = endpoint
                            .parse_response(&server_response)
                            .map_err(|error| AttemptFailure::protocol(error.into()))?
                        {
                            if pending_response.replace(parsed).is_some() {
                                return Err(AttemptFailure::protocol(anyhow!(
                                    "websocket response observer queue is full"
                                )));
                            }
                        }
                    }
                    if let Some(content) = state
                        .content_for_observation(&event, timestamp_ns)
                        .map_err(AttemptFailure::protocol)?
                    {
                        let text = std::str::from_utf8(&content).map_err(|error| {
                            AttemptFailure::protocol(anyhow!(
                                "websocket classified content was not UTF-8: {error}"
                            ))
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
                        ResponsesEvent::Created { .. } | ResponsesEvent::Content { .. } => {}
                        ResponsesEvent::Reasoning { .. } => observer.on_classified_token(
                            request.uuid,
                            timestamp_ns.saturating_sub(self.start_ns.get()) as f64 / 1_000_000.0,
                            ObservedTokenKind::Reasoning,
                        ),
                        ResponsesEvent::Audio { .. } => {}
                        ResponsesEvent::Usage { usage, .. } => {
                            observer.on_usage(request.uuid, *usage)
                        }
                        ResponsesEvent::Terminal {
                            response_id: id,
                            usage,
                            status,
                            ..
                        } => {
                            observer.on_usage(request.uuid, *usage);
                            response_id = Some(id.clone());
                            terminal = *status;
                        }
                        ResponsesEvent::RetriableContinuationRejection => {
                            return Err(AttemptFailure::retryable_protocol(
                                anyhow!("websocket continuation identity was rejected"),
                                state.can_retry(),
                            ));
                        }
                        ResponsesEvent::Error { .. } => {
                            return Err(AttemptFailure::protocol(anyhow!(
                                "correlated websocket error was not handled before observation"
                            )));
                        }
                        ResponsesEvent::Ignored => {}
                    }
                    if is_terminal {
                        has_terminal = true;
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

    async fn shutdown_idle_sockets(&self) {
        let sockets = std::mem::take(&mut *self.idle.borrow_mut());
        futures::future::join_all(
            sockets
                .into_iter()
                .map(|cached| close_socket(cached.socket, self.clock.clone(), None)),
        )
        .await;
    }
}

async fn next_while_delivering_response(
    driver: &mut SocketOperationDriver<WebSocket>,
    pending_response: &mut Option<crate::endpoints::ParsedResponse>,
    responses: Option<&dyn TurnResponseObserver>,
    state: &TurnOperationState,
) -> std::result::Result<Option<DriverEvent>, AttemptFailure> {
    if pending_response.is_none() {
        return driver
            .next()
            .await
            .map(Some)
            .map_err(|error| AttemptFailure::from_driver(error, state));
    }
    let responses = responses.ok_or_else(|| {
        AttemptFailure::protocol(anyhow!("websocket response delivery lost its observer"))
    })?;
    tokio::select! {
        biased;
        ready = poll_fn(|context| responses.poll_ready(context)) => {
            ready.map_err(AttemptFailure::terminal)?;
            let response = pending_response.take().ok_or_else(|| {
                AttemptFailure::protocol(anyhow!(
                    "websocket response delivery lost its pending frame"
                ))
            })?;
            responses
                .start_send(response)
                .map_err(AttemptFailure::terminal)?;
            Ok(None)
        }
        event = driver.next() => event
            .map(Some)
            .map_err(|error| AttemptFailure::from_driver(error, state)),
    }
}

const fn socket_pool_requires_eviction(
    idle_socket_count: usize,
    available_operation_slots: usize,
) -> bool {
    idle_socket_count > available_operation_slots
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
    kind: ErrorKind,
    can_retry: bool,
}

impl AttemptFailure {
    fn terminal(error: anyhow::Error) -> Self {
        Self {
            error,
            kind: ErrorKind::Other,
            can_retry: false,
        }
    }

    fn connect(error: anyhow::Error) -> Self {
        Self {
            error,
            kind: ErrorKind::Connect,
            can_retry: false,
        }
    }

    fn protocol(error: anyhow::Error) -> Self {
        Self::retryable_protocol(error, false)
    }

    fn retryable_protocol(error: anyhow::Error, can_retry: bool) -> Self {
        Self {
            error,
            kind: ErrorKind::Protocol,
            can_retry,
        }
    }

    fn from_driver(error: WsDriverError, state: &TurnOperationState) -> Self {
        let is_connection_failure = matches!(
            &error,
            WsDriverError::Write(_)
                | WsDriverError::Read(_)
                | WsDriverError::PeerClosed
                | WsDriverError::CloseHandshakeTimeout
                | WsDriverError::StreamIdleTimeout
                | WsDriverError::ConnectionRotation
        );
        let kind = match &error {
            WsDriverError::OperationDeadline
            | WsDriverError::StreamIdleTimeout
            | WsDriverError::CloseHandshakeTimeout
            | WsDriverError::ConnectionRotation => ErrorKind::Timeout,
            WsDriverError::RequestCancellation => ErrorKind::Cancelled,
            WsDriverError::Write(_) | WsDriverError::Read(_) | WsDriverError::PeerClosed => {
                ErrorKind::Connect
            }
            _ => ErrorKind::Protocol,
        };
        Self {
            error: error.into(),
            kind,
            can_retry: is_connection_failure && state.can_retry(),
        }
    }

    fn into_details(self) -> ErrorDetails {
        let message = self.error.to_string();
        if self.kind == ErrorKind::Cancelled {
            ErrorDetails::cancelled(message)
        } else {
            ErrorDetails {
                kind: self.kind,
                code: None,
                message,
            }
        }
    }
}

const fn websocket_error_kind(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Http => "http",
        ErrorKind::Sse => "sse",
        ErrorKind::Cancelled => "cancelled",
        ErrorKind::Connect => "connect",
        ErrorKind::Protocol => "protocol",
        ErrorKind::Timeout => "timeout",
        ErrorKind::Other => "other",
    }
}

const fn failure_terminal(kind: ErrorKind) -> ReplayTerminalStatus {
    if matches!(kind, ErrorKind::Cancelled) {
        ReplayTerminalStatus::Canceled
    } else {
        ReplayTerminalStatus::Failed
    }
}

const fn cancellation_timestamp(status: ReplayTerminalStatus, end_ns: i64) -> Option<i64> {
    if matches!(status, ReplayTerminalStatus::Canceled) {
        Some(end_ns)
    } else {
        None
    }
}

fn reset_record_for_retry(record: &mut RequestRecord) {
    record.recv_start_ns = None;
    record.status = None;
    record.responses.clear();
    record.error = None;
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
    if now_ns >= deadline_ns {
        tracing::trace!(
            component = "websocket",
            "websocket close handshake reached its deadline"
        );
        return;
    }
    let handshake = async {
        socket.send(Message::Close(None)).await?;
        loop {
            match socket.next().await {
                Some(Ok(Message::Close(_))) => {
                    socket.flush().await?;
                    return Ok::<(), tokio_tungstenite::tungstenite::Error>(());
                }
                Some(Ok(Message::Ping(_))) => socket.flush().await?,
                Some(Ok(_)) => {}
                Some(Err(error)) => return Err(error),
                None => return Err(tokio_tungstenite::tungstenite::Error::ConnectionClosed),
            }
        }
    };
    tokio::select! {
        result = handshake => {
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

    async fn shutdown(&self) -> Result<()> {
        self.shutdown_idle_sockets().await;
        Ok(())
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

    fn supports_response_streaming(&self) -> bool {
        true
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

    async fn execute_measured_streaming(
        &self,
        turn: PreparedTurn,
        context: MeasuredContext,
        on_first_token: &dyn Fn(i64),
        responses: &dyn TurnResponseObserver,
    ) -> Result<MeasuredOutcome> {
        let observer = self.measurement_observer()?;
        let uuid = turn.request.uuid;
        let result = self
            .dispatch_measured(&observer, turn, &context, on_first_token, Some(responses))
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

    async fn shutdown(&self) -> Result<()> {
        self.shutdown_idle_sockets().await;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::task::{Context, Poll};

    use crate::body_plan::PreparedWsMessage;
    use crate::clock::RealClock;
    use crate::endpoints::{EndpointId, EndpointRegistry, RawEndpointConfig};
    use crate::multiturn::{PreparedEndpointReference, TurnDataPolicy};
    use futures::StreamExt;
    use serde_json::Value;
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;
    use tokio_tungstenite::accept_async;
    use tokio_tungstenite::tungstenite::Message;
    use uuid::Uuid;

    struct StalledResponseObserver;

    impl TurnResponseObserver for StalledResponseObserver {
        fn poll_ready(&self, _context: &mut Context<'_>) -> Poll<Result<()>> {
            Poll::Pending
        }

        fn start_send(&self, _response: crate::endpoints::ParsedResponse) -> Result<()> {
            anyhow::bail!("stalled observer must never accept a response")
        }
    }

    #[derive(Default)]
    struct TerminalObserver(RefCell<Vec<ReplayTerminalStatus>>);

    impl RequestObserver for TerminalObserver {
        fn on_arrival(&self, _uuid: Uuid, _at_ms: f64, _input: usize, _output: usize) {}

        fn on_admit(&self, _uuid: Uuid, _at_ms: f64, _reused: usize) {}

        fn on_token(&self, _uuid: Uuid, _at_ms: f64) {}

        fn on_terminal(&self, _uuid: Uuid, status: ReplayTerminalStatus) {
            self.0.borrow_mut().push(status);
        }
    }

    fn realtime_sink(
        clock: Rc<dyn Clock>,
        url: String,
        transport: crate::transport::http::TransportSinkConfig,
    ) -> (WebSocketTransportSink, PreparedEndpointReference) {
        let endpoint_id = EndpointId::new("realtime").expect("Realtime endpoint id is valid");
        let endpoint = EndpointRegistry::builtin()
            .expect("builtin endpoints register")
            .prepare(&endpoint_id, RawEndpointConfig::default())
            .expect("Realtime endpoint prepares");
        let mut endpoints = crate::endpoints::PreparedEndpointTable::new();
        let key = endpoints
            .push(endpoint)
            .expect("Realtime endpoint is stored");
        let sink = WebSocketTransportSink::new(
            clock,
            vec![url],
            "model".to_owned(),
            transport,
            WebSocketTransportConfig::default(),
            endpoints,
            true,
        )
        .expect("Realtime sink builds");
        (sink, PreparedEndpointReference { key, endpoint_id })
    }

    fn responses_sink(
        clock: Rc<dyn Clock>,
        url: String,
    ) -> (WebSocketTransportSink, PreparedEndpointReference) {
        let endpoint_id = EndpointId::new("responses").expect("Responses endpoint id is valid");
        let endpoint = EndpointRegistry::builtin()
            .expect("builtin endpoints register")
            .prepare(&endpoint_id, RawEndpointConfig::default())
            .expect("Responses endpoint prepares");
        let mut endpoints = crate::endpoints::PreparedEndpointTable::new();
        let key = endpoints
            .push(endpoint)
            .expect("Responses endpoint is stored");
        let sink = WebSocketTransportSink::new(
            clock,
            vec![url],
            "model".to_owned(),
            crate::transport::http::TransportSinkConfig::default(),
            WebSocketTransportConfig::default(),
            endpoints,
            true,
        )
        .expect("Responses sink builds");
        (sink, PreparedEndpointReference { key, endpoint_id })
    }

    fn responses_turn(reference: PreparedEndpointReference) -> PreparedTurn {
        let operation = PreparedWsOperation::new(
            [PreparedWsMessage::text(
                Bytes::from_static(
                    br#"{"type":"response.create","model":"model","input":[{"role":"user","content":"hi"}]}"#,
                ),
                PreparedWsMessageRole::MeasuredInput,
            )],
            None,
        );
        PreparedTurn {
            runtime_session_id: "request".to_owned(),
            request: crate::transport::core::Request {
                uuid: Uuid::from_u128(43),
                input_length: 1,
                max_output_tokens: 1,
                prompt_text: None,
                body: Some(RequestBody::WebSocket(Arc::new(operation))),
                headers: BTreeMap::new(),
                parameters: BTreeMap::new(),
                endpoint_path: None,
                streaming: true,
                x_correlation_id: None,
                is_final_turn: true,
                cancel_after_ns: None,
                url_index: None,
                image_count: None,
                recorded_api_time_ns: None,
                recorded_ttft_ns: None,
            },
            model: "model".to_owned(),
            endpoint: PreparedEndpointBinding::Prepared(reference),
            endpoint_aware: true,
            data_policy: TurnDataPolicy::ordinary(),
            deferred: None,
        }
    }

    #[test]
    fn reused_responses_socket_quarantines_stale_response_before_raw_attribution() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test listener binds");
            let address = listener.local_addr().expect("listener has an address");
            let server = tokio::task::spawn_local(async move {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                let mut socket = accept_async(stream).await.expect("server upgrades client");
                let request = match socket.next().await {
                    Some(Ok(Message::Text(payload))) => {
                        serde_json::from_str::<Value>(&payload).expect("request is JSON")
                    }
                    message => panic!("expected correlated text request, got {message:?}"),
                };
                let operation_id = request["metadata"]["_aiperf_ws_operation"]
                    .as_str()
                    .expect("request has an operation marker");
                for event in [
                    serde_json::json!({
                        "type":"response.created",
                        "response":{"id":"stale","metadata":{"_aiperf_ws_operation":"previous"}},
                    }),
                    serde_json::json!({
                        "type":"response.output_text.delta",
                        "response_id":"stale",
                        "delta":"wrong",
                    }),
                    serde_json::json!({
                        "type":"response.completed",
                        "response":{"id":"stale","status":"completed"},
                    }),
                    serde_json::json!({
                        "type":"response.created",
                        "response":{"id":"current","metadata":{"_aiperf_ws_operation":operation_id}},
                    }),
                    serde_json::json!({
                        "type":"response.output_text.delta",
                        "response_id":"current",
                        "delta":"right",
                    }),
                    serde_json::json!({
                        "type":"response.completed",
                        "response":{
                            "id":"current",
                            "status":"completed",
                            "output":[{"content":[{"text":"right"}]}],
                        },
                    }),
                ] {
                    socket
                        .send(Message::Text(event.to_string().into()))
                        .await
                        .expect("server sends event");
                }
                while let Some(message) = socket.next().await {
                    if matches!(message, Ok(Message::Close(_))) {
                        socket.flush().await.expect("server flushes Close");
                        break;
                    }
                }
            });

            let clock: Rc<dyn Clock> = RealClock::new();
            let url = format!("ws://{address}/v1/responses");
            let cached = connect(
                &Url::parse(&url).expect("test URL is valid"),
                &BTreeMap::new(),
                &crate::transport::http::config::ClientConfig::default(),
                WebSocketConfig::default(),
                clock.clone(),
                None,
            )
            .await
            .expect("cached socket upgrades");
            let (sink, reference) = responses_sink(clock, url.clone());
            let now_ns = sink.clock.now_ns();
            sink.idle.borrow_mut().push(CachedSocket {
                affinity_key: None,
                continuation_id: None,
                url,
                headers: BTreeMap::new(),
                connected_ns: now_ns,
                retained_ns: now_ns,
                socket: cached,
            });
            let observer = TerminalObserver::default();
            let result = sink
                .dispatch_inner(responses_turn(reference), &observer, &|_| {}, None)
                .await
                .expect("correlated response completes");

            assert_eq!(result.outcome.response_text, "right");
            assert_eq!(result.record.responses.len(), 3);
            assert!(result.record.responses.iter().all(|response| match response {
                Response::Text(response) => !response.text.contains("stale"),
                _ => true,
            }));
            assert_eq!(
                observer.0.borrow().as_slice(),
                [ReplayTerminalStatus::Completed]
            );
            server.abort();
        });
    }

    fn realtime_continuation(reference: PreparedEndpointReference) -> PreparedTurn {
        let operation = PreparedWsOperation::new(
            [
                PreparedWsMessage::text(
                    Bytes::from_static(
                        br#"{"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_text","text":"next"}]}}"#,
                    ),
                    PreparedWsMessageRole::MeasuredInput,
                ),
                PreparedWsMessage::text(
                    Bytes::from_static(
                        br#"{"type":"response.create","response":{"modalities":["text"]}}"#,
                    ),
                    PreparedWsMessageRole::Control,
                ),
            ],
            None,
        )
        .requiring_affinity_state();
        PreparedTurn {
            runtime_session_id: "session".to_owned(),
            request: crate::transport::core::Request {
                uuid: Uuid::from_u128(42),
                input_length: 1,
                max_output_tokens: 1,
                prompt_text: None,
                body: Some(RequestBody::WebSocket(Arc::new(operation))),
                headers: BTreeMap::new(),
                parameters: BTreeMap::new(),
                endpoint_path: None,
                streaming: true,
                x_correlation_id: Some("session".to_owned()),
                is_final_turn: true,
                cancel_after_ns: None,
                url_index: None,
                image_count: None,
                recorded_api_time_ns: None,
                recorded_ttft_ns: None,
            },
            model: "model".to_owned(),
            endpoint: PreparedEndpointBinding::Prepared(reference),
            endpoint_aware: true,
            data_policy: TurnDataPolicy::ordinary(),
            deferred: None,
        }
    }

    #[derive(Clone, Copy)]
    enum RealtimeAffinityLoss {
        Rotation,
        KeepaliveExpiry,
        Eviction,
        RouteChange,
    }

    async fn assert_realtime_affinity_loss_is_a_failed_dispatch(loss: RealtimeAffinityLoss) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("test listener binds");
        let address = listener.local_addr().expect("listener has an address");
        let expected_connections = match loss {
            RealtimeAffinityLoss::Rotation
            | RealtimeAffinityLoss::KeepaliveExpiry
            | RealtimeAffinityLoss::RouteChange => 2,
            RealtimeAffinityLoss::Eviction => 1,
        };
        let server = tokio::task::spawn_local(async move {
            for _ in 0..expected_connections {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                tokio::task::spawn_local(async move {
                    let mut socket = accept_async(stream).await.expect("server upgrades client");
                    while let Some(message) = socket.next().await {
                        if matches!(message, Ok(Message::Close(_))) {
                            socket
                                .flush()
                                .await
                                .expect("server flushes reciprocal Close");
                            break;
                        }
                    }
                });
            }
        });

        let clock: Rc<dyn Clock> = RealClock::new();
        let url = format!("ws://{address}/v1/realtime");
        let mut transport = crate::transport::http::TransportSinkConfig::default();
        if matches!(loss, RealtimeAffinityLoss::KeepaliveExpiry) {
            transport.client.keepalive_ns = Some(1);
        }
        let (sink, reference) = realtime_sink(clock.clone(), url.clone(), transport);
        if !matches!(loss, RealtimeAffinityLoss::Eviction) {
            let cached = connect(
                &Url::parse(&url).expect("test URL is valid"),
                &BTreeMap::new(),
                &crate::transport::http::config::ClientConfig::default(),
                WebSocketConfig::default(),
                clock.clone(),
                None,
            )
            .await
            .expect("cached socket upgrades");
            let now_ns = clock.now_ns();
            let connected_ns = if matches!(loss, RealtimeAffinityLoss::Rotation) {
                now_ns.saturating_sub(SOCKET_ROTATION_NS)
            } else {
                now_ns
            };
            let retained_ns = if matches!(loss, RealtimeAffinityLoss::KeepaliveExpiry) {
                now_ns.saturating_sub(2)
            } else {
                now_ns
            };
            let cached_url = if matches!(loss, RealtimeAffinityLoss::RouteChange) {
                "ws://previous-route.invalid/v1/realtime".to_owned()
            } else {
                url.clone()
            };
            sink.idle.borrow_mut().push(CachedSocket {
                affinity_key: Some("session".to_owned()),
                continuation_id: None,
                url: cached_url,
                headers: BTreeMap::new(),
                connected_ns,
                retained_ns,
                socket: cached,
            });
        }
        let observer = TerminalObserver::default();
        let result = sink
            .dispatch_inner(realtime_continuation(reference), &observer, &|_| {}, None)
            .await
            .expect("lost affinity is a per-request failed dispatch");

        assert_eq!(result.outcome.terminal, ReplayTerminalStatus::Failed);
        assert_eq!(
            result.record.error.as_ref().map(|error| error.kind),
            Some(ErrorKind::Protocol)
        );
        assert!(result.record.end_ns.is_some(), "failed record is terminal");
        assert!(
            !result.request_payload.is_empty(),
            "raw failed request is retained"
        );
        assert_eq!(
            observer.0.borrow().as_slice(),
            [ReplayTerminalStatus::Failed]
        );
        sink.shutdown_idle_sockets().await;
        server
            .await
            .expect("server accepts every expected connection");
    }

    #[test]
    fn rotated_realtime_affinity_is_a_typed_failed_dispatch() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            assert_realtime_affinity_loss_is_a_failed_dispatch(RealtimeAffinityLoss::Rotation)
                .await;
        });
    }

    #[test]
    fn expired_realtime_affinity_is_a_typed_failed_dispatch() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            assert_realtime_affinity_loss_is_a_failed_dispatch(
                RealtimeAffinityLoss::KeepaliveExpiry,
            )
            .await;
        });
    }

    #[test]
    fn evicted_realtime_affinity_is_a_typed_failed_dispatch() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            assert_realtime_affinity_loss_is_a_failed_dispatch(RealtimeAffinityLoss::Eviction)
                .await;
        });
    }

    #[test]
    fn route_changed_realtime_affinity_is_a_typed_failed_dispatch() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            assert_realtime_affinity_loss_is_a_failed_dispatch(RealtimeAffinityLoss::RouteChange)
                .await;
        });
    }

    #[test]
    fn fallback_url_preserves_authority_and_path() {
        assert_eq!(
            websocket_fallback_url("wss://example.test/v1/responses?x=1").expect("URL maps"),
            "https://example.test/v1/responses?x=1"
        );
    }

    #[test]
    fn rotation_happens_before_the_sixty_minute_service_limit() {
        assert!(SOCKET_ROTATION_NS < SOCKET_SERVICE_LIMIT_NS);
        assert_eq!(SOCKET_ROTATION_NS, 55 * 60 * 1_000_000_000);
        assert_eq!(SOCKET_SERVICE_LIMIT_NS, 60 * 60 * 1_000_000_000);
    }

    #[test]
    fn unmatched_affinity_socket_is_evicted_only_at_pool_capacity() {
        assert!(!socket_pool_requires_eviction(1, 3));
        assert!(!socket_pool_requires_eviction(3, 3));
        assert!(socket_pool_requires_eviction(4, 3));
    }

    #[test]
    fn timed_out_affinity_waiter_does_not_leak_the_gate() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            let clock: Rc<dyn Clock> = RealClock::new();
            let gates = RefCell::new(BTreeMap::new());
            let gate = Rc::new(Semaphore::new(1));
            gates
                .borrow_mut()
                .insert("session".to_owned(), gate.clone());
            let owner_cleanup = AffinityGateCleanup {
                gates: &gates,
                affinity_key: "session",
                gate: gate.clone(),
            };
            let owner = gate.acquire().await.expect("owner acquires affinity");

            let waiter_gate = gate.clone();
            let waiter_cleanup = AffinityGateCleanup {
                gates: &gates,
                affinity_key: "session",
                gate: waiter_gate.clone(),
            };
            let result = acquire_before_deadline(
                waiter_gate.as_ref(),
                clock.clone(),
                Some(clock.now_ns().saturating_add(1_000_000)),
            )
            .await;
            assert!(result.is_err(), "waiter must reach its Clock deadline");
            drop(result);
            drop(waiter_cleanup);
            drop(waiter_gate);
            assert_eq!(gates.borrow().len(), 1, "owner still owns the gate");

            drop(owner);
            drop(owner_cleanup);
            assert!(gates.borrow().is_empty(), "last owner removes the gate");
        });
    }

    #[test]
    fn worker_shutdown_sends_close_for_every_idle_socket() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test listener binds");
            let address = listener.local_addr().expect("listener has an address");
            let (close_seen, close_result) = oneshot::channel();
            let (allow_reply, reply_allowed) = oneshot::channel();
            tokio::task::spawn_local(async move {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                let mut socket = accept_async(stream).await.expect("server upgrades client");
                let is_close = matches!(socket.next().await, Some(Ok(Message::Close(_))));
                let _ = close_seen.send(is_close);
                reply_allowed.await.expect("test releases reciprocal close");
                socket
                    .flush()
                    .await
                    .expect("server flushes reciprocal Close");
            });

            let clock: Rc<dyn Clock> = RealClock::new();
            let url =
                Url::parse(&format!("ws://{address}/v1/responses")).expect("test URL is valid");
            let socket = connect(
                &url,
                &BTreeMap::new(),
                &crate::transport::http::config::ClientConfig::default(),
                WebSocketConfig::default(),
                clock.clone(),
                None,
            )
            .await
            .expect("client upgrades server");
            let sink = WebSocketTransportSink::new(
                clock,
                vec![url.to_string()],
                "model".to_owned(),
                crate::transport::http::TransportSinkConfig::default(),
                WebSocketTransportConfig::default(),
                crate::endpoints::PreparedEndpointTable::new(),
                false,
            )
            .expect("test sink builds");
            sink.idle.borrow_mut().push(CachedSocket {
                affinity_key: None,
                continuation_id: None,
                url: url.to_string(),
                headers: BTreeMap::new(),
                connected_ns: sink.clock.now_ns(),
                retained_ns: sink.clock.now_ns(),
                socket,
            });

            let shutdown = WorkerSink::shutdown(&sink);
            tokio::pin!(shutdown);
            let has_close = tokio::select! {
                result = &mut shutdown => panic!(
                    "shutdown completed before reciprocal Close was allowed: {result:?}"
                ),
                result = tokio::time::timeout(
                    std::time::Duration::from_secs(1),
                    close_result,
                ) => result
                    .expect("server observes shutdown")
                    .expect("server reports close"),
            };
            assert!(has_close);
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(20), &mut shutdown)
                    .await
                    .is_err(),
                "shutdown must wait for the reciprocal Close rather than only flush its own frame"
            );
            allow_reply.send(()).expect("test releases server reply");
            tokio::time::timeout(std::time::Duration::from_secs(1), &mut shutdown)
                .await
                .expect("reciprocal close completes before the Clock bound")
                .expect("worker shutdown succeeds");
            assert!(sink.idle.borrow().is_empty());
        });
    }

    #[test]
    fn close_handshake_stops_at_the_clock_deadline_when_peer_is_silent() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test listener binds");
            let address = listener.local_addr().expect("listener has an address");
            let (close_seen, close_result) = oneshot::channel();
            tokio::task::spawn_local(async move {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                let mut socket = accept_async(stream).await.expect("server upgrades client");
                let is_close = matches!(socket.next().await, Some(Ok(Message::Close(_))));
                let _ = close_seen.send(is_close);
                std::future::pending::<()>().await;
            });

            let clock: Rc<dyn Clock> = RealClock::new();
            let url =
                Url::parse(&format!("ws://{address}/v1/responses")).expect("test URL is valid");
            let socket = connect(
                &url,
                &BTreeMap::new(),
                &crate::transport::http::config::ClientConfig::default(),
                WebSocketConfig::default(),
                clock.clone(),
                None,
            )
            .await
            .expect("client upgrades server");
            let deadline_ns = clock.now_ns().saturating_add(20_000_000);
            tokio::time::timeout(
                std::time::Duration::from_secs(1),
                close_socket(socket, clock, Some(deadline_ns)),
            )
            .await
            .expect("silent peer cannot hold shutdown past its Clock deadline");
            assert!(
                close_result
                    .await
                    .expect("server reports whether it received Close")
            );
        });
    }

    #[test]
    fn stalled_response_observer_does_not_suspend_pong_or_operation_deadline() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime builds");
        tokio::task::LocalSet::new().block_on(&runtime, async {
            let listener = TcpListener::bind("127.0.0.1:0")
                .await
                .expect("test listener binds");
            let address = listener.local_addr().expect("listener has an address");
            let (pong_seen, pong_result) = oneshot::channel();
            tokio::task::spawn_local(async move {
                let (stream, _) = listener.accept().await.expect("server accepts client");
                let mut socket = accept_async(stream).await.expect("server upgrades client");
                loop {
                    match socket.next().await {
                        Some(Ok(Message::Text(_))) => break,
                        Some(Ok(_)) => {}
                        Some(Err(error)) => panic!("server read failed: {error}"),
                        None => panic!("client closed before its application message"),
                    }
                }
                socket
                    .send(Message::Ping(Bytes::from_static(b"observer")))
                    .await
                    .expect("server sends Ping");
                let has_pong = loop {
                    match socket.next().await {
                        Some(Ok(Message::Pong(payload))) => {
                            break payload == Bytes::from_static(b"observer");
                        }
                        Some(Ok(_)) => {}
                        Some(Err(_)) | None => break false,
                    }
                };
                let _ = pong_seen.send(has_pong);
                std::future::pending::<()>().await;
            });

            let clock: Rc<dyn Clock> = RealClock::new();
            let url =
                Url::parse(&format!("ws://{address}/v1/responses")).expect("test URL is valid");
            let socket = connect(
                &url,
                &BTreeMap::new(),
                &crate::transport::http::config::ClientConfig::default(),
                WebSocketConfig::default(),
                clock.clone(),
                None,
            )
            .await
            .expect("client upgrades server");
            let operation = PreparedWsOperation::new(
                [PreparedWsMessage::text(
                    Bytes::from_static(br#"{"type":"response.create"}"#),
                    PreparedWsMessageRole::MeasuredInput,
                )],
                None,
            );
            let now_ns = clock.now_ns();
            let mut driver = SocketOperationDriver::start(
                socket,
                clock,
                &operation,
                ApplicationQueueLimits::new(1, 64),
                DriverTiming {
                    deadline_ns: Some(now_ns.saturating_add(50_000_000)),
                    rotation_deadline_ns: now_ns.saturating_add(1_000_000_000),
                    ping_interval_ns: 1_000_000_000,
                    stream_idle_timeout_ns: 1_000_000_000,
                    cancel_after_ns: None,
                },
                64,
            )
            .expect("driver starts");
            while !matches!(driver.next().await.unwrap(), DriverEvent::Flushed { .. }) {}
            let mut pending = Some(crate::endpoints::ParsedResponse {
                perf_ns: 1,
                data: None,
                usage: None,
                sources: None,
            });
            let failure = next_while_delivering_response(
                &mut driver,
                &mut pending,
                Some(&StalledResponseObserver),
                &TurnOperationState::default(),
            )
            .await
            .expect_err("operation deadline must remain live");

            assert_eq!(failure.kind, ErrorKind::Timeout);
            assert!(
                tokio::time::timeout(std::time::Duration::from_secs(1), pong_result)
                    .await
                    .expect("server observes control progress")
                    .expect("server received the Pong")
            );
        });
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
    fn cancellation_has_canceled_terminal_and_raw_timestamp() {
        assert_eq!(
            failure_terminal(ErrorKind::Cancelled),
            ReplayTerminalStatus::Canceled
        );
        assert_eq!(
            cancellation_timestamp(ReplayTerminalStatus::Canceled, 42),
            Some(42)
        );
        assert_eq!(
            failure_terminal(ErrorKind::Timeout),
            ReplayTerminalStatus::Failed
        );
        assert_eq!(
            cancellation_timestamp(ReplayTerminalStatus::Failed, 42),
            None
        );
        let mut raw_record = RequestRecord::started(1);
        raw_record.cancellation_ns = cancellation_timestamp(ReplayTerminalStatus::Canceled, 42);
        assert!(raw_record.was_cancelled());
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
            &operation
                .to_artifact_bytes()
                .expect("operation capture serializes"),
        )
        .expect("operation capture is JSON");
        assert_eq!(captured["transport"], "websocket");
        assert_eq!(captured["messages"].as_array().map(Vec::len), Some(2));
        assert_eq!(captured["messages"][0]["role"], "measured_input");
        assert_eq!(captured["messages"][1]["role"], "control");
    }

    #[test]
    fn retry_raw_capture_discards_every_abandoned_attempt_fact() {
        let mut record = RequestRecord::started(10);
        record.recv_start_ns = Some(20);
        record.status = Some(101);
        record.responses.push(Response::Text(TextResponse {
            perf_ns: 20,
            body: Bytes::from_static(br#"{"type":"ignored"}"#),
            text: r#"{"type":"ignored"}"#.to_owned(),
            content_type: Some("application/json".to_owned()),
        }));
        record.error = Some(ErrorDetails::other("abandoned"));

        reset_record_for_retry(&mut record);

        assert_eq!(record.recv_start_ns, None);
        assert_eq!(record.status, None);
        assert!(record.responses.is_empty());
        assert_eq!(record.error, None);
    }

    #[test]
    fn direct_request_executor_advertises_live_response_streaming() {
        let sink = WebSocketTransportSink::new(
            RealClock::new(),
            vec!["ws://127.0.0.1:1".to_owned()],
            "model".to_owned(),
            crate::transport::http::TransportSinkConfig::default(),
            WebSocketTransportConfig::default(),
            crate::endpoints::PreparedEndpointTable::new(),
            false,
        )
        .expect("test sink builds");

        assert!(RequestExecutor::supports_response_streaming(&sink));
    }

    #[test]
    fn tungstenite_buffer_reserves_frame_header_and_control_capacity() {
        let config = WebSocketTransportConfig {
            max_queued_bytes: 1_048_576,
            max_frame_bytes: 1_048_576,
            ..WebSocketTransportConfig::default()
        };
        let sink = WebSocketTransportSink::new(
            RealClock::new(),
            vec!["ws://127.0.0.1:1".to_owned()],
            "model".to_owned(),
            crate::transport::http::TransportSinkConfig::default(),
            config,
            crate::endpoints::PreparedEndpointTable::new(),
            false,
        )
        .expect("test sink builds");
        let tungstenite = sink.websocket_config();

        assert!(
            tungstenite.max_write_buffer_size >= 1_048_576 + 14 + 131,
            "a maximum client data frame and one maximum control frame must fit together"
        );
    }
}
