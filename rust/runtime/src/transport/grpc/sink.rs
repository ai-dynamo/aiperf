// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduled-runtime composition for the protocol-v2 native gRPC transport.
//!
//! This module intentionally accepts only worker-local prepared endpoints. It
//! has no protocol-v1 endpoint adapter, no closed endpoint enum dispatch, and
//! no Python transport fallback. The scheduler-facing command/result types are
//! temporarily shared with the established online execution-placement seam;
//! all actual wire IO and timing are owned by `aiperf-transport-grpc`.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::Value;

use crate::clock::Clock;
use crate::endpoints::{
    EndpointKey, ParsedResponse, PreparedEndpoint, PreparedEndpointTable,
    RequestRecord as EndpointRequestRecord, ResponseData, ServerResponse, Turn, UsageView,
};
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest, RequestTrace};
use crate::transport::core::{
    ErrorDetails, ErrorKind, RequestRecord, Response, TextResponse, TraceData,
};
use crate::transport::grpc::{
    ConnectionReuseStrategy as GrpcConnectionReuseStrategy, GrpcBindingRegistry, GrpcClientConfig,
    GrpcErrorKind, GrpcRequestConfig, GrpcRequestRecord, GrpcTransport,
};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    Dispatchable, ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
    RequestSink,
};
use uuid::Uuid;

use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata};
use crate::multiturn::TurnToSend;
use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use crate::transport::core::{DispatchResult, MeasuredContext, MeasuredOutcome, Request};
use crate::transport::http::{Dispatcher, PreparedHttpEndpoint, PreparedTurn, RequestExecutor};

/// Worker-local gRPC scheduled sink policy.
#[derive(Clone, Debug)]
pub struct GrpcTransportSinkConfig {
    /// Low-level Clock deadline and message-size policy.
    pub client: GrpcClientConfig,
    /// Channel reuse policy.
    pub connection_reuse: GrpcConnectionReuseStrategy,
    /// Optional additional correlation metadata name.
    pub session_header: Option<String>,
}

/// Transport-native request retaining its protocol-v2 prepared endpoint.
///
/// The richer runner placement seam uses [`PreparedTurn`] directly until
/// its historical HTTP naming is generalized. This wrapper keeps the canonical
/// [`RequestSink`] extension seam available without discarding the dense
/// endpoint reference required by gRPC serialization.
#[derive(Clone, Debug)]
pub struct GrpcRequest {
    turn: PreparedTurn,
}

impl GrpcRequest {
    /// Wrap one worker-resolved protocol-v2 turn.
    pub fn new(turn: PreparedTurn) -> Self {
        Self { turn }
    }

    /// Recover the richer prepared turn.
    pub fn into_inner(self) -> PreparedTurn {
        self.turn
    }
}

impl Dispatchable for GrpcRequest {
    fn uuid(&self) -> Uuid {
        self.turn.request.uuid
    }

    fn input_length(&self) -> usize {
        self.turn.request.input_length
    }

    fn max_output_tokens(&self) -> usize {
        self.turn.request.max_output_tokens
    }
}

impl Default for GrpcTransportSinkConfig {
    fn default() -> Self {
        Self {
            client: GrpcClientConfig::default(),
            connection_reuse: GrpcConnectionReuseStrategy::Pooled,
            session_header: None,
        }
    }
}

/// Prepared protocol-v2 scheduled sink over native gRPC.
pub struct GrpcTransportSink {
    transport: GrpcTransport,
    clock: Rc<dyn Clock>,
    start_ns: Cell<i64>,
    base_urls: Vec<String>,
    model: String,
    connection_reuse: GrpcConnectionReuseStrategy,
    binding_registry: GrpcBindingRegistry,
    prepared_endpoints: Option<Rc<PreparedEndpointTable>>,
    prepared_bindings: Vec<Box<dyn crate::transport::grpc::GrpcEndpointBinding>>,
    /// Worker-local metric accumulator for the scheduled runner's measured
    /// execution path (`None` until `configure_measurement`).
    measurement: RefCell<Option<Rc<NativeMetricsObserver>>>,
}

impl std::fmt::Debug for GrpcTransportSink {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GrpcTransportSink")
            .field("base_urls", &self.base_urls)
            .field("model", &self.model)
            .field("connection_reuse", &self.connection_reuse)
            .field("prepared_bindings", &self.prepared_bindings.len())
            .finish_non_exhaustive()
    }
}

impl GrpcTransportSink {
    /// Construct a v2-only sink over one non-empty gRPC target list.
    pub fn new(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        base_urls: &[String],
        model: impl Into<String>,
        config: GrpcTransportSinkConfig,
        binding_registry: GrpcBindingRegistry,
    ) -> Result<Self> {
        ensure!(!base_urls.is_empty(), "at least one gRPC URL is required");
        ensure!(
            base_urls.len() <= u32::MAX as usize,
            "gRPC URL count exceeds the u32 request-index representation"
        );
        if let Some(header) = &config.session_header {
            ensure!(
                !header.trim().is_empty(),
                "gRPC session metadata name must be non-empty"
            );
        }
        let mut transport =
            GrpcTransport::new(clock.clone(), config.client, base_urls.iter().cloned())?;
        if let Some(header) = config.session_header {
            transport = transport.with_session_header(header);
        }
        Ok(Self {
            transport,
            clock,
            start_ns: Cell::new(start_ns),
            base_urls: base_urls.to_vec(),
            model: model.into(),
            connection_reuse: config.connection_reuse,
            binding_registry,
            prepared_endpoints: None,
            prepared_bindings: Vec::new(),
            measurement: RefCell::new(None),
        })
    }

    /// Prepare a dense gRPC wire binding for every worker-local endpoint key.
    pub fn with_prepared_endpoints(mut self, endpoints: Rc<PreparedEndpointTable>) -> Result<Self> {
        let mut bindings = Vec::with_capacity(endpoints.len());
        for index in 0..endpoints.len() {
            let index = u32::try_from(index).context("prepared endpoint index exceeds u32")?;
            let endpoint = endpoints.get(EndpointKey::from_index(index))?;
            bindings.push(
                self.binding_registry
                    .prepare(&crate::endpoints::EndpointId::new(
                        endpoint.descriptor().id,
                    )?)
                    .with_context(|| {
                        format!(
                            "preparing gRPC binding for endpoint {:?}",
                            endpoint.descriptor().id
                        )
                    })?,
            );
        }
        self.prepared_endpoints = Some(endpoints);
        self.prepared_bindings = bindings;
        Ok(self)
    }

    /// Set the benchmark origin after channel-stack startup.
    pub fn set_run_origin(&self, start_ns: i64) {
        self.start_ns.set(start_ns);
    }

    fn ms(&self, ns: i64) -> f64 {
        ns.saturating_sub(self.start_ns.get()) as f64 / 1_000_000.0
    }

    fn selected_url(&self, index: Option<u32>) -> Option<String> {
        self.base_urls.get(index.unwrap_or(0) as usize).cloned()
    }

    /// Execute one scheduler-free v2 command and retain compatibility raw facts.
    /// Access the worker-local measurement observer, erroring if the measured
    /// execution path is used before `configure_measurement`.
    fn measurement_observer(&self) -> Result<Rc<NativeMetricsObserver>> {
        self.measurement.borrow().clone().ok_or_else(|| {
            anyhow::anyhow!("worker-local measurement was not configured before dispatch")
        })
    }

    /// Register coordinator-known arrival facts on `observer`, dispatch the
    /// prepared gRPC turn into it, and record the terminal transport facts — the
    /// gRPC twin of [`TransportSink::dispatch_measured`].
    ///
    /// [`TransportSink::dispatch_measured`]: crate::transport::http::TransportSink::dispatch_measured
    pub async fn dispatch_measured(
        &self,
        observer: &NativeMetricsObserver,
        turn: PreparedTurn,
        context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        let uuid = turn.request.uuid;
        observer.register_metadata(uuid, context.metadata.clone());
        observer.on_arrival(
            uuid,
            context.arrival_ms,
            context.input_length,
            context.requested_output_length,
        );
        let result = self.dispatch_collect(turn, observer, on_first_token).await;
        match &result {
            Ok(collected) => {
                let outcome = &collected.outcome;
                observer.record_response(
                    uuid,
                    NativeResponseMetadata {
                        start_ns: Some(outcome.start_ns),
                        end_ns: Some(outcome.end_ns),
                        prompt_tokens: outcome.prompt_tokens,
                        completion_tokens: outcome.completion_tokens,
                        http: outcome.http,
                    },
                );
            }
            Err(_) => {
                let now = self.clock.now_ns();
                observer.on_terminal(uuid, ReplayTerminalStatus::Failed);
                observer.record_response(
                    uuid,
                    NativeResponseMetadata {
                        start_ns: Some(now),
                        end_ns: Some(now),
                        ..NativeResponseMetadata::default()
                    },
                );
            }
        }
        result
    }

    pub async fn dispatch_collect(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        ensure!(
            turn.endpoint_aware,
            "native gRPC execution requires endpoint-aware protocol-v2 materialization"
        );
        let model = turn.model;
        let PreparedHttpEndpoint::Prepared(reference) = turn.endpoint;
        let table = self.prepared_endpoints.as_ref().ok_or_else(|| {
            anyhow::anyhow!("gRPC worker received a prepared endpoint without a prepared table")
        })?;
        let endpoint = table.get(reference.key)?;
        ensure!(
            endpoint.descriptor().id == reference.endpoint_id.as_str(),
            "prepared endpoint key {} resolved to {:?}, expected {:?}",
            reference.key.index(),
            endpoint.descriptor().id,
            reference.endpoint_id.as_str()
        );
        let binding = self
            .prepared_bindings
            .get(reference.key.index() as usize)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "gRPC endpoint key {} has no prepared wire binding",
                    reference.key.index()
                )
            })?;
        ensure!(
            binding.endpoint_id() == &reference.endpoint_id,
            "prepared gRPC binding identity does not match endpoint reference"
        );
        self.dispatch_endpoint(
            turn.request,
            &model,
            endpoint,
            binding.as_ref(),
            observer,
            on_first_token,
        )
        .await
    }

    async fn dispatch_endpoint(
        &self,
        request: Request,
        model: &str,
        endpoint: &dyn PreparedEndpoint,
        binding: &dyn crate::transport::grpc::GrpcEndpointBinding,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        let Request {
            uuid,
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
        } = request;
        ensure!(
            parameters.is_empty(),
            "gRPC endpoint requests do not support URL query parameters"
        );
        if let Some(endpoint_path) = endpoint_path.as_deref() {
            let expected_path = endpoint
                .config()
                .as_raw()
                .path
                .as_deref()
                .or_else(|| {
                    streaming
                        .then_some(endpoint.descriptor().streaming_path)
                        .flatten()
                })
                .or(endpoint.descriptor().endpoint_path);
            ensure!(
                Some(endpoint_path) == expected_path,
                "gRPC binding ignores only the selected dialect's HTTP path; authored per-turn endpoint_path {endpoint_path:?} is unsupported"
            );
        }
        ensure!(
            request_body.is_none() || request_body_bytes.is_none(),
            "a gRPC request cannot supply both JSON and serialized canonical bodies"
        );
        let body = match (request_body, request_body_bytes) {
            (Some(value), None) => value,
            (None, Some(bytes)) => serde_json::from_slice(&bytes)
                .context("decoding prepared endpoint JSON before gRPC serialization")?,
            (None, None) => anyhow::bail!(
                "gRPC protocol-v2 execution requires a canonical prepared endpoint body"
            ),
            (Some(_), Some(_)) => unreachable!("exclusivity checked above"),
        };
        observer.on_admit(uuid, self.ms(self.clock.now_ns()), 0);
        let mut endpoint_metrics = ObservedEndpointMetrics {
            num_images: nonzero_usize(endpoint.extract_payload_inputs(&body).image_count as usize),
            ..ObservedEndpointMetrics::default()
        };
        let mut metadata = endpoint.headers().clone();
        metadata.extend(headers);
        let mut config = GrpcRequestConfig::new(model)
            .streaming(streaming)
            .reuse(self.connection_reuse)
            .request_id(uuid.to_string())
            .final_turn(is_final_turn);
        config.metadata = metadata;
        config.cancel_after_ns = cancel_after_ns;
        config.correlation_id = x_correlation_id;
        config.url_index = url_index;

        let first_token_released = Cell::new(false);
        let mut first_response_filter = |ttft_ns: i64, response: &Value| {
            let server_response = ServerResponse::from_json(
                u64::try_from(self.clock.now_ns()).unwrap_or(u64::MAX),
                response.clone(),
            );
            if !meaningful_response(endpoint, &server_response) {
                return false;
            }
            if !first_token_released.replace(true) {
                on_first_token(ttft_ns);
            }
            true
        };
        let record = self
            .transport
            .send_request(binding, &config, &body, &mut first_response_filter)
            .await;

        let mut parsed_content = false;
        let mut parse_failed = false;
        let mut response_text = String::new();
        let mut model_response = ModelResponseMetadata::default();
        let mut usage = ObservedUsage::default();
        let mut endpoint_responses = Vec::with_capacity(record.responses.len());
        for response in &record.responses {
            let server_response = ServerResponse {
                perf_ns: u64::try_from(response.perf_ns).unwrap_or(u64::MAX),
                json: Some(response.json.clone()),
                raw: serde_json::to_string(&response.json).ok(),
            };
            endpoint_responses.push(server_response.clone());
            let parsed = match endpoint.parse_response(&server_response) {
                Ok(parsed) => parsed,
                Err(error) => {
                    tracing::warn!(uuid = %uuid, error = %error, "gRPC endpoint response parsing failed");
                    parse_failed = true;
                    continue;
                }
            };
            let Some(parsed) = parsed else { continue };
            absorb_usage(&parsed, &mut usage);
            let Some(data) = parsed.data.as_ref() else {
                continue;
            };
            parsed_content = true;
            absorb_endpoint_metrics(data, &mut endpoint_metrics);
            let text = absorb_response_data(data, &mut model_response);
            response_text.push_str(&text);
            if endpoint.descriptor().produces_tokens {
                let at_ns = i64::try_from(parsed.perf_ns).unwrap_or(i64::MAX);
                if let ResponseData::TokenIds { token_ids } = data
                    && !token_ids.is_empty()
                {
                    if !first_token_released.replace(true) {
                        on_first_token(at_ns.saturating_sub(record.start_ns));
                    }
                    let timestamps = vec![self.ms(at_ns); token_ids.len()];
                    observer.on_output_tokens(uuid, &timestamps);
                } else if !text.is_empty() {
                    if !first_token_released.replace(true) {
                        on_first_token(at_ns.saturating_sub(record.start_ns));
                    }
                    observer.on_classified_token(uuid, self.ms(at_ns), token_kind(data));
                }
            }
        }
        if endpoint.captures_assistant_turn() {
            match endpoint.build_assistant_turn(&EndpointRequestRecord {
                responses: endpoint_responses,
            }) {
                Ok(Some(turn)) => model_response.assistant_message = assistant_message(&turn),
                Ok(None) => {}
                Err(error) => {
                    tracing::warn!(uuid = %uuid, error = %error, "gRPC assistant replay failed");
                    parse_failed = true;
                }
            }
        }
        let terminal = match record.error.as_ref().map(|error| error.kind) {
            Some(GrpcErrorKind::RequestCancellation) => ReplayTerminalStatus::Canceled,
            Some(_) => ReplayTerminalStatus::Failed,
            None if record
                .status
                .is_some_and(|status| (200..300).contains(&status))
                && parsed_content
                && !parse_failed =>
            {
                ReplayTerminalStatus::Completed
            }
            None => ReplayTerminalStatus::Failed,
        };
        absorb_grpc_error(&record, terminal, &mut model_response);
        observer.on_usage(uuid, usage);
        observer.on_endpoint_metrics(uuid, endpoint_metrics);
        observer.on_terminal(uuid, terminal);

        let prompt_tokens = usage
            .prompt_tokens
            .and_then(|value| u64::try_from(value).ok());
        let completion_tokens = usage
            .completion_tokens
            .and_then(|value| u64::try_from(value).ok());
        let outcome = TurnDispatchOutcome {
            start_ns: record.start_ns,
            end_ns: record.end_ns.unwrap_or_else(|| self.clock.now_ns()),
            terminal,
            response_text,
            model_response,
            prompt_tokens,
            completion_tokens,
            http: grpc_metrics_trace(&record),
        };
        let request_payload = Bytes::from(serde_json::to_vec(&body)?);
        let compatibility_record = compatibility_http_record(&record);
        Ok(DispatchResult {
            outcome,
            request_payload,
            record: compatibility_record,
        })
    }
}

#[async_trait(?Send)]
impl Dispatcher for GrpcTransportSink {
    async fn dispatch_collect(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        GrpcTransportSink::dispatch_collect(self, turn, observer, on_first_token).await
    }

    // The inherent gRPC `inference_dimensions` resolves from a `&TurnToSend`
    // (scheduled path); the `Dispatcher` seam is keyed on the transport-neutral
    // `Request`. Build the same dimensions gRPC produces — selected URL by
    // the request's url index, model falling back to the sink's model.
    fn inference_dimensions(&self, request: &Request) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: self.selected_url(request.url_index),
            model: Some(self.model.clone()),
        }
    }
}

#[async_trait(?Send)]
impl RequestExecutor for GrpcTransportSink {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        GrpcTransportSink::set_run_origin(self, start_ns);
        Ok(())
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: self.selected_url(turn.url_index),
            model: turn
                .effective_model
                .clone()
                .or_else(|| Some(self.model.clone())),
        }
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        let observer = NativeMetricsObserver::new(self.clock.clone(), origin_ns, config);
        *self.measurement.borrow_mut() = Some(Rc::new(observer));
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
            .dispatch_measured(&observer, turn, &context, on_first_token)
            .await?;
        let live_record = context
            .wants_live_record
            .then(|| {
                // Metrics-only (sketch) mode moves the record out of the observer
                // so its token storage is freed as the run streams; every other
                // mode clones it and leaves the authoritative copy for the drain.
                if context.consume_record {
                    observer.drain_terminal_record(uuid, 0)
                } else {
                    observer.snapshot_record(uuid, 0)
                }
            })
            .flatten();
        Ok(MeasuredOutcome {
            result,
            live_record,
        })
    }

    fn drain_records(&self, end_ns: i64) -> Result<Vec<(Uuid, RecordIngest)>> {
        match self.measurement.borrow_mut().take() {
            Some(observer) => Ok(observer
                .take_finalizer_at(end_ns)
                .finish_with_records()
                .records),
            None => Ok(Vec::new()),
        }
    }
}

#[async_trait(?Send)]
impl RequestSink<GrpcRequest> for GrpcTransportSink {
    async fn dispatch(&self, request: GrpcRequest, observer: &dyn RequestObserver) -> Result<()> {
        self.dispatch_collect(request.into_inner(), observer, &|_| {})
            .await?;
        Ok(())
    }
}

fn meaningful_response(endpoint: &dyn PreparedEndpoint, response: &ServerResponse) -> bool {
    endpoint
        .parse_response(response)
        .ok()
        .flatten()
        .and_then(|parsed| parsed.data)
        .is_some_and(|data| match data {
            ResponseData::Audio(audio) => !audio.audio_bytes.is_empty(),
            ResponseData::Images(images) => !images.images.is_empty(),
            ResponseData::TokenIds { token_ids } => !token_ids.is_empty(),
            ResponseData::Video(_) => true,
            other => !other.get_text().is_empty() || !endpoint.descriptor().produces_tokens,
        })
}

fn token_kind(data: &ResponseData) -> ObservedTokenKind {
    match data {
        ResponseData::Reasoning { reasoning, .. } if !reasoning.is_empty() => {
            ObservedTokenKind::Reasoning
        }
        _ => ObservedTokenKind::Output,
    }
}

fn absorb_response_data(data: &ResponseData, metadata: &mut ModelResponseMetadata) -> String {
    match data {
        ResponseData::Text { text } => append_text(&mut metadata.content, text),
        ResponseData::Reasoning { content, reasoning } => {
            metadata.content.get_or_insert_with(String::new);
            append_text(&mut metadata.reasoning, reasoning);
            if let Some(content) = content {
                append_text(&mut metadata.content, content);
            }
        }
        ResponseData::ToolCall {
            tool_call_text,
            content,
        } => {
            if let Some(content) = content {
                append_text(&mut metadata.content, content);
            }
            append_text(&mut metadata.content, tool_call_text);
        }
        ResponseData::TokenIds { token_ids } => {
            metadata
                .output_token_ids
                .get_or_insert_with(Vec::new)
                .extend_from_slice(token_ids);
        }
        ResponseData::Embeddings { .. }
        | ResponseData::Rankings { .. }
        | ResponseData::ImageRetrieval { .. }
        | ResponseData::Images(_)
        | ResponseData::Audio(_)
        | ResponseData::Video(_) => {}
    }
    data.get_text()
}

fn append_text(target: &mut Option<String>, text: &str) {
    target.get_or_insert_with(String::new).push_str(text);
}

fn absorb_usage(parsed: &ParsedResponse, observed: &mut ObservedUsage) {
    let Some(usage) = parsed.usage.as_ref().and_then(UsageView::from_value) else {
        return;
    };
    observed.prompt_tokens = usage
        .prompt_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_tokens);
    observed.completion_tokens = usage
        .completion_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.completion_tokens);
    observed.total_tokens = usage
        .total_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.total_tokens);
    observed.reasoning_tokens = usage
        .reasoning_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.reasoning_tokens);
    observed.prompt_cache_read_tokens = usage
        .prompt_cache_read_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_cache_read_tokens);
    observed.prompt_cache_write_tokens = usage
        .prompt_cache_write_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_cache_write_tokens);
    observed.prompt_cache_miss_tokens = usage
        .prompt_cache_miss_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_cache_miss_tokens);
    observed.prompt_audio_tokens = usage
        .prompt_audio_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.prompt_audio_tokens);
    observed.completion_audio_tokens = usage
        .completion_audio_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.completion_audio_tokens);
    observed.accepted_prediction_tokens = usage
        .accepted_prediction_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.accepted_prediction_tokens);
    observed.rejected_prediction_tokens = usage
        .rejected_prediction_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.rejected_prediction_tokens);
    observed.tool_use_prompt_tokens = usage
        .tool_use_prompt_tokens()
        .and_then(|value| usize::try_from(value).ok())
        .or(observed.tool_use_prompt_tokens);
    observed.prompt_audio_seconds = usage
        .prompt_audio_seconds()
        .or(observed.prompt_audio_seconds);
}

fn absorb_endpoint_metrics(data: &ResponseData, metrics: &mut ObservedEndpointMetrics) {
    let ResponseData::Video(video) = data else {
        return;
    };
    metrics.video_inference_seconds = video
        .inference_time_s
        .filter(|value| value.is_finite())
        .or(metrics.video_inference_seconds);
    metrics.video_peak_memory_mb = video
        .peak_memory_mb
        .filter(|value| value.is_finite())
        .or(metrics.video_peak_memory_mb);
}

fn assistant_message(turn: &Turn) -> Option<Value> {
    if let Some(message) = turn
        .raw_messages
        .as_ref()
        .and_then(|messages| messages.first())
    {
        return Some(message.clone());
    }
    let content = turn
        .texts
        .iter()
        .flat_map(|media| &media.contents)
        .cloned()
        .collect::<String>();
    (!content.is_empty()).then(|| {
        serde_json::json!({
            "role": turn.role.as_deref().unwrap_or("assistant"),
            "content": content,
        })
    })
}

fn absorb_grpc_error(
    record: &GrpcRequestRecord,
    terminal: ReplayTerminalStatus,
    metadata: &mut ModelResponseMetadata,
) {
    if terminal == ReplayTerminalStatus::Completed {
        metadata.error_kind = None;
        metadata.error_message = None;
        return;
    }
    if let Some(error) = &record.error {
        metadata.error_kind = Some(format!("grpc_{:?}", error.kind).to_ascii_lowercase());
        metadata.error_message = Some(error.message.clone());
    } else {
        metadata.error_kind = Some("grpc_incomplete_response".to_string());
        metadata.error_message = Some("gRPC request completed without parsed content".to_string());
    }
}

fn grpc_metrics_trace(record: &GrpcRequestRecord) -> RequestTrace {
    RequestTrace {
        stream_setup_ns: record
            .trace
            .response_receive_start_ns
            .map(|value| value.saturating_sub(record.start_ns)),
        connecting_ns: match (record.trace.connect_start_ns, record.trace.connect_end_ns) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        },
        sending_ns: record.trace.sending_ns(),
        waiting_ns: record.trace.waiting_ns(),
        receiving_ns: record.trace.receiving_ns(),
        duration_ns: record.trace.duration_ns(),
        connection_reused: Some(record.trace.channel_reused_ns.is_some()),
        data_sent_bytes: Some(record.trace.request_bytes_total),
        data_received_bytes: Some(record.trace.response_bytes_total),
        chunks_sent: Some(u64::from(record.trace.request_chunks_count)),
        chunks_received: Some(u64::from(record.trace.response_chunks_count)),
        ..RequestTrace::default()
    }
}

fn compatibility_http_record(record: &GrpcRequestRecord) -> RequestRecord {
    let responses = record
        .responses
        .iter()
        .map(|response| {
            let body = Bytes::from(serde_json::to_vec(&response.json).unwrap_or_default());
            Response::Text(TextResponse {
                perf_ns: response.perf_ns,
                text: String::from_utf8_lossy(&body).into_owned(),
                body,
                content_type: Some("application/json".to_string()),
            })
        })
        .collect();
    let mut response_headers = record.trace.response_metadata.clone();
    if let Some(status) = record.trace.grpc_status_code {
        response_headers.insert("grpc-status".to_string(), status.to_string());
    }
    if let Some(message) = &record.trace.grpc_status_message {
        response_headers.insert("grpc-message".to_string(), message.clone());
    }
    RequestRecord {
        start_ns: record.start_ns,
        request_body: record.request_body.clone(),
        request_headers: record.trace.request_metadata.clone(),
        end_ns: record.end_ns,
        recv_start_ns: record.trace.response_receive_start_ns,
        status: record.status,
        response_headers,
        responses,
        error: record.error.as_ref().map(|error| ErrorDetails {
            kind: match error.kind {
                GrpcErrorKind::RequestCancellation => ErrorKind::Cancelled,
                GrpcErrorKind::RequestTimeout | GrpcErrorKind::RequestSendTimeout => {
                    ErrorKind::Timeout
                }
                GrpcErrorKind::Rpc | GrpcErrorKind::Stream => ErrorKind::Http,
                GrpcErrorKind::InvalidRequest | GrpcErrorKind::Decode | GrpcErrorKind::Other => {
                    ErrorKind::Other
                }
            },
            code: (error.code != 0).then_some(error.code),
            message: error.message.clone(),
        }),
        trace: Some(TraceData {
            tcp_connect_start_ns: record.trace.connect_start_ns,
            tcp_connect_end_ns: record.trace.connect_end_ns,
            connection_reused_ns: record.trace.channel_reused_ns,
            request_send_start_ns: record.trace.request_send_start_ns,
            request_headers_sent_ns: record.trace.request_headers_sent_ns,
            request_send_end_ns: record.trace.request_send_end_ns,
            request_chunks_count: record.trace.request_chunks_count,
            request_bytes_total: record.trace.request_bytes_total,
            request_chunks: record.trace.request_chunks.clone(),
            response_status_code: record.trace.response_status_code,
            response_reason: record.trace.response_reason.clone(),
            response_receive_start_ns: record.trace.response_receive_start_ns,
            response_headers_received_ns: record.trace.response_headers_received_ns,
            response_chunks_count: record.trace.response_chunks_count,
            response_bytes_total: record.trace.response_bytes_total,
            response_chunks: record.trace.response_chunks.clone(),
            response_receive_end_ns: record.trace.response_receive_end_ns,
            error_timestamp_ns: record.trace.error_timestamp_ns,
            ..TraceData::default()
        }),
        cancellation_ns: record.cancellation_ns,
        reusable_connection: false,
    }
}

fn nonzero_usize(value: usize) -> Option<usize> {
    (value > 0).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::{EndpointId, EndpointRegistry, RawEndpointConfig};

    #[test]
    fn usage_absorption_retains_extended_endpoint_facts() {
        let parsed = ParsedResponse {
            perf_ns: 1,
            data: None,
            usage: Some(serde_json::json!({
                "prompt_tokens_details": {"audio_tokens": 2},
                "completion_tokens_details": {
                    "audio_tokens": 3,
                    "accepted_prediction_tokens": 4,
                    "rejected_prediction_tokens": 5
                },
                "toolUsePromptTokenCount": 6,
                "prompt_audio_seconds": 1.5
            })),
            sources: None,
        };
        let mut observed = ObservedUsage::default();
        absorb_usage(&parsed, &mut observed);

        assert_eq!(observed.prompt_audio_tokens, Some(2));
        assert_eq!(observed.completion_audio_tokens, Some(3));
        assert_eq!(observed.accepted_prediction_tokens, Some(4));
        assert_eq!(observed.rejected_prediction_tokens, Some(5));
        assert_eq!(observed.tool_use_prompt_tokens, Some(6));
        assert_eq!(observed.prompt_audio_seconds, Some(1.5));
    }

    #[test]
    fn riva_tts_audio_is_meaningful_without_becoming_model_text() {
        let endpoint = EndpointRegistry::builtin()
            .unwrap()
            .prepare(
                &EndpointId::new("riva_tts").unwrap(),
                RawEndpointConfig {
                    urls: vec!["grpc://127.0.0.1:50051".to_string()],
                    ..RawEndpointConfig::default()
                },
            )
            .unwrap();
        let response = ServerResponse::from_json(1, serde_json::json!({"audio": "AQI="}));
        assert!(meaningful_response(endpoint.as_ref(), &response));

        let parsed = endpoint.parse_response(&response).unwrap().unwrap();
        let data = parsed.data.as_ref().unwrap();
        let mut metadata = ModelResponseMetadata::default();
        assert_eq!(absorb_response_data(data, &mut metadata), "");
        assert_eq!(metadata.content, None);

        let empty = ServerResponse::from_json(2, serde_json::json!({"audio": ""}));
        assert!(!meaningful_response(endpoint.as_ref(), &empty));
    }
}
