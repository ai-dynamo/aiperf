// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduled-runtime composition for protocol-v2 gRPC.
//!
//! The sink accepts worker-local prepared endpoints and delegates wire I/O and
//! timing to [`GrpcTransport`].

use std::cell::Cell;
use std::rc::Rc;

use anyhow::{Context, Result, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::Value;

use crate::clock::Clock;
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{
    Dispatchable, ObservedEndpointMetrics, ObservedUsage, RequestObserver, RequestSink,
};
use crate::endpoints::{
    EndpointKey, PreparedEndpoint, PreparedEndpointTable, RequestRecord as EndpointRequestRecord,
    ResponseData, ServerResponse,
};
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest, RequestTrace};
use crate::transport::core::{
    ErrorDetails, ErrorKind, RequestRecord, Response, TextResponse, TraceData,
};
use crate::transport::grpc::{
    ConnectionReuseStrategy as GrpcConnectionReuseStrategy, GrpcBindingRegistry, GrpcClientConfig,
    GrpcErrorKind, GrpcRequestConfig, GrpcRequestRecord, GrpcTransport,
};
use crate::transport::reduce::{
    EndpointReduceAccumulators, TokenEmitter, assistant_message, reduce_parsed_response,
};
use uuid::Uuid;

use crate::metrics::NativeMetricsObserver;
use crate::multiturn::TurnToSend;
use crate::scheduled::{ModelResponseMetadata, TurnDispatchOutcome};
use crate::transport::core::{
    DispatchResult, Dispatcher, MeasuredContext, MeasuredOutcome, PreparedTurn, Request,
    RequestExecutor,
};
use crate::transport::measure::{self, WorkerMeasurement};

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

/// A [`RequestSink`] request retaining its prepared endpoint binding.
#[derive(Clone, Debug)]
pub struct GrpcRequest {
    turn: PreparedTurn,
}

impl GrpcRequest {
    /// Wrap one worker-resolved protocol-v2 turn.
    pub fn new(turn: PreparedTurn) -> Self {
        Self { turn }
    }

    /// Recover the prepared turn.
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
    /// Worker-local metric accumulator, unset until `configure_measurement`.
    measurement: WorkerMeasurement,
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
            measurement: WorkerMeasurement::default(),
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

    /// Access the worker-local measurement observer, erroring if the measured
    /// execution path is used before `configure_measurement`.
    fn measurement_observer(&self) -> Result<Rc<NativeMetricsObserver>> {
        self.measurement.observer()
    }

    /// Register coordinator-known arrival facts on `observer`, dispatch the
    /// prepared gRPC turn into it, and record the terminal transport facts.
    pub async fn dispatch_measured(
        &self,
        observer: &NativeMetricsObserver,
        turn: PreparedTurn,
        context: &MeasuredContext,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        let uuid = turn.request.uuid;
        measure::measure_dispatch(
            observer,
            self.clock.as_ref(),
            uuid,
            context,
            self.dispatch_collect(turn, observer, on_first_token),
        )
        .await
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
        let crate::transport::core::PreparedEndpointBinding::Prepared(reference) = turn.endpoint;
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
        // Assistant-turn capture is the only consumer of `endpoint_responses`; skip the
        // per-response retain (and its clone) entirely when the endpoint does not capture.
        let captures_turn = endpoint.captures_assistant_turn();
        let mut endpoint_responses = if captures_turn {
            Vec::with_capacity(record.responses.len())
        } else {
            Vec::new()
        };
        let to_ms = |ns| self.ms(ns);
        let emitter = TokenEmitter {
            uuid,
            produces_tokens: endpoint.descriptor().produces_tokens,
            start_ns: record.start_ns,
            obs: observer,
            to_ms: &to_ms,
            first_token_released: &first_token_released,
            on_first_token,
        };
        for response in &record.responses {
            let server_response = ServerResponse {
                perf_ns: u64::try_from(response.perf_ns).unwrap_or(u64::MAX),
                json: Some(response.json.clone()),
                // `raw` is only consulted as a fallback when `json` is `None`
                // (see `parse_flexible_response`); on the gRPC path `json` is always
                // `Some`, so reconstructing it here is pure per-token waste.
                raw: None,
            };
            let parsed = match endpoint.parse_response(&server_response) {
                Ok(parsed) => parsed,
                Err(error) => {
                    tracing::warn!(uuid = %uuid, error = %error, "gRPC endpoint response parsing failed");
                    parse_failed = true;
                    if captures_turn {
                        endpoint_responses.push(server_response);
                    }
                    continue;
                }
            };
            let Some(parsed) = parsed else {
                if captures_turn {
                    endpoint_responses.push(server_response);
                }
                continue;
            };
            let carried_content = reduce_parsed_response(
                &parsed,
                &emitter,
                EndpointReduceAccumulators {
                    response_text: &mut response_text,
                    model_response: &mut model_response,
                    endpoint_metrics: &mut endpoint_metrics,
                    observed_usage: &mut usage,
                },
            );
            parsed_content |= carried_content;
            // Retain the response only for assistant-turn replay; move rather than clone.
            if captures_turn {
                endpoint_responses.push(server_response);
            }
        }
        if captures_turn {
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
            .dispatch_measured(&observer, turn, &context, on_first_token)
            .await?;
        let live_record = measure::live_record(&observer, uuid, &context);
        Ok(MeasuredOutcome {
            result,
            live_record,
        })
    }

    fn drain_records(&self, end_ns: i64) -> Result<Vec<(Uuid, RecordIngest)>> {
        Ok(self.measurement.drain(end_ns))
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
    use crate::transport::reduce::absorb_response_data;

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
