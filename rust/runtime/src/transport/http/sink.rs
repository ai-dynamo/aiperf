// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online HTTP dispatch over Hyper.
//!
//! [`TransportSink`] implements `crate::dispatch`'s [`RequestSink`] using the
//! clock-injected HTTP client. It is single-threaded (`!Send`, `Rc`-based) and
//! driven on a `LocalSet`;
//! admit/token times are stamped from the same clock origin the run loop uses for
//! arrival, so all events share one timeline.
//!
//! Per-request cancellation and endpoint resolution consume the timing scalars
//! and preserve the full-send timer invariant.

use std::cell::{Cell, RefCell};
#[cfg(test)]
use std::collections::BTreeMap;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use uuid::Uuid;

use crate::clock::Clock;
use crate::endpoints::PreparedEndpointTable;
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest, RequestTrace};

use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::{RequestObserver, RequestSink};
use crate::metrics::NativeMetricsObserver;
use crate::transport::core::{
    ConnectionReuseStrategy, DispatchResult, Dispatcher, ErrorDetails, ErrorKind, MeasuredContext,
    MeasuredOutcome, PreparedEndpointBinding, PreparedTurn, Request, RequestExecutor,
    RequestRecord,
};
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::HttpVersion;
use crate::transport::http::transport::http_transport::HttpTransport;
use crate::transport::measure::{self, WorkerMeasurement};
use serde_json::Value;

pub use crate::multiturn::PreparedEndpointReference;
use crate::multiturn::{TurnDataPolicy, TurnToSend};
use crate::scheduled::{
    ModelResponseMetadata, TurnDispatchOutcome, TurnDispatcher, TurnResponseObserver,
};

mod endpoint_dispatch;

use endpoint_dispatch::EndpointDispatchHooks;


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
    pub http: RequestTrace,
}

struct HttpCollectedDispatch {
    result: HttpDispatchResult,
    request_payload: Bytes,
    record: RequestRecord,
}

fn enforce_turn_data_policy(
    data_policy: TurnDataPolicy,
    request_payload: &mut Bytes,
    record: &mut RequestRecord,
    model_response: &mut ModelResponseMetadata,
) {
    if !data_policy.retain_raw_exchange() || !data_policy.allow_public_content_hash() {
        *request_payload = Bytes::new();
        record.request_headers.clear();
        record.response_headers.clear();
        record.responses.clear();
    }
    if !data_policy.allow_content_diagnostics() {
        if let Some(error) = &mut record.error {
            error.message = "restricted evaluator HTTP operation failed".to_string();
        }
        if model_response.error_message.is_some() {
            model_response.error_message =
                Some("restricted evaluator inference failed".to_string());
        }
    }
}

/// Construction policy for one online HTTP sink.
///
/// The client config owns Clock-enforced transport deadlines and protocol
/// selection; reuse and affinity remain per-request policies applied when a
/// materialized turn becomes a [`RequestConfig`].
#[derive(Clone, Debug)]
pub struct TransportSinkConfig {
    /// Low-level HTTP client policy.
    pub client: ClientConfig,
    /// Connection pooling/lease strategy.
    pub connection_reuse: ConnectionReuseStrategy,
    /// Optional replacement for the default `X-Correlation-ID` header.
    pub session_header: Option<String>,
    /// Content-server origin (`http://host:port`) for this run, when the server
    /// is enabled. Media URLs starting with it are tagged at dispatch with
    /// `?rid&mi&td`; `None` disables tagging.
    pub content_server_base: Option<Arc<str>>,
    /// Whether a raw artifact will consume this run's per-dispatch request
    /// payload.
    pub capture_raw: bool,
    /// Whether this run retains canonical request payloads for `inputs.json`.
    ///
    /// Together with [`Self::capture_raw`] this decides whether the canonical
    /// body handle is taken at all.
    ///
    /// **Both default to `true`, the opposite of the identically-named fields on
    /// [`GrpcTransportSinkConfig`], which default to `false`.** The asymmetry is
    /// deliberate: this type is built by struct literals that inherit the rest of
    /// their fields from [`Default`], so a site that has not been told the run's
    /// artifact selection must fail toward capturing (losing the per-dispatch
    /// saving) rather than toward silently emptying `inputs.json` or the raw
    /// artifact. The gRPC config is only ever constructed field-complete, so it
    /// has no such site to protect.
    ///
    /// [`GrpcTransportSinkConfig`]: crate::transport::grpc::GrpcTransportSinkConfig
    pub inputs_enabled: bool,
}

impl Default for TransportSinkConfig {
    fn default() -> Self {
        Self {
            client: ClientConfig::default(),
            connection_reuse: ConnectionReuseStrategy::default(),
            session_header: None,
            content_server_base: None,
            capture_raw: true,
            inputs_enabled: true,
        }
    }
}

/// Response-capturing request-dispatch seam used by the shared paced issuer.
///
/// The online implementation is [`TransportSink`]; the optional in-process
/// simulator implements the same contract, so pacing, admission, adaptive
/// control, observers, and report construction do not branch on a backend.
#[async_trait(?Send)]
pub trait HttpRequestDispatcher: RequestSink<Request> {
    /// Resolve report dimensions using the same endpoint selection as dispatch.
    fn inference_dimensions(&self, _request: &Request) -> InferenceDimensions {
        InferenceDimensions::default()
    }

    /// Dispatch one request, retain its terminal response facts, and invoke
    /// `on_first_token` exactly once with TTFT in nanoseconds.
    async fn dispatch_collect(
        &self,
        req: Request,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<HttpDispatchResult>;
}

/// Live OpenAI-chat sink over [`crate::transport::http`]. Shares the caller's clock and
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
    content_server_base: Option<Arc<str>>,
    /// Whether any configured artifact consumes the canonical request payload.
    /// With no consumer the dispatch paths below never take the handle, which
    /// leaves the assembled body with a single live handle and so avoids
    /// promoting it to a shared (heap-allocated) control block.
    capture_request_payload: bool,
    prepared_endpoints: Option<Rc<PreparedEndpointTable>>,
    /// Worker-local metric accumulator for measured execution. Unset until
    /// [`configure_measurement`] is called.
    ///
    /// [`configure_measurement`]: RequestExecutor::configure_measurement
    measurement: WorkerMeasurement,
    /// Single-entry memo for [`selected_url`](Self::selected_url), keyed by the
    /// request's `(url index, endpoint path)`.
    ///
    /// URL selection is pure — it depends only on the run's fixed `model` and
    /// `base_urls` plus that key — yet it ran three `String` allocations and four
    /// `{model_name}` pattern scans (or a full `Url::parse`) on every request to
    /// rebuild a byte-identical string. A scheduled run reuses one key for its
    /// whole lifetime, so one entry collapses the hot path to a single clone.
    /// Only successful renders are memoized, so template validation still fails
    /// closed on every request that would have failed before.
    url_memo: RefCell<Option<(usize, Option<Box<str>>, String)>>,
    /// Whether a raw HTTP-exchange artifact will consume the retained responses.
    ///
    /// When false the responses are released on the worker that produced them,
    /// because the only consumer — `RunCapture::record_http_exchange` — drops
    /// them behind its own `raw_enabled` guard. Under `GlobalHop` that consumer
    /// runs on the single coordinator thread, so leaving them attached funnels
    /// every request's response strings (151 per request at OSL 150) through
    /// one thread's allocator; a profile put `mi_free` + `_mi_page_malloc_zero`
    /// at 17.5% there versus 5.4% under `global`. The gRPC sink already skips
    /// building this record for the same reason.
    ///
    /// Defaults to true so every other construction site keeps today's
    /// behavior; only the worker builder opts out.
    retain_raw_responses: Cell<bool>,
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
            content_server_base: config.content_server_base,
            capture_request_payload: config.capture_raw || config.inputs_enabled,
            prepared_endpoints: None,
            measurement: WorkerMeasurement::default(),
            url_memo: RefCell::new(None),
            retain_raw_responses: Cell::new(true),
        })
    }

    /// Declare whether a raw artifact will consume retained responses.
    ///
    /// See [`Self::retain_raw_responses`]. Called by the worker sink builder
    /// once per worker; the default keeps responses attached.
    pub fn set_retain_raw_responses(&self, retain: bool) {
        self.retain_raw_responses.set(retain);
    }

    /// Whether this sink retains response bodies for a raw artifact.
    #[cfg(test)]
    pub(crate) fn retains_raw_responses(&self) -> bool {
        self.retain_raw_responses.get()
    }

    /// Install worker-local prepared endpoint bindings.
    ///
    /// Execution commands carry only dense keys and canonical IDs; endpoint
    /// configuration and credentials remain in this worker-owned table.
    pub fn with_prepared_endpoints(mut self, endpoints: Rc<PreparedEndpointTable>) -> Self {
        self.prepared_endpoints = Some(endpoints);
        self
    }

    /// Whether this run has any consumer for the canonical request payload.
    ///
    /// The raw artifact writes it verbatim and `inputs.json` retains it; with
    /// neither selected the payload handle is never taken, and the assembled
    /// body stays uniquely owned through dispatch.
    pub(super) fn captures_request_payload(&self) -> bool {
        self.capture_request_payload
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
        if let Some((index, path, rendered)) = self.url_memo.borrow().as_ref()
            && *index == selected_index
            && path.as_deref() == endpoint_path
        {
            return Ok(rendered.clone());
        }
        let selected_url = self.urls.get(selected_index).ok_or_else(|| {
            anyhow::anyhow!(
                "URL index {selected_index} is out of range for {} configured endpoints",
                self.urls.len()
            )
        })?;
        let rendered = match endpoint_path {
            None => selected_url.clone(),
            Some(path) if path.starts_with('/') => {
                // Expand the supported path template before removing a duplicate
                // `/v1` prefix.
                let template_remainder = path.replace("{model_name}", "");
                anyhow::ensure!(
                    !template_remainder.contains('{') && !template_remainder.contains('}'),
                    "endpoint path {path:?} contains an unsupported template placeholder"
                );
                let rendered = path.replace("{model_name}", &self.model);
                let base_url = self
                    .base_urls
                    .get(selected_index)
                    .expect("base/default URL vectors have equal length");
                let rendered = if base_url.ends_with("/v1") && rendered.starts_with("/v1/") {
                    &rendered[3..]
                } else {
                    rendered.as_str()
                };
                format!("{base_url}{rendered}")
            }
            Some(url) if url::Url::parse(url).is_ok() => url.to_string(),
            Some(value) => {
                anyhow::bail!("dataset endpoint target {value:?} must be an absolute path or URL")
            }
        };
        *self.url_memo.borrow_mut() =
            Some((selected_index, endpoint_path.map(Box::from), rendered.clone()));
        Ok(rendered)
    }



}

/// Follow a fixed JSON path without [`Value::pointer`].
///
/// `Value::pointer` splits its path on `/` and runs two `String::replace` calls
/// per segment to undo `~0`/`~1` escaping, so every lookup allocates and scans.
/// [`absorb_wire_response_metadata`] issues up to seven of them per decoded
/// response, which on a streamed reply is per generated token — profiling put
/// the resulting `str::replace` + `StrSearcher::new` + `Split<char>` at ~11% of
/// load-phase samples.
///
/// Every call site here uses a static, escape-free path, so walking the
/// segments directly is byte-identical and allocation-free. A segment naming an
/// array is indexed by its decimal value, matching pointer semantics.
fn dig<'v>(value: &'v Value, path: &[&str]) -> Option<&'v Value> {
    let mut current = value;
    for segment in path {
        current = match current {
            Value::Array(items) => items.get(segment.parse::<usize>().ok()?)?,
            _ => current.get(segment)?,
        };
    }
    Some(current)
}




pub(super) fn absorb_wire_response_metadata(value: &Value, metadata: &mut ModelResponseMetadata) {
    if let Some(response_id) = value
        .get("id")
        .or_else(|| value.get("request_id"))
        .or_else(|| dig(value, &["response", "id"]))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        // Streaming repeats the same `id` on every chunk, so a 150-chunk response
        // built and discarded 150 identical Strings. Allocate only on an actual
        // change; last-wins semantics are unchanged because an equal value would
        // have overwritten itself.
        if metadata.response_id.as_deref() != Some(response_id) {
            metadata.response_id = Some(response_id.to_string());
        }
    }
    if let Some(finish_reason) = dig(value, &["choices", "0", "finish_reason"])
        .or_else(|| dig(value, &["response", "incomplete_details", "reason"]))
        .or_else(|| dig(value, &["incomplete_details", "reason"]))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        metadata.finish_reason = Some(normalize_finish_reason(finish_reason));
    }
    let usage = value
        .get("usage")
        .or_else(|| dig(value, &["response", "usage"]));
    metadata.cached_prompt_tokens = usage
        .and_then(|usage| {
            dig(usage, &["prompt_tokens_details", "cached_tokens"])
                .or_else(|| dig(usage, &["input_tokens_details", "cached_tokens"]))
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
impl Dispatcher for TransportSink {
    async fn dispatch_collect(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        TransportSink::dispatch_collect(self, turn, observer, on_first_token).await
    }

    fn inference_dimensions(&self, request: &Request) -> InferenceDimensions {
        InferenceDimensions {
            endpoint_url: self
                .selected_url(request.url_index, request.endpoint_path.as_deref())
                .ok(),
            model: Some(self.model.clone()),
        }
    }

    fn supports_response_streaming(&self) -> bool {
        true
    }
}

#[async_trait(?Send)]
impl TurnDispatcher for TransportSink {
    fn supports_response_streaming(&self) -> bool {
        true
    }

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

    async fn dispatch_turn_streaming(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: &dyn TurnResponseObserver,
    ) -> Result<TurnDispatchOutcome> {
        Ok(self
            .dispatch_turn_collect_record_streaming(turn, observer, on_first_token, responses)
            .await?
            .outcome)
    }
}

#[async_trait(?Send)]
impl RequestExecutor for TransportSink {
    fn set_run_origin(&self, start_ns: i64) -> Result<()> {
        TransportSink::set_run_origin(self, start_ns);
        Ok(())
    }

    fn supports_response_streaming(&self) -> bool {
        true
    }

    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        <Self as TurnDispatcher>::inference_dimensions(self, turn)
    }

    fn configure_measurement(&self, config: MetricsConfig, origin_ns: i64) -> Result<()> {
        // The workers==1 observer accumulates on the coordinator thread.
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
        let live_record = measure::live_record(&observer, uuid, &context);
        Ok(MeasuredOutcome {
            result,
            live_record,
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
        let live_record = measure::live_record(&observer, uuid, &context);
        Ok(MeasuredOutcome {
            result,
            live_record,
        })
    }

    async fn prewarm(&self, turn: PreparedTurn) -> Result<()> {
        // Warm connection and materialization state without recording metrics;
        // persistent failures are reported by the timed run.
        let observer = PrewarmObserver;
        let _ = self.dispatch_collect(turn, &observer, &|_| {}).await;
        Ok(())
    }

    fn drain_records(&self, end_ns: i64) -> Result<Vec<(Uuid, RecordIngest)>> {
        Ok(self.measurement.drain(end_ns))
    }
}

/// No-op [`RequestObserver`] used only by warmup dispatches, so a prewarm
/// round-trip warms the transport without entering the metrics.
struct PrewarmObserver;

impl RequestObserver for PrewarmObserver {
    fn on_arrival(&self, _uuid: Uuid, _arrival_ms: f64, _input_length: usize, _requested: usize) {}
    fn on_admit(&self, _uuid: Uuid, _admit_ms: f64, _reused_input_tokens: usize) {}
    fn on_token(&self, _uuid: Uuid, _at_ms: f64) {}
    fn on_terminal(&self, _uuid: Uuid, _status: ReplayTerminalStatus) {}
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
    ) -> Result<DispatchResult> {
        let turn = PreparedTurn::from_turn(turn, &self.model);
        self.dispatch_collect(turn, observer, on_first_token).await
    }

    /// Dispatch one scheduled turn while publishing live endpoint-normalized
    /// response frames before terminal completion.
    pub async fn dispatch_turn_collect_record_streaming(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: &dyn TurnResponseObserver,
    ) -> Result<DispatchResult> {
        let turn = PreparedTurn::from_turn(turn, &self.model);
        self.dispatch_collect_streaming(turn, observer, on_first_token, Some(responses))
            .await
    }

    /// Execute an owned scheduler-free HTTP command and retain the exact wire
    /// exchange. Execution-placement adapters use this method on their local
    /// worker reactor while the ordinary direct path calls it in place.
    pub async fn dispatch_collect(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult> {
        self.dispatch_collect_streaming(turn, observer, on_first_token, None)
            .await
    }

    /// Access the worker-local measurement observer, erroring if the measured
    /// execution path is used before [`configure_measurement`] runs.
    ///
    /// [`configure_measurement`]: RequestExecutor::configure_measurement
    fn measurement_observer(&self) -> Result<Rc<NativeMetricsObserver>> {
        self.measurement.observer()
    }

    /// Register coordinator-known arrival facts on `observer`, dispatch the
    /// prepared turn into it, and record the terminal transport facts.
    ///
    /// This is the shared worker-local measurement wrapper used by both the
    /// workers==1 sink and each thread-per-core worker. The observer accumulates
    /// the complete record (arrival → admit → tokens → usage → terminal →
    /// response) so the end-of-run drain yields one authoritative
    /// [`RecordIngest`] per request. `phase`, `session_num`, the global
    /// `request_index`, and the credit-issued `admit_ns` are patched onto the
    /// drained record coordinator-side; they are intentionally not set here.
    pub async fn dispatch_measured(
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
            self.dispatch_collect_streaming(turn, observer, on_first_token, responses),
        )
        .await
    }

    /// Dispatch one prepared turn and retain the exact wire exchange. This is
    /// the single collect primitive: `responses` is `Some` for the live-frame
    /// streaming path and `None` for terminal-only collection, so the surface is
    /// two methods ([`dispatch_collect`](Self::dispatch_collect) is the `None`
    /// convenience) rather than a family of near-duplicates.
    pub async fn dispatch_collect_streaming(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
        responses: Option<&dyn TurnResponseObserver>,
    ) -> Result<DispatchResult> {
        let PreparedTurn {
            mut request,
            model,
            endpoint,
            endpoint_aware: _,
            data_policy,
            deferred: _,
        } = turn;
        if !data_policy.allow_result_cache() {
            request
                .headers
                .insert("cache-control".to_string(), "no-store".to_string());
            request
                .headers
                .insert("pragma".to_string(), "no-cache".to_string());
        }
        // Every product turn carries a prebuilt request body, so `PreparedTurn`
        // always reports endpoint_aware (see PreparedTurn::from_turn). The
        // non-endpoint-aware alternative this used to branch to was reachable
        // only from tests.
        let collected = {
            match endpoint {
                PreparedEndpointBinding::Prepared(reference) => {
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
                        &model,
                        EndpointDispatchHooks::new(
                            observer,
                            on_first_token,
                            responses,
                            data_policy,
                        ),
                    )
                    .await?
                }
            }
        };
        let HttpCollectedDispatch {
            result,
            mut request_payload,
            mut record,
        } = collected;
        let HttpDispatchResult {
            start_ns,
            end_ns,
            terminal,
            response_text,
            mut model_response,
            prompt_tokens,
            completion_tokens,
            http,
            ..
        } = result;
        enforce_turn_data_policy(
            data_policy,
            &mut request_payload,
            &mut record,
            &mut model_response,
        );
        Ok(DispatchResult {
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

/// Tag content-server media URLs in `body` with `?rid&mi&td`, returning the
/// possibly-reserialized bytes along with the parsed payload (so the caller can
/// reuse it instead of re-parsing). Only URLs starting with `base` are rewritten;
/// bytes are reserialized only when a URL actually changed, and a non-JSON body
/// is returned unchanged.
fn tag_content_urls(body: Bytes, base: &str, rid: &str, wall_ns: u64) -> (Bytes, Option<Value>) {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return (body, None);
    };
    if crate::content_server::tag_media_urls(&mut value, base, rid, wall_ns) == 0 {
        return (body, Some(value));
    }
    match serde_json::to_vec(&value) {
        Ok(bytes) => (Bytes::from(bytes), Some(value)),
        // Unreachable: `value` came from valid JSON. Keep the original on the
        // impossible error rather than panicking on the hot path.
        Err(_) => (body, Some(value)),
    }
}

#[cfg(test)]
mod tests {
    use crate::dispatch::sink::ObservedUsage;
    use crate::transport::http::sse::ChatChunk;
    use crate::endpoints::chat_request_body;
    use crate::transport::core::Response;
    use crate::transport::http::models::RequestConfig;
    use std::cell::Cell;

    use super::*;
    use crate::clock::RealClock;

    #[test]
    fn transport_and_grpc_sinks_are_dispatchers() {
        fn assert_dispatcher<T: Dispatcher>() {}
        assert_dispatcher::<TransportSink>();
        #[cfg(feature = "grpc")]
        assert_dispatcher::<crate::transport::grpc::GrpcTransportSink>();
        fn _takes_dyn(_: &dyn Dispatcher) {}
    }

    #[test]
    fn tag_content_urls_tags_only_matching_base() {
        let base = "http://127.0.0.1:8090";
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "http://127.0.0.1:8090/content/images/a.png"}},
                    {"type": "image_url", "image_url": {"url": "https://cdn.example.com/user.jpg"}},
                ]}]
            }))
            .unwrap(),
        );
        let (tagged, _) = tag_content_urls(body, base, "req-9", 777);
        let value: Value = serde_json::from_slice(&tagged).unwrap();
        assert_eq!(
            value
                .pointer("/messages/0/content/0/image_url/url")
                .unwrap(),
            "http://127.0.0.1:8090/content/images/a.png?rid=req-9&mi=0&td=777"
        );
        assert_eq!(
            value
                .pointer("/messages/0/content/1/image_url/url")
                .unwrap(),
            "https://cdn.example.com/user.jpg"
        );
    }

    #[test]
    fn tag_content_urls_returns_body_unchanged_when_no_match() {
        let base = "http://127.0.0.1:8090";
        let original = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "messages": [{"role": "user", "content": "plain text only"}]
            }))
            .unwrap(),
        );
        let (out, _) = tag_content_urls(original.clone(), base, "req", 1);
        assert_eq!(out, original);
    }

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
    fn endpoint_path_expands_primary_model_name() {
        let sink = TransportSink::new(
            RealClock::new(),
            0,
            "http://localhost:8000",
            "sklearn-iris",
            false,
        );
        assert_eq!(
            sink.selected_url(None, Some("/v1/models/{model_name}:predict"))
                .unwrap(),
            "http://localhost:8000/v1/models/sklearn-iris:predict"
        );
    }

    #[test]
    fn endpoint_path_deduplicates_v1_base_prefix() {
        let sink = TransportSink::new(
            RealClock::new(),
            0,
            "http://localhost:8000/v1",
            "sklearn-iris",
            false,
        );
        assert_eq!(
            sink.selected_url(None, Some("/v1/models/{model_name}:predict"))
                .unwrap(),
            "http://localhost:8000/v1/models/sklearn-iris:predict"
        );
    }

    #[test]
    fn endpoint_path_rejects_unknown_template_placeholders() {
        let sink = TransportSink::new(
            RealClock::new(),
            0,
            "http://localhost:8000",
            "fixture-model",
            false,
        );
        assert!(
            sink.selected_url(None, Some("/v1/models/{unknown}:predict"))
                .unwrap_err()
                .to_string()
                .contains("unsupported template placeholder")
        );
    }


    #[test]
    fn vllm_request_id_and_finish_reason_enter_normalized_metadata() {
        let mut metadata = ModelResponseMetadata::default();
        absorb_wire_response_metadata(
            &serde_json::json!({
                "request_id": "vllm-request-1",
                "choices": [{"finish_reason": "stop"}]
            }),
            &mut metadata,
        );

        assert_eq!(metadata.response_id.as_deref(), Some("vllm-request-1"));
        assert_eq!(metadata.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn prepared_turn_is_send_between_reactor_threads() {
        fn assert_send<T: Send>() {}
        assert_send::<PreparedTurn>();
    }

    #[test]
    fn restricted_turn_policy_erases_raw_exchange_and_content_diagnostics() {
        const SENTINEL: &str = "hidden-restricted-request-sentinel";

        let mut request_payload = Bytes::from_static(SENTINEL.as_bytes());
        let mut record = RequestRecord {
            request_headers: BTreeMap::from([("x-hidden".into(), SENTINEL.into())]),
            response_headers: BTreeMap::from([("x-hidden".into(), SENTINEL.into())]),
            responses: vec![Response::Text(crate::transport::core::TextResponse {
                perf_ns: 1,
                text: SENTINEL.into(),
                body: Bytes::from_static(SENTINEL.as_bytes()),
                content_type: Some("text/plain".into()),
            })],
            error: Some(ErrorDetails::other(SENTINEL)),
            ..RequestRecord::started(0)
        };
        let mut model_response = ModelResponseMetadata {
            error_message: Some(SENTINEL.into()),
            ..ModelResponseMetadata::default()
        };

        enforce_turn_data_policy(
            TurnDataPolicy::restricted_transient(),
            &mut request_payload,
            &mut record,
            &mut model_response,
        );

        assert!(request_payload.is_empty());
        assert!(record.request_headers.is_empty());
        assert!(record.response_headers.is_empty());
        assert!(record.responses.is_empty());
        assert!(!record.error.as_ref().unwrap().message.contains(SENTINEL));
        assert!(!model_response.error_message.unwrap().contains(SENTINEL));
        assert!(!format!("{record:?}").contains(SENTINEL));
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
                            // Inlined from the removed non-endpoint-aware path:
                            // a frame counts only if it carries delta text.
                            message.data().is_some_and(|data| {
                                serde_json::from_str::<ChatChunk>(data)
                                    .is_ok_and(|chunk| !chunk.delta_text().is_empty())
                            })
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

}
