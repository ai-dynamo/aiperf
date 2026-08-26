// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral request, result, and measurement types.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{self, Read};
use std::num::NonZeroUsize;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytes::Bytes;
use uuid::Uuid;

use crate::body_plan::RequestBody;
use crate::dispatch::sink::{Dispatchable, RequestObserver};

use crate::metrics::RequestMetricMetadata;
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest};
use crate::multiturn::{
    CreditIdentity, PreparedEndpointReference, TurnDataPolicy, TurnEndpoint, TurnToSend,
};
use crate::scheduled::{TurnDispatchOutcome, TurnResponseObserver};
use crate::transport::core::record::RequestRecord;

/// A transport-neutral inference request.
#[derive(Clone)]
pub struct Request {
    /// Stable per-request identifier used to correlate observer events.
    pub uuid: Uuid,
    /// Prompt length in tokens, for measurement accounting.
    pub input_length: usize,
    /// Maximum number of output tokens to request.
    pub max_output_tokens: usize,
    /// Prompt text placed on the wire.
    pub prompt_text: Option<String>,
    /// The request body, in whichever form the dataset seam produced. Absent
    /// for a synthetic request that carries only `prompt_text` and lets the
    /// transport build a body from the shared chat builder.
    pub body: Option<RequestBody>,
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
    /// Wire image count known from composition, when trustworthy without
    /// re-parsing the body. `None` means the dispatch path derives `num_images`
    /// by parsing the serialized body (raw payloads, history-accumulating turns).
    pub image_count: Option<u32>,
    /// Pre-known recorded response latency (api_time) in nanoseconds, lowered from
    /// a recorded trace. Consumed only by the `dry_run` transport under the
    /// `recorded` latency model to reproduce the recorded timeline exactly; `None`
    /// on every non-recorded path (analytic fallback).
    pub recorded_api_time_ns: Option<i64>,
    /// Pre-known recorded time-to-first-token in nanoseconds, when the trace
    /// supplies it. Splits the recorded api_time into TTFT + generated-token span;
    /// `None` falls back to an even split of the recorded api_time.
    pub recorded_ttft_ns: Option<i64>,
}

impl fmt::Debug for Request {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Request")
            .field("uuid", &self.uuid)
            .field("input_length", &self.input_length)
            .field("max_output_tokens", &self.max_output_tokens)
            .field("has_prompt_text", &self.prompt_text.is_some())
            // Summarized, never rendered: a multimodal body runs to megabytes,
            // and this `Debug` is reached from tracing.
            .field(
                "body",
                &self.body.as_ref().map(|body| match body {
                    RequestBody::Wire(bytes) => format!("wire({} bytes)", bytes.len()),
                    RequestBody::Plan(_) => "plan".to_string(),
                    RequestBody::WebSocket(plan) => {
                        format!("websocket({} messages)", plan.messages().len())
                    }
                    RequestBody::Value(_) => "value".to_string(),
                }),
            )
            .field("header_names", &self.headers.keys().collect::<Vec<_>>())
            .field("parameters", &self.parameters)
            .field("endpoint_path", &self.endpoint_path)
            .field("streaming", &self.streaming)
            .field("x_correlation_id", &self.x_correlation_id)
            .field("is_final_turn", &self.is_final_turn)
            .field("cancel_after_ns", &self.cancel_after_ns)
            .field("url_index", &self.url_index)
            .field("image_count", &self.image_count)
            .field("recorded_api_time_ns", &self.recorded_api_time_ns)
            .field("recorded_ttft_ns", &self.recorded_ttft_ns)
            .finish()
    }
}

impl Dispatchable for Request {
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

/// Immutable byte cap for one opt-in model-decision response.
///
/// A decision transport must admit every decoded decision byte through
/// [`BoundedDecisionAdmission`] before it can expose a
/// [`BoundedDecisionReader`]. The cap is deliberately separate from general
/// response limits: callers opt into this mode only for package-selected policy
/// decisions that must not reach response/raw-record accumulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundedDecisionMode {
    max_decision_bytes: NonZeroUsize,
}

impl BoundedDecisionMode {
    // A JSON `\u00xx` escape is the largest wire representation of one
    // admitted byte. The fixed allowance covers the SSE `data:` prefix and
    // endpoint envelope without making that metadata caller-controlled.
    const JSON_ESCAPED_BYTE_MULTIPLIER: usize = 6;
    const SSE_FRAME_OVERHEAD_BYTES: usize = 4 * 1024;

    /// Create a bounded decision mode with a positive byte cap.
    pub fn new(max_decision_bytes: usize) -> Result<Self, DecisionAdmissionError> {
        let max_decision_bytes =
            NonZeroUsize::new(max_decision_bytes).ok_or(DecisionAdmissionError::ZeroLimit)?;
        if max_decision_bytes.get()
            > (usize::MAX - Self::SSE_FRAME_OVERHEAD_BYTES) / Self::JSON_ESCAPED_BYTE_MULTIPLIER
        {
            return Err(DecisionAdmissionError::WireLimitOverflow);
        }
        Ok(Self { max_decision_bytes })
    }

    /// Return the exact maximum number of admitted decision bytes.
    pub const fn max_decision_bytes(self) -> usize {
        self.max_decision_bytes.get()
    }

    /// Return the bounded raw SSE frame allowance for one decision message.
    ///
    /// This is intentionally derived from the immutable decoded-decision cap,
    /// rather than from the general response-body cap: a JSON escape occupies
    /// at most six raw bytes per admitted decision byte, and endpoint framing
    /// has a fixed finite allowance. A transport that cannot honor this cap
    /// must refuse bounded-decision dispatch.
    pub const fn max_sse_frame_bytes(self) -> usize {
        self.max_decision_bytes.get() * Self::JSON_ESCAPED_BYTE_MULTIPLIER
            + Self::SSE_FRAME_OVERHEAD_BYTES
    }
}

/// Failure while admitting a bounded model-decision response.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecisionAdmissionError {
    /// The selected package supplied a zero-byte decision cap.
    ZeroLimit,
    /// The selected cap cannot derive a finite raw wire-frame allowance.
    WireLimitOverflow,
    /// The incoming chunk would exceed the immutable decision byte cap.
    LimitExceeded {
        /// Immutable cap selected before dispatch.
        max_decision_bytes: usize,
    },
    /// A previous admission failure permanently closed this response.
    Rejected,
    /// The host could not reserve the bounded reader buffer.
    AllocationFailed,
}

impl fmt::Display for DecisionAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroLimit => formatter.write_str("bounded decision byte limit must be positive"),
            Self::WireLimitOverflow => {
                formatter.write_str("bounded decision byte limit cannot derive a wire frame cap")
            }
            Self::LimitExceeded { max_decision_bytes } => write!(
                formatter,
                "bounded decision exceeds the selected {max_decision_bytes}-byte limit"
            ),
            Self::Rejected => formatter.write_str("bounded decision admission is closed"),
            Self::AllocationFailed => {
                formatter.write_str("unable to reserve the bounded decision buffer")
            }
        }
    }
}

impl std::error::Error for DecisionAdmissionError {}

/// Host-owned incremental admission for one bounded model decision.
///
/// The admission permanently closes on the first rejected chunk. This lets the
/// transport abort that response immediately instead of completing a terminal
/// record that a later consumer would have to discard.
pub struct BoundedDecisionAdmission {
    mode: BoundedDecisionMode,
    bytes: Vec<u8>,
    is_rejected: bool,
}

impl BoundedDecisionAdmission {
    /// Start admitting one response under an immutable byte cap.
    pub fn new(mode: BoundedDecisionMode) -> Self {
        Self {
            mode,
            bytes: Vec::new(),
            is_rejected: false,
        }
    }

    /// Admit one already-decoded decision chunk.
    ///
    /// Capacity is checked before reserve or copy, so the first byte over the
    /// selected limit cannot force a terminal response/raw-record allocation.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), DecisionAdmissionError> {
        if self.is_rejected {
            return Err(DecisionAdmissionError::Rejected);
        }
        let Some(total) = self.bytes.len().checked_add(bytes.len()) else {
            self.is_rejected = true;
            return Err(DecisionAdmissionError::LimitExceeded {
                max_decision_bytes: self.mode.max_decision_bytes(),
            });
        };
        if total > self.mode.max_decision_bytes() {
            self.is_rejected = true;
            return Err(DecisionAdmissionError::LimitExceeded {
                max_decision_bytes: self.mode.max_decision_bytes(),
            });
        }
        if self.bytes.try_reserve(bytes.len()).is_err() {
            self.is_rejected = true;
            return Err(DecisionAdmissionError::AllocationFailed);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }

    /// Seal admission at transport EOF and expose the host-owned reader.
    pub fn finish(self) -> Result<BoundedDecisionReader, DecisionAdmissionError> {
        if self.is_rejected {
            return Err(DecisionAdmissionError::Rejected);
        }
        Ok(BoundedDecisionReader {
            bytes: self.bytes,
            offset: 0,
        })
    }
}

/// Immutable host-owned decision bytes exposed only after bounded admission.
pub struct BoundedDecisionReader {
    bytes: Vec<u8>,
    offset: usize,
}

impl BoundedDecisionReader {
    /// Return the number of unread admitted bytes.
    pub const fn remaining_len(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

impl fmt::Debug for BoundedDecisionReader {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundedDecisionReader")
            .field("remaining_len", &self.remaining_len())
            .finish()
    }
}

impl Read for BoundedDecisionReader {
    fn read(&mut self, destination: &mut [u8]) -> io::Result<usize> {
        let available = &self.bytes[self.offset..];
        let count = available.len().min(destination.len());
        destination[..count].copy_from_slice(&available[..count]);
        self.offset = self.offset.saturating_add(count);
        Ok(count)
    }
}

/// A terminal outcome with the compatibility record required by raw artifacts.
#[derive(Clone, Debug)]
pub struct DispatchResult {
    /// Backend-neutral result consumed by scheduling and record processors.
    pub outcome: TurnDispatchOutcome,
    /// Canonical JSON payload before transport-specific body preparation.
    pub request_payload: Bytes,
    /// Exact HTTP transport record.
    pub record: RequestRecord,
}

/// Coordinator-known facts registered by a worker-local metrics observer.
///
/// `metadata` excludes `phase`, `session_num`, `has_credit_timestamp`, and the
/// global `request_index`; the coordinator adds them after joining drained
/// records by UUID.
#[derive(Clone, Debug)]
pub struct MeasuredContext {
    /// Arrival timestamp in milliseconds relative to the run origin.
    pub arrival_ms: f64,
    /// Coordinator-known input length forwarded to the worker's `on_arrival`.
    pub input_length: usize,
    /// Coordinator-known requested output length forwarded to `on_arrival`.
    pub requested_output_length: usize,
    /// Issue-time request metadata (turn index, conversation, correlation,
    /// dimensions, audio duration); no `request_index`/`phase`/`session_num`.
    pub metadata: RequestMetricMetadata,
    /// Whether a live-results sink is attached and the worker must return a
    /// non-consuming cloned record for live emission.
    pub wants_live_record: bool,
    /// Whether an artifact will consume this request's raw HTTP exchange.
    ///
    /// Only the raw artifact reads it. When nothing does, a worker that returns
    /// a credit releases the request payload and transport record locally
    /// instead of shipping both to the coordinator to be dropped there --
    /// dropping them was measured at ~7% of the single issuer's CPU, on the one
    /// thread that bounds the run.
    pub wants_http_exchange: bool,
    /// Whether the returned record should be *moved out* of the worker observer
    /// (freeing its token storage) rather than cloned. Set in metrics-only
    /// (sketch) mode, where the coordinator folds each record into a bounded
    /// streaming accumulator and immediately drops it, so retaining the
    /// observer's copy would defeat the O(sketch) memory bound. Only consulted
    /// when `wants_live_record` is set.
    pub consume_record: bool,
}

/// A transport outcome and optional record for live-results ingestion.
#[derive(Debug)]
pub struct MeasuredOutcome {
    /// Backend-neutral dispatch result consumed by scheduling/record processors.
    pub result: DispatchResult,
    /// Snapshot or consumed record for a live sink, according to
    /// [`MeasuredContext::consume_record`].
    pub live_record: Option<RecordIngest>,
}

/// An owned, scheduler-free execution command.
///
/// The scheduling-only [`TurnToSend`] retains an `Rc` session backend so that
/// continuations can be materialized locally. This projection deliberately
/// removes that scheduler state: every remaining field is owned and `Send`, and
/// the endpoint's stable identity is carried by the worker-local prepared
/// binding key in [`PreparedEndpointReference`]. Local backends retain the
/// already-resolved prepared adapter allocation.
///
/// Consumed by HTTP, gRPC, graph, and dry-run execution.
#[derive(Clone)]
pub struct PreparedTurn {
    /// Runtime-owned session identity used for sticky transport affinity.
    pub runtime_session_id: String,
    /// Transport-ready request fields.
    pub request: Request,
    /// Effective model selected for this turn.
    pub model: String,
    /// Worker-resolved endpoint binding selected during preparation.
    pub endpoint: PreparedEndpointBinding,
    /// Whether the request came from the endpoint-aware dataset seam.
    pub endpoint_aware: bool,
    /// Content retention/cache/diagnostic policy fixed by materialization.
    pub data_policy: TurnDataPolicy,
    /// Present when the issuer routed this credit WITHOUT materializing its
    /// body. Every body-bearing field above is then a placeholder and the
    /// receiving worker rebuilds the request from the resident dataset, the way
    /// Python's `Credit` carries ids and lets the worker build the request.
    pub deferred: Option<CreditIdentity>,
}

impl fmt::Debug for PreparedTurn {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedTurn")
            .field("runtime_session_id", &self.runtime_session_id)
            .field("request", &self.request)
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("endpoint_aware", &self.endpoint_aware)
            .field("data_policy", &self.data_policy)
            .finish()
    }
}

/// Endpoint selection retained by one scheduler-free execution command.
#[derive(Clone)]
pub enum PreparedEndpointBinding {
    /// Protocol-v2 worker-local prepared binding.
    Prepared(PreparedEndpointReference),
}

impl fmt::Debug for PreparedEndpointBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepared(reference) => formatter
                .debug_tuple("PreparedEndpointBinding")
                .field(reference)
                .finish(),
        }
    }
}

impl PreparedTurn {
    /// Remove scheduler-local session state and build one owned execution command.
    pub fn from_turn(turn: TurnToSend, model: &str) -> Self {
        let is_final_turn = turn.is_final_turn();
        let runtime_session_id = turn.x_correlation_id.clone();
        let deferred = turn.deferred_body.then(|| CreditIdentity {
            conversation_id: turn.conversation_id.clone(),
            x_correlation_id: turn.x_correlation_id.clone(),
            turn_index: turn.turn_index,
            num_turns: turn.num_turns,
        });
        // A deferred credit has no body yet; the worker's materialization sets
        // this from what it actually builds.
        let endpoint_aware = turn.request_body.is_some();
        let data_policy = turn.data_policy;
        let model = turn
            .effective_model
            .clone()
            .unwrap_or_else(|| model.to_string());
        let endpoint = match turn.endpoint {
            TurnEndpoint::Prepared(reference) => PreparedEndpointBinding::Prepared(reference),
        };
        Self {
            runtime_session_id,
            request: Request {
                uuid: turn.uuid,
                input_length: turn.input_length,
                max_output_tokens: turn.max_output_tokens,
                prompt_text: None,
                body: turn.request_body,
                headers: turn.request_headers,
                parameters: turn.request_parameters,
                endpoint_path: turn.endpoint_path,
                streaming: turn.streaming,
                x_correlation_id: Some(turn.request_correlation_id),
                is_final_turn,
                cancel_after_ns: turn.cancel_after_ns,
                url_index: turn.url_index,
                image_count: turn.image_count,
                recorded_api_time_ns: None,
                recorded_ttft_ns: None,
            },
            model,
            endpoint,
            endpoint_aware,
            data_policy,
            deferred,
        }
    }
}

/// One worker's out-of-band report about a credit the issuer sent it.
///
/// The Rust counterpart of Python's `WorkerToRouterMessage`. `--dispatch
/// global-push` sends a credit to a worker and returns; the worker owns the
/// whole round-trip and reports back on the placement's single shared return
/// stream, tagged by `uuid` because the reports of every in-flight credit are
/// interleaved on it. Per-credit ordering is preserved by that stream, so
/// `FirstToken` still precedes `CreditReturn`.
pub struct WorkerCreditReport {
    /// Request this report belongs to.
    pub uuid: Uuid,
    /// Index of the reporting worker within its placement.
    ///
    /// Echoed back because a returned credit has to release the in-flight depth
    /// of the worker that held it, and the coordinator deliberately keeps no
    /// per-request routing table to look that up in.
    pub worker: usize,
    /// What the worker observed.
    pub kind: CreditReportKind,
}

/// The kinds of report a worker sends back about a credit.
pub enum CreditReportKind {
    /// First token observed, in nanoseconds since dispatch start. Releases the
    /// issuer's prefill slot while the request keeps decoding.
    FirstToken(i64),
    /// The credit is returned: terminal outcome, always the last report for a
    /// given `uuid`. Returning it is what releases the issuer's admission slot
    /// and the worker's in-flight depth.
    CreditReturn(Box<Result<MeasuredOutcome>>),
    /// The worker abandoned the credit because the placement was cancelled.
    /// This still returns the credit and releases worker depth, but is not a
    /// transport failure.
    Cancelled,
}

/// Pluggable execution placement behind the one logical turn dispatcher.
///
/// Implementations may execute on the caller's reactor, a thread-per-core
/// local pool, or a remote transport such as ZMQ. Scheduling, phase policy,
/// admission, adaptive control, and record capture remain above this seam and
/// therefore do not change when execution placement changes.
#[async_trait(?Send)]
pub trait RequestExecutor {
    /// Set the shared run origin after backend startup and before dispatch.
    fn set_run_origin(&self, start_ns: i64) -> Result<()>;

    /// Whether endpoint-normalized response frames can cross this placement.
    fn supports_response_streaming(&self) -> bool {
        false
    }

    /// Resolve labels using the same endpoint/model selection as execution.
    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions;

    /// Configure worker-local metric accumulation.
    ///
    /// Called once after [`set_run_origin`](Self::set_run_origin) and before any
    /// [`execute_measured`](Self::execute_measured). Builds one
    /// [`crate::metrics::NativeMetricsObserver`] per execution worker from the
    /// single resolved [`MetricsConfig`] so every worker accumulator shares an
    /// identical configuration. Backends that do not support worker-local
    /// measurement reject the call.
    fn configure_measurement(&self, _config: MetricsConfig, _origin_ns: i64) -> Result<()> {
        Err(anyhow!(
            "selected HTTP execution placement does not support worker-local measurement"
        ))
    }

    /// Execute one prepared request while accumulating its metrics into the
    /// worker-local observer (no per-token replay across the coordinator).
    async fn execute_measured(
        &self,
        _turn: PreparedTurn,
        _context: MeasuredContext,
        _on_first_token: &dyn Fn(i64),
    ) -> Result<MeasuredOutcome> {
        Err(anyhow!(
            "selected HTTP execution placement does not support worker-local measurement"
        ))
    }

    /// Worker-local measured execution with live response-frame forwarding.
    async fn execute_measured_streaming(
        &self,
        _turn: PreparedTurn,
        _context: MeasuredContext,
        _on_first_token: &dyn Fn(i64),
        _responses: &dyn TurnResponseObserver,
    ) -> Result<MeasuredOutcome> {
        Err(anyhow!(
            "selected HTTP execution placement does not support worker-local measurement"
        ))
    }

    /// Whether this placement accepts credits and returns them out of band
    /// (`--dispatch global-push`).
    fn supports_credit_dispatch(&self) -> bool {
        false
    }

    /// Route one credit to its worker and return WITHOUT awaiting the
    /// round-trip, after Python's `StickyCreditRouter::send_credit`.
    ///
    /// Synchronous on purpose: the issuer calls this from the single
    /// coordinator scheduling loop, and the whole point of the mode is that no
    /// coordinator future stays resident for the request's lifetime. A
    /// placement that cannot accept a credit immediately must queue it in
    /// routed order rather than block.
    fn send_credit(&self, _turn: PreparedTurn, _context: MeasuredContext) -> Result<()> {
        Err(anyhow!(
            "selected execution placement does not accept dispatched credits"
        ))
    }

    /// Receive the next out-of-band worker report, or `None` once no worker can
    /// report again.
    async fn next_credit_report(&self) -> Option<WorkerCreditReport> {
        None
    }

    /// Ask every worker to abandon the credit it is driving.
    ///
    /// Each abandoned credit is still RETURNED with a cancelled terminal, so the
    /// issuer's accounting closes normally instead of being fabricated
    /// coordinator-side.
    fn cancel_credits(&self) {}

    /// Warm each worker's dispatch path before timed issuance.
    ///
    /// The throwaway request uses the real sink with a no-op observer and is not
    /// recorded. Failures are non-fatal because the timed run reports persistent
    /// transport errors. Placements that require no warmup use the default no-op.
    async fn prewarm(&self, _turn: PreparedTurn) -> Result<()> {
        Ok(())
    }

    /// Drain every worker observer into flat `(uuid, record)` pairs after all
    /// dispatched turns reach terminal.
    ///
    /// Each record carries the worker's dense-local `request_index`; the
    /// coordinator reassigns the global dispatch ordinal and rewrites `admit_ns`
    /// during its uuid-join, then re-ingests in dispatch order. Backends without
    /// worker-local measurement return an empty vector.
    fn drain_records(&self, _end_ns: i64) -> Result<Vec<(Uuid, RecordIngest)>> {
        Ok(Vec::new())
    }

    /// Drain backend-owned execution resources after all dispatched turns have
    /// reached terminal. In-process direct execution owns no extra resources;
    /// thread pools and remote clients override this lifecycle hook.
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Object-safe [`PreparedTurn`] dispatch used by graph workers.
///
/// The trait is `#[async_trait(?Send)]` because each graph worker owns its sink
/// in `Rc`/`RefCell` on a thread-local `LocalSet`.
#[async_trait(?Send)]
pub trait Dispatcher {
    /// Execute one owned scheduler-free command, retaining its terminal
    /// response facts, and invoke `on_first_token` once with TTFT in nanoseconds.
    async fn dispatch_collect(
        &self,
        turn: PreparedTurn,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<DispatchResult>;

    /// Dispatch through a transport-owned, no-record bounded decision path.
    ///
    /// The default deliberately fails before dispatch. A transport must opt in
    /// only when it can enforce [`BoundedDecisionMode`] while streaming, before
    /// it creates terminal response or raw-record accumulation.
    async fn dispatch_bounded_decision(
        &self,
        _turn: PreparedTurn,
        _observer: &dyn RequestObserver,
        _on_first_token: &dyn Fn(i64),
        _mode: BoundedDecisionMode,
    ) -> Result<BoundedDecisionReader> {
        Err(anyhow!(
            "selected transport does not support bounded decisions"
        ))
    }

    /// Resolve report dimensions using the same endpoint selection as dispatch.
    fn inference_dimensions(&self, request: &Request) -> InferenceDimensions;

    /// Whether the transport can publish live response frames before terminal
    /// completion.
    fn supports_response_streaming(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    use crate::endpoints::{EndpointId, EndpointKey};
    use crate::multiturn::TurnDataPolicy;

    fn assert_send_sync<T: Send + Sync>() {}

    /// `Dispatchable` is `Send + Sync` and `Request` implements it, so anything
    /// reachable from a `Request` field must be too. A body form carrying
    /// interior mutability (a `OnceCell` materialization cache, say) breaks this
    /// as a compile error rather than a lint, which is what this pins.
    #[test]
    fn request_stays_send_and_sync() {
        assert_send_sync::<Request>();
        assert_send_sync::<RequestBody>();
    }

    #[test]
    fn bounded_decision_admits_the_exact_limit_before_exposing_a_reader() {
        let mode = BoundedDecisionMode::new(4).unwrap();
        let mut admission = BoundedDecisionAdmission::new(mode);
        admission.push(b"ab").unwrap();
        admission.push(b"cd").unwrap();

        let mut reader = admission.finish().unwrap();
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"abcd");
    }

    #[test]
    fn bounded_decision_derives_a_fixed_escaped_json_frame_allowance() {
        let mode = BoundedDecisionMode::new(3).unwrap();
        assert_eq!(mode.max_sse_frame_bytes(), 4 * 1024 + 18);
    }

    #[test]
    fn bounded_decision_rejects_the_first_byte_over_the_limit_while_streaming() {
        let mode = BoundedDecisionMode::new(4).unwrap();
        let mut admission = BoundedDecisionAdmission::new(mode);
        admission.push(b"ab").unwrap();
        admission.push(b"cd").unwrap();

        let error = admission.push(b"e").unwrap_err();
        assert!(error.to_string().contains("exceeds"));
        assert!(admission.finish().is_err());
    }

    #[test]
    fn bounded_decision_reader_reports_eof_after_admitted_bytes() {
        let mode = BoundedDecisionMode::new(3).unwrap();
        let mut admission = BoundedDecisionAdmission::new(mode);
        admission.push(b"abc").unwrap();

        let mut reader = admission.finish().unwrap();
        let mut bytes = [0_u8; 8];
        assert_eq!(reader.read(&mut bytes).unwrap(), 3);
        assert_eq!(&bytes[..3], b"abc");
        assert_eq!(reader.read(&mut bytes).unwrap(), 0);
    }

    #[tokio::test]
    async fn unsupported_dispatcher_fails_closed_for_bounded_decision_mode() {
        struct UnsupportedDispatcher;

        #[async_trait(?Send)]
        impl Dispatcher for UnsupportedDispatcher {
            async fn dispatch_collect(
                &self,
                _turn: PreparedTurn,
                _observer: &dyn RequestObserver,
                _on_first_token: &dyn Fn(i64),
            ) -> Result<DispatchResult> {
                unreachable!("the bounded-decision default must not dispatch")
            }

            fn inference_dimensions(&self, _request: &Request) -> InferenceDimensions {
                InferenceDimensions::default()
            }
        }

        struct SilentObserver;

        impl RequestObserver for SilentObserver {
            fn on_arrival(&self, _uuid: Uuid, _arrival_ms: f64, _input: usize, _output: usize) {}

            fn on_admit(&self, _uuid: Uuid, _admit_ms: f64, _reused_input: usize) {}

            fn on_token(&self, _uuid: Uuid, _at_ms: f64) {}

            fn on_terminal(
                &self,
                _uuid: Uuid,
                _status: crate::dispatch::collector::ReplayTerminalStatus,
            ) {
            }
        }

        let turn = PreparedTurn {
            runtime_session_id: "test-session".to_string(),
            request: Request {
                uuid: Uuid::nil(),
                input_length: 1,
                max_output_tokens: 1,
                prompt_text: None,
                body: None,
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
            model: "test-model".to_string(),
            endpoint: PreparedEndpointBinding::Prepared(PreparedEndpointReference {
                key: EndpointKey::from_index(0),
                endpoint_id: EndpointId::new("chat").unwrap(),
            }),
            endpoint_aware: true,
            data_policy: TurnDataPolicy::ordinary(),
            deferred: None,
        };

        let error = UnsupportedDispatcher
            .dispatch_bounded_decision(
                turn,
                &SilentObserver,
                &|_| {},
                BoundedDecisionMode::new(4).unwrap(),
            )
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("does not support bounded decisions")
        );
    }
}
