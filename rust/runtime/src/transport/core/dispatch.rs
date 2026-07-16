// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral dispatch DTOs shared by every transport.
//!
//! [`Request`] is the load generator's own transport-neutral request type,
//! dispatched by every transport (http, grpc, dynosim, dry_run) through the
//! `loadgen_core` [`Dispatchable`] seam. [`DispatchResult`], [`MeasuredContext`],
//! and [`MeasuredOutcome`] are the backend-neutral result/measurement carriers
//! consumed by scheduling, record processing, and the worker-local measured
//! execution path. None of these carry transport-specific state, so they live in
//! `transport::core` rather than any one transport module.

use std::collections::BTreeMap;
use std::fmt;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::Value;
use uuid::Uuid;

use loadgen_core::sink::{Dispatchable, RequestObserver};

use crate::metrics::RequestMetricMetadata;
use crate::metrics_core::{InferenceDimensions, MetricsConfig, RecordIngest};
use crate::multiturn::{PreparedEndpointReference, TurnDataPolicy, TurnEndpoint, TurnToSend};
use crate::scheduled::{TurnDispatchOutcome, TurnResponseObserver};
use crate::transport::core::record::RequestRecord;

/// A slim online request carrying prompt text — the load generator's own
/// transport-neutral request type, dispatched by every transport (http, grpc,
/// dynosim, dry_run). Implementing [`Dispatchable`] is all the dispatch seam
/// requires.
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

impl fmt::Debug for Request {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Request")
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

/// HTTP-specific terminal result retained by raw-artifact consumers.
///
/// Policy-neutral workloads continue to consume [`TurnDispatchOutcome`]. The
/// native subprocess runner calls the concrete collection method only when it
/// must preserve HTTP wire facts; alternate backends do not inherit an HTTP
/// dependency through the shared `TurnDispatcher` seam.
#[derive(Clone, Debug)]
pub struct DispatchResult {
    /// Backend-neutral result consumed by scheduling and record processors.
    pub outcome: TurnDispatchOutcome,
    /// Canonical JSON payload before transport-specific body preparation.
    pub request_payload: Bytes,
    /// Exact HTTP transport record.
    pub record: RequestRecord,
}

/// Coordinator-supplied per-turn measurement facts for the worker-local
/// accumulation path.
///
/// The scheduled runner no longer replays every token onto one coordinator
/// observer. Instead, each execution worker owns its own metrics observer
/// and this context carries the coordinator-known arrival facts and metadata the
/// worker registers locally before dispatch. Fields the coordinator only learns
/// after dispatch (`phase`, `session_num`, `has_credit_timestamp`, and the global
/// dispatch `request_index`) are deliberately absent from `metadata`; the
/// coordinator patches them onto the drained record at finish so all
/// credit/phase logic stays exactly where the single-observer path had it.
#[derive(Clone, Debug)]
pub struct MeasuredContext {
    /// Arrival timestamp in milliseconds relative to the run origin, computed
    /// coordinator-side at issue exactly as the single-observer path did.
    pub arrival_ms: f64,
    /// Coordinator-known input length forwarded to the worker's `on_arrival`.
    pub input_length: usize,
    /// Coordinator-known requested output length forwarded to `on_arrival`.
    pub requested_output_length: usize,
    /// Begin-known request metadata (turn index, conversation, correlation,
    /// dimensions, audio duration); no `request_index`/`phase`/`session_num`.
    pub metadata: RequestMetricMetadata,
    /// Whether a live-results sink is attached and the worker must return a
    /// non-consuming cloned record for live emission.
    pub wants_live_record: bool,
    /// Whether the returned record should be *moved out* of the worker observer
    /// (freeing its token storage) rather than cloned. Set in metrics-only
    /// (sketch) mode, where the coordinator folds each record into a bounded
    /// streaming accumulator and immediately drops it, so retaining the
    /// observer's copy would defeat the O(sketch) memory bound. Only consulted
    /// when `wants_live_record` is set.
    pub consume_record: bool,
}

/// Result of a worker-local measured execution: the transport outcome plus an
/// optional non-consuming cloned record for the live-results sink.
///
/// The authoritative record stays inside the worker observer for the end-of-run
/// drain; `live_record` (present only when [`MeasuredContext::wants_live_record`]
/// is set) is a clone so live emission never removes it from the final merge.
#[derive(Debug)]
pub struct MeasuredOutcome {
    /// Backend-neutral dispatch result consumed by scheduling/record processors.
    pub result: DispatchResult,
    /// Non-consuming cloned record for a live sink, when requested.
    pub live_record: Option<RecordIngest>,
}

/// Owned execution command handed from the single logical dispatcher to an
/// injected execution backend.
///
/// The scheduling-only [`TurnToSend`] retains an `Rc` session backend so that
/// continuations can be materialized locally. This projection deliberately
/// removes that scheduler state: every remaining field is owned and `Send`, and
/// the endpoint's stable identity is carried by the worker-local prepared
/// binding key in [`PreparedEndpointReference`]. Local backends retain the
/// already-resolved prepared adapter allocation.
///
/// Transport-neutral: consumed by http, grpc, graph, and dry-run execution, so
/// it lives in `transport::core` rather than any one transport module.
#[derive(Clone)]
pub struct PreparedTurn {
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
}

impl fmt::Debug for PreparedTurn {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedTurn")
            .field("request", &self.request)
            .field("model", &self.model)
            .field("endpoint", &self.endpoint)
            .field("endpoint_aware", &self.endpoint_aware)
            .field("data_policy", &self.data_policy)
            .finish()
    }
}

/// Endpoint selection retained by one scheduler-free execution command.
///
/// This is an enum rather than a bare [`PreparedEndpointReference`] to keep the
/// seam open for a future non-prepared execution binding. Transport-neutral
/// despite the historical `Http` name it carried: grpc, graph, and dry-run
/// execution all consume it.
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
            request: Request {
                uuid: turn.uuid,
                input_length: turn.input_length,
                max_output_tokens: turn.max_output_tokens,
                prompt_text: None,
                request_body: None,
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
            model,
            endpoint,
            endpoint_aware,
            data_policy,
        }
    }
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

    /// Warm the dispatch cold path on every execution worker before timed
    /// issuance begins, so the first authored request is not delayed relative to
    /// its schedule by one-time setup (connection establishment, endpoint body
    /// materialization, tokenizer/JIT warmup).
    ///
    /// This is the Rust-native analogue of the Python engine's ZMQ "workers
    /// ready, go" barrier: each worker sends one throwaway request through its
    /// real sink and discards the result, so the timed run starts warm and all
    /// workers begin issuing from the same warmed state. The warmup request is
    /// **never recorded** (a no-op observer, discarded outcome), so it does not
    /// enter the metrics. Failures are non-fatal — the timed run surfaces any
    /// real transport error itself. The default is a no-op for placements that
    /// need no warmup.
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
    fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Transport-neutral [`PreparedTurn`] dispatch seam shared by the graph path.
///
/// This is deliberately distinct from the HTTP `HttpRequestDispatcher` (the
/// scheduled path, keyed on [`Request`]). Both the HTTP `TransportSink` and the
/// native gRPC sink already expose an identical inherent `dispatch_collect(turn:
/// PreparedTurn, …)`; this object-safe trait unifies them so a graph sink can
/// hold its transport as `Rc<dyn Dispatcher>` and later dispatch a graph dataset
/// over gRPC without branching on a concrete backend. It is `#[async_trait(?Send)]`
/// because graph workers own their sink in `Rc`/`RefCell` on a thread-local
/// `LocalSet`.
///
/// Extension point: a future non-HTTP/non-gRPC `PreparedTurn` transport
/// implements this trait and nothing in the graph runtime changes.
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

    /// Resolve report dimensions using the same endpoint selection as dispatch.
    fn inference_dimensions(&self, request: &Request) -> InferenceDimensions;

    /// Whether the transport can publish live response frames before terminal
    /// completion.
    fn supports_response_streaming(&self) -> bool {
        false
    }
}
