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

use bytes::Bytes;
use serde_json::Value;
use uuid::Uuid;

use loadgen_core::sink::Dispatchable;

use crate::metrics::RequestMetricMetadata;
use crate::metrics_core::RecordIngest;
use crate::scheduled::TurnDispatchOutcome;
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
