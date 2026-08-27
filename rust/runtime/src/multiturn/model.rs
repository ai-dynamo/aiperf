// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary values for multi-turn request execution.

use std::collections::BTreeMap;
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::body_plan::RequestBody;
use crate::dataset::{Handle, SegmentStore};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::endpoints::{EndpointId, EndpointKey};
use crate::graph::wire::OpenAiChatMessage;

/// Metadata and static user content for one turn in a conversation template.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnMetadata {
    /// Absolute trace timestamp in milliseconds, when present.
    pub timestamp_ms: Option<f64>,
    /// Relative think time after the previous response, in milliseconds.
    pub delay_ms: Option<f64>,
    /// Source-trace cache identities retained for offline simulator adapters.
    #[serde(default)]
    pub trace_hash_ids: Option<Handle>,
    /// User message placed on the wire for this turn.
    pub prompt_text: String,
    /// Input-token accounting for this static user message.
    pub input_length: usize,
    /// Maximum output tokens requested for this turn.
    pub max_output_tokens: usize,
}

/// Terminal response data needed to construct a continuation request.
#[derive(Clone, Debug)]
pub struct TurnResponse {
    /// Endpoint-normalized assistant text.
    pub text: String,
    /// Endpoint-normalized assistant message for lossless history replay.
    pub assistant_message: Option<Value>,
    /// Authoritative completion-token usage, when emitted by the server.
    pub completion_tokens: Option<u64>,
    /// Dispatch terminal state.
    pub terminal: ReplayTerminalStatus,
}

pub(super) trait RuntimeSessionBackend: fmt::Debug {
    /// Whether conversation-walking state has been built (see
    /// [`SampledSession::has_walking_state`]). Backends with no lazy state
    /// report `true`.
    #[cfg(test)]
    fn has_walking_state(&self) -> bool {
        true
    }

    fn available_turns(&self) -> usize;
    fn build_first_turn(
        &self,
        owner: &SampledSession,
        max_turns: Option<usize>,
    ) -> Result<TurnToSend>;
    fn build_turn_at(
        &self,
        owner: &SampledSession,
        start_index: usize,
        max_turns: Option<usize>,
    ) -> Result<TurnToSend>;
    fn next_metadata(&self, turn_index: usize) -> Result<TurnMetadata>;
    fn build_next_turn(
        &self,
        owner: &SampledSession,
        current: &TurnToSend,
        response: TurnResponse,
    ) -> Result<TurnToSend>;

    /// Build turn `index`'s scheduling identity WITHOUT materializing its
    /// request body, for a placement that materializes on the worker.
    ///
    /// The body-bearing fields come out as placeholders and `deferred_body` is
    /// set; `input_length`/`max_output_tokens` are filled from authored metadata
    /// so issuer-side accounting is unchanged, and the worker replaces them with
    /// what it actually built. The default materializes as usual, so a backend
    /// with no worker-side materializer keeps working unchanged.
    fn build_deferred_turn_at(
        &self,
        owner: &SampledSession,
        index: usize,
        max_turns: Option<usize>,
    ) -> Result<TurnToSend> {
        self.build_turn_at(owner, index, max_turns)
    }
}

/// A sampled runtime session for one reusable conversation template.
#[derive(Clone)]
pub struct SampledSession {
    /// Template identifier.
    pub conversation_id: String,
    /// Runtime session identifier used for scheduling and sticky routing.
    pub x_correlation_id: String,
    pub(super) backend: Rc<dyn RuntimeSessionBackend>,
}

/// One handle-bound view of authored trace block identities.
///
/// The hash vector stays in the unified segment pool. Simulator-aware
/// dispatchers resolve it only when they need a backend-specific prompt
/// representation; ordinary transports never copy or inspect it.
#[derive(Clone)]
pub struct StoredTraceHashIds {
    pub(super) handle: Handle,
    pub(super) segments: Arc<dyn SegmentStore>,
}

/// One handle-bound view of exact authored input token IDs.
///
/// Online materialization uses the same handle to build the vLLM JSON body;
/// Dynamo-offline resolves it directly and bypasses request-body decoding.
#[derive(Clone)]
pub struct StoredRawTokenIds {
    pub(super) handle: Handle,
    pub(super) segments: Arc<dyn SegmentStore>,
}

/// Copyable open endpoint identity carried from scheduling to execution.
///
/// Every execution worker prepares profiles in the same deterministic order.
/// The dense key selects the hot-path binding, while the canonical ID detects
/// a mismatched local or remote registry before any request is sent.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreparedEndpointReference {
    /// Worker-local dense table key.
    pub key: EndpointKey,
    /// Canonical open endpoint identity expected at that key.
    pub endpoint_id: EndpointId,
}

/// Endpoint selection retained by one schedulable turn.
///
/// The enum keeps endpoint-binding details explicit at turn-consumer boundaries.
#[derive(Clone)]
pub enum TurnEndpoint {
    /// Protocol-v2 open prepared binding selected only by stable key and ID.
    Prepared(PreparedEndpointReference),
}

/// Retention and disclosure policy carried with one materialized turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnDataPolicy {
    pub(super) restricted_transient: bool,
}

/// A turn awaiting issuance by a timing workload.
#[derive(Clone, Debug)]
pub struct TurnToSend {
    /// Stable request id used by measurement observers.
    pub uuid: Uuid,
    /// Effective wire model when dataset materialization selected or overrode it.
    pub effective_model: Option<String>,
    /// Template identifier.
    pub conversation_id: String,
    /// Runtime session identifier.
    pub x_correlation_id: String,
    /// Correlation identity sent to the backend and metrics association. This
    /// differs from the runtime session id for typed accuracy datasets.
    pub request_correlation_id: String,
    /// Zero-based turn index.
    pub turn_index: usize,
    /// Total turns this runtime session will send.
    pub num_turns: usize,
    /// Full accumulated input length used for accounting.
    pub input_length: usize,
    /// Requested output tokens.
    pub max_output_tokens: usize,
    /// Full OpenAI message history, including captured prior replies.
    pub messages: Vec<OpenAiChatMessage>,
    /// Exact segment-backed request body, when the native dataset seam built it.
    pub request_body: Option<RequestBody>,
    /// Per-turn HTTP headers.
    pub request_headers: BTreeMap<String, String>,
    /// Per-turn URL query parameters.
    pub request_parameters: BTreeMap<String, String>,
    /// Endpoint path selected by the endpoint resolver.
    pub endpoint_path: Option<String>,
    /// Prepared endpoint identity used for execution.
    pub endpoint: TurnEndpoint,
    /// Whether this endpoint returns an SSE stream.
    pub streaming: bool,
    /// Audio duration propagated into ASR metrics.
    pub audio_duration_seconds: Option<f64>,
    /// Exact wire image count from composition, letting dispatch skip a
    /// full-body re-parse for the `num_images` metric. `None` when unknown.
    pub image_count: Option<u32>,
    /// Absolute trace timestamp for this turn, if any.
    pub timestamp_ms: Option<f64>,
    /// Relative delay for this turn, if any.
    pub delay_ms: Option<f64>,
    /// Source-trace cache identities for a simulator-aware dispatch adapter.
    pub trace_hash_ids: Option<StoredTraceHashIds>,
    /// Exact authored IDs for a token-native backend dispatch adapter.
    pub raw_token_ids: Option<StoredRawTokenIds>,
    /// Content retention/cache/diagnostic policy fixed during materialization.
    pub data_policy: TurnDataPolicy,
    /// Fixed cancellation delay selected at issuance and armed at send-complete.
    pub cancel_after_ns: Option<i64>,
    /// Effective endpoint index, including a continuation's session pin.
    pub url_index: Option<u32>,
    /// Whether this turn's request body was deliberately NOT materialized by the
    /// issuer, leaving the worker that receives the credit to build it from the
    /// resident dataset (`--dispatch global-push`). The body-bearing fields are
    /// placeholders until then; see [`WorkerMaterializer`](super::WorkerMaterializer).
    pub deferred_body: bool,
    pub(super) session: SampledSession,
}

/// The identity half of a routed credit: everything a worker needs to rebuild
/// the request, and nothing more.
///
/// The Rust counterpart of Python's `Credit`
/// (`src/aiperf/credit/structs.py`), which likewise carries ids and indices
/// rather than a materialized body.
#[derive(Clone, Debug)]
pub struct CreditIdentity {
    /// Dataset template this credit draws from.
    pub conversation_id: String,
    /// Runtime session id; the sticky-routing key and the session-cache key.
    pub x_correlation_id: String,
    /// Zero-based turn index within the conversation.
    pub turn_index: usize,
    /// Total turns this runtime session will send.
    pub num_turns: usize,
}
