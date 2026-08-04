// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dataset-backed multi-turn conversation sourcing and prompt materialization.
//!
//! A [`ConversationSource`] samples
//! reusable templates, mints a distinct runtime correlation id for each session,
//! caps virtual-history sessions safely to the sampled template length, and
//! builds continuation turns. Static user messages live in the unified
//! `aiperf-dataset` content-addressed [`SegmentStore`]; real assistant replies
//! are appended dynamically before the next user segment, preserving growing
//! multi-turn context without reserializing stored static messages.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::rc::Rc;
use std::sync::Arc;

use crate::body_plan::RequestBody;
use crate::cellular::partition::{CellPartition, ModuloCellPartition};
use crate::dataset::{
    ConversationSession as NativeConversationSession, Dataset as NativeDataset,
    EndpointRequestMaterializer, Handle, Overrides, Payload, RequestMaterializer, Sampler,
    SamplerRegistry, SegmentStore, SequentialSampler, TextTokenizer, TiktokenTokenizer,
};
use crate::dataset::TurnEndpointLookup;
use crate::dataset::request::reply_image_count;
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::endpoints::{
    CreditPhase, Endpoint, EndpointId, EndpointKey, Media as EndpointMedia, PreparedEndpoint,
    PreparedEndpointTable, ShapeLowerer, Turn as EndpointTurn, TurnMessageLowerer,
};
use crate::graph::wire::OpenAiChatMessage;
use crate::rng::RngRoot;
use crate::timing::{RunState, StopConfig};
use anyhow::{Result, anyhow, bail};
use bytes::Bytes;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// Policy for deriving the input length attached to one materialized request.
///
/// The endpoint and decoded body are passed together because endpoint dialects
/// own their extraction rules. Implementations may preserve authored dataset
/// counts or retokenize the exact body that will be serialized on the wire.
pub trait InputTokenCounter: Send + Sync {
    /// Count input tokens for one fully materialized endpoint request.
    fn count_input_tokens(
        &self,
        endpoint: &dyn Endpoint,
        body: &[u8],
        authored_input_tokens: u64,
    ) -> Result<u64>;

    /// Count input tokens through a worker-local prepared endpoint binding.
    ///
    /// Policies that preserve authored counts inherit this default.
    fn count_prepared_input_tokens(
        &self,
        _endpoint: &dyn PreparedEndpoint,
        _body: &[u8],
        authored_input_tokens: u64,
    ) -> Result<u64> {
        Ok(authored_input_tokens)
    }

    /// Count input tokens for a materialized dispatch that may already carry the
    /// endpoint-reported input structure.
    ///
    /// `extracted` is `Some` only when the endpoint reported the structure of
    /// this exact body through [`PreparedEndpoint::extracted`]. The default
    /// ignores it and defers to [`Self::count_prepared_input_tokens`], which
    /// parses the body and is always correct.
    ///
    /// An implementation overriding this must also override
    /// [`Self::count_prepared_input_tokens`]: graph dispatch counts through that
    /// method directly and never sees a [`crate::dataset::MaterializedRequest`].
    fn count_materialized_input_tokens(
        &self,
        endpoint: &dyn PreparedEndpoint,
        body: &[u8],
        _extracted: Option<&crate::endpoints::ExtractedPayload>,
        authored_input_tokens: u64,
    ) -> Result<u64> {
        self.count_prepared_input_tokens(endpoint, body, authored_input_tokens)
    }

    /// Whether immutable first turns may reuse a previously computed count.
    ///
    /// Stateful or externally backed counters keep the conservative default.
    /// Deterministic body/tokenizer policies opt in so repeated dataset samples
    /// do not rerun tokenization for byte-identical static prompts.
    fn caches_static_first_turns(&self) -> bool {
        false
    }
}

/// Input-count policy that preserves the count authored by the dataset.
#[derive(Debug, Default)]
pub struct AuthoredInputTokenCounter;

impl InputTokenCounter for AuthoredInputTokenCounter {
    fn count_input_tokens(
        &self,
        _endpoint: &dyn Endpoint,
        _body: &[u8],
        authored_input_tokens: u64,
    ) -> Result<u64> {
        Ok(authored_input_tokens)
    }
}

/// Exact wire-body token accounting with optional Hugging Face chat templates.
///
/// Endpoint-specific
/// extraction runs against the final request body, chat templates are
/// best-effort, tool text is added outside the role/content template, and the
/// ordinary path joins extracted text with one space.
pub struct EndpointInputTokenCounter {
    tokenizer: Arc<dyn TextTokenizer>,
    apply_chat_template: bool,
}

impl EndpointInputTokenCounter {
    /// Construct exact endpoint-aware input accounting.
    pub fn new(tokenizer: Arc<dyn TextTokenizer>, apply_chat_template: bool) -> Self {
        Self {
            tokenizer,
            apply_chat_template,
        }
    }

    fn add_text_count(&self, count: u64, texts: &[String]) -> Result<u64> {
        if texts.is_empty() {
            return Ok(count);
        }
        let tokens = u64::try_from(self.tokenizer.count(&texts.join(" "))?)
            .map_err(|_| anyhow!("input token count exceeds u64"))?;
        count
            .checked_add(tokens)
            .ok_or_else(|| anyhow!("input token count overflowed u64"))
    }

    fn count_extracted(
        &self,
        extracted: &crate::endpoints::ExtractedPayload,
        authored_input_tokens: u64,
    ) -> Result<u64> {
        if self.apply_chat_template
            && let Some(messages) = extracted
                .messages
                .as_deref()
                .filter(|items| !items.is_empty())
            && let Some(tokens) = self
                .tokenizer
                .apply_chat_template(messages, true)
                .ok()
                .flatten()
        {
            let templated = u64::try_from(tokens.len())
                .map_err(|_| anyhow!("templated input token count exceeds u64"))?;
            let count = extracted
                .pretokenised_token_count
                .checked_add(templated)
                .ok_or_else(|| anyhow!("input token count overflowed u64"))?;
            return self.add_text_count(count, &extracted.tool_texts);
        }
        if !extracted.texts.is_empty() {
            return self.add_text_count(extracted.pretokenised_token_count, &extracted.texts);
        }
        if extracted.pretokenised_token_count > 0 {
            return Ok(extracted.pretokenised_token_count);
        }
        Ok(authored_input_tokens)
    }
}

impl fmt::Debug for EndpointInputTokenCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EndpointInputTokenCounter")
            .field("tokenizer", &self.tokenizer.name())
            .field("apply_chat_template", &self.apply_chat_template)
            .finish()
    }
}

impl InputTokenCounter for EndpointInputTokenCounter {
    fn count_input_tokens(
        &self,
        endpoint: &dyn Endpoint,
        body: &[u8],
        authored_input_tokens: u64,
    ) -> Result<u64> {
        let Ok(body) = serde_json::from_slice(body) else {
            return Ok(authored_input_tokens);
        };
        self.count_extracted(
            &endpoint.extract_payload_inputs(&body),
            authored_input_tokens,
        )
    }

    fn count_prepared_input_tokens(
        &self,
        endpoint: &dyn PreparedEndpoint,
        body: &[u8],
        authored_input_tokens: u64,
    ) -> Result<u64> {
        let Ok(body) = serde_json::from_slice(body) else {
            return Ok(authored_input_tokens);
        };
        self.count_extracted(
            &endpoint.extract_payload_inputs(&body),
            authored_input_tokens,
        )
    }

    fn count_materialized_input_tokens(
        &self,
        endpoint: &dyn PreparedEndpoint,
        body: &[u8],
        extracted: Option<&crate::endpoints::ExtractedPayload>,
        authored_input_tokens: u64,
    ) -> Result<u64> {
        match extracted {
            Some(extracted) => self.count_extracted(extracted, authored_input_tokens),
            None => self.count_prepared_input_tokens(endpoint, body, authored_input_tokens),
        }
    }

    fn caches_static_first_turns(&self) -> bool {
        true
    }
}

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

impl TurnMetadata {
    /// Build one turn with no trace timing metadata.
    pub fn untimed(
        prompt_text: impl Into<String>,
        input_length: usize,
        max_output_tokens: usize,
    ) -> Self {
        Self {
            timestamp_ms: None,
            delay_ms: None,
            trace_hash_ids: None,
            prompt_text: prompt_text.into(),
            input_length,
            max_output_tokens,
        }
    }
}

/// Reusable conversation template loaded from a dataset.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConversationMetadata {
    /// Stable template id. Runtime sessions mint separate correlation ids.
    pub conversation_id: String,
    /// Ordered turns in this conversation.
    pub turns: Vec<TurnMetadata>,
}

/// One conversation's canonical per-turn request bodies, generated up front from
/// the resident dataset for a pure `inputs.json` export.
///
/// This is the dataset-derived analog of the during-run `record_input_payload`
/// capture: the session id is the authored conversation id and each payload is the
/// exact canonical body `crate::transport::http::transport::prepare_request` would
/// retain as `canonical_body()` when the same turn is dispatched, so an
/// `inputs.json` written from these bytes is byte-identical to the capture-based
/// output (see [`NativeDatasetConversationSource::materialize_input_payloads`]).
#[derive(Clone, Debug)]
pub struct UpFrontInputSession {
    /// Authored conversation id shared with `SampledSession::conversation_id`.
    pub session_id: String,
    /// One canonical request body per turn, ordered by turn index.
    pub payloads: Vec<Bytes>,
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

trait RuntimeSessionBackend: fmt::Debug {
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
}

/// A sampled runtime session for one reusable conversation template.
#[derive(Clone)]
pub struct SampledSession {
    /// Template identifier.
    pub conversation_id: String,
    /// Runtime session identifier used for scheduling and sticky routing.
    pub x_correlation_id: String,
    backend: Rc<dyn RuntimeSessionBackend>,
}

impl fmt::Debug for SampledSession {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SampledSession")
            .field("conversation_id", &self.conversation_id)
            .field("x_correlation_id", &self.x_correlation_id)
            .field("turns", &self.backend.available_turns())
            .finish()
    }
}

impl SampledSession {
    /// Number of turns available in the sampled template.
    pub fn available_turns(&self) -> usize {
        self.backend.available_turns()
    }

    /// Build the first turn, clamping virtual history to the sampled template.
    pub fn build_first_turn(&self, max_turns: Option<usize>) -> Result<TurnToSend> {
        self.backend.build_first_turn(self, max_turns)
    }

    /// Jump-resume the session at `start_index`, reconstructing the recorded
    /// 0..=`start_index` context directly from the dataset, and materialize that turn
    /// with `turn_index == start_index`.
    ///
    /// `build_turn_at(0, max_turns)` is identical to
    /// [`build_first_turn`](Self::build_first_turn). Non-zero resume is only faithful
    /// for context modes whose context is self-contained in the recorded conversation
    /// (see [`ConversationSession::seek_to`](crate::dataset::ConversationSession::seek_to));
    /// live-reply-capture modes fail closed. This is the primitive the
    /// accelerated-cache-warmup workload uses to start a lane at a runtime-determined
    /// drained frontier without replaying every prior turn's dispatch.
    pub fn build_turn_at(
        &self,
        start_index: usize,
        max_turns: Option<usize>,
    ) -> Result<TurnToSend> {
        self.backend.build_turn_at(self, start_index, max_turns)
    }

    fn next_metadata(&self, turn_index: usize) -> Result<TurnMetadata> {
        self.backend.next_metadata(turn_index)
    }

    fn build_next_turn(&self, current: &TurnToSend, response: TurnResponse) -> Result<TurnToSend> {
        self.backend.build_next_turn(self, current, response)
    }
}

/// One handle-bound view of authored trace block identities.
///
/// The hash vector stays in the unified segment pool. Simulator-aware
/// dispatchers resolve it only when they need a backend-specific prompt
/// representation; ordinary transports never copy or inspect it.
#[derive(Clone)]
pub struct StoredTraceHashIds {
    handle: Handle,
    segments: Arc<dyn SegmentStore>,
}

impl StoredTraceHashIds {
    fn new(handle: Handle, segments: Arc<dyn SegmentStore>) -> Self {
        Self { handle, segments }
    }

    /// Resolve the authored block identities and source block size.
    pub fn resolve(&self) -> Result<(&[i64], usize)> {
        match self.segments.get(self.handle)? {
            Payload::TraceHashIds {
                hash_ids,
                block_size,
            } => Ok((hash_ids, *block_size)),
            payload => bail!(
                "segment {} contains {}, expected trace-hash-ids",
                self.handle,
                payload.kind_name()
            ),
        }
    }

    /// Dense segment handle retained for diagnostics and tests.
    pub const fn handle(&self) -> Handle {
        self.handle
    }
}

impl fmt::Debug for StoredTraceHashIds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoredTraceHashIds")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
}

/// One handle-bound view of exact authored input token IDs.
///
/// Online materialization uses the same handle to build the vLLM JSON body;
/// Dynamo-offline resolves it directly and bypasses request-body decoding.
#[derive(Clone)]
pub struct StoredRawTokenIds {
    handle: Handle,
    segments: Arc<dyn SegmentStore>,
}

impl StoredRawTokenIds {
    fn new(handle: Handle, segments: Arc<dyn SegmentStore>) -> Self {
        Self { handle, segments }
    }

    /// Resolve the validated, non-empty raw token sequence.
    pub fn resolve(&self) -> Result<&[u32]> {
        match self.segments.get(self.handle)? {
            Payload::TokenIds { token_ids } if !token_ids.is_empty() => Ok(token_ids),
            payload => bail!(
                "segment {} contains {}, expected non-empty token-ids",
                self.handle,
                payload.kind_name()
            ),
        }
    }

    /// Dense segment handle retained for diagnostics and tests.
    pub const fn handle(&self) -> Handle {
        self.handle
    }
}

impl fmt::Debug for StoredRawTokenIds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoredRawTokenIds")
            .field("handle", &self.handle)
            .finish_non_exhaustive()
    }
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

/// One coordinator-local prepared binding selected for request materialization.
pub struct ResolvedPreparedEndpoint<'a> {
    /// Stable identity carried to the execution worker.
    pub reference: PreparedEndpointReference,
    /// Coordinator-local binding used to build the exact request body.
    pub endpoint: &'a dyn PreparedEndpoint,
}

/// Open selection seam for prepared dataset endpoint overrides.
///
/// [`PreparedEndpointTableResolver`] provides local online resolution without
/// coupling session, scheduling, or dispatch code to table lookup.
pub trait PreparedTurnEndpointResolver: fmt::Debug {
    /// Resolve an authored per-turn endpoint name, or the run default when
    /// absent, to one dense prepared binding.
    fn resolve(&self, name: Option<&str>) -> Result<ResolvedPreparedEndpoint<'_>>;
}

/// Every prepared resolver is also the endpoint lookup dataset precompute needs,
/// so image counting reaches per-turn overrides without a second seam.
impl<T: PreparedTurnEndpointResolver + ?Sized> TurnEndpointLookup for T {
    fn endpoint_for(&self, name: Option<&str>) -> Option<&dyn PreparedEndpoint> {
        self.resolve(name).ok().map(|resolved| resolved.endpoint)
    }
}

/// Dense-table prepared endpoint resolver used by local online execution.
pub struct PreparedEndpointTableResolver {
    table: Rc<PreparedEndpointTable>,
    default: PreparedEndpointReference,
    named: HashMap<String, PreparedEndpointReference>,
}

impl fmt::Debug for PreparedEndpointTableResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names = self.named.keys().collect::<Vec<_>>();
        names.sort_unstable();
        formatter
            .debug_struct("PreparedEndpointTableResolver")
            .field("default", &self.default)
            .field("names", &names)
            .finish_non_exhaustive()
    }
}

impl PreparedEndpointTableResolver {
    /// Bind one default endpoint and all descriptor aliases to a prepared table.
    pub fn single(
        table: Rc<PreparedEndpointTable>,
        default: PreparedEndpointReference,
    ) -> Result<Self> {
        let endpoint = table.get(default.key)?;
        if endpoint.descriptor().id != default.endpoint_id.as_str() {
            bail!(
                "prepared endpoint key {} contains {:?}, expected {:?}",
                default.key.index(),
                endpoint.descriptor().id,
                default.endpoint_id.as_str()
            );
        }
        let mut named = HashMap::new();
        for name in std::iter::once(endpoint.descriptor().id)
            .chain(endpoint.descriptor().aliases.iter().copied())
        {
            named.insert(normalize_endpoint_name(name), default.clone());
        }
        Ok(Self {
            table,
            default,
            named,
        })
    }
}

impl PreparedTurnEndpointResolver for PreparedEndpointTableResolver {
    fn resolve(&self, name: Option<&str>) -> Result<ResolvedPreparedEndpoint<'_>> {
        let reference = match name {
            None => self.default.clone(),
            Some(name) => self
                .named
                .get(&normalize_endpoint_name(name))
                .cloned()
                .ok_or_else(|| anyhow!("dataset endpoint override {name:?} was not prepared"))?,
        };
        let endpoint = self.table.get(reference.key)?;
        if endpoint.descriptor().id != reference.endpoint_id.as_str() {
            bail!(
                "prepared endpoint key {} contains {:?}, expected {:?}",
                reference.key.index(),
                endpoint.descriptor().id,
                reference.endpoint_id.as_str()
            );
        }
        Ok(ResolvedPreparedEndpoint {
            reference,
            endpoint,
        })
    }
}

fn normalize_endpoint_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '/'], "_")
}

/// Pre-serialize static message turns for the default prepared endpoint.
/// Endpoints without per-turn message arrays remain on the live rendering path,
/// and per-turn endpoint overrides are not lowered.
fn lower_static_messages(
    dataset: &mut NativeDataset,
    endpoint_resolver: &dyn PreparedTurnEndpointResolver,
    primary_model_name: &str,
) -> Result<()> {
    let resolved = endpoint_resolver.resolve(None)?;
    if let Some(lowerer) = ShapeLowerer::for_descriptor_id(resolved.endpoint.descriptor().id) {
        dataset
            .lower_messages_for_endpoint(&lowerer)
            .map_err(|error| anyhow!("failed to lower static messages: {error}"))?;
    }
    // Cache eligible static body plans against the default prepared endpoint
    // after lowering so dispatch can reuse byte-identical plans.
    dataset
        .precompute_body_plans(resolved.endpoint, primary_model_name)
        .map_err(|error| anyhow!("failed to precompute body plans: {error}"))?;
    // Establish each turn's wire image count here too, so dispatch never
    // deserializes the body it just assembled to recover one number.
    dataset
        .precompute_image_counts(endpoint_resolver, primary_model_name)
        .map_err(|error| anyhow!("failed to precompute image counts: {error}"))?;
    Ok(())
}

/// Endpoint selection retained by one schedulable turn.
///
/// The enum keeps endpoint-binding details explicit at turn-consumer boundaries.
#[derive(Clone)]
pub enum TurnEndpoint {
    /// Protocol-v2 open prepared binding selected only by stable key and ID.
    Prepared(PreparedEndpointReference),
}

impl fmt::Debug for TurnEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepared(reference) => formatter
                .debug_tuple("PreparedEndpoint")
                .field(reference)
                .finish(),
        }
    }
}

/// Retention and disclosure policy carried with one materialized turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnDataPolicy {
    restricted_transient: bool,
}

impl TurnDataPolicy {
    /// Ordinary benchmark content may use configured capture/export paths.
    pub const fn ordinary() -> Self {
        Self {
            restricted_transient: false,
        }
    }

    /// Restricted evaluator content may exist only for the live dispatch.
    pub const fn restricted_transient() -> Self {
        Self {
            restricted_transient: true,
        }
    }

    /// Whether raw request/response content may be retained after dispatch.
    pub const fn retain_raw_exchange(self) -> bool {
        !self.restricted_transient
    }

    /// Whether a result cache may retain content from this turn.
    pub const fn allow_result_cache(self) -> bool {
        !self.restricted_transient
    }

    /// Whether content-bearing failure detail may enter diagnostics.
    pub const fn allow_content_diagnostics(self) -> bool {
        !self.restricted_transient
    }

    /// Whether raw content may participate in a public digest.
    pub const fn allow_public_content_hash(self) -> bool {
        !self.restricted_transient
    }
}

impl Default for TurnDataPolicy {
    fn default() -> Self {
        Self::ordinary()
    }
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
    session: SampledSession,
}

impl TurnToSend {
    /// Whether this is the session's final root turn.
    pub fn is_final_turn(&self) -> bool {
        self.turn_index + 1 >= self.num_turns
    }
}

/// Metadata retained from issue until a dispatch returns.
#[derive(Clone, Debug)]
pub struct IssuedCredit {
    /// Monotonic credit id assigned before dispatch.
    pub id: u64,
    /// Clock timestamp at the issuer boundary before backend dispatch.
    pub issued_ns: i64,
    /// The issued turn, including materialized prompt history.
    pub turn: TurnToSend,
    /// Per-request cancellation scalar selected at issuance.
    pub cancel_after_ns: Option<i64>,
    /// URL selector output carried only by turn 0; continuation credits leave
    /// this absent and use session-pinned state.
    pub url_index: Option<u32>,
}

impl IssuedCredit {
    /// Build issued-credit metadata from a turn and assigned id.
    pub fn from_turn(id: u64, issued_ns: i64, turn: &TurnToSend) -> Self {
        Self::from_issued_turn(id, issued_ns, turn, turn.url_index)
    }

    /// Build issued metadata while keeping the selector's turn-0 output
    /// distinct from the effective endpoint pin on `turn`.
    pub fn from_issued_turn(
        id: u64,
        issued_ns: i64,
        turn: &TurnToSend,
        url_index: Option<u32>,
    ) -> Self {
        Self {
            id,
            issued_ns,
            turn: turn.clone(),
            cancel_after_ns: turn.cancel_after_ns,
            url_index,
        }
    }

    /// Whether this credit represents the final turn for its session.
    pub fn is_final_turn(&self) -> bool {
        self.turn.is_final_turn()
    }
}

/// Source seam for sampling sessions and materializing continuation turns.
///
/// Dataset-backed, generated, remote, or mmap sources can implement this trait;
/// timing workloads never branch on a concrete loader kind.
pub trait ConversationSource {
    /// Stable dataset metadata used by strategy setup.
    fn conversations(&self) -> &[ConversationMetadata];

    /// Sample the next runtime session, optionally using a caller-supplied
    /// correlation id (user-centric uses the monotonically assigned user id).
    fn next(&mut self, x_correlation_id: Option<String>) -> Result<SampledSession>;

    /// Build a runtime session for a specific template. Fixed-schedule replay
    /// uses this instead of the sampler so every trace entry is replayed once.
    fn session_for(
        &self,
        conversation_id: &str,
        x_correlation_id: String,
    ) -> Result<SampledSession>;

    /// Generate every conversation's canonical per-turn request bodies up front for a
    /// pure `inputs.json` export, or `None` when this source cannot reproduce them
    /// without dispatching (the caller then keeps the during-run capture path).
    ///
    /// The default is `None` (unsupported); the dataset-backed
    /// [`NativeDatasetConversationSource`] overrides it (see
    /// [`NativeDatasetConversationSource::materialize_input_payloads`]).
    fn materialize_input_payloads(&self) -> Result<Option<Vec<UpFrontInputSession>>> {
        Ok(None)
    }

    /// Metadata for the turn after `credit`, with a checked out-of-range error.
    fn next_turn_metadata(&self, credit: &IssuedCredit) -> Result<TurnMetadata> {
        credit.turn.session.next_metadata(credit.turn.turn_index)
    }

    /// Build the continuation after `credit`, splicing `response_text` into its
    /// growing prompt. Returns `None` for a final credit.
    fn next_turn(
        &self,
        credit: &IssuedCredit,
        response: TurnResponse,
    ) -> Result<Option<TurnToSend>> {
        if credit.is_final_turn() {
            return Ok(None);
        }
        credit
            .turn
            .session
            .build_next_turn(&credit.turn, response)
            .map(Some)
    }
}

#[derive(Clone)]
enum NativeSessionEndpoint {
    Prepared {
        primary_model_name: String,
        endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    },
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum StaticInputCountEndpoint {
    Prepared(EndpointKey),
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct StaticInputCountKey {
    template_index: usize,
    endpoint: StaticInputCountEndpoint,
}

type StaticInputCountCache = Rc<RefCell<FxHashMap<StaticInputCountKey, u64>>>;

impl fmt::Debug for NativeSessionEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Prepared {
                primary_model_name,
                endpoint_resolver,
            } => formatter
                .debug_struct("PreparedSessionEndpoint")
                .field("primary_model_name", primary_model_name)
                .field("endpoint_resolver", endpoint_resolver)
                .finish_non_exhaustive(),
        }
    }
}

#[derive(Clone)]
struct NativeSessionBackend {
    session: RefCell<NativeConversationSession>,
    template_index: usize,
    metadata: ConversationMetadata,
    endpoint: NativeSessionEndpoint,
    materializer: Arc<dyn RequestMaterializer>,
    response_tokenizer: Arc<dyn TextTokenizer>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    static_input_count_cache: StaticInputCountCache,
    segments: Arc<dyn SegmentStore>,
    default_output_tokens: usize,
}

impl fmt::Debug for NativeSessionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeSessionBackend")
            .field("conversation_id", &self.metadata.conversation_id)
            .field("turns", &self.metadata.turns.len())
            .finish()
    }
}

impl RuntimeSessionBackend for NativeSessionBackend {
    fn available_turns(&self) -> usize {
        self.metadata.turns.len()
    }

    fn build_first_turn(
        &self,
        owner: &SampledSession,
        max_turns: Option<usize>,
    ) -> Result<TurnToSend> {
        self.build_turn_at(owner, 0, max_turns)
    }

    fn build_turn_at(
        &self,
        owner: &SampledSession,
        start_index: usize,
        max_turns: Option<usize>,
    ) -> Result<TurnToSend> {
        if self.metadata.turns.is_empty() {
            bail!("conversation {} has no turns", owner.conversation_id);
        }
        let num_turns = max_turns
            .unwrap_or(self.metadata.turns.len())
            .min(self.metadata.turns.len())
            .max(1);
        // Turn 0 uses the strictly sequential advance so the existing first-turn path
        // is unchanged; a non-zero frontier jump-resumes via `seek_to`.
        self.materialize(owner, start_index, num_turns, start_index != 0)
    }

    fn next_metadata(&self, turn_index: usize) -> Result<TurnMetadata> {
        let next_index = turn_index + 1;
        self.metadata.turns.get(next_index).cloned().ok_or_else(|| {
            anyhow!(
                "no turn {next_index} in conversation {} (only {} turns exist)",
                self.metadata.conversation_id,
                self.metadata.turns.len()
            )
        })
    }

    fn build_next_turn(
        &self,
        owner: &SampledSession,
        current: &TurnToSend,
        response: TurnResponse,
    ) -> Result<TurnToSend> {
        if response.terminal == ReplayTerminalStatus::Completed {
            // Lower a completed reply once against the default prepared endpoint
            // so subsequent turns splice its stored wire. Dialects without message
            // arrays remain on the live rendering path.
            let resolved = match &self.endpoint {
                NativeSessionEndpoint::Prepared {
                    endpoint_resolver, ..
                } => endpoint_resolver.resolve(None)?,
            };
            let lowerer = ShapeLowerer::for_descriptor_id(resolved.endpoint.descriptor().id);
            let mut session = self.session.borrow_mut();
            if session.should_capture_response() {
                let tokens = match response.completion_tokens {
                    Some(tokens) => tokens,
                    None => u64::try_from(self.response_tokenizer.count(&response.text)?)
                        .map_err(|_| anyhow!("assistant token count exceeds u64"))?,
                };
                let mut reply = response.assistant_message.map_or_else(
                    || EndpointTurn {
                        role: Some("assistant".into()),
                        texts: vec![EndpointMedia::new(vec![response.text])],
                        ..EndpointTurn::default()
                    },
                    |message| EndpointTurn {
                        raw_messages: Some(vec![message]),
                        ..EndpointTurn::default()
                    },
                );
                if let Some(lowerer) = &lowerer {
                    reply.lowered = Some(lowerer.lower_turn(&reply)?);
                }
                // Count this reply's own image parts now, so a later turn's
                // image count stays known without inspecting its whole body.
                let images = reply_image_count(&reply, resolved.endpoint);
                session.capture_response(reply, tokens, images)?;
            }
        }
        self.materialize(owner, current.turn_index + 1, current.num_turns, false)
    }
}

impl NativeSessionBackend {
    fn materialize(
        &self,
        owner: &SampledSession,
        turn_index: usize,
        num_turns: usize,
        jump: bool,
    ) -> Result<TurnToSend> {
        let mut session = self.session.borrow_mut();
        if jump {
            session.seek_to(turn_index)?;
        } else {
            session.advance_to(turn_index)?;
        }
        let endpoint_name = session.endpoint_override()?.map(str::to_string);
        let (materialized, turn_endpoint, prepared_endpoint) = match &self.endpoint {
            NativeSessionEndpoint::Prepared {
                primary_model_name,
                endpoint_resolver,
            } => {
                let selected = endpoint_resolver.resolve(endpoint_name.as_deref())?;
                let materialized = self.materializer.materialize_prepared(
                    &session,
                    selected.endpoint,
                    primary_model_name,
                    CreditPhase::Profiling,
                    &Overrides::new(),
                )?;
                let reference = selected.reference.clone();
                (
                    materialized,
                    TurnEndpoint::Prepared(selected.reference),
                    (reference, selected.endpoint),
                )
            }
        };
        let timing = self
            .metadata
            .turns
            .get(turn_index)
            .ok_or_else(|| anyhow!("missing native turn metadata {turn_index}"))?;
        let static_count_key = (turn_index == 0
            && self.input_token_counter.caches_static_first_turns())
        .then(|| StaticInputCountKey {
            template_index: self.template_index,
            endpoint: match &turn_endpoint {
                TurnEndpoint::Prepared(reference) => {
                    StaticInputCountEndpoint::Prepared(reference.key)
                }
            },
        });
        let input_tokens =
            if timing.trace_hash_ids.is_some() || materialized.raw_token_ids.is_some() {
                u64::try_from(timing.input_length)
                    .map_err(|_| anyhow!("authored trace input count exceeds u64"))?
            } else if let Some(cached) = static_count_key
                .and_then(|key| self.static_input_count_cache.borrow().get(&key).copied())
            {
                cached
            } else {
                let (_, endpoint) = &prepared_endpoint;
                // Opaque bodies report absent as 0 at the u64 counter boundary for now.
                let counted = self.input_token_counter.count_materialized_input_tokens(
                    *endpoint,
                    &materialized.body.to_wire()?,
                    materialized.extracted.as_ref(),
                    materialized.input_tokens.unwrap_or(0),
                )?;
                if let Some(key) = static_count_key {
                    self.static_input_count_cache
                        .borrow_mut()
                        .insert(key, counted);
                }
                counted
            };
        let input_length = usize::try_from(input_tokens)
            .map_err(|_| anyhow!("materialized input token count exceeds usize"))?;
        let max_output_tokens = materialized
            .max_tokens
            .map(|tokens| tokens as usize)
            .unwrap_or(self.default_output_tokens);
        let request_correlation_id = materialized
            .accuracy
            .as_ref()
            .map(|accuracy| accuracy.correlation_id.as_str().to_string())
            .unwrap_or_else(|| owner.x_correlation_id.clone());
        let endpoint_path = materialized.endpoint_path;
        Ok(TurnToSend {
            uuid: Uuid::new_v4(),
            effective_model: Some(materialized.model),
            conversation_id: owner.conversation_id.clone(),
            x_correlation_id: owner.x_correlation_id.clone(),
            request_correlation_id,
            turn_index,
            num_turns,
            input_length,
            max_output_tokens,
            messages: Vec::new(),
            request_body: Some(materialized.body),
            request_headers: materialized.headers,
            request_parameters: materialized.parameters,
            endpoint_path,
            endpoint: turn_endpoint,
            streaming: materialized.streaming,
            audio_duration_seconds: materialized.audio_duration_seconds,
            image_count: materialized.image_count,
            timestamp_ms: timing.timestamp_ms,
            delay_ms: timing.delay_ms,
            trace_hash_ids: timing
                .trace_hash_ids
                .map(|handle| StoredTraceHashIds::new(handle, self.segments.clone())),
            raw_token_ids: materialized
                .raw_token_ids
                .map(|handle| StoredRawTokenIds::new(handle, self.segments.clone())),
            data_policy: TurnDataPolicy::ordinary(),
            cancel_after_ns: None,
            url_index: None,
            session: owner.clone(),
        })
    }
}

/// Native handle-only dataset source used by online scheduled workloads.
pub struct NativeDatasetConversationSource {
    dataset: Arc<NativeDataset>,
    /// This shard's fixed giver corpus: the sampleable conversations it owns.
    /// Under a multi-worker/cell partition that is the authored-index residue
    /// class; the sampler recycles only within this set. Unpartitioned sources
    /// hold every sampleable conversation.
    metadata: Vec<ConversationMetadata>,
    metadata_by_id: HashMap<String, usize>,
    sampler: Box<dyn Sampler>,
    endpoint: NativeSessionEndpoint,
    materializer: Arc<dyn RequestMaterializer>,
    response_tokenizer: Arc<dyn TextTokenizer>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    static_input_count_cache: StaticInputCountCache,
    default_output_tokens: usize,
}

impl NativeDatasetConversationSource {
    /// Honor loader sampling through a directly prepared open endpoint table.
    pub fn preferred_with_prepared_endpoint(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        rng_root: RngRoot,
        table: Rc<PreparedEndpointTable>,
        endpoint: PreparedEndpointReference,
    ) -> Result<Self> {
        let samplers = SamplerRegistry::with_builtin_strategies()?;
        let resolver: Rc<dyn PreparedTurnEndpointResolver> =
            Rc::new(PreparedEndpointTableResolver::single(table, endpoint)?);
        Self::preferred_with_prepared_resolver(
            dataset,
            model,
            default_output_tokens,
            rng_root,
            &samplers,
            resolver,
        )
    }

    /// Honor loader sampling with injected sampler and prepared endpoint
    /// resolution registries.
    #[allow(clippy::too_many_arguments)]
    pub fn preferred_with_prepared_resolver(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        rng_root: RngRoot,
        samplers: &SamplerRegistry,
        endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    ) -> Result<Self> {
        Self::preferred_with_prepared_resolver_for_partition(
            dataset,
            model,
            default_output_tokens,
            rng_root,
            samplers,
            endpoint_resolver,
            None,
        )
    }

    /// Inject a per-thread cell partition. `None` reads
    /// `AIPERF_CELL_ID`/`AIPERF_CELL_COUNT` from the process environment.
    #[allow(clippy::too_many_arguments)]
    pub fn preferred_with_prepared_resolver_for_partition(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        rng_root: RngRoot,
        samplers: &SamplerRegistry,
        endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
        cell_partition: Option<ModuloCellPartition>,
    ) -> Result<Self> {
        let mut dataset = dataset;
        let primary_model_name = model.into();
        lower_static_messages(
            &mut dataset,
            endpoint_resolver.as_ref(),
            &primary_model_name,
        )?;
        let dataset = Arc::new(dataset);
        let strategy = dataset.metadata().sampling_strategy.clone();
        Self::new_with_endpoint(
            dataset,
            |owned| Ok(samplers.create(&strategy, owned, rng_root)?),
            NativeSessionEndpoint::Prepared {
                primary_model_name,
                endpoint_resolver,
            },
            Arc::new(EndpointRequestMaterializer),
            Arc::new(TiktokenTokenizer::builtin()),
            default_output_tokens,
            cell_partition,
        )
    }

    /// Construct a sequential source through one directly prepared endpoint.
    pub fn sequential_with_prepared_endpoint(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        table: Rc<PreparedEndpointTable>,
        endpoint: PreparedEndpointReference,
    ) -> Result<Self> {
        let endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver> =
            Rc::new(PreparedEndpointTableResolver::single(table, endpoint)?);
        Self::sequential_with_prepared_resolver(
            dataset,
            model,
            default_output_tokens,
            endpoint_resolver,
        )
    }

    /// Construct a sequential source with a prepared resolver built once by
    /// the owning harness.
    pub fn sequential_with_prepared_resolver(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    ) -> Result<Self> {
        Self::sequential_with_prepared_resolver_for_partition(
            dataset,
            model,
            default_output_tokens,
            endpoint_resolver,
            None,
        )
    }

    /// Inject a per-thread cell partition. `None` reads
    /// `AIPERF_CELL_ID`/`AIPERF_CELL_COUNT` from the process environment.
    pub fn sequential_with_prepared_resolver_for_partition(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
        cell_partition: Option<ModuloCellPartition>,
    ) -> Result<Self> {
        let mut dataset = dataset;
        let primary_model_name = model.into();
        lower_static_messages(
            &mut dataset,
            endpoint_resolver.as_ref(),
            &primary_model_name,
        )?;
        let dataset = Arc::new(dataset);
        Self::new_with_endpoint(
            dataset,
            |owned| Ok(Box::new(SequentialSampler::from_metadata(owned)?)),
            NativeSessionEndpoint::Prepared {
                primary_model_name,
                endpoint_resolver,
            },
            Arc::new(EndpointRequestMaterializer),
            Arc::new(TiktokenTokenizer::builtin()),
            default_output_tokens,
            cell_partition,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_endpoint(
        dataset: Arc<NativeDataset>,
        make_sampler: impl FnOnce(
            &[crate::dataset::model::ConversationMetadata],
        ) -> Result<Box<dyn Sampler>>,
        endpoint: NativeSessionEndpoint,
        materializer: Arc<dyn RequestMaterializer>,
        response_tokenizer: Arc<dyn TextTokenizer>,
        default_output_tokens: usize,
        cell_partition: Option<ModuloCellPartition>,
    ) -> Result<Self> {
        if default_output_tokens == 0 {
            bail!("native dataset default output tokens must be positive");
        }
        // Resolve this shard's authored-index ownership. Explicit thread partitions
        // win; otherwise a multi-cell env partition applies. The owned residue is
        // the fixed giver corpus for both enumeration and sampling — recycle wraps
        // stay inside it rather than reaching foreign sessions via a position-based
        // PartitionedSampler over the full dataset.
        let ownership = match cell_partition {
            Some(partition) if partition.cell_count() > 1 => Some(partition),
            None => ModuloCellPartition::from_env().filter(|partition| partition.cell_count() > 1),
            Some(_) => None,
        };
        let owned_model: Vec<crate::dataset::model::ConversationMetadata> = dataset
            .sampleable_metadata()
            .enumerate()
            .filter(|(index, _)| {
                ownership
                    .map(|partition| partition.owns(*index as u64))
                    .unwrap_or(true)
            })
            .map(|(_, conversation)| conversation.clone())
            .collect();
        // One sampler over the owned corpus only — the single giver for this shard.
        // Do not wrap with PartitionedSampler: position filtering over a full-corpus
        // sequential draw is what requested foreign sessions after recycle.
        if ownership.is_some() && owned_model.is_empty() {
            bail!(
                "conversation partition owns no sampleable sessions; reduce workers/cells \
                 so every shard receives at least one conversation"
            );
        }
        let sampler = make_sampler(&owned_model)?;
        let metadata = owned_model
            .iter()
            .map(|conversation| {
                let authored = dataset
                    .get(&conversation.conversation_id)
                    .expect("metadata is projected from a validated conversation");
                ConversationMetadata {
                    conversation_id: conversation.conversation_id.as_str().to_string(),
                    turns: conversation
                        .turns
                        .iter()
                        .enumerate()
                        .map(|(index, turn)| TurnMetadata {
                            timestamp_ms: turn.timestamp_ms,
                            delay_ms: turn.delay_ms,
                            trace_hash_ids: turn.trace_hash_ids,
                            prompt_text: String::new(),
                            // Opaque bodies report absent as 0 at the usize observer boundary for now.
                            input_length: usize::try_from(turn.input_tokens.unwrap_or(0))
                                .unwrap_or(usize::MAX)
                                .max(1),
                            max_output_tokens: authored.turns[index]
                                .max_tokens
                                .map(|tokens| tokens as usize)
                                .unwrap_or(default_output_tokens),
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let metadata_by_id = metadata
            .iter()
            .enumerate()
            .map(|(index, metadata)| (metadata.conversation_id.clone(), index))
            .collect();
        Ok(Self {
            dataset,
            metadata,
            metadata_by_id,
            sampler,
            endpoint,
            materializer,
            response_tokenizer,
            input_token_counter: Arc::new(AuthoredInputTokenCounter),
            static_input_count_cache: Rc::new(RefCell::new(FxHashMap::default())),
            default_output_tokens,
        })
    }

    /// Replace authored input lengths with an injected materialized-body policy.
    pub fn with_input_token_counter(mut self, counter: Arc<dyn InputTokenCounter>) -> Self {
        self.input_token_counter = counter;
        self.static_input_count_cache.borrow_mut().clear();
        self
    }

    /// Replace the tokenizer used when a response omits authoritative usage.
    pub fn with_response_tokenizer(mut self, tokenizer: Arc<dyn TextTokenizer>) -> Self {
        self.response_tokenizer = tokenizer;
        self
    }

    /// Replace request-body construction with an injected materialization
    /// policy.
    ///
    /// Simulator adapters use this to consume segment-backed trace identities
    /// without formatting bytes that their dispatcher will never read. Online
    /// sources retain [`EndpointRequestMaterializer`] by default.
    pub fn with_request_materializer(mut self, materializer: Arc<dyn RequestMaterializer>) -> Self {
        self.materializer = materializer;
        self
    }

    fn session(
        &self,
        conversation_id: &str,
        correlation_id: Option<String>,
    ) -> Result<SampledSession> {
        let id = crate::dataset::SessionId::from(conversation_id);
        self.dataset.get(&id)?;
        let metadata_index = self
            .metadata_by_id
            .get(conversation_id)
            .copied()
            .ok_or_else(|| {
                anyhow!("native dataset session {conversation_id:?} is not sampleable")
            })?;
        let metadata = self.metadata[metadata_index].clone();
        let x_correlation_id = correlation_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let backend = NativeSessionBackend {
            session: RefCell::new(NativeConversationSession::new(self.dataset.clone(), id)?),
            template_index: metadata_index,
            metadata,
            endpoint: self.endpoint.clone(),
            materializer: self.materializer.clone(),
            response_tokenizer: self.response_tokenizer.clone(),
            input_token_counter: self.input_token_counter.clone(),
            static_input_count_cache: self.static_input_count_cache.clone(),
            segments: self.dataset.segments().clone(),
            default_output_tokens: self.default_output_tokens,
        };
        Ok(SampledSession {
            conversation_id: conversation_id.to_string(),
            x_correlation_id,
            backend: Rc::new(backend),
        })
    }

    /// Generate every sampleable conversation's canonical per-turn request bodies up
    /// front, WITHOUT dispatching, for a pure `inputs.json` export.
    ///
    /// Each turn is materialized through the same session machinery dispatch drives
    /// ([`SampledSession::build_first_turn`] / `build_next_turn` over the resident
    /// [`Dataset`](NativeDataset)), so the retained `request_body` equals the
    /// `canonical_body()` the capture path records for the same turn — the two
    /// `inputs.json` files are byte-identical (same conversation-id sort via the
    /// returned [`BTreeMap`] ordering, same per-turn dedup: one payload per authored
    /// turn). The session id is the authored conversation id, matching the capture
    /// path's `owner.conversation_id` key.
    ///
    /// Returns `None` (fallback: the caller keeps the during-run capture path) when a
    /// MULTI-turn conversation captures live responses — context modes
    /// [`DeltasWithoutResponses`](crate::dataset::ConversationContextMode::DeltasWithoutResponses)
    /// or
    /// [`MessageArrayWithoutResponses`](crate::dataset::ConversationContextMode::MessageArrayWithoutResponses)
    /// — because a later turn's body then splices the live model reply, which an
    /// up-front pass cannot reproduce. Single-turn conversations never capture a reply
    /// into a subsequent turn, so they are always reproducible regardless of mode.
    pub fn build_input_payloads(&self) -> Result<Option<Vec<UpFrontInputSession>>> {
        // A dummy terminal for `build_next_turn`; the only conversations that reach it
        // here are non-response-capture (with-responses or single-turn), where
        // `build_next_turn` ignores the live reply and splices the authored turn.
        let no_capture_reply = || TurnResponse {
            text: String::new(),
            assistant_message: None,
            completion_tokens: None,
            terminal: ReplayTerminalStatus::Completed,
        };
        // Sort by conversation id so the emitted session order matches the capture
        // path's `BTreeMap<conversation_id, …>` iteration exactly.
        let mut sessions: BTreeMap<String, Vec<Bytes>> = BTreeMap::new();
        for metadata in &self.metadata {
            let conversation_id = metadata.conversation_id.as_str();
            let id = crate::dataset::SessionId::from(conversation_id);
            let conversation = self.dataset.get(&id)?;
            let captures_response = matches!(
                self.dataset.context_mode(conversation),
                crate::dataset::ConversationContextMode::DeltasWithoutResponses
                    | crate::dataset::ConversationContextMode::MessageArrayWithoutResponses
            );
            let session = self.session(conversation_id, None)?;
            let num_turns = session.available_turns();
            if num_turns > 1 && captures_response {
                // A live-reply-dependent multi-turn conversation cannot be reproduced
                // up front: the whole run falls back to the during-run capture path.
                return Ok(None);
            }
            let mut payloads = Vec::with_capacity(num_turns);
            let mut current = session.build_first_turn(None)?;
            payloads.push(
                current
                    .request_body
                    .as_ref()
                    .ok_or_else(|| {
                        anyhow!(
                            "materialized turn for conversation {conversation_id:?} produced no body"
                        )
                    })?
                    .to_wire()?,
            );
            for _ in 1..num_turns {
                let next = session.build_next_turn(&current, no_capture_reply())?;
                payloads.push(
                    next.request_body
                        .as_ref()
                        .ok_or_else(|| {
                            anyhow!(
                                "materialized turn for conversation {conversation_id:?} produced no body"
                            )
                        })?
                        .to_wire()?,
                );
                current = next;
            }
            sessions.insert(conversation_id.to_string(), payloads);
        }
        Ok(Some(
            sessions
                .into_iter()
                .map(|(session_id, payloads)| UpFrontInputSession {
                    session_id,
                    payloads,
                })
                .collect(),
        ))
    }
}

impl ConversationSource for NativeDatasetConversationSource {
    fn conversations(&self) -> &[ConversationMetadata] {
        &self.metadata
    }

    fn materialize_input_payloads(&self) -> Result<Option<Vec<UpFrontInputSession>>> {
        self.build_input_payloads()
    }

    fn next(&mut self, x_correlation_id: Option<String>) -> Result<SampledSession> {
        let id = self.sampler.next();
        self.session(id.as_str(), x_correlation_id)
    }

    fn session_for(
        &self,
        conversation_id: &str,
        x_correlation_id: String,
    ) -> Result<SampledSession> {
        self.session(conversation_id, Some(x_correlation_id))
    }
}

/// Lock-free-by-serialization counters for a single issuer loop.
#[derive(Default)]
pub struct CreditCounter {
    requests_sent: u64,
    root_requests_sent: u64,
    sent_sessions: u64,
    total_session_turns: u64,
}

impl CreditCounter {
    /// Increment sent counters and return `(credit_id, is_final_credit)`.
    pub fn increment_sent(&mut self, turn: &TurnToSend, stop: &StopConfig) -> (u64, bool) {
        let credit_id = self.requests_sent;
        let new_sent = self.requests_sent + 1;
        let new_root_sent = self.root_requests_sent + 1;
        let mut new_sessions = self.sent_sessions;
        let mut new_total_turns = self.total_session_turns;

        if turn.turn_index == 0 {
            new_sessions += 1;
            new_total_turns += turn.num_turns as u64;
        }

        let is_final_credit = stop
            .total_expected_requests
            .is_some_and(|total| new_sent >= total)
            || stop.expected_num_sessions.is_some_and(|expected| {
                new_sessions >= expected && new_root_sent >= new_total_turns
            });

        self.requests_sent = new_sent;
        self.root_requests_sent = new_root_sent;
        self.sent_sessions = new_sessions;
        self.total_session_turns = new_total_turns;

        (credit_id, is_final_credit)
    }

    /// Snapshot counters as a StopChecker [`RunState`].
    pub fn run_state(&self, started_at_ns: i64, sending_complete: bool) -> RunState {
        RunState {
            requests_sent: self.requests_sent,
            root_requests_sent: self.root_requests_sent,
            sent_sessions: self.sent_sessions,
            total_session_turns: self.total_session_turns,
            cancelled: false,
            sending_complete,
            started_at_ns,
        }
    }

    /// Number of issued requests.
    pub fn requests_sent(&self) -> u64 {
        self.requests_sent
    }

    /// Number of started runtime sessions.
    pub fn sent_sessions(&self) -> u64 {
        self.sent_sessions
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::dataset::{ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry};
    use crate::endpoints::ChatEndpoint;
    use crate::rng::RngRoot;
    use serde_json::json;

    use super::*;

    /// The wire bytes a built turn dispatches.
    fn dispatched_body(turn: &TurnToSend) -> Bytes {
        turn.request_body
            .as_ref()
            .expect("a built turn carries a request body")
            .to_wire()
            .expect("the request body materializes")
    }

    struct FixedTemplateTokenizer;

    #[derive(Default)]
    struct CountingStaticInputTokenCounter {
        calls: AtomicUsize,
    }

    impl InputTokenCounter for CountingStaticInputTokenCounter {
        fn count_input_tokens(
            &self,
            _endpoint: &dyn Endpoint,
            _body: &[u8],
            authored_input_tokens: u64,
        ) -> Result<u64> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(authored_input_tokens.saturating_sub(1))
        }

        fn count_prepared_input_tokens(
            &self,
            _endpoint: &dyn PreparedEndpoint,
            _body: &[u8],
            authored_input_tokens: u64,
        ) -> Result<u64> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            Ok(authored_input_tokens.saturating_sub(1))
        }

        fn caches_static_first_turns(&self) -> bool {
            true
        }
    }

    impl TextTokenizer for FixedTemplateTokenizer {
        fn encode(&self, text: &str) -> crate::dataset::Result<Vec<u32>> {
            Ok((0..text.split_whitespace().count() as u32).collect())
        }

        fn decode(&self, _token_ids: &[u32]) -> crate::dataset::Result<String> {
            Ok(String::new())
        }

        fn bos_token_id(&self) -> Option<u32> {
            None
        }

        fn eos_token_id(&self) -> Option<u32> {
            None
        }

        fn name(&self) -> &str {
            "fixed-template"
        }

        fn apply_chat_template(
            &self,
            _messages: &[Value],
            _add_generation_prompt: bool,
        ) -> crate::dataset::Result<Option<Vec<u32>>> {
            Ok(Some(vec![0; 5]))
        }
    }

    fn prepared_source(
        dataset: NativeDataset,
        model: &str,
        default_output_tokens: usize,
        endpoint_name: &str,
    ) -> NativeDatasetConversationSource {
        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let endpoint_id = EndpointId::new(endpoint_name).unwrap();
        let endpoint = registry
            .prepare(
                &endpoint_id,
                crate::endpoints::RawEndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        NativeDatasetConversationSource::sequential_with_prepared_endpoint(
            dataset,
            model,
            default_output_tokens,
            Rc::new(table),
            PreparedEndpointReference { key, endpoint_id },
        )
        .unwrap()
    }

    fn prepared_chat_source(
        dataset: NativeDataset,
        model: &str,
        default_output_tokens: usize,
    ) -> NativeDatasetConversationSource {
        prepared_source(dataset, model, default_output_tokens, "chat")
    }

    async fn inline_multi_turn_dataset(
        turns: usize,
        output_tokens: usize,
        model: &str,
    ) -> NativeDataset {
        let mut turn_objs = Vec::new();
        for index in 0..turns.max(1) {
            let mut turn = json!({"text": format!("turn {index}"), "output_length": output_tokens});
            if index > 0 {
                turn["delay"] = json!(0);
            }
            turn_objs.push(turn);
        }
        LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("multi_turn"),
                &LoadConfig::new(DatasetSource::Inline(
                    json!([{"session_id":"synthetic","turns": turn_objs}]),
                )),
                &ComposeConfig::new(model, RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap()
    }

    /// A static single-session dataset via the `single_turn` format: multiple rows
    /// sharing a `session_id` compose into one MessageArrayWithResponses conversation
    /// (authored responses, no live-reply capture), so its turns can be materialized
    /// up front byte-identically to dispatch.
    async fn inline_static_multi_turn_dataset(model: &str) -> NativeDataset {
        LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("single_turn"),
                &LoadConfig::new(DatasetSource::Inline(json!([
                    {"session_id":"s","text":"q0 alpha","output_length":4},
                    {"session_id":"s","text":"q1 beta gamma","output_length":4},
                ]))),
                &ComposeConfig::new(model, RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap()
    }

    async fn inline_single_turn_dataset(model: &str) -> NativeDataset {
        LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("single_turn"),
                &LoadConfig::new(DatasetSource::Inline(json!([
                    {"session_id":"solo","text":"hello world","output_length":4},
                ]))),
                &ComposeConfig::new(model, RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap()
    }

    /// Up-front `inputs.json` generation for a single-turn dataset reproduces the
    /// exact dispatch body: the generated payload equals the `request_body` the same
    /// session's `build_first_turn` (the dispatch turn producer) yields — the bytes the
    /// capture path would record as `canonical_body`.
    #[tokio::test]
    async fn up_front_input_payloads_single_turn_match_dispatch_body() {
        let dataset = inline_single_turn_dataset("test-model").await;
        let source = prepared_chat_source(dataset, "test-model", 4);

        let sessions = source
            .build_input_payloads()
            .unwrap()
            .expect("single-turn dataset is reproducible up front");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "solo");
        assert_eq!(sessions[0].payloads.len(), 1);

        // The dispatch turn producer for the same conversation yields the identical
        // body bytes the up-front pass emits.
        let dispatch_turn = source
            .session_for("solo", "corr-0".into())
            .unwrap()
            .build_first_turn(None)
            .unwrap();
        assert_eq!(sessions[0].payloads[0], dispatched_body(&dispatch_turn));
    }

    /// Up-front generation of a static (authored-response) multi-turn conversation
    /// yields every turn in order; each payload is a distinct, growing message array.
    #[tokio::test]
    async fn up_front_input_payloads_static_multi_turn_are_ordered_and_complete() {
        let dataset = inline_static_multi_turn_dataset("test-model").await;
        let source = prepared_chat_source(dataset, "test-model", 4);

        let sessions = source
            .build_input_payloads()
            .unwrap()
            .expect("authored-response multi-turn is reproducible up front");
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "s");
        assert_eq!(sessions[0].payloads.len(), 2);

        // The two turns are materialized in order into distinct request bodies: turn 0
        // carries q0's content, turn 1 carries q1's — proving the walk drives every
        // authored turn sequentially.
        let turn0: Value = serde_json::from_slice(sessions[0].payloads[0].as_ref()).unwrap();
        let turn1: Value = serde_json::from_slice(sessions[0].payloads[1].as_ref()).unwrap();
        assert_ne!(
            sessions[0].payloads[0].as_ref(),
            sessions[0].payloads[1].as_ref(),
            "each authored turn materializes to a distinct body"
        );
        let turn0_text = turn0["messages"].to_string();
        let turn1_text = turn1["messages"].to_string();
        assert!(
            turn0_text.contains("q0"),
            "turn 0 body carries q0: {turn0_text}"
        );
        assert!(
            turn1_text.contains("q1"),
            "turn 1 body carries q1: {turn1_text}"
        );
        // The first turn body matches the dispatch producer's first turn.
        let dispatch_turn0 = source
            .session_for("s", "corr-0".into())
            .unwrap()
            .build_first_turn(None)
            .unwrap();
        assert_eq!(sessions[0].payloads[0], dispatched_body(&dispatch_turn0));
    }

    /// Jump-resume: `build_turn_at(k)` on a fresh session yields the turn at `k`
    /// with the recorded 0..=k context reconstructed, byte-identical to sequentially
    /// advancing the same conversation from 0 to k. This is the resume primitive the
    /// accelerated-cache-warmup workload needs to start a lane at a non-zero drained
    /// frontier without replaying every prior turn's dispatch.
    #[tokio::test]
    async fn build_turn_at_jump_resume_equals_sequential_advance() {
        let dataset = inline_static_multi_turn_dataset("test-model").await;
        let source = prepared_chat_source(dataset, "test-model", 4);
        let no_capture_reply = || TurnResponse {
            text: String::new(),
            assistant_message: None,
            completion_tokens: None,
            terminal: ReplayTerminalStatus::Completed,
        };

        // Oracle: sequentially advance a fresh session 0 -> 1.
        let seq_session = source.session_for("s", "corr-0".into()).unwrap();
        let seq_turn0 = seq_session.build_first_turn(None).unwrap();
        let seq_turn1 = seq_session
            .build_next_turn(&seq_turn0, no_capture_reply())
            .unwrap();

        // Jump: resume a fresh session directly at turn 1.
        let jump_session = source.session_for("s", "corr-0".into()).unwrap();
        let jump_turn1 = jump_session.build_turn_at(1, None).unwrap();

        assert_eq!(jump_turn1.turn_index, 1);
        assert_eq!(jump_turn1.num_turns, seq_turn1.num_turns);
        assert_eq!(
            dispatched_body(&jump_turn1),
            dispatched_body(&seq_turn1),
            "jump-resume to turn 1 is byte-identical to sequential advance to turn 1"
        );

        // The reconstructed context reflects turns 0..=1: the recorded message array
        // carries the current turn's content (q1) and is distinct from turn 0 (q0).
        let jump_body: Value = serde_json::from_slice(&dispatched_body(&jump_turn1)).unwrap();
        let messages = jump_body["messages"].to_string();
        assert!(
            messages.contains("q1"),
            "turn-1 context carries q1: {messages}"
        );
        assert_ne!(
            dispatched_body(&jump_turn1),
            dispatched_body(&seq_turn0),
            "turn-1 body differs from turn-0 body"
        );

        // build_turn_at(0) is identical to build_first_turn (sequential path unchanged).
        let jump_turn0 = source
            .session_for("s", "corr-0".into())
            .unwrap()
            .build_turn_at(0, None)
            .unwrap();
        assert_eq!(jump_turn0.turn_index, 0);
        assert_eq!(dispatched_body(&jump_turn0), dispatched_body(&seq_turn0),);
    }

    /// A live-reply-dependent multi-turn dataset (the default DeltasWithoutResponses
    /// `multi_turn` format) cannot be reproduced up front, so the generator declines
    /// (returns `None`) and the caller keeps the during-run capture path.
    #[tokio::test]
    async fn up_front_input_payloads_declines_live_reply_multi_turn() {
        let dataset = inline_multi_turn_dataset(3, 4, "test-model").await;
        let source = prepared_chat_source(dataset, "test-model", 4);
        assert!(
            source.build_input_payloads().unwrap().is_none(),
            "capture-mode multi-turn must fall back to the during-run capture path"
        );
    }

    #[test]
    fn endpoint_counter_handles_template_and_bare_paths() {
        let body = serde_json::to_vec(&json!({
            "messages":[{"role":"user","content":"hello world"}],
            "tools":[{"type":"function","function":{"name":"weather"}}]
        }))
        .unwrap();
        let tokenizer: Arc<dyn TextTokenizer> = Arc::new(FixedTemplateTokenizer);
        let templated = EndpointInputTokenCounter::new(tokenizer.clone(), true);
        let bare = EndpointInputTokenCounter::new(tokenizer, false);

        assert_eq!(
            templated
                .count_input_tokens(&ChatEndpoint, &body, 99)
                .unwrap(),
            6
        );
        assert_eq!(
            bare.count_input_tokens(&ChatEndpoint, &body, 99).unwrap(),
            3
        );
        assert_eq!(
            bare.count_input_tokens(&ChatEndpoint, b"not-json", 99)
                .unwrap(),
            99
        );
    }

    /// The differential every [`PreparedEndpoint::extracted`] override owes,
    /// swept across every built-in endpoint.
    ///
    /// This arms itself: today no endpoint reports structure, so the inner
    /// assertion is skipped and only the catalog-breadth floor below is
    /// asserted — nothing here pins that endpoints report nothing. The first
    /// override makes the assertion live against the body that endpoint actually
    /// dispatches — `format_payload` through `materialize_standalone`, not a
    /// hand-written payload. A divergence is a silent ISL shift, not a failure
    /// the run surfaces.
    #[test]
    fn extracted_structure_matches_the_dispatched_body_for_every_endpoint() {
        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let turns = [EndpointTurn {
            role: Some("user".into()),
            texts: vec![EndpointMedia::new(vec!["hello world".to_string()])],
            ..EndpointTurn::default()
        }];
        let request = crate::endpoints::PreparedRequest::new(
            "test-model",
            &turns,
            None,
            None,
            CreditPhase::Profiling,
            None,
            None,
            None,
        );

        let mut swept = 0usize;
        for id in registry.canonical_ids() {
            let Ok(prepared) = registry.prepare(id, crate::endpoints::RawEndpointConfig::default())
            else {
                continue;
            };
            // A dialect that cannot express a bare text turn (media-only,
            // token-native, raw) has nothing to compare here.
            let Ok(plan) = prepared.format_payload(&request) else {
                continue;
            };
            swept += 1;
            let Some(direct) = prepared.extracted(&request, &plan) else {
                continue;
            };
            let dispatched = plan.materialize_standalone().unwrap();
            let parsed: Value = serde_json::from_slice(&dispatched).unwrap();
            assert_eq!(
                direct,
                prepared.extract_payload_inputs(&parsed),
                "endpoint {id} reports structure that diverges from its dispatched body"
            );
        }
        // A floor, not the exact count: the sweep must keep covering the bulk of
        // the catalog rather than silently narrowing to one dialect.
        assert!(
            swept >= 10,
            "the sweep covered only {swept} endpoints; it must exercise the catalog"
        );
    }

    /// The counter honors a reported structure and falls back to parsing without
    /// one. This pins the plumbing between `MaterializedRequest.extracted` and
    /// the counter; the structure/body agreement itself is the sweep above.
    #[test]
    fn materialized_counter_prefers_the_reported_structure() {
        let payload = json!({
            "messages":[{"role":"user","content":"hello world"}],
            "tools":[{"type":"function","function":{"name":"weather"}}]
        });
        let body = serde_json::to_vec(&payload).unwrap();
        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let prepared = registry
            .prepare(
                &EndpointId::new("chat").unwrap(),
                crate::endpoints::RawEndpointConfig::default(),
            )
            .unwrap();
        let counter = EndpointInputTokenCounter::new(Arc::new(FixedTemplateTokenizer), false);

        let parsed = counter
            .count_prepared_input_tokens(prepared.as_ref(), &body, 99)
            .unwrap();
        assert_eq!(
            counter
                .count_materialized_input_tokens(prepared.as_ref(), &body, None, 99)
                .unwrap(),
            parsed
        );
        // A structure the body does not describe must reach the count, proving
        // the reported value is used rather than silently re-derived.
        let reported = crate::endpoints::ExtractedPayload {
            texts: vec!["one two three four".into()],
            ..crate::endpoints::ExtractedPayload::default()
        };
        assert_eq!(
            counter
                .count_materialized_input_tokens(prepared.as_ref(), &body, Some(&reported), 99)
                .unwrap(),
            4
        );
    }

    #[tokio::test]
    async fn credit_counter_counts_root_turns_for_session_bounds() {
        let dataset = inline_multi_turn_dataset(2, 2, "model").await;
        let mut source = prepared_chat_source(dataset, "model", 2);
        let mut counter = CreditCounter::default();
        let stop = StopConfig {
            total_expected_requests: None,
            expected_num_sessions: Some(2),
            expected_duration_ns: None,
        };

        let first = source.next(None).unwrap().build_first_turn(None).unwrap();
        let (id0, final0) = counter.increment_sent(&first, &stop);
        assert_eq!(id0, 0);
        assert!(!final0);
        let next = source
            .next_turn(
                &IssuedCredit::from_turn(id0, 0, &first),
                TurnResponse {
                    text: String::new(),
                    assistant_message: None,
                    completion_tokens: None,
                    terminal: ReplayTerminalStatus::Completed,
                },
            )
            .unwrap()
            .unwrap();
        let (_, final1) = counter.increment_sent(&next, &stop);
        assert!(!final1);
        let second = source.next(None).unwrap().build_first_turn(None).unwrap();
        let (_, final2) = counter.increment_sent(&second, &stop);
        assert!(!final2, "second session still has a continuation");
        let second_next = source
            .next_turn(
                &IssuedCredit::from_turn(2, 0, &second),
                TurnResponse {
                    text: String::new(),
                    assistant_message: None,
                    completion_tokens: None,
                    terminal: ReplayTerminalStatus::Completed,
                },
            )
            .unwrap()
            .unwrap();
        let (_, final3) = counter.increment_sent(&second_next, &stop);
        assert!(final3);
        assert_eq!(counter.run_state(10, false).requests_sent, 4);
        assert_eq!(counter.run_state(10, false).sent_sessions, 2);
        assert_eq!(counter.run_state(10, false).total_session_turns, 4);
    }

    #[tokio::test]
    async fn native_dataset_session_materializes_and_splices_live_replies() {
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let source = DatasetSource::Inline(json!([{
            "session_id":"native",
            "turns":[
                {"text":"first question","timestamp":0,"output_length":2},
                {"text":"second question","delay":5,"output_length":3}
            ]
        }]));
        let compose = ComposeConfig::new("model", RngRoot::new(Some(4)));
        let dataset = registry
            .build_dataset(
                Some("multi_turn"),
                &LoadConfig::new(source),
                &compose,
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let mut source = prepared_chat_source(dataset, "model", 8);
        let first = source
            .next(Some("runtime-session".into()))
            .unwrap()
            .build_first_turn(None)
            .unwrap();
        let first_body: Value = serde_json::from_slice(&dispatched_body(&first)).unwrap();
        assert_eq!(first_body["messages"][0]["content"], "first question");
        let next = source
            .next_turn(
                &IssuedCredit::from_turn(0, 0, &first),
                TurnResponse {
                    text: "live answer".into(),
                    assistant_message: None,
                    completion_tokens: Some(2),
                    terminal: ReplayTerminalStatus::Completed,
                },
            )
            .unwrap()
            .unwrap();
        let next_body: Value = serde_json::from_slice(&dispatched_body(&next)).unwrap();
        assert_eq!(
            next_body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(|message| message["role"].as_str().unwrap())
                .collect::<Vec<_>>(),
            vec!["user", "assistant", "user"]
        );
        assert_eq!(next_body["messages"][1]["content"], "live answer");
        assert_eq!(next.input_length, first.input_length + 2 + 2);
        assert_eq!(next.delay_ms, Some(5.0));
    }

    #[tokio::test]
    async fn captured_reply_is_lowered_once_and_spliced_identically_each_dispatch() {
        // The three-turn fixture exposes the first reply in two later request
        // bodies, detecting any byte drift between dispatches.
        let registry = LoaderRegistry::with_builtin_formats().unwrap();
        let source = DatasetSource::Inline(json!([{
            "session_id": "native",
            "turns": [
                {"text": "q0", "timestamp": 0, "output_length": 2},
                {"text": "q1", "delay": 0, "output_length": 2},
                {"text": "q2", "delay": 0, "output_length": 2}
            ]
        }]));
        let dataset = registry
            .build_dataset(
                Some("multi_turn"),
                &LoadConfig::new(source),
                &ComposeConfig::new("model", RngRoot::new(Some(7))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let mut source = prepared_chat_source(dataset, "model", 8);
        let first = source
            .next(Some("runtime-session".into()))
            .unwrap()
            .build_first_turn(None)
            .unwrap();
        let turn1 = source
            .next_turn(
                &IssuedCredit::from_turn(0, 0, &first),
                TurnResponse {
                    text: "reply-0".into(),
                    assistant_message: None,
                    completion_tokens: Some(2),
                    terminal: ReplayTerminalStatus::Completed,
                },
            )
            .unwrap()
            .unwrap();
        let turn2 = source
            .next_turn(
                &IssuedCredit::from_turn(1, 1, &turn1),
                TurnResponse {
                    text: "reply-1".into(),
                    assistant_message: None,
                    completion_tokens: Some(2),
                    terminal: ReplayTerminalStatus::Completed,
                },
            )
            .unwrap()
            .unwrap();

        let body1: Value = serde_json::from_slice(&dispatched_body(&turn1)).unwrap();
        let body2: Value = serde_json::from_slice(&dispatched_body(&turn2)).unwrap();
        // reply-0 is message[1] in both dispatch bodies and must be identical.
        assert_eq!(body1["messages"][1], body2["messages"][1]);
        assert_eq!(body1["messages"][1]["content"], "reply-0");
        assert_eq!(body2["messages"][3]["content"], "reply-1");

        // A single-text assistant turn has this contractually serialized shape.
        assert_eq!(
            body1["messages"][1],
            json!({"role": "assistant", "content": "reply-0"})
        );
    }

    #[tokio::test]
    async fn native_messages_session_replays_lossless_assistant_blocks() {
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("multi_turn"),
                &LoadConfig::new(DatasetSource::Inline(json!([{
                    "session_id":"messages",
                    "turns":[
                        {"text":"first","output_length":4},
                        {"text":"second","output_length":4}
                    ]
                }]))),
                &ComposeConfig::new("claude", RngRoot::new(Some(9))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let mut source = prepared_source(dataset, "claude", 4, "messages");
        let first = source.next(None).unwrap().build_first_turn(None).unwrap();
        let assistant = json!({
            "role":"assistant",
            "content":[
                {"type":"thinking","thinking":"why","signature":"sig"},
                {"type":"text","text":"answer"},
                {"type":"tool_use","id":"tool-1","name":"lookup","input":{"q":"x"}}
            ]
        });
        let next = source
            .next_turn(
                &IssuedCredit::from_turn(0, 0, &first),
                TurnResponse {
                    text: "whyanswerlookup{\"q\":\"x\"}".into(),
                    assistant_message: Some(assistant.clone()),
                    completion_tokens: Some(9),
                    terminal: ReplayTerminalStatus::Completed,
                },
            )
            .unwrap()
            .unwrap();
        let body: Value = serde_json::from_slice(&dispatched_body(&next)).unwrap();
        assert_eq!(body["messages"][1], assistant);
        assert_eq!(
            body["messages"][2],
            json!({"role":"user","content":"second"})
        );
    }

    #[tokio::test]
    async fn native_raw_payload_reaches_turn_dispatch_byte_identically() {
        let authored = Bytes::from_static(
            b"{ \"messages\": [{\"role\":\"user\",\"content\":\"exact\"}], \"stream\": true }",
        );
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("raw_payload"),
                &LoadConfig::new(DatasetSource::Bytes(authored.clone())),
                &ComposeConfig::new("model", RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let mut source = prepared_chat_source(dataset, "model", 4);
        let turn = source.next(None).unwrap().build_first_turn(None).unwrap();
        assert_eq!(dispatched_body(&turn), authored);
        assert!(turn.streaming);
    }

    #[tokio::test]
    async fn native_source_materializes_directly_through_prepared_endpoint() {
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("single_turn"),
                &LoadConfig::new(DatasetSource::Inline(json!([{"text":"hello"}]))),
                &ComposeConfig::new("prepared-model", RngRoot::new(Some(5))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let endpoint_id = EndpointId::new("chat").unwrap();
        let endpoint = registry
            .prepare(
                &endpoint_id,
                crate::endpoints::RawEndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        let counter = Arc::new(CountingStaticInputTokenCounter::default());
        let mut source = NativeDatasetConversationSource::sequential_with_prepared_endpoint(
            dataset,
            "prepared-model",
            4,
            Rc::new(table),
            PreparedEndpointReference {
                key,
                endpoint_id: endpoint_id.clone(),
            },
        )
        .unwrap()
        .with_input_token_counter(counter.clone());

        let turn = source.next(None).unwrap().build_first_turn(None).unwrap();
        let repeated = source.next(None).unwrap().build_first_turn(None).unwrap();

        let body: Value = serde_json::from_slice(&dispatched_body(&turn)).unwrap();
        let TurnEndpoint::Prepared(reference) = turn.endpoint;
        assert_eq!(reference.key, key);
        assert_eq!(reference.endpoint_id, endpoint_id);
        assert_eq!(body["model"], "prepared-model");
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["stream"], true);
        assert_eq!(turn.input_length, repeated.input_length);
        assert_eq!(counter.calls.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn native_source_resolves_the_loader_sampling_strategy() {
        let built = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("single_turn"),
                &LoadConfig::new(DatasetSource::Inline(json!([{"text":"hello"}]))),
                &ComposeConfig::new("model", RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        let dataset = NativeDataset::new(
            built.conversations().to_vec(),
            built.segments().clone(),
            "not_registered",
            built.metadata().default_context_mode,
        )
        .unwrap();
        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let endpoint_id = EndpointId::new("chat").unwrap();
        let endpoint = registry
            .prepare(
                &endpoint_id,
                crate::endpoints::RawEndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        let error = NativeDatasetConversationSource::preferred_with_prepared_endpoint(
            dataset,
            "model",
            4,
            RngRoot::new(Some(1)),
            Rc::new(table),
            PreparedEndpointReference { key, endpoint_id },
        )
        .err()
        .unwrap();
        assert!(error.to_string().contains("unknown sampler strategy"));
    }

    /// A partitioned shard's sampler is a fixed giver over its authored-index
    /// residue only. Recycle wraps inside that set (never into a foreign
    /// session), which is what `--request-count` recycling with `workers > 1`
    /// requires when `conversation_count % grid != 0`.
    #[tokio::test]
    async fn partitioned_sequential_recycle_stays_inside_owned_raw_payloads() {
        let dataset = raw_payload_jsonl_dataset(3).await;
        let mut source = partitioned_sequential_source(dataset, 0, 2).await;
        assert_eq!(source.conversations().len(), 2);
        let owned = owned_ids(&source);

        let mut seen = Vec::new();
        for _ in 0..4 {
            let session = source
                .next(None)
                .expect("owned recycle must remain sampleable");
            assert!(
                owned.contains(&session.conversation_id),
                "giver must not hand out a foreign session {}",
                session.conversation_id
            );
            seen.push(session.conversation_id.clone());
            session.build_first_turn(None).unwrap();
        }
        // Sequential wrap inside the owned pair: a, b, a, b.
        assert_eq!(seen[0], seen[2]);
        assert_eq!(seen[1], seen[3]);
        assert_ne!(seen[0], seen[1]);
    }

    /// Both shards of a 2-wide grid own disjoint residues whose union is the
    /// full raw_payload corpus, and each recycles only inside its own giver.
    #[tokio::test]
    async fn partitioned_raw_payload_givers_are_disjoint_and_cover_the_corpus() {
        let dataset = raw_payload_jsonl_dataset(3).await;
        let all_ids: Vec<String> = dataset
            .conversations()
            .iter()
            .map(|conversation| conversation.session_id.as_str().to_string())
            .collect();

        let mut cell0 = partitioned_sequential_source(dataset.clone(), 0, 2).await;
        let mut cell1 = partitioned_sequential_source(dataset, 1, 2).await;
        let owned0 = owned_ids(&cell0);
        let owned1 = owned_ids(&cell1);

        assert_eq!(owned0.len(), 2);
        assert_eq!(owned1.len(), 1);
        for id in &owned0 {
            assert!(
                !owned1.contains(id),
                "shards must not share a giver session"
            );
        }
        let mut union = owned0
            .iter()
            .chain(owned1.iter())
            .cloned()
            .collect::<Vec<_>>();
        union.sort();
        let mut expected = all_ids.clone();
        expected.sort();
        assert_eq!(union, expected);

        for _ in 0..5 {
            let session = cell0.next(None).unwrap();
            assert!(owned0.contains(&session.conversation_id));
            session.build_first_turn(None).unwrap();
            let session = cell1.next(None).unwrap();
            assert!(owned1.contains(&session.conversation_id));
            session.build_first_turn(None).unwrap();
        }
    }

    /// Preferred (shuffle) sampling under a partition still recycles only the
    /// shard's owned corpus — the fixed giver, not a full-corpus draw filter.
    #[tokio::test]
    async fn partitioned_preferred_shuffle_recycles_only_owned_raw_payloads() {
        let jsonl = (0..5)
            .map(|n| {
                format!(r#"{{"messages":[{{"role":"user","content":"p{n}"}}],"stream":false}}"#)
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        let mut load = LoadConfig::new(DatasetSource::Bytes(Bytes::from(jsonl)));
        load.sampling_strategy = Some("shuffle".into());
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("raw_payload"),
                &load,
                &ComposeConfig::new("model", RngRoot::new(Some(7))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();
        assert_eq!(dataset.metadata().sampling_strategy, "shuffle");

        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let endpoint_id = EndpointId::new("chat").unwrap();
        let endpoint = registry
            .prepare(
                &endpoint_id,
                crate::endpoints::RawEndpointConfig {
                    streaming: false,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        let resolver: Rc<dyn PreparedTurnEndpointResolver> = Rc::new(
            PreparedEndpointTableResolver::single(
                Rc::new(table),
                PreparedEndpointReference { key, endpoint_id },
            )
            .unwrap(),
        );
        let samplers = SamplerRegistry::with_builtin_strategies().unwrap();
        let mut source =
            NativeDatasetConversationSource::preferred_with_prepared_resolver_for_partition(
                dataset,
                "model",
                4,
                RngRoot::new(Some(7)),
                &samplers,
                resolver,
                Some(ModuloCellPartition::new(0, 2).unwrap()),
            )
            .unwrap();

        let owned = owned_ids(&source);
        assert_eq!(owned.len(), 3); // indices 0,2,4 of five rows
        for _ in 0..9 {
            let session = source.next(None).unwrap();
            assert!(
                owned.contains(&session.conversation_id),
                "shuffle giver leaked {}",
                session.conversation_id
            );
            session.build_first_turn(None).unwrap();
        }
    }

    /// Request-rate construction caches one first turn from the owned giver;
    /// further giver draws after wraparound must stay sampleable (the issuer
    /// refills from `ConversationSource::next` after each successful issue —
    /// the original "is not sampleable" failure mode).
    #[tokio::test]
    async fn request_rate_builds_and_owned_recycle_stays_sampleable() {
        use crate::request_rate::{RequestRateConfig, RequestRateWorkload};
        use crate::timing::ArrivalPattern;

        let dataset = raw_payload_jsonl_dataset(3).await;
        RequestRateWorkload::new(
            RequestRateConfig {
                arrival_pattern: ArrivalPattern::Constant,
                request_rate: Some(10.0),
                arrival_smoothness: None,
                session_concurrency: None,
                prefill_concurrency: None,
                seed: 1,
            },
            Box::new(partitioned_sequential_source(dataset.clone(), 0, 2).await),
        )
        .expect("request-rate must accept a non-empty owned giver");

        let mut source = partitioned_sequential_source(dataset, 0, 2).await;
        let owned = owned_ids(&source);
        for _ in 0..6 {
            let session = source
                .next(None)
                .expect("recycle refill must stay sampleable");
            assert!(owned.contains(&session.conversation_id));
            session.build_first_turn(None).unwrap();
        }
    }

    async fn raw_payload_jsonl_dataset(rows: usize) -> NativeDataset {
        let jsonl = (0..rows)
            .map(|n| {
                format!(r#"{{"messages":[{{"role":"user","content":"p{n}"}}],"stream":false}}"#)
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("raw_payload"),
                &LoadConfig::new(DatasetSource::Bytes(Bytes::from(jsonl))),
                &ComposeConfig::new("model", RngRoot::new(Some(1))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap()
    }

    async fn partitioned_sequential_source(
        dataset: NativeDataset,
        cell_id: u32,
        cell_count: u32,
    ) -> NativeDatasetConversationSource {
        let registry = crate::endpoints::EndpointRegistry::builtin().unwrap();
        let endpoint_id = EndpointId::new("chat").unwrap();
        let endpoint = registry
            .prepare(
                &endpoint_id,
                crate::endpoints::RawEndpointConfig {
                    streaming: false,
                    ..crate::endpoints::RawEndpointConfig::default()
                },
            )
            .unwrap();
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint).unwrap();
        let resolver: Rc<dyn PreparedTurnEndpointResolver> = Rc::new(
            PreparedEndpointTableResolver::single(
                Rc::new(table),
                PreparedEndpointReference { key, endpoint_id },
            )
            .unwrap(),
        );
        NativeDatasetConversationSource::sequential_with_prepared_resolver_for_partition(
            dataset,
            "model",
            4,
            resolver,
            Some(ModuloCellPartition::new(cell_id, cell_count).unwrap()),
        )
        .unwrap()
    }

    fn owned_ids(source: &NativeDatasetConversationSource) -> Vec<String> {
        source
            .conversations()
            .iter()
            .map(|conversation| conversation.conversation_id.clone())
            .collect()
    }
}
