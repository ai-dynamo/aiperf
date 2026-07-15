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

use crate::cellular::partition::ModuloCellPartition;
use crate::dataset::{
    ConversationSession as NativeConversationSession, Dataset as NativeDataset,
    EndpointRequestMaterializer, Handle, Overrides, Payload, RequestMaterializer, Sampler,
    SamplerRegistry, SegmentStore, SequentialSampler, TextTokenizer, TiktokenTokenizer,
};
use crate::endpoints::{
    CreditPhase, Endpoint, EndpointId, EndpointKey, Media as EndpointMedia, PreparedEndpoint,
    PreparedEndpointTable, ShapeLowerer, Turn as EndpointTurn, TurnMessageLowerer,
};
use crate::graph::wire::OpenAiChatMessage;
use crate::rng::RngRoot;
use crate::timing::{RunState, StopConfig};
use anyhow::{Result, anyhow, bail};
use bytes::Bytes;
use loadgen_core::collector::ReplayTerminalStatus;
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
    /// Policies that preserve authored counts inherit this default. Exact
    /// endpoint-aware counters override it without reconstructing a legacy
    /// endpoint/configuration pair.
    fn count_prepared_input_tokens(
        &self,
        _endpoint: &dyn PreparedEndpoint,
        _body: &[u8],
        authored_input_tokens: u64,
    ) -> Result<u64> {
        Ok(authored_input_tokens)
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
        extracted: crate::endpoints::ExtractedPayload,
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
            endpoint.extract_payload_inputs(&body),
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
            endpoint.extract_payload_inputs(&body),
            authored_input_tokens,
        )
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
/// The ordinary single-profile implementation is
/// [`PreparedEndpointTableResolver`]. A future remote catalog or workload-local
/// routing policy can implement the same contract without changing session,
/// scheduling, or dispatch code.
pub trait PreparedTurnEndpointResolver: fmt::Debug {
    /// Resolve an authored per-turn endpoint name, or the run default when
    /// absent, to one dense prepared binding.
    fn resolve(&self, name: Option<&str>) -> Result<ResolvedPreparedEndpoint<'_>>;
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

/// Lower every static content turn to a pre-serialized `Message` segment for the
/// run's default prepared endpoint (segment spec §5), so dispatch splices the
/// stored wire instead of re-rendering. Endpoints whose body is not a per-turn
/// message array (embeddings, completions, media, …) have no lowerer and are
/// left on the live render path; per-turn endpoint overrides are skipped inside
/// the lowering pass itself.
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
    // Segment spec §3a: after lowering, precompute and cache each eligible static
    // turn's BodyPlan against the default prepared endpoint so dispatch clones it
    // instead of reformatting. Runs on the same (lowered) segment store dispatch
    // sees, so a cache hit is byte-identical to the live formatter output.
    dataset
        .precompute_body_plans(resolved.endpoint, primary_model_name)
        .map_err(|error| anyhow!("failed to precompute body plans: {error}"))?;
    Ok(())
}

/// Endpoint selection retained by one schedulable turn.
///
/// This is an enum rather than a bare [`PreparedEndpointReference`] to keep the
/// seam open: a future non-prepared execution binding can be added as a new
/// variant without touching every turn consumer.
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
    pub request_body: Option<Bytes>,
    /// Per-turn HTTP headers.
    pub request_headers: BTreeMap<String, String>,
    /// Per-turn URL query parameters.
    pub request_parameters: BTreeMap<String, String>,
    /// Endpoint path selected by the endpoint resolver.
    pub endpoint_path: Option<String>,
    /// Legacy adapter or open prepared endpoint identity used for execution.
    pub endpoint: TurnEndpoint,
    /// Whether this endpoint returns an SSE stream.
    pub streaming: bool,
    /// Audio duration propagated into ASR metrics.
    pub audio_duration_seconds: Option<f64>,
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
        if self.metadata.turns.is_empty() {
            bail!("conversation {} has no turns", owner.conversation_id);
        }
        let num_turns = max_turns
            .unwrap_or(self.metadata.turns.len())
            .min(self.metadata.turns.len())
            .max(1);
        self.materialize(owner, 0, num_turns)
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
            // Lower the captured live reply once, here at capture (segment spec §5),
            // so every later multi-turn dispatch splices the stored wire instead of
            // re-serializing the reply body through `RenderedMessage::Value` on each
            // request. The shape is bound to the run's DEFAULT prepared endpoint —
            // the same commit `lower_static_messages` makes at load — never the
            // per-turn override. Non-message-array dialects (completions, embeddings,
            // …) have no lowerer; there the reply stays on the live render path
            // (today's behavior). Resolving the shape does not touch the session, so
            // it happens before the mutable borrow below.
            let lowerer = match &self.endpoint {
                NativeSessionEndpoint::Prepared {
                    endpoint_resolver, ..
                } => {
                    let resolved = endpoint_resolver.resolve(None)?;
                    ShapeLowerer::for_descriptor_id(resolved.endpoint.descriptor().id)
                }
            };
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
                session.capture_response(reply, tokens)?;
            }
        }
        self.materialize(owner, current.turn_index + 1, current.num_turns)
    }
}

impl NativeSessionBackend {
    fn materialize(
        &self,
        owner: &SampledSession,
        turn_index: usize,
        num_turns: usize,
    ) -> Result<TurnToSend> {
        let mut session = self.session.borrow_mut();
        session.advance_to(turn_index)?;
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
                let counted = self.input_token_counter.count_prepared_input_tokens(
                    *endpoint,
                    &materialized.body,
                    materialized.input_tokens,
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

    /// Same as [`Self::preferred_with_prepared_resolver`] but with an explicitly
    /// injected cell partition instead of the process-environment default. `None`
    /// reads the partition from the env (`AIPERF_CELL_ID`/`_COUNT`) — the
    /// byte-unchanged default every current caller takes; `Some` restricts this
    /// source to the partition's owned instances, a per-thread `(cell_id,
    /// cell_count)` slice the process-global env vars cannot express, for a future
    /// single-process thread-per-core scheduled run whose `W` sub-cell threads each
    /// own a distinct partition.
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
        let sampler = samplers.create(
            &dataset.metadata().sampling_strategy,
            &dataset.metadata().conversations,
            rng_root,
        )?;
        Self::new_with_endpoint(
            dataset,
            sampler,
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

    /// Same as [`Self::sequential_with_prepared_resolver`] but with an explicitly
    /// injected cell partition instead of the process-environment default. `None`
    /// reads the partition from the env (`AIPERF_CELL_ID`/`_COUNT`) — the
    /// byte-unchanged default every current caller takes; `Some` restricts this
    /// source to the partition's owned instances, a per-thread `(cell_id,
    /// cell_count)` slice the process-global env vars cannot express, for a future
    /// single-process thread-per-core scheduled run whose `W` sub-cell threads each
    /// own a distinct partition.
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
        let sampler = SequentialSampler::from_metadata(&dataset.metadata().conversations)?;
        Self::new_with_endpoint(
            dataset,
            Box::new(sampler),
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
        sampler: Box<dyn Sampler>,
        endpoint: NativeSessionEndpoint,
        materializer: Arc<dyn RequestMaterializer>,
        response_tokenizer: Arc<dyn TextTokenizer>,
        default_output_tokens: usize,
        cell_partition: Option<ModuloCellPartition>,
    ) -> Result<Self> {
        if default_output_tokens == 0 {
            bail!("native dataset default output tokens must be positive");
        }
        // A cell of a multi-cell run yields only its owned instances (roadmap S4);
        // the single-process path returns the sampler unchanged (byte-identical).
        // `None` reads the partition from the process env — the byte-unchanged
        // default for every current caller; `Some` injects a per-thread partition
        // the process-global env vars cannot express (thread-per-core scheduled run).
        let sampler = match cell_partition {
            None => crate::dataset::sampler::PartitionedSampler::from_env(sampler),
            Some(partition) => {
                crate::dataset::sampler::PartitionedSampler::for_partition(sampler, Some(partition))
            }
        };
        let metadata = dataset
            .sampleable_metadata()
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
                            input_length: usize::try_from(turn.input_tokens)
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
}

impl ConversationSource for NativeDatasetConversationSource {
    fn conversations(&self) -> &[ConversationMetadata] {
        &self.metadata
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

/// One fully preformatted profiling turn: everything the dispatch path pulls
/// from the resident `Dataset`/`SegmentStore` for a single-turn profiling
/// request, captured up front so a store-free source can reproduce a byte- and
/// metrics-identical [`TurnToSend`].
///
/// This is the owned-bytes counterpart to the store-resident materialization in
/// [`NativeSessionBackend::materialize`]: the fields mirror exactly what that
/// method reads out of a [`crate::dataset::MaterializedRequest`] and the
/// authored [`TurnMetadata`] to build a `TurnToSend`. The one deliberately
/// deferred field is the runtime session correlation id, which is minted per
/// runtime session at sampling time; [`Self::to_turn_to_send`] splices in the
/// owner's id (or the captured accuracy id when present), matching the
/// `request_correlation_id` rule at the store-backed construction site.
///
/// Constructed only by
/// [`Dataset::prebuild_profiling_dispatch`](crate::dataset::Dataset::prebuild_profiling_dispatch)
/// for datasets that pass
/// [`qualifies_for_prebuilt_free`](crate::dataset::Dataset::qualifies_for_prebuilt_free);
/// trace-hash, token-native, and raw-body turns are excluded by that gate, so
/// the corresponding `TurnToSend` fields are always absent here.
#[derive(Clone, Debug)]
pub struct PreparedTurn {
    /// Effective wire model resolved during materialization.
    pub effective_model: String,
    /// Captured accuracy correlation id, when the conversation carried one. When
    /// absent the runtime session id is used for `request_correlation_id`.
    pub accuracy_correlation_id: Option<String>,
    /// Zero-based authored turn index (always `0` for a single-turn prebuild).
    pub turn_index: usize,
    /// Total turns the runtime session will send (always `1` here).
    pub num_turns: usize,
    /// Final input-token count after the run's input-token counting policy.
    pub input_length: usize,
    /// Requested output-token cap after endpoint/default resolution.
    pub max_output_tokens: usize,
    /// Fully materialized request body ready to clone onto the wire.
    pub request_body: Bytes,
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
    /// Absolute trace timestamp for this turn, if any.
    pub timestamp_ms: Option<f64>,
    /// Relative delay for this turn, if any.
    pub delay_ms: Option<f64>,
}

impl PreparedTurn {
    /// Assemble a [`TurnToSend`] identical to the store-backed one for this turn,
    /// cloning the preformatted bytes and copying the captured ancillaries. The
    /// owning [`SampledSession`] supplies the per-session identity that is not
    /// known until sampling; every metrics-relevant field (body, input length,
    /// headers, parameters, model, output cap, streaming) is copied verbatim.
    fn to_turn_to_send(&self, owner: &SampledSession) -> TurnToSend {
        // Store-backed rule (multiturn.rs materialize): the correlation id sent to
        // the backend is the accuracy identity when the conversation carried one,
        // otherwise the runtime session id minted at sampling time.
        let request_correlation_id = self
            .accuracy_correlation_id
            .clone()
            .unwrap_or_else(|| owner.x_correlation_id.clone());
        TurnToSend {
            uuid: Uuid::new_v4(),
            effective_model: Some(self.effective_model.clone()),
            conversation_id: owner.conversation_id.clone(),
            x_correlation_id: owner.x_correlation_id.clone(),
            request_correlation_id,
            turn_index: self.turn_index,
            num_turns: self.num_turns,
            input_length: self.input_length,
            max_output_tokens: self.max_output_tokens,
            messages: Vec::new(),
            request_body: Some(self.request_body.clone()),
            request_headers: self.request_headers.clone(),
            request_parameters: self.request_parameters.clone(),
            endpoint_path: self.endpoint_path.clone(),
            endpoint: self.endpoint.clone(),
            streaming: self.streaming,
            audio_duration_seconds: self.audio_duration_seconds,
            timestamp_ms: self.timestamp_ms,
            delay_ms: self.delay_ms,
            // Excluded by the qualification gate: a prebuilt-and-freed turn never
            // carries simulator trace identities or token-native handles, and its
            // content is ordinary benchmark content.
            trace_hash_ids: None,
            raw_token_ids: None,
            data_policy: TurnDataPolicy::ordinary(),
            cancel_after_ns: None,
            url_index: None,
            session: owner.clone(),
        }
    }
}

/// One preformatted runtime session: all of a conversation's prepared turns.
///
/// Single-turn for now (the store-free dispatch path only targets single-turn
/// static runs), but modeled as a `Vec` so a later multi-turn extension reuses
/// the same type without a breaking change.
#[derive(Clone, Debug)]
pub struct PreparedSession {
    /// Authored conversation/template identifier.
    pub session_id: String,
    /// Ordered prepared turns for this conversation.
    pub turns: Vec<PreparedTurn>,
}

/// Store-free [`RuntimeSessionBackend`] over one preformatted session.
///
/// Holds no `Arc<Dataset>` / `Arc<dyn SegmentStore>`: turn production clones the
/// owned prepared bytes and copies the captured ancillaries. Single-turn only —
/// continuation/live-reply construction fails closed.
#[derive(Debug)]
struct PreformattedSessionBackend {
    prepared: Rc<PreparedSession>,
}

impl RuntimeSessionBackend for PreformattedSessionBackend {
    fn available_turns(&self) -> usize {
        self.prepared.turns.len()
    }

    fn build_first_turn(
        &self,
        owner: &SampledSession,
        _max_turns: Option<usize>,
    ) -> Result<TurnToSend> {
        // Single-turn: `num_turns` is fixed at prebuild time, so a caller cap is
        // irrelevant and intentionally ignored.
        let turn = self.prepared.turns.first().ok_or_else(|| {
            anyhow!(
                "preformatted session {:?} has no prepared turns",
                self.prepared.session_id
            )
        })?;
        Ok(turn.to_turn_to_send(owner))
    }

    fn next_metadata(&self, _turn_index: usize) -> Result<TurnMetadata> {
        bail!("preformatted conversation source is single-turn only")
    }

    fn build_next_turn(
        &self,
        _owner: &SampledSession,
        _current: &TurnToSend,
        _response: TurnResponse,
    ) -> Result<TurnToSend> {
        bail!("preformatted conversation source cannot build a continuation turn")
    }
}

/// Store-free [`ConversationSource`] backed by preformatted profiling turns.
///
/// Produced from [`Dataset::prebuild_profiling_dispatch`](crate::dataset::Dataset::prebuild_profiling_dispatch)
/// output; it retains only owned [`PreparedSession`] bytes and a sampler, so the
/// resident `Dataset`/`SegmentStore` can be dropped before dispatch. Its turn
/// production is byte- and metrics-identical to
/// [`NativeDatasetConversationSource`] for the same single-turn profiling turn.
pub struct PreformattedConversationSource {
    metadata: Vec<ConversationMetadata>,
    sessions_by_id: HashMap<String, Rc<PreparedSession>>,
    sampler: Box<dyn Sampler>,
}

impl fmt::Debug for PreformattedConversationSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreformattedConversationSource")
            .field("sessions", &self.sessions_by_id.len())
            .finish_non_exhaustive()
    }
}

impl PreformattedConversationSource {
    /// Build a source over preformatted sessions with an injected sampler.
    ///
    /// Rejects an empty session set or any non-single-turn session. The
    /// [`ConversationMetadata`] projection is derived from the prepared turns so
    /// the source needs no residual dataset reference; its `input_length` is the
    /// counted dispatch value (this projection feeds strategy setup and request
    /// counting, not the dispatch body).
    pub fn new(sessions: Vec<PreparedSession>, sampler: Box<dyn Sampler>) -> Result<Self> {
        if sessions.is_empty() {
            bail!("preformatted conversation source requires at least one session");
        }
        for session in &sessions {
            if session.turns.len() != 1 {
                bail!(
                    "preformatted conversation source is single-turn only; session {:?} has {} turns",
                    session.session_id,
                    session.turns.len()
                );
            }
        }
        let metadata = sessions
            .iter()
            .map(|session| ConversationMetadata {
                conversation_id: session.session_id.clone(),
                turns: session
                    .turns
                    .iter()
                    .map(|turn| TurnMetadata {
                        timestamp_ms: turn.timestamp_ms,
                        delay_ms: turn.delay_ms,
                        trace_hash_ids: None,
                        prompt_text: String::new(),
                        input_length: turn.input_length,
                        max_output_tokens: turn.max_output_tokens,
                    })
                    .collect(),
            })
            .collect();
        let sessions_by_id = sessions
            .into_iter()
            .map(|session| (session.session_id.clone(), Rc::new(session)))
            .collect();
        Ok(Self {
            metadata,
            sessions_by_id,
            sampler,
        })
    }

    /// Build a source with insertion-order sampling over the prepared sessions,
    /// mirroring a sequential-sampler dataset run.
    pub fn sequential(sessions: Vec<PreparedSession>) -> Result<Self> {
        let ids = sessions
            .iter()
            .map(|session| crate::dataset::SessionId::from(session.session_id.as_str()))
            .collect();
        let sampler = SequentialSampler::new(ids)
            .map_err(|error| anyhow!("preformatted sampler: {error}"))?;
        Self::new(sessions, Box::new(sampler))
    }

    fn build_session(
        &self,
        conversation_id: &str,
        x_correlation_id: String,
    ) -> Result<SampledSession> {
        let prepared = self
            .sessions_by_id
            .get(conversation_id)
            .cloned()
            .ok_or_else(|| {
                anyhow!("preformatted source has no conversation {conversation_id:?}")
            })?;
        Ok(SampledSession {
            conversation_id: conversation_id.to_string(),
            x_correlation_id,
            backend: Rc::new(PreformattedSessionBackend { prepared }),
        })
    }
}

impl ConversationSource for PreformattedConversationSource {
    fn conversations(&self) -> &[ConversationMetadata] {
        &self.metadata
    }

    fn next(&mut self, x_correlation_id: Option<String>) -> Result<SampledSession> {
        let id = self.sampler.next();
        let x_correlation_id = x_correlation_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        self.build_session(id.as_str(), x_correlation_id)
    }

    fn session_for(
        &self,
        conversation_id: &str,
        x_correlation_id: String,
    ) -> Result<SampledSession> {
        self.build_session(conversation_id, x_correlation_id)
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

    #[test]
    fn endpoint_counter_matches_python_template_and_bare_paths() {
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

    #[tokio::test]
    async fn counter_matches_python_root_counting_rules() {
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
        let first_body: Value =
            serde_json::from_slice(first.request_body.as_ref().unwrap()).unwrap();
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
        let next_body: Value = serde_json::from_slice(next.request_body.as_ref().unwrap()).unwrap();
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
        // Segment spec §5: a live reply captured after turn 0 is lowered once at
        // capture and its stored wire is spliced verbatim into the context of every
        // later dispatch. A 3-turn conversation surfaces reply-0 in both the turn-1
        // and turn-2 request bodies, so byte-drift between dispatches would show up.
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

        let body1: Value = serde_json::from_slice(turn1.request_body.as_ref().unwrap()).unwrap();
        let body2: Value = serde_json::from_slice(turn2.request_body.as_ref().unwrap()).unwrap();
        // reply-0 is message[1] in both dispatch bodies and must be identical.
        assert_eq!(body1["messages"][1], body2["messages"][1]);
        assert_eq!(body1["messages"][1]["content"], "reply-0");
        assert_eq!(body2["messages"][3]["content"], "reply-1");

        // The spliced reply equals the pre-change re-rendered body: a single-text
        // assistant turn renders to `{"role":"assistant","content":"reply-0"}`
        // (byte-parity of `lower_turn` vs that render is pinned in the endpoints
        // `reply_constructors_lower_to_value_dispatch_wire_all_shapes` test).
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
        let body: Value = serde_json::from_slice(next.request_body.as_ref().unwrap()).unwrap();
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
        assert_eq!(turn.request_body.unwrap(), authored);
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

        let TurnEndpoint::Prepared(reference) = turn.endpoint;
        assert_eq!(reference.key, key);
        assert_eq!(reference.endpoint_id, endpoint_id);
        let body: Value = serde_json::from_slice(turn.request_body.as_ref().unwrap()).unwrap();
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

    #[tokio::test]
    async fn preformatted_source_matches_store_backed_turn() {
        let dataset = LoaderRegistry::with_builtin_formats()
            .unwrap()
            .build_dataset(
                Some("single_turn"),
                &LoadConfig::new(DatasetSource::Inline(
                    json!([{"text":"hello world"},{"text":"a different prompt"}]),
                )),
                &ComposeConfig::new("model", RngRoot::new(Some(11))),
                &TiktokenTokenizer::builtin(),
            )
            .await
            .unwrap();

        // Store-backed source lowers its own copy of the dataset internally.
        let native = prepared_chat_source(dataset.clone(), "model", 16);

        // Prebuild over a copy lowered exactly the way the online source lowers,
        // through an identically configured `chat` prepared endpoint.
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
        let resolver = PreparedEndpointTableResolver::single(
            Rc::new(table),
            PreparedEndpointReference {
                key,
                endpoint_id: endpoint_id.clone(),
            },
        )
        .unwrap();

        let mut lowered = dataset;
        {
            let resolved = resolver.resolve(None).unwrap();
            if let Some(lowerer) =
                ShapeLowerer::for_descriptor_id(resolved.endpoint.descriptor().id)
            {
                lowered.lower_messages_for_endpoint(&lowerer).unwrap();
            }
            lowered
                .precompute_body_plans(resolved.endpoint, "model")
                .unwrap();
        }
        let lowered = Arc::new(lowered);
        let prepared = lowered
            .prebuild_profiling_dispatch(&resolver, "model", &AuthoredInputTokenCounter, 16)
            .unwrap();
        let preformatted = PreformattedConversationSource::sequential(prepared).unwrap();

        // Same conversation, same runtime correlation id: sampling is bypassed so
        // the two turns must be byte- and metrics-identical.
        let id = native.conversations()[0].conversation_id.clone();
        let native_turn = native
            .session_for(&id, "runtime-correlation".into())
            .unwrap()
            .build_first_turn(Some(1))
            .unwrap();
        let preformatted_turn = preformatted
            .session_for(&id, "runtime-correlation".into())
            .unwrap()
            .build_first_turn(Some(1))
            .unwrap();

        assert_eq!(
            native_turn.request_body, preformatted_turn.request_body,
            "request body byte-parity"
        );
        assert_eq!(native_turn.input_length, preformatted_turn.input_length);
        assert_eq!(
            native_turn.request_headers,
            preformatted_turn.request_headers
        );
        assert_eq!(
            native_turn.request_parameters,
            preformatted_turn.request_parameters
        );
        assert_eq!(
            native_turn.effective_model,
            preformatted_turn.effective_model
        );
        assert_eq!(
            native_turn.max_output_tokens,
            preformatted_turn.max_output_tokens
        );
        assert_eq!(native_turn.streaming, preformatted_turn.streaming);
        assert_eq!(
            native_turn.request_correlation_id,
            preformatted_turn.request_correlation_id
        );
        assert_eq!(native_turn.endpoint_path, preformatted_turn.endpoint_path);
        assert_eq!(native_turn.num_turns, preformatted_turn.num_turns);
        assert_eq!(native_turn.turn_index, preformatted_turn.turn_index);
        assert_eq!(native_turn.timestamp_ms, preformatted_turn.timestamp_ms);
        assert_eq!(native_turn.delay_ms, preformatted_turn.delay_ms);
        let TurnEndpoint::Prepared(native_reference) = &native_turn.endpoint;
        let TurnEndpoint::Prepared(preformatted_reference) = &preformatted_turn.endpoint;
        assert_eq!(native_reference.key, preformatted_reference.key);
        assert_eq!(
            native_reference.endpoint_id,
            preformatted_reference.endpoint_id
        );

        // The single-turn contract: no continuation turn exists.
        let credit = IssuedCredit::from_turn(0, 0, &preformatted_turn);
        assert!(
            preformatted
                .next_turn(
                    &credit,
                    TurnResponse {
                        text: String::new(),
                        assistant_message: None,
                        completion_tokens: None,
                        terminal: ReplayTerminalStatus::Completed,
                    },
                )
                .unwrap()
                .is_none()
        );
    }
}
