// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dataset-backed multi-turn conversation sourcing and prompt materialization.
//!
//! This is the native counterpart of
//! `src/aiperf/timing/conversation_source.py:23-198` and
//! `src/aiperf/credit/structs.py:113-163`. A [`ConversationSource`] samples
//! reusable templates, mints a distinct runtime correlation id for each session,
//! caps virtual-history sessions safely to the sampled template length, and
//! builds continuation turns. Static user messages live in the unified
//! `aiperf-dataset` content-addressed [`SegmentStore`]; real assistant replies
//! are appended dynamically before the next user segment, preserving growing
//! multi-turn context without reserializing stored static messages.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use aiperf_dataset::{
    BuiltinEndpointResolver, ConversationSession as NativeConversationSession,
    Dataset as NativeDataset, EndpointRequestMaterializer, EndpointResolver, Handle, Overrides,
    Payload, RequestMaterializer, Sampler, SegmentPool, SegmentStore, SequentialSampler,
    TextTokenizer, TiktokenTokenizer,
};
use aiperf_endpoints::{
    CreditPhase, EndpointConfig, Media as EndpointMedia, ModelEndpoint, Turn as EndpointTurn,
};
use aiperf_graph::segment::intern_message;
use aiperf_graph::wire::OpenAiChatMessage;
use aiperf_timing::{RunState, StopConfig};
use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;
use loadgen_core::collector::ReplayTerminalStatus;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::workload::SkeletonWorkload;

/// Metadata and static user content for one turn in a conversation template.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TurnMetadata {
    /// Absolute trace timestamp in milliseconds, when present.
    pub timestamp_ms: Option<f64>,
    /// Relative think time after the previous response, in milliseconds.
    pub delay_ms: Option<f64>,
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
            prompt_text: prompt_text.into(),
            input_length,
            max_output_tokens,
        }
    }

    fn validate(&self, conversation_id: &str, turn_index: usize) -> Result<()> {
        if self.timestamp_ms.is_some_and(|v| !v.is_finite()) {
            bail!("turn {turn_index} of {conversation_id} has non-finite timestamp_ms");
        }
        if self.delay_ms.is_some_and(|v| !v.is_finite() || v < 0.0) {
            bail!("turn {turn_index} of {conversation_id} has invalid delay_ms");
        }
        if self.input_length == 0 {
            bail!("turn {turn_index} of {conversation_id} has zero input_length");
        }
        if self.max_output_tokens == 0 {
            bail!("turn {turn_index} of {conversation_id} has zero max_output_tokens");
        }
        Ok(())
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

/// A validated dataset plus its content-addressed static-message store.
#[derive(Clone)]
pub struct ConversationDataset {
    conversations: Vec<ConversationMetadata>,
    by_id: HashMap<String, usize>,
    segment_ids: HashMap<String, Vec<Handle>>,
    segments: Rc<SegmentPool>,
}

impl fmt::Debug for ConversationDataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConversationDataset")
            .field("conversations", &self.conversations)
            .field("segments", &self.segments.len())
            .finish()
    }
}

impl ConversationDataset {
    /// Validate `conversations` and build their prefix-dependent segment pool.
    /// Empty datasets are retained so fixed-schedule setup can produce its
    /// strategy-specific “no valid conversations” error.
    pub fn new(conversations: Vec<ConversationMetadata>) -> Result<Self> {
        let mut by_id = HashMap::with_capacity(conversations.len());
        let mut segment_ids = HashMap::with_capacity(conversations.len());
        let mut segments = SegmentPool::new();
        let tokenizer = TiktokenTokenizer::builtin();

        for (conversation_index, conversation) in conversations.iter().enumerate() {
            if conversation.conversation_id.is_empty() {
                bail!("conversation id cannot be empty");
            }
            if by_id
                .insert(conversation.conversation_id.clone(), conversation_index)
                .is_some()
            {
                bail!("duplicate conversation id {}", conversation.conversation_id);
            }

            let mut parent: Option<Handle> = None;
            let mut ids = Vec::with_capacity(conversation.turns.len());
            for (turn_index, turn) in conversation.turns.iter().enumerate() {
                turn.validate(&conversation.conversation_id, turn_index)?;
                let message = OpenAiChatMessage::new("user", turn.prompt_text.clone());
                let id = intern_message(&mut segments, &message, parent, &tokenizer)?;
                parent = Some(id);
                ids.push(id);
            }
            segment_ids.insert(conversation.conversation_id.clone(), ids);
        }

        Ok(Self {
            conversations,
            by_id,
            segment_ids,
            segments: Rc::new(segments),
        })
    }

    /// Load native JSON/JSONL trace data from `path`.
    ///
    /// Supported row aliases intentionally cover the inherited Mooncake-style
    /// examples: `session_id|conversation_id`, `timestamp|timestamp_ms`,
    /// `delay|delay_ms`, `text_input|input_text|prompt|prompt_text`, and
    /// `output_length|max_tokens|max_output_tokens`. Rows sharing a session id
    /// become ordered turns; rows without one become independent conversations.
    /// A native object with `{conversation_id, turns:[...]}` or a top-level
    /// `{conversations:[...]}` is accepted as well.
    pub fn from_path(
        path: impl AsRef<Path>,
        default_input_length: usize,
        default_output_tokens: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        let input = std::fs::read_to_string(path)
            .with_context(|| format!("reading conversation dataset {}", path.display()))?;
        Self::from_json_or_jsonl(&input, default_input_length, default_output_tokens)
            .with_context(|| format!("parsing conversation dataset {}", path.display()))
    }

    /// Parse native JSON or JSONL conversation data.
    pub fn from_json_or_jsonl(
        input: &str,
        default_input_length: usize,
        default_output_tokens: usize,
    ) -> Result<Self> {
        if default_input_length == 0 || default_output_tokens == 0 {
            bail!("dataset token defaults must be positive");
        }
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Self::new(Vec::new());
        }

        let values = if trimmed.starts_with('[') {
            serde_json::from_str::<Vec<Value>>(trimmed).context("parsing JSON array")?
        } else if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
            match value {
                Value::Object(ref object) if object.contains_key("conversations") => object
                    .get("conversations")
                    .and_then(Value::as_array)
                    .cloned()
                    .ok_or_else(|| anyhow!("conversations must be an array"))?,
                value => vec![value],
            }
        } else {
            input
                .lines()
                .enumerate()
                .filter(|(_, line)| !line.trim().is_empty())
                .map(|(index, line)| {
                    serde_json::from_str::<Value>(line)
                        .with_context(|| format!("parsing JSONL line {}", index + 1))
                })
                .collect::<Result<Vec<_>>>()?
        };

        let mut conversations: Vec<ConversationMetadata> = Vec::new();
        let mut row_groups: HashMap<String, usize> = HashMap::new();
        for (row_index, value) in values.into_iter().enumerate() {
            let object = value
                .as_object()
                .ok_or_else(|| anyhow!("dataset entry {} must be an object", row_index + 1))?;
            if let Some(turn_values) = object.get("turns") {
                let conversation_id = string_field(object, &["conversation_id", "session_id"])
                    .unwrap_or_else(|| format!("conversation-{row_index}"));
                let turns = turn_values
                    .as_array()
                    .ok_or_else(|| anyhow!("turns for {conversation_id} must be an array"))?
                    .iter()
                    .enumerate()
                    .map(|(turn_index, value)| {
                        let object = value.as_object().ok_or_else(|| {
                            anyhow!("turn {turn_index} of {conversation_id} must be an object")
                        })?;
                        parse_turn(object, default_input_length, default_output_tokens)
                    })
                    .collect::<Result<Vec<_>>>()?;
                conversations.push(ConversationMetadata {
                    conversation_id,
                    turns,
                });
                continue;
            }

            let supplied_id = string_field(object, &["conversation_id", "session_id"]);
            let conversation_id = supplied_id
                .clone()
                .unwrap_or_else(|| format!("trace-row-{row_index}"));
            let turn = parse_turn(object, default_input_length, default_output_tokens)?;
            if supplied_id.is_none() {
                conversations.push(ConversationMetadata {
                    conversation_id,
                    turns: vec![turn],
                });
            } else if let Some(index) = row_groups.get(&conversation_id).copied() {
                conversations[index].turns.push(turn);
            } else {
                row_groups.insert(conversation_id.clone(), conversations.len());
                conversations.push(ConversationMetadata {
                    conversation_id,
                    turns: vec![turn],
                });
            }
        }

        Self::new(conversations)
    }

    /// Dataset conversations in stable loader order.
    pub fn conversations(&self) -> &[ConversationMetadata] {
        &self.conversations
    }

    /// Average turn count across all conversations, or `0.0` when empty.
    pub fn average_turn_count(&self) -> f64 {
        if self.conversations.is_empty() {
            return 0.0;
        }
        self.conversations
            .iter()
            .map(|conversation| conversation.turns.len())
            .sum::<usize>() as f64
            / self.conversations.len() as f64
    }

    /// Return a rebuilt dataset containing only conversations whose first-turn
    /// timestamp lies inside the inclusive `[start_ms, end_ms]` window.
    /// Empty conversations are excluded because they have no replay timestamp.
    pub fn filter_first_turn_window(
        &self,
        start_ms: Option<f64>,
        end_ms: Option<f64>,
    ) -> Result<Self> {
        if start_ms.is_some_and(|v| !v.is_finite() || v < 0.0)
            || end_ms.is_some_and(|v| !v.is_finite() || v < 0.0)
        {
            bail!("fixed-schedule offsets must be finite and non-negative");
        }
        if let (Some(start), Some(end)) = (start_ms, end_ms)
            && start > end
        {
            bail!("fixed-schedule start offset must be <= end offset");
        }

        let conversations = self
            .conversations
            .iter()
            .filter(|conversation| {
                let Some(timestamp) = conversation
                    .turns
                    .first()
                    .and_then(|turn| turn.timestamp_ms)
                else {
                    return false;
                };
                start_ms.is_none_or(|start| timestamp >= start)
                    && end_ms.is_none_or(|end| timestamp <= end)
            })
            .cloned()
            .collect();
        Self::new(conversations)
    }

    fn session(&self, conversation_id: &str, x_correlation_id: String) -> Result<SampledSession> {
        let index = self
            .by_id
            .get(conversation_id)
            .copied()
            .ok_or_else(|| anyhow!("no metadata for conversation {conversation_id}"))?;
        let metadata = self.conversations[index].clone();
        let segment_ids = self
            .segment_ids
            .get(conversation_id)
            .cloned()
            .expect("validated dataset has segment ids for every conversation");
        let segments: Rc<dyn SegmentStore> = self.segments.clone();
        Ok(SampledSession {
            conversation_id: conversation_id.to_string(),
            x_correlation_id,
            backend: Rc::new(LegacySessionBackend {
                metadata,
                segment_ids,
                segments,
            }),
        })
    }
}

fn string_field(object: &Map<String, Value>, names: &[&str]) -> Option<String> {
    names
        .iter()
        .find_map(|name| object.get(*name).and_then(Value::as_str))
        .map(ToString::to_string)
}

fn number_field(object: &Map<String, Value>, names: &[&str]) -> Option<f64> {
    names
        .iter()
        .find_map(|name| object.get(*name).and_then(Value::as_f64))
}

fn usize_field(object: &Map<String, Value>, names: &[&str]) -> Option<usize> {
    names
        .iter()
        .find_map(|name| object.get(*name).and_then(Value::as_u64))
        .and_then(|value| usize::try_from(value).ok())
}

fn parse_turn(
    object: &Map<String, Value>,
    default_input_length: usize,
    default_output_tokens: usize,
) -> Result<TurnMetadata> {
    let explicit_prompt = string_field(
        object,
        &["prompt_text", "text_input", "input_text", "prompt"],
    );
    let input_length =
        usize_field(object, &["input_length", "input_tokens"]).unwrap_or_else(|| {
            explicit_prompt
                .as_deref()
                .map(|prompt| prompt.split_whitespace().count().max(1))
                .unwrap_or(default_input_length)
        });
    let prompt_text = explicit_prompt.unwrap_or_else(|| vec!["lorem"; input_length].join(" "));
    let max_output_tokens = usize_field(
        object,
        &["max_output_tokens", "output_length", "max_tokens"],
    )
    .unwrap_or(default_output_tokens);
    Ok(TurnMetadata {
        timestamp_ms: number_field(object, &["timestamp_ms", "timestamp"]),
        delay_ms: number_field(object, &["delay_ms", "delay"]),
        prompt_text,
        input_length,
        max_output_tokens,
    })
}

fn materialize_messages(store: &dyn SegmentStore, leaf: Handle) -> Result<Vec<OpenAiChatMessage>> {
    let segment = store
        .segment(leaf)
        .ok_or_else(|| anyhow!("unknown conversation segment {leaf}"))?;
    let message = match &segment.payload {
        Payload::Message { wire, .. } => serde_json::from_slice(wire)
            .with_context(|| format!("decoding conversation segment {leaf}"))?,
        Payload::Text { role, bytes, .. } => OpenAiChatMessage::new(
            role.as_str(),
            std::str::from_utf8(bytes)
                .with_context(|| format!("decoding conversation text {leaf}"))?,
        ),
        payload => bail!(
            "conversation segment {leaf} has unsupported {} payload",
            payload.kind_name()
        ),
    };
    Ok(vec![message])
}

/// Terminal response data needed to construct a continuation request.
#[derive(Clone, Debug)]
pub struct TurnResponse {
    /// Endpoint-normalized assistant text.
    pub text: String,
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

#[derive(Clone)]
struct LegacySessionBackend {
    metadata: ConversationMetadata,
    segment_ids: Vec<Handle>,
    segments: Rc<dyn SegmentStore>,
}

impl fmt::Debug for LegacySessionBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LegacySessionBackend")
            .field("turns", &self.metadata.turns.len())
            .finish()
    }
}

impl RuntimeSessionBackend for LegacySessionBackend {
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
        self.build_turn(owner, 0, num_turns, Vec::new(), None)
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
        self.build_turn(
            owner,
            current.turn_index + 1,
            current.num_turns,
            current.messages.clone(),
            Some(response.text),
        )
    }
}

impl LegacySessionBackend {
    fn build_turn(
        &self,
        owner: &SampledSession,
        turn_index: usize,
        num_turns: usize,
        mut messages: Vec<OpenAiChatMessage>,
        prior_reply: Option<String>,
    ) -> Result<TurnToSend> {
        let metadata = self
            .metadata
            .turns
            .get(turn_index)
            .ok_or_else(|| anyhow!("missing turn {turn_index} in {}", owner.conversation_id))?;
        if let Some(reply) = prior_reply.filter(|reply| !reply.is_empty()) {
            messages.push(OpenAiChatMessage::new("assistant", reply));
        }
        let segment_id = self
            .segment_ids
            .get(turn_index)
            .ok_or_else(|| anyhow!("missing segment for turn {turn_index}"))?;
        messages.extend(materialize_messages(self.segments.as_ref(), *segment_id)?);

        let assistant_tokens = messages
            .iter()
            .filter(|message| message.role == "assistant")
            .map(|message| message.content.split_whitespace().count().max(1))
            .sum::<usize>();
        let static_input_tokens = self.metadata.turns[..=turn_index]
            .iter()
            .map(|turn| turn.input_length)
            .sum::<usize>();

        Ok(TurnToSend {
            uuid: Uuid::new_v4(),
            conversation_id: owner.conversation_id.clone(),
            x_correlation_id: owner.x_correlation_id.clone(),
            request_correlation_id: owner.x_correlation_id.clone(),
            turn_index,
            num_turns,
            input_length: static_input_tokens + assistant_tokens,
            max_output_tokens: metadata.max_output_tokens,
            messages,
            request_body: None,
            request_headers: BTreeMap::new(),
            request_parameters: BTreeMap::new(),
            endpoint_path: None,
            streaming: true,
            audio_duration_seconds: None,
            timestamp_ms: metadata.timestamp_ms,
            delay_ms: metadata.delay_ms,
            cancel_after_ns: None,
            url_index: None,
            session: owner.clone(),
        })
    }
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

/// A turn awaiting issuance by a timing workload.
#[derive(Clone, Debug)]
pub struct TurnToSend {
    /// Stable request id used by measurement observers.
    pub uuid: Uuid,
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
    /// Whether this endpoint returns an SSE stream.
    pub streaming: bool,
    /// Audio duration propagated into ASR metrics.
    pub audio_duration_seconds: Option<f64>,
    /// Absolute trace timestamp for this turn, if any.
    pub timestamp_ms: Option<f64>,
    /// Relative delay for this turn, if any.
    pub delay_ms: Option<f64>,
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
    pub fn from_turn(id: u64, turn: &TurnToSend) -> Self {
        Self::from_issued_turn(id, turn, turn.url_index)
    }

    /// Build issued metadata while keeping the selector's turn-0 output
    /// distinct from the effective endpoint pin on `turn`.
    pub fn from_issued_turn(id: u64, turn: &TurnToSend, url_index: Option<u32>) -> Self {
        Self {
            id,
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

/// Sequential dataset-backed [`ConversationSource`]. Sampling wraps at the end
/// in stable loader order, matching Python's `SequentialSampler`.
pub struct DatasetConversationSource {
    dataset: Rc<ConversationDataset>,
    next_index: usize,
}

impl DatasetConversationSource {
    /// Create a source over `dataset`.
    pub fn new(dataset: ConversationDataset) -> Self {
        Self {
            dataset: Rc::new(dataset),
            next_index: 0,
        }
    }

    /// Create a source sharing an existing dataset allocation.
    pub fn from_shared(dataset: Rc<ConversationDataset>) -> Self {
        Self {
            dataset,
            next_index: 0,
        }
    }

    /// Shared dataset handle for constructing additional source views.
    pub fn dataset(&self) -> Rc<ConversationDataset> {
        self.dataset.clone()
    }
}

impl ConversationSource for DatasetConversationSource {
    fn conversations(&self) -> &[ConversationMetadata] {
        self.dataset.conversations()
    }

    fn next(&mut self, x_correlation_id: Option<String>) -> Result<SampledSession> {
        if self.dataset.conversations.is_empty() {
            bail!("conversation dataset cannot be empty");
        }
        if self.next_index >= self.dataset.conversations.len() {
            self.next_index = 0;
        }
        let conversation_id = self.dataset.conversations[self.next_index]
            .conversation_id
            .clone();
        self.next_index += 1;
        self.dataset.session(
            &conversation_id,
            x_correlation_id.unwrap_or_else(|| Uuid::new_v4().to_string()),
        )
    }

    fn session_for(
        &self,
        conversation_id: &str,
        x_correlation_id: String,
    ) -> Result<SampledSession> {
        self.dataset.session(conversation_id, x_correlation_id)
    }
}

#[derive(Clone)]
struct NativeSessionBackend {
    session: RefCell<NativeConversationSession>,
    metadata: ConversationMetadata,
    model_endpoint: ModelEndpoint,
    endpoint_resolver: Arc<dyn EndpointResolver>,
    materializer: Arc<dyn RequestMaterializer>,
    response_tokenizer: Arc<dyn TextTokenizer>,
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
            let mut session = self.session.borrow_mut();
            if session.should_capture_response() {
                let tokens = match response.completion_tokens {
                    Some(tokens) => tokens,
                    None => u64::try_from(self.response_tokenizer.count(&response.text)?)
                        .map_err(|_| anyhow!("assistant token count exceeds u64"))?,
                };
                session.capture_response(
                    EndpointTurn {
                        role: Some("assistant".into()),
                        texts: vec![EndpointMedia::new(vec![response.text])],
                        ..EndpointTurn::default()
                    },
                    tokens,
                )?;
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
        let endpoint = self.endpoint_resolver.resolve(endpoint_name.as_deref())?;
        let mut model_endpoint = self.model_endpoint.clone();
        let endpoint_metadata = endpoint.metadata();
        model_endpoint.endpoint.endpoint_type = endpoint_metadata.endpoint_type;
        model_endpoint.endpoint.streaming &= endpoint_metadata.supports_streaming;
        let materialized = self.materializer.materialize(
            &session,
            endpoint,
            &model_endpoint,
            CreditPhase::Profiling,
            &Overrides::new(),
        )?;
        let input_length = usize::try_from(materialized.input_tokens)
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
        let timing = self
            .metadata
            .turns
            .get(turn_index)
            .ok_or_else(|| anyhow!("missing native turn metadata {turn_index}"))?;
        Ok(TurnToSend {
            uuid: Uuid::new_v4(),
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
            streaming: materialized.streaming,
            audio_duration_seconds: materialized.audio_duration_seconds,
            timestamp_ms: timing.timestamp_ms,
            delay_ms: timing.delay_ms,
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
    sampler: Box<dyn Sampler>,
    model_endpoint: ModelEndpoint,
    endpoint_resolver: Arc<dyn EndpointResolver>,
    materializer: Arc<dyn RequestMaterializer>,
    response_tokenizer: Arc<dyn TextTokenizer>,
    default_output_tokens: usize,
}

impl NativeDatasetConversationSource {
    /// Construct the normal sequential source with all built endpoint adapters.
    pub fn sequential(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
    ) -> Result<Self> {
        let dataset = Arc::new(dataset);
        let sampler = SequentialSampler::from_metadata(&dataset.metadata().conversations)?;
        let endpoint = EndpointConfig {
            streaming: true,
            use_server_token_count: true,
            ..EndpointConfig::default()
        }
        .validate()?;
        Self::new(
            dataset,
            Box::new(sampler),
            ModelEndpoint {
                primary_model_name: model.into(),
                endpoint,
            },
            Arc::new(BuiltinEndpointResolver::default()),
            Arc::new(EndpointRequestMaterializer),
            Arc::new(TiktokenTokenizer::builtin()),
            default_output_tokens,
        )
    }

    /// Construct a source with injected sampler, endpoint registry,
    /// materializer, and response tokenizer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Arc<NativeDataset>,
        sampler: Box<dyn Sampler>,
        model_endpoint: ModelEndpoint,
        endpoint_resolver: Arc<dyn EndpointResolver>,
        materializer: Arc<dyn RequestMaterializer>,
        response_tokenizer: Arc<dyn TextTokenizer>,
        default_output_tokens: usize,
    ) -> Result<Self> {
        if default_output_tokens == 0 {
            bail!("native dataset default output tokens must be positive");
        }
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
            .collect();
        Ok(Self {
            dataset,
            metadata,
            sampler,
            model_endpoint,
            endpoint_resolver,
            materializer,
            response_tokenizer,
            default_output_tokens,
        })
    }

    fn session(
        &self,
        conversation_id: &str,
        correlation_id: Option<String>,
    ) -> Result<SampledSession> {
        let id = aiperf_dataset::SessionId::from(conversation_id);
        self.dataset.get(&id)?;
        let metadata = self
            .metadata
            .iter()
            .find(|metadata| metadata.conversation_id == conversation_id)
            .cloned()
            .ok_or_else(|| {
                anyhow!("native dataset session {conversation_id:?} is not sampleable")
            })?;
        let x_correlation_id = correlation_id.unwrap_or_else(|| Uuid::new_v4().to_string());
        let backend = NativeSessionBackend {
            session: RefCell::new(NativeConversationSession::new(self.dataset.clone(), id)?),
            metadata,
            model_endpoint: self.model_endpoint.clone(),
            endpoint_resolver: self.endpoint_resolver.clone(),
            materializer: self.materializer.clone(),
            response_tokenizer: self.response_tokenizer.clone(),
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

/// Synthetic conversation source used when the CLI has no dataset file.
pub struct SyntheticConversationSource {
    inner: DatasetConversationSource,
}

impl SyntheticConversationSource {
    /// Create one fixed K-turn template from the current online workload knobs.
    pub fn new(workload: SkeletonWorkload) -> Result<Self> {
        let turns = (0..workload.turns.max(1))
            .map(|turn_index| TurnMetadata {
                timestamp_ms: None,
                delay_ms: (turn_index > 0)
                    .then_some(workload.think_time_ms.unwrap_or_default() as f64),
                prompt_text: format!(
                    "turn {turn_index}: {}",
                    vec!["lorem"; workload.input_tokens].join(" ")
                ),
                input_length: workload.input_tokens,
                max_output_tokens: workload.output_tokens,
            })
            .collect();
        let dataset = ConversationDataset::new(vec![ConversationMetadata {
            conversation_id: "synthetic".to_string(),
            turns,
        }])?;
        Ok(Self {
            inner: DatasetConversationSource::new(dataset),
        })
    }
}

impl ConversationSource for SyntheticConversationSource {
    fn conversations(&self) -> &[ConversationMetadata] {
        self.inner.conversations()
    }

    fn next(&mut self, x_correlation_id: Option<String>) -> Result<SampledSession> {
        self.inner.next(x_correlation_id)
    }

    fn session_for(
        &self,
        conversation_id: &str,
        x_correlation_id: String,
    ) -> Result<SampledSession> {
        self.inner.session_for(conversation_id, x_correlation_id)
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
    use aiperf_dataset::{ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry};
    use aiperf_rng::RngRoot;
    use serde_json::json;

    use super::*;

    fn workload(turns: usize) -> SkeletonWorkload {
        SkeletonWorkload {
            num_requests: 0,
            input_tokens: 4,
            output_tokens: 2,
            turns,
            think_time_ms: Some(7),
        }
    }

    #[test]
    fn synthetic_source_reuses_template_but_mints_session_ids() {
        let mut source = SyntheticConversationSource::new(workload(3)).unwrap();
        let a = source.next(None).unwrap().build_first_turn(None).unwrap();
        let b = source.next(None).unwrap().build_first_turn(None).unwrap();
        assert_eq!(a.conversation_id, "synthetic");
        assert_eq!(b.conversation_id, "synthetic");
        assert_ne!(a.x_correlation_id, b.x_correlation_id);
        assert_eq!(a.turn_index, 0);
        assert_eq!(a.num_turns, 3);
        assert_eq!(a.delay_ms, None);
        assert_eq!(a.input_length, 4);
        assert_eq!(a.max_output_tokens, 2);
        assert_eq!(a.messages.len(), 1);
    }

    #[test]
    fn continuation_splices_real_reply_and_carries_timing() {
        let mut source = SyntheticConversationSource::new(workload(3)).unwrap();
        let first = source.next(None).unwrap().build_first_turn(None).unwrap();
        let credit = IssuedCredit::from_turn(0, &first);
        let next = source
            .next_turn(
                &credit,
                TurnResponse {
                    text: "server reply".to_string(),
                    completion_tokens: None,
                    terminal: ReplayTerminalStatus::Completed,
                },
            )
            .unwrap()
            .unwrap();
        assert_eq!(next.x_correlation_id, first.x_correlation_id);
        assert_eq!(next.turn_index, 1);
        assert_eq!(next.num_turns, 3);
        assert_eq!(next.delay_ms, Some(7.0));
        assert_eq!(
            next.messages
                .iter()
                .map(|message| message.role.as_str())
                .collect::<Vec<_>>(),
            vec!["user", "assistant", "user"]
        );
        assert_eq!(next.messages[1].content, "server reply");
    }

    #[test]
    fn virtual_history_cap_clamps_to_sampled_length() {
        let mut source = SyntheticConversationSource::new(workload(2)).unwrap();
        let session = source.next(Some("u-1".to_string())).unwrap();
        let first = session.build_first_turn(Some(99)).unwrap();
        assert_eq!(first.num_turns, 2);
        assert_eq!(first.x_correlation_id, "u-1");
    }

    #[test]
    fn jsonl_loader_groups_sessions_and_accepts_trace_aliases() {
        let input = r#"
{"session_id":"a","timestamp":1000,"text_input":"hello there","output_length":2}
{"session_id":"a","delay":25,"input_length":3,"max_tokens":4}
{"timestamp":1050,"input_length":2,"output_length":1}
"#;
        let dataset = ConversationDataset::from_json_or_jsonl(input, 8, 6).unwrap();
        assert_eq!(dataset.conversations().len(), 2);
        assert_eq!(dataset.conversations()[0].turns.len(), 2);
        assert_eq!(
            dataset.conversations()[0].turns[0].timestamp_ms,
            Some(1000.0)
        );
        assert_eq!(dataset.conversations()[0].turns[1].delay_ms, Some(25.0));
        assert_eq!(dataset.conversations()[1].conversation_id, "trace-row-2");
    }

    #[test]
    fn first_turn_window_rebuilds_only_in_range_conversations() {
        let dataset = ConversationDataset::from_json_or_jsonl(
            concat!(
                "{\"timestamp\":1000,\"input_length\":2}\n",
                "{\"timestamp\":2000,\"input_length\":2}\n",
                "{\"timestamp\":3000,\"input_length\":2}\n"
            ),
            2,
            1,
        )
        .unwrap();
        let filtered = dataset
            .filter_first_turn_window(Some(1500.0), Some(2500.0))
            .unwrap();
        assert_eq!(filtered.conversations().len(), 1);
        assert_eq!(
            filtered.conversations()[0].turns[0].timestamp_ms,
            Some(2000.0)
        );
    }

    #[test]
    fn counter_matches_python_root_counting_rules() {
        let mut source = SyntheticConversationSource::new(workload(2)).unwrap();
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
                &IssuedCredit::from_turn(id0, &first),
                TurnResponse {
                    text: String::new(),
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
                &IssuedCredit::from_turn(2, &second),
                TurnResponse {
                    text: String::new(),
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
        let mut source = NativeDatasetConversationSource::sequential(dataset, "model", 8).unwrap();
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
                &IssuedCredit::from_turn(0, &first),
                TurnResponse {
                    text: "live answer".into(),
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
        let mut source = NativeDatasetConversationSource::sequential(dataset, "model", 4).unwrap();
        let turn = source.next(None).unwrap().build_first_turn(None).unwrap();
        assert_eq!(turn.request_body.unwrap(), authored);
        assert!(turn.streaming);
    }
}
