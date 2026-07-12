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
    Payload, RequestMaterializer, Sampler, SamplerRegistry, SegmentPool, SegmentStore,
    SequentialSampler, TextTokenizer, TiktokenTokenizer,
};
use aiperf_endpoints::{
    ChatEndpoint, CreditPhase, Endpoint, EndpointConfig, EndpointId, EndpointKey, EndpointType,
    Media as EndpointMedia, ModelEndpoint, PreparedEndpoint, PreparedEndpointTable,
    Turn as EndpointTurn,
};
use aiperf_graph::segment::intern_message;
use aiperf_graph::wire::OpenAiChatMessage;
use aiperf_rng::RngRoot;
use aiperf_timing::{RunState, StopConfig};
use anyhow::{Context, Result, anyhow, bail};
use bytes::Bytes;
use loadgen_core::collector::ReplayTerminalStatus;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::workload::SkeletonWorkload;

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
/// This is the native counterpart of
/// `src/aiperf/records/inference_result_parser.py:320-428`: endpoint-specific
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
        extracted: aiperf_endpoints::ExtractedPayload,
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
        trace_hash_ids: None,
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
            effective_model: None,
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
            endpoint: TurnEndpoint::Legacy(Arc::new(LegacyTurnEndpointBinding {
                endpoint: Arc::new(ChatEndpoint),
                config: EndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    ..EndpointConfig::default()
                },
            })),
            streaming: true,
            audio_duration_seconds: None,
            timestamp_ms: metadata.timestamp_ms,
            delay_ms: metadata.delay_ms,
            trace_hash_ids: None,
            raw_token_ids: None,
            data_policy: TurnDataPolicy::ordinary(),
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

/// Endpoint selection retained by one schedulable turn.
#[derive(Clone)]
pub enum TurnEndpoint {
    /// Protocol-v1 compatibility adapter and closed configuration.
    Legacy(Arc<LegacyTurnEndpointBinding>),
    /// Protocol-v2 open prepared binding selected only by stable key and ID.
    Prepared(PreparedEndpointReference),
}

/// Shared legacy adapter/configuration retained by compatibility turns.
///
/// Indirection keeps [`TurnEndpoint`] cheap to clone without boxing or copying
/// the comparatively large closed [`EndpointConfig`] on the scheduler path.
pub struct LegacyTurnEndpointBinding {
    /// Stateless legacy endpoint implementation.
    pub endpoint: Arc<dyn Endpoint>,
    /// Effective closed compatibility configuration.
    pub config: EndpointConfig,
}

impl fmt::Debug for LegacyTurnEndpointBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LegacyTurnEndpointBinding")
            .field("endpoint", &self.endpoint.metadata().endpoint_type)
            .field("config", &self.config)
            .finish()
    }
}

impl fmt::Debug for TurnEndpoint {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Legacy(binding) => fmt::Debug::fmt(binding, formatter),
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
enum NativeSessionEndpoint {
    Legacy(Arc<LegacyNativeSessionEndpoint>),
    Prepared {
        primary_model_name: String,
        endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    },
}

struct LegacyNativeSessionEndpoint {
    model_endpoint: ModelEndpoint,
    endpoint_resolver: Arc<dyn EndpointResolver>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum StaticInputCountEndpoint {
    Legacy(EndpointType),
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
            Self::Legacy(legacy) => formatter
                .debug_struct("LegacySessionEndpoint")
                .field(
                    "endpoint_type",
                    &legacy.model_endpoint.endpoint.endpoint_type,
                )
                .finish_non_exhaustive(),
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
            let mut session = self.session.borrow_mut();
            if session.should_capture_response() {
                let tokens = match response.completion_tokens {
                    Some(tokens) => tokens,
                    None => u64::try_from(self.response_tokenizer.count(&response.text)?)
                        .map_err(|_| anyhow!("assistant token count exceeds u64"))?,
                };
                session.capture_response(
                    response.assistant_message.map_or_else(
                        || EndpointTurn {
                            role: Some("assistant".into()),
                            texts: vec![EndpointMedia::new(vec![response.text])],
                            ..EndpointTurn::default()
                        },
                        |message| EndpointTurn {
                            raw_messages: Some(vec![message]),
                            ..EndpointTurn::default()
                        },
                    ),
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
        let (materialized, turn_endpoint, prepared_endpoint) = match &self.endpoint {
            NativeSessionEndpoint::Legacy(legacy) => {
                let endpoint = match endpoint_name.as_deref() {
                    Some(name) => legacy.endpoint_resolver.resolve(Some(name))?,
                    None => legacy
                        .endpoint_resolver
                        .resolve_type(legacy.model_endpoint.endpoint.endpoint_type)?,
                };
                let mut effective_model_endpoint = legacy.model_endpoint.clone();
                let endpoint_metadata = endpoint.metadata();
                effective_model_endpoint.endpoint.endpoint_type = endpoint_metadata.endpoint_type;
                effective_model_endpoint.endpoint.streaming &= endpoint_metadata.supports_streaming;
                let materialized = self.materializer.materialize(
                    &session,
                    endpoint.as_ref(),
                    &effective_model_endpoint,
                    CreditPhase::Profiling,
                    &Overrides::new(),
                )?;
                let turn_endpoint = TurnEndpoint::Legacy(Arc::new(LegacyTurnEndpointBinding {
                    endpoint,
                    config: effective_model_endpoint.endpoint,
                }));
                (materialized, turn_endpoint, None)
            }
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
                    Some((reference, selected.endpoint)),
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
                TurnEndpoint::Legacy(binding) => {
                    StaticInputCountEndpoint::Legacy(binding.endpoint.metadata().endpoint_type)
                }
                TurnEndpoint::Prepared(reference) => {
                    StaticInputCountEndpoint::Prepared(reference.key)
                }
            },
        });
        let input_tokens = if timing.trace_hash_ids.is_some()
            || materialized.raw_token_ids.is_some()
        {
            u64::try_from(timing.input_length)
                .map_err(|_| anyhow!("authored trace input count exceeds u64"))?
        } else if let Some(cached) = static_count_key
            .and_then(|key| self.static_input_count_cache.borrow().get(&key).copied())
        {
            cached
        } else {
            let counted = match &prepared_endpoint {
                Some((_, endpoint)) => self.input_token_counter.count_prepared_input_tokens(
                    *endpoint,
                    &materialized.body,
                    materialized.input_tokens,
                )?,
                None => match &turn_endpoint {
                    TurnEndpoint::Legacy(binding) => self.input_token_counter.count_input_tokens(
                        binding.endpoint.as_ref(),
                        &materialized.body,
                        materialized.input_tokens,
                    )?,
                    TurnEndpoint::Prepared(_) => {
                        unreachable!("prepared endpoint retained above for token counting")
                    }
                },
            };
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
    /// Construct a source that honors the loader's preferred sampler strategy.
    pub fn preferred(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        rng_root: RngRoot,
    ) -> Result<Self> {
        let endpoint = EndpointConfig {
            streaming: true,
            use_server_token_count: true,
            ..EndpointConfig::default()
        };
        Self::preferred_with_endpoint_config(
            dataset,
            model,
            default_output_tokens,
            rng_root,
            endpoint,
        )
    }

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
                primary_model_name: model.into(),
                endpoint_resolver,
            },
            Arc::new(EndpointRequestMaterializer),
            Arc::new(TiktokenTokenizer::builtin()),
            default_output_tokens,
        )
    }

    /// Honor loader sampling policy with caller-supplied compile-time registries.
    pub fn preferred_with_registries(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        rng_root: RngRoot,
        samplers: &SamplerRegistry,
        endpoint_resolver: Arc<dyn EndpointResolver>,
    ) -> Result<Self> {
        let endpoint = EndpointConfig {
            streaming: true,
            use_server_token_count: true,
            ..EndpointConfig::default()
        };
        Self::preferred_with_endpoint_config_and_registries(
            dataset,
            model,
            default_output_tokens,
            rng_root,
            endpoint,
            samplers,
            endpoint_resolver,
        )
    }

    /// Honor loader sampling policy with caller-selected endpoint configuration.
    pub fn preferred_with_endpoint_config(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        rng_root: RngRoot,
        endpoint: EndpointConfig,
    ) -> Result<Self> {
        let samplers = SamplerRegistry::with_builtin_strategies()?;
        Self::preferred_with_endpoint_config_and_registries(
            dataset,
            model,
            default_output_tokens,
            rng_root,
            endpoint,
            &samplers,
            Arc::new(BuiltinEndpointResolver::default()),
        )
    }

    /// Honor loader sampling and endpoint policy from caller-supplied registries.
    #[allow(clippy::too_many_arguments)]
    pub fn preferred_with_endpoint_config_and_registries(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        rng_root: RngRoot,
        endpoint: EndpointConfig,
        samplers: &SamplerRegistry,
        endpoint_resolver: Arc<dyn EndpointResolver>,
    ) -> Result<Self> {
        let dataset = Arc::new(dataset);
        let sampler = samplers.create(
            &dataset.metadata().sampling_strategy,
            &dataset.metadata().conversations,
            rng_root,
        )?;
        let endpoint = endpoint.validate()?;
        Self::new(
            dataset,
            sampler,
            ModelEndpoint {
                primary_model_name: model.into(),
                endpoint,
            },
            endpoint_resolver,
            Arc::new(EndpointRequestMaterializer),
            Arc::new(TiktokenTokenizer::builtin()),
            default_output_tokens,
        )
    }

    /// Construct the normal sequential source with all built endpoint adapters.
    pub fn sequential(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
    ) -> Result<Self> {
        let endpoint = EndpointConfig {
            streaming: true,
            use_server_token_count: true,
            ..EndpointConfig::default()
        };
        Self::sequential_with_endpoint_config(dataset, model, default_output_tokens, endpoint)
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
        let dataset = Arc::new(dataset);
        let sampler = SequentialSampler::from_metadata(&dataset.metadata().conversations)?;
        Self::new_with_endpoint(
            dataset,
            Box::new(sampler),
            NativeSessionEndpoint::Prepared {
                primary_model_name: model.into(),
                endpoint_resolver,
            },
            Arc::new(EndpointRequestMaterializer),
            Arc::new(TiktokenTokenizer::builtin()),
            default_output_tokens,
        )
    }

    /// Construct a sequential source with caller-selected endpoint policy.
    ///
    /// Endpoint configuration is part of the ordinary dataset pipeline; callers
    /// use this for compatibility flags such as legacy chat `max_tokens` without
    /// rebuilding or intercepting HTTP requests.
    pub fn sequential_with_endpoint_config(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        endpoint: EndpointConfig,
    ) -> Result<Self> {
        Self::sequential_with_endpoint_config_and_resolver(
            dataset,
            model,
            default_output_tokens,
            endpoint,
            Arc::new(BuiltinEndpointResolver::default()),
        )
    }

    /// Construct a sequential source with injected endpoint registration.
    pub fn sequential_with_endpoint_config_and_resolver(
        dataset: NativeDataset,
        model: impl Into<String>,
        default_output_tokens: usize,
        endpoint: EndpointConfig,
        endpoint_resolver: Arc<dyn EndpointResolver>,
    ) -> Result<Self> {
        let dataset = Arc::new(dataset);
        let sampler = SequentialSampler::from_metadata(&dataset.metadata().conversations)?;
        let endpoint = endpoint.validate()?;
        Self::new(
            dataset,
            Box::new(sampler),
            ModelEndpoint {
                primary_model_name: model.into(),
                endpoint,
            },
            endpoint_resolver,
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
        Self::new_with_endpoint(
            dataset,
            sampler,
            NativeSessionEndpoint::Legacy(Arc::new(LegacyNativeSessionEndpoint {
                model_endpoint,
                endpoint_resolver,
            })),
            materializer,
            response_tokenizer,
            default_output_tokens,
        )
    }

    fn new_with_endpoint(
        dataset: Arc<NativeDataset>,
        sampler: Box<dyn Sampler>,
        endpoint: NativeSessionEndpoint,
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
        let id = aiperf_dataset::SessionId::from(conversation_id);
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
                trace_hash_ids: None,
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aiperf_dataset::{ComposeConfig, DatasetSource, LoadConfig, LoaderRegistry};
    use aiperf_rng::RngRoot;
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
        fn encode(&self, text: &str) -> aiperf_dataset::Result<Vec<u32>> {
            Ok((0..text.split_whitespace().count() as u32).collect())
        }

        fn decode(&self, _token_ids: &[u32]) -> aiperf_dataset::Result<String> {
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
        ) -> aiperf_dataset::Result<Option<Vec<u32>>> {
            Ok(Some(vec![0; 5]))
        }
    }

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
        let credit = IssuedCredit::from_turn(0, 0, &first);
        let next = source
            .next_turn(
                &credit,
                TurnResponse {
                    text: "server reply".to_string(),
                    assistant_message: None,
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
        let mut source =
            NativeDatasetConversationSource::sequential_with_endpoint_config_and_resolver(
                dataset,
                "claude",
                4,
                EndpointConfig {
                    endpoint_type: aiperf_endpoints::EndpointType::Messages,
                    streaming: true,
                    ..EndpointConfig::default()
                },
                Arc::new(BuiltinEndpointResolver::default()),
            )
            .unwrap();
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
        let mut source = NativeDatasetConversationSource::sequential(dataset, "model", 4).unwrap();
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
        let registry = aiperf_endpoints::EndpointRegistry::builtin().unwrap();
        let endpoint_id = EndpointId::new("chat").unwrap();
        let endpoint = registry
            .prepare(
                &endpoint_id,
                aiperf_endpoints::RawEndpointConfig {
                    streaming: true,
                    use_server_token_count: true,
                    ..aiperf_endpoints::RawEndpointConfig::default()
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

        let TurnEndpoint::Prepared(reference) = turn.endpoint else {
            panic!("prepared source constructed a legacy endpoint turn")
        };
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
        let error =
            NativeDatasetConversationSource::preferred(dataset, "model", 4, RngRoot::new(Some(1)))
                .err()
                .unwrap();
        assert!(error.to_string().contains("unknown sampler strategy"));
    }
}
