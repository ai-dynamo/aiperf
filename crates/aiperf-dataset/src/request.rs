// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful conversation reconstruction and endpoint request materialization.
//!
//! The history rules are ported from
//! `src/aiperf/workers/session_manager.py:85-122`; raw payload bypass and endpoint
//! formatting are ported from `src/aiperf/workers/inference_client.py:114-126`;
//! current-turn header precedence follows
//! `src/aiperf/transports/base_transports.py:113-127`. The Python-reserved
//! `message_array_without_responses` case is completed here by prefix-diffing
//! successive authored snapshots and interleaving each captured live reply.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use aiperf_endpoints::{
    ChatEmbeddingsEndpoint, ChatEndpoint, CohereRankingsEndpoint, CompletionsEndpoint, CreditPhase,
    EmbeddingsEndpoint, Endpoint, HfTeiRankingsEndpoint, HuggingFaceGenerateEndpoint,
    ImageEditEndpoint, ImageGenerationEndpoint, ImageRetrievalEndpoint, Media, ModelEndpoint,
    NimEmbeddingsEndpoint, NimRankingsEndpoint, RawEndpoint, RequestInfo, ResponsesEndpoint,
    SolidoRagEndpoint, TemplateEndpoint, Turn as EndpointTurn, VideoGenerationEndpoint,
};
use bytes::Bytes;
use serde_json::{Map, Value};

use crate::dataset::Dataset;
use crate::error::{DatasetError, Result};
use crate::materialize::Overrides;
use crate::model::{
    AccuracyAssociation, Conversation, ConversationContextMode, MediaKind, SessionId, Turn,
};
use crate::segment::{Handle, Payload, SegmentStore};

/// One fully built dispatch request and its media-free accounting metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedRequest {
    /// Serialized request bytes. Raw payloads retain their authored bytes exactly
    /// unless explicit dispatch overrides were supplied.
    pub body: Bytes,
    /// Per-turn headers, after validating every value as a string.
    pub headers: BTreeMap<String, String>,
    /// Per-turn URL query parameters.
    pub parameters: BTreeMap<String, String>,
    /// Authored endpoint/dialect override, when present.
    pub endpoint: Option<String>,
    /// Resolved endpoint path after endpoint configuration and capability metadata.
    pub endpoint_path: Option<String>,
    /// Effective model name.
    pub model: String,
    /// Requested output-token cap.
    pub max_tokens: Option<u32>,
    /// Effective response streaming mode.
    pub streaming: bool,
    /// Precomputed input-token count including selected history and captured replies.
    pub input_tokens: u64,
    /// Audio duration used by ASR metrics.
    pub audio_duration_seconds: Option<f64>,
    /// Opaque evaluator association propagated without positional matching.
    pub accuracy: Option<AccuracyAssociation>,
    /// Zero-based authored turn index.
    pub turn_index: usize,
    /// Whether this is the final authored turn.
    pub is_final_turn: bool,
}

/// Endpoint-independent request-materialization extension point.
pub trait RequestMaterializer: Send + Sync {
    /// Build the current session turn through `endpoint` and apply explicit
    /// dispatch overrides after authored/endpoint fields.
    fn materialize(
        &self,
        session: &ConversationSession,
        endpoint: &dyn Endpoint,
        model_endpoint: &ModelEndpoint,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest>;
}

/// Lookup seam for authored per-turn endpoint/dialect overrides.
pub trait EndpointResolver: Send + Sync {
    /// Resolve an optional authored endpoint name, falling back to the registry default.
    fn resolve(&self, name: Option<&str>) -> Result<Arc<dyn Endpoint>>;
}

/// Extensible name-to-endpoint registry containing the endpoint implementations
/// currently built by `aiperf-endpoints` plus statically linked extensions.
#[derive(Clone)]
pub struct BuiltinEndpointResolver {
    default_name: String,
    endpoints: HashMap<String, Arc<dyn Endpoint>>,
}

impl std::fmt::Debug for BuiltinEndpointResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut names = self.endpoints.keys().collect::<Vec<_>>();
        names.sort();
        f.debug_struct("BuiltinEndpointResolver")
            .field("default_name", &self.default_name)
            .field("names", &names)
            .finish()
    }
}

impl BuiltinEndpointResolver {
    /// Construct the registry with Chat Completions as its default.
    pub fn chat_default() -> Self {
        let mut resolver = Self {
            default_name: "chat".into(),
            endpoints: HashMap::new(),
        };
        resolver
            .register("chat", ChatEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("chat_completions", ChatEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("completions", CompletionsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("responses", ResponsesEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("embeddings", EmbeddingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("chat_embeddings", ChatEmbeddingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("nim_embeddings", NimEmbeddingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("cohere_rankings", CohereRankingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("hf_tei_rankings", HfTeiRankingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("nim_rankings", NimRankingsEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("huggingface_generate", HuggingFaceGenerateEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("image_generation", ImageGenerationEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("image_edit", ImageEditEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("video_generation", VideoGenerationEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("image_retrieval", ImageRetrievalEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("solido_rag", SolidoRagEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("raw", RawEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
            .register("template", TemplateEndpoint)
            .expect("built-in endpoint names are unique");
        resolver
    }

    /// Select the fallback endpoint name after verifying it is registered.
    pub fn with_default(mut self, name: &str) -> Result<Self> {
        self.set_default(name)?;
        Ok(self)
    }

    /// Change the fallback endpoint after verifying it is registered.
    pub fn set_default(&mut self, name: &str) -> Result<()> {
        let normalized = normalize_endpoint_name(name);
        if !self.endpoints.contains_key(&normalized) {
            return Err(DatasetError::Validation(format!(
                "default endpoint {name:?} is not registered"
            )));
        }
        self.default_name = normalized;
        Ok(())
    }

    /// Register one endpoint implementation, rejecting duplicate normalized names.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        endpoint: impl Endpoint + 'static,
    ) -> Result<()> {
        let authored = name.into();
        let normalized = normalize_endpoint_name(&authored);
        if normalized.is_empty() {
            return Err(DatasetError::Validation(
                "endpoint registration name cannot be empty".into(),
            ));
        }
        if self.endpoints.contains_key(&normalized) {
            return Err(DatasetError::Validation(format!(
                "duplicate endpoint registration {authored:?}"
            )));
        }
        self.endpoints.insert(normalized, Arc::new(endpoint));
        Ok(())
    }
}

impl Default for BuiltinEndpointResolver {
    fn default() -> Self {
        Self::chat_default()
    }
}

impl EndpointResolver for BuiltinEndpointResolver {
    fn resolve(&self, name: Option<&str>) -> Result<Arc<dyn Endpoint>> {
        let normalized = name
            .map(normalize_endpoint_name)
            .unwrap_or_else(|| self.default_name.clone());
        self.endpoints.get(&normalized).cloned().ok_or_else(|| {
            let mut available = self.endpoints.keys().cloned().collect::<Vec<_>>();
            available.sort();
            DatasetError::Validation(format!(
                "unknown dataset endpoint {normalized:?}; registered endpoints: {}",
                available.join(", ")
            ))
        })
    }
}

fn normalize_endpoint_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', '/'], "_")
}

/// Native materializer backed by `aiperf-endpoints` formatters.
#[derive(Debug, Clone, Copy, Default)]
pub struct EndpointRequestMaterializer;

impl RequestMaterializer for EndpointRequestMaterializer {
    fn materialize(
        &self,
        session: &ConversationSession,
        endpoint: &dyn Endpoint,
        model_endpoint: &ModelEndpoint,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        let (conversation, current, turn_index) = session.current()?;
        let store = session.dataset.segments().as_ref();
        let (body, effective) = if let Some(raw) = current.raw_payload {
            (
                store.build_body(&[raw], overrides)?,
                EffectiveRequest {
                    model: effective_model(current, model_endpoint, overrides)?,
                    max_tokens: effective_max_tokens(current, overrides)?,
                    streaming: effective_streaming(current, model_endpoint, endpoint, overrides)?,
                },
            )
        } else {
            let streaming = effective_streaming(current, model_endpoint, endpoint, overrides)?;
            let mut effective_model_endpoint = model_endpoint.clone();
            effective_model_endpoint.endpoint.streaming = streaming;
            let turns = session.endpoint_turns(store)?;
            let request_info = RequestInfo {
                model_endpoint: effective_model_endpoint,
                turns,
                system_message: resolve_prompt(store, conversation.system)?,
                user_context_message: resolve_prompt(store, conversation.user_context)?,
                credit_phase: phase,
                x_request_id: None,
                x_correlation_id: None,
                conversation_id: Some(session.conversation_id().as_str().to_string()),
            };
            let mut value = endpoint.format_payload(&request_info)?;
            merge_overrides(&mut value, overrides)?;
            let effective = effective_from_structured_body(
                &mut value,
                current,
                model_endpoint,
                endpoint,
                overrides,
            )?;
            (Bytes::from(serde_json::to_vec(&value)?), effective)
        };

        let endpoint_path = model_endpoint.endpoint.path.clone().or_else(|| {
            if effective.streaming {
                endpoint.metadata().streaming_path
            } else {
                None
            }
            .or(endpoint.metadata().endpoint_path)
            .map(str::to_string)
        });
        Ok(MaterializedRequest {
            body,
            headers: raw_string_map(store, current.extra_headers, "extra_headers")?,
            parameters: raw_string_map(store, current.request_parameters, "request_parameters")?,
            endpoint: current.endpoint.clone(),
            endpoint_path,
            model: effective.model,
            max_tokens: effective.max_tokens,
            streaming: effective.streaming,
            input_tokens: session.input_tokens(store)?,
            audio_duration_seconds: current.audio_duration_seconds,
            accuracy: conversation.accuracy.clone(),
            turn_index,
            is_final_turn: turn_index + 1 == conversation.turns.len(),
        })
    }
}

struct EffectiveRequest {
    model: String,
    max_tokens: Option<u32>,
    streaming: bool,
}

fn effective_from_structured_body(
    value: &mut Value,
    turn: &Turn,
    model_endpoint: &ModelEndpoint,
    endpoint: &dyn Endpoint,
    overrides: &Overrides,
) -> Result<EffectiveRequest> {
    let object = value.as_object_mut().ok_or_else(|| {
        DatasetError::Validation("endpoint formatter returned a non-object body".into())
    })?;
    let model = match object.get("model") {
        Some(Value::String(model)) => model.clone(),
        Some(_) => {
            return Err(DatasetError::Validation(
                "effective request model must be a string".into(),
            ));
        }
        None => effective_model(turn, model_endpoint, overrides)?,
    };
    let mut max_tokens = effective_max_tokens(turn, overrides)?;
    for (field, value) in object.iter().filter(|(field, _)| {
        matches!(
            field.as_str(),
            "max_tokens" | "max_completion_tokens" | "max_output_tokens"
        )
    }) {
        max_tokens = Some(positive_u32(value, field)?);
    }
    let requested_streaming = match object.get("stream") {
        Some(Value::Bool(streaming)) => *streaming,
        Some(_) => {
            return Err(DatasetError::Validation(
                "effective request stream must be boolean".into(),
            ));
        }
        None => effective_streaming(turn, model_endpoint, endpoint, overrides)?,
    };
    let streaming = requested_streaming && endpoint.metadata().supports_streaming;
    if requested_streaming != streaming {
        object.insert("stream".into(), Value::Bool(streaming));
    }
    Ok(EffectiveRequest {
        model,
        max_tokens,
        streaming,
    })
}

fn effective_model(
    turn: &Turn,
    model_endpoint: &ModelEndpoint,
    overrides: &Overrides,
) -> Result<String> {
    match overrides.fields().get("model") {
        Some(Value::String(model)) => Ok(model.clone()),
        Some(_) => Err(DatasetError::Validation(
            "request override model must be a string".into(),
        )),
        None => Ok(turn
            .model
            .as_ref()
            .map(|model| model.as_str().to_string())
            .unwrap_or_else(|| model_endpoint.primary_model_name.clone())),
    }
}

fn effective_max_tokens(turn: &Turn, overrides: &Overrides) -> Result<Option<u32>> {
    let mut effective = turn.max_tokens;
    for field in ["max_tokens", "max_completion_tokens", "max_output_tokens"] {
        let Some(value) = overrides.fields().get(field) else {
            continue;
        };
        effective = Some(positive_u32(value, field)?);
    }
    Ok(effective)
}

fn positive_u32(value: &Value, field: &str) -> Result<u32> {
    value
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            DatasetError::Validation(format!("effective request {field} must be a positive u32"))
        })
}

fn effective_streaming(
    turn: &Turn,
    model_endpoint: &ModelEndpoint,
    endpoint: &dyn Endpoint,
    overrides: &Overrides,
) -> Result<bool> {
    let requested = match overrides.fields().get("stream") {
        Some(Value::Bool(streaming)) => *streaming,
        Some(_) => {
            return Err(DatasetError::Validation(
                "request override stream must be boolean".into(),
            ));
        }
        None => turn.streaming.unwrap_or(model_endpoint.endpoint.streaming),
    };
    Ok(requested && endpoint.metadata().supports_streaming)
}

#[derive(Debug, Clone, PartialEq)]
struct CapturedReply {
    after_turn: usize,
    turn: EndpointTurn,
    tokens: u64,
}

/// Per-runtime-session state over one immutable, reusable dataset conversation.
///
/// The session keeps only small dynamic assistant turns. Authored request bytes
/// remain in the dataset's shared segment store and are resolved on dispatch.
#[derive(Clone)]
pub struct ConversationSession {
    dataset: Arc<Dataset>,
    conversation_id: SessionId,
    context_mode: ConversationContextMode,
    current_turn: Option<usize>,
    replies: Vec<CapturedReply>,
}

impl std::fmt::Debug for ConversationSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConversationSession")
            .field("conversation_id", &self.conversation_id)
            .field("context_mode", &self.context_mode)
            .field("current_turn", &self.current_turn)
            .field("captured_replies", &self.replies.len())
            .finish()
    }
}

impl ConversationSession {
    /// Bind a new runtime session to one authored conversation.
    pub fn new(dataset: Arc<Dataset>, conversation_id: SessionId) -> Result<Self> {
        let conversation = dataset.get(&conversation_id)?;
        let context_mode = dataset.context_mode(conversation);
        Ok(Self {
            dataset,
            conversation_id,
            context_mode,
            current_turn: None,
            replies: Vec::new(),
        })
    }

    /// Authored conversation identifier.
    pub fn conversation_id(&self) -> &SessionId {
        &self.conversation_id
    }

    /// Effective conversation context mode.
    pub const fn context_mode(&self) -> ConversationContextMode {
        self.context_mode
    }

    /// Authored endpoint override on the current turn, if any.
    pub fn endpoint_override(&self) -> Result<Option<&str>> {
        let (_, turn, _) = self.current()?;
        Ok(turn.endpoint.as_deref())
    }

    /// Advance sequentially to the requested authored turn.
    pub fn advance_to(&mut self, turn_index: usize) -> Result<&Turn> {
        let conversation = self.dataset.get(&self.conversation_id)?;
        let expected = self.current_turn.map_or(0, |current| current + 1);
        if turn_index != expected {
            return Err(DatasetError::Validation(format!(
                "conversation {:?} expected turn {expected}, got {turn_index}",
                self.conversation_id.as_str()
            )));
        }
        let turn = conversation.turns.get(turn_index).ok_or_else(|| {
            DatasetError::Validation(format!(
                "turn {turn_index} is out of range for conversation {:?} with {} turns",
                self.conversation_id.as_str(),
                conversation.turns.len()
            ))
        })?;
        self.current_turn = Some(turn_index);
        Ok(turn)
    }

    /// Whether a successful response must be retained for later context.
    pub const fn should_capture_response(&self) -> bool {
        matches!(
            self.context_mode,
            ConversationContextMode::DeltasWithoutResponses
                | ConversationContextMode::MessageArrayWithoutResponses
        )
    }

    /// Store one endpoint-normalized assistant turn after the current request.
    /// `tokens` should be the server's authoritative completion count when
    /// available, so later request accounting needs no hot-path tokenization.
    pub fn capture_response(&mut self, turn: EndpointTurn, tokens: u64) -> Result<()> {
        if !self.should_capture_response() {
            return Err(DatasetError::Validation(format!(
                "context mode {:?} does not accept live assistant responses",
                self.context_mode
            )));
        }
        let after_turn = self.current_turn.ok_or_else(|| {
            DatasetError::Validation("cannot capture a response before advancing a turn".into())
        })?;
        if self
            .replies
            .last()
            .is_some_and(|reply| reply.after_turn >= after_turn)
        {
            return Err(DatasetError::Validation(format!(
                "conversation {:?} already captured a response for turn {after_turn}",
                self.conversation_id.as_str()
            )));
        }
        self.replies.push(CapturedReply {
            after_turn,
            turn,
            tokens,
        });
        Ok(())
    }

    /// Materialize the current request through an injected implementation.
    pub fn materialize(
        &self,
        materializer: &dyn RequestMaterializer,
        endpoint: &dyn Endpoint,
        model_endpoint: &ModelEndpoint,
        phase: CreditPhase,
        overrides: &Overrides,
    ) -> Result<MaterializedRequest> {
        materializer.materialize(self, endpoint, model_endpoint, phase, overrides)
    }

    fn current(&self) -> Result<(&Conversation, &Turn, usize)> {
        let conversation = self.dataset.get(&self.conversation_id)?;
        let turn_index = self.current_turn.ok_or_else(|| {
            DatasetError::Validation("cannot materialize before advancing a turn".into())
        })?;
        let turn = conversation
            .turns
            .get(turn_index)
            .expect("advance_to validates the current turn");
        Ok((conversation, turn, turn_index))
    }

    fn endpoint_turns(&self, store: &dyn SegmentStore) -> Result<Vec<EndpointTurn>> {
        let (conversation, _, current) = self.current()?;
        match self.context_mode {
            ConversationContextMode::DeltasWithoutResponses => {
                let mut out = Vec::new();
                for index in 0..=current {
                    out.push(resolve_turn(store, &conversation.turns[index])?);
                    if let Some(reply) = self.replies.iter().find(|reply| reply.after_turn == index)
                    {
                        out.push(reply.turn.clone());
                    }
                }
                Ok(out)
            }
            ConversationContextMode::DeltasWithResponses => conversation.turns[..=current]
                .iter()
                .map(|turn| resolve_turn(store, turn))
                .collect(),
            ConversationContextMode::MessageArrayWithResponses => {
                Ok(vec![resolve_turn(store, &conversation.turns[current])?])
            }
            ConversationContextMode::MessageArrayWithoutResponses => {
                self.merge_message_array_snapshots(store, conversation, current)
            }
        }
    }

    fn merge_message_array_snapshots(
        &self,
        store: &dyn SegmentStore,
        conversation: &Conversation,
        current: usize,
    ) -> Result<Vec<EndpointTurn>> {
        let mut previous = Vec::<EndpointTurn>::new();
        let mut out = Vec::new();
        for index in 0..=current {
            let snapshot = split_snapshot(resolve_turn(store, &conversation.turns[index])?);
            if !snapshot.starts_with(&previous) {
                return Err(DatasetError::Validation(format!(
                    "conversation {:?} turn {index} is not a prefix-extending message-array snapshot",
                    conversation.session_id.as_str()
                )));
            }
            out.extend(snapshot[previous.len()..].iter().cloned());
            if index < current {
                let reply = self
                    .replies
                    .iter()
                    .find(|reply| reply.after_turn == index)
                    .ok_or_else(|| {
                        DatasetError::Validation(format!(
                            "conversation {:?} is missing the live response after snapshot turn {index}",
                            conversation.session_id.as_str()
                        ))
                    })?;
                out.push(reply.turn.clone());
            }
            previous = snapshot;
        }
        Ok(out)
    }

    fn input_tokens(&self, store: &dyn SegmentStore) -> Result<u64> {
        let (conversation, current_turn, current) = self.current()?;
        if current_turn.raw_payload.is_some() {
            return Ok(current_turn.input_tokens);
        }
        let mut count = match self.context_mode {
            ConversationContextMode::DeltasWithoutResponses
            | ConversationContextMode::DeltasWithResponses => conversation.turns[..=current]
                .iter()
                .try_fold(0_u64, |count, turn| checked_add(count, turn.input_tokens))?,
            ConversationContextMode::MessageArrayWithResponses
            | ConversationContextMode::MessageArrayWithoutResponses => current_turn.input_tokens,
        };
        if matches!(
            self.context_mode,
            ConversationContextMode::DeltasWithoutResponses
                | ConversationContextMode::MessageArrayWithoutResponses
        ) {
            for reply in self
                .replies
                .iter()
                .filter(|reply| reply.after_turn < current)
            {
                count = checked_add(count, reply.tokens)?;
            }
        }
        if let Some(tool_tokens) = selected_tool_tokens(conversation, current, self.context_mode) {
            count = checked_add(count, tool_tokens)?;
        }
        for handle in [conversation.system, conversation.user_context]
            .into_iter()
            .flatten()
        {
            if let Some(tokens) = store.get(handle)?.token_count() {
                count = checked_add(count, tokens as u64)?;
            }
        }
        Ok(count)
    }
}

fn selected_tool_tokens(
    conversation: &Conversation,
    current: usize,
    mode: ConversationContextMode,
) -> Option<u64> {
    let turns = match mode {
        ConversationContextMode::MessageArrayWithResponses
        | ConversationContextMode::MessageArrayWithoutResponses => {
            &conversation.turns[current..=current]
        }
        ConversationContextMode::DeltasWithResponses
        | ConversationContextMode::DeltasWithoutResponses => &conversation.turns[..=current],
    };
    turns
        .iter()
        .rev()
        .find(|turn| turn.tools.is_some())
        .map(|turn| turn.tool_tokens)
}

fn checked_add(left: u64, right: u64) -> Result<u64> {
    left.checked_add(right)
        .ok_or_else(|| DatasetError::Validation("request input token count overflowed u64".into()))
}

fn split_snapshot(mut turn: EndpointTurn) -> Vec<EndpointTurn> {
    match turn.raw_messages.take() {
        Some(messages) if !messages.is_empty() => messages
            .into_iter()
            .map(|message| EndpointTurn {
                raw_messages: Some(vec![message]),
                ..EndpointTurn::default()
            })
            .collect(),
        _ => vec![turn],
    }
}

fn resolve_turn(store: &dyn SegmentStore, turn: &Turn) -> Result<EndpointTurn> {
    let mut raw_messages = Vec::new();
    for handle in &turn.messages {
        raw_messages.push(raw_value(store, *handle, "message")?);
    }
    if let Some(handle) = turn.raw_messages {
        match raw_value(store, handle, "raw_messages")? {
            Value::Array(messages) => raw_messages.extend(messages),
            _ => {
                return Err(DatasetError::Validation(format!(
                    "raw_messages handle {handle} must contain a JSON array"
                )));
            }
        }
    }

    let mut resolved = EndpointTurn {
        model: turn.model.as_ref().map(|model| model.as_str().to_string()),
        role: turn.role.as_ref().map(|role| role.as_str().to_string()),
        max_tokens: turn.max_tokens,
        raw_messages: (!raw_messages.is_empty()).then_some(raw_messages),
        raw_tools: raw_array(store, turn.tools, "tools")?,
        extra_body: raw_object(store, turn.extra_body, "extra_body")?,
        ..EndpointTurn::default()
    };
    for group in &turn.content {
        let contents = group
            .handles
            .iter()
            .map(|handle| content_string(store, *handle, group.kind))
            .collect::<Result<Vec<_>>>()?;
        let media = Media {
            name: group.name.clone(),
            contents,
        };
        match group.kind {
            MediaKind::Text => resolved.texts.push(media),
            MediaKind::Image => resolved.images.push(media),
            MediaKind::Audio => resolved.audios.push(media),
            MediaKind::Video => resolved.videos.push(media),
        }
    }
    Ok(resolved)
}

fn content_string(store: &dyn SegmentStore, handle: Handle, kind: MediaKind) -> Result<String> {
    let bytes = match (kind, store.get(handle)?) {
        (MediaKind::Text, Payload::Text { bytes, .. }) => bytes,
        (expected, Payload::Media { kind, bytes }) if expected == *kind => bytes,
        (_, payload) => {
            return Err(DatasetError::PayloadKind {
                handle,
                expected: "matching text/media content",
                actual: payload.kind_name(),
            });
        }
    };
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|error| DatasetError::InvalidWire(format!("handle {handle}: {error}")))
}

fn resolve_prompt(store: &dyn SegmentStore, handle: Option<Handle>) -> Result<Option<String>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    match store.get(handle)? {
        Payload::Text { bytes, .. } => std::str::from_utf8(bytes)
            .map(|text| Some(text.to_string()))
            .map_err(|error| DatasetError::InvalidWire(format!("handle {handle}: {error}"))),
        Payload::Message { wire, .. } => {
            let value: Value = serde_json::from_slice(wire)?;
            value
                .get("content")
                .and_then(Value::as_str)
                .map(|text| Some(text.to_string()))
                .ok_or_else(|| {
                    DatasetError::InvalidWire(format!(
                        "prompt message handle {handle} has no string content"
                    ))
                })
        }
        payload => Err(DatasetError::PayloadKind {
            handle,
            expected: "text or message",
            actual: payload.kind_name(),
        }),
    }
}

fn raw_value(store: &dyn SegmentStore, handle: Handle, field: &str) -> Result<Value> {
    let wire = match store.get(handle)? {
        Payload::Raw { wire } | Payload::Message { wire, .. } => wire,
        payload => {
            return Err(DatasetError::PayloadKind {
                handle,
                expected: "raw or message",
                actual: payload.kind_name(),
            });
        }
    };
    serde_json::from_slice(wire)
        .map_err(|error| DatasetError::InvalidWire(format!("{field} handle {handle}: {error}")))
}

fn raw_array(
    store: &dyn SegmentStore,
    handle: Option<Handle>,
    field: &str,
) -> Result<Option<Vec<Value>>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    match raw_value(store, handle, field)? {
        Value::Array(values) => Ok(Some(values)),
        _ => Err(DatasetError::InvalidWire(format!(
            "{field} handle {handle} must contain a JSON array"
        ))),
    }
}

fn raw_object(
    store: &dyn SegmentStore,
    handle: Option<Handle>,
    field: &str,
) -> Result<Option<Map<String, Value>>> {
    let Some(handle) = handle else {
        return Ok(None);
    };
    match raw_value(store, handle, field)? {
        Value::Object(values) => Ok(Some(values)),
        _ => Err(DatasetError::InvalidWire(format!(
            "{field} handle {handle} must contain a JSON object"
        ))),
    }
}

fn raw_string_map(
    store: &dyn SegmentStore,
    handle: Option<Handle>,
    field: &str,
) -> Result<BTreeMap<String, String>> {
    let Some(values) = raw_object(store, handle, field)? else {
        return Ok(BTreeMap::new());
    };
    values
        .into_iter()
        .map(|(key, value)| match value {
            Value::String(value) => Ok((key, value)),
            _ => Err(DatasetError::Validation(format!(
                "{field}.{key} must be a string"
            ))),
        })
        .collect()
}

fn merge_overrides(value: &mut Value, overrides: &Overrides) -> Result<()> {
    let object = value.as_object_mut().ok_or_else(|| {
        DatasetError::Validation("endpoint formatter returned a non-object body".into())
    })?;
    for (key, value) in overrides.fields() {
        object.insert(key.clone(), value.clone());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use aiperf_endpoints::{ChatEndpoint, EndpointConfig, EndpointType};
    use bytes::Bytes;
    use smallvec::smallvec;

    use super::*;
    use crate::model::{ContentGroup, CorrelationId, ModelId};
    use crate::segment::{Role, SegmentPool};

    fn model_endpoint() -> ModelEndpoint {
        ModelEndpoint {
            primary_model_name: "default-model".into(),
            endpoint: EndpointConfig {
                endpoint_type: EndpointType::Chat,
                streaming: true,
                use_server_token_count: true,
                ..EndpointConfig::default()
            },
        }
    }

    #[test]
    fn endpoint_registry_rejects_normalized_duplicates() {
        let mut resolver = BuiltinEndpointResolver::default();
        let error = resolver.register("CHAT", ChatEndpoint).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("duplicate endpoint registration")
        );
    }

    #[test]
    fn endpoint_registry_contains_every_native_dialect() {
        let resolver = BuiltinEndpointResolver::default();
        for name in [
            "chat",
            "chat_completions",
            "completions",
            "responses",
            "embeddings",
            "chat_embeddings",
            "nim_embeddings",
            "cohere_rankings",
            "hf_tei_rankings",
            "nim_rankings",
            "huggingface_generate",
            "image_generation",
            "image_edit",
            "video_generation",
            "image_retrieval",
            "solido_rag",
            "raw",
            "template",
        ] {
            assert!(resolver.resolve(Some(name)).is_ok(), "missing {name}");
        }
    }

    fn message(pool: &mut SegmentPool, parent: Option<Handle>, role: &str, text: &str) -> Handle {
        pool.intern_message(
            parent,
            role,
            Bytes::from(
                serde_json::to_vec(&serde_json::json!({
                    "role": role,
                    "content": text
                }))
                .unwrap(),
            ),
            vec![text.len() as u32],
        )
        .unwrap()
    }

    fn dataset(mode: ConversationContextMode, turns: Vec<Turn>, pool: SegmentPool) -> Arc<Dataset> {
        let mut conversation = Conversation::new("session");
        conversation.turns = turns;
        conversation.context_mode = Some(mode);
        conversation.accuracy = Some(AccuracyAssociation {
            correlation_id: CorrelationId::from("corr"),
            task: "task".into(),
        });
        Arc::new(
            Dataset::new(
                vec![conversation],
                Arc::new(pool.freeze()),
                "sequential",
                ConversationContextMode::DeltasWithoutResponses,
            )
            .unwrap(),
        )
    }

    #[test]
    fn raw_payload_is_byte_exact_and_explicit_overrides_are_tail_spliced() {
        let mut pool = SegmentPool::new();
        let wire = Bytes::from_static(b"{ \"messages\" : [ ], \"model\":\"authored\" }\n");
        let raw = pool.intern_raw(None, wire.clone()).unwrap();
        let data = dataset(
            ConversationContextMode::MessageArrayWithResponses,
            vec![Turn {
                raw_payload: Some(raw),
                model: Some(ModelId::from("metadata-only")),
                input_tokens: 7,
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let exact = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        assert_eq!(exact.body, wire);
        assert_eq!(exact.input_tokens, 7);

        let mut overrides = Overrides::new();
        overrides.set_stream(true);
        let spliced = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &overrides,
            )
            .unwrap();
        assert_eq!(
            spliced.body,
            b"{ \"messages\" : [ ], \"model\":\"authored\" ,\"stream\":true}\n"[..]
        );
    }

    #[test]
    fn deltas_without_responses_interleaves_live_assistant_and_accounts_tokens() {
        let mut pool = SegmentPool::new();
        let q0 = message(&mut pool, None, "user", "q0");
        let q1 = message(&mut pool, Some(q0), "user", "q1");
        let data = dataset(
            ConversationContextMode::DeltasWithoutResponses,
            vec![
                Turn {
                    messages: smallvec![q0],
                    input_tokens: 2,
                    max_tokens: Some(4),
                    ..Turn::default()
                },
                Turn {
                    messages: smallvec![q1],
                    input_tokens: 3,
                    max_tokens: Some(5),
                    audio_duration_seconds: Some(2.5),
                    ..Turn::default()
                },
            ],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        session
            .capture_response(
                EndpointTurn {
                    role: Some("assistant".into()),
                    texts: vec![Media::new(vec!["a0".into()])],
                    ..EndpointTurn::default()
                },
                11,
            )
            .unwrap();
        session.advance_to(1).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["messages"][0]["content"], "q0");
        assert_eq!(body["messages"][1]["content"], "a0");
        assert_eq!(body["messages"][2]["content"], "q1");
        assert_eq!(request.input_tokens, 16);
        assert_eq!(request.max_tokens, Some(5));
        assert_eq!(request.audio_duration_seconds, Some(2.5));
        assert_eq!(request.accuracy.unwrap().correlation_id.as_str(), "corr");
    }

    #[test]
    fn message_array_without_responses_prefix_diffs_and_interleaves() {
        let mut pool = SegmentPool::new();
        let q0 = message(&mut pool, None, "user", "q0");
        let q1 = message(&mut pool, Some(q0), "user", "q1");
        let data = dataset(
            ConversationContextMode::MessageArrayWithoutResponses,
            vec![
                Turn {
                    messages: smallvec![q0],
                    input_tokens: 2,
                    ..Turn::default()
                },
                Turn {
                    messages: smallvec![q0, q1],
                    input_tokens: 5,
                    ..Turn::default()
                },
            ],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        session
            .capture_response(
                EndpointTurn {
                    role: Some("assistant".into()),
                    texts: vec![Media::new(vec!["a0".into()])],
                    ..EndpointTurn::default()
                },
                7,
            )
            .unwrap();
        session.advance_to(1).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["messages"].as_array().unwrap().len(), 3);
        assert_eq!(body["messages"][0]["content"], "q0");
        assert_eq!(body["messages"][1]["content"], "a0");
        assert_eq!(body["messages"][2]["content"], "q1");
        assert_eq!(request.input_tokens, 12);
    }

    #[test]
    fn structured_content_headers_and_parameters_are_resolved_from_handles() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let headers = pool
            .intern_raw(None, Bytes::from_static(br#"{"x-custom":"yes"}"#))
            .unwrap();
        let params = pool
            .intern_raw(None, Bytes::from_static(br#"{"api-version":"2026-01"}"#))
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                }],
                input_tokens: 2,
                streaming: Some(false),
                extra_headers: Some(headers),
                request_parameters: Some(params),
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        assert_eq!(request.headers["x-custom"], "yes");
        assert_eq!(request.parameters["api-version"], "2026-01");
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["stream"], false);
        assert!(!request.streaming);
    }

    #[test]
    fn explicit_overrides_update_wire_and_effective_request_metadata() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                model: Some(ModelId::from("authored-model")),
                max_tokens: Some(4),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                }],
                input_tokens: 2,
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let mut overrides = Overrides::new();
        overrides.set_model("dispatch-model");
        overrides.set_max_tokens("max_completion_tokens", 13);
        overrides.set_stream(false);
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &overrides,
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["model"], "dispatch-model");
        assert_eq!(body["max_completion_tokens"], 13);
        assert_eq!(body["stream"], false);
        assert_eq!(request.model, "dispatch-model");
        assert_eq!(request.max_tokens, Some(13));
        assert!(!request.streaming);
    }

    #[test]
    fn authored_extra_body_updates_effective_request_metadata() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello"),
                vec![1, 2],
            )
            .unwrap();
        let extra = pool
            .intern_raw(
                Some(text),
                Bytes::from_static(
                    br#"{"model":"body-model","stream":false,"max_completion_tokens":15}"#,
                ),
            )
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                model: Some(ModelId::from("turn-model")),
                max_tokens: Some(4),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                }],
                extra_body: Some(extra),
                input_tokens: 2,
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                &ChatEndpoint,
                &model_endpoint(),
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert_eq!(body["model"], "body-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_completion_tokens"], 15);
        assert_eq!(request.model, "body-model");
        assert_eq!(request.max_tokens, Some(15));
        assert!(!request.streaming);
    }

    #[test]
    fn responses_override_resolves_dialect_body_and_default_path() {
        let mut pool = SegmentPool::new();
        let text = pool
            .intern_text(
                None,
                Role::from("user"),
                Bytes::from_static(b"hello responses"),
                vec![1, 2],
            )
            .unwrap();
        let data = dataset(
            ConversationContextMode::DeltasWithResponses,
            vec![Turn {
                role: Some(Role::from("user")),
                endpoint: Some("responses".into()),
                max_tokens: Some(9),
                content: smallvec![ContentGroup {
                    kind: MediaKind::Text,
                    name: String::new(),
                    handles: smallvec![text],
                }],
                input_tokens: 2,
                ..Turn::default()
            }],
            pool,
        );
        let mut session = ConversationSession::new(data, SessionId::from("session")).unwrap();
        session.advance_to(0).unwrap();
        let resolver = BuiltinEndpointResolver::default();
        let endpoint = resolver
            .resolve(session.endpoint_override().unwrap())
            .unwrap();
        let mut configured = model_endpoint();
        configured.endpoint.endpoint_type = endpoint.metadata().endpoint_type;
        let request = session
            .materialize(
                &EndpointRequestMaterializer,
                endpoint.as_ref(),
                &configured,
                CreditPhase::Profiling,
                &Overrides::new(),
            )
            .unwrap();
        let body: Value = serde_json::from_slice(&request.body).unwrap();
        assert!(body.get("messages").is_none());
        assert_eq!(body["input"][0]["content"], "hello responses");
        assert_eq!(body["max_output_tokens"], 9);
        assert_eq!(request.endpoint.as_deref(), Some("responses"));
        assert_eq!(request.endpoint_path.as_deref(), Some("/v1/responses"));
    }
}
