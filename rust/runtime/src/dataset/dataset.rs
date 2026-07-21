// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen shared dataset and cross-record validation.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use crate::endpoints::EndpointDescriptor;

use crate::body_plan::BodyPlan;
use crate::dataset::error::{DatasetError, Result};
use crate::dataset::model::{
    Conversation, ConversationBranchMode, ConversationContextMode, ConversationMetadata,
    DispatchTiming, MediaKind, PrerequisiteKind, SessionId, Turn,
};
use crate::dataset::request::{raw_body_handle, resolve_prompt, resolve_turn, token_ids_handle};
use crate::dataset::segment::{Handle, Payload, Role, SegmentDomain, SegmentPool, SegmentStore};
use crate::endpoints::{
    CreditPhase, PreparedEndpoint, PreparedRequest, ShapeLowerer, Turn as EndpointTurn,
    TurnMessageLowerer,
};
use smallvec::SmallVec;

/// Media-free structural metadata for one frozen dataset.
#[derive(Debug, Clone)]
pub struct DatasetMetadata {
    /// Conversation projections in authored insertion order.
    pub conversations: Arc<[ConversationMetadata]>,
    /// User-selected sampler implementation name.
    pub sampling_strategy: String,
    /// Whether any turn carries an absolute timestamp or relative delay.
    pub has_timing_data: bool,
    /// Dataset-level context behavior used when a conversation has no override.
    pub default_context_mode: ConversationContextMode,
    /// Total authored turn count.
    pub total_turn_count: usize,
    /// Arithmetic mean turns per conversation, or zero for an empty dataset.
    pub average_turn_count: f64,
}

/// Immutable conversations plus their one shared segment store.
#[derive(Clone)]
pub struct Dataset {
    conversations: Arc<[Conversation]>,
    index: HashMap<SessionId, usize>,
    segments: Arc<dyn SegmentStore>,
    metadata: DatasetMetadata,
    /// Precomputed profiling-phase [`BodyPlan`] per `[conversation position][turn
    /// index]` for eligible static message-array turns. Empty
    /// until [`precompute_body_plans`](Dataset::precompute_body_plans) runs after
    /// endpoint-bind lowering; a `None` slot (or an empty outer vector) means the
    /// turn falls back to per-dispatch formatting. Keyed by dense position so the
    /// hot-path lookup is two indexed reads, never a hash.
    body_plans: Vec<Vec<Option<BodyPlan>>>,
}

impl fmt::Debug for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dataset")
            .field("conversations", &self.conversations.len())
            .field("segments", &self.segments.len())
            .field("metadata", &self.metadata)
            .finish()
    }
}

impl Dataset {
    /// Validate and freeze conversations while preserving insertion order.
    pub fn new(
        conversations: Vec<Conversation>,
        segments: Arc<dyn SegmentStore>,
        sampling_strategy: impl Into<String>,
        default_context_mode: ConversationContextMode,
    ) -> Result<Self> {
        let mut index = HashMap::with_capacity(conversations.len());
        for (position, conversation) in conversations.iter().enumerate() {
            if conversation.session_id.as_str().is_empty() {
                return Err(DatasetError::Validation(format!(
                    "conversation at index {position} has an empty session id"
                )));
            }
            if index
                .insert(conversation.session_id.clone(), position)
                .is_some()
            {
                return Err(DatasetError::DuplicateSession(
                    conversation.session_id.to_string(),
                ));
            }
            validate_conversation_handles(conversation, segments.as_ref())?;
        }
        validate_dag(&conversations, &index)?;

        let projections: Vec<_> = conversations.iter().map(Conversation::metadata).collect();
        let total_turn_count = conversations
            .iter()
            .map(|conversation| conversation.turns.len())
            .sum();
        let has_timing_data = conversations.iter().any(|conversation| {
            conversation
                .turns
                .iter()
                .any(|turn| turn.timestamp_ms.is_some() || turn.delay_ms.is_some())
        });
        let average_turn_count = if conversations.is_empty() {
            0.0
        } else {
            total_turn_count as f64 / conversations.len() as f64
        };
        let metadata = DatasetMetadata {
            conversations: projections.into(),
            sampling_strategy: sampling_strategy.into(),
            has_timing_data,
            default_context_mode,
            total_turn_count,
            average_turn_count,
        };

        Ok(Self {
            conversations: conversations.into(),
            index,
            segments,
            metadata,
            body_plans: Vec::new(),
        })
    }

    /// Borrow conversations in authored insertion order.
    pub fn conversations(&self) -> &[Conversation] {
        &self.conversations
    }

    /// Whether every conversation is at most a single turn.
    ///
    /// Multi-turn conversations introduce continuation think-time, which a
    /// scheduled runtime realizes as clock-scheduled deferral; a single-turn
    /// dataset issues each session exactly once with no such clock event. Used
    /// to decide whether a closed-loop concurrency run can be driven by an
    /// engine that cannot stop at a finite clock deadline.
    pub fn is_single_turn(&self) -> bool {
        self.conversations
            .iter()
            .all(|conversation| conversation.turns.len() <= 1)
    }

    /// Borrow conversation identifiers in authored insertion order.
    pub fn conversation_ids(&self) -> impl ExactSizeIterator<Item = &SessionId> {
        self.conversations
            .iter()
            .map(|conversation| &conversation.session_id)
    }

    /// Borrow only sampleable root conversation metadata in authored order.
    pub fn sampleable_metadata(&self) -> impl Iterator<Item = &ConversationMetadata> {
        self.metadata
            .conversations
            .iter()
            .filter(|metadata| metadata.dag.as_ref().map(|dag| dag.is_root).unwrap_or(true))
    }

    /// Resolve a conversation directly by authored identifier.
    pub fn get(&self, id: &SessionId) -> Result<&Conversation> {
        let position = self
            .index
            .get(id)
            .copied()
            .ok_or_else(|| DatasetError::UnknownSession(id.to_string()))?;
        Ok(&self.conversations[position])
    }

    /// Borrow the shared segment-store trait object.
    pub fn segments(&self) -> &Arc<dyn SegmentStore> {
        &self.segments
    }

    /// Lower every static content turn's message to a pre-serialized `Message`
    /// segment for the bound endpoint, so dispatch
    /// splices the stored wire instead of re-rendering and re-serializing the
    /// turn's content on every request.
    ///
    /// Run once at load, after the endpoint is bound and before the dataset is
    /// shared (`Arc::new`). The store is thawed (preserving every existing
    /// handle), each eligible turn is rendered through the injected
    /// [`TurnMessageLowerer`] to the exact wire the dispatch path would emit,
    /// interned as a `Message` segment, and recorded on `turn.messages`; the
    /// unified `body` handles are refreshed and the frozen store is swapped in.
    ///
    /// Carve-outs (kept on the live render path for byte-parity): turns with a
    /// per-turn `endpoint` override, complete `raw_payload` bodies, token-native
    /// `raw_token_ids`, preformatted `raw_messages`, or already-lowered/authored
    /// `messages`. The turn's `content` is intentionally retained so
    /// input-token accounting and the warmup first-turn re-render still resolve
    /// it. A turn whose content cannot render for this shape (Responses video,
    /// Messages audio/video, …) is skipped so the identical error still surfaces
    /// at dispatch. Idempotent: a second call finds every eligible turn already
    /// carrying `messages` and is a no-op.
    pub fn lower_messages_for_endpoint(&mut self, lowerer: &dyn TurnMessageLowerer) -> Result<()> {
        // Thaw preserves every existing handle; new `Message` wires are appended
        // and the store is refrozen once. Conversations are mutated in place when
        // uniquely owned (the load path) so lowering never clones the whole
        // O(entries) conversation vector to high-water alongside the growing pool,
        // and each turn's transient `EndpointTurn` — with any `serde_json::Value`
        // it built — is confined to `lower_turn_in_place` and dropped before the
        // next turn. The lowering high-water mark is therefore one turn's
        // intermediates plus the pool, flat in entry count.
        let mut pool = SegmentPool::thaw(self.segments.as_ref());
        let total = self.conversations.len();
        let mut changed = false;

        if let Some(conversations) = Arc::get_mut(&mut self.conversations) {
            for (position, conversation) in conversations.iter_mut().enumerate() {
                for turn in &mut conversation.turns {
                    changed |= lower_turn_in_place(turn, &mut pool, lowerer)?;
                }
                report_build_progress("lowering", position + 1, total);
            }
        } else {
            // The conversation arena is shared (e.g. the dataset was cloned before
            // lowering); fall back to a single clone-mutate-replace pass.
            let mut conversations: Vec<Conversation> = self.conversations.to_vec();
            for (position, conversation) in conversations.iter_mut().enumerate() {
                for turn in &mut conversation.turns {
                    changed |= lower_turn_in_place(turn, &mut pool, lowerer)?;
                }
                report_build_progress("lowering", position + 1, total);
            }
            if changed {
                self.conversations = conversations.into();
            }
        }

        if changed {
            self.segments = Arc::new(pool.freeze());
        }
        Ok(())
    }

    /// Cache profiling-phase [`BodyPlan`] values for eligible static
    /// message-array turns against the run's default prepared endpoint.
    ///
    /// Call after [`lower_messages_for_endpoint`](Dataset::lower_messages_for_endpoint)
    /// and before sharing the dataset. Dispatch clones the cached plan, folds the
    /// same [`Overrides`](crate::dataset::materialize::Overrides), applies the same
    /// effective-field pass, and materializes with an empty override set, preserving
    /// byte identity with per-dispatch formatting.
    ///
    /// A turn is cached only when every reuse invariant holds:
    /// - the endpoint's body is [`precomputable`](PreparedEndpoint::precomputable_body)
    ///   (excludes template, raw passthrough, and token-native dialects), and it
    ///   is a per-turn message-array shape (`chat`/`responses`/`messages`/…),
    ///   which excludes completions, embeddings, rankings, and media endpoints;
    /// - the conversation uses a static context mode (`MessageArrayWithResponses`
    ///   or `DeltasWithResponses`), where the assembled turns do not depend on
    ///   live replies, and is not a graph/DAG conversation;
    /// - the turn carries no per-turn `endpoint` override, no complete raw body,
    ///   and no token-native `raw_token_ids`.
    ///
    /// Only the profiling phase is cached; warmup folds the system prompt into the
    /// first message inside the formatter and always takes the live path. Formatter
    /// failures are non-fatal — the slot stays `None` and the identical error
    /// resurfaces on the live dispatch path. Idempotent: it rebuilds the whole
    /// cache from the current conversations each call.
    pub fn precompute_body_plans(
        &mut self,
        endpoint: &dyn PreparedEndpoint,
        primary_model_name: &str,
    ) -> Result<()> {
        // Endpoint-level gate: only precomputable message-array dialects qualify.
        // A dialect that is not a per-turn message array has no shape lowerer and
        // is left entirely on the live path.
        if !endpoint.precomputable_body()
            || ShapeLowerer::for_descriptor_id(endpoint.descriptor().id).is_none()
        {
            self.body_plans = Vec::new();
            return Ok(());
        }
        let plans = {
            let store = self.segments.as_ref();
            let mut plans: Vec<Vec<Option<BodyPlan>>> =
                Vec::with_capacity(self.conversations.len());
            for conversation in self.conversations.iter() {
                let mut turn_plans: Vec<Option<BodyPlan>> = vec![None; conversation.turns.len()];
                let mode = self.context_mode(conversation);
                let static_mode = matches!(
                    mode,
                    ConversationContextMode::MessageArrayWithResponses
                        | ConversationContextMode::DeltasWithResponses
                );
                // Graph/DAG conversations dispatch through a separate execution
                // path; never cache their turns here.
                if static_mode && conversation.dag.is_none() {
                    let system = resolve_prompt(store, conversation.system)?;
                    let user_context = resolve_prompt(store, conversation.user_context)?;
                    let conversation_id = conversation.session_id.as_str().to_string();
                    for (turn_index, (turn, slot)) in conversation
                        .turns
                        .iter()
                        .zip(turn_plans.iter_mut())
                        .enumerate()
                    {
                        // Per-turn override, raw body, and token-native turns take
                        // the live path and dispatch fallback branches.
                        if turn.endpoint.is_some()
                            || raw_body_handle(turn, store)?.is_some()
                            || token_ids_handle(turn, store)?.is_some()
                        {
                            continue;
                        }
                        let turns = static_endpoint_turns(store, conversation, turn_index, mode)?;
                        let request = PreparedRequest::new(
                            primary_model_name,
                            &turns,
                            system.as_deref(),
                            user_context.as_deref(),
                            CreditPhase::Profiling,
                            None,
                            None,
                            Some(&conversation_id),
                        );
                        // Non-fatal: an unrenderable turn simply stays uncached and
                        // surfaces its identical error at dispatch.
                        if let Ok(plan) = endpoint.format_payload(&request) {
                            *slot = Some(plan);
                        }
                    }
                }
                plans.push(turn_plans);
                report_build_progress("body plans", plans.len(), self.conversations.len());
            }
            plans
        };
        self.body_plans = plans;
        Ok(())
    }

    /// Borrow the cached profiling-phase [`BodyPlan`] for one conversation turn, if
    /// [`precompute_body_plans`](Dataset::precompute_body_plans) cached it. Dispatch
    /// clones the returned plan instead of reformatting; a `None` means fall back.
    pub(crate) fn cached_body_plan(&self, id: &SessionId, turn_index: usize) -> Option<&BodyPlan> {
        let position = *self.index.get(id)?;
        self.body_plans.get(position)?.get(turn_index)?.as_ref()
    }

    /// Borrow media-free structural metadata.
    pub fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }

    /// Resolve the effective context mode for a conversation.
    pub fn context_mode(&self, conversation: &Conversation) -> ConversationContextMode {
        conversation
            .context_mode
            .unwrap_or(self.metadata.default_context_mode)
    }

    /// Validate endpoint-owned representation requirements after composition.
    ///
    /// This pass runs before scheduling, so a prepared endpoint never discovers
    /// missing or mixed raw-token data on the dispatch path.
    pub fn validate_for_endpoint(&self, descriptor: &EndpointDescriptor) -> Result<()> {
        if !descriptor.requires_raw_token_ids {
            return Ok(());
        }
        for conversation in self.conversations.iter() {
            if conversation.system.is_some() || conversation.user_context.is_some() {
                return Err(DatasetError::Validation(format!(
                    "endpoint {:?} requires raw token IDs and does not accept system or user-context text in conversation {:?}",
                    descriptor.id,
                    conversation.session_id.as_str()
                )));
            }
            if conversation.turns.len() > 1
                && self.context_mode(conversation)
                    != ConversationContextMode::MessageArrayWithResponses
            {
                return Err(DatasetError::Validation(format!(
                    "endpoint {:?} requires independent one-turn dispatches; multi-turn conversation {:?} must use message_array_with_responses context",
                    descriptor.id,
                    conversation.session_id.as_str()
                )));
            }
            for (turn_index, turn) in conversation.turns.iter().enumerate() {
                if token_ids_handle(turn, self.segments.as_ref())?.is_none() {
                    return Err(DatasetError::Validation(format!(
                        "endpoint {:?} requires raw_token_ids, but conversation {:?} turn {turn_index} has none",
                        descriptor.id,
                        conversation.session_id.as_str()
                    )));
                }
                if raw_body_handle(turn, self.segments.as_ref())?.is_some() {
                    return Err(DatasetError::Validation(format!(
                        "endpoint {:?} requires token-native composition, but conversation {:?} turn {turn_index} retained raw payload bytes",
                        descriptor.id,
                        conversation.session_id.as_str()
                    )));
                }
                if !descriptor.supports_streaming && turn.streaming == Some(true) {
                    return Err(DatasetError::Validation(format!(
                        "endpoint {:?} does not accept streaming token-native turns (conversation {:?} turn {turn_index})",
                        descriptor.id,
                        conversation.session_id.as_str()
                    )));
                }
                if !turn.branch_ids.is_empty() || !turn.prerequisites.is_empty() {
                    return Err(DatasetError::Validation(format!(
                        "endpoint {:?} token-native scheduled input cannot carry graph control on conversation {:?} turn {turn_index}",
                        descriptor.id,
                        conversation.session_id.as_str()
                    )));
                }
            }
        }
        Ok(())
    }

    /// Rebuild a dataset containing roots whose first authored timestamp lies
    /// inside an inclusive millisecond window. DAG descendants of selected
    /// roots are retained as a complete lineage even when they have no own
    /// schedule timestamp. The segment arena remains shared.
    pub fn filter_first_turn_window(
        &self,
        start_ms: Option<f64>,
        end_ms: Option<f64>,
    ) -> Result<Self> {
        for (name, value) in [("start", start_ms), ("end", end_ms)] {
            if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
                return Err(DatasetError::Validation(format!(
                    "fixed-schedule {name} offset must be finite and non-negative"
                )));
            }
        }
        if start_ms.zip(end_ms).is_some_and(|(start, end)| start > end) {
            return Err(DatasetError::Validation(
                "fixed-schedule start offset must be <= end offset".into(),
            ));
        }
        let mut selected = HashSet::new();
        let mut pending = Vec::new();
        for conversation in self.conversations.iter().filter(|conversation| {
            conversation
                .dag
                .as_ref()
                .map(|dag| dag.is_root)
                .unwrap_or(true)
        }) {
            let Some(timestamp) = conversation
                .turns
                .first()
                .and_then(|turn| turn.timestamp_ms)
            else {
                continue;
            };
            if start_ms.is_none_or(|start| timestamp >= start)
                && end_ms.is_none_or(|end| timestamp <= end)
                && selected.insert(conversation.session_id.clone())
            {
                pending.push(conversation.session_id.clone());
            }
        }
        while let Some(id) = pending.pop() {
            let conversation = self.get(&id)?;
            for child in conversation
                .dag
                .iter()
                .flat_map(|dag| dag.branches.iter())
                .flat_map(|branch| branch.child_conversation_ids.iter())
            {
                if selected.insert(child.clone()) {
                    pending.push(child.clone());
                }
            }
        }
        let conversations = self
            .conversations
            .iter()
            .filter(|conversation| selected.contains(&conversation.session_id))
            .cloned()
            .collect();
        Self::new(
            conversations,
            self.segments.clone(),
            self.metadata.sampling_strategy.clone(),
            self.metadata.default_context_mode,
        )
    }
}

fn validate_conversation_handles(
    conversation: &Conversation,
    segments: &dyn SegmentStore,
) -> Result<()> {
    if conversation.turns.is_empty() {
        return Err(DatasetError::Validation(format!(
            "conversation {:?} contains no turns",
            conversation.session_id.as_str()
        )));
    }
    for handle in [conversation.system, conversation.user_context]
        .into_iter()
        .flatten()
    {
        let payload = segments.get(handle)?;
        if !matches!(payload, Payload::Text { .. } | Payload::Message { .. }) {
            return payload_error(handle, "text or message", payload);
        }
    }
    for (turn_index, turn) in conversation.turns.iter().enumerate() {
        validate_turn(conversation, turn_index, turn, segments)?;
    }
    Ok(())
}

fn validate_turn(
    conversation: &Conversation,
    turn_index: usize,
    turn: &Turn,
    segments: &dyn SegmentStore,
) -> Result<()> {
    let context = || {
        format!(
            "conversation {:?} turn {turn_index}",
            conversation.session_id.as_str()
        )
    };
    validate_optional_finite(turn.timestamp_ms, "timestamp_ms", &context)?;
    validate_optional_nonnegative(turn.delay_ms, "delay_ms", &context)?;
    validate_optional_nonnegative(
        turn.audio_duration_seconds,
        "audio_duration_seconds",
        &context,
    )?;
    if turn.max_tokens == Some(0) {
        return Err(DatasetError::Validation(format!(
            "{} has max_tokens=0",
            context()
        )));
    }
    if turn
        .tool_walk_start
        .is_some_and(|start| start as usize > turn_index)
    {
        return Err(DatasetError::Validation(format!(
            "{} has tool_walk_start after the current turn",
            context()
        )));
    }
    // The body admits only raw, token-ID, and message segment domains.
    let mut body_has_raw = false;
    let mut body_token_ids: Option<Handle> = None;
    let mut body_has_messages = false;
    for &handle in &turn.body {
        let payload = segments.get(handle)?;
        match payload.domain() {
            SegmentDomain::Raw => body_has_raw = true,
            SegmentDomain::TokenIds => body_token_ids = Some(handle),
            SegmentDomain::Message => body_has_messages = true,
            _ => return payload_error(handle, "raw, token-ids, or message", payload),
        }
    }
    let formatted = !turn.content.is_empty() || turn.raw_messages.is_some() || body_has_messages;
    if body_has_raw && formatted {
        return Err(DatasetError::Validation(format!(
            "{} combines raw_payload with formatted content",
            context()
        )));
    }
    if body_token_ids.is_some() && formatted {
        return Err(DatasetError::Validation(format!(
            "{} combines raw_token_ids with formatted content",
            context()
        )));
    }
    if turn.raw_messages.is_some() && body_has_messages {
        return Err(DatasetError::Validation(format!(
            "{} combines raw_messages with message handles",
            context()
        )));
    }

    for group in &turn.content {
        if group.handles.is_empty() {
            return Err(DatasetError::Validation(format!(
                "{} has an empty {:?} content group {:?}",
                context(),
                group.kind,
                group.name
            )));
        }
        for handle in &group.handles {
            let payload = segments.get(*handle)?;
            match (group.kind, payload) {
                (MediaKind::Text, Payload::Text { .. }) => {}
                (expected, Payload::Media { kind, .. }) if expected == *kind => {}
                (MediaKind::Text, payload) => return payload_error(*handle, "text", payload),
                (_, payload) => return payload_error(*handle, "matching media", payload),
            }
        }
    }
    for handle in [
        turn.raw_messages,
        turn.tools,
        turn.raw_system,
        turn.extra_body,
        turn.extra_headers,
        turn.request_parameters,
    ]
    .into_iter()
    .flatten()
    {
        let payload = segments.get(handle)?;
        if !matches!(payload, Payload::Raw { .. }) {
            return payload_error(handle, "raw", payload);
        }
    }
    if let Some(handle) = body_token_ids {
        let payload = segments.get(handle)?;
        if !matches!(payload, Payload::TokenIds { token_ids } if !token_ids.is_empty()) {
            return payload_error(handle, "non-empty token-ids", payload);
        }
        let count = u64::try_from(payload.token_count().unwrap_or_default()).map_err(|_| {
            DatasetError::Validation(format!("{} raw token count exceeds u64", context()))
        })?;
        if turn.input_tokens != count {
            return Err(DatasetError::Validation(format!(
                "{} declares input_tokens={} but raw_token_ids contains {count} IDs",
                context(),
                turn.input_tokens
            )));
        }
    }
    if let Some(handle) = turn.trace_hash_ids {
        let payload = segments.get(handle)?;
        if !matches!(payload, Payload::TraceHashIds { .. }) {
            return payload_error(handle, "trace-hash-ids", payload);
        }
    }
    for prerequisite in &turn.prerequisites {
        match prerequisite.kind {
            PrerequisiteKind::SpawnJoin if prerequisite.branch_id.is_none() => {
                return Err(DatasetError::Validation(format!(
                    "{} has spawn_join without branch_id",
                    context()
                )));
            }
            PrerequisiteKind::ChildSessionComplete
                if prerequisite.child_conversation_ids.is_empty() =>
            {
                return Err(DatasetError::Validation(format!(
                    "{} has child_session_complete without child ids",
                    context()
                )));
            }
            PrerequisiteKind::Timer
                if prerequisite
                    .timer_seconds
                    .is_none_or(|seconds| !seconds.is_finite() || seconds < 0.0) =>
            {
                return Err(DatasetError::Validation(format!(
                    "{} has timer prerequisite without a finite non-negative duration",
                    context()
                )));
            }
            PrerequisiteKind::ExternalEvent
                if prerequisite.event_name.as_deref().is_none_or(str::is_empty) =>
            {
                return Err(DatasetError::Validation(format!(
                    "{} has external_event without event_name",
                    context()
                )));
            }
            PrerequisiteKind::Barrier
                if prerequisite.barrier_id.as_deref().is_none_or(str::is_empty) =>
            {
                return Err(DatasetError::Validation(format!(
                    "{} has barrier without barrier_id",
                    context()
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Assemble the endpoint turns for one static-context turn exactly as
/// `ConversationSession::endpoint_turns` does for these two modes, so a plan
/// precomputed from them is byte-identical to the dispatch-time plan. Restricted
/// to the static modes (the only modes `precompute_body_plans` caches), where the
/// turn sequence is a pure function of the frozen conversation.
fn static_endpoint_turns(
    store: &dyn SegmentStore,
    conversation: &Conversation,
    current: usize,
    mode: ConversationContextMode,
) -> Result<Vec<EndpointTurn>> {
    match mode {
        ConversationContextMode::MessageArrayWithResponses => {
            Ok(vec![resolve_turn(store, &conversation.turns[current])?])
        }
        ConversationContextMode::DeltasWithResponses => conversation.turns[..=current]
            .iter()
            .map(|turn| resolve_turn(store, turn))
            .collect(),
        // The dynamic modes interleave live replies and are never precomputed.
        ConversationContextMode::DeltasWithoutResponses
        | ConversationContextMode::MessageArrayWithoutResponses => Err(DatasetError::Validation(
            "static_endpoint_turns called for a dynamic context mode".into(),
        )),
    }
}

/// Lower one static content turn to pre-serialized `Message` segment(s) in
/// `pool`, returning whether the turn's `body` was rewritten. The transient
/// [`EndpointTurn`] (and any `serde_json::Value` it built) lives only for this
/// call, so a streaming lowering pass keeps at most one turn's intermediates
/// alive at a time rather than the whole dataset's.
fn lower_turn_in_place(
    turn: &mut Turn,
    pool: &mut SegmentPool,
    lowerer: &dyn TurnMessageLowerer,
) -> Result<bool> {
    if !turn_is_lowerable(turn) {
        return Ok(false);
    }
    let endpoint_turn = resolve_turn(&*pool, turn)?;
    // A shape that cannot render this content (e.g. audio under the Anthropic
    // Messages shape) is left unlowered so the identical error surfaces at
    // dispatch, not at load.
    let Ok(wires) = lowerer.lower_turn(&endpoint_turn) else {
        return Ok(false);
    };
    let role = turn.role.clone().unwrap_or_else(|| Role::new("user"));
    let mut handles: SmallVec<[Handle; 1]> = SmallVec::new();
    let mut parent = None;
    for wire in wires {
        // Empty token vector: the lowered `Message` identity keys on its wire +
        // role + prefix (never re-read for accounting, which uses the turn's
        // precomputed `input_tokens`), so identical content dedups regardless of
        // token IDs.
        let handle = pool.intern_message(parent, role.clone(), wire, Vec::<u32>::new())?;
        parent = Some(handle);
        handles.push(handle);
    }
    turn.body = handles;
    Ok(true)
}

/// Emit a throttled dataset-build progress line — at roughly every 5% and always
/// on the final conversation — so long synthetic builds and lowering passes are
/// observable without per-conversation log spam. Cheap enough to call in the hot
/// loop: one modulo and an early return off the logging path.
pub(crate) fn report_build_progress(phase: &str, done: usize, total: usize) {
    if total == 0 {
        return;
    }
    let step = (total / 20).max(1);
    if done == total || done % step == 0 {
        tracing::info!("dataset build: {done}/{total} conversations ({phase})");
    }
}

fn turn_is_lowerable(turn: &Turn) -> bool {
    !turn.content.is_empty()
        && turn.body.is_empty()
        && turn.raw_messages.is_none()
        && turn.endpoint.is_none()
}

fn payload_error<T>(handle: Handle, expected: &'static str, payload: &Payload) -> Result<T> {
    Err(DatasetError::PayloadKind {
        handle,
        expected,
        actual: payload.kind_name(),
    })
}

fn validate_optional_finite(
    value: Option<f64>,
    field: &str,
    context: &dyn Fn() -> String,
) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite()) {
        return Err(DatasetError::Validation(format!(
            "{} has non-finite {field}",
            context()
        )));
    }
    Ok(())
}

fn validate_optional_nonnegative(
    value: Option<f64>,
    field: &str,
    context: &dyn Fn() -> String,
) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(DatasetError::Validation(format!(
            "{} has invalid {field}",
            context()
        )));
    }
    Ok(())
}

fn validate_dag(conversations: &[Conversation], index: &HashMap<SessionId, usize>) -> Result<()> {
    let mut branch_owner = HashMap::new();
    let mut edges: HashMap<&SessionId, Vec<&SessionId>> = HashMap::new();

    for conversation in conversations {
        let Some(dag) = &conversation.dag else {
            continue;
        };
        if dag.is_root {
            if dag.agent_depth != 0
                || dag.parent_conversation_id.is_some()
                || dag.root_conversation_id != conversation.session_id
            {
                return Err(DatasetError::Validation(format!(
                    "root conversation {:?} has inconsistent DAG lineage",
                    conversation.session_id.as_str()
                )));
            }
        } else if (dag.parent_conversation_id.is_some() && dag.agent_depth == 0)
            || (dag.parent_conversation_id.is_none() && dag.agent_depth != 0)
        {
            return Err(DatasetError::Validation(format!(
                "child conversation {:?} has inconsistent DAG lineage",
                conversation.session_id.as_str()
            )));
        }

        let mut declared = HashSet::new();
        for branch in &dag.branches {
            if branch.child_conversation_ids.is_empty() {
                return Err(DatasetError::Validation(format!(
                    "branch {:?} contains no children",
                    branch.branch_id.as_str()
                )));
            }
            if branch.mode == ConversationBranchMode::Fork
                && branch.dispatch_timing == DispatchTiming::Pre
            {
                return Err(DatasetError::Validation(format!(
                    "fork branch {:?} cannot dispatch pre-session",
                    branch.branch_id.as_str()
                )));
            }
            if !declared.insert(branch.branch_id.clone())
                || branch_owner
                    .insert(branch.branch_id.clone(), &conversation.session_id)
                    .is_some()
            {
                return Err(DatasetError::Validation(format!(
                    "duplicate DAG branch id {:?}",
                    branch.branch_id.as_str()
                )));
            }
            let mut branch_children = HashSet::new();
            for child_id in &branch.child_conversation_ids {
                if !branch_children.insert(child_id) {
                    return Err(DatasetError::Validation(format!(
                        "branch {:?} repeats child {:?}",
                        branch.branch_id.as_str(),
                        child_id.as_str()
                    )));
                }
                let child = index
                    .get(child_id)
                    .map(|position| &conversations[*position]);
                let Some(child) = child else {
                    return Err(DatasetError::Validation(format!(
                        "branch {:?} references unknown child {:?}",
                        branch.branch_id.as_str(),
                        child_id.as_str()
                    )));
                };
                let Some(child_dag) = &child.dag else {
                    return Err(DatasetError::Validation(format!(
                        "DAG child {:?} has no lineage metadata",
                        child_id.as_str()
                    )));
                };
                let valid_lineage = match branch.mode {
                    ConversationBranchMode::Fork => {
                        child_dag.parent_conversation_id.as_ref() == Some(&conversation.session_id)
                            && child_dag.agent_depth == dag.agent_depth + 1
                            && child_dag.root_conversation_id == dag.root_conversation_id
                            && !child_dag.is_root
                    }
                    ConversationBranchMode::Spawn => {
                        !child_dag.is_root
                            && (child_dag.parent_conversation_id.is_none()
                                || child_dag.parent_conversation_id.as_ref()
                                    == Some(&conversation.session_id))
                    }
                };
                if !valid_lineage {
                    return Err(DatasetError::Validation(format!(
                        "DAG child {:?} lineage does not match {:?} parent {:?}",
                        child_id.as_str(),
                        branch.mode,
                        conversation.session_id.as_str()
                    )));
                }
                edges
                    .entry(&conversation.session_id)
                    .or_default()
                    .push(child_id);
            }
        }

        let mut seen_on_turns = HashSet::new();
        let mut earlier = HashSet::new();
        for (turn_index, turn) in conversation.turns.iter().enumerate() {
            for branch_id in &turn.branch_ids {
                if !declared.contains(branch_id) || !seen_on_turns.insert(branch_id.clone()) {
                    return Err(DatasetError::Validation(format!(
                        "conversation {:?} turn {turn_index} references unknown or repeated branch {:?}",
                        conversation.session_id.as_str(),
                        branch_id.as_str()
                    )));
                }
            }
            for prerequisite in &turn.prerequisites {
                if let Some(branch_id) = &prerequisite.branch_id
                    && !earlier.contains(branch_id)
                {
                    return Err(DatasetError::Validation(format!(
                        "conversation {:?} turn {turn_index} prerequisite references branch {:?} before it is declared",
                        conversation.session_id.as_str(),
                        branch_id.as_str()
                    )));
                }
            }
            earlier.extend(turn.branch_ids.iter().cloned());
        }
        if seen_on_turns != declared {
            return Err(DatasetError::Validation(format!(
                "conversation {:?} has branch descriptors not attached to exactly one turn",
                conversation.session_id.as_str()
            )));
        }
    }

    for conversation in conversations {
        if let Some(dag) = &conversation.dag
            && !dag.is_root
            && !edges
                .values()
                .any(|children| children.contains(&&conversation.session_id))
        {
            return Err(DatasetError::Validation(format!(
                "DAG child {:?} is not referenced by its parent",
                conversation.session_id.as_str()
            )));
        }
    }

    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    for conversation in conversations {
        visit_dag(
            &conversation.session_id,
            &edges,
            &mut visiting,
            &mut visited,
        )?;
    }
    Ok(())
}

fn visit_dag<'a>(
    id: &'a SessionId,
    edges: &HashMap<&'a SessionId, Vec<&'a SessionId>>,
    visiting: &mut HashSet<&'a SessionId>,
    visited: &mut HashSet<&'a SessionId>,
) -> Result<()> {
    if visited.contains(id) {
        return Ok(());
    }
    if !visiting.insert(id) {
        return Err(DatasetError::Validation(format!(
            "DAG cycle reaches conversation {:?}",
            id.as_str()
        )));
    }
    if let Some(children) = edges.get(id) {
        for child in children {
            visit_dag(child, edges, visiting, visited)?;
        }
    }
    visiting.remove(id);
    visited.insert(id);
    Ok(())
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;
    use crate::dataset::model::{
        BranchId, ContentGroup, ConversationBranch, DagMetadata, DispatchTiming, SessionId,
    };
    use crate::dataset::segment::SegmentPool;

    fn one_turn(id: &str, handle: Handle) -> Conversation {
        let mut conversation = Conversation::new(id);
        conversation.turns.push(Turn {
            body: smallvec::smallvec![handle],
            ..Turn::default()
        });
        conversation
    }

    #[test]
    fn insertion_order_lookup_and_metadata_are_preserved() {
        let mut pool = SegmentPool::new();
        let message = pool
            .intern_message(
                None,
                "user",
                Bytes::from_static(br#"{"role":"user","content":"hi"}"#),
                vec![1_u32].into_boxed_slice(),
            )
            .unwrap();
        let dataset = Dataset::new(
            vec![one_turn("b", message), one_turn("a", message)],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap();

        assert_eq!(
            dataset
                .conversation_ids()
                .map(SessionId::as_str)
                .collect::<Vec<_>>(),
            vec!["b", "a"]
        );
        assert_eq!(
            dataset.get(&SessionId::from("a")).unwrap().session_id,
            "a".into()
        );
        assert_eq!(dataset.metadata().total_turn_count, 2);
        assert_eq!(dataset.metadata().average_turn_count, 1.0);
    }

    #[test]
    fn dag_lineage_and_branch_attachment_are_validated() {
        let mut pool = SegmentPool::new();
        let message = pool
            .intern_message(
                None,
                "user",
                Bytes::from_static(br#"{"role":"user","content":"hi"}"#),
                vec![1_u32].into_boxed_slice(),
            )
            .unwrap();
        let branch_id = BranchId::from("root:0");
        let mut root = one_turn("root", message);
        root.turns[0].branch_ids.push(branch_id.clone());
        root.dag = Some(DagMetadata {
            branches: smallvec::smallvec![ConversationBranch {
                branch_id,
                child_conversation_ids: smallvec::smallvec![SessionId::from("child")],
                mode: ConversationBranchMode::Spawn,
                dispatch_timing: DispatchTiming::Post,
                background: false,
            }],
            is_root: true,
            agent_depth: 0,
            parent_conversation_id: None,
            root_conversation_id: SessionId::from("root"),
        });
        let mut child = one_turn("child", message);
        child.dag = Some(DagMetadata {
            branches: smallvec::smallvec![],
            is_root: false,
            agent_depth: 1,
            parent_conversation_id: Some(SessionId::from("root")),
            root_conversation_id: SessionId::from("root"),
        });

        Dataset::new(
            vec![root, child],
            Arc::new(pool.freeze()),
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap();
    }

    #[test]
    fn coexisting_raw_body_and_token_ids_keep_the_token_count_validation() {
        let mut pool = SegmentPool::new();
        let raw = pool
            .intern_raw(
                None,
                Bytes::from_static(br#"{"messages":[],"token_ids":[1,2,3]}"#),
            )
            .unwrap();
        let token = pool.intern_token_ids(Some(raw), [1_u32, 2, 3]).unwrap();
        let store: Arc<dyn SegmentStore> = Arc::new(pool.freeze());

        let mut ok = Conversation::new("ok");
        ok.turns.push(Turn {
            input_tokens: 3,
            body: Turn::dispatch_body(Some(raw), Some(token), &[]),
            ..Turn::default()
        });
        let dataset = Dataset::new(
            vec![ok],
            store.clone(),
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap();
        assert_eq!(
            dataset.conversations()[0].turns[0].body.as_slice(),
            &[raw, token]
        );

        let mut bad = Conversation::new("bad");
        bad.turns.push(Turn {
            input_tokens: 99,
            body: Turn::dispatch_body(Some(raw), Some(token), &[]),
            ..Turn::default()
        });
        let error = Dataset::new(
            vec![bad],
            store,
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("raw_token_ids contains 3 IDs"),
            "unexpected error: {error}"
        );
    }

    // One-domain-per-body invariant: only raw, token-ids, and message segments
    // are valid dispatch-body handles, and a raw body cannot coexist with the
    // formatter-driven content/raw_messages representations.
    #[test]
    fn body_rejects_non_dispatch_domains_and_mixed_representations() {
        let mut pool = SegmentPool::new();
        let media = pool
            .intern_media(None, MediaKind::Image, Bytes::from_static(b"http://x"))
            .unwrap();
        let raw = pool.intern_raw(None, Bytes::from_static(b"{}")).unwrap();
        let text = pool
            .intern_text(None, "user", Bytes::from_static(b"hi"), vec![1_u32])
            .unwrap();
        let store: Arc<dyn SegmentStore> = Arc::new(pool.freeze());

        let mut media_body = Conversation::new("media");
        media_body.turns.push(Turn {
            body: smallvec::smallvec![media],
            ..Turn::default()
        });
        let error = Dataset::new(
            vec![media_body],
            store.clone(),
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("raw, token-ids, or message"),
            "unexpected error: {error}"
        );

        let mut mixed = Conversation::new("mixed");
        mixed.turns.push(Turn {
            body: Turn::dispatch_body(Some(raw), None, &[]),
            content: smallvec::smallvec![ContentGroup {
                kind: MediaKind::Text,
                name: "text".into(),
                handles: smallvec::smallvec![text],
                uuids: smallvec::smallvec![],
            }],
            ..Turn::default()
        });
        let error = Dataset::new(
            vec![mixed],
            store,
            "sequential",
            ConversationContextMode::DeltasWithoutResponses,
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("combines raw_payload with formatted content"),
            "unexpected error: {error}"
        );
    }
}
