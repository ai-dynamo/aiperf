// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Frozen shared dataset and cross-record validation.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

use aiperf_endpoints::EndpointDescriptor;

use crate::error::{DatasetError, Result};
use crate::model::{
    Conversation, ConversationBranchMode, ConversationContextMode, ConversationMetadata,
    DispatchTiming, MediaKind, PrerequisiteKind, SessionId, Turn,
};
use crate::segment::{Handle, Payload, SegmentStore};

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
        })
    }

    /// Borrow conversations in authored insertion order.
    pub fn conversations(&self) -> &[Conversation] {
        &self.conversations
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
    /// missing or mixed raw-token data on the dispatch path. Future endpoint
    /// representations extend [`EndpointDescriptor`] and validate here rather
    /// than branching on a concrete endpoint ID.
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
                if turn.raw_token_ids.is_none() {
                    return Err(DatasetError::Validation(format!(
                        "endpoint {:?} requires raw_token_ids, but conversation {:?} turn {turn_index} has none",
                        descriptor.id,
                        conversation.session_id.as_str()
                    )));
                }
                if turn.raw_payload.is_some() {
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
    if turn.raw_payload.is_some()
        && (!turn.messages.is_empty() || !turn.content.is_empty() || turn.raw_messages.is_some())
    {
        return Err(DatasetError::Validation(format!(
            "{} combines raw_payload with formatted content",
            context()
        )));
    }
    if turn.raw_token_ids.is_some()
        && (!turn.messages.is_empty() || !turn.content.is_empty() || turn.raw_messages.is_some())
    {
        return Err(DatasetError::Validation(format!(
            "{} combines raw_token_ids with formatted content",
            context()
        )));
    }
    if turn.raw_messages.is_some() && !turn.messages.is_empty() {
        return Err(DatasetError::Validation(format!(
            "{} combines raw_messages with message handles",
            context()
        )));
    }

    for handle in &turn.messages {
        let payload = segments.get(*handle)?;
        if !matches!(payload, Payload::Message { .. }) {
            return payload_error(*handle, "message", payload);
        }
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
        turn.raw_payload,
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
    if let Some(handle) = turn.raw_token_ids {
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
    use crate::model::{BranchId, ConversationBranch, DagMetadata, DispatchTiming, SessionId};
    use crate::segment::SegmentPool;

    fn one_turn(id: &str, handle: Handle) -> Conversation {
        let mut conversation = Conversation::new(id);
        conversation.turns.push(Turn {
            messages: smallvec::smallvec![handle],
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
}
