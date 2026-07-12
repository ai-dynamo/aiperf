// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Agentic DAG JSONL lowering into native conversation topology.
//!
//! The schema/desugaring is ported from
//! `src/aiperf/dataset/loader/dag_jsonl_models.py:22-204`,
//! `src/aiperf/dataset/loader/dag_jsonl.py:181-503`, and
//! `src/aiperf/dataset/loader/_dag_jsonl_helpers.py:24-301`. Branches become
//! native descriptors and prerequisites; no Python credit-orchestrator protocol
//! is reproduced.

use std::collections::{HashMap, HashSet};

use aiperf_endpoints::extract_payload;
use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;
use serde_json::{Map, Value};
use smallvec::SmallVec;

use crate::compose::{ComposeConfig, Composer};
use crate::error::{DatasetError, Result};
use crate::loader::{DatasetLoader, DatasetProbe, LoadConfig, RawRow, jsonl_rows};
use crate::model::{
    BranchId, Conversation, ConversationBranch, ConversationBranchMode, ConversationContextMode,
    DagMetadata, DispatchTiming, PrerequisiteKind, SessionId, Turn, TurnPrerequisite,
};
use crate::segment::SegmentPool;
use crate::tokenizer::TextTokenizer;

/// Loader for one DAG conversation per JSONL row.
#[derive(Debug, Clone, Copy, Default)]
pub struct DagJsonlDatasetLoader;

/// Composer/lowerer for DAG JSONL rows.
#[derive(Debug, Clone, Copy, Default)]
pub struct DagJsonlComposer;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DagConversationRaw<'a> {
    session_id: String,
    #[serde(borrow)]
    turns: Vec<DagTurnRaw<'a>>,
    #[serde(default)]
    pre_session_spawns: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct DagTurnRaw<'a> {
    #[serde(borrow)]
    messages: &'a RawValue,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    endpoint: Option<String>,
    #[serde(default)]
    streaming: Option<bool>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default, borrow)]
    tools: Option<&'a RawValue>,
    #[serde(default, borrow)]
    raw_system: Option<&'a RawValue>,
    #[serde(default, borrow)]
    extra: Option<&'a RawValue>,
    #[serde(default, borrow)]
    extra_headers: Option<&'a RawValue>,
    #[serde(default, borrow)]
    request_parameters: Option<&'a RawValue>,
    #[serde(default)]
    forks: Vec<ForkEntry>,
    #[serde(default)]
    spawns: Vec<SpawnEntry>,
    #[serde(default)]
    delay: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum ForkEntry {
    Child(String),
    Config {
        child: String,
        #[serde(default)]
        background: bool,
    },
}

impl ForkEntry {
    fn child(&self) -> &str {
        match self {
            Self::Child(child) | Self::Config { child, .. } => child,
        }
    }

    fn background(&self) -> bool {
        matches!(
            self,
            Self::Config {
                background: true,
                ..
            }
        )
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum SpawnEntry {
    Child(String),
    Config {
        children: Vec<String>,
        #[serde(default)]
        join_at: Option<usize>,
    },
}

impl DagConversationRaw<'_> {
    fn validate(&self, origin: &impl std::fmt::Display) -> Result<()> {
        if self.session_id.is_empty() || self.turns.is_empty() {
            return Err(DatasetError::Validation(format!(
                "{origin}: DAG session_id and turns must be non-empty"
            )));
        }
        reject_duplicates(
            &self.pre_session_spawns,
            &format!("{origin}: pre_session_spawns"),
        )?;
        for (index, turn) in self.turns.iter().enumerate() {
            if turn.max_tokens == Some(0) {
                return Err(DatasetError::Validation(format!(
                    "{origin}: turn {index} max_tokens must be positive"
                )));
            }
            if !turn.delay.is_finite() || turn.delay < 0.0 {
                return Err(DatasetError::Validation(format!(
                    "{origin}: turn {index} delay must be finite and non-negative"
                )));
            }
            for (field, value) in [("model", &turn.model), ("endpoint", &turn.endpoint)] {
                if value.as_ref().is_some_and(|value| value.trim().is_empty()) {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} {field} must be non-empty when configured"
                    )));
                }
            }
            validate_messages(turn.messages, origin, index)?;
            validate_optional_array(turn.tools, "tools", origin, index)?;
            validate_optional_array(turn.raw_system, "raw_system", origin, index)?;
            validate_optional_object(turn.extra, "extra", origin, index)?;
            validate_optional_string_object(turn.extra_headers, "extra_headers", origin, index)?;
            validate_optional_string_object(
                turn.request_parameters,
                "request_parameters",
                origin,
                index,
            )?;
            let fork_children = turn
                .forks
                .iter()
                .map(|fork| fork.child().to_string())
                .collect::<Vec<_>>();
            reject_duplicates(&fork_children, &format!("{origin}: turn {index} forks"))?;
            for fork in &turn.forks {
                if fork.child().is_empty() {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} fork child cannot be empty"
                    )));
                }
            }
            let spawn_groups = spawn_groups(&turn.spawns)?;
            let mut all_spawn_children = Vec::new();
            for (children, join_at) in &spawn_groups {
                if children.is_empty() || children.iter().any(String::is_empty) {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} spawn children must be non-empty"
                    )));
                }
                reject_duplicates(children, &format!("{origin}: turn {index} spawn group"))?;
                if let Some(join_at) = join_at
                    && (*join_at <= index || *join_at >= self.turns.len())
                {
                    return Err(DatasetError::Validation(format!(
                        "{origin}: turn {index} spawn join_at={join_at} must be after the spawning turn and inside the conversation"
                    )));
                }
                all_spawn_children.extend(children.iter().cloned());
            }
            reject_duplicates(
                &all_spawn_children,
                &format!("{origin}: turn {index} spawn groups"),
            )?;
            if index + 1 < self.turns.len() && turn.forks.iter().any(|fork| !fork.background()) {
                return Err(DatasetError::Validation(format!(
                    "{origin}: turn {index} has a foreground fork before the terminal turn"
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl DatasetLoader for DagJsonlDatasetLoader {
    fn name(&self) -> &str {
        "dag_jsonl"
    }

    fn can_load(&self, probe: &DatasetProbe) -> bool {
        let Some(value) = probe.value.as_ref().and_then(Value::as_object) else {
            return false;
        };
        value.get("session_id").is_some_and(Value::is_string)
            && value
                .get("turns")
                .and_then(Value::as_array)
                .is_some_and(|turns| {
                    !turns.is_empty()
                        && turns.iter().all(|turn| {
                            turn.get("messages").is_some_and(Value::is_array)
                                || turn.get("forks").is_some()
                                || turn.get("spawns").is_some()
                        })
                })
    }

    async fn load(&self, config: &LoadConfig) -> Result<Vec<RawRow>> {
        let rows = jsonl_rows(&config.source)?;
        if rows.is_empty() {
            return Err(DatasetError::Validation(
                "DAG JSONL source contains no conversations".into(),
            ));
        }
        let mut ids = HashSet::new();
        for row in &rows {
            let wire = row.wire.as_ref().ok_or_else(|| {
                DatasetError::Validation(format!(
                    "{}: DAG wire bytes were not retained",
                    row.origin
                ))
            })?;
            let parsed: DagConversationRaw<'_> = serde_json::from_slice(wire).map_err(|error| {
                DatasetError::Validation(format!(
                    "{}: invalid DAG conversation: {error}",
                    row.origin
                ))
            })?;
            parsed.validate(&row.origin)?;
            if !ids.insert(parsed.session_id.clone()) {
                return Err(DatasetError::DuplicateSession(parsed.session_id));
            }
        }
        Ok(rows)
    }

    fn preferred_sampling_strategy(&self) -> &str {
        "random"
    }

    fn default_context_mode(&self) -> Option<ConversationContextMode> {
        Some(ConversationContextMode::DeltasWithoutResponses)
    }
}

#[derive(Debug)]
struct LoweredConversation {
    conversation: Conversation,
    referenced_children: HashSet<SessionId>,
    system_message_turns: Vec<bool>,
}

impl Composer for DagJsonlComposer {
    fn compose(
        &self,
        rows: Vec<RawRow>,
        config: &ComposeConfig,
        tokenizer: &dyn TextTokenizer,
        segments: &mut SegmentPool,
    ) -> Result<Vec<Conversation>> {
        let delay_cap_ms = config
            .format_options
            .get("inter_turn_delay_cap_seconds")
            .map(|value| {
                value
                    .as_f64()
                    .filter(|value| value.is_finite() && *value >= 0.0)
                    .map(|seconds| seconds * 1000.0)
                    .ok_or_else(|| {
                        DatasetError::Validation(
                            "inter_turn_delay_cap_seconds must be finite and non-negative".into(),
                        )
                    })
            })
            .transpose()?;
        let mut lowered = Vec::with_capacity(rows.len());
        let mut finalizer = config.finalizer()?;
        let mut all_ids = HashSet::new();
        for row in rows {
            let wire = row.wire.as_ref().expect("loader retains DAG wire");
            let parsed: DagConversationRaw<'_> = serde_json::from_slice(wire)?;
            parsed.validate(&row.origin)?;
            all_ids.insert(SessionId::from(parsed.session_id.as_str()));
            lowered.push(lower_conversation(
                parsed,
                delay_cap_ms,
                tokenizer,
                segments,
                &mut finalizer,
            )?);
        }
        validate_targets(&lowered, &all_ids)?;
        stamp_topology(&mut lowered)?;
        Ok(lowered
            .into_iter()
            .map(|lowered| lowered.conversation)
            .collect())
    }
}

fn lower_conversation(
    parsed: DagConversationRaw<'_>,
    delay_cap_ms: Option<f64>,
    tokenizer: &dyn TextTokenizer,
    segments: &mut SegmentPool,
    finalizer: &mut crate::compose::TurnFinalizer<'_>,
) -> Result<LoweredConversation> {
    let session_id = SessionId::from(parsed.session_id.as_str());
    let mut conversation = Conversation::new(session_id.clone());
    conversation.context_mode = Some(ConversationContextMode::DeltasWithoutResponses);
    let mut parent = None;
    let mut branch_descriptors = SmallVec::new();
    let mut referenced_children = HashSet::new();
    let mut system_message_turns = Vec::with_capacity(parsed.turns.len());
    let mut pending_prerequisites = vec![SmallVec::new(); parsed.turns.len()];

    for (index, authored) in parsed.turns.iter().enumerate() {
        system_message_turns.push(raw_messages_have_system(authored.messages)?);
        let messages_handle = segments.intern_raw(
            parent,
            Bytes::copy_from_slice(authored.messages.get().as_bytes()),
        )?;
        parent = Some(messages_handle);
        let tools_handle = authored
            .tools
            .map(|tools| {
                segments.intern_raw(parent, Bytes::copy_from_slice(tools.get().as_bytes()))
            })
            .transpose()?;
        if tools_handle.is_some() {
            parent = tools_handle;
        }
        let raw_system_handle = authored
            .raw_system
            .map(|system| {
                segments.intern_raw(parent, Bytes::copy_from_slice(system.get().as_bytes()))
            })
            .transpose()?;
        if raw_system_handle.is_some() {
            parent = raw_system_handle;
        }
        let extra_handle = authored
            .extra
            .map(|extra| {
                segments.intern_raw(parent, Bytes::copy_from_slice(extra.get().as_bytes()))
            })
            .transpose()?;
        let extra_headers = authored
            .extra_headers
            .map(|headers| {
                segments.intern_raw(parent, Bytes::copy_from_slice(headers.get().as_bytes()))
            })
            .transpose()?;
        let request_parameters = authored
            .request_parameters
            .map(|parameters| {
                segments.intern_raw(parent, Bytes::copy_from_slice(parameters.get().as_bytes()))
            })
            .transpose()?;
        let payload = message_payload(authored.messages, authored.tools)?;
        let extracted = extract_payload(&payload);
        let input_tokens = input_tokens_excluding_tools(&extracted, tokenizer)?;
        let tool_tokens = extracted
            .tool_texts
            .iter()
            .try_fold(0_u64, |count, text| add_token_count(count, text, tokenizer))?;
        let mut turn = Turn {
            model: authored.model.as_deref().map(crate::model::ModelId::from),
            endpoint: authored.endpoint.clone(),
            streaming: authored.streaming,
            max_tokens: authored.max_tokens,
            input_tokens,
            tool_tokens,
            delay_ms: Some(delay_cap_ms.map_or(authored.delay, |cap| authored.delay.min(cap))),
            raw_messages: Some(messages_handle),
            tools: tools_handle,
            raw_system: raw_system_handle,
            extra_body: extra_handle,
            extra_headers,
            request_parameters,
            prerequisites: std::mem::take(&mut pending_prerequisites[index]),
            ..Turn::default()
        };
        let forks = authored.forks.iter().collect::<Vec<_>>();
        let foreground = forks
            .iter()
            .filter(|fork| !fork.background())
            .map(|fork| SessionId::from(fork.child()))
            .collect::<SmallVec<[_; 1]>>();
        let background = forks
            .iter()
            .filter(|fork| fork.background())
            .map(|fork| SessionId::from(fork.child()))
            .collect::<SmallVec<[_; 1]>>();
        let spawn_groups = spawn_groups(&authored.spawns)?;
        let spawn_group_count = spawn_groups.len();
        let mixed = (!foreground.is_empty() || !background.is_empty()) && !spawn_groups.is_empty();
        let split_forks = !foreground.is_empty() && !background.is_empty();
        if !foreground.is_empty() {
            add_branch(
                &mut turn,
                &mut branch_descriptors,
                branch_id(&session_id, index, (mixed || split_forks).then_some("fork")),
                foreground,
                ConversationBranchMode::Fork,
                DispatchTiming::Post,
                false,
                &mut referenced_children,
            );
        }
        if !background.is_empty() {
            add_branch(
                &mut turn,
                &mut branch_descriptors,
                branch_id(
                    &session_id,
                    index,
                    (mixed || split_forks).then_some("bg_fork"),
                ),
                background,
                ConversationBranchMode::Fork,
                DispatchTiming::Post,
                true,
                &mut referenced_children,
            );
        }
        for (group_index, (children, join_at)) in spawn_groups.into_iter().enumerate() {
            let suffix = if mixed || spawn_group_count > 1 {
                Some(if group_index == 0 {
                    "spawn".to_string()
                } else {
                    format!("spawn{group_index}")
                })
            } else {
                None
            };
            let id = branch_id(&session_id, index, suffix.as_deref());
            let children = children
                .iter()
                .map(|child| SessionId::from(child.as_str()))
                .collect::<SmallVec<[_; 1]>>();
            add_branch(
                &mut turn,
                &mut branch_descriptors,
                id.clone(),
                children,
                ConversationBranchMode::Spawn,
                DispatchTiming::Post,
                false,
                &mut referenced_children,
            );
            let effective_join = join_at.unwrap_or(index + 1);
            if effective_join < parsed.turns.len() {
                pending_prerequisites[effective_join].push(TurnPrerequisite {
                    kind: PrerequisiteKind::SpawnJoin,
                    branch_id: Some(id),
                    child_conversation_ids: SmallVec::new(),
                    barrier_id: None,
                    timer_seconds: None,
                    event_name: None,
                });
            }
        }
        finalizer.finalize_turn(&mut turn)?;
        conversation.turns.push(turn);
    }

    if !parsed.pre_session_spawns.is_empty() {
        let id = BranchId::from(format!("{}:pre", session_id.as_str()));
        let children = parsed
            .pre_session_spawns
            .iter()
            .map(|child| SessionId::from(child.as_str()))
            .collect::<SmallVec<[_; 1]>>();
        add_branch(
            &mut conversation.turns[0],
            &mut branch_descriptors,
            id,
            children,
            ConversationBranchMode::Spawn,
            DispatchTiming::Pre,
            false,
            &mut referenced_children,
        );
    }
    conversation.dag = Some(DagMetadata {
        branches: branch_descriptors,
        is_root: true,
        agent_depth: 0,
        parent_conversation_id: None,
        root_conversation_id: session_id,
    });
    Ok(LoweredConversation {
        conversation,
        referenced_children,
        system_message_turns,
    })
}

#[allow(clippy::too_many_arguments)]
fn add_branch(
    turn: &mut Turn,
    descriptors: &mut SmallVec<[ConversationBranch; 0]>,
    id: BranchId,
    children: SmallVec<[SessionId; 1]>,
    mode: ConversationBranchMode,
    dispatch_timing: DispatchTiming,
    background: bool,
    referenced: &mut HashSet<SessionId>,
) {
    referenced.extend(children.iter().cloned());
    turn.branch_ids.push(id.clone());
    descriptors.push(ConversationBranch {
        branch_id: id,
        child_conversation_ids: children,
        mode,
        dispatch_timing,
        background,
    });
}

fn validate_targets(lowered: &[LoweredConversation], all_ids: &HashSet<SessionId>) -> Result<()> {
    let mut fork_parent = HashMap::<SessionId, SessionId>::new();
    let mut pre_spawns = HashSet::new();
    for lowered_conversation in lowered {
        let parent = &lowered_conversation.conversation.session_id;
        for branch in &lowered_conversation
            .conversation
            .dag
            .as_ref()
            .expect("lowerer always stamps DAG metadata")
            .branches
        {
            for child in &branch.child_conversation_ids {
                if !all_ids.contains(child) {
                    return Err(DatasetError::Validation(format!(
                        "DAG session {:?} references unknown child {:?}",
                        parent.as_str(),
                        child.as_str()
                    )));
                }
                if branch.mode == ConversationBranchMode::Fork
                    && let Some(previous) = fork_parent.insert(child.clone(), parent.clone())
                {
                    return Err(DatasetError::Validation(format!(
                        "DAG child {:?} has multiple fork parents {:?} and {:?}",
                        child.as_str(),
                        previous.as_str(),
                        parent.as_str()
                    )));
                }
                if branch.dispatch_timing == DispatchTiming::Pre {
                    pre_spawns.insert(child.clone());
                }
            }
        }
    }
    if let Some(child) = pre_spawns
        .iter()
        .find(|child| fork_parent.contains_key(*child))
    {
        return Err(DatasetError::Validation(format!(
            "DAG child {:?} is both a pre-session spawn and a fork target",
            child.as_str()
        )));
    }
    Ok(())
}

fn stamp_topology(lowered: &mut [LoweredConversation]) -> Result<()> {
    let index = lowered
        .iter()
        .enumerate()
        .map(|(index, lowered)| (lowered.conversation.session_id.clone(), index))
        .collect::<HashMap<_, _>>();
    let referenced = lowered
        .iter()
        .flat_map(|lowered| lowered.referenced_children.iter().cloned())
        .collect::<HashSet<_>>();
    let mut fork_parent = HashMap::<SessionId, SessionId>::new();
    let mut edges = HashMap::<SessionId, Vec<SessionId>>::new();
    for lowered_conversation in lowered.iter() {
        let parent = lowered_conversation.conversation.session_id.clone();
        for branch in &lowered_conversation
            .conversation
            .dag
            .as_ref()
            .expect("lowered DAG")
            .branches
        {
            for child in &branch.child_conversation_ids {
                edges.entry(parent.clone()).or_default().push(child.clone());
                if branch.mode == ConversationBranchMode::Fork {
                    fork_parent.insert(child.clone(), parent.clone());
                }
            }
        }
    }
    detect_cycles(&index, &edges)?;
    for lowered_conversation in lowered.iter_mut() {
        let id = lowered_conversation.conversation.session_id.clone();
        let dag = lowered_conversation
            .conversation
            .dag
            .as_mut()
            .expect("lowered DAG");
        dag.is_root = !referenced.contains(&id);
        if !dag.is_root && !fork_parent.contains_key(&id) {
            dag.root_conversation_id = id;
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for (child, parent) in &fork_parent {
            let parent_index = index[parent];
            let child_index = index[child];
            let (parent_depth, parent_root) = {
                let dag = lowered[parent_index].conversation.dag.as_ref().unwrap();
                (dag.agent_depth, dag.root_conversation_id.clone())
            };
            let dag = lowered[child_index].conversation.dag.as_mut().unwrap();
            if dag.parent_conversation_id.as_ref() != Some(parent)
                || dag.agent_depth != parent_depth + 1
                || dag.root_conversation_id != parent_root
            {
                dag.parent_conversation_id = Some(parent.clone());
                dag.agent_depth = parent_depth + 1;
                dag.root_conversation_id = parent_root;
                changed = true;
            }
        }
    }
    validate_system_placement(lowered, &fork_parent)?;
    Ok(())
}

fn validate_system_placement(
    lowered: &[LoweredConversation],
    fork_parent: &HashMap<SessionId, SessionId>,
) -> Result<()> {
    for lowered_conversation in lowered {
        let id = &lowered_conversation.conversation.session_id;
        for (index, has_system) in lowered_conversation
            .system_message_turns
            .iter()
            .copied()
            .enumerate()
        {
            if index == 0 && !fork_parent.contains_key(id) {
                continue;
            }
            if has_system {
                return Err(DatasetError::Validation(format!(
                    "DAG session {:?} turn {index} contains a non-root system message",
                    id.as_str()
                )));
            }
        }
    }
    Ok(())
}

fn detect_cycles(
    index: &HashMap<SessionId, usize>,
    edges: &HashMap<SessionId, Vec<SessionId>>,
) -> Result<()> {
    fn visit(
        id: &SessionId,
        edges: &HashMap<SessionId, Vec<SessionId>>,
        visiting: &mut Vec<SessionId>,
        visited: &mut HashSet<SessionId>,
    ) -> Result<()> {
        if let Some(position) = visiting.iter().position(|candidate| candidate == id) {
            let mut cycle = visiting[position..]
                .iter()
                .map(|id| id.as_str())
                .collect::<Vec<_>>();
            cycle.push(id.as_str());
            return Err(DatasetError::Validation(format!(
                "DAG cycle detected: {}",
                cycle.join(" -> ")
            )));
        }
        if !visited.insert(id.clone()) {
            return Ok(());
        }
        visiting.push(id.clone());
        for child in edges.get(id).into_iter().flatten() {
            visit(child, edges, visiting, visited)?;
        }
        visiting.pop();
        Ok(())
    }
    let mut visiting = Vec::new();
    let mut visited = HashSet::new();
    for id in index.keys() {
        visit(id, edges, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn spawn_groups(entries: &[SpawnEntry]) -> Result<Vec<(Vec<String>, Option<usize>)>> {
    let mut groups = Vec::new();
    let mut bare = Vec::new();
    for entry in entries {
        match entry {
            SpawnEntry::Child(child) => bare.push(child.clone()),
            SpawnEntry::Config { children, join_at } => {
                if !bare.is_empty() {
                    groups.push((std::mem::take(&mut bare), None));
                }
                groups.push((children.clone(), *join_at));
            }
        }
    }
    if !bare.is_empty() {
        groups.push((bare, None));
    }
    Ok(groups)
}

fn branch_id(session: &SessionId, turn: usize, suffix: Option<&str>) -> BranchId {
    BranchId::from(match suffix {
        Some(suffix) => format!("{}:{turn}:{suffix}", session.as_str()),
        None => format!("{}:{turn}", session.as_str()),
    })
}

fn reject_duplicates(values: &[String], context: &str) -> Result<()> {
    let mut seen = HashSet::new();
    if let Some(value) = values.iter().find(|value| !seen.insert(value.as_str())) {
        return Err(DatasetError::Validation(format!(
            "{context}: duplicate child {value:?}"
        )));
    }
    Ok(())
}

fn validate_messages(raw: &RawValue, origin: &impl std::fmt::Display, turn: usize) -> Result<()> {
    let messages: Value = serde_json::from_str(raw.get())?;
    let Some(messages) = messages.as_array() else {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} messages must be an array"
        )));
    };
    if messages.is_empty()
        || messages.iter().any(|message| {
            !message.is_object() || !message.get("role").is_some_and(Value::is_string)
        })
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} messages must be non-empty objects with roles"
        )));
    }
    Ok(())
}

fn raw_messages_have_system(raw: &RawValue) -> Result<bool> {
    let messages: Value = serde_json::from_str(raw.get())?;
    Ok(messages.as_array().is_some_and(|messages| {
        messages
            .iter()
            .any(|message| message.get("role").and_then(Value::as_str) == Some("system"))
    }))
}

fn validate_optional_array(
    raw: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
    turn: usize,
) -> Result<()> {
    if let Some(raw) = raw
        && !serde_json::from_str::<Value>(raw.get())?.is_array()
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} must be an array"
        )));
    }
    Ok(())
}

fn validate_optional_object(
    raw: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
    turn: usize,
) -> Result<()> {
    if let Some(raw) = raw
        && !serde_json::from_str::<Value>(raw.get())?.is_object()
    {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} must be an object"
        )));
    }
    Ok(())
}

fn validate_optional_string_object(
    raw: Option<&RawValue>,
    field: &str,
    origin: &impl std::fmt::Display,
    turn: usize,
) -> Result<()> {
    let Some(raw) = raw else { return Ok(()) };
    let value: Value = serde_json::from_str(raw.get())?;
    let Some(object) = value.as_object() else {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} must be an object"
        )));
    };
    if object.values().any(|value| !value.is_string()) {
        return Err(DatasetError::Validation(format!(
            "{origin}: turn {turn} {field} values must be strings"
        )));
    }
    Ok(())
}

fn message_payload(messages: &RawValue, tools: Option<&RawValue>) -> Result<Value> {
    let mut object = Map::new();
    object.insert("messages".into(), serde_json::from_str(messages.get())?);
    if let Some(tools) = tools {
        object.insert("tools".into(), serde_json::from_str(tools.get())?);
    }
    Ok(Value::Object(object))
}

fn input_tokens_excluding_tools(
    extracted: &aiperf_endpoints::ExtractedPayload,
    tokenizer: &dyn TextTokenizer,
) -> Result<u64> {
    let mut excluded = HashMap::<&str, usize>::new();
    for text in &extracted.tool_texts {
        *excluded.entry(text).or_default() += 1;
    }
    let mut count = extracted.pretokenised_token_count;
    for text in &extracted.texts {
        if let Some(remaining) = excluded.get_mut(text.as_str())
            && *remaining > 0
        {
            *remaining -= 1;
            continue;
        }
        count = add_token_count(count, text, tokenizer)?;
    }
    Ok(count)
}

fn add_token_count(count: u64, text: &str, tokenizer: &dyn TextTokenizer) -> Result<u64> {
    count
        .checked_add(tokenizer.encode(text)?.len() as u64)
        .ok_or_else(|| DatasetError::Validation("DAG token count overflowed u64".into()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use aiperf_rng::RngRoot;
    use serde_json::json;

    use super::*;
    use crate::loader::{DatasetFormatRegistration, DatasetSource, LoaderRegistry};
    use crate::tokenizer::TiktokenTokenizer;

    async fn build(value: Value) -> Result<crate::Dataset> {
        let mut registry = LoaderRegistry::new();
        registry.register(DatasetFormatRegistration::new(
            Arc::new(DagJsonlDatasetLoader),
            Arc::new(DagJsonlComposer),
        ))?;
        registry
            .build_dataset(
                Some("dag_jsonl"),
                &LoadConfig::new(DatasetSource::Inline(value)),
                &ComposeConfig::new("model", RngRoot::new(Some(4))),
                &TiktokenTokenizer::builtin(),
            )
            .await
    }

    #[tokio::test]
    async fn forks_spawns_prebranches_and_delayed_join_lower_completely() {
        let dataset = build(json!([
            {
                "session_id":"root",
                "pre_session_spawns":["pre"],
                "turns":[
                    {"messages":[{"role":"system","content":"s"},{"role":"user","content":"q0"}],
                     "forks":[{"child":"fork","background":true}],
                     "spawns":[{"children":["spawn"],"join_at":2}]},
                    {"messages":[{"role":"user","content":"q1"}]},
                    {"messages":[{"role":"user","content":"q2"}]}
                ]
            },
            {"session_id":"fork","turns":[{"messages":[{"role":"user","content":"f"}]}]},
            {"session_id":"spawn","turns":[{"messages":[{"role":"user","content":"x"}]}]},
            {"session_id":"pre","turns":[{"messages":[{"role":"user","content":"p"}]}]}
        ]))
        .await
        .unwrap();
        let root = dataset.get(&SessionId::from("root")).unwrap();
        let dag = root.dag.as_ref().unwrap();
        assert_eq!(dag.branches.len(), 3);
        assert_eq!(root.turns[2].prerequisites.len(), 1);
        assert_eq!(
            root.turns[2].prerequisites[0].kind,
            PrerequisiteKind::SpawnJoin
        );
        let fork = dataset.get(&SessionId::from("fork")).unwrap();
        assert_eq!(
            fork.dag
                .as_ref()
                .unwrap()
                .parent_conversation_id
                .as_ref()
                .unwrap()
                .as_str(),
            "root"
        );
        assert_eq!(fork.dag.as_ref().unwrap().agent_depth, 1);
        assert!(
            !dataset
                .get(&SessionId::from("spawn"))
                .unwrap()
                .dag
                .as_ref()
                .unwrap()
                .is_root
        );
        assert_eq!(dataset.sampleable_metadata().count(), 1);
    }

    #[tokio::test]
    async fn dag_turns_preserve_endpoint_headers_parameters_and_streaming() {
        let dataset = build(json!([{
            "session_id":"root",
            "turns":[{
                "messages":[{"role":"user","content":"q"}],
                "model":"turn-model",
                "endpoint":"responses",
                "streaming":false,
                "extra_headers":{"x-agent":"yes"},
                "request_parameters":{"api-version":"2026-07"}
            }]
        }]))
        .await
        .unwrap();
        let turn = &dataset.get(&SessionId::from("root")).unwrap().turns[0];
        assert_eq!(turn.model.as_ref().unwrap().as_str(), "turn-model");
        assert_eq!(turn.endpoint.as_deref(), Some("responses"));
        assert_eq!(turn.streaming, Some(false));
        assert!(turn.extra_headers.is_some());
        assert!(turn.request_parameters.is_some());
    }

    #[tokio::test]
    async fn unknown_targets_cycles_duplicates_and_foreground_midturn_are_rejected() {
        for value in [
            json!([{"session_id":"a","turns":[{"messages":[{"role":"user"}],"forks":["missing"]}]}]),
            json!([
                {"session_id":"a","turns":[{"messages":[{"role":"user"}],"spawns":["b"]}]},
                {"session_id":"b","turns":[{"messages":[{"role":"user"}],"spawns":["a"]}]}
            ]),
            json!([{"session_id":"a","turns":[{"messages":[{"role":"user"}],"spawns":["b","b"]}]},{"session_id":"b","turns":[{"messages":[{"role":"user"}]}]}]),
            json!([{"session_id":"a","turns":[{"messages":[{"role":"user"}],"forks":["b"]},{"messages":[{"role":"user"}]}]},{"session_id":"b","turns":[{"messages":[{"role":"user"}]}]}]),
        ] {
            assert!(build(value).await.is_err());
        }
    }
}
