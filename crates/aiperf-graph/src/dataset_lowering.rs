// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native dataset-DAG to Graph-IR lowering.
//!
//! The semantic source is the complete Python DAG loader and branch runtime:
//! `src/aiperf/dataset/loader/dag_jsonl.py:189-503`,
//! `src/aiperf/dataset/loader/_dag_jsonl_helpers.py:52-301`,
//! `src/aiperf/timing/branch_orchestrator.py:128-480`, and
//! `src/aiperf/timing/_branch_orchestrator_spawn.py:45-315`.
//! The lowering keeps the behavior while deleting the credit protocol:
//!
//! - a post-turn branch is an out-edge from the declaring turn;
//! - a pre-session SPAWN is a `START` entry scheduled before the root;
//! - a FORK child receives the parent's ordered prompt program;
//! - a SPAWN child starts with fresh context;
//! - a `SPAWN_JOIN` is a static requirement on every selected child's terminal
//!   channel, so the channel store owns fan-in and cannot satisfy early.
//!
//! Every static prompt item retains the dense [`aiperf_dataset::Handle`] from
//! the one shared store. No graph-private content arena is constructed.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

use aiperf_dataset::{
    Conversation, ConversationBranch, ConversationBranchMode, ConversationContextMode, Dataset,
    DispatchTiming, Handle, MediaKind, Payload, PrerequisiteKind, SegmentStore, SessionId, Turn,
};

use crate::model::{
    ChannelRequirement, ChannelSpec, ChannelType, Count, END_NODE_ID, GraphRecord, LlmNode,
    ParsedGraph, PromptItem, ReducerName, START_NODE_ID, StaticEdge, TraceRecord,
};

const LOWERING_VERSION: &str = "aiperf-dataset-dag-v1";

/// One lowered multi-root dataset plus the unchanged shared segment store.
#[derive(Clone)]
pub struct LoweredDatasetGraph {
    /// Graph-per-root records and one trace referencing each graph.
    pub parsed: ParsedGraph,
    /// The dataset's immutable content-addressed segment arena.
    pub segments: Arc<dyn SegmentStore>,
}

impl fmt::Debug for LoweredDatasetGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoweredDatasetGraph")
            .field("graphs", &self.parsed.graphs.len())
            .field("traces", &self.parsed.traces.len())
            .field("segments", &self.segments.len())
            .finish()
    }
}

/// Dataset-to-Graph-IR extension point.
pub trait DatasetGraphLowerer: Send + Sync {
    /// Lower every sampleable DAG root in authored order.
    fn lower(&self, dataset: &Dataset) -> Result<LoweredDatasetGraph, DatasetGraphLowerError>;
}

/// Canonical static lowering for native [`aiperf_dataset::DagMetadata`].
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeDatasetGraphLowerer;

impl DatasetGraphLowerer for NativeDatasetGraphLowerer {
    fn lower(&self, dataset: &Dataset) -> Result<LoweredDatasetGraph, DatasetGraphLowerError> {
        lower_dataset(dataset)
    }
}

/// Lower every sampleable root conversation into its own graph and trace.
pub fn lower_dataset(dataset: &Dataset) -> Result<LoweredDatasetGraph, DatasetGraphLowerError> {
    let roots = dataset
        .conversations()
        .iter()
        .filter(|conversation| {
            conversation
                .dag
                .as_ref()
                .is_some_and(|metadata| metadata.is_root)
        })
        .collect::<Vec<_>>();
    if roots.is_empty() {
        return Err(DatasetGraphLowerError::NoDagRoots);
    }

    let mut parsed = ParsedGraph::default();
    for root in roots {
        let graph_name = format!("dag:{}", root.session_id.as_str());
        let graph = GraphBuilder::new(dataset).lower_root(root)?;
        parsed.graphs.insert(graph_name.clone(), graph);
        parsed.traces.push(TraceRecord {
            id: root.session_id.as_str().to_string(),
            graph_ref: Some(graph_name),
            initial_state: BTreeMap::new(),
        });
    }

    Ok(LoweredDatasetGraph {
        parsed,
        segments: dataset.segments().clone(),
    })
}

/// A deterministic graph-build failure with authored dataset context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetGraphLowerError {
    /// No conversation is both DAG-authored and sampleable as a root.
    NoDagRoots,
    /// A referenced conversation unexpectedly lacks DAG lineage.
    MissingDagMetadata(String),
    /// The Graph-IR prompt program cannot represent this context mode faithfully.
    UnsupportedContextMode {
        /// Authored conversation identifier.
        conversation_id: String,
        /// Rejected mode.
        mode: ConversationContextMode,
    },
    /// A turn contains a request shape outside the chat-message Graph-IR seam.
    UnsupportedTurn {
        /// Authored conversation identifier.
        conversation_id: String,
        /// Zero-based turn index.
        turn_index: usize,
        /// Actionable reason.
        reason: String,
    },
    /// A prerequisite has no static Graph-IR equivalent.
    UnsupportedPrerequisite {
        /// Authored conversation identifier.
        conversation_id: String,
        /// Zero-based gated turn index.
        turn_index: usize,
        /// Rejected prerequisite kind.
        kind: PrerequisiteKind,
    },
    /// A branch descriptor or reference is inconsistent.
    Branch(String),
    /// A dense handle has the wrong payload kind.
    Payload {
        /// Dense segment handle.
        handle: Handle,
        /// Expected payload description.
        expected: &'static str,
        /// Actual payload kind.
        actual: &'static str,
    },
}

impl Display for DatasetGraphLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoDagRoots => write!(f, "dataset contains no sampleable DAG root conversations"),
            Self::MissingDagMetadata(id) => {
                write!(f, "DAG conversation {id:?} has no lineage metadata")
            }
            Self::UnsupportedContextMode {
                conversation_id,
                mode,
            } => write!(
                f,
                "DAG conversation {conversation_id:?} uses unsupported context mode {mode:?}; native Graph-IR DAG lowering requires deltas_without_responses"
            ),
            Self::UnsupportedTurn {
                conversation_id,
                turn_index,
                reason,
            } => write!(
                f,
                "DAG conversation {conversation_id:?} turn {turn_index} cannot lower to chat Graph-IR: {reason}"
            ),
            Self::UnsupportedPrerequisite {
                conversation_id,
                turn_index,
                kind,
            } => write!(
                f,
                "DAG conversation {conversation_id:?} turn {turn_index} uses unsupported prerequisite {kind:?}"
            ),
            Self::Branch(message) => write!(f, "invalid dataset DAG for Graph-IR: {message}"),
            Self::Payload {
                handle,
                expected,
                actual,
            } => write!(
                f,
                "segment handle {handle} contains {actual}, expected {expected} during Graph-IR lowering"
            ),
        }
    }
}

impl Error for DatasetGraphLowerError {}

#[derive(Debug, Clone)]
enum EntryTrigger {
    Start,
    After(String),
}

#[derive(Debug, Clone)]
struct ConversationExpansion {
    terminal_channel: String,
}

#[derive(Debug, Clone)]
struct BranchExpansion {
    mode: ConversationBranchMode,
    child_terminals: Vec<(SessionId, String)>,
}

struct GraphBuilder<'a> {
    dataset: &'a Dataset,
    graph: GraphRecord,
    next_node: u64,
    next_occurrence: u64,
    active_path: Vec<SessionId>,
}

impl<'a> GraphBuilder<'a> {
    fn new(dataset: &'a Dataset) -> Self {
        Self {
            dataset,
            graph: GraphRecord {
                version: Some(LOWERING_VERSION.to_string()),
                ..GraphRecord::default()
            },
            next_node: 0,
            next_occurrence: 0,
            active_path: Vec::new(),
        }
    }

    fn lower_root(mut self, root: &Conversation) -> Result<GraphRecord, DatasetGraphLowerError> {
        self.graph.system = Some(root.session_id.as_str().to_string());
        self.expand_conversation(root, Vec::new(), EntryTrigger::Start, true)?;
        self.attach_terminal_edges();
        Ok(self.graph)
    }

    fn expand_conversation(
        &mut self,
        conversation: &Conversation,
        inherited: Vec<PromptItem>,
        trigger: EntryTrigger,
        root_instance: bool,
    ) -> Result<ConversationExpansion, DatasetGraphLowerError> {
        if self
            .active_path
            .iter()
            .any(|id| id == &conversation.session_id)
        {
            let mut cycle = self
                .active_path
                .iter()
                .map(SessionId::as_str)
                .collect::<Vec<_>>();
            cycle.push(conversation.session_id.as_str());
            return Err(DatasetGraphLowerError::Branch(format!(
                "cycle reached while expanding {}",
                cycle.join(" -> ")
            )));
        }
        self.active_path.push(conversation.session_id.clone());
        let result =
            self.expand_conversation_inner(conversation, inherited, trigger, root_instance);
        self.active_path.pop();
        result
    }

    fn expand_conversation_inner(
        &mut self,
        conversation: &Conversation,
        inherited: Vec<PromptItem>,
        trigger: EntryTrigger,
        root_instance: bool,
    ) -> Result<ConversationExpansion, DatasetGraphLowerError> {
        let dag = conversation.dag.as_ref().ok_or_else(|| {
            DatasetGraphLowerError::MissingDagMetadata(conversation.session_id.as_str().to_string())
        })?;
        let mode = self.dataset.context_mode(conversation);
        if mode != ConversationContextMode::DeltasWithoutResponses {
            return Err(DatasetGraphLowerError::UnsupportedContextMode {
                conversation_id: conversation.session_id.as_str().to_string(),
                mode,
            });
        }
        if conversation.turns.is_empty() {
            return Err(DatasetGraphLowerError::Branch(format!(
                "conversation {:?} has no turns",
                conversation.session_id.as_str()
            )));
        }

        let occurrence = self.next_occurrence;
        self.next_occurrence = self.next_occurrence.saturating_add(1);
        let branches = dag
            .branches
            .iter()
            .map(|branch| (branch.branch_id.as_str(), branch))
            .collect::<HashMap<_, _>>();
        let mut expanded_branches = HashMap::<String, BranchExpansion>::new();

        // Python dispatches pre-session SPAWNs before root turn zero
        // (`branch_orchestrator.py:128-163`). Insertion order is the LocalSet's
        // deterministic same-instant order, so expand these START entries first.
        for branch_id in &conversation.turns[0].branch_ids {
            let branch = branches.get(branch_id.as_str()).ok_or_else(|| {
                DatasetGraphLowerError::Branch(format!(
                    "conversation {:?} turn 0 references unknown branch {:?}",
                    conversation.session_id.as_str(),
                    branch_id.as_str()
                ))
            })?;
            if branch.dispatch_timing != DispatchTiming::Pre {
                continue;
            }
            if !root_instance || branch.mode != ConversationBranchMode::Spawn {
                return Err(DatasetGraphLowerError::Branch(format!(
                    "pre-session branch {:?} must be a SPAWN attached to a sampleable root",
                    branch.branch_id.as_str()
                )));
            }
            let expansion = self.expand_branch(branch, Vec::new(), EntryTrigger::Start)?;
            expanded_branches.insert(branch.branch_id.as_str().to_string(), expansion);
        }

        let mut context = inherited;
        if context.is_empty() {
            self.append_conversation_context(conversation, &mut context)?;
        }
        let mut previous_node = None::<String>;
        let mut terminal_channel = None::<String>;

        for (turn_index, turn) in conversation.turns.iter().enumerate() {
            let mut items = context.clone();
            self.append_turn_items(conversation, turn_index, turn, &mut items)?;
            let output = self.allocate_channel();
            let node_id = self.allocate_node(
                conversation,
                occurrence,
                turn_index,
                turn,
                items.clone(),
                output.clone(),
            )?;
            self.add_prerequisites(conversation, turn_index, turn, &expanded_branches, &node_id)?;

            let delay_us = millis_to_micros(turn.delay_ms).ok_or_else(|| {
                DatasetGraphLowerError::UnsupportedTurn {
                    conversation_id: conversation.session_id.as_str().to_string(),
                    turn_index,
                    reason: "delay_ms is non-finite or outside Graph-IR time range".into(),
                }
            })?;
            match previous_node.as_ref() {
                Some(previous) => self.graph.edges.push(StaticEdge {
                    source: previous.clone(),
                    target: node_id.clone(),
                    delay_after_predecessor_us: nonzero(delay_us),
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: None,
                }),
                None => self.add_entry_edge(&trigger, &node_id, delay_us),
            }

            let mut after = items;
            after.push(PromptItem::Splice {
                splice: output.clone(),
            });

            for branch_id in &turn.branch_ids {
                let branch = branches.get(branch_id.as_str()).ok_or_else(|| {
                    DatasetGraphLowerError::Branch(format!(
                        "conversation {:?} turn {turn_index} references unknown branch {:?}",
                        conversation.session_id.as_str(),
                        branch_id.as_str()
                    ))
                })?;
                if branch.dispatch_timing == DispatchTiming::Pre {
                    if turn_index != 0 {
                        return Err(DatasetGraphLowerError::Branch(format!(
                            "pre-session branch {:?} is attached after turn 0",
                            branch.branch_id.as_str()
                        )));
                    }
                    continue;
                }
                if branch.mode == ConversationBranchMode::Fork
                    && !branch.background
                    && turn_index + 1 != conversation.turns.len()
                {
                    return Err(DatasetGraphLowerError::Branch(format!(
                        "foreground FORK {:?} is attached before the terminal parent turn",
                        branch.branch_id.as_str()
                    )));
                }
                let inherited = match branch.mode {
                    ConversationBranchMode::Fork => after.clone(),
                    ConversationBranchMode::Spawn => Vec::new(),
                };
                let expansion =
                    self.expand_branch(branch, inherited, EntryTrigger::After(node_id.clone()))?;
                expanded_branches.insert(branch.branch_id.as_str().to_string(), expansion);
            }

            context = after;
            previous_node = Some(node_id);
            terminal_channel = Some(output);
        }

        Ok(ConversationExpansion {
            terminal_channel: terminal_channel.expect("non-empty conversation has a terminal"),
        })
    }

    fn expand_branch(
        &mut self,
        branch: &ConversationBranch,
        inherited: Vec<PromptItem>,
        trigger: EntryTrigger,
    ) -> Result<BranchExpansion, DatasetGraphLowerError> {
        let mut child_terminals = Vec::with_capacity(branch.child_conversation_ids.len());
        for child_id in &branch.child_conversation_ids {
            let child = self.dataset.get(child_id).map_err(|error| {
                DatasetGraphLowerError::Branch(format!(
                    "branch {:?} cannot resolve child {:?}: {error}",
                    branch.branch_id.as_str(),
                    child_id.as_str()
                ))
            })?;
            let expansion =
                self.expand_conversation(child, inherited.clone(), trigger.clone(), false)?;
            child_terminals.push((child_id.clone(), expansion.terminal_channel));
        }
        Ok(BranchExpansion {
            mode: branch.mode,
            child_terminals,
        })
    }

    fn append_conversation_context(
        &self,
        conversation: &Conversation,
        items: &mut Vec<PromptItem>,
    ) -> Result<(), DatasetGraphLowerError> {
        for (handle, role) in [
            (conversation.system, "system"),
            (conversation.user_context, "user"),
        ] {
            let Some(handle) = handle else { continue };
            match self.dataset.segments().get(handle).map_err(|error| {
                DatasetGraphLowerError::Branch(format!(
                    "conversation {:?} context handle {handle}: {error}",
                    conversation.session_id.as_str()
                ))
            })? {
                Payload::Message { .. } => items.push(PromptItem::Seg { seg: handle }),
                Payload::Text { .. } => items.push(PromptItem::Text {
                    text: handle,
                    role: role.to_string(),
                }),
                payload => {
                    return Err(DatasetGraphLowerError::Payload {
                        handle,
                        expected: "message or text-only context",
                        actual: payload.kind_name(),
                    });
                }
            }
        }
        Ok(())
    }

    fn append_turn_items(
        &self,
        conversation: &Conversation,
        turn_index: usize,
        turn: &Turn,
        items: &mut Vec<PromptItem>,
    ) -> Result<(), DatasetGraphLowerError> {
        if turn.raw_payload.is_some() {
            return Err(DatasetGraphLowerError::UnsupportedTurn {
                conversation_id: conversation.session_id.as_str().to_string(),
                turn_index,
                reason: "raw_payload bypasses chat-message materialization".into(),
            });
        }
        for handle in &turn.messages {
            let payload = self.dataset.segments().get(*handle).map_err(|error| {
                DatasetGraphLowerError::Branch(format!(
                    "conversation {:?} turn {turn_index} message handle {handle}: {error}",
                    conversation.session_id.as_str()
                ))
            })?;
            if !matches!(payload, Payload::Message { .. }) {
                return Err(DatasetGraphLowerError::Payload {
                    handle: *handle,
                    expected: "message",
                    actual: payload.kind_name(),
                });
            }
            items.push(PromptItem::Seg { seg: *handle });
        }
        if let Some(raw_messages) = turn.raw_messages {
            let payload = self.dataset.segments().get(raw_messages).map_err(|error| {
                DatasetGraphLowerError::Branch(format!(
                    "conversation {:?} turn {turn_index} raw_messages handle {raw_messages}: {error}",
                    conversation.session_id.as_str()
                ))
            })?;
            if !matches!(payload, Payload::Raw { .. }) {
                return Err(DatasetGraphLowerError::Payload {
                    handle: raw_messages,
                    expected: "raw message array",
                    actual: payload.kind_name(),
                });
            }
            items.push(PromptItem::RawMessages { raw_messages });
        }
        for group in &turn.content {
            if group.kind != MediaKind::Text {
                return Err(DatasetGraphLowerError::UnsupportedTurn {
                    conversation_id: conversation.session_id.as_str().to_string(),
                    turn_index,
                    reason: format!(
                        "{:?} content requires an endpoint-specific multimodal materializer",
                        group.kind
                    ),
                });
            }
            let role = turn.role.as_ref().map_or("user", |role| role.as_str());
            for handle in &group.handles {
                let payload = self.dataset.segments().get(*handle).map_err(|error| {
                    DatasetGraphLowerError::Branch(format!(
                        "conversation {:?} turn {turn_index} text handle {handle}: {error}",
                        conversation.session_id.as_str()
                    ))
                })?;
                if !matches!(payload, Payload::Text { .. }) {
                    return Err(DatasetGraphLowerError::Payload {
                        handle: *handle,
                        expected: "text-only",
                        actual: payload.kind_name(),
                    });
                }
                items.push(PromptItem::Text {
                    text: *handle,
                    role: role.to_string(),
                });
            }
        }
        if items.is_empty() {
            return Err(DatasetGraphLowerError::UnsupportedTurn {
                conversation_id: conversation.session_id.as_str().to_string(),
                turn_index,
                reason: "turn materializes no messages".into(),
            });
        }
        Ok(())
    }

    fn allocate_channel(&mut self) -> String {
        let channel = format!("reply_{:08}", self.next_node);
        self.graph.state.insert(
            channel.clone(),
            ChannelSpec {
                channel_type: ChannelType::Messages,
                reducer: ReducerName::AddMessages,
            },
        );
        channel
    }

    fn allocate_node(
        &mut self,
        conversation: &Conversation,
        occurrence: u64,
        turn_index: usize,
        turn: &Turn,
        items: Vec<PromptItem>,
        output: String,
    ) -> Result<String, DatasetGraphLowerError> {
        let node_id = format!("n{:08}", self.next_node);
        self.next_node = self.next_node.saturating_add(1);
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "conversation_id".into(),
            serde_json::Value::String(conversation.session_id.as_str().to_string()),
        );
        metadata.insert("turn_index".into(), serde_json::Value::from(turn_index));
        metadata.insert("occurrence".into(), serde_json::Value::from(occurrence));
        metadata.insert(
            "input_tokens".into(),
            serde_json::Value::from(turn.input_tokens),
        );
        if let Some(model) = &turn.model {
            metadata.insert(
                "model".into(),
                serde_json::Value::String(model.as_str().to_string()),
            );
        }
        if let Some(endpoint) = &turn.endpoint {
            metadata.insert(
                "endpoint".into(),
                serde_json::Value::String(endpoint.clone()),
            );
        }
        if let Some(streaming) = turn.streaming {
            metadata.insert("streaming".into(), serde_json::Value::Bool(streaming));
        }
        for (name, handle) in [
            ("tools_handle", turn.tools),
            ("extra_body_handle", turn.extra_body),
            ("extra_headers_handle", turn.extra_headers),
            ("request_parameters_handle", turn.request_parameters),
        ] {
            if let Some(handle) = handle {
                metadata.insert(name.into(), serde_json::Value::from(handle.index()));
            }
        }
        self.graph.nodes.insert(
            node_id.clone(),
            LlmNode {
                output,
                streaming: turn.streaming.unwrap_or(true),
                inputs: Vec::new(),
                min_start_delay_us: turn.timestamp_ms.map(|milliseconds| milliseconds * 1_000.0),
                max_tokens: turn.max_tokens.map(|tokens| tokens as usize),
                items,
                metadata,
            },
        );
        Ok(node_id)
    }

    fn add_prerequisites(
        &mut self,
        conversation: &Conversation,
        turn_index: usize,
        turn: &Turn,
        branches: &HashMap<String, BranchExpansion>,
        node_id: &str,
    ) -> Result<(), DatasetGraphLowerError> {
        let mut required_channels = Vec::new();
        for prerequisite in &turn.prerequisites {
            match prerequisite.kind {
                PrerequisiteKind::SpawnJoin => {
                    let branch_id = prerequisite.branch_id.as_ref().ok_or_else(|| {
                        DatasetGraphLowerError::Branch(format!(
                            "conversation {:?} turn {turn_index} has SPAWN_JOIN without branch_id",
                            conversation.session_id.as_str()
                        ))
                    })?;
                    let expansion = branches.get(branch_id.as_str()).ok_or_else(|| {
                        DatasetGraphLowerError::Branch(format!(
                            "conversation {:?} turn {turn_index} joins branch {:?} before it expands",
                            conversation.session_id.as_str(),
                            branch_id.as_str()
                        ))
                    })?;
                    if expansion.mode != ConversationBranchMode::Spawn {
                        return Err(DatasetGraphLowerError::Branch(format!(
                            "conversation {:?} turn {turn_index} SPAWN_JOIN references non-SPAWN branch {:?}",
                            conversation.session_id.as_str(),
                            branch_id.as_str()
                        )));
                    }
                    let selected = select_terminals(
                        &expansion.child_terminals,
                        &prerequisite.child_conversation_ids,
                        conversation,
                        turn_index,
                    )?;
                    required_channels.extend(selected);
                }
                PrerequisiteKind::ChildSessionComplete => {
                    for child_id in &prerequisite.child_conversation_ids {
                        let mut found = false;
                        for expansion in branches.values() {
                            for (candidate, channel) in &expansion.child_terminals {
                                if candidate == child_id {
                                    required_channels.push(channel.clone());
                                    found = true;
                                }
                            }
                        }
                        if !found {
                            return Err(DatasetGraphLowerError::Branch(format!(
                                "conversation {:?} turn {turn_index} waits for undeclared child {:?}",
                                conversation.session_id.as_str(),
                                child_id.as_str()
                            )));
                        }
                    }
                }
                PrerequisiteKind::Timer
                | PrerequisiteKind::ExternalEvent
                | PrerequisiteKind::Barrier => {
                    return Err(DatasetGraphLowerError::UnsupportedPrerequisite {
                        conversation_id: conversation.session_id.as_str().to_string(),
                        turn_index,
                        kind: prerequisite.kind,
                    });
                }
            }
        }
        let mut seen = HashSet::new();
        let inputs = self
            .graph
            .nodes
            .get_mut(node_id)
            .expect("node was inserted before prerequisites");
        for channel in required_channels {
            if seen.insert(channel.clone()) {
                inputs.inputs.push(ChannelRequirement {
                    channel,
                    count: Count::N(1),
                });
            }
        }
        Ok(())
    }

    fn add_entry_edge(&mut self, trigger: &EntryTrigger, node_id: &str, delay_us: f64) {
        let source = match trigger {
            EntryTrigger::Start => START_NODE_ID.to_string(),
            EntryTrigger::After(node) => node.clone(),
        };
        self.graph.edges.push(StaticEdge {
            source,
            target: node_id.to_string(),
            delay_after_predecessor_us: match trigger {
                EntryTrigger::After(_) => nonzero(delay_us),
                EntryTrigger::Start => None,
            },
            min_start_delay_us: match trigger {
                EntryTrigger::Start => nonzero(delay_us),
                EntryTrigger::After(_) => None,
            },
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        });
    }

    fn attach_terminal_edges(&mut self) {
        let sources = self
            .graph
            .edges
            .iter()
            .filter(|edge| edge.target != END_NODE_ID)
            .map(|edge| edge.source.as_str())
            .collect::<BTreeSet<_>>();
        let terminals = self
            .graph
            .nodes
            .keys()
            .filter(|node| !sources.contains(node.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        for source in terminals {
            self.graph.edges.push(StaticEdge {
                source,
                target: END_NODE_ID.to_string(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            });
        }
    }
}

fn select_terminals(
    terminals: &[(SessionId, String)],
    subset: &[SessionId],
    conversation: &Conversation,
    turn_index: usize,
) -> Result<Vec<String>, DatasetGraphLowerError> {
    if subset.is_empty() {
        return Ok(terminals
            .iter()
            .map(|(_, channel)| channel.clone())
            .collect());
    }
    let mut selected = Vec::with_capacity(subset.len());
    for child in subset {
        let Some((_, channel)) = terminals.iter().find(|(candidate, _)| candidate == child) else {
            return Err(DatasetGraphLowerError::Branch(format!(
                "conversation {:?} turn {turn_index} prerequisite selects child {:?} outside its branch",
                conversation.session_id.as_str(),
                child.as_str()
            )));
        };
        selected.push(channel.clone());
    }
    Ok(selected)
}

fn millis_to_micros(value: Option<f64>) -> Option<f64> {
    let value = value.unwrap_or(0.0);
    (value.is_finite() && value >= 0.0).then_some(value * 1_000.0)
}

fn nonzero(value: f64) -> Option<f64> {
    (value != 0.0).then_some(value)
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use aiperf_dataset::loader::{
        DagJsonlComposer, DagJsonlDatasetLoader, DatasetFormatRegistration, DatasetSource,
        LoadConfig, LoaderRegistry,
    };
    use aiperf_dataset::{ComposeConfig, TiktokenTokenizer};
    use aiperf_rng::RngRoot;
    use async_trait::async_trait;
    use bytes::Bytes;
    use serde_json::{Value, json};

    use super::*;
    use crate::executor::{ExecutorFlags, TraceExecutor};
    use crate::materialize::{PromptMaterializer, SegmentItemsMaterializer};
    use crate::runtime::{Handle as RuntimeHandle, drive_sim};
    use crate::sink::{GraphReply, GraphSink};
    use crate::wire::OpenAiChatMessage;
    use aiperf_clock::sim_clock::SimClock;

    async fn dataset(value: Value) -> Arc<Dataset> {
        let mut registry = LoaderRegistry::new();
        registry
            .register(DatasetFormatRegistration::new(
                Arc::new(DagJsonlDatasetLoader),
                Arc::new(DagJsonlComposer),
            ))
            .unwrap();
        Arc::new(
            registry
                .build_dataset(
                    Some("dag_jsonl"),
                    &LoadConfig::new(DatasetSource::Inline(value)),
                    &ComposeConfig::new("model", RngRoot::new(Some(7))),
                    &TiktokenTokenizer::builtin(),
                )
                .await
                .unwrap(),
        )
    }

    fn metadata_node<'a>(
        graph: &'a GraphRecord,
        conversation: &str,
        turn: u64,
    ) -> (&'a str, &'a LlmNode) {
        graph
            .nodes
            .iter()
            .find(|(_, node)| {
                node.metadata.get("conversation_id").and_then(Value::as_str) == Some(conversation)
                    && node.metadata.get("turn_index").and_then(Value::as_u64) == Some(turn)
            })
            .map(|(id, node)| (id.as_str(), node))
            .unwrap()
    }

    #[tokio::test]
    async fn lowers_pre_spawn_fork_inheritance_and_delayed_join() {
        let dataset = dataset(json!([
            {"session_id":"root","pre_session_spawns":["pre"],"turns":[
                {"messages":[{"role":"system","content":"sys"},{"role":"user","content":"r0"}],
                 "forks":[{"child":"fork","background":true}],
                 "spawns":[{"children":["spawn"],"join_at":2}]},
                {"messages":[{"role":"user","content":"r1"}]},
                {"messages":[{"role":"user","content":"r2"}]}
            ]},
            {"session_id":"fork","turns":[{"messages":[{"role":"user","content":"f0"}]}]},
            {"session_id":"spawn","turns":[{"messages":[{"role":"system","content":"spawn-sys"},{"role":"user","content":"s0"}]}]},
            {"session_id":"pre","turns":[{"messages":[{"role":"system","content":"pre-sys"},{"role":"user","content":"p0"}]}]}
        ]))
        .await;

        let lowered = lower_dataset(&dataset).unwrap();
        let trace = &lowered.parsed.traces[0];
        let graph = &lowered.parsed.graphs[trace.graph_ref.as_ref().unwrap()];
        assert!(crate::validate::validate(graph).is_empty());

        let (root0_id, root0) = metadata_node(graph, "root", 0);
        let (_, root2) = metadata_node(graph, "root", 2);
        let (fork_id, fork0) = metadata_node(graph, "fork", 0);
        let (spawn_id, spawn0) = metadata_node(graph, "spawn", 0);
        let (pre_id, pre0) = metadata_node(graph, "pre", 0);

        // Pre child is inserted as a START entry before the root entry.
        let entries = graph
            .edges
            .iter()
            .filter(|edge| edge.source == START_NODE_ID)
            .map(|edge| edge.target.as_str())
            .collect::<Vec<_>>();
        assert_eq!(entries, vec![pre_id, root0_id]);

        // FORK inherits the parent's complete prompt program including reply;
        // SPAWN and pre-session children start from their own raw-message arrays.
        assert!(fork0.items.iter().any(|item| {
            matches!(item, PromptItem::Splice { splice } if splice == &root0.output)
        }));
        assert!(
            !spawn0
                .items
                .iter()
                .any(|item| matches!(item, PromptItem::Splice { .. }))
        );
        assert!(
            !pre0
                .items
                .iter()
                .any(|item| matches!(item, PromptItem::Splice { .. }))
        );
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.source == root0_id && edge.target == fork_id)
        );
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| edge.source == root0_id && edge.target == spawn_id)
        );

        // Delayed join is attached to root turn 2 and waits on spawn terminal.
        assert_eq!(root2.inputs.len(), 1);
        assert_eq!(root2.inputs[0].channel, spawn0.output);
    }

    struct RecordingSink {
        order: Rc<RefCell<Vec<String>>>,
        prompts: Rc<RefCell<HashMap<String, Vec<Value>>>>,
    }

    #[async_trait(?Send)]
    impl GraphSink<OpenAiChatMessage> for RecordingSink {
        async fn dispatch(
            &self,
            node_id: &str,
            messages: Vec<Bytes>,
            _max_tokens: Option<usize>,
            on_first_token: &dyn Fn(),
        ) -> anyhow::Result<GraphReply<OpenAiChatMessage>> {
            self.order.borrow_mut().push(node_id.to_string());
            self.prompts.borrow_mut().insert(
                node_id.to_string(),
                messages
                    .iter()
                    .map(|wire| serde_json::from_slice(wire).unwrap())
                    .collect(),
            );
            on_first_token();
            Ok(GraphReply::from_text(format!("reply:{node_id}")))
        }
    }

    #[test]
    fn simclock_executes_join_and_materializes_fork_without_context_leakage() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let dataset = runtime.block_on(dataset(json!([
                {"session_id":"root","turns":[
                    {"messages":[{"role":"user","content":"root"}],"forks":[{"child":"fork","background":true}],"spawns":["spawn"]},
                    {"messages":[{"role":"user","content":"joined"}]}
                ]},
                {"session_id":"fork","turns":[{"messages":[{"role":"user","content":"fork"}]}]},
                {"session_id":"spawn","turns":[{"messages":[{"role":"user","content":"spawn"}]}]}
            ])));
        drop(runtime);
        let lowered = lower_dataset(&dataset).unwrap();
        let trace = lowered.parsed.traces[0].clone();
        let graph = Rc::new(lowered.parsed.resolve_trace_graph(&trace).clone());
        let (root0_id, root0) = metadata_node(&graph, "root", 0);
        let root0_id = root0_id.to_string();
        let root_output = root0.output.clone();
        let (root1_id, _) = metadata_node(&graph, "root", 1);
        let root1_id = root1_id.to_string();
        let (fork_id, _) = metadata_node(&graph, "fork", 0);
        let fork_id = fork_id.to_string();
        let (spawn_id, _) = metadata_node(&graph, "spawn", 0);
        let spawn_id = spawn_id.to_string();
        let order = Rc::new(RefCell::new(Vec::new()));
        let prompts = Rc::new(RefCell::new(HashMap::new()));
        let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(RecordingSink {
            order: order.clone(),
            prompts: prompts.clone(),
        });
        let materializer: Rc<dyn PromptMaterializer> =
            Rc::new(SegmentItemsMaterializer::new(lowered.segments));
        let out = Rc::new(RefCell::new(None));
        let out_slot = out.clone();
        let clock = Rc::new(SimClock::new());
        let outcome = drive_sim(clock, move |handle: RuntimeHandle| async move {
            let exec = TraceExecutor::new(
                graph,
                materializer,
                sink,
                handle.clone(),
                ExecutorFlags::default(),
            )
            .unwrap();
            let context = exec.build_context(trace).unwrap();
            exec.schedule_entries(&context);
            handle.wait_idle().await;
            *out_slot.borrow_mut() = Some(TraceExecutor::<OpenAiChatMessage>::result(&context));
        });
        assert!(!outcome.deadlocked);
        out.borrow_mut().take().unwrap().unwrap();

        let order = order.borrow();
        assert_eq!(order[0], root0_id);
        assert!(
            order.iter().position(|id| id == &root1_id).unwrap()
                > order.iter().position(|id| id == &spawn_id).unwrap()
        );
        let prompts = prompts.borrow();
        let fork_prompt = &prompts[&fork_id];
        assert!(fork_prompt.iter().any(|message| {
            message.get("content").and_then(Value::as_str)
                == Some(format!("reply:{root0_id}").as_str())
        }));
        let spawn_prompt = &prompts[&spawn_id];
        assert!(!spawn_prompt.iter().any(|message| {
            message.get("content").and_then(Value::as_str)
                == Some(format!("reply:{root0_id}").as_str())
        }));
        assert!(prompts[&root1_id].iter().any(|message| {
            message.get("content").and_then(Value::as_str)
                == Some(format!("reply:{root0_id}").as_str())
        }));
        assert_eq!(
            root_output,
            graph_output_for(&prompts, &root0_id, &root_output)
        );
    }

    fn graph_output_for(
        _prompts: &HashMap<String, Vec<Value>>,
        _node_id: &str,
        output: &str,
    ) -> String {
        output.to_string()
    }

    #[tokio::test]
    async fn reused_spawn_template_is_instantiated_once_per_parent_root() {
        let dataset = dataset(json!([
            {"session_id":"a","turns":[{"messages":[{"role":"user","content":"a"}],"spawns":["shared"]}]},
            {"session_id":"b","turns":[{"messages":[{"role":"user","content":"b"}],"spawns":["shared"]}]},
            {"session_id":"shared","turns":[{"messages":[{"role":"user","content":"s"}]}]}
        ]))
        .await;
        let lowered = lower_dataset(&dataset).unwrap();
        assert_eq!(lowered.parsed.traces.len(), 2);
        for trace in &lowered.parsed.traces {
            let graph = lowered.parsed.resolve_trace_graph(trace);
            assert_eq!(
                graph
                    .nodes
                    .values()
                    .filter(
                        |node| node.metadata.get("conversation_id").and_then(Value::as_str)
                            == Some("shared")
                    )
                    .count(),
                1
            );
        }
    }

    #[tokio::test]
    async fn three_level_fork_chain_inherits_each_ancestor_reply() {
        let dataset = dataset(json!([
            {"session_id":"root","turns":[
                {"messages":[{"role":"user","content":"root"}],"forks":[{"child":"level-one"}]}
            ]},
            {"session_id":"level-one","turns":[
                {"messages":[{"role":"user","content":"one"}],"forks":[{"child":"level-two"}]}
            ]},
            {"session_id":"level-two","turns":[
                {"messages":[{"role":"user","content":"two"}]}
            ]}
        ]))
        .await;

        let lowered = lower_dataset(&dataset).unwrap();
        let graph = lowered
            .parsed
            .resolve_trace_graph(&lowered.parsed.traces[0]);
        assert!(crate::validate::validate(graph).is_empty());
        let (root_id, root) = metadata_node(graph, "root", 0);
        let (one_id, one) = metadata_node(graph, "level-one", 0);
        let (two_id, two) = metadata_node(graph, "level-two", 0);

        assert!(
            graph
                .edges
                .iter()
                .any(|edge| { edge.source == root_id && edge.target == one_id })
        );
        assert!(
            graph
                .edges
                .iter()
                .any(|edge| { edge.source == one_id && edge.target == two_id })
        );
        let inherited = two
            .items
            .iter()
            .filter_map(|item| match item {
                PromptItem::Splice { splice } => Some(splice.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(inherited, vec![root.output.as_str(), one.output.as_str()]);
    }
}
