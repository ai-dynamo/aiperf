// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical authored DAG to Graph-IR lowering.
//!
//! The lowering keeps the behavior while deleting the credit protocol:
//!
//! - a post-turn branch is an out-edge from the declaring turn;
//! - a pre-session SPAWN is a `START` entry scheduled before the root;
//! - a FORK child receives the parent's ordered prompt program;
//! - a SPAWN child starts with fresh context;
//! - a `SPAWN_JOIN` is a static requirement on every selected child's terminal
//!   channel, so the channel store owns fan-in and cannot satisfy early.
//!
//! Every static prompt item retains the dense [`crate::dataset::Handle`] from
//! the direct input adapter's one shared store. No graph-private content arena
//! or alternate `Dataset` conversion is constructed.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::error::Error;
use std::fmt::{self, Display};
use std::sync::Arc;

use crate::dataset::{
    ConversationBranchMode, ConversationContextMode, DispatchTiming, Handle, MediaKind, Payload,
    PrerequisiteKind, SegmentStore,
};

use crate::graph::model::{
    ChannelRequirement, ChannelSpec, ChannelType, Count, END_NODE_ID, GraphRecord, LlmNode,
    ParsedGraph, PromptItem, ReducerName, START_NODE_ID, StaticEdge, TraceRecord,
};

const LOWERING_VERSION: &str = "aiperf-authored-dag-v1";

/// One lowered multi-root authored input plus its unchanged shared segment store.
#[derive(Clone)]
pub(crate) struct LoweredGraphInput {
    /// Graph-per-root records and one trace referencing each graph.
    pub parsed: ParsedGraph,
    /// The dataset's immutable content-addressed segment arena.
    pub segments: Arc<dyn SegmentStore>,
}

impl fmt::Debug for LoweredGraphInput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LoweredGraphInput")
            .field("graphs", &self.parsed.graphs.len())
            .field("traces", &self.parsed.traces.len())
            .field("segments", &self.segments.len())
            .finish()
    }
}

pub(crate) fn lower_catalog(
    catalog: &GraphCatalog,
) -> Result<LoweredGraphInput, GraphLoweringError> {
    let mut parsed = ParsedGraph::default();
    for root_id in &catalog.roots {
        let root = catalog.conversations.get(root_id).ok_or_else(|| {
            GraphLoweringError::Branch(format!("DAG root {root_id:?} is not in its catalog"))
        })?;
        let graph_name = format!("dag:{}", root.id);
        let graph = GraphBuilder::new(catalog).lower_root(root)?;
        parsed.graphs.insert(graph_name.clone(), graph);
        parsed.traces.push(TraceRecord {
            id: root.id.clone(),
            graph_ref: Some(graph_name),
            initial_state: BTreeMap::new(),
        });
    }

    Ok(LoweredGraphInput {
        parsed,
        segments: catalog.segments.clone(),
    })
}

#[derive(Clone)]
pub(crate) struct GraphCatalog {
    pub(crate) conversations: HashMap<String, CatalogConversation>,
    pub(crate) roots: Vec<String>,
    pub(crate) segments: Arc<dyn SegmentStore>,
}

#[derive(Clone)]
pub(crate) struct CatalogConversation {
    pub(crate) id: String,
    pub(crate) context_mode: ConversationContextMode,
    pub(crate) system: Option<Handle>,
    pub(crate) user_context: Option<Handle>,
    pub(crate) turns: Vec<CatalogTurn>,
    pub(crate) branches: Vec<CatalogBranch>,
}

#[derive(Clone)]
pub(crate) struct CatalogBranch {
    pub(crate) id: String,
    pub(crate) children: Vec<String>,
    pub(crate) mode: ConversationBranchMode,
    pub(crate) dispatch_timing: DispatchTiming,
    pub(crate) background: bool,
}

#[derive(Clone)]
pub(crate) struct CatalogTurn {
    pub(crate) messages: Vec<Handle>,
    pub(crate) raw_messages: Option<Handle>,
    pub(crate) raw_payload: bool,
    pub(crate) content: Vec<(MediaKind, Vec<Handle>)>,
    pub(crate) role: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) endpoint: Option<String>,
    pub(crate) streaming: Option<bool>,
    pub(crate) max_tokens: Option<u32>,
    pub(crate) input_tokens: u64,
    pub(crate) timestamp_ms: Option<f64>,
    pub(crate) delay_ms: Option<f64>,
    pub(crate) tools: Option<Handle>,
    pub(crate) raw_system: Option<Handle>,
    pub(crate) extra_body: Option<Handle>,
    pub(crate) extra_headers: Option<Handle>,
    pub(crate) request_parameters: Option<Handle>,
    pub(crate) branch_ids: Vec<String>,
    pub(crate) prerequisites: Vec<CatalogPrerequisite>,
}

#[derive(Clone)]
pub(crate) struct CatalogPrerequisite {
    pub(crate) kind: PrerequisiteKind,
    pub(crate) branch_id: Option<String>,
    pub(crate) child_ids: Vec<String>,
}

/// A deterministic graph-build failure with authored input context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GraphLoweringError {
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

impl Display for GraphLoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
            Self::Branch(message) => write!(f, "invalid authored DAG for Graph-IR: {message}"),
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

impl Error for GraphLoweringError {}

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
    child_terminals: Vec<(String, String)>,
}

struct GraphBuilder<'a> {
    catalog: &'a GraphCatalog,
    graph: GraphRecord,
    next_node: u64,
    next_occurrence: u64,
    active_path: Vec<String>,
}

impl<'a> GraphBuilder<'a> {
    fn new(catalog: &'a GraphCatalog) -> Self {
        Self {
            catalog,
            graph: GraphRecord {
                version: Some(LOWERING_VERSION.to_string()),
                ..GraphRecord::default()
            },
            next_node: 0,
            next_occurrence: 0,
            active_path: Vec::new(),
        }
    }

    fn lower_root(mut self, root: &CatalogConversation) -> Result<GraphRecord, GraphLoweringError> {
        self.graph.system = Some(root.id.clone());
        self.expand_conversation(root, Vec::new(), EntryTrigger::Start, true)?;
        self.attach_terminal_edges();
        Ok(self.graph)
    }

    fn expand_conversation(
        &mut self,
        conversation: &CatalogConversation,
        inherited: Vec<PromptItem>,
        trigger: EntryTrigger,
        root_instance: bool,
    ) -> Result<ConversationExpansion, GraphLoweringError> {
        if self.active_path.iter().any(|id| id == &conversation.id) {
            let mut cycle = self
                .active_path
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>();
            cycle.push(conversation.id.as_str());
            return Err(GraphLoweringError::Branch(format!(
                "cycle reached while expanding {}",
                cycle.join(" -> ")
            )));
        }
        self.active_path.push(conversation.id.clone());
        let result =
            self.expand_conversation_inner(conversation, inherited, trigger, root_instance);
        self.active_path.pop();
        result
    }

    fn expand_conversation_inner(
        &mut self,
        conversation: &CatalogConversation,
        inherited: Vec<PromptItem>,
        trigger: EntryTrigger,
        root_instance: bool,
    ) -> Result<ConversationExpansion, GraphLoweringError> {
        let mode = conversation.context_mode;
        if mode != ConversationContextMode::DeltasWithoutResponses {
            return Err(GraphLoweringError::UnsupportedContextMode {
                conversation_id: conversation.id.clone(),
                mode,
            });
        }
        if conversation.turns.is_empty() {
            return Err(GraphLoweringError::Branch(format!(
                "conversation {:?} has no turns",
                conversation.id
            )));
        }

        let occurrence = self.next_occurrence;
        self.next_occurrence = self.next_occurrence.saturating_add(1);
        let branches = conversation
            .branches
            .iter()
            .map(|branch| (branch.id.as_str(), branch))
            .collect::<HashMap<_, _>>();
        let mut expanded_branches = HashMap::<String, BranchExpansion>::new();

        // Pre-session SPAWN entries precede root turn zero; insertion order
        // preserves deterministic same-instant execution.
        for branch_id in &conversation.turns[0].branch_ids {
            let branch = branches.get(branch_id.as_str()).ok_or_else(|| {
                GraphLoweringError::Branch(format!(
                    "conversation {:?} turn 0 references unknown branch {:?}",
                    conversation.id, branch_id
                ))
            })?;
            if branch.dispatch_timing != DispatchTiming::Pre {
                continue;
            }
            if !root_instance || branch.mode != ConversationBranchMode::Spawn {
                return Err(GraphLoweringError::Branch(format!(
                    "pre-session branch {:?} must be a SPAWN attached to a sampleable root",
                    branch.id
                )));
            }
            let expansion = self.expand_branch(branch, Vec::new(), EntryTrigger::Start)?;
            expanded_branches.insert(branch.id.clone(), expansion);
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
                GraphLoweringError::UnsupportedTurn {
                    conversation_id: conversation.id.clone(),
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
                    GraphLoweringError::Branch(format!(
                        "conversation {:?} turn {turn_index} references unknown branch {:?}",
                        conversation.id, branch_id
                    ))
                })?;
                if branch.dispatch_timing == DispatchTiming::Pre {
                    if turn_index != 0 {
                        return Err(GraphLoweringError::Branch(format!(
                            "pre-session branch {:?} is attached after turn 0",
                            branch.id
                        )));
                    }
                    continue;
                }
                if branch.mode == ConversationBranchMode::Fork
                    && !branch.background
                    && turn_index + 1 != conversation.turns.len()
                {
                    return Err(GraphLoweringError::Branch(format!(
                        "foreground FORK {:?} is attached before the terminal parent turn",
                        branch.id
                    )));
                }
                let inherited = match branch.mode {
                    ConversationBranchMode::Fork => after.clone(),
                    ConversationBranchMode::Spawn => Vec::new(),
                };
                let expansion =
                    self.expand_branch(branch, inherited, EntryTrigger::After(node_id.clone()))?;
                expanded_branches.insert(branch.id.clone(), expansion);
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
        branch: &CatalogBranch,
        inherited: Vec<PromptItem>,
        trigger: EntryTrigger,
    ) -> Result<BranchExpansion, GraphLoweringError> {
        let mut child_terminals = Vec::with_capacity(branch.children.len());
        for child_id in &branch.children {
            let child = self.catalog.conversations.get(child_id).ok_or_else(|| {
                GraphLoweringError::Branch(format!(
                    "branch {:?} cannot resolve child {:?}",
                    branch.id, child_id
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
        conversation: &CatalogConversation,
        items: &mut Vec<PromptItem>,
    ) -> Result<(), GraphLoweringError> {
        for (handle, role) in [
            (conversation.system, "system"),
            (conversation.user_context, "user"),
        ] {
            let Some(handle) = handle else { continue };
            match self.catalog.segments.get(handle).map_err(|error| {
                GraphLoweringError::Branch(format!(
                    "conversation {:?} context handle {handle}: {error}",
                    conversation.id
                ))
            })? {
                Payload::Message { .. } => items.push(PromptItem::Seg { seg: handle }),
                Payload::Text { .. } => items.push(PromptItem::Text {
                    text: handle,
                    role: role.to_string(),
                }),
                payload => {
                    return Err(GraphLoweringError::Payload {
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
        conversation: &CatalogConversation,
        turn_index: usize,
        turn: &CatalogTurn,
        items: &mut Vec<PromptItem>,
    ) -> Result<(), GraphLoweringError> {
        if turn.raw_payload {
            return Err(GraphLoweringError::UnsupportedTurn {
                conversation_id: conversation.id.clone(),
                turn_index,
                reason: "raw_payload bypasses chat-message materialization".into(),
            });
        }
        for handle in &turn.messages {
            let payload = self.catalog.segments.get(*handle).map_err(|error| {
                GraphLoweringError::Branch(format!(
                    "conversation {:?} turn {turn_index} message handle {handle}: {error}",
                    conversation.id
                ))
            })?;
            if !matches!(payload, Payload::Message { .. }) {
                return Err(GraphLoweringError::Payload {
                    handle: *handle,
                    expected: "message",
                    actual: payload.kind_name(),
                });
            }
            items.push(PromptItem::Seg { seg: *handle });
        }
        if let Some(raw_messages) = turn.raw_messages {
            let payload = self.catalog.segments.get(raw_messages).map_err(|error| {
                GraphLoweringError::Branch(format!(
                    "conversation {:?} turn {turn_index} raw_messages handle {raw_messages}: {error}",
                    conversation.id
                ))
            })?;
            if !matches!(payload, Payload::Raw { .. }) {
                return Err(GraphLoweringError::Payload {
                    handle: raw_messages,
                    expected: "raw message array",
                    actual: payload.kind_name(),
                });
            }
            items.push(PromptItem::RawMessages { raw_messages });
        }
        for (kind, handles) in &turn.content {
            if *kind != MediaKind::Text {
                return Err(GraphLoweringError::UnsupportedTurn {
                    conversation_id: conversation.id.clone(),
                    turn_index,
                    reason: format!(
                        "{:?} content requires an endpoint-specific multimodal materializer",
                        kind
                    ),
                });
            }
            let role = turn.role.as_deref().unwrap_or("user");
            for handle in handles {
                let payload = self.catalog.segments.get(*handle).map_err(|error| {
                    GraphLoweringError::Branch(format!(
                        "conversation {:?} turn {turn_index} text handle {handle}: {error}",
                        conversation.id
                    ))
                })?;
                if !matches!(payload, Payload::Text { .. }) {
                    return Err(GraphLoweringError::Payload {
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
            return Err(GraphLoweringError::UnsupportedTurn {
                conversation_id: conversation.id.clone(),
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
        conversation: &CatalogConversation,
        occurrence: u64,
        turn_index: usize,
        turn: &CatalogTurn,
        items: Vec<PromptItem>,
        output: String,
    ) -> Result<String, GraphLoweringError> {
        let node_id = format!("n{:08}", self.next_node);
        self.next_node = self.next_node.saturating_add(1);
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "conversation_id".into(),
            serde_json::Value::String(conversation.id.clone()),
        );
        metadata.insert("turn_index".into(), serde_json::Value::from(turn_index));
        metadata.insert("occurrence".into(), serde_json::Value::from(occurrence));
        metadata.insert(
            "input_tokens".into(),
            serde_json::Value::from(turn.input_tokens),
        );
        if let Some(model) = &turn.model {
            metadata.insert("model".into(), serde_json::Value::String(model.clone()));
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
            ("raw_system_handle", turn.raw_system),
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
        conversation: &CatalogConversation,
        turn_index: usize,
        turn: &CatalogTurn,
        branches: &HashMap<String, BranchExpansion>,
        node_id: &str,
    ) -> Result<(), GraphLoweringError> {
        let mut required_channels = Vec::new();
        for prerequisite in &turn.prerequisites {
            match prerequisite.kind {
                PrerequisiteKind::SpawnJoin => {
                    let branch_id = prerequisite.branch_id.as_ref().ok_or_else(|| {
                        GraphLoweringError::Branch(format!(
                            "conversation {:?} turn {turn_index} has SPAWN_JOIN without branch_id",
                            conversation.id
                        ))
                    })?;
                    let expansion = branches.get(branch_id.as_str()).ok_or_else(|| {
                        GraphLoweringError::Branch(format!(
                            "conversation {:?} turn {turn_index} joins branch {:?} before it expands",
                            conversation.id, branch_id
                        ))
                    })?;
                    if expansion.mode != ConversationBranchMode::Spawn {
                        return Err(GraphLoweringError::Branch(format!(
                            "conversation {:?} turn {turn_index} SPAWN_JOIN references non-SPAWN branch {:?}",
                            conversation.id, branch_id
                        )));
                    }
                    let selected = select_terminals(
                        &expansion.child_terminals,
                        &prerequisite.child_ids,
                        conversation,
                        turn_index,
                    )?;
                    required_channels.extend(selected);
                }
                PrerequisiteKind::ChildSessionComplete => {
                    for child_id in &prerequisite.child_ids {
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
                            return Err(GraphLoweringError::Branch(format!(
                                "conversation {:?} turn {turn_index} waits for undeclared child {:?}",
                                conversation.id, child_id
                            )));
                        }
                    }
                }
                PrerequisiteKind::Timer
                | PrerequisiteKind::ExternalEvent
                | PrerequisiteKind::Barrier => {
                    return Err(GraphLoweringError::UnsupportedPrerequisite {
                        conversation_id: conversation.id.clone(),
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
    terminals: &[(String, String)],
    subset: &[String],
    conversation: &CatalogConversation,
    turn_index: usize,
) -> Result<Vec<String>, GraphLoweringError> {
    if subset.is_empty() {
        return Ok(terminals
            .iter()
            .map(|(_, channel)| channel.clone())
            .collect());
    }
    let mut selected = Vec::with_capacity(subset.len());
    for child in subset {
        let Some((_, channel)) = terminals.iter().find(|(candidate, _)| candidate == child) else {
            return Err(GraphLoweringError::Branch(format!(
                "conversation {:?} turn {turn_index} prerequisite selects child {:?} outside its branch",
                conversation.id, child
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

    use crate::dataset::TiktokenTokenizer;
    use crate::dataset::loader::{DatasetSource, LoadConfig};
    use async_trait::async_trait;
    use bytes::Bytes;
    use serde_json::{Value, json};

    use super::*;
    use crate::clock::sim_clock::SimClock;
    use crate::graph::executor::{ExecutorFlags, TraceExecutor};
    use crate::graph::input::{GraphInputConfig, compile_dag_jsonl_input};
    use crate::graph::materialize::{PromptMaterializer, SegmentItemsMaterializer};
    use crate::graph::runtime::{Handle as RuntimeHandle, drive_sim};
    use crate::graph::sink::{GraphReply, GraphSink};
    use crate::graph::wire::OpenAiChatMessage;

    async fn lowered(value: Value) -> LoweredGraphInput {
        let bundle = compile_dag_jsonl_input(
            GraphInputConfig {
                load: LoadConfig::new(DatasetSource::Inline(value)),
                root_limit: None,
            },
            &TiktokenTokenizer::builtin(),
        )
        .await
        .unwrap();
        let mut parsed = ParsedGraph::default();
        for plan in bundle.plans {
            let graph_name = format!("dag:{}", plan.trace.id);
            parsed.graphs.insert(graph_name.clone(), plan.graph);
            let mut trace = plan.trace;
            trace.graph_ref = Some(graph_name);
            parsed.traces.push(trace);
        }
        LoweredGraphInput {
            parsed,
            segments: bundle.segments,
        }
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
        let lowered = lowered(json!([
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
        let trace = &lowered.parsed.traces[0];
        let graph = &lowered.parsed.graphs[trace.graph_ref.as_ref().unwrap()];
        assert!(crate::graph::validate::validate(graph).is_empty());

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
        let lowered = runtime.block_on(lowered(json!([
                {"session_id":"root","turns":[
                    {"messages":[{"role":"user","content":"root"}],"forks":[{"child":"fork","background":true}],"spawns":["spawn"]},
                    {"messages":[{"role":"user","content":"joined"}]}
                ]},
                {"session_id":"fork","turns":[{"messages":[{"role":"user","content":"fork"}]}]},
                {"session_id":"spawn","turns":[{"messages":[{"role":"user","content":"spawn"}]}]}
            ])));
        drop(runtime);
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
        let lowered = lowered(json!([
            {"session_id":"a","turns":[{"messages":[{"role":"user","content":"a"}],"spawns":["shared"]}]},
            {"session_id":"b","turns":[{"messages":[{"role":"user","content":"b"}],"spawns":["shared"]}]},
            {"session_id":"shared","turns":[{"messages":[{"role":"user","content":"s"}]}]}
        ]))
        .await;
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
        let lowered = lowered(json!([
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
        let graph = lowered
            .parsed
            .resolve_trace_graph(&lowered.parsed.traces[0]);
        assert!(crate::graph::validate::validate(graph).is_empty());
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
