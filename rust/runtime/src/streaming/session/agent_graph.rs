// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Append-only cross-chunk agent and graph session state.
//!
//! Nodes and edges arrive out of order and across arbitrary partition
//! boundaries. State is append-only: a node identity is never rewritten, an
//! edge is never removed, and a declared-but-unseen predecessor is retained
//! rather than resolved optimistically. Readiness is incremental — each edge
//! insertion contributes exactly one pending count and each terminal decrements
//! exactly one — so a chunk costs work proportional to the fragments it carries,
//! not to the graph accumulated so far.
//!
//! Two refusals keep the structure safe. An edge that would close a cycle is
//! refused at ingest, because the graph is append-only and every prior insertion
//! left it acyclic. An edge naming a target that already released or terminated
//! is refused for the same reason: it would retroactively change a dependency
//! the action host has already acted on.
//!
//! Recorded agent events are retained as *inert* nodes. They participate in
//! readiness and reach terminal, but never dispatch — the streaming analogue of
//! `ExecutableGraphNode::static_request_count()` returning zero for a recorded
//! tool. Modelling inertness as "drop the node" would break predecessor
//! counting; modelling it as "never terminal" would deadlock the graph.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;
use smallvec::SmallVec;

use crate::engine::registry::WorkloadDescriptor;
use crate::streaming::{
    action::{ActionExecutionEvent, DatasetActionSchema, canonical_action_schema},
    budget::{BudgetError, BudgetLease, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::SessionFailureCode,
    format::{SessionWatermark, StreamingFormatDescriptor},
    identity::{
        ContentDigest, DuplicateDisposition, GlobalSequence, LogicalRecordReceipt,
        SessionCausalFrontier, StableActionId, StableOrderKey, StableRecordId, StableSessionKey,
        classify_logical_duplicate, stable_action_id, stable_record_id_from_key,
    },
    session::{
        DatasetActionSink, SessionClosureCapability, SessionCoordinatorError, SessionPlacement,
        SessionSealReceipt, SessionStateRetention, StreamingSessionCoordinator,
        StreamingSessionPrepareContext, StreamingSessionProgramDescriptor,
        StreamingSessionProgramFactory, ValidatedStreamingSessionProgramConfig,
    },
    source::SourceSeal,
    unit::{
        ActionContentLeaseSet, DatasetActionKind, DatasetActionV1, EventTimeUtc,
        ExecutableDatasetAction, SessionFragmentLease, SessionGraphAction, SessionMutationV1,
        SessionTerminalAction, SourcePosition, StateBudgetFailureCode, StreamingSessionFragment,
        UnitProvenance,
    },
};

/// Stable registry identity of this session program.
pub const AGENT_GRAPH_PROGRAM_ID: &str = "agent_graph";

/// Canonical fragment schema this program joins.
pub const AGENT_GRAPH_FRAGMENT_SCHEMA: &str = "aiperf.stream.session-fragment.v1";

/// Stable schema of the graph-node action this program emits.
pub const SESSION_GRAPH_ACTION_SCHEMA: &str = "session_graph.v1";

/// Stable schema of the terminal action this program emits.
pub const SESSION_TERMINAL_ACTION_SCHEMA: &str = "session_terminal.v1";

/// Checkpoint schema identity for retained agent-graph state.
const AGENT_GRAPH_CHECKPOINT_SCHEMA_ID: &str = "aiperf.stream.session.agent_graph";

/// Checkpoint schema version for retained agent-graph state.
const AGENT_GRAPH_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Immutable registry metadata for the incremental agent/graph program.
///
/// The program can emit both a graph-node action and a terminal action, so both
/// schemas are declared: the action host validates exactly one prepared binding
/// per emitted schema, and an undeclared schema surfaces as a run-time
/// capability-agreement failure rather than a design decision.
pub static AGENT_GRAPH_SESSION_PROGRAM: StreamingSessionProgramDescriptor =
    StreamingSessionProgramDescriptor {
        id: AGENT_GRAPH_PROGRAM_ID,
        description: "Append-only cross-chunk agent and graph session state",
        fragment_input_schemas: &[AGENT_GRAPH_FRAGMENT_SCHEMA],
        action_schemas: &[SESSION_GRAPH_ACTION_SCHEMA, SESSION_TERMINAL_ACTION_SCHEMA],
        // Partition exhaustion is a decoder event, not a closure proof: this
        // program closes only on an authored close or a proven finite seal.
        closure: &[
            SessionClosureCapability::ExplicitClose,
            SessionClosureCapability::FiniteSeal,
        ],
        // Spill authority belongs to a separate owner; until it exists this
        // program proves its bound in memory and refuses beyond it.
        retention: SessionStateRetention::BoundedMemory,
        placement: SessionPlacement::RoutedByStableSession,
        supports_virtual_clock: true,
    };

/// Return the exact action schemas an `agent_graph` run can emit.
///
/// The action host binds exactly one prepared binding per emitted schema, so
/// this is the set a run plan must cover before the host is constructed.
#[must_use]
pub fn agent_graph_emitted_action_schemas() -> BTreeSet<DatasetActionSchema> {
    [
        canonical_action_schema(DatasetActionKind::GraphNode),
        canonical_action_schema(DatasetActionKind::SessionTerminal),
    ]
    .into_iter()
    .collect()
}

/// Whether a node dispatches or is inert recorded state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeRole {
    /// Dispatches one graph-node action.
    Llm,
    /// Recorded tool or agent event; never executed.
    InertRecorded,
}

/// Append-only node lifecycle. Transitions are strictly forward.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeState {
    /// Declared, with at least one declared predecessor still outstanding.
    Waiting,
    /// Every declared predecessor is terminal; the action has been emitted.
    Released,
    /// The node reached a terminal receipt, or is inert and never dispatches.
    Terminal,
}

/// One append-only graph node.
///
/// `pending_predecessors` counts declared predecessors that have not yet
/// reached terminal. It is the whole readiness mechanism: an edge insertion
/// contributes one, a terminal removes exactly one, and zero means "release".
#[derive(Debug)]
struct GraphSessionNode {
    node_key: String,
    request: Vec<u8>,
    role: GraphNodeRole,
    pending_predecessors: usize,
    state: GraphNodeState,
    action_id: Option<StableActionId>,
    // Retained separately from the node so a growing graph charges
    // incrementally rather than reallocating one lease per session.
    lease: BudgetLease,
}

/// One session's complete append-only graph.
#[derive(Debug)]
pub struct GraphSessionScope {
    session_key: StableSessionKey,
    nodes: BTreeMap<StableRecordId, GraphSessionNode>,
    /// Complete append-only adjacency, including edges whose endpoints are
    /// still undeclared. Successor lookup is therefore total.
    edges: BTreeMap<StableRecordId, SmallVec<[StableRecordId; 2]>>,
    /// Inbound edges naming a `to` node that has not been declared yet.
    ///
    /// Each entry is removed the moment its target node is declared, so this is
    /// a transient hidden-parent buffer, never a second copy of the graph.
    orphan_edges: BTreeMap<StableRecordId, SmallVec<[StableRecordId; 2]>>,
    receipts: BTreeMap<StableRecordId, LogicalRecordReceipt>,
    ready: VecDeque<StableRecordId>,
    causal_ordinal: u64,
    pending_close_reason: Option<String>,
    is_terminal_emitted: bool,
    last_action: Option<StableActionId>,
    last_provenance: UnitProvenance,
    last_stable_order: StableOrderKey,
    last_source_position: SourcePosition,
    first_source_position: SourcePosition,
    last_event_time: Option<EventTimeUtc>,
    version: u64,
}

impl GraphSessionScope {
    fn new(session_key: StableSessionKey, fragment: &StreamingSessionFragment) -> Self {
        Self {
            session_key,
            nodes: BTreeMap::new(),
            edges: BTreeMap::new(),
            orphan_edges: BTreeMap::new(),
            receipts: BTreeMap::new(),
            ready: VecDeque::new(),
            causal_ordinal: 0,
            pending_close_reason: None,
            is_terminal_emitted: false,
            last_action: None,
            last_provenance: fragment.provenance.clone(),
            last_stable_order: fragment.stable_tie_break,
            last_source_position: fragment.source_position,
            first_source_position: fragment.source_position,
            last_event_time: fragment.event_time,
            version: 0,
        }
    }

    /// Return the stable session key joining fragments across partitions.
    #[must_use]
    pub const fn session_key(&self) -> StableSessionKey {
        self.session_key
    }

    /// Return the number of declared nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of inbound edges parked on undeclared targets.
    #[must_use]
    pub fn orphan_edge_count(&self) -> usize {
        self.orphan_edges.values().map(SmallVec::len).sum()
    }

    /// Derive the stable record identity of one producer node key.
    ///
    /// The session key is the namespace, so two producers in different input
    /// domains cannot collide on an identical authored key.
    #[must_use]
    pub fn node_record_id(&self, node_key: &str) -> StableRecordId {
        stable_record_id_from_key(self.session_key.as_bytes(), node_key.as_bytes())
    }

    /// Return one declared node's lifecycle state.
    #[must_use]
    pub fn node_state(&self, record_id: StableRecordId) -> Option<GraphNodeState> {
        self.nodes.get(&record_id).map(|node| node.state)
    }

    /// Return one declared node's outstanding declared-predecessor count.
    #[must_use]
    pub fn pending_predecessors(&self, record_id: StableRecordId) -> Option<usize> {
        self.nodes
            .get(&record_id)
            .map(|node| node.pending_predecessors)
    }

    /// Drain the released-but-unemitted node keys in release order.
    #[must_use]
    pub fn take_ready(&mut self) -> Vec<String> {
        let drained: Vec<StableRecordId> = self.ready.drain(..).collect();
        drained
            .into_iter()
            .filter_map(|record_id| self.nodes.get(&record_id).map(|node| node.node_key.clone()))
            .collect()
    }

    /// Whether every declared node has reached terminal.
    #[must_use]
    fn is_quiescent(&self) -> bool {
        self.nodes
            .values()
            .all(|node| matches!(node.state, GraphNodeState::Terminal))
    }

    /// Whether `target` is reachable from `start` over declared adjacency.
    ///
    /// The search terminates at the first witness and visits each node at most
    /// once, so it is bounded by the session's retained node budget.
    fn is_reachable(&self, start: StableRecordId, target: StableRecordId) -> bool {
        if start == target {
            return true;
        }
        let mut seen = BTreeSet::new();
        let mut frontier = vec![start];
        seen.insert(start);
        while let Some(current) = frontier.pop() {
            let Some(successors) = self.edges.get(&current) else {
                continue;
            };
            for successor in successors {
                if *successor == target {
                    return true;
                }
                if seen.insert(*successor) {
                    frontier.push(*successor);
                }
            }
        }
        false
    }

    /// Insert one edge, refusing a cycle or a post-execution dependency.
    ///
    /// Because the graph is append-only and every prior insertion left it
    /// acyclic, `from -> to` closes a cycle if and only if `from` is already
    /// reachable from `to`. Both refusals happen before any mutation, so a
    /// refused edge leaves the graph byte-identical.
    fn insert_edge(
        &mut self,
        from: StableRecordId,
        to: StableRecordId,
    ) -> Result<(), SessionCoordinatorError> {
        if self.is_reachable(to, from) {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::GraphCycle,
            ));
        }
        // An edge into a node the host already acted on would retroactively
        // change a dependency; refuse rather than emit a second, differently
        // caused action for one identity.
        if matches!(
            self.nodes.get(&to).map(|node| node.state),
            Some(GraphNodeState::Released | GraphNodeState::Terminal)
        ) {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::EdgeAfterExecution,
            ));
        }
        let successors = self.edges.entry(from).or_default();
        if successors.contains(&to) {
            // An identical redeclared edge is idempotent, not a second
            // dependency: counting it twice would strand the target forever.
            return Ok(());
        }
        successors.push(to);

<<<<<<< HEAD
        let is_source_terminal = matches!(
            self.nodes.get(&from).map(|source| source.state),
            Some(GraphNodeState::Terminal)
        );
        match self.nodes.get_mut(&to) {
            Some(node) => {
                // A predecessor that is already terminal never contributes a count.
                if !is_source_terminal {
                    node.pending_predecessors = node.pending_predecessors.saturating_add(1);
                }
            }
            None => {
                self.orphan_edges.entry(to).or_default().push(from);
=======
        // The terminal probe needs a shared borrow of the node map, so the
        // target's presence is resolved before any mutable borrow is taken.
        if self.nodes.contains_key(&to) {
            let is_source_terminal = matches!(
                self.nodes.get(&from).map(|source| source.state),
                Some(GraphNodeState::Terminal)
            );
            if !is_source_terminal && let Some(node) = self.nodes.get_mut(&to) {
                node.pending_predecessors = node.pending_predecessors.saturating_add(1);
>>>>>>> ajc/streaming-task-c3
            }
        }
        self.version = self.version.saturating_add(1);
        Ok(())
    }

    /// Declare one node and release it when it is already causally ready.
    fn declare_node(
        &mut self,
        record_id: StableRecordId,
        node_key: String,
        request: Vec<u8>,
        role: GraphNodeRole,
        lease: BudgetLease,
    ) -> Result<(), SessionCoordinatorError> {
        if self.nodes.contains_key(&record_id) {
            // Identity is immutable: a redeclaration is classified by the
            // coordinator's receipt map before it reaches here.
            return Ok(());
        }
        // Inbound edges parked while this node was hidden resolve here, and a
        // predecessor that is already terminal never contributes a count.
        let parked = self.orphan_edges.remove(&record_id).unwrap_or_default();
        let pending = parked
            .iter()
            .filter(|from| {
                !matches!(
                    self.nodes.get(*from).map(|node| node.state),
                    Some(GraphNodeState::Terminal)
                )
            })
            .count();
        self.nodes.insert(
            record_id,
            GraphSessionNode {
                node_key,
                request,
                role,
                pending_predecessors: pending,
                state: GraphNodeState::Waiting,
                action_id: None,
                lease,
            },
        );
        self.version = self.version.saturating_add(1);
        self.settle(record_id);
        Ok(())
    }

    /// Record one terminal and release every successor that became ready.
    fn mark_terminal(&mut self, record_id: StableRecordId) {
        let mut worklist = vec![record_id];
        while let Some(current) = worklist.pop() {
            match self.nodes.get_mut(&current) {
                Some(node) if !matches!(node.state, GraphNodeState::Terminal) => {
                    node.state = GraphNodeState::Terminal;
                }
                // A repeated terminal for one logical identity — a retried
                // attempt, or a replayed receipt — is absorbed, not counted
                // twice.
                _ => continue,
            }
            self.version = self.version.saturating_add(1);
            let Some(successors) = self.edges.get(&current).cloned() else {
                continue;
            };
            for successor in successors {
                let Some(node) = self.nodes.get_mut(&successor) else {
                    continue;
                };
                node.pending_predecessors = node.pending_predecessors.saturating_sub(1);
                if let Some(cascaded) = self.settle_returning_inert(successor) {
                    worklist.push(cascaded);
                }
            }
        }
    }

    /// Release a node whose declared predecessors are all terminal.
    fn settle(&mut self, record_id: StableRecordId) {
        if let Some(cascaded) = self.settle_returning_inert(record_id) {
            self.mark_terminal(cascaded);
        }
    }

    /// Release `record_id` when ready, returning it when it is inert.
    ///
    /// An inert node reaches terminal instead of dispatching, so its own
    /// terminal must cascade. Returning it rather than recursing keeps the
    /// cascade on the caller's explicit worklist.
    fn settle_returning_inert(&mut self, record_id: StableRecordId) -> Option<StableRecordId> {
        let node = self.nodes.get_mut(&record_id)?;
        if !matches!(node.state, GraphNodeState::Waiting) || node.pending_predecessors != 0 {
            return None;
        }
        match node.role {
            GraphNodeRole::InertRecorded => Some(record_id),
            GraphNodeRole::Llm => {
                node.state = GraphNodeState::Released;
                self.ready.push_back(record_id);
                None
            }
        }
    }
}

/// Validated startup-only configuration for the `agent_graph` program.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgentGraphProgramConfig {
    /// Maximum simultaneously live graph sessions.
    #[serde(default = "default_max_active_sessions")]
    pub max_active_sessions: usize,
    /// Maximum declared nodes retained for one session.
    #[serde(default = "default_max_nodes_per_session")]
    pub max_nodes_per_session: usize,
    /// Maximum inbound edges parked on undeclared targets for one session.
    #[serde(default = "default_max_orphan_edges")]
    pub max_orphan_edges_per_session: usize,
    /// Maximum retained authored request bytes for one session.
    #[serde(default = "default_max_request_bytes")]
    pub max_request_bytes_per_session: usize,
}

const fn default_max_active_sessions() -> usize {
    4096
}

const fn default_max_nodes_per_session() -> usize {
    4096
}

const fn default_max_orphan_edges() -> usize {
    4096
}

const fn default_max_request_bytes() -> usize {
    1 << 22
}

impl Default for AgentGraphProgramConfig {
    fn default() -> Self {
        Self {
            max_active_sessions: default_max_active_sessions(),
            max_nodes_per_session: default_max_nodes_per_session(),
            max_orphan_edges_per_session: default_max_orphan_edges(),
            max_request_bytes_per_session: default_max_request_bytes(),
        }
    }
}

impl AgentGraphProgramConfig {
    fn validate_limits(self) -> Result<Self, SessionCoordinatorError> {
        if self.max_active_sessions == 0
            || self.max_nodes_per_session == 0
            || self.max_orphan_edges_per_session == 0
            || self.max_request_bytes_per_session == 0
        {
            // A zero bound is an unbounded causality state with extra steps.
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::UnboundedCausalityState,
            ));
        }
        Ok(self)
    }
}

/// Startup factory for the `agent_graph` session program.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamingAgentGraphProgramFactory;

impl StreamingSessionProgramFactory for StreamingAgentGraphProgramFactory {
    fn descriptor(&self) -> &'static StreamingSessionProgramDescriptor {
        &AGENT_GRAPH_SESSION_PROGRAM
    }

    fn validate(
        &self,
        authored: &RawValue,
        format: &StreamingFormatDescriptor,
        _workload: &WorkloadDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingSessionProgramConfig>, SessionCoordinatorError> {
        if !AGENT_GRAPH_SESSION_PROGRAM
            .fragment_input_schemas
            .contains(&format.output_schema)
        {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::UnsupportedMutation,
            ));
        }
        let config: AgentGraphProgramConfig =
            serde_json::from_str(authored.get()).map_err(|_| {
                SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation)
            })?;
        Ok(Box::new(config.validate_limits()?))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSessionProgramConfig>,
        context: &StreamingSessionPrepareContext,
    ) -> Result<Box<dyn StreamingSessionCoordinator>, SessionCoordinatorError> {
        // The erased value must be reached through an explicit reborrow:
        // `config.as_any()` would erase the box rather than the value.
        let config = *ValidatedStreamingSessionProgramConfig::as_any(config.as_ref())
            .downcast_ref::<AgentGraphProgramConfig>()
            .ok_or_else(|| {
                SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation)
            })?;
        Ok(Box::new(StreamingAgentGraphCoordinator::new(
            config, context,
        )))
    }
}

/// Run-scoped owner of every live `agent_graph` session.
#[derive(Debug)]
pub struct StreamingAgentGraphCoordinator {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    program_semantic_digest: ContentDigest,
    stream_identity: ContentDigest,
    limits: AgentGraphProgramConfig,
    state_budget: StreamingResourceBudget,
    checkpoint_budget: StreamingResourceBudget,
    sessions: BTreeMap<StableSessionKey, GraphSessionScope>,
    in_flight: BTreeMap<StableActionId, (StableSessionKey, StableRecordId)>,
    initialization: ParticipantInitialization,
    causal_frontier: SessionCausalFrontier,
    next_global_sequence: u64,
    latest_event_time: Option<EventTimeUtc>,
}

impl StreamingAgentGraphCoordinator {
    /// Construct one run-scoped coordinator from a validated configuration.
    #[must_use]
    pub fn new(config: AgentGraphProgramConfig, context: &StreamingSessionPrepareContext) -> Self {
        Self {
            run: context.run,
            participant_id: context.participant_id.clone(),
            program_semantic_digest: context.program_semantic_digest,
            stream_identity: context.stream_semantic_digest,
            limits: config,
            state_budget: context.session_state_budget.clone(),
            checkpoint_budget: context.checkpoint_budget.clone(),
            sessions: BTreeMap::new(),
            in_flight: BTreeMap::new(),
            initialization: ParticipantInitialization::default(),
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(0),
                event_time: None,
                digest: ContentDigest::from_bytes([0; 32]),
            },
            next_global_sequence: 0,
            latest_event_time: None,
        }
    }

    /// Return the number of live graph sessions.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Borrow one session's append-only graph.
    #[must_use]
    pub fn scope(&self, session_key: StableSessionKey) -> Option<&GraphSessionScope> {
        self.sessions.get(&session_key)
    }

    /// Mutably borrow one session's append-only graph.
    pub fn scope_mut(&mut self, session_key: StableSessionKey) -> Option<&mut GraphSessionScope> {
        self.sessions.get_mut(&session_key)
    }

    /// Return the causal frontier this coordinator has proven complete.
    #[must_use]
    pub const fn causal_frontier(&self) -> &SessionCausalFrontier {
        &self.causal_frontier
    }

    async fn ingest_mutation(
        &mut self,
        fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        let session_key = fragment.session_key;
        // Vocabulary acceptance precedes every state change, so an unaccepted
        // mutation never opens a session.
        let accepted = AcceptedGraphMutation::classify(&fragment.mutation)?;
        let receipt = LogicalRecordReceipt {
            record_id: fragment.record_id,
            content_digest: accepted.canonical_content_digest(),
            provenance: fragment.provenance.clone(),
        };

        if let Some(scope) = self.sessions.get(&session_key)
            && let Some(existing) = scope.receipts.get(&fragment.record_id)
        {
            let disposition = classify_logical_duplicate(existing, &receipt).map_err(|_| {
                SessionCoordinatorError::session(SessionFailureCode::ConflictingMutation)
            })?;
            if matches!(disposition, DuplicateDisposition::Identical) {
                // Idempotent replay: dropping the fragment returns its charge.
                return Ok(());
            }
        }

        self.admit_session(session_key, &fragment)?;
        self.incorporate(session_key, &fragment, accepted, receipt)?;
        drop(fragment.lease);
        self.drain_ready(session_key, output).await
    }

    fn admit_session(
        &mut self,
        session_key: StableSessionKey,
        fragment: &StreamingSessionFragment,
    ) -> Result<(), SessionCoordinatorError> {
        if self.sessions.contains_key(&session_key) {
            return Ok(());
        }
        if self.sessions.len() >= self.limits.max_active_sessions {
            return Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            ));
        }
        self.sessions
            .insert(session_key, GraphSessionScope::new(session_key, fragment));
        Ok(())
    }

    fn incorporate(
        &mut self,
        session_key: StableSessionKey,
        fragment: &StreamingSessionFragment,
        accepted: AcceptedGraphMutation,
        receipt: LogicalRecordReceipt,
    ) -> Result<(), SessionCoordinatorError> {
        match accepted {
            AcceptedGraphMutation::Node {
                node_key,
                request,
                role,
            } => {
                self.declare_node(session_key, node_key, request, role)?;
            }
            AcceptedGraphMutation::Edge { from, to } => {
                // Read the authored bound before borrowing the session scope:
                // the limits live on `self`, which `session_mut` borrows.
                let max_orphan_edges = self.limits.max_orphan_edges_per_session;
                let scope = self.session_mut(session_key)?;
                let from_id = scope.node_record_id(&from);
                let to_id = scope.node_record_id(&to);
                if !scope.nodes.contains_key(&to_id)
                    && scope.orphan_edge_count() >= max_orphan_edges
                {
                    return Err(SessionCoordinatorError::state_budget(
                        StateBudgetFailureCode::ItemCapacity,
                    ));
                }
                scope.insert_edge(from_id, to_id)?;
            }
            AcceptedGraphMutation::Close { reason } => {
                let scope = self.session_mut(session_key)?;
                scope.pending_close_reason = Some(reason);
            }
        }

        let event_time = fragment.event_time;
        let scope = self.session_mut(session_key)?;
        scope.receipts.insert(fragment.record_id, receipt);
        scope.version = scope.version.saturating_add(1);
        scope.last_provenance = fragment.provenance.clone();
        scope.last_stable_order = fragment.stable_tie_break;
        scope.last_source_position = fragment.source_position;
        if event_time.is_some() {
            scope.last_event_time = event_time;
        }
        if fragment.source_position < scope.first_source_position {
            scope.first_source_position = fragment.source_position;
        }
        self.observe_event_time(event_time);
        Ok(())
    }

    fn declare_node(
        &mut self,
        session_key: StableSessionKey,
        node_key: String,
        request: Vec<u8>,
        role: GraphNodeRole,
    ) -> Result<(), SessionCoordinatorError> {
        let limits = self.limits;
        {
            let scope = self.session_mut(session_key)?;
            if scope.nodes.len() >= limits.max_nodes_per_session {
                return Err(SessionCoordinatorError::state_budget(
                    StateBudgetFailureCode::ItemCapacity,
                ));
            }
            if request.len() > limits.max_request_bytes_per_session {
                return Err(SessionCoordinatorError::state_budget(
                    StateBudgetFailureCode::ByteCapacity,
                ));
            }
        }
        let lease = self
            .state_budget
            .try_acquire(1, request.len().saturating_add(node_key.len()))
            .map_err(map_budget_error)?;
        let scope = self.session_mut(session_key)?;
        let record_id = scope.node_record_id(&node_key);
        scope.declare_node(record_id, node_key, request, role, lease)
    }

    fn session_mut(
        &mut self,
        session_key: StableSessionKey,
    ) -> Result<&mut GraphSessionScope, SessionCoordinatorError> {
        self.sessions
            .get_mut(&session_key)
            .ok_or_else(|| SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor))
    }

    async fn drain_ready(
        &mut self,
        session_key: StableSessionKey,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        loop {
            let Some(record_id) = self
                .sessions
                .get_mut(&session_key)
                .and_then(|scope| scope.ready.pop_front())
            else {
                break;
            };
            let action = self.build_graph_action(session_key, record_id)?;
            let action_id = action.action_id();
            output.send_action(action).await?;
            self.next_global_sequence = self.next_global_sequence.saturating_add(1);
            self.in_flight.insert(action_id, (session_key, record_id));
            let scope = self.session_mut(session_key)?;
            scope.last_action = Some(action_id);
            scope.causal_ordinal = scope.causal_ordinal.saturating_add(1);
            if let Some(node) = scope.nodes.get_mut(&record_id) {
                node.action_id = Some(action_id);
            }
        }
        self.settle_pending_close(session_key, output).await?;
        self.publish_frontier(output).await
    }

    /// Build one graph-node action over the node's authored request bytes.
    ///
    /// The identity is a pure function of program digest, session key,
    /// incorporated record causes, and the session's causal ordinal, so
    /// re-deriving it after a restart yields the identical [`StableActionId`]
    /// and a retried attempt never becomes a second graph node.
    fn build_graph_action(
        &self,
        session_key: StableSessionKey,
        record_id: StableRecordId,
    ) -> Result<ExecutableDatasetAction, SessionCoordinatorError> {
        let scope = self.sessions.get(&session_key).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        let node = scope.nodes.get(&record_id).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        let node_key = node.node_key.clone().into_boxed_str().into_string();
        let request = node.request.clone().into_boxed_slice().into_vec();
        let causes: Vec<StableRecordId> = scope.receipts.keys().copied().collect();
        let action_id = stable_action_id(
            self.program_semantic_digest.as_bytes(),
            session_key,
            &causes,
            DatasetActionKind::GraphNode,
            scope.causal_ordinal,
        );
        let predecessors: SmallVec<[StableActionId; 2]> = scope.last_action.into_iter().collect();
        let envelope = acquire_retained(
            &self.state_budget,
            node_key.len().saturating_add(request.len()),
        )?;
        ExecutableDatasetAction::new(
            action_id,
            session_key,
            predecessors,
            scope.last_event_time,
            scope.last_stable_order,
            scope.last_source_position,
            scope.last_provenance.clone(),
            DatasetActionV1::GraphNode(SessionGraphAction { node_key, request }),
            ActionContentLeaseSet::from_retained(envelope),
        )
        .map_err(map_budget_error)
    }

    async fn settle_pending_close(
        &mut self,
        session_key: StableSessionKey,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        let Some(scope) = self.sessions.get(&session_key) else {
            return Ok(());
        };
        // An authored close is terminal only once every declared node is
        // terminal: a node still waiting on a hidden parent is not closure.
        if scope.pending_close_reason.is_none()
            || scope.is_terminal_emitted
            || !scope.ready.is_empty()
            || !scope.is_quiescent()
        {
            return Ok(());
        }
        let action = self.build_terminal_action(session_key)?;
        output.send_action(action).await?;
        self.next_global_sequence = self.next_global_sequence.saturating_add(1);
        let scope = self.session_mut(session_key)?;
        scope.is_terminal_emitted = true;
        scope.pending_close_reason = None;
        Ok(())
    }

    fn build_terminal_action(
        &self,
        session_key: StableSessionKey,
    ) -> Result<ExecutableDatasetAction, SessionCoordinatorError> {
        let scope = self.sessions.get(&session_key).ok_or_else(|| {
            SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
        })?;
        let reason = scope
            .pending_close_reason
            .clone()
            .ok_or_else(|| {
                SessionCoordinatorError::session(SessionFailureCode::MissingPredecessor)
            })?
            .into_boxed_str()
            .into_string();
        let causes: Vec<StableRecordId> = scope.receipts.keys().copied().collect();
        let action_id = stable_action_id(
            self.program_semantic_digest.as_bytes(),
            session_key,
            &causes,
            DatasetActionKind::SessionTerminal,
            scope.causal_ordinal,
        );
        let predecessors: SmallVec<[StableActionId; 2]> = scope.last_action.into_iter().collect();
        let envelope = acquire_retained(&self.state_budget, reason.capacity())?;
        ExecutableDatasetAction::new(
            action_id,
            session_key,
            predecessors,
            scope.last_event_time,
            scope.last_stable_order,
            scope.last_source_position,
            scope.last_provenance.clone(),
            DatasetActionV1::SessionTerminal(SessionTerminalAction { reason }),
            ActionContentLeaseSet::from_retained(envelope),
        )
        .map_err(map_budget_error)
    }

    async fn apply_watermark(
        &mut self,
        watermark: SessionWatermark,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        // Format-proven completeness is recorded here; the closure policy that
        // consumes it belongs to the inferred-closure owner.
        self.observe_event_time(Some(watermark.through));
        self.publish_frontier(output).await
    }

    async fn apply_execution(
        &mut self,
        event: ActionExecutionEvent,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        match event {
            ActionExecutionEvent::Admitted(_)
            | ActionExecutionEvent::FirstToken(_)
            | ActionExecutionEvent::SessionUpdate(_) => Ok(()),
            ActionExecutionEvent::Terminal(receipt) => {
                // Keyed by the stable logical action, never the physical
                // attempt: a retried attempt resolves the same graph node, and
                // a second terminal for one action finds nothing to resolve.
                let Some((session_key, record_id)) =
                    self.in_flight.remove(&receipt.event.action_id)
                else {
                    return Ok(());
                };
                if let Some(scope) = self.sessions.get_mut(&session_key) {
                    scope.mark_terminal(record_id);
                }
                self.drain_ready(session_key, output).await
            }
        }
    }

    async fn seal_finite(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError> {
        self.publish_frontier(output).await?;
        Ok(SessionSealReceipt {
            digest: self.seal_digest(&seal),
            causal_frontier: self.causal_frontier.clone(),
        })
    }

    fn seal_digest(&self, seal: &SourceSeal) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.session.agent_graph.seal.v1");
        hasher.update(seal.digest.as_bytes());
        hasher.update(
            &seal
                .final_position
                .map_or(u64::MAX, SourcePosition::get)
                .to_le_bytes(),
        );
        hasher.update(self.stream_identity.as_bytes());
        hasher.update(&(self.sessions.len() as u64).to_le_bytes());
        for (session_key, scope) in &self.sessions {
            hasher.update(session_key.as_bytes());
            hasher.update(&scope.version.to_le_bytes());
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }

    async fn publish_frontier(
        &mut self,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.session.agent_graph.frontier.v1");
        hasher.update(&self.next_global_sequence.to_le_bytes());
        for (session_key, scope) in &self.sessions {
            hasher.update(session_key.as_bytes());
            hasher.update(&scope.version.to_le_bytes());
        }
        self.causal_frontier = SessionCausalFrontier {
            through_sequence: GlobalSequence::new(self.next_global_sequence),
            event_time: self.latest_event_time,
            digest: ContentDigest::from_bytes(*hasher.finalize().as_bytes()),
        };
        output
            .advance_causal_frontier(self.causal_frontier.clone())
            .await
    }

    fn observe_event_time(&mut self, event_time: Option<EventTimeUtc>) {
        let Some(candidate) = event_time else {
            return;
        };
        self.latest_event_time = Some(match self.latest_event_time {
            Some(existing) if existing >= candidate => existing,
            _ => candidate,
        });
    }

    /// Encode complete state, dropping whole sessions from the tail of the
    /// encode order until the payload fits the checkpoint budget.
    fn encode_within_budget(
        &mut self,
    ) -> Result<(Vec<u8>, BudgetLease, Option<SourcePosition>, u64), CheckpointError> {
        let mut records: Vec<AgentGraphSessionRecordV1> =
            self.sessions.values().map(record_of).collect();
        let mut first_unrepresented: Option<SourcePosition> = None;
        loop {
            let state = AgentGraphCheckpointStateV1 {
                program_semantic_digest: self.program_semantic_digest,
                stream_identity: self.stream_identity,
                sessions: records,
            };
            let bytes = rmp_serde::to_vec(&state).map_err(|error| CheckpointError::Storage {
                message: format!("could not encode agent-graph session state: {error}"),
            })?;
            match self.checkpoint_budget.try_acquire(1, bytes.len()) {
                Ok(lease) => {
                    let item_count = u64::try_from(state.sessions.len()).unwrap_or(u64::MAX);
                    return Ok((bytes, lease, first_unrepresented, item_count));
                }
                Err(BudgetError::Closed) => {
                    return Err(CheckpointError::ParticipantUnavailable {
                        participant: self.participant_id.clone(),
                    });
                }
                Err(_) => {}
            }
            records = state.sessions;
            let Some(dropped) = records.pop() else {
                return Err(CheckpointError::StateBudget {
                    participant: self.participant_id.clone(),
                    code: StateBudgetFailureCode::ByteCapacity,
                });
            };
            first_unrepresented = Some(match first_unrepresented {
                Some(existing) if existing <= dropped.first_source_position => existing,
                _ => dropped.first_source_position,
            });
        }
    }

    fn prepare_complete_or_rolled_back_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let (bytes, lease, first_unrepresented, item_count) = self.encode_within_budget()?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(bytes), lease)?;
        let mut represented = barrier.cut.clone();
        if let Some(position) = first_unrepresented {
            represented.decoded = crate::streaming::checkpoint::DecodeHorizon::new(position);
        }
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            AGENT_GRAPH_CHECKPOINT_SCHEMA_ID,
            AGENT_GRAPH_CHECKPOINT_SCHEMA_VERSION,
            represented,
            item_count,
            payload,
        )
    }

    fn restore_sessions(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        let Some(state) = state else {
            return Ok(());
        };
        let decoded: AgentGraphCheckpointStateV1 = rmp_serde::from_slice(state.payload_bytes())
            .map_err(|error| CheckpointError::Storage {
                message: format!("could not decode agent-graph session state: {error}"),
            })?;
        if decoded.program_semantic_digest != self.program_semantic_digest
            || decoded.stream_identity != self.stream_identity
        {
            return Err(CheckpointError::ObjectVerification);
        }
        for record in decoded.sessions {
            let mut nodes = BTreeMap::new();
            for node in record.nodes {
                let lease = self
                    .state_budget
                    .try_acquire(1, node.request.len().saturating_add(node.node_key.len()))
                    .map_err(|_| CheckpointError::StateBudget {
                        participant: self.participant_id.clone(),
                        code: StateBudgetFailureCode::ByteCapacity,
                    })?;
                nodes.insert(
                    node.record_id,
                    GraphSessionNode {
                        node_key: node.node_key,
                        request: node.request,
                        role: node.role,
                        pending_predecessors: node.pending_predecessors,
                        state: node.state,
                        action_id: node.action_id,
                        lease,
                    },
                );
            }
            let edges = record
                .edges
                .into_iter()
                .map(|edge| (edge.from, edge.to.into_iter().collect()))
                .collect();
            let orphan_edges = record
                .orphan_edges
                .into_iter()
                .map(|edge| (edge.from, edge.to.into_iter().collect()))
                .collect();
            let receipts = record
                .receipts
                .into_iter()
                .map(|receipt| (receipt.record_id, receipt))
                .collect();
            // A node released before the cut is already emitted; only the
            // still-unemitted ready order is restored, so restart never
            // re-emits a node the host already admitted.
            self.sessions.insert(
                record.session_key,
                GraphSessionScope {
                    session_key: record.session_key,
                    nodes,
                    edges,
                    orphan_edges,
                    receipts,
                    ready: record.ready.into_iter().collect(),
                    causal_ordinal: record.causal_ordinal,
                    pending_close_reason: record.pending_close_reason,
                    is_terminal_emitted: record.is_terminal_emitted,
                    last_action: record.last_action,
                    last_provenance: record.last_provenance,
                    last_stable_order: record.last_stable_order,
                    last_source_position: record.last_source_position,
                    first_source_position: record.first_source_position,
                    last_event_time: record.last_event_time,
                    version: record.version,
                },
            );
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingSessionCoordinator for StreamingAgentGraphCoordinator {
    async fn ingest(
        &mut self,
        fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.ingest_mutation(fragment, output).await
    }

    async fn advance_watermark(
        &mut self,
        watermark: SessionWatermark,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.apply_watermark(watermark, output).await
    }

    async fn observe_execution(
        &mut self,
        event: ActionExecutionEvent,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.apply_execution(event, output).await
    }

    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError> {
        self.seal_finite(seal, output).await
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for StreamingAgentGraphCoordinator {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        self.prepare_complete_or_rolled_back_view(barrier)
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.restore_sessions(state)
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        // Retained state is the live state; there is no separate pre-cut copy
        // to release, so the notification is idempotent by construction.
        let _ = receipt;
        Ok(())
    }
}

/// Mutation vocabulary this program accepts, already projected to its content.
#[derive(Debug)]
enum AcceptedGraphMutation {
    Node {
        node_key: String,
        request: Vec<u8>,
        role: GraphNodeRole,
    },
    Edge {
        from: String,
        to: String,
    },
    Close {
        reason: String,
    },
}

impl AcceptedGraphMutation {
    fn classify(mutation: &SessionMutationV1) -> Result<Self, SessionCoordinatorError> {
        match mutation {
            SessionMutationV1::GraphNode(node) => Ok(Self::Node {
                node_key: node.node_key.clone(),
                request: node.request.clone(),
                role: GraphNodeRole::Llm,
            }),
            SessionMutationV1::GraphEdge(edge) => Ok(Self::Edge {
                from: edge.from.clone(),
                to: edge.to.clone(),
            }),
            // A recorded agent event is retained as inert graph state: it
            // orders successors but never reaches an endpoint.
            SessionMutationV1::AgentEvent(event) => Ok(Self::Node {
                node_key: format!("{}#{}", event.event_kind, event.event_ordinal),
                request: event.payload.clone(),
                role: GraphNodeRole::InertRecorded,
            }),
            SessionMutationV1::SessionClose(close) => Ok(Self::Close {
                reason: close.reason.clone(),
            }),
            // The `conversation` program owns these; neither program silently
            // ignores the other's fragments.
            SessionMutationV1::ConversationTurn(_)
            | SessionMutationV1::DeferredRecordedRequest(_) => Err(
                SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation),
            ),
        }
    }

    fn canonical_content_digest(&self) -> ContentDigest {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.stream.session.agent_graph.content.v1");
        match self {
            Self::Node {
                node_key,
                request,
                role,
            } => {
                hasher.update(&[0u8]);
                hasher.update(&[u8::from(matches!(role, GraphNodeRole::InertRecorded))]);
                hasher.update(&(node_key.len() as u64).to_le_bytes());
                hasher.update(node_key.as_bytes());
                hasher.update(&(request.len() as u64).to_le_bytes());
                hasher.update(request);
            }
            Self::Edge { from, to } => {
                hasher.update(&[1u8]);
                hasher.update(&(from.len() as u64).to_le_bytes());
                hasher.update(from.as_bytes());
                hasher.update(&(to.len() as u64).to_le_bytes());
                hasher.update(to.as_bytes());
            }
            Self::Close { reason } => {
                hasher.update(&[2u8]);
                hasher.update(&(reason.len() as u64).to_le_bytes());
                hasher.update(reason.as_bytes());
            }
        }
        ContentDigest::from_bytes(*hasher.finalize().as_bytes())
    }
}

fn acquire_retained(
    budget: &StreamingResourceBudget,
    bytes: usize,
) -> Result<crate::streaming::unit::RetainedContentLease, SessionCoordinatorError> {
    let lease = budget.try_acquire(1, bytes).map_err(map_budget_error)?;
    Ok(SessionFragmentLease::try_from(lease)
        .map_err(map_budget_error)?
        .into_retained())
}

const fn map_budget_error(error: BudgetError) -> SessionCoordinatorError {
    match error {
        BudgetError::CapacityUnavailable
        | BudgetError::RequestExceedsCapacity
        | BudgetError::ActionPayloadUndercharged { .. } => {
            SessionCoordinatorError::state_budget(StateBudgetFailureCode::ByteCapacity)
        }
        _ => SessionCoordinatorError::state_budget(StateBudgetFailureCode::ItemCapacity),
    }
}

fn record_of(scope: &GraphSessionScope) -> AgentGraphSessionRecordV1 {
    AgentGraphSessionRecordV1 {
        session_key: scope.session_key,
        nodes: scope
            .nodes
            .iter()
            .map(|(record_id, node)| AgentGraphNodeRecordV1 {
                record_id: *record_id,
                node_key: node.node_key.clone(),
                request: node.request.clone(),
                role: node.role,
                pending_predecessors: node.pending_predecessors,
                state: node.state,
                action_id: node.action_id,
            })
            .collect(),
        edges: scope
            .edges
            .iter()
            .map(|(from, to)| AgentGraphAdjacencyRecordV1 {
                from: *from,
                to: to.to_vec(),
            })
            .collect(),
        orphan_edges: scope
            .orphan_edges
            .iter()
            .map(|(to, from)| AgentGraphAdjacencyRecordV1 {
                from: *to,
                to: from.to_vec(),
            })
            .collect(),
        receipts: scope.receipts.values().cloned().collect(),
        ready: scope.ready.iter().copied().collect(),
        causal_ordinal: scope.causal_ordinal,
        pending_close_reason: scope.pending_close_reason.clone(),
        is_terminal_emitted: scope.is_terminal_emitted,
        last_action: scope.last_action,
        last_provenance: scope.last_provenance.clone(),
        last_stable_order: scope.last_stable_order,
        last_source_position: scope.last_source_position,
        first_source_position: scope.first_source_position,
        last_event_time: scope.last_event_time,
        version: scope.version,
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentGraphCheckpointStateV1 {
    program_semantic_digest: ContentDigest,
    stream_identity: ContentDigest,
    sessions: Vec<AgentGraphSessionRecordV1>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentGraphSessionRecordV1 {
    session_key: StableSessionKey,
    nodes: Vec<AgentGraphNodeRecordV1>,
    edges: Vec<AgentGraphAdjacencyRecordV1>,
    orphan_edges: Vec<AgentGraphAdjacencyRecordV1>,
    receipts: Vec<LogicalRecordReceipt>,
    ready: Vec<StableRecordId>,
    causal_ordinal: u64,
    pending_close_reason: Option<String>,
    is_terminal_emitted: bool,
    last_action: Option<StableActionId>,
    last_provenance: UnitProvenance,
    last_stable_order: StableOrderKey,
    last_source_position: SourcePosition,
    first_source_position: SourcePosition,
    last_event_time: Option<EventTimeUtc>,
    version: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentGraphNodeRecordV1 {
    record_id: StableRecordId,
    node_key: String,
    request: Vec<u8>,
    role: GraphNodeRole,
    pending_predecessors: usize,
    state: GraphNodeState,
    action_id: Option<StableActionId>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentGraphAdjacencyRecordV1 {
    from: StableRecordId,
    to: Vec<StableRecordId>,
}
