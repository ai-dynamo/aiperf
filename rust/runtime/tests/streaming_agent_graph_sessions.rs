// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Incremental append-only graph state for the `agent_graph` session program.
//!
//! Nodes and edges arrive in any order and across arbitrary partition
//! boundaries. A node whose declared predecessor has not been seen yet is
//! retained rather than released, an edge that would close a cycle or that
//! names an already-released target is refused at ingest, and a recorded agent
//! event is retained as inert state that orders successors without ever
//! reaching an endpoint.
#![cfg(feature = "streaming")]

#[path = "support/streaming_session_conformance.rs"]
mod support;

use aiperf_runtime::streaming::{
    action::{
        ActionEventIdentity, ActionExecutionEvent, ActionTerminalDisposition, ActionTerminalReceipt,
    },
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        CheckpointParticipantId, StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::{
        SessionCoordinatorError, SessionFailureCode, StreamingIssueReporter,
        StreamingIssueReporterHandle,
    },
    identity::{
        ActionAttemptId, ContentDigest, ImmutableObjectIdentity, LogicalReplayRunId,
        SessionOwnershipEpoch, StableActionId, StableOrderKey, stable_record_id_from_key,
        stable_session_key,
    },
    reliability::{
        BudgetOwnedStreamingIssueReporter, PreparedStreamingIssuePolicy, StreamingIssueClass,
        StreamingIssueComponentId, StreamingIssueDisposition, StreamingIssueScopeKind,
        StreamingIssueThresholdRule, submission_queue_charge_bytes,
    },
    session::{
        StreamingSessionCoordinator, StreamingSessionPrepareContext,
        agent_graph::{
            AgentGraphProgramConfig, GraphNodeState, StreamingAgentGraphCoordinator,
        },
    },
    unit::{
        AgentEventFragment, ConversationTurnFragment, EventTimeUtc, GraphEdgeFragment,
        GraphNodeFragment, SessionCloseFragment, SessionFragmentLease, SessionMutationV1,
        SourcePosition, StreamingSessionFragment, UnitProvenance,
    },
};
use smallvec::SmallVec;
use support::{CapturedActionPayload, CollectingActionSink};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const NAMESPACE: &[u8] = b"aiperf.test.agent_graph";
const SESSION: &str = "trace-1";

fn run_id() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x21; 32]))
}

fn program_digest() -> ContentDigest {
    ContentDigest::from_bytes([0x1a; 32])
}

fn stream_digest() -> ContentDigest {
    ContentDigest::from_bytes([0x1b; 32])
}

fn format_digest() -> ContentDigest {
    ContentDigest::from_bytes([0x1c; 32])
}

fn budget(items: usize, bytes: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: items,
        max_bytes: bytes,
    })
    .expect("valid harness budget")
}

fn reporter() -> BudgetOwnedStreamingIssueReporter {
    let reporter_budget = budget(65, submission_queue_charge_bytes() + 64 * 1024);
    let policy = PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
        StreamingIssueComponentId::new("record_default").expect("valid rule ID"),
        StreamingIssueScopeKind::Record,
        StreamingIssueClass::Permanent,
        None,
        0,
        StreamingIssueDisposition::Quarantine,
        None,
    )
    .expect("valid record rule")])
    .expect("valid record policy");
    BudgetOwnedStreamingIssueReporter::new(run_id(), policy, reporter_budget)
        .expect("budget-owned reporter")
}

fn prepare_context(
    state_budget: &StreamingResourceBudget,
    checkpoint_budget: &StreamingResourceBudget,
    issue_reporter: StreamingIssueReporterHandle,
) -> StreamingSessionPrepareContext {
    StreamingSessionPrepareContext {
        program_semantic_digest: program_digest(),
        run: run_id(),
        participant_id: CheckpointParticipantId::new("session_coordinator"),
        stream_semantic_digest: stream_digest(),
        source_identity: ImmutableObjectIdentity::from_bytes([0x40; 32]),
        session_state_budget: state_budget.clone(),
        checkpoint_budget: checkpoint_budget.clone(),
        issue_reporter,
    }
}

/// Build one coordinator plus the reporter whose handle it borrows.
///
/// The reporter is returned alongside so it outlives the injected handle.
fn coordinator() -> (
    StreamingAgentGraphCoordinator,
    BudgetOwnedStreamingIssueReporter,
    StreamingResourceBudget,
    StreamingResourceBudget,
) {
    let state_budget = budget(4096, 4_194_304);
    let checkpoint_budget = budget(64, 4_194_304);
    let reporter = reporter();
    let context = prepare_context(&state_budget, &checkpoint_budget, reporter.handle());
    (
        StreamingAgentGraphCoordinator::new(AgentGraphProgramConfig::default(), &context),
        reporter,
        state_budget,
        checkpoint_budget,
    )
}

fn provenance(position: u64) -> UnitProvenance {
    UnitProvenance {
        source_partition: ImmutableObjectIdentity::from_bytes([0x40; 32]),
        source_position: SourcePosition::new(position),
        format_semantic_digest: format_digest(),
    }
}

fn fragment(
    state_budget: &StreamingResourceBudget,
    producer_record: &str,
    partition: u8,
    position: u64,
    mutation: SessionMutationV1,
    charged_bytes: usize,
) -> StreamingSessionFragment {
    let lease = state_budget
        .try_acquire(1, charged_bytes.max(1))
        .expect("fragment charge fits the harness budget");
    StreamingSessionFragment {
        record_id: stable_record_id_from_key(NAMESPACE, producer_record.as_bytes()),
        session_key: stable_session_key(NAMESPACE, SESSION.as_bytes()),
        source_position: SourcePosition::new(position),
        source_partition: ImmutableObjectIdentity::from_bytes([partition; 32]),
        event_time: Some(
            EventTimeUtc::new(i64::try_from(position).unwrap_or(0) + 1).expect("event time"),
        ),
        stable_tie_break: StableOrderKey::from_bytes([partition; 32]),
        predecessors: SmallVec::new(),
        mutation,
        provenance: provenance(position),
        lease: SessionFragmentLease::try_from(lease).expect("one-item fragment charge"),
    }
}

fn node(
    state_budget: &StreamingResourceBudget,
    producer_record: &str,
    partition: u8,
    position: u64,
    node_key: &str,
    request: &str,
) -> StreamingSessionFragment {
    fragment(
        state_budget,
        producer_record,
        partition,
        position,
        SessionMutationV1::GraphNode(GraphNodeFragment {
            node_key: node_key.to_string(),
            request: request.as_bytes().to_vec(),
        }),
        request.len(),
    )
}

fn edge(
    state_budget: &StreamingResourceBudget,
    producer_record: &str,
    partition: u8,
    position: u64,
    from: &str,
    to: &str,
) -> StreamingSessionFragment {
    fragment(
        state_budget,
        producer_record,
        partition,
        position,
        SessionMutationV1::GraphEdge(GraphEdgeFragment {
            from: from.to_string(),
            to: to.to_string(),
        }),
        from.len() + to.len(),
    )
}

fn agent_event(
    state_budget: &StreamingResourceBudget,
    producer_record: &str,
    partition: u8,
    position: u64,
    event_kind: &str,
    event_ordinal: u64,
    payload: &str,
) -> StreamingSessionFragment {
    fragment(
        state_budget,
        producer_record,
        partition,
        position,
        SessionMutationV1::AgentEvent(AgentEventFragment {
            event_kind: event_kind.to_string(),
            payload: payload.as_bytes().to_vec(),
            event_ordinal,
        }),
        payload.len(),
    )
}

fn close(
    state_budget: &StreamingResourceBudget,
    producer_record: &str,
    position: u64,
    reason: &str,
) -> StreamingSessionFragment {
    fragment(
        state_budget,
        producer_record,
        1,
        position,
        SessionMutationV1::SessionClose(SessionCloseFragment {
            reason: reason.to_string(),
        }),
        reason.len(),
    )
}

fn terminal_event(action_id: StableActionId, attempt: u8) -> ActionExecutionEvent {
    ActionExecutionEvent::Terminal(ActionTerminalReceipt {
        event: ActionEventIdentity {
            action_id,
            attempt_id: ActionAttemptId::from_bytes([attempt; 32]),
            ownership_epoch: SessionOwnershipEpoch::new(0),
            event_ordinal: 0,
        },
        disposition: ActionTerminalDisposition::Completed,
    })
}

/// Return the emitted graph-node keys in emission order.
fn emitted_nodes(sink: &CollectingActionSink) -> Vec<String> {
    sink.actions
        .iter()
        .filter_map(|action| match &action.payload {
            CapturedActionPayload::GraphNode(node_key) => Some(node_key.clone()),
            _ => None,
        })
        .collect()
}

fn state_of(coordinator: &StreamingAgentGraphCoordinator, node_key: &str) -> Option<GraphNodeState> {
    let session_key = stable_session_key(NAMESPACE, SESSION.as_bytes());
    let scope = coordinator.scope(session_key)?;
    scope.node_state(scope.node_record_id(node_key))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// An edge whose target is undeclared parks, resolves on declaration, and the
/// child releases only after its now-visible parent reaches terminal.
#[tokio::test(flavor = "current_thread")]
async fn hidden_parent_edge_parks_then_gates_the_child_across_chunks() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();
    let session_key = stable_session_key(NAMESPACE, SESSION.as_bytes());

    // Chunk one carries only the edge: neither endpoint is declared yet.
    coordinator
        .ingest(
            edge(&state_budget, "e-1", 1, 0, "parent", "child"),
            &mut sink,
        )
        .await
        .expect("parked edge is accepted");
    assert_eq!(
        coordinator
            .scope(session_key)
            .expect("session opened")
            .orphan_edge_count(),
        1,
        "an edge naming an undeclared target parks rather than being dropped"
    );
    assert!(emitted_nodes(&sink).is_empty());

    // Chunk two declares the child. Its parked inbound edge resolves into one
    // outstanding predecessor, so the child must not release.
    coordinator
        .ingest(
            node(&state_budget, "n-child", 2, 1, "child", "ask-child"),
            &mut sink,
        )
        .await
        .expect("child declaration is accepted");
    let scope = coordinator.scope(session_key).expect("session");
    assert_eq!(scope.orphan_edge_count(), 0, "the parked edge is consumed");
    assert_eq!(scope.pending_predecessors(scope.node_record_id("child")), Some(1));
    assert!(
        emitted_nodes(&sink).is_empty(),
        "a child owing a predecessor is retained, not released"
    );

    // Chunk three declares the parent, which owes nothing and releases.
    coordinator
        .ingest(
            node(&state_budget, "n-parent", 3, 2, "parent", "ask-parent"),
            &mut sink,
        )
        .await
        .expect("parent declaration is accepted");
    assert_eq!(emitted_nodes(&sink), ["parent"]);
    assert_eq!(state_of(&coordinator, "child"), Some(GraphNodeState::Waiting));

    let parent_action = sink.actions[0].action_id;
    coordinator
        .observe_execution(terminal_event(parent_action, 0x01), &mut sink)
        .await
        .expect("parent terminal is accepted");
    assert_eq!(emitted_nodes(&sink), ["parent", "child"]);
    assert_eq!(
        state_of(&coordinator, "parent"),
        Some(GraphNodeState::Terminal)
    );
}

/// A second edge closing a loop over declared adjacency is refused before it
/// mutates the graph.
#[tokio::test(flavor = "current_thread")]
async fn cycle_closing_edge_is_refused_and_state_is_unchanged() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();
    let session_key = stable_session_key(NAMESPACE, SESSION.as_bytes());

    coordinator
        .ingest(edge(&state_budget, "e-1", 1, 0, "left", "right"), &mut sink)
        .await
        .expect("first edge is accepted");
    let parked_before = coordinator
        .scope(session_key)
        .expect("session")
        .orphan_edge_count();

    let error = coordinator
        .ingest(edge(&state_budget, "e-2", 1, 1, "right", "left"), &mut sink)
        .await
        .expect_err("the reverse edge closes a cycle");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::GraphCycle)
    );
    assert_eq!(
        coordinator
            .scope(session_key)
            .expect("session")
            .orphan_edge_count(),
        parked_before,
        "a refused edge leaves parked adjacency unchanged"
    );
}

/// An edge naming a target the host already acted on is refused rather than
/// retroactively changing that node's dependencies.
#[tokio::test(flavor = "current_thread")]
async fn edge_into_released_node_is_refused_and_state_is_unchanged() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();

    coordinator
        .ingest(
            node(&state_budget, "n-root", 1, 0, "root", "ask-root"),
            &mut sink,
        )
        .await
        .expect("root declaration is accepted");
    assert_eq!(emitted_nodes(&sink), ["root"]);
    assert_eq!(state_of(&coordinator, "root"), Some(GraphNodeState::Released));

    let error = coordinator
        .ingest(edge(&state_budget, "e-1", 1, 1, "late", "root"), &mut sink)
        .await
        .expect_err("an edge into a released node is refused");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::EdgeAfterExecution)
    );
    assert_eq!(state_of(&coordinator, "root"), Some(GraphNodeState::Released));
    assert_eq!(
        emitted_nodes(&sink),
        ["root"],
        "the refusal emits no second action for one identity"
    );
}

/// A recorded agent event is inert: it never dispatches, yet it still reaches
/// terminal and releases the successors that declared it as a predecessor.
#[tokio::test(flavor = "current_thread")]
async fn recorded_agent_event_is_inert_and_still_releases_successors() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();

    coordinator
        .ingest(
            edge(&state_budget, "e-1", 1, 0, "shell#7", "child"),
            &mut sink,
        )
        .await
        .expect("parked edge is accepted");
    coordinator
        .ingest(
            agent_event(&state_budget, "a-1", 1, 1, "shell", 7, "recorded-output"),
            &mut sink,
        )
        .await
        .expect("recorded agent event is accepted");
    assert!(
        emitted_nodes(&sink).is_empty(),
        "an inert node dispatches nothing"
    );
    assert_eq!(
        state_of(&coordinator, "shell#7"),
        Some(GraphNodeState::Terminal),
        "inertness is terminal at declaration, not a dropped node"
    );

    coordinator
        .ingest(
            node(&state_budget, "n-child", 2, 2, "child", "ask-child"),
            &mut sink,
        )
        .await
        .expect("child declaration is accepted");
    assert_eq!(
        emitted_nodes(&sink),
        ["child"],
        "a successor of a terminal inert node owes nothing"
    );
}

/// Two attempts of one logical action collapse to a single node terminal.
#[tokio::test(flavor = "current_thread")]
async fn retry_attempts_share_one_logical_action() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();

    coordinator
        .ingest(edge(&state_budget, "e-1", 1, 0, "root", "leaf"), &mut sink)
        .await
        .expect("parked edge is accepted");
    coordinator
        .ingest(
            node(&state_budget, "n-leaf", 1, 1, "leaf", "ask-leaf"),
            &mut sink,
        )
        .await
        .expect("leaf declaration is accepted");
    coordinator
        .ingest(
            node(&state_budget, "n-root", 1, 2, "root", "ask-root"),
            &mut sink,
        )
        .await
        .expect("root declaration is accepted");
    let root_action = sink.actions[0].action_id;

    coordinator
        .observe_execution(terminal_event(root_action, 0x01), &mut sink)
        .await
        .expect("first attempt terminal is accepted");
    coordinator
        .observe_execution(terminal_event(root_action, 0x02), &mut sink)
        .await
        .expect("a retried attempt under one action id is absorbed");

    assert_eq!(
        emitted_nodes(&sink),
        ["root", "leaf"],
        "the second attempt releases no second successor action"
    );
}

/// The other program's vocabulary is refused before any session is opened.
#[tokio::test(flavor = "current_thread")]
async fn conversation_turn_mutation_is_refused_before_state_change() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();

    let error = coordinator
        .ingest(
            fragment(
                &state_budget,
                "t-1",
                1,
                0,
                SessionMutationV1::ConversationTurn(ConversationTurnFragment {
                    role: "user".to_string(),
                    content: b"hello".to_vec(),
                    turn_ordinal: 0,
                }),
                5,
            ),
            &mut sink,
        )
        .await
        .expect_err("a conversation turn is not this program's vocabulary");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation)
    );
    assert_eq!(coordinator.active_session_count(), 0);
}

/// An authored close publishes its terminal only once every declared node is
/// terminal, and exactly once.
#[tokio::test(flavor = "current_thread")]
async fn session_close_is_terminal_only_after_every_declared_node() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();

    coordinator
        .ingest(
            node(&state_budget, "n-root", 1, 0, "root", "ask-root"),
            &mut sink,
        )
        .await
        .expect("root declaration is accepted");
    coordinator
        .ingest(close(&state_budget, "c-1", 1, "producer-close"), &mut sink)
        .await
        .expect("close is accepted");
    assert!(
        sink.terminal_actions().is_empty(),
        "an outstanding node still owes an action"
    );

    let root_action = sink.actions[0].action_id;
    coordinator
        .observe_execution(terminal_event(root_action, 0x01), &mut sink)
        .await
        .expect("root terminal is accepted");
    let terminals = sink.terminal_actions();
    assert_eq!(terminals.len(), 1);
    assert_eq!(terminals[0].payload, CapturedActionPayload::Terminal("producer-close".to_string()));
}

/// A same-identity redeclaration is idempotent; conflicting content under one
/// identity is refused.
#[tokio::test(flavor = "current_thread")]
async fn duplicate_node_declaration_is_idempotent_and_conflict_is_refused() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();

    coordinator
        .ingest(
            node(&state_budget, "n-root", 1, 0, "root", "ask-root"),
            &mut sink,
        )
        .await
        .expect("root declaration is accepted");
    coordinator
        .ingest(
            node(&state_budget, "n-root", 1, 0, "root", "ask-root"),
            &mut sink,
        )
        .await
        .expect("an identical redeclaration is idempotent");
    assert_eq!(emitted_nodes(&sink), ["root"]);

    let error = coordinator
        .ingest(
            node(&state_budget, "n-root", 1, 0, "root", "ask-something-else"),
            &mut sink,
        )
        .await
        .expect_err("one record identity cannot carry conflicting content");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::ConflictingMutation)
    );
}

/// A checkpoint round trip preserves outstanding predecessor counts, parked
/// adjacency, and node lifecycle, so a restarted run resumes the same graph.
#[tokio::test(flavor = "current_thread")]
async fn checkpoint_restore_preserves_waiting_graph_state() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = coordinator();
    let mut sink = CollectingActionSink::default();
    let session_key = stable_session_key(NAMESPACE, SESSION.as_bytes());

    coordinator
        .ingest(edge(&state_budget, "e-1", 1, 0, "root", "leaf"), &mut sink)
        .await
        .expect("parked edge is accepted");
    coordinator
        .ingest(
            edge(&state_budget, "e-2", 1, 1, "hidden", "leaf"),
            &mut sink,
        )
        .await
        .expect("second parked edge is accepted");
    coordinator
        .ingest(
            node(&state_budget, "n-leaf", 1, 2, "leaf", "ask-leaf"),
            &mut sink,
        )
        .await
        .expect("leaf declaration is accepted");
    coordinator
        .ingest(
            edge(&state_budget, "e-3", 1, 3, "leaf", "tail"),
            &mut sink,
        )
        .await
        .expect("outbound edge onto an undeclared tail is accepted");

    let prepared = coordinator
        .checkpoint_view(&support::barrier_at(run_id(), 4))
        .await
        .expect("prepared participant state");
    let committed = support::commit_and_restore(prepared).await;

    let restored_state = budget(4096, 4_194_304);
    let restored_checkpoint = budget(64, 4_194_304);
    let restored_reporter = reporter();
    let context = prepare_context(
        &restored_state,
        &restored_checkpoint,
        restored_reporter.handle(),
    );
    let mut restored =
        StreamingAgentGraphCoordinator::new(AgentGraphProgramConfig::default(), &context);
    restored
        .initialize(Some(committed))
        .await
        .expect("restore committed agent-graph state");

    let scope = restored.scope(session_key).expect("restored session");
    let leaf = scope.node_record_id("leaf");
    assert_eq!(scope.node_count(), 1);
    assert_eq!(scope.node_state(leaf), Some(GraphNodeState::Waiting));
    assert_eq!(
        scope.pending_predecessors(leaf),
        Some(2),
        "both outstanding predecessors survive the cut"
    );
    assert_eq!(
        scope.orphan_edge_count(),
        1,
        "the edge onto the still-undeclared tail survives the cut"
    );
}
