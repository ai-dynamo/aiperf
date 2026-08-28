// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-chunk conversation continuity for the `conversation` session program.
//!
//! A conversation is joined by its stable session key across arbitrary source
//! partitions, survives a checkpoint and restart under a stable action
//! identity, and folds endpoint replies into the same durable transcript as
//! authored turns.
#![cfg(feature = "streaming")]

#[path = "support/streaming_session_conformance.rs"]
mod support;

use aiperf_runtime::engine::registry::WorkloadDescriptor;
use aiperf_runtime::streaming::{
    action::{
        ActionEventIdentity, ActionExecutionEvent, ActionTerminalDisposition,
        ActionTerminalReceipt, BudgetedActionUpdate, EndpointSessionUpdate,
    },
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        CheckpointParticipantId, PreparedParticipantState, StreamRunIdentity,
        StreamingCheckpointParticipant,
    },
    failure::{
        SessionCoordinatorError, SessionFailureCode, StreamingIssueReporter,
        StreamingIssueReporterHandle,
    },
    format::{FormatProjection, FormatStateRetention, SessionWatermark, StreamingFormatDescriptor},
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
        StreamingSessionProgramFactory,
        conversation::{
            ConversationProgramConfig, ConversationSessionScope, StreamingConversationCoordinator,
            StreamingConversationProgramFactory, TranscriptOrigin,
        },
    },
    source::{PartitionAccessKind, SourceSeal},
    unit::{
        ConversationTurnFragment, EventTimeUtc, GraphNodeFragment, SessionCloseFragment,
        SessionFragmentLease, SessionMutationV1, SourcePosition, StreamingSessionFragment,
        UnitProvenance,
    },
};
use bytes::Bytes;
use serde_json::value::RawValue;
use smallvec::SmallVec;
use std::rc::Rc;
use support::{CollectingActionSink, SessionConformanceCases, assert_session_conformance};

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const NAMESPACE: &[u8] = b"aiperf.test.conversation";

fn run_id() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x11; 32]))
}

fn program_digest() -> ContentDigest {
    ContentDigest::from_bytes([0x0a; 32])
}

fn stream_digest() -> ContentDigest {
    ContentDigest::from_bytes([0x0b; 32])
}

fn source_identity() -> aiperf_runtime::streaming::identity::ImmutableObjectIdentity {
    aiperf_runtime::streaming::identity::ImmutableObjectIdentity::from_bytes([0x0d; 32])
}

fn format_digest() -> ContentDigest {
    ContentDigest::from_bytes([0x0c; 32])
}

fn participant() -> CheckpointParticipantId {
    CheckpointParticipantId::new("session_coordinator")
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
        participant_id: participant(),
        stream_semantic_digest: stream_digest(),
        source_identity: source_identity(),
        session_state_budget: state_budget.clone(),
        checkpoint_budget: checkpoint_budget.clone(),
        issue_reporter,
    }
}

/// Build one coordinator plus the reporter whose handle it borrows.
///
/// The reporter is returned alongside so it outlives the injected handle.
fn make_coordinator(
    config: ConversationProgramConfig,
    state_budget: &StreamingResourceBudget,
    checkpoint_budget: &StreamingResourceBudget,
) -> (
    StreamingConversationCoordinator,
    BudgetOwnedStreamingIssueReporter,
) {
    let reporter = reporter();
    let context = prepare_context(state_budget, checkpoint_budget, reporter.handle());
    (
        StreamingConversationCoordinator::new(config, &context),
        reporter,
    )
}

fn default_coordinator() -> (
    StreamingConversationCoordinator,
    BudgetOwnedStreamingIssueReporter,
    StreamingResourceBudget,
    StreamingResourceBudget,
) {
    let state_budget = budget(4096, 4_194_304);
    let checkpoint_budget = budget(64, 4_194_304);
    let (coordinator, reporter) = make_coordinator(
        ConversationProgramConfig::default(),
        &state_budget,
        &checkpoint_budget,
    );
    (coordinator, reporter, state_budget, checkpoint_budget)
}

fn scope_for(producer_session: &str) -> ConversationSessionScope {
    ConversationSessionScope::new(
        stream_digest(),
        stable_session_key(NAMESPACE, producer_session.as_bytes()),
    )
}

fn provenance(partition: u8, position: u64) -> UnitProvenance {
    UnitProvenance {
        source_partition: ImmutableObjectIdentity::from_bytes([partition; 32]),
        source_position: SourcePosition::new(position),
        format_semantic_digest: format_digest(),
    }
}

#[allow(clippy::too_many_arguments)]
fn fragment(
    state_budget: &StreamingResourceBudget,
    producer_session: &str,
    producer_record: &str,
    partition: u8,
    position: u64,
    mutation: SessionMutationV1,
    charged_bytes: usize,
) -> StreamingSessionFragment {
    let lease = state_budget
        .try_acquire(1, charged_bytes)
        .expect("fragment charge fits the harness budget");
    StreamingSessionFragment {
        record_id: stable_record_id_from_key(NAMESPACE, producer_record.as_bytes()),
        session_key: stable_session_key(NAMESPACE, producer_session.as_bytes()),
        source_position: SourcePosition::new(position),
        source_partition: ImmutableObjectIdentity::from_bytes([partition; 32]),
        event_time: Some(
            EventTimeUtc::new(i64::try_from(position).unwrap_or(0) + 1).expect("time"),
        ),
        stable_tie_break: StableOrderKey::from_bytes([partition; 32]),
        predecessors: SmallVec::new(),
        mutation,
        provenance: provenance(partition, position),
        lease: SessionFragmentLease::try_from(lease).expect("one-item fragment charge"),
    }
}

#[allow(clippy::too_many_arguments)]
fn turn(
    state_budget: &StreamingResourceBudget,
    producer_session: &str,
    producer_record: &str,
    partition: u8,
    position: u64,
    role: &str,
    content: &str,
    turn_ordinal: u64,
) -> StreamingSessionFragment {
    fragment(
        state_budget,
        producer_session,
        producer_record,
        partition,
        position,
        SessionMutationV1::ConversationTurn(ConversationTurnFragment {
            role: role.to_string(),
            content: content.as_bytes().to_vec(),
            turn_ordinal,
        }),
        content.len().max(1),
    )
}

fn close(
    state_budget: &StreamingResourceBudget,
    producer_session: &str,
    producer_record: &str,
    partition: u8,
    position: u64,
    reason: &str,
) -> StreamingSessionFragment {
    fragment(
        state_budget,
        producer_session,
        producer_record,
        partition,
        position,
        SessionMutationV1::SessionClose(SessionCloseFragment {
            reason: reason.to_string(),
        }),
        reason.len().max(1),
    )
}

fn seal_at(position: u64) -> SourceSeal {
    SourceSeal {
        final_position: Some(SourcePosition::new(position)),
        digest: ContentDigest::from_bytes([0x77; 32]),
    }
}

fn watermark_at(value: i64) -> SessionWatermark {
    SessionWatermark {
        through: EventTimeUtc::new(value).expect("non-negative watermark"),
        digest: ContentDigest::from_bytes([0x78; 32]),
    }
}

fn terminal_event(action_id: StableActionId) -> ActionExecutionEvent {
    ActionExecutionEvent::Terminal(ActionTerminalReceipt {
        event: event_identity(action_id),
        disposition: ActionTerminalDisposition::Completed,
    })
}

fn event_identity(action_id: StableActionId) -> ActionEventIdentity {
    ActionEventIdentity {
        action_id,
        attempt_id: ActionAttemptId::from_bytes([0x33; 32]),
        ownership_epoch: SessionOwnershipEpoch::new(0),
        event_ordinal: 0,
    }
}

fn session_update(
    update_budget: &StreamingResourceBudget,
    action_id: StableActionId,
    reply: &str,
) -> ActionExecutionEvent {
    let lease = update_budget
        .try_acquire(1, reply.len())
        .expect("update charge fits the harness budget");
    let payload = BudgetedActionUpdate::new(Bytes::copy_from_slice(reply.as_bytes()), lease)
        .expect("exact update charge");
    ActionExecutionEvent::SessionUpdate(EndpointSessionUpdate {
        event: event_identity(action_id),
        payload,
    })
}

fn authored(text: &str) -> Box<RawValue> {
    RawValue::from_string(text.to_string()).expect("valid authored JSON")
}

static CONVERSATION_FORMAT: StreamingFormatDescriptor = StreamingFormatDescriptor {
    id: "test_conversation_format",
    description: "test-only conversation fragment format",
    semantic_digest: ContentDigest::from_bytes([0x0c; 32]),
    media_types: &["application/jsonl"],
    input_schemas: &["test.source.v1"],
    required_access: PartitionAccessKind::Sequential,
    projection: FormatProjection::FullRecord,
    output_schema: "aiperf.stream.session-fragment.v1",
    has_event_time: true,
    has_stable_record_ids: true,
    retention: FormatStateRetention::BoundedMemory,
    supports_virtual_clock: true,
};

static CONVERSATION_WORKLOAD: WorkloadDescriptor = WorkloadDescriptor {
    id: "request_rate",
    description: "test-only workload descriptor",
};

async fn prepared_state(
    coordinator: &mut StreamingConversationCoordinator,
    at: u64,
) -> PreparedParticipantState {
    coordinator
        .checkpoint_view(&support::barrier_at(run_id(), at))
        .await
        .expect("coordinator prepares a non-destructive view")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn one_conversation_spans_partitions_and_checkpoint() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();

    // Turn 0 arrives in partition 0; turn 1 arrives three partitions later.
    for (partition, position, record, content, ordinal) in [
        (0u8, 0u64, "r0", "hello", 0u64),
        (3u8, 9u64, "r1", "again", 1u64),
    ] {
        coordinator
            .ingest(
                turn(
                    &state_budget,
                    "s-1",
                    record,
                    partition,
                    position,
                    "user",
                    content,
                    ordinal,
                ),
                &mut sink,
            )
            .await
            .expect("cross-partition turn is incorporated");
    }

    let scope = scope_for("s-1");
    assert_eq!(coordinator.active_session_count(), 1);
    assert_eq!(
        coordinator
            .continuity(&scope)
            .expect("live session")
            .folded_through_turn(),
        Some(1)
    );
    let last = sink
        .request_actions()
        .last()
        .copied()
        .expect("a request action per contiguous turn")
        .clone();
    assert_eq!(last.messages(), ["hello", "again"]);

    let restored = support::commit_and_restore(prepared_state(&mut coordinator, 12).await).await;

    let restore_state_budget = budget(4096, 4_194_304);
    let restore_checkpoint_budget = budget(64, 4_194_304);
    let (mut resumed, _resumed_reporter) = make_coordinator(
        ConversationProgramConfig::default(),
        &restore_state_budget,
        &restore_checkpoint_budget,
    );
    resumed
        .initialize(Some(restored))
        .await
        .expect("restore from a committed generation");

    assert_eq!(resumed.active_session_count(), 1);
    let messages: Vec<String> = resumed
        .transcript(&scope)
        .expect("restored session")
        .iter()
        .map(|entry| String::from_utf8(entry.content.clone()).expect("text content"))
        .collect();
    assert_eq!(messages, ["hello", "again"]);
}

#[tokio::test(flavor = "current_thread")]
async fn partition_eof_never_closes_a_session() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();
    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("first turn is incorporated");

    // Partition exhaustion reaches the coordinator only as a format watermark;
    // neither it nor a source seal is session closure.
    coordinator
        .advance_watermark(watermark_at(20), &mut sink)
        .await
        .expect("watermark is recorded");
    assert_eq!(coordinator.active_session_count(), 1);

    let receipt = coordinator
        .seal(seal_at(9), &mut sink)
        .await
        .expect("a source seal is accepted over open sessions");
    assert_ne!(receipt.digest, ContentDigest::from_bytes([0; 32]));
    assert_eq!(coordinator.active_session_count(), 1);
    assert!(
        coordinator
            .continuity(&scope_for("s-1"))
            .expect("session stays live through the seal")
            .folded_through_turn()
            .is_some()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn identical_producer_mutation_is_idempotent() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();
    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("first observation is incorporated");
    let scope = scope_for("s-1");
    let version = coordinator
        .continuity(&scope)
        .expect("live session")
        .version();
    let actions = sink.request_actions().len();

    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 1, 4, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("an identical replay is a no-op");

    assert_eq!(sink.request_actions().len(), actions);
    assert_eq!(
        coordinator
            .continuity(&scope)
            .expect("live session")
            .version(),
        version
    );
}

#[tokio::test(flavor = "current_thread")]
async fn conflicting_mutation_content_is_refused() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();
    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("first observation is incorporated");
    let scope = scope_for("s-1");
    let version = coordinator
        .continuity(&scope)
        .expect("live session")
        .version();
    let transcript = coordinator
        .transcript(&scope)
        .expect("live session")
        .to_vec();

    let error = coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 1, 4, "user", "different", 0),
            &mut sink,
        )
        .await
        .expect_err("one record identity cannot carry conflicting content");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::ConflictingMutation)
    );
    assert_eq!(
        coordinator
            .continuity(&scope)
            .expect("live session")
            .version(),
        version
    );
    assert_eq!(
        coordinator.transcript(&scope).expect("live session"),
        transcript
    );
}

#[tokio::test(flavor = "current_thread")]
async fn explicit_close_is_terminal_only_after_declared_actions() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();
    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("first turn is incorporated");
    let action_id = sink
        .request_actions()
        .first()
        .copied()
        .expect("one request action")
        .action_id;

    coordinator
        .ingest(close(&state_budget, "s-1", "r1", 0, 1, "done"), &mut sink)
        .await
        .expect("producer-authored close is retained");
    let scope = scope_for("s-1");
    assert!(
        coordinator
            .continuity(&scope)
            .expect("live session")
            .has_pending_close(),
        "a close with an unterminated declared action stays pending"
    );
    assert!(
        sink.terminal_actions().is_empty(),
        "no terminal action is emitted while a declared action is in flight"
    );

    coordinator
        .observe_execution(terminal_event(action_id), &mut sink)
        .await
        .expect("terminal receipt settles the in-flight turn");
    assert_eq!(
        sink.terminal_actions().len(),
        1,
        "the close becomes exactly one terminal action"
    );
    assert!(
        !coordinator
            .continuity(&scope)
            .expect("live session")
            .has_pending_close()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn in_flight_turn_is_reemitted_with_the_same_stable_action_id_after_restart() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();
    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("first turn is incorporated");
    let before = sink
        .request_actions()
        .first()
        .copied()
        .expect("one request action")
        .action_id;

    let restored = support::commit_and_restore(prepared_state(&mut coordinator, 6).await).await;
    let restore_state_budget = budget(4096, 4_194_304);
    let restore_checkpoint_budget = budget(64, 4_194_304);
    let (mut resumed, _resumed_reporter) = make_coordinator(
        ConversationProgramConfig::default(),
        &restore_state_budget,
        &restore_checkpoint_budget,
    );
    resumed
        .initialize(Some(restored))
        .await
        .expect("restore from a committed generation");

    let scope = scope_for("s-1");
    assert_eq!(
        resumed
            .continuity(&scope)
            .expect("restored session")
            .in_flight_turn(),
        Some(0)
    );

    let mut resumed_sink = CollectingActionSink::default();
    resumed
        .advance_watermark(watermark_at(30), &mut resumed_sink)
        .await
        .expect("the first sink-bearing entry point flushes re-emissions");
    let reemitted = resumed_sink.request_actions();
    assert_eq!(reemitted.len(), 1, "an in-flight turn is re-emitted once");
    assert_eq!(reemitted[0].action_id, before);

    // A second sink-bearing entry point does not re-emit again.
    resumed_sink.clear();
    resumed
        .advance_watermark(watermark_at(31), &mut resumed_sink)
        .await
        .expect("watermark advances");
    assert!(resumed_sink.request_actions().is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_rolls_back_the_decoded_horizon_when_state_does_not_fit() {
    async fn two_sessions(
        checkpoint_bytes: usize,
    ) -> (
        StreamingConversationCoordinator,
        BudgetOwnedStreamingIssueReporter,
    ) {
        let state_budget = budget(4096, 4_194_304);
        let checkpoint_budget = budget(64, checkpoint_bytes);
        let (mut coordinator, reporter) = make_coordinator(
            ConversationProgramConfig::default(),
            &state_budget,
            &checkpoint_budget,
        );
        let mut sink = CollectingActionSink::default();
        for (session, record, position, content) in
            [("s-a", "ra", 5u64, "alpha"), ("s-b", "rb", 9u64, "beta")]
        {
            coordinator
                .ingest(
                    turn(
                        &state_budget,
                        session,
                        record,
                        0,
                        position,
                        "user",
                        content,
                        0,
                    ),
                    &mut sink,
                )
                .await
                .expect("turn is incorporated");
        }
        (coordinator, reporter)
    }

    // Measure the complete payload first, then re-run under a budget that
    // provably cannot retain it.
    let (mut complete, _complete_reporter) = two_sessions(4_194_304).await;
    let full = prepared_state(&mut complete, 100).await;
    let full_len = full.payload_bytes().len();
    assert_eq!(
        full.descriptor().represented_cut.decoded,
        support::cut_at(100).decoded,
        "a complete view represents the barrier's decode horizon verbatim"
    );
    drop(full);

    let (mut constrained, _constrained_reporter) = two_sessions(full_len - 1).await;
    let partial = prepared_state(&mut constrained, 100).await;

    // The dropped session is the last in scope encode order.
    let mut scopes = [scope_for("s-a"), scope_for("s-b")];
    scopes.sort_unstable();
    let dropped_position = if scopes[1] == scope_for("s-a") {
        SourcePosition::new(5)
    } else {
        SourcePosition::new(9)
    };
    assert!(
        *partial.descriptor().represented_cut.decoded.get() < SourcePosition::new(100),
        "an incomplete payload rolls the decode horizon back"
    );
    assert_eq!(
        *partial.descriptor().represented_cut.decoded.get(),
        dropped_position,
        "the rolled-back horizon is the first unrepresented source position"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn endpoint_reply_is_folded_into_the_next_turn_transcript() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let update_budget = budget(16, 4096);
    let mut sink = CollectingActionSink::default();
    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("first turn is incorporated");
    let action_id = sink
        .request_actions()
        .first()
        .copied()
        .expect("one request action")
        .action_id;

    coordinator
        .observe_execution(
            session_update(&update_budget, action_id, "world"),
            &mut sink,
        )
        .await
        .expect("an endpoint reply folds into the transcript");

    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r1", 1, 4, "user", "again", 1),
            &mut sink,
        )
        .await
        .expect("second turn is incorporated");

    let last = sink
        .request_actions()
        .last()
        .copied()
        .expect("a second request action")
        .clone();
    assert_eq!(last.messages(), ["hello", "world", "again"]);

    let scope = scope_for("s-1");
    let origins: Vec<TranscriptOrigin> = coordinator
        .transcript(&scope)
        .expect("live session")
        .iter()
        .map(|entry| entry.origin)
        .collect();
    assert_eq!(
        origins,
        [
            TranscriptOrigin::Authored,
            TranscriptOrigin::Endpoint,
            TranscriptOrigin::Authored
        ]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unsupported_mutation_is_refused_before_state_change() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();
    let error = coordinator
        .ingest(
            fragment(
                &state_budget,
                "s-1",
                "r0",
                0,
                0,
                SessionMutationV1::GraphNode(GraphNodeFragment {
                    node_key: "node".to_string(),
                    request: b"{}".to_vec(),
                }),
                8,
            ),
            &mut sink,
        )
        .await
        .expect_err("a conversation program does not accept graph mutations");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation)
    );
    assert_eq!(coordinator.active_session_count(), 0);
    assert!(sink.actions.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn stable_ordinal_is_restart_stable_and_map_free() {
    let (mut coordinator, _reporter, state_budget, _checkpoint_budget) = default_coordinator();
    let mut sink = CollectingActionSink::default();
    coordinator
        .ingest(
            turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
            &mut sink,
        )
        .await
        .expect("first turn is incorporated");

    let scope = scope_for("s-1");
    let before = scope.stable_ordinal();

    let restored = support::commit_and_restore(prepared_state(&mut coordinator, 6).await).await;
    let restore_state_budget = budget(4096, 4_194_304);
    let restore_checkpoint_budget = budget(64, 4_194_304);
    let (mut resumed, _resumed_reporter) = make_coordinator(
        ConversationProgramConfig::default(),
        &restore_state_budget,
        &restore_checkpoint_budget,
    );
    resumed
        .initialize(Some(restored))
        .await
        .expect("restore from a committed generation");

    let restored_scope = *resumed
        .continuity(&scope)
        .expect("restored session")
        .scope();
    assert_eq!(restored_scope.stable_ordinal(), before);
    // The projection reads only the durable session key, so it needs no entry in
    // any runtime-owned session-number map.
    assert_eq!(
        ConversationSessionScope::new(stream_digest(), scope.session_key()).stable_ordinal(),
        before
    );
}

#[tokio::test(flavor = "current_thread")]
async fn state_budget_exhaustion_is_refused_not_silently_dropped() {
    let state_budget = budget(4096, 4_194_304);
    let checkpoint_budget = budget(64, 4_194_304);
    let (mut coordinator, _reporter) = make_coordinator(
        ConversationProgramConfig {
            max_transcript_bytes: 4,
            ..ConversationProgramConfig::default()
        },
        &state_budget,
        &checkpoint_budget,
    );
    let mut sink = CollectingActionSink::default();
    let error = coordinator
        .ingest(
            turn(
                &state_budget,
                "s-1",
                "r0",
                0,
                0,
                "user",
                "a transcript beyond the proven bound",
                0,
            ),
            &mut sink,
        )
        .await
        .expect_err("retained state beyond the proven bound is refused");
    assert!(matches!(
        error,
        SessionCoordinatorError::StateBudget(
            aiperf_runtime::streaming::unit::StateBudgetFailureCode::ByteCapacity
        )
    ));
    assert!(
        coordinator
            .transcript(&scope_for("s-1"))
            .expect("session slot exists")
            .is_empty(),
        "a refused mutation never leaves a truncated transcript"
    );
    assert!(sink.actions.is_empty());
}

#[tokio::test(flavor = "current_thread")]
async fn conversation_program_satisfies_the_session_conformance_contract() {
    let state_budget = budget(4096, 4_194_304);
    let checkpoint_budget = budget(64, 4_194_304);
    let fragments = vec![
        turn(&state_budget, "s-1", "r0", 0, 0, "user", "hello", 0),
        turn(&state_budget, "s-1", "r1", 2, 5, "user", "again", 1),
    ];
    assert_session_conformance(
        &StreamingConversationProgramFactory,
        Box::new(reporter()),
        SessionConformanceCases {
            authored: authored("{}"),
            rejected_authored: authored("{\"unknown_field\":1}"),
            format_descriptor: &CONVERSATION_FORMAT,
            workload_descriptor: &CONVERSATION_WORKLOAD,
            run: run_id(),
            participant_id: participant(),
            program_semantic_digest: program_digest(),
            stream_semantic_digest: stream_digest(),
            source_identity: source_identity(),
            session_state_budget: state_budget.clone(),
            checkpoint_budget: checkpoint_budget.clone(),
            fragments,
            seal: seal_at(5),
            expected_action_count: 2,
            expected_issue_count: 0,
            advance: Rc::new(|| {}),
        },
    )
    .await;
}

#[test]
fn descriptor_declares_the_canonical_conversation_capability_axis() {
    let descriptor = StreamingConversationProgramFactory.descriptor();
    assert_eq!(descriptor.id, "conversation");
    assert_eq!(
        descriptor.fragment_input_schemas,
        ["aiperf.stream.session-fragment.v1"]
    );
    assert_eq!(descriptor.action_schemas, ["aiperf.stream.action.v1"]);
    assert!(descriptor.supports_virtual_clock);
}

#[test]
fn zero_bounds_are_refused_as_unbounded_causality_state() {
    let error = StreamingConversationProgramFactory
        .validate(
            &authored("{\"max_active_sessions\":0}"),
            &CONVERSATION_FORMAT,
            &CONVERSATION_WORKLOAD,
        )
        .expect_err("a zero bound is unbounded causality state with extra steps");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::UnboundedCausalityState)
    );
}

static OTHER_FORMAT: StreamingFormatDescriptor = StreamingFormatDescriptor {
    id: "test_other_format",
    description: "test-only alternate fragment format",
    semantic_digest: ContentDigest::from_bytes([0x0d; 32]),
    media_types: &["application/jsonl"],
    input_schemas: &["test.source.v1"],
    required_access: PartitionAccessKind::Sequential,
    projection: FormatProjection::FullRecord,
    output_schema: "other.fragment.v1",
    has_event_time: true,
    has_stable_record_ids: true,
    retention: FormatStateRetention::BoundedMemory,
    supports_virtual_clock: true,
};

#[test]
fn unmatched_fragment_schema_is_refused_at_validation() {
    let error = StreamingConversationProgramFactory
        .validate(&authored("{}"), &OTHER_FORMAT, &CONVERSATION_WORKLOAD)
        .expect_err("an unmatched canonical fragment schema is refused");
    assert_eq!(
        error,
        SessionCoordinatorError::session(SessionFailureCode::UnsupportedMutation)
    );
}
