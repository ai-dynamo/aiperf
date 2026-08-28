// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversation turn closure for the native streaming session plane.
//!
//! A conversation turn pair closes on one terminal action receipt and on
//! nothing else. Failed, cancelled, and dropped turns close exactly as
//! completed ones do, ordinary endpoint faults never inflate the reliability
//! counters, and the durable closure ledger survives a checkpoint under a
//! contiguous — never sparse — terminal horizon.
#![cfg(feature = "streaming")]

#[path = "support/streaming_session_conformance.rs"]
mod support;

mod common;

use std::cell::RefCell;
use std::future::Future;
use std::pin::pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::metrics_core::RequestTrace;
use aiperf_runtime::multiturn::IssuedCredit;
use aiperf_runtime::scheduled::{ModelResponseMetadata, TurnDispatchOutcome, TurnRecordProcessor};
use aiperf_runtime::streaming::action::{
    ActionEventIdentity, ActionExecutionEvent, ActionTerminalDisposition,
};
use aiperf_runtime::streaming::budget::{BudgetLimits, StreamingResourceBudget};
use aiperf_runtime::streaming::checkpoint::{
    CheckpointParticipantId, StreamRunIdentity, StreamingCheckpointParticipant,
};
use aiperf_runtime::streaming::closure::{
    ClosedTurnDisposition, ClosureSeamContext, ConversationClosureSeam, ConversationTurnCloser,
    StreamingTurnBinding, TurnClosureToken, disposition_of,
};
use aiperf_runtime::streaming::failure::SessionCoordinatorError;
use aiperf_runtime::streaming::format::SessionWatermark;
use aiperf_runtime::streaming::identity::{
    ActionAttemptId, ContentDigest, GlobalSequence, ImmutableObjectIdentity, LogicalReplayRunId,
    SessionCausalFrontier, SessionOwnershipEpoch, StableActionId, StableSessionKey,
};
use aiperf_runtime::streaming::reliability::{
    OrdinaryStreamingIssue, StreamingIssueClass, StreamingIssueReportError,
    StreamingIssueReportStatus, StreamingIssueReporterEndpoint, StreamingIssueReporterHandle,
    StreamingIssueScope,
};
use aiperf_runtime::streaming::session::conversation::ConversationSessionScope;
use aiperf_runtime::streaming::session::{
    DatasetActionSink, SessionSealReceipt, StreamingSessionCoordinator,
};
use aiperf_runtime::streaming::source::SourceSeal;
use aiperf_runtime::streaming::unit::{
    EventTimeUtc, SourcePosition, StateBudgetFailureCode, StreamingSessionFragment,
};
use async_trait::async_trait;
use bytes::Bytes;
use support::CollectingActionSink;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn run_id() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x21; 32]))
}

fn stream_digest() -> ContentDigest {
    ContentDigest::from_bytes([0x2b; 32])
}

fn participant() -> CheckpointParticipantId {
    CheckpointParticipantId::new("aiperf.stream.session.closure")
}

fn budget(items: usize, bytes: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: items,
        max_bytes: bytes,
    })
    .expect("valid harness budget")
}

fn scope_for(index: u8) -> ConversationSessionScope {
    ConversationSessionScope::new(
        stream_digest(),
        StableSessionKey::from_bytes([0x40 + index; 32]),
    )
}

fn action_of(index: u8) -> StableActionId {
    StableActionId::from_bytes([0x50 + index; 32])
}

fn event_identity(action_id: StableActionId) -> ActionEventIdentity {
    ActionEventIdentity {
        action_id,
        attempt_id: ActionAttemptId::from_bytes([0x33; 32]),
        ownership_epoch: SessionOwnershipEpoch::new(0),
        event_ordinal: 0,
    }
}

fn binding_for(
    index: u8,
    scope: ConversationSessionScope,
    sequence: u64,
    position: u64,
) -> StreamingTurnBinding {
    let action_id = action_of(index);
    StreamingTurnBinding {
        credit_id: u64::from(index),
        action_id,
        scope,
        global_sequence: GlobalSequence::new(sequence),
        source_position: SourcePosition::new(position),
        source_partition: ImmutableObjectIdentity::from_bytes([0x60; 32]),
        event_identity: event_identity(action_id),
    }
}

/// Every ordinary issue the seam submitted, in submission order.
#[derive(Default)]
struct IssueLedger {
    issues: RefCell<Vec<(StreamingIssueClass, bool)>>,
}

impl IssueLedger {
    fn count(&self) -> usize {
        self.issues.borrow().len()
    }

    fn classes(&self) -> Vec<StreamingIssueClass> {
        self.issues
            .borrow()
            .iter()
            .map(|(class, _)| *class)
            .collect()
    }

    fn all_session_scoped(&self) -> bool {
        self.issues
            .borrow()
            .iter()
            .all(|(_, is_session_scoped)| *is_session_scoped)
    }
}

struct RecordingIssueReporter {
    ledger: Rc<IssueLedger>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for RecordingIssueReporter {
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        let is_session_scoped = matches!(issue.scope(), StreamingIssueScope::Session { .. });
        self.ledger
            .issues
            .borrow_mut()
            .push((issue.class(), is_session_scoped));
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

fn reporter() -> (StreamingIssueReporterHandle, Rc<IssueLedger>) {
    let ledger = Rc::new(IssueLedger::default());
    let handle = StreamingIssueReporterHandle::new(RecordingIssueReporter {
        ledger: Rc::clone(&ledger),
    });
    (handle, ledger)
}

/// Execution events the seam delivered, in delivery order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ObservedEvent {
    Reply(StableActionId),
    Terminal(StableActionId),
}

#[derive(Default)]
struct RecordingCoordinator {
    events: Vec<ObservedEvent>,
    replies: Vec<Vec<u8>>,
}

#[async_trait(?Send)]
impl StreamingSessionCoordinator for RecordingCoordinator {
    async fn ingest(
        &mut self,
        _fragment: StreamingSessionFragment,
        _output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        Ok(())
    }

    async fn advance_watermark(
        &mut self,
        _watermark: SessionWatermark,
        _output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        Ok(())
    }

    async fn observe_execution(
        &mut self,
        event: ActionExecutionEvent,
        _output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        match event {
            ActionExecutionEvent::SessionUpdate(update) => {
                self.events
                    .push(ObservedEvent::Reply(update.event.action_id));
                self.replies.push(update.payload.as_bytes().to_vec());
            }
            ActionExecutionEvent::Terminal(receipt) => {
                self.events
                    .push(ObservedEvent::Terminal(receipt.event.action_id));
            }
            ActionExecutionEvent::Admitted(_) | ActionExecutionEvent::FirstToken(_) => {}
        }
        Ok(())
    }

    async fn seal(
        &mut self,
        _seal: SourceSeal,
        _output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError> {
        Ok(SessionSealReceipt {
            digest: ContentDigest::from_bytes([0x77; 32]),
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(0),
                event_time: None,
                digest: ContentDigest::from_bytes([0; 32]),
            },
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for RecordingCoordinator {
    fn participant_id(&self) -> CheckpointParticipantId {
        CheckpointParticipantId::new("recording_coordinator")
    }

    async fn checkpoint_view(
        &mut self,
        _barrier: &aiperf_runtime::streaming::checkpoint::CheckpointBarrier,
    ) -> Result<
        aiperf_runtime::streaming::checkpoint::PreparedParticipantState,
        aiperf_runtime::streaming::checkpoint::CheckpointError,
    > {
        Err(aiperf_runtime::streaming::checkpoint::CheckpointError::ObjectVerification)
    }

    async fn initialize(
        &mut self,
        _state: Option<aiperf_runtime::streaming::checkpoint::CommittedParticipantState>,
    ) -> Result<(), aiperf_runtime::streaming::checkpoint::CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &aiperf_runtime::streaming::checkpoint::CommittedParticipantReceipt,
    ) -> Result<(), aiperf_runtime::streaming::checkpoint::CheckpointError> {
        Ok(())
    }
}

/// Build one seam with explicit slot, reply, and checkpoint capacity.
fn seam_with(
    max_bound_turns: usize,
    reply_bytes: usize,
    checkpoint_bytes: usize,
) -> (ConversationClosureSeam, Rc<IssueLedger>) {
    let (issue_reporter, ledger) = reporter();
    let seam = ConversationClosureSeam::new(ClosureSeamContext {
        run: run_id(),
        participant_id: participant(),
        stream_identity: stream_digest(),
        max_bound_turns,
        slot_budget: budget(64, 1 << 20),
        reply_budget: budget(64, reply_bytes),
        checkpoint_budget: budget(64, checkpoint_bytes),
        issue_reporter,
    })
    .expect("valid closure seam");
    (seam, ledger)
}

fn default_seam() -> (ConversationClosureSeam, Rc<IssueLedger>) {
    seam_with(64, 1 << 20, 1 << 20)
}

/// Drive one closure delivery pass against a recording coordinator.
async fn deliver(
    seam: &mut ConversationClosureSeam,
    coordinator: &mut RecordingCoordinator,
    sink: &mut CollectingActionSink,
) -> aiperf_runtime::streaming::closure::ClosureDeliveryReceipt {
    seam.deliver_closures(coordinator, sink)
        .await
        .expect("closure delivery never fails on ordinary endpoint faults")
}

// ---------------------------------------------------------------------------
// 1-2: what closes, and what does not
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn terminal_receipt_is_the_only_signal_that_closes_a_turn() {
    let (mut seam, ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();
    let scope = scope_for(0);

    assert_eq!(
        seam.bind_turn(binding_for(0, scope, 0, 1))
            .expect("first bind reserves a terminal slot"),
        TurnClosureToken::Bound
    );

    seam.observe_reply(action_of(0), Bytes::from_static(b"an endpoint reply"));
    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.replies_folded, 1);
    assert_eq!(receipt.closed, 0, "a reply alone closes no turn");
    let readiness = seam.readiness(&scope).expect("bound scope is known");
    assert!(
        !readiness.has_no_inflight_turn(),
        "the turn is still in flight after only a reply"
    );

    seam.observe_terminal(action_of(0), ActionTerminalDisposition::Completed);
    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.closed, 1);
    assert!(
        seam.readiness(&scope)
            .expect("scope is known")
            .has_no_inflight_turn(),
        "the terminal receipt is what closes the turn"
    );
    assert_eq!(ledger.count(), 0, "an ordinary closure reports no issue");
}

#[tokio::test(flavor = "current_thread")]
async fn partition_eof_and_source_seal_close_no_turn() {
    let (mut seam, _ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    seam.bind_turn(binding_for(0, scope_for(0), 0, 1))
        .expect("bind");
    let before = seam.terminal_horizon();

    // Partition exhaustion and a source seal are decoder facts. Neither says
    // anything about an in-flight request.
    coordinator
        .advance_watermark(
            SessionWatermark {
                through: EventTimeUtc::new(9).expect("non-negative watermark"),
                digest: ContentDigest::from_bytes([0x78; 32]),
            },
            &mut sink,
        )
        .await
        .expect("watermark");
    coordinator
        .seal(
            SourceSeal {
                final_position: Some(SourcePosition::new(9)),
                digest: ContentDigest::from_bytes([0x79; 32]),
            },
            &mut sink,
        )
        .await
        .expect("seal");

    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.closed, 0);
    assert!(seam.closed_turn(action_of(0)).is_none());
    assert_eq!(seam.terminal_horizon(), before);
    assert_eq!(seam.inflight_turns(), 1);
}

// ---------------------------------------------------------------------------
// 3-5: fold order, dispositions, and issue silence
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn reply_is_folded_before_its_terminal_in_one_pass() {
    let (mut seam, _ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    seam.bind_turn(binding_for(0, scope_for(0), 0, 1))
        .expect("bind");
    seam.observe_reply(action_of(0), Bytes::from_static(b"reply bytes"));
    seam.observe_terminal(action_of(0), ActionTerminalDisposition::Completed);

    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.replies_folded, 1);
    assert_eq!(receipt.closed, 1);
    assert_eq!(
        coordinator.events,
        vec![
            ObservedEvent::Reply(action_of(0)),
            ObservedEvent::Terminal(action_of(0)),
        ],
        "delivering the terminal first would settle a turn whose reply the transcript lacks"
    );
    assert_eq!(coordinator.replies, vec![b"reply bytes".to_vec()]);
    assert!(
        seam.closed_turn(action_of(0))
            .expect("closed receipt")
            .reply_digest
            .is_some()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn failed_and_cancelled_turns_still_close() {
    let (mut seam, ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    let cases = [
        (
            0u8,
            ReplayTerminalStatus::Failed,
            ClosedTurnDisposition::Failed,
        ),
        (
            1u8,
            ReplayTerminalStatus::Canceled,
            ClosedTurnDisposition::Cancelled,
        ),
        (
            2u8,
            ReplayTerminalStatus::Rejected,
            ClosedTurnDisposition::Dropped,
        ),
    ];
    for (index, status, _) in cases {
        seam.bind_turn(binding_for(
            index,
            scope_for(0),
            u64::from(index),
            u64::from(index) + 1,
        ))
        .expect("bind");
        seam.observe_terminal(action_of(index), disposition_of(status));
    }

    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.closed, 3, "a failed turn is a closed turn");
    for (index, _, expected) in cases {
        assert_eq!(
            seam.closed_turn(action_of(index))
                .expect("closed receipt")
                .disposition,
            expected
        );
    }
    assert_eq!(
        ledger.count(),
        0,
        "endpoint fault conversion belongs to the action host, not this seam"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn successful_closure_reports_no_issue() {
    let (mut seam, ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    for index in 0..8u8 {
        seam.bind_turn(binding_for(
            index,
            scope_for(0),
            u64::from(index),
            u64::from(index) + 1,
        ))
        .expect("bind");
        seam.observe_reply(action_of(index), Bytes::from_static(b"ok"));
        seam.observe_terminal(action_of(index), ActionTerminalDisposition::Completed);
    }

    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.closed, 8);
    assert_eq!(receipt.replies_folded, 8);
    assert_eq!(
        ledger.count(),
        0,
        "ordinary success must never increment a reliability counter"
    );
}

// ---------------------------------------------------------------------------
// 6-7: the two issues the seam owns
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn orphan_terminal_reports_one_session_issue_and_closes_nothing() {
    let (mut seam, ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    seam.bind_turn(binding_for(0, scope_for(0), 0, 1))
        .expect("bind");
    seam.observe_terminal(action_of(0), ActionTerminalDisposition::Completed);
    assert_eq!(
        deliver(&mut seam, &mut coordinator, &mut sink).await.closed,
        1
    );

    // A second terminal for a settled action is claimed by no live turn.
    seam.observe_terminal(action_of(0), ActionTerminalDisposition::Completed);
    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.orphan_terminals, 1);
    assert_eq!(receipt.closed, 0, "an orphan terminal closes nothing");
    assert_eq!(seam.closed_turns(), 1, "no second closure was minted");
    assert_eq!(ledger.classes(), vec![StreamingIssueClass::Permanent]);
    assert!(ledger.all_session_scoped());

    // A terminal for an action the ledger has never seen is still an orphan,
    // and the seam reports nothing it cannot truthfully scope.
    seam.observe_terminal(action_of(9), ActionTerminalDisposition::Completed);
    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.orphan_terminals, 1);
    assert_eq!(ledger.count(), 1);
}

#[tokio::test(flavor = "current_thread")]
async fn refused_reply_is_absent_state_not_a_fatal_error() {
    // One byte of reply capacity cannot admit the reply below.
    let (mut seam, ledger) = seam_with(8, 1, 1 << 20);
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    seam.bind_turn(binding_for(0, scope_for(0), 0, 1))
        .expect("bind");
    seam.observe_reply(
        action_of(0),
        Bytes::from_static(b"a reply that does not fit"),
    );
    seam.observe_terminal(action_of(0), ActionTerminalDisposition::Completed);

    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.refused_replies, 1);
    assert_eq!(receipt.replies_folded, 0);
    assert_eq!(receipt.closed, 1, "the turn still closes without its reply");
    assert!(
        seam.closed_turn(action_of(0))
            .expect("closed receipt")
            .reply_digest
            .is_none()
    );
    assert_eq!(ledger.classes(), vec![StreamingIssueClass::Capacity]);
    assert!(ledger.all_session_scoped());
    assert_eq!(
        coordinator.events,
        vec![ObservedEvent::Terminal(action_of(0))]
    );
}

// ---------------------------------------------------------------------------
// 8-10: reservation, idempotence, contiguity
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn terminal_slot_is_reserved_at_bind_so_terminal_delivery_cannot_be_refused() {
    let (mut seam, _ledger) = seam_with(3, 1, 1 << 20);
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    for index in 0..3u8 {
        assert_eq!(
            seam.bind_turn(binding_for(
                index,
                scope_for(0),
                u64::from(index),
                u64::from(index) + 1
            ))
            .expect("bind within the declared bound"),
            TurnClosureToken::Bound
        );
        seam.observe_reply(action_of(index), Bytes::from_static(b"exhausting reply"));
        seam.observe_terminal(action_of(index), ActionTerminalDisposition::Completed);
    }

    assert_eq!(
        seam.bind_turn(binding_for(3, scope_for(0), 3, 4)),
        Err(SessionCoordinatorError::state_budget(
            StateBudgetFailureCode::ItemCapacity
        )),
        "binding past the bound is refused before dispatch"
    );

    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(
        receipt.closed, 3,
        "an exhausted reply budget never drops a terminal"
    );
    assert_eq!(receipt.refused_replies, 3);
}

#[tokio::test(flavor = "current_thread")]
async fn rebinding_a_reemitted_action_is_idempotent() {
    let (mut seam, _ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();
    let scope = scope_for(0);

    assert_eq!(
        seam.bind_turn(binding_for(0, scope, 0, 1)).expect("bind"),
        TurnClosureToken::Bound
    );
    assert_eq!(
        seam.bind_turn(binding_for(0, scope, 0, 1))
            .expect("re-emission rebind"),
        TurnClosureToken::AlreadyBound
    );
    assert_eq!(
        seam.readiness(&scope).expect("scope is known").bound_turns,
        1,
        "a re-emitted action is one turn, not two"
    );

    seam.observe_terminal(action_of(0), ActionTerminalDisposition::Completed);
    let receipt = deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.closed, 1);
    assert_eq!(
        seam.bind_turn(binding_for(0, scope, 0, 1))
            .expect("rebind after closure"),
        TurnClosureToken::AlreadyBound
    );
}

#[tokio::test(flavor = "current_thread")]
async fn terminal_horizon_is_contiguous_not_maximal() {
    let (mut seam, _ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();

    for index in [0u8, 1, 2, 4] {
        seam.bind_turn(binding_for(
            index,
            scope_for(0),
            u64::from(index),
            u64::from(index) + 1,
        ))
        .expect("bind");
        seam.observe_terminal(action_of(index), ActionTerminalDisposition::Completed);
    }
    deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(
        seam.terminal_horizon().get().get(),
        2,
        "publishing 4 over an open 3 would be an untruthful cut"
    );

    seam.bind_turn(binding_for(3, scope_for(0), 3, 4))
        .expect("bind");
    seam.observe_terminal(action_of(3), ActionTerminalDisposition::Completed);
    deliver(&mut seam, &mut coordinator, &mut sink).await;
    assert_eq!(seam.terminal_horizon().get().get(), 4);
}

// ---------------------------------------------------------------------------
// 11-12: checkpoint
// ---------------------------------------------------------------------------

/// Close three turns at dense sequences 0..2 and source positions 1..3.
async fn seam_with_three_closures(checkpoint_bytes: usize) -> ConversationClosureSeam {
    let (mut seam, _ledger) = seam_with(64, 1 << 20, checkpoint_bytes);
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();
    seam.initialize(None).await.expect("fresh initialization");
    for index in 0..3u8 {
        seam.bind_turn(binding_for(
            index,
            scope_for(0),
            u64::from(index),
            u64::from(index) + 1,
        ))
        .expect("bind");
        seam.observe_reply(action_of(index), Bytes::from_static(b"reply"));
        seam.observe_terminal(action_of(index), ActionTerminalDisposition::Completed);
    }
    deliver(&mut seam, &mut coordinator, &mut sink).await;
    seam
}

#[tokio::test(flavor = "current_thread")]
async fn closure_ledger_survives_checkpoint_and_restore() {
    let mut seam = seam_with_three_closures(1 << 20).await;
    let horizon = seam.terminal_horizon();
    let before = seam
        .closed_turn(action_of(1))
        .expect("closed receipt")
        .clone();

    let prepared = seam
        .checkpoint_view(&support::barrier_at(run_id(), 64))
        .await
        .expect("closure ledger prepares a view");
    assert_eq!(prepared.descriptor().participant_id, participant());
    let committed = support::commit_and_restore(prepared).await;

    let (mut restored, ledger) = seam_with(64, 1 << 20, 1 << 20);
    restored
        .initialize(Some(committed))
        .await
        .expect("restore the committed ledger");
    assert_eq!(
        restored
            .closed_turn(action_of(1))
            .expect("restored receipt"),
        &before
    );
    assert_eq!(restored.terminal_horizon(), horizon);

    // A replayed terminal for a restored action is an orphan, never a second
    // closure.
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();
    restored.observe_terminal(action_of(1), ActionTerminalDisposition::Completed);
    let receipt = deliver(&mut restored, &mut coordinator, &mut sink).await;
    assert_eq!(receipt.orphan_terminals, 1);
    assert_eq!(receipt.closed, 0);
    assert_eq!(restored.closed_turns(), 3);
    assert_eq!(ledger.classes(), vec![StreamingIssueClass::Permanent]);
}

#[tokio::test(flavor = "current_thread")]
async fn checkpoint_rolls_back_both_horizons_when_the_ledger_does_not_fit() {
    let mut complete = seam_with_three_closures(1 << 20).await;
    let barrier = support::barrier_at(run_id(), 64);
    let full = complete
        .checkpoint_view(&barrier)
        .await
        .expect("complete view");
    assert_eq!(
        full.descriptor().represented_cut.decoded,
        barrier.cut.decoded,
        "a complete payload represents the barrier's decode horizon"
    );
    let full_len = full.payload_bytes().len();

    let mut undersized = seam_with_three_closures(full_len - 1).await;
    let rolled = undersized
        .checkpoint_view(&barrier)
        .await
        .expect("an undersized budget rolls back rather than failing");
    let cut = &rolled.descriptor().represented_cut;
    assert!(
        cut.decoded.get() < barrier.cut.decoded.get(),
        "the decode horizon must move below the first unrepresented turn"
    );
    assert!(
        cut.terminal.get().get() < barrier.cut.terminal.get().get(),
        "the terminal horizon must move below the first unrepresented closure"
    );
    assert!(
        rolled.descriptor().item_count < 3,
        "a rolled-back payload holds fewer receipts than the ledger"
    );
}

// ---------------------------------------------------------------------------
// 13-14: the terminal-lane processor
// ---------------------------------------------------------------------------

fn completed_outcome(response_text: &str) -> TurnDispatchOutcome {
    TurnDispatchOutcome {
        start_ns: 0,
        end_ns: 0,
        terminal: ReplayTerminalStatus::Completed,
        response_text: response_text.to_string(),
        model_response: ModelResponseMetadata::default(),
        prompt_tokens: None,
        completion_tokens: None,
        http: RequestTrace::default(),
    }
}

/// Build one real issued credit with the supplied credit id.
async fn issued_credit(id: u64) -> IssuedCredit {
    let mut source = common::prepared_source_from_conversations(
        serde_json::json!([{"session_id": "closure", "turns": [{
            "text": "closure seam turn",
            "input_length": 4,
            "output_length": 1,
        }]}]),
        "closure-model",
        1,
    )
    .await;
    let session = source
        .next(Some("closure-0".to_string()))
        .expect("sampled session");
    let turn = session.build_first_turn(None).expect("materialized turn");
    IssuedCredit::from_turn(id, 0, &turn)
}

#[tokio::test(flavor = "current_thread")]
async fn unbound_credit_is_ignored_by_the_lane_processor() {
    let (seam, ledger) = default_seam();
    let processor = seam.processor();
    let credit = issued_credit(4242).await;

    processor
        .process(&credit, &completed_outcome("ignored reply"))
        .await
        .expect("a foreign credit is not an error");

    assert_eq!(
        seam.intake().queued(),
        Some(0),
        "the shared lane serves ordinary non-streaming workloads too"
    );
    assert_eq!(ledger.count(), 0);
}

/// Run one future to completion, refusing to suspend.
///
/// A pending poll is the exact failure this asserts against: the lane's single
/// drain owner cannot afford an await inside a record processor.
fn run_without_suspending<F: Future>(future: F) -> F::Output {
    let mut future = pin!(future);
    let mut context = Context::from_waker(Waker::noop());
    match future.as_mut().poll(&mut context) {
        Poll::Ready(value) => value,
        Poll::Pending => panic!("the lane processor must not await"),
    }
}

#[test]
fn processor_never_borrows_the_coordinator_or_awaits_the_reporter() {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime for dataset materialization");
    let credit = runtime.block_on(issued_credit(7));

    let (mut seam, ledger) = default_seam();
    seam.bind_turn(StreamingTurnBinding {
        credit_id: 7,
        ..binding_for(0, scope_for(0), 0, 1)
    })
    .expect("bind");
    let processor = seam.processor();

    run_without_suspending(processor.process(&credit, &completed_outcome("a reply")))
        .expect("the processor completes without suspending");

    assert_eq!(
        seam.intake().queued(),
        Some(2),
        "no intake borrow outlives the push that produced the entries"
    );
    assert_eq!(
        ledger.count(),
        0,
        "the lane owner never awaits the reliability reporter"
    );
}

// ---------------------------------------------------------------------------
// 15: readiness is not closure
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn readiness_is_necessary_but_never_closes_a_session() {
    let (mut seam, ledger) = default_seam();
    let mut coordinator = RecordingCoordinator::default();
    let mut sink = CollectingActionSink::default();
    let scope = scope_for(0);

    for index in 0..2u8 {
        seam.bind_turn(binding_for(
            index,
            scope,
            u64::from(index),
            u64::from(index) + 1,
        ))
        .expect("bind");
        seam.observe_terminal(action_of(index), ActionTerminalDisposition::Completed);
    }
    deliver(&mut seam, &mut coordinator, &mut sink).await;

    let readiness = seam.readiness(&scope).expect("scope is known");
    assert_eq!(readiness.bound_turns, 2);
    assert_eq!(readiness.closed_turns, 2);
    assert!(readiness.has_no_inflight_turn());

    // Readiness is a fact, not a decision: the seam emits no action of any
    // kind, so it cannot promote a producer-authored close to a terminal.
    assert!(
        sink.actions.is_empty(),
        "the closure seam never emits a session-terminal action"
    );
    assert!(sink.frontiers.is_empty());
    assert_eq!(ledger.count(), 0);
    assert!(
        seam.readiness(&scope_for(7)).is_none(),
        "an unknown conversation has no readiness fact to report"
    );
}
