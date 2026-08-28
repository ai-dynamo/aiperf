// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conversation turn closure seam.
//!
//! One conversation turn pair — the authored turn the session coordinator
//! emitted plus the endpoint reply it produced — becomes durable here. A turn
//! closes on `ActionExecutionEvent::Terminal` and on nothing else: partition
//! exhaustion, source seals, and watermarks say nothing about an in-flight
//! request. Session closure is a different unit with different proofs and is
//! owned by the session-closure policy, not by this seam: the only fact
//! published for that owner is [`SessionClosureReadiness`], which is necessary
//! and never sufficient.
//!
//! The seam is split across the two owners that cannot borrow each other. The
//! terminal lane pushes into a bounded worker-local intake through
//! [`StreamingTurnClosureProcessor`], which performs no await and never touches
//! the session coordinator. The pipeline task, which owns
//! `&mut dyn StreamingSessionCoordinator`, drains that intake in
//! [`ConversationTurnCloser::deliver_closures`]. Every intake entry is moved out
//! of its `RefCell` before any suspension point.

use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet, VecDeque},
    rc::Rc,
};

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    dispatch::collector::ReplayTerminalStatus,
    multiturn::IssuedCredit,
    scheduled::{TurnDispatchOutcome, TurnRecordProcessor},
    streaming::{
        action::{
            ActionEventIdentity, ActionExecutionEvent, ActionTerminalDisposition,
            ActionTerminalReceipt, BudgetedActionUpdate, EndpointSessionUpdate,
        },
        budget::{BudgetError, BudgetLease, StreamingResourceBudget},
        checkpoint::{
            BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
            CommittedParticipantReceipt, CommittedParticipantState, DecodeHorizon,
            ParticipantInitialization, PreparedParticipantState, StreamRunIdentity,
            StreamingCheckpointParticipant, TerminalActionHorizon,
        },
        failure::{OrdinaryStreamingFailure, SessionFailureCode},
        identity::{
            ContentDigest, GlobalSequence, ImmutableObjectIdentity, SessionCausalFrontier,
            StableActionId,
        },
        reliability::{
            OrdinaryStreamingIssue, StreamingInputDomainIdentity, StreamingIssueClass,
            StreamingIssueReportStatus, StreamingIssueReporterHandle,
        },
        session::{
            DatasetActionSink, SessionCoordinatorError, StreamingSessionCoordinator,
            conversation::ConversationSessionScope,
        },
        unit::{ExecutableDatasetAction, SourcePosition, StateBudgetFailureCode},
    },
};

/// Checkpoint schema and participant identity of the closure ledger.
pub const CLOSURE_PARTICIPANT_ID: &str = "aiperf.stream.session.closure";

/// Checkpoint schema version for the retained closure ledger.
const CLOSURE_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

/// Dense per-conversation ordinal of one closed turn pair.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TurnClosureOrdinal(u64);

impl TurnClosureOrdinal {
    /// Ordinal of the first closed turn in a conversation.
    pub const FIRST: Self = Self(0);

    /// Return the underlying dense ordinal.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Advance one ordinal, refusing instead of wrapping.
    pub const fn next(self) -> Result<Self, SessionCoordinatorError> {
        match self.0.checked_add(1) {
            Some(value) => Ok(Self(value)),
            None => Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            )),
        }
    }
}

/// Serializable projection of one observed terminal disposition.
///
/// `ActionTerminalDisposition` is owned by the action-binding contract and
/// carries no serde derives; the durable ledger mirrors it here rather than
/// widening a type this seam only reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClosedTurnDisposition {
    /// The turn completed successfully.
    Completed,
    /// The turn failed after admission.
    Failed,
    /// The turn was cancelled.
    Cancelled,
    /// The turn was dropped before endpoint issue.
    Dropped,
}

impl From<ActionTerminalDisposition> for ClosedTurnDisposition {
    fn from(disposition: ActionTerminalDisposition) -> Self {
        match disposition {
            ActionTerminalDisposition::Completed => Self::Completed,
            ActionTerminalDisposition::Failed => Self::Failed,
            ActionTerminalDisposition::Cancelled => Self::Cancelled,
            ActionTerminalDisposition::Dropped => Self::Dropped,
        }
    }
}

impl From<ClosedTurnDisposition> for ActionTerminalDisposition {
    fn from(disposition: ClosedTurnDisposition) -> Self {
        match disposition {
            ClosedTurnDisposition::Completed => Self::Completed,
            ClosedTurnDisposition::Failed => Self::Failed,
            ClosedTurnDisposition::Cancelled => Self::Cancelled,
            ClosedTurnDisposition::Dropped => Self::Dropped,
        }
    }
}

/// Map one reduced dispatch outcome onto its terminal disposition.
///
/// `Rejected` is refusal before endpoint issue, which is `Dropped` rather than
/// `Failed`; every other pairing is nominal.
#[must_use]
pub const fn disposition_of(terminal: ReplayTerminalStatus) -> ActionTerminalDisposition {
    match terminal {
        ReplayTerminalStatus::Completed => ActionTerminalDisposition::Completed,
        ReplayTerminalStatus::Canceled => ActionTerminalDisposition::Cancelled,
        ReplayTerminalStatus::Rejected => ActionTerminalDisposition::Dropped,
        ReplayTerminalStatus::Failed => ActionTerminalDisposition::Failed,
    }
}

/// Issue-time facts binding one dispatched credit to one logical turn action.
#[derive(Clone, Debug)]
pub struct StreamingTurnBinding {
    /// Credit id assigned by the issuer before backend dispatch.
    pub credit_id: u64,
    /// Stable logical action identity emitted by the session coordinator.
    pub action_id: StableActionId,
    /// Cross-partition conversation owning the turn.
    pub scope: ConversationSessionScope,
    /// Dense global sequence assigned by the action host.
    pub global_sequence: GlobalSequence,
    /// Stable source coordinate that caused the turn.
    pub source_position: SourcePosition,
    /// Immutable partition that supplied the authored turn.
    pub source_partition: ImmutableObjectIdentity,
    /// Host-minted event identity; the seam varies only `event_ordinal`.
    pub event_identity: ActionEventIdentity,
}

/// Result of binding one turn, distinguishing a restart re-emission.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TurnClosureToken {
    /// A new turn was bound and its terminal slot reserved.
    Bound,
    /// The identical stable action was already bound or already closed.
    AlreadyBound,
}

/// Durable proof that one conversation turn pair reached terminal.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClosedTurnReceipt {
    /// Cross-partition conversation owning the closed turn.
    pub scope: ConversationSessionScope,
    /// Stable logical action identity of the closed turn.
    pub action_id: StableActionId,
    /// Dense per-conversation closure ordinal.
    pub closure_ordinal: TurnClosureOrdinal,
    /// Dense global sequence the action host assigned.
    pub global_sequence: GlobalSequence,
    /// Terminal disposition observed exactly once.
    pub disposition: ClosedTurnDisposition,
    /// Digest of the folded endpoint reply, absent when no reply was retained.
    ///
    /// The reply bytes themselves belong to the session transcript. Retaining
    /// them twice would double-charge state and create a second place to drift.
    pub reply_digest: Option<ContentDigest>,
    /// Restart-stable external session ordinal reported as `session_num`.
    pub stable_session_ordinal: u64,
    /// Stable source coordinate that caused the closed turn.
    pub source_position: SourcePosition,
    /// Immutable partition that supplied the authored turn.
    pub source_partition: ImmutableObjectIdentity,
}

/// Per-conversation counts a session-closure policy may read.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionClosureReadiness {
    /// Conversation these counts describe.
    pub scope: ConversationSessionScope,
    /// Turns bound for dispatch.
    pub bound_turns: u64,
    /// Turns that reached exactly one terminal receipt.
    pub closed_turns: u64,
}

impl SessionClosureReadiness {
    /// Whether no request belonging to this conversation is in flight.
    ///
    /// This is necessary but never sufficient for session closure: a closure
    /// proof is owned by the session-closure policy, not by this seam.
    #[must_use]
    pub const fn has_no_inflight_turn(&self) -> bool {
        self.bound_turns == self.closed_turns
    }
}

/// Outcome of one closure delivery pass.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClosureDeliveryReceipt {
    /// Turns transitioned to closed in this pass.
    pub closed: u64,
    /// Endpoint replies folded into the coordinator in this pass.
    pub replies_folded: u64,
    /// Terminal events observed for an unbound or already-closed action.
    pub orphan_terminals: u64,
    /// Endpoint replies refused for bounded reply capacity.
    pub refused_replies: u64,
    /// Advisory issues the reporter refused for bounded capacity.
    pub backpressured_issues: u64,
}

/// Bounded turn-closure contract consumed by the streaming pipeline.
#[async_trait(?Send)]
pub trait ConversationTurnCloser {
    /// Reserve one infallible terminal slot for a turn about to be dispatched.
    fn bind_turn(
        &mut self,
        binding: StreamingTurnBinding,
    ) -> Result<TurnClosureToken, SessionCoordinatorError>;

    /// Drain observed execution facts into the session coordinator.
    async fn deliver_closures(
        &mut self,
        coordinator: &mut dyn StreamingSessionCoordinator,
        output: &mut dyn DatasetActionSink,
    ) -> Result<ClosureDeliveryReceipt, SessionCoordinatorError>;

    /// Borrow the durable receipt for one closed turn.
    fn closed_turn(&self, action_id: StableActionId) -> Option<&ClosedTurnReceipt>;

    /// Return the greatest contiguous terminal action sequence.
    fn terminal_horizon(&self) -> TerminalActionHorizon;

    /// Return per-conversation in-flight counts.
    fn readiness(&self, scope: &ConversationSessionScope) -> Option<SessionClosureReadiness>;
}

#[derive(Debug)]
struct BoundTurn {
    binding: StreamingTurnBinding,
    reply_digest: Option<ContentDigest>,
    // Reserved at bind time so the terminal push is infallible; released with
    // the entry when the turn closes.
    _slot_lease: BudgetLease,
}

#[derive(Debug)]
enum IntakeEntry {
    Reply {
        action_id: StableActionId,
        bytes: Bytes,
        lease: BudgetLease,
    },
    ReplyRefused {
        action_id: StableActionId,
    },
    Terminal {
        action_id: StableActionId,
        disposition: ActionTerminalDisposition,
    },
}

/// Bounded worker-local queue shared by the terminal lane and the pipeline.
///
/// `Rc<RefCell<_>>` is deliberate rather than reflexive: the lane drain owner
/// and the pipeline task are the same current-thread runtime and `LocalSet`,
/// and neither `TurnRecordProcessor` nor the intake ever crosses a thread.
#[derive(Clone, Debug, Default)]
pub struct TurnClosureIntake {
    queue: Rc<RefCell<VecDeque<IntakeEntry>>>,
}

impl TurnClosureIntake {
    fn push(&self, entry: IntakeEntry) {
        self.queue.borrow_mut().push_back(entry);
    }

    fn pop(&self) -> Option<IntakeEntry> {
        self.queue.borrow_mut().pop_front()
    }

    /// Return the queued entry count, or `None` while the queue is borrowed.
    ///
    /// A `Some` result is the structural proof that no borrow outlived the
    /// call that produced the entries.
    #[must_use]
    pub fn queued(&self) -> Option<usize> {
        self.queue.try_borrow().ok().map(|queue| queue.len())
    }
}

/// Synchronous terminal-lane processor that never borrows the coordinator.
///
/// `TurnRecordProcessor::process` runs on the terminal lane's single
/// `spawn_local` drain owner. It therefore performs no await and no coordinator
/// mutation: it converts one reduced dispatch outcome into intake entries and
/// returns, leaving delivery to the pipeline task that owns the coordinator.
#[derive(Debug)]
pub struct StreamingTurnClosureProcessor {
    intake: TurnClosureIntake,
    credits: Rc<RefCell<BTreeMap<u64, StableActionId>>>,
    reply_budget: StreamingResourceBudget,
}

impl StreamingTurnClosureProcessor {
    /// Borrow the bounded intake this processor pushes into.
    #[must_use]
    pub const fn intake(&self) -> &TurnClosureIntake {
        &self.intake
    }
}

#[async_trait(?Send)]
impl TurnRecordProcessor for StreamingTurnClosureProcessor {
    async fn process(&self, credit: &IssuedCredit, outcome: &TurnDispatchOutcome) -> Result<()> {
        // The borrow is released by the end of this statement: the lane owner
        // must never hold it across the pushes below.
        let bound = self.credits.borrow().get(&credit.id).copied();
        let Some(action_id) = bound else {
            // Ordinary non-streaming workloads share this lane.
            return Ok(());
        };
        if !outcome.response_text.is_empty() {
            push_reply(
                &self.intake,
                &self.reply_budget,
                action_id,
                Bytes::copy_from_slice(outcome.response_text.as_bytes()),
            );
        }
        self.intake.push(IntakeEntry::Terminal {
            action_id,
            disposition: disposition_of(outcome.terminal),
        });
        Ok(())
    }
}

/// Charge and queue one endpoint reply, recording a refusal instead of losing it.
fn push_reply(
    intake: &TurnClosureIntake,
    reply_budget: &StreamingResourceBudget,
    action_id: StableActionId,
    bytes: Bytes,
) {
    match reply_budget.try_acquire(1, bytes.len()) {
        Ok(lease) => intake.push(IntakeEntry::Reply {
            action_id,
            bytes,
            lease,
        }),
        // An absent reply is recoverable state; an absent terminal is not, so
        // only the reply is refused here.
        Err(_) => intake.push(IntakeEntry::ReplyRefused { action_id }),
    }
}

/// Run-frozen construction inputs for one closure seam.
#[derive(Clone, Debug)]
pub struct ClosureSeamContext {
    /// Logical run owning every closed turn.
    pub run: StreamRunIdentity,
    /// Stable checkpoint-participant identity frozen in the run plan.
    pub participant_id: CheckpointParticipantId,
    /// Semantic namespace of the selected stream.
    pub stream_identity: ContentDigest,
    /// Maximum simultaneously bound turns.
    pub max_bound_turns: usize,
    /// Budget charged for one reserved terminal slot per bound turn.
    pub slot_budget: StreamingResourceBudget,
    /// Budget charged for retained endpoint reply bytes.
    pub reply_budget: StreamingResourceBudget,
    /// Budget charged for prepared checkpoint payloads.
    pub checkpoint_budget: StreamingResourceBudget,
    /// Host-owned reliability issue reporting boundary.
    pub issue_reporter: StreamingIssueReporterHandle,
}

#[derive(Clone, Copy, Debug, Default)]
struct ScopeCounts {
    bound: u64,
    closed: u64,
    next_ordinal: u64,
}

/// Run-scoped owner of conversation turn closure.
#[derive(Debug)]
pub struct ConversationClosureSeam {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    stream_identity: ContentDigest,
    max_bound_turns: usize,
    slot_budget: StreamingResourceBudget,
    reply_budget: StreamingResourceBudget,
    checkpoint_budget: StreamingResourceBudget,
    issue_reporter: StreamingIssueReporterHandle,
    intake: TurnClosureIntake,
    credits: Rc<RefCell<BTreeMap<u64, StableActionId>>>,
    bound: BTreeMap<StableActionId, BoundTurn>,
    closed: BTreeMap<StableActionId, ClosedTurnReceipt>,
    observed: BTreeMap<StableActionId, ConversationSessionScope>,
    counts: BTreeMap<ConversationSessionScope, ScopeCounts>,
    closed_sequences: BTreeSet<u64>,
    next_expected_sequence: u64,
    initialization: ParticipantInitialization,
}

impl ConversationClosureSeam {
    /// Construct one run-scoped closure seam.
    ///
    /// # Errors
    ///
    /// Returns a session failure when the bound-turn limit is zero, which is an
    /// unbounded causality state with extra steps.
    pub fn new(context: ClosureSeamContext) -> Result<Self, SessionCoordinatorError> {
        if context.max_bound_turns == 0 {
            return Err(SessionCoordinatorError::session(
                SessionFailureCode::UnboundedCausalityState,
            ));
        }
        Ok(Self {
            run: context.run,
            participant_id: context.participant_id,
            stream_identity: context.stream_identity,
            max_bound_turns: context.max_bound_turns,
            slot_budget: context.slot_budget,
            reply_budget: context.reply_budget,
            checkpoint_budget: context.checkpoint_budget,
            issue_reporter: context.issue_reporter,
            intake: TurnClosureIntake::default(),
            credits: Rc::new(RefCell::new(BTreeMap::new())),
            bound: BTreeMap::new(),
            closed: BTreeMap::new(),
            observed: BTreeMap::new(),
            counts: BTreeMap::new(),
            closed_sequences: BTreeSet::new(),
            next_expected_sequence: 0,
            initialization: ParticipantInitialization::default(),
        })
    }

    /// Return the lane processor that feeds this seam.
    #[must_use]
    pub fn processor(&self) -> Rc<StreamingTurnClosureProcessor> {
        Rc::new(StreamingTurnClosureProcessor {
            intake: self.intake.clone(),
            credits: Rc::clone(&self.credits),
            reply_budget: self.reply_budget.clone(),
        })
    }

    /// Borrow the bounded intake shared with the terminal lane.
    #[must_use]
    pub const fn intake(&self) -> &TurnClosureIntake {
        &self.intake
    }

    /// Return the sink decorator that observes emitted actions.
    pub fn observing_sink<'a>(
        &'a mut self,
        inner: &'a mut dyn DatasetActionSink,
    ) -> ClosureObservingSink<'a> {
        ClosureObservingSink { seam: self, inner }
    }

    /// Return the conversation an emitted action was observed under.
    #[must_use]
    pub fn observed_scope(&self, action_id: StableActionId) -> Option<ConversationSessionScope> {
        self.observed.get(&action_id).copied()
    }

    /// Queue one observed endpoint reply for a bound turn.
    ///
    /// Reply bytes are charged separately from the reserved terminal slot and
    /// may be refused; the turn still closes without them.
    pub fn observe_reply(&self, action_id: StableActionId, bytes: Bytes) {
        push_reply(&self.intake, &self.reply_budget, action_id, bytes);
    }

    /// Queue one observed terminal fact.
    ///
    /// The slot was reserved at bind time, so this cannot be refused.
    pub fn observe_terminal(
        &self,
        action_id: StableActionId,
        disposition: ActionTerminalDisposition,
    ) {
        self.intake.push(IntakeEntry::Terminal {
            action_id,
            disposition,
        });
    }

    /// Return the number of turns bound but not yet closed.
    #[must_use]
    pub fn inflight_turns(&self) -> usize {
        self.bound.len()
    }

    /// Return the number of retained closure receipts.
    #[must_use]
    pub fn closed_turns(&self) -> usize {
        self.closed.len()
    }

    fn note_emitted(&mut self, action: &ExecutableDatasetAction) {
        // P1 keys its live map by exactly this pair and `stream_identity` is
        // run-frozen, so the scope is exact rather than approximate.
        let scope = ConversationSessionScope::new(self.stream_identity, action.session_key());
        self.observed.insert(action.action_id(), scope);
        if let Some(turn) = self.bound.get_mut(&action.action_id()) {
            turn.binding.scope = scope;
        }
    }

    fn contiguous_terminal(&self) -> u64 {
        self.next_expected_sequence.saturating_sub(1)
    }

    fn advance_contiguous(&mut self, sequence: GlobalSequence) {
        self.closed_sequences.insert(sequence.get());
        // The horizon is the greatest *contiguous* closed sequence, never a
        // sparse maximum: publishing a gap would be an untruthful cut.
        while self.closed_sequences.remove(&self.next_expected_sequence) {
            self.next_expected_sequence = self.next_expected_sequence.saturating_add(1);
        }
    }

    async fn report_session_issue(
        &self,
        facts: &IssueFacts,
        class: StreamingIssueClass,
        failure: SessionCoordinatorError,
    ) -> u64 {
        let domain =
            StreamingInputDomainIdentity::new(self.stream_identity, facts.source_partition);
        let Ok(issue) = OrdinaryStreamingIssue::session(
            self.run,
            domain,
            facts.scope.session_key(),
            class,
            self.stream_identity,
            facts.source_position,
            0,
            ContentDigest::from_bytes(*facts.action_id.as_bytes()),
            OrdinaryStreamingFailure::Session(failure),
        ) else {
            return 0;
        };
        // No intake borrow and no coordinator borrow is live here.
        match self.issue_reporter.report(issue).await {
            Ok(StreamingIssueReportStatus::Accepted) | Err(_) => 0,
            Ok(StreamingIssueReportStatus::Backpressured) => 1,
        }
    }

    /// Facts describing an orphan terminal, when the ledger retains any.
    fn orphan_facts(&self, action_id: StableActionId) -> Option<IssueFacts> {
        let receipt = self.closed.get(&action_id)?;
        Some(IssueFacts {
            action_id,
            scope: receipt.scope,
            source_position: receipt.source_position,
            source_partition: receipt.source_partition,
        })
    }

    fn encode_within_budget(
        &self,
    ) -> Result<
        (
            Vec<u8>,
            BudgetLease,
            Option<SourcePosition>,
            Option<u64>,
            u64,
        ),
        CheckpointError,
    > {
        let mut retained: Vec<ClosedTurnReceipt> = self.closed.values().cloned().collect();
        let mut first_unrepresented_position: Option<SourcePosition> = None;
        let mut first_unrepresented_sequence: Option<u64> = None;
        loop {
            let state = ClosureCheckpointStateV1 {
                stream_identity: self.stream_identity,
                next_expected_sequence: self.next_expected_sequence,
                closed: retained,
            };
            let bytes = rmp_serde::to_vec(&state).map_err(|error| CheckpointError::Storage {
                message: format!("could not encode closure ledger state: {error}"),
            })?;
            match self.checkpoint_budget.try_acquire(1, bytes.len()) {
                Ok(lease) => {
                    let item_count = u64::try_from(state.closed.len()).unwrap_or(u64::MAX);
                    return Ok((
                        bytes,
                        lease,
                        first_unrepresented_position,
                        first_unrepresented_sequence,
                        item_count,
                    ));
                }
                Err(BudgetError::Closed) => {
                    return Err(CheckpointError::ParticipantUnavailable {
                        participant: self.participant_id.clone(),
                    });
                }
                Err(_) => {}
            }
            retained = state.closed;
            // Roll back rather than truncate: whole receipts leave the payload
            // and both horizons move below the first one they no longer hold.
            let Some(dropped) = retained.pop() else {
                return Err(CheckpointError::StateBudget {
                    participant: self.participant_id.clone(),
                    code: StateBudgetFailureCode::ByteCapacity,
                });
            };
            first_unrepresented_position = Some(match first_unrepresented_position {
                Some(existing) if existing <= dropped.source_position => existing,
                _ => dropped.source_position,
            });
            let dropped_sequence = dropped.global_sequence.get();
            first_unrepresented_sequence = Some(match first_unrepresented_sequence {
                Some(existing) if existing <= dropped_sequence => existing,
                _ => dropped_sequence,
            });
        }
    }

    fn restore_ledger(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        let Some(state) = state else {
            return Ok(());
        };
        let decoded: ClosureCheckpointStateV1 = rmp_serde::from_slice(state.payload_bytes())
            .map_err(|error| CheckpointError::Storage {
                message: format!("could not decode closure ledger state: {error}"),
            })?;
        if decoded.stream_identity != self.stream_identity {
            return Err(CheckpointError::ObjectVerification);
        }
        for receipt in decoded.closed {
            let counts = self.counts.entry(receipt.scope).or_default();
            counts.bound = counts.bound.saturating_add(1);
            counts.closed = counts.closed.saturating_add(1);
            counts.next_ordinal = counts
                .next_ordinal
                .max(receipt.closure_ordinal.get().saturating_add(1));
            self.closed_sequences.insert(receipt.global_sequence.get());
            self.closed.insert(receipt.action_id, receipt);
        }
        self.next_expected_sequence = decoded.next_expected_sequence;
        while self.closed_sequences.remove(&self.next_expected_sequence) {
            self.next_expected_sequence = self.next_expected_sequence.saturating_add(1);
        }
        Ok(())
    }

    /// Release receipts a committed generation already proves durable.
    fn release_through(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.participant_id() != &self.participant_id {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        let through = receipt.represented_cut().terminal.get().get();
        // Unlike the session coordinator, this participant has a real release
        // point: a closure covered by a committed generation is proven durable
        // and its live map entry is redundant. The per-scope counts stay, so
        // readiness is unchanged by the release.
        self.closed
            .retain(|_, entry| entry.global_sequence.get() > through);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct IssueFacts {
    action_id: StableActionId,
    scope: ConversationSessionScope,
    source_position: SourcePosition,
    source_partition: ImmutableObjectIdentity,
}

#[async_trait(?Send)]
impl ConversationTurnCloser for ConversationClosureSeam {
    fn bind_turn(
        &mut self,
        binding: StreamingTurnBinding,
    ) -> Result<TurnClosureToken, SessionCoordinatorError> {
        // A restart re-emits an in-flight turn under its identical stable
        // identity, so rebinding is idempotent rather than a second turn.
        if self.closed.contains_key(&binding.action_id)
            || self.bound.contains_key(&binding.action_id)
        {
            return Ok(TurnClosureToken::AlreadyBound);
        }
        if self.bound.len() >= self.max_bound_turns {
            return Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            ));
        }
        // Charged before dispatch so the terminal push is infallible.
        let lease = self
            .slot_budget
            .try_acquire(1, size_of::<BoundTurn>())
            .map_err(map_budget_error)?;
        let scope = self
            .observed
            .get(&binding.action_id)
            .copied()
            .unwrap_or(binding.scope);
        let counts = self.counts.entry(scope).or_default();
        counts.bound = counts.bound.saturating_add(1);
        self.credits
            .borrow_mut()
            .insert(binding.credit_id, binding.action_id);
        let action_id = binding.action_id;
        self.bound.insert(
            action_id,
            BoundTurn {
                binding: StreamingTurnBinding { scope, ..binding },
                reply_digest: None,
                _slot_lease: lease,
            },
        );
        Ok(TurnClosureToken::Bound)
    }

    async fn deliver_closures(
        &mut self,
        coordinator: &mut dyn StreamingSessionCoordinator,
        output: &mut dyn DatasetActionSink,
    ) -> Result<ClosureDeliveryReceipt, SessionCoordinatorError> {
        let mut receipt = ClosureDeliveryReceipt::default();
        // Each entry is moved out before any await: no `RefCell` borrow and no
        // reporter borrow is ever live across a suspension point.
        while let Some(entry) = self.intake.pop() {
            match entry {
                IntakeEntry::Reply {
                    action_id,
                    bytes,
                    lease,
                } => {
                    let Some(turn) = self.bound.get_mut(&action_id) else {
                        continue;
                    };
                    let event = next_event(&turn.binding.event_identity, 1);
                    let digest = ContentDigest::from_bytes(*blake3::hash(&bytes).as_bytes());
                    let payload =
                        BudgetedActionUpdate::new(bytes, lease).map_err(|_| budget_invariant())?;
                    turn.reply_digest = Some(digest);
                    // The reply is folded before its terminal: delivering the
                    // terminal first would settle a turn whose reply the
                    // transcript does not yet hold.
                    coordinator
                        .observe_execution(
                            ActionExecutionEvent::SessionUpdate(EndpointSessionUpdate {
                                event,
                                payload,
                            }),
                            output,
                        )
                        .await?;
                    receipt.replies_folded = receipt.replies_folded.saturating_add(1);
                }
                IntakeEntry::ReplyRefused { action_id } => {
                    receipt.refused_replies = receipt.refused_replies.saturating_add(1);
                    let facts = self.bound.get(&action_id).map(|turn| IssueFacts {
                        action_id,
                        scope: turn.binding.scope,
                        source_position: turn.binding.source_position,
                        source_partition: turn.binding.source_partition,
                    });
                    // Absent reply state made observable, never a fatal
                    // invariant: the turn still closes without the reply.
                    if let Some(facts) = facts {
                        receipt.backpressured_issues = receipt.backpressured_issues.saturating_add(
                            self.report_session_issue(
                                &facts,
                                StreamingIssueClass::Capacity,
                                SessionCoordinatorError::state_budget(
                                    StateBudgetFailureCode::ByteCapacity,
                                ),
                            )
                            .await,
                        );
                    }
                }
                IntakeEntry::Terminal {
                    action_id,
                    disposition,
                } => {
                    let Some(turn) = self.bound.remove(&action_id) else {
                        receipt.orphan_terminals = receipt.orphan_terminals.saturating_add(1);
                        let facts = self.orphan_facts(action_id);
                        if let Some(facts) = facts {
                            receipt.backpressured_issues =
                                receipt.backpressured_issues.saturating_add(
                                    self.report_session_issue(
                                        &facts,
                                        StreamingIssueClass::Permanent,
                                        SessionCoordinatorError::session(
                                            SessionFailureCode::UnclaimedTerminal,
                                        ),
                                    )
                                    .await,
                                );
                        }
                        continue;
                    };
                    let binding = turn.binding;
                    let event = next_event(&binding.event_identity, 2);
                    coordinator
                        .observe_execution(
                            ActionExecutionEvent::Terminal(ActionTerminalReceipt {
                                event,
                                disposition,
                            }),
                            output,
                        )
                        .await?;
                    let counts = self.counts.entry(binding.scope).or_default();
                    let ordinal = TurnClosureOrdinal(counts.next_ordinal);
                    counts.next_ordinal = ordinal.next()?.get();
                    counts.closed = counts.closed.saturating_add(1);
                    self.credits.borrow_mut().remove(&binding.credit_id);
                    self.closed.insert(
                        action_id,
                        ClosedTurnReceipt {
                            scope: binding.scope,
                            action_id,
                            closure_ordinal: ordinal,
                            global_sequence: binding.global_sequence,
                            disposition: disposition.into(),
                            reply_digest: turn.reply_digest,
                            stable_session_ordinal: binding.scope.stable_ordinal(),
                            source_position: binding.source_position,
                            source_partition: binding.source_partition,
                        },
                    );
                    self.advance_contiguous(binding.global_sequence);
                    receipt.closed = receipt.closed.saturating_add(1);
                }
            }
        }
        Ok(receipt)
    }

    fn closed_turn(&self, action_id: StableActionId) -> Option<&ClosedTurnReceipt> {
        self.closed.get(&action_id)
    }

    fn terminal_horizon(&self) -> TerminalActionHorizon {
        TerminalActionHorizon::new(GlobalSequence::new(self.contiguous_terminal()))
    }

    fn readiness(&self, scope: &ConversationSessionScope) -> Option<SessionClosureReadiness> {
        let counts = self.counts.get(scope)?;
        Some(SessionClosureReadiness {
            scope: *scope,
            bound_turns: counts.bound,
            closed_turns: counts.closed,
        })
    }
}

/// Sink decorator that learns which action belongs to which conversation.
///
/// It adds no charge and mutates no action: the wrapped sink still receives the
/// exact move-only value the session coordinator emitted.
pub struct ClosureObservingSink<'a> {
    seam: &'a mut ConversationClosureSeam,
    inner: &'a mut dyn DatasetActionSink,
}

#[async_trait(?Send)]
impl DatasetActionSink for ClosureObservingSink<'_> {
    async fn send_action(
        &mut self,
        action: ExecutableDatasetAction,
    ) -> Result<(), SessionCoordinatorError> {
        self.seam.note_emitted(&action);
        self.inner.send_action(action).await
    }

    async fn advance_causal_frontier(
        &mut self,
        frontier: SessionCausalFrontier,
    ) -> Result<(), SessionCoordinatorError> {
        self.inner.advance_causal_frontier(frontier).await
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for ConversationClosureSeam {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let (bytes, lease, first_position, first_sequence, item_count) =
            self.encode_within_budget()?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(bytes), lease)?;
        let mut represented = barrier.cut.clone();
        if let Some(position) = first_position {
            represented.decoded = DecodeHorizon::new(position);
        }
        let contiguous = self.contiguous_terminal();
        let terminal = match first_sequence {
            Some(sequence) => contiguous.min(sequence.saturating_sub(1)),
            None => contiguous,
        };
        represented.terminal = TerminalActionHorizon::new(GlobalSequence::new(terminal));
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            CLOSURE_PARTICIPANT_ID,
            CLOSURE_CHECKPOINT_SCHEMA_VERSION,
            represented,
            item_count,
            payload,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.restore_ledger(state)
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        self.release_through(receipt)
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ClosureCheckpointStateV1 {
    stream_identity: ContentDigest,
    next_expected_sequence: u64,
    closed: Vec<ClosedTurnReceipt>,
}

/// Derive one event identity from the host-minted prototype.
///
/// Attempt identity stays with the action host, which is the sole owner of the
/// run incarnation; the seam varies only the event ordinal.
fn next_event(prototype: &ActionEventIdentity, offset: u64) -> ActionEventIdentity {
    ActionEventIdentity {
        action_id: prototype.action_id,
        attempt_id: prototype.attempt_id,
        ownership_epoch: prototype.ownership_epoch,
        event_ordinal: prototype.event_ordinal.saturating_add(offset),
    }
}

const fn budget_invariant() -> SessionCoordinatorError {
    SessionCoordinatorError::state_budget(StateBudgetFailureCode::ByteCapacity)
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
