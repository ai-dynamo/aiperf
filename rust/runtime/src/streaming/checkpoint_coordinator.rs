// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Single-writer checkpoint publication sequencing and post-CAS notification.
//!
//! One coordinator owns one logical run, one checkpoint backend, and the exact
//! frozen participant set. It converts one [`CheckpointBarrier`] into one
//! atomically published generation and one idempotent notification fan-out.
//!
//! The coordinator selects nothing. Its backend is injected already prepared by
//! the checkpoint backend factory registry, and its [`StreamRunIdentity`] is
//! resolved by the product run-lifecycle owner. The coordinator only ever
//! *verifies* both: it computes no digest, writes no object, acquires no budget
//! lease, and takes no lock.

use tracing::debug;

use super::{
    checkpoint::{
        CheckpointBarrier, CheckpointEpoch, CheckpointError, CheckpointGeneration,
        CheckpointParticipantId, CheckpointParticipantPlan, CheckpointTerminalReason,
        CommittedCheckpointGeneration, CommittedParticipantReceipt, ParticipantStateDescriptor,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations, CurrentV4CheckpointGeneration,
        LeasedCheckpointGenerationView, StreamingCheckpointBackend,
    },
    reliability::{
        HandledIssueCut, PreparedIssueReceiptResultPartition, StreamingIssueDisposition,
        StreamingIssueReporter, classify_checkpoint_attempt_failure,
    },
    results::ResultPartition,
};

/// Ordinary and detailed-receipt result inputs prepared for one barrier.
///
/// The two fields move together so a refused staging cannot consume one without
/// the other. The reliability ledger's receipt-partition producer builds the
/// value; the coordinator is its only consumer.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::checkpoint_coordinator::PreparedCheckpointResultInput;
/// # fn cannot_separate(value: PreparedCheckpointResultInput) {
/// let _partitions = value.partitions;
/// # }
/// ```
pub struct PreparedCheckpointResultInput {
    partitions: Vec<ResultPartition>,
    issue_receipts: Option<PreparedIssueReceiptResultPartition>,
}

impl PreparedCheckpointResultInput {
    /// Carry ordinary partitions and at most one detailed-receipt partition.
    #[must_use]
    pub fn new(
        partitions: Vec<ResultPartition>,
        issue_receipts: Option<PreparedIssueReceiptResultPartition>,
    ) -> Self {
        Self {
            partitions,
            issue_receipts,
        }
    }

    /// Carry no result input at all.
    #[must_use]
    pub fn empty() -> Self {
        Self::new(Vec::new(), None)
    }

    /// Borrow the ordinary partitions without disturbing either authority.
    #[must_use]
    pub fn partitions(&self) -> &[ResultPartition] {
        &self.partitions
    }

    /// Borrow the detailed-receipt handoff, when one was prepared.
    #[must_use]
    pub const fn issue_receipts(&self) -> Option<&PreparedIssueReceiptResultPartition> {
        self.issue_receipts.as_ref()
    }

    /// Expose both mutable staging inputs to the backend transaction.
    fn stage_inputs(
        &mut self,
    ) -> (
        &mut Vec<ResultPartition>,
        &mut Option<PreparedIssueReceiptResultPartition>,
    ) {
        (&mut self.partitions, &mut self.issue_receipts)
    }
}

impl std::ops::Deref for PreparedCheckpointResultInput {
    type Target = [ResultPartition];

    fn deref(&self) -> &[ResultPartition] {
        self.partitions()
    }
}

/// One published barrier and the generation it made authoritative.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublishedBarrier {
    barrier: CheckpointBarrier,
    committed: CommittedCheckpointGeneration,
}

impl PublishedBarrier {
    /// Borrow the exact barrier that produced this publication.
    #[must_use]
    pub const fn barrier(&self) -> &CheckpointBarrier {
        &self.barrier
    }

    /// Borrow the authoritative committed generation.
    #[must_use]
    pub const fn committed(&self) -> &CommittedCheckpointGeneration {
        &self.committed
    }
}

/// Finality selected by the caller for one barrier publication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointBarrierFinality {
    /// The run continues after this generation.
    Continuing,
    /// This generation terminates the logical run for the stated reason.
    Terminal(CheckpointTerminalReason),
}

/// Reliability routing selected for one pre-CAS publication failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreCasFailureRouting {
    /// The host classifier verified a terminal invariant; the run must fail.
    FailRun,
    /// Bounded capacity was unavailable; fence admission and retry.
    CapacityBackpressure,
    /// The attempt failed without invariant violation; retry the same head.
    Retryable,
}

/// Single-writer checkpoint publication sequencer for one logical run.
pub struct StreamingCheckpointCoordinator {
    run: StreamRunIdentity,
    backend: Box<dyn StreamingCheckpointBackend>,
    plan: CheckpointParticipantPlan,
    generation_expectations: CheckpointGenerationExpectations,
    /// Every non-reporter owner, in no required order; the plan defines the set.
    participants: Vec<Box<dyn StreamingCheckpointParticipant>>,
    /// The reliability ledger, which is also a participant in `plan`.
    reporter: Box<dyn StreamingIssueReporter>,
    /// Cloneable immutable expected-head identity. Never a reader token.
    expected: Option<CheckpointGeneration>,
    /// The most recent publication, retained for exact-barrier idempotency.
    published: Option<PublishedBarrier>,
    /// Whether `published` still owes at least one participant its callback.
    is_notification_pending: bool,
    /// Routing the classifier assigned to the most recent pre-CAS failure.
    last_pre_cas_routing: Option<PreCasFailureRouting>,
}

impl StreamingCheckpointCoordinator {
    /// Construct a coordinator over one resolved run and one prepared backend.
    ///
    /// `initial_expected` is `None` for a fresh run and the exact generation
    /// identity of the verified restored reader for a resume. The constructor
    /// resolves nothing: the run and the head identity are both supplied by the
    /// product run-lifecycle owner.
    pub fn new(
        run: StreamRunIdentity,
        backend: Box<dyn StreamingCheckpointBackend>,
        generation_expectations: CheckpointGenerationExpectations,
        participants: Vec<Box<dyn StreamingCheckpointParticipant>>,
        reporter: Box<dyn StreamingIssueReporter>,
        initial_expected: Option<CheckpointGeneration>,
    ) -> Result<Self, CheckpointError> {
        if generation_expectations.run != run {
            return Err(CheckpointError::ObjectVerification);
        }
        let plan = generation_expectations.participant_plan.clone();
        let mut observed = participants
            .iter()
            .map(|participant| participant.participant_id())
            .collect::<Vec<_>>();
        observed.push(reporter.participant_id());
        observed.sort_unstable();
        if observed.iter().ne(plan.ids().iter()) {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        Ok(Self {
            run,
            backend,
            plan,
            generation_expectations,
            participants,
            reporter,
            expected: initial_expected,
            published: None,
            is_notification_pending: false,
            last_pre_cas_routing: None,
        })
    }

    /// Borrow the retained expected-head identity.
    #[must_use]
    pub const fn expected(&self) -> Option<&CheckpointGeneration> {
        self.expected.as_ref()
    }

    /// Borrow the generation whose notifications are still owed, if any.
    #[must_use]
    pub fn pending_notification_generation(&self) -> Option<&CheckpointGeneration> {
        if !self.is_notification_pending {
            return None;
        }
        self.published
            .as_ref()
            .map(|published| published.committed.generation_ref())
    }

    /// Borrow the complete retained publication.
    #[must_use]
    pub const fn published_barrier(&self) -> Option<&PublishedBarrier> {
        self.published.as_ref()
    }

    /// Return the reliability routing of the most recent pre-CAS failure.
    #[must_use]
    pub const fn last_pre_cas_routing(&self) -> Option<PreCasFailureRouting> {
        self.last_pre_cas_routing
    }

    /// Publish one barrier atomically and notify every participant.
    pub async fn commit_barrier(
        &mut self,
        barrier: CheckpointBarrier,
        results: &mut PreparedCheckpointResultInput,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        self.commit_barrier_with_finality(barrier, CheckpointBarrierFinality::Continuing, results)
            .await
    }

    /// Publish one barrier with explicit finality.
    pub async fn commit_barrier_with_finality(
        &mut self,
        barrier: CheckpointBarrier,
        finality: CheckpointBarrierFinality,
        results: &mut PreparedCheckpointResultInput,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        // Run and frozen plan first: neither a pending retry nor the caller's
        // move-only inputs may be touched on behalf of a foreign barrier.
        self.validate_barrier(&barrier)
            .inspect_err(|error| self.route_pre_cas_failure(error))?;

        // Pending publication retry precedes any inspection of `results`, so a
        // failure or cancellation here leaves both the retained publication and
        // every newly supplied uncommitted input exactly as they were.
        if self.is_notification_pending {
            let (is_exact_repeat, committed) = {
                let published = self
                    .published
                    .as_ref()
                    .ok_or(CheckpointError::ObjectVerification)?;
                (published.barrier == barrier, published.committed.clone())
            };
            self.notify_committed(&committed).await?;
            // Synchronous clear, with no intervening cancellation point.
            self.is_notification_pending = false;
            if is_exact_repeat {
                return Ok(committed);
            }
        } else if let Some(published) = self.published.as_ref()
            && published.barrier == barrier
        {
            return Ok(published.committed.clone());
        }

        let committed = match self.publish(&barrier, finality, results).await {
            Ok(committed) => committed,
            Err(error) => {
                self.route_pre_cas_failure(&error);
                return Err(error);
            }
        };

        // Post-CAS authority transition, before any fallible callback.
        self.expected = Some(committed.generation());
        self.published = Some(PublishedBarrier {
            barrier,
            committed: committed.clone(),
        });
        self.is_notification_pending = true;
        debug!(
            run = ?self.run,
            epoch = ?committed.generation_ref().epoch(),
            component = "streaming.checkpoint.coordinator",
            "published checkpoint generation"
        );
        self.notify_committed(&committed).await?;
        self.is_notification_pending = false;
        Ok(committed)
    }

    /// Replay committed notifications idempotently after a restart.
    pub async fn replay_committed_notifications(
        &mut self,
        committed: &CommittedCheckpointGeneration,
    ) -> Result<(), CheckpointError> {
        self.notify_committed(committed).await
    }

    fn validate_barrier(&self, barrier: &CheckpointBarrier) -> Result<(), CheckpointError> {
        if barrier.run != self.run
            || self.generation_expectations.run != self.run
            || barrier.plan_digest != self.generation_expectations.execution_plan_digest
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }

    /// Pass one pre-CAS attempt failure through the reliability classifier.
    ///
    /// Only a reporter-checked terminal invariant becomes [`PreCasFailureRouting::FailRun`];
    /// everything else is retryable at the same expected head, with bounded
    /// capacity separated out so the caller can fence admission truthfully.
    fn route_pre_cas_failure(&mut self, error: &CheckpointError) {
        let routing = match classify_checkpoint_attempt_failure(error) {
            Some(decision) if decision.disposition() == StreamingIssueDisposition::FailRun => {
                PreCasFailureRouting::FailRun
            }
            Some(_) => PreCasFailureRouting::Retryable,
            None => match error {
                CheckpointError::StateBudget { .. }
                | CheckpointError::BackendBudget { .. }
                | CheckpointError::ResultIndexReadBudgetTooSmall { .. } => {
                    PreCasFailureRouting::CapacityBackpressure
                }
                _ => PreCasFailureRouting::Retryable,
            },
        };
        self.last_pre_cas_routing = Some(routing);
    }

    /// One atomic publication attempt. Every failure here is pre-CAS.
    async fn publish(
        &mut self,
        barrier: &CheckpointBarrier,
        finality: CheckpointBarrierFinality,
        results: &mut PreparedCheckpointResultInput,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        let views = self.collect_views(barrier).await?;
        self.validate_exact_set(&views)?;
        self.validate_issue_receipt_input(barrier, &views, results)?;
        let metadata = self.metadata(barrier, finality)?;

        // Early stale-head detection. Opening before any staging means a
        // concurrent advance is refused before a whole epoch has been charged,
        // and the minted move-only authority is the only thing that can follow
        // the head into the transaction.
        let expected = match self.expected.clone() {
            None => None,
            Some(expected) => Some(self.open_verified_current_predecessor(&expected).await?),
        };

        let mut transaction = self
            .backend
            .begin_generation(self.run, expected, self.generation_expectations.clone())
            .await?;
        for view in views {
            transaction.stage_participant(view).await?;
        }
        let (partitions, issue_receipts) = results.stage_inputs();
        let prepared = transaction
            .stage_results(partitions, issue_receipts)
            .await?;

        // The staged index root is only knowable now, so this is the earliest
        // point the ledger can bind it, and it must precede CAS.
        self.reporter
            .bind_prepared_result_epoch(&prepared)
            .map_err(|_| CheckpointError::ObjectVerification)?;

        let staged_index_root = *prepared.index_root();
        let committed = transaction.commit(metadata).await?;

        // Post-fence accounting check. The head is already authoritative, so a
        // mismatch is corruption and must never be silently tolerated.
        if committed.result_index_root() != &staged_index_root || committed.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(committed)
    }

    async fn collect_views(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<Vec<PreparedParticipantState>, CheckpointError> {
        let mut views = Vec::with_capacity(self.participants.len() + 1);
        for participant in &mut self.participants {
            views.push(participant.checkpoint_view(barrier).await?);
        }
        views.push(self.reporter.checkpoint_view(barrier).await?);
        Ok(views)
    }

    fn validate_exact_set(
        &self,
        views: &[PreparedParticipantState],
    ) -> Result<(), CheckpointError> {
        if views.iter().any(|view| view.run() != &self.run) {
            return Err(CheckpointError::ObjectVerification);
        }
        let mut observed = views
            .iter()
            .map(|view| view.descriptor().participant_id.clone())
            .collect::<Vec<_>>();
        observed.sort_unstable();
        if observed.iter().ne(self.plan.ids().iter()) {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        Ok(())
    }

    /// Require exactly one issue-receipt partition agreeing with the ledger.
    ///
    /// The ledger's own participant view carries the handled cut it wired at
    /// this barrier, so the equality is a pure comparison over values already
    /// held: no second receipt-partition view, no extra budget charge, and no
    /// cloned detailed receipt.
    fn validate_issue_receipt_input(
        &self,
        barrier: &CheckpointBarrier,
        views: &[PreparedParticipantState],
        results: &PreparedCheckpointResultInput,
    ) -> Result<(), CheckpointError> {
        let ledger_id = self.reporter.participant_id();
        let handled = views
            .iter()
            .find(|view| view.descriptor().participant_id == ledger_id)
            .map(|view| &view.descriptor().represented_cut.handled_issues)
            .ok_or(CheckpointError::ParticipantSetMismatch)?;
        if &barrier.cut.handled_issues != handled {
            return Err(CheckpointError::ObjectVerification);
        }
        match results.issue_receipts() {
            Some(receipts) => {
                if receipts.run() != &self.run
                    || receipts.barrier_epoch() != barrier.epoch
                    || receipts.receipt_root() != handled.receipt_root()
                    || receipts.handled_cut() != handled
                {
                    return Err(CheckpointError::ObjectVerification);
                }
                Ok(())
            }
            // Admissible only against the canonical empty cut, and refused for
            // any retained receipt, input-frontier, or tombstone authority.
            None if handled == &HandledIssueCut::empty() => Ok(()),
            None => Err(CheckpointError::ObjectVerification),
        }
    }

    fn metadata(
        &self,
        barrier: &CheckpointBarrier,
        finality: CheckpointBarrierFinality,
    ) -> Result<CheckpointCommitMetadata, CheckpointError> {
        let epoch = match self.expected.as_ref() {
            None => CheckpointEpoch::new(1),
            Some(previous) => {
                CheckpointEpoch::new(previous.epoch().get().checked_add(1).ok_or_else(|| {
                    CheckpointError::GenerationEpochOverflow {
                        previous: previous.clone(),
                    }
                })?)
            }
        };
        if barrier.epoch != epoch {
            return Err(CheckpointError::GenerationConflict {
                expected: self.expected.clone(),
                actual: None,
            });
        }
        let (is_final, terminal_reason) = match finality {
            CheckpointBarrierFinality::Continuing => (false, None),
            CheckpointBarrierFinality::Terminal(reason) => (true, Some(reason)),
        };
        Ok(CheckpointCommitMetadata {
            previous: self.expected.clone(),
            epoch,
            cut: barrier.cut.clone(),
            execution_plan_digest: self.generation_expectations.execution_plan_digest,
            result_plan_digest: self.generation_expectations.result_plan_digest,
            is_final,
            terminal_reason,
        })
    }

    /// Open the current head and mint successor authority for the exact
    /// retained expectation.
    ///
    /// A concurrent advance is reported without mutating `self.expected`, so it
    /// is refused rather than adopted. A legacy read-only head has no successor
    /// authority at all and refuses here rather than at transaction start.
    async fn open_verified_current_predecessor(
        &self,
        expected: &CheckpointGeneration,
    ) -> Result<CurrentV4CheckpointGeneration, CheckpointError> {
        let opened = self
            .backend
            .open_latest(&self.run, &self.generation_expectations)
            .await?
            .ok_or_else(|| CheckpointError::GenerationConflict {
                expected: Some(expected.clone()),
                actual: None,
            })?;
        match opened.view() {
            LeasedCheckpointGenerationView::CurrentV4(reader) => {
                reader.current_v4_predecessor(expected)
            }
            LeasedCheckpointGenerationView::LegacyV3ReadOnly(_) => {
                Err(CheckpointError::LegacyReadOnlyHead)
            }
        }
    }

    /// Deliver one idempotent receipt to every participant in the plan.
    async fn notify_committed(
        &mut self,
        committed: &CommittedCheckpointGeneration,
    ) -> Result<(), CheckpointError> {
        if committed.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        // The descriptors are borrowed from the committed generation the caller
        // owns, so cloning the identity list keeps `self` mutably borrowable
        // across the fan-out without cloning any payload.
        let descriptors = committed.participant_descriptors().to_vec();
        for descriptor in &descriptors {
            let receipt = CommittedParticipantReceipt::new(committed, descriptor)?;
            if receipt.run() != &self.run {
                return Err(CheckpointError::ObjectVerification);
            }
            self.notify_one(&descriptor.participant_id, &receipt)
                .await?;
        }
        Ok(())
    }

    async fn notify_one(
        &mut self,
        participant_id: &CheckpointParticipantId,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if &self.reporter.participant_id() == participant_id {
            return self.reporter.checkpoint_committed(receipt).await;
        }
        let participant = self
            .participants
            .iter_mut()
            .find(|participant| &participant.participant_id() == participant_id)
            .ok_or(CheckpointError::ParticipantSetMismatch)?;
        participant.checkpoint_committed(receipt).await
    }
}

/// Borrow the descriptor a receipt was minted from, for callers that only hold
/// the committed generation.
#[must_use]
pub fn committed_descriptor<'a>(
    committed: &'a CommittedCheckpointGeneration,
    participant_id: &CheckpointParticipantId,
) -> Option<&'a ParticipantStateDescriptor> {
    committed
        .participant_descriptors()
        .iter()
        .find(|descriptor| &descriptor.participant_id == participant_id)
}
