// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fixtures for the checkpoint coordinator suite.
//!
//! These live apart from `support/streaming_checkpoint.rs` so the coordinator
//! task never collides with the concurrently executing backend, GC, and result
//! index tasks that all extend the shared file.

use std::{cell::Cell, cell::RefCell, rc::Rc};

use async_trait::async_trait;
use bytes::Bytes;

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, BudgetedCheckpointBytes, CheckpointBarrier,
        CheckpointCut, CheckpointEpoch, CheckpointError, CheckpointGeneration,
        CheckpointParticipantId, CheckpointParticipantPlan, CommittedCheckpointGeneration,
        CommittedParticipantReceipt, CommittedParticipantState, DecodeHorizon, DiscoveryHorizon,
        EventTimeWatermark, OrderedActionHorizon, ParticipantInitialization,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
        TerminalActionHorizon,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations, CurrentV4CheckpointGeneration,
        LeasedCheckpointGeneration, StreamingCheckpointBackend, StreamingGenerationTransaction,
    },
    checkpoint_coordinator::StreamingCheckpointCoordinator,
    checkpoints::memory::{
        ImmutableObjectInventory, MemoryCheckpointBackend, MemoryCheckpointLimits,
    },
    identity::{ContentDigest, GlobalSequence, LogicalReplayRunId, SessionCausalFrontier},
    reliability::{
        HandledIssueCut, OrdinaryStreamingIssue, StreamingIssueReportError,
        StreamingIssueReportStatus, StreamingIssueReporter, StreamingIssueReporterEndpoint,
        StreamingIssueReporterHandle,
    },
    results::{PreparedResultEpoch, ResultPartition},
    unit::{EventTimeUtc, SourcePosition},
};

pub const PARTICIPANT_ID: &str = "coordinator_participant";
pub const LEDGER_ID: &str = "streaming_issue_ledger";
pub const PLAN_DIGEST: [u8; 32] = [0x31; 32];

pub fn run_id(value: u8) -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([value; 32]))
}

pub fn cut_at(value: u64) -> CheckpointCut {
    let event_time = EventTimeUtc::new(i64::try_from(value).unwrap_or(i64::MAX))
        .expect("non-negative test event time");
    CheckpointCut {
        discovered: DiscoveryHorizon::new(SourcePosition::new(value)),
        acquired: AcquisitionHorizon::new(SourcePosition::new(value)),
        decoded: DecodeHorizon::new(SourcePosition::new(value)),
        ordered: OrderedActionHorizon::new(GlobalSequence::new(value)),
        admitted: AdmissionHorizon::new(GlobalSequence::new(value)),
        terminal: TerminalActionHorizon::new(GlobalSequence::new(value)),
        event_watermark: EventTimeWatermark::Hard {
            through: event_time,
        },
        causal_frontier: SessionCausalFrontier {
            through_sequence: GlobalSequence::new(value),
            event_time: Some(event_time),
            digest: ContentDigest::from_bytes([value as u8; 32]),
        },
        handled_issues: HandledIssueCut::empty(),
    }
}

pub fn barrier_for_run(run: StreamRunIdentity, epoch: u64) -> CheckpointBarrier {
    CheckpointBarrier {
        run,
        epoch: CheckpointEpoch::new(epoch),
        cut: cut_at(epoch),
        plan_digest: ContentDigest::from_bytes(PLAN_DIGEST),
    }
}

pub fn barrier_at(epoch: u64) -> CheckpointBarrier {
    barrier_for_run(run_id(1), epoch)
}

pub fn expectations(run: StreamRunIdentity) -> CheckpointGenerationExpectations {
    CheckpointGenerationExpectations {
        run,
        participant_plan: CheckpointParticipantPlan::new([
            CheckpointParticipantId::new(PARTICIPANT_ID),
            CheckpointParticipantId::new(LEDGER_ID),
        ])
        .expect("valid coordinator participant plan"),
        execution_plan_digest: ContentDigest::from_bytes(PLAN_DIGEST),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
    }
}

pub fn backend_limits() -> MemoryCheckpointLimits {
    let limits = BudgetLimits {
        max_items: 128,
        max_bytes: 1_048_576,
    };
    MemoryCheckpointLimits {
        transactions: limits,
        prepared_indexes: limits,
        storage: limits,
        result_summaries: limits,
        reads: limits,
    }
}

pub async fn checkpoint_payload(bytes: Bytes) -> BudgetedCheckpointBytes {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: bytes.len().max(1),
    })
    .expect("valid test budget");
    let lease = budget
        .acquire(1, bytes.len())
        .await
        .expect("checkpoint payload budget");
    BudgetedCheckpointBytes::new(bytes, lease).expect("exact payload charge")
}

/// Shared observable state for one participant handed to a coordinator.
#[derive(Default)]
pub struct ParticipantControl {
    view_calls: Cell<u64>,
    commit_notifications: Cell<u64>,
    failing_notifications: Cell<u64>,
    last_receipt: RefCell<Option<CommittedParticipantReceipt>>,
}

impl ParticipantControl {
    /// Refuse the next `count` commit notifications.
    pub fn fail_next_commit_notifications(&self, count: u64) {
        self.failing_notifications.set(count);
    }

    pub fn view_calls(&self) -> u64 {
        self.view_calls.get()
    }

    pub fn commit_notifications(&self) -> u64 {
        self.commit_notifications.get()
    }

    pub fn last_notified_generation(&self) -> Option<CheckpointGeneration> {
        self.last_receipt
            .borrow()
            .as_ref()
            .map(|receipt| receipt.generation().clone())
    }
}

/// Participant that counts views and notifications and can refuse callbacks.
pub struct NotifyingParticipant {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    control: Rc<ParticipantControl>,
    initialization: ParticipantInitialization,
}

impl NotifyingParticipant {
    pub fn new(run: StreamRunIdentity, participant_id: &str) -> (Self, Rc<ParticipantControl>) {
        let control = Rc::new(ParticipantControl::default());
        (
            Self {
                run,
                participant_id: CheckpointParticipantId::new(participant_id),
                control: Rc::clone(&control),
                initialization: ParticipantInitialization::default(),
            },
            control,
        )
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for NotifyingParticipant {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if barrier.run != self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        self.control
            .view_calls
            .set(self.control.view_calls.get() + 1);
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            "test.coordinator",
            1,
            barrier.cut.clone(),
            1,
            checkpoint_payload(Bytes::from(barrier.epoch.get().to_le_bytes().to_vec())).await,
        )
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let remaining = self.control.failing_notifications.get();
        if remaining > 0 {
            self.control.failing_notifications.set(remaining - 1);
            return Err(CheckpointError::PostCommitNotification {
                participant: self.participant_id.clone(),
            });
        }
        self.control
            .commit_notifications
            .set(self.control.commit_notifications.get() + 1);
        *self.control.last_receipt.borrow_mut() = Some(receipt.clone());
        Ok(())
    }
}

struct SilentEndpoint;

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for SilentEndpoint {
    async fn report(
        &self,
        _issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

/// Shared observable state for the ledger participant.
#[derive(Default)]
pub struct ReporterControl {
    bound_roots: RefCell<Vec<ContentDigest>>,
    acknowledged_roots: RefCell<Vec<ContentDigest>>,
    is_bind_refused: Cell<bool>,
}

impl ReporterControl {
    /// Refuse every subsequent pre-CAS result-epoch binding.
    pub fn refuse_bind(&self) {
        self.is_bind_refused.set(true);
    }

    pub fn bound_roots(&self) -> Vec<ContentDigest> {
        self.bound_roots.borrow().clone()
    }

    /// Roots the ledger acknowledged post-CAS, in delivery order.
    pub fn acknowledged_roots(&self) -> Vec<ContentDigest> {
        self.acknowledged_roots.borrow().clone()
    }
}

/// Ledger participant that binds a staged index root and checks it post-CAS.
pub struct FakeIssueReporter {
    run: StreamRunIdentity,
    control: Rc<ReporterControl>,
    initialization: ParticipantInitialization,
}

impl FakeIssueReporter {
    pub fn new(run: StreamRunIdentity) -> (Self, Rc<ReporterControl>) {
        let control = Rc::new(ReporterControl::default());
        (
            Self {
                run,
                control: Rc::clone(&control),
                initialization: ParticipantInitialization::default(),
            },
            control,
        )
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for FakeIssueReporter {
    fn participant_id(&self) -> CheckpointParticipantId {
        CheckpointParticipantId::new(LEDGER_ID)
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if barrier.run != self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        PreparedParticipantState::new(
            self.run,
            CheckpointParticipantId::new(LEDGER_ID),
            "test.ledger",
            1,
            barrier.cut.clone(),
            1,
            checkpoint_payload(Bytes::from_static(b"ledger-state")).await,
        )
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        // Retire receipts only against the exact root bound before CAS.
        if !self
            .control
            .bound_roots
            .borrow()
            .contains(receipt.result_index_root())
        {
            return Err(CheckpointError::ObjectVerification);
        }
        self.control
            .acknowledged_roots
            .borrow_mut()
            .push(*receipt.result_index_root());
        Ok(())
    }
}

impl StreamingIssueReporter for FakeIssueReporter {
    fn handle(&self) -> StreamingIssueReporterHandle {
        StreamingIssueReporterHandle::new(SilentEndpoint)
    }

    fn bind_prepared_result_epoch(
        &mut self,
        prepared: &PreparedResultEpoch,
    ) -> Result<(), aiperf_runtime::streaming::reliability::StreamingReliabilityError> {
        if self.control.is_bind_refused.get() {
            return Err(
                aiperf_runtime::streaming::reliability::StreamingReliabilityError::ReliabilityStateUnavailable,
            );
        }
        self.control
            .bound_roots
            .borrow_mut()
            .push(*prepared.index_root());
        Ok(())
    }
}

/// Injectable faults and call counters shared with a decorated backend.
#[derive(Default)]
pub struct BackendControl {
    begin_generation_calls: Cell<u64>,
    stage_results_calls: Cell<u64>,
    commit_calls: Cell<u64>,
    is_begin_refused: Cell<bool>,
    is_stage_results_refused: Cell<bool>,
    is_commit_refused: Cell<bool>,
}

impl BackendControl {
    pub fn fail_next_begin_generation(&self) {
        self.is_begin_refused.set(true);
    }

    pub fn fail_next_stage_results(&self) {
        self.is_stage_results_refused.set(true);
    }

    pub fn fail_next_commit(&self) {
        self.is_commit_refused.set(true);
    }

    pub fn begin_generation_calls(&self) -> u64 {
        self.begin_generation_calls.get()
    }

    pub fn stage_results_calls(&self) -> u64 {
        self.stage_results_calls.get()
    }

    pub fn commit_calls(&self) -> u64 {
        self.commit_calls.get()
    }
}

/// Memory backend decorator that injects pre-CAS faults.
pub struct FaultingCheckpointBackend {
    inner: MemoryCheckpointBackend,
    control: Rc<BackendControl>,
}

#[async_trait(?Send)]
impl StreamingCheckpointBackend for FaultingCheckpointBackend {
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<LeasedCheckpointGeneration>, CheckpointError> {
        self.inner.open_latest(run, expected).await
    }

    async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CurrentV4CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError> {
        self.control
            .begin_generation_calls
            .set(self.control.begin_generation_calls.get() + 1);
        if self.control.is_begin_refused.replace(false) {
            return Err(CheckpointError::Storage {
                message: "injected begin_generation refusal".to_owned(),
            });
        }
        let inner = self
            .inner
            .begin_generation(run, expected, expectations)
            .await?;
        Ok(Box::new(FaultingTransaction {
            inner: Box::new(inner),
            control: Rc::clone(&self.control),
        }))
    }
}

struct FaultingTransaction {
    inner: Box<dyn StreamingGenerationTransaction>,
    control: Rc<BackendControl>,
}

#[async_trait(?Send)]
impl StreamingGenerationTransaction for FaultingTransaction {
    async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        self.inner.stage_participant(state).await
    }

    async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<
            aiperf_runtime::streaming::reliability::PreparedIssueReceiptResultPartition,
        >,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        self.control
            .stage_results_calls
            .set(self.control.stage_results_calls.get() + 1);
        if self.control.is_stage_results_refused.replace(false) {
            // Refuse without touching either move-only input, exactly as the
            // trait contract requires of every implementor.
            return Err(CheckpointError::Storage {
                message: "injected stage_results refusal".to_owned(),
            });
        }
        self.inner.stage_results(partitions, issue_receipts).await
    }

    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        self.control
            .commit_calls
            .set(self.control.commit_calls.get() + 1);
        if self.control.is_commit_refused.replace(false) {
            return Err(CheckpointError::Storage {
                message: "injected commit refusal".to_owned(),
            });
        }
        self.inner.commit(metadata).await
    }
}

/// One coordinator plus every handle a test needs to observe it.
pub struct CoordinatorFixture {
    pub coordinator: StreamingCheckpointCoordinator,
    pub run: StreamRunIdentity,
    pub backend: MemoryCheckpointBackend,
    pub backend_control: Rc<BackendControl>,
    pub participant: Rc<ParticipantControl>,
    pub reporter: Rc<ReporterControl>,
}

impl CoordinatorFixture {
    /// Return the backend's authoritative head identity for this run.
    pub async fn latest_generation(&self) -> Option<CheckpointGeneration> {
        self.backend
            .open_latest(&self.run, &expectations(self.run))
            .await
            .expect("open memory head")
            .map(|opened| opened.generation().clone())
    }

    pub fn immutable_object_inventory(&self) -> ImmutableObjectInventory {
        self.backend.immutable_object_inventory(&self.run)
    }
}

pub fn coordinator_fixture() -> CoordinatorFixture {
    coordinator_fixture_for_run(run_id(1))
}

pub fn coordinator_fixture_for_run(run: StreamRunIdentity) -> CoordinatorFixture {
    let backend = MemoryCheckpointBackend::new(backend_limits()).expect("valid memory backend");
    let backend_control = Rc::new(BackendControl::default());
    let decorated = FaultingCheckpointBackend {
        inner: backend.clone(),
        control: Rc::clone(&backend_control),
    };
    let (participant, participant_control) = NotifyingParticipant::new(run, PARTICIPANT_ID);
    let (reporter, reporter_control) = FakeIssueReporter::new(run);
    let coordinator = StreamingCheckpointCoordinator::new(
        run,
        Box::new(decorated),
        expectations(run),
        vec![Box::new(participant)],
        Box::new(reporter),
        None,
    )
    .expect("valid coordinator");
    CoordinatorFixture {
        coordinator,
        run,
        backend,
        backend_control,
        participant: participant_control,
        reporter: reporter_control,
    }
}
