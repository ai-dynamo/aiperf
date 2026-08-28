// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic fixture for the bounded streaming pipeline.
//!
//! Every stage is a scripted fake so the pipeline's own arm discipline is what
//! the tests observe. The action submitter is the fixture's stand-in for the
//! terminal-record permit, which is the innermost reservation in the chain: it
//! parks until the test explicitly releases it, exactly as a saturated terminal
//! lane would.

#![allow(dead_code)]

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeSet, VecDeque},
    rc::Rc,
};

use aiperf_runtime::streaming::{
    action::{
        ActionAdmissionReceipt, ActionCancelReceipt, ActionDrainReceipt, ActionEventIdentity,
        ActionExecutionError, ActionExecutionEvent,
        ActionFailureCode, ActionHandleId, ActionTerminalDisposition, ActionTerminalReceipt,
        DatasetActionSchema, OrderedDatasetAction, PreparedStreamingActionBinding,
        StreamingActionBindingSet, StreamingActionDriver, StreamingActionDriverControl,
        StreamingActionDriverControlOps, StreamingActionHost, StreamingActionSubmitter,
        SubmittedAction, action_execution_control, canonical_action_schema,
    },
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        CheckpointBarrier, CheckpointError, CheckpointParticipantId, CheckpointParticipantPlan,
        CommittedParticipantReceipt, CommittedParticipantState, PreparedParticipantState,
        StreamRunIdentity, StreamingCheckpointParticipant,
    },
    checkpoint_backend::CheckpointGenerationExpectations,
    checkpoint_coordinator::StreamingCheckpointCoordinator,
    checkpoints::memory::{MemoryCheckpointBackend, MemoryCheckpointLimits},
    failure::{StreamFormatError, StreamSourceError},
    format::{
        DecodeBatchBudget, DecodeReceipt, DecodeStep, DecoderCheckpoint, DecoderResumeState,
        FormatEvent, FormatEventSink, FormatSealReceipt, SessionWatermark, StreamingDatasetFormat,
        StreamingPartitionDecoder,
    },
    identity::{
        ActionAttemptId, ContentDigest, GlobalSequence, ImmutableObjectIdentity,
        LogicalReplayRunId, SessionCausalFrontier, SessionOwnershipEpoch, StableActionId,
        StableOrderKey, StableSessionKey,
    },
    pipeline::{
        PreparedStreamingComponents, StreamingPhaseContext, StreamingPipeline,
        StreamingPipelineControl, StreamingPipelineError, StreamingPipelineLimits,
    },
    placement::local_placement_binding,
    reliability::{StreamingIssueReporter, StreamingIssueReporterHandle},
    session::{
        DatasetActionSink, SessionCoordinatorError, SessionSealReceipt, StreamingSessionCoordinator,
    },
    source::{
        AcquiredPartition, AcquisitionBudget, SourceEvent, SourceFrontier, SourceSeal,
        SourceSnapshotReceipt, StreamingDatasetSource, StreamingSourceControl,
        streaming_stop_channel,
    },
    unit::{
        ActionContentLeaseSet, DatasetActionKind, DatasetActionV1, EventTimeUtc,
        ExecutableDatasetAction, SessionFragmentLease, SessionRequestAction, SourcePosition,
        StreamingSessionFragment, UnitProvenance,
    },
};
use async_trait::async_trait;
use bytes::Bytes;

/// Run a `!Send` future on a current-thread runtime and `LocalSet`.
pub fn local<T>(future: impl Future<Output = T>) -> T {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current-thread runtime");
    tokio::task::LocalSet::new().block_on(&runtime, future)
}

pub fn run_id(value: u8) -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([value; 32]))
}

pub fn budget(max_items: usize, max_bytes: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items,
        max_bytes,
    })
    .expect("valid limits")
}

fn backend_limits() -> MemoryCheckpointLimits {
    let limits = BudgetLimits {
        max_items: 512,
        max_bytes: 4_194_304,
    };
    MemoryCheckpointLimits {
        transactions: limits,
        prepared_indexes: limits,
        storage: limits,
        result_summaries: limits,
        reads: limits,
    }
}

async fn empty_payload() -> aiperf_runtime::streaming::checkpoint::BudgetedCheckpointBytes {
    let budget = budget(1, 1);
    let lease = budget.acquire(1, 0).await.expect("payload charge");
    aiperf_runtime::streaming::checkpoint::BudgetedCheckpointBytes::new(Bytes::new(), lease)
        .expect("exact payload charge")
}

/// Everything the tests observe about one pipeline run.
#[derive(Default)]
pub struct Probes {
    /// Source `next_event` calls, counted by the source itself.
    pub source_pulls: Cell<u64>,
    /// Source pulls observed after the submitter began parking.
    pub source_pulls_after_saturation: Cell<u64>,
    /// Decoder pulls issued by the pipeline.
    pub decode_pulls: Cell<u64>,
    /// Session `advance_watermark` calls.
    pub watermarks: Cell<u64>,
    /// Session `ingest` calls.
    pub ingests: Cell<u64>,
    /// Session `seal` calls.
    pub seals: Cell<u64>,
    /// Every `observe_execution` call, tagged by the event it carried.
    pub observed: RefCell<Vec<&'static str>>,
    /// Actions the submitter accepted, in submission order.
    pub submitted: RefCell<Vec<StableActionId>>,
    /// Whether the submitter is currently parked, standing in for a full lane.
    pub is_lane_blocked: Cell<bool>,
    /// Whether the submitter is currently parked inside `submit`.
    pub is_submit_parked: Cell<bool>,
    /// Events the driver has yet to deliver.
    pub events: RefCell<VecDeque<ActionExecutionEvent>>,
    /// Whether both drivers were joined during shutdown.
    pub is_action_drained: Cell<bool>,
    /// Whether the action control fenced issue during shutdown.
    pub is_issue_stopped: Cell<bool>,
    wake: tokio::sync::Notify,
}

impl Probes {
    /// Park every subsequent submission, standing in for a saturated lane.
    pub fn block_terminal_lane(&self) {
        self.is_lane_blocked.set(true);
    }

    /// Release the lane and wake a parked submission.
    pub fn release_terminal_lane(&self) {
        self.is_lane_blocked.set(false);
        self.wake.notify_waiters();
        self.wake.notify_one();
    }

    /// Queue one terminal receipt for the action at `index` in submission order.
    pub fn emit_terminal(&self, index: usize, disposition: ActionTerminalDisposition) {
        let action_id = self.submitted.borrow()[index];
        self.events
            .borrow_mut()
            .push_back(ActionExecutionEvent::Terminal(ActionTerminalReceipt {
                event: ActionEventIdentity {
                    action_id,
                    attempt_id: ActionAttemptId::from_bytes([9; 32]),
                    ownership_epoch: SessionOwnershipEpoch::new(0),
                    event_ordinal: 2,
                },
                disposition,
            }));
        self.wake.notify_waiters();
        self.wake.notify_one();
    }

    /// Queue one admission receipt for the action at `index`.
    pub fn emit_admitted(&self, index: usize) {
        let action_id = self.submitted.borrow()[index];
        self.events
            .borrow_mut()
            .push_back(ActionExecutionEvent::Admitted(ActionAdmissionReceipt {
                event: ActionEventIdentity {
                    action_id,
                    attempt_id: ActionAttemptId::from_bytes([9; 32]),
                    ownership_epoch: SessionOwnershipEpoch::new(0),
                    event_ordinal: 1,
                },
            }));
        self.wake.notify_waiters();
        self.wake.notify_one();
    }

    /// Terminate every action submitted so far, in order.
    pub fn emit_all_terminal(&self) {
        let count = self.submitted.borrow().len();
        for index in 0..count {
            self.emit_terminal(index, ActionTerminalDisposition::Completed);
        }
    }
}

/// Scripted immutable source that counts its own pulls.
struct FakeSource {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    events: RefCell<VecDeque<SourceEvent>>,
    snapshot: SourceSnapshotReceipt,
    probes: Rc<Probes>,
}

#[async_trait(?Send)]
impl StreamingDatasetSource for FakeSource {
    fn snapshot(&self) -> &SourceSnapshotReceipt {
        &self.snapshot
    }

    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError> {
        self.probes
            .source_pulls
            .set(self.probes.source_pulls.get() + 1);
        if self.probes.is_submit_parked.get() {
            self.probes
                .source_pulls_after_saturation
                .set(self.probes.source_pulls_after_saturation.get() + 1);
        }
        if let Some(event) = self.events.borrow_mut().pop_front() {
            return Ok(event);
        }
        // A quiet follow source: never resolves until the run stops.
        std::future::pending().await
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for FakeSource {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        stub_view(self.run, self.participant_id.clone(), barrier).await
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

/// Format that turns every source frontier into exactly one session watermark.
struct FakeFormat {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    next_event_time: Cell<i64>,
}

#[async_trait(?Send)]
impl StreamingDatasetFormat for FakeFormat {
    async fn begin_partition(
        &mut self,
        _partition: AcquiredPartition,
        _resume: Option<DecoderCheckpoint>,
    ) -> Result<Box<dyn StreamingPartitionDecoder>, StreamFormatError> {
        Ok(Box::new(EmptyDecoder))
    }

    async fn advance_source_frontier(
        &mut self,
        _frontier: SourceFrontier,
        output: &mut dyn FormatEventSink,
    ) -> Result<(), StreamFormatError> {
        let through = self.next_event_time.get();
        self.next_event_time.set(through + 1);
        output
            .send(FormatEvent::SessionFrontier(SessionWatermark {
                through: EventTimeUtc::new(through).expect("non-negative event time"),
                digest: ContentDigest::from_bytes([7; 32]),
            }))
            .await
    }

    async fn seal(
        &mut self,
        _seal: SourceSeal,
        _output: &mut dyn FormatEventSink,
    ) -> Result<FormatSealReceipt, StreamFormatError> {
        Ok(FormatSealReceipt {
            digest: ContentDigest::from_bytes([8; 32]),
            partition_count: 0,
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for FakeFormat {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        stub_view(self.run, self.participant_id.clone(), barrier).await
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

/// Decoder that exhausts its partition immediately.
struct EmptyDecoder;

#[async_trait(?Send)]
impl StreamingPartitionDecoder for EmptyDecoder {
    async fn next_batch(
        &mut self,
        _budget: DecodeBatchBudget,
    ) -> Result<DecodeStep, StreamFormatError> {
        Ok(DecodeStep::End(DecodeReceipt {
            partition: ImmutableObjectIdentity::from_bytes([4; 32]),
            fragment_count: 0,
            final_state: resume_state(),
        }))
    }

    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError> {
        Ok(resume_state())
    }
}

fn resume_state() -> DecoderResumeState {
    let budget = budget(1, 1);
    let lease = budget.try_acquire(1, 0).expect("cursor charge");
    DecoderResumeState::new(Bytes::new(), lease).expect("exact cursor charge")
}

/// Session program that emits one action per watermark and records call sites.
struct FakeSession {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    probes: Rc<Probes>,
    content_budget: StreamingResourceBudget,
    actions_per_watermark: usize,
    next_action: Cell<u8>,
    sequence: Cell<u64>,
}

impl FakeSession {
    async fn emit_actions(
        &self,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        for _ in 0..self.actions_per_watermark {
            let tag = self.next_action.get();
            self.next_action.set(tag.wrapping_add(1));
            let action =
                request_action(&self.content_budget, StableActionId::from_bytes([tag; 32])).await;
            output.send_action(action).await?;
        }
        let sequence = self.sequence.get() + 1;
        self.sequence.set(sequence);
        output
            .advance_causal_frontier(SessionCausalFrontier {
                through_sequence: GlobalSequence::new(sequence),
                event_time: None,
                digest: ContentDigest::from_bytes([5; 32]),
            })
            .await
    }
}

#[async_trait(?Send)]
impl StreamingSessionCoordinator for FakeSession {
    async fn ingest(
        &mut self,
        _fragment: StreamingSessionFragment,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.probes.ingests.set(self.probes.ingests.get() + 1);
        self.emit_actions(output).await
    }

    async fn advance_watermark(
        &mut self,
        _watermark: SessionWatermark,
        output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        self.probes.watermarks.set(self.probes.watermarks.get() + 1);
        self.emit_actions(output).await
    }

    async fn observe_execution(
        &mut self,
        event: ActionExecutionEvent,
        _output: &mut dyn DatasetActionSink,
    ) -> Result<(), SessionCoordinatorError> {
        let tag = match event {
            ActionExecutionEvent::Admitted(_) => "admitted",
            ActionExecutionEvent::FirstToken(_) => "first_token",
            ActionExecutionEvent::SessionUpdate(_) => "session_update",
            ActionExecutionEvent::Terminal(_) => "terminal",
        };
        self.probes.observed.borrow_mut().push(tag);
        Ok(())
    }

    async fn seal(
        &mut self,
        _seal: SourceSeal,
        _output: &mut dyn DatasetActionSink,
    ) -> Result<SessionSealReceipt, SessionCoordinatorError> {
        self.probes.seals.set(self.probes.seals.get() + 1);
        Ok(SessionSealReceipt {
            digest: ContentDigest::from_bytes([6; 32]),
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(self.sequence.get()),
                event_time: None,
                digest: ContentDigest::from_bytes([5; 32]),
            },
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for FakeSession {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        stub_view(self.run, self.participant_id.clone(), barrier).await
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

/// Fully charged request action carrying a distinguishable identity.
pub async fn request_action(
    content_budget: &StreamingResourceBudget,
    action_id: StableActionId,
) -> ExecutableDatasetAction {
    let request = vec![0_u8; 8];
    let charge = request.capacity() + 64;
    let fragment = SessionFragmentLease::try_from(
        content_budget
            .acquire(1, charge)
            .await
            .expect("content charge"),
    )
    .expect("one-item fragment");
    ExecutableDatasetAction::new(
        action_id,
        StableSessionKey::from_bytes([2; 32]),
        Default::default(),
        None,
        StableOrderKey::from_bytes([3; 32]),
        SourcePosition::new(4),
        UnitProvenance {
            source_partition: ImmutableObjectIdentity::from_bytes([5; 32]),
            source_position: SourcePosition::new(4),
            format_semantic_digest: ContentDigest::from_bytes([6; 32]),
        },
        DatasetActionV1::Request(SessionRequestAction { request }),
        ActionContentLeaseSet::from_retained(fragment.into_retained()),
    )
    .expect("fully charged action")
}

/// Submitter that parks while the fixture's stand-in terminal lane is full.
///
/// Parking here rather than in a separate stage is deliberate: the terminal
/// permit is the innermost reservation, and it is reached transitively through
/// the action binding's own `submit`.
struct FakeSubmitter {
    schema: DatasetActionSchema,
    probes: Rc<Probes>,
}

#[async_trait(?Send)]
impl StreamingActionSubmitter for FakeSubmitter {
    fn accepted_schema(&self) -> DatasetActionSchema {
        self.schema.clone()
    }

    async fn submit(
        &mut self,
        action: OrderedDatasetAction,
    ) -> Result<SubmittedAction, ActionExecutionError> {
        if self.probes.is_issue_stopped.get() {
            return Err(ActionExecutionError::action(ActionFailureCode::Cancelled));
        }
        while self.probes.is_lane_blocked.get() {
            self.probes.is_submit_parked.set(true);
            self.probes.wake.notified().await;
        }
        self.probes.is_submit_parked.set(false);
        let action_id = action.action().action_id();
        self.probes.submitted.borrow_mut().push(action_id);
        let (control, _receiver) = action_execution_control();
        Ok(SubmittedAction {
            handle_id: ActionHandleId::new(self.probes.submitted.borrow().len() as u64),
            control,
        })
    }
}

/// Driver delivering the events the fixture queued.
struct FakeDriver {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    probes: Rc<Probes>,
}

#[async_trait(?Send)]
impl StreamingActionDriver for FakeDriver {
    async fn next_event(&mut self) -> Result<ActionExecutionEvent, ActionExecutionError> {
        loop {
            if let Some(event) = self.probes.events.borrow_mut().pop_front() {
                return Ok(event);
            }
            self.probes.wake.notified().await;
        }
    }

    async fn drain(&mut self) -> Result<ActionDrainReceipt, ActionExecutionError> {
        self.probes.is_action_drained.set(true);
        Ok(ActionDrainReceipt {
            submitted: self.probes.submitted.borrow().len() as u64,
            terminal: self.probes.submitted.borrow().len() as u64,
            digest: ContentDigest::from_bytes([1; 32]),
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for FakeDriver {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        stub_view(self.run, self.participant_id.clone(), barrier).await
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

struct FakeActionControl {
    probes: Rc<Probes>,
}

#[async_trait(?Send)]
impl StreamingActionDriverControlOps for FakeActionControl {
    fn stop_issuing(&self) {
        self.probes.is_issue_stopped.set(true);
        self.probes.wake.notify_waiters();
    }

    fn cancel_pending(&self) {
        self.probes.wake.notify_waiters();
    }

    async fn cancel_inflight(&self) -> Result<ActionCancelReceipt, ActionExecutionError> {
        Ok(ActionCancelReceipt {
            cancelled: 0,
            digest: ContentDigest::from_bytes([2; 32]),
        })
    }
}

/// Checkpoint participant that retains nothing.
pub struct StubParticipant {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
}

impl StubParticipant {
    pub fn new(run: StreamRunIdentity, id: &str) -> Self {
        Self {
            participant_id: CheckpointParticipantId::new(id),
            run,
        }
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for StubParticipant {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        stub_view(self.run, self.participant_id.clone(), barrier).await
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

async fn stub_view(
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    barrier: &CheckpointBarrier,
) -> Result<PreparedParticipantState, CheckpointError> {
    PreparedParticipantState::new(
        run,
        participant_id,
        "test.stub",
        1,
        barrier.cut.clone(),
        0,
        empty_payload().await,
    )
}

/// Reliability ledger that accepts every barrier without minting a fact.
pub struct StubReporter {
    run: StreamRunIdentity,
}

struct SilentEndpoint;

#[async_trait(?Send)]
impl aiperf_runtime::streaming::reliability::StreamingIssueReporterEndpoint for SilentEndpoint {
    async fn report(
        &self,
        _issue: aiperf_runtime::streaming::reliability::OrdinaryStreamingIssue,
    ) -> Result<
        aiperf_runtime::streaming::reliability::StreamingIssueReportStatus,
        aiperf_runtime::streaming::reliability::StreamingIssueReportError,
    > {
        Ok(aiperf_runtime::streaming::reliability::StreamingIssueReportStatus::Accepted)
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for StubReporter {
    fn participant_id(&self) -> CheckpointParticipantId {
        CheckpointParticipantId::new("streaming_issue_ledger")
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        stub_view(self.run, self.participant_id(), barrier).await
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

impl StreamingIssueReporter for StubReporter {
    fn handle(&self) -> StreamingIssueReporterHandle {
        StreamingIssueReporterHandle::new(SilentEndpoint)
    }

    fn bind_prepared_result_epoch(
        &mut self,
        _prepared: &aiperf_runtime::streaming::results::PreparedResultEpoch,
    ) -> Result<(), aiperf_runtime::streaming::reliability::StreamingReliabilityError> {
        Ok(())
    }
}

/// How a spec deliberately corrupts the frozen participant plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OmittedOwner {
    /// Give the blocking-work owner the source's identity.
    BlockingOwner,
}

/// Authored shape of one fixture pipeline.
pub struct FixtureSpec {
    /// Source events delivered in order before the source goes quiet.
    pub source_events: Vec<SourceEvent>,
    /// Actions the session emits per watermark.
    pub actions_per_watermark: usize,
    /// Maximum prepared placements.
    pub max_prepared: usize,
    /// Maximum actions buffered between sink drains.
    pub max_pending_actions: usize,
    /// Nanoseconds between checkpoint barriers.
    pub checkpoint_interval_ns: i64,
    /// A required participant to omit, for plan-validation tests.
    pub omit: Option<OmittedOwner>,
}

impl Default for FixtureSpec {
    fn default() -> Self {
        Self {
            source_events: Vec::new(),
            actions_per_watermark: 1,
            max_prepared: 64,
            max_pending_actions: 16,
            // Long enough that no ordinary test trips a barrier by accident.
            checkpoint_interval_ns: 1_000_000_000_000,
            omit: None,
        }
    }
}

/// One assembled pipeline plus every handle a test needs.
pub struct PipelineFixture {
    pub pipeline: StreamingPipeline,
    pub phase: StreamingPhaseContext,
    pub probes: Rc<Probes>,
    pub control: StreamingPipelineControl,
    pub stop: StreamingSourceControl,
}

/// Emit `count` source frontiers followed by nothing.
pub fn frontiers(count: u64) -> Vec<SourceEvent> {
    (1..=count)
        .map(|position| {
            SourceEvent::Frontier(SourceFrontier {
                through: SourcePosition::new(position),
            })
        })
        .collect()
}

/// Emit `count` source frontiers followed by an explicit seal.
pub fn frontiers_then_seal(count: u64) -> Vec<SourceEvent> {
    let mut events = frontiers(count);
    events.push(SourceEvent::Seal(SourceSeal {
        final_position: Some(SourcePosition::new(count)),
        digest: ContentDigest::from_bytes([0xAB; 32]),
    }));
    events
}

/// Build one fixture, or return the plan error the spec deliberately provokes.
pub fn build(spec: FixtureSpec) -> Result<PipelineFixture, StreamingPipelineError> {
    let run = run_id(1);
    let probes = Rc::new(Probes::default());
    let (stop_control, stop_receiver) = streaming_stop_channel();

    let source = FakeSource {
        participant_id: CheckpointParticipantId::new("source"),
        run,
        events: RefCell::new(spec.source_events.into_iter().collect()),
        snapshot: SourceSnapshotReceipt {
            digest: ContentDigest::from_bytes([0x11; 32]),
        },
        probes: Rc::clone(&probes),
    };
    let format = FakeFormat {
        participant_id: CheckpointParticipantId::new("format"),
        run,
        next_event_time: Cell::new(0),
    };
    let session = FakeSession {
        participant_id: CheckpointParticipantId::new("session"),
        run,
        probes: Rc::clone(&probes),
        content_budget: budget(4096, 4_194_304),
        actions_per_watermark: spec.actions_per_watermark,
        next_action: Cell::new(0),
        sequence: Cell::new(0),
    };

    let schema = canonical_action_schema(DatasetActionKind::Request);
    let control = StreamingActionDriverControl::new(FakeActionControl {
        probes: Rc::clone(&probes),
    });
    let mut bindings = StreamingActionBindingSet::new();
    bindings
        .insert(PreparedStreamingActionBinding {
            submitter: Box::new(FakeSubmitter {
                schema: schema.clone(),
                probes: Rc::clone(&probes),
            }),
            driver: Box::new(FakeDriver {
                participant_id: CheckpointParticipantId::new("action_driver"),
                run,
                probes: Rc::clone(&probes),
            }),
            control,
        })
        .expect("one binding per schema");
    let emitted: BTreeSet<_> = [schema].into_iter().collect();
    let (action, _controls) =
        StreamingActionHost::new(run, &emitted, bindings, budget(4096, 4_194_304))
            .expect("valid action host");

    let placement = local_placement_binding(
        run,
        CheckpointParticipantId::new("placement_policy"),
        CheckpointParticipantId::new("placement_driver"),
        spec.max_prepared,
        budget(64, 65_536),
        budget(64, 65_536),
    );

    let blocking_owner: Box<dyn StreamingCheckpointParticipant> = match spec.omit {
        // Two owners claiming one identity is the observable form of an invalid
        // frozen plan, and it must be refused before the first source poll.
        Some(OmittedOwner::BlockingOwner) => Box::new(StubParticipant::new(run, "source")),
        None => Box::new(StubParticipant::new(run, "blocking")),
    };

    let components = PreparedStreamingComponents {
        source: Box::new(source),
        format: Box::new(format),
        session: Box::new(session),
        action,
        placement,
        acquisition: AcquisitionBudget::new(budget(64, 1_048_576), budget(64, 1_048_576)),
        event_time_order_policy: Box::new(StubParticipant::new(run, "event_time")),
        action_driver_bindings: vec![Box::new(StubParticipant::new(run, "action_driver"))],
        active_execution_set: Box::new(StubParticipant::new(run, "active_execution")),
        blocking_owner,
        result_epoch: Box::new(StubParticipant::new(run, "result_epoch")),
    };

    let limits = StreamingPipelineLimits {
        decode_batch: DecodeBatchBudget {
            max_fragments: 8,
            max_bytes: 4096,
        },
        max_pending_actions: spec.max_pending_actions,
        checkpoint_interval_ns: spec.checkpoint_interval_ns,
    };

    let (pipeline, control, participants) =
        StreamingPipeline::prepare(run, limits, components, stop_control.clone())?;

    // The reliability ledger is a coordinator participant the pipeline's own
    // owner inventory has no slot for, so the assembler adds it to the frozen
    // plan the coordinator validates against.
    let ledger = StubReporter { run };
    let plan = CheckpointParticipantPlan::new(
        pipeline
            .participant_plan()
            .ids()
            .iter()
            .cloned()
            .chain([ledger.participant_id()]),
    )
    .expect("valid coordinator plan");
    let expectations = CheckpointGenerationExpectations {
        run,
        participant_plan: plan,
        // The pipeline binds every barrier to its own frozen plan digest.
        execution_plan_digest: pipeline.participant_plan().digest(),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
    };
    let backend = MemoryCheckpointBackend::new(backend_limits()).expect("valid memory backend");
    let checkpoint = StreamingCheckpointCoordinator::new(
        run,
        Box::new(backend),
        expectations,
        participants,
        Box::new(ledger),
        None,
    )
    .expect("valid coordinator");

    let phase = StreamingPhaseContext {
        clock: Rc::new(aiperf_runtime::clock::sim_clock::SimClock::new()),
        checkpoint,
        stop: stop_receiver,
    };

    Ok(PipelineFixture {
        pipeline,
        phase,
        probes,
        control,
        stop: stop_control,
    })
}
