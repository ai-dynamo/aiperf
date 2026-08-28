// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded fused composition of every prepared streaming component.
//!
//! One current-thread future owns source acquisition, decoding, session
//! coordination, action ordering, placement, submission, settlement,
//! checkpointing, and shutdown. A new unit is pulled only when every downstream
//! stage owns the permits that unit will consume, so an unbounded input produces
//! a bounded resident footprint. The permit chain is strictly nested — the
//! terminal-record permit owned inside the action submitter is innermost — which
//! is why a saturated downstream pins every upstream stage instead of merely
//! slowing it:
//!
//! ```text
//! acquisition lease ⊃ session state charge ⊃ placement route reservation
//!                                          ⊃ placement slab entry
//!                                          ⊃ active-execution lease
//!                                          ⊃ terminal-lane permit (innermost)
//! ```
//!
//! The pipeline assigns no [`GlobalSequence`]: the action host numbers an action
//! as part of submitting it, and the pipeline binds that number back onto the
//! prepared placement afterwards. The pipeline mints no reliability facts
//! either; it submits `IssueSequenceUpdate`s and applies the returned
//! `StreamingIssueOutcome`, and only the reliability module's private classifier
//! can answer `FailRun`.
//!
//! Every stage lives behind an [`Rc`]`<`[`RefCell`]`<_>>` so the checkpoint
//! coordinator, which owns its participants by value, can hold a
//! [`CheckpointProxy`] over the same cell the fused loop drives. The whole
//! streaming plane is `!Send` and current-thread, so this is a checked borrow
//! discipline rather than a lock.

use std::{
    cell::{Cell, RefCell, RefMut},
    rc::Rc,
};

use async_trait::async_trait;
use futures::{FutureExt as _, future::LocalBoxFuture, select_biased};
use tracing::debug;

use crate::clock::Clock;

use super::{
    action::{ActionExecutionError, ActionExecutionEvent, StreamingActionHost},
    checkpoint::{
        CheckpointBarrier, CheckpointEpoch, CheckpointError, CheckpointParticipantId,
        CheckpointParticipantOwners, CheckpointParticipantPlan, CheckpointParticipantPlanError,
        CheckpointTerminalReason, CommittedCheckpointGeneration, CommittedParticipantReceipt,
        CommittedParticipantState, PreparedParticipantState, StreamRunIdentity,
        StreamingCheckpointParticipant,
    },
    checkpoint_coordinator::{
        CheckpointBarrierFinality, PreparedCheckpointResultInput, StreamingCheckpointCoordinator,
    },
    failure::{StreamFormatError, StreamSourceError},
    format::{
        DecodeBatchBudget, DecodeStep, FormatEvent, FormatEventSink, StreamingDatasetFormat,
        StreamingPartitionDecoder,
    },
    identity::{ContentDigest, GlobalSequence, SessionCausalFrontier, StableSessionKey},
    placement::{
        PlacementError, PlacementEvent, PlacementHandle, PlacementHandleId,
        PreparedStreamingPlacementBinding, StreamingPlacementAdmission, StreamingPlacementControl,
        StreamingPlacementDriver, StreamingPlacementPolicy, StreamingPlacementSubmitter,
    },
    reliability::{StreamingIssueDisposition, StreamingIssueOutcome},
    session::{DatasetActionSink, SessionCoordinatorError, StreamingSessionCoordinator},
    source::{
        AcquisitionBudget, PartitionAccessRequest, SourceEvent, StreamingDatasetSource,
        StreamingStopReceiver,
    },
    unit::{ExecutableDatasetAction, StateBudgetFailureCode},
};

/// Shared single-threaded cell for one stateful streaming owner.
///
/// The pipeline drives the stage through this handle; the checkpoint coordinator
/// receives a [`CheckpointProxy`] over the same cell. Both are single-threaded,
/// and the pipeline only calls the coordinator at a quiescent point in its fused
/// loop where no stage borrow is live, so the checked borrow below is an
/// assertion of that discipline rather than a lock.
pub type StageCell<T> = Rc<RefCell<T>>;

/// Erased access to a stage that is also a checkpoint participant.
trait ParticipantCell {
    /// Borrow the stage as a participant, or report that it is in use.
    fn try_borrow_participant(&self) -> Option<RefMut<'_, dyn StreamingCheckpointParticipant>>;
}

macro_rules! participant_cell {
    ($stage:ty) => {
        impl ParticipantCell for RefCell<Box<$stage>> {
            fn try_borrow_participant(
                &self,
            ) -> Option<RefMut<'_, dyn StreamingCheckpointParticipant>> {
                self.try_borrow_mut().ok().map(|stage| {
                    RefMut::map(stage, |inner| {
                        let participant: &mut dyn StreamingCheckpointParticipant = inner.as_mut();
                        participant
                    })
                })
            }
        }
    };
}

participant_cell!(dyn StreamingDatasetSource);
participant_cell!(dyn StreamingDatasetFormat);
participant_cell!(dyn StreamingSessionCoordinator);
participant_cell!(dyn StreamingPlacementPolicy);
participant_cell!(dyn StreamingPlacementDriver);

/// Checkpoint participant view over a stage the pipeline also drives.
///
/// The borrow is taken inside each call and released before it returns; the
/// pipeline holds no stage borrow while a barrier is in flight, so a failure to
/// borrow is a discipline violation and is reported as a typed
/// [`CheckpointError::ParticipantUnavailable`] at a barrier rather than a panic
/// in the fused loop.
pub struct CheckpointProxy {
    participant_id: CheckpointParticipantId,
    stage: Rc<dyn ParticipantCell>,
}

impl CheckpointProxy {
    fn borrow_stage(&self) -> Result<RefMut<'_, dyn StreamingCheckpointParticipant>, CheckpointError> {
        self.stage.try_borrow_participant().ok_or_else(|| {
            CheckpointError::ParticipantUnavailable {
                participant: self.participant_id.clone(),
            }
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for CheckpointProxy {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let mut stage = self.borrow_stage()?;
        stage.checkpoint_view(barrier).await
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        let mut stage = self.borrow_stage()?;
        stage.initialize(state).await
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        let mut stage = self.borrow_stage()?;
        stage.checkpoint_committed(receipt).await
    }
}

/// Phase-owned services injected into one pipeline run.
pub struct StreamingPhaseContext {
    /// Run clock; every pipeline timer routes through it.
    pub clock: Rc<dyn Clock>,
    /// Single-writer checkpoint publication sequencer.
    pub checkpoint: StreamingCheckpointCoordinator,
    /// Run-level stop signal driven by phase cancellation.
    pub stop: StreamingStopReceiver,
}

/// Exact reason a pipeline run terminated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamingTerminalReason {
    /// The source sealed and every accepted action reached terminal.
    Sealed,
    /// Run-level cancellation fenced admission and the prefix drained.
    Cancelled,
    /// A frozen threshold fenced admission; the truthful prefix drained.
    DrainedAfterReliabilityFence {
        /// Deterministic identity of the fencing issue.
        issue_id: ContentDigest,
    },
    /// The private host classifier verified a terminal invariant.
    FailedInvariant {
        /// Deterministic identity of the invariant issue.
        issue_id: ContentDigest,
    },
}

/// Outcome of one bounded pipeline run.
#[derive(Clone, Debug)]
pub struct StreamingRunOutcome {
    /// Exact terminal reason.
    pub terminal_reason: StreamingTerminalReason,
    /// Last generation that became authoritative, when any.
    pub last_committed_generation: Option<CommittedCheckpointGeneration>,
}

/// Bounded pipeline failure.
#[derive(Debug)]
pub enum StreamingPipelineError {
    /// More than one phase, or a warmup phase, was composed with a stream.
    ///
    /// A run installs and closes exactly one terminal lane, so two phases would
    /// either share one lane across independent stream generations or need a
    /// second lane the drain loop was never told about. The pipeline supplies
    /// this vocabulary; capability agreement raises it before any component is
    /// prepared.
    UnsupportedPhaseComposition,
    /// The frozen participant set was invalid or incomplete.
    ParticipantPlan(CheckpointParticipantPlanError),
    /// Source acquisition failed and no scoped disposition applied.
    Source(StreamSourceError),
    /// Decoding failed and no scoped disposition applied.
    Format(StreamFormatError),
    /// Session coordination failed and no scoped disposition applied.
    Session(SessionCoordinatorError),
    /// Placement failed and no scoped disposition applied.
    Placement(PlacementError),
    /// Action dispatch failed and no scoped disposition applied.
    Action(ActionExecutionError),
    /// A checkpoint could not represent a truthful cut.
    Checkpoint(CheckpointError),
}

impl std::fmt::Display for StreamingPipelineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPhaseComposition => {
                write!(formatter, "streaming supports exactly one profiling phase")
            }
            Self::ParticipantPlan(error) => {
                write!(formatter, "streaming participant plan: {error:?}")
            }
            Self::Source(error) => write!(formatter, "streaming source: {error}"),
            Self::Format(error) => write!(formatter, "streaming format: {error}"),
            Self::Session(error) => write!(formatter, "streaming session: {error}"),
            Self::Placement(error) => write!(formatter, "streaming placement: {error}"),
            Self::Action(error) => write!(formatter, "streaming action: {error}"),
            Self::Checkpoint(error) => write!(formatter, "streaming checkpoint: {error:?}"),
        }
    }
}

impl std::error::Error for StreamingPipelineError {}

/// Authored bounds for one pipeline run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingPipelineLimits {
    /// Strict per-pull decoder bound.
    pub decode_batch: DecodeBatchBudget,
    /// Maximum actions the session sink may buffer between drains.
    pub max_pending_actions: usize,
    /// Nanoseconds between checkpoint barriers, on the run clock.
    pub checkpoint_interval_ns: i64,
}

/// Cheaply cloneable pipeline control surface.
#[derive(Clone)]
pub struct StreamingPipelineControl {
    stop: super::source::StreamingSourceControl,
    is_admission_fenced: Rc<Cell<bool>>,
}

impl StreamingPipelineControl {
    /// Request an orderly stop and wake every pending stage.
    pub fn stop(&self) {
        self.stop.stop();
    }

    /// Fence new admission without cancelling accepted work.
    pub fn fence_admission(&self) {
        self.is_admission_fenced.set(true);
    }

    /// Return whether admission is currently fenced.
    #[must_use]
    pub fn is_admission_fenced(&self) -> bool {
        self.is_admission_fenced.get()
    }
}

/// Why the pipeline stopped admitting new units.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DrainReason {
    /// The source sealed; drain the accepted prefix.
    Sealed,
    /// Run-level cancellation; cancel accepted work under phase policy.
    Cancelled,
    /// A frozen reliability threshold fenced admission.
    ReliabilityFence {
        /// Deterministic identity of the fencing issue.
        issue_id: ContentDigest,
    },
    /// The private host classifier verified a terminal invariant.
    FailedInvariant {
        /// Deterministic identity of the invariant issue.
        issue_id: ContentDigest,
    },
}

impl DrainReason {
    /// Return whether accepted in-flight work is cancelled rather than drained.
    #[must_use]
    pub const fn cancels_inflight(self) -> bool {
        matches!(self, Self::Cancelled | Self::FailedInvariant { .. })
    }

    const fn terminal_reason(self) -> StreamingTerminalReason {
        match self {
            Self::Sealed => StreamingTerminalReason::Sealed,
            Self::Cancelled => StreamingTerminalReason::Cancelled,
            Self::ReliabilityFence { issue_id } => {
                StreamingTerminalReason::DrainedAfterReliabilityFence { issue_id }
            }
            Self::FailedInvariant { issue_id } => {
                StreamingTerminalReason::FailedInvariant { issue_id }
            }
        }
    }

    const fn checkpoint_terminal_reason(self) -> CheckpointTerminalReason {
        match self {
            Self::Sealed => CheckpointTerminalReason::Completed,
            Self::Cancelled | Self::ReliabilityFence { .. } => CheckpointTerminalReason::Cancelled,
            Self::FailedInvariant { .. } => CheckpointTerminalReason::Aborted,
        }
    }
}

/// Where the fused loop currently is in the reserve-process-settle cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PipelinePhase {
    /// Free to acquire the next unit's permits and pull.
    Pulling,
    /// Admission is fenced; only settle and drain arms remain live.
    Draining(DrainReason),
}

/// Bounded pipeline-owned collection point for session-emitted actions.
///
/// This is not a channel. `send_action` is called synchronously inside one
/// [`StreamingSessionCoordinator::ingest`], and the fused loop drains it on the
/// very next statement, so a bounded `Vec` is both the correct capacity bound
/// and one allocation for the run.
#[derive(Debug)]
struct PipelineActionSink {
    actions: Vec<ExecutableDatasetAction>,
    frontier: Option<SessionCausalFrontier>,
    max_pending: usize,
}

impl PipelineActionSink {
    fn new(max_pending: usize) -> Self {
        Self {
            actions: Vec::with_capacity(max_pending),
            frontier: None,
            max_pending,
        }
    }

    fn drain(&mut self) -> Vec<ExecutableDatasetAction> {
        std::mem::take(&mut self.actions)
    }
}

#[async_trait(?Send)]
impl DatasetActionSink for PipelineActionSink {
    async fn send_action(
        &mut self,
        action: ExecutableDatasetAction,
    ) -> Result<(), SessionCoordinatorError> {
        if self.actions.len() >= self.max_pending {
            return Err(SessionCoordinatorError::state_budget(
                StateBudgetFailureCode::ItemCapacity,
            ));
        }
        self.actions.push(action);
        Ok(())
    }

    async fn advance_causal_frontier(
        &mut self,
        frontier: SessionCausalFrontier,
    ) -> Result<(), SessionCoordinatorError> {
        // Monotonic by contract; the greatest proven prefix wins so a replayed
        // fragment cannot walk the frontier backwards.
        let is_newer = self
            .frontier
            .as_ref()
            .is_none_or(|current| frontier.through_sequence >= current.through_sequence);
        if is_newer {
            self.frontier = Some(frontier);
        }
        Ok(())
    }
}

/// Bounded pipeline-owned collection point for decoder-emitted format events.
#[derive(Default)]
struct PipelineFormatSink {
    events: Vec<FormatEvent>,
}

#[async_trait(?Send)]
impl FormatEventSink for PipelineFormatSink {
    async fn send(&mut self, event: FormatEvent) -> Result<(), StreamFormatError> {
        self.events.push(event);
        Ok(())
    }
}

/// Every prepared component one bounded pipeline composes.
///
/// The five stages the pipeline drives arrive by value and are installed into
/// stage cells. The remaining checkpoint participants — the event-time policy,
/// the action-driver bindings, the active-execution set, the blocking owner, and
/// the result-epoch coordinator — are owned elsewhere, so the assembler supplies
/// their participant views directly and the pipeline only records their
/// identities in the frozen plan.
pub struct PreparedStreamingComponents {
    /// Immutable-source discovery and acquisition owner.
    pub source: Box<dyn StreamingDatasetSource>,
    /// Decode owner.
    pub format: Box<dyn StreamingDatasetFormat>,
    /// Cross-partition session-state owner.
    pub session: Box<dyn StreamingSessionCoordinator>,
    /// Multiplexing action host, which owns sequencing and dispatch.
    pub action: StreamingActionHost,
    /// Prepared placement binding.
    pub placement: PreparedStreamingPlacementBinding,
    /// Immutable-acquisition budget.
    pub acquisition: AcquisitionBudget,
    /// Event-time and global-order policy participant.
    pub event_time_order_policy: Box<dyn StreamingCheckpointParticipant>,
    /// One participant per prepared action-driver binding.
    pub action_driver_bindings: Vec<Box<dyn StreamingCheckpointParticipant>>,
    /// Active-execution set participant.
    pub active_execution_set: Box<dyn StreamingCheckpointParticipant>,
    /// Blocking-work owner participant.
    pub blocking_owner: Box<dyn StreamingCheckpointParticipant>,
    /// Result and terminal epoch participant.
    pub result_epoch: Box<dyn StreamingCheckpointParticipant>,
}

/// Bounded owner of one complete streaming run.
pub struct StreamingPipeline {
    run: StreamRunIdentity,
    limits: StreamingPipelineLimits,
    participant_plan: CheckpointParticipantPlan,
    source: StageCell<Box<dyn StreamingDatasetSource>>,
    format: StageCell<Box<dyn StreamingDatasetFormat>>,
    session: StageCell<Box<dyn StreamingSessionCoordinator>>,
    policy: StageCell<Box<dyn StreamingPlacementPolicy>>,
    placement_driver: StageCell<Box<dyn StreamingPlacementDriver>>,
    admission: RefCell<Box<dyn StreamingPlacementAdmission>>,
    submitter: RefCell<Box<dyn StreamingPlacementSubmitter>>,
    placement_control: Rc<dyn StreamingPlacementControl>,
    action: RefCell<StreamingActionHost>,
    acquisition: AcquisitionBudget,
    sink: RefCell<PipelineActionSink>,
    decoder: RefCell<Option<Box<dyn StreamingPartitionDecoder>>>,
    control: StreamingPipelineControl,
    /// Placements retained until their action reaches terminal.
    placements: RefCell<std::collections::BTreeMap<super::identity::StableActionId, PlacementHandle>>,
    /// In-flight action count per placed session, so route capacity is released
    /// at the session's causal terminal rather than at each action's.
    session_inflight: RefCell<std::collections::BTreeMap<StableSessionKey, usize>>,
    accepted: Cell<u64>,
    settled: Cell<u64>,
    source_pulls: Cell<u64>,
    decode_pulls: Cell<u64>,
    is_source_sealed: Cell<bool>,
}

impl StreamingPipeline {
    /// Freeze the exact participant inventory and build the control surface.
    ///
    /// This is the only place the participant plan is computed. Every proxy the
    /// checkpoint coordinator receives is minted here, so a stage that appears
    /// in the pipeline but not in the plan is a construction-time
    /// [`CheckpointParticipantPlanError`], never a barrier-time surprise. The
    /// plan is frozen before the first source poll.
    ///
    /// # Errors
    ///
    /// Returns [`StreamingPipelineError::ParticipantPlan`] when a required
    /// stateful owner is absent or two owners claim the same identity.
    pub fn prepare(
        run: StreamRunIdentity,
        limits: StreamingPipelineLimits,
        components: PreparedStreamingComponents,
        stop: super::source::StreamingSourceControl,
    ) -> Result<
        (
            Self,
            StreamingPipelineControl,
            Vec<Box<dyn StreamingCheckpointParticipant>>,
        ),
        StreamingPipelineError,
    > {
        let PreparedStreamingComponents {
            source,
            format,
            session,
            action,
            placement,
            acquisition,
            event_time_order_policy,
            action_driver_bindings,
            active_execution_set,
            blocking_owner,
            result_epoch,
        } = components;

        let source = Rc::new(RefCell::new(source));
        let format = Rc::new(RefCell::new(format));
        let session = Rc::new(RefCell::new(session));
        let policy = Rc::new(RefCell::new(placement.policy));
        let placement_driver = Rc::new(RefCell::new(placement.driver));

        // Identities are read from the decorated stage itself, never from a
        // hard-coded string, so a counting or tracing decorator installed around
        // a stage still lands in the frozen plan under its own identity.
        let owners = CheckpointParticipantOwners {
            source: Some(source.borrow().participant_id()),
            format: Some(format.borrow().participant_id()),
            event_time_order_policy: Some(event_time_order_policy.participant_id()),
            session_coordinator: Some(session.borrow().participant_id()),
            action_driver_bindings: action_driver_bindings
                .iter()
                .map(|binding| binding.participant_id())
                .collect(),
            placement_policy: Some(policy.borrow().participant_id()),
            placement_driver: Some(placement_driver.borrow().participant_id()),
            active_execution_set: Some(active_execution_set.participant_id()),
            blocking_owner: Some(blocking_owner.participant_id()),
            result_epoch: Some(result_epoch.participant_id()),
        };
        let source_id = source.borrow().participant_id();
        let format_id = format.borrow().participant_id();
        let session_id = session.borrow().participant_id();
        let policy_id = policy.borrow().participant_id();
        let placement_driver_id = placement_driver.borrow().participant_id();
        let participant_plan = CheckpointParticipantPlan::from_required_owners(owners)
            .map_err(StreamingPipelineError::ParticipantPlan)?;

        let mut participants: Vec<Box<dyn StreamingCheckpointParticipant>> = vec![
            Box::new(proxy(&source, source_id)),
            Box::new(proxy(&format, format_id)),
            Box::new(proxy(&session, session_id)),
            Box::new(proxy(&policy, policy_id)),
            Box::new(proxy(&placement_driver, placement_driver_id)),
            event_time_order_policy,
            active_execution_set,
            blocking_owner,
            result_epoch,
        ];
        participants.extend(action_driver_bindings);

        let control = StreamingPipelineControl {
            stop,
            is_admission_fenced: Rc::new(Cell::new(false)),
        };
        let pipeline = Self {
            run,
            limits,
            participant_plan,
            source,
            format,
            session,
            policy,
            placement_driver,
            admission: RefCell::new(placement.admission),
            submitter: RefCell::new(placement.submitter),
            placement_control: placement.control,
            action: RefCell::new(action),
            acquisition,
            sink: RefCell::new(PipelineActionSink::new(limits.max_pending_actions)),
            decoder: RefCell::new(None),
            control: control.clone(),
            placements: RefCell::new(std::collections::BTreeMap::new()),
            session_inflight: RefCell::new(std::collections::BTreeMap::new()),
            accepted: Cell::new(0),
            settled: Cell::new(0),
            source_pulls: Cell::new(0),
            decode_pulls: Cell::new(0),
            is_source_sealed: Cell::new(false),
        };
        Ok((pipeline, control, participants))
    }

    /// Borrow the frozen participant inventory.
    #[must_use]
    pub const fn participant_plan(&self) -> &CheckpointParticipantPlan {
        &self.participant_plan
    }

    /// Return the number of source events pulled so far.
    #[must_use]
    pub fn source_pull_count(&self) -> u64 {
        self.source_pulls.get()
    }

    /// Return the number of decoder pulls issued so far.
    #[must_use]
    pub fn decode_pull_count(&self) -> u64 {
        self.decode_pulls.get()
    }

    /// Return whether every accepted action has settled.
    #[must_use]
    pub fn is_quiescent(&self) -> bool {
        self.accepted.get() == self.settled.get()
    }

    /// Run the complete bounded pipeline to a terminal outcome.
    ///
    /// # Errors
    ///
    /// Returns the first unscoped stage failure, or a checkpoint failure when no
    /// truthful cut can be published.
    pub async fn run(
        self,
        phase: StreamingPhaseContext,
    ) -> Result<StreamingRunOutcome, StreamingPipelineError> {
        self.run_fused(phase).await
    }

    async fn run_fused(
        self,
        phase: StreamingPhaseContext,
    ) -> Result<StreamingRunOutcome, StreamingPipelineError> {
        let StreamingPhaseContext {
            clock,
            mut checkpoint,
            mut stop,
        } = phase;

        let this = &self;
        let mut state = PipelinePhase::Pulling;
        let mut last_committed: Option<CommittedCheckpointGeneration> = None;
        let mut epoch: u64 = 0;
        let mut next_barrier_ns = clock
            .now_ns()
            .saturating_add(this.limits.checkpoint_interval_ns);
        // Exactly one owned in-flight admission cycle. Retaining it here rather
        // than rebuilding it every iteration is what makes "no source pull while
        // a downstream reservation is pending" a property of the loop instead of
        // a convention: a settle event resolving first must not cancel the
        // submission that settle is meant to unblock.
        let mut inflight: Option<
            LocalBoxFuture<'_, Result<Option<DrainReason>, StreamingPipelineError>>,
        > = None;

        loop {
            if let PipelinePhase::Draining(reason) = state
                && this.is_quiescent()
            {
                // Load-bearing drop, not a dead store: a retained admission
                // cycle may be parked inside the action host's `submit` and so
                // still holds that borrow, which `shutdown` needs to join the
                // drivers. The unit it was building was never accepted, so
                // nothing settles for it.
                drop(inflight.take());
                this.shutdown(reason).await?;
                epoch = epoch.saturating_add(1);
                let generation = this
                    .commit_barrier(
                        &mut checkpoint,
                        epoch,
                        CheckpointBarrierFinality::Terminal(reason.checkpoint_terminal_reason()),
                    )
                    .await?;
                return Ok(StreamingRunOutcome {
                    terminal_reason: reason.terminal_reason(),
                    last_committed_generation: Some(generation),
                });
            }

            let is_admitting = matches!(state, PipelinePhase::Pulling)
                && !this.control.is_admission_fenced()
                && !this.is_source_sealed.get();
            if !is_admitting {
                // A fenced or sealed pipeline pulls nothing, so a parked
                // admission cycle is dropped rather than retained: the unit it
                // was building was never accepted and nothing settles for it.
                inflight = None;
            } else if inflight.is_none() {
                inflight = Some(this.admit_next_unit().boxed_local());
            }
            // The action host owns submission and its multiplexed event stream
            // behind one exclusive borrow, so the event arm is armed only when
            // no admission cycle can be holding that borrow. While admission is
            // live the events keep accumulating in the drivers; the loop
            // consumes them as soon as admission fences, which every terminal
            // path does before it waits for quiescence.
            let is_event_arm_live = inflight.is_none();

            let step = select_biased! {
                // 1. Settlement first: it is the only arm that returns capacity.
                //    Preferring it over admission is what keeps a saturated
                //    pipeline from livelocking on a submit only a settle can
                //    unblock.
                event = next_placement_event(&this.placement_driver).fuse() => {
                    LoopStep::Placement(event)
                }
                // 2. Action events, republished as the single `PlacementEvent::Action`
                //    route back into session state.
                event = next_action_event(&this.action, is_event_arm_live).fuse() => {
                    LoopStep::Action(event)
                }
                // 3. Shutdown, which fences admission but never drops accepted work.
                _ = stop.stopped().fuse() => LoopStep::Stopped,
                // 4. Checkpoint tick, on the run clock.
                _ = clock_tick(&clock, next_barrier_ns).fuse() => LoopStep::Barrier,
                // 5. Admission LAST, and only while nothing downstream is pending.
                //    While an admission cycle is in flight this arm owns the task,
                //    so there is deliberately no concurrent source pull.
                admitted = poll_inflight(&mut inflight).fuse() => {
                    LoopStep::Admitted(admitted)
                }
            };

            match step {
                LoopStep::Placement(event) => {
                    let event = event.map_err(StreamingPipelineError::Placement)?;
                    this.settle_placement_event(event).await?;
                }
                LoopStep::Action(event) => {
                    let event = event.map_err(StreamingPipelineError::Action)?;
                    // The action host's events reach session state only by being
                    // republished here as `PlacementEvent::Action`; the pipeline
                    // holds no second path into `observe_execution`.
                    this.settle_placement_event(PlacementEvent::Action(event))
                        .await?;
                }
                LoopStep::Stopped => {
                    state = PipelinePhase::Draining(DrainReason::Cancelled);
                }
                LoopStep::Barrier => {
                    epoch = epoch.saturating_add(1);
                    let generation = this
                        .commit_barrier(
                            &mut checkpoint,
                            epoch,
                            CheckpointBarrierFinality::Continuing,
                        )
                        .await?;
                    last_committed = Some(generation);
                    next_barrier_ns = clock
                        .now_ns()
                        .saturating_add(this.limits.checkpoint_interval_ns);
                }
                LoopStep::Admitted(result) => {
                    inflight = None;
                    if let Some(reason) = result? {
                        state = PipelinePhase::Draining(reason);
                    }
                }
            }

            let _ = &last_committed;
        }
    }

    /// Reserve every downstream permit, pull one unit, and admit its actions.
    ///
    /// Returns the drain reason when the source sealed. While this future is
    /// pending on any downstream reservation, the fused loop has no other arm
    /// that can pull from the source, which is why
    /// `source_pull_count` cannot advance past a saturated downstream.
    async fn admit_next_unit(&self) -> Result<Option<DrainReason>, StreamingPipelineError> {
        if self.decoder.borrow().is_some() {
            return self.decode_and_admit().await;
        }

        let event = {
            let mut source = self.source.borrow_mut();
            source.next_event().await
        };
        self.source_pulls.set(self.source_pulls.get() + 1);
        let event = match event {
            Ok(event) => event,
            Err(error) if error.is_stopped() => return Ok(Some(DrainReason::Cancelled)),
            Err(error) => return Err(StreamingPipelineError::Source(error)),
        };

        match event {
            SourceEvent::Partition(partition) => {
                // Reservation 1: the acquisition lease, outermost of the chain.
                let acquired = partition
                    .content()
                    .acquire(
                        PartitionAccessRequest::Sequential { resume_offset: 0 },
                        &self.acquisition,
                    )
                    .await
                    .map_err(StreamingPipelineError::Source)?;
                let decoder = {
                    let mut format = self.format.borrow_mut();
                    format
                        .begin_partition(acquired, None)
                        .await
                        .map_err(StreamingPipelineError::Format)?
                };
                *self.decoder.borrow_mut() = Some(decoder);
                Ok(None)
            }
            SourceEvent::Frontier(frontier) => {
                let mut events = PipelineFormatSink::default();
                {
                    let mut format = self.format.borrow_mut();
                    format
                        .advance_source_frontier(frontier, &mut events)
                        .await
                        .map_err(StreamingPipelineError::Format)?;
                }
                self.apply_format_events(events.events).await?;
                Ok(None)
            }
            SourceEvent::Seal(seal) => {
                let mut events = PipelineFormatSink::default();
                {
                    let mut format = self.format.borrow_mut();
                    format
                        .seal(seal.clone(), &mut events)
                        .await
                        .map_err(StreamingPipelineError::Format)?;
                }
                self.apply_format_events(events.events).await?;
                {
                    let mut session = self.session.borrow_mut();
                    let mut sink = self.sink.borrow_mut();
                    session
                        .seal(seal, &mut *sink)
                        .await
                        .map_err(StreamingPipelineError::Session)?;
                }
                self.drain_sink().await?;
                self.is_source_sealed.set(true);
                Ok(Some(DrainReason::Sealed))
            }
        }
    }

    async fn decode_and_admit(&self) -> Result<Option<DrainReason>, StreamingPipelineError> {
        // Reservation 2: the strict per-pull decode bound. Not a lease — an
        // exact upper bound on what one pull may materialize.
        let step = {
            let mut decoder = self.decoder.borrow_mut();
            let Some(decoder) = decoder.as_mut() else {
                return Ok(None);
            };
            decoder.next_batch(self.limits.decode_batch).await
        };
        self.decode_pulls.set(self.decode_pulls.get() + 1);
        match step.map_err(StreamingPipelineError::Format)? {
            DecodeStep::Batch(batch) => {
                for fragment in batch.fragments {
                    // Reservation 3: the session retained-state charge, which
                    // refuses rather than waits.
                    {
                        let mut session = self.session.borrow_mut();
                        let mut sink = self.sink.borrow_mut();
                        session
                            .ingest(fragment, &mut *sink)
                            .await
                            .map_err(StreamingPipelineError::Session)?;
                    }
                    self.drain_sink().await?;
                }
                Ok(None)
            }
            DecodeStep::End(_) => {
                *self.decoder.borrow_mut() = None;
                Ok(None)
            }
        }
    }

    async fn apply_format_events(
        &self,
        events: Vec<FormatEvent>,
    ) -> Result<(), StreamingPipelineError> {
        for event in events {
            {
                let mut session = self.session.borrow_mut();
                let mut sink = self.sink.borrow_mut();
                match event {
                    FormatEvent::Fragment(fragment) => session
                        .ingest(fragment, &mut *sink)
                        .await
                        .map_err(StreamingPipelineError::Session)?,
                    FormatEvent::SessionFrontier(watermark) => session
                        .advance_watermark(watermark, &mut *sink)
                        .await
                        .map_err(StreamingPipelineError::Session)?,
                }
            }
            self.drain_sink().await?;
        }
        Ok(())
    }

    async fn drain_sink(&self) -> Result<(), StreamingPipelineError> {
        let actions = self.sink.borrow_mut().drain();
        for action in actions {
            self.admit_action(action).await?;
        }
        Ok(())
    }

    /// Route, place, and submit exactly one causally ready action.
    ///
    /// `reserve_route` is polled on the separately owned admission object, so a
    /// terminal event arriving while the reservation is pending can still call
    /// `observe_session_terminal` on the policy and release route capacity. Once
    /// the reservation resolves, `install_route_reservation` and `place` run
    /// with no intervening `.await`, so the route map and the proven capacity
    /// cannot diverge.
    async fn admit_action(
        &self,
        action: ExecutableDatasetAction,
    ) -> Result<(), StreamingPipelineError> {
        // Reservation 4: the placement route charge, when the policy declares one.
        let charge = {
            let policy = self.policy.borrow();
            policy
                .route_admission(&action)
                .map_err(StreamingPipelineError::Placement)?
        };
        let reservation = match charge {
            Some(charge) => Some(
                self.admission
                    .borrow_mut()
                    .reserve_route(charge)
                    .await
                    .map_err(StreamingPipelineError::Placement)?,
            ),
            None => None,
        };

        // Synchronous block: no `.await` between installing the proven capacity
        // and deciding the route it pays for.
        let decision = {
            let mut policy = self.policy.borrow_mut();
            if let Some(reservation) = reservation {
                policy
                    .install_route_reservation(reservation)
                    .map_err(StreamingPipelineError::Placement)?;
            }
            policy
                .place(&action)
                .map_err(StreamingPipelineError::Placement)?
        };

        // Reservation 5: the bounded placement slab entry.
        let handle = {
            let mut submitter = self.submitter.borrow_mut();
            submitter
                .prepare(decision, &action)
                .await
                .map_err(StreamingPipelineError::Placement)?
        };
        let action_id = action.action_id();
        self.placements.borrow_mut().insert(action_id, handle);
        *self
            .session_inflight
            .borrow_mut()
            .entry(handle.session)
            .or_insert(0) += 1;

        // Reservations 6 and 7: the active-execution lease and, transitively
        // inside the binding's own `submit`, the terminal-record permit. The
        // host assigns the dense global sequence here; the pipeline does not.
        let sequence = {
            let mut action_host = self.action.borrow_mut();
            action_host.submit(action).await
        };
        let sequence = match sequence {
            Ok(sequence) => sequence,
            Err(error) => {
                // A refused submission consumes no sequence, so the prepared
                // placement must not outlive it.
                self.release_placement(action_id).await?;
                return Err(StreamingPipelineError::Action(error));
            }
        };
        self.bind_sequence(action_id, handle.id, sequence)?;
        self.accepted.set(self.accepted.get() + 1);
        Ok(())
    }

    fn bind_sequence(
        &self,
        action_id: super::identity::StableActionId,
        handle: PlacementHandleId,
        sequence: GlobalSequence,
    ) -> Result<(), StreamingPipelineError> {
        self.submitter
            .borrow_mut()
            .bind_sequence(handle, sequence)
            .map_err(StreamingPipelineError::Placement)?;
        if let Some(entry) = self.placements.borrow_mut().get_mut(&action_id) {
            entry.global_sequence = Some(sequence);
        }
        Ok(())
    }

    async fn release_placement(
        &self,
        action_id: super::identity::StableActionId,
    ) -> Result<(), StreamingPipelineError> {
        let handle = self.placements.borrow_mut().remove(&action_id);
        if let Some(handle) = handle {
            self.submitter
                .borrow_mut()
                .release(handle.id)
                .await
                .map_err(StreamingPipelineError::Placement)?;
        }
        Ok(())
    }

    /// Decrement one session's in-flight count and report whether it reached zero.
    fn is_session_causally_terminal(&self, session: StableSessionKey) -> bool {
        let mut inflight = self.session_inflight.borrow_mut();
        let Some(count) = inflight.get_mut(&session) else {
            return false;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            inflight.remove(&session);
            return true;
        }
        false
    }

    /// Apply one settle event.
    ///
    /// `PlacementEvent::Action` is the only variant that reaches session state.
    async fn settle_placement_event(
        &self,
        event: PlacementEvent,
    ) -> Result<(), StreamingPipelineError> {
        match event {
            PlacementEvent::Prepared(_) => Ok(()),
            PlacementEvent::Released(receipt) => {
                self.submitter
                    .borrow_mut()
                    .release(receipt.handle)
                    .await
                    .map_err(StreamingPipelineError::Placement)?;
                Ok(())
            }
            PlacementEvent::Failed(receipt) => {
                debug!(
                    code = ?receipt.code,
                    handle = ?receipt.handle,
                    component = "streaming.pipeline",
                    "placement reported a scoped failure"
                );
                Ok(())
            }
            PlacementEvent::Action(event) => self.settle_action_event(event).await,
        }
    }

    async fn settle_action_event(
        &self,
        event: ActionExecutionEvent,
    ) -> Result<(), StreamingPipelineError> {
        let terminal = match &event {
            ActionExecutionEvent::Terminal(receipt) => {
                Some((receipt.event.action_id, receipt.disposition))
            }
            _ => None,
        };

        {
            let mut session = self.session.borrow_mut();
            let mut sink = self.sink.borrow_mut();
            session
                .observe_execution(event, &mut *sink)
                .await
                .map_err(StreamingPipelineError::Session)?;
        }

        if let Some((action_id, disposition)) = terminal {
            let handle = self.placements.borrow().get(&action_id).copied();
            if let Some(handle) = handle {
                // A route serves a session, not an action, so it is fenced only
                // once every action placed on it has reached terminal. Releasing
                // per action would fence the epoch out from under the session's
                // own siblings.
                let frontier = self.sink.borrow().frontier.clone();
                if self.is_session_causally_terminal(handle.session)
                    && let Some(frontier) = frontier
                {
                    // Route capacity returns here, which is what wakes a pending
                    // reservation in the next reserve phase.
                    self.policy
                        .borrow_mut()
                        .observe_session_terminal(handle.session, handle.ownership_epoch, &frontier)
                        .map_err(StreamingPipelineError::Placement)?;
                }
                self.release_placement(action_id).await?;
            }
            self.settled.set(self.settled.get() + 1);
            debug!(
                ?disposition,
                settled = self.settled.get(),
                component = "streaming.pipeline",
                "action reached terminal"
            );
        }

        // Actions the fold emitted in response are admitted under the same
        // permit chain as authored ones.
        self.drain_sink().await
    }

    /// Fence admission, cancel or drain accepted work, and join every owner.
    async fn shutdown(&self, reason: DrainReason) -> Result<(), StreamingPipelineError> {
        self.control.fence_admission();
        self.placement_control.stop_preparing();
        self.placement_control.cancel_pending();

        if reason.cancels_inflight() {
            self.placement_control
                .cancel_inflight()
                .await
                .map_err(StreamingPipelineError::Placement)?;
        }

        {
            let mut action = self.action.borrow_mut();
            action
                .cancel_and_join()
                .await
                .map_err(StreamingPipelineError::Action)?;
        }
        self.placement_driver
            .borrow_mut()
            .drain()
            .await
            .map_err(StreamingPipelineError::Placement)?;
        Ok(())
    }

    async fn commit_barrier(
        &self,
        checkpoint: &mut StreamingCheckpointCoordinator,
        epoch: u64,
        finality: CheckpointBarrierFinality,
    ) -> Result<CommittedCheckpointGeneration, StreamingPipelineError> {
        let barrier = CheckpointBarrier {
            run: self.run.clone(),
            epoch: CheckpointEpoch::new(epoch),
            cut: self.current_cut(),
            plan_digest: self.participant_plan.digest(),
        };
        let mut results = PreparedCheckpointResultInput::empty();
        checkpoint
            .commit_barrier_with_finality(barrier, finality, &mut results)
            .await
            .map_err(StreamingPipelineError::Checkpoint)
    }

    fn current_cut(&self) -> super::checkpoint::CheckpointCut {
        use super::checkpoint::{
            AcquisitionHorizon, AdmissionHorizon, CheckpointCut, DecodeHorizon, DiscoveryHorizon,
            EventTimeWatermark, OrderedActionHorizon, TerminalActionHorizon,
        };
        use super::unit::SourcePosition;

        let frontier = self.sink.borrow().frontier.clone().unwrap_or_else(|| {
            SessionCausalFrontier {
                through_sequence: GlobalSequence::new(0),
                event_time: None,
                digest: ContentDigest::from_bytes([0; 32]),
            }
        });
        let position = SourcePosition::new(self.source_pulls.get());
        CheckpointCut {
            discovered: DiscoveryHorizon::new(position),
            acquired: AcquisitionHorizon::new(position),
            decoded: DecodeHorizon::new(position),
            ordered: OrderedActionHorizon::new(GlobalSequence::new(self.accepted.get())),
            admitted: AdmissionHorizon::new(GlobalSequence::new(self.accepted.get())),
            terminal: TerminalActionHorizon::new(GlobalSequence::new(self.settled.get())),
            event_watermark: EventTimeWatermark::Unknown,
            causal_frontier: frontier,
            handled_issues: super::reliability::HandledIssueCut::empty(),
        }
    }

    /// Apply one host-selected disposition to the pipeline's own state.
    ///
    /// The pipeline selects nothing. `Retry`, `Backpressure`, `Quarantine`,
    /// `Hole`, `Continue`, `TerminalActionReceipt`, and `ExportIncomplete` all
    /// keep the run alive; only `FailRun` — which only the reliability module's
    /// private classifier can produce — reaches failed-run shutdown.
    ///
    /// A `StreamingIssueOutcome` has private fields and is constructed only
    /// inside `reliability.rs`, so a caller outside that module cannot hand the
    /// pipeline a forged `Continue`.
    pub fn apply_disposition(
        &self,
        outcome: StreamingIssueOutcome,
        state: PipelinePhase,
    ) -> PipelinePhase {
        if outcome.needs_admission_fence() {
            self.control.fence_admission();
        }
        match outcome.disposition() {
            StreamingIssueDisposition::FailRun => PipelinePhase::Draining(
                DrainReason::FailedInvariant {
                    issue_id: outcome.issue_id(),
                },
            ),
            StreamingIssueDisposition::Retry
            | StreamingIssueDisposition::Backpressure
            | StreamingIssueDisposition::Quarantine
            | StreamingIssueDisposition::Hole
            | StreamingIssueDisposition::Continue
            | StreamingIssueDisposition::TerminalActionReceipt
            | StreamingIssueDisposition::ExportIncomplete => {
                if outcome.needs_admission_fence() {
                    PipelinePhase::Draining(DrainReason::ReliabilityFence {
                        issue_id: outcome.issue_id(),
                    })
                } else {
                    state
                }
            }
        }
    }
}

/// One resolved arm of the fused loop.
enum LoopStep {
    Placement(Result<PlacementEvent, PlacementError>),
    Action(Result<ActionExecutionEvent, ActionExecutionError>),
    Stopped,
    Barrier,
    Admitted(Result<Option<DrainReason>, StreamingPipelineError>),
}

/// Mint one checkpoint proxy over a stage the pipeline also drives.
///
/// The identity is the one already read from the decorated stage in
/// [`StreamingPipeline::prepare`], so the proxy and the frozen plan cannot
/// disagree.
fn proxy<S: ?Sized + 'static>(
    stage: &Rc<RefCell<Box<S>>>,
    participant_id: CheckpointParticipantId,
) -> CheckpointProxy
where
    RefCell<Box<S>>: ParticipantCell,
{
    CheckpointProxy {
        participant_id,
        stage: Rc::clone(stage) as Rc<dyn ParticipantCell>,
    }
}

/// Poll the retained admission cycle, or park when there is none.
///
/// The boxed future stays in `slot`, so the wrapper this returns can be dropped
/// by `select_biased!` without cancelling the admission cycle it polls.
async fn poll_inflight<T>(slot: &mut Option<LocalBoxFuture<'_, T>>) -> T {
    match slot.as_mut() {
        Some(inflight) => inflight.await,
        None => std::future::pending().await,
    }
}

/// Await the next placement event without holding a borrow across the loop body.
async fn next_placement_event(
    cell: &StageCell<Box<dyn StreamingPlacementDriver>>,
) -> Result<PlacementEvent, PlacementError> {
    let mut driver = cell.borrow_mut();
    driver.next_event().await
}

/// Await the next action event, or park when the host borrow belongs elsewhere.
///
/// The caller arms this only when no admission cycle is in flight, because the
/// action host serves both submission and the event stream through one
/// exclusive borrow. Parking without touching the cell keeps that discipline a
/// property of the loop rather than a runtime borrow check.
async fn next_action_event(
    cell: &RefCell<StreamingActionHost>,
    is_live: bool,
) -> Result<ActionExecutionEvent, ActionExecutionError> {
    if !is_live {
        return std::future::pending().await;
    }
    let mut host = cell.borrow_mut();
    host.next_event().await
}

/// Sleep on the run clock until the next barrier deadline.
async fn clock_tick(clock: &Rc<dyn Clock>, deadline_ns: i64) {
    let remaining = deadline_ns.saturating_sub(clock.now_ns());
    Rc::clone(clock).sleep(remaining).await;
}
