// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sole mutable telemetry archive owner and LocalSet receipt Clock bridge.
//!
//! Source pipelines send one command per decoded attempt or compact missed
//! range. Exactly one local task assigns global sequence, projects Arrow/WAL,
//! mutates the sink, observes durability on the injected Clock, and persists
//! the corresponding non-self-referential receipt before acknowledging the
//! source driver.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::fmt::{self, Debug, Display, Formatter};
use std::rc::Rc;
use std::sync::Arc;

use aiperf_clock::Clock;
use aiperf_telemetry_archive::{
    AdmissionRejection, AppendReceipt, ArchiveAdmissionPolicy, ArchiveBudgetClass,
    ArchiveFinalizationPermit, ArchiveFrameSequencerV1, ArchiveFrameTimingV1, ArchiveId,
    ArchiveProjectionFootprint, ArchiveProjectionLease, ArchiveSink, ArchiveSinkError,
    ArchiveSpoolBudgetAuthority, ArchiveSpoolBudgetSnapshot, ArchiveWalFrame, BoundaryReference,
    CheckpointCompletion, ControlFrameCodecV1, DurabilityCompletion, FinalizeCompletion,
    FixedLossLedgerV1, FrameId, LifecycleCompletionReasonV1, LifecycleMarkerKindV1,
    LifecycleMarkerV1, LifecyclePhaseStateV1, LocalArchiveRepository, LossFrameIdentityV1,
    LossKindV1, LossLedgerRecordOutcomeV1, LossLedgerViewV1, LossReasonV1, ObservationKind,
    ReceiptEventDraft, ReceiptEventV1, ReceiptObserverEpochId, ReceiptObserverEpochV1,
    ReceiptTargetV1, ScrapeReasonV1, SessionId, SourceFrameCodecV1, TelemetryAttemptDisposition,
    TerminationReason, UnsequencedLossV1, WalRangeTargetV1, receipt_range_coverage,
};
use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::telemetry_attachment::{AttachedLifecycleIngress, AttachedTelemetryError};
use crate::telemetry_pipeline::{
    ArchiveAttemptObservation, ArchiveAttemptOwner, ArchiveIssuedLossObservation,
    ArchiveMissedObservation, ArchiveOwnerError, AttachedArchiveAttemptOwner,
    AttachedAttemptAdmission,
};

/// Bounded owner task configuration frozen before source activation.
pub struct TelemetryArchiveOwnerConfig {
    /// Archive identity bound to the prepared sink and sequencer.
    pub archive_id: ArchiveId,
    /// Collection session bound to the prepared WAL segment.
    pub session_id: SessionId,
    /// Sole sequence/projection state machine.
    pub sequencer: ArchiveFrameSequencerV1,
    /// Closed source-frame Arrow/WAL codec.
    pub codec: SourceFrameCodecV1,
    /// Closed lifecycle/loss control-frame codec over the same schemas.
    pub control_codec: ControlFrameCodecV1,
    /// Prepared and lifetime-qualified durability sink.
    pub sink: Box<dyn ArchiveSink>,
    /// Run-owned Clock used only for completion observation.
    pub clock: Rc<dyn Clock>,
    /// Execution-specific durable receipt epoch.
    pub receipt_epoch: ReceiptObserverEpochV1,
    /// Whether sink preparation already persisted this epoch alone.
    pub receipt_epoch_registered: bool,
    /// First receipt-journal sequence after recovery.
    pub next_receipt_seq: u64,
    /// Preparation-bounded exact/saturation health authority.
    pub loss_ledger: FixedLossLedgerV1,
    /// Positive bounded command capacity.
    pub queue_capacity: usize,
    /// Positive capacity reserved exclusively for loss/lifecycle/finalize work.
    pub control_queue_capacity: usize,
    /// Whether writer failure degrades into retained report health instead of failing the run.
    pub best_effort: bool,
    /// Whether source attempts use native-first nonblocking attached admission.
    pub attached: bool,
    /// Prepared atomic authority for spool/queue quota and protected reserves.
    pub budget_authority: Arc<dyn ArchiveSpoolBudgetAuthority>,
    /// Product-mode admission semantics applied to both resource lanes.
    pub admission_policy: Arc<dyn ArchiveAdmissionPolicy>,
    /// Worst-case complete source-frame transaction.
    pub ordinary_projection_footprint: ArchiveProjectionFootprint,
    /// Worst-case lifecycle/loss control-frame transaction.
    pub control_projection_footprint: ArchiveProjectionFootprint,
}

impl Debug for TelemetryArchiveOwnerConfig {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TelemetryArchiveOwnerConfig")
            .field("archive_id", &self.archive_id)
            .field("session_id", &self.session_id)
            .field("sequencer", &self.sequencer)
            .field("codec", &self.codec)
            .field("control_codec", &self.control_codec)
            .field("sink", &self.sink)
            .field("virtual_clock", &self.clock.is_virtual())
            .field(
                "receipt_observer_epoch_id",
                &self.receipt_epoch.observer_epoch_id,
            )
            .field("receipt_epoch_registered", &self.receipt_epoch_registered)
            .field("next_receipt_seq", &self.next_receipt_seq)
            .field("loss_ledger", &self.loss_ledger)
            .field("queue_capacity", &self.queue_capacity)
            .field("control_queue_capacity", &self.control_queue_capacity)
            .field("best_effort", &self.best_effort)
            .field("attached", &self.attached)
            .field("budget_authority", &self.budget_authority)
            .field("admission_policy", &self.admission_policy)
            .field(
                "ordinary_projection_footprint",
                &self.ordinary_projection_footprint,
            )
            .field(
                "control_projection_footprint",
                &self.control_projection_footprint,
            )
            .finish()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct AttachedLossCounters {
    missed_ticks: u64,
    missed_ranges: u64,
    archive_rejected: u64,
    projection_failed: u64,
    writer_failed: u64,
    shutdown_abandoned: u64,
}

struct AttachedOwnerShared {
    archive_id: ArchiveId,
    session_id: SessionId,
    admission_open: Cell<bool>,
    writer_alive: Cell<bool>,
    shutdown_deadline_ns: Cell<Option<i64>>,
    source_next: RefCell<BTreeMap<String, u64>>,
    loss_ledger: RefCell<FixedLossLedgerV1>,
    budget_authority: Arc<dyn ArchiveSpoolBudgetAuthority>,
    admission_policy: Arc<dyn ArchiveAdmissionPolicy>,
    ordinary_projection_footprint: ArchiveProjectionFootprint,
    control_projection_footprint: ArchiveProjectionFootprint,
    counters: RefCell<AttachedLossCounters>,
    first_failure: RefCell<Option<String>>,
}

impl AttachedOwnerShared {
    #[allow(clippy::too_many_arguments)]
    fn new(
        archive_id: ArchiveId,
        session_id: SessionId,
        loss_ledger: FixedLossLedgerV1,
        budget_authority: Arc<dyn ArchiveSpoolBudgetAuthority>,
        admission_policy: Arc<dyn ArchiveAdmissionPolicy>,
        ordinary_projection_footprint: ArchiveProjectionFootprint,
        control_projection_footprint: ArchiveProjectionFootprint,
    ) -> Self {
        let source_next = loss_ledger
            .prepared_source_ids()
            .map(|source_id| (source_id.to_owned(), 0))
            .collect();
        Self {
            archive_id,
            session_id,
            admission_open: Cell::new(true),
            writer_alive: Cell::new(true),
            shutdown_deadline_ns: Cell::new(None),
            source_next: RefCell::new(source_next),
            loss_ledger: RefCell::new(loss_ledger),
            budget_authority,
            admission_policy,
            ordinary_projection_footprint,
            control_projection_footprint,
            counters: RefCell::new(AttachedLossCounters::default()),
            first_failure: RefCell::new(None),
        }
    }

    fn check_source_sequence(
        &self,
        source_id: &str,
        source_record_seq: u64,
    ) -> Result<(), TelemetryArchiveOwnerError> {
        let source_next = self.source_next.borrow();
        let expected = source_next
            .get(source_id)
            .copied()
            .ok_or_else(|| TelemetryArchiveOwnerError::UnknownLossSource(source_id.to_owned()))?;
        if source_record_seq != expected {
            return Err(TelemetryArchiveOwnerError::SourceAdmissionSequence {
                source_id: source_id.to_owned(),
                expected,
                actual: source_record_seq,
            });
        }
        Ok(())
    }

    fn commit_source_sequence(
        &self,
        source_id: &str,
        source_record_seq: u64,
    ) -> Result<(), TelemetryArchiveOwnerError> {
        let next = source_record_seq
            .checked_add(1)
            .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
        *self
            .source_next
            .borrow_mut()
            .get_mut(source_id)
            .ok_or_else(|| TelemetryArchiveOwnerError::UnknownLossSource(source_id.to_owned()))? =
            next;
        Ok(())
    }

    fn record_issued_loss(
        &self,
        observation: ArchiveIssuedLossObservation,
        register_source: bool,
    ) -> Result<(), TelemetryArchiveOwnerError> {
        if observation.source_id.is_empty() || observation.loss_kind.reason() != observation.reason
        {
            return Err(TelemetryArchiveOwnerError::InvalidIssuedLoss);
        }
        if register_source {
            self.check_source_sequence(&observation.source_id, observation.source_record_seq)?;
            observation
                .source_record_seq
                .checked_add(1)
                .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
        }
        let boundary_count = observation.boundary_refs.len();
        let boundary_capacity = self.loss_ledger.borrow().max_boundary_refs_per_range();
        if boundary_count > boundary_capacity {
            return Err(TelemetryArchiveOwnerError::BoundaryLossReserve {
                maximum: boundary_capacity,
                actual: boundary_count,
            });
        }
        let next_counter = {
            let counters = self.counters.borrow();
            let current = match observation.loss_kind {
                LossKindV1::ArchiveRejected => counters.archive_rejected,
                LossKindV1::ProjectionFailed => counters.projection_failed,
                LossKindV1::WriterFailed => counters.writer_failed,
                LossKindV1::ShutdownAbandoned => counters.shutdown_abandoned,
                LossKindV1::MissedCadence => {
                    return Err(TelemetryArchiveOwnerError::InvalidIssuedLoss);
                }
            };
            checked_increment(current)?
        };
        let outcome = self
            .loss_ledger
            .borrow_mut()
            .record_unsequenced(UnsequencedLossV1 {
                source_id: Some(observation.source_id.clone()),
                count: 1,
                loss_kind: observation.loss_kind,
                reason: observation.reason,
                first_source_record_seq: Some(observation.source_record_seq),
                last_source_record_seq: Some(observation.source_record_seq),
                first_request_attempt_seq: observation.request_attempt_seq,
                last_request_attempt_seq: observation.request_attempt_seq,
                first_tick: None,
                last_tick: None,
                first_deadline_ns: None,
                last_deadline_ns: None,
                loss_observed_ns: observation.observed_ns,
                boundary_refs: observation.boundary_refs,
                boundary_overflow_count: 0,
                boundary_overflow_digest: None,
            })
            .map_err(|error| TelemetryArchiveOwnerError::LossLedger(error.to_string()))?;
        if register_source {
            self.commit_source_sequence(&observation.source_id, observation.source_record_seq)?;
        }
        let mut counters = self.counters.borrow_mut();
        let counter = match observation.loss_kind {
            LossKindV1::ArchiveRejected => &mut counters.archive_rejected,
            LossKindV1::ProjectionFailed => &mut counters.projection_failed,
            LossKindV1::WriterFailed => &mut counters.writer_failed,
            LossKindV1::ShutdownAbandoned => &mut counters.shutdown_abandoned,
            LossKindV1::MissedCadence => return Err(TelemetryArchiveOwnerError::InvalidIssuedLoss),
        };
        *counter = next_counter;
        if boundary_count > 0 && matches!(outcome, LossLedgerRecordOutcomeV1::Saturated { .. }) {
            return Err(TelemetryArchiveOwnerError::BoundaryLossSaturated);
        }
        Ok(())
    }

    fn record_missed_loss(
        &self,
        observation: ArchiveMissedObservation,
    ) -> Result<(), TelemetryArchiveOwnerError> {
        if observation.source_id.is_empty() || observation.missed.count == 0 {
            return Err(TelemetryArchiveOwnerError::InvalidMissedRange);
        }
        let (next_missed_ticks, next_missed_ranges) = {
            let counters = self.counters.borrow();
            (
                counters
                    .missed_ticks
                    .checked_add(observation.missed.count)
                    .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?,
                checked_increment(counters.missed_ranges)?,
            )
        };
        self.loss_ledger
            .borrow_mut()
            .record_unsequenced(UnsequencedLossV1 {
                source_id: Some(observation.source_id),
                count: observation.missed.count,
                loss_kind: LossKindV1::MissedCadence,
                reason: LossReasonV1::CadenceOverrun,
                first_source_record_seq: None,
                last_source_record_seq: None,
                first_request_attempt_seq: None,
                last_request_attempt_seq: None,
                first_tick: Some(observation.missed.first_tick),
                last_tick: Some(observation.missed.last_tick),
                first_deadline_ns: Some(observation.missed.first_deadline_ns),
                last_deadline_ns: Some(observation.missed.last_deadline_ns),
                loss_observed_ns: observation.observed_ns,
                boundary_refs: Vec::new(),
                boundary_overflow_count: 0,
                boundary_overflow_digest: None,
            })
            .map_err(|error| TelemetryArchiveOwnerError::LossLedger(error.to_string()))?;
        let mut counters = self.counters.borrow_mut();
        counters.missed_ticks = next_missed_ticks;
        counters.missed_ranges = next_missed_ranges;
        Ok(())
    }

    fn record_global_loss(
        &self,
        kind: LossKindV1,
        observed_ns: i64,
    ) -> Result<(), TelemetryArchiveOwnerError> {
        if !matches!(
            kind,
            LossKindV1::WriterFailed | LossKindV1::ShutdownAbandoned
        ) {
            return Err(TelemetryArchiveOwnerError::InvalidIssuedLoss);
        }
        let next_counter = {
            let counters = self.counters.borrow();
            let current = if kind == LossKindV1::WriterFailed {
                counters.writer_failed
            } else {
                counters.shutdown_abandoned
            };
            checked_increment(current)?
        };
        self.loss_ledger
            .borrow_mut()
            .record_unsequenced(UnsequencedLossV1 {
                source_id: None,
                count: 1,
                loss_kind: kind,
                reason: kind.reason(),
                first_source_record_seq: None,
                last_source_record_seq: None,
                first_request_attempt_seq: None,
                last_request_attempt_seq: None,
                first_tick: None,
                last_tick: None,
                first_deadline_ns: None,
                last_deadline_ns: None,
                loss_observed_ns: observed_ns,
                boundary_refs: Vec::new(),
                boundary_overflow_count: 0,
                boundary_overflow_digest: None,
            })
            .map_err(|error| TelemetryArchiveOwnerError::LossLedger(error.to_string()))?;
        let mut counters = self.counters.borrow_mut();
        let counter = if kind == LossKindV1::WriterFailed {
            &mut counters.writer_failed
        } else {
            &mut counters.shutdown_abandoned
        };
        *counter = next_counter;
        Ok(())
    }

    fn fail_writer(&self, message: impl Into<String>) {
        self.writer_alive.set(false);
        self.admission_open.set(false);
        let mut first_failure = self.first_failure.borrow_mut();
        if first_failure.is_none() {
            *first_failure = Some(message.into());
        }
    }

    fn reserve(
        &self,
        class: ArchiveBudgetClass,
    ) -> Result<ArchiveProjectionLease, AdmissionRejection> {
        let footprint = match class {
            ArchiveBudgetClass::Ordinary => self.ordinary_projection_footprint,
            ArchiveBudgetClass::Control => self.control_projection_footprint,
        };
        Arc::clone(&self.budget_authority).try_reserve(
            self.admission_policy.as_ref(),
            class,
            footprint,
        )
    }
}

impl Debug for AttachedOwnerShared {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AttachedOwnerShared")
            .field("archive_id", &self.archive_id)
            .field("session_id", &self.session_id)
            .field("admission_open", &self.admission_open.get())
            .field("writer_alive", &self.writer_alive.get())
            .field("shutdown_deadline_ns", &self.shutdown_deadline_ns.get())
            .field("source_next", &self.source_next.borrow())
            .field("counters", &self.counters.borrow())
            .field("first_failure", &self.first_failure.borrow())
            .field("budget", &self.budget_authority.snapshot())
            .finish_non_exhaustive()
    }
}

/// Cloneable source-facing handle to one owner task.
#[derive(Clone)]
pub struct TelemetryArchiveOwnerHandle {
    data_commands: mpsc::Sender<OwnerCommand>,
    control_commands: mpsc::Sender<OwnerCommand>,
    shared: Rc<AttachedOwnerShared>,
}

impl Debug for TelemetryArchiveOwnerHandle {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TelemetryArchiveOwnerHandle")
            .field("data_closed", &self.data_commands.is_closed())
            .field("data_capacity", &self.data_commands.capacity())
            .field("control_closed", &self.control_commands.is_closed())
            .field("control_capacity", &self.control_commands.capacity())
            .field("shared", &self.shared)
            .finish()
    }
}

#[async_trait(?Send)]
impl ArchiveAttemptOwner for TelemetryArchiveOwnerHandle {
    async fn observe_attempt(
        &self,
        observation: ArchiveAttemptObservation,
    ) -> Result<(), ArchiveOwnerError> {
        if !self.shared.admission_open.get() || !self.shared.writer_alive.get() {
            return Err(owner_error("telemetry archive attempt admission is closed"));
        }
        let lease = self
            .shared
            .reserve(ArchiveBudgetClass::Ordinary)
            .map_err(|error| owner_error(error.to_string()))?;
        let (response, receiver) = oneshot::channel();
        self.data_commands
            .send(OwnerCommand::Attempt {
                observation,
                attached: false,
                lease,
                receipt_response: Some(response),
                terminal_response: None,
            })
            .await
            .map_err(|_| owner_error("telemetry archive owner stopped before attempt admission"))?;
        receiver
            .await
            .map_err(|_| owner_error("telemetry archive owner stopped before attempt completion"))?
            .map(|_| ())
            .map_err(|error| owner_error(error.to_string()))
    }

    async fn observe_missed(
        &self,
        observation: ArchiveMissedObservation,
    ) -> Result<(), ArchiveOwnerError> {
        if !self.shared.admission_open.get() || !self.shared.writer_alive.get() {
            return Err(owner_error("telemetry archive loss admission is closed"));
        }
        let lease = self
            .shared
            .reserve(ArchiveBudgetClass::Control)
            .map_err(|error| owner_error(error.to_string()))?;
        let (response, receiver) = oneshot::channel();
        self.control_commands
            .send(OwnerCommand::Missed {
                observation,
                lease,
                response: Some(response),
            })
            .await
            .map_err(|_| owner_error("telemetry archive owner stopped before loss admission"))?;
        receiver
            .await
            .map_err(|_| owner_error("telemetry archive owner stopped before loss completion"))?
            .map_err(|error| owner_error(error.to_string()))
    }
}

impl AttachedArchiveAttemptOwner for TelemetryArchiveOwnerHandle {
    fn try_observe_attempt(
        &self,
        observation: ArchiveAttemptObservation,
    ) -> Result<AttachedAttemptAdmission, AdmissionRejection> {
        if !self.shared.admission_open.get() || !self.shared.writer_alive.get() {
            return Err(AdmissionRejection::Closed);
        }
        let source_id = observation.decoded.facts.source_id.clone();
        let source_record_seq = observation.decoded.facts.source_record_seq;
        if let Err(error) = self
            .shared
            .check_source_sequence(&source_id, source_record_seq)
        {
            self.shared.fail_writer(error.to_string());
            return Err(AdmissionRejection::Closed);
        }
        if source_record_seq == u64::MAX {
            self.shared
                .fail_writer(TelemetryArchiveOwnerError::SequenceOverflow.to_string());
            return Err(AdmissionRejection::Closed);
        }
        let lease = self.shared.reserve(ArchiveBudgetClass::Ordinary)?;
        let boundary = observation.projection_context.reason == ScrapeReasonV1::Boundary;
        let (terminal_response, boundary_terminal) = if boundary {
            let (response, receiver) = oneshot::channel::<
                Result<TelemetryAttemptDisposition, TelemetryArchiveOwnerError>,
            >();
            let future = Box::pin(async move {
                receiver
                    .await
                    .map_err(|_| {
                        owner_error(
                            "telemetry archive owner stopped before boundary terminalization",
                        )
                    })?
                    .map_err(|error| owner_error(error.to_string()))
            });
            (
                Some(response),
                Some(future as crate::telemetry_pipeline::AttachedAttemptTerminalFuture),
            )
        } else {
            (None, None)
        };
        self.data_commands
            .try_send(OwnerCommand::Attempt {
                observation,
                attached: true,
                lease,
                receipt_response: None,
                terminal_response,
            })
            .map_err(|error| match error {
                mpsc::error::TrySendError::Full(_) => AdmissionRejection::Capacity,
                mpsc::error::TrySendError::Closed(_) => AdmissionRejection::Closed,
            })?;
        self.shared
            .commit_source_sequence(&source_id, source_record_seq)
            .expect("attached source sequence was validated before nonblocking admission");
        Ok(AttachedAttemptAdmission { boundary_terminal })
    }

    fn record_visible_loss(
        &self,
        observation: ArchiveIssuedLossObservation,
    ) -> Result<(), ArchiveOwnerError> {
        self.shared
            .record_issued_loss(observation, true)
            .map_err(|error| {
                self.shared.fail_writer(error.to_string());
                owner_error(error.to_string())
            })
    }

    fn record_missed(
        &self,
        observation: ArchiveMissedObservation,
    ) -> Result<(), ArchiveOwnerError> {
        self.shared
            .record_missed_loss(observation)
            .map_err(|error| {
                self.shared.fail_writer(error.to_string());
                owner_error(error.to_string())
            })
    }
}

impl AttachedLifecycleIngress for TelemetryArchiveOwnerHandle {
    fn try_observe_lifecycle(
        &self,
        observation: ArchiveLifecycleObservation,
    ) -> Result<(), AttachedTelemetryError> {
        if !self.shared.admission_open.get() || !self.shared.writer_alive.get() {
            return Err(AttachedTelemetryError::Component(
                "telemetry archive lifecycle admission is closed".to_owned(),
            ));
        }
        let lease = self
            .shared
            .reserve(ArchiveBudgetClass::Control)
            .map_err(|error| AttachedTelemetryError::Component(error.to_string()))?;
        self.control_commands
            .try_send(OwnerCommand::Lifecycle {
                observation,
                lease,
                response: None,
            })
            .map_err(|error| {
                AttachedTelemetryError::Component(format!(
                    "reserved telemetry lifecycle lane rejected marker: {error}"
                ))
            })
    }
}

/// Lifecycle handle retained by the prepared workload operation.
pub struct RunningTelemetryArchiveOwner {
    data_commands: mpsc::Sender<OwnerCommand>,
    control_commands: mpsc::Sender<OwnerCommand>,
    shared: Rc<AttachedOwnerShared>,
    task: Option<JoinHandle<Result<OwnerTaskExit, TelemetryArchiveOwnerError>>>,
}

impl Debug for RunningTelemetryArchiveOwner {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RunningTelemetryArchiveOwner")
            .field("data_closed", &self.data_commands.is_closed())
            .field("control_closed", &self.control_commands.is_closed())
            .field("shared", &self.shared)
            .field("task_present", &self.task.is_some())
            .finish()
    }
}

impl RunningTelemetryArchiveOwner {
    /// Persist one owner-sequenced lifecycle marker through receipt durability.
    pub async fn observe_lifecycle(
        &self,
        observation: ArchiveLifecycleObservation,
    ) -> Result<AppendReceipt, TelemetryArchiveOwnerError> {
        if !self.shared.admission_open.get() || !self.shared.writer_alive.get() {
            return Err(TelemetryArchiveOwnerError::WriterUnavailable);
        }
        let lease = self
            .shared
            .reserve(ArchiveBudgetClass::Control)
            .map_err(TelemetryArchiveOwnerError::Admission)?;
        let (response, receiver) = oneshot::channel();
        self.control_commands
            .send(OwnerCommand::Lifecycle {
                observation,
                lease,
                response: Some(response),
            })
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?;
        receiver
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?
    }

    /// Freeze all losses visible behind a data-lane fence, append them, and checkpoint.
    pub async fn checkpoint(&self) -> Result<CheckpointCompletion, TelemetryArchiveOwnerError> {
        let (fence_response, fence_receiver) = oneshot::channel();
        self.data_commands
            .send(OwnerCommand::Fence {
                response: fence_response,
            })
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?;
        fence_receiver
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?;
        let (response, receiver) = oneshot::channel();
        self.control_commands
            .send(OwnerCommand::Checkpoint { response })
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?;
        receiver
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?
    }

    /// Finalize after every source driver has stopped and drained.
    pub async fn finalize(
        mut self,
        reason: TerminationReason,
    ) -> Result<TelemetryArchiveOwnerFinalization, TelemetryArchiveOwnerError> {
        self.finalize_inner(reason, None).await
    }

    /// Finalize while converting accepted work left after an expired deadline to loss.
    pub async fn finalize_before(
        mut self,
        reason: TerminationReason,
        shutdown_deadline_ns: i64,
    ) -> Result<TelemetryArchiveOwnerFinalization, TelemetryArchiveOwnerError> {
        self.finalize_inner(reason, Some(shutdown_deadline_ns))
            .await
    }

    async fn finalize_inner(
        &mut self,
        reason: TerminationReason,
        shutdown_deadline_ns: Option<i64>,
    ) -> Result<TelemetryArchiveOwnerFinalization, TelemetryArchiveOwnerError> {
        self.shared.admission_open.set(false);
        self.shared.shutdown_deadline_ns.set(shutdown_deadline_ns);
        let (fence_response, fence_receiver) = oneshot::channel();
        self.data_commands
            .send(OwnerCommand::Fence {
                response: fence_response,
            })
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?;
        fence_receiver
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?;
        let (response, receiver) = oneshot::channel();
        self.control_commands
            .send(OwnerCommand::Finalize { reason, response })
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)?;
        let completion = receiver
            .await
            .map_err(|_| TelemetryArchiveOwnerError::TaskStopped)??;
        let task = self
            .task
            .take()
            .ok_or(TelemetryArchiveOwnerError::AlreadyJoined)?;
        let exit = task
            .await
            .map_err(|error| TelemetryArchiveOwnerError::Join(error.to_string()))??;
        Ok(TelemetryArchiveOwnerFinalization {
            completion,
            summary: exit.summary,
            repository: exit.repository,
        })
    }
}

impl Drop for RunningTelemetryArchiveOwner {
    fn drop(&mut self) {
        self.shared.admission_open.set(false);
        self.shared.budget_authority.close_admission();
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

/// Owner counters and final sink completion.
#[derive(Debug)]
pub struct TelemetryArchiveOwnerFinalization {
    /// Verified final local checkpoint and sealed WAL.
    pub completion: Option<FinalizeCompletion>,
    /// Terminal owner counters.
    pub summary: TelemetryArchiveOwnerSummary,
    /// Still-locked local authority transferred directly to remote publication.
    pub repository: Option<LocalArchiveRepository>,
}

/// Fixed-size health facts retained for report assembly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TelemetryArchiveOwnerSummary {
    /// Frames durably appended and receipt-observed.
    pub durable_frames: u64,
    /// Durable local receipt events.
    pub receipts: u64,
    /// Cadence targets compacted into missed ranges.
    pub missed_ticks: u64,
    /// Compact missed ranges observed by the owner.
    pub missed_ranges: u64,
    /// Last acknowledged terminal frame.
    pub last_frame_id: Option<FrameId>,
    /// Bounded exact/saturation view of every owner-observed loss.
    pub loss_ledger: Option<LossLedgerViewV1>,
    /// Whether the durability writer remained usable through finalization.
    pub writer_alive: bool,
    /// Native-delivered attempts rejected by bounded archive admission.
    pub archive_rejected: u64,
    /// Accepted attempts whose archive projection failed.
    pub projection_failed: u64,
    /// Source/global work affected after the writer failed.
    pub writer_failed: u64,
    /// Accepted work abandoned at an explicit shutdown deadline.
    pub shutdown_abandoned: u64,
    /// First bounded archive degradation detail, absent for a healthy writer.
    pub first_failure: Option<String>,
    /// Terminal quota, reserve, outstanding-lease, and high-water accounting.
    pub budget: Option<ArchiveSpoolBudgetSnapshot>,
}

impl Default for TelemetryArchiveOwnerSummary {
    fn default() -> Self {
        Self {
            durable_frames: 0,
            receipts: 0,
            missed_ticks: 0,
            missed_ranges: 0,
            last_frame_id: None,
            loss_ledger: None,
            writer_alive: true,
            archive_rejected: 0,
            projection_failed: 0,
            writer_failed: 0,
            shutdown_abandoned: 0,
            first_failure: None,
            budget: None,
        }
    }
}

/// Owner-stamped lifecycle marker before archive/session/sequence identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveLifecycleObservation {
    /// Closed lifecycle class.
    pub kind: LifecycleMarkerKindV1,
    /// Injected-Clock instant observed on the owning LocalSet.
    pub observed_ns: i64,
    /// Optional benchmark run identity.
    pub run_id: Option<String>,
    /// Optional phase identity.
    pub phase_id: Option<String>,
    /// Optional physical source identity.
    pub source_id: Option<String>,
    /// Optional exact phase state.
    pub phase_state: Option<LifecyclePhaseStateV1>,
    /// Optional terminal lifecycle reason.
    pub completion_reason: Option<LifecycleCompletionReasonV1>,
    /// Optional complete phase-boundary join key.
    pub boundary: Option<BoundaryReference>,
    /// Optional phase start instant.
    pub phase_start_ns: Option<i64>,
    /// Optional sending-complete instant.
    pub sent_end_ns: Option<i64>,
    /// Optional request-complete instant.
    pub requests_end_ns: Option<i64>,
    /// Optional source attribute-epoch identity.
    pub attribute_epoch_id: Option<aiperf_telemetry_archive::Digest>,
    /// Sanitized marker attributes.
    pub attributes: BTreeMap<String, String>,
}

impl ArchiveLifecycleObservation {
    /// Construct the first durable marker of a new collection session.
    #[must_use]
    pub fn session_started(observed_ns: i64) -> Self {
        Self {
            kind: LifecycleMarkerKindV1::SessionStarted,
            observed_ns,
            run_id: None,
            phase_id: None,
            source_id: None,
            phase_state: None,
            completion_reason: None,
            boundary: None,
            phase_start_ns: None,
            sent_end_ns: None,
            requests_end_ns: None,
            attribute_epoch_id: None,
            attributes: BTreeMap::new(),
        }
    }

    /// Construct the final session marker before the owner closes admission.
    #[must_use]
    pub fn session_stopped(
        observed_ns: i64,
        completion_reason: LifecycleCompletionReasonV1,
    ) -> Self {
        Self {
            kind: LifecycleMarkerKindV1::SessionStopped,
            completion_reason: Some(completion_reason),
            ..Self::session_started(observed_ns)
        }
    }
}

/// Start one owner after static validation and sink preparation.
pub async fn start_telemetry_archive_owner(
    mut config: TelemetryArchiveOwnerConfig,
) -> Result<
    (
        Rc<TelemetryArchiveOwnerHandle>,
        RunningTelemetryArchiveOwner,
    ),
    TelemetryArchiveOwnerError,
> {
    if config.queue_capacity == 0 {
        return Err(TelemetryArchiveOwnerError::ZeroQueueCapacity);
    }
    if config.control_queue_capacity == 0 {
        return Err(TelemetryArchiveOwnerError::ZeroControlQueueCapacity);
    }
    let recovered = config
        .sink
        .recover()
        .await
        .map_err(TelemetryArchiveOwnerError::Sink)?;
    if recovered.next_record_seq != config.sequencer.next_record_seq() {
        return Err(TelemetryArchiveOwnerError::RecoverySequence {
            recovered: recovered.next_record_seq,
            sequencer: config.sequencer.next_record_seq(),
        });
    }
    let TelemetryArchiveOwnerConfig {
        archive_id,
        session_id,
        sequencer,
        codec,
        control_codec,
        sink,
        clock,
        receipt_epoch,
        receipt_epoch_registered,
        next_receipt_seq,
        loss_ledger,
        queue_capacity,
        control_queue_capacity,
        best_effort,
        attached,
        budget_authority,
        admission_policy,
        ordinary_projection_footprint,
        control_projection_footprint,
    } = config;
    let shared = Rc::new(AttachedOwnerShared::new(
        archive_id,
        session_id,
        loss_ledger,
        budget_authority,
        admission_policy,
        ordinary_projection_footprint,
        control_projection_footprint,
    ));
    let runtime = OwnerRuntimeConfig {
        archive_id,
        session_id,
        sequencer,
        codec,
        control_codec,
        sink,
        clock,
        receipt_epoch,
        receipt_epoch_registered,
        next_receipt_seq,
        best_effort,
        attached,
    };
    let (data_commands, data_receiver) = mpsc::channel(queue_capacity);
    let (control_commands, control_receiver) = mpsc::channel(control_queue_capacity);
    let handle = Rc::new(TelemetryArchiveOwnerHandle {
        data_commands: data_commands.clone(),
        control_commands: control_commands.clone(),
        shared: shared.clone(),
    });
    let task = tokio::task::spawn_local(run_owner(
        runtime,
        shared.clone(),
        data_receiver,
        control_receiver,
    ));
    Ok((
        handle,
        RunningTelemetryArchiveOwner {
            data_commands,
            control_commands,
            shared,
            task: Some(task),
        },
    ))
}

struct OwnerRuntimeConfig {
    archive_id: ArchiveId,
    session_id: SessionId,
    sequencer: ArchiveFrameSequencerV1,
    codec: SourceFrameCodecV1,
    control_codec: ControlFrameCodecV1,
    sink: Box<dyn ArchiveSink>,
    clock: Rc<dyn Clock>,
    receipt_epoch: ReceiptObserverEpochV1,
    receipt_epoch_registered: bool,
    next_receipt_seq: u64,
    best_effort: bool,
    attached: bool,
}

enum OwnerCommand {
    Attempt {
        observation: ArchiveAttemptObservation,
        attached: bool,
        lease: ArchiveProjectionLease,
        receipt_response:
            Option<oneshot::Sender<Result<AppendReceipt, TelemetryArchiveOwnerError>>>,
        terminal_response: Option<
            oneshot::Sender<Result<TelemetryAttemptDisposition, TelemetryArchiveOwnerError>>,
        >,
    },
    Missed {
        observation: ArchiveMissedObservation,
        lease: ArchiveProjectionLease,
        response: Option<oneshot::Sender<Result<(), TelemetryArchiveOwnerError>>>,
    },
    Lifecycle {
        observation: ArchiveLifecycleObservation,
        lease: ArchiveProjectionLease,
        response: Option<oneshot::Sender<Result<AppendReceipt, TelemetryArchiveOwnerError>>>,
    },
    Fence {
        response: oneshot::Sender<()>,
    },
    Checkpoint {
        response: oneshot::Sender<Result<CheckpointCompletion, TelemetryArchiveOwnerError>>,
    },
    Finalize {
        reason: TerminationReason,
        response: oneshot::Sender<Result<Option<FinalizeCompletion>, TelemetryArchiveOwnerError>>,
    },
}

struct OwnerTaskExit {
    summary: TelemetryArchiveOwnerSummary,
    repository: Option<LocalArchiveRepository>,
}

async fn run_owner(
    mut config: OwnerRuntimeConfig,
    shared: Rc<AttachedOwnerShared>,
    mut data_commands: mpsc::Receiver<OwnerCommand>,
    mut control_commands: mpsc::Receiver<OwnerCommand>,
) -> Result<OwnerTaskExit, TelemetryArchiveOwnerError> {
    let mut summary = TelemetryArchiveOwnerSummary::default();
    let receipt_epoch = config.receipt_epoch.clone();
    let receipt_epoch_id = receipt_epoch.observer_epoch_id;
    let mut epoch_pending = (!config.receipt_epoch_registered).then_some(receipt_epoch.clone());
    let mut receipt_seq = config.next_receipt_seq;
    let mut loss_seq = 0_u64;
    let mut processed_source_next: BTreeMap<String, u64> = shared
        .loss_ledger
        .borrow()
        .prepared_source_ids()
        .map(|source_id| (source_id.to_owned(), 0))
        .collect();
    loop {
        let command = tokio::select! {
            biased;
            command = control_commands.recv() => command,
            command = data_commands.recv() => command,
        };
        let Some(command) = command else {
            break;
        };
        match command {
            OwnerCommand::Attempt {
                observation,
                attached,
                lease,
                receipt_response,
                terminal_response,
            } => {
                process_owner_attempt(
                    &mut config,
                    &shared,
                    &mut summary,
                    &mut processed_source_next,
                    receipt_epoch_id,
                    &mut epoch_pending,
                    &mut receipt_seq,
                    observation,
                    attached,
                    lease,
                    receipt_response,
                    terminal_response,
                )
                .await?;
            }
            OwnerCommand::Missed {
                observation,
                lease,
                response,
            } => {
                let result = shared.record_missed_loss(observation);
                if result.is_ok()
                    && shared.writer_alive.get()
                    && let Err(error) = flush_pending_losses(
                        &mut config,
                        &shared,
                        &mut summary,
                        receipt_epoch_id,
                        &mut epoch_pending,
                        &mut receipt_seq,
                        &mut loss_seq,
                        Some(lease),
                    )
                    .await
                {
                    mark_writer_failed(&config, &shared, &mut summary, &error)?;
                    if let Some(response) = response {
                        let _ = response.send(Err(error));
                    }
                    if !degrades(&config) {
                        return Err(TelemetryArchiveOwnerError::FailStopped(
                            "standalone missed-loss durability failed".to_owned(),
                        ));
                    }
                    continue;
                }
                if let Some(response) = response {
                    let _ = response.send(result);
                }
            }
            OwnerCommand::Lifecycle {
                observation,
                lease,
                response,
            } => {
                if !shared.writer_alive.get() {
                    if let Some(response) = response {
                        let _ = response.send(Err(TelemetryArchiveOwnerError::WriterUnavailable));
                    }
                    continue;
                }
                let result = append_lifecycle(
                    &mut config.sequencer,
                    &config.control_codec,
                    config.sink.as_mut(),
                    config.clock.as_ref(),
                    config.archive_id,
                    config.session_id,
                    &receipt_epoch,
                    receipt_epoch_id,
                    &mut epoch_pending,
                    &mut receipt_seq,
                    observation,
                )
                .await;
                let response_result = match result {
                    Ok((receipt, frame_id)) => {
                        summary.durable_frames = checked_increment(summary.durable_frames)?;
                        summary.receipts = checked_increment(summary.receipts)?;
                        summary.last_frame_id = Some(frame_id);
                        lease.commit();
                        Ok(receipt)
                    }
                    Err(error) => Err(error),
                };
                let fatal = response_result.as_ref().err().map(ToString::to_string);
                if let Some(message) = fatal {
                    mark_writer_failed(
                        &config,
                        &shared,
                        &mut summary,
                        &TelemetryArchiveOwnerError::FailStopped(message.clone()),
                    )?;
                    if let Some(response) = response {
                        let _ = response.send(if config.best_effort {
                            Err(TelemetryArchiveOwnerError::WriterUnavailable)
                        } else {
                            response_result
                        });
                    }
                    if !degrades(&config) {
                        return Err(TelemetryArchiveOwnerError::FailStopped(message));
                    }
                } else if let Some(response) = response {
                    let _ = response.send(response_result);
                }
            }
            OwnerCommand::Fence { response } => {
                let _ = response.send(());
            }
            OwnerCommand::Checkpoint { response } => {
                let result = checkpoint_owner(
                    &mut config,
                    &shared,
                    &mut summary,
                    receipt_epoch_id,
                    &mut epoch_pending,
                    &mut receipt_seq,
                    &mut loss_seq,
                )
                .await;
                if let Err(error) = &result {
                    mark_writer_failed(&config, &shared, &mut summary, error)?;
                }
                let fatal = result.as_ref().err().map(ToString::to_string);
                let _ = response.send(result);
                if let Some(message) = fatal
                    && !degrades(&config)
                {
                    return Err(TelemetryArchiveOwnerError::FailStopped(message));
                }
            }
            OwnerCommand::Finalize { reason, response } => {
                if let Err(error) = terminalize_trailing_source_losses(
                    &mut config.sequencer,
                    &mut processed_source_next,
                    &shared.source_next.borrow(),
                ) {
                    mark_writer_failed(&config, &shared, &mut summary, &error)?;
                    if !degrades(&config) {
                        let message = error.to_string();
                        let _ = response.send(Err(error));
                        return Err(TelemetryArchiveOwnerError::FailStopped(message));
                    }
                }
                sync_summary(&shared, &mut summary);
                if shared.writer_alive.get()
                    && let Err(error) = flush_pending_losses(
                        &mut config,
                        &shared,
                        &mut summary,
                        receipt_epoch_id,
                        &mut epoch_pending,
                        &mut receipt_seq,
                        &mut loss_seq,
                        None,
                    )
                    .await
                {
                    mark_writer_failed(&config, &shared, &mut summary, &error)?;
                    if !degrades(&config) {
                        let message = error.to_string();
                        let _ = response.send(Err(error));
                        return Err(TelemetryArchiveOwnerError::FailStopped(message));
                    }
                }
                let finalization_permit = match begin_finalization_budget(&config, &shared) {
                    Ok(permit) => Some(permit),
                    Err(error) => {
                        mark_writer_failed(&config, &shared, &mut summary, &error)?;
                        if !degrades(&config) {
                            let message = error.to_string();
                            let _ = response.send(Err(error));
                            return Err(TelemetryArchiveOwnerError::FailStopped(message));
                        }
                        None
                    }
                };
                let (completion, repository) =
                    if shared.writer_alive.get() && finalization_permit.is_some() {
                        match config.sink.finalize(reason).await {
                            Ok(completion) => {
                                let repository = config
                                    .sink
                                    .into_local_repository()
                                    .map_err(TelemetryArchiveOwnerError::Sink)?;
                                (Some(completion), repository)
                            }
                            Err(error) if degrades(&config) => {
                                mark_writer_failed(
                                    &config,
                                    &shared,
                                    &mut summary,
                                    &TelemetryArchiveOwnerError::Sink(error),
                                )?;
                                (None, None)
                            }
                            Err(error) => {
                                let error = TelemetryArchiveOwnerError::Sink(error);
                                let message = error.to_string();
                                let _ = response.send(Err(error));
                                return Err(TelemetryArchiveOwnerError::FailStopped(message));
                            }
                        }
                    } else {
                        (None, None)
                    };
                sync_summary(&shared, &mut summary);
                let _ = response.send(Ok(completion));
                return Ok(OwnerTaskExit {
                    summary,
                    repository,
                });
            }
        }
    }
    Err(TelemetryArchiveOwnerError::FinalizeNotRequested)
}

#[allow(clippy::too_many_arguments)]
async fn process_owner_attempt(
    config: &mut OwnerRuntimeConfig,
    shared: &AttachedOwnerShared,
    summary: &mut TelemetryArchiveOwnerSummary,
    processed_source_next: &mut BTreeMap<String, u64>,
    receipt_epoch_id: ReceiptObserverEpochId,
    epoch_pending: &mut Option<ReceiptObserverEpochV1>,
    receipt_seq: &mut u64,
    observation: ArchiveAttemptObservation,
    attached: bool,
    lease: ArchiveProjectionLease,
    receipt_response: Option<oneshot::Sender<Result<AppendReceipt, TelemetryArchiveOwnerError>>>,
    terminal_response: Option<
        oneshot::Sender<Result<TelemetryAttemptDisposition, TelemetryArchiveOwnerError>>,
    >,
) -> Result<(), TelemetryArchiveOwnerError> {
    if attached != config.attached {
        return Err(TelemetryArchiveOwnerError::Invariant(
            "attempt admission mode differs from prepared owner mode".to_owned(),
        ));
    }
    let source_id = observation.decoded.facts.source_id.clone();
    let source_record_seq = observation.decoded.facts.source_record_seq;
    let request_attempt_seq = observation.decoded.facts.request_attempt_seq;
    let boundary_refs = observation.projection_context.boundary_refs.clone();
    if attached {
        catch_up_rejected_sources(
            &mut config.sequencer,
            processed_source_next,
            &source_id,
            source_record_seq,
        )?;
    }

    let shutdown_abandoned = attached
        && shared
            .shutdown_deadline_ns
            .get()
            .is_some_and(|deadline| config.clock.now_ns() >= deadline);
    if shutdown_abandoned || !shared.writer_alive.get() {
        if attached {
            terminalize_current_source_loss(
                &mut config.sequencer,
                processed_source_next,
                &source_id,
                source_record_seq,
            )?;
        }
        let kind = if shutdown_abandoned {
            LossKindV1::ShutdownAbandoned
        } else {
            LossKindV1::WriterFailed
        };
        shared.record_issued_loss(
            ArchiveIssuedLossObservation {
                source_id,
                source_record_seq,
                request_attempt_seq,
                loss_kind: kind,
                reason: kind.reason(),
                observed_ns: config.clock.now_ns(),
                boundary_refs,
            },
            false,
        )?;
        if let Some(response) = receipt_response {
            let _ = response.send(Err(TelemetryArchiveOwnerError::WriterUnavailable));
        }
        if let Some(response) = terminal_response {
            let _ = response.send(Ok(TelemetryAttemptDisposition::Loss {
                kind,
                reason: kind.reason(),
            }));
        }
        return Ok(());
    }

    let result = append_observed_attempt(
        &mut config.sequencer,
        &config.codec,
        config.sink.as_mut(),
        config.clock.as_ref(),
        config.archive_id,
        config.session_id,
        receipt_epoch_id,
        epoch_pending,
        receipt_seq,
        observation,
    )
    .await;
    match result {
        Ok((receipt, frame_id)) => {
            if attached {
                commit_processed_source(processed_source_next, &source_id, source_record_seq)?;
            }
            summary.durable_frames = checked_increment(summary.durable_frames)?;
            summary.receipts = checked_increment(summary.receipts)?;
            summary.last_frame_id = Some(frame_id);
            lease.commit();
            if let Some(response) = receipt_response {
                let _ = response.send(Ok(receipt));
            }
            if let Some(response) = terminal_response {
                let _ = response.send(Ok(TelemetryAttemptDisposition::Attempt));
            }
            Ok(())
        }
        Err(error @ TelemetryArchiveOwnerError::Sequencing(_)) if attached => {
            terminalize_current_source_loss(
                &mut config.sequencer,
                processed_source_next,
                &source_id,
                source_record_seq,
            )?;
            shared.record_issued_loss(
                ArchiveIssuedLossObservation {
                    source_id,
                    source_record_seq,
                    request_attempt_seq,
                    loss_kind: LossKindV1::ProjectionFailed,
                    reason: LossReasonV1::ProjectionError,
                    observed_ns: config.clock.now_ns(),
                    boundary_refs,
                },
                false,
            )?;
            if let Some(response) = receipt_response {
                let _ = response.send(Err(error));
            }
            if let Some(response) = terminal_response {
                let _ = response.send(Ok(TelemetryAttemptDisposition::Loss {
                    kind: LossKindV1::ProjectionFailed,
                    reason: LossReasonV1::ProjectionError,
                }));
            }
            Ok(())
        }
        Err(error @ TelemetryArchiveOwnerError::Projection(_)) if attached => {
            commit_processed_source(processed_source_next, &source_id, source_record_seq)?;
            shared.record_issued_loss(
                ArchiveIssuedLossObservation {
                    source_id,
                    source_record_seq,
                    request_attempt_seq,
                    loss_kind: LossKindV1::ProjectionFailed,
                    reason: LossReasonV1::ProjectionError,
                    observed_ns: config.clock.now_ns(),
                    boundary_refs,
                },
                false,
            )?;
            mark_writer_failed(config, shared, summary, &error)?;
            if let Some(response) = receipt_response {
                let _ = response.send(Err(TelemetryArchiveOwnerError::FailStopped(
                    error.to_string(),
                )));
            }
            if let Some(response) = terminal_response {
                let _ = response.send(Ok(TelemetryAttemptDisposition::Loss {
                    kind: LossKindV1::ProjectionFailed,
                    reason: LossReasonV1::ProjectionError,
                }));
            }
            Ok(())
        }
        Err(error) => {
            if attached {
                commit_processed_source(processed_source_next, &source_id, source_record_seq)?;
                shared.record_issued_loss(
                    ArchiveIssuedLossObservation {
                        source_id,
                        source_record_seq,
                        request_attempt_seq,
                        loss_kind: LossKindV1::WriterFailed,
                        reason: LossReasonV1::WriterError,
                        observed_ns: config.clock.now_ns(),
                        boundary_refs,
                    },
                    false,
                )?;
            }
            mark_writer_failed(config, shared, summary, &error)?;
            if let Some(response) = receipt_response {
                let _ = response.send(Err(TelemetryArchiveOwnerError::FailStopped(
                    error.to_string(),
                )));
            }
            if let Some(response) = terminal_response {
                let _ = response.send(Ok(TelemetryAttemptDisposition::Loss {
                    kind: LossKindV1::WriterFailed,
                    reason: LossReasonV1::WriterError,
                }));
            }
            if degrades(config) {
                Ok(())
            } else {
                Err(TelemetryArchiveOwnerError::FailStopped(error.to_string()))
            }
        }
    }
}

fn catch_up_rejected_sources(
    sequencer: &mut ArchiveFrameSequencerV1,
    processed_source_next: &mut BTreeMap<String, u64>,
    source_id: &str,
    target: u64,
) -> Result<(), TelemetryArchiveOwnerError> {
    let next = processed_source_next
        .get_mut(source_id)
        .ok_or_else(|| TelemetryArchiveOwnerError::UnknownLossSource(source_id.to_owned()))?;
    if *next > target {
        return Err(TelemetryArchiveOwnerError::SourceAdmissionSequence {
            source_id: source_id.to_owned(),
            expected: *next,
            actual: target,
        });
    }
    while *next < target {
        sequencer
            .terminalize_source_loss(source_id, *next)
            .map_err(|error| TelemetryArchiveOwnerError::Sequencing(error.to_string()))?;
        *next = next
            .checked_add(1)
            .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
    }
    Ok(())
}

fn terminalize_current_source_loss(
    sequencer: &mut ArchiveFrameSequencerV1,
    processed_source_next: &mut BTreeMap<String, u64>,
    source_id: &str,
    source_record_seq: u64,
) -> Result<(), TelemetryArchiveOwnerError> {
    let next = processed_source_next
        .get_mut(source_id)
        .ok_or_else(|| TelemetryArchiveOwnerError::UnknownLossSource(source_id.to_owned()))?;
    if *next != source_record_seq {
        return Err(TelemetryArchiveOwnerError::SourceAdmissionSequence {
            source_id: source_id.to_owned(),
            expected: *next,
            actual: source_record_seq,
        });
    }
    sequencer
        .terminalize_source_loss(source_id, source_record_seq)
        .map_err(|error| TelemetryArchiveOwnerError::Sequencing(error.to_string()))?;
    *next = source_record_seq
        .checked_add(1)
        .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
    Ok(())
}

fn commit_processed_source(
    processed_source_next: &mut BTreeMap<String, u64>,
    source_id: &str,
    source_record_seq: u64,
) -> Result<(), TelemetryArchiveOwnerError> {
    let next = processed_source_next
        .get_mut(source_id)
        .ok_or_else(|| TelemetryArchiveOwnerError::UnknownLossSource(source_id.to_owned()))?;
    if *next != source_record_seq {
        return Err(TelemetryArchiveOwnerError::SourceAdmissionSequence {
            source_id: source_id.to_owned(),
            expected: *next,
            actual: source_record_seq,
        });
    }
    *next = source_record_seq
        .checked_add(1)
        .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
    Ok(())
}

fn terminalize_trailing_source_losses(
    sequencer: &mut ArchiveFrameSequencerV1,
    processed_source_next: &mut BTreeMap<String, u64>,
    admitted_source_next: &BTreeMap<String, u64>,
) -> Result<(), TelemetryArchiveOwnerError> {
    for (source_id, target) in admitted_source_next {
        catch_up_rejected_sources(sequencer, processed_source_next, source_id, *target)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn flush_pending_losses(
    config: &mut OwnerRuntimeConfig,
    shared: &AttachedOwnerShared,
    summary: &mut TelemetryArchiveOwnerSummary,
    receipt_epoch_id: ReceiptObserverEpochId,
    epoch_pending: &mut Option<ReceiptObserverEpochV1>,
    receipt_seq: &mut u64,
    next_loss_seq: &mut u64,
    mut prepaid_control_lease: Option<ArchiveProjectionLease>,
) -> Result<(), TelemetryArchiveOwnerError> {
    if !shared.writer_alive.get() {
        return Err(TelemetryArchiveOwnerError::WriterUnavailable);
    }
    let plan = shared.loss_ledger.borrow().freeze_plan();
    let frame_count = u64::try_from(plan.frame_count())
        .map_err(|_| TelemetryArchiveOwnerError::SequenceOverflow)?;
    if frame_count == 0 {
        if prepaid_control_lease.is_some() {
            return Err(TelemetryArchiveOwnerError::Invariant(
                "a prepaid loss projection produced no frozen frame".to_owned(),
            ));
        }
        return Ok(());
    }
    let first_record_seq = config.sequencer.next_record_seq();
    first_record_seq
        .checked_add(frame_count)
        .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
    next_loss_seq
        .checked_add(frame_count)
        .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
    let identities: Vec<_> = (0..frame_count)
        .map(|offset| LossFrameIdentityV1 {
            record_seq: first_record_seq + offset,
            loss_seq: *next_loss_seq + offset,
        })
        .collect();
    let frozen = shared
        .loss_ledger
        .borrow_mut()
        .freeze_checkpoint(identities.iter().copied())
        .map_err(|error| TelemetryArchiveOwnerError::LossLedger(error.to_string()))?;
    for identity in &identities {
        let assigned = config
            .sequencer
            .assign_control_record_seq()
            .map_err(|error| TelemetryArchiveOwnerError::Sequencing(error.to_string()))?;
        if assigned != identity.record_seq {
            return Err(TelemetryArchiveOwnerError::Invariant(
                "loss freeze and sequencer record identities diverged".to_owned(),
            ));
        }
    }
    *next_loss_seq += frame_count;

    let mut frames =
        Vec::with_capacity(frozen.exact_ranges.len() + frozen.saturation_snapshots.len());
    for loss in frozen.exact_ranges {
        frames.push(
            config
                .control_codec
                .encode_exact_loss_frame(loss)
                .map_err(|error| TelemetryArchiveOwnerError::Projection(error.to_string()))?,
        );
    }
    for snapshot in frozen.saturation_snapshots {
        frames.push(
            config
                .control_codec
                .encode_loss_saturation_frame(snapshot)
                .map_err(|error| TelemetryArchiveOwnerError::Projection(error.to_string()))?,
        );
    }
    for frame in frames {
        let lease = match prepaid_control_lease.take() {
            Some(lease) => lease,
            None => shared
                .reserve(ArchiveBudgetClass::Control)
                .map_err(TelemetryArchiveOwnerError::Admission)?,
        };
        let (_, frame_id) = append_frame_with_receipt(
            config.sink.as_mut(),
            config.clock.as_ref(),
            config.archive_id,
            config.session_id,
            receipt_epoch_id,
            epoch_pending,
            receipt_seq,
            frame,
        )
        .await?;
        summary.durable_frames = checked_increment(summary.durable_frames)?;
        summary.receipts = checked_increment(summary.receipts)?;
        summary.last_frame_id = Some(frame_id);
        lease.commit();
    }
    if prepaid_control_lease.is_some() {
        return Err(TelemetryArchiveOwnerError::Invariant(
            "a prepaid loss projection was not consumed".to_owned(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn checkpoint_owner(
    config: &mut OwnerRuntimeConfig,
    shared: &AttachedOwnerShared,
    summary: &mut TelemetryArchiveOwnerSummary,
    receipt_epoch_id: ReceiptObserverEpochId,
    epoch_pending: &mut Option<ReceiptObserverEpochV1>,
    receipt_seq: &mut u64,
    next_loss_seq: &mut u64,
) -> Result<CheckpointCompletion, TelemetryArchiveOwnerError> {
    flush_pending_losses(
        config,
        shared,
        summary,
        receipt_epoch_id,
        epoch_pending,
        receipt_seq,
        next_loss_seq,
        None,
    )
    .await?;
    config
        .sink
        .checkpoint()
        .await
        .map_err(TelemetryArchiveOwnerError::Sink)
}

fn begin_finalization_budget(
    config: &OwnerRuntimeConfig,
    shared: &AttachedOwnerShared,
) -> Result<ArchiveFinalizationPermit, TelemetryArchiveOwnerError> {
    if let Some(repository) = config.sink.local_repository() {
        let observation = repository
            .spool()
            .budget_observation()
            .map_err(|error| TelemetryArchiveOwnerError::Budget(error.to_string()))?;
        shared
            .budget_authority
            .refresh(observation)
            .map_err(|error| TelemetryArchiveOwnerError::Budget(error.to_string()))?;
    }
    shared
        .budget_authority
        .begin_finalization()
        .map_err(|error| TelemetryArchiveOwnerError::Budget(error.to_string()))
}

fn mark_writer_failed(
    config: &OwnerRuntimeConfig,
    shared: &AttachedOwnerShared,
    summary: &mut TelemetryArchiveOwnerSummary,
    error: &TelemetryArchiveOwnerError,
) -> Result<(), TelemetryArchiveOwnerError> {
    if shared.writer_alive.get() {
        shared.record_global_loss(LossKindV1::WriterFailed, config.clock.now_ns())?;
    }
    shared.fail_writer(error.to_string());
    summary.writer_alive = false;
    Ok(())
}

fn sync_summary(shared: &AttachedOwnerShared, summary: &mut TelemetryArchiveOwnerSummary) {
    let counters = *shared.counters.borrow();
    summary.missed_ticks = counters.missed_ticks;
    summary.missed_ranges = counters.missed_ranges;
    summary.archive_rejected = counters.archive_rejected;
    summary.projection_failed = counters.projection_failed;
    summary.writer_failed = counters.writer_failed;
    summary.shutdown_abandoned = counters.shutdown_abandoned;
    summary.writer_alive = shared.writer_alive.get();
    summary
        .first_failure
        .clone_from(&shared.first_failure.borrow());
    summary.loss_ledger = Some(shared.loss_ledger.borrow().bounded_view());
    summary.budget = Some(shared.budget_authority.snapshot());
}

const fn degrades(config: &OwnerRuntimeConfig) -> bool {
    config.attached || config.best_effort
}

#[allow(clippy::too_many_arguments)]
async fn append_observed_attempt(
    sequencer: &mut ArchiveFrameSequencerV1,
    codec: &SourceFrameCodecV1,
    sink: &mut dyn ArchiveSink,
    clock: &dyn Clock,
    archive_id: ArchiveId,
    session_id: SessionId,
    receipt_epoch_id: ReceiptObserverEpochId,
    epoch_pending: &mut Option<ReceiptObserverEpochV1>,
    receipt_seq: &mut u64,
    observation: ArchiveAttemptObservation,
) -> Result<(AppendReceipt, FrameId), TelemetryArchiveOwnerError> {
    let sequenced = sequencer
        .project_attempt_with_context(
            observation.decoded,
            ArchiveFrameTimingV1 {
                parse_done_ns: observation.parse_done_ns,
                archive_enqueue_ns: observation.archive_enqueue_ns,
            },
            observation.projection_context,
        )
        .map_err(|error| TelemetryArchiveOwnerError::Sequencing(error.to_string()))?;
    let frame: ArchiveWalFrame = codec
        .encode_source_frame(sequenced)
        .map_err(|error| TelemetryArchiveOwnerError::Projection(error.to_string()))?;
    append_frame_with_receipt(
        sink,
        clock,
        archive_id,
        session_id,
        receipt_epoch_id,
        epoch_pending,
        receipt_seq,
        frame,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn append_lifecycle(
    sequencer: &mut ArchiveFrameSequencerV1,
    codec: &ControlFrameCodecV1,
    sink: &mut dyn ArchiveSink,
    clock: &dyn Clock,
    archive_id: ArchiveId,
    session_id: SessionId,
    receipt_epoch: &ReceiptObserverEpochV1,
    receipt_epoch_id: ReceiptObserverEpochId,
    epoch_pending: &mut Option<ReceiptObserverEpochV1>,
    receipt_seq: &mut u64,
    observation: ArchiveLifecycleObservation,
) -> Result<(AppendReceipt, FrameId), TelemetryArchiveOwnerError> {
    let record_seq = sequencer
        .assign_control_record_seq()
        .map_err(|error| TelemetryArchiveOwnerError::Sequencing(error.to_string()))?;
    let unix_epoch_ns = receipt_epoch
        .unix_ns_at(observation.observed_ns)
        .map_err(|error| TelemetryArchiveOwnerError::Receipt(error.to_string()))?;
    let frame = codec
        .encode_lifecycle_frame(LifecycleMarkerV1 {
            archive_id,
            session_id,
            record_seq,
            marker_seq: record_seq,
            kind: observation.kind,
            clock_ns: observation.observed_ns,
            unix_epoch_ns,
            run_id: observation.run_id,
            phase_id: observation.phase_id,
            source_id: observation.source_id,
            phase_state: observation.phase_state,
            completion_reason: observation.completion_reason,
            boundary: observation.boundary,
            phase_start_ns: observation.phase_start_ns,
            sent_end_ns: observation.sent_end_ns,
            requests_end_ns: observation.requests_end_ns,
            attribute_epoch_id: observation.attribute_epoch_id,
            attributes: observation.attributes,
        })
        .map_err(|error| TelemetryArchiveOwnerError::Projection(error.to_string()))?;
    append_frame_with_receipt(
        sink,
        clock,
        archive_id,
        session_id,
        receipt_epoch_id,
        epoch_pending,
        receipt_seq,
        frame,
    )
    .await
}

#[allow(clippy::too_many_arguments)]
async fn append_frame_with_receipt(
    sink: &mut dyn ArchiveSink,
    clock: &dyn Clock,
    archive_id: ArchiveId,
    session_id: SessionId,
    receipt_epoch_id: ReceiptObserverEpochId,
    epoch_pending: &mut Option<ReceiptObserverEpochV1>,
    receipt_seq: &mut u64,
    frame: ArchiveWalFrame,
) -> Result<(AppendReceipt, FrameId), TelemetryArchiveOwnerError> {
    let frame_id = frame.wal_frame.header().frame_id;
    let completion = sink
        .append_frame(frame)
        .await
        .map_err(TelemetryArchiveOwnerError::Sink)?;
    let target = wal_receipt_target(archive_id, session_id, completion)?;
    let event = ReceiptEventV1::new(
        target.archive_id(),
        *receipt_seq,
        target.receipt_target_id,
        receipt_epoch_id,
        ObservationKind::ResponseObserved,
        clock.now_ns(),
    );
    let receipt = sink
        .record_receipt(ReceiptEventDraft {
            observer_epoch: epoch_pending.take(),
            target,
            event,
        })
        .await
        .map_err(TelemetryArchiveOwnerError::Sink)?;
    *receipt_seq = receipt_seq
        .checked_add(1)
        .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)?;
    Ok((receipt, frame_id))
}

fn wal_receipt_target(
    archive_id: ArchiveId,
    session_id: SessionId,
    completion: DurabilityCompletion,
) -> Result<ReceiptTargetV1, TelemetryArchiveOwnerError> {
    let projection_coverage_digest = receipt_range_coverage(vec![(
        completion.first_record_seq,
        completion.projection_coverage_digest,
    )])
    .map_err(|error| TelemetryArchiveOwnerError::Receipt(error.to_string()))?;
    ReceiptTargetV1::wal_range(WalRangeTargetV1 {
        archive_id,
        session_id,
        wal_segment_id: completion.wal_segment_id,
        durable_prefix_hash: completion.durable_prefix_hash,
        first_record_seq: completion.first_record_seq,
        last_record_seq: completion.last_record_seq,
        projection_coverage_digest,
    })
    .map_err(|error| TelemetryArchiveOwnerError::Receipt(error.to_string()))
}

fn checked_increment(value: u64) -> Result<u64, TelemetryArchiveOwnerError> {
    value
        .checked_add(1)
        .ok_or(TelemetryArchiveOwnerError::SequenceOverflow)
}

fn owner_error(message: impl Into<String>) -> ArchiveOwnerError {
    ArchiveOwnerError {
        message: message.into(),
    }
}

/// Owner composition, sequencing, receipt, or lifecycle failure.
#[derive(Debug)]
pub enum TelemetryArchiveOwnerError {
    /// Owner command queue capacity must be positive.
    ZeroQueueCapacity,
    /// Reserved control command capacity must be positive.
    ZeroControlQueueCapacity,
    /// Recovered WAL sequence and fresh sequencing authority disagreed.
    RecoverySequence {
        /// First sequence absent from recovered WAL.
        recovered: u64,
        /// Sequencer's next sequence.
        sequencer: u64,
    },
    /// Source-frame sequencing failed before WAL construction.
    Sequencing(String),
    /// Arrow/WAL projection failed before sink mutation.
    Projection(String),
    /// Sink durability or finalization failed.
    Sink(ArchiveSinkError),
    /// Receipt target/event construction failed.
    Receipt(String),
    /// Prepared spool/queue admission rejected a complete transaction.
    Admission(AdmissionRejection),
    /// Spool quota refresh or protected finalization reserve failed.
    Budget(String),
    /// A compact missed range was internally invalid.
    InvalidMissedRange,
    /// An issued-work loss violated its closed source/kind/reason matrix.
    InvalidIssuedLoss,
    /// A loss named a source outside the prepared owner universe.
    UnknownLossSource(String),
    /// Source event admission was duplicated, skipped, or reordered.
    SourceAdmissionSequence {
        /// Stable physical source identity.
        source_id: String,
        /// Next source event required by the owner.
        expected: u64,
        /// Source event presented by the caller.
        actual: u64,
    },
    /// A boundary loss exceeded its preparation-time exact reference reserve.
    BoundaryLossReserve {
        /// Prepared exact reference capacity.
        maximum: usize,
        /// References carried by the source-cardinal command.
        actual: usize,
    },
    /// Fixed exact capacity was exhausted for a boundary-bearing loss.
    BoundaryLossSaturated,
    /// Fixed-memory loss accounting rejected an owner-stamped loss.
    LossLedger(String),
    /// A monotone counter overflowed.
    SequenceOverflow,
    /// Owner task stopped before a response.
    TaskStopped,
    /// The best-effort owner retained health after its durability writer died.
    WriterUnavailable,
    /// Owner task join failed.
    Join(String),
    /// Owner handle was joined twice.
    AlreadyJoined,
    /// Every lifecycle owner disappeared without an ordered finalize command.
    FinalizeNotRequested,
    /// A terminal error poisoned the owner and stopped later admission.
    FailStopped(String),
    /// Internal owner topology or sequencing invariant failed.
    Invariant(String),
}

impl Display for TelemetryArchiveOwnerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroQueueCapacity => {
                formatter.write_str("telemetry archive owner queue capacity must be positive")
            }
            Self::ZeroControlQueueCapacity => formatter
                .write_str("telemetry archive owner control queue capacity must be positive"),
            Self::RecoverySequence {
                recovered,
                sequencer,
            } => write!(
                formatter,
                "archive recovery next sequence {recovered} disagrees with sequencer {sequencer}"
            ),
            Self::Sequencing(message) => write!(formatter, "archive sequencing failed: {message}"),
            Self::Projection(message) => write!(formatter, "archive projection failed: {message}"),
            Self::Sink(error) => write!(formatter, "archive sink failed: {error}"),
            Self::Receipt(message) => write!(formatter, "archive receipt failed: {message}"),
            Self::Admission(error) => write!(formatter, "archive admission failed: {error}"),
            Self::Budget(message) => write!(formatter, "archive budget failed: {message}"),
            Self::InvalidMissedRange => formatter.write_str("invalid telemetry missed range"),
            Self::InvalidIssuedLoss => formatter.write_str("invalid issued telemetry loss"),
            Self::UnknownLossSource(source_id) => {
                write!(
                    formatter,
                    "telemetry loss named unknown source {source_id:?}"
                )
            }
            Self::SourceAdmissionSequence {
                source_id,
                expected,
                actual,
            } => write!(
                formatter,
                "telemetry source {source_id:?} admitted sequence {actual}, expected {expected}"
            ),
            Self::BoundaryLossReserve { maximum, actual } => write!(
                formatter,
                "boundary loss carries {actual} references, exact reserve is {maximum}"
            ),
            Self::BoundaryLossSaturated => formatter
                .write_str("boundary loss exhausted exact reserve and cannot use saturation"),
            Self::LossLedger(message) => write!(formatter, "archive loss ledger failed: {message}"),
            Self::SequenceOverflow => formatter.write_str("archive owner sequence overflow"),
            Self::TaskStopped => formatter.write_str("telemetry archive owner task stopped"),
            Self::WriterUnavailable => {
                formatter.write_str("telemetry archive writer is unavailable")
            }
            Self::Join(message) => {
                write!(formatter, "telemetry archive owner join failed: {message}")
            }
            Self::AlreadyJoined => formatter.write_str("telemetry archive owner already joined"),
            Self::FinalizeNotRequested => {
                formatter.write_str("telemetry archive owner closed without finalization")
            }
            Self::FailStopped(message) => {
                write!(formatter, "archive owner fail-stopped: {message}")
            }
            Self::Invariant(message) => write!(formatter, "archive owner invariant: {message}"),
        }
    }
}

impl std::error::Error for TelemetryArchiveOwnerError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use aiperf_clock::SimClock;
    use aiperf_prometheus::StrictExpositionParser;
    use aiperf_telemetry_archive::{
        ArchiveId, ArchiveSchemasV1, ArchiveSpoolBudgetLimits, ArchiveSpoolObservation,
        ArchiveSpoolReservePlan, ArchiveSpoolResources, AtomicArchiveSpoolBudget,
        AttachedBestEffortAdmissionPolicy, AttemptDecoder, Blake3ArchiveKeyProvider, BoundaryRole,
        DecodeLimits, EpochAnchor, ExecutionId, FetchDisposition, FetchedAttempt,
        MemoryArchiveSink, MemoryArchiveSinkFault, NativeEntityDecoder, NoopNativeEntityDecoder,
        ParquetRotationConfigV1, PrimaryWatchAdmissionPolicy, PrometheusAttemptDecoder,
        ReceiptObserverEpochV1, SessionId, SourceProjectionPolicyV1, TimeDomain,
        WalSegmentHeaderV1,
    };
    use bytes::Bytes;

    use super::*;

    fn id(seed: u8) -> [u8; 16] {
        let mut bytes = [seed; 16];
        bytes[15] = seed.wrapping_add(1);
        bytes
    }

    fn decoded(
        body: &'static [u8],
    ) -> aiperf_telemetry_archive::DecodedAttempt<aiperf_prometheus::Exposition, ()> {
        decoded_at(0, Some(10), body)
    }

    fn decoded_at(
        sequence: u64,
        scheduled_ns: Option<i64>,
        body: &'static [u8],
    ) -> aiperf_telemetry_archive::DecodedAttempt<aiperf_prometheus::Exposition, ()> {
        let decoder = PrometheusAttemptDecoder::new(
            Arc::new(StrictExpositionParser),
            Arc::new(NoopNativeEntityDecoder) as Arc<dyn NativeEntityDecoder<()>>,
        );
        decoder.decode(
            FetchedAttempt {
                source_id: "node-a".to_owned(),
                source_record_seq: sequence,
                request_attempt_seq: Some(sequence),
                scheduled_ns,
                request_start_ns: Some(11 + i64::try_from(sequence).unwrap()),
                first_byte_ns: Some(12 + i64::try_from(sequence).unwrap()),
                capture_ns: Some(13 + i64::try_from(sequence).unwrap()),
                latency_ns: Some(2),
                disposition: FetchDisposition::Response {
                    status: 200,
                    content_type: Some("text/plain; version=0.0.4; charset=utf-8".to_owned()),
                    content_encoding: None,
                    encoded_body: Bytes::from_static(body),
                    decoded_body: Bytes::from_static(body),
                },
            },
            &DecodeLimits::default(),
        )
    }

    fn config(clock: Rc<dyn Clock>) -> TelemetryArchiveOwnerConfig {
        config_with_fault(clock, MemoryArchiveSinkFault::None)
    }

    const fn resources(bytes: u64, files: u64) -> ArchiveSpoolResources {
        ArchiveSpoolResources { bytes, files }
    }

    fn budget_with_ordinary_capacity(
        ordinary_frame_capacity: u64,
    ) -> Arc<dyn ArchiveSpoolBudgetAuthority> {
        AtomicArchiveSpoolBudget::new(
            ArchiveSpoolBudgetLimits {
                quota: resources(10_000_000, 10_000),
                ordinary_frame_capacity,
                control_frame_capacity: 64,
                reserve: ArchiveSpoolReservePlan {
                    largest_wal_frame: resources(1, 1),
                    fallback_wal_window: resources(1, 1),
                    open_parquet_builders: resources(1, 1),
                    cow_index_path: resources(1, 1),
                    generation_and_head: resources(1, 1),
                    receipt_transaction: resources(1, 1),
                    optional_raw_object: resources(0, 0),
                    wal_seal: resources(1, 1),
                    emergency_finalization: resources(1, 1),
                    control_lane: resources(100_000, 1_000),
                },
            },
            ArchiveSpoolObservation {
                logical_bytes: 0,
                logical_files: 0,
                filesystem_available_bytes: 10_000_000,
                filesystem_available_files: 10_000,
            },
        )
        .unwrap()
    }

    fn config_with_fault(
        clock: Rc<dyn Clock>,
        fault: MemoryArchiveSinkFault,
    ) -> TelemetryArchiveOwnerConfig {
        let archive_id = ArchiveId::new(id(1)).unwrap();
        let session_id = SessionId::new(id(2)).unwrap();
        let schemas = ArchiveSchemasV1::load().unwrap();
        let segment = WalSegmentHeaderV1::new(
            archive_id,
            session_id,
            aiperf_telemetry_archive::Digest::from_bytes([3; 32]),
            aiperf_telemetry_archive::Digest::from_bytes([4; 32]),
            aiperf_telemetry_archive::Digest::from_bytes([5; 32]),
            0,
            aiperf_telemetry_archive::SessionAnchorV1::new(TimeDomain::Virtual, None).unwrap(),
            schemas
                .iter()
                .map(|schema| (schema.table(), schema.fingerprint()))
                .collect(),
        )
        .unwrap();
        let mut sink =
            MemoryArchiveSink::new(segment, schemas.clone(), ParquetRotationConfigV1::default())
                .unwrap();
        sink.set_fault(fault);
        let key = Arc::new(Blake3ArchiveKeyProvider::new("fixture_key", [7; 32]).unwrap());
        let sequencer = ArchiveFrameSequencerV1::new(
            archive_id,
            session_id,
            Some(EpochAnchor {
                clock_ns: 0,
                unix_epoch_ns: 1_700_000_000_000_000_000,
                capture_uncertainty_ns: 0,
            }),
            key,
            BTreeMap::from([(
                "node-a".to_owned(),
                SourceProjectionPolicyV1 {
                    attributes: BTreeMap::new(),
                },
            )]),
        )
        .unwrap();
        let receipt_epoch = ReceiptObserverEpochV1::new(
            ExecutionId::new(id(9)).unwrap(),
            Some(session_id),
            TimeDomain::Virtual,
            0,
            None,
            0,
            aiperf_telemetry_archive::Digest::from_bytes([8; 32]),
        )
        .unwrap();
        let loss_ledger = aiperf_telemetry_archive::FixedLossLedgerV1::new(
            archive_id,
            session_id,
            ["node-a"],
            aiperf_telemetry_archive::LossLedgerLimitsV1 {
                max_exact_ranges: 8,
                max_sources: 1,
                max_source_id_bytes: 64,
                max_boundary_refs_per_range: 4,
                max_boundary_identifier_bytes: 64,
            },
        )
        .unwrap();
        TelemetryArchiveOwnerConfig {
            archive_id,
            session_id,
            sequencer,
            codec: SourceFrameCodecV1::with_schemas(schemas),
            control_codec: ControlFrameCodecV1::new().unwrap(),
            sink: Box::new(sink),
            clock,
            receipt_epoch,
            receipt_epoch_registered: false,
            next_receipt_seq: 0,
            loss_ledger,
            queue_capacity: 2,
            control_queue_capacity: 4,
            best_effort: false,
            attached: false,
            budget_authority: budget_with_ordinary_capacity(64),
            admission_policy: Arc::new(PrimaryWatchAdmissionPolicy),
            ordinary_projection_footprint: ArchiveProjectionFootprint {
                bytes: 128,
                frames: 1,
                files: 1,
            },
            control_projection_footprint: ArchiveProjectionFootprint {
                bytes: 128,
                frames: 1,
                files: 1,
            },
        }
    }

    fn attached_config(clock: Rc<dyn Clock>) -> TelemetryArchiveOwnerConfig {
        let mut config = config(clock);
        config.attached = true;
        config.best_effort = true;
        config.admission_policy = Arc::new(AttachedBestEffortAdmissionPolicy);
        config
    }

    fn observation(
        sequence: u64,
        boundary_refs: Vec<BoundaryReference>,
    ) -> ArchiveAttemptObservation {
        let boundary = !boundary_refs.is_empty();
        ArchiveAttemptObservation {
            decoded: decoded_at(
                sequence,
                (!boundary).then_some(10 + i64::try_from(sequence).unwrap()),
                b"# TYPE temperature gauge\ntemperature 42\n",
            ),
            parse_done_ns: 20 + i64::try_from(sequence).unwrap(),
            archive_enqueue_ns: 21 + i64::try_from(sequence).unwrap(),
            projection_context: aiperf_telemetry_archive::ArchiveAttemptProjectionContextV1 {
                reason: if boundary {
                    ScrapeReasonV1::Boundary
                } else {
                    ScrapeReasonV1::Continuous
                },
                boundary_refs,
            },
        }
    }

    fn issued_loss(sequence: u64, kind: LossKindV1) -> ArchiveIssuedLossObservation {
        ArchiveIssuedLossObservation {
            source_id: "node-a".to_owned(),
            source_record_seq: sequence,
            request_attempt_seq: Some(sequence),
            loss_kind: kind,
            reason: kind.reason(),
            observed_ns: 100 + i64::try_from(sequence).unwrap(),
            boundary_refs: Vec::new(),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn acknowledgment_crosses_wal_and_receipt_before_finalization() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let (owner, running) = start_telemetry_archive_owner(config(clock)).await.unwrap();
                owner
                    .observe_attempt(ArchiveAttemptObservation {
                        decoded: decoded(b"# TYPE temperature gauge\ntemperature 42\n"),
                        parse_done_ns: 14,
                        archive_enqueue_ns: 15,
                        projection_context:
                            aiperf_telemetry_archive::ArchiveAttemptProjectionContextV1::continuous(
                            ),
                    })
                    .await
                    .unwrap();
                let finalization = running.finalize(TerminationReason::Duration).await.unwrap();
                assert_eq!(finalization.summary.durable_frames, 1);
                assert_eq!(finalization.summary.receipts, 1);
                assert!(finalization.summary.last_frame_id.is_some());
                let budget = finalization.summary.budget.unwrap();
                assert!(budget.closed);
                assert!(budget.finalizing);
                assert_eq!(budget.outstanding_leases, 0);
                assert_eq!(
                    finalization.completion.as_ref().unwrap().archive_state,
                    aiperf_telemetry_archive::ArchiveState::LocallyFinalized
                );
                assert_eq!(
                    finalization
                        .completion
                        .as_ref()
                        .unwrap()
                        .sealed_wal
                        .frame_count(),
                    1
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn lifecycle_marker_is_owner_sequenced_and_receipt_durable() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let (_source_owner, running) =
                    start_telemetry_archive_owner(config(clock)).await.unwrap();
                running
                    .observe_lifecycle(ArchiveLifecycleObservation::session_started(5))
                    .await
                    .unwrap();
                let finalization = running.finalize(TerminationReason::Duration).await.unwrap();
                assert_eq!(finalization.summary.durable_frames, 1);
                assert_eq!(finalization.summary.receipts, 1);
                assert_eq!(
                    finalization
                        .completion
                        .as_ref()
                        .unwrap()
                        .sealed_wal
                        .frame_count(),
                    1
                );
                assert!(finalization.repository.is_none());
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missed_ranges_are_bounded_owner_health_facts() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let (owner, running) = start_telemetry_archive_owner(config(clock)).await.unwrap();
                owner
                    .observe_missed(ArchiveMissedObservation {
                        source_id: "node-a".to_owned(),
                        missed: aiperf_telemetry_archive::MissedCadenceRange {
                            first_tick: 1,
                            last_tick: 3,
                            first_deadline_ns: 10,
                            last_deadline_ns: 30,
                            count: 3,
                        },
                        observed_ns: 31,
                    })
                    .await
                    .unwrap();
                owner
                    .observe_attempt(ArchiveAttemptObservation {
                        decoded: decoded(b"# TYPE temperature gauge\ntemperature 42\n"),
                        parse_done_ns: 40,
                        archive_enqueue_ns: 41,
                        projection_context:
                            aiperf_telemetry_archive::ArchiveAttemptProjectionContextV1::continuous(
                            ),
                    })
                    .await
                    .unwrap();
                let finalization = running.finalize(TerminationReason::Duration).await.unwrap();
                assert_eq!(finalization.summary.missed_ranges, 1);
                assert_eq!(finalization.summary.missed_ticks, 3);
                let ledger = finalization.summary.loss_ledger.unwrap();
                assert_eq!(ledger.exact_ranges.len(), 1);
                assert!(ledger.complete_ranges);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn attached_backpressure_records_loss_and_preserves_later_source_sequence() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let mut prepared = attached_config(clock);
                prepared.queue_capacity = 1;
                let (owner, running) = start_telemetry_archive_owner(prepared).await.unwrap();

                owner
                    .try_observe_attempt(observation(0, Vec::new()))
                    .unwrap();
                assert!(matches!(
                    owner.try_observe_attempt(observation(1, Vec::new())),
                    Err(AdmissionRejection::Capacity)
                ));
                owner
                    .record_visible_loss(issued_loss(1, LossKindV1::ArchiveRejected))
                    .unwrap();
                tokio::task::yield_now().await;
                owner
                    .try_observe_attempt(observation(2, Vec::new()))
                    .unwrap();

                let finalization = running.finalize(TerminationReason::Duration).await.unwrap();
                assert_eq!(finalization.summary.archive_rejected, 1);
                assert!(finalization.summary.writer_alive);
                let losses = finalization.summary.loss_ledger.unwrap();
                assert_eq!(losses.exact_ranges.len(), 1);
                assert_eq!(losses.exact_ranges[0].first_source_record_seq, Some(1));
                assert_eq!(
                    finalization
                        .completion
                        .as_ref()
                        .unwrap()
                        .sealed_wal
                        .frame_count(),
                    3
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn attached_budget_rejection_is_visible_before_data_queue_fills() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let mut prepared = attached_config(clock);
                prepared.queue_capacity = 2;
                prepared.budget_authority = budget_with_ordinary_capacity(1);
                let (owner, running) = start_telemetry_archive_owner(prepared).await.unwrap();

                owner
                    .try_observe_attempt(observation(0, Vec::new()))
                    .unwrap();
                assert_eq!(owner.data_commands.capacity(), 1);
                assert!(matches!(
                    owner.try_observe_attempt(observation(1, Vec::new())),
                    Err(AdmissionRejection::Capacity)
                ));
                assert_eq!(owner.data_commands.capacity(), 1);
                owner
                    .record_visible_loss(issued_loss(1, LossKindV1::ArchiveRejected))
                    .unwrap();
                tokio::task::yield_now().await;
                owner
                    .try_observe_attempt(observation(2, Vec::new()))
                    .unwrap();

                let finalization = running.finalize(TerminationReason::Duration).await.unwrap();
                assert_eq!(finalization.summary.archive_rejected, 1);
                let budget = finalization.summary.budget.unwrap();
                assert!(budget.closed);
                assert!(budget.finalizing);
                assert_eq!(budget.outstanding_leases, 0);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn attached_atomic_projection_rejection_is_loss_without_killing_writer() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let (owner, running) = start_telemetry_archive_owner(attached_config(clock))
                    .await
                    .unwrap();
                let mut invalid = observation(0, Vec::new());
                invalid.parse_done_ns = invalid.archive_enqueue_ns + 1;
                owner.try_observe_attempt(invalid).unwrap();
                tokio::task::yield_now().await;

                let finalization = running.finalize(TerminationReason::Duration).await.unwrap();
                assert!(finalization.summary.writer_alive);
                assert_eq!(finalization.summary.projection_failed, 1);
                assert_eq!(finalization.summary.writer_failed, 0);
                assert_eq!(
                    finalization.summary.loss_ledger.unwrap().exact_ranges[0].loss_kind,
                    LossKindV1::ProjectionFailed
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn repeated_checkpoints_emit_only_new_exact_and_dirty_latest_saturation() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let (owner, running) = start_telemetry_archive_owner(attached_config(clock))
                    .await
                    .unwrap();
                let kinds = [
                    LossKindV1::ArchiveRejected,
                    LossKindV1::ProjectionFailed,
                    LossKindV1::WriterFailed,
                    LossKindV1::ShutdownAbandoned,
                ];
                for sequence in 0..9_u64 {
                    owner
                        .record_visible_loss(issued_loss(
                            sequence,
                            kinds[usize::try_from(sequence % 4).unwrap()],
                        ))
                        .unwrap();
                }
                running.checkpoint().await.unwrap();
                owner
                    .record_visible_loss(issued_loss(9, LossKindV1::ArchiveRejected))
                    .unwrap();
                running.checkpoint().await.unwrap();

                let finalization = running.finalize(TerminationReason::Duration).await.unwrap();
                let losses = finalization.summary.loss_ledger.unwrap();
                assert_eq!(losses.exact_ranges.len(), 8);
                assert_eq!(losses.saturation_snapshots.len(), 1);
                assert!(!losses.complete_ranges);
                assert_eq!(
                    finalization
                        .completion
                        .as_ref()
                        .unwrap()
                        .sealed_wal
                        .frame_count(),
                    10
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn writer_failure_keeps_native_run_health_and_boundary_loss_visible() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let mut prepared =
                    config_with_fault(clock, MemoryArchiveSinkFault::AppendUncertainAfterApply);
                prepared.attached = true;
                prepared.best_effort = true;
                prepared.admission_policy = Arc::new(AttachedBestEffortAdmissionPolicy);
                let (owner, running) = start_telemetry_archive_owner(prepared).await.unwrap();
                let reference = BoundaryReference {
                    transition_id: "warmup-to-profile".to_owned(),
                    boundary_id: "node-a-end".to_owned(),
                    phase_id: "warmup".to_owned(),
                    source_id: "node-a".to_owned(),
                    role: BoundaryRole::PhaseEnd,
                    coalescing_group_id: None,
                };
                let admission = owner
                    .try_observe_attempt(observation(0, vec![reference.clone()]))
                    .unwrap();
                assert_eq!(
                    admission.boundary_terminal.unwrap().await.unwrap(),
                    TelemetryAttemptDisposition::Loss {
                        kind: LossKindV1::WriterFailed,
                        reason: LossReasonV1::WriterError,
                    }
                );
                let finalization = running.finalize(TerminationReason::Failure).await.unwrap();
                assert!(finalization.completion.is_none());
                assert!(!finalization.summary.writer_alive);
                assert!(finalization.summary.writer_failed >= 2);
                assert!(finalization.summary.first_failure.is_some());
                let losses = finalization.summary.loss_ledger.unwrap();
                assert!(
                    losses
                        .exact_ranges
                        .iter()
                        .any(|loss| loss.boundary_refs == vec![reference.clone()])
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn expired_shutdown_converts_accepted_boundary_to_exact_loss() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
                let (owner, running) = start_telemetry_archive_owner(attached_config(clock))
                    .await
                    .unwrap();
                let references = vec![
                    BoundaryReference {
                        transition_id: "adjacent".to_owned(),
                        boundary_id: "node-a-end".to_owned(),
                        phase_id: "warmup".to_owned(),
                        source_id: "node-a".to_owned(),
                        role: BoundaryRole::PhaseEnd,
                        coalescing_group_id: Some("node-a-adjacent".to_owned()),
                    },
                    BoundaryReference {
                        transition_id: "adjacent".to_owned(),
                        boundary_id: "node-a-start".to_owned(),
                        phase_id: "profiling".to_owned(),
                        source_id: "node-a".to_owned(),
                        role: BoundaryRole::PhaseStart,
                        coalescing_group_id: Some("node-a-adjacent".to_owned()),
                    },
                ];
                let admission = owner
                    .try_observe_attempt(observation(0, references.clone()))
                    .unwrap();
                let finalization_future = running.finalize_before(TerminationReason::Signal, 0);
                let finalization = finalization_future.await.unwrap();
                assert_eq!(finalization.summary.shutdown_abandoned, 1);
                let losses = finalization.summary.loss_ledger.unwrap();
                assert_eq!(losses.exact_ranges.len(), 1);
                assert_eq!(losses.exact_ranges[0].boundary_refs, references);
                assert_eq!(
                    admission.boundary_terminal.unwrap().await.unwrap(),
                    TelemetryAttemptDisposition::Loss {
                        kind: LossKindV1::ShutdownAbandoned,
                        reason: LossReasonV1::ShutdownDeadline,
                    }
                );
            })
            .await;
    }
}
