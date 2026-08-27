// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, cooperatively cancellable ownership of blocking streaming work.

use std::{
    cell::{Cell, RefCell},
    fmt,
    ops::Deref,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use bytes::Bytes;
use futures::FutureExt;
use tokio::{sync::watch, task::JoinHandle};

use super::{
    budget::{BudgetError, BudgetLease, BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, DecodeHorizon,
        ParticipantInitialization, ParticipantStateDescriptor, PreparedParticipantState,
        StreamingCheckpointParticipant,
    },
    unit::SourcePosition,
};

/// Stable schema identity for the blocking owner's checkpoint payload.
pub const BLOCKING_CHECKPOINT_SCHEMA_ID: &str = "aiperf.streaming.blocking-owner";
/// Current blocking-owner checkpoint schema version.
pub const BLOCKING_CHECKPOINT_SCHEMA_VERSION: u32 = 1;

const BLOCKING_CHECKPOINT_PAYLOAD_BYTES: usize = 16;
const DEFAULT_BLOCKING_PARTICIPANT_ID: &str = "streaming-blocking";

/// Host-classified category of blocking work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BlockingWorkClass {
    /// Network-file or object acquisition.
    Acquisition,
    /// Format or record decoding.
    Decode,
    /// Compression or decompression.
    Compression,
    /// Bounded external sorting.
    ExternalSort,
    /// Durable synchronization such as `fsync`.
    DurableSync,
    /// Immutable index construction.
    IndexBuild,
    /// Final result compaction.
    FinalCompaction,
}

/// Capacity reserved before one blocking closure is enqueued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockingWorkBudget {
    /// Bytes retained as inputs for the lifetime of the closure.
    pub input_bytes: usize,
    /// Caller-declared retained output allocation reservation.
    pub output_bytes: usize,
}

/// Cooperative cancellation token passed into a blocking closure.
#[derive(Clone, Debug)]
pub struct BlockingCancellation {
    is_cancelled: Arc<AtomicBool>,
}

impl BlockingCancellation {
    /// Return whether shutdown has requested cancellation.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.is_cancelled.load(Ordering::Acquire)
    }

    fn cancel(&self) {
        self.is_cancelled.store(true, Ordering::Release);
    }
}

/// Blocking submission, execution, accounting, or join failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BlockingWorkError {
    /// The executor no longer accepts work.
    SubmissionClosed,
    /// Cooperative cancellation terminated the work.
    Cancelled,
    /// A streaming resource budget could not be acquired or maintained.
    Budget(BudgetError),
    /// The process exhausted the monotonic accepted-job identity space.
    JobIdExhausted,
    /// The blocking task panicked or was cancelled by Tokio.
    Join {
        /// Stable diagnostic from Tokio's join failure.
        message: String,
    },
    /// A worker exited without publishing its typed result.
    MissingResult,
}

impl fmt::Display for BlockingWorkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "streaming blocking-work error: {self:?}")
    }
}

impl std::error::Error for BlockingWorkError {}

impl From<BudgetError> for BlockingWorkError {
    fn from(error: BudgetError) -> Self {
        Self::Budget(error)
    }
}

/// Typed output inseparably retaining its output allocation lease.
#[derive(Debug)]
pub struct BudgetedBlockingOutput<T> {
    value: T,
    _lease: BudgetLease,
    class: BlockingWorkClass,
}

impl<T> BudgetedBlockingOutput<T> {
    /// Return the blocking work category that produced this value.
    #[must_use]
    pub const fn class(&self) -> BlockingWorkClass {
        self.class
    }
}

impl<T> Deref for BudgetedBlockingOutput<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

/// Durable blocking-owner state containing no replayable closure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockingCheckpointState {
    completed_horizon: DecodeHorizon,
    inflight_job_count: usize,
}

impl BlockingCheckpointState {
    /// Construct a state value for checkpoint encoding or restore validation.
    #[must_use]
    pub const fn new(completed_horizon: DecodeHorizon, inflight_job_count: usize) -> Self {
        Self {
            completed_horizon,
            inflight_job_count,
        }
    }

    /// Borrow the greatest fully completed typed decode horizon.
    #[must_use]
    pub const fn completed_horizon(&self) -> &DecodeHorizon {
        &self.completed_horizon
    }

    /// Return the encoded in-flight claim, which must be zero on restore.
    #[must_use]
    pub const fn inflight_job_count(&self) -> usize {
        self.inflight_job_count
    }

    /// Encode the fixed-size version-one state payload.
    #[must_use]
    pub fn encode(&self) -> [u8; BLOCKING_CHECKPOINT_PAYLOAD_BYTES] {
        let mut bytes = [0_u8; BLOCKING_CHECKPOINT_PAYLOAD_BYTES];
        bytes[..8].copy_from_slice(&self.completed_horizon.get().get().to_le_bytes());
        let inflight = u64::try_from(self.inflight_job_count).unwrap_or(u64::MAX);
        bytes[8..].copy_from_slice(&inflight.to_le_bytes());
        bytes
    }

    /// Decode one exact version-one state payload.
    pub fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let bytes: &[u8; BLOCKING_CHECKPOINT_PAYLOAD_BYTES] = bytes
            .try_into()
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let horizon = u64::from_le_bytes(
            bytes[..8]
                .try_into()
                .map_err(|_| CheckpointError::ObjectVerification)?,
        );
        let inflight = u64::from_le_bytes(
            bytes[8..]
                .try_into()
                .map_err(|_| CheckpointError::ObjectVerification)?,
        );
        Ok(Self {
            completed_horizon: DecodeHorizon::new(SourcePosition::new(horizon)),
            inflight_job_count: usize::try_from(inflight)
                .map_err(|_| CheckpointError::ObjectVerification)?,
        })
    }
}

/// Current bounded blocking-owner resource use and committed progress.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockingExecutorSnapshot {
    /// Accepted closures not yet joined.
    pub accepted_jobs: usize,
    /// Input bytes retained by accepted or pending submissions.
    pub input_bytes: usize,
    /// Output allocation bytes reserved or retained by live outputs.
    pub output_bytes: usize,
    /// Greatest decode horizon advanced by an authoritative commit or restore.
    pub completed_horizon: Option<DecodeHorizon>,
    /// Whether new submissions may enter budget acquisition.
    pub is_accepting: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct JobId(u64);

struct AcceptedJob {
    id: JobId,
    cancellation: BlockingCancellation,
    _reaper: JoinHandle<()>,
    join_status: watch::Receiver<Option<Result<(), BlockingWorkError>>>,
}

struct ExecutorInner {
    participant_id: CheckpointParticipantId,
    is_accepting: Cell<bool>,
    is_shutdown: Cell<bool>,
    next_job_id: Cell<u64>,
    accepted_budget: StreamingResourceBudget,
    input_budget: StreamingResourceBudget,
    output_budget: StreamingResourceBudget,
    checkpoint_budget: StreamingResourceBudget,
    jobs: RefCell<Vec<Option<AcceptedJob>>>,
    initialization: RefCell<ParticipantInitialization>,
    completed_horizon: RefCell<Option<DecodeHorizon>>,
    prepared_descriptor: RefCell<Option<ParticipantStateDescriptor>>,
    committed_receipt: RefCell<Option<CommittedParticipantReceipt>>,
}

/// Worker-local owner of a bounded set of Tokio blocking tasks.
#[derive(Clone)]
pub struct StreamingBlockingExecutor {
    inner: Rc<ExecutorInner>,
}

impl fmt::Debug for StreamingBlockingExecutor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingBlockingExecutor")
            .field("participant_id", &self.inner.participant_id)
            .field("snapshot", &self.snapshot())
            .finish_non_exhaustive()
    }
}

impl StreamingBlockingExecutor {
    /// Construct a bounded blocking owner with a stable participant identity.
    pub fn new(
        participant_id: CheckpointParticipantId,
        max_accepted_jobs: usize,
        max_input_bytes: usize,
        max_output_bytes: usize,
    ) -> Result<Self, BlockingWorkError> {
        let accepted_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: max_accepted_jobs,
            max_bytes: 1,
        })?;
        let input_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: max_accepted_jobs,
            max_bytes: max_input_bytes,
        })?;
        let output_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: max_accepted_jobs,
            max_bytes: max_output_bytes,
        })?;
        let checkpoint_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: BLOCKING_CHECKPOINT_PAYLOAD_BYTES,
        })?;
        Ok(Self {
            inner: Rc::new(ExecutorInner {
                participant_id,
                is_accepting: Cell::new(true),
                is_shutdown: Cell::new(false),
                next_job_id: Cell::new(0),
                accepted_budget,
                input_budget,
                output_budget,
                checkpoint_budget,
                jobs: RefCell::new(Vec::with_capacity(max_accepted_jobs)),
                initialization: RefCell::new(ParticipantInitialization::default()),
                completed_horizon: RefCell::new(None),
                prepared_descriptor: RefCell::new(None),
                committed_receipt: RefCell::new(None),
            }),
        })
    }

    /// Construct the standard test owner.
    pub fn for_test(
        max_accepted_jobs: usize,
        max_input_bytes: usize,
        max_output_bytes: usize,
    ) -> Result<Self, BlockingWorkError> {
        Self::new(
            CheckpointParticipantId::new(DEFAULT_BLOCKING_PARTICIPANT_ID),
            max_accepted_jobs,
            max_input_bytes,
            max_output_bytes,
        )
    }

    /// Run one closure after reserving input, output, and accepted-job capacity.
    pub async fn run<T, F>(
        &self,
        class: BlockingWorkClass,
        budget: BlockingWorkBudget,
        work: F,
    ) -> Result<BudgetedBlockingOutput<T>, BlockingWorkError>
    where
        F: FnOnce(BlockingCancellation) -> Result<T, BlockingWorkError> + Send + 'static,
        T: Send + 'static,
    {
        self.ensure_accepting()?;
        let input_lease = self
            .inner
            .input_budget
            .acquire(1, budget.input_bytes)
            .await?;
        self.ensure_accepting()?;
        let output_lease = self
            .inner
            .output_budget
            .acquire(1, budget.output_bytes)
            .await?;
        self.ensure_accepting()?;
        let accepted_lease = self.inner.accepted_budget.acquire(1, 0).await?;
        self.ensure_accepting()?;

        let next = self.inner.next_job_id.get();
        let next_job_id = next
            .checked_add(1)
            .ok_or(BlockingWorkError::JobIdExhausted)?;
        let id = JobId(next);
        self.inner.next_job_id.set(next_job_id);
        let cancellation = BlockingCancellation {
            is_cancelled: Arc::new(AtomicBool::new(false)),
        };
        let worker_cancellation = cancellation.clone();
        self.reap_completed_jobs()?;
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let handle = tokio::task::spawn_blocking(move || {
            let result = work(worker_cancellation);
            drop(input_lease);
            let result = result.map(|value| (value, output_lease));
            let _ = sender.send(result);
        });
        let (join_sender, join_status) = watch::channel(None);
        // This bounded driver is the lifetime owner of both join authority and
        // accepted capacity, so cancellation of any caller cannot detach work.
        let reaper = tokio::spawn(async move {
            let result = handle.await.map_err(|error| BlockingWorkError::Join {
                message: error.to_string(),
            });
            drop(accepted_lease);
            let _ = join_sender.send(Some(result));
        });

        self.insert_job(id, cancellation, reaper, join_status.clone());
        let received = receiver.await;
        self.join_job(id, join_status).await?;
        let (value, lease) = received.map_err(|_| BlockingWorkError::MissingResult)??;
        Ok(BudgetedBlockingOutput {
            value,
            _lease: lease,
            class,
        })
    }

    /// Stop submission, signal every accepted closure, and join all of them.
    pub async fn cancel_and_join(&self) -> Result<(), BlockingWorkError> {
        self.inner.is_accepting.set(false);
        self.inner.is_shutdown.set(true);
        self.inner.accepted_budget.close();
        self.inner.input_budget.close();
        self.inner.output_budget.close();
        self.inner.checkpoint_budget.close();

        let jobs: Vec<_> = self
            .inner
            .jobs
            .borrow()
            .iter()
            .flatten()
            .map(|job| {
                job.cancellation.cancel();
                (job.id, job.join_status.clone())
            })
            .collect();
        let mut first_error = None;
        for (id, status) in jobs {
            if let Err(error) = self.join_job(id, status).await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    /// Snapshot current owned capacity and authoritative progress.
    #[must_use]
    pub fn snapshot(&self) -> BlockingExecutorSnapshot {
        BlockingExecutorSnapshot {
            accepted_jobs: self.inner.accepted_budget.snapshot().used_items,
            input_bytes: self.inner.input_budget.snapshot().used_bytes,
            output_bytes: self.inner.output_budget.snapshot().used_bytes,
            completed_horizon: self.inner.completed_horizon.borrow().clone(),
            is_accepting: self.inner.is_accepting.get(),
        }
    }

    fn ensure_accepting(&self) -> Result<(), BlockingWorkError> {
        if self.inner.is_accepting.get() {
            Ok(())
        } else {
            Err(BlockingWorkError::SubmissionClosed)
        }
    }

    fn insert_job(
        &self,
        id: JobId,
        cancellation: BlockingCancellation,
        reaper: JoinHandle<()>,
        join_status: watch::Receiver<Option<Result<(), BlockingWorkError>>>,
    ) {
        let job = AcceptedJob {
            id,
            cancellation,
            _reaper: reaper,
            join_status,
        };
        let mut jobs = self.inner.jobs.borrow_mut();
        if let Some(slot) = jobs.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(job);
        } else {
            jobs.push(Some(job));
        }
    }

    async fn join_job(
        &self,
        id: JobId,
        mut status: watch::Receiver<Option<Result<(), BlockingWorkError>>>,
    ) -> Result<(), BlockingWorkError> {
        loop {
            if let Some(result) = status.borrow().clone() {
                self.remove_joined_job(id);
                return result;
            }
            status
                .changed()
                .await
                .map_err(|_| BlockingWorkError::MissingResult)?;
        }
    }

    fn remove_joined_job(&self, id: JobId) {
        let mut jobs = self.inner.jobs.borrow_mut();
        let _ = jobs
            .iter_mut()
            .find(|slot| slot.as_ref().is_some_and(|job| job.id == id))
            .and_then(Option::take);
    }

    fn reap_completed_jobs(&self) -> Result<(), BlockingWorkError> {
        let finished: Vec<_> = self
            .inner
            .jobs
            .borrow()
            .iter()
            .flatten()
            .filter_map(|job| {
                job.join_status
                    .borrow()
                    .clone()
                    .map(|result| (job.id, result))
            })
            .collect();
        let mut first_error = None;
        for (id, result) in finished {
            self.remove_joined_job(id);
            if let Err(error) = result
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    async fn prepare_quiescent_view_or_refuse(
        &self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if self.inner.is_shutdown.get() {
            return Err(CheckpointError::ParticipantUnavailable {
                participant: self.inner.participant_id.clone(),
            });
        }
        self.inner.is_accepting.set(false);
        self.reap_completed_jobs()
            .map_err(|error| CheckpointError::Storage {
                message: error.to_string(),
            })?;
        let job_count = self.inner.accepted_budget.snapshot().used_items;
        if job_count != 0 {
            return Err(CheckpointError::CutBlockedByInflight {
                participant: self.inner.participant_id.clone(),
                job_count,
            });
        }
        self.ensure_horizon_not_regressed(&barrier.cut.decoded)?;

        let state = BlockingCheckpointState::new(barrier.cut.decoded.clone(), 0);
        let bytes = Bytes::from(state.encode().to_vec().into_boxed_slice());
        let lease = self
            .inner
            .checkpoint_budget
            .acquire(1, bytes.len())
            .now_or_never()
            .ok_or_else(|| CheckpointError::StateBudget {
                participant: self.inner.participant_id.clone(),
                code: super::unit::StateBudgetFailureCode::ItemCapacity,
            })?
            .map_err(|error| match error {
                BudgetError::Closed => CheckpointError::ParticipantUnavailable {
                    participant: self.inner.participant_id.clone(),
                },
                _ => CheckpointError::StateBudget {
                    participant: self.inner.participant_id.clone(),
                    code: super::unit::StateBudgetFailureCode::ItemCapacity,
                },
            })?;
        if self.inner.is_shutdown.get() {
            return Err(CheckpointError::ParticipantUnavailable {
                participant: self.inner.participant_id.clone(),
            });
        }
        let payload = BudgetedCheckpointBytes::new(bytes, lease)?;
        let prepared = PreparedParticipantState::new(
            self.inner.participant_id.clone(),
            BLOCKING_CHECKPOINT_SCHEMA_ID,
            BLOCKING_CHECKPOINT_SCHEMA_VERSION,
            barrier.cut.clone(),
            1,
            payload,
        )?;
        *self.inner.prepared_descriptor.borrow_mut() = Some(prepared.descriptor().clone());
        Ok(prepared)
    }

    async fn restore_completed_horizon_only(
        &self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.inner.initialization.borrow_mut().initialize_once()?;
        if self.inner.accepted_budget.snapshot().used_items != 0 {
            return Err(CheckpointError::ObjectVerification);
        }
        let completed_horizon = match state {
            Some(state) => {
                let descriptor = state.descriptor();
                if descriptor.participant_id != self.inner.participant_id
                    || descriptor.schema_id != BLOCKING_CHECKPOINT_SCHEMA_ID
                    || descriptor.schema_version != BLOCKING_CHECKPOINT_SCHEMA_VERSION
                    || descriptor.item_count != 1
                {
                    return Err(CheckpointError::ObjectVerification);
                }
                let restored = BlockingCheckpointState::decode(state.payload_bytes())?;
                if restored.inflight_job_count != 0
                    || restored.completed_horizon != descriptor.represented_cut.decoded
                {
                    return Err(CheckpointError::ObjectVerification);
                }
                Some(restored.completed_horizon)
            }
            None => None,
        };
        *self.inner.completed_horizon.borrow_mut() = completed_horizon;
        Ok(())
    }

    fn advance_committed(
        &self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.participant_id() != &self.inner.participant_id {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        if self.inner.committed_receipt.borrow().as_ref() == Some(receipt) {
            return Ok(());
        }
        if let Some(committed) = self.inner.committed_receipt.borrow().as_ref()
            && receipt.generation().epoch() <= committed.generation().epoch()
        {
            return Err(CheckpointError::GenerationConflict {
                expected: Some(committed.generation().clone()),
                actual: Some(receipt.generation().clone()),
            });
        }
        self.ensure_horizon_not_regressed(&receipt.represented_cut().decoded)?;
        {
            let prepared = self.inner.prepared_descriptor.borrow();
            let prepared = prepared
                .as_ref()
                .ok_or(CheckpointError::ObjectVerification)?;
            if receipt.descriptor_digest() != &prepared.digest()?
                || receipt.represented_cut() != &prepared.represented_cut
            {
                return Err(CheckpointError::ObjectVerification);
            }
        }
        *self.inner.completed_horizon.borrow_mut() =
            Some(receipt.represented_cut().decoded.clone());
        *self.inner.committed_receipt.borrow_mut() = Some(receipt.clone());
        self.inner.prepared_descriptor.borrow_mut().take();
        if !self.inner.is_shutdown.get() {
            self.inner.is_accepting.set(true);
        }
        Ok(())
    }

    fn ensure_horizon_not_regressed(
        &self,
        proposed: &DecodeHorizon,
    ) -> Result<(), CheckpointError> {
        if let Some(completed) = self.inner.completed_horizon.borrow().as_ref()
            && proposed.get().get() < completed.get().get()
        {
            return Err(CheckpointError::DecodeHorizonRegression {
                participant: self.inner.participant_id.clone(),
                completed: completed.clone(),
                proposed: proposed.clone(),
            });
        }
        Ok(())
    }
}

#[async_trait::async_trait(?Send)]
impl StreamingCheckpointParticipant for StreamingBlockingExecutor {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.inner.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        self.prepare_quiescent_view_or_refuse(barrier).await
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.restore_completed_horizon_only(state).await
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        self.advance_committed(receipt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{
        checkpoint::{
            AcquisitionHorizon, AdmissionHorizon, CheckpointCut, CheckpointEpoch,
            CheckpointGenerationCandidate, CheckpointGenerationPublicationProof,
            CheckpointParticipantPlan, DiscoveryHorizon, EventTimeWatermark, OrderedActionHorizon,
            TerminalActionHorizon,
        },
        identity::{ContentDigest, GlobalSequence, SessionCausalFrontier},
    };

    fn cut_at(value: u64) -> CheckpointCut {
        CheckpointCut {
            discovered: DiscoveryHorizon::new(SourcePosition::new(value)),
            acquired: AcquisitionHorizon::new(SourcePosition::new(value)),
            decoded: DecodeHorizon::new(SourcePosition::new(value)),
            ordered: OrderedActionHorizon::new(GlobalSequence::new(value)),
            admitted: AdmissionHorizon::new(GlobalSequence::new(value)),
            terminal: TerminalActionHorizon::new(GlobalSequence::new(value)),
            event_watermark: EventTimeWatermark::Unknown,
            causal_frontier: SessionCausalFrontier {
                through_sequence: GlobalSequence::new(value),
                event_time: None,
                digest: ContentDigest::from_bytes([value as u8; 32]),
            },
        }
    }

    fn authoritative_receipt(
        participant_id: CheckpointParticipantId,
        epoch: u64,
        previous: Option<ContentDigest>,
        cut: CheckpointCut,
    ) -> (
        CommittedParticipantReceipt,
        ParticipantStateDescriptor,
        ContentDigest,
    ) {
        let descriptor = ParticipantStateDescriptor {
            participant_id: participant_id.clone(),
            schema_id: BLOCKING_CHECKPOINT_SCHEMA_ID.into(),
            schema_version: BLOCKING_CHECKPOINT_SCHEMA_VERSION,
            represented_cut: cut.clone(),
            content_digest: ContentDigest::from_bytes([epoch as u8; 32]),
            item_count: 1,
            byte_length: BLOCKING_CHECKPOINT_PAYLOAD_BYTES as u64,
        };
        let plan = CheckpointParticipantPlan::new([participant_id]).expect("participant plan");
        let execution_plan = ContentDigest::from_bytes([0x41; 32]);
        let result_plan = ContentDigest::from_bytes([0x42; 32]);
        let candidate = CheckpointGenerationCandidate::new(
            CheckpointEpoch::new(epoch),
            previous,
            cut,
            &plan,
            execution_plan,
            result_plan,
            vec![descriptor.clone()],
            ContentDigest::from_bytes([0x43; 32]),
            false,
            None,
        )
        .expect("candidate");
        let generation_digest = *candidate.generation().digest();
        let proof = CheckpointGenerationPublicationProof::for_generation(candidate.generation());
        let committed = candidate
            .promote(&plan, &execution_plan, &result_plan, proof)
            .expect("authoritative generation");
        let receipt = CommittedParticipantReceipt::new(&committed, &descriptor)
            .expect("authoritative participant receipt");
        (receipt, descriptor, generation_digest)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn greater_epoch_receipt_cannot_regress_committed_decode_horizon() {
        let mut owner = StreamingBlockingExecutor::for_test(1, 8, 8).expect("executor");
        let participant_id = owner.participant_id();
        let (baseline, baseline_descriptor, baseline_digest) =
            authoritative_receipt(participant_id.clone(), 1, None, cut_at(7));
        *owner.inner.prepared_descriptor.borrow_mut() = Some(baseline_descriptor);
        owner
            .checkpoint_committed(&baseline)
            .await
            .expect("baseline commit");

        let (regressing, regressing_descriptor, _) =
            authoritative_receipt(participant_id, 2, Some(baseline_digest), cut_at(3));
        *owner.inner.prepared_descriptor.borrow_mut() = Some(regressing_descriptor);
        assert!(matches!(
            owner.checkpoint_committed(&regressing).await,
            Err(CheckpointError::DecodeHorizonRegression {
                completed,
                proposed,
                ..
            }) if completed == cut_at(7).decoded && proposed == cut_at(3).decoded
        ));
        assert_eq!(owner.snapshot().completed_horizon, Some(cut_at(7).decoded));
    }
}
