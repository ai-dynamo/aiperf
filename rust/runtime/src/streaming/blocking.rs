// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, cooperatively cancellable ownership of blocking streaming work.

use std::{
    cell::{Cell, RefCell},
    fmt,
    mem::size_of,
    ops::Deref,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use bytes::Bytes;
use tokio::{sync::Notify, task::JoinHandle};

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
    /// Maximum retained output allocation bytes.
    pub output_bytes: usize,
}

/// Explicit retained-allocation sizing contract for a blocking output.
///
/// Implementations must include the complete allocation retained by the value,
/// not merely its logical length. The executor validates this size against the
/// reservation before returning the value.
pub trait BlockingOutputSize {
    /// Return the number of allocation bytes retained by this value.
    fn retained_allocation_bytes(&self) -> Result<usize, BlockingWorkError>;
}

mod flat_element {
    pub trait Sealed {}

    macro_rules! flat_elements {
        ($($type:ty),+ $(,)?) => {
            $(impl Sealed for $type {})+
        };
    }

    flat_elements!(
        u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool, char
    );
}

impl<T: flat_element::Sealed> BlockingOutputSize for Vec<T> {
    fn retained_allocation_bytes(&self) -> Result<usize, BlockingWorkError> {
        self.capacity()
            .checked_mul(size_of::<T>())
            .ok_or(BlockingWorkError::OutputSizeOverflow)
    }
}

impl BlockingOutputSize for String {
    fn retained_allocation_bytes(&self) -> Result<usize, BlockingWorkError> {
        Ok(self.capacity())
    }
}

impl<T: flat_element::Sealed> BlockingOutputSize for Box<[T]> {
    fn retained_allocation_bytes(&self) -> Result<usize, BlockingWorkError> {
        self.len()
            .checked_mul(size_of::<T>())
            .ok_or(BlockingWorkError::OutputSizeOverflow)
    }
}

macro_rules! fixed_size_outputs {
    ($($type:ty),+ $(,)?) => {
        $(
            impl BlockingOutputSize for $type {
                fn retained_allocation_bytes(&self) -> Result<usize, BlockingWorkError> {
                    Ok(size_of::<Self>())
                }
            }
        )+
    };
}

fixed_size_outputs!(
    (),
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    f32,
    f64,
    bool,
    char
);

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
    /// The output retained more allocation capacity than was reserved.
    OutputExceedsReservation {
        /// Capacity reserved before enqueue.
        reserved_bytes: usize,
        /// Capacity retained by the returned value.
        retained_bytes: usize,
    },
    /// Retained output allocation size overflowed `usize`.
    OutputSizeOverflow,
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
    lease: BudgetLease,
    class: BlockingWorkClass,
}

impl<T> BudgetedBlockingOutput<T> {
    /// Return the blocking work category that produced this value.
    #[must_use]
    pub const fn class(&self) -> BlockingWorkClass {
        self.class
    }

    /// Return the retained allocation capacity charged to this output.
    #[must_use]
    pub fn retained_allocation_bytes(&self) -> usize {
        self.lease.charged_bytes()
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

struct JoinStatus {
    result: RefCell<Option<Result<(), BlockingWorkError>>>,
    notify: Notify,
}

impl JoinStatus {
    fn new() -> Self {
        Self {
            result: RefCell::new(None),
            notify: Notify::new(),
        }
    }

    fn complete(&self, result: Result<(), BlockingWorkError>) {
        *self.result.borrow_mut() = Some(result);
        self.notify.notify_waiters();
    }

    async fn wait(&self) -> Result<(), BlockingWorkError> {
        loop {
            let notified = self.notify.notified();
            if let Some(result) = self.result.borrow().clone() {
                return result;
            }
            notified.await;
        }
    }
}

struct AcceptedJob {
    id: JobId,
    cancellation: BlockingCancellation,
    accepted_lease: BudgetLease,
    handle: Option<JoinHandle<()>>,
    join_status: Rc<JoinStatus>,
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
        T: BlockingOutputSize + Send + 'static,
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
        let join_status = Rc::new(JoinStatus::new());
        let run_join_status = Rc::clone(&join_status);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let handle = tokio::task::spawn_blocking(move || {
            let result = work(worker_cancellation);
            drop(input_lease);
            let result = match result {
                Ok(value) => match value.retained_allocation_bytes() {
                    Ok(retained_bytes) if retained_bytes <= budget.output_bytes => {
                        let mut output_lease = output_lease;
                        match output_lease.shrink_to(1, retained_bytes) {
                            Ok(()) => Ok((value, output_lease)),
                            Err(error) => Err(BlockingWorkError::Budget(error)),
                        }
                    }
                    Ok(retained_bytes) => Err(BlockingWorkError::OutputExceedsReservation {
                        reserved_bytes: budget.output_bytes,
                        retained_bytes,
                    }),
                    Err(error) => Err(error),
                },
                Err(error) => Err(error),
            };
            let _ = sender.send(result);
        });

        self.insert_job(id, cancellation, accepted_lease, handle, join_status);
        let received = receiver.await;
        self.join_job(id, run_join_status).await?;
        let (value, lease) = received.map_err(|_| BlockingWorkError::MissingResult)??;
        Ok(BudgetedBlockingOutput {
            value,
            lease,
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

        let jobs: Vec<_> = self
            .inner
            .jobs
            .borrow()
            .iter()
            .flatten()
            .map(|job| {
                job.cancellation.cancel();
                (job.id, Rc::clone(&job.join_status))
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
        accepted_lease: BudgetLease,
        handle: JoinHandle<()>,
        join_status: Rc<JoinStatus>,
    ) {
        let job = AcceptedJob {
            id,
            cancellation,
            accepted_lease,
            handle: Some(handle),
            join_status,
        };
        let mut jobs = self.inner.jobs.borrow_mut();
        if let Some(slot) = jobs.iter_mut().find(|slot| slot.is_none()) {
            *slot = Some(job);
        } else {
            jobs.push(Some(job));
        }
    }

    async fn join_job(&self, id: JobId, status: Rc<JoinStatus>) -> Result<(), BlockingWorkError> {
        let handle = self
            .inner
            .jobs
            .borrow_mut()
            .iter_mut()
            .flatten()
            .find(|job| job.id == id)
            .and_then(|job| job.handle.take());
        if let Some(handle) = handle {
            let result = handle.await.map_err(|error| BlockingWorkError::Join {
                message: error.to_string(),
            });
            status.complete(result.clone());
            self.remove_joined_job(id);
            result
        } else {
            status.wait().await
        }
    }

    fn remove_joined_job(&self, id: JobId) {
        let mut jobs = self.inner.jobs.borrow_mut();
        if let Some(AcceptedJob { accepted_lease, .. }) = jobs
            .iter_mut()
            .find(|slot| slot.as_ref().is_some_and(|job| job.id == id))
            .and_then(Option::take)
        {
            drop(accepted_lease);
        }
    }

    async fn reap_finished_jobs(&self) -> Result<(), BlockingWorkError> {
        let finished: Vec<_> = self
            .inner
            .jobs
            .borrow()
            .iter()
            .flatten()
            .filter(|job| job.handle.as_ref().is_some_and(JoinHandle::is_finished))
            .map(|job| (job.id, Rc::clone(&job.join_status)))
            .collect();
        for (id, status) in finished {
            self.join_job(id, status).await?;
        }
        Ok(())
    }

    async fn prepare_quiescent_view_or_refuse(
        &self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        self.inner.is_accepting.set(false);
        self.reap_finished_jobs()
            .await
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

        let state = BlockingCheckpointState::new(barrier.cut.decoded.clone(), 0);
        let bytes = Bytes::from(state.encode().to_vec().into_boxed_slice());
        let lease = self
            .inner
            .checkpoint_budget
            .acquire(1, bytes.len())
            .await
            .map_err(|error| CheckpointError::Storage {
                message: error.to_string(),
            })?;
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
