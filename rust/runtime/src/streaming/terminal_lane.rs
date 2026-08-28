// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded terminal record-processing lane for native streaming runs.
//!
//! The finite scheduled runtime detaches one `spawn_local` task and retains one
//! `JoinHandle` per completed record. A stream has no record count, so that
//! shape cannot be reused. This module replaces it with a two-dimensional
//! item-and-byte reservation taken *before* dispatch, one drain owner for the
//! whole phase, and fixed-size counters.
//!
//! The ordering contract is: reserve (may wait) -> issue -> settle (cannot wait,
//! cannot fail for capacity) -> drain. Capacity is proven at reservation, so
//! the terminal edge of a request never blocks on the lane.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::fmt;
use std::num::NonZeroUsize;
use std::rc::Rc;

use serde_json::Value;
use tokio::sync::Notify;
use uuid::Uuid;

use crate::multiturn::IssuedCredit;
use crate::scheduled::{TurnDispatchOutcome, TurnRecordProcessor};
use crate::streaming::budget::{BudgetError, BudgetLease, BudgetLimits, StreamingResourceBudget};
use crate::streaming::checkpoint::{CheckpointGeneration, StreamRunIdentity};
use crate::streaming::failure::{
    OrdinaryStreamingFailure, ResultExportError, ResultExportFailureCode,
};
use crate::streaming::identity::ContentDigest;
use crate::streaming::reliability::{
    OrdinaryStreamingIssue, StreamingIssueClass, StreamingIssueComponentId,
    StreamingIssueReportError, StreamingIssueReportStatus, StreamingIssueReporterHandle,
    StreamingTerminalInvariant,
};

/// Structural cost of one queued terminal record, excluding owned heap bytes.
const TERMINAL_RECORD_FIXED_BYTES: usize =
    size_of::<IssuedCredit>() + size_of::<TurnDispatchOutcome>();

/// Per-node structural cost charged when measuring a retained JSON value.
const JSON_NODE_FIXED_BYTES: usize = size_of::<Value>();

/// Component identity used for lane-authored export issues.
const TERMINAL_LANE_COMPONENT_ID: &str = "streaming_terminal_lane";

/// Fixed item and byte capacity for the whole terminal lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalLaneLimits {
    /// Maximum simultaneously reserved or queued terminal records.
    pub max_items: usize,
    /// Maximum simultaneously reserved or retained terminal bytes.
    pub max_bytes: usize,
}

/// Proven finite upper bound on the bytes one terminal record may retain.
///
/// This is never inferred from an observed record. It is proven once, before
/// dispatch, from declared endpoint and capture limits (see
/// [`TerminalRecordSizeInputs::prove`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalRecordSizeBound(NonZeroUsize);

impl TerminalRecordSizeBound {
    /// Construct a bound from an already-checked non-zero byte count.
    #[must_use]
    pub const fn new(bytes: NonZeroUsize) -> Self {
        Self(bytes)
    }

    /// Construct a bound, refusing a zero byte count.
    #[must_use]
    pub fn try_from_bytes(bytes: usize) -> Option<Self> {
        NonZeroUsize::new(bytes).map(Self)
    }

    /// Return the proven maximum byte count.
    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

/// Declared finite limits from which a conservative terminal bound is proven.
///
/// Every field is an authored or transport-enforced limit. Nothing here is a
/// heuristic cushion: if a dimension cannot be proven finite, [`Self::prove`]
/// refuses rather than guessing. The streaming run assembler fills this in on
/// the post-acceptance side of the capability agreement, from the accepted plan
/// plus the resolved endpoint, output-token, and capture policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalRecordSizeInputs {
    /// Transport-enforced maximum response body bytes, when authored.
    ///
    /// This is the HTTP client's `max_response_body_bytes`, the only limit the
    /// runtime actually enforces on a peer that ignores `max_tokens`.
    pub max_response_body_bytes: Option<u64>,
    /// Authored maximum output tokens for any turn in this run.
    pub max_output_tokens: u64,
    /// Maximum bytes any single generated token may decode to.
    pub max_bytes_per_output_token: u64,
    /// Fixed envelope for usage counters and per-record metric fields.
    pub usage_metric_envelope_bytes: u64,
    /// Maximum request-side bytes the retained credit clone carries.
    pub max_request_retained_bytes: u64,
    /// Whether verbatim raw payload capture is configured.
    pub captures_raw_payload: bool,
}

/// Why a conservative terminal bound could not be proven.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalBoundError {
    /// No transport-enforced response limit was authored.
    UnprovenResponseLimit,
    /// The declared output-token product is not finite in this address space.
    UnprovenTokenLimit,
    /// The proven bound does not fit `usize` on this target.
    BoundNotRepresentable,
    /// Every declared dimension is zero.
    EmptyBound,
}

impl fmt::Display for TerminalBoundError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "unbounded streaming terminal payload: {self:?}")
    }
}

impl std::error::Error for TerminalBoundError {}

impl TerminalRecordSizeInputs {
    /// Prove one conservative maximum terminal record size, or refuse.
    ///
    /// The response term is `min(transport cap, token product)`: the transport
    /// cap is the enforced ceiling, and the token product lowers it when the
    /// authored request is smaller. The transport cap is mandatory because a
    /// peer that ignores `max_tokens` is bounded by nothing else.
    pub fn prove(&self) -> Result<TerminalRecordSizeBound, TerminalBoundError> {
        let transport_cap = self
            .max_response_body_bytes
            .ok_or(TerminalBoundError::UnprovenResponseLimit)?;
        let token_product = self
            .max_output_tokens
            .checked_mul(self.max_bytes_per_output_token)
            .ok_or(TerminalBoundError::UnprovenTokenLimit)?;
        let response_term = if token_product == 0 {
            transport_cap
        } else {
            transport_cap.min(token_product)
        };
        // Raw capture retains a second verbatim copy of the same body.
        let raw_term = if self.captures_raw_payload {
            response_term
        } else {
            0
        };
        let total = response_term
            .checked_add(raw_term)
            .and_then(|value| value.checked_add(self.usage_metric_envelope_bytes))
            .and_then(|value| value.checked_add(self.max_request_retained_bytes))
            .ok_or(TerminalBoundError::BoundNotRepresentable)?;
        let total =
            usize::try_from(total).map_err(|_| TerminalBoundError::BoundNotRepresentable)?;
        let total = total
            .checked_add(TERMINAL_RECORD_FIXED_BYTES)
            .ok_or(TerminalBoundError::BoundNotRepresentable)?;
        TerminalRecordSizeBound::try_from_bytes(total).ok_or(TerminalBoundError::EmptyBound)
    }
}

/// Bounded terminal-lane failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalLaneError {
    /// Configured item or byte capacity is zero or unrepresentable.
    InvalidLimits(BudgetError),
    /// The proven bound exceeds the whole lane's byte capacity.
    BoundExceedsCapacity {
        /// Bytes the caller asked to reserve.
        requested_bytes: usize,
        /// Bytes the lane can ever hold.
        capacity_bytes: usize,
    },
    /// The lane closed before this reservation completed.
    Closed,
    /// A drain owner is already running for this lane.
    DrainOwnerAlreadyStarted,
    /// No drain owner was started, so the lane can never drain.
    NoDrainOwner,
    /// `drain` was called before `close`, which would never terminate.
    DrainBeforeClose,
    /// Settlement measured more bytes than the validated reservation.
    ActualExceedsReservedBound {
        /// Bytes proven and reserved before dispatch.
        reserved_bytes: usize,
        /// Bytes actually retained by the terminal record.
        actual_bytes: usize,
    },
    /// Lane accounting could not represent a state transition.
    AccountingCorruption,
}

impl fmt::Display for TerminalLaneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "streaming terminal lane error: {self:?}")
    }
}

impl std::error::Error for TerminalLaneError {}

/// Current bounded terminal-lane resource use and progress.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalLaneSnapshot {
    /// Settled records waiting for the drain owner.
    pub queued_items: usize,
    /// Reserved-or-queued items currently charged.
    pub reserved_items: usize,
    /// Reserved-or-queued bytes currently charged.
    pub reserved_bytes: usize,
    /// Greatest observed item charge.
    pub high_water_items: usize,
    /// Greatest observed byte charge.
    pub high_water_bytes: usize,
    /// Drain owners ever started; the lane permits exactly one.
    pub drain_tasks_started: u64,
    /// Permits settled into the queue.
    pub submitted_records: u64,
    /// Records the drain owner ran every processor for.
    pub processed_records: u64,
    /// Ordinary processor or export faults observed.
    pub ordinary_processor_failures: u64,
    /// Ordinary faults the scoped reporter could not accept.
    pub unreported_issue_count: u64,
    /// First checked invariant, when one was latched.
    pub checked_invariant: Option<StreamingTerminalInvariant>,
}

/// Scoped reliability authority the drain owner reports ordinary faults through.
///
/// The lane never classifies. It submits typed ordinary facts and lets the host
/// reliability owner select the disposition.
pub struct TerminalLaneIssueScope {
    run: StreamRunIdentity,
    exporter_id: StreamingIssueComponentId,
    generation: CheckpointGeneration,
    semantic_context_digest: ContentDigest,
    reporter: StreamingIssueReporterHandle,
}

impl TerminalLaneIssueScope {
    /// Bind one lane to a run, exporter identity, generation, and reporter.
    #[must_use]
    pub fn new(
        run: StreamRunIdentity,
        exporter_id: StreamingIssueComponentId,
        generation: CheckpointGeneration,
        semantic_context_digest: ContentDigest,
        reporter: StreamingIssueReporterHandle,
    ) -> Self {
        Self {
            run,
            exporter_id,
            generation,
            semantic_context_digest,
            reporter,
        }
    }
}

impl fmt::Debug for TerminalLaneIssueScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TerminalLaneIssueScope")
            .field("exporter_id", &self.exporter_id)
            .finish_non_exhaustive()
    }
}

enum TerminalWorkPayload {
    Record {
        credit: Box<IssuedCredit>,
        outcome: Box<TurnDispatchOutcome>,
        request_id: Uuid,
    },
    /// Boundedness-only payload used by the lane's own test seam.
    Probe { index: u64 },
}

struct TerminalWork {
    payload: TerminalWorkPayload,
    /// Held for exactly as long as this record occupies the lane. `BudgetLease`
    /// returns the exact charge on drop, so nothing hand-releases capacity.
    _lease: BudgetLease,
}

struct LaneInner {
    limits: TerminalLaneLimits,
    budget: StreamingResourceBudget,
    processors: RefCell<Vec<Rc<dyn TurnRecordProcessor>>>,
    queue: RefCell<VecDeque<TerminalWork>>,
    scope: Option<TerminalLaneIssueScope>,
    work_ready: Notify,
    drain_complete: Notify,
    invariant_latched: Notify,
    is_closed: Cell<bool>,
    is_drained: Cell<bool>,
    drain_tasks_started: Cell<u64>,
    submitted_records: Cell<u64>,
    processed_records: Cell<u64>,
    ordinary_processor_failures: Cell<u64>,
    unreported_issue_count: Cell<u64>,
    checked_invariant: Cell<Option<StreamingTerminalInvariant>>,
}

impl LaneInner {
    fn snapshot(&self) -> TerminalLaneSnapshot {
        let budget = self.budget.snapshot();
        TerminalLaneSnapshot {
            queued_items: self.queue.borrow().len(),
            reserved_items: budget.used_items,
            reserved_bytes: budget.used_bytes,
            high_water_items: budget.high_water_items,
            high_water_bytes: budget.high_water_bytes,
            drain_tasks_started: self.drain_tasks_started.get(),
            submitted_records: self.submitted_records.get(),
            processed_records: self.processed_records.get(),
            ordinary_processor_failures: self.ordinary_processor_failures.get(),
            unreported_issue_count: self.unreported_issue_count.get(),
            checked_invariant: self.checked_invariant.get(),
        }
    }

    /// Latch the first checked invariant and wake the phase owner exactly once.
    fn latch_invariant(&self, invariant: StreamingTerminalInvariant) {
        if self.checked_invariant.get().is_some() {
            return;
        }
        self.checked_invariant.set(Some(invariant));
        tracing::error!(
            invariant = ?invariant,
            component = TERMINAL_LANE_COMPONENT_ID,
            "streaming terminal lane latched a checked invariant"
        );
        self.invariant_latched.notify_waiters();
    }

    fn count_ordinary_failure(&self) {
        self.ordinary_processor_failures
            .set(self.ordinary_processor_failures.get().saturating_add(1));
    }

    fn count_unreported_issue(&self) {
        self.unreported_issue_count
            .set(self.unreported_issue_count.get().saturating_add(1));
    }
}

/// Move-only proof that one terminal record's exact worst-case capacity is held.
///
/// A dropped permit returns the whole charge, which is how a cancelled dispatch
/// releases lane capacity without a bespoke cancellation path.
pub struct TerminalLanePermit {
    inner: Rc<LaneInner>,
    lease: BudgetLease,
    bound: TerminalRecordSizeBound,
}

impl fmt::Debug for TerminalLanePermit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TerminalLanePermit")
            .field("reserved_bytes", &self.bound.get())
            .finish_non_exhaustive()
    }
}

impl TerminalLanePermit {
    /// Return the proven bound this permit reserved.
    #[must_use]
    pub const fn bound(&self) -> TerminalRecordSizeBound {
        self.bound
    }

    /// Settle one terminal record into the lane.
    ///
    /// This measures the actual retained size, verifies it against the proven
    /// bound, shrinks the owned lease to the actual size, and moves the lease
    /// into the queued work. It never awaits and never fails for capacity.
    pub fn settle(
        self,
        credit: IssuedCredit,
        outcome: TurnDispatchOutcome,
    ) -> Result<(), TerminalLaneError> {
        let request_id = credit.turn.uuid;
        let actual_bytes = terminal_record_bytes(&credit, &outcome);
        self.settle_measured(
            TerminalWorkPayload::Record {
                credit: Box::new(credit),
                outcome: Box::new(outcome),
                request_id,
            },
            actual_bytes,
        )
    }

    /// Settle one boundedness-only record of an exact measured size.
    ///
    /// `TurnToSend` has a private field, so an out-of-crate test cannot build a
    /// real terminal record; this is the only way to drive the oversize branch
    /// from the integration suite.
    #[doc(hidden)]
    pub fn settle_measured_for_test(self, actual_bytes: usize) -> Result<(), TerminalLaneError> {
        self.settle_measured(TerminalWorkPayload::Probe { index: 0 }, actual_bytes)
    }

    fn settle_measured(
        mut self,
        payload: TerminalWorkPayload,
        actual_bytes: usize,
    ) -> Result<(), TerminalLaneError> {
        if actual_bytes > self.bound.get() {
            // The proof was wrong, which is an accounting fact about this run,
            // not an ordinary export fault. Latch and refuse; the lease is
            // returned in full by the drop below.
            self.inner
                .latch_invariant(StreamingTerminalInvariant::AccountingCorruption);
            return Err(TerminalLaneError::ActualExceedsReservedBound {
                reserved_bytes: self.bound.get(),
                actual_bytes,
            });
        }
        if let Err(error) = self.lease.shrink_to(1, actual_bytes) {
            self.inner
                .latch_invariant(StreamingTerminalInvariant::AccountingCorruption);
            tracing::error!(
                error = %error,
                component = TERMINAL_LANE_COMPONENT_ID,
                "terminal lease shrink failed after a validated reservation"
            );
            return Err(TerminalLaneError::AccountingCorruption);
        }
        let TerminalLanePermit { inner, lease, .. } = self;
        inner.queue.borrow_mut().push_back(TerminalWork {
            payload,
            _lease: lease,
        });
        inner
            .submitted_records
            .set(inner.submitted_records.get().saturating_add(1));
        inner.work_ready.notify_one();
        Ok(())
    }
}

/// Cloneable control surface for reservation, closure, drain, and inspection.
#[derive(Clone)]
pub struct TerminalLaneControl {
    inner: Rc<LaneInner>,
}

impl fmt::Debug for TerminalLaneControl {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TerminalLaneControl")
            .field("snapshot", &self.snapshot())
            .finish_non_exhaustive()
    }
}

impl TerminalLaneControl {
    /// Reserve one item plus the exact proven conservative maximum.
    ///
    /// This is the only waiting point in the lane. Waiting here is the lane's
    /// backpressure: an issuer cannot dispatch a request whose terminal record
    /// has nowhere to land.
    pub async fn reserve(
        &self,
        bound: TerminalRecordSizeBound,
    ) -> Result<TerminalLanePermit, TerminalLaneError> {
        if self.inner.is_closed.get() {
            return Err(TerminalLaneError::Closed);
        }
        if bound.get() > self.inner.limits.max_bytes {
            return Err(TerminalLaneError::BoundExceedsCapacity {
                requested_bytes: bound.get(),
                capacity_bytes: self.inner.limits.max_bytes,
            });
        }
        let lease = self
            .inner
            .budget
            .acquire(1, bound.get())
            .await
            .map_err(map_reservation_error)?;
        Ok(TerminalLanePermit {
            inner: Rc::clone(&self.inner),
            lease,
            bound,
        })
    }

    /// Stop admitting reservations and let the drain owner finish.
    pub fn close(&self) {
        if self.inner.is_closed.replace(true) {
            return;
        }
        // Closing the budget wakes every pending `reserve` with `Closed`.
        // Already-owned leases are unaffected; `BudgetLease::drop` still
        // returns their charge.
        self.inner.budget.close();
        self.inner.work_ready.notify_one();
    }

    /// Wait until the single drain owner has processed every settled record.
    pub async fn drain(&self) -> Result<(), TerminalLaneError> {
        if !self.inner.is_closed.get() {
            return Err(TerminalLaneError::DrainBeforeClose);
        }
        if self.inner.drain_tasks_started.get() == 0 {
            return Err(TerminalLaneError::NoDrainOwner);
        }
        loop {
            let completed = self.inner.drain_complete.notified();
            if self.inner.is_drained.get() {
                break;
            }
            completed.await;
        }
        let budget = self.inner.budget.snapshot();
        if budget.used_items != 0 || budget.used_bytes != 0 {
            self.inner
                .latch_invariant(StreamingTerminalInvariant::AccountingCorruption);
            tracing::error!(
                used_items = budget.used_items,
                used_bytes = budget.used_bytes,
                component = TERMINAL_LANE_COMPONENT_ID,
                "terminal lane retained capacity after a complete drain"
            );
        }
        if self.inner.submitted_records.get() != self.inner.processed_records.get() {
            self.inner
                .latch_invariant(StreamingTerminalInvariant::AccountingCorruption);
        }
        Ok(())
    }

    /// Wait until the lane latches a checked invariant.
    ///
    /// Only an accounting or authority contradiction resolves this. Ordinary
    /// processor and export faults never do.
    pub async fn wait_for_invariant(&self) -> StreamingTerminalInvariant {
        loop {
            let latched = self.inner.invariant_latched.notified();
            if let Some(invariant) = self.inner.checked_invariant.get() {
                return invariant;
            }
            latched.await;
        }
    }

    /// Return the first checked invariant, when one was latched.
    #[must_use]
    pub fn checked_invariant(&self) -> Option<StreamingTerminalInvariant> {
        self.inner.checked_invariant.get()
    }

    /// Read fixed-size lane counters.
    #[must_use]
    pub fn snapshot(&self) -> TerminalLaneSnapshot {
        self.inner.snapshot()
    }
}

/// Bounded owner of the terminal record-processing lane.
pub struct BoundedTerminalProcessorLane {
    inner: Rc<LaneInner>,
}

impl fmt::Debug for BoundedTerminalProcessorLane {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundedTerminalProcessorLane")
            .field("snapshot", &self.inner.snapshot())
            .finish_non_exhaustive()
    }
}

impl BoundedTerminalProcessorLane {
    /// Construct a lane bound to one scoped reliability reporter.
    pub fn new(
        limits: TerminalLaneLimits,
        scope: TerminalLaneIssueScope,
    ) -> Result<Self, TerminalLaneError> {
        Self::build(limits, Some(scope))
    }

    /// Construct a lane with no reliability scope, for boundedness tests.
    ///
    /// An ordinary processor fault is counted and traced but has nowhere to be
    /// reported, so production construction always supplies a scope.
    #[doc(hidden)]
    pub fn new_for_test(limits: TerminalLaneLimits) -> Result<Self, TerminalLaneError> {
        Self::build(limits, None)
    }

    fn build(
        limits: TerminalLaneLimits,
        scope: Option<TerminalLaneIssueScope>,
    ) -> Result<Self, TerminalLaneError> {
        let budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: limits.max_items,
            max_bytes: limits.max_bytes,
        })
        .map_err(TerminalLaneError::InvalidLimits)?;
        Ok(Self {
            inner: Rc::new(LaneInner {
                limits,
                budget,
                processors: RefCell::new(Vec::new()),
                queue: RefCell::new(VecDeque::with_capacity(limits.max_items)),
                scope,
                work_ready: Notify::new(),
                drain_complete: Notify::new(),
                invariant_latched: Notify::new(),
                is_closed: Cell::new(false),
                is_drained: Cell::new(false),
                drain_tasks_started: Cell::new(0),
                submitted_records: Cell::new(0),
                processed_records: Cell::new(0),
                ordinary_processor_failures: Cell::new(0),
                unreported_issue_count: Cell::new(0),
                checked_invariant: Cell::new(None),
            }),
        })
    }

    /// Attach one terminal record processor before the drain owner starts.
    pub fn add_processor(&self, processor: Rc<dyn TurnRecordProcessor>) {
        self.inner.processors.borrow_mut().push(processor);
    }

    /// Return the cloneable control surface.
    #[must_use]
    pub fn control(&self) -> TerminalLaneControl {
        TerminalLaneControl {
            inner: Rc::clone(&self.inner),
        }
    }

    /// Read fixed-size lane counters.
    #[must_use]
    pub fn snapshot(&self) -> TerminalLaneSnapshot {
        self.inner.snapshot()
    }

    /// Start the single `spawn_local` drain owner for this lane.
    ///
    /// The processor list is frozen into an `Rc<[_]>` here so the drain loop
    /// pays no per-record allocation, and the returned `JoinHandle` is
    /// deliberately dropped: completion is observed through
    /// [`TerminalLaneControl::drain`], never through a retained handle.
    pub fn start_local_drain(&self) -> Result<(), TerminalLaneError> {
        if self.inner.drain_tasks_started.get() != 0 {
            return Err(TerminalLaneError::DrainOwnerAlreadyStarted);
        }
        self.inner.drain_tasks_started.set(1);
        let frozen: Rc<[Rc<dyn TurnRecordProcessor>]> =
            Rc::from(self.inner.processors.borrow().as_slice());
        let inner = Rc::clone(&self.inner);
        drop(tokio::task::spawn_local(async move {
            drain_loop(inner, frozen).await
        }));
        Ok(())
    }

    /// Reserve one item plus the exact proven conservative maximum.
    pub async fn reserve(
        &self,
        bound: TerminalRecordSizeBound,
    ) -> Result<TerminalLanePermit, TerminalLaneError> {
        self.control().reserve(bound).await
    }

    /// Reserve and settle one boundedness-only record.
    #[doc(hidden)]
    pub async fn submit_test_terminal(
        &self,
        index: u64,
        bytes: usize,
    ) -> Result<(), TerminalLaneError> {
        let bound = TerminalRecordSizeBound::try_from_bytes(bytes)
            .ok_or(TerminalLaneError::AccountingCorruption)?;
        let permit = self.reserve(bound).await?;
        permit.settle_measured(TerminalWorkPayload::Probe { index }, bytes)
    }
}

async fn drain_loop(inner: Rc<LaneInner>, processors: Rc<[Rc<dyn TurnRecordProcessor>]>) {
    loop {
        // Register the wake-up BEFORE inspecting the queue so a settle that
        // lands between the check and the await cannot be lost.
        let ready = inner.work_ready.notified();
        let work = inner.queue.borrow_mut().pop_front();
        match work {
            Some(work) => process_one(&inner, &processors, work).await,
            None => {
                if inner.is_closed.get() {
                    break;
                }
                ready.await;
            }
        }
    }
    inner.is_drained.set(true);
    inner.drain_complete.notify_waiters();
}

async fn process_one(
    inner: &Rc<LaneInner>,
    processors: &Rc<[Rc<dyn TurnRecordProcessor>]>,
    work: TerminalWork,
) {
    match &work.payload {
        TerminalWorkPayload::Record {
            credit,
            outcome,
            request_id,
        } => {
            for processor in processors.iter() {
                if let Err(error) = processor.process(credit, outcome).await {
                    inner.count_ordinary_failure();
                    tracing::debug!(
                        uuid = %request_id,
                        error = %error,
                        component = TERMINAL_LANE_COMPONENT_ID,
                        "terminal record processor reported an ordinary fault"
                    );
                    report_ordinary_export_fault(inner).await;
                }
            }
        }
        TerminalWorkPayload::Probe { index } => {
            tracing::trace!(
                index,
                component = TERMINAL_LANE_COMPONENT_ID,
                "terminal lane drained a boundedness probe"
            );
        }
    }
    inner
        .processed_records
        .set(inner.processed_records.get().saturating_add(1));
    // Explicit: the lease inside `work` returns its exact charge here, which is
    // what unblocks the next reservation.
    drop(work);
}

/// Report one ordinary export fault through the scoped Task 1D-R reporter.
///
/// This can never latch the checked invariant. `Backpressured`, `Closed`, and
/// `InvalidIssue` are all counted and traced; the drain continues in every case.
async fn report_ordinary_export_fault(inner: &Rc<LaneInner>) {
    let Some(scope) = inner.scope.as_ref() else {
        inner.count_unreported_issue();
        return;
    };
    let issue = match OrdinaryStreamingIssue::export(
        scope.run,
        scope.exporter_id.clone(),
        scope.generation.clone(),
        StreamingIssueClass::Retryable,
        scope.semantic_context_digest,
        0,
        scope.semantic_context_digest,
        OrdinaryStreamingFailure::Export(ResultExportError::failure(
            ResultExportFailureCode::Attempt,
        )),
    ) {
        Ok(issue) => issue,
        Err(error) => {
            inner.count_unreported_issue();
            tracing::debug!(
                error = %error,
                component = TERMINAL_LANE_COMPONENT_ID,
                "terminal lane could not construct an ordinary export issue"
            );
            return;
        }
    };
    match scope.reporter.report(issue).await {
        Ok(StreamingIssueReportStatus::Accepted) => {}
        Ok(StreamingIssueReportStatus::Backpressured) => {
            inner.count_unreported_issue();
            tracing::trace!(
                component = TERMINAL_LANE_COMPONENT_ID,
                "reliability reporter backpressured a terminal export issue"
            );
        }
        Err(StreamingIssueReportError::Closed) => {
            inner.count_unreported_issue();
            tracing::debug!(
                component = TERMINAL_LANE_COMPONENT_ID,
                "reliability reporter closed before a terminal export issue"
            );
        }
        Err(StreamingIssueReportError::InvalidIssue) => {
            inner.count_unreported_issue();
            tracing::debug!(
                component = TERMINAL_LANE_COMPONENT_ID,
                "reliability reporter rejected a terminal export issue"
            );
        }
    }
}

fn map_reservation_error(error: BudgetError) -> TerminalLaneError {
    match error {
        BudgetError::Closed => TerminalLaneError::Closed,
        BudgetError::ZeroCapacity
        | BudgetError::PermitCountTooLarge
        | BudgetError::RequestExceedsCapacity => TerminalLaneError::InvalidLimits(error),
        BudgetError::CapacityUnavailable
        | BudgetError::CannotGrowLease
        | BudgetError::InvalidFragmentItemCharge { .. }
        | BudgetError::ActionPayloadUndercharged { .. }
        | BudgetError::PartialLeasedBuffer { .. }
        | BudgetError::AccountingOverflow => TerminalLaneError::AccountingCorruption,
    }
}

/// Measure the exact bytes one terminal record retains inside the lane.
///
/// This walks only owned heap the queued clone keeps alive. It allocates
/// nothing.
#[must_use]
pub fn terminal_record_bytes(credit: &IssuedCredit, outcome: &TurnDispatchOutcome) -> usize {
    let model = &outcome.model_response;
    let mut bytes = TERMINAL_RECORD_FIXED_BYTES;
    bytes = bytes.saturating_add(outcome.response_text.len());
    bytes = bytes.saturating_add(option_string_bytes(model.content.as_ref()));
    bytes = bytes.saturating_add(option_string_bytes(model.reasoning.as_ref()));
    bytes = bytes.saturating_add(option_string_bytes(model.response_id.as_ref()));
    bytes = bytes.saturating_add(option_string_bytes(model.finish_reason.as_ref()));
    bytes = bytes.saturating_add(option_string_bytes(model.error_kind.as_ref()));
    bytes = bytes.saturating_add(option_string_bytes(model.error_message.as_ref()));
    bytes = bytes.saturating_add(
        model
            .output_token_ids
            .as_ref()
            .map_or(0, |ids| ids.len().saturating_mul(size_of::<u32>())),
    );
    bytes = bytes.saturating_add(model.assistant_message.as_ref().map_or(0, json_value_bytes));
    bytes = bytes.saturating_add(credit.turn.conversation_id.len());
    bytes = bytes.saturating_add(credit.turn.x_correlation_id.len());
    bytes = bytes.saturating_add(credit.turn.request_correlation_id.len());
    bytes
}

fn option_string_bytes(value: Option<&String>) -> usize {
    value.map_or(0, String::len)
}

/// Measure one retained JSON value.
///
/// Depth is bounded by the endpoint parser that produced the value, so the
/// recursion terminates on every value this lane can observe.
fn json_value_bytes(value: &Value) -> usize {
    let mut bytes = JSON_NODE_FIXED_BYTES;
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
        Value::String(text) => bytes = bytes.saturating_add(text.len()),
        Value::Array(items) => {
            for item in items {
                bytes = bytes.saturating_add(json_value_bytes(item));
            }
        }
        Value::Object(entries) => {
            for (key, item) in entries {
                bytes = bytes.saturating_add(key.len());
                bytes = bytes.saturating_add(json_value_bytes(item));
            }
        }
    }
    bytes
}
