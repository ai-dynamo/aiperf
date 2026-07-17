// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-ready multi-trace Graph-IR workload policy around the one executor.
//!
//! Arrival pacing, root-session admission, phase observation, node policy, and
//! run-failure behavior are independent traits. Every admitted trace still
//! dispatches exclusively through [`crate::graph::executor::TraceExecutor`]; this
//! module does not implement a second benchmark or backend path.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{self, Display};
use std::rc::Rc;

use crate::clock::Clock;
use crate::timing::{
    FirstArrival, IntervalGenerator, SlotGuard, SlotPool, WhenBehind, next_arrival_target,
};
use async_trait::async_trait;

use crate::graph::errors::TraceError;
use crate::graph::execution::TracePlacement;
pub use crate::graph::model::GraphTracePlan;
use crate::graph::policy::{ContinueRunFailurePolicy, RunFailurePolicy};
use crate::graph::tstar::PermutationDraw;

/// Stateful root-trace selection seam.
pub trait GraphTraceSource {
    /// Return the next trace plan, or `None` when sending is complete.
    fn next_trace(&self) -> Result<Option<GraphTracePlan>, GraphWorkloadError>;
}

/// Authored-order finite trace source.
pub struct VecGraphTraceSource {
    plans: RefCell<VecDeque<GraphTracePlan>>,
}

impl VecGraphTraceSource {
    /// Construct from plans in desired admission order.
    pub fn new(plans: impl IntoIterator<Item = GraphTracePlan>) -> Self {
        Self {
            plans: RefCell::new(plans.into_iter().collect()),
        }
    }
}

impl GraphTraceSource for VecGraphTraceSource {
    fn next_trace(&self) -> Result<Option<GraphTracePlan>, GraphWorkloadError> {
        Ok(self.plans.borrow_mut().pop_front())
    }
}

/// Sequential root-template cycling with unique execution-instance identities.
///
/// This source gives bounded phase request/session counts their ordinary
/// resampling behavior without reparsing or relowering the authored DAG. Both
/// budgets may be absent when a duration policy owns termination.
pub struct CyclingGraphTraceSource {
    templates: Vec<GraphTracePlan>,
    session_limit: Option<u64>,
    request_limit: Option<u64>,
    next: Cell<u64>,
    admitted_requests: Cell<u64>,
    instance_sequence: GraphTraceInstanceSequence,
    /// Corpus draw offset inherited from an earlier phase.
    start_ordinal: u64,
    /// Sampling strategy used to map draw ordinals to templates.
    draw: PermutationDraw,
}

/// Run-scoped identity sequence shared by independently prepared graph phases.
///
/// Phase-local session and request budgets remain independent, while every
/// emitted root instance receives one run-unique ordinal for correlation and
/// metric identity.
#[derive(Clone, Default)]
pub struct GraphTraceInstanceSequence {
    next: Rc<Cell<u64>>,
}

impl GraphTraceInstanceSequence {
    fn take(&self) -> Result<u64, GraphWorkloadError> {
        let ordinal = self.next.get();
        self.next.set(ordinal.checked_add(1).ok_or_else(|| {
            GraphWorkloadError("graph trace instance identity exhausted u64".into())
        })?);
        Ok(ordinal)
    }
}

impl CyclingGraphTraceSource {
    /// Construct a sequential cycle over at least one root template.
    pub fn new(
        templates: Vec<GraphTracePlan>,
        session_limit: Option<u64>,
    ) -> Result<Self, GraphWorkloadError> {
        Self::with_budgets(templates, session_limit, None)
    }

    /// Construct with independent whole-trace session and static-node budgets.
    ///
    /// The next authored template is never split or skipped: if its static node
    /// count would exceed `request_limit`, the finite source is exhausted.
    pub fn with_budgets(
        templates: Vec<GraphTracePlan>,
        session_limit: Option<u64>,
        request_limit: Option<u64>,
    ) -> Result<Self, GraphWorkloadError> {
        Self::with_budgets_and_sequence(
            templates,
            session_limit,
            request_limit,
            GraphTraceInstanceSequence::default(),
        )
    }

    /// Construct with budgets and a run-scoped instance identity sequence.
    pub fn with_budgets_and_sequence(
        templates: Vec<GraphTracePlan>,
        session_limit: Option<u64>,
        request_limit: Option<u64>,
        instance_sequence: GraphTraceInstanceSequence,
    ) -> Result<Self, GraphWorkloadError> {
        if templates.is_empty() {
            return Err(GraphWorkloadError(
                "graph trace cycling requires at least one root template".into(),
            ));
        }
        if session_limit == Some(0) || request_limit == Some(0) {
            return Err(GraphWorkloadError(
                "graph trace session/request budgets must be positive when configured".into(),
            ));
        }
        Ok(Self {
            templates,
            session_limit,
            request_limit,
            next: Cell::new(0),
            admitted_requests: Cell::new(0),
            instance_sequence,
            start_ordinal: 0,
            draw: PermutationDraw::sequential(),
        })
    }

    /// Resume the corpus draw from `start_ordinal` (the handoff `corpus_cursor`).
    ///
    /// The first `next_trace` then serves template `start_ordinal % len` and the
    /// cycle continues from there, so the bounded profiling recycle does not
    /// re-serve the templates a cache-pressure warmup already consumed. Session
    /// and static-request budgets are unaffected because they count from `0`.
    pub fn starting_at(mut self, start_ordinal: u64) -> Self {
        self.start_ordinal = start_ordinal;
        self
    }

    /// Apply `draw` when mapping recycle ordinals to templates.
    pub fn with_sampling(mut self, draw: PermutationDraw) -> Self {
        self.draw = draw;
        self
    }
}

impl GraphTraceSource for CyclingGraphTraceSource {
    fn next_trace(&self) -> Result<Option<GraphTracePlan>, GraphWorkloadError> {
        let ordinal = self.next.get();
        if self.session_limit.is_some_and(|limit| ordinal >= limit) {
            return Ok(None);
        }
        // The corpus draw position is the session ordinal shifted by the handoff
        // resume cursor; the session-limit gate above stays on the raw ordinal, so
        // the cursor moves only which template each draw picks.
        let draw = ordinal
            .checked_add(self.start_ordinal)
            .ok_or_else(|| GraphWorkloadError("graph resumed draw ordinal exceeds u64".into()))?;
        // The remap changes template selection without changing draw counters.
        let template_index = self.draw.index(draw, self.templates.len());
        let mut plan = self.templates[template_index].clone();
        let requests = u64::try_from(plan.graph.nodes.len()).map_err(|_| {
            GraphWorkloadError("graph template static node count exceeds u64".into())
        })?;
        let admitted_requests = self.admitted_requests.get();
        let next_requests = admitted_requests.checked_add(requests).ok_or_else(|| {
            GraphWorkloadError("graph admitted static node count exceeds u64".into())
        })?;
        if self
            .request_limit
            .is_some_and(|limit| next_requests > limit)
        {
            return Ok(None);
        }
        let next_ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| GraphWorkloadError("graph admitted root count exceeds u64".into()))?;
        let instance = self.instance_sequence.take()?;
        plan.trace.id = format!("{}::instance-{instance}", plan.trace.id);
        self.next.set(next_ordinal);
        self.admitted_requests.set(next_requests);
        Ok(Some(plan))
    }
}

/// A cell-partitioned graph trace source: cell `cell_id` of `cell_count` owns the
/// interleaved GLOBAL session ordinals `cell_id, cell_id + cell_count, cell_id + 2·C, …`
/// and cycles the root templates by that global ordinal. The union across all cells
/// therefore reproduces a single-cell run's trace set and per-template distribution
/// (the design's *deterministic-per-topology* contract), and each trace's
/// globally-unique ordinal rides its `trace.id` (`"{template}::instance-{global}"`) so
/// the controller can merge cells' records in one global order — the graph analogue of
/// the scheduled path's `CellularAutonomousIssuer` absolute slot (`base + within·C + id`).
///
/// One formula covers both partition modes the runtime needs:
/// - **finite** (SharedIterations, e.g. `--num-conversations`): a `session_limit` bounds
///   the global ordinal, so each cell stops once its interleave passes the shared cap;
/// - **unbounded sampler-loop** (duration-driven GraphAgentic): `session_limit = None`
///   and a phase duration policy owns termination.
///
/// Ownership uses deterministic modulo interleaving of global trace ordinals.
/// The partition applies to session limits; static-node `request_limit` values
/// are not partitioned.
pub struct PartitionedGraphTraceSource {
    templates: Vec<GraphTracePlan>,
    session_limit: Option<u64>,
    cell_id: u64,
    cell_count: u64,
    next_local: Cell<u64>,
    /// Strategy-aware corpus-index remap keyed on the GLOBAL session ordinal.
    ///
    /// `Sequential` (the default) is `global_ordinal % len`. Under
    /// `Shuffle`/`Random` each cell draws
    /// `epoch[global / len][global % len]`; because the union of all cells' global
    /// ordinals is the contiguous `0..N`, each persistent-epoch pass is still
    /// covered exactly once across the cells, so the deterministic-per-topology
    /// cover-the-corpus-once contract holds and equals a single-cell cycling run
    /// under the identical draw.
    draw: PermutationDraw,
}

impl PartitionedGraphTraceSource {
    /// Construct a partitioned cycle for cell `cell_id` of `cell_count`
    /// (`cell_count >= 1`, `cell_id < cell_count`). `session_limit` bounds the GLOBAL
    /// session ordinal (the same cap a 1-cell run would use), or `None` for the
    /// unbounded duration-driven case. `cell_count == 1` reproduces
    /// [`CyclingGraphTraceSource`] exactly.
    pub fn new(
        templates: Vec<GraphTracePlan>,
        session_limit: Option<u64>,
        cell_id: u32,
        cell_count: u32,
    ) -> Result<Self, GraphWorkloadError> {
        if templates.is_empty() {
            return Err(GraphWorkloadError(
                "graph trace cycling requires at least one root template".into(),
            ));
        }
        if cell_count == 0 || cell_id >= cell_count {
            return Err(GraphWorkloadError(
                "graph cell partition requires cell_count >= 1 and cell_id < cell_count".into(),
            ));
        }
        if session_limit == Some(0) {
            return Err(GraphWorkloadError(
                "graph trace session budget must be positive when configured".into(),
            ));
        }
        Ok(Self {
            templates,
            session_limit,
            cell_id: u64::from(cell_id),
            cell_count: u64::from(cell_count),
            next_local: Cell::new(0),
            draw: PermutationDraw::sequential(),
        })
    }

    /// Route the interleave template pick through a resolved strategy-aware draw.
    ///
    /// See [`CyclingGraphTraceSource::with_sampling`]: `Sequential` is the
    /// byte-unchanged `global_ordinal % len` pick; `Shuffle`/`Random` picks the
    /// same draw pressure warmup and the single-cell cycler use, keyed
    /// on the global ordinal so the per-topology cover-the-corpus-once union is
    /// preserved (`Shuffle`; `Random` is with replacement, so the union matches a
    /// single-cell run but need not cover every template each pass).
    pub fn with_sampling(mut self, draw: PermutationDraw) -> Self {
        self.draw = draw;
        self
    }
}

impl GraphTraceSource for PartitionedGraphTraceSource {
    fn next_trace(&self) -> Result<Option<GraphTracePlan>, GraphWorkloadError> {
        let local = self.next_local.get();
        let global_ordinal = local
            .checked_mul(self.cell_count)
            .and_then(|scaled| scaled.checked_add(self.cell_id))
            .ok_or_else(|| GraphWorkloadError("graph partitioned ordinal exceeds u64".into()))?;
        if self
            .session_limit
            .is_some_and(|limit| global_ordinal >= limit)
        {
            return Ok(None);
        }
        let template_index = self.draw.index(global_ordinal, self.templates.len());
        let mut plan = self.templates[template_index].clone();
        plan.trace.id = format!("{}::instance-{global_ordinal}", plan.trace.id);
        self.next_local.set(local.checked_add(1).ok_or_else(|| {
            GraphWorkloadError("graph partitioned local ordinal exceeds u64".into())
        })?);
        Ok(Some(plan))
    }
}

/// Arrival-pacing extension point.
#[async_trait(?Send)]
pub trait GraphArrivalPolicy {
    /// Wait until `plan` may arrive. `run_start_ns` anchors authored offsets.
    async fn wait_for_arrival(
        &self,
        clock: Rc<dyn Clock>,
        run_start_ns: i64,
        ordinal: u64,
        plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError>;
}

/// Run-level admission stop policy owned by the graph coordinator.
pub trait GraphStopPolicy {
    /// Optional absolute stop deadline derived from the workload start.
    fn deadline_ns(&self, run_start_ns: i64) -> Option<i64>;
}

/// No run-level admission deadline.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnlimitedGraphStop;

impl GraphStopPolicy for UnlimitedGraphStop {
    fn deadline_ns(&self, _run_start_ns: i64) -> Option<i64> {
        None
    }
}

/// Workload-relative duration bound that stops new roots and drains active traces.
#[derive(Debug, Clone, Copy)]
pub struct DurationGraphStop {
    duration_ns: i64,
}

impl DurationGraphStop {
    /// Construct a non-negative duration admission bound.
    pub fn new(duration_ns: i64) -> Result<Self, GraphWorkloadError> {
        if duration_ns < 0 {
            return Err(GraphWorkloadError(
                "graph stop duration must be non-negative".into(),
            ));
        }
        Ok(Self { duration_ns })
    }
}

impl GraphStopPolicy for DurationGraphStop {
    fn deadline_ns(&self, run_start_ns: i64) -> Option<i64> {
        Some(run_start_ns.saturating_add(self.duration_ns))
    }
}

/// Immediate arrivals; session capacity governs throughput.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImmediateGraphArrival;

#[async_trait(?Send)]
impl GraphArrivalPolicy for ImmediateGraphArrival {
    async fn wait_for_arrival(
        &self,
        _clock: Rc<dyn Clock>,
        _run_start_ns: i64,
        _ordinal: u64,
        _plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError> {
        Ok(())
    }
}

/// Arrival policy that schedules each trace at its authored offset from the run start.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScheduledGraphArrival;

#[async_trait(?Send)]
impl GraphArrivalPolicy for ScheduledGraphArrival {
    async fn wait_for_arrival(
        &self,
        clock: Rc<dyn Clock>,
        run_start_ns: i64,
        _ordinal: u64,
        plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError> {
        let Some(offset) = plan.arrival_offset_ns else {
            return Ok(());
        };
        if offset < 0 {
            return Err(GraphWorkloadError(format!(
                "trace {:?} has negative arrival offset {offset}ns",
                plan.trace.id
            )));
        }
        let target = run_start_ns.saturating_add(offset);
        let delay_ns = target.saturating_sub(clock.now_ns());
        clock.sleep(delay_ns).await;
        Ok(())
    }
}

/// Interval-generator arrivals whose live rate is shared with ramp/adaptive controls.
pub struct IntervalGraphArrival {
    generator: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    next_at_ns: Cell<Option<i64>>,
}

impl IntervalGraphArrival {
    /// Bind to a live interval generator.
    pub fn new(generator: Rc<RefCell<Box<dyn IntervalGenerator>>>) -> Self {
        Self {
            generator,
            next_at_ns: Cell::new(None),
        }
    }

    /// Clone the generator handle used by request-rate ramp/adaptive actuators.
    pub fn generator(&self) -> Rc<RefCell<Box<dyn IntervalGenerator>>> {
        self.generator.clone()
    }
}

#[async_trait(?Send)]
impl GraphArrivalPolicy for IntervalGraphArrival {
    async fn wait_for_arrival(
        &self,
        clock: Rc<dyn Clock>,
        run_start_ns: i64,
        ordinal: u64,
        _plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError> {
        // The first arrival targets run_start; subsequent targets remain anchored
        // to the prior target.
        let prev = if ordinal == 0 {
            None
        } else {
            Some(self.next_at_ns.get().unwrap_or(run_start_ns))
        };
        let target = next_arrival_target(
            prev,
            run_start_ns,
            clock.now_ns(),
            FirstArrival::AtStart,
            WhenBehind::KeepAbsolute,
            || self.generator.borrow_mut().next_interval_ns().max(0),
        );
        self.next_at_ns.set(Some(target));
        let delay_ns = target.saturating_sub(clock.now_ns());
        clock.sleep(delay_ns).await;
        Ok(())
    }
}

/// Immutable context supplied to root-session admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceAdmissionInfo {
    /// Trace identifier.
    pub trace_id: String,
    /// Static node/request count in the trace.
    pub node_count: usize,
    /// Clock time at which arrival pacing completed.
    pub arrival_ns: i64,
}

/// Permit held for the complete root trace, including every child node.
pub trait TraceAdmissionPermit {}

/// Root-session admission seam. DAG children never reacquire this permit.
#[async_trait(?Send)]
pub trait TraceAdmissionPolicy {
    /// Acquire capacity for one whole trace.
    async fn acquire(
        &self,
        info: &TraceAdmissionInfo,
    ) -> Result<Box<dyn TraceAdmissionPermit>, GraphWorkloadError>;
}

/// Admission policy with no cap.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnlimitedTraceAdmission;

struct UnlimitedTracePermit;

impl TraceAdmissionPermit for UnlimitedTracePermit {}

#[async_trait(?Send)]
impl TraceAdmissionPolicy for UnlimitedTraceAdmission {
    async fn acquire(
        &self,
        _info: &TraceAdmissionInfo,
    ) -> Result<Box<dyn TraceAdmissionPermit>, GraphWorkloadError> {
        Ok(Box::new(UnlimitedTracePermit))
    }
}

/// Dynamic root-session cap over the shared timing [`SlotPool`].
pub struct SlotPoolTraceAdmission {
    pool: Rc<SlotPool>,
}

impl SlotPoolTraceAdmission {
    /// Bind whole-trace admission to a live pool.
    pub fn new(pool: Rc<SlotPool>) -> Self {
        Self { pool }
    }

    /// Clone the pool used by phase resources, ramps, and adaptive control.
    pub fn pool(&self) -> Rc<SlotPool> {
        self.pool.clone()
    }
}

struct SlotPoolTracePermit {
    _guard: SlotGuard,
}

impl TraceAdmissionPermit for SlotPoolTracePermit {}

#[async_trait(?Send)]
impl TraceAdmissionPolicy for SlotPoolTraceAdmission {
    async fn acquire(
        &self,
        _info: &TraceAdmissionInfo,
    ) -> Result<Box<dyn TraceAdmissionPermit>, GraphWorkloadError> {
        Ok(Box::new(SlotPoolTracePermit {
            _guard: self.pool.acquire().await,
        }))
    }
}

/// Phase/run hooks emitted by the graph workload.
pub trait GraphWorkloadObserver {
    /// Arrival pacing completed.
    fn on_trace_arrival(&self, _info: &TraceAdmissionInfo) {}
    /// Root-session admission completed.
    fn on_trace_admit(&self, _info: &TraceAdmissionInfo, _admit_ns: i64) {}
    /// One trace drained through the executor.
    fn on_trace_complete(&self, _result: &GraphTraceRunResult) {}
    /// The finite source stopped producing new roots.
    fn on_sending_complete(&self, _at_ns: i64) {}
}

/// No-op workload observer.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopGraphWorkloadObserver;

impl GraphWorkloadObserver for NoopGraphWorkloadObserver {}

/// One drained trace outcome.
#[derive(Debug, Clone)]
pub struct GraphTraceRunResult {
    /// Trace identifier.
    pub trace_id: String,
    /// Success or trace-aborting failure.
    pub result: Result<(), TraceError>,
}

/// Aggregate workload outcome.
#[derive(Debug, Clone, Default)]
pub struct GraphWorkloadReport {
    /// Root traces that acquired session admission.
    pub admitted: u64,
    /// Successfully drained traces.
    pub completed: u64,
    /// Traces terminated by configured or phase-driven cancellation.
    pub cancelled: u64,
    /// Traces that aborted.
    pub failed: u64,
    /// Results in completion order.
    pub traces: Vec<GraphTraceRunResult>,
}

/// Policy-composed coordinator delegating complete traces to one backend.
pub struct GraphWorkload {
    clock: Rc<dyn Clock>,
    source: Rc<dyn GraphTraceSource>,
    arrival: Rc<dyn GraphArrivalPolicy>,
    admission: Rc<dyn TraceAdmissionPolicy>,
    stop: Rc<dyn GraphStopPolicy>,
    backend: Rc<dyn TracePlacement>,
    run_failure: Rc<dyn RunFailurePolicy>,
    observer: Rc<dyn GraphWorkloadObserver>,
    cancelled: Rc<Cell<bool>>,
}

impl GraphWorkload {
    /// Construct the default immediate/unlimited/resilient workload.
    pub fn new(
        clock: Rc<dyn Clock>,
        source: Rc<dyn GraphTraceSource>,
        backend: Rc<dyn TracePlacement>,
    ) -> Self {
        Self {
            clock,
            source,
            arrival: Rc::new(ImmediateGraphArrival),
            admission: Rc::new(UnlimitedTraceAdmission),
            stop: Rc::new(UnlimitedGraphStop),
            backend,
            run_failure: Rc::new(ContinueRunFailurePolicy),
            observer: Rc::new(NoopGraphWorkloadObserver),
            cancelled: Rc::new(Cell::new(false)),
        }
    }

    /// Inject arrival pacing.
    pub fn with_arrival(mut self, arrival: Rc<dyn GraphArrivalPolicy>) -> Self {
        self.arrival = arrival;
        self
    }

    /// Inject root-session admission.
    pub fn with_admission(mut self, admission: Rc<dyn TraceAdmissionPolicy>) -> Self {
        self.admission = admission;
        self
    }

    /// Inject a run-level admission stop policy.
    pub fn with_stop_policy(mut self, stop: Rc<dyn GraphStopPolicy>) -> Self {
        self.stop = stop;
        self
    }

    /// Inject run-level admission-after-failure behavior.
    pub fn with_run_failure(mut self, policy: Rc<dyn RunFailurePolicy>) -> Self {
        self.run_failure = policy;
        self
    }

    /// Inject phase/report observation.
    pub fn with_observer(mut self, observer: Rc<dyn GraphWorkloadObserver>) -> Self {
        self.observer = observer;
        self
    }

    /// Stop admitting new traces. Existing traces drain through their sink.
    pub fn cancel(&self) {
        self.cancelled.set(true);
    }

    /// Whether external cancellation has latched.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.get()
    }

    /// Execute all admitted traces on the caller's current-thread `LocalSet`.
    pub async fn execute(&self) -> Result<GraphWorkloadReport, GraphWorkloadError> {
        let run_start_ns = self.clock.now_ns();
        let deadline_ns = self.stop.deadline_ns(run_start_ns);
        let (completed_tx, mut completed_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut active = 0_u64;
        let mut admitted = 0_u64;
        let mut ordinal = 0_u64;
        let mut report = GraphWorkloadReport::default();

        loop {
            self.drain_ready_results(&mut completed_rx, &mut active, &mut report);
            if self.cancelled.get()
                || !self.run_failure.may_admit()
                || deadline_ns.is_some_and(|deadline| self.clock.now_ns() >= deadline)
            {
                break;
            }
            let Some(plan) = self.source.next_trace()? else {
                break;
            };
            {
                let arrival =
                    self.arrival
                        .wait_for_arrival(self.clock.clone(), run_start_ns, ordinal, &plan);
                tokio::pin!(arrival);
                if let Some(deadline) = deadline_ns {
                    let stop = self
                        .clock
                        .clone()
                        .sleep(deadline.saturating_sub(self.clock.now_ns()));
                    tokio::pin!(stop);
                    tokio::select! {
                        biased;
                        () = &mut stop => break,
                        result = &mut arrival => result?,
                    }
                } else {
                    arrival.as_mut().await?;
                }
            }
            ordinal = ordinal.saturating_add(1);
            self.drain_ready_results(&mut completed_rx, &mut active, &mut report);
            if self.cancelled.get()
                || !self.run_failure.may_admit()
                || deadline_ns.is_some_and(|deadline| self.clock.now_ns() >= deadline)
            {
                break;
            }

            let info = TraceAdmissionInfo {
                trace_id: plan.trace.id.clone(),
                node_count: plan.graph.nodes.len(),
                arrival_ns: self.clock.now_ns(),
            };
            self.observer.on_trace_arrival(&info);
            let acquire = self.admission.acquire(&info);
            tokio::pin!(acquire);
            let permit = if let Some(deadline) = deadline_ns {
                let stop = self
                    .clock
                    .clone()
                    .sleep(deadline.saturating_sub(self.clock.now_ns()));
                tokio::pin!(stop);
                tokio::select! {
                    biased;
                    () = &mut stop => break,
                    permit = &mut acquire => permit?,
                }
            } else {
                acquire.await?
            };
            self.drain_ready_results(&mut completed_rx, &mut active, &mut report);
            if self.cancelled.get()
                || !self.run_failure.may_admit()
                || deadline_ns.is_some_and(|deadline| self.clock.now_ns() >= deadline)
            {
                drop(permit);
                break;
            }
            self.observer.on_trace_admit(&info, self.clock.now_ns());
            admitted = admitted.saturating_add(1);
            active = active.saturating_add(1);

            let trace_id = plan.trace.id.clone();
            let backend = self.backend.clone();
            let observer = self.observer.clone();
            let run_failure = self.run_failure.clone();
            let completed_tx = completed_tx.clone();
            tokio::task::spawn_local(async move {
                let result = backend.execute_trace(plan).await;
                run_failure.on_trace_result(&trace_id, &result);
                let outcome = GraphTraceRunResult { trace_id, result };
                observer.on_trace_complete(&outcome);
                let _ = completed_tx.send(outcome);
                drop(permit);
            });

            // Let a same-instant failure latch before another burst admission.
            tokio::task::yield_now().await;
        }

        self.observer.on_sending_complete(self.clock.now_ns());
        while active > 0 {
            let outcome = completed_rx.recv().await.ok_or_else(|| {
                GraphWorkloadError("trace completion channel closed with active work".into())
            })?;
            active -= 1;
            push_result(&mut report, outcome);
        }
        report.admitted = admitted;
        Ok(report)
    }

    fn drain_ready_results(
        &self,
        completed_rx: &mut tokio::sync::mpsc::UnboundedReceiver<GraphTraceRunResult>,
        active: &mut u64,
        report: &mut GraphWorkloadReport,
    ) {
        while let Ok(outcome) = completed_rx.try_recv() {
            *active = active.saturating_sub(1);
            push_result(report, outcome);
        }
    }
}

fn push_result(report: &mut GraphWorkloadReport, outcome: GraphTraceRunResult) {
    match &outcome.result {
        Ok(()) => report.completed = report.completed.saturating_add(1),
        Err(TraceError::Cancelled(_)) => {
            report.cancelled = report.cancelled.saturating_add(1);
        }
        Err(_) => report.failed = report.failed.saturating_add(1),
    }
    report.traces.push(outcome);
}

/// Workload/source/admission error outside an individual trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphWorkloadError(pub String);

impl Display for GraphWorkloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for GraphWorkloadError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use anyhow::anyhow;
    use bytes::Bytes;

    use super::*;
    use crate::clock::sim_clock::SimClock;
    use crate::dataset::TiktokenTokenizer;
    use crate::graph::materialize::{PromptMaterializer, SegmentItemsMaterializer};
    use crate::graph::model::{
        ChannelSpec, ChannelType, GraphRecord, LlmNode, PromptItem, ReducerName, StaticEdge,
        TraceRecord,
    };
    use crate::graph::policy::{AbortTraceNodeFailurePolicy, FailFastRunFailurePolicy};
    use crate::graph::segment::{SegmentPool, intern_message};
    use crate::graph::sink::{GraphReply, GraphSink};
    use crate::graph::wire::OpenAiChatMessage;

    fn one_node_plan(id: &str, handle: crate::dataset::Handle) -> GraphTracePlan {
        let output = format!("out-{id}");
        let mut graph = GraphRecord::default();
        graph.state.insert(
            output.clone(),
            ChannelSpec {
                channel_type: ChannelType::Messages,
                reducer: ReducerName::AddMessages,
            },
        );
        graph.nodes.insert(
            id.to_string(),
            LlmNode {
                output,
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: handle }],
                metadata: BTreeMap::new(),
            },
        );
        graph.edges.push(StaticEdge {
            source: crate::graph::model::START_NODE_ID.into(),
            target: id.into(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        });
        GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        }
    }

    #[test]
    fn cycling_source_reuses_templates_with_unique_instance_ids() {
        let mut pool = SegmentPool::new();
        let handle = intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "hello"),
            None,
            &TiktokenTokenizer::builtin(),
        )
        .unwrap();
        let source = CyclingGraphTraceSource::new(
            vec![one_node_plan("a", handle), one_node_plan("b", handle)],
            Some(3),
        )
        .unwrap();
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "a::instance-0"
        );
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "b::instance-1"
        );
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "a::instance-2"
        );
        assert!(source.next_trace().unwrap().is_none());
    }

    #[test]
    fn cycling_source_resumes_from_start_ordinal() {
        let mut pool = SegmentPool::new();
        let handle = intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "hello"),
            None,
            &TiktokenTokenizer::builtin(),
        )
        .unwrap();
        // corpus_cursor = 3 over a 2-template cycle: first draw is template
        // 3 % 2 = 1 ("b"), NOT the 0th; the session budget still counts from 0.
        let source = CyclingGraphTraceSource::new(
            vec![one_node_plan("a", handle), one_node_plan("b", handle)],
            Some(3),
        )
        .unwrap()
        .starting_at(3);
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "b::instance-0"
        );
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "a::instance-1"
        );
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "b::instance-2"
        );
        // session_limit (3) is anchored to the raw ordinal, not the shifted draw.
        assert!(source.next_trace().unwrap().is_none());
    }

    #[test]
    fn cycling_source_default_start_ordinal_is_unchanged() {
        let mut pool = SegmentPool::new();
        let handle = intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "hello"),
            None,
            &TiktokenTokenizer::builtin(),
        )
        .unwrap();
        let source = CyclingGraphTraceSource::new(
            vec![one_node_plan("a", handle), one_node_plan("b", handle)],
            Some(2),
        )
        .unwrap();
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "a::instance-0"
        );
        assert_eq!(
            source.next_trace().unwrap().unwrap().trace.id,
            "b::instance-1"
        );
        assert!(source.next_trace().unwrap().is_none());
    }

    #[test]
    fn cycling_source_shuffle_matches_shared_draw_and_covers_each_pass() {
        // (b) Shuffle: the profiling recycle `next_trace` template order equals the
        // shared persistent-epoch `PermutationDraw` on the same
        // `ShuffleSampler` child seed, and every full pass covers each template once.
        let handle = sample_handle();
        let letters = ["a", "b", "c", "d", "e"];
        let total = letters.len();
        let templates: Vec<GraphTracePlan> =
            letters.iter().map(|id| one_node_plan(id, handle)).collect();
        let base_seed = 5203359018791016587u64;
        // Two full passes over the 5-template corpus.
        let source = CyclingGraphTraceSource::new(templates, Some(2 * total as u64))
            .unwrap()
            .with_sampling(PermutationDraw::shuffle(base_seed));
        let drawn: Vec<String> = std::iter::from_fn(|| {
            source
                .next_trace()
                .unwrap()
                .map(|plan| plan.trace.id.split_once("::").unwrap().0.to_owned())
        })
        .collect();
        let reference = PermutationDraw::shuffle(base_seed);
        for pass in 0u64..2 {
            let mut seen = Vec::new();
            for offset in 0..total {
                let x = pass * total as u64 + offset as u64;
                let drawn_letter = &drawn[pass as usize * total + offset];
                let idx = reference.index(x, total);
                assert_eq!(drawn_letter, letters[idx]);
                seen.push(idx);
            }
            seen.sort_unstable();
            assert_eq!(seen, (0..total).collect::<Vec<_>>(), "pass {pass} coverage");
        }
    }

    #[test]
    fn cycling_source_shuffle_continues_pressure_shared_sampler_order() {
        // (c) Shared-sampler-with-pressure: a freed profiling lane that resumes at
        // corpus cursor `k` (`starting_at(k)`) serves the SAME template the shared
        // `PermutationDraw` (the pressure-warmup remap) yields for counter `k+i`,
        // so profiling never re-serves a template pressure warmup replayed
        // under a different order. Both route through the identical remap on the
        // identical `(base_seed, total)`.
        let handle = sample_handle();
        let letters = ["a", "b", "c", "d", "e"];
        let total = letters.len();
        let templates: Vec<GraphTracePlan> =
            letters.iter().map(|id| one_node_plan(id, handle)).collect();
        let base_seed = 7u64;
        let start = 3u64;
        let source = CyclingGraphTraceSource::new(templates, Some(total as u64))
            .unwrap()
            .starting_at(start)
            .with_sampling(PermutationDraw::shuffle(base_seed));
        // The reference sampler pressure warmup draws from on the same counter.
        let pressure = PermutationDraw::shuffle(base_seed);
        for i in 0..total as u64 {
            let id = source.next_trace().unwrap().unwrap().trace.id;
            let letter = id.split_once("::").unwrap().0;
            let expected = letters[pressure.index(start + i, total)];
            assert_eq!(letter, expected, "counter {}", start + i);
        }
    }

    #[test]
    fn cycling_source_sequential_default_ignores_seed_and_is_byte_unchanged() {
        // (a) HARD guard: default (no `with_sampling`) and an explicit
        // `Sequential` draw (even with a nonzero seed) both keep `draw % len`.
        let handle = sample_handle();
        let templates = || vec![one_node_plan("a", handle), one_node_plan("b", handle)];
        let default_source = CyclingGraphTraceSource::new(templates(), Some(3)).unwrap();
        let sequential_source = CyclingGraphTraceSource::new(templates(), Some(3))
            .unwrap()
            .with_sampling(PermutationDraw::sequential());
        for _ in 0..3 {
            assert_eq!(
                default_source.next_trace().unwrap().unwrap().trace.id,
                sequential_source.next_trace().unwrap().unwrap().trace.id
            );
        }
        assert!(default_source.next_trace().unwrap().is_none());
    }

    fn sample_handle() -> crate::dataset::Handle {
        let mut pool = SegmentPool::new();
        intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "hello"),
            None,
            &TiktokenTokenizer::builtin(),
        )
        .unwrap()
    }

    fn instance_ordinal(id: &str) -> u64 {
        id.split_once("::instance-").unwrap().1.parse().unwrap()
    }

    #[test]
    fn partitioned_source_interleaves_and_covers_the_single_cell_set() {
        let handle = sample_handle();
        let templates = || vec![one_node_plan("a", handle), one_node_plan("b", handle)];
        // 3 cells over a 2-template cycle, shared global session cap 10.
        let mut owned: Vec<Vec<String>> = Vec::new();
        for cell_id in 0..3u32 {
            let source =
                PartitionedGraphTraceSource::new(templates(), Some(10), cell_id, 3).unwrap();
            let mut ids = Vec::new();
            while let Some(plan) = source.next_trace().unwrap() {
                ids.push(plan.trace.id);
            }
            owned.push(ids);
        }
        // Interleave: cell k owns exactly the global ordinals ≡ k (mod 3), below the cap.
        assert_eq!(
            owned[0]
                .iter()
                .map(|id| instance_ordinal(id))
                .collect::<Vec<_>>(),
            vec![0, 3, 6, 9]
        );
        assert_eq!(
            owned[1]
                .iter()
                .map(|id| instance_ordinal(id))
                .collect::<Vec<_>>(),
            vec![1, 4, 7]
        );
        assert_eq!(
            owned[2]
                .iter()
                .map(|id| instance_ordinal(id))
                .collect::<Vec<_>>(),
            vec![2, 5, 8]
        );
        // Union is exactly the single-cell set 0..10, each template drawn by ordinal
        // parity (the same template a 1-cell run assigns that global ordinal).
        let mut all: Vec<String> = owned.into_iter().flatten().collect();
        for id in &all {
            let (template, _) = id.split_once("::instance-").unwrap();
            let expected = if instance_ordinal(id) % 2 == 0 {
                "a"
            } else {
                "b"
            };
            assert_eq!(template, expected, "template drift at {id}");
        }
        all.sort_by_key(|id| instance_ordinal(id));
        assert_eq!(
            all.iter()
                .map(|id| instance_ordinal(id))
                .collect::<Vec<_>>(),
            (0..10).collect::<Vec<_>>()
        );
    }

    #[test]
    fn partitioned_shuffle_union_matches_single_cell_cycling_shuffle() {
        // Under Shuffle the cellular union still covers the corpus once per pass:
        // each cell draws `perm[global/len][global%len]` on its interleaved global
        // ordinals, and the union (contiguous 0..cap) reproduces exactly the
        // single-cell shuffle cycling sequence — the deterministic-per-topology
        // contract, preserved by routing both through the shared `PermutationDraw`.
        let handle = sample_handle();
        let letters = ["a", "b", "c", "d", "e"];
        let templates = || {
            letters
                .iter()
                .map(|id| one_node_plan(id, handle))
                .collect::<Vec<_>>()
        };
        let base_seed = 3u64;
        let cap = 10u64;
        // Single-cell shuffle cycler = the reference global order.
        let single = CyclingGraphTraceSource::new(templates(), Some(cap))
            .unwrap()
            .with_sampling(PermutationDraw::shuffle(base_seed));
        let mut reference: BTreeMap<u64, String> = BTreeMap::new();
        while let Some(plan) = single.next_trace().unwrap() {
            let (letter, ord) = plan.trace.id.split_once("::instance-").unwrap();
            reference.insert(ord.parse().unwrap(), letter.to_owned());
        }
        // The three cells' interleaved draws, keyed by global ordinal, must equal
        // that reference exactly (same template at each global ordinal).
        let mut union: BTreeMap<u64, String> = BTreeMap::new();
        for cell_id in 0..3u32 {
            let source = PartitionedGraphTraceSource::new(templates(), Some(cap), cell_id, 3)
                .unwrap()
                .with_sampling(PermutationDraw::shuffle(base_seed));
            while let Some(plan) = source.next_trace().unwrap() {
                let (letter, ord) = plan.trace.id.split_once("::instance-").unwrap();
                union.insert(ord.parse().unwrap(), letter.to_owned());
            }
        }
        assert_eq!(union, reference);
        // And each full pass covers every template once (music-shuffle contract).
        for pass in 0u64..(cap / letters.len() as u64) {
            let mut seen: Vec<&String> = (0..letters.len() as u64)
                .map(|o| &union[&(pass * letters.len() as u64 + o)])
                .collect();
            seen.sort();
            let mut expected: Vec<&str> = letters.to_vec();
            expected.sort();
            assert_eq!(
                seen.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn single_cell_partition_matches_the_cycling_source() {
        let handle = sample_handle();
        let part =
            PartitionedGraphTraceSource::new(vec![one_node_plan("a", handle)], Some(3), 0, 1)
                .unwrap();
        let cyc = CyclingGraphTraceSource::new(vec![one_node_plan("a", handle)], Some(3)).unwrap();
        for _ in 0..3 {
            assert_eq!(
                part.next_trace().unwrap().unwrap().trace.id,
                cyc.next_trace().unwrap().unwrap().trace.id
            );
        }
        assert!(part.next_trace().unwrap().is_none());
        assert!(cyc.next_trace().unwrap().is_none());
    }

    #[test]
    fn unbounded_partition_owns_only_its_interleave() {
        let handle = sample_handle();
        // cell 1 of 3, no session cap (duration-driven): owns 1, 4, 7, 10, …
        let source =
            PartitionedGraphTraceSource::new(vec![one_node_plan("a", handle)], None, 1, 3).unwrap();
        for expected in [1u64, 4, 7, 10] {
            assert_eq!(
                instance_ordinal(&source.next_trace().unwrap().unwrap().trace.id),
                expected
            );
        }
    }

    #[test]
    fn partitioned_source_rejects_bad_partitions() {
        let handle = sample_handle();
        assert!(
            PartitionedGraphTraceSource::new(vec![one_node_plan("a", handle)], Some(4), 4, 4)
                .is_err(),
            "cell_id must be < cell_count"
        );
        assert!(
            PartitionedGraphTraceSource::new(vec![one_node_plan("a", handle)], None, 0, 0).is_err(),
            "cell_count must be >= 1"
        );
        assert!(
            PartitionedGraphTraceSource::new(Vec::new(), None, 0, 1).is_err(),
            "at least one template required"
        );
    }

    #[test]
    fn cycling_source_enforces_static_request_budget_at_trace_boundaries() {
        let mut pool = SegmentPool::new();
        let handle = intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "hello"),
            None,
            &TiktokenTokenizer::builtin(),
        )
        .unwrap();
        let small = one_node_plan("small", handle);
        let mut large = one_node_plan("large", handle);
        let node = large.graph.nodes.values().next().unwrap().clone();
        large
            .graph
            .nodes
            .insert("large-extra-1".into(), node.clone());
        large.graph.nodes.insert("large-extra-2".into(), node);

        let exact = CyclingGraphTraceSource::with_budgets(
            vec![small.clone(), large.clone()],
            Some(10),
            Some(4),
        )
        .unwrap();
        assert_eq!(exact.next_trace().unwrap().unwrap().graph.nodes.len(), 1);
        assert_eq!(exact.next_trace().unwrap().unwrap().graph.nodes.len(), 3);
        assert!(exact.next_trace().unwrap().is_none());

        let no_split = CyclingGraphTraceSource::with_budgets(
            vec![small.clone(), large.clone()],
            Some(10),
            Some(3),
        )
        .unwrap();
        assert_eq!(no_split.next_trace().unwrap().unwrap().graph.nodes.len(), 1);
        assert!(no_split.next_trace().unwrap().is_none());

        let sessions_first = CyclingGraphTraceSource::with_budgets(
            vec![small.clone(), large.clone()],
            Some(1),
            Some(100),
        )
        .unwrap();
        assert_eq!(
            sessions_first
                .next_trace()
                .unwrap()
                .unwrap()
                .graph
                .nodes
                .len(),
            1
        );
        assert!(sessions_first.next_trace().unwrap().is_none());

        let requests_first =
            CyclingGraphTraceSource::with_budgets(vec![small, large], Some(100), Some(3)).unwrap();
        assert_eq!(
            requests_first
                .next_trace()
                .unwrap()
                .unwrap()
                .graph
                .nodes
                .len(),
            1
        );
        assert!(requests_first.next_trace().unwrap().is_none());
    }

    #[test]
    fn independently_budgeted_phase_sources_share_run_unique_instance_ids() {
        let mut pool = SegmentPool::new();
        let handle = intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "hello"),
            None,
            &TiktokenTokenizer::builtin(),
        )
        .unwrap();
        let sequence = GraphTraceInstanceSequence::default();
        let warmup = CyclingGraphTraceSource::with_budgets_and_sequence(
            vec![one_node_plan("root", handle)],
            Some(1),
            None,
            sequence.clone(),
        )
        .unwrap();
        let profiling = CyclingGraphTraceSource::with_budgets_and_sequence(
            vec![one_node_plan("root", handle)],
            Some(1),
            None,
            sequence,
        )
        .unwrap();

        assert_eq!(
            warmup.next_trace().unwrap().unwrap().trace.id,
            "root::instance-0"
        );
        assert_eq!(
            profiling.next_trace().unwrap().unwrap().trace.id,
            "root::instance-1"
        );
    }

    struct SelectiveSink;

    #[async_trait(?Send)]
    impl GraphSink<OpenAiChatMessage> for SelectiveSink {
        async fn dispatch(
            &self,
            node_id: &str,
            _messages: Vec<Bytes>,
            _max_tokens: Option<usize>,
            on_first_token: &dyn Fn(),
        ) -> anyhow::Result<GraphReply<OpenAiChatMessage>> {
            if node_id == "fail" {
                return Err(anyhow!("selected failure"));
            }
            on_first_token();
            Ok(GraphReply::from_text("ok".into()))
        }
    }

    struct RecordingBackend {
        plans: Rc<RefCell<Vec<GraphTracePlan>>>,
    }

    #[async_trait(?Send)]
    impl crate::graph::execution::TracePlacement for RecordingBackend {
        async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
            self.plans.borrow_mut().push(plan);
            Ok(())
        }
    }

    #[test]
    fn coordinator_delegates_a_complete_trace_through_the_backend_trait() {
        let clock = Rc::new(SimClock::new());
        let mut graph = GraphRecord::default();
        graph.nodes.insert(
            "left".into(),
            LlmNode {
                output: "left-output".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(7),
                items: Vec::new(),
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "right".into(),
            LlmNode {
                output: "right-output".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(11),
                items: Vec::new(),
                metadata: BTreeMap::new(),
            },
        );
        let source: Rc<dyn GraphTraceSource> =
            Rc::new(VecGraphTraceSource::new([GraphTracePlan {
                graph,
                trace: TraceRecord {
                    id: "whole-trace".into(),
                    graph_ref: Some("resolved-before-placement".into()),
                    initial_state: BTreeMap::from([(
                        "seed".into(),
                        serde_json::Value::String("value".into()),
                    )]),
                },
                arrival_offset_ns: Some(123),
            }]));
        let received = Rc::new(RefCell::new(Vec::new()));
        let backend: Rc<dyn crate::graph::execution::TracePlacement> = Rc::new(RecordingBackend {
            plans: received.clone(),
        });
        let workload = GraphWorkload::new(clock.clone(), source, backend);
        let report = Rc::new(RefCell::new(None));
        let report_slot = report.clone();
        let outcome = crate::graph::runtime::drive_sim(clock, move |_handle| async move {
            *report_slot.borrow_mut() = Some(workload.execute().await.unwrap());
        });

        assert!(!outcome.deadlocked);
        assert_eq!(report.borrow().as_ref().unwrap().completed, 1);
        let received = received.borrow();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].trace.id, "whole-trace");
        assert_eq!(received[0].trace.initial_state["seed"], "value");
        assert_eq!(received[0].arrival_offset_ns, Some(123));
        assert_eq!(
            received[0].graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["left", "right"]
        );
    }

    struct SleepingBackend {
        clock: Rc<dyn Clock>,
        completed: Rc<RefCell<Vec<String>>>,
    }

    #[async_trait(?Send)]
    impl crate::graph::execution::TracePlacement for SleepingBackend {
        async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
            self.clock.clone().sleep(20).await;
            self.completed.borrow_mut().push(plan.trace.id);
            Ok(())
        }
    }

    #[test]
    fn duration_stop_cancels_root_admission_and_drains_active_trace() {
        let clock = Rc::new(SimClock::new());
        let plans = [
            GraphTracePlan {
                graph: GraphRecord::default(),
                trace: TraceRecord {
                    id: "active".into(),
                    graph_ref: None,
                    initial_state: BTreeMap::new(),
                },
                arrival_offset_ns: Some(0),
            },
            GraphTracePlan {
                graph: GraphRecord::default(),
                trace: TraceRecord {
                    id: "after-deadline".into(),
                    graph_ref: None,
                    initial_state: BTreeMap::new(),
                },
                arrival_offset_ns: Some(15),
            },
        ];
        let source: Rc<dyn GraphTraceSource> = Rc::new(VecGraphTraceSource::new(plans));
        let completed = Rc::new(RefCell::new(Vec::new()));
        let backend: Rc<dyn crate::graph::execution::TracePlacement> = Rc::new(SleepingBackend {
            clock: clock.clone(),
            completed: completed.clone(),
        });
        let workload = GraphWorkload::new(clock.clone(), source, backend)
            .with_arrival(Rc::new(ScheduledGraphArrival))
            .with_stop_policy(Rc::new(DurationGraphStop::new(10).unwrap()));
        let report = Rc::new(RefCell::new(None));
        let report_slot = report.clone();
        let outcome = crate::graph::runtime::drive_sim(clock, move |_handle| async move {
            *report_slot.borrow_mut() = Some(workload.execute().await.unwrap());
        });

        assert!(!outcome.deadlocked);
        assert_eq!(report.borrow().as_ref().unwrap().admitted, 1);
        assert_eq!(report.borrow().as_ref().unwrap().completed, 1);
        assert_eq!(&*completed.borrow(), &["active"]);
    }

    #[test]
    fn fail_fast_stops_new_trace_admission_while_resilient_runs_all() {
        fn run(fail_fast: bool) -> GraphWorkloadReport {
            let clock = Rc::new(SimClock::new());
            let tokenizer = TiktokenTokenizer::builtin();
            let mut pool = SegmentPool::new();
            let message = intern_message(
                &mut pool,
                &OpenAiChatMessage::new("user", "u"),
                None,
                &tokenizer,
            )
            .unwrap();
            let source: Rc<dyn GraphTraceSource> = Rc::new(VecGraphTraceSource::new([
                one_node_plan("fail", message),
                one_node_plan("after", message),
            ]));
            let materializer: Rc<dyn PromptMaterializer> =
                Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(SelectiveSink);
            let slots = Rc::new(SlotPool::new(1));
            let mut backend = crate::graph::execution::LocalGraphTraceExecutionBackend::new(
                clock.clone(),
                materializer,
                sink,
            );
            if fail_fast {
                backend = backend.with_node_failure(Rc::new(AbortTraceNodeFailurePolicy));
            }
            let mut workload = GraphWorkload::new(clock.clone(), source, Rc::new(backend))
                .with_admission(Rc::new(SlotPoolTraceAdmission::new(slots)));
            if fail_fast {
                workload = workload.with_run_failure(Rc::new(FailFastRunFailurePolicy::default()));
            }
            let result = Rc::new(RefCell::new(None));
            let result_slot = result.clone();
            let outcome = crate::graph::runtime::drive_sim(clock, move |_handle| async move {
                *result_slot.borrow_mut() = Some(workload.execute().await.unwrap());
            });
            assert!(!outcome.deadlocked);
            result.borrow_mut().take().unwrap()
        }

        let resilient = run(false);
        assert_eq!(resilient.admitted, 2);
        assert_eq!(resilient.completed, 2);
        assert_eq!(resilient.failed, 0);

        let fail_fast = run(true);
        assert_eq!(fail_fast.admitted, 1);
        assert_eq!(fail_fast.completed, 0);
        assert_eq!(fail_fast.failed, 1);
        assert_eq!(fail_fast.traces[0].trace_id, "fail");
    }

    #[test]
    fn interval_arrival_and_session_pool_share_simclock_policy_path() {
        let clock = Rc::new(SimClock::new());
        let generator: Rc<RefCell<Box<dyn IntervalGenerator>>> = Rc::new(RefCell::new(Box::new(
            crate::timing::intervals::Constant::new(10.0),
        )));
        let arrival = IntervalGraphArrival::new(generator.clone());
        let plan = GraphTracePlan {
            graph: GraphRecord::default(),
            trace: TraceRecord {
                id: "t".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        };
        let observed = Rc::new(RefCell::new(Vec::new()));
        let observed_slot = observed.clone();
        let drive_clock = clock.clone();
        let outcome = crate::graph::runtime::drive_sim(clock, move |_handle| async move {
            let start = drive_clock.now_ns();
            arrival
                .wait_for_arrival(drive_clock.clone(), start, 0, &plan)
                .await
                .unwrap();
            observed_slot.borrow_mut().push(drive_clock.now_ns());
            arrival
                .wait_for_arrival(drive_clock.clone(), start, 1, &plan)
                .await
                .unwrap();
            observed_slot.borrow_mut().push(drive_clock.now_ns());
            generator.borrow_mut().set_rate(20.0);
            arrival
                .wait_for_arrival(drive_clock.clone(), start, 2, &plan)
                .await
                .unwrap();
            observed_slot.borrow_mut().push(drive_clock.now_ns());
        });
        assert!(!outcome.deadlocked);
        assert_eq!(*observed.borrow(), vec![0, 100_000_000, 150_000_000]);
    }

    #[test]
    fn scheduled_arrival_honors_exact_virtual_offset() {
        let clock = Rc::new(SimClock::new());
        let plan = GraphTracePlan {
            graph: GraphRecord::default(),
            trace: TraceRecord {
                id: "scheduled".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: Some(42_000),
        };
        let observed = Rc::new(Cell::new(-1));
        let observed_slot = observed.clone();
        let drive_clock = clock.clone();
        let outcome = crate::graph::runtime::drive_sim(clock, move |_handle| async move {
            ScheduledGraphArrival
                .wait_for_arrival(drive_clock.clone(), 0, 0, &plan)
                .await
                .unwrap();
            observed_slot.set(drive_clock.now_ns());
        });
        assert!(!outcome.deadlocked);
        assert_eq!(observed.get(), 42_000);
    }

    #[test]
    fn fail_fast_wakes_fan_in_waiting_on_a_never_scheduled_producer() {
        let clock = Rc::new(SimClock::new());
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let message = intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "u"),
            None,
            &tokenizer,
        )
        .unwrap();
        let mut graph = GraphRecord::default();
        for channel in ["a", "b", "gate"] {
            graph.state.insert(
                channel.into(),
                ChannelSpec {
                    channel_type: ChannelType::Messages,
                    reducer: ReducerName::AddMessages,
                },
            );
        }
        graph.nodes.insert(
            "fail".into(),
            LlmNode {
                output: "a".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "never".into(),
            LlmNode {
                output: "b".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "waiting".into(),
            LlmNode {
                output: "gate".into(),
                streaming: true,
                inputs: vec![crate::graph::model::ChannelRequirement {
                    channel: "b".into(),
                    count: crate::graph::model::Count::N(1),
                }],
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "first-token-waiting".into(),
            LlmNode {
                output: "gate".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.edges.extend([
            StaticEdge {
                source: crate::graph::model::START_NODE_ID.into(),
                target: "fail".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
            StaticEdge {
                source: crate::graph::model::START_NODE_ID.into(),
                target: "waiting".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
            StaticEdge {
                source: "fail".into(),
                target: "never".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
            StaticEdge {
                source: "never".into(),
                target: "first-token-waiting".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: Some(1.0),
            },
            StaticEdge {
                source: crate::graph::model::START_NODE_ID.into(),
                target: "first-token-waiting".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
        ]);
        let source: Rc<dyn GraphTraceSource> =
            Rc::new(VecGraphTraceSource::new([GraphTracePlan {
                graph,
                trace: TraceRecord {
                    id: "stranded".into(),
                    graph_ref: None,
                    initial_state: BTreeMap::new(),
                },
                arrival_offset_ns: None,
            }]));
        let materializer: Rc<dyn PromptMaterializer> =
            Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));
        let backend = crate::graph::execution::LocalGraphTraceExecutionBackend::new(
            clock.clone(),
            materializer,
            Rc::new(SelectiveSink),
        )
        .with_node_failure(Rc::new(AbortTraceNodeFailurePolicy));
        let workload = GraphWorkload::new(clock.clone(), source, Rc::new(backend))
            .with_run_failure(Rc::new(FailFastRunFailurePolicy::default()));
        let report = Rc::new(RefCell::new(None));
        let report_slot = report.clone();
        let outcome = crate::graph::runtime::drive_sim(clock, move |_handle| async move {
            *report_slot.borrow_mut() = Some(workload.execute().await.unwrap());
        });

        assert!(!outcome.deadlocked);
        let report = report.borrow_mut().take().unwrap();
        assert_eq!(report.failed, 1);
        assert_eq!(report.admitted, 1);
    }
}
