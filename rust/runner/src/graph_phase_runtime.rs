// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral Graph-IR phase orchestration.
//!
//! Authored root selection, arrival, admission, lifecycle, ramps, adaptive
//! control, exact node/trace terminal accounting, and reporting records live
//! here once. Online HTTP and in-process offline simulation inject only a
//! phase-local whole-trace execution backend.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use aiperf::adaptive::{AdaptiveControlVariable, AdaptiveRunConfig, build_adaptive_scale};
use aiperf::adaptive_core::{
    AdaptiveError, ControlActuator, ControlSnapshot, RequestRateActuator,
    SessionConcurrencyActuator, SharedWindowSampler, TumblingWindowSampler,
};
use aiperf::ancillary::RATE_RAMP_UPDATE_INTERVAL_NS;
use aiperf::cellular::{CellPartition, ModuloCellPartition};
use aiperf::clock::Clock;
use aiperf::failure::OnFailure;
use aiperf::graph::errors::TraceError;
use aiperf::graph::execution::TracePlacement;
use aiperf::graph::input::GraphInputBundle;
use aiperf::graph::model::{GraphTracePlan, ParsedGraph, TraceRecord};
use aiperf::graph::policy::{ContinueRunFailurePolicy, FailFastRunFailurePolicy, RunFailurePolicy};
use aiperf::graph::snapshot::{chop_trie_at_tstar, rewrite_for_warmup};
use aiperf::graph::tstar::{TStarSampler, WindowTStarSampler, trace_duration_us};
use aiperf::graph::workload::{
    CyclingGraphTraceSource, GraphArrivalPolicy, GraphTraceInstanceSequence, GraphTraceRunResult,
    GraphTraceSource, GraphWorkload, GraphWorkloadObserver, GraphWorkloadReport,
    ImmediateGraphArrival, IntervalGraphArrival, PartitionedGraphTraceSource,
    SlotPoolTraceAdmission, TraceAdmissionInfo,
};
use aiperf::metrics_core::Phase as MetricsPhase;
use aiperf::phase_runtime::{
    RampScheduledPhaseController, ScheduledPhaseController, ScheduledPhaseSidecar,
};
use aiperf::rng::{RngRoot, namespace};
use aiperf::timing::{
    ClockPhaseOrchestrator, ClockPhaseRunnerFactory, LocalPhaseFuture, NoopPhaseObserver,
    PhaseConfig, PhaseContext, PhaseExecution, PhaseExecutionError, PhaseExecutionFactory,
    PhaseObserver, PhaseOrchestrator, PhaseReturn, PhaseSend, PhaseStats, RampDriver, SlotPool,
    make_interval_generator,
};
use anyhow::{Context, Result, anyhow, bail, ensure};
use tokio::sync::{Notify, mpsc};
use uuid::Uuid;

use crate::execute::{
    AdaptiveScheduledPhaseController, RampActuatorRngRoots, adaptive_run_config,
    integer_adaptive_bound, metrics_phase, phase_config, phase_seamless_to_next, ramp_strategy,
    seconds_to_u64_ns,
};
use crate::graph_execution::{
    ChannelRunnerGraphExecutionEventSink, GraphCancellationConfig, ObservedRunnerGraphPlacement,
    RunnerGraphExecutionEvent, RunnerGraphExecutionEventSink,
};
use crate::graph_input::TStarWindow;
use crate::protocol::{AdaptiveControlVariableSpec, PhaseSpec};
use crate::protocol_v2::RunnerFailureStageV2;
use crate::records::CapturedRecord;
use crate::registry::PreparedRunFailure;

/// Backend-owned inputs for one already lowered Graph-IR phase.
pub(crate) struct GraphPhaseBackendConfig {
    pub(crate) metrics_phase: MetricsPhase,
    pub(crate) prefill_concurrency: Option<usize>,
    pub(crate) cancellation: Option<GraphCancellationConfig>,
    pub(crate) events: Arc<dyn RunnerGraphExecutionEventSink>,
}

/// One phase-local whole-trace backend returned by an injected implementation.
pub(crate) struct PreparedGraphPhaseBackend {
    pub(crate) placement: Rc<dyn TracePlacement>,
    pub(crate) requires_node_records: bool,
}

/// Backend construction seam beneath the one shared graph phase driver.
pub(crate) trait RunnerGraphPhaseBackendFactory {
    fn prepare_backend(&self, config: GraphPhaseBackendConfig)
    -> Result<PreparedGraphPhaseBackend>;
}

/// Complete result of one shared Graph-IR phase-orchestrator run.
pub(crate) struct GraphPhaseRunOutput {
    pub(crate) captured: Vec<CapturedRecord>,
    pub(crate) phases: Vec<PhaseStats>,
    pub(crate) workload: GraphWorkloadReport,
}

/// Validate transport-neutral authored Graph-IR phase policy.
pub(crate) fn validate_graph_phases(phases: &[PhaseSpec]) -> Result<()> {
    ensure!(
        !phases.is_empty(),
        "graph execution requires at least one phase"
    );
    for (phase_index, phase) in phases.iter().enumerate() {
        ensure!(
            matches!(
                phase,
                PhaseSpec::Concurrency { .. }
                    | PhaseSpec::Poisson { .. }
                    | PhaseSpec::Gamma { .. }
                    | PhaseSpec::Constant { .. }
            ),
            "graph phase {phase_index} must use concurrency, poisson, gamma, or constant scheduling"
        );
        let common = phase.common();
        ensure!(
            common.requests != Some(0) && common.sessions != Some(0),
            "graph phase {phase_index} request/session bounds must be positive when configured"
        );
        ensure!(
            phase.concurrency() != Some(0),
            "graph phase {phase_index} concurrency must be positive when configured"
        );
        ensure!(
            common.prefill_concurrency != Some(0),
            "graph phase {phase_index} prefill_concurrency must be positive when configured"
        );
        if let (Some(prefill), Some(concurrency)) =
            (common.prefill_concurrency, phase.concurrency())
        {
            ensure!(
                prefill <= concurrency,
                "graph phase {phase_index} prefill_concurrency must be <= concurrency"
            );
        }
        if let Some(duration) = common.duration {
            seconds_to_u64_ns(duration)
                .with_context(|| format!("validating graph phase {phase_index} duration"))?;
            ensure!(duration > 0.0, "graph phase duration must be positive");
        }
        match phase {
            PhaseSpec::Poisson { rate, .. } | PhaseSpec::Constant { rate, .. } => ensure!(
                rate.is_finite() && *rate > 0.0,
                "graph phase {phase_index} rate must be finite and positive"
            ),
            PhaseSpec::Gamma {
                rate, smoothness, ..
            } => {
                ensure!(
                    rate.is_finite() && *rate > 0.0,
                    "graph phase {phase_index} rate must be finite and positive"
                );
                ensure!(
                    smoothness.is_none_or(|value| value.is_finite() && value > 0.0),
                    "graph phase {phase_index} gamma smoothness must be finite and positive"
                );
            }
            PhaseSpec::Concurrency { .. } => {}
            PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } => unreachable!(),
        }
        if let Some(cancellation) = common.cancellation {
            ensure!(
                cancellation.rate.is_finite() && (0.0..=100.0).contains(&cancellation.rate),
                "graph phase {phase_index} cancellation.rate must be finite and in 0..=100"
            );
            ensure!(
                cancellation.delay.is_finite() && cancellation.delay >= 0.0,
                "graph phase {phase_index} cancellation.delay must be finite and non-negative"
            );
        }
        if let Some(ramp) = &common.concurrency_ramp {
            ensure!(
                phase.concurrency().is_some(),
                "graph phase {phase_index} concurrency_ramp requires a concurrency target"
            );
            validate_graph_ramp(phase_index, "concurrency_ramp", ramp.duration)?;
        }
        if let Some(ramp) = &common.prefill_ramp {
            ensure!(
                common.prefill_concurrency.is_some(),
                "graph phase {phase_index} prefill_ramp requires prefill_concurrency"
            );
            validate_graph_ramp(phase_index, "prefill_ramp", ramp.duration)?;
        }
        if let Some(ramp) = &common.rate_ramp {
            ensure!(
                phase
                    .request_arrival()
                    .and_then(|(_, rate, _)| rate)
                    .is_some(),
                "graph phase {phase_index} rate_ramp requires a rate-controlled phase"
            );
            validate_graph_ramp(phase_index, "rate_ramp", ramp.duration)?;
        }
        let _ = graph_adaptive_config(phase, "validation", Path::new("."))?;
    }
    Ok(())
}

fn validate_graph_ramp(phase_index: usize, name: &str, duration: f64) -> Result<()> {
    ensure!(
        duration.is_finite() && duration > 0.0,
        "graph phase {phase_index} {name}.duration must be finite and positive"
    );
    let _ = seconds_to_u64_ns(duration)?;
    Ok(())
}

/// Upper bound in seconds on the extended (cache-pressure) warmup phase's
/// drain grace period.
///
/// Port of `Environment.GRAPH.PRESSURE_DRAIN_GRACE_CAP` (default `300.0`) in
/// `src/aiperf/common/environment.py:657`, consumed by
/// `timing/config.py::_graph_pressure_grace_sec`.
pub const PRESSURE_DRAIN_GRACE_CAP_SEC: f64 = 300.0;

/// Derive the drain grace (seconds) for a cache-pressure-mode graph warmup.
///
/// Port of `timing/config.py::_graph_pressure_grace_sec`
/// (`src/aiperf/timing/config.py:771-791`). An EXPLICIT `user_grace`
/// (`Some`) is honored verbatim -- the operator's escape hatch when a healthy
/// drain outlives the pressure duration (e.g. a 45s prefill in flight at a
/// 30s deadline). Otherwise the drain is bounded by
/// `min(cache_pressure, PRESSURE_DRAIN_GRACE_CAP_SEC)` so a wedged or lost
/// return cannot hang the run. AgentX's benchmark-grace floor branch is
/// intentionally not ported. Callers gate on a set pressure duration first.
pub fn graph_pressure_grace_sec(user_grace: Option<f64>, cache_pressure: f64) -> f64 {
    match user_grace {
        Some(grace) => grace,
        None => cache_pressure.min(PRESSURE_DRAIN_GRACE_CAP_SEC),
    }
}

struct PreparedGraphPhase {
    workload: GraphWorkload,
    placement: Rc<dyn TracePlacement>,
    events: mpsc::UnboundedReceiver<RunnerGraphExecutionEvent>,
    intervals: Rc<RefCell<Box<dyn aiperf::timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_initial: Option<usize>,
    controller: Rc<dyn ScheduledPhaseController>,
    failures: Rc<GraphPhaseFailures>,
    adaptive: Option<AdaptiveRunConfig>,
    /// Whether this is the WARMUP phase (gates warmup-failure accounting).
    is_warmup: bool,
}

struct GraphTracePhaseProgress {
    expected_nodes: usize,
    returned_nodes: usize,
    first_token_uuids: HashSet<Uuid>,
    returned_uuids: HashSet<Uuid>,
}

trait GraphPhaseProgressSink {
    fn record_sent_batch(&self, sent: &[PhaseSend]) -> Result<(), String>;
    fn record_first_token(&self);
    fn record_returned(&self, returned: PhaseReturn);
    fn mark_all_sent(&self);
}

impl GraphPhaseProgressSink for PhaseContext {
    fn record_sent_batch(&self, sent: &[PhaseSend]) -> Result<(), String> {
        PhaseContext::record_sent_batch(self, sent)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn record_returned(&self, returned: PhaseReturn) {
        PhaseContext::record_returned(self, returned);
    }

    fn record_first_token(&self) {
        PhaseContext::record_first_token(self);
    }

    fn mark_all_sent(&self) {
        PhaseContext::mark_all_sent(self);
    }
}

struct GraphPhaseProgress {
    sink: Rc<dyn GraphPhaseProgressSink>,
    failures: Rc<GraphPhaseFailures>,
    traces: RefCell<HashMap<String, GraphTracePhaseProgress>>,
    outcome: Rc<RefCell<GraphWorkloadReport>>,
    /// Whether this phase is the WARMUP phase, gating warmup-failure accounting.
    is_warmup: bool,
    /// Run-scoped ledger of WARMUP-phase trace ids that produced a terminal,
    /// non-cancelled failure. Shared across every phase's progress so the
    /// sequencer can abort before PROFILING. Mirrors agentx
    /// `graph_ir_replay.py:_record_warmup_failure` (a WARMUP return carrying a
    /// non-None error that is NOT cancelled).
    warmup_failed_trace_ids: Rc<RefCell<Vec<String>>>,
}

impl GraphPhaseProgress {
    fn new(
        sink: Rc<dyn GraphPhaseProgressSink>,
        failures: Rc<GraphPhaseFailures>,
        outcome: Rc<RefCell<GraphWorkloadReport>>,
        is_warmup: bool,
        warmup_failed_trace_ids: Rc<RefCell<Vec<String>>>,
    ) -> Self {
        Self {
            sink,
            failures,
            traces: RefCell::new(HashMap::new()),
            outcome,
            is_warmup,
            warmup_failed_trace_ids,
        }
    }

    /// Ledger one terminal (non-cancelled) WARMUP-phase failure for `trace_id`,
    /// deduplicated so each failed trace is reported once. No-op outside the
    /// WARMUP phase. Port of `graph_ir_replay.py:908` (`_record_warmup_failure`);
    /// the run sequencer consumes the ledger to abort before PROFILING
    /// (`phase/runner.py:578` `report_warmup_failures`).
    fn note_warmup_failure(&self, trace_id: &str) {
        if !self.is_warmup {
            return;
        }
        let mut ledger = self.warmup_failed_trace_ids.borrow_mut();
        if !ledger.iter().any(|existing| existing == trace_id) {
            ledger.push(trace_id.to_owned());
        }
    }

    fn admit(&self, info: &TraceAdmissionInfo) {
        if info.node_count == 0 {
            self.failures.record(format!(
                "graph trace {:?} contains no dispatchable nodes",
                info.trace_id
            ));
            return;
        }
        if self
            .traces
            .borrow_mut()
            .insert(
                info.trace_id.clone(),
                GraphTracePhaseProgress {
                    expected_nodes: info.node_count,
                    returned_nodes: 0,
                    first_token_uuids: HashSet::new(),
                    returned_uuids: HashSet::new(),
                },
            )
            .is_some()
        {
            self.failures.record(format!(
                "graph trace {:?} was admitted more than once",
                info.trace_id
            ));
            return;
        }
        let mut sent = Vec::with_capacity(info.node_count);
        sent.push(PhaseSend::single_turn_session());
        sent.extend((1..info.node_count).map(|_| PhaseSend::dag_child()));
        if let Err(error) = self.sink.record_sent_batch(&sent) {
            self.failures.record(format!(
                "recording graph trace {:?} admitted send batch: {error}",
                info.trace_id
            ));
        }
        let mut outcome = self.outcome.borrow_mut();
        outcome.admitted = outcome.admitted.saturating_add(1);
    }

    fn record(&self, record: &CapturedRecord) {
        let (completes_session, released_at_first_token) = {
            let mut traces = self.traces.borrow_mut();
            let Some(trace) = traces.get_mut(&record.x_correlation_id) else {
                self.failures.record(format!(
                    "graph trace {:?} emitted a node record before admission or after completion",
                    record.x_correlation_id
                ));
                return;
            };
            if trace.returned_uuids.contains(&record.uuid) {
                self.failures.record(format!(
                    "graph trace {:?} emitted duplicate terminal record for request {}",
                    record.x_correlation_id, record.uuid
                ));
                return;
            }
            if trace.returned_nodes >= trace.expected_nodes {
                self.failures.record(format!(
                    "graph trace {:?} emitted more than {} node records",
                    record.x_correlation_id, trace.expected_nodes
                ));
                return;
            }
            let released_at_first_token = trace.first_token_uuids.contains(&record.uuid);
            let metadata_has_first_token = record.ingest.first_token_ns.is_some();
            if released_at_first_token != metadata_has_first_token {
                self.failures.record(format!(
                    "graph trace {:?} request {} first-token event/record mismatch: event={} record={}",
                    record.x_correlation_id,
                    record.uuid,
                    released_at_first_token,
                    metadata_has_first_token
                ));
            }
            trace.returned_uuids.insert(record.uuid);
            trace.returned_nodes += 1;
            (
                trace.returned_nodes == trace.expected_nodes,
                released_at_first_token,
            )
        };
        // AgentX warmup gate: a WARMUP node return carrying a non-cancelled
        // error is a terminal warmup failure. The graph node-failure policy for
        // the warmup priming turns is resilient (a failed reply does not abort
        // the boundary trace), so the failure surfaces here as an errored
        // per-node record rather than a trace-level `complete()` error — the
        // exact analogue of Python `_on_graph_return`'s per-return dispatch to
        // `_record_warmup_failure` (`graph_ir_replay.py:964`).
        if record.ingest.errored && !record.ingest.canceled {
            self.note_warmup_failure(&record.x_correlation_id);
        }
        self.sink.record_returned(PhaseReturn {
            completes_session,
            cancelled: record.ingest.canceled,
            errored: record.ingest.errored,
            releases_prefill: !released_at_first_token,
        });
    }

    fn first_token(&self, trace_id: &str, uuid: Uuid) {
        let mut traces = self.traces.borrow_mut();
        let Some(trace) = traces.get_mut(trace_id) else {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted a first token before admission or after completion"
            ));
            return;
        };
        if trace.returned_uuids.contains(&uuid) {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted first-token event after terminal record for request {uuid}"
            ));
            return;
        }
        if trace.first_token_uuids.contains(&uuid) {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted duplicate first-token event for request {uuid}"
            ));
            return;
        }
        if trace.first_token_uuids.len() >= trace.expected_nodes {
            self.failures.record(format!(
                "graph trace {trace_id:?} emitted more than {} first-token events",
                trace.expected_nodes
            ));
            return;
        }
        trace.first_token_uuids.insert(uuid);
        self.sink.record_first_token();
    }

    fn complete(
        &self,
        trace_id: &str,
        node_count: usize,
        requires_node_records: bool,
        result: &Result<(), TraceError>,
    ) {
        let Some(trace) = self.traces.borrow_mut().remove(trace_id) else {
            self.failures.record(format!(
                "graph trace {trace_id:?} completed before admission or more than once"
            ));
            return;
        };
        if trace.expected_nodes != node_count {
            self.failures.record(format!(
                "graph trace {trace_id:?} completed with {node_count} nodes after reserving {}",
                trace.expected_nodes
            ));
        }
        let missing = trace.expected_nodes.saturating_sub(trace.returned_nodes);
        let orphan_first_tokens = trace
            .first_token_uuids
            .difference(&trace.returned_uuids)
            .count();
        if orphan_first_tokens > missing {
            self.failures.record(format!(
                "graph trace {trace_id:?} has {orphan_first_tokens} first-token events without terminals but only {missing} missing terminal records"
            ));
        }
        let first_tokens_without_terminals = orphan_first_tokens.min(missing);
        let (cancelled, mut errored) = match result {
            Ok(()) => (false, false),
            Err(TraceError::Cancelled(_)) => (true, false),
            Err(_) => (false, true),
        };
        if result.is_ok() && missing > 0 && requires_node_records {
            errored = true;
            self.failures.record(format!(
                "graph trace {trace_id:?} completed successfully without {missing} reserved node records"
            ));
        }
        for index in 0..missing {
            self.sink.record_returned(PhaseReturn {
                completes_session: index + 1 == missing,
                cancelled,
                errored,
                releases_prefill: index >= first_tokens_without_terminals,
            });
        }
        if let Err(error) = result
            && !cancelled
        {
            self.failures
                .record(format!("graph trace {trace_id:?} failed: {error}"));
            // A WARMUP trace that aborts (fail-fast node policy) rather than
            // completing with an errored record still counts as a terminal
            // warmup failure for the pre-profiling abort gate.
            self.note_warmup_failure(trace_id);
        }
        let mut outcome = self.outcome.borrow_mut();
        match result {
            Ok(()) => outcome.completed = outcome.completed.saturating_add(1),
            Err(TraceError::Cancelled(_)) => {
                outcome.cancelled = outcome.cancelled.saturating_add(1);
            }
            Err(_) => outcome.failed = outcome.failed.saturating_add(1),
        }
        outcome.traces.push(GraphTraceRunResult {
            trace_id: trace_id.to_owned(),
            result: result.clone(),
        });
    }
}

#[derive(Default)]
struct GraphPhaseFailures {
    messages: RefCell<Vec<String>>,
    notify: Notify,
}

impl GraphPhaseFailures {
    fn record(&self, message: impl Into<String>) {
        self.messages.borrow_mut().push(message.into());
        self.notify.notify_waiters();
    }

    fn first(&self) -> Option<String> {
        self.messages.borrow().first().cloned()
    }

    async fn wait(&self) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.first().is_some() {
                return;
            }
            notified.await;
        }
    }
}

struct GraphPhaseWorkloadObserver {
    progress: Rc<GraphPhaseProgress>,
}

impl GraphWorkloadObserver for GraphPhaseWorkloadObserver {
    fn on_trace_admit(&self, info: &TraceAdmissionInfo, _admit_ns: i64) {
        self.progress.admit(info);
    }

    fn on_sending_complete(&self, _at_ns: i64) {
        self.progress.sink.mark_all_sent();
    }
}

#[derive(Default)]
struct GraphRecordDrainStop {
    stopped: Cell<bool>,
    notify: Notify,
}

impl GraphRecordDrainStop {
    fn stop(&self) {
        if !self.stopped.replace(true) {
            self.notify.notify_waiters();
        }
    }
}

struct GraphPhaseExecution {
    clock: Rc<dyn Clock>,
    context: PhaseContext,
    workload: Rc<GraphWorkload>,
    placement: Rc<dyn TracePlacement>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_initial: Option<usize>,
    adaptive_control_variable: Option<AdaptiveControlVariable>,
    controller: Rc<dyn ScheduledPhaseController>,
    failures: Rc<GraphPhaseFailures>,
    events: RefCell<Option<mpsc::UnboundedReceiver<RunnerGraphExecutionEvent>>>,
    captured: Rc<RefCell<Vec<CapturedRecord>>>,
    progress: Rc<GraphPhaseProgress>,
    adaptive_sampler: Option<SharedWindowSampler>,
    sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    drain_stop: Rc<GraphRecordDrainStop>,
    drain_task: RefCell<Option<tokio::task::JoinHandle<()>>>,
    setup_error: Option<String>,
}

impl GraphPhaseExecution {
    fn start_record_drain(&self) -> Result<()> {
        ensure!(
            self.drain_task.borrow().is_none(),
            "graph record drain was already started"
        );
        let mut events = self
            .events
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("graph record receiver was already consumed"))?;
        let captured = self.captured.clone();
        let sampler = self.adaptive_sampler.clone();
        let progress = self.progress.clone();
        let stop = self.drain_stop.clone();
        *self.drain_task.borrow_mut() = Some(tokio::task::spawn_local(async move {
            loop {
                while let Ok(event) = events.try_recv() {
                    ingest_graph_execution_event(&captured, sampler.as_ref(), &progress, event);
                }
                if stop.stopped.get() {
                    return;
                }
                let stopped = stop.notify.notified();
                tokio::pin!(stopped);
                stopped.as_mut().enable();
                if stop.stopped.get() {
                    continue;
                }
                tokio::select! {
                    biased;
                    event = events.recv() => match event {
                        Some(event) => ingest_graph_execution_event(
                            &captured,
                            sampler.as_ref(),
                            &progress,
                            event,
                        ),
                        None => return,
                    },
                    () = &mut stopped => {}
                }
            }
        }));
        Ok(())
    }
}

fn ingest_graph_execution_event(
    captured: &Rc<RefCell<Vec<CapturedRecord>>>,
    sampler: Option<&SharedWindowSampler>,
    progress: &GraphPhaseProgress,
    event: RunnerGraphExecutionEvent,
) {
    match event {
        RunnerGraphExecutionEvent::FirstToken { trace_id, uuid } => {
            progress.first_token(&trace_id, uuid);
        }
        RunnerGraphExecutionEvent::Record(record) => {
            if let Some(sampler) = sampler {
                sampler.borrow_mut().on_record(&record.ingest);
            }
            progress.record(&record);
            captured.borrow_mut().push(*record);
        }
        RunnerGraphExecutionEvent::TraceComplete {
            trace_id,
            node_count,
            requires_node_records,
            result,
        } => {
            progress.complete(&trace_id, node_count, requires_node_records, &result);
        }
    }
}

impl PhaseExecution for GraphPhaseExecution {
    fn configure(&self, config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        if let Some(error) = &self.setup_error {
            return Err(PhaseExecutionError::new(error.clone()));
        }
        if self.adaptive_control_variable != Some(AdaptiveControlVariable::Concurrency)
            && let (Some(limit), Some(slots)) = (config.concurrency, &self.session_slots)
        {
            slots.set_limit(limit);
        }
        if self.adaptive_control_variable != Some(AdaptiveControlVariable::PrefillConcurrency)
            && let Some(limit) = self.prefill_initial
        {
            self.placement
                .set_prefill_limit(limit)
                .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
        }
        Ok(())
    }

    fn setup(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let sidecars = self.sidecars.clone();
        let clock = self.clock.clone();
        Box::pin(async move {
            for sidecar in &sidecars {
                sidecar.start().await.map_err(|error| {
                    PhaseExecutionError::new(format!("starting graph phase sidecar: {error:#}"))
                })?;
            }
            let phase_start_ns = clock.now_ns();
            for sidecar in &sidecars {
                sidecar.on_phase_start(phase_start_ns);
            }
            Ok(())
        })
    }

    fn start_ramps(&self) -> Result<(), PhaseExecutionError> {
        self.start_record_drain()
            .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
        self.controller
            .start()
            .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
        if let Some(error) = self.failures.first() {
            return Err(PhaseExecutionError::new(error));
        }
        Ok(())
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let workload = self.workload.clone();
        let context = self.context.clone();
        let controller = self.controller.clone();
        let failures = self.failures.clone();
        Box::pin(async move {
            let execute = workload.execute();
            let adaptive_stop = controller.wait_until_stop();
            let failed = failures.wait();
            tokio::pin!(execute);
            tokio::pin!(adaptive_stop);
            tokio::pin!(failed);
            tokio::select! {
                biased;
                () = &mut failed => Err(PhaseExecutionError::new(
                    failures.first().unwrap_or_else(|| "graph phase failed".into())
                )),
                () = &mut adaptive_stop => {
                    workload.cancel();
                    context.mark_all_sent();
                    Ok(())
                }
                result = &mut execute => result
                    .map(|_| ())
                    .map_err(|error| PhaseExecutionError::new(error.to_string())),
            }
        })
    }

    fn stop_issuing(&self) {
        self.workload.cancel();
    }

    fn cancel_inflight(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let result = self
            .placement
            .cancel_inflight()
            .map_err(|error| PhaseExecutionError::new(error.to_string()));
        Box::pin(async move { result })
    }

    fn stop_ramps(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let controller = self.controller.clone();
        let failures = self.failures.clone();
        Box::pin(async move {
            controller
                .stop()
                .await
                .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
            match failures.first() {
                Some(error) => Err(PhaseExecutionError::new(error)),
                None => Ok(()),
            }
        })
    }

    fn finalize(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        self.drain_stop.stop();
        let drain = self.drain_task.borrow_mut().take();
        let failures = self.failures.clone();
        let sidecars = self.sidecars.clone();
        let clock = self.clock.clone();
        let progress = self.progress.clone();
        Box::pin(async move {
            if let Some(drain) = drain {
                drain.await.map_err(|error| {
                    PhaseExecutionError::new(format!("graph record drain failed: {error}"))
                })?;
            }
            let phase_end_ns = clock.now_ns();
            for sidecar in &sidecars {
                sidecar.on_phase_end(phase_end_ns);
            }
            for sidecar in &sidecars {
                sidecar.finish().await.map_err(|error| {
                    PhaseExecutionError::new(format!("finishing graph phase sidecar: {error:#}"))
                })?;
            }
            if let Some(error) = failures.first() {
                return Err(PhaseExecutionError::new(error));
            }
            // The record drain has now processed every WARMUP return, so the
            // shared warmup-failure ledger is complete. A WARMUP phase that
            // recorded a terminal (non-cancelled) trace failure fails HERE so
            // the orchestrator never advances to PROFILING — the resilient
            // warmup node policy lets the boundary trace complete, so this
            // finalize-time gate is the point at which the run must stop.
            // Mirrors agentx `report_warmup_failures` raising after warmup
            // returning-complete (`phase/runner.py:578`); `run_graph_phases`
            // then renders the structured `trajectory_warmup_failed` envelope
            // from the same ledger.
            if progress.is_warmup && !progress.warmup_failed_trace_ids.borrow().is_empty() {
                return Err(PhaseExecutionError::new(
                    "warmup phase recorded terminal trace failures; aborting before profiling",
                ));
            }
            Ok(())
        })
    }
}

struct GraphPhaseExecutionFactory {
    phases: RefCell<HashMap<String, PreparedGraphPhase>>,
    sidecars: RefCell<HashMap<String, Vec<Rc<dyn ScheduledPhaseSidecar>>>>,
    placements: Vec<Rc<dyn TracePlacement>>,
    captured: Rc<RefCell<Vec<CapturedRecord>>>,
    outcome: Rc<RefCell<GraphWorkloadReport>>,
    /// Run-scoped ledger of terminal WARMUP-phase trace failures, consumed by
    /// [`run_graph_phases`] to abort before PROFILING.
    warmup_failed_trace_ids: Rc<RefCell<Vec<String>>>,
}

impl PhaseExecutionFactory for GraphPhaseExecutionFactory {
    fn create(&self, config: &PhaseConfig, context: PhaseContext) -> Rc<dyn PhaseExecution> {
        let Some(prepared) = self.phases.borrow_mut().remove(&config.id) else {
            return Rc::new(FailedGraphPhaseExecution {
                error: format!("graph phase {:?} has no prepared execution plan", config.id),
            });
        };
        let progress = Rc::new(GraphPhaseProgress::new(
            Rc::new(context.clone()),
            prepared.failures.clone(),
            self.outcome.clone(),
            prepared.is_warmup,
            self.warmup_failed_trace_ids.clone(),
        ));
        let observer = Rc::new(GraphPhaseWorkloadObserver {
            progress: progress.clone(),
        });
        let workload = Rc::new(prepared.workload.with_observer(observer));
        let mut setup_error = None;
        let mut controller = prepared.controller;
        let adaptive_control_variable = prepared
            .adaptive
            .as_ref()
            .map(|adaptive| adaptive.control_variable);
        let adaptive_sampler = prepared.adaptive.map(|adaptive| {
            let sampler: SharedWindowSampler = Rc::new(RefCell::new(Box::new(
                TumblingWindowSampler::new(context.clock().now_ns()),
            )));
            match graph_adaptive_actuator(
                &adaptive,
                prepared.session_slots.clone(),
                prepared.intervals.clone(),
                prepared.placement.clone(),
            )
            .and_then(|actuator| {
                build_adaptive_scale(adaptive, context.clock(), actuator, sampler.clone())
            }) {
                Ok(scale) => {
                    controller = Rc::new(AdaptiveScheduledPhaseController::new(
                        scale,
                        controller.clone(),
                    ));
                }
                Err(error) => setup_error = Some(error.to_string()),
            }
            sampler
        });
        let sidecars = self
            .sidecars
            .borrow_mut()
            .remove(&config.id)
            .unwrap_or_default();
        Rc::new(GraphPhaseExecution {
            clock: context.clock(),
            context,
            workload,
            placement: prepared.placement,
            session_slots: prepared.session_slots,
            prefill_initial: prepared.prefill_initial,
            adaptive_control_variable,
            controller,
            failures: prepared.failures,
            events: RefCell::new(Some(prepared.events)),
            captured: self.captured.clone(),
            progress,
            adaptive_sampler,
            sidecars,
            drain_stop: Rc::new(GraphRecordDrainStop::default()),
            drain_task: RefCell::new(None),
            setup_error,
        })
    }

    fn cancel_all(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let errors = self
            .placements
            .iter()
            .enumerate()
            .filter_map(|(index, placement)| {
                placement
                    .cancel_inflight()
                    .err()
                    .map(|error| format!("placement {index}: {error}"))
            })
            .collect::<Vec<_>>();
        let result = if errors.is_empty() {
            Ok(())
        } else {
            Err(PhaseExecutionError::new(format!(
                "cancelling graph placements: {}",
                errors.join("; ")
            )))
        };
        Box::pin(async move { result })
    }
}

struct FailedGraphPhaseExecution {
    error: String,
}

impl PhaseExecution for FailedGraphPhaseExecution {
    fn configure(&self, _config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        Err(PhaseExecutionError::new(self.error.clone()))
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let error = PhaseExecutionError::new(self.error.clone());
        Box::pin(async move { Err(error) })
    }
}

fn graph_adaptive_actuator(
    config: &AdaptiveRunConfig,
    session_slots: Option<Rc<SlotPool>>,
    intervals: Rc<RefCell<Box<dyn aiperf::timing::IntervalGenerator>>>,
    placement: Rc<dyn TracePlacement>,
) -> Result<Rc<dyn ControlActuator>> {
    Ok(match config.control_variable {
        AdaptiveControlVariable::Concurrency => Rc::new(SessionConcurrencyActuator::new(
            session_slots
                .ok_or_else(|| anyhow!("adaptive graph concurrency requires session admission"))?,
            integer_adaptive_bound(config.minimum, "concurrency minimum")?,
            integer_adaptive_bound(config.maximum, "concurrency maximum")?,
        )?),
        AdaptiveControlVariable::PrefillConcurrency => Rc::new(GraphPrefillActuator::new(
            placement,
            integer_adaptive_bound(config.minimum, "prefill minimum")?,
            integer_adaptive_bound(config.maximum, "prefill maximum")?,
        )?),
        AdaptiveControlVariable::RequestRate => Rc::new(RequestRateActuator::new(
            intervals,
            config.minimum,
            config.maximum,
        )?),
        AdaptiveControlVariable::Users => {
            bail!("adaptive users is not defined for Graph-IR phases")
        }
    })
}

struct GraphPrefillActuator {
    placement: Rc<dyn TracePlacement>,
    minimum: usize,
    maximum: usize,
    current: Cell<usize>,
}

impl GraphPrefillActuator {
    fn new(placement: Rc<dyn TracePlacement>, minimum: usize, maximum: usize) -> Result<Self> {
        ensure!(
            minimum > 0,
            "adaptive graph prefill minimum must be positive"
        );
        ensure!(
            maximum > minimum,
            "adaptive graph prefill maximum must be greater than minimum"
        );
        Ok(Self {
            placement,
            minimum,
            maximum,
            current: Cell::new(minimum),
        })
    }
}

impl ControlActuator for GraphPrefillActuator {
    fn variable(&self) -> &'static str {
        "prefill_concurrency"
    }

    fn minimum(&self) -> f64 {
        self.minimum as f64
    }

    fn maximum(&self) -> f64 {
        self.maximum as f64
    }

    fn current(&self) -> f64 {
        self.current.get() as f64
    }

    fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
        if !value.is_finite() {
            return Err(AdaptiveError::Actuator(format!(
                "graph prefill control value must be finite, got {value}"
            )));
        }
        let value = value
            .clamp(self.minimum as f64, self.maximum as f64)
            .trunc() as usize;
        self.placement
            .set_prefill_limit(value)
            .map_err(|error| AdaptiveError::Actuator(error.to_string()))?;
        self.current.set(value);
        Ok(value as f64)
    }

    fn snapshot(&self) -> ControlSnapshot {
        ControlSnapshot {
            target_value: self.current(),
            actual_value: self.current(),
            active_users: None,
            retiring_users: None,
            cancelled: None,
        }
    }
}

/// Lower and execute every authored Graph-IR phase through one lifecycle path.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_graph_phases(
    phases: &[PhaseSpec],
    benchmark_id: &str,
    artifact_dir: &Path,
    input: &GraphInputBundle,
    clock: Rc<dyn Clock>,
    rng_root: RngRoot,
    allow_dataset_wrap: bool,
    t_star: TStarWindow,
    phase_sidecars: Vec<Vec<Rc<dyn ScheduledPhaseSidecar>>>,
    backends: &dyn RunnerGraphPhaseBackendFactory,
    on_failure: OnFailure,
) -> Result<GraphPhaseRunOutput> {
    validate_dataset_wrap_policy(phases, input, allow_dataset_wrap)?;
    ensure!(
        phase_sidecars.len() == phases.len(),
        "graph phase sidecars must be provided one list per phase"
    );
    let trace_instances = GraphTraceInstanceSequence::default();
    let session_slots = phases
        .iter()
        .any(graph_phase_uses_session_admission)
        .then(|| Rc::new(SlotPool::new(1)));
    let mut prepared = Vec::with_capacity(phases.len());
    for (phase_index, phase) in phases.iter().enumerate() {
        prepared.push(prepare_graph_phase(
            phase_index,
            phase,
            benchmark_id,
            artifact_dir,
            input,
            clock.clone(),
            rng_root,
            t_star,
            trace_instances.clone(),
            session_slots.clone(),
            backends,
            on_failure,
        )?);
    }

    let captured = Rc::new(RefCell::new(Vec::new()));
    let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
    let placements = prepared
        .iter()
        .map(|phase| phase.placement.clone())
        .collect();
    let phase_configs = phases
        .iter()
        .enumerate()
        .map(|(index, spec)| phase_config(spec, phase_seamless_to_next(phases, index)))
        .collect::<Result<Vec<_>>>()?;
    let prepared = prepared
        .into_iter()
        .zip(&phase_configs)
        .map(|(phase, config)| (config.id.clone(), phase))
        .collect::<HashMap<_, _>>();
    // Side-channel telemetry producers are barrier-synchronized, not per-token,
    // so the graph runtime drives them through the shared phase-sidecar seam the
    // scheduled runtime uses: start before issuance, finish after drain.
    let sidecars = phase_configs
        .iter()
        .zip(phase_sidecars)
        .map(|(config, sidecars)| (config.id.clone(), sidecars))
        .collect::<HashMap<_, _>>();
    let warmup_failed_trace_ids = Rc::new(RefCell::new(Vec::new()));
    let execution_factory: Rc<dyn PhaseExecutionFactory> = Rc::new(GraphPhaseExecutionFactory {
        phases: RefCell::new(prepared),
        sidecars: RefCell::new(sidecars),
        placements,
        captured: captured.clone(),
        outcome: outcome.clone(),
        warmup_failed_trace_ids: warmup_failed_trace_ids.clone(),
    });
    let phase_observer: Rc<dyn PhaseObserver> = Rc::new(NoopPhaseObserver);
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock,
        phase_observer.clone(),
        execution_factory,
    ));
    let orchestrator = ClockPhaseOrchestrator::new(phase_configs, runner_factory, phase_observer)?;
    // A terminal (non-cancelled) failure during the WARMUP phase already fails
    // that phase, so `run_all` returns before PROFILING starts. Before
    // propagating any error we consult the warmup ledger: if the WARMUP phase
    // recorded terminal trace failures, the run aborts with the structured
    // `trajectory_warmup_failed` envelope (the `TrajectoryWarmupFailedError`
    // analogue) so benchmark numbers are never taken from a pool the warmup
    // could not faithfully prime. Mirrors agentx `phase/runner.py:578`
    // (`report_warmup_failures`) raising before PROFILING.
    let run_result = orchestrator.run_all().await;
    let warmup_failures = std::mem::take(&mut *warmup_failed_trace_ids.borrow_mut());
    if !warmup_failures.is_empty() {
        return Err(trajectory_warmup_failed_error(&warmup_failures));
    }
    let phase_stats = run_result?;

    let mut captured = std::mem::take(&mut *captured.borrow_mut());
    captured.sort_by(|left, right| {
        left.ingest
            .start_ns
            .cmp(&right.ingest.start_ns)
            .then_with(|| left.uuid.cmp(&right.uuid))
    });
    for (request_index, record) in captured.iter_mut().enumerate() {
        record.ingest.request_index = Some(request_index);
    }
    let workload = std::mem::take(&mut *outcome.borrow_mut());
    Ok(GraphPhaseRunOutput {
        captured,
        phases: phase_stats,
        workload,
    })
}

/// Build the structured `trajectory_warmup_failed` protocol-v2 execution
/// failure for a WARMUP phase that recorded terminal trace failures.
///
/// The wire envelope is a stable lowercase-snake-case `code`
/// (`trajectory_warmup_failed`, the `kind`) plus a message listing the failed
/// trace ids — the `{ kind, failed_trace_ids }` analogue in the runner's
/// `code` + `message` diagnostic shape. Returned as a downcastable
/// [`PreparedRunFailure`] so the process coordinator emits an
/// execution-stage failure envelope rather than the generic `execution_failed`
/// fallback. Mirrors Python `TrajectoryWarmupFailedError`
/// (`src/aiperf/common/scenario/base.py:182`), whose message likewise embeds the
/// failed trace ids, raised by `report_warmup_failures`
/// (`graph_ir_replay.py:908-931`, `phase/runner.py:578`).
fn trajectory_warmup_failed_error(failed_trace_ids: &[String]) -> anyhow::Error {
    let message = format!(
        "Trajectory warmup failed for {} trace(s): {}. Run aborted to preserve metrics integrity.",
        failed_trace_ids.len(),
        failed_trace_ids.join(", ")
    );
    match PreparedRunFailure::new(
        RunnerFailureStageV2::Execution,
        "trajectory_warmup_failed",
        message,
    ) {
        Ok(failure) => anyhow::Error::new(failure),
        Err(error) => error,
    }
}

fn validate_dataset_wrap_policy(
    phases: &[PhaseSpec],
    input: &GraphInputBundle,
    allow_dataset_wrap: bool,
) -> Result<()> {
    if allow_dataset_wrap {
        return Ok(());
    }
    let distinct = u64::try_from(input.plans.len()).context("graph root count exceeds u64")?;
    for phase in phases {
        let common = phase.common();
        let one_pass =
            common.sessions.is_none() && common.requests.is_none() && common.duration.is_none();
        if one_pass || common.sessions.is_some_and(|sessions| sessions <= distinct) {
            continue;
        }
        let concurrency = phase.concurrency().unwrap_or(1);
        if u64::try_from(concurrency).context("graph concurrency exceeds u64")? > distinct {
            bail!(
                "graph phase {:?} concurrency {} exceeds the {} distinct loaded traces while dataset wrapping is disabled; reduce concurrency to at most {}, bound sessions within the loaded corpus, or set dataset.synthesis.allow_dataset_wrap=true",
                common.name,
                concurrency,
                distinct,
                distinct
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prepare_graph_phase(
    phase_index: usize,
    phase: &PhaseSpec,
    benchmark_id: &str,
    artifact_dir: &Path,
    input: &GraphInputBundle,
    clock: Rc<dyn Clock>,
    rng_root: RngRoot,
    t_star: TStarWindow,
    trace_instances: GraphTraceInstanceSequence,
    session_slots: Option<Rc<SlotPool>>,
    backends: &dyn RunnerGraphPhaseBackendFactory,
    on_failure: OnFailure,
) -> Result<PreparedGraphPhase> {
    let phase_index = u64::try_from(phase_index).context("graph phase index exceeds u64")?;
    let phase_rng = rng_root.derive_indexed_root(namespace::GRAPH_PHASE, phase_index);
    let common = phase.common();
    let one_pass =
        common.sessions.is_none() && common.requests.is_none() && common.duration.is_none();
    let session_limit = if one_pass {
        Some(u64::try_from(input.plans.len()).context("graph root count exceeds u64")?)
    } else {
        common.sessions
    };
    // A multi-cell process owns only its interleaved slice of the global session
    // ordinals, so it swaps the single-cell cycler for the partitioned source. The
    // `common.requests.is_none()` guard keeps a configured static-node request budget
    // on the cycler because `PartitionedGraphTraceSource` does not yet slice that
    // budget across cells; a later controller step must own the split before a
    // per-cell run can honor a static request_limit.
    // Per-phase trajectory-start snapshot split: the warmup phase primes each
    // trace's boundary turns (`rewrite_for_warmup`), the profiling (and any other
    // non-warmup) phase replays only the post-`t*` frontier (`chop_trie_at_tstar`).
    // At the default `[0, 0]` window `t*` is `0` for every trace, so profiling is
    // the unchanged full graph and warmup is empty. Both phases sample the SAME
    // deterministic `t*` per trace, so warmup primes exactly what profiling resumes.
    let phase_plans = apply_tstar_split(&input.plans, phase, t_star);
    let source: Rc<dyn GraphTraceSource> = match ModuloCellPartition::from_env() {
        Some(partition) if partition.cell_count() > 1 && common.requests.is_none() => {
            tracing::debug!(
                cell_id = partition.cell_id(),
                cell_count = partition.cell_count(),
                "graph phase using partitioned trace source for cell"
            );
            Rc::new(PartitionedGraphTraceSource::new(
                phase_plans,
                session_limit,
                partition.cell_id(),
                partition.cell_count(),
            )?)
        }
        _ => Rc::new(CyclingGraphTraceSource::with_budgets_and_sequence(
            phase_plans,
            session_limit,
            common.requests,
            trace_instances,
        )?),
    };
    let seed = phase_rng.derive_seed_or_entropy(namespace::GRAPH_ARRIVAL);
    let intervals = Rc::new(RefCell::new(match phase.request_arrival() {
        Some((pattern, rate, smoothness)) => {
            make_interval_generator(pattern, rate, smoothness, seed)
        }
        None => make_interval_generator(
            aiperf::timing::ArrivalPattern::ConcurrencyBurst,
            None,
            None,
            seed,
        ),
    }));
    let arrival: Rc<dyn GraphArrivalPolicy> = match phase {
        PhaseSpec::Concurrency { .. } => Rc::new(ImmediateGraphArrival),
        PhaseSpec::Poisson { .. } | PhaseSpec::Gamma { .. } | PhaseSpec::Constant { .. } => {
            Rc::new(IntervalGraphArrival::new(intervals.clone()))
        }
        PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } => {
            unreachable!("unsupported graph phase rejected before input acquisition")
        }
    };
    let (events_tx, events_rx) = mpsc::unbounded_channel();
    let event_sink: Arc<dyn RunnerGraphExecutionEventSink> =
        Arc::new(ChannelRunnerGraphExecutionEventSink::new(events_tx));
    let adaptive = graph_adaptive_config(phase, benchmark_id, artifact_dir)?;
    let prefill_initial = match (common.prefill_concurrency, adaptive.as_ref()) {
        (Some(limit), _) => Some(limit),
        (None, Some(config))
            if config.control_variable == AdaptiveControlVariable::PrefillConcurrency =>
        {
            Some(integer_adaptive_bound(config.minimum, "prefill minimum")?)
        }
        (None, _) => None,
    };
    let cancellation = common
        .cancellation
        .map(|cancellation| GraphCancellationConfig {
            rate: cancellation.rate,
            delay_seconds: cancellation.delay,
            rng_root: phase_rng.derive_root(namespace::GRAPH_NODE_CANCELLATION),
            phase: if common.name == "warmup" {
                aiperf::timing::Phase::Warmup
            } else {
                aiperf::timing::Phase::Profiling
            },
        });
    let backend = backends.prepare_backend(GraphPhaseBackendConfig {
        metrics_phase: metrics_phase(phase)?,
        prefill_concurrency: prefill_initial,
        cancellation,
        events: event_sink.clone(),
    })?;
    let placement: Rc<dyn TracePlacement> = Rc::new(ObservedRunnerGraphPlacement::new(
        backend.placement,
        event_sink,
        backend.requires_node_records,
    ));
    let failures = Rc::new(GraphPhaseFailures::default());
    let controller = graph_ramp_controller(
        phase,
        clock.clone(),
        intervals.clone(),
        session_slots.clone(),
        placement.clone(),
        phase_rng,
        failures.clone(),
    )?;
    // Run-level failure discipline is config-selected (default fail-fast for the
    // graph path). `Abort` latches the whole run on the first non-cancellation
    // trace failure; `Continue` keeps admitting unrelated roots and records the
    // failed traces (the coordinator relaxes its `failed == 0` assertion to
    // match). See `specs/2026-07-13-scheduled-graph-convergence-implementation.md`.
    let run_failure: Rc<dyn RunFailurePolicy> = match on_failure {
        OnFailure::Abort => Rc::new(FailFastRunFailurePolicy::default()),
        OnFailure::Continue => Rc::new(ContinueRunFailurePolicy),
    };
    let mut workload = GraphWorkload::new(clock, source, placement.clone())
        .with_arrival(arrival)
        .with_run_failure(run_failure);
    if graph_phase_uses_session_admission(phase) {
        workload = workload.with_admission(Rc::new(SlotPoolTraceAdmission::new(
            session_slots
                .clone()
                .ok_or_else(|| anyhow!("graph phase requires shared session admission"))?,
        )));
    }
    Ok(PreparedGraphPhase {
        workload,
        placement,
        events: events_rx,
        intervals,
        session_slots,
        prefill_initial,
        controller,
        failures,
        adaptive,
        is_warmup: common.name == "warmup",
    })
}

/// Apply the per-trace trajectory-start (`t*`) snapshot split for one phase.
///
/// For each root plan the intrinsic replayable span is measured
/// ([`trace_duration_us`]), a per-trace `t*` is drawn from the window
/// ([`WindowTStarSampler`]), and the plan graph is rewritten: the WARMUP phase
/// keeps only the boundary priming turns ([`rewrite_for_warmup`]); every other
/// phase (profiling) keeps only the live post-`t*` frontier
/// ([`chop_trie_at_tstar`]).
///
/// The default `[0, 0]` window is a TRUE NO-OP: BOTH warmup and profiling return
/// the plans UNCHANGED. This restores pre-`t*`-split behavior — a plain
/// (`dag_jsonl` / no-scenario) run has a normal warmup that dispatches the full
/// graph. The t* warmup/profiling split only engages when a snapshot window is
/// actually configured (non-default). In the Python source the warmup graph is
/// routed through [`rewrite_for_warmup`] only under the GRAPH_IR replay strategy
/// that a t* scenario activates; without a scenario, `rewrite_for_warmup` at
/// `t* = 0` would empty the warmup graph and every trace would report "contains
/// no dispatchable nodes". The default-window early return skips both the
/// per-trace sample and the rewrite (also avoiding a needless per-plan clone).
///
/// Warmup and profiling are independent phases that each clone `plans`, but they
/// sample the identical deterministic `t*` per trace (same window, seed, trace
/// id, and lane), so warmup primes exactly the prefix profiling resumes from.
///
/// Lane index: the base single-pass product selection dispatches each root
/// template once, matching Python's first/only lane (`0`). The trace source mints
/// per-instance ids only AFTER this seam, so recycled-corpus per-lane `t*`
/// decorrelation is a future refinement; lane `0` is correct for the one-pass
/// case. Mirrors `graph_ir_source.py:_plan_trace` selecting the chop vs the
/// warmup rewrite per phase.
fn apply_tstar_split(
    plans: &[GraphTracePlan],
    phase: &PhaseSpec,
    t_star: TStarWindow,
) -> Vec<GraphTracePlan> {
    // Default `[0, 0]` window: the split is inactive, so leave both warmup and
    // profiling graphs untouched (pre-split behavior). Only a configured
    // (non-default) window engages the per-trace warmup/profiling rewrite.
    if t_star.start_min_ratio == 0.0 && t_star.start_max_ratio == 0.0 {
        return plans.to_vec();
    }
    let is_warmup = phase.common().name == "warmup";
    let sampler = WindowTStarSampler {
        start_min_ratio: t_star.start_min_ratio,
        start_max_ratio: t_star.start_max_ratio,
        random_seed: t_star.random_seed,
    };
    plans
        .iter()
        .map(|plan| {
            // A single-trace `ParsedGraph` view over the already-resolved plan
            // graph. The view trace clears `graph_ref` so `resolve_trace_graph`
            // returns this graph directly (recorded plans carry the resolved
            // graph on the plan, not a named `graph_ref`).
            let view_trace = TraceRecord {
                id: plan.trace.id.clone(),
                graph_ref: None,
                initial_state: plan.trace.initial_state.clone(),
            };
            let parsed = ParsedGraph {
                graph: plan.graph.clone(),
                graphs: std::collections::BTreeMap::new(),
                traces: vec![view_trace.clone()],
            };
            let duration_us = trace_duration_us(&parsed, &view_trace);
            let t = sampler.sample_t_star(&plan.trace.id, 0, duration_us);
            let rewritten = if is_warmup {
                rewrite_for_warmup(&parsed, t)
            } else {
                chop_trie_at_tstar(&parsed, t)
            };
            GraphTracePlan {
                graph: rewritten.graph,
                trace: plan.trace.clone(),
                arrival_offset_ns: plan.arrival_offset_ns,
            }
        })
        .collect()
}

fn graph_phase_uses_session_admission(phase: &PhaseSpec) -> bool {
    phase.concurrency().is_some()
        || phase
            .common()
            .adaptive_scale
            .as_ref()
            .is_some_and(|adaptive| {
                matches!(
                    adaptive.control_variable,
                    AdaptiveControlVariableSpec::Concurrency
                )
            })
}

#[allow(clippy::too_many_arguments)]
fn graph_ramp_controller(
    spec: &PhaseSpec,
    clock: Rc<dyn Clock>,
    intervals: Rc<RefCell<Box<dyn aiperf::timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    placement: Rc<dyn TracePlacement>,
    rng_root: RngRoot,
    failures: Rc<GraphPhaseFailures>,
) -> Result<Rc<dyn ScheduledPhaseController>> {
    let common = spec.common();
    let rng_roots = RampActuatorRngRoots::from_phase_root(rng_root);
    let target_rate = spec
        .request_arrival()
        .and_then(|(_, target_rate, _)| target_rate);
    let mut drivers = Vec::new();
    if let Some(ramp) = &common.concurrency_ramp {
        let target = spec
            .concurrency()
            .ok_or_else(|| anyhow!("concurrency_ramp requires a concurrency target"))?;
        let slots = session_slots
            .clone()
            .ok_or_else(|| anyhow!("concurrency_ramp requires graph session admission"))?;
        let strategy = ramp_strategy(ramp, 1.0, target as f64, false, rng_roots.concurrency())?;
        drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
            slots.set_limit(value.round() as usize)
        }));
    }
    if let Some(ramp) = &common.prefill_ramp {
        let target = common
            .prefill_concurrency
            .ok_or_else(|| anyhow!("prefill_ramp requires prefill_concurrency"))?;
        let strategy = ramp_strategy(
            ramp,
            1.0,
            target as f64,
            false,
            rng_roots.prefill_concurrency(),
        )?;
        let placement = placement.clone();
        let failures = failures.clone();
        drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
            if let Err(error) = placement.set_prefill_limit(value.round() as usize) {
                failures.record(format!("applying graph prefill ramp: {error}"));
            }
        }));
    }
    if let Some(ramp) = &common.rate_ramp {
        let target = target_rate.ok_or_else(|| anyhow!("rate_ramp requires a rate phase"))?;
        let duration_ns = seconds_to_u64_ns(ramp.duration)?;
        let start = target * RATE_RAMP_UPDATE_INTERVAL_NS as f64 / duration_ns as f64;
        let strategy = ramp_strategy(ramp, start, target, true, rng_roots.request_rate())?;
        drivers.push(RampDriver::new(clock, strategy, move |value| {
            intervals.borrow_mut().set_rate(value)
        }));
    }
    if drivers.is_empty() {
        Ok(Rc::new(aiperf::phase_runtime::NoopScheduledPhaseController))
    } else {
        Ok(Rc::new(RampScheduledPhaseController::new(drivers)))
    }
}

fn graph_adaptive_config(
    phase: &PhaseSpec,
    benchmark_id: &str,
    artifact_dir: &Path,
) -> Result<Option<AdaptiveRunConfig>> {
    let Some(config) = adaptive_run_config(phase, benchmark_id, artifact_dir)? else {
        return Ok(None);
    };
    match config.control_variable {
        AdaptiveControlVariable::Concurrency => {
            ensure!(
                phase.common().concurrency_ramp.is_none(),
                "adaptive graph concurrency cannot be combined with concurrency_ramp"
            );
        }
        AdaptiveControlVariable::PrefillConcurrency => {
            ensure!(
                phase.common().prefill_ramp.is_none(),
                "adaptive graph prefill_concurrency cannot be combined with prefill_ramp"
            );
            let session_target = phase.concurrency().ok_or_else(|| {
                anyhow!("adaptive graph prefill_concurrency requires a session concurrency cap")
            })?;
            ensure!(
                config.maximum <= session_target as f64,
                "adaptive graph prefill_concurrency maximum must be <= concurrency"
            );
        }
        AdaptiveControlVariable::RequestRate => {
            ensure!(
                phase.request_arrival().is_some(),
                "adaptive graph request_rate requires a rate-controlled phase"
            );
            ensure!(
                phase.common().rate_ramp.is_none(),
                "adaptive graph request_rate cannot be combined with rate_ramp"
            );
        }
        AdaptiveControlVariable::Users => {
            bail!("adaptive users is not defined for Graph-IR phases")
        }
    }
    Ok(Some(config))
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::rc::Rc;

    use aiperf::adaptive_core::{SharedWindowSampler, TumblingWindowSampler};
    use aiperf::dataset::SegmentPool;
    use aiperf::graph::errors::TraceError;
    use aiperf::graph::model::{GraphRecord, GraphTracePlan, TraceRecord};
    use aiperf::graph::workload::{GraphWorkloadReport, TraceAdmissionInfo};
    use aiperf::timing::{PhaseReturn, PhaseSend};
    use uuid::Uuid;

    use super::*;
    use crate::records::CapturedModelOutput;

    fn wrap_policy_input(root_count: usize) -> GraphInputBundle {
        let plans = (0..root_count)
            .map(|index| GraphTracePlan {
                graph: GraphRecord::default(),
                trace: TraceRecord {
                    id: format!("root-{index}"),
                    graph_ref: None,
                    initial_state: Default::default(),
                },
                arrival_offset_ns: None,
            })
            .collect();
        GraphInputBundle {
            plans,
            segments: Arc::new(SegmentPool::new().freeze()),
            metadata: aiperf::graph::input::GraphInputMetadata {
                format: "weka_trace".into(),
                root_count,
                node_count: 0,
            },
        }
    }

    fn concurrency_phase(concurrency: usize, sessions: Option<u64>) -> PhaseSpec {
        let mut value = serde_json::json!({
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "concurrency": concurrency,
        });
        if let Some(sessions) = sessions {
            value["sessions"] = serde_json::json!(sessions);
        }
        serde_json::from_value(value).unwrap()
    }

    #[test]
    fn graph_pressure_grace_matches_python_derived_values() {
        // _graph_pressure_grace_sec: explicit user_grace honored verbatim.
        assert_eq!(graph_pressure_grace_sec(Some(30.0), 45.0), 30.0);
        assert_eq!(graph_pressure_grace_sec(Some(45.0), 30.0), 45.0);
        // Explicit grace of 0.0 is still honored (not treated as absent).
        assert_eq!(graph_pressure_grace_sec(Some(0.0), 100.0), 0.0);
        // None -> min(cache_pressure, cap); below cap returns the duration.
        assert_eq!(graph_pressure_grace_sec(None, 10.0), 10.0);
        // At the cap boundary (300.0) returns the cap exactly.
        assert_eq!(
            graph_pressure_grace_sec(None, PRESSURE_DRAIN_GRACE_CAP_SEC),
            300.0
        );
        // Above the cap clamps to the cap.
        assert_eq!(graph_pressure_grace_sec(None, 10_000.0), 300.0);
    }

    #[test]
    fn recorded_graph_wrap_policy_rejects_unintentional_lane_cloning() {
        let input = wrap_policy_input(2);
        let phases = [concurrency_phase(3, Some(3))];
        let error = validate_dataset_wrap_policy(&phases, &input, false).unwrap_err();
        assert!(format!("{error:#}").contains("dataset wrapping is disabled"));
        validate_dataset_wrap_policy(&phases, &input, true).unwrap();
    }

    #[test]
    fn recorded_graph_wrap_policy_allows_bounded_or_one_pass_corpora() {
        let input = wrap_policy_input(2);
        validate_dataset_wrap_policy(&[concurrency_phase(3, Some(2))], &input, false).unwrap();
        validate_dataset_wrap_policy(&[concurrency_phase(3, None)], &input, false).unwrap();
    }

    fn named_concurrency_phase(name: &str) -> PhaseSpec {
        serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": name,
            "exclude_from_results": false,
            "concurrency": 1,
        }))
        .unwrap()
    }

    /// One linear-chain recorded plan `n_0 -> n_1 -> n_2` at arrivals 0/1e6/2e6
    /// us, so the `n` chain straddles a mid-chain `t*` and the split is
    /// observable in the surviving node ids.
    fn tstar_chain_plan() -> Vec<GraphTracePlan> {
        use aiperf::graph::model::{ChannelRequirement, LlmNode, StaticEdge};
        use serde_json::json;
        use std::collections::BTreeMap;

        let node = |arrival: u64, inputs: &[&str]| {
            let mut metadata = BTreeMap::new();
            metadata.insert("arrival_offset_us".to_owned(), json!(arrival));
            LlmNode {
                output: "out".to_owned(),
                streaming: true,
                inputs: inputs
                    .iter()
                    .map(|c| ChannelRequirement {
                        channel: (*c).to_owned(),
                        count: Default::default(),
                    })
                    .collect(),
                min_start_delay_us: None,
                max_tokens: None,
                items: Vec::new(),
                metadata,
            }
        };
        let edge = |source: &str, target: &str| StaticEdge {
            source: source.to_owned(),
            target: target.to_owned(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        };
        let mut nodes = BTreeMap::new();
        nodes.insert("n_0".to_owned(), node(0, &[]));
        nodes.insert("n_1".to_owned(), node(1_000_000, &["n_0_out"]));
        nodes.insert("n_2".to_owned(), node(2_000_000, &["n_1_out"]));
        let graph = GraphRecord {
            nodes,
            edges: vec![
                edge("START", "n_0"),
                edge("n_0", "n_1"),
                edge("n_1", "n_2"),
                edge("n_2", "END"),
            ],
            ..Default::default()
        };
        vec![GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: "t".to_owned(),
                graph_ref: None,
                initial_state: Default::default(),
            },
            arrival_offset_ns: None,
        }]
    }

    fn node_ids(plans: &[GraphTracePlan]) -> Vec<String> {
        plans[0].graph.nodes.keys().cloned().collect()
    }

    #[test]
    fn tstar_split_profiling_keeps_only_post_tstar_frontier() {
        // Collapsed window min==max==0.5 draws no RNG: t* = 0.5 * dur(2e6) = 1e6.
        // Survivors are the nodes arriving at/after t*: n_1, n_2 (n_0 dropped).
        let window = TStarWindow {
            start_min_ratio: 0.5,
            start_max_ratio: 0.5,
            random_seed: 0,
        };
        let split = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
        );
        assert_eq!(node_ids(&split), vec!["n_1".to_owned(), "n_2".to_owned()]);
    }

    #[test]
    fn tstar_split_warmup_primes_boundary_turns() {
        // Same t* = 1e6. The `n` chain's last pre-t* turn is n_0; it primes the
        // chain prefix while n_1/n_2 (post-t*) are profiled, not warmed.
        let window = TStarWindow {
            start_min_ratio: 0.5,
            start_max_ratio: 0.5,
            random_seed: 0,
        };
        let split = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("warmup"),
            window,
        );
        assert_eq!(node_ids(&split), vec!["n_0".to_owned()]);
        // Boundary priming node is flattened: fan-in cleared, re-rooted at START.
        assert!(split[0].graph.nodes["n_0"].inputs.is_empty());
        assert_eq!(split[0].graph.edges.len(), 1);
        assert_eq!(split[0].graph.edges[0].source, "START");
        assert_eq!(split[0].graph.edges[0].target, "n_0");
    }

    #[test]
    fn tstar_default_window_is_unchanged_full_replay() {
        // Regression guard (fixA): the default [0, 0] window is a TRUE no-op, so
        // BOTH profiling and warmup return the full graph unchanged. Emptying the
        // warmup graph here (via rewrite_for_warmup at t* = 0) regressed plain
        // dag_jsonl runs with "contains no dispatchable nodes".
        let window = TStarWindow::default();
        let original = tstar_chain_plan();
        let profiling = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("profiling"),
            window,
        );
        assert_eq!(
            node_ids(&profiling),
            vec!["n_0".to_owned(), "n_1".to_owned(), "n_2".to_owned()]
        );
        // Edges carried through verbatim (default window returns plans as-is).
        assert_eq!(
            profiling[0].graph.edges.len(),
            original[0].graph.edges.len()
        );

        // The warmup phase must ALSO be unchanged at the default window — a plain
        // (no-scenario) warmup runs the full graph, not an empty priming rewrite.
        let warmup = apply_tstar_split(
            &tstar_chain_plan(),
            &named_concurrency_phase("warmup"),
            window,
        );
        assert_eq!(
            node_ids(&warmup),
            vec!["n_0".to_owned(), "n_1".to_owned(), "n_2".to_owned()]
        );
        assert_eq!(warmup[0].graph.edges.len(), original[0].graph.edges.len());
    }

    #[derive(Default)]
    struct RecordingGraphPhaseProgressSink {
        sent: RefCell<Vec<PhaseSend>>,
        returned: RefCell<Vec<PhaseReturn>>,
        first_tokens: Cell<u64>,
        all_sent: Cell<bool>,
    }

    impl GraphPhaseProgressSink for RecordingGraphPhaseProgressSink {
        fn record_sent_batch(&self, sent: &[PhaseSend]) -> Result<(), String> {
            self.sent.borrow_mut().extend_from_slice(sent);
            Ok(())
        }

        fn record_first_token(&self) {
            self.first_tokens
                .set(self.first_tokens.get().saturating_add(1));
        }

        fn record_returned(&self, returned: PhaseReturn) {
            self.returned.borrow_mut().push(returned);
        }

        fn mark_all_sent(&self) {
            self.all_sent.set(true);
        }
    }

    fn progress(
        sink: Rc<RecordingGraphPhaseProgressSink>,
        failures: Rc<GraphPhaseFailures>,
    ) -> (GraphPhaseProgress, Rc<RefCell<GraphWorkloadReport>>) {
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        (
            GraphPhaseProgress::new(
                sink,
                failures,
                outcome.clone(),
                false,
                Rc::new(RefCell::new(Vec::new())),
            ),
            outcome,
        )
    }

    /// Build a WARMUP-phase progress plus the shared warmup-failure ledger it
    /// records into, so a test can assert which trace ids the pre-profiling
    /// abort gate would report.
    fn warmup_progress(
        sink: Rc<RecordingGraphPhaseProgressSink>,
        failures: Rc<GraphPhaseFailures>,
    ) -> (GraphPhaseProgress, Rc<RefCell<Vec<String>>>) {
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        let ledger = Rc::new(RefCell::new(Vec::new()));
        (
            GraphPhaseProgress::new(sink, failures, outcome, true, ledger.clone()),
            ledger,
        )
    }

    fn graph_phase_record(trace_id: &str, errored: bool, canceled: bool) -> CapturedRecord {
        let mut ingest = aiperf::metrics_core::RecordIngest::minimal(
            0,
            1,
            aiperf::metrics_core::Phase::Profiling,
        );
        ingest.errored = errored;
        ingest.canceled = canceled;
        CapturedRecord {
            uuid: Uuid::nil(),
            x_correlation_id: trace_id.into(),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest,
        }
    }

    #[test]
    fn graph_phase_progress_preserves_completed_nodes_before_trace_cancellation() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-cancelled".into(),
            node_count: 3,
            arrival_ns: 0,
        });
        progress.record(&graph_phase_record("trace-cancelled", false, false));
        progress.complete(
            "trace-cancelled",
            3,
            true,
            &Err(TraceError::Cancelled("phase grace expired".into())),
        );

        assert_eq!(
            *sink.sent.borrow(),
            vec![
                PhaseSend::single_turn_session(),
                PhaseSend::dag_child(),
                PhaseSend::dag_child(),
            ]
        );
        assert_eq!(
            *sink.returned.borrow(),
            vec![
                PhaseReturn {
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    cancelled: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    completes_session: true,
                    cancelled: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
            ]
        );
        assert_eq!(outcome.borrow().admitted, 1);
        assert_eq!(outcome.borrow().cancelled, 1);
        assert!(failures.first().is_none());
    }

    #[test]
    fn graph_phase_progress_never_masks_backend_failure_as_cancellation() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-failed".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        progress.record(&graph_phase_record("trace-failed", false, false));
        progress.complete(
            "trace-failed",
            2,
            true,
            &Err(TraceError::Other("backend failed".into())),
        );

        assert_eq!(
            *sink.returned.borrow(),
            vec![
                PhaseReturn {
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    completes_session: true,
                    errored: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
            ]
        );
        assert_eq!(outcome.borrow().failed, 1);
        assert_eq!(
            failures.first().as_deref(),
            Some("graph trace \"trace-failed\" failed: backend failed")
        );
    }

    #[test]
    fn graph_first_token_reaches_progress_before_adaptive_terminal_record() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-adaptive".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        let mut record = graph_phase_record("trace-adaptive", false, false);
        record.ingest.first_token_ns = Some(5);
        record.ingest.token_arrival_ns.push(5);
        record.ingest.usage.completion_tokens = Some(1);
        let sampler: SharedWindowSampler =
            Rc::new(RefCell::new(Box::new(TumblingWindowSampler::new(0))));
        let captured = Rc::new(RefCell::new(Vec::new()));

        ingest_graph_execution_event(
            &captured,
            Some(&sampler),
            &progress,
            RunnerGraphExecutionEvent::FirstToken {
                trace_id: "trace-adaptive".into(),
                uuid: Uuid::nil(),
            },
        );
        ingest_graph_execution_event(
            &captured,
            Some(&sampler),
            &progress,
            RunnerGraphExecutionEvent::Record(Box::new(record)),
        );

        let window = sampler.borrow_mut().take(10);
        assert_eq!(window.completed(), 1);
        assert_eq!(captured.borrow().len(), 1);
        assert_eq!(sink.first_tokens.get(), 1);
        assert_eq!(
            *sink.returned.borrow(),
            vec![PhaseReturn {
                completes_session: true,
                releases_prefill: false,
                ..PhaseReturn::default()
            }]
        );
    }

    #[test]
    fn graph_phase_rejects_duplicate_first_token_for_one_request() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-duplicate-token".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        let uuid = Uuid::from_u128(7);
        progress.first_token("trace-duplicate-token", uuid);
        progress.first_token("trace-duplicate-token", uuid);

        assert_eq!(sink.first_tokens.get(), 1);
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-duplicate-token\" emitted duplicate first-token event for request 00000000-0000-0000-0000-000000000007"
            )
        );
    }

    #[test]
    fn graph_phase_rejects_duplicate_terminal_record_without_double_return() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-duplicate-terminal".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        let record = graph_phase_record("trace-duplicate-terminal", false, false);
        progress.record(&record);
        progress.record(&record);

        assert_eq!(sink.returned.borrow().len(), 1);
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-duplicate-terminal\" emitted duplicate terminal record for request 00000000-0000-0000-0000-000000000000"
            )
        );
    }

    #[test]
    fn graph_phase_uses_live_first_token_membership_when_record_metadata_disagrees() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-token-mismatch".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        let mut record = graph_phase_record("trace-token-mismatch", false, false);
        record.ingest.first_token_ns = Some(5);
        progress.record(&record);

        assert_eq!(
            *sink.returned.borrow(),
            vec![PhaseReturn {
                completes_session: true,
                releases_prefill: true,
                ..PhaseReturn::default()
            }]
        );
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-token-mismatch\" request 00000000-0000-0000-0000-000000000000 first-token event/record mismatch: event=false record=true"
            )
        );
    }

    #[test]
    fn graph_phase_rejects_first_token_after_terminal_record() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-late-token".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        let record = graph_phase_record("trace-late-token", false, false);
        progress.record(&record);
        progress.first_token("trace-late-token", record.uuid);

        assert_eq!(sink.first_tokens.get(), 0);
        assert_eq!(
            failures.first().as_deref(),
            Some(
                "graph trace \"trace-late-token\" emitted first-token event after terminal record for request 00000000-0000-0000-0000-000000000000"
            )
        );
    }

    #[test]
    fn graph_phase_does_not_double_release_orphan_first_token_on_cancellation() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, _outcome) = progress(sink.clone(), failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "trace-orphan-token".into(),
            node_count: 2,
            arrival_ns: 0,
        });
        progress.first_token("trace-orphan-token", Uuid::from_u128(11));
        progress.complete(
            "trace-orphan-token",
            2,
            true,
            &Err(TraceError::Cancelled("phase cancelled".into())),
        );

        assert_eq!(sink.first_tokens.get(), 1);
        assert_eq!(
            *sink.returned.borrow(),
            vec![
                PhaseReturn {
                    cancelled: true,
                    releases_prefill: false,
                    ..PhaseReturn::default()
                },
                PhaseReturn {
                    completes_session: true,
                    cancelled: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                },
            ]
        );
        assert!(failures.first().is_none());
    }

    #[test]
    fn warmup_terminal_node_failure_is_ledgered_for_abort_gate() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-fail".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        // Resilient warmup priming: a failed reply surfaces as an errored,
        // non-cancelled per-node record (Python `_on_graph_return` -> error).
        progress.record(&graph_phase_record("warmup-fail", true, false));
        assert_eq!(*ledger.borrow(), vec!["warmup-fail".to_owned()]);
    }

    #[test]
    fn warmup_trace_abort_is_ledgered_for_abort_gate() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-abort".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        // Fail-fast node policy: the trace itself aborts with a non-cancelled
        // error and never emits a terminal record.
        progress.complete(
            "warmup-abort",
            1,
            true,
            &Err(TraceError::Other("boom".into())),
        );
        assert_eq!(*ledger.borrow(), vec!["warmup-abort".to_owned()]);
        assert!(failures.first().is_some());
    }

    #[test]
    fn warmup_cancelled_return_is_not_a_terminal_failure() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-cancelled".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        // Cancelled returns are excluded even when they carry error text
        // (agentx parity: a drain-cancel is self-inflicted teardown).
        progress.record(&graph_phase_record("warmup-cancelled", true, true));
        assert!(ledger.borrow().is_empty());
        // And a trace cancelled at completion is likewise not ledgered.
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let (progress, ledger) = warmup_progress(sink, failures);
        progress.admit(&TraceAdmissionInfo {
            trace_id: "warmup-drain".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        progress.complete(
            "warmup-drain",
            1,
            true,
            &Err(TraceError::Cancelled("phase grace expired".into())),
        );
        assert!(ledger.borrow().is_empty());
    }

    #[test]
    fn profiling_phase_never_ledgers_warmup_failures() {
        let sink = Rc::new(RecordingGraphPhaseProgressSink::default());
        let failures = Rc::new(GraphPhaseFailures::default());
        let outcome = Rc::new(RefCell::new(GraphWorkloadReport::default()));
        let ledger = Rc::new(RefCell::new(Vec::new()));
        // is_warmup = false: the same terminal failure is a normal profiling
        // failure, never a pre-profiling abort trigger.
        let progress = GraphPhaseProgress::new(sink, failures, outcome, false, ledger.clone());
        progress.admit(&TraceAdmissionInfo {
            trace_id: "profiling-fail".into(),
            node_count: 1,
            arrival_ns: 0,
        });
        progress.record(&graph_phase_record("profiling-fail", true, false));
        assert!(ledger.borrow().is_empty());
    }

    #[test]
    fn trajectory_warmup_failed_error_downcasts_to_execution_stage_failure() {
        let error = trajectory_warmup_failed_error(&["trace-a".to_owned(), "trace-b".to_owned()]);
        let failure = error
            .downcast_ref::<PreparedRunFailure>()
            .expect("structured warmup failure");
        assert_eq!(failure.stage, RunnerFailureStageV2::Execution);
        assert_eq!(failure.code, "trajectory_warmup_failed");
        assert!(failure.message.contains("trace-a"));
        assert!(failure.message.contains("trace-b"));
        assert!(failure.message.contains("2 trace(s)"));
    }
}
