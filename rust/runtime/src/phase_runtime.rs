// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production composition of scheduled workloads through the phase driver.
//!
//! [`run_scheduled_phases`] connects the transport-neutral
//! [`crate::scheduled::TurnDispatcher`] and
//! [`crate::scheduled::Workload`] seams to
//! `crate::timing::{PhaseRunner, PhaseOrchestrator}`. The adapter records sends,
//! first tokens, and terminal returns at the dispatcher boundary, so workload
//! implementations remain schedule generators and do not learn phase lifecycle
//! policy. One factory is shared across every phase; each phase still receives
//! a fresh runtime, observer graph, counter set, and report.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::observer::CollectorObserver;
use crate::dispatch::sink::RequestObserver;
use crate::metrics_core::MetricsConfig;
use crate::timing::{
    ClockPhaseOrchestrator, ClockPhaseRunnerFactory, LocalPhaseFuture, PhaseBranchStats,
    PhaseConfig, PhaseContext, PhaseExecution, PhaseExecutionError, PhaseExecutionFactory,
    PhaseKind, PhaseObserver, PhaseReturn, PhaseSend, PhaseStats, RampDriver, RampHandle,
    ReleasedStuckSlots, SlotPool, drive_phases,
};
use anyhow::{Result, anyhow};
use rustc_hash::FxHashMap;
use serde::Serialize;
use uuid::Uuid;

use crate::metrics::{NativeMetricsObserver, ObserverTee};
use crate::multiturn::TurnToSend;
use crate::scheduled::{
    IssuanceGate, ScheduledAncillaryPolicies, ScheduledRunReport, ScheduledRuntime,
    TurnDispatchOutcome, TurnDispatcher, TurnLifecycleObserver, TurnRecordProcessor,
    UserControlSnapshot, Workload,
};
use crate::scheduler::LocalTaskScheduler;

/// Optional phase-owned actuator/ramp lifecycle.
pub trait ScheduledPhaseController {
    /// Start actuators before phase issuance begins.
    fn start(&self) -> Result<()>;

    /// Stop and join actuators at sending handoff.
    fn stop(&self) -> LocalPhaseFuture<Result<()>>;

    /// Resolve when controller policy independently requires issuance to stop.
    ///
    /// Ramps never resolve this future. Adaptive controllers use it to wake a
    /// workload that may otherwise be sleeping until a distant arrival time.
    fn wait_until_stop(&self) -> LocalPhaseFuture<()> {
        Box::pin(std::future::pending())
    }
}

/// Runtime additions constructed at the actual start of one phase.
///
/// This is the extension seam for policies that consume request observations,
/// gate issuance, and own an asynchronous controller. Construction occurs only
/// after the ordinary collector and native metrics observer exist, so an
/// extension decorates rather than replaces benchmark measurement.
pub struct ScheduledRuntimeExtensionParts {
    /// Observer that includes the supplied ordinary measurement delegate.
    pub observer: Rc<dyn RequestObserver>,
    /// Optional policy gate consulted before every root or continuation issue.
    pub issuance_gate: Option<Rc<dyn IssuanceGate>>,
    /// Effective phase controller, normally wrapping the supplied controller.
    pub controller: Rc<dyn ScheduledPhaseController>,
    /// Per-turn record processors the extension needs registered on the runtime.
    ///
    /// An extension whose policy consumes finished worker records (rather than
    /// the callback observer) contributes them here, since the sampler and its
    /// feed are both created at phase start. They run alongside the plan's own
    /// processors, after normal measurement and credit return.
    pub record_processors: Vec<Rc<dyn TurnRecordProcessor>>,
}

/// Object-safe factory for one phase-local runtime policy extension.
pub trait ScheduledRuntimeExtension {
    /// Decorate ordinary measurement and controller policy for this phase.
    ///
    /// `observer_origin_ns` is the timestamp origin used by transport callback
    /// offsets. `phase_start_ns` is the actual phase boundary and should anchor
    /// phase-local windows. They differ when warmup and profiling share one
    /// transport timeline.
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        clock: Rc<dyn crate::clock::Clock>,
        observer_origin_ns: i64,
        phase_start_ns: i64,
        delegate: Rc<dyn RequestObserver>,
        controller: Rc<dyn ScheduledPhaseController>,
    ) -> Result<ScheduledRuntimeExtensionParts>;
}

/// Controller used when a phase has no actuators.
#[derive(Default)]
pub struct NoopScheduledPhaseController;

impl ScheduledPhaseController for NoopScheduledPhaseController {
    fn start(&self) -> Result<()> {
        Ok(())
    }

    fn stop(&self) -> LocalPhaseFuture<Result<()>> {
        Box::pin(async { Ok(()) })
    }
}

/// Phase-owned controller for prepared Clock-native ramp drivers.
///
/// Drivers apply their initial value synchronously when the runner invokes
/// [`ScheduledPhaseController::start`], before workload execution can issue a
/// request. The controller then stops and joins every task at the phase's
/// sending handoff.
pub struct RampScheduledPhaseController {
    drivers: RefCell<Option<Vec<RampDriver>>>,
    handles: RefCell<Option<Vec<RampHandle>>>,
}

impl RampScheduledPhaseController {
    /// Take ownership of drivers prepared for one phase.
    pub fn new(drivers: Vec<RampDriver>) -> Self {
        Self {
            drivers: RefCell::new(Some(drivers)),
            handles: RefCell::new(None),
        }
    }
}

impl ScheduledPhaseController for RampScheduledPhaseController {
    fn start(&self) -> Result<()> {
        let drivers = self
            .drivers
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("phase ramp controller was already started or stopped"))?;
        *self.handles.borrow_mut() =
            Some(drivers.into_iter().map(RampDriver::spawn_local).collect());
        Ok(())
    }

    fn stop(&self) -> LocalPhaseFuture<Result<()>> {
        // Failure before start still owns prepared drivers; dropping them is
        // the complete cleanup because no task or actuator mutation occurred.
        self.drivers.borrow_mut().take();
        let handles = self.handles.borrow_mut().take().unwrap_or_default();
        Box::pin(async move {
            for handle in handles {
                if handle.is_running() {
                    handle.stop();
                }
                if let Err(error) = handle.wait().await
                    && !error.is_cancelled()
                {
                    return Err(anyhow!("phase ramp task failed: {error}"));
                }
            }
            Ok(())
        })
    }
}

/// Shared admission resources configured and cleaned up at phase boundaries.
///
/// Implementations keep long-lived slot pools outside individual workloads so
/// a lower cap can debt-drain returns from a seamless predecessor. A workload
/// that stores guards outside scheduler tasks also releases them here after the
/// cancellation-drain backstop fires.
pub trait ScheduledPhaseResources {
    /// Apply this phase's targets before setup or issuance begins.
    fn configure(&self, _config: &PhaseConfig) -> Result<()> {
        Ok(())
    }

    /// Release phase-owned admission state after cancellation drain fails.
    fn release_stuck(&self) -> ReleasedStuckSlots;
}

/// Asynchronous control-plane work synchronized to one phase's hard barriers.
///
/// Sidecars are outside [`RequestObserver`] so low-rate telemetry and profilers
/// can force samples at phase boundaries without adding per-token work.
/// The phase driver awaits [`start`](Self::start) before it can issue the first
/// turn and awaits [`finish`](Self::finish) after dispatch has fully drained.
pub trait ScheduledPhaseSidecar {
    /// Prepare and force any baseline sample before issuance begins.
    fn start(&self) -> LocalPhaseFuture<Result<()>>;

    /// Record the common phase-start instant after every sidecar is prepared.
    fn on_phase_start(&self, _timestamp_ns: i64) {}

    /// Record the common phase-end instant before any final sample is taken.
    fn on_phase_end(&self, _timestamp_ns: i64) {}

    /// Force any final sample after every dispatch has drained.
    fn finish(&self) -> LocalPhaseFuture<Result<()>>;
}

/// Start every phase sidecar, then mark the phase-start instant on each.
///
/// Shared by the scheduled and graph `PhaseExecution::setup` paths; `label`
/// selects the workload word in the error context ("scheduled"/"graph").
pub(crate) async fn start_phase_sidecars(
    sidecars: &[Rc<dyn ScheduledPhaseSidecar>],
    clock: &dyn crate::clock::Clock,
    label: &str,
) -> Result<(), PhaseExecutionError> {
    for sidecar in sidecars {
        sidecar.start().await.map_err(|error| {
            PhaseExecutionError::new(format!("starting {label} phase sidecar: {error:#}"))
        })?;
    }
    let phase_start_ns = clock.now_ns();
    for sidecar in sidecars {
        sidecar.on_phase_start(phase_start_ns);
    }
    Ok(())
}

/// Mark the phase-end instant on every sidecar, then finish each.
///
/// Shared by the scheduled and graph `PhaseExecution::execute` finish paths.
pub(crate) async fn finish_phase_sidecars(
    sidecars: &[Rc<dyn ScheduledPhaseSidecar>],
    clock: &dyn crate::clock::Clock,
    label: &str,
) -> Result<(), PhaseExecutionError> {
    let phase_end_ns = clock.now_ns();
    for sidecar in sidecars {
        sidecar.on_phase_end(phase_end_ns);
    }
    for sidecar in sidecars {
        sidecar.finish().await.map_err(|error| {
            PhaseExecutionError::new(format!("finishing {label} phase sidecar: {error:#}"))
        })?;
    }
    Ok(())
}

/// Resources used by workloads with no shared admission state.
#[derive(Default)]
pub struct NoopScheduledPhaseResources;

impl ScheduledPhaseResources for NoopScheduledPhaseResources {
    fn release_stuck(&self) -> ReleasedStuckSlots {
        ReleasedStuckSlots::default()
    }
}

/// Shared session/prefill pools whose debt survives phase boundaries.
pub struct SlotPoolPhaseResources {
    session: Option<Rc<SlotPool>>,
    prefill: Option<Rc<SlotPool>>,
}

impl SlotPoolPhaseResources {
    /// Bind optional long-lived pools used by every phase workload.
    pub fn new(session: Option<Rc<SlotPool>>, prefill: Option<Rc<SlotPool>>) -> Self {
        Self { session, prefill }
    }
}

impl ScheduledPhaseResources for SlotPoolPhaseResources {
    fn configure(&self, config: &PhaseConfig) -> Result<()> {
        match (config.concurrency, &self.session) {
            (Some(limit), Some(pool)) => pool.set_limit(limit),
            (Some(_), None) => {
                return Err(anyhow!(
                    "phase {:?} configures session concurrency without a shared session pool",
                    config.id
                ));
            }
            (None, _) => {}
        }
        match (config.prefill_concurrency, &self.prefill) {
            (Some(limit), Some(pool)) => pool.set_limit(limit),
            (Some(_), None) => {
                return Err(anyhow!(
                    "phase {:?} configures prefill concurrency without a shared prefill pool",
                    config.id
                ));
            }
            (None, _) => {}
        }
        Ok(())
    }

    fn release_stuck(&self) -> ReleasedStuckSlots {
        // Slot guards remain the authoritative ownership record. Cancelling
        // their local tasks drops them; fabricating releases here would make a
        // later guard drop over-credit the pool.
        ReleasedStuckSlots::default()
    }
}

/// One prepared phase lowered into the shared scheduled runtime.
pub struct ScheduledPhasePlan {
    /// Lifecycle, stop, grace, concurrency, and seamless policy.
    pub config: PhaseConfig,
    /// Schedule generator for this phase.
    pub workload: Rc<dyn Workload>,
    /// Cancellation and endpoint-selection policies.
    pub ancillary: ScheduledAncillaryPolicies,
    /// Terminal record processors attached to this phase.
    pub record_processors: Vec<Rc<dyn TurnRecordProcessor>>,
    /// Whether the scheduled runtime enforces this phase's stop bounds.
    pub enforce_stop: bool,
    /// Optional pre-captured observer/transport timeline origin.
    pub start_ns: Option<i64>,
    /// Phase-owned actuator/ramp lifecycle.
    pub controller: Rc<dyn ScheduledPhaseController>,
    /// Long-lived admission state and force cleanup.
    pub resources: Rc<dyn ScheduledPhaseResources>,
    /// Low-rate control-plane services synchronized to phase barriers.
    pub sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    /// Optional phase-local observer/gate/controller decorator.
    pub runtime_extension: Option<Rc<dyn ScheduledRuntimeExtension>>,
    /// Native metric policy for this phase's owned accumulator.
    pub metrics_config: MetricsConfig,
    /// Whether the compatibility collector retains export-only request detail.
    pub capture_performance_records: bool,
    /// Whether the phase-local compatibility collector observes the request stream.
    ///
    /// Backends that already own the canonical compatibility report may disable
    /// this duplicate observer while retaining AIPerf's native metrics observer.
    /// The default stays enabled for online transports and independent parity
    /// tests.
    pub collect_performance_summary: bool,
    /// Whether native records retain exporter/join-only row identities.
    pub retain_native_metric_record_dimensions: bool,
    /// Whether full per-turn timing rows are retained in the report.
    pub capture_timing_records: bool,
    /// Whether post-drain compatibility and native reductions may share the
    /// bounded reduction pool.
    pub parallel_report_reduction: bool,
    /// Run-wide observers that receive the exact phase-local measurement
    /// stream in addition to the phase's own collector and native metrics.
    ///
    /// This is the aggregation seam for a backend that owns one engine across
    /// multiple phases. It avoids reconstructing a whole-run report by merging
    /// already-finalized phase summaries.
    pub additional_observers: Vec<Rc<dyn RequestObserver>>,
}

impl ScheduledPhasePlan {
    /// Build a phase plan with no ramps, processors, or external guard cleanup.
    pub fn new(
        config: PhaseConfig,
        workload: Rc<dyn Workload>,
        ancillary: ScheduledAncillaryPolicies,
    ) -> Self {
        Self {
            config,
            workload,
            ancillary,
            record_processors: Vec::new(),
            enforce_stop: true,
            start_ns: None,
            controller: Rc::new(NoopScheduledPhaseController),
            resources: Rc::new(NoopScheduledPhaseResources),
            sidecars: Vec::new(),
            runtime_extension: None,
            metrics_config: MetricsConfig::default(),
            capture_performance_records: true,
            collect_performance_summary: true,
            retain_native_metric_record_dimensions: true,
            capture_timing_records: true,
            parallel_report_reduction: false,
            additional_observers: Vec::new(),
        }
    }

    /// Preserve natural-exhaustion workloads that own their authored bounds.
    pub fn with_enforce_stop(mut self, enforce_stop: bool) -> Self {
        self.enforce_stop = enforce_stop;
        self
    }

    /// Reuse an origin captured while constructing a phase-specific dispatcher.
    pub fn with_start_ns(mut self, start_ns: i64) -> Self {
        self.start_ns = Some(start_ns);
        self
    }

    /// Attach terminal consumers that run after credit return.
    pub fn with_record_processors(
        mut self,
        record_processors: Vec<Rc<dyn TurnRecordProcessor>>,
    ) -> Self {
        self.record_processors = record_processors;
        self
    }

    /// Attach the phase-owned actuator lifecycle.
    pub fn with_controller(mut self, controller: Rc<dyn ScheduledPhaseController>) -> Self {
        self.controller = controller;
        self
    }

    /// Attach shared admission resources used across phase workloads.
    pub fn with_resources(mut self, resources: Rc<dyn ScheduledPhaseResources>) -> Self {
        self.resources = resources;
        self
    }

    /// Attach control-plane services that share this phase's hard barriers.
    pub fn with_sidecars(mut self, sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>) -> Self {
        self.sidecars = sidecars;
        self
    }

    /// Attach a phase-local observer, issuance-gate, and controller extension.
    pub fn with_runtime_extension(mut self, extension: Rc<dyn ScheduledRuntimeExtension>) -> Self {
        self.runtime_extension = Some(extension);
        self
    }

    /// Configure the phase-local native metric accumulator.
    pub fn with_metrics_config(mut self, metrics_config: MetricsConfig) -> Self {
        self.metrics_config = metrics_config;
        self
    }

    /// Configure compatibility-only per-request record retention.
    pub fn with_performance_record_capture(mut self, capture: bool) -> Self {
        self.capture_performance_records = capture;
        self
    }

    /// Configure whether this phase independently collects compatibility metrics.
    pub fn with_performance_summary_collection(mut self, collect: bool) -> Self {
        self.collect_performance_summary = collect;
        self
    }

    /// Configure retention of native exporter/join-only row identities.
    pub fn with_native_metric_record_dimensions(mut self, retain: bool) -> Self {
        self.retain_native_metric_record_dimensions = retain;
        self
    }

    /// Configure retention of full per-turn timing records.
    pub fn with_timing_record_capture(mut self, capture: bool) -> Self {
        self.capture_timing_records = capture;
        self
    }

    /// Attach observers that span phase boundaries on the shared dispatcher
    /// timeline.
    pub fn with_additional_observers(mut self, observers: Vec<Rc<dyn RequestObserver>>) -> Self {
        self.additional_observers = observers;
        self
    }
}

/// Scheduled performance report tagged with its phase identity.
#[derive(Debug, Serialize)]
pub struct ScheduledPhaseReport {
    /// Stable phase identifier.
    pub phase_id: String,
    /// Warmup or profiling role.
    pub kind: PhaseKind,
    /// Normal scheduled workload report.
    pub report: ScheduledRunReport,
}

/// Complete ordered result of a phased scheduled run.
#[derive(Debug, Serialize)]
pub struct PhasedScheduledRunReport {
    /// Final lifecycle/progress snapshots in configured order.
    pub phases: Vec<PhaseStats>,
    /// Performance reports in configured order.
    pub reports: Vec<ScheduledPhaseReport>,
}

enum PendingScheduledPhaseReport {
    Finalized(Box<ScheduledRunReport>),
    Deferred {
        runtime: Rc<ScheduledRuntime>,
        end_ns: i64,
        strategy: &'static str,
        user_control: Option<UserControlSnapshot>,
    },
}

impl PendingScheduledPhaseReport {
    fn finish(self) -> ScheduledRunReport {
        match self {
            Self::Finalized(report) => *report,
            Self::Deferred {
                runtime,
                end_ns,
                strategy,
                user_control,
            } => runtime.finish_at(end_ns, strategy, user_control),
        }
    }
}

/// Drained phase runtimes whose aggregate reduction has not run yet.
///
/// The value is intentionally `!Send`: it preserves the worker-local
/// `Rc`/`RefCell` observer graph while allowing an offline driver to leave its
/// Tokio `LocalSet` before performing metric sweeps and report construction on
/// the same OS thread.
pub struct DeferredPhasedScheduledRunReport {
    phases: Vec<PhaseStats>,
    order: BTreeMap<String, (usize, PhaseKind)>,
    reports: Vec<(String, PendingScheduledPhaseReport)>,
}

impl DeferredPhasedScheduledRunReport {
    /// Reduce captured observer facts into ordered phase reports.
    pub fn finish(mut self) -> PhasedScheduledRunReport {
        self.reports.sort_by_key(|(phase_id, _)| {
            self.order
                .get(phase_id)
                .map(|(index, _)| *index)
                .unwrap_or(usize::MAX)
        });
        let reports = self
            .reports
            .into_iter()
            .map(|(phase_id, report)| ScheduledPhaseReport {
                kind: self
                    .order
                    .get(&phase_id)
                    .map(|(_, kind)| *kind)
                    .unwrap_or(PhaseKind::Profiling),
                phase_id,
                report: report.finish(),
            })
            .collect();
        PhasedScheduledRunReport {
            phases: self.phases,
            reports,
        }
    }
}

/// Phased result plus one compatibility report accumulated directly from the
/// live observer stream across the complete shared clock/dispatcher lifecycle.
#[derive(Debug, Serialize)]
pub struct AggregatedPhasedScheduledRunReport {
    /// Independently finalized phase lifecycle and performance reports.
    pub phased: PhasedScheduledRunReport,
    /// Whole-run compatibility metrics observed before phase finalization.
    pub performance: crate::dispatch::collector::TraceSimulationReport,
}

/// Drained phased execution plus its still-unreduced whole-run collector.
pub struct DeferredAggregatedPhasedScheduledRunReport {
    phased: DeferredPhasedScheduledRunReport,
    aggregate: Box<dyn DeferredAggregateStrategy>,
    wall_ms: f64,
}

impl DeferredAggregatedPhasedScheduledRunReport {
    /// Finalize phase-local and whole-run collectors after the live runtime has
    /// been torn down.
    pub fn finish(self) -> AggregatedPhasedScheduledRunReport {
        let phased = self.phased.finish();
        let performance = self.aggregate.finish(&phased, self.wall_ms);
        AggregatedPhasedScheduledRunReport {
            phased,
            performance,
        }
    }
}

trait DeferredAggregateStrategy {
    fn observer(&self) -> Option<Rc<dyn RequestObserver>>;

    fn finish(
        self: Box<Self>,
        phased: &PhasedScheduledRunReport,
        wall_ms: f64,
    ) -> crate::dispatch::collector::TraceSimulationReport;
}

struct DedicatedAggregateCollector {
    collector: Rc<CollectorObserver>,
}

impl DeferredAggregateStrategy for DedicatedAggregateCollector {
    fn observer(&self) -> Option<Rc<dyn RequestObserver>> {
        Some(self.collector.clone())
    }

    fn finish(
        self: Box<Self>,
        _phased: &PhasedScheduledRunReport,
        wall_ms: f64,
    ) -> crate::dispatch::collector::TraceSimulationReport {
        self.collector.finish(wall_ms)
    }
}

struct SinglePhaseAggregateReuse;

impl DeferredAggregateStrategy for SinglePhaseAggregateReuse {
    fn observer(&self) -> Option<Rc<dyn RequestObserver>> {
        None
    }

    fn finish(
        self: Box<Self>,
        phased: &PhasedScheduledRunReport,
        _wall_ms: f64,
    ) -> crate::dispatch::collector::TraceSimulationReport {
        phased
            .reports
            .first()
            .expect("single-phase aggregate strategy requires one phase report")
            .report
            .performance
            .clone()
    }
}

/// Run scheduled phases to quiescence while deferring all aggregate reduction.
///
/// This is the virtual-time backend boundary: the returned value owns the
/// drained worker-local observer graph and can be finalized after the DES
/// `LocalSet` has exited. No request, engine, or clock decision is deferred.
pub async fn run_scheduled_phases_with_aggregate_deferred(
    mut plans: Vec<ScheduledPhasePlan>,
    clock: Rc<dyn crate::clock::Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    observer: Rc<dyn PhaseObserver>,
) -> Result<DeferredAggregatedPhasedScheduledRunReport> {
    let aggregate: Box<dyn DeferredAggregateStrategy> = if plans.len() == 1 {
        Box::new(SinglePhaseAggregateReuse)
    } else {
        Box::new(DedicatedAggregateCollector {
            collector: Rc::new(CollectorObserver::new(true)),
        })
    };
    if let Some(observer) = aggregate.observer() {
        for plan in &mut plans {
            plan.additional_observers.push(observer.clone());
        }
    }
    let phased = run_scheduled_phases_deferred(plans, clock.clone(), dispatcher, observer).await?;
    let wall_ms = clock.now_ns().saturating_sub(start_ns) as f64 / 1_000_000.0;
    Ok(DeferredAggregatedPhasedScheduledRunReport {
        phased,
        aggregate,
        wall_ms,
    })
}

/// Run prepared scheduled workloads through the shared phase orchestrator.
pub async fn run_scheduled_phases(
    plans: Vec<ScheduledPhasePlan>,
    clock: Rc<dyn crate::clock::Clock>,
    dispatcher: Rc<dyn TurnDispatcher>,
    observer: Rc<dyn PhaseObserver>,
) -> Result<PhasedScheduledRunReport> {
    Ok(
        run_scheduled_phases_inner(plans, clock, dispatcher, observer, false)
            .await?
            .finish(),
    )
}

/// Run prepared phases to quiescence without reducing their retained metrics.
pub async fn run_scheduled_phases_deferred(
    plans: Vec<ScheduledPhasePlan>,
    clock: Rc<dyn crate::clock::Clock>,
    dispatcher: Rc<dyn TurnDispatcher>,
    observer: Rc<dyn PhaseObserver>,
) -> Result<DeferredPhasedScheduledRunReport> {
    run_scheduled_phases_inner(plans, clock, dispatcher, observer, true).await
}

async fn run_scheduled_phases_inner(
    plans: Vec<ScheduledPhasePlan>,
    clock: Rc<dyn crate::clock::Clock>,
    dispatcher: Rc<dyn TurnDispatcher>,
    observer: Rc<dyn PhaseObserver>,
    defer_reports: bool,
) -> Result<DeferredPhasedScheduledRunReport> {
    let configs = plans
        .iter()
        .map(|plan| plan.config.clone())
        .collect::<Vec<_>>();
    let order = configs
        .iter()
        .enumerate()
        .map(|(index, config)| (config.id.clone(), (index, config.kind)))
        .collect::<BTreeMap<_, _>>();
    // Emit the profiling marker only after readiness and signal cancellation
    // are active, so an immediate interrupt can still produce partial results.
    let observer: Rc<dyn PhaseObserver> = Rc::new(ProfilingBannerObserver::new(observer));
    let reports = Rc::new(RefCell::new(Vec::new()));
    let execution_factory = Rc::new(ScheduledPhaseExecutionFactory {
        clock: clock.clone(),
        dispatcher,
        plans: RefCell::new(
            plans
                .into_iter()
                .map(|plan| (plan.config.id.clone(), plan))
                .collect(),
        ),
        reports: reports.clone(),
        runtimes: RefCell::new(Vec::new()),
        defer_reports,
    });
    let phase_execution_factory: Rc<dyn PhaseExecutionFactory> = execution_factory.clone();
    // The virtual (offline) clock builds a bare `current_thread` runtime with no
    // I/O/signal driver, so `tokio::signal` would panic there; capture the axis
    // before the clock is moved and arm the listener only under the wall clock.
    let clock_is_virtual = clock.is_virtual();
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock,
        observer.clone(),
        phase_execution_factory,
    ));
    let orchestrator = ClockPhaseOrchestrator::new(configs, runner_factory, observer)
        .map_err(|error| anyhow!(error))?;
    // `drive_phases` arms run-level cancellation on the first SIGINT/SIGTERM (only
    // under a wall clock) and drives the orchestrator to completion. The active
    // phase then drains through the existing cancellation latch and yields
    // `PhaseStats { was_cancelled: true }` while the runner still writes its
    // partial native report.
    let phase_result = drive_phases(orchestrator, clock_is_virtual)
        .await
        .map_err(|error| anyhow!(error));
    let processor_result = execution_factory.wait_record_processors().await;
    let phases = match (phase_result, processor_result) {
        (Ok(phases), Ok(())) => phases,
        (Err(phase_error), Ok(())) => return Err(phase_error),
        (Ok(_), Err(processor_error)) => return Err(processor_error),
        (Err(phase_error), Err(processor_error)) => {
            return Err(phase_error.context(format!(
                "terminal record processing also failed: {processor_error:#}"
            )));
        }
    };

    let reports = reports.borrow_mut().drain(..).collect::<Vec<_>>();
    Ok(DeferredPhasedScheduledRunReport {
        phases,
        order,
        reports,
    })
}

/// Emitted to stderr when the first profiling phase starts; stdout is reserved
/// for terminal JSON.
const PROFILING_BANNER: &str = "AIPerf System is PROFILING";

/// Observer decorator that emits [`PROFILING_BANNER`] once, when the first
/// profiling phase starts, then delegates every event to the inner observer.
struct ProfilingBannerObserver {
    inner: Rc<dyn PhaseObserver>,
    announced: Cell<bool>,
}

impl ProfilingBannerObserver {
    fn new(inner: Rc<dyn PhaseObserver>) -> Self {
        Self {
            inner,
            announced: Cell::new(false),
        }
    }
}

impl PhaseObserver for ProfilingBannerObserver {
    fn on_phase_start(&self, config: &PhaseConfig, stats: PhaseStats) {
        if config.kind == PhaseKind::Profiling && !self.announced.replace(true) {
            eprintln!("{PROFILING_BANNER}");
        }
        let requests = optional_count(stats.total_expected_requests);
        let duration_s = stats
            .expected_duration_ns
            .map(|ns| ns as f64 / 1e9)
            .map(|s| format!("{s:.0}s"))
            .unwrap_or_else(|| "-".to_owned());
        let sessions = optional_count(stats.expected_num_sessions);
        tracing::info!(
            "Phase {} started | target: {requests} requests, {duration_s} duration, {sessions} sessions",
            stats.phase_id,
        );
        self.inner.on_phase_start(config, stats);
    }

    fn on_progress(&self, stats: PhaseStats) {
        self.inner.on_progress(stats);
    }

    fn on_sending_complete(&self, stats: PhaseStats) {
        tracing::info!(
            "Phase {} sending complete | sent={}, completed={}, in_flight={}",
            stats.phase_id,
            stats.requests_sent,
            stats.requests_completed,
            stats.in_flight_requests,
        );
        self.inner.on_sending_complete(stats);
    }

    fn on_phase_complete(&self, stats: PhaseStats, branch_stats: Option<PhaseBranchStats>) {
        let elapsed_s = match (stats.start_ns, stats.requests_end_ns) {
            (Some(start), Some(end)) => (end - start) as f64 / 1e9,
            _ => 0.0,
        };
        tracing::info!(
            "Phase {} complete | completed={}, cancelled={}, errors={} | elapsed={elapsed_s:.2}s",
            stats.phase_id,
            stats
                .final_requests_completed
                .unwrap_or(stats.requests_completed),
            stats
                .final_requests_cancelled
                .unwrap_or(stats.requests_cancelled),
            stats.final_request_errors.unwrap_or(stats.request_errors),
        );
        self.inner.on_phase_complete(stats, branch_stats);
    }

    fn on_phases_complete(&self, stats: Vec<PhaseStats>) {
        tracing::info!("All credits completed");
        self.inner.on_phases_complete(stats);
    }
}

/// Render an optional expected count as a number or `unbounded`.
fn optional_count(value: Option<u64>) -> String {
    value.map_or_else(|| "unbounded".to_owned(), |v| v.to_string())
}

struct ScheduledPhaseExecutionFactory {
    clock: Rc<dyn crate::clock::Clock>,
    dispatcher: Rc<dyn TurnDispatcher>,
    plans: RefCell<BTreeMap<String, ScheduledPhasePlan>>,
    reports: Rc<RefCell<Vec<(String, PendingScheduledPhaseReport)>>>,
    runtimes: RefCell<Vec<Rc<ScheduledRuntime>>>,
    defer_reports: bool,
}

impl ScheduledPhaseExecutionFactory {
    async fn wait_record_processors(&self) -> Result<()> {
        let runtimes = self.runtimes.borrow().clone();
        for runtime in runtimes {
            runtime.wait_record_processors().await?;
        }
        Ok(())
    }
}

impl PhaseExecutionFactory for ScheduledPhaseExecutionFactory {
    fn create(&self, config: &PhaseConfig, context: PhaseContext) -> Rc<dyn PhaseExecution> {
        let Some(mut plan) = self.plans.borrow_mut().remove(&config.id) else {
            return Rc::new(MissingScheduledPhaseExecution {
                phase_id: config.id.clone(),
            });
        };
        plan.ancillary.phase = match config.kind {
            PhaseKind::Warmup => crate::timing::Phase::Warmup,
            PhaseKind::Profiling => crate::timing::Phase::Profiling,
        };
        let tracker = Rc::new(PhaseDispatchTracker::new(context));
        let phase_start_ns = self.clock.now_ns();
        let start_ns = plan.start_ns.unwrap_or(phase_start_ns);
        let mut controller = plan.controller.clone();
        // Live metrics for the profiling phase's periodic realtime block: a
        // persistent accumulator fed one complete record per completion (see
        // `crate::realtime`). Built with the phase's metrics config, cloned here
        // before it is moved into the observer below. Only for the profiling
        // phase AND only when the realtime block is enabled, so a default run
        // pays zero per-completion cost.
        let realtime_live = (config.kind == PhaseKind::Profiling
            && crate::realtime::stats_interval_ns().is_some())
        .then(|| crate::realtime::LiveMetrics::new(plan.metrics_config.clone()));
        // Dedicated retain-mode observer feeding the realtime block. The
        // authoritative `native_metrics` below commonly runs in aggregate-only
        // mode, which drops each request slot inside `record_response` before the
        // detached record-processor task runs — so a snapshot against it would
        // always miss. This side observer is an ordinary phase delegate (receives
        // the same callbacks) whose slots the `LiveMetricsProcessor` drains per
        // completion, so retention stays bounded to in-flight requests. Built only
        // when the realtime block is enabled (see `realtime_live`).
        let realtime_observer = realtime_live.as_ref().map(|_| {
            Rc::new(NativeMetricsObserver::new(
                self.clock.clone(),
                start_ns,
                plan.metrics_config.clone(),
            ))
        });
        let collector = Rc::new(CollectorObserver::new(plan.capture_performance_records));
        let native_metrics = Rc::new(if !plan.retain_native_metric_record_dimensions {
            NativeMetricsObserver::new_aggregate_only(
                self.clock.clone(),
                start_ns,
                plan.metrics_config,
            )
        } else {
            NativeMetricsObserver::new(self.clock.clone(), start_ns, plan.metrics_config)
        });
        let mut delegates: Vec<Rc<dyn RequestObserver>> = Vec::with_capacity(
            usize::from(plan.collect_performance_summary) + 1 + plan.additional_observers.len(),
        );
        if plan.collect_performance_summary {
            delegates.push(collector.clone());
        }
        delegates.push(native_metrics.clone());
        delegates.append(&mut plan.additional_observers);
        // Fan the same request callbacks into the realtime block's dedicated
        // observer so its per-completion drain sees fully assembled records.
        if let Some(observer) = realtime_observer.clone() {
            delegates.push(observer);
        }
        // The common offline phase has only native metrics. Avoid routing every
        // callback through a fan-out allocation and loop when there is no fan-out.
        let delegate: Rc<dyn RequestObserver> = if delegates.len() == 1 {
            delegates.pop().expect("one observer delegate was counted")
        } else {
            Rc::new(ObserverTee::new(delegates))
        };
        let (observer, issuance_gate, extension_processors) =
            if let Some(extension) = plan.runtime_extension.take() {
                let extension = match extension.build(
                    self.clock.clone(),
                    start_ns,
                    phase_start_ns,
                    delegate,
                    controller,
                ) {
                    Ok(extension) => extension,
                    Err(error) => {
                        return Rc::new(FailedScheduledPhaseExecution {
                            phase_id: config.id.clone(),
                            error: format!("building phase runtime extension: {error:#}"),
                        });
                    }
                };
                controller = extension.controller;
                (
                    extension.observer,
                    extension.issuance_gate,
                    extension.record_processors,
                )
            } else {
                (delegate, None, Vec::new())
            };
        // Build the live-metrics processor over the dedicated realtime observer
        // (both are `Some` together, gated on the profiling phase + enabled
        // realtime block); registered below.
        let realtime_processor =
            realtime_live
                .as_ref()
                .zip(realtime_observer)
                .map(|(live, observer)| {
                    Rc::new(crate::realtime::LiveMetricsProcessor::new(
                        observer,
                        live.clone(),
                    )) as Rc<dyn TurnRecordProcessor>
                });
        let runtime = ScheduledRuntime::new_with_observer(
            self.clock.clone(),
            start_ns,
            self.dispatcher.clone(),
            config.stop,
            plan.enforce_stop,
            collector,
            native_metrics,
            observer,
            issuance_gate,
        );
        runtime.set_parallel_report_reduction(plan.parallel_report_reduction);
        runtime.set_timing_record_capture(plan.capture_timing_records);
        runtime.set_credit_latency_enabled(plan.workload.has_credit_timestamps());
        runtime.set_turn_lifecycle_observer(tracker.clone());
        for processor in plan.record_processors {
            runtime.add_record_processor(processor);
        }
        for processor in extension_processors {
            runtime.add_record_processor(processor);
        }
        // Fold each completed profiling request into the live accumulator the
        // realtime reporter reads.
        if let Some(processor) = realtime_processor {
            runtime.add_record_processor(processor);
        }
        runtime.configure_ancillary(
            plan.ancillary.cancellation_policy,
            plan.ancillary.url_selector,
            plan.ancillary.phase,
        );
        self.runtimes.borrow_mut().push(runtime.clone());
        Rc::new(ScheduledPhaseExecution {
            phase_id: config.id.clone(),
            clock: self.clock.clone(),
            workload: plan.workload,
            runtime,
            tracker,
            wait_for_natural_drain: !plan.enforce_stop,
            controller,
            resources: plan.resources,
            sidecars: plan.sidecars,
            reports: self.reports.clone(),
            defer_report: self.defer_reports,
            finalized: Cell::new(false),
            realtime_live,
            // Elapsed for the realtime line is measured from the actual phase
            // boundary, not the (possibly earlier) shared transport timeline
            // origin `start_ns` warmup and profiling share.
            realtime_origin_ns: phase_start_ns,
        })
    }

    fn cancel_all(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let runtimes = self.runtimes.borrow().clone();
        Box::pin(async move {
            for runtime in runtimes {
                runtime.scheduler().cancel_all();
            }
            tokio::task::yield_now().await;
            Ok(())
        })
    }
}

struct FailedScheduledPhaseExecution {
    phase_id: String,
    error: String,
}

impl PhaseExecution for FailedScheduledPhaseExecution {
    fn configure(&self, _config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        Err(PhaseExecutionError::new(format!(
            "phase {:?}: {}",
            self.phase_id, self.error
        )))
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }
}

struct MissingScheduledPhaseExecution {
    phase_id: String,
}

impl PhaseExecution for MissingScheduledPhaseExecution {
    fn configure(&self, _config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        Err(PhaseExecutionError::new(format!(
            "missing scheduled phase plan for {:?}",
            self.phase_id
        )))
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async { Ok(()) })
    }
}

struct ScheduledPhaseExecution {
    phase_id: String,
    clock: Rc<dyn crate::clock::Clock>,
    workload: Rc<dyn Workload>,
    runtime: Rc<ScheduledRuntime>,
    tracker: Rc<PhaseDispatchTracker>,
    wait_for_natural_drain: bool,
    controller: Rc<dyn ScheduledPhaseController>,
    resources: Rc<dyn ScheduledPhaseResources>,
    sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    reports: Rc<RefCell<Vec<(String, PendingScheduledPhaseReport)>>>,
    defer_report: bool,
    finalized: Cell<bool>,
    /// Live-metrics accumulator + phase-start origin for the periodic realtime
    /// block (`crate::realtime`). `Some` only for the profiling phase; a warmup
    /// phase never emits the block.
    realtime_live: Option<crate::realtime::LiveMetrics>,
    realtime_origin_ns: i64,
}

impl PhaseExecution for ScheduledPhaseExecution {
    fn configure(&self, config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        self.resources.configure(config).map_err(|error| {
            PhaseExecutionError::new(format!("configuring shared phase resources: {error:#}"))
        })
    }

    fn start_ramps(&self) -> Result<(), PhaseExecutionError> {
        self.controller
            .start()
            .map_err(|error| PhaseExecutionError::new(format!("starting phase ramps: {error:#}")))
    }

    fn setup(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let sidecars = self.sidecars.clone();
        let clock = self.clock.clone();
        Box::pin(async move { start_phase_sidecars(&sidecars, clock.as_ref(), "scheduled").await })
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let workload = self.workload.clone();
        let runtime = self.runtime.clone();
        let wait_for_natural_drain = self.wait_for_natural_drain;
        let controller = self.controller.clone();
        // Periodic realtime-metrics block for the profiling phase: a
        // Clock-driven task summarizes the live accumulator every interval and
        // logs one `[realtime …]` block, aborted when this phase completes.
        let realtime = self.realtime_live.clone();
        let realtime_clock = self.clock.clone();
        let realtime_origin_ns = self.realtime_origin_ns;
        Box::pin(async move {
            let realtime_task =
                realtime
                    .zip(crate::realtime::stats_interval_ns())
                    .map(|(live, interval_ns)| {
                        tokio::task::spawn_local(crate::realtime::realtime_reporter_loop(
                            realtime_clock,
                            live,
                            realtime_origin_ns,
                            interval_ns,
                        ))
                    });

            let execution = workload.execute(runtime.clone());
            let stop = controller.wait_until_stop();
            tokio::pin!(execution);
            tokio::pin!(stop);
            let result = tokio::select! {
                result = &mut execution => result.map_err(|error| {
                    PhaseExecutionError::new(format!("scheduled workload: {error:#}"))
                }),
                () = &mut stop => {
                    runtime.scheduler().cancel_pending();
                    Ok(())
                }
            };
            // Keep the realtime reporter running through the natural drain so its
            // last block reflects the final completions, then abort it (all exit
            // paths) so no block fires after the phase ends.
            if result.is_ok() && wait_for_natural_drain {
                // Authored/naturally exhausted workloads do not have a stop
                // counter that can publish the last-send edge. Their scheduler
                // is therefore the authoritative completion signal.
                runtime.scheduler().wait_idle().await;
            }
            if let Some(task) = &realtime_task {
                task.abort();
            }
            result
        })
    }

    fn stop_issuing(&self) {
        self.runtime.scheduler().cancel_pending();
    }

    fn cancel_pending(&self) {
        self.runtime.scheduler().cancel_pending();
    }

    fn cancel_inflight(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let runtime = self.runtime.clone();
        let tracker = self.tracker.clone();
        Box::pin(async move {
            runtime.scheduler().cancel_all();
            tracker.cancel_active();
            tokio::task::yield_now().await;
            Ok(())
        })
    }

    fn release_stuck_slots(&self) -> ReleasedStuckSlots {
        let active = self.tracker.cancel_active();
        let cleanup = self.resources.release_stuck();
        ReleasedStuckSlots {
            session: cleanup.session,
            prefill: cleanup.prefill.saturating_add(active),
        }
    }

    fn stop_ramps(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let controller = self.controller.clone();
        Box::pin(async move {
            controller.stop().await.map_err(|error| {
                PhaseExecutionError::new(format!("stopping phase ramps: {error:#}"))
            })
        })
    }

    fn finalize(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        if self.finalized.replace(true) {
            return Box::pin(async { Ok(()) });
        }
        let phase_id = self.phase_id.clone();
        let strategy = self.workload.name();
        let snapshot = self.workload.user_control_snapshot();
        let runtime = self.runtime.clone();
        let sidecars = self.sidecars.clone();
        let clock = self.clock.clone();
        let reports = self.reports.clone();
        let defer_report = self.defer_report;
        Box::pin(async move {
            runtime.scheduler().wait_idle().await;
            finish_phase_sidecars(&sidecars, clock.as_ref(), "scheduled").await?;
            let report = if defer_report {
                PendingScheduledPhaseReport::Deferred {
                    runtime,
                    end_ns: clock.now_ns(),
                    strategy,
                    user_control: snapshot,
                }
            } else {
                PendingScheduledPhaseReport::Finalized(Box::new(runtime.finish(strategy, snapshot)))
            };
            reports.borrow_mut().push((phase_id, report));
            Ok(())
        })
    }
}

#[derive(Clone, Copy)]
struct ActiveTurn {
    completes_session: bool,
    first_token_seen: bool,
}

struct PhaseDispatchTracker {
    context: PhaseContext,
    // Lifecycle updates address requests by UUID and cancellation only reduces
    // counts, so maintaining key order adds work without observable ordering.
    active: RefCell<FxHashMap<Uuid, ActiveTurn>>,
}

impl TurnLifecycleObserver for PhaseDispatchTracker {
    fn on_issue(&self, turn: &TurnToSend) {
        // The trait method returns `()`, so `start`'s `Result` cannot propagate;
        // a duplicate UUID is an invariant violation rather than a recoverable error.
        self.start(turn)
            .expect("phase runtime must observe each accepted turn exactly once");
    }

    fn on_first_token(&self, uuid: Uuid) {
        self.first_token(uuid);
    }

    fn on_terminal(&self, turn: &TurnToSend, outcome: &TurnDispatchOutcome) {
        self.finish(turn.uuid, outcome.terminal);
    }
}

impl PhaseDispatchTracker {
    fn new(context: PhaseContext) -> Self {
        Self {
            context,
            active: RefCell::new(FxHashMap::default()),
        }
    }

    fn start(&self, turn: &TurnToSend) -> Result<()> {
        if self.active.borrow().contains_key(&turn.uuid) {
            return Err(anyhow!("duplicate active phase request {}", turn.uuid));
        }
        self.context
            .record_sent(PhaseSend {
                is_root: true,
                starts_session: turn.turn_index == 0,
                planned_session_turns: if turn.turn_index == 0 {
                    turn.num_turns as u64
                } else {
                    0
                },
            })
            .map_err(|error| anyhow!(error))?;
        self.active.borrow_mut().insert(
            turn.uuid,
            ActiveTurn {
                completes_session: turn.turn_index + 1 >= turn.num_turns,
                first_token_seen: false,
            },
        );
        Ok(())
    }

    fn first_token(&self, uuid: Uuid) {
        if let Some(active) = self.active.borrow_mut().get_mut(&uuid)
            && !active.first_token_seen
        {
            active.first_token_seen = true;
            self.context.record_first_token();
        }
    }

    fn finish(&self, uuid: Uuid, terminal: ReplayTerminalStatus) {
        let Some(active) = self.active.borrow_mut().remove(&uuid) else {
            return;
        };
        self.context.record_returned(PhaseReturn {
            completes_session: active.completes_session,
            cancelled: terminal == ReplayTerminalStatus::Canceled,
            errored: matches!(
                terminal,
                ReplayTerminalStatus::Failed | ReplayTerminalStatus::Rejected
            ),
            releases_prefill: !active.first_token_seen,
        });
    }

    fn cancel_active(&self) -> u64 {
        let active = std::mem::take(&mut *self.active.borrow_mut());
        let count = active.len() as u64;
        for (_, active) in active {
            self.context.record_returned(PhaseReturn {
                completes_session: active.completes_session,
                cancelled: true,
                releases_prefill: !active.first_token_seen,
                ..PhaseReturn::default()
            });
        }
        count
    }
}
