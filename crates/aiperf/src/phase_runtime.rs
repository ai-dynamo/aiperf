// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production composition of scheduled workloads through the phase driver.
//!
//! [`run_scheduled_phases`] connects the transport-neutral
//! [`TurnDispatcher`](crate::scheduled::TurnDispatcher) and
//! [`Workload`](crate::scheduled::Workload) seams to
//! `aiperf_timing::{PhaseRunner, PhaseOrchestrator}`. The adapter records sends,
//! first tokens, and terminal returns at the dispatcher boundary, so workload
//! implementations remain schedule generators and do not learn phase lifecycle
//! policy. One factory is shared across every phase; each phase still receives
//! a fresh runtime, observer graph, counter set, and report.

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use aiperf_core::observer::CollectorObserver;
use aiperf_metrics::MetricsConfig;
use aiperf_timing::{
    ClockPhaseOrchestrator, ClockPhaseRunnerFactory, LocalPhaseFuture, PhaseConfig, PhaseContext,
    PhaseExecution, PhaseExecutionError, PhaseExecutionFactory, PhaseKind, PhaseObserver,
    PhaseOrchestrator, PhaseReturn, PhaseSend, PhaseStats, RampDriver, RampHandle,
    ReleasedStuckSlots, SlotPool,
};
use anyhow::{Result, anyhow};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use serde::Serialize;
use uuid::Uuid;

use crate::metrics::{NativeMetricsObserver, ObserverTee};
use crate::multiturn::TurnToSend;
use crate::scheduled::{
    IssuanceGate, ScheduledAncillaryPolicies, ScheduledRunReport, ScheduledRuntime,
    TurnDispatchOutcome, TurnDispatcher, TurnLifecycleObserver, TurnRecordProcessor, Workload,
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
        clock: Rc<dyn aiperf_clock::Clock>,
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
/// Sidecars are intentionally outside [`RequestObserver`]: low-rate telemetry,
/// profilers, and future extension runtimes may need a forced sample before
/// issuance and after every return without adding work to the per-token path.
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

/// Phased result plus one compatibility report accumulated directly from the
/// live observer stream across the complete shared clock/dispatcher lifecycle.
#[derive(Debug, Serialize)]
pub struct AggregatedPhasedScheduledRunReport {
    /// Independently finalized phase lifecycle and performance reports.
    pub phased: PhasedScheduledRunReport,
    /// Whole-run compatibility metrics observed before phase finalization.
    pub performance: loadgen_core::collector::TraceSimulationReport,
}

/// Run scheduled phases while one collector observes every phase on the
/// caller-supplied timeline.
///
/// This is the backend-neutral contract for a single engine that spans warmup
/// and profiling. It accumulates original callbacks and never merges already
/// finalized phase summaries.
pub async fn run_scheduled_phases_with_aggregate(
    mut plans: Vec<ScheduledPhasePlan>,
    clock: Rc<dyn aiperf_clock::Clock>,
    start_ns: i64,
    dispatcher: Rc<dyn TurnDispatcher>,
    observer: Rc<dyn PhaseObserver>,
) -> Result<AggregatedPhasedScheduledRunReport> {
    let collector = Rc::new(CollectorObserver::new(true));
    let aggregate_observer: Rc<dyn RequestObserver> = collector.clone();
    for plan in &mut plans {
        plan.additional_observers.push(aggregate_observer.clone());
    }
    let phased = run_scheduled_phases(plans, clock.clone(), dispatcher, observer).await?;
    let wall_ms = clock.now_ns().saturating_sub(start_ns) as f64 / 1_000_000.0;
    Ok(AggregatedPhasedScheduledRunReport {
        phased,
        performance: collector.finish(wall_ms),
    })
}

/// Run prepared scheduled workloads through the shared phase orchestrator.
pub async fn run_scheduled_phases(
    plans: Vec<ScheduledPhasePlan>,
    clock: Rc<dyn aiperf_clock::Clock>,
    dispatcher: Rc<dyn TurnDispatcher>,
    observer: Rc<dyn PhaseObserver>,
) -> Result<PhasedScheduledRunReport> {
    let configs = plans
        .iter()
        .map(|plan| plan.config.clone())
        .collect::<Vec<_>>();
    let order = configs
        .iter()
        .enumerate()
        .map(|(index, config)| (config.id.clone(), (index, config.kind)))
        .collect::<BTreeMap<_, _>>();
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
    });
    let phase_execution_factory: Rc<dyn PhaseExecutionFactory> = execution_factory.clone();
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock,
        observer.clone(),
        phase_execution_factory,
    ));
    let orchestrator = ClockPhaseOrchestrator::new(configs, runner_factory, observer)
        .map_err(|error| anyhow!(error))?;
    let phase_result = orchestrator.run_all().await.map_err(|error| anyhow!(error));
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

    let mut reports = reports.borrow_mut().drain(..).collect::<Vec<_>>();
    reports.sort_by_key(|(phase_id, _)| {
        order
            .get(phase_id)
            .map(|(index, _)| *index)
            .unwrap_or(usize::MAX)
    });
    let reports = reports
        .into_iter()
        .map(|(phase_id, report)| ScheduledPhaseReport {
            kind: order
                .get(&phase_id)
                .map(|(_, kind)| *kind)
                .unwrap_or(PhaseKind::Profiling),
            phase_id,
            report,
        })
        .collect();
    Ok(PhasedScheduledRunReport { phases, reports })
}

struct ScheduledPhaseExecutionFactory {
    clock: Rc<dyn aiperf_clock::Clock>,
    dispatcher: Rc<dyn TurnDispatcher>,
    plans: RefCell<BTreeMap<String, ScheduledPhasePlan>>,
    reports: Rc<RefCell<Vec<(String, ScheduledRunReport)>>>,
    runtimes: RefCell<Vec<Rc<ScheduledRuntime>>>,
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
            PhaseKind::Warmup => aiperf_timing::Phase::Warmup,
            PhaseKind::Profiling => aiperf_timing::Phase::Profiling,
        };
        let tracker = Rc::new(PhaseDispatchTracker::new(context));
        let phase_start_ns = self.clock.now_ns();
        let start_ns = plan.start_ns.unwrap_or(phase_start_ns);
        let mut controller = plan.controller.clone();
        let collector = Rc::new(CollectorObserver::new(true));
        let native_metrics = Rc::new(NativeMetricsObserver::new(
            self.clock.clone(),
            start_ns,
            plan.metrics_config,
        ));
        let mut delegates: Vec<Rc<dyn RequestObserver>> =
            vec![collector.clone(), native_metrics.clone()];
        delegates.append(&mut plan.additional_observers);
        let delegate: Rc<dyn RequestObserver> = Rc::new(ObserverTee::new(delegates));
        let (observer, issuance_gate) = if let Some(extension) = plan.runtime_extension.take() {
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
            (extension.observer, extension.issuance_gate)
        } else {
            (delegate, None)
        };
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
        runtime.set_credit_latency_enabled(plan.workload.has_credit_timestamps());
        runtime.set_turn_lifecycle_observer(tracker.clone());
        for processor in plan.record_processors {
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
            finalized: Cell::new(false),
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
    clock: Rc<dyn aiperf_clock::Clock>,
    workload: Rc<dyn Workload>,
    runtime: Rc<ScheduledRuntime>,
    tracker: Rc<PhaseDispatchTracker>,
    wait_for_natural_drain: bool,
    controller: Rc<dyn ScheduledPhaseController>,
    resources: Rc<dyn ScheduledPhaseResources>,
    sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>>,
    reports: Rc<RefCell<Vec<(String, ScheduledRunReport)>>>,
    finalized: Cell<bool>,
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
        Box::pin(async move {
            for sidecar in &sidecars {
                sidecar.start().await.map_err(|error| {
                    PhaseExecutionError::new(format!("starting scheduled phase sidecar: {error:#}"))
                })?;
            }
            let phase_start_ns = clock.now_ns();
            for sidecar in &sidecars {
                sidecar.on_phase_start(phase_start_ns);
            }
            Ok(())
        })
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let workload = self.workload.clone();
        let runtime = self.runtime.clone();
        let wait_for_natural_drain = self.wait_for_natural_drain;
        let controller = self.controller.clone();
        Box::pin(async move {
            let execution = workload.execute(runtime.clone());
            let stop = controller.wait_until_stop();
            tokio::pin!(execution);
            tokio::pin!(stop);
            tokio::select! {
                result = &mut execution => result.map_err(|error| {
                    PhaseExecutionError::new(format!("scheduled workload: {error:#}"))
                })?,
                () = &mut stop => runtime.scheduler().cancel_pending(),
            }
            if wait_for_natural_drain {
                // Authored/naturally exhausted workloads do not have a stop
                // counter that can publish the last-send edge. Their scheduler
                // is therefore the authoritative completion signal.
                runtime.scheduler().wait_idle().await;
            }
            Ok(())
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
        Box::pin(async move {
            runtime.scheduler().wait_idle().await;
            let phase_end_ns = clock.now_ns();
            for sidecar in &sidecars {
                sidecar.on_phase_end(phase_end_ns);
            }
            for sidecar in &sidecars {
                sidecar.finish().await.map_err(|error| {
                    PhaseExecutionError::new(format!(
                        "finishing scheduled phase sidecar: {error:#}"
                    ))
                })?;
            }
            reports
                .borrow_mut()
                .push((phase_id, runtime.finish(strategy, snapshot)));
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
    active: RefCell<BTreeMap<Uuid, ActiveTurn>>,
}

impl TurnLifecycleObserver for PhaseDispatchTracker {
    fn on_issue(&self, turn: &TurnToSend) {
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
            active: RefCell::new(BTreeMap::new()),
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
