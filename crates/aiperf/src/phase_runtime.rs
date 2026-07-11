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

use aiperf_timing::{
    ClockPhaseOrchestrator, ClockPhaseRunnerFactory, LocalPhaseFuture, PhaseConfig, PhaseContext,
    PhaseExecution, PhaseExecutionError, PhaseExecutionFactory, PhaseKind, PhaseObserver,
    PhaseOrchestrator, PhaseReturn, PhaseSend, PhaseStats, ReleasedStuckSlots,
};
use anyhow::{Result, anyhow};
use loadgen_core::collector::ReplayTerminalStatus;
use serde::Serialize;
use uuid::Uuid;

use crate::multiturn::TurnToSend;
use crate::scheduled::{
    ScheduledAncillaryPolicies, ScheduledRunReport, ScheduledRuntime, TurnDispatchOutcome,
    TurnDispatcher, TurnLifecycleObserver, TurnRecordProcessor, Workload,
};
use crate::scheduler::LocalTaskScheduler;

/// Optional phase-owned actuator/ramp lifecycle.
pub trait ScheduledPhaseController {
    /// Start actuators before phase issuance begins.
    fn start(&self) -> Result<()>;

    /// Stop and join actuators at sending handoff.
    fn stop(&self) -> LocalPhaseFuture<Result<()>>;
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

/// Workload-specific force-cleanup seam for admission guards stored outside
/// scheduler tasks.
pub trait ScheduledPhaseCleanup {
    /// Release phase-owned admission state after cancellation drain fails.
    fn release_stuck(&self) -> ReleasedStuckSlots;
}

/// Cleanup used by workloads whose guards live entirely in scheduler tasks.
#[derive(Default)]
pub struct NoopScheduledPhaseCleanup;

impl ScheduledPhaseCleanup for NoopScheduledPhaseCleanup {
    fn release_stuck(&self) -> ReleasedStuckSlots {
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
    /// Phase-owned actuator/ramp lifecycle.
    pub controller: Rc<dyn ScheduledPhaseController>,
    /// Workload-specific force cleanup.
    pub cleanup: Rc<dyn ScheduledPhaseCleanup>,
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
            controller: Rc::new(NoopScheduledPhaseController),
            cleanup: Rc::new(NoopScheduledPhaseCleanup),
        }
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
    let execution_factory: Rc<dyn PhaseExecutionFactory> =
        Rc::new(ScheduledPhaseExecutionFactory {
            clock: clock.clone(),
            dispatcher,
            plans: RefCell::new(
                plans
                    .into_iter()
                    .map(|plan| (plan.config.id.clone(), plan))
                    .collect(),
            ),
            reports: reports.clone(),
        });
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock,
        observer.clone(),
        execution_factory,
    ));
    let orchestrator = ClockPhaseOrchestrator::new(configs, runner_factory, observer)
        .map_err(|error| anyhow!(error))?;
    let phases = orchestrator
        .run_all()
        .await
        .map_err(|error| anyhow!(error))?;

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
        let start_ns = self.clock.now_ns();
        let runtime = ScheduledRuntime::new(
            self.clock.clone(),
            start_ns,
            self.dispatcher.clone(),
            config.stop,
            true,
        );
        runtime.set_turn_lifecycle_observer(tracker.clone());
        for processor in plan.record_processors {
            runtime.add_record_processor(processor);
        }
        runtime.configure_ancillary(
            plan.ancillary.cancellation_policy,
            plan.ancillary.url_selector,
            plan.ancillary.phase,
        );
        Rc::new(ScheduledPhaseExecution {
            phase_id: config.id.clone(),
            workload: plan.workload,
            runtime,
            tracker,
            controller: plan.controller,
            cleanup: plan.cleanup,
            reports: self.reports.clone(),
            finalized: Cell::new(false),
        })
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
    workload: Rc<dyn Workload>,
    runtime: Rc<ScheduledRuntime>,
    tracker: Rc<PhaseDispatchTracker>,
    controller: Rc<dyn ScheduledPhaseController>,
    cleanup: Rc<dyn ScheduledPhaseCleanup>,
    reports: Rc<RefCell<Vec<(String, ScheduledRunReport)>>>,
    finalized: Cell<bool>,
}

impl PhaseExecution for ScheduledPhaseExecution {
    fn start_ramps(&self) -> Result<(), PhaseExecutionError> {
        self.controller
            .start()
            .map_err(|error| PhaseExecutionError::new(format!("starting phase ramps: {error:#}")))
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let workload = self.workload.clone();
        let runtime = self.runtime.clone();
        Box::pin(async move {
            workload
                .execute(runtime)
                .await
                .map_err(|error| PhaseExecutionError::new(format!("scheduled workload: {error:#}")))
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
        let cleanup = self.cleanup.release_stuck();
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
        let reports = self.reports.clone();
        Box::pin(async move {
            runtime.scheduler().wait_idle().await;
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
