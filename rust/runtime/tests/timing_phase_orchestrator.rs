// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Warmup/profiling handoff and shared debt-drain integration coverage.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use aiperf_runtime::clock::{Clock, sim_clock::SimClock};
use aiperf_runtime::timing::{
    ClockPhaseOrchestrator, ClockPhaseRunnerFactory, GracePeriod, LocalPhaseFuture,
    PhaseBranchStats, PhaseCompletionReason, PhaseConfig, PhaseContext, PhaseExecution,
    PhaseExecutionError, PhaseExecutionFactory, PhaseKind, PhaseObserver, PhaseOrchestrator,
    PhaseOrchestratorError, PhaseReturn, PhaseRunError, PhaseSend, PhaseStats, SlotPool,
    StopConfig,
};
use tokio::task::LocalSet;

#[derive(Clone, Debug, PartialEq, Eq)]
enum TimelineEvent {
    Start(String, i64),
    Complete(String, i64),
    Issue(String, i64),
    Configure(String, i64, usize),
    RunComplete(i64),
}

struct TimelineObserver {
    clock: Rc<SimClock>,
    events: Rc<RefCell<Vec<TimelineEvent>>>,
}

impl PhaseObserver for TimelineObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, stats: PhaseStats) {
        self.events
            .borrow_mut()
            .push(TimelineEvent::Start(stats.phase_id, self.clock.now_ns()));
    }

    fn on_progress(&self, _stats: PhaseStats) {}

    fn on_sending_complete(&self, _stats: PhaseStats) {}

    fn on_phase_complete(&self, stats: PhaseStats, _branch_stats: Option<PhaseBranchStats>) {
        self.events
            .borrow_mut()
            .push(TimelineEvent::Complete(stats.phase_id, self.clock.now_ns()));
    }

    fn on_phases_complete(&self, _stats: Vec<PhaseStats>) {
        self.events
            .borrow_mut()
            .push(TimelineEvent::RunComplete(self.clock.now_ns()));
    }
}

struct DebtExecutionFactory {
    clock: Rc<SimClock>,
    slots: Rc<SlotPool>,
    events: Rc<RefCell<Vec<TimelineEvent>>>,
}

impl PhaseExecutionFactory for DebtExecutionFactory {
    fn create(&self, config: &PhaseConfig, context: PhaseContext) -> Rc<dyn PhaseExecution> {
        Rc::new(DebtExecution {
            phase_id: config.id.clone(),
            kind: config.kind,
            limit: config.concurrency.expect("test config has a limit"),
            clock: self.clock.clone(),
            slots: self.slots.clone(),
            events: self.events.clone(),
            context,
        })
    }
}

struct DebtExecution {
    phase_id: String,
    kind: PhaseKind,
    limit: usize,
    clock: Rc<SimClock>,
    slots: Rc<SlotPool>,
    events: Rc<RefCell<Vec<TimelineEvent>>>,
    context: PhaseContext,
}

impl PhaseExecution for DebtExecution {
    fn configure(&self, _config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        self.slots.set_limit(self.limit);
        self.events.borrow_mut().push(TimelineEvent::Configure(
            self.phase_id.clone(),
            self.clock.now_ns(),
            self.slots.debt(),
        ));
        Ok(())
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let phase_id = self.phase_id.clone();
        let kind = self.kind;
        let clock = self.clock.clone();
        let slots = self.slots.clone();
        let events = self.events.clone();
        let context = self.context.clone();
        Box::pin(async move {
            let requests = match kind {
                PhaseKind::Warmup => 4,
                PhaseKind::Profiling => 1,
            };
            for request_index in 0..requests {
                let guard = slots.acquire().await;
                events
                    .borrow_mut()
                    .push(TimelineEvent::Issue(phase_id.clone(), clock.now_ns()));
                context
                    .record_sent(PhaseSend::single_turn_session())
                    .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
                let returned = context.clone();
                let return_clock = clock.clone();
                let delay = match kind {
                    PhaseKind::Warmup => (request_index as i64 + 1) * 10,
                    PhaseKind::Profiling => 5,
                };
                tokio::task::spawn_local(async move {
                    return_clock.sleep(delay).await;
                    returned.record_first_token();
                    returned.record_returned(PhaseReturn {
                        completes_session: true,
                        ..PhaseReturn::default()
                    });
                    drop(guard);
                });
            }
            Ok(())
        })
    }
}

#[test]
fn seamless_handoff_overlaps_runners_and_debt_drains_shared_capacity() {
    let clock = Rc::new(SimClock::new());
    let (orchestrator, events, slots) = orchestrator(clock.clone(), true);

    let stats = drive_sim(clock.clone(), orchestrator.run_all()).unwrap();

    assert_eq!(stats.len(), 2);
    assert_eq!(stats[0].phase_id, "warmup");
    assert_eq!(stats[1].phase_id, "profiling");
    assert_eq!(
        stats[0].completion_reason,
        Some(PhaseCompletionReason::Completed)
    );
    assert_eq!(stats[0].final_requests_completed, Some(4));
    assert_eq!(stats[1].final_requests_completed, Some(1));
    assert_eq!(orchestrator.active_phase_count(), 0);

    let events = events.borrow();
    assert!(events.contains(&TimelineEvent::Start("warmup".into(), 0)));
    assert!(events.contains(&TimelineEvent::Start("profiling".into(), 0)));
    assert!(events.contains(&TimelineEvent::Configure("profiling".into(), 0, 1)));
    assert!(events.contains(&TimelineEvent::Issue("profiling".into(), 20)));
    assert!(events.contains(&TimelineEvent::Complete("profiling".into(), 25)));
    assert!(events.contains(&TimelineEvent::Complete("warmup".into(), 40)));
    assert!(events.contains(&TimelineEvent::RunComplete(40)));
    assert_eq!(clock.now_ns(), 40);
    assert_eq!(slots.current_limit(), 3);
    assert_eq!(slots.debt(), 0);
    assert_eq!(slots.effective_slots(), 3);
}

#[test]
fn non_seamless_handoff_waits_for_warmup_to_fully_drain() {
    let clock = Rc::new(SimClock::new());
    let (orchestrator, events, slots) = orchestrator(clock.clone(), false);

    let stats = drive_sim(clock.clone(), orchestrator.run_all()).unwrap();

    assert_eq!(stats.len(), 2);
    let events = events.borrow();
    assert!(events.contains(&TimelineEvent::Complete("warmup".into(), 40)));
    assert!(events.contains(&TimelineEvent::Start("profiling".into(), 40)));
    assert!(events.contains(&TimelineEvent::Configure("profiling".into(), 40, 0)));
    assert!(events.contains(&TimelineEvent::Issue("profiling".into(), 40)));
    assert!(events.contains(&TimelineEvent::RunComplete(45)));
    assert_eq!(clock.now_ns(), 45);
    assert_eq!(slots.debt(), 0);
}

#[test]
fn named_workflow_allows_warmup_after_profiling() {
    let clock = Rc::new(SimClock::new());
    let events = Rc::new(RefCell::new(Vec::new()));
    let observer: Rc<dyn PhaseObserver> = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events,
    });
    let factory: Rc<dyn PhaseExecutionFactory> = Rc::new(DebtExecutionFactory {
        clock: clock.clone(),
        slots: Rc::new(SlotPool::new(0)),
        events: Rc::new(RefCell::new(Vec::new())),
    });
    let clock_dyn: Rc<dyn Clock> = clock;
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock_dyn,
        observer.clone(),
        factory,
    ));
    let configs = vec![profiling_config(), warmup_config(false)];

    ClockPhaseOrchestrator::new(configs, runner_factory, observer)
        .expect("named workflows may place warmup after profiling");
}

#[test]
fn validation_rejects_a_warmup_only_benchmark() {
    let clock = Rc::new(SimClock::new());
    let events = Rc::new(RefCell::new(Vec::new()));
    let observer: Rc<dyn PhaseObserver> = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events: events.clone(),
    });
    let factory: Rc<dyn PhaseExecutionFactory> = Rc::new(DebtExecutionFactory {
        clock: clock.clone(),
        slots: Rc::new(SlotPool::new(0)),
        events,
    });
    let clock_dyn: Rc<dyn Clock> = clock;
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock_dyn,
        observer.clone(),
        factory,
    ));

    let error =
        match ClockPhaseOrchestrator::new(vec![warmup_config(false)], runner_factory, observer) {
            Ok(_) => panic!("warmup-only benchmark was accepted"),
            Err(error) => error,
        };
    assert_eq!(error, PhaseOrchestratorError::ProfilingPhaseRequired);
}

#[test]
fn external_cancel_never_advances_into_the_next_phase() {
    let clock = Rc::new(SimClock::new());
    let (orchestrator, events, _slots) = orchestrator(clock.clone(), false);
    let cancellation = orchestrator.clone();
    let cancellation_clock = clock.clone();

    let stats = drive_sim(clock.clone(), async move {
        tokio::task::spawn_local(async move {
            cancellation_clock.sleep(5).await;
            cancellation.cancel().await.unwrap();
        });
        orchestrator.run_all().await
    })
    .unwrap();

    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0].phase_id, "warmup");
    assert!(stats[0].was_cancelled);
    assert_eq!(
        stats[0].completion_reason,
        Some(PhaseCompletionReason::Cancelled)
    );
    assert_eq!(clock.now_ns(), 40);
    assert!(
        !events
            .borrow()
            .iter()
            .any(|event| matches!(event, TimelineEvent::Start(id, _) if id == "profiling"))
    );
}

struct SeamlessFailureFactory {
    clock: Rc<SimClock>,
    cancel_all_calls: Rc<Cell<usize>>,
    profiling_cancellations: Rc<Cell<usize>>,
}

impl PhaseExecutionFactory for SeamlessFailureFactory {
    fn create(&self, config: &PhaseConfig, context: PhaseContext) -> Rc<dyn PhaseExecution> {
        match config.id.as_str() {
            "warmup" => Rc::new(DelayedFinalizeFailure {
                clock: self.clock.clone(),
                context,
            }),
            "profiling-active" | "profiling-must-not-start" => Rc::new(BlockingExecution {
                cancellations: self.profiling_cancellations.clone(),
            }),
            id => panic!("unexpected phase {id:?}"),
        }
    }

    fn cancel_all(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let calls = self.cancel_all_calls.clone();
        Box::pin(async move {
            calls.set(calls.get() + 1);
            Ok(())
        })
    }
}

struct DelayedFinalizeFailure {
    clock: Rc<SimClock>,
    context: PhaseContext,
}

impl PhaseExecution for DelayedFinalizeFailure {
    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let clock = self.clock.clone();
        let context = self.context.clone();
        Box::pin(async move {
            context
                .record_sent(PhaseSend::single_turn_session())
                .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
            tokio::task::spawn_local(async move {
                clock.sleep(10).await;
                context.record_first_token();
                context.record_returned(PhaseReturn {
                    completes_session: true,
                    ..PhaseReturn::default()
                });
            });
            Ok(())
        })
    }

    fn finalize(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(async {
            Err(PhaseExecutionError::new(
                "intentional seamless predecessor failure",
            ))
        })
    }
}

struct BlockingExecution {
    cancellations: Rc<Cell<usize>>,
}

impl PhaseExecution for BlockingExecution {
    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        Box::pin(std::future::pending())
    }

    fn cancel_inflight(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let cancellations = self.cancellations.clone();
        Box::pin(async move {
            cancellations.set(cancellations.get() + 1);
            Ok(())
        })
    }
}

#[test]
fn seamless_predecessor_failure_cancels_active_phase_before_advancing() {
    let clock = Rc::new(SimClock::new());
    let events = Rc::new(RefCell::new(Vec::new()));
    let observer: Rc<dyn PhaseObserver> = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events: events.clone(),
    });
    let cancel_all_calls = Rc::new(Cell::new(0));
    let profiling_cancellations = Rc::new(Cell::new(0));
    let execution_factory: Rc<dyn PhaseExecutionFactory> = Rc::new(SeamlessFailureFactory {
        clock: clock.clone(),
        cancel_all_calls: cancel_all_calls.clone(),
        profiling_cancellations: profiling_cancellations.clone(),
    });
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock.clone(),
        observer.clone(),
        execution_factory,
    ));
    let orchestrator = ClockPhaseOrchestrator::new(
        vec![
            PhaseConfig::new(
                "warmup",
                PhaseKind::Warmup,
                StopConfig {
                    total_expected_requests: Some(1),
                    ..StopConfig::default()
                },
            )
            .with_seamless(true),
            PhaseConfig::new(
                "profiling-active",
                PhaseKind::Profiling,
                StopConfig {
                    expected_duration_ns: Some(100),
                    ..StopConfig::default()
                },
            ),
            PhaseConfig::new(
                "profiling-must-not-start",
                PhaseKind::Profiling,
                StopConfig {
                    total_expected_requests: Some(1),
                    ..StopConfig::default()
                },
            ),
        ],
        runner_factory,
        observer,
    )
    .unwrap();

    let error = drive_sim(clock.clone(), orchestrator.run_all()).unwrap_err();

    assert_eq!(
        error,
        PhaseOrchestratorError::Runner {
            phase_id: "warmup".into(),
            source: PhaseRunError::Execution(PhaseExecutionError::new(
                "intentional seamless predecessor failure"
            )),
        }
    );
    assert_eq!(clock.now_ns(), 10);
    assert_eq!(cancel_all_calls.get(), 1);
    assert_eq!(profiling_cancellations.get(), 1);
    let events = events.borrow();
    assert!(events.contains(&TimelineEvent::Start("warmup".into(), 0)));
    assert!(events.contains(&TimelineEvent::Start("profiling-active".into(), 0)));
    assert!(!events.iter().any(
        |event| matches!(event, TimelineEvent::Start(id, _) if id == "profiling-must-not-start")
    ));
}

fn orchestrator(
    clock: Rc<SimClock>,
    seamless: bool,
) -> (
    ClockPhaseOrchestrator,
    Rc<RefCell<Vec<TimelineEvent>>>,
    Rc<SlotPool>,
) {
    let events = Rc::new(RefCell::new(Vec::new()));
    let slots = Rc::new(SlotPool::new(0));
    let observer: Rc<dyn PhaseObserver> = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events: events.clone(),
    });
    let execution_factory: Rc<dyn PhaseExecutionFactory> = Rc::new(DebtExecutionFactory {
        clock: clock.clone(),
        slots: slots.clone(),
        events: events.clone(),
    });
    let clock_dyn: Rc<dyn Clock> = clock;
    let runner_factory = Rc::new(ClockPhaseRunnerFactory::new(
        clock_dyn,
        observer.clone(),
        execution_factory,
    ));
    let orchestrator = ClockPhaseOrchestrator::new(
        vec![warmup_config(seamless), profiling_config()],
        runner_factory,
        observer,
    )
    .unwrap();
    (orchestrator, events, slots)
}

fn warmup_config(seamless: bool) -> PhaseConfig {
    PhaseConfig::new(
        "warmup",
        PhaseKind::Warmup,
        StopConfig {
            total_expected_requests: Some(4),
            ..StopConfig::default()
        },
    )
    .with_seamless(seamless)
    .with_concurrency(Some(4), None)
    .with_runtime_intervals(100, 50)
}

fn profiling_config() -> PhaseConfig {
    PhaseConfig::new(
        "profiling",
        PhaseKind::Profiling,
        StopConfig {
            total_expected_requests: Some(1),
            ..StopConfig::default()
        },
    )
    .with_grace_period(GracePeriod::Disabled)
    .with_concurrency(Some(3), None)
    .with_runtime_intervals(100, 50)
}

fn drive_sim<F, T>(clock: Rc<SimClock>, body: F) -> T
where
    F: Future<Output = T>,
{
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime");
    let _guard = runtime.enter();
    let local = LocalSet::new();
    let future = local.run_until(body);
    let mut future = pin!(future);
    let flag = Arc::new(AtomicBool::new(true));
    let waker = Waker::from(Arc::new(FlagWaker(flag.clone())));
    let mut context = Context::from_waker(&waker);

    loop {
        flag.store(false, Ordering::SeqCst);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending if flag.load(Ordering::SeqCst) => {}
            Poll::Pending => match clock.next_event_time() {
                Some(at_ns) => clock.advance_to(at_ns),
                None => panic!("phase orchestrator deadlocked with no clock event"),
            },
        }
    }
}

struct FlagWaker(Arc<AtomicBool>);

impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}
