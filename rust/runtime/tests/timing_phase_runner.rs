// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic integration coverage for the clock-native phase runner.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use aiperf_runtime::clock::{Clock, sim_clock::SimClock};
use aiperf_runtime::timing::{
    ClockPhaseRunner, GracePeriod, LocalPhaseFuture, PhaseCompletionReason, PhaseConfig,
    PhaseContext, PhaseEventKind, PhaseExecution, PhaseExecutionError, PhaseExecutionFactory,
    PhaseKind, PhaseObserver, PhaseReturn, PhaseRunError, PhaseRunner, PhaseSend, PhaseStats,
    RecordingPhaseObserver, ReleasedStuckSlots, StopConfig,
};
use tokio::task::LocalSet;

#[derive(Clone, Copy)]
enum Behavior {
    ReturnAfter(i64),
    HangAndDrainOnCancel,
    HangForever,
    FailSetup,
}

#[derive(Default)]
struct ExecutionState {
    log: RefCell<Vec<&'static str>>,
    releases: Cell<u64>,
}

struct TestExecutionFactory {
    behavior: Behavior,
    state: Rc<ExecutionState>,
}

impl PhaseExecutionFactory for TestExecutionFactory {
    fn create(&self, _config: &PhaseConfig, context: PhaseContext) -> Rc<dyn PhaseExecution> {
        Rc::new(TestExecution {
            behavior: self.behavior,
            context,
            state: self.state.clone(),
        })
    }
}

struct TestExecution {
    behavior: Behavior,
    context: PhaseContext,
    state: Rc<ExecutionState>,
}

impl PhaseExecution for TestExecution {
    fn configure(&self, _config: &PhaseConfig) -> Result<(), PhaseExecutionError> {
        self.state.log.borrow_mut().push("configure");
        Ok(())
    }

    fn setup(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let state = self.state.clone();
        let fails = matches!(self.behavior, Behavior::FailSetup);
        Box::pin(async move {
            state.log.borrow_mut().push("setup");
            if fails {
                Err(PhaseExecutionError::new("setup failed"))
            } else {
                Ok(())
            }
        })
    }

    fn start_ramps(&self) -> Result<(), PhaseExecutionError> {
        self.state.log.borrow_mut().push("start_ramps");
        Ok(())
    }

    fn execute(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let behavior = self.behavior;
        let context = self.context.clone();
        let state = self.state.clone();
        Box::pin(async move {
            state.log.borrow_mut().push("execute");
            context
                .record_sent(PhaseSend::single_turn_session())
                .map_err(|error| PhaseExecutionError::new(error.to_string()))?;
            match behavior {
                Behavior::ReturnAfter(delay_ns) => {
                    let returned = context.clone();
                    let clock = context.clock();
                    tokio::task::spawn_local(async move {
                        clock.sleep(delay_ns).await;
                        returned.record_first_token();
                        returned.record_returned(PhaseReturn {
                            completes_session: true,
                            ..PhaseReturn::default()
                        });
                    });
                    Ok(())
                }
                Behavior::HangAndDrainOnCancel | Behavior::HangForever => {
                    std::future::pending::<()>().await;
                    Ok(())
                }
                Behavior::FailSetup => unreachable!("setup prevents execute"),
            }
        })
    }

    fn stop_issuing(&self) {
        self.state.log.borrow_mut().push("stop_issuing");
    }

    fn cancel_pending(&self) {
        self.state.log.borrow_mut().push("cancel_pending");
    }

    fn cancel_inflight(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let behavior = self.behavior;
        let context = self.context.clone();
        let state = self.state.clone();
        Box::pin(async move {
            state.log.borrow_mut().push("cancel_inflight");
            if matches!(behavior, Behavior::HangAndDrainOnCancel) {
                context.record_returned(PhaseReturn {
                    completes_session: true,
                    cancelled: true,
                    releases_prefill: true,
                    ..PhaseReturn::default()
                });
            }
            Ok(())
        })
    }

    fn release_stuck_slots(&self) -> ReleasedStuckSlots {
        self.state.log.borrow_mut().push("release_stuck");
        self.state.releases.set(self.state.releases.get() + 1);
        ReleasedStuckSlots {
            session: 1,
            prefill: 1,
        }
    }

    fn stop_ramps(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let state = self.state.clone();
        Box::pin(async move {
            state.log.borrow_mut().push("stop_ramps");
            Ok(())
        })
    }

    fn finalize(&self) -> LocalPhaseFuture<Result<(), PhaseExecutionError>> {
        let state = self.state.clone();
        Box::pin(async move {
            state.log.borrow_mut().push("finalize");
            Ok(())
        })
    }
}

fn runner(
    clock: Rc<SimClock>,
    config: PhaseConfig,
    behavior: Behavior,
) -> (
    ClockPhaseRunner,
    Rc<RecordingPhaseObserver>,
    Rc<ExecutionState>,
) {
    let observer = Rc::new(RecordingPhaseObserver::default());
    let state = Rc::new(ExecutionState::default());
    let factory: Rc<dyn PhaseExecutionFactory> = Rc::new(TestExecutionFactory {
        behavior,
        state: state.clone(),
    });
    let clock_dyn: Rc<dyn Clock> = clock;
    let runner = ClockPhaseRunner::new(config, clock_dyn, observer.clone(), factory).unwrap();
    (runner, observer, state)
}

fn profiling_config(duration_ns: Option<i64>, grace: GracePeriod, drain_ns: i64) -> PhaseConfig {
    PhaseConfig::new(
        "profiling",
        PhaseKind::Profiling,
        StopConfig {
            // Duration cases deliberately leave request count uncapped so the
            // execution future remains active until the runner's sending timer
            // fires. Count-bounded cases close sending on the first request.
            total_expected_requests: duration_ns.is_none().then_some(1),
            expected_num_sessions: None,
            expected_duration_ns: duration_ns,
        },
    )
    .with_grace_period(grace)
    .with_runtime_intervals(10, drain_ns)
}

#[test]
fn happy_path_orders_setup_ramps_sending_and_returns() {
    let clock = Rc::new(SimClock::new());
    let (runner, observer, state) = runner(
        clock.clone(),
        profiling_config(None, GracePeriod::Disabled, 30),
        Behavior::ReturnAfter(25),
    );

    let stats = drive_sim(clock.clone(), runner.run(true)).unwrap();

    assert_eq!(clock.now_ns(), 25);
    assert_eq!(
        stats.completion_reason,
        Some(PhaseCompletionReason::Completed)
    );
    assert_eq!(stats.final_requests_completed, Some(1));
    assert_eq!(stats.in_flight_requests, 0);
    let log = state.log.borrow();
    assert_eq!(&log[..4], &["configure", "setup", "start_ramps", "execute"]);
    assert!(
        log.iter().position(|entry| *entry == "cancel_pending")
            < log.iter().position(|entry| *entry == "stop_ramps")
    );
    assert!(
        log.iter().position(|entry| *entry == "finalize")
            < log.iter().position(|entry| *entry == "stop_ramps")
    );
    let events = observer.events();
    let sending = events
        .iter()
        .position(|event| event.kind == PhaseEventKind::SendingComplete)
        .unwrap();
    let complete = events
        .iter()
        .position(|event| event.kind == PhaseEventKind::Complete)
        .unwrap();
    assert!(sending < complete);
}

#[test]
fn non_final_outbound_seamless_hands_off_before_returns_complete() {
    let clock = Rc::new(SimClock::new());
    let config = profiling_config(None, GracePeriod::Disabled, 30).with_seamless(true);
    let (runner, _observer, _state) = runner(clock.clone(), config, Behavior::ReturnAfter(25));

    let (handoff_at, handoff_stats, complete_at, complete_stats) =
        drive_sim(clock.clone(), async {
            let handoff_stats = runner.run(false).await.unwrap();
            let handoff_at = clock.now_ns();
            let complete_stats = runner.wait_complete().await.unwrap();
            (handoff_at, handoff_stats, clock.now_ns(), complete_stats)
        });

    assert_eq!(handoff_at, 0);
    assert_eq!(handoff_stats.completion_reason, None);
    assert_eq!(handoff_stats.requests_completed, 0);
    assert_eq!(complete_at, 25);
    assert_eq!(
        complete_stats.completion_reason,
        Some(PhaseCompletionReason::Completed)
    );
    assert_eq!(complete_stats.final_requests_completed, Some(1));
}

#[test]
fn final_phase_waits_for_returns_even_with_outbound_seamless_set() {
    let clock = Rc::new(SimClock::new());
    let config = profiling_config(None, GracePeriod::Disabled, 30).with_seamless(true);
    let (runner, _observer, _state) = runner(clock.clone(), config, Behavior::ReturnAfter(25));

    let stats = drive_sim(clock.clone(), runner.run(true)).unwrap();

    assert_eq!(clock.now_ns(), 25);
    assert_eq!(
        stats.completion_reason,
        Some(PhaseCompletionReason::Completed)
    );
    assert_eq!(stats.final_requests_completed, Some(1));
}

#[test]
fn grace_timeout_cancels_and_drains_without_forcing() {
    let clock = Rc::new(SimClock::new());
    let (runner, _observer, state) = runner(
        clock.clone(),
        profiling_config(Some(100), GracePeriod::Finite(20), 30),
        Behavior::HangAndDrainOnCancel,
    );

    let stats = drive_sim(clock.clone(), runner.run(true)).unwrap();

    assert_eq!(clock.now_ns(), 120);
    assert!(stats.timeout_triggered);
    assert!(stats.grace_period_timeout_triggered);
    assert!(!stats.cancel_drain_timeout_triggered);
    assert!(!stats.forced_completion);
    assert_eq!(stats.final_requests_cancelled, Some(1));
    assert_eq!(
        stats.completion_reason,
        Some(PhaseCompletionReason::GraceTimeout)
    );
    assert!(state.log.borrow().contains(&"cancel_inflight"));
}

#[test]
fn drain_timeout_releases_stuck_slots_at_exact_virtual_instant() {
    let clock = Rc::new(SimClock::new());
    let (runner, _observer, state) = runner(
        clock.clone(),
        profiling_config(Some(100), GracePeriod::Finite(20), 30),
        Behavior::HangForever,
    );

    let stats = drive_sim(clock.clone(), runner.run(true)).unwrap();

    assert_eq!(clock.now_ns(), 150);
    assert!(stats.timeout_triggered);
    assert!(stats.grace_period_timeout_triggered);
    assert!(stats.cancel_drain_timeout_triggered);
    assert!(stats.forced_completion);
    assert_eq!(stats.stuck_session_slots_released, 1);
    assert_eq!(stats.stuck_prefill_slots_released, 1);
    assert_eq!(
        stats.completion_reason,
        Some(PhaseCompletionReason::ForceCompleted)
    );
    assert_eq!(state.releases.get(), 1);
}

#[test]
fn external_cancel_during_sending_short_circuits_returns() {
    let clock = Rc::new(SimClock::new());
    let (runner, _observer, _state) = runner(
        clock.clone(),
        profiling_config(Some(100), GracePeriod::Finite(20), 30),
        Behavior::HangForever,
    );
    let cancellation_runner = runner.clone();
    let cancellation_clock = clock.clone();

    let stats = drive_sim(clock.clone(), async move {
        tokio::task::spawn_local(async move {
            cancellation_clock.sleep(10).await;
            cancellation_runner.cancel();
        });
        runner.run(true).await
    })
    .unwrap();

    assert_eq!(clock.now_ns(), 10);
    assert!(stats.was_cancelled);
    assert_eq!(
        stats.completion_reason,
        Some(PhaseCompletionReason::Cancelled)
    );
}

#[test]
fn setup_failure_flushes_terminal_lifecycle_events() {
    let clock = Rc::new(SimClock::new());
    let (runner, observer, _state) = runner(
        clock.clone(),
        profiling_config(None, GracePeriod::Disabled, 30),
        Behavior::FailSetup,
    );

    let error = drive_sim(clock, runner.run(true)).unwrap_err();

    assert!(matches!(error, PhaseRunError::Execution(_)));
    let events = observer.events();
    assert!(
        events
            .iter()
            .any(|event| event.kind == PhaseEventKind::Start)
    );
    assert!(
        events
            .iter()
            .any(|event| event.kind == PhaseEventKind::SendingComplete)
    );
    let complete = events
        .iter()
        .find(|event| event.kind == PhaseEventKind::Complete)
        .unwrap();
    assert_eq!(
        complete.stats.completion_reason,
        Some(PhaseCompletionReason::Failed)
    );
}

#[test]
fn progress_loop_ticks_on_the_injected_clock() {
    let clock = Rc::new(SimClock::new());
    let state = Rc::new(ExecutionState::default());
    let factory: Rc<dyn PhaseExecutionFactory> = Rc::new(TestExecutionFactory {
        behavior: Behavior::ReturnAfter(25),
        state,
    });
    let observer = Rc::new(TimelineObserver {
        clock: clock.clone(),
        progress_at: RefCell::new(Vec::new()),
    });
    let clock_dyn: Rc<dyn Clock> = clock.clone();
    let runner = ClockPhaseRunner::new(
        profiling_config(None, GracePeriod::Disabled, 30),
        clock_dyn,
        observer.clone(),
        factory,
    )
    .unwrap();

    drive_sim(clock, runner.run(true)).unwrap();

    let progress_at = observer.progress_at.borrow();
    assert!(progress_at.contains(&10));
    assert!(progress_at.contains(&20));
    assert!(!progress_at.iter().any(|at| *at > 25));
}

struct TimelineObserver {
    clock: Rc<SimClock>,
    progress_at: RefCell<Vec<i64>>,
}

impl PhaseObserver for TimelineObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, _stats: PhaseStats) {}

    fn on_progress(&self, _stats: PhaseStats) {
        self.progress_at.borrow_mut().push(self.clock.now_ns());
    }

    fn on_sending_complete(&self, _stats: PhaseStats) {}

    fn on_phase_complete(
        &self,
        _stats: PhaseStats,
        _branch_stats: Option<aiperf_runtime::timing::PhaseBranchStats>,
    ) {
    }
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
                None => panic!("phase runner deadlocked with no clock event"),
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
