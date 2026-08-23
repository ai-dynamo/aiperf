// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end phased scheduled-runtime coverage over virtual time.
#![cfg(feature = "engine")]

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use aiperf_runtime::clock::{Clock, sim_clock::SimClock};
use aiperf_runtime::dispatch::collector::ReplayTerminalStatus;
use aiperf_runtime::dispatch::sink::RequestObserver;
use aiperf_runtime::metrics_core::RequestTrace;
use aiperf_runtime::multiturn::{ConversationSource, IssuedCredit, TurnToSend};
use aiperf_runtime::phase_runtime::{
    RampScheduledPhaseController, ScheduledPhaseController, ScheduledPhasePlan,
    ScheduledPhaseResources, SlotPoolPhaseResources, run_scheduled_phases,
};
use aiperf_runtime::scheduled::{
    ScheduledAncillaryPolicies, ScheduledRuntime, SingleTurnDatasetWorkload, TurnDispatchOutcome,
    TurnDispatcher, TurnRecordProcessor, Workload,
};
use aiperf_runtime::timing::{
    GracePeriod, LinearRamp, PhaseBranchStats, PhaseConfig, PhaseKind, PhaseObserver, PhaseStats,
    RampDriver, RamperConfig, SlotPool, StopConfig,
};
use async_trait::async_trait;
use tokio::task::LocalSet;

mod common;

#[derive(Clone, Debug, PartialEq, Eq)]
enum PhaseEvent {
    Start(String, i64),
    Complete(String, i64),
}

struct TimelineObserver {
    clock: Rc<SimClock>,
    events: RefCell<Vec<PhaseEvent>>,
}

impl PhaseObserver for TimelineObserver {
    fn on_phase_start(&self, _config: &PhaseConfig, stats: PhaseStats) {
        self.events
            .borrow_mut()
            .push(PhaseEvent::Start(stats.phase_id, self.clock.now_ns()));
    }

    fn on_progress(&self, _stats: PhaseStats) {}

    fn on_sending_complete(&self, _stats: PhaseStats) {}

    fn on_phase_complete(&self, stats: PhaseStats, _branch_stats: Option<PhaseBranchStats>) {
        self.events
            .borrow_mut()
            .push(PhaseEvent::Complete(stats.phase_id, self.clock.now_ns()));
    }
}

struct DelayedDispatcher {
    clock: Rc<SimClock>,
    dispatched: Cell<usize>,
}

#[async_trait(?Send)]
impl TurnDispatcher for DelayedDispatcher {
    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        let index = self.dispatched.get();
        self.dispatched.set(index + 1);
        let delay_ns = if index == 0 { 20 } else { 5 };
        let start_ns = self.clock.now_ns();
        observer.on_admit(turn.uuid, start_ns as f64 / 1_000_000.0, 0);
        self.clock.clone().sleep(delay_ns).await;
        on_first_token(delay_ns);
        observer.on_token(turn.uuid, self.clock.now_ns() as f64 / 1_000_000.0);
        observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
        Ok(TurnDispatchOutcome {
            start_ns,
            end_ns: self.clock.now_ns(),
            terminal: ReplayTerminalStatus::Completed,
            response_text: "ok".into(),
            model_response: aiperf_runtime::scheduled::ModelResponseMetadata::default(),
            prompt_tokens: Some(turn.input_length as u64),
            completion_tokens: Some(1),
            http: RequestTrace::default(),
        })
    }
}

#[test]
fn scheduled_runtime_uses_real_phase_handoff_and_finalizes_reports_after_returns() {
    let clock = Rc::new(SimClock::new());
    let observer = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events: RefCell::new(Vec::new()),
    });
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(DelayedDispatcher {
        clock: clock.clone(),
        dispatched: Cell::new(0),
    });
    let clock_dyn: Rc<dyn Clock> = clock.clone();
    let phase_observer: Rc<dyn PhaseObserver> = observer.clone();
    let plans = vec![
        ScheduledPhasePlan::new(
            phase_config("warmup", PhaseKind::Warmup, true),
            one_request_workload(),
            ScheduledAncillaryPolicies::default(),
        ),
        ScheduledPhasePlan::new(
            phase_config("profiling", PhaseKind::Profiling, false),
            one_request_workload(),
            ScheduledAncillaryPolicies::default(),
        ),
    ];

    let report = drive_sim(clock.clone(), async move {
        run_scheduled_phases(plans, clock_dyn, dispatcher, phase_observer).await
    })
    .unwrap();

    assert_eq!(clock.now_ns(), 20);
    assert_eq!(report.phases.len(), 2);
    assert_eq!(report.phases[0].final_requests_completed, Some(1));
    assert_eq!(report.phases[1].final_requests_completed, Some(1));
    assert_eq!(report.reports.len(), 2);
    assert_eq!(report.reports[0].phase_id, "warmup");
    assert_eq!(report.reports[1].phase_id, "profiling");
    assert_eq!(
        report.reports[0]
            .report
            .performance
            .request_counts
            .completed_requests,
        1
    );
    assert_eq!(
        report.reports[1]
            .report
            .performance
            .request_counts
            .completed_requests,
        1
    );
    let events = observer.events.borrow();
    assert!(events.contains(&PhaseEvent::Start("warmup".into(), 0)));
    assert!(events.contains(&PhaseEvent::Start("profiling".into(), 0)));
    assert!(events.contains(&PhaseEvent::Complete("profiling".into(), 5)));
    assert!(events.contains(&PhaseEvent::Complete("warmup".into(), 20)));
}

#[test]
fn production_adapter_debt_drains_shared_capacity_during_seamless_handoff() {
    let clock = Rc::new(SimClock::new());
    let starts = Rc::new(RefCell::new(Vec::new()));
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(DebtDispatcher {
        clock: clock.clone(),
        starts: starts.clone(),
        dispatched: Cell::new(0),
    });
    let observer = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events: RefCell::new(Vec::new()),
    });
    let pool = Rc::new(SlotPool::new(0));
    let resources: Rc<dyn ScheduledPhaseResources> =
        Rc::new(SlotPoolPhaseResources::new(Some(pool.clone()), None));
    let plans = vec![
        ScheduledPhasePlan::new(
            phase_config_with_count("warmup", PhaseKind::Warmup, true, 4)
                .with_concurrency(Some(4), None),
            shared_slot_workload(4, pool.clone()),
            ScheduledAncillaryPolicies::default(),
        )
        .with_resources(resources.clone()),
        ScheduledPhasePlan::new(
            phase_config("profiling", PhaseKind::Profiling, false).with_concurrency(Some(3), None),
            shared_slot_workload(1, pool.clone()),
            ScheduledAncillaryPolicies::default(),
        )
        .with_resources(resources),
    ];
    let clock_dyn: Rc<dyn Clock> = clock.clone();
    let phase_observer: Rc<dyn PhaseObserver> = observer.clone();

    let report = drive_sim(clock.clone(), async move {
        run_scheduled_phases(plans, clock_dyn, dispatcher, phase_observer).await
    })
    .unwrap();

    assert_eq!(report.phases[0].final_requests_completed, Some(4));
    assert_eq!(report.phases[1].final_requests_completed, Some(1));
    assert_eq!(starts.borrow().as_slice(), &[0, 0, 0, 0, 5]);
    assert_eq!(pool.current_limit(), 3);
    assert_eq!(pool.debt(), 0);
    assert_eq!(clock.now_ns(), 20);
    let events = observer.events.borrow();
    assert!(events.contains(&PhaseEvent::Start("profiling".into(), 0)));
    assert!(events.contains(&PhaseEvent::Complete("warmup".into(), 20)));
}

#[test]
fn phased_api_joins_terminal_processors_after_the_phase_window_closes() {
    let clock = Rc::new(SimClock::new());
    let observer = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events: RefCell::new(Vec::new()),
    });
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(DelayedDispatcher {
        clock: clock.clone(),
        dispatched: Cell::new(1),
    });
    let processor = Rc::new(DelayedProcessor {
        clock: clock.clone(),
        completed_at: Cell::new(None),
    });
    let processors: Vec<Rc<dyn TurnRecordProcessor>> = vec![processor.clone()];
    let plans = vec![
        ScheduledPhasePlan::new(
            phase_config("profiling", PhaseKind::Profiling, false),
            one_request_workload(),
            ScheduledAncillaryPolicies::default(),
        )
        .with_record_processors(processors),
    ];
    let clock_dyn: Rc<dyn Clock> = clock.clone();
    let phase_observer: Rc<dyn PhaseObserver> = observer.clone();

    let report = drive_sim(clock.clone(), async move {
        run_scheduled_phases(plans, clock_dyn, dispatcher, phase_observer).await
    })
    .unwrap();

    assert_eq!(report.phases[0].requests_end_ns, Some(5));
    assert_eq!(processor.completed_at.get(), Some(12));
    assert_eq!(clock.now_ns(), 12);
    assert!(
        observer
            .events
            .borrow()
            .contains(&PhaseEvent::Complete("profiling".into(), 5))
    );
}

#[test]
fn prepared_ramps_apply_before_issuance_and_stop_at_sending_handoff() {
    let clock = Rc::new(SimClock::new());
    let pool = Rc::new(SlotPool::new(0));
    let pool_for_driver = pool.clone();
    let clock_dyn: Rc<dyn Clock> = clock.clone();
    let driver = RampDriver::new(
        clock_dyn.clone(),
        Box::new(LinearRamp::new(RamperConfig::new(1.0, 4.0, 100).unwrap())),
        move |value| pool_for_driver.set_limit(value as usize),
    );
    let controller: Rc<dyn ScheduledPhaseController> =
        Rc::new(RampScheduledPhaseController::new(vec![driver]));
    let resources: Rc<dyn ScheduledPhaseResources> =
        Rc::new(SlotPoolPhaseResources::new(Some(pool.clone()), None));
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(DelayedDispatcher {
        clock: clock.clone(),
        dispatched: Cell::new(0),
    });
    let observer: Rc<dyn PhaseObserver> = Rc::new(TimelineObserver {
        clock: clock.clone(),
        events: RefCell::new(Vec::new()),
    });
    let plan = ScheduledPhasePlan::new(
        phase_config("profiling", PhaseKind::Profiling, false).with_concurrency(Some(4), None),
        shared_slot_workload(1, pool.clone()),
        ScheduledAncillaryPolicies::default(),
    )
    .with_resources(resources)
    .with_controller(controller);

    let report = drive_sim(clock.clone(), async move {
        run_scheduled_phases(vec![plan], clock_dyn, dispatcher, observer).await
    })
    .unwrap();

    assert_eq!(report.phases[0].final_requests_completed, Some(1));
    assert_eq!(clock.now_ns(), 20);
    assert_eq!(pool.current_limit(), 1);
}

fn one_request_workload() -> Rc<dyn Workload> {
    let source = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(common::synthetic_prepared_source(1, 4, 1, None, "model"));
    Rc::new(SingleTurnDatasetWorkload::new(source, 1).unwrap())
}

struct SharedSlotWorkload {
    conversations: RefCell<Box<dyn ConversationSource>>,
    count: usize,
    slots: Rc<SlotPool>,
}

#[async_trait(?Send)]
impl Workload for SharedSlotWorkload {
    fn name(&self) -> &'static str {
        "shared_slot_test"
    }

    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> anyhow::Result<()> {
        for _ in 0..self.count {
            let guard = self.slots.acquire().await;
            let turn = self
                .conversations
                .borrow_mut()
                .next(None)?
                .build_first_turn(Some(1))?;
            if !runtime.issue_turn(
                turn,
                runtime.now_ns(),
                None,
                Box::new(move |_credit, _outcome| Box::pin(async move { drop(guard) })),
            ) {
                break;
            }
        }
        Ok(())
    }
}

fn shared_slot_workload(count: usize, slots: Rc<SlotPool>) -> Rc<dyn Workload> {
    let source = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(common::synthetic_prepared_source(1, 4, 1, None, "model"));
    Rc::new(SharedSlotWorkload {
        conversations: RefCell::new(source),
        count,
        slots,
    })
}

struct DebtDispatcher {
    clock: Rc<SimClock>,
    starts: Rc<RefCell<Vec<i64>>>,
    dispatched: Cell<usize>,
}

#[async_trait(?Send)]
impl TurnDispatcher for DebtDispatcher {
    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> anyhow::Result<TurnDispatchOutcome> {
        let index = self.dispatched.get();
        self.dispatched.set(index + 1);
        let start_ns = self.clock.now_ns();
        self.starts.borrow_mut().push(start_ns);
        let delay_ns = if index == 0 { 20 } else { 5 };
        observer.on_admit(turn.uuid, start_ns as f64 / 1_000_000.0, 0);
        self.clock.clone().sleep(delay_ns).await;
        on_first_token(delay_ns);
        observer.on_token(turn.uuid, self.clock.now_ns() as f64 / 1_000_000.0);
        observer.on_terminal(turn.uuid, ReplayTerminalStatus::Completed);
        Ok(TurnDispatchOutcome {
            start_ns,
            end_ns: self.clock.now_ns(),
            terminal: ReplayTerminalStatus::Completed,
            response_text: "ok".into(),
            model_response: aiperf_runtime::scheduled::ModelResponseMetadata::default(),
            prompt_tokens: Some(turn.input_length as u64),
            completion_tokens: Some(1),
            http: RequestTrace::default(),
        })
    }
}

struct DelayedProcessor {
    clock: Rc<SimClock>,
    completed_at: Cell<Option<i64>>,
}

#[async_trait(?Send)]
impl TurnRecordProcessor for DelayedProcessor {
    async fn process(
        &self,
        _credit: &IssuedCredit,
        _outcome: &TurnDispatchOutcome,
    ) -> anyhow::Result<()> {
        self.clock.clone().sleep(7).await;
        self.completed_at.set(Some(self.clock.now_ns()));
        Ok(())
    }
}

fn phase_config(id: &str, kind: PhaseKind, seamless: bool) -> PhaseConfig {
    phase_config_with_count(id, kind, seamless, 1)
}

fn phase_config_with_count(id: &str, kind: PhaseKind, seamless: bool, count: u64) -> PhaseConfig {
    PhaseConfig::new(
        id,
        kind,
        StopConfig {
            total_expected_requests: Some(count),
            ..StopConfig::default()
        },
    )
    .with_grace_period(if kind == PhaseKind::Warmup {
        GracePeriod::Infinite
    } else {
        GracePeriod::Disabled
    })
    .with_seamless(seamless)
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
                None => panic!("phased scheduled runtime deadlocked"),
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
