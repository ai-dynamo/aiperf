// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end phased scheduled-runtime coverage over virtual time.

use std::cell::{Cell, RefCell};
use std::future::Future;
use std::pin::pin;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};

use aiperf::multiturn::{ConversationSource, SyntheticConversationSource, TurnToSend};
use aiperf::phase_runtime::{ScheduledPhasePlan, run_scheduled_phases};
use aiperf::scheduled::{
    ScheduledAncillaryPolicies, SingleTurnDatasetWorkload, TurnDispatchOutcome, TurnDispatcher,
    Workload,
};
use aiperf::workload::SkeletonWorkload;
use aiperf_clock::{Clock, sim_clock::SimClock};
use aiperf_metrics::HttpTrace;
use aiperf_timing::{
    GracePeriod, PhaseBranchStats, PhaseConfig, PhaseKind, PhaseObserver, PhaseStats, StopConfig,
};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use tokio::task::LocalSet;

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
            prompt_tokens: Some(turn.input_length as u64),
            completion_tokens: Some(1),
            http: HttpTrace::default(),
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

fn one_request_workload() -> Rc<dyn Workload> {
    let source: Box<dyn ConversationSource> = Box::new(
        SyntheticConversationSource::new(SkeletonWorkload {
            num_requests: 1,
            input_tokens: 4,
            output_tokens: 1,
            turns: 1,
            think_time_ms: None,
        })
        .unwrap(),
    );
    Rc::new(SingleTurnDatasetWorkload::new(source, 1).unwrap())
}

fn phase_config(id: &str, kind: PhaseKind, seamless: bool) -> PhaseConfig {
    PhaseConfig::new(
        id,
        kind,
        StopConfig {
            total_expected_requests: Some(1),
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
