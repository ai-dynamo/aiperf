// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Async task runtime for the dataflow executor, on **tokio**.
//!
//! The dataflow is genuinely async: nodes are `spawn_local` tasks on a tokio
//! `current_thread` runtime + `LocalSet`. Each trace runs single-threaded, so
//! its state stays `Rc`/`RefCell` with no locks on the hot path; futures are
//! `!Send`, and parallelism comes from running many traces across threads.
//!
//! Time is **not** `tokio::time` — that timer wheel is 1 ms-granular and would
//! destroy the µs/ns firing gates. Instead virtual time is the ns-exact
//! [`SimClock`] (ai-dynamo dynosim's discrete-event pattern), advanced by an
//! idle-pump: [`drive_sim`] polls the `LocalSet` to quiescence (draining all
//! same-instant work), then fast-forwards the clock to the next scheduled event
//! (waking heap-ordered sleepers), and repeats — the offline-DES driver loop,
//! with tokio owning task scheduling.

use aiperf_clock::clock::Clock;
use aiperf_clock::sim_clock::SimClock;
use std::cell::Cell;
use std::fmt::{self, Display};
use std::future::Future;
use std::pin::{Pin, pin};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::task::{Context, Poll, Wake, Waker};
use tokio::sync::Notify;
use tokio::task::LocalSet;

/// Shared handle threaded into task futures: spawn, clock access, sleeping.
///
/// Clock-agnostic: `Rc<dyn Clock>` is either the virtual [`SimClock`] or the
/// real [`RealClock`](aiperf_clock::real_clock::RealClock), so the executor is
/// identical in both modes.
#[derive(Clone)]
pub struct Handle {
    clock: Rc<dyn Clock>,
    inflight: Rc<Cell<usize>>,
    done: Rc<Notify>,
}

impl Handle {
    /// Create a handle bound to `clock`. Each trace gets its own handle (with
    /// its own in-flight counter), so many traces can share one worker's runtime
    /// and clock — the basis of the thread-per-core parallel engine.
    pub fn new(clock: Rc<dyn Clock>) -> Self {
        Handle {
            clock,
            inflight: Rc::new(Cell::new(0)),
            done: Rc::new(Notify::new()),
        }
    }

    /// Current time in nanoseconds (virtual for sim, monotonic for real).
    pub fn now_ns(&self) -> i64 {
        self.clock.now_ns()
    }

    /// Spawn a task onto the tokio `LocalSet`. An in-flight counter tracks
    /// liveness so [`drive_sim`] knows when the whole trace has drained.
    pub fn spawn<F>(&self, fut: F)
    where
        F: Future<Output = ()> + 'static,
    {
        self.inflight.set(self.inflight.get() + 1);
        let inflight = self.inflight.clone();
        let done = self.done.clone();
        tokio::task::spawn_local(async move {
            fut.await;
            let remaining = inflight.get() - 1;
            inflight.set(remaining);
            if remaining == 0 {
                done.notify_one();
            }
        });
    }

    /// Await until every spawned task has completed.
    pub async fn wait_idle(&self) {
        if self.inflight.get() == 0 {
            return;
        }
        self.done.notified().await;
    }

    /// Sleep for `duration_ns` of this clock's time. Non-positive durations
    /// yield once to the runtime instead of parking.
    pub fn sleep_ns(&self, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
        self.clock.clone().sleep(duration_ns)
    }
}

/// The result of driving a graph to quiescence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOutcome {
    /// Tasks remained parked with no future clock event to wake them.
    pub deadlocked: bool,
}

/// One externally clocked discrete-event source consumed by the virtual-time
/// pump. Implementations own their event queue and route any events produced by
/// [`step`](SimEventSource::step) to the futures they wake.
///
/// This is deliberately engine-neutral: a mock inference engine is one
/// implementation, while a future telemetry or network simulator can implement
/// the same boundary without adding a backend dependency to `aiperf-graph`.
pub trait SimEventSource {
    /// Earliest virtual nanosecond at which this source can make progress.
    fn next_event_ns(&self) -> Result<Option<i64>, SimDriveError>;

    /// Synchronize an idle source to a caller-selected clock time.
    fn set_time_ns(&self, now_ns: i64) -> Result<(), SimDriveError>;

    /// Process the source event due at `now_ns` and route resulting wakeups.
    fn step(&self, now_ns: i64) -> Result<SimStep, SimDriveError>;

    /// True when the source has no pending or in-flight work.
    fn is_idle(&self) -> bool;
}

/// Outcome of one external discrete-event step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimStep {
    /// Source time after the step. It must not precede the caller's `now_ns`.
    pub end_ns: i64,
    /// Whether the step changed source state or emitted an event.
    pub made_progress: bool,
}

/// Deterministic virtual-time pump failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimDriveError {
    /// The external source rejected an operation.
    EventSource(String),
    /// An event source returned a timestamp behind the shared clock.
    TimeRegression {
        /// Shared-clock time before the operation.
        now_ns: i64,
        /// Invalid timestamp returned by the source.
        event_ns: i64,
    },
    /// A source step crossed a parked clock deadline. Crossing it would delay
    /// an arrival or firing gate and silently change batch composition.
    OvershotClockEvent {
        /// Source time before the step.
        started_ns: i64,
        /// Source time returned by the step.
        ended_ns: i64,
        /// Clock deadline that had to run first.
        clock_event_ns: i64,
    },
    /// Repeated same-instant steps made no observable progress.
    Stalled {
        /// Virtual time at which the source stalled.
        at_ns: i64,
    },
}

impl Display for SimDriveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EventSource(message) => write!(f, "external event source failed: {message}"),
            Self::TimeRegression { now_ns, event_ns } => write!(
                f,
                "external event time regressed from shared clock {now_ns}ns to {event_ns}ns"
            ),
            Self::OvershotClockEvent {
                started_ns,
                ended_ns,
                clock_event_ns,
            } => write!(
                f,
                "external step {started_ns}ns..{ended_ns}ns crossed clock event at {clock_event_ns}ns"
            ),
            Self::Stalled { at_ns } => {
                write!(f, "external event source stalled at {at_ns}ns")
            }
        }
    }
}

impl std::error::Error for SimDriveError {}

/// Run `body` on a single-threaded tokio runtime + `LocalSet`, advancing the
/// ns-exact [`SimClock`] between same-instant rounds (the offline-DES idle-pump).
/// Returns once `body` resolves (the trace drained), or once the `LocalSet` is
/// idle with no clock event left to advance (deadlock).
///
/// This is the **virtual-time** driver. The real/wall clock drives via tokio's
/// reactor instead — see [`drive_real`].
pub fn drive_sim<F>(clock: Rc<SimClock>, make_body: impl FnOnce(Handle) -> F) -> RunOutcome
where
    F: Future<Output = ()>,
{
    drive_sim_inner(clock, None, make_body)
        .expect("the clock-only virtual driver has no fallible event source")
}

/// Run `body` against a [`SimClock`] and one passive external event source.
///
/// Ready clock tasks are always polled to quiescence before an equal-time
/// external event. This preserves the arrival-before-pass ordering required by
/// inference batching. A source step is rejected if it jumps across an already
/// parked clock event; accepting that jump would make request-rate, fixed
/// schedules, and firing gates backend-dependent.
pub fn drive_sim_with_source<F>(
    clock: Rc<SimClock>,
    source: Rc<dyn SimEventSource>,
    make_body: impl FnOnce(Handle) -> F,
) -> Result<RunOutcome, SimDriveError>
where
    F: Future<Output = ()>,
{
    drive_sim_inner(clock, Some(source), make_body)
}

fn drive_sim_inner<F>(
    clock: Rc<SimClock>,
    source: Option<Rc<dyn SimEventSource>>,
    make_body: impl FnOnce(Handle) -> F,
) -> Result<RunOutcome, SimDriveError>
where
    F: Future<Output = ()>,
{
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current_thread runtime");
    let _guard = rt.enter();
    let handle = Handle::new(clock.clone());
    let local = LocalSet::new();
    let body = make_body(handle);
    let fut = local.run_until(body);
    let mut fut = pin!(fut);

    let flag = Arc::new(AtomicBool::new(true));
    let waker = Waker::from(Arc::new(FlagWaker(flag.clone())));
    let mut cx = Context::from_waker(&waker);
    let mut no_progress_steps = 0_u32;

    loop {
        flag.store(false, Ordering::SeqCst);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(()) => {
                if let Some(source) = source.as_ref() {
                    drain_source(clock.as_ref(), source.as_ref())?;
                }
                return Ok(RunOutcome { deadlocked: false });
            }
            Poll::Pending => {
                // A wake during this poll (yield_now, sibling wake, or the
                // drain-complete Notify) means more same-instant work is ready:
                // re-poll without advancing virtual time.
                if flag.load(Ordering::SeqCst) {
                    continue;
                }
                // Genuinely idle: choose the earliest clock or external event.
                // Clock wins ties so arrivals/gates due at `t` are submitted
                // before an engine pass at the same `t` observes its batch.
                let clock_event = clock.next_event_time();
                let source_event = source
                    .as_ref()
                    .map(|source| source.next_event_ns())
                    .transpose()?
                    .flatten();
                validate_event_time(clock.now_ns(), source_event)?;

                if let Some(at_ns) = clock_event
                    && source_event.is_none_or(|source_ns| at_ns <= source_ns)
                {
                    clock.advance_to(at_ns);
                    if let Some(source) = source.as_ref() {
                        source.set_time_ns(at_ns)?;
                    }
                    no_progress_steps = 0;
                    continue;
                }

                let Some(at_ns) = source_event else {
                    return Ok(RunOutcome { deadlocked: true });
                };
                if at_ns > clock.now_ns() {
                    clock.advance_to(at_ns);
                }
                let step = source
                    .as_ref()
                    .expect("a source event requires a source")
                    .step(at_ns)?;
                if step.end_ns < at_ns {
                    return Err(SimDriveError::TimeRegression {
                        now_ns: at_ns,
                        event_ns: step.end_ns,
                    });
                }
                if let Some(clock_event_ns) = clock.next_event_time()
                    && clock_event_ns < step.end_ns
                {
                    return Err(SimDriveError::OvershotClockEvent {
                        started_ns: at_ns,
                        ended_ns: step.end_ns,
                        clock_event_ns,
                    });
                }
                if step.end_ns > clock.now_ns() {
                    clock.advance_to(step.end_ns);
                }
                if step.made_progress || step.end_ns > at_ns {
                    no_progress_steps = 0;
                } else {
                    no_progress_steps += 1;
                    if no_progress_steps >= 100_000 {
                        return Err(SimDriveError::Stalled { at_ns });
                    }
                }
            }
        }
    }
}

fn validate_event_time(now_ns: i64, event_ns: Option<i64>) -> Result<(), SimDriveError> {
    if let Some(event_ns) = event_ns
        && event_ns < now_ns
    {
        return Err(SimDriveError::TimeRegression { now_ns, event_ns });
    }
    Ok(())
}

fn drain_source(clock: &SimClock, source: &dyn SimEventSource) -> Result<(), SimDriveError> {
    let mut no_progress_steps = 0_u32;
    while !source.is_idle() {
        let Some(at_ns) = source.next_event_ns()? else {
            return Err(SimDriveError::EventSource(
                "source is non-idle but exposes no next event".to_string(),
            ));
        };
        validate_event_time(clock.now_ns(), Some(at_ns))?;
        if at_ns > clock.now_ns() {
            clock.advance_to(at_ns);
        }
        let step = source.step(at_ns)?;
        if step.end_ns < at_ns {
            return Err(SimDriveError::TimeRegression {
                now_ns: at_ns,
                event_ns: step.end_ns,
            });
        }
        if step.end_ns > clock.now_ns() {
            clock.advance_to(step.end_ns);
        }
        if step.made_progress || step.end_ns > at_ns {
            no_progress_steps = 0;
        } else {
            no_progress_steps += 1;
            if no_progress_steps >= 100_000 {
                return Err(SimDriveError::Stalled { at_ns });
            }
        }
    }
    Ok(())
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

/// Drive `body` on a single-threaded tokio runtime + `LocalSet` using **real**
/// time (the reactor drives `timerfd`/IO wakeups). For the live
/// [`RealClock`](aiperf_clock::real_clock::RealClock) backend. `enable_all`
/// turns on the IO + time drivers.
pub fn drive_real<F>(make_body: impl FnOnce(Handle) -> F) -> RunOutcome
where
    F: Future<Output = ()>,
{
    let clock: Rc<dyn Clock> = aiperf_clock::real_clock::RealClock::new();
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("current_thread runtime");
    let local = LocalSet::new();
    let handle = Handle::new(clock);
    local.block_on(&rt, make_body(handle));
    RunOutcome { deadlocked: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct RecordingSource {
        next_ns: Cell<Option<i64>>,
        step_end_ns: i64,
        log: Rc<RefCell<Vec<&'static str>>>,
    }

    impl SimEventSource for RecordingSource {
        fn next_event_ns(&self) -> Result<Option<i64>, SimDriveError> {
            Ok(self.next_ns.get())
        }

        fn set_time_ns(&self, _now_ns: i64) -> Result<(), SimDriveError> {
            Ok(())
        }

        fn step(&self, _now_ns: i64) -> Result<SimStep, SimDriveError> {
            self.log.borrow_mut().push("source");
            self.next_ns.set(None);
            Ok(SimStep {
                end_ns: self.step_end_ns,
                made_progress: true,
            })
        }

        fn is_idle(&self) -> bool {
            self.next_ns.get().is_none()
        }
    }

    #[test]
    fn tasks_run_in_spawn_order_same_instant() {
        let clock = Rc::new(SimClock::new());
        let log = Rc::new(RefCell::new(Vec::<u32>::new()));
        let log2 = log.clone();
        let outcome = drive_sim(clock, move |handle| async move {
            for i in 0..3u32 {
                let log = log2.clone();
                handle.spawn(async move {
                    log.borrow_mut().push(i);
                });
            }
            handle.wait_idle().await;
        });
        assert!(!outcome.deadlocked);
        assert_eq!(*log.borrow(), vec![0, 1, 2]);
    }

    #[test]
    fn sleep_orders_wakes_by_virtual_time() {
        let clock = Rc::new(SimClock::new());
        let log = Rc::new(RefCell::new(Vec::<u32>::new()));
        let log2 = log.clone();
        drive_sim(clock, move |handle| async move {
            {
                let (h, log) = (handle.clone(), log2.clone());
                handle.spawn(async move {
                    h.sleep_ns(200).await;
                    log.borrow_mut().push(200);
                });
            }
            {
                let (h, log) = (handle.clone(), log2.clone());
                handle.spawn(async move {
                    h.sleep_ns(100).await;
                    log.borrow_mut().push(100);
                });
            }
            handle.wait_idle().await;
        });
        assert_eq!(*log.borrow(), vec![100, 200]);
    }

    #[test]
    fn clock_task_wins_an_equal_time_tie_with_external_source() {
        let clock = Rc::new(SimClock::new());
        let log = Rc::new(RefCell::new(Vec::new()));
        let source: Rc<dyn SimEventSource> = Rc::new(RecordingSource {
            next_ns: Cell::new(Some(100)),
            step_end_ns: 100,
            log: log.clone(),
        });
        let body_clock: Rc<dyn Clock> = clock.clone();
        let body_log = log.clone();

        let outcome = drive_sim_with_source(clock, source, move |_handle| async move {
            body_clock.sleep(100).await;
            body_log.borrow_mut().push("clock");
        })
        .unwrap();

        assert!(!outcome.deadlocked);
        assert_eq!(*log.borrow(), vec!["clock", "source"]);
    }

    #[test]
    fn rejects_external_step_that_crosses_a_clock_deadline() {
        let clock = Rc::new(SimClock::new());
        let source: Rc<dyn SimEventSource> = Rc::new(RecordingSource {
            next_ns: Cell::new(Some(10)),
            step_end_ns: 30,
            log: Rc::new(RefCell::new(Vec::new())),
        });
        let body_clock: Rc<dyn Clock> = clock.clone();

        let error = drive_sim_with_source(clock, source, move |_handle| async move {
            body_clock.sleep(20).await;
            std::future::pending::<()>().await;
        })
        .unwrap_err();

        assert_eq!(
            error,
            SimDriveError::OvershotClockEvent {
                started_ns: 10,
                ended_ns: 30,
                clock_event_ns: 20,
            }
        );
    }
}
