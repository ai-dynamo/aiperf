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

    loop {
        flag.store(false, Ordering::SeqCst);
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(()) => return RunOutcome { deadlocked: false },
            Poll::Pending => {
                // A wake during this poll (yield_now, sibling wake, or the
                // drain-complete Notify) means more same-instant work is ready:
                // re-poll without advancing virtual time.
                if flag.load(Ordering::SeqCst) {
                    continue;
                }
                // Genuinely idle: every task is parked on the clock. Fast-forward
                // to the next scheduled event, which wakes heap-ordered sleepers.
                match clock.next_event_time() {
                    Some(t) => clock.advance_to(t),
                    None => return RunOutcome { deadlocked: true },
                }
            }
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
#[allow(unused_imports)]
use std::cell::RefCell;

#[cfg(test)]
mod tests {
    use super::*;

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
}
