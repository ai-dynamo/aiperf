// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Integer-nanosecond discrete-event clock.
//!
//! Events are ordered by `(at_ns, seq_no)`, making same-time wakes deterministic
//! in registration order.

use crate::clock::clock::Clock;
use std::cell::{Cell, RefCell};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;
use std::task::{Context, Poll, Waker};

/// One parked sleeper: fire `waker` once virtual time reaches `at_ns`.
struct Sleeper {
    at_ns: i64,
    seq_no: u64,
    waker: Waker,
}

impl PartialEq for Sleeper {
    fn eq(&self, other: &Self) -> bool {
        self.at_ns == other.at_ns && self.seq_no == other.seq_no
    }
}
impl Eq for Sleeper {}

impl Ord for Sleeper {
    /// Reverse ordering makes the max-heap yield the earliest deadline and
    /// registration sequence first.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .at_ns
            .cmp(&self.at_ns)
            .then_with(|| other.seq_no.cmp(&self.seq_no))
    }
}
impl PartialOrd for Sleeper {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Virtual-time clock advanced by the runtime driver pump.
pub struct SimClock {
    now_ns: Cell<i64>,
    seq: Cell<u64>,
    heap: RefCell<BinaryHeap<Sleeper>>,
}

impl Default for SimClock {
    fn default() -> Self {
        Self::new()
    }
}

impl SimClock {
    pub fn new() -> Self {
        SimClock {
            now_ns: Cell::new(0),
            seq: Cell::new(0),
            heap: RefCell::new(BinaryHeap::new()),
        }
    }

    /// Current virtual time in nanoseconds.
    pub fn now_ns(&self) -> i64 {
        self.now_ns.get()
    }

    /// Register `waker` to fire when virtual time reaches `at_ns`.
    ///
    /// Equal deadlines wake in registration order.
    pub fn schedule(&self, at_ns: i64, waker: Waker) {
        let seq_no = self.seq.get();
        self.seq.set(seq_no + 1);
        self.heap.borrow_mut().push(Sleeper {
            at_ns,
            seq_no,
            waker,
        });
    }

    /// The deadline to fast-forward to — `max(earliest parked deadline, now)`, so
    /// an already-due top sleeper yields `now` rather than a past time — or
    /// `None` when idle.
    ///
    /// The driver pump reads this to know how far to fast-forward.
    pub fn next_event_time(&self) -> Option<i64> {
        let heap = self.heap.borrow();
        heap.peek().map(|s| {
            if s.at_ns > self.now_ns.get() {
                s.at_ns
            } else {
                self.now_ns.get()
            }
        })
    }

    /// Advance virtual time to `ns` (time does not move when `ns <= now`), waking
    /// every sleeper whose deadline is `<= ns` in `(deadline, seq_no)` heap order.
    ///
    pub fn advance_to(&self, ns: i64) {
        if ns <= self.now_ns.get() {
            // Still drain any already-due sleepers (deadline <= now) so a
            // zero-advance tick makes progress.
            self.drain_due(self.now_ns.get());
            return;
        }
        self.now_ns.set(ns);
        self.drain_due(ns);
    }

    fn drain_due(&self, ns: i64) {
        // Collect crossed wakers under the borrow, then wake outside it so a
        // wake that re-schedules cannot re-borrow the heap mid-iteration.
        let mut fired: Vec<Waker> = Vec::new();
        {
            let mut heap = self.heap.borrow_mut();
            while let Some(top) = heap.peek() {
                if top.at_ns <= ns {
                    fired.push(heap.pop().unwrap().waker);
                } else {
                    break;
                }
            }
        }
        for waker in fired {
            waker.wake();
        }
    }

    /// True when at least one sleeper is parked.
    pub fn has_sleepers(&self) -> bool {
        !self.heap.borrow().is_empty()
    }

    /// Total sleepers registered since construction.
    ///
    /// Monotonic and never reset, so the driver pump can use it as a progress
    /// signal: a body that parks on the clock bumps this even when the deadline
    /// is already due and virtual time does not move. A body that only
    /// self-wakes (`yield_now`, or a non-positive [`Clock::sleep`]) leaves both
    /// this and [`now_ns`](Self::now_ns) unchanged.
    pub fn scheduled_count(&self) -> u64 {
        self.seq.get()
    }
}

impl Clock for SimClock {
    fn now_ns(&self) -> i64 {
        self.now_ns.get()
    }

    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
        if duration_ns <= 0 {
            Box::pin(SimSleep::Yield { yielded: false })
        } else {
            let deadline = self.now_ns.get() + duration_ns;
            Box::pin(SimSleep::Until {
                clock: self,
                deadline,
                scheduled: false,
            })
        }
    }

    fn is_virtual(&self) -> bool {
        true
    }

    fn drive(
        self: Rc<Self>,
        body: Pin<Box<dyn Future<Output = ()> + '_>>,
    ) -> crate::graph::runtime::RunOutcome {
        // The idle-pump advances virtual time to each next event, so the body's
        // `Clock::sleep`s resolve instantly in wall time. `body` is already built
        // against this clock; the driver's own `Handle` clock is unused.
        crate::graph::runtime::drive_sim(self, move |_handle| body)
    }
}

/// Sleep future parked on the ns-exact virtual clock (woken by `advance_to`).
enum SimSleep {
    Yield {
        yielded: bool,
    },
    Until {
        clock: Rc<SimClock>,
        deadline: i64,
        scheduled: bool,
    },
}

impl Future for SimSleep {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        match self.get_mut() {
            SimSleep::Yield { yielded } => {
                if *yielded {
                    Poll::Ready(())
                } else {
                    *yielded = true;
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
            }
            SimSleep::Until {
                clock,
                deadline,
                scheduled,
            } => {
                if clock.now_ns.get() >= *deadline {
                    Poll::Ready(())
                } else {
                    if !*scheduled {
                        *scheduled = true;
                        clock.schedule(*deadline, cx.waker().clone());
                    }
                    Poll::Pending
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;
    use std::sync::{Arc, Mutex};
    use std::task::Wake;

    struct RecordingWaker {
        label: u64,
        log: Arc<Mutex<Vec<u64>>>,
    }
    impl Wake for RecordingWaker {
        fn wake(self: Arc<Self>) {
            self.log.lock().unwrap().push(self.label);
        }
    }
    fn waker(label: u64, log: &Arc<Mutex<Vec<u64>>>) -> Waker {
        Waker::from(Arc::new(RecordingWaker {
            label,
            log: log.clone(),
        }))
    }

    #[test]
    fn advances_and_wakes_in_deadline_then_seq_order() {
        let clock = Rc::new(SimClock::new());
        let log = Arc::new(Mutex::new(Vec::new()));
        // Schedule out of order; same deadline for 10/11 must wake in seq order.
        clock.schedule(200, waker(2, &log));
        clock.schedule(100, waker(10, &log));
        clock.schedule(100, waker(11, &log));

        assert_eq!(clock.next_event_time(), Some(100));
        clock.advance_to(100);
        assert_eq!(*log.lock().unwrap(), vec![10, 11]);
        assert_eq!(clock.now_ns(), 100);

        assert_eq!(clock.next_event_time(), Some(200));
        clock.advance_to(200);
        assert_eq!(*log.lock().unwrap(), vec![10, 11, 2]);
        assert!(!clock.has_sleepers());
    }

    #[test]
    fn advance_is_monotonic() {
        let clock = SimClock::new();
        clock.advance_to(500);
        clock.advance_to(300); // ignored
        assert_eq!(clock.now_ns(), 500);
    }
}
