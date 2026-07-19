// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-thread global request-rate pacing for `global` dispatch mode.
//!
//! [`GlobalRateGate`] replaces per-thread rate slicing (`rate / workers` in
//! `sharded_scheduled::slice_phase_for_thread`) with one shared next-fire-time
//! counter so the aggregate issuance rate across all worker threads matches a
//! single global limiter exactly, the way Python's implementation does.
//!
//! The shared counter only models the **fixed-interval base grid** (`0`,
//! `interval_ns`, `2*interval_ns`, ...). A caller draws a mean-zero offset
//! from its own [`IntervalGenerator`](crate::timing::IntervalGenerator)
//! (`next_interval_ns() - interval_ns()`) and adds it to its claimed base slot,
//! so the aggregate arrival rate remains exactly the configured global rate
//! regardless of jitter. This is **not** a reproduction of Poisson/Gamma
//! arrival-process statistics: each thread's own generator still contributes a
//! bounded, mean-zero scatter around its slot, but the resulting inter-arrival
//! times are not exponentially distributed and do not carry the authored
//! distribution's shape (constant-variance grid + offset, not a true renewal
//! process). Exact global concurrency and exact global *rate* (this type's
//! job) are the byte-exactness guarantees `Global` mode makes; full
//! Poisson/Gamma arrival-*pattern* parity requires `global-hop`. See
//! [`GlobalRateGate::claim_offset_ns`].

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use crate::clock::Clock;

/// A shared, clock-driven admission gate enforcing one global request rate
/// across every worker thread in a `global`-dispatch cell.
pub struct GlobalRateGate {
    /// Nanoseconds between successive fires (`1e9 / rate_per_sec`).
    interval_ns: i64,
    /// The next unclaimed fire time, in run-clock nanoseconds. Claimed via an
    /// atomic fetch-add so concurrent callers on different threads each get a
    /// distinct slot.
    next_fire_ns: AtomicI64,
}

impl GlobalRateGate {
    /// Create a gate gating admission to `rate_per_sec` requests per second,
    /// with the first caller's fire time starting at `0` (relative to phase
    /// start; the caller's `Clock` origin already anchors this to run start).
    pub fn new(rate_per_sec: f64) -> Arc<Self> {
        assert!(rate_per_sec > 0.0, "GlobalRateGate rate must be positive");
        Arc::new(Self {
            interval_ns: (1e9 / rate_per_sec).round() as i64,
            next_fire_ns: AtomicI64::new(0),
        })
    }

    /// The fixed base interval between successive slots, in nanoseconds
    /// (`round(1e9 / rate_per_sec)`). Callers use this both to add a phase-start
    /// anchor and to compute a mean-zero jitter offset relative to the grid.
    pub fn interval_ns(&self) -> i64 {
        self.interval_ns
    }

    /// Atomically claim the next base-grid offset (`0`, `interval_ns`,
    /// `2*interval_ns`, ...) and return it without waiting.
    ///
    /// Each concurrent caller across every worker thread gets a distinct,
    /// gapless offset via a single `fetch_add`, so the union of all claimed
    /// offsets is exactly the evenly-spaced grid — the property that keeps the
    /// aggregate rate exact. The returned value is a *relative* offset from the
    /// gate's origin; the caller anchors it to phase start and optionally adds a
    /// mean-zero jitter offset before pacing to it on its own `Clock`.
    pub fn claim_offset_ns(self: &Arc<Self>) -> i64 {
        self.next_fire_ns
            .fetch_add(self.interval_ns, Ordering::SeqCst)
    }

    /// Claim the next fire slot and wait for it via `clock`.
    ///
    /// Each caller claims a monotonically-increasing slot by atomically
    /// advancing `next_fire_ns`. The caller then sleeps until their fire time
    /// has arrived according to `clock`.
    pub async fn wait_turn(self: &Arc<Self>, clock: std::rc::Rc<dyn Clock>) {
        let fire_ns = self.claim_offset_ns();
        let now_ns = clock.now_ns();
        let duration_ns = fire_ns.saturating_sub(now_ns);
        clock.sleep(duration_ns).await;
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use super::*;
    use crate::clock::SimClock;

    #[test]
    fn global_rate_gate_serializes_fire_times_across_threads() {
        let clock = Rc::new(SimClock::new());
        let gate = GlobalRateGate::new(1000.0); // 1000 req/s => 1ms apart
        let fire_times = Rc::new(RefCell::new(Vec::new()));

        let body = {
            let fire_times = fire_times.clone();
            let clock = clock.clone();
            async move {
                for _ in 0..5 {
                    gate.wait_turn(clock.clone()).await;
                    fire_times.borrow_mut().push(clock.now_ns());
                }
            }
        };

        let body = Box::pin(body);
        clock.clone().drive(body);

        let times = fire_times.borrow();
        for pair in times.windows(2) {
            assert_eq!(
                pair[1] - pair[0],
                1_000_000,
                "fire times must be exactly 1ms apart"
            );
        }
    }

    /// Proves the cross-thread serialization property the doc comments
    /// assert: real `std::thread::spawn` OS threads racing `wait_turn`
    /// concurrently each claim a distinct, gapless slot in the shared
    /// `next_fire_ns` sequence. Each thread builds its own current-thread
    /// runtime, `LocalSet`, and `Rc<RealClock>` (both `!Send`), all anchored
    /// to one shared [`RealClockAnchor`] (`Copy`/`Send`) so every thread's
    /// `now_ns()` sits on the same timeline; the only thing actually shared
    /// across threads for the claim itself is the `Arc<GlobalRateGate>` and
    /// its internal `AtomicI64`.
    #[test]
    fn global_rate_gate_serializes_fire_times_across_real_os_threads() {
        use std::sync::Mutex;
        use std::thread;

        use crate::clock::{RealClock, RealClockAnchor};

        const RATE_PER_SEC: f64 = 500.0; // interval_ns == 2_000_000 (2ms)
        const INTERVAL_NS: i64 = 2_000_000;
        const N_THREADS: usize = 6;
        const CALLS_PER_THREAD: usize = 12;
        const EXPECTED_TOTAL: usize = N_THREADS * CALLS_PER_THREAD;

        let gate = GlobalRateGate::new(RATE_PER_SEC);
        assert_eq!(gate.interval_ns, INTERVAL_NS);
        let anchor = RealClockAnchor::now();
        let all_fire_times: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));

        let handles: Vec<_> = (0..N_THREADS)
            .map(|_| {
                let gate = gate.clone();
                let all_fire_times = all_fire_times.clone();
                thread::spawn(move || {
                    // `Rc<dyn Clock>` and the current-thread runtime are both
                    // `!Send`, so each OS thread constructs its own; sharing
                    // `anchor` keeps every thread's `now_ns()` on one
                    // timeline so claimed slots line up with observed fire
                    // times across threads.
                    let clock: Rc<dyn Clock> = RealClock::from_anchor(anchor);
                    let local_fire_times =
                        Rc::new(RefCell::new(Vec::with_capacity(CALLS_PER_THREAD)));
                    let body = {
                        let clock = clock.clone();
                        let local_fire_times = local_fire_times.clone();
                        async move {
                            for _ in 0..CALLS_PER_THREAD {
                                gate.wait_turn(clock.clone()).await;
                                local_fire_times.borrow_mut().push(clock.now_ns());
                            }
                        }
                    };
                    clock.drive(Box::pin(body));
                    all_fire_times
                        .lock()
                        .unwrap()
                        .extend(local_fire_times.borrow().iter().copied());
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker thread panicked");
        }

        let times = all_fire_times.lock().unwrap();
        assert_eq!(times.len(), EXPECTED_TOTAL);

        // The gate never wakes a caller before its claimed fire time, and
        // real-clock scheduling jitter is far smaller than `INTERVAL_NS`, so
        // rounding each observed fire time down to the interval grid
        // recovers exactly the slot index that thread claimed via
        // `fetch_add`. If two threads ever raced to the same slot (a lost
        // update), two entries would collide on the same bucket here.
        let mut slots: Vec<i64> = times.iter().map(|&t| t / INTERVAL_NS).collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(
            slots.len(),
            EXPECTED_TOTAL,
            "every claimed slot must be distinct across all OS threads (no duplicates from a lost update)"
        );
        // No gaps: the recovered slot set is exactly {0, 1, ..., N-1}, i.e.
        // the full claimed set is {0, interval_ns, 2*interval_ns, ...}.
        let expected_slots: Vec<i64> = (0..EXPECTED_TOTAL as i64).collect();
        assert_eq!(
            slots, expected_slots,
            "claimed slots across all threads must be exactly \
             {{0, interval_ns, 2*interval_ns, ...}} with no gaps"
        );

        // The shared atomic itself: after `EXPECTED_TOTAL` successful
        // `fetch_add` claims from racing OS threads, it must land at
        // precisely `EXPECTED_TOTAL * interval_ns` with no lost updates.
        let final_next_fire_ns = gate.next_fire_ns.load(Ordering::SeqCst);
        assert_eq!(
            final_next_fire_ns,
            EXPECTED_TOTAL as i64 * INTERVAL_NS,
            "shared counter must reflect exactly one fetch_add per wait_turn call \
             across all threads, with no lost updates"
        );
    }
}
