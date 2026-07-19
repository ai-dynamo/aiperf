// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cross-thread global request-rate pacing for `global` dispatch mode.
//!
//! [`GlobalRateGate`] replaces per-thread rate slicing (`rate / workers` in
//! `sharded_scheduled::slice_phase_for_thread`) with one shared next-fire-time
//! counter so the aggregate issuance rate across all worker threads matches a
//! single global limiter exactly, the way Python's implementation does.

use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use crate::clock::Clock;

/// A shared, clock-driven admission gate enforcing one global request rate
/// across every worker thread in a `global`-dispatch cell.
pub struct GlobalRateGate {
    /// Nanoseconds between successive fires (`1e9 / rate_per_sec`).
    interval_ns: i64,
    /// The next unclaimed fire time, in run-clock nanoseconds. Claimed via CAS
    /// so concurrent callers on different threads each get a distinct slot.
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

    /// Claim the next fire slot and wait for it via `clock`.
    ///
    /// Each caller claims a monotonically-increasing slot by atomically
    /// advancing `next_fire_ns`. The caller then sleeps until their fire time
    /// has arrived according to `clock`.
    pub async fn wait_turn(self: &Arc<Self>, clock: std::rc::Rc<dyn Clock>) {
        let fire_ns = self
            .next_fire_ns
            .fetch_add(self.interval_ns, Ordering::SeqCst);
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
}
