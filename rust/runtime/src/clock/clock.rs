// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The time source the async dataflow runs against.
//!
//! Two implementations back the same tokio executor:
//!
//! * [`SimClock`](crate::clock::sim_clock::SimClock) — **virtual** discrete-event time
//!   (ai-dynamo dynosim's pattern), ns-exact and deterministic. Driven by the
//!   downstream sim idle-pump driver (`drive_sim`); used for fast, reproducible
//!   runs where timers cost nothing.
//! * [`RealClock`](crate::clock::real_clock::RealClock) — **real** wall-clock time with
//!   ns-precision `timerfd` timers on Linux, integrated with tokio's IO reactor.
//!   The live backend for dispatching in real time.

use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

/// A sleepable time source.
pub trait Clock {
    /// Current time in nanoseconds (virtual for sim, monotonic for real).
    fn now_ns(&self) -> i64;

    /// A future that resolves after `duration_ns` of this clock's time.
    /// Non-positive durations resolve after a single task yield.
    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>>;

    // Virtual-time control (fast-forward to the next event, advance-and-wake) is
    // intentionally NOT on this trait: it is meaningful only for `SimClock` and
    // is driven by `drive_sim` through the concrete `Rc<SimClock>`, via
    // `SimClock`'s inherent `next_event_time`/`advance_to` methods. Keeping it off
    // the trait avoids no-op stubs on `RealClock`.

    /// True when this clock needs the sim idle-pump driver (the downstream
    /// `drive_sim` pump); false when it drives via tokio's reactor (the
    /// downstream `drive_real` pump).
    fn is_virtual(&self) -> bool {
        false
    }
}
