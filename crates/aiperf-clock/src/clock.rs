// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The time source the async dataflow runs against.
//!
//! Two implementations back the same tokio executor:
//!
//! * [`SimClock`](crate::sim_clock::SimClock) — **virtual** discrete-event time
//!   (ai-dynamo dynosim's pattern), ns-exact and deterministic. Driven by the
//!   downstream sim idle-pump driver (`drive_sim`); used for fast, reproducible
//!   runs where timers cost nothing.
//! * [`RealClock`](crate::real_clock::RealClock) — **real** wall-clock time with
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

    /// Virtual clocks only: the deadline the sim driver should fast-forward to —
    /// `max(earliest scheduled deadline, now)`, so an already-due sleeper yields
    /// `now` rather than a past time. Real clocks return `None` (time flows on
    /// its own via the OS/reactor).
    fn next_event_time(&self) -> Option<i64> {
        None
    }

    /// Virtual clocks only: advance to `ns`, waking crossed sleepers. No-op for
    /// real clocks.
    fn advance_to(&self, _ns: i64) {}

    /// True when this clock needs the sim idle-pump driver (the downstream
    /// `drive_sim` pump); false when it drives via tokio's reactor (the
    /// downstream `drive_real` pump).
    fn is_virtual(&self) -> bool {
        false
    }
}
