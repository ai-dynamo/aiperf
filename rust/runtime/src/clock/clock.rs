// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Clock abstraction for real and deterministic virtual execution.

use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

use crate::graph::runtime::RunOutcome;

/// A sleepable time source.
pub trait Clock {
    /// Current time in nanoseconds (virtual for sim, monotonic for real).
    fn now_ns(&self) -> i64;

    /// A future that resolves after `duration_ns` of this clock's time.
    /// Non-positive durations resolve after a single task yield.
    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>>;

    // Virtual-time control stays on `SimClock` because `RealClock` cannot
    // advance explicitly.

    /// Whether this clock requires an idle pump to advance virtual time.
    fn is_virtual(&self) -> bool {
        false
    }

    /// Drive `body` to completion using this clock's reactor discipline.
    ///
    /// The default drives on a current-thread tokio runtime whose IO/timer
    /// reactor wakes real sleepers. [`SimClock`](crate::clock::sim_clock::SimClock)
    /// overrides this with deterministic event-by-event advancement; a
    /// [`RunOutcome::deadlocked`] result means no virtual event can make progress.
    fn drive(self: Rc<Self>, body: Pin<Box<dyn Future<Output = ()> + '_>>) -> RunOutcome {
        // IO + time only: this driver needs no signal handling. See
        // turn_execution::run_worker_thread for why this does not remove
        // tokio's child-process orphan sweep from the park path.
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_io()
            .enable_time()
            .build()
            .expect("current-thread runtime for real-clock run driver");
        tokio::task::LocalSet::new().block_on(&runtime, body);
        RunOutcome { deadlocked: false }
    }
}
