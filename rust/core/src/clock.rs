// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Boundary-owned clock abstraction for real and deterministic virtual execution.
//!
//! Every measurement and firing gate in the product routes its time through
//! [`Clock`]. The trait lives here rather than in the runtime so a plugin can
//! be authored, and its determinism tested, against the boundary alone.

use std::future::Future;
use std::pin::Pin;
use std::rc::Rc;

/// The result of driving a graph to quiescence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunOutcome {
    /// Tasks remained parked with no future clock event to wake them.
    pub deadlocked: bool,
}

/// A sleepable time source.
pub trait Clock {
    /// Current time in nanoseconds (virtual for sim, monotonic for real).
    fn now_ns(&self) -> i64;

    /// A future that resolves after `duration_ns` of this clock's time.
    /// Non-positive durations resolve after a single task yield.
    fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>>;

    // Virtual-time control stays on the simulation clock because a real clock
    // cannot advance explicitly.

    /// Whether this clock requires an idle pump to advance virtual time.
    fn is_virtual(&self) -> bool {
        false
    }

    /// Drive `body` to completion using this clock's reactor discipline.
    ///
    /// The default drives on a current-thread tokio runtime whose IO/timer
    /// reactor wakes real sleepers. A virtual clock overrides this with
    /// deterministic event-by-event advancement; a [`RunOutcome::deadlocked`]
    /// result means no virtual event can make progress.
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
