// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wall-vs-virtual clock abstraction.
//!
//! A single [`Clock`] trait is implemented by:
//!   - [`RealClock`] — real (monotonic) time; `sleep` actually waits.
//!   - [`SimClock`] — a virtual discrete-event clock advanced explicitly, so a
//!     run completes in simulated time with no wall-clock waits.
//!
//! The same async executor runs identically on either clock — the foundation
//! for driving one front-end both live (real time) and simulated (virtual time).

pub mod clock;
pub mod real_clock;
pub mod sim_clock;

pub use clock::Clock;
pub use real_clock::{RealClock, RealClockAnchor, sleep_ns};
pub use sim_clock::SimClock;

/// Drive `body` to completion under a virtual [`SimClock`] with an idle pump:
/// poll to quiescence, advance the clock to the next scheduled event, repeat.
///
/// Shared by unit tests across the crate so this discrete-event pump — whose
/// poll/advance ordering is easy to get subtly wrong — lives in exactly one
/// place. This is the general-purpose twin of the graph-specific
/// [`crate::graph::runtime::drive_sim`], which is coupled to a graph `Handle`.
#[cfg(test)]
pub(crate) fn drive_sim<T>(
    clock: std::rc::Rc<SimClock>,
    body: impl std::future::Future<Output = T>,
) -> T {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll, Wake, Waker};

    struct FlagWaker(Arc<AtomicBool>);
    impl Wake for FlagWaker {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap();
    let _guard = runtime.enter();
    let local = tokio::task::LocalSet::new();
    let future = local.run_until(body);
    let mut future = std::pin::pin!(future);
    let flag = Arc::new(AtomicBool::new(true));
    let waker = Waker::from(Arc::new(FlagWaker(flag.clone())));
    let mut context = Context::from_waker(&waker);

    loop {
        flag.store(false, Ordering::SeqCst);
        match future.as_mut().poll(&mut context) {
            Poll::Ready(value) => return value,
            Poll::Pending if flag.load(Ordering::SeqCst) => continue,
            Poll::Pending => match clock.next_event_time() {
                Some(next) => clock.advance_to(next),
                None => panic!("deadlock: no simulated clock event"),
            },
        }
    }
}
