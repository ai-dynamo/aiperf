// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real and deterministic virtual clock implementations.

pub mod clock;
pub mod real_clock;
pub mod sim_clock;

pub use clock::Clock;
pub use real_clock::{RealClock, RealClockAnchor, sleep_ns};
pub use sim_clock::SimClock;

/// Drive `body` to completion under a virtual [`SimClock`] with an idle pump:
/// poll to quiescence, advance the clock to the next scheduled event, repeat.
///
/// Polling must reach quiescence before each clock advance so same-time wakes
/// run in deterministic registration order.
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
