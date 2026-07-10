// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellation timing. Port of `AioHttpClient._request_with_cancellation`:
//! the cancel timer starts once the request body is sent, and on timeout the
//! request future is dropped and recorded as a 499 cancellation.

use std::future::Future;
use std::rc::Rc;

use aiperf_clock::Clock;

/// Outcome of racing a request future against a cancel timer.
pub enum CancelOutcome<T> {
    Completed(T),
    Cancelled,
}

/// Resolve `fut`, or cancel it after `cancel_after_ns` of clock time.
pub async fn race_cancel<T>(
    clock: Rc<dyn Clock>,
    cancel_after_ns: i64,
    fut: impl Future<Output = T>,
) -> CancelOutcome<T> {
    let timer = clock.sleep(cancel_after_ns);
    futures::pin_mut!(fut);
    tokio::select! {
        biased;
        out = &mut fut => CancelOutcome::Completed(out),
        _ = timer => CancelOutcome::Cancelled,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aiperf_clock::SimClock;
    use std::pin::pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::task::{Context, Poll, Wake, Waker};
    use tokio::task::LocalSet;

    struct FlagWaker(Arc<AtomicBool>);
    impl Wake for FlagWaker {
        fn wake(self: Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
        fn wake_by_ref(self: &Arc<Self>) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    /// Drive `make_body` under a virtual [`SimClock`] idle-pump (a local copy of
    /// `aiperf_graph::runtime::drive_sim`), returning the body's output.
    fn drive_sim<T>(clock: Rc<SimClock>, body: impl Future<Output = T>) -> T {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let _guard = rt.enter();
        let local = LocalSet::new();
        let fut = local.run_until(body);
        let mut fut = pin!(fut);

        let flag = Arc::new(AtomicBool::new(true));
        let waker = Waker::from(Arc::new(FlagWaker(flag.clone())));
        let mut cx = Context::from_waker(&waker);

        loop {
            flag.store(false, Ordering::SeqCst);
            match fut.as_mut().poll(&mut cx) {
                Poll::Ready(v) => return v,
                Poll::Pending => {
                    if flag.load(Ordering::SeqCst) {
                        continue;
                    }
                    match clock.next_event_time() {
                        Some(t) => clock.advance_to(t),
                        None => panic!("deadlock: no clock event to advance"),
                    }
                }
            }
        }
    }

    #[test]
    fn cancels_a_never_completing_future_after_delay() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let outcome = drive_sim(clock.clone(), async move {
            let never = futures::future::pending::<u32>();
            race_cancel(clk, 1_000_000, never).await
        });
        assert!(matches!(outcome, CancelOutcome::Cancelled));
    }

    #[test]
    fn completes_when_future_resolves_before_timer() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let outcome = drive_sim(clock.clone(), async move {
            // Resolves immediately (before the 1ms cancel timer).
            let ready = async { 42u32 };
            race_cancel(clk, 1_000_000, ready).await
        });
        assert!(matches!(outcome, CancelOutcome::Completed(42)));
    }
}
