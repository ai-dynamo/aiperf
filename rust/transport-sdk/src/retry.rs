// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral connect-phase retry with clock-driven linear backoff.
//!
//! Every request transport shares one connect-retry policy: a single initial
//! attempt plus up to `max_connect_retries` further tries, retrying only genuine
//! pre-send connect failures and sleeping `n * connect_retry_backoff_ns` on the
//! injected [`Clock`] before retry `n`. Transports differ only in their error
//! type and in which failures count as retryable, expressed here by the
//! `is_retryable` predicate.
//!
//! Retrying only *pre-send* failures is the load-bearing part: a request that
//! reached the server and failed afterwards must not be replayed, or the
//! benchmark silently issues more work than it reports.

use std::rc::Rc;

use aiperf_core::clock::Clock;

/// Drive a connect-phase `attempt` with retry-on-connect-failure and linear
/// backoff.
///
/// `attempt` is invoked once, then up to `max_connect_retries` more times
/// whenever it fails with an error `is_retryable` accepts. Any other failure is
/// returned immediately without consuming a retry. Retry `n` (1-based) sleeps
/// `n * connect_retry_backoff_ns` on the injected [`Clock`] before re-invoking
/// `attempt`, so virtual-time replays stay deterministic. `attempt` is an
/// [`AsyncFnMut`] so callers can mutably borrow per-attempt state (such as trace
/// records) across each await.
pub async fn retry_connect<T, E>(
    clock: &Rc<dyn Clock>,
    max_connect_retries: u32,
    connect_retry_backoff_ns: i64,
    is_retryable: impl Fn(&E) -> bool,
    mut attempt: impl AsyncFnMut() -> Result<T, E>,
) -> Result<T, E> {
    let mut retries_taken: u32 = 0;
    loop {
        match attempt().await {
            Ok(value) => return Ok(value),
            Err(error) => {
                if !is_retryable(&error) || retries_taken >= max_connect_retries {
                    return Err(error);
                }
                retries_taken += 1;
                let wait_ns = connect_retry_backoff_ns.saturating_mul(i64::from(retries_taken));
                if wait_ns > 0 {
                    clock.clone().sleep(wait_ns).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::future::Future;
    use std::pin::Pin;
    use std::task::{Context, Poll, Wake, Waker};

    use super::*;

    /// Records every requested sleep instead of waiting for one.
    struct RecordingClock {
        slept_ns: RefCell<Vec<i64>>,
    }

    impl Clock for RecordingClock {
        fn now_ns(&self) -> i64 {
            0
        }
        fn sleep(self: Rc<Self>, duration_ns: i64) -> Pin<Box<dyn Future<Output = ()>>> {
            self.slept_ns.borrow_mut().push(duration_ns);
            Box::pin(async {})
        }
    }

    fn block_on<T>(future: impl Future<Output = T>) -> T {
        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = Waker::from(std::sync::Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = Box::pin(future);
        loop {
            if let Poll::Ready(value) = future.as_mut().poll(&mut context) {
                return value;
            }
        }
    }

    #[test]
    fn backoff_is_linear_in_the_retry_number() {
        let recording = Rc::new(RecordingClock {
            slept_ns: RefCell::new(Vec::new()),
        });
        let clock: Rc<dyn Clock> = recording.clone();
        let attempts = Cell::new(0u32);
        let outcome: Result<(), &str> = block_on(retry_connect(
            &clock,
            3,
            100,
            |_| true,
            async || {
                attempts.set(attempts.get() + 1);
                Err("connection refused")
            },
        ));
        assert!(outcome.is_err());
        assert_eq!(attempts.get(), 4, "one initial attempt plus three retries");
        assert_eq!(recording.slept_ns.borrow().as_slice(), &[100, 200, 300]);
    }

    #[test]
    fn a_non_retryable_error_consumes_no_retry() {
        let recording = Rc::new(RecordingClock {
            slept_ns: RefCell::new(Vec::new()),
        });
        let clock: Rc<dyn Clock> = recording.clone();
        let attempts = Cell::new(0u32);
        let outcome: Result<(), &str> = block_on(retry_connect(
            &clock,
            5,
            100,
            |_| false,
            async || {
                attempts.set(attempts.get() + 1);
                Err("http 500")
            },
        ));
        assert!(outcome.is_err());
        assert_eq!(attempts.get(), 1);
        assert!(recording.slept_ns.borrow().is_empty());
    }

    #[test]
    fn a_later_success_is_returned() {
        let recording = Rc::new(RecordingClock {
            slept_ns: RefCell::new(Vec::new()),
        });
        let clock: Rc<dyn Clock> = recording.clone();
        let attempts = Cell::new(0u32);
        let outcome: Result<u8, &str> = block_on(retry_connect(
            &clock,
            5,
            10,
            |_| true,
            async || {
                attempts.set(attempts.get() + 1);
                if attempts.get() < 3 {
                    Err("connection refused")
                } else {
                    Ok(7u8)
                }
            },
        ));
        assert_eq!(outcome.unwrap(), 7);
        assert_eq!(attempts.get(), 3);
    }
}
