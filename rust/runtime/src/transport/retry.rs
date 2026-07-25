// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-neutral connect-phase retry with clock-driven linear backoff.
//!
//! The HTTP and gRPC transports share the same connect-retry policy: one initial
//! attempt plus up to `max_connect_retries` further tries, retrying only genuine
//! pre-send connect failures and sleeping `n * connect_retry_backoff_ns` on the
//! injected [`Clock`] before retry `n`. They differ only in their error type and
//! in which failures count as retryable, expressed here by the `is_retryable`
//! predicate.

use std::rc::Rc;

use crate::clock::Clock;

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
pub(crate) async fn retry_connect<T, E>(
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
