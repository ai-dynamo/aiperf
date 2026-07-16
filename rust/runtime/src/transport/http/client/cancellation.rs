// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellation timing. Port of
//! `AioHttpClient._request_with_cancellation`: the cancel timer starts once the
//! request body is sent, and on timeout the request future is dropped and
//! recorded as a 499 cancellation.

use std::future::Future;
use std::rc::Rc;

use crate::clock::Clock;

use crate::transport::http::client::connection::SendCompletion;

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

/// Resolve `fut`, or wait for the request body's send-completion signal and
/// cancel at `sent_ns + cancel_after_ns`.
///
/// A request that fails or finishes before send completion is returned normally.
/// The response branch is biased at exact ties, guaranteeing one terminal
/// outcome and allowing an already-completed response to win over a zero-delay
/// timer.
pub async fn race_cancel_after_send<T>(
    clock: Rc<dyn Clock>,
    cancel_after_ns: i64,
    completion: Rc<SendCompletion>,
    fut: impl Future<Output = T>,
) -> CancelOutcome<T> {
    futures::pin_mut!(fut);
    let sent = completion.wait();
    futures::pin_mut!(sent);

    let sent_ns = tokio::select! {
        biased;
        out = &mut fut => return CancelOutcome::Completed(out),
        sent_ns = &mut sent => sent_ns,
    };

    let deadline_ns = sent_ns.saturating_add(cancel_after_ns.max(0));
    let remaining_ns = deadline_ns.saturating_sub(clock.now_ns()).max(0);
    race_cancel(clock, remaining_ns, fut).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{SimClock, drive_sim};
    use std::rc::Rc;

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

    #[test]
    fn post_send_timer_is_anchored_to_captured_send_time() {
        use crate::transport::http::client::connection::TimedBody;
        use bytes::Bytes;
        use http_body::Body;

        let clock = Rc::new(SimClock::new());
        let clock_for_body = clock.clone();
        let clk: Rc<dyn Clock> = clock.clone();
        let completion = Rc::new(crate::transport::http::client::connection::SendCompletion::new());
        let completion_for_body = completion.clone();
        let body_clock: Rc<dyn Clock> = clock.clone();

        let outcome = drive_sim(clock.clone(), async move {
            // Polling the single-frame body stamps send completion at t=250ns.
            clock_for_body.advance_to(250);
            let mut body = Box::pin(TimedBody::with_completion(
                Bytes::from_static(b"complete request"),
                body_clock,
                completion_for_body,
            ));
            let frame = futures::future::poll_fn(|cx| body.as_mut().poll_frame(cx)).await;
            assert!(matches!(frame, Some(Ok(_))));
            // Simulate executor lag before the cancellation driver observes the
            // signal. The deadline must remain 250 + 1000, not 500 + 1000.
            clock_for_body.advance_to(500);

            let never = futures::future::pending::<u32>();
            race_cancel_after_send(clk, 1_000, completion, never).await
        });
        assert!(matches!(outcome, CancelOutcome::Cancelled));
        assert_eq!(clock.now_ns(), 1_250);
    }

    #[test]
    fn request_can_complete_before_send_signal() {
        let clock = Rc::new(SimClock::new());
        let clk: Rc<dyn Clock> = clock.clone();
        let completion = Rc::new(crate::transport::http::client::connection::SendCompletion::new());
        let outcome = drive_sim(clock, async move {
            race_cancel_after_send(clk, 0, completion, async { 7_u32 }).await
        });
        assert!(matches!(outcome, CancelOutcome::Completed(7)));
    }

    #[test]
    fn response_wins_an_exact_post_send_deadline_tie() {
        use crate::transport::http::client::connection::TimedBody;
        use bytes::Bytes;
        use http_body::Body;

        let clock = Rc::new(SimClock::new());
        let body_clock: Rc<dyn Clock> = clock.clone();
        let completion = Rc::new(crate::transport::http::client::connection::SendCompletion::new());
        let completion_for_body = completion.clone();
        let request_clock: Rc<dyn Clock> = clock.clone();
        let race_clock: Rc<dyn Clock> = clock.clone();
        let outcome = drive_sim(clock, async move {
            let mut body = Box::pin(TimedBody::with_completion(
                Bytes::from_static(b"body"),
                body_clock,
                completion_for_body,
            ));
            let _ = futures::future::poll_fn(|cx| body.as_mut().poll_frame(cx)).await;
            race_cancel_after_send(race_clock, 100, completion, async move {
                request_clock.sleep(100).await;
                42_u32
            })
            .await
        });
        assert!(matches!(outcome, CancelOutcome::Completed(42)));
    }
}
