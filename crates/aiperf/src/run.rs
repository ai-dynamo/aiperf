// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The online run loop: a Clock-driven arrival pacer, gated by `StopChecker`
//! (request-count / duration) and `SlotPool` (concurrency), dispatching a synthetic
//! workload through [`TransportSink`] and measuring it with the shared
//! `TraceCollector`.
//!
//! One loop serves both modes via the timing-plane seam:
//! - **request-rate** — Poisson/Gamma/Constant inter-arrivals ([`IntervalGenerator`](crate::timing::IntervalGenerator)),
//! - **concurrency** — the degenerate `ConcurrencyBurst` (zero interval) bounded by a
//!   session [`SlotPool`](crate::timing::SlotPool).
//!
//! Stopping is condition-driven ([`StopChecker`](crate::timing::StopChecker)): the
//! loop pulls requests on demand until the request-count and/or duration bound fires,
//! not until a fixed list is exhausted. Arrival timing uses only `clock.now_ns()` +
//! `clock.sleep()`, so the identical loop runs on `RealClock` (online) or `SimClock`
//! (offline) — backend-agnostic by construction.
//!
//! The transport is `!Send` (`Rc<dyn Clock>`), so the loop runs on a single
//! `LocalSet` with `spawn_local`; a shared clock is the one time authority.

use std::rc::Rc;
use std::sync::Arc;

use aiperf_clock::{Clock, RealClock};
use aiperf_core::observer::CollectorObserver;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use loadgen_core::sink::{RequestObserver, RequestSink};

use aiperf_timing::{
    ArrivalPattern, RunState, SlotPool, StopChecker, StopConfig, make_interval_generator,
};

use crate::http::TransportSink;
use crate::workload::SkeletonWorkload;

/// Run a closed-loop concurrency-`concurrency` benchmark of `workload` against
/// `base_url` for `model`, bounded by `workload.num_requests`. Thin wrapper over
/// [`run_paced`] with `ConcurrencyBurst` arrival (zero delay; throughput bounded by
/// the session slot pool). Must be driven inside a `LocalSet`.
pub async fn run(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    concurrency: usize,
) -> anyhow::Result<TraceSimulationReport> {
    let stop = StopConfig {
        total_expected_requests: Some(workload.num_requests as u64),
        ..Default::default()
    };
    run_paced(
        base_url,
        model,
        workload,
        ArrivalPattern::ConcurrencyBurst,
        None,
        None,
        Some(concurrency),
        stop,
        0,
    )
    .await
}

/// Run `workload` against `base_url` with an explicit arrival `pattern`, `stop`
/// bounds, and optional concurrency cap.
///
/// - `rate` (req/s) is required for every pattern except `ConcurrencyBurst`.
/// - `smoothness` tunes `Gamma` burstiness (`None` -> Poisson-equivalent).
/// - `concurrency` caps in-flight requests via a `SlotPool`; `None` = open-loop.
/// - `stop` bounds the run (request-count and/or duration; first-hit wins). At least
///   one bound must be set or the loop never terminates.
/// - `seed` seeds the arrival RNG for bit-reproducible spacing.
///
/// Pacing is **absolute-schedule** with catch-up re-anchoring; the next interval is
/// drawn before dispatch. Must be driven inside a `LocalSet`.
#[allow(clippy::too_many_arguments)]
pub async fn run_paced(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
    stop: StopConfig,
    seed: u64,
) -> anyhow::Result<TraceSimulationReport> {
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let ms = |ns: i64| (ns - start_ns) as f64 / 1_000_000.0;

    let obs = Arc::new(CollectorObserver::new(false));
    let sink = Rc::new(TransportSink::new(
        clock.clone(),
        start_ns,
        &base_url,
        model,
        false,
    ));
    let slots = concurrency.map(SlotPool::new);
    let mut intervals = make_interval_generator(pattern, rate, smoothness, seed);

    let checker = StopChecker::new(&stop);
    let mut state = RunState {
        started_at_ns: start_ns,
        ..Default::default()
    };

    // Absolute schedule: the next arrival's target time on the clock's timeline.
    let mut next_target_ns = start_ns + intervals.next_interval_ns();

    let mut handles = Vec::new();
    while checker.can_send_any(&state, clock.now_ns()) {
        // Pace to the next arrival target. Falling behind re-anchors to `now` rather
        // than firing a catch-up salvo.
        let now = clock.now_ns();
        if next_target_ns < now {
            next_target_ns = now;
        }
        let wait_ns = next_target_ns - now;
        if wait_ns > 0 {
            clock.clone().sleep(wait_ns).await;
        }
        // Draw the next interval BEFORE dispatch so issue latency doesn't skew it.
        next_target_ns += intervals.next_interval_ns();

        // Duration may have elapsed during the sleep — re-check before dispatching.
        if !checker.can_send_any(&state, clock.now_ns()) {
            break;
        }

        // Session slot (if capped): acquire before dispatch; the guard releases the
        // slot when the dispatch task completes. Open-loop rate passes `None`.
        let guard = match &slots {
            Some(pool) => Some(pool.acquire().await),
            None => None,
        };

        let req = workload.make_request();
        obs.on_arrival(
            req.uuid,
            ms(clock.now_ns()),
            req.input_length,
            req.max_output_tokens,
        );
        // Single-turn synthetic: each request is its own session (turn 0 = final).
        state.requests_sent += 1;
        state.root_requests_sent += 1;
        state.sent_sessions += 1;

        let obs2 = obs.clone();
        let sink2 = sink.clone();
        handles.push(tokio::task::spawn_local(async move {
            let _guard = guard; // releases the session slot on completion
            let uuid = req.uuid;
            if let Err(e) = sink2.dispatch(req, obs2.as_ref()).await {
                obs2.on_terminal(uuid, ReplayTerminalStatus::Failed);
                tracing::warn!(%uuid, error = %e, "request dispatch failed");
            }
        }));
    }
    for h in handles {
        if let Err(e) = h.await {
            tracing::warn!(error = %e, "request task join failed");
        }
    }

    let wall_ms = ms(clock.now_ns());
    Ok(obs.finish(wall_ms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::SkeletonWorkload;

    #[tokio::test]
    async fn e2e_reports_finite_metrics() {
        // The transport sink is `!Send`, so drive the run on a LocalSet.
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let wl = SkeletonWorkload {
                    num_requests: 4,
                    input_tokens: 8,
                    output_tokens: 2,
                };
                let report = run(base, "m".into(), wl, 2).await.unwrap();
                assert_eq!(report.request_counts.num_requests, 4);
                // 4 requests * 2 content chunks each.
                assert_eq!(report.request_counts.total_output_tokens, 8);
                assert!(report.latency.ttft.mean_ms.is_finite());
                assert!(report.throughput.output_throughput_tok_s.is_finite());
            })
            .await;
    }

    #[tokio::test]
    async fn request_rate_paces_arrivals_by_the_clock() {
        // Constant 1000 req/s over a fast mock: the pacer sleeps ~1ms between the N
        // arrivals, so wall time >= (N-1)ms even though the mock replies instantly.
        // Open-loop (no concurrency cap); bounded by request count.
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let n = 20usize;
                let rate = 1000.0; // 1ms interval
                let wl = SkeletonWorkload {
                    num_requests: n,
                    input_tokens: 8,
                    output_tokens: 1,
                };
                let stop = StopConfig {
                    total_expected_requests: Some(n as u64),
                    ..Default::default()
                };
                let report = run_paced(
                    base,
                    "m".into(),
                    wl,
                    ArrivalPattern::Constant,
                    Some(rate),
                    None,
                    None,
                    stop,
                    0,
                )
                .await
                .unwrap();
                assert_eq!(report.request_counts.num_requests, n);
                let floor_ms = (n as f64 - 1.0) / rate * 1000.0 * 0.75;
                assert!(
                    report.throughput.wall_time_ms >= floor_ms,
                    "wall {:.2}ms should reflect pacing floor {:.2}ms",
                    report.throughput.wall_time_ms,
                    floor_ms
                );
                assert!(report.latency.ttft.mean_ms.is_finite());
            })
            .await;
    }

    #[tokio::test]
    async fn duration_bound_stops_the_run() {
        // No request-count cap: the run is bounded purely by duration. Burst arrivals
        // + concurrency 4 against a fast mock; a 60ms duration must stop it (and admit
        // more than the handful a count-bound test would).
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let base = crate::test_util::spawn_mock().await;
                let wl = SkeletonWorkload {
                    num_requests: 0, // unused: count bound is None below
                    input_tokens: 4,
                    output_tokens: 1,
                };
                let stop = StopConfig {
                    total_expected_requests: None,
                    expected_num_sessions: None,
                    expected_duration_ns: Some(60_000_000), // 60ms
                };
                let report = run_paced(
                    base,
                    "m".into(),
                    wl,
                    ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    Some(4),
                    stop,
                    0,
                )
                .await
                .unwrap();
                assert!(
                    report.request_counts.num_requests > 0,
                    "duration run should admit at least one request"
                );
                assert!(report.throughput.output_throughput_tok_s.is_finite());
            })
            .await;
    }
}
