// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The online run loop: a Clock-driven arrival pacer that dispatches a synthetic
//! workload through [`TransportSink`] (the `aiperf-transport` hyper client) and
//! measures it with the shared `TraceCollector`.
//!
//! One loop serves both modes via the [`IntervalGenerator`](crate::timing::IntervalGenerator)
//! seam: **request-rate** (Poisson/Gamma/Constant inter-arrivals) and
//! **concurrency** (the degenerate `ConcurrencyBurst` = zero interval, bounded by a
//! session semaphore). Arrival timing uses only `clock.now_ns()` +
//! `clock.sleep()`, so the identical loop runs on `RealClock` (online) or `SimClock`
//! (offline) — the pacer is backend-agnostic by construction.
//!
//! The transport is `!Send` (`Rc<dyn Clock>`), so the loop runs on a single
//! `LocalSet` with `spawn_local`; a shared clock is the one time authority, so
//! arrival, admit, and token timestamps all sit on the same timeline.

use std::rc::Rc;
use std::sync::Arc;

use tokio::sync::Semaphore;

use aiperf_clock::{Clock, RealClock};
use aiperf_core::observer::CollectorObserver;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use loadgen_core::sink::{RequestObserver, RequestSink};

use crate::http::TransportSink;
use crate::timing::{ArrivalPattern, make_interval_generator};
use crate::workload::SkeletonWorkload;

/// Run a closed-loop concurrency-`concurrency` load test of `workload` against
/// `base_url` for `model`, returning the aggregated report. Thin wrapper over
/// [`run_paced`] with the `ConcurrencyBurst` arrival pattern (zero inter-arrival
/// delay; throughput bounded by the semaphore).
///
/// Must be driven inside a `LocalSet` (the transport sink is `!Send`).
pub async fn run(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    concurrency: usize,
) -> anyhow::Result<TraceSimulationReport> {
    run_paced(
        base_url,
        model,
        workload,
        ArrivalPattern::ConcurrencyBurst,
        None,
        None,
        Some(concurrency),
        0,
    )
    .await
}

/// Run `workload` against `base_url` with an explicit arrival `pattern`.
///
/// - `rate` (req/s) is required for every pattern except `ConcurrencyBurst`.
/// - `smoothness` tunes `Gamma` burstiness (`None` -> Poisson-equivalent).
/// - `concurrency` caps requests in flight; `None` = open-loop (rate mode's
///   natural shape). `ConcurrencyBurst` + `Some(n)` = the closed-loop path.
/// - `seed` seeds the arrival RNG for bit-reproducible spacing.
///
/// Pacing is **absolute-schedule**: cumulative target times, re-anchored to `now`
/// when the loop falls behind (so dispatch latency never compounds into drift or a
/// catch-up burst). The next interval is drawn *before* dispatch so issue latency
/// doesn't skew it. Must be driven inside a `LocalSet`.
#[allow(clippy::too_many_arguments)]
pub async fn run_paced(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    concurrency: Option<usize>,
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
    let sem = concurrency.map(|c| Arc::new(Semaphore::new(c.max(1))));
    let mut intervals = make_interval_generator(pattern, rate, smoothness, seed);

    // Absolute schedule: the next arrival's target time on the clock's timeline.
    let mut next_target_ns = start_ns + intervals.next_interval_ns();

    let mut handles = Vec::new();
    for req in workload.generate() {
        // Pace to the next arrival target. Falling behind re-anchors to `now`
        // (drop the burst, keep throughput) rather than firing a catch-up salvo.
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

        // Concurrency cap (if configured): acquire a slot before dispatch. Open-loop
        // rate mode passes `None` and never blocks here.
        let permit = match &sem {
            Some(s) => Some(s.clone().acquire_owned().await?),
            None => None,
        };

        obs.on_arrival(
            req.uuid,
            ms(clock.now_ns()),
            req.input_length,
            req.max_output_tokens,
        );
        let obs2 = obs.clone();
        let sink2 = sink.clone();
        handles.push(tokio::task::spawn_local(async move {
            let _permit = permit;
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
        // Constant 1000 req/s over a fast mock: the pacer must sleep ~1ms between
        // the N arrivals, so wall time >= (N-1)ms even though the mock replies
        // near-instantly. Open-loop (no concurrency cap).
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
                let report = run_paced(
                    base,
                    "m".into(),
                    wl,
                    ArrivalPattern::Constant,
                    Some(rate),
                    None,
                    None,
                    0,
                )
                .await
                .unwrap();
                assert_eq!(report.request_counts.num_requests, n);
                // (n-1) inter-arrival gaps of 1ms = 19ms floor; allow scheduler slack.
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
}
