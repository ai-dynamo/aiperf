// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The skeleton run loop: closed-loop concurrency over a synthetic workload,
//! dispatched through [`TransportSink`] (the `aiperf-transport` hyper client) and
//! measured by the shared `TraceCollector`.
//!
//! The transport is `!Send` (`Rc<dyn Clock>`), so the loop runs on a single
//! `LocalSet` with `spawn_local`; a shared `RealClock` is the one time authority,
//! so arrival, admit, and token timestamps all sit on the same timeline.

use std::rc::Rc;
use std::sync::Arc;

use tokio::sync::Semaphore;

use aiperf_clock::{Clock, RealClock};
use aiperf_core::observer::CollectorObserver;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use loadgen_core::sink::{RequestObserver, RequestSink};

use crate::http::TransportSink;
use crate::workload::SkeletonWorkload;

/// Run a concurrency-`concurrency` load test of `workload` against `base_url`
/// for `model`, returning the aggregated report.
///
/// Must be driven inside a `LocalSet` (e.g. `LocalSet::block_on`) because the
/// transport sink is `!Send`.
pub async fn run(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    concurrency: usize,
) -> anyhow::Result<TraceSimulationReport> {
    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let ms = |ns: i64| (ns - start_ns) as f64 / 1_000_000.0;

    let obs = Arc::new(CollectorObserver::new(false));
    let sink = Rc::new(TransportSink::new(clock.clone(), start_ns, &base_url, model, false));
    let sem = Arc::new(Semaphore::new(concurrency.max(1)));

    let mut handles = Vec::new();
    for req in workload.generate() {
        let permit = sem.clone().acquire_owned().await?;
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
}
