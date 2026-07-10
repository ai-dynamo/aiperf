// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The skeleton run loop: closed-loop concurrency over a synthetic workload,
//! dispatched through [`HttpSink`] and measured by the shared `TraceCollector`.

use std::sync::Arc;
use std::time::Instant;

use tokio::sync::Semaphore;

use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use loadgen_core::sink::{RequestObserver, RequestSink};

use crate::workload::SkeletonWorkload;
use aiperf_core::http_sink::HttpSink;
use aiperf_core::observer::CollectorObserver;

/// Run a concurrency-`concurrency` load test of `workload` against `base_url`
/// for `model`, returning the aggregated report.
pub async fn run(
    base_url: String,
    model: String,
    workload: SkeletonWorkload,
    concurrency: usize,
) -> anyhow::Result<TraceSimulationReport> {
    let start = Instant::now();
    let obs = Arc::new(CollectorObserver::new(start, false));
    let sink = Arc::new(HttpSink::new(base_url, model, start));
    let sem = Arc::new(Semaphore::new(concurrency.max(1)));

    let mut handles = Vec::new();
    for req in workload.generate() {
        let permit = sem.clone().acquire_owned().await?;
        obs.on_arrival(
            req.uuid,
            obs.now_ms(),
            req.input_length,
            req.max_output_tokens,
        );
        let obs2 = obs.clone();
        let sink2 = sink.clone();
        handles.push(tokio::spawn(async move {
            let _permit = permit;
            let uuid = req.uuid;
            if let Err(e) = sink2.dispatch(req, obs2.as_ref()).await {
                obs2.on_terminal(uuid, ReplayTerminalStatus::Failed);
                tracing::warn!("request {uuid} failed: {e}");
            }
        }));
    }
    for h in handles {
        let _ = h.await;
    }

    Ok(obs.finish(obs.now_ms()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workload::SkeletonWorkload;

    #[tokio::test]
    async fn e2e_reports_finite_metrics() {
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
    }
}
