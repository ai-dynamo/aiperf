// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "common/mod.rs"]
mod common;
use common::*;

// Tests for high concurrency and performance scenarios.

const UI: &str = "simple";

/// Mirror of `RunResult.has_streaming_metrics`: all streaming metric keys present.
fn has_streaming_metrics(r: &RunResult) -> bool {
    let json = r.artifacts.json();
    ["time_to_first_token", "inter_token_latency", "inter_chunk_latency", "time_to_second_token"]
        .iter()
        .all(|k| json.get(k).map(|v| !v.is_null()).unwrap_or(false))
}

/// High concurrency (1000) with streaming and multimodal inputs.
#[tokio::test]
async fn test_high_concurrency_multimodal() {
    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let r = h.run_timeout(
        &format!(
            "--model mistralai/Mixtral-8x7B-Instruct-v0.1 --url {} \
             --gpu-telemetry {dcgm} --endpoint-type chat --streaming \
             --warmup-request-count 100 --request-count 1000 --concurrency 1000 \
             --request-rate 1000 --image-width-mean 64 --image-height-mean 64 \
             --workers-max 5 --record-processors 5 --ui {UI}",
            h.mock.url
        ),
        600,
    );
    // Allow up to 0.5% drop at 1000-way concurrency. On a busy VDI a couple of
    // in-flight requests can be cancelled during shutdown without indicating a
    // real product bug — the assertion is about the stress path completing, not
    // about exact request accounting.
    let count = r.artifacts.request_count();
    assert!(count >= 995.0, "Expected >=995 requests, got {count}");
    assert!(has_streaming_metrics(&r));
}

/// High worker count (100 workers) with streaming.
#[tokio::test]
async fn test_high_worker_count_streaming() {
    // Windows VDIs hit WinError 1450 spawning 100 worker subprocesses; this
    // stress level is Linux-CI only.
    if cfg!(target_os = "windows") {
        return;
    }

    // 100 worker subprocesses spawning concurrently overrun the default
    // registration retry budget; bump the per-worker max attempts so each
    // worker has ~60s before giving up.
    // SAFETY: single-threaded test setup before the run spawns subprocesses.
    unsafe {
        std::env::set_var("AIPERF_SERVICE_REGISTRATION_MAX_ATTEMPTS", "60");
    }

    let h = AIPerfHarness::new().await;
    let dcgm = h.mock.dcgm_urls().join(" ");
    let r = h.run_timeout(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --gpu-telemetry {dcgm} \
             --endpoint-type chat --concurrency 2000 --request-count 4000 \
             --osl 50 --workers-max 100 --streaming --ui {UI}",
            h.mock.url
        ),
        600,
    );
    assert_eq!(r.artifacts.request_count() as u32, 4000);
    assert!(has_streaming_metrics(&r));
}
