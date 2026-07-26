// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

const UI: &str = "simple";

fn has_streaming_metrics(r: &RunResult) -> bool {
    let json = r.artifacts.json();
    [
        "time_to_first_token",
        "inter_token_latency",
        "inter_chunk_latency",
        "time_to_second_token",
    ]
    .iter()
    .all(|k| json.get(k).map(|v| !v.is_null()).unwrap_or(false))
}

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
    // Busy hosts may cancel a few in-flight requests during shutdown.
    let count = r.artifacts.request_count();
    assert!(count >= 995.0, "Expected >=995 requests, got {count}");
    assert!(has_streaming_metrics(&r));
}

#[tokio::test]
async fn test_high_worker_count_streaming() {
    if cfg!(target_os = "windows") {
        return;
    }

    // The larger retry budget gives 100 concurrent workers time to register.
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
