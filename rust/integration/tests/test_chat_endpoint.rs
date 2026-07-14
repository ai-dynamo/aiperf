// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#[path = "common/mod.rs"]
mod common;
use common::*;

// Tests for /v1/chat/completions endpoint.

/// IntegrationTestDefaults (from tests/integration/conftest.py).
const WORKERS_MAX: u32 = 1;
const CONCURRENCY: u32 = 2;
const REQUEST_COUNT: u32 = 10;
const UI: &str = "simple";

/// Mirror of `RunResult.has_streaming_metrics`: all streaming metric keys present.
fn has_streaming_metrics(r: &RunResult) -> bool {
    let json = r.artifacts.json();
    ["time_to_first_token", "inter_token_latency", "inter_chunk_latency", "time_to_second_token"]
        .iter()
        .all(|k| json.get(k).map(|v| !v.is_null()).unwrap_or(false))
}

/// Basic non-streaming chat completion.
#[tokio::test]
async fn test_basic_chat() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model microsoft/phi-4 --url {} --endpoint-type chat \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
}

/// Streaming chat completion with metrics validation.
#[tokio::test]
async fn test_streaming_chat() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model Qwen/Qwen2.5-32B-Instruct --url {} --endpoint-type chat --streaming \
         --request-count {REQUEST_COUNT} --concurrency {CONCURRENCY} \
         --workers-max {WORKERS_MAX} --ui {UI}",
        h.mock.url
    ));
    assert_eq!(r.artifacts.request_count() as u32, REQUEST_COUNT);
    assert!(has_streaming_metrics(&r));
}
