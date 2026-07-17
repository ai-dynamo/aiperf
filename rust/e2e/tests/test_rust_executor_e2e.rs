// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
#[ignore = "requires Python OTLP collector and adaptive chat-handler fixtures"]
async fn test_otel_fixture_profile_completes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 80 --concurrency 2 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 80);
}

#[tokio::test]
#[ignore = "requires Python server-metrics handler and Parquet-reader fixtures"]
async fn test_server_metrics_fixture_profile_completes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --warmup-request-count 2 --request-count 3 --concurrency 1 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 3);
}

#[tokio::test]
#[ignore = "requires a Python DCGM scrape fixture with monotonic energy counters"]
async fn test_gpu_telemetry_fixture_profile_completes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 4 --concurrency 2 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 4);
}

#[tokio::test]
#[ignore = "requires the Python network-latency probe-target harness"]
async fn test_network_latency_fixture_profile_completes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 2 --concurrency 1 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 2);
}

#[tokio::test]
#[ignore = "requires Python fixed-RTT configuration and adaptive chat fixtures"]
async fn test_fixed_network_rtt_fixture_profile_completes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 2 --concurrency 1 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 2);
}

#[tokio::test]
#[ignore = "requires Python request-capture, tokenizer, and dataset fixtures"]
async fn test_request_capture_fixture_profile_completes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 4 --concurrency 2 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 4);
}

#[tokio::test]
#[ignore = "requires a Python handler that captures client peer ports"]
async fn test_connection_reuse_fixture_profile_completes() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 3 --concurrency 1 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert_eq!(r.artifacts.request_count() as u32, 3);
}

#[tokio::test]
#[ignore = "requires a Python handler with controllable response delay"]
async fn test_request_timeout_fixture_fails_after_deadline() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 1 --concurrency 1 --request-timeout-seconds 0.01 --ui none",
        h.mock.url
    ));
    assert!(!r.success());
}

#[tokio::test]
#[ignore = "requires a Python handler that records peak in-flight concurrency"]
async fn test_adaptive_fixture_profile_sustains_load() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --benchmark-duration 8 --concurrency 2 --ui none",
        h.mock.url
    ));
    assert!(r.success());
    assert!(r.artifacts.request_count() > 10.0);
}
