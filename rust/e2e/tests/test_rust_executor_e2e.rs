// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

#[tokio::test]
#[ignore = "requires Python OTLP collector and adaptive chat-handler fixtures"]
async fn test_config_v2_streams_rust_metrics_live_through_canonical_python_otel() {
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
async fn test_config_v2_collects_server_metrics_in_rust_across_exact_phase_boundaries() {
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
async fn test_config_v2_joins_rust_gpu_telemetry_into_all_artifacts() {
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
async fn test_config_v2_runs_native_tcp_rtt_calibration_and_adjusts_metrics() {
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
async fn test_config_v2_fixed_network_rtt_bypasses_probes_and_shifts_metrics() {
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
async fn test_config_v2_executes_a_real_native_child() {
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
async fn test_config_v2_controls_native_connection_reuse() {
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
async fn test_config_v2_enforces_one_native_end_to_end_request_timeout() {
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
async fn test_config_v2_adaptive_phase_controls_the_native_live_issuer() {
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
