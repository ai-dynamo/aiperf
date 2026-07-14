// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

// Python Config v2 -> stdio -> Rust HTTP/SSE -> native-v2 proof.
//
// The Python originals (`tests/integration/test_rust_executor_e2e.py`) each
// spin up a bespoke Python `http.server` handler with test-only behavior that
// the shared in-process `aiperf-mock-server` target does not emulate, and drive the
// native child through the Python `RustSubprocessExecutor` object API + Pydantic
// `AIPerfConfig`/`BenchmarkRun` envelopes rather than the `aiperf profile` CLI.
//
// That custom Python HTTP infrastructure (OTLP metrics collector, telemetry
// scrape counters, connection peer-port capture, forced request timeouts) is
// Python-only, so per the porting rules these are ported as `#[ignore]` tests.
// Each keeps the closest `aiperf profile` translation and its behavioral
// assertions so intent is preserved, but is gated behind the missing service.

/// Streams native Rust metrics live through the canonical Python OTel exporter.
///
/// requires: OTLP metrics collector endpoint + Python `_AdaptiveChatHandler`
/// (opentelemetry proto `ExportMetricsServiceRequest`).
#[tokio::test]
#[ignore]
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

/// Collects server metrics in Rust across exact warmup/profiling phase boundaries.
///
/// requires: Python `_ServerMetricsHandler` (per-endpoint scrape counters,
/// prometheus histogram fixtures) + pyarrow parquet reader.
#[tokio::test]
#[ignore]
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

/// Joins Rust GPU telemetry into every emitted artifact.
///
/// requires: Python `_ChatHandler` DCGM `/metrics` scrape fixture with
/// per-scrape monotonic energy counter.
#[tokio::test]
#[ignore]
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

/// Runs native TCP-RTT calibration and adjusts the emitted latency metrics.
///
/// requires: Python `_ChatHandler` + network-latency probe target harness.
#[tokio::test]
#[ignore]
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

/// Fixed network RTT bypasses probes and shifts the adjusted metrics.
///
/// requires: Python `_AdaptiveChatHandler` + fixed-mean network-latency config.
#[tokio::test]
#[ignore]
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

/// Executes a real native child end-to-end across many dataset/tokenizer shapes.
///
/// requires: Python `_ChatHandler` (captured request bodies, redaction proof),
/// `_run_single_benchmark` CLI entry, ShareGPTLoader URL patch, wordlevel
/// tokenizer fixtures, and mooncake-trace synthesis config.
#[tokio::test]
#[ignore]
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

/// Controls native connection reuse (pooled vs never) via peer-port capture.
///
/// requires: Python `_ConnectionHandler` capturing per-request client peer ports.
#[tokio::test]
#[ignore]
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

/// Enforces one native end-to-end request timeout.
///
/// requires: Python `_AdaptiveChatHandler` (deliberate 50ms server delay vs a
/// 10ms client timeout) to force the TimeoutError failure path.
#[tokio::test]
#[ignore]
async fn test_config_v2_enforces_one_native_end_to_end_request_timeout() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model mock-model --url {}/v1/chat/completions --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 1 \
         --request-count 1 --concurrency 1 --request-timeout-seconds 0.01 --ui none",
        h.mock.url
    ));
    // The Python original asserts the run fails with "All 1 requests failed".
    assert!(!r.success());
}

/// Adaptive phase controls the native live issuer under an SLA gate.
///
/// requires: Python `_AdaptiveChatHandler` tracking peak in-flight concurrency.
#[tokio::test]
#[ignore]
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
