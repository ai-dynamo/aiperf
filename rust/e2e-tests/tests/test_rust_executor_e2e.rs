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

/// `--request-timeout-seconds` shorter than the server's TTFT errors every request.
///
/// Paired with [`test_requests_without_timeout_succeed_at_same_ttft`], which runs the
/// identical workload against the identical server with the flag removed. Both halves
/// are needed: asserting only that requests error would also pass if the fixture were
/// broken (an unreachable port errors just as thoroughly).
#[tokio::test]
async fn test_request_timeout_errors_requests_past_the_deadline() {
    // TTFT well above the deadline, but low enough that the control run passes
    // comfortably. Jitter off so the margin is not probabilistic.
    let h = AIPerfHarness::new_with(tuned_mock_config(TIMEOUT_TTFT_MS, TIMEOUT_ITL_MS)).await;
    let r = h.run(&timeout_probe_args(&h, "--request-timeout-seconds 0.5"));

    // The run itself completes: a timed-out request is a recorded error, not a
    // harness failure. Only the per-request outcome differs.
    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    let export = r.artifacts.json();
    assert_eq!(
        export["error_request_count"]["sum"].as_f64(),
        Some(f64::from(TIMEOUT_PROBE_REQUESTS)),
        "every request should have exceeded the deadline; export: {export:#}"
    );
    assert!(
        export["request_count"].is_null(),
        "no request should have succeeded, got request_count: {}",
        export["request_count"]
    );
}

/// The control for [`test_request_timeout_errors_requests_past_the_deadline`].
#[tokio::test]
async fn test_requests_without_timeout_succeed_at_same_ttft() {
    let h = AIPerfHarness::new_with(tuned_mock_config(TIMEOUT_TTFT_MS, TIMEOUT_ITL_MS)).await;
    let r = h.run(&timeout_probe_args(&h, ""));

    assert_eq!(r.exit_code, 0, "stderr: {}", r.stderr);
    let export = r.artifacts.json();
    assert_eq!(
        export["request_count"]["sum"].as_f64(),
        Some(f64::from(TIMEOUT_PROBE_REQUESTS)),
        "the same workload must succeed without a deadline; export: {export:#}"
    );
    assert!(
        export["error_request_count"].is_null(),
        "unexpected errors without a deadline: {}",
        export["error_request_count"]
    );
}

const TIMEOUT_TTFT_MS: f64 = 1500.0;
const TIMEOUT_ITL_MS: f64 = 5.0;
const TIMEOUT_PROBE_REQUESTS: u32 = 2;

/// The workload shared by the timeout pair; `extra` is the only difference.
fn timeout_probe_args(h: &AIPerfHarness, extra: &str) -> String {
    format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --streaming \
         --synthetic-input-tokens-mean 8 --output-tokens-mean 2 \
         --request-count {TIMEOUT_PROBE_REQUESTS} --concurrency 1 {extra} --ui none",
        h.mock.url
    )
}

