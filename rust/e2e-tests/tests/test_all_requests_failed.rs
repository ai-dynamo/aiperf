// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A run in which every inference request failed must not report success.
//!
//! The native coordinator decided `success: true, exit_code: 0` purely on
//! whether execution and report persistence returned `Ok`, so a run against a
//! closed port printed a full metrics report and exited zero — the exact
//! false-green the python engine guards against in `system_controller`.

mod common;
use common::*;

use aiperf_mock_server::config::MockServerConfig;
use serde_json::Value;

/// A port nothing is listening on, so every request fails before send.
const CLOSED_PORT_URL: &str = "http://127.0.0.1:59999";

fn counter_total(report: &Value, name: &str) -> f64 {
    report
        .pointer(&format!("/{name}/avg"))
        .or_else(|| report.get(name))
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

#[tokio::test]
async fn every_request_failing_at_connect_exits_non_zero() {
    // The harness only supplies the artifact dir and tokenizer here; the run
    // deliberately targets a closed port rather than the mock server.
    let h = AIPerfHarness::new_with(MockServerConfig {
        fast: true,
        no_tokenizer: true,
        ..MockServerConfig::default()
    })
    .await;

    let r = h.run(&format!(
        "--model gpt-4 --url {CLOSED_PORT_URL} --endpoint-type chat \
         --request-count 4 --concurrency 2 --workers-max 1 --ui simple"
    ));

    assert_ne!(
        r.exit_code, 0,
        "a run in which every request failed must exit non-zero; stdout: {}\nstderr: {}",
        r.stdout, r.stderr
    );

    // The report is still persisted so the per-request errors stay diagnosable.
    let report = r.artifacts.json();
    assert!(
        !report.is_null(),
        "the summary report must still be written for a fully failed run"
    );
    assert_eq!(
        counter_total(&report, "error_request_count"),
        4.0,
        "all four requests should be recorded as errors: {report}"
    );
    assert_eq!(
        counter_total(&report, "request_count"),
        0.0,
        "no request should be recorded as successful: {report}"
    );
}

#[tokio::test]
async fn fully_injected_server_errors_exit_non_zero() {
    // Same contract when the server answers but every answer is an error: the
    // requests reached a live endpoint, so this covers the post-send path.
    let h = AIPerfHarness::new_with(MockServerConfig {
        fast: true,
        no_tokenizer: true,
        random_seed: Some(7),
        error_rate: 100.0,
        error_status_codes: vec![503],
        ..MockServerConfig::default()
    })
    .await;

    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat \
         --request-count 8 --concurrency 2 --workers-max 1 \
         --random-seed 7 --ui simple",
        h.mock.url,
    ));

    assert_ne!(
        r.exit_code, 0,
        "a 100% server-error run must exit non-zero; stdout: {}\nstderr: {}",
        r.stdout, r.stderr
    );
    let report = r.artifacts.json();
    assert_eq!(
        counter_total(&report, "error_request_count"),
        8.0,
        "all eight requests should be recorded as errors: {report}"
    );
}

#[tokio::test]
async fn partially_failed_run_still_exits_zero() {
    // The guard must fire only when *zero* requests succeeded; a run with a mix
    // of successes and errors remains a successful run.
    let h = AIPerfHarness::new_with(MockServerConfig {
        fast: true,
        no_tokenizer: true,
        random_seed: Some(7),
        error_rate: 50.0,
        error_status_codes: vec![503],
        ..MockServerConfig::default()
    })
    .await;

    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat \
         --request-count 40 --concurrency 4 --workers-max 1 \
         --random-seed 7 --ui simple",
        h.mock.url,
    ));

    assert_eq!(
        r.exit_code, 0,
        "a partially failed run must still exit zero; stdout: {}\nstderr: {}",
        r.stdout, r.stderr
    );
    let report = r.artifacts.json();
    let errors = counter_total(&report, "error_request_count");
    let successes = counter_total(&report, "request_count");
    assert!(
        errors > 0.0 && successes > 0.0,
        "the 50% error rate should yield both successes and errors \
         (errors={errors}, successes={successes}): {report}"
    );
}
