// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Product-level check that `--steady-state` emits a closed-loop steady-state
//! summary in the `*_aiperf.json` artifact for a concurrency-target run, and
//! that it is absent (no behavior change) when the flag is not set.

mod common;
use common::*;

/// A concurrency-target run with `--steady-state` emits a steady-state block
/// whose threshold matches `ceil(0.8 * concurrency)` and whose metrics are a
/// non-empty subset scoped to the detected saturated window.
#[tokio::test]
async fn test_steady_state_emitted_for_concurrency_run() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --concurrency 4 --request-count 400 --steady-state --ui simple",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "run must succeed");

    let json = r.artifacts.json();
    let steady = &json["steady_state"];
    assert!(
        steady.is_object(),
        "steady_state block must be emitted, got: {steady}"
    );

    // Threshold is ceil(0.8 * 4) = 4. Measured in-flight overlap (from record
    // start/end timestamps) is at least the threshold inside the window.
    assert_eq!(steady["threshold_concurrency"].as_u64(), Some(4));
    let peak = steady["peak_concurrency"]
        .as_u64()
        .expect("peak_concurrency");
    assert!(
        peak >= 4,
        "peak concurrency must reach the threshold, got {peak}"
    );

    // The window is a real half-open interval.
    let start = steady["window_start_ns"].as_i64().expect("window_start_ns");
    let end = steady["window_end_ns"].as_i64().expect("window_end_ns");
    assert!(end > start, "window end must exceed start ({start}..{end})");
    assert!(
        steady["duration_s"].as_f64().unwrap_or(0.0) > 0.0,
        "window duration must be positive"
    );

    // The steady metrics reuse the ordinary per-metric summary shape and are a
    // subset scoped to the window, so core inference metrics are present.
    let metrics = &steady["metrics"];
    assert!(metrics.is_object(), "steady metrics must be an object");
    assert!(
        metrics.get("request_latency").is_some(),
        "steady metrics must carry request_latency, got keys: {:?}",
        metrics.as_object().map(|m| m.keys().collect::<Vec<_>>())
    );
    assert!(
        metrics.get("request_throughput").is_some(),
        "steady metrics must carry request_throughput"
    );
}

/// Without `--steady-state` the artifact carries no steady-state block, proving
/// the feature is fully gated (no behavior change when disabled).
#[tokio::test]
async fn test_steady_state_absent_when_disabled() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat --concurrency 4 --request-count 100 --ui simple",
        h.mock.url
    ));
    assert_eq!(r.exit_code, 0, "run must succeed");
    assert!(
        r.artifacts.json()["steady_state"].is_null(),
        "steady_state must be absent when the flag is not set"
    );
}
