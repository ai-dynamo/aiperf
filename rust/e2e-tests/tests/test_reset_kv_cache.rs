// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for `endpoint.reset_kv_cache`.

mod common;
use common::*;

fn mock_chat_requests(state: &aiperf_mock_server::AppState) -> u64 {
    state
        .recorder
        .metrics
        .aiperf
        .REQUESTS_TOTAL
        .with_label_values(&["/v1/chat/completions", "POST", "200"])
        .get()
}

#[tokio::test]
async fn reset_kv_cache_runs_once_per_sweep_cell() {
    let h = AIPerfHarness::new().await;
    assert_eq!(h.mock.state.prefix_cache_generation(), 0);

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --concurrency 1,2 --request-count 4 --workers-max 1 \
         --reset-kv-cache --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "sweep with reset_kv_cache failed: {}", r.stderr);
    assert_eq!(
        h.mock.state.prefix_cache_generation(),
        2,
        "each sweep cell must POST reset exactly once"
    );
}

#[tokio::test]
async fn reset_kv_cache_failure_aborts_before_warmup() {
    let h = AIPerfHarness::new().await;

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 8 --concurrency 2 --workers-max 1 \
         --warmup-request-count 3 \
         --reset-kv-cache --reset-kv-cache-path /missing_reset \
         --ui simple",
        h.mock.url
    ));
    assert!(!r.success(), "reset failure must abort the profile run");
    assert!(
        r.stderr.contains("reset_kv_cache") || r.stderr.contains("reset"),
        "stderr should mention the failed reset hook: {}",
        r.stderr
    );
    assert_eq!(h.mock.state.prefix_cache_generation(), 0);
    assert_eq!(
        mock_chat_requests(&h.mock.state),
        0,
        "failed reset must not admit warmup or profiling traffic"
    );
}

#[tokio::test]
async fn cellular_reset_kv_cache_runs_once_on_controller() {
    let h = AIPerfHarness::new().await;

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 12 --concurrency 3 --cells 3 --workers-max 1 \
         --reset-kv-cache --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "cellular reset run failed: {}", r.stderr);
    assert_eq!(
        h.mock.state.prefix_cache_generation(),
        1,
        "cellular runs must reset once on the controller, not per worker cell"
    );
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "cellular path must remain active with reset_kv_cache enabled"
    );
}
