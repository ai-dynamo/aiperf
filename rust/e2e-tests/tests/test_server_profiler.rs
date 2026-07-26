// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for `endpoint.server_profiler`.

mod common;
use common::*;

#[tokio::test]
async fn server_profiler_starts_and_stops_around_profiling() {
    let h = AIPerfHarness::new().await;
    assert_eq!(h.mock.state.profiler_state().starts(), 0);
    assert_eq!(h.mock.state.profiler_state().stops(), 0);

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 8 --concurrency 2 --workers-max 1 \
         --warmup-request-count 3 \
         --server-profiler --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "server_profiler run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 8);
    assert_eq!(
        h.mock.state.profiler_state().starts(),
        1,
        "profiler must start once for the profiling phase"
    );
    assert_eq!(
        h.mock.state.profiler_state().stops(),
        1,
        "profiler must stop once after profiling drain"
    );
}

#[tokio::test]
async fn server_profiler_stop_failure_preserves_successful_run() {
    let h = AIPerfHarness::new().await;

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 6 --concurrency 2 --workers-max 1 \
         --server-profiler --server-profiler-stop-path /missing_stop \
         --ui simple",
        h.mock.url
    ));
    assert!(
        r.success(),
        "stop failure must warn without failing the benchmark: {}",
        r.stderr
    );
    assert_eq!(r.artifacts.request_count() as u32, 6);
    assert_eq!(h.mock.state.profiler_state().starts(), 1);
    assert_eq!(
        h.mock.state.profiler_state().stops(),
        0,
        "failed stop path must not increment the mock stop counter"
    );
}

#[tokio::test]
async fn cellular_server_profiler_is_controller_owned() {
    let h = AIPerfHarness::new().await;

    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 12 --concurrency 3 --cells 3 --workers-max 1 \
         --server-profiler --ui simple",
        h.mock.url
    ));
    assert!(
        r.success(),
        "cellular server_profiler run failed: {}",
        r.stderr
    );
    assert_eq!(r.artifacts.request_count() as u32, 12);
    assert_eq!(
        h.mock.state.profiler_state().starts(),
        1,
        "cellular profiler start must be controller-owned (one POST, not one per cell)"
    );
    assert_eq!(
        h.mock.state.profiler_state().stops(),
        1,
        "cellular profiler stop must be controller-owned"
    );
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "cellular path must remain active with server_profiler enabled"
    );
}
