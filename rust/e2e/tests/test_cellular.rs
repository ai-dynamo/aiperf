// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for cellular (multi-process) mode, reached through the
//! ordinary Python frontend via `--cells N`.
//!
//! `--cells N` sets `runtime.cells` in the projected protocol-v2 envelope; the
//! launched `aiperf-runner` becomes a controller that spawns `N`
//! `aiperf-runner --cell` subprocesses over a `(cell_id, cell_count)` partition of
//! the request budget and merges their records into one report. These tests prove
//! the whole path works from `aiperf profile` — not just the Rust internals — and
//! that an `N`-cell run reproduces the single-cell run's dataset-deterministic
//! metrics byte-for-byte through the full presentation pipeline.

mod common;
use common::*;

/// The dataset-deterministic metrics a cellular merge must reproduce exactly: they
/// depend only on the seeded synthetic dataset (input tokens) and the deterministic
/// mock's response (output tokens), not on wall-clock timing.
const DETERMINISTIC_METRICS: &[&str] = &["input_sequence_length", "output_sequence_length"];

/// `aiperf profile --cells 3` runs end-to-end and reports the full request budget.
///
/// Exercises the entire product path: Python projects `runtime.cells = 3`, the
/// controller spawns three cell subprocesses, each dispatches its slice over the
/// shared HTTP transport, ships its records back, and the controller merges them
/// into one report the Python frontend then presents.
#[tokio::test]
async fn test_cellular_run_from_python_frontend() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 60 --concurrency 6 --cells 3 --random-seed 42 \
         --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
         --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "cellular run failed: {}", r.stderr);
    assert_eq!(
        r.artifacts.request_count() as u32,
        60,
        "merged cellular report must carry every cell's records"
    );
    // Non-vacuous proof the CONTROLLER (multi-cell) path actually ran: the
    // cellular-heartbeat.json sidecar is written only by the controller after
    // aggregating the cells' shipped heartbeats. If `--cells` were stripped from the
    // wire (or otherwise inert) this run would be a plain single process and the
    // sidecar would be absent — so success()+request_count alone cannot mask it.
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "cellular run must emit the controller's cellular-heartbeat.json sidecar; \
         its absence means --cells did not reach the runner (single-process run)"
    );
}

/// A 3-cell run reproduces the 1-cell run's dataset-deterministic metrics exactly.
///
/// Same seed → same instance space; the 3-cell run partitions it across cells and
/// merges in global dispatch order, so the input/output sequence-length
/// distributions in the presented report must be byte-identical to the single-cell
/// run. Wall-clock metrics (throughput/latency) are intentionally not compared.
#[tokio::test]
async fn test_cellular_matches_single_cell() {
    let args = |cells: u32, url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 60 --concurrency 6 --cells {cells} --random-seed 42 \
             --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple"
        )
    };

    let h1 = AIPerfHarness::new().await;
    let baseline = h1.run(&args(1, &h1.mock.url));
    assert!(baseline.success(), "1-cell run failed: {}", baseline.stderr);

    let h3 = AIPerfHarness::new().await;
    let cellular = h3.run(&args(3, &h3.mock.url));
    assert!(cellular.success(), "3-cell run failed: {}", cellular.stderr);

    // Guard against a vacuous pass: prove the two runs really differ in topology.
    // The 3-cell run goes through the controller (emits cellular-heartbeat.json); the
    // 1-cell baseline is single-process (no sidecar). Without this, a stripped
    // `--cells` would make both runs 1-cell and byte-identical by construction.
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "3-cell run must go through the controller (cellular-heartbeat.json sidecar)"
    );
    assert!(
        baseline
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_none(),
        "1-cell baseline must be single-process (no cellular sidecar)"
    );

    let base_json = baseline.artifacts.json();
    let cell_json = cellular.artifacts.json();

    assert_eq!(
        baseline.artifacts.request_count() as u32,
        cellular.artifacts.request_count() as u32,
        "1-cell and 3-cell must dispatch the same request count"
    );

    for metric in DETERMINISTIC_METRICS {
        let base = &base_json[metric];
        let cell = &cell_json[metric];
        assert!(
            !base.is_null(),
            "baseline report missing dataset-deterministic metric {metric}"
        );
        assert_eq!(
            base, cell,
            "cellular {metric} diverged from the single-cell run: \
             1-cell={base}  3-cell={cell}"
        );
    }
}
