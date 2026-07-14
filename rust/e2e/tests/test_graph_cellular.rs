// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for graph-mode cellular: a `dag_jsonl` graph run through the
//! ordinary Python frontend with `--cells N`.
//!
//! `--cells 3` on a graph dataset makes the launched `aiperf-runner` a controller that
//! spawns three `aiperf-runner --cell` children; each partitions the trace instances by
//! `instance_ordinal % cell_count` (PartitionedGraphTraceSource), runs its interleaved
//! slice, and ships its graph records; the controller concatenation-merges them (records
//! carry local per-cell indices, wall-clock ordered) into one report. This proves the
//! whole graph-cellular path from `aiperf profile`, not just the Rust internals.

mod common;
use common::*;

/// A multi-root single-turn DAG fixture: cheap to run many instances of, so a
/// `--num-conversations` sweep gives every cell a non-empty interleaved slice.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/dag/multi_root_single_turn.dag.jsonl"
);

/// `aiperf profile --cells 3` over a graph dataset runs end-to-end and merges every
/// cell's graph records into one report.
#[tokio::test]
async fn test_graph_cellular_from_python_frontend() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 3 --random-seed 7 --ui simple",
            h.mock.url
        ),
        120,
    );
    assert!(r.success(), "graph cellular run failed: {}", r.stderr);

    // Non-vacuous proof the CONTROLLER (multi-cell) path ran: only the controller emits
    // the cellular-heartbeat.json sidecar (after aggregating the cells' heartbeats). If
    // `--cells` were inert this would be a single-process graph run and the sidecar absent.
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "graph --cells 3 must go through the controller (cellular-heartbeat.json sidecar)"
    );

    // The concatenation-merged report exists and carries the cells' graph records.
    let json = r.artifacts.json();
    assert!(!json.is_null(), "graph cellular merged report must exist");
    assert!(
        r.artifacts.request_count() > 0.0,
        "merged graph report must carry records from the cells"
    );

    // Prove the partition covers the FULL trace set (not a subset stuck on one cell):
    // a 1-cell run of the same seeded config dispatches the same conversations, so its
    // total record count and input-token distribution must match the 3-cell run
    // (deterministic-per-topology — the trace set is identical, only cell ownership and
    // merge order differ). The 1-cell run takes the single-process path (no controller
    // sidecar), so it also confirms the sidecar above is a genuine multi-cell signal.
    let h1 = AIPerfHarness::new().await;
    let baseline = h1.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 1 --random-seed 7 --ui simple",
            h1.mock.url
        ),
        120,
    );
    assert!(baseline.success(), "1-cell graph run failed: {}", baseline.stderr);
    assert!(
        baseline
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_none(),
        "a 1-cell graph run must take the single-process path (no controller sidecar)"
    );
    assert_eq!(
        r.artifacts.request_count() as u64,
        baseline.artifacts.request_count() as u64,
        "3-cell graph run must dispatch the same total record count as 1-cell (same trace set)"
    );
    let cell_isl = &json["input_sequence_length"];
    let base_isl = &baseline.artifacts.json()["input_sequence_length"];
    assert!(!base_isl.is_null(), "baseline must report input_sequence_length");
    assert_eq!(
        cell_isl, base_isl,
        "3-cell graph run must reproduce the 1-cell input-token distribution \
         (the partition covers the full trace set): 1-cell={base_isl} 3-cell={cell_isl}"
    );
}
