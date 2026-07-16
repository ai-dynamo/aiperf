// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for cellular (multi-process) mode, reached through the
//! ordinary Python frontend via `--cells N`.
//!
//! `--cells N` sets `runtime.cells` in the projected protocol-v2 envelope; the
//! launched `aiperf-runner` becomes a controller that (via the `LocalLauncher`)
//! spawns `N` `aiperf-runner --cell` subprocesses over a `(cell_id, cell_count)`
//! partition of the request budget. Cells fetch their sliced envelope over the
//! **velo** transport, await the controller's synchronized START, run their slice,
//! ship their records over velo, and the controller merges them into one report.
//! These tests prove the whole path works from `aiperf profile` — not just the
//! Rust internals — and that an `N`-cell run reproduces the single-cell run's
//! dataset-deterministic metrics byte-for-byte through the full presentation
//! pipeline.
//!
//! Requires the launched `aiperf-runner` (`AIPERF_EXEC_BIN`) to include the
//! `velo` cell transport — it is in the default runner build, so a default
//! `cargo test` run drives it; a lean `--no-default-features` runner fails closed.

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

/// A seedless `--cells N` run auto-derives one shared seed and still runs multi-cell.
///
/// Previously rejected (cellular required an explicit `--random-seed`). The controller
/// now derives a single seed from the run identity and injects it into every cell, so
/// all cells compose the same dataset space without the operator supplying one.
#[tokio::test]
async fn test_cellular_autoderives_seed_when_absent() {
    let h = AIPerfHarness::new().await;
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --request-count 30 --concurrency 6 --cells 3 \
         --synthetic-input-tokens-mean 64 --output-tokens-mean 4 --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "seedless cellular run failed: {}", r.stderr);
    assert_eq!(r.artifacts.request_count() as u32, 30);
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "a seedless cellular run must still go multi-cell (auto-derived shared seed), \
         not fall back to single-process"
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

/// The sorted multiset of each profiling record's dataset-deterministic
/// `(input_sequence_length, output_sequence_length)` projection, read from
/// `profile_export.jsonl`. Wall-clock fields (timing) differ run to run, but this
/// projection depends only on the seeded dataset + the deterministic mock, so it is a
/// stable per-record key for a row-SET comparison across topologies.
fn record_isl_osl_multiset(r: &RunResult) -> Vec<(i64, i64)> {
    let metric = |record: &serde_json::Value, tag: &str| -> i64 {
        record["metrics"][tag]["value"].as_f64().unwrap_or(f64::NAN) as i64
    };
    let mut rows: Vec<(i64, i64)> = r
        .artifacts
        .jsonl()
        .iter()
        .filter(|record| record["metadata"]["benchmark_phase"] == "profiling")
        .map(|record| {
            (
                metric(record, "input_sequence_length"),
                metric(record, "output_sequence_length"),
            )
        })
        .collect();
    rows.sort_unstable();
    rows
}

/// Stage D: a `--cells N` run with per-record artifacts enabled (`--export-level raw`)
/// EMITS the merged per-record files in the run artifact dir, and their row SET equals
/// the single-cell run's for the same seed.
///
/// Each cell runs its ordinary execute path with a controller-local `temp_root/cell-{id}`
/// dir as its artifact_dir, so it writes its own `profile_export.jsonl` / `_raw.jsonl`
/// there (streaming lane under the default exact-fold path); the controller concatenates
/// them into the real artifact dir at finalize. Completion order is accepted, so the row
/// SET — not byte order — is compared, exactly as the in-process sharded concat is
/// (`shard_artifacts::per_shard_concat_matches_batch_over_union`).
#[tokio::test]
async fn test_cellular_emits_per_record_artifacts_matching_single_cell() {
    let args = |cells: u32, url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 60 --concurrency 6 --cells {cells} --random-seed 42 \
             --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --export-level raw --ui simple"
        )
    };

    let h1 = AIPerfHarness::new().await;
    let baseline = h1.run(&args(1, &h1.mock.url));
    assert!(baseline.success(), "1-cell run failed: {}", baseline.stderr);

    let h3 = AIPerfHarness::new().await;
    let cellular = h3.run(&args(3, &h3.mock.url));
    assert!(cellular.success(), "3-cell run failed: {}", cellular.stderr);

    // Topology guard: the 3-cell run went through the controller; the baseline did not.
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

    // The controller actually emitted the per-record files (Stage D): before this they
    // were written only into the discarded scratch tree and never reached the run dir.
    assert!(
        cellular
            .artifacts
            .find_file("**/profile_export.jsonl")
            .is_some(),
        "3-cell run must emit the merged profile_export.jsonl (Stage D concat)"
    );
    assert!(
        cellular
            .artifacts
            .find_file("**/profile_export_raw.jsonl")
            .is_some(),
        "3-cell run must emit the merged profile_export_raw.jsonl (Stage D concat)"
    );

    let base_records = record_isl_osl_multiset(&baseline);
    let cell_records = record_isl_osl_multiset(&cellular);
    assert!(
        !base_records.is_empty(),
        "1-cell baseline must emit per-record rows"
    );
    assert_eq!(
        base_records.len(),
        cell_records.len(),
        "1-cell and 3-cell must emit the same number of per-record rows"
    );
    assert_eq!(
        base_records, cell_records,
        "3-cell merged per-record row SET must equal the single-cell run's for the same seed"
    );

    // Raw records must also be present and the same count (byte-append concat).
    assert_eq!(
        baseline.artifacts.raw_records().len(),
        cellular.artifacts.raw_records().len(),
        "1-cell and 3-cell must emit the same number of raw per-record rows"
    );

    // inputs.json (always-on per rust_wire) must be emitted by the controller (Stage D)
    // and be IDENTICAL to the single-cell run's: every cell generates the same full-dataset
    // inputs.json from the shared seed, so the controller copies one cell's copy verbatim.
    // Before Stage D it was silently dropped (written only into the discarded scratch tree).
    let base_inputs = baseline.artifacts.inputs();
    let cell_inputs = cellular.artifacts.inputs();
    assert!(
        !base_inputs.is_null(),
        "1-cell baseline must emit inputs.json"
    );
    assert!(
        !cell_inputs.is_null(),
        "3-cell run must emit inputs.json (Stage D controller copy)"
    );
    assert_eq!(
        base_inputs, cell_inputs,
        "3-cell inputs.json must equal the single-cell run's (identical full-dataset doc)"
    );
}

/// Stage C: a `--cells N` metrics-only run with the DEFAULT exact-fold path (each cell
/// folds its records into its own EXACT store and ships that folded store to the
/// controller, which appends them) reproduces the same run on the legacy cellular
/// RETAIN path (`AIPERF_RUNTIME_EXACT_FOLD=0`, cells ship raw record `Vec`s merged in
/// global dispatch order) for the dataset-deterministic metrics.
///
/// Both runs go through the controller (multi-cell), differing ONLY in whether the
/// cells shipped a folded store or a record `Vec`. The compared metrics are the
/// INTEGER input/output sequence lengths, whose summary stats (avg/percentiles/min/max/
/// std) are all order-independent (integer sums are exact regardless of summation
/// order; std is computed over sorted values), so the concatenated store-merge is
/// byte-identical to the global-order record-merge for them — while float metrics
/// (throughput/latency, intentionally not compared) may drift a few ULPs.
#[tokio::test]
async fn test_cellular_exact_fold_matches_retain() {
    let args = |url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 60 --concurrency 6 --cells 3 --random-seed 42 \
             --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple"
        )
    };

    // Default engine: exact-fold — each cell ships a folded CellMessage::StorePartition.
    let h_fold = AIPerfHarness::new().await;
    let folded = h_fold.run(&args(&h_fold.mock.url));
    assert!(
        folded.success(),
        "exact-fold cellular run failed: {}",
        folded.stderr
    );

    // Legacy retain — each cell ships its raw record Vec, merged in global order.
    let h_retain = AIPerfHarness::new().await;
    let retained = h_retain.run_env(
        &args(&h_retain.mock.url),
        &[("AIPERF_RUNTIME_EXACT_FOLD", "0")],
    );
    assert!(
        retained.success(),
        "retain cellular run failed: {}",
        retained.stderr
    );

    // Both must have gone through the controller (multi-cell), else this is vacuous.
    for (label, run) in [("exact-fold", &folded), ("retain", &retained)] {
        assert!(
            run.artifacts
                .find_file("**/cellular-heartbeat.json")
                .is_some(),
            "{label} 3-cell run must go through the controller (cellular-heartbeat.json sidecar)"
        );
    }

    assert_eq!(
        folded.artifacts.request_count() as u32,
        retained.artifacts.request_count() as u32,
        "exact-fold and retain cellular runs must dispatch the same request count"
    );

    let fold_json = folded.artifacts.json();
    let retain_json = retained.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        let fold = &fold_json[metric];
        let retain = &retain_json[metric];
        assert!(
            !retain.is_null(),
            "retain report missing dataset-deterministic metric {metric}"
        );
        assert_eq!(
            fold, retain,
            "cellular exact-fold {metric} diverged from the retain path: \
             fold={fold}  retain={retain}"
        );
    }
}

/// Tier T1 (bounded-memory horizontal scale): a `--cells N` run with SKETCH metric
/// storage (`AIPERF_METRICS_SKETCH=1`) reproduces the single-cell sketch run's EXACT
/// aggregates. Each cell folds its records into a per-`(phase, tag)` t-digest store
/// that retains no rows and ships that folded store (`CellMessage::StorePartition`,
/// the same wire form exact-fold uses); the controller merges the sketches
/// associatively (`merge_store_partitions` → `append_store` → t-digest merge). Counts,
/// sums, and extrema stay exact across the merge (exact Welford aggregates + anchored
/// min/max), so `request_count` and the INTEGER ISL/OSL `avg`/`min`/`max` match the
/// single-cell run exactly. Percentiles are t-digest-approximate (a merged digest
/// differs slightly from a single-ingestion digest) and are intentionally NOT compared
/// for equality — the exact-aggregate contract is what tier T1 guarantees.
///
/// This proves the whole path from `aiperf profile --cells N --sketch`: Python projects
/// `metrics.sketch` + `runtime.cells`, each cell folds-and-drops into a bounded sketch
/// (never retaining its record stream), ships the bounded store, and the controller
/// merges O(cells) sketches into one report — the change that unblocked sketch cellular.
#[tokio::test]
async fn test_cellular_sketch_matches_single_cell() {
    let args = |cells: u32, url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 60 --concurrency 6 --cells {cells} --random-seed 42 \
             --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple"
        )
    };
    let sketch_env = [("AIPERF_METRICS_SKETCH", "1")];

    let h1 = AIPerfHarness::new().await;
    let baseline = h1.run_env(&args(1, &h1.mock.url), &sketch_env);
    assert!(
        baseline.success(),
        "1-cell sketch run failed: {}",
        baseline.stderr
    );

    let h3 = AIPerfHarness::new().await;
    let cellular = h3.run_env(&args(3, &h3.mock.url), &sketch_env);
    assert!(
        cellular.success(),
        "3-cell sketch run failed: {}",
        cellular.stderr
    );

    // Topology guard: the 3-cell run went through the controller; the baseline did not
    // — otherwise the merge is never exercised and the parity check is vacuous.
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "3-cell sketch run must go through the controller (cellular-heartbeat.json sidecar)"
    );
    assert!(
        baseline
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_none(),
        "1-cell baseline must be single-process (no cellular sidecar)"
    );

    // The record total survives fold-and-clear + ship + merge (the store carries it;
    // a sketch store's row count is 0).
    assert_eq!(
        baseline.artifacts.request_count() as u32,
        cellular.artifacts.request_count() as u32,
        "request_count must be exact across the cellular sketch merge"
    );

    // Exact aggregates: the INTEGER ISL/OSL avg/min/max are order-independent, so a
    // merged sketch reproduces the single-cell sketch exactly (percentiles are not).
    let base = baseline.artifacts.json();
    let cell = cellular.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        for field in ["avg", "min", "max"] {
            let b = &base[metric][field];
            let c = &cell[metric][field];
            assert!(
                !b.is_null(),
                "1-cell sketch report missing {metric}.{field}"
            );
            assert_eq!(
                b, c,
                "cellular sketch {metric}.{field} diverged from the single-cell sketch: \
                 base={b}  cell={c}"
            );
        }
    }
}

/// Tier T2 (hierarchical tree-merge): with `AIPERF_CELL_AGG_FANOUT` set, the controller
/// routes the cells through an aggregator tier — `M = ceil(cells / fanout)` extra
/// `aiperf-runner --aggregator` processes, each collecting its round-robin subtree of
/// cells' folded stores, merging them, and shipping ONE merged store up — instead of the
/// flat star where all cells ship to the controller. Because the fold-mode store merge
/// (`merge_store_partitions` → t-digest merge) is associative and deterministic-at-topology,
/// the tree-merged report is BYTE-IDENTICAL to the flat-star report on the dataset-
/// deterministic metrics. Sketch storage (a fold path) so the cells ship a `StorePartition`
/// the aggregator can merge; a base port well clear of the default avoids any collision.
///
/// This exercises the full multi-process tree end-to-end (6 cells + 2 aggregators +
/// controller) from `aiperf profile`, proving the single-controller fan-in ceiling is
/// liftable by inserting aggregator tiers without changing the report.
#[tokio::test]
async fn test_cellular_tree_merge_matches_flat_star() {
    let args = |url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 60 --concurrency 6 --cells 6 --random-seed 42 \
             --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple"
        )
    };
    let sketch = ("AIPERF_METRICS_SKETCH", "1");

    // Flat star: 6 cells ship folded sketches straight to the controller.
    let h_flat = AIPerfHarness::new().await;
    let flat = h_flat.run_env(&args(&h_flat.mock.url), &[sketch]);
    assert!(flat.success(), "flat cellular run failed: {}", flat.stderr);

    // Tree: 6 cells → 2 aggregators (fanout 3) → controller. A base port clear of the
    // 9700 default so a concurrent test never collides on the fixed aggregator port.
    let h_tree = AIPerfHarness::new().await;
    let tree = h_tree.run_env(
        &args(&h_tree.mock.url),
        &[
            sketch,
            ("AIPERF_CELL_AGG_FANOUT", "3"),
            ("AIPERF_CELL_AGG_BASE_PORT", "9764"),
        ],
    );
    assert!(tree.success(), "tree cellular run failed: {}", tree.stderr);

    // Both went through the controller (multi-cell), else the merge is never exercised.
    for (label, run) in [("flat", &flat), ("tree", &tree)] {
        assert!(
            run.artifacts
                .find_file("**/cellular-heartbeat.json")
                .is_some(),
            "{label} run must go through the controller (cellular-heartbeat.json sidecar)"
        );
    }
    assert_eq!(
        flat.artifacts.request_count() as u32,
        tree.artifacts.request_count() as u32,
        "flat and tree must dispatch the same request count through the aggregator tier"
    );

    // The tree-merged report equals the flat-star report on the dataset-deterministic
    // metrics (integer ISL/OSL: avg/percentiles/min/max/std are order-independent, so the
    // associative aggregator-tier merge reproduces the flat star exactly at this size).
    let flat_json = flat.artifacts.json();
    let tree_json = tree.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        let a = &flat_json[metric];
        let b = &tree_json[metric];
        assert!(
            !a.is_null(),
            "flat report missing deterministic metric {metric}"
        );
        assert_eq!(
            a, b,
            "tier-T2 tree merge {metric} diverged from the flat star: flat={a}  tree={b}"
        );
    }
}

/// Tier T3 (master-less barrier-free start): `AIPERF_CELL_BARRIER_FREE=1` makes the
/// controller trigger START immediately instead of gathering all N cell registrations
/// first (the O(N) fan-in rendezvous). Start timing does not affect the dataset-
/// deterministic metrics, so a barrier-free run reproduces the synchronized-start run's
/// deterministic metrics exactly — proving the rendezvous is removable for unbounded scale.
#[tokio::test]
async fn test_cellular_barrier_free_matches_synchronized() {
    let args = |url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 48 --concurrency 6 --cells 4 --random-seed 42 \
             --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple"
        )
    };
    let sketch = ("AIPERF_METRICS_SKETCH", "1");

    // Synchronized (default): the controller waits for all 4 cells to register.
    let h_sync = AIPerfHarness::new().await;
    let sync = h_sync.run_env(&args(&h_sync.mock.url), &[sketch]);
    assert!(
        sync.success(),
        "synchronized-start run failed: {}",
        sync.stderr
    );

    // Barrier-free: START triggers immediately; cells start on their own registration.
    let h_bf = AIPerfHarness::new().await;
    let bf = h_bf.run_env(
        &args(&h_bf.mock.url),
        &[sketch, ("AIPERF_CELL_BARRIER_FREE", "1")],
    );
    assert!(bf.success(), "barrier-free run failed: {}", bf.stderr);

    for (label, run) in [("synchronized", &sync), ("barrier-free", &bf)] {
        assert!(
            run.artifacts
                .find_file("**/cellular-heartbeat.json")
                .is_some(),
            "{label} run must go through the controller (cellular-heartbeat.json sidecar)"
        );
    }
    assert_eq!(
        sync.artifacts.request_count() as u32,
        bf.artifacts.request_count() as u32,
        "barrier-free must dispatch the same request count as synchronized start"
    );
    let sync_json = sync.artifacts.json();
    let bf_json = bf.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        let a = &sync_json[metric];
        let b = &bf_json[metric];
        assert!(!a.is_null(), "synchronized report missing metric {metric}");
        assert_eq!(
            a, b,
            "tier-T3 barrier-free {metric} diverged from synchronized start: \
             sync={a}  barrier-free={b}"
        );
    }
}

/// Ultimate spec §4 (monotonic phaser control plane): `AIPERF_CELL_PHASER_START=1`
/// routes the run-wide START through the distributed phaser (the controller binds a
/// `PhaserServer` and advances generation 1 = `Started`; cells subscribe with
/// `PhaserClient` and await generation 1) instead of the single-shot velo event. START
/// timing does not affect the dataset-deterministic metrics, so a phaser-START run
/// reproduces the event-START run's deterministic metrics exactly — proving the
/// broadcast → phaser → phaser_velo control plane drives a real benchmark end-to-end.
#[tokio::test]
async fn test_cellular_phaser_start_matches_event_start() {
    let args = |url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 40 --concurrency 6 --cells 4 --random-seed 42 \
             --synthetic-input-tokens-mean 220 --synthetic-input-tokens-stddev 40 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple"
        )
    };
    let sketch = ("AIPERF_METRICS_SKETCH", "1");

    // Default: the single-shot velo event drives START.
    let h_event = AIPerfHarness::new().await;
    let event = h_event.run_env(&args(&h_event.mock.url), &[sketch]);
    assert!(event.success(), "event-START run failed: {}", event.stderr);

    // §4: the monotonic phaser drives START.
    let h_phaser = AIPerfHarness::new().await;
    let phaser = h_phaser.run_env(
        &args(&h_phaser.mock.url),
        &[sketch, ("AIPERF_CELL_PHASER_START", "1")],
    );
    assert!(
        phaser.success(),
        "phaser-START run failed: {}",
        phaser.stderr
    );

    for (label, run) in [("event", &event), ("phaser", &phaser)] {
        assert!(
            run.artifacts
                .find_file("**/cellular-heartbeat.json")
                .is_some(),
            "{label} run must go through the controller (cellular-heartbeat.json sidecar)"
        );
    }
    assert_eq!(
        event.artifacts.request_count() as u32,
        phaser.artifacts.request_count() as u32,
        "phaser-START must dispatch the same request count as event-START"
    );
    let event_json = event.artifacts.json();
    let phaser_json = phaser.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        let a = &event_json[metric];
        let b = &phaser_json[metric];
        assert!(!a.is_null(), "event report missing metric {metric}");
        assert_eq!(
            a, b,
            "§4 phaser-START {metric} diverged from event-START: event={a}  phaser={b}"
        );
    }
}

/// Ultimate spec §3 (dataset fan-out data plane) + §4.5 (dispatch state machine): with
/// `AIPERF_CELL_DATASET_FANOUT=1` the controller generates the dataset's request-ids once
/// and broadcasts them (advancing the phaser `ShardsAvailable` per chunk); each cell
/// subscribes over velo, builds its owned index (round-robin owned-filter → O(1/N) RAM),
/// and runs the dispatch state machine over its owned slice (exactly-once, counted
/// `DistributionMiss`). The fan-out is additive verification — it does not change the
/// benchmark — so an ON run reproduces the OFF run's deterministic metrics exactly, and
/// (via the runner's fail-closed miss guard) succeeding proves every cell received its
/// full owned shard with zero distribution misses.
#[tokio::test]
async fn test_cellular_dataset_fanout_matches_baseline() {
    let args = |url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 48 --concurrency 6 --cells 4 --random-seed 42 \
             --synthetic-input-tokens-mean 220 --synthetic-input-tokens-stddev 40 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple"
        )
    };
    let sketch = ("AIPERF_METRICS_SKETCH", "1");

    let h_off = AIPerfHarness::new().await;
    let off = h_off.run_env(&args(&h_off.mock.url), &[sketch]);
    assert!(off.success(), "baseline run failed: {}", off.stderr);

    // Fan-out ON (with the phaser availability interlock). The cell fails closed on any
    // distribution miss, so success == every owned shard delivered completely.
    let h_on = AIPerfHarness::new().await;
    let on = h_on.run_env(
        &args(&h_on.mock.url),
        &[
            sketch,
            ("AIPERF_CELL_DATASET_FANOUT", "1"),
            ("AIPERF_CELL_PHASER_START", "1"),
        ],
    );
    assert!(on.success(), "dataset fan-out run failed: {}", on.stderr);

    assert_eq!(
        off.artifacts.request_count() as u32,
        on.artifacts.request_count() as u32,
        "dataset fan-out must not change the dispatched request count"
    );
    let off_json = off.artifacts.json();
    let on_json = on.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        let a = &off_json[metric];
        let b = &on_json[metric];
        assert!(!a.is_null(), "baseline report missing metric {metric}");
        assert_eq!(
            a, b,
            "§3 dataset-fanout {metric} diverged from baseline: off={a}  on={b}"
        );
    }
}

/// The smallest `metadata.request_start_ns` across a run's raw records — the first
/// request's start on the run's timing origin. Panics if the run emitted no raw
/// records (wrong `--export-level`, or the run failed).
fn min_request_start_ns(run: &RunResult) -> i64 {
    let records = run.artifacts.raw_records();
    assert!(
        !records.is_empty(),
        "expected raw records (did the run pass --export-level raw and succeed?)"
    );
    records
        .iter()
        .filter_map(|r| r.get("metadata")?.get("request_start_ns")?.as_i64())
        .min()
        .expect("at least one record carries metadata.request_start_ns")
}

/// `AIPERF_CELL_SHARED_ORIGIN=1` zeroes every cell's record timeline at the shared
/// velo START barrier instead of at each cell's own post-setup local run start.
///
/// Proves the barrier-synchronized origin end-to-end at the raw-record level:
///
/// 1. **No regression** — the shifted origin only moves ABSOLUTE timestamps; all
///    latency/throughput/token metrics are differences, so an ON run reproduces a
///    single-cell baseline's deterministic ISL/OSL exactly.
/// 2. **The origin actually moved to the barrier** — with the flag ON the first
///    request's `request_start_ns` is measured from the START barrier, so it is
///    LATER than the flag-OFF run's (whose origin is each cell's post-setup start)
///    by the whole per-cell setup span (dataset compile + the large-tokenizer
///    load). That the ON minimum strictly exceeds the OFF minimum by a substantial
///    margin is the deterministic signature of the origin having been pulled back
///    to the shared barrier.
#[tokio::test]
async fn test_cellular_shared_origin_zeroes_at_the_barrier() {
    let args = |url: &str| {
        format!(
            "--model {DEFAULT_MODEL} --url {url} --endpoint-type chat \
             --request-count 60 --concurrency 6 --random-seed 42 \
             --synthetic-input-tokens-mean 256 --synthetic-input-tokens-stddev 64 \
             --output-tokens-mean 8 --output-tokens-stddev 0 --ui simple --export-level raw"
        )
    };

    // Single-cell baseline for the deterministic-metric parity check.
    let h_base = AIPerfHarness::new().await;
    let base = h_base.run(&format!("{} --cells 1", args(&h_base.mock.url)));
    assert!(base.success(), "baseline run failed: {}", base.stderr);

    // Flag OFF: each cell's origin is its own post-setup local run start.
    let h_off = AIPerfHarness::new().await;
    let off = h_off.run(&format!("{} --cells 3", args(&h_off.mock.url)));
    assert!(off.success(), "flag-off cellular run failed: {}", off.stderr);

    // Flag ON: every cell zeroes at the shared velo START barrier.
    let h_on = AIPerfHarness::new().await;
    let on = h_on.run_env(
        &format!("{} --cells 3", args(&h_on.mock.url)),
        &[("AIPERF_CELL_SHARED_ORIGIN", "1")],
    );
    assert!(on.success(), "shared-origin cellular run failed: {}", on.stderr);

    // Both cellular runs went through the controller.
    for (label, run) in [("off", &off), ("on", &on)] {
        assert!(
            run.artifacts
                .find_file("**/cellular-heartbeat.json")
                .is_some(),
            "{label} run must go through the controller (cellular-heartbeat.json sidecar)"
        );
    }
    assert_eq!(on.artifacts.request_count() as u32, 60);

    // (1) No regression: shared origin preserves the deterministic ISL/OSL exactly.
    let base_json = base.artifacts.json();
    let on_json = on.artifacts.json();
    for metric in DETERMINISTIC_METRICS {
        let a = &base_json[metric];
        let b = &on_json[metric];
        assert!(!a.is_null(), "baseline report missing metric {metric}");
        assert_eq!(
            a, b,
            "shared-origin {metric} diverged from the single-cell baseline: base={a}  on={b}"
        );
    }

    // (2) The origin moved to the barrier: the ON first-request timestamp is later
    // than the OFF one by the per-cell setup span. Assert a strict, substantial
    // shift (>50ms — setup includes the large-tokenizer load, reliably seconds).
    let off_min = min_request_start_ns(&off);
    let on_min = min_request_start_ns(&on);
    assert!(
        off_min > 0 && on_min > 0,
        "request_start_ns must be positive on both runs (off={off_min}, on={on_min})"
    );
    assert!(
        on_min > off_min + 50_000_000,
        "shared-origin run must measure the first request from the START barrier, so its \
         min request_start_ns ({on_min} ns) must exceed the flag-off run's ({off_min} ns) by \
         the per-cell setup span (>50ms); diff={} ns",
        on_min - off_min
    );
}
