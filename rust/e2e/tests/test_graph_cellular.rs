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
    assert!(
        baseline.success(),
        "1-cell graph run failed: {}",
        baseline.stderr
    );
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
    assert!(
        !base_isl.is_null(),
        "baseline must report input_sequence_length"
    );
    assert_eq!(
        cell_isl, base_isl,
        "3-cell graph run must reproduce the 1-cell input-token distribution \
         (the partition covers the full trace set): 1-cell={base_isl} 3-cell={cell_isl}"
    );
}

/// Stage G for GRAPH: a cross-host (forced-HTTP) `--cells N` graph run ships its
/// SINGLE-FILE `dag_jsonl` trace controller->cell over the SAME HTTP+zstd dataset plane
/// the scheduled `file`/`path` datasets use (proving `cellular_file_dataset_path` is
/// format-blind end-to-end), and each cell recompiles the trace and reproduces the
/// 1-cell trace set.
///
/// Drives the real cross-host mechanism same-host via the `AIPERF_CELL_ARTIFACT_HTTP_FORCE`
/// seam (true multi-host k8s cannot run in-sandbox): the controller binds its artifact
/// server on loopback, registers the single-file trace source, injects the authority into
/// each locally-launched cell, and the cells `GET /dataset/{name}` it back over real TCP +
/// zstd. The load-bearing proof is the controller's `served dataset source over HTTP …
/// content_encoding=zstd` log line (target `aiperf_cellular_artifact`, surfaced at `info`).
#[tokio::test]
async fn test_graph_cellular_single_file_dataset_shipping() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    // Baseline: single-cell, default path (no controller, no dataset shipping).
    let h_base = AIPerfHarness::new().await;
    let baseline = h_base.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 1 --random-seed 7 --ui simple",
            h_base.mock.url
        ),
        120,
    );
    assert!(
        baseline.success(),
        "1-cell graph baseline failed: {}",
        baseline.stderr
    );
    assert!(
        dataset_serve_observables(&baseline).is_empty(),
        "single-cell baseline must not serve a dataset over HTTP: {:?}",
        dataset_serve_observables(&baseline)
    );

    // Forced multi-process HTTP shipping: 3 cell subprocesses GET the single-file
    // dag_jsonl trace from the controller's loopback server over zstd, then recompile it.
    let h_cell = AIPerfHarness::new().await;
    let cellular = h_cell.run_env(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 3 --random-seed 7 --ui simple",
            h_cell.mock.url
        ),
        &[
            ("AIPERF_RUNNER_LOG", "warn,aiperf_cellular_artifact=info"),
            ("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"),
        ],
    );
    assert!(
        cellular.success(),
        "forced-HTTP 3-cell graph dataset-ship run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );

    // Topology guard: the multi-cell run went through the controller.
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "graph --cells 3 must go through the controller (cellular-heartbeat.json sidecar)"
    );

    // --- THE HTTP+zstd GRAPH DATASET-SHIP PROOF -----------------------------------
    let observables = dataset_serve_observables(&cellular);
    assert!(
        !observables.is_empty(),
        "no dataset-serve observable in logs/aiperf.log — the single-file graph trace did \
         not go over HTTP (or the force seam did not engage). Log tail:\n{}",
        aiperf_log(&cellular)
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .join("\n")
    );
    for line in &observables {
        assert!(
            line.contains("content_encoding=\"zstd\"") || line.contains("content_encoding=zstd"),
            "graph dataset-serve observable is not zstd-encoded: {line}"
        );
    }
    eprintln!(
        "HTTP+zstd GRAPH dataset shipping observed: {} serve(s):\n{}",
        observables.len(),
        observables.join("\n")
    );

    // --- TRACE-SET PARITY vs the 1-cell baseline over the shipped trace -----------
    assert_eq!(
        cellular.artifacts.request_count() as u64,
        baseline.artifacts.request_count() as u64,
        "HTTP-shipped 3-cell graph run must dispatch the same total record count as 1-cell"
    );
    let cell_isl = &cellular.artifacts.json()["input_sequence_length"];
    let base_isl = &baseline.artifacts.json()["input_sequence_length"];
    assert!(
        !base_isl.is_null(),
        "baseline must report input_sequence_length"
    );
    assert_eq!(
        cell_isl, base_isl,
        "HTTP-shipped 3-cell graph run must reproduce the 1-cell input-token distribution: \
         1-cell={base_isl} 3-cell={cell_isl}"
    );
}

/// A METRICS-ONLY (`--export-level summary`, no per-record artifacts) `--cells N` graph
/// run takes the exact-fold path (task G1): each cell folds every record into its own
/// exact accumulator, DROPS it, and ships the folded EXACT STORE (`StorePartition`)
/// instead of the full record `Vec`; the controller `merge_store_partitions`-merges the
/// cells' stores into one report. This proves (1) the store-shipping path runs end-to-end
/// through the Python frontend, and (2) its merged summary matches a 1-cell metrics-only
/// graph run within tolerance (counts/min/max/percentiles exact; sums/means a few ULPs).
///
/// The store-vs-records observable is the controller's `cellular-heartbeat.json`: a cell
/// shipping a folded store ships EMPTY latency sketches (the fold dropped the per-record
/// samples) but EXACT counters, so the merged heartbeat has `latency_ms.count == 0` with
/// `counters.issued > 0`. A retain-path (records-shipping) run would carry populated
/// sketches (`latency_ms.count > 0`) — so `count == 0` uniquely proves the store path.
#[tokio::test]
async fn test_graph_cellular_metrics_only_exact_fold_ships_store() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 3 --random-seed 7 --export-level summary --ui simple",
            h.mock.url
        ),
        120,
    );
    assert!(
        r.success(),
        "metrics-only graph cellular run failed: {}",
        r.stderr
    );

    // The controller ran (only it writes the cellular-heartbeat.json sidecar).
    let heartbeat = r.artifacts.read_json_file("**/cellular-heartbeat.json");
    assert!(
        !heartbeat.is_null(),
        "metrics-only graph --cells 3 must go through the controller (cellular-heartbeat.json)"
    );

    // STORE-path proof: exact counters but EMPTY latency sketches (the fold dropped the
    // per-record samples). A records-shipping (retain) run would have `latency_ms.count > 0`.
    let issued = heartbeat["counters"]["issued"].as_u64().unwrap_or(0);
    assert!(
        issued > 0,
        "merged heartbeat must carry the cells' exact issued counter; got {heartbeat}"
    );
    let latency_count = heartbeat["latency_ms"]["count"].as_u64();
    assert_eq!(
        latency_count,
        Some(0),
        "a folded-store (exact-fold) cell ships EMPTY latency sketches, so the merged \
         heartbeat latency count must be 0 (proving StorePartition, not records); got {heartbeat}"
    );

    // The merged summary report exists and carries the cells' folded records.
    let json = r.artifacts.json();
    assert!(!json.is_null(), "metrics-only merged report must exist");
    assert!(
        r.artifacts.request_count() > 0.0,
        "merged metrics-only report must carry records folded from the cells"
    );

    // Parity vs a 1-cell metrics-only graph run (also exact-fold, single-process — no
    // controller sidecar): the same seeded trace set, so summary metrics match within
    // tolerance (min/max/percentiles exact; the integer sequence-length averages exact
    // under the order-independent store concat).
    let h1 = AIPerfHarness::new().await;
    let baseline = h1.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 1 --random-seed 7 --export-level summary --ui simple",
            h1.mock.url
        ),
        120,
    );
    assert!(
        baseline.success(),
        "1-cell metrics-only graph run failed: {}",
        baseline.stderr
    );
    assert!(
        baseline
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_none(),
        "a 1-cell metrics-only run must take the single-process path (no controller sidecar)"
    );
    assert_eq!(
        r.artifacts.request_count() as u64,
        baseline.artifacts.request_count() as u64,
        "3-cell metrics-only graph run must fold the same total record count as 1-cell"
    );
    let base_json = baseline.artifacts.json();
    for metric in [
        "input_sequence_length",
        "output_sequence_length",
        "request_count",
    ] {
        let base = &base_json[metric];
        let cell = &json[metric];
        assert!(!base.is_null(), "baseline missing metric {metric}");
        for stat in ["min", "max", "avg", "p50", "p99"] {
            if let (Some(b), Some(c)) = (base[stat].as_f64(), cell[stat].as_f64()) {
                assert!(
                    (b - c).abs() <= 1e-9 * b.abs().max(1.0),
                    "metrics-only graph cellular {metric}.{stat} diverged: 1-cell={b} 3-cell={c}"
                );
            }
        }
    }
}

/// The sorted multiset of each PROFILING graph record's dataset-deterministic
/// `(input_sequence_length, output_sequence_length)` projection, read from
/// `profile_export.jsonl`. Wall-clock timing differs run to run, but this projection
/// depends only on the seeded trace + deterministic mock, so it is a stable per-record
/// key for a row-SET comparison across topologies (completion order accepted).
fn graph_record_isl_osl_multiset(r: &RunResult) -> Vec<(i64, i64)> {
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

/// Task G2: a `--cells N` graph run WITH per-record artifacts (`--export-level raw`)
/// takes the exact-fold STREAMING LANE path — each cell folds each record into its exact
/// accumulator and DROPS it while STREAMING that record's artifact rows to its cell dir
/// (records/raw via `RecordArtifactLane`), then ships the folded EXACT STORE
/// (`StorePartition`: empty heartbeat sketches, exact counters) instead of the full
/// record `Vec`. The controller concatenates the per-cell artifact files into the run
/// dir (Stage D concat) and `merge_store_partitions`-merges the stores.
///
/// Proof it was EXACT-FOLD (not the legacy retain path) WITH artifacts present: the
/// merged `cellular-heartbeat.json` has `latency_ms.count == 0` (a folded store ships
/// empty sketches; a records-shipping retain cell would populate them) with
/// `counters.issued > 0`, AND the per-record `profile_export.jsonl`/`_raw.jsonl` are
/// emitted anyway. Its merged per-record row SET equals a 1-cell run's for the same
/// seed, and the summary matches within tolerance.
#[tokio::test]
async fn test_graph_cellular_exact_fold_streams_per_record_artifacts() {
    let h3 = AIPerfHarness::new().await;
    let cellular = h3.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 3 --random-seed 7 --export-level raw --ui simple",
            h3.mock.url
        ),
        120,
    );
    assert!(
        cellular.success(),
        "graph cellular artifact run failed: {}",
        cellular.stderr
    );

    // Topology guard: the 3-cell run went through the controller.
    let heartbeat = cellular
        .artifacts
        .read_json_file("**/cellular-heartbeat.json");
    assert!(
        !heartbeat.is_null(),
        "graph --cells 3 with artifacts must go through the controller (cellular-heartbeat.json)"
    );

    // EXACT-FOLD-WITH-ARTIFACTS proof: a folded store ships EMPTY latency sketches
    // (count 0) but EXACT counters. A retain (records-shipping) cell would carry
    // populated sketches — so count == 0 with issued > 0 uniquely proves the store path
    // was taken even though per-record files were requested (task G2 relaxed the gate).
    let issued = heartbeat["counters"]["issued"].as_u64().unwrap_or(0);
    assert!(
        issued > 0,
        "merged heartbeat must carry the cells' exact issued counter; got {heartbeat}"
    );
    assert_eq!(
        heartbeat["latency_ms"]["count"].as_u64(),
        Some(0),
        "an exact-fold (folded-store) graph cell ships EMPTY latency sketches, so the merged \
         heartbeat latency count must be 0 (proving StorePartition, not records) even with \
         per-record artifacts requested; got {heartbeat}"
    );

    // The controller emitted the merged per-record files (Stage D concat over the cells'
    // streaming-lane output) — the whole point of G2: exact-fold no longer drops them.
    assert!(
        cellular
            .artifacts
            .find_file("**/profile_export.jsonl")
            .is_some(),
        "graph --cells 3 must emit the merged profile_export.jsonl (lane + Stage D concat)"
    );
    assert!(
        cellular
            .artifacts
            .find_file("**/profile_export_raw.jsonl")
            .is_some(),
        "graph --cells 3 must emit the merged profile_export_raw.jsonl (lane + Stage D concat)"
    );

    // A 1-cell graph run over the same seed also takes the single-process exact-fold lane
    // path (no controller sidecar) and writes its records directly.
    let h1 = AIPerfHarness::new().await;
    let baseline = h1.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 1 --random-seed 7 --export-level raw --ui simple",
            h1.mock.url
        ),
        120,
    );
    assert!(
        baseline.success(),
        "1-cell graph artifact run failed: {}",
        baseline.stderr
    );
    assert!(
        baseline
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_none(),
        "a 1-cell graph run must take the single-process path (no controller sidecar)"
    );

    // Row-SET parity: the 3-cell merged per-record rows equal the 1-cell run's (same
    // seeded trace set; completion order accepted, so SET — not byte order — is compared).
    let base_records = graph_record_isl_osl_multiset(&baseline);
    let cell_records = graph_record_isl_osl_multiset(&cellular);
    assert!(
        !base_records.is_empty(),
        "1-cell graph baseline must emit per-record rows"
    );
    assert_eq!(
        base_records, cell_records,
        "3-cell merged graph per-record row SET must equal the 1-cell run's for the same seed"
    );
    assert_eq!(
        baseline.artifacts.raw_records().len(),
        cellular.artifacts.raw_records().len(),
        "1-cell and 3-cell graph runs must emit the same number of raw per-record rows"
    );

    // Summary parity within tolerance (counts/min/max/percentiles exact; sums/means ULPs).
    let base_json = baseline.artifacts.json();
    let cell_json = cellular.artifacts.json();
    assert_eq!(
        cellular.artifacts.request_count() as u64,
        baseline.artifacts.request_count() as u64,
        "3-cell graph run must dispatch the same total record count as 1-cell"
    );
    for metric in [
        "input_sequence_length",
        "output_sequence_length",
        "request_count",
    ] {
        let base = &base_json[metric];
        let cell = &cell_json[metric];
        assert!(!base.is_null(), "baseline missing metric {metric}");
        for stat in ["min", "max", "avg", "p50", "p99"] {
            if let (Some(b), Some(c)) = (base[stat].as_f64(), cell[stat].as_f64()) {
                assert!(
                    (b - c).abs() <= 1e-9 * b.abs().max(1.0),
                    "graph cellular artifact {metric}.{stat} diverged: 1-cell={b} 3-cell={c}"
                );
            }
        }
    }
}

/// The full text of the run's `logs/aiperf.log` (empty if absent).
fn aiperf_log(r: &RunResult) -> String {
    match r.artifacts.find_file("**/aiperf.log") {
        Some(path) => std::fs::read_to_string(&path).unwrap_or_default(),
        None => String::new(),
    }
}

/// The HTTP+zstd dataset-serve observable lines: one per served source, naming the
/// dataset and the encoding. The load-bearing proof the trace crossed a real HTTP
/// socket compressed, not a shared filesystem.
fn dataset_serve_observables(r: &RunResult) -> Vec<String> {
    aiperf_log(r)
        .lines()
        .filter(|l| l.contains("served dataset source over HTTP"))
        .map(str::to_string)
        .collect()
}
