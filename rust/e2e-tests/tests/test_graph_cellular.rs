// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for graph-mode cellular with `dag_jsonl` and `--cells N`.
//!
//! `--cells 3` on a graph dataset makes the launched `aiperf` a controller that
//! spawns three `aiperf --cell` children; each partitions the trace instances by
//! `instance_ordinal % cell_count` (PartitionedGraphTraceSource), runs its interleaved
//! slice, and ships its graph records; the controller concatenation-merges them (records
//! carry local per-cell indices, wall-clock ordered) into one report.

mod common;
use arrow::array::{Array, Float64Array, StringArray};
use common::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

/// A multi-root single-turn DAG fixture: cheap to run many instances of, so a
/// `--num-conversations` sweep gives every cell a non-empty interleaved slice.
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/fixtures/dag/multi_root_single_turn.dag.jsonl"
);

/// `aiperf profile --cells 3` over a graph dataset runs end-to-end and merges every
/// cell's graph records into one report.
#[tokio::test]
async fn test_graph_cellular() {
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

    // Only the controller emits the heartbeat after aggregating cell heartbeats.
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

    // A 1-cell run of the same seeded config dispatches the same conversations, so its
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

/// A forced-HTTP `--cells N` graph run ships its single-file `dag_jsonl` trace
/// from controller to cells over the HTTP+zstd dataset plane, and each cell
/// recompiles the trace and reproduces the
/// 1-cell trace set.
///
/// Drives the real cross-host mechanism same-host via the `AIPERF_CELL_ARTIFACT_HTTP_FORCE`
/// seam (true multi-host k8s cannot run in-sandbox): the controller binds its artifact
/// server on loopback, registers the single-file trace source, injects the authority into
/// each locally-launched cell, and the cells `GET /dataset/{name}` it back over real TCP +
/// zstd. The controller logs `served dataset source over HTTP …
/// content_encoding=zstd` for each transfer.
#[tokio::test]
async fn test_graph_cellular_single_file_dataset_shipping() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

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
            // `aiperf=info` is required: the serve event fires in the `--execute` child and the
            // parent re-emits forwarded child lines under target `aiperf` into `logs/aiperf.log`.
            (
                "AIPERF_LOG",
                "warn,aiperf=info,aiperf_cellular_artifact=info",
            ),
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

    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "graph --cells 3 must go through the controller (cellular-heartbeat.json sidecar)"
    );

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
/// run uses exact folding: each cell folds every record into its own
/// exact accumulator, DROPS it, and ships the folded EXACT STORE (`StorePartition`)
/// instead of the full record `Vec`; the controller merges the cells' stores into
/// one report matching a 1-cell metrics-only graph run within tolerance.
///
/// The store-vs-records observable is the controller's `cellular-heartbeat.json`: a cell
/// shipping a folded store ships EMPTY latency sketches (the fold dropped the per-record
/// samples) but EXACT counters, so the merged heartbeat has `latency_ms.count == 0` with
/// `counters.issued > 0`. Records shipping would populate the latency sketches.
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

    let heartbeat = r.artifacts.read_json_file("**/cellular-heartbeat.json");
    assert!(
        !heartbeat.is_null(),
        "metrics-only graph --cells 3 must go through the controller (cellular-heartbeat.json)"
    );

    // Folded stores carry counters but no per-record latency samples.
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
         heartbeat latency count must be 0; got {heartbeat}"
    );

    let json = r.artifacts.json();
    assert!(!json.is_null(), "metrics-only merged report must exist");
    assert!(
        r.artifacts.request_count() > 0.0,
        "merged metrics-only report must carry records folded from the cells"
    );

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

/// SKETCH graph cellular (bounded memory): `--cells 3 --sketch-metrics` over a graph
/// dataset folds each record into the per-`(phase, tag)` t-digest and DROPS it, then a
/// cell ships its folded SKETCH STORE (a `CellMessage::StorePartition`), which the
/// controller merges (`merge_store_partitions` → t-digest merge) into one report — the
/// SAME store path exact-fold uses, so it works through the shared graph fold gate
/// (`graph_fold = graph_exact_fold || sketch`). Counts/sums/min/max stay EXACT; only
/// percentiles are approximate.
///
/// The merged heartbeat carries exact counters but no retained latency samples,
/// identifying the folded-store path.
#[tokio::test]
async fn test_graph_cellular_sketch_ships_store() {
    let h = AIPerfHarness::new().await;
    let r = h.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 3 --random-seed 7 --sketch-metrics --export-level summary --ui simple",
            h.mock.url
        ),
        120,
    );
    assert!(
        r.success(),
        "sketch graph cellular run failed: {}",
        r.stderr
    );

    let heartbeat = r.artifacts.read_json_file("**/cellular-heartbeat.json");
    assert!(
        !heartbeat.is_null(),
        "sketch graph --cells 3 must go through the controller (cellular-heartbeat.json)"
    );

    // Folded sketch stores carry counters but no per-record latency samples.
    let issued = heartbeat["counters"]["issued"].as_u64().unwrap_or(0);
    assert!(
        issued > 0,
        "merged sketch heartbeat must carry the cells' exact issued counter; got {heartbeat}"
    );
    assert_eq!(
        heartbeat["latency_ms"]["count"].as_u64(),
        Some(0),
        "a folded-store (sketch) cell ships EMPTY latency sketches, so the merged heartbeat \
         latency count must be 0; got {heartbeat}"
    );

    let json = r.artifacts.json();
    assert!(!json.is_null(), "sketch merged report must exist");
    assert!(
        r.artifacts.request_count() > 0.0,
        "merged sketch report must carry the cells' folded record total"
    );

    let h1 = AIPerfHarness::new().await;
    let baseline = h1.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {FIXTURE} \
             --custom-dataset-type dag_jsonl --num-conversations 6 --concurrency 3 \
             --cells 1 --random-seed 7 --sketch-metrics --export-level summary --ui simple",
            h1.mock.url
        ),
        120,
    );
    assert!(
        baseline.success(),
        "1-cell sketch graph run failed: {}",
        baseline.stderr
    );
    assert_eq!(
        r.artifacts.request_count() as u64,
        baseline.artifacts.request_count() as u64,
        "3-cell sketch graph run must fold the same total record count as 1-cell"
    );
    let base_json = baseline.artifacts.json();
    for metric in ["input_sequence_length", "output_sequence_length"] {
        let base = &base_json[metric];
        let cell = &json[metric];
        assert!(!base.is_null(), "sketch baseline missing metric {metric}");
        for stat in ["min", "max", "avg"] {
            if let (Some(b), Some(c)) = (base[stat].as_f64(), cell[stat].as_f64()) {
                assert!(
                    (b - c).abs() <= 1e-9 * b.abs().max(1.0),
                    "sketch graph cellular {metric}.{stat} diverged: 1-cell={b} 3-cell={c}"
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

/// A `--cells N` graph run with per-record artifacts (`--export-level raw`)
/// takes the exact-fold STREAMING LANE path — each cell folds each record into its exact
/// accumulator and DROPS it while STREAMING that record's artifact rows to its cell dir
/// (records/raw via `RecordArtifactLane`), then ships the folded EXACT STORE
/// (`StorePartition`: empty heartbeat sketches, exact counters) instead of the full
/// record `Vec`. The controller concatenates the per-cell artifact files into the run
/// dir and merges the store partitions.
///
/// Exact-fold execution with artifacts has `latency_ms.count == 0` in the merged
/// `cellular-heartbeat.json` (a folded store ships
/// empty sketches; a records-shipping retain cell would populate them) with
/// `counters.issued > 0`, AND the per-record `profile_export.jsonl`/`_raw.jsonl` are
/// emitted anyway. Its merged per-record row set equals a 1-cell run's for the same
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

    let heartbeat = cellular
        .artifacts
        .read_json_file("**/cellular-heartbeat.json");
    assert!(
        !heartbeat.is_null(),
        "graph --cells 3 with artifacts must go through the controller (cellular-heartbeat.json)"
    );

    // Folded stores carry counters but no per-record latency samples.
    let issued = heartbeat["counters"]["issued"].as_u64().unwrap_or(0);
    assert!(
        issued > 0,
        "merged heartbeat must carry the cells' exact issued counter; got {heartbeat}"
    );
    assert_eq!(
        heartbeat["latency_ms"]["count"].as_u64(),
        Some(0),
        "an exact-fold (folded-store) graph cell ships EMPTY latency sketches, so the merged \
         heartbeat latency count must be 0 even with \
         per-record artifacts requested; got {heartbeat}"
    );

    // The controller concatenates the cells' streaming-lane output.
    assert!(
        cellular
            .artifacts
            .find_file("**/profile_export.jsonl")
            .is_some(),
        "graph --cells 3 must emit the merged profile_export.jsonl"
    );
    assert!(
        cellular
            .artifacts
            .find_file("**/profile_export_raw.jsonl")
            .is_some(),
        "graph --cells 3 must emit the merged profile_export_raw.jsonl"
    );

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

/// Base conversation id with the topology-dependent instance ordinal stripped.
///
/// Graph cellular partitions the trace instances across cells and each cell
/// renumbers its slice DENSELY (see `cellular/shard.rs` — "renumbers densely"), so a
/// record's `session_num` and its `conversation_id` instance suffix (`r1::instance-4`)
/// depend on how many cells ran and which cell owned the instance. The BASE session
/// (`r1`) plus the seeded token metrics are topology-independent, so the row-SET
/// comparison keys on the base id, exactly as [`graph_record_isl_osl_multiset`] keys on
/// `(ISL, OSL)` alone. Everything before the first `::instance-` marker is the base id.
fn strip_instance(conversation_id: &str) -> &str {
    match conversation_id.find("::instance-") {
        Some(pos) => &conversation_id[..pos],
        None => conversation_id,
    }
}

/// A single-turn `dag_jsonl` graph config with ALL per-record artifacts enabled
/// (`records: [jsonl, csv, parquet]`, `raw`, `export_outputs_json`), pointed at `url`
/// and seeded so a 1-cell and a `--cells N` run over the same trace produce the same
/// topology-independent record content. `--cells` is supplied on the CLI (a runtime
/// axis), so this config is identical across the two arms.
///
/// CLI `--export-level` does not select CSV or Parquet, so the artifact formats
/// are configured through the YAML `records` list.
fn graph_artifact_config(url: &str) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         \n\
         benchmark:\n\
        \x20 model: Qwen3-0.6B\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   path: {FIXTURE}\n\
        \x20   format: dag_jsonl\n\
        \x20   randomSeed: 7\n\
        \x20 phases:\n\
        \x20   type: concurrency\n\
        \x20   concurrency: 3\n\
        \x20   sessions: 6\n\
        \x20 artifacts:\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20     - csv\n\
        \x20     - parquet\n\
        \x20   raw: true\n\
        \x20   export_outputs_json: true\n",
    )
}

/// Split one RFC4180 CSV line into fields, including quoted delimiters and quotes.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == ',' {
            fields.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    fields.push(cur);
    fields
}

/// The dataset-deterministic, topology-independent records-CSV columns compared across
/// the two graph runs: the BASE conversation id (instance ordinal stripped), turn index,
/// phase, and the seeded ISL/OSL metrics (headers follow
/// `RecordMetricColumn::csv_display_name`). Excludes `session_num` / the instance-suffixed
/// `conversation_id` (topology-dependent dense renumbering) and every wall-clock/UUID/timing
/// column that legitimately differs between two online runs.
const CSV_DETERMINISTIC_COLUMNS: &[&str] = &[
    "conversation_id",
    "turn_index",
    "benchmark_phase",
    "Input Sequence Length (tokens)",
    "Output Sequence Length (tokens)",
];

/// Read the records CSV as (header line, sorted deterministic-column projection multiset).
/// `conversation_id` is projected through [`strip_instance`] so the SET is
/// topology-independent. Sorted `Vec` (not a set) so a divergent row MULTIPLICITY still
/// diverges the comparison.
fn read_graph_records_csv_projection(r: &RunResult) -> (String, Vec<String>) {
    let p = r
        .artifacts
        .find_file("**/profile_export_records.csv")
        .expect("records csv");
    let text = std::fs::read_to_string(&p).unwrap();
    let mut it = text.lines();
    let header_line = it.next().unwrap_or_default().to_string();
    let header = parse_csv_line(&header_line);
    let cid_idx = header
        .iter()
        .position(|c| c == "conversation_id")
        .expect("records CSV missing conversation_id");
    let col_idx: Vec<usize> = CSV_DETERMINISTIC_COLUMNS
        .iter()
        .map(|name| {
            header
                .iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("records CSV missing column {name}: {header:?}"))
        })
        .collect();

    let mut rows = Vec::new();
    for line in it {
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_line(line);
        let projected: Vec<String> = col_idx
            .iter()
            .map(|&i| {
                let cell = fields.get(i).map(String::as_str).unwrap_or("");
                if i == cid_idx {
                    strip_instance(cell).to_string()
                } else {
                    cell.to_string()
                }
            })
            .collect();
        // ASCII unit separator: cannot appear in the projected cells, so the join is
        // collision-free.
        rows.push(projected.join("\u{1f}"));
    }
    rows.sort();
    (header_line, rows)
}

/// Read a graph `profile_export.parquet` as (schema column names, sorted
/// topology-independent projection multiset). The projection is the base conversation id
/// plus seeded ISL/OSL. Dense per-cell renumbering makes `session_num` and instance
/// ordinals topology-dependent.
fn read_graph_parquet_projection(path: &std::path::Path) -> (Vec<String>, Vec<String>) {
    let file = std::fs::File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader builder");
    let reader = builder.build().expect("parquet reader");
    let batches: Vec<_> = reader.map(|b| b.expect("parquet batch")).collect();
    let schema = batches[0].schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();

    let conv_idx = schema
        .index_of("conversation_id")
        .expect("conversation_id column");
    let isl_idx = schema
        .index_of("input_sequence_length")
        .expect("input_sequence_length column");
    let osl_idx = schema
        .index_of("output_sequence_length")
        .expect("output_sequence_length column");

    let opt_f64 = |a: &Float64Array, i: usize| -> String {
        if a.is_null(i) {
            "null".to_string()
        } else {
            a.value(i).to_string()
        }
    };

    let mut rows = Vec::new();
    for b in &batches {
        let conv = b
            .column(conv_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("conversation_id Utf8");
        let isl = b
            .column(isl_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("input_sequence_length Float64");
        let osl = b
            .column(osl_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("output_sequence_length Float64");
        for i in 0..b.num_rows() {
            let conv_cell = if conv.is_null(i) {
                "null".to_string()
            } else {
                strip_instance(conv.value(i)).to_string()
            };
            rows.push(format!(
                "{}|{}|{}",
                conv_cell,
                opt_f64(isl, i),
                opt_f64(osl, i),
            ));
        }
    }
    rows.sort();
    (names, rows)
}

/// The topology-independent `outputs.json` projection multiset: base conversation id +
/// turn index + the mock's deterministic generated text. Excludes `session_num` /
/// `x_request_id` and the timing block. This JSON-MERGE concat (`data`-array merge, not a
/// line append — `shard_artifacts.rs::concat_outputs_json`) is the highest-value graph
/// concatenation path to exercise end to end.
fn outputs_json_projection(r: &RunResult) -> Vec<String> {
    let p = r
        .artifacts
        .find_file("**/outputs.json")
        .expect("outputs.json");
    let doc: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    assert!(
        !doc["schema_version"].is_null(),
        "merged outputs.json must carry schema_version; got {doc}"
    );
    let data = doc["data"].as_array().cloned().unwrap_or_default();
    let mut rows: Vec<String> = data
        .iter()
        .map(|row| {
            let cid = row["conversation_id"].as_str().unwrap_or("");
            serde_json::json!({
                "conversation_id": strip_instance(cid),
                "turn_index": row["turn_index"],
                "response_text": row["response_text"],
                "reasoning_text": row["reasoning_text"],
            })
            .to_string()
        })
        .collect();
    rows.sort();
    rows
}

/// A `--cells N` graph run with CSV, Parquet, and `outputs.json`
/// per-record artifacts enabled takes the exact-fold STREAMING LANE, and the controller
/// concatenates each lane's per-cell file into the run dir: header-once +
/// data-append CSV, row-group-concat Parquet,
/// and — the highest-value path — a `data`-array JSON MERGE for `outputs.json`
/// (`shard_artifacts.rs::concat_outputs_json`, not a line concat).
///
/// A 1-cell run over the same seeded trace takes the single-process lane path (no
/// controller sidecar) and writes each file directly. Each merged file must equal the
/// 1-cell run's as a topology-independent row SET — keyed on the base conversation id +
/// seeded ISL/OSL (+ deterministic generated text for outputs.json), because graph cells
/// renumber their instance slice densely (`session_num`/instance ordinals are
/// topology-dependent, exactly why `graph_record_isl_osl_multiset` keys on ISL/OSL alone).
#[tokio::test]
async fn test_graph_cellular_exact_fold_concats_csv_parquet_outputs() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    let run = |h: &AIPerfHarness, cells: u32| -> RunResult {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg = tmp.path().join("graph_artifacts.yaml");
        std::fs::write(&cfg, graph_artifact_config(&h.mock.url)).unwrap();
        h.run_timeout(
            &format!("--config {} --cells {cells} --ui simple", cfg.display()),
            120,
        )
    };

    // The controller concatenates the three cells' streaming lanes.
    let h3 = AIPerfHarness::new().await;
    let cellular = run(&h3, 3);
    assert!(
        cellular.success(),
        "3-cell graph artifact run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );

    // Only the controller emits the heartbeat; folded stores contain no latency samples.
    let heartbeat = cellular
        .artifacts
        .read_json_file("**/cellular-heartbeat.json");
    assert!(
        !heartbeat.is_null(),
        "graph --cells 3 with artifacts must go through the controller (cellular-heartbeat.json)"
    );
    assert!(
        heartbeat["counters"]["issued"].as_u64().unwrap_or(0) > 0,
        "merged heartbeat must carry the cells' exact issued counter; got {heartbeat}"
    );
    assert_eq!(
        heartbeat["latency_ms"]["count"].as_u64(),
        Some(0),
        "an exact-fold graph cell ships EMPTY latency sketches, so the merged heartbeat \
         latency count must be 0 even with per-record artifacts requested; got {heartbeat}"
    );

    for glob in [
        "**/profile_export_records.csv",
        "**/profile_export.parquet",
        "**/outputs.json",
    ] {
        assert!(
            cellular.artifacts.find_file(glob).is_some(),
            "graph --cells 3 must emit the merged {glob}"
        );
    }

    // 1-cell baseline over the same seeded trace: single-process lane, no controller.
    let h1 = AIPerfHarness::new().await;
    let baseline = run(&h1, 1);
    assert!(
        baseline.success(),
        "1-cell graph artifact run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );
    assert!(
        baseline
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_none(),
        "a 1-cell graph run must take the single-process path (no controller sidecar)"
    );

    let (base_csv_header, base_csv) = read_graph_records_csv_projection(&baseline);
    let (cell_csv_header, cell_csv) = read_graph_records_csv_projection(&cellular);
    assert!(
        !base_csv.is_empty(),
        "1-cell graph baseline must emit records-CSV data rows"
    );
    assert_eq!(
        base_csv_header, cell_csv_header,
        "merged records-CSV header diverged from the 1-cell run"
    );
    assert_eq!(
        base_csv.len(),
        cell_csv.len(),
        "merged records-CSV row count diverged: 1-cell={} 3-cell={}",
        base_csv.len(),
        cell_csv.len()
    );
    assert_eq!(
        base_csv, cell_csv,
        "merged graph records-CSV topology-independent content SET (base cid + ISL/OSL) \
         diverged from the 1-cell run"
    );

    let base_out = outputs_json_projection(&baseline);
    let cell_out = outputs_json_projection(&cellular);
    assert!(
        !base_out.is_empty(),
        "1-cell graph baseline must emit outputs.json data rows"
    );
    assert_eq!(
        base_out.len(),
        cell_out.len(),
        "merged outputs.json row count diverged: 1-cell={} 3-cell={}",
        base_out.len(),
        cell_out.len()
    );
    assert_eq!(
        base_out, cell_out,
        "merged graph outputs.json topology-independent (base cid + generated text) SET \
         diverged from the 1-cell run — the data-array JSON-merge concat is broken"
    );

    let (base_pq_cols, base_pq) = read_graph_parquet_projection(
        &baseline
            .artifacts
            .find_file("**/profile_export.parquet")
            .expect("baseline parquet"),
    );
    let (cell_pq_cols, cell_pq) = read_graph_parquet_projection(
        &cellular
            .artifacts
            .find_file("**/profile_export.parquet")
            .expect("cellular parquet"),
    );
    assert_eq!(
        base_pq_cols, cell_pq_cols,
        "merged parquet schema (column names) diverged from the 1-cell run"
    );
    assert_eq!(
        base_pq.len(),
        cell_pq.len(),
        "merged parquet row count diverged: 1-cell={} 3-cell={}",
        base_pq.len(),
        cell_pq.len()
    );
    assert_eq!(
        base_pq, cell_pq,
        "merged graph parquet topology-independent projection SET (base cid + ISL/OSL) \
         diverged from the 1-cell run"
    );

    // Cross-artifact consistency: all three lanes carry the same number of records, and
    // it matches the merged summary's dispatched count.
    assert_eq!(
        cell_csv.len(),
        cell_out.len(),
        "merged CSV and outputs.json row counts must agree"
    );
    assert_eq!(
        cell_csv.len(),
        cell_pq.len(),
        "merged CSV and parquet row counts must agree"
    );
    assert_eq!(
        cellular.artifacts.request_count() as u64,
        baseline.artifacts.request_count() as u64,
        "3-cell graph run must dispatch the same total record count as 1-cell"
    );
}

/// Build a DIRECTORY multi-file `weka_trace` fixture: `count` sibling `.json`
/// files under a fresh temp dir, each a whole single-root WEKA trace document (the
/// shape the WEKA dir loader reads — every `.json` in the dir, non-recursive). The
/// roots differ only in id / hash / requested output length, so the compiled trace
/// set is deterministic per topology. Returns the `TempDir` (kept alive by the
/// caller) and the directory path string.
fn write_weka_dir_fixture(count: usize) -> (tempfile::TempDir, String) {
    let dir = tempfile::tempdir().expect("weka fixture tempdir");
    for i in 0..count {
        // Vary the requested output length so OSL is a non-trivial distribution to
        // compare across topologies; keep input at one 16-token block per root.
        let out = 5 + (i % 5);
        let doc = serde_json::json!({
            "id": format!("root{i}"),
            "models": ["Qwen3-0.6B"],
            "block_size": 16,
            "hash_id_scope": "global",
            "requests": [{
                "t": 0,
                "type": "n",
                "model": "Qwen3-0.6B",
                "in": 16,
                "out": out,
                "hash_ids": [(i as i64) + 1]
            }]
        });
        std::fs::write(
            dir.path().join(format!("shard-{i:03}.json")),
            serde_json::to_vec(&doc).unwrap(),
        )
        .expect("write weka shard");
    }
    let path = dir.path().display().to_string();
    (dir, path)
}

/// A forced-HTTP `--cells N` graph run over a
/// DIRECTORY multi-shard `weka_trace` must ship EVERY shard the loader reads over
/// the HTTP+zstd plane (one `served dataset source over HTTP` per shard), each cell
/// reconstructs the whole directory from the manifest, and the merged record SET
/// matches a 1-cell run over the same directory.
///
/// The controller enumerates the shard set, serves it with a manifest, and cells
/// rebuild the directory tree through the force-HTTP loopback seam.
#[tokio::test]
async fn test_graph_cellular_directory_multi_file_dataset_shipping() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    const SHARDS: usize = 6;
    let (_fixture_guard, trace_dir) = write_weka_dir_fixture(SHARDS);

    let h_base = AIPerfHarness::new().await;
    let baseline = h_base.run_timeout(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {trace_dir} \
             --custom-dataset-type weka_trace --num-conversations 6 --concurrency 3 --cells 1 --random-seed 7 \
             --ui simple",
            h_base.mock.url
        ),
        120,
    );
    assert!(
        baseline.success(),
        "1-cell weka directory baseline failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );
    assert!(
        dataset_serve_observables(&baseline).is_empty(),
        "single-cell baseline must not serve any dataset over HTTP: {:?}",
        dataset_serve_observables(&baseline)
    );
    assert!(
        baseline.artifacts.request_count() > 0.0,
        "1-cell weka directory baseline must dispatch records"
    );

    // Forced multi-process HTTP shipping: 3 cells fetch the manifest and every shard
    // of the directory trace from the controller's loopback server over zstd, then
    // each recompiles the reconstructed directory.
    let h_cell = AIPerfHarness::new().await;
    let cellular = h_cell.run_env(
        &format!(
            "--model Qwen3-0.6B --url {} --endpoint-type chat --input-file {trace_dir} \
             --custom-dataset-type weka_trace --num-conversations 6 --concurrency 3 --cells 3 --random-seed 7 \
             --ui simple",
            h_cell.mock.url
        ),
        &[
            // `aiperf=info` is required: the serve event fires in the `--execute` child and the
            // parent re-emits forwarded child lines under target `aiperf` into `logs/aiperf.log`.
            ("AIPERF_LOG", "warn,aiperf=info,aiperf_cellular_artifact=info"),
            ("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"),
        ],
    );
    assert!(
        cellular.success(),
        "forced-HTTP 3-cell weka directory ship run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );

    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "weka directory --cells 3 must go through the controller (cellular-heartbeat.json sidecar)"
    );

    // Every shard must be served over HTTP+zstd, and enough serves must cover the whole
    // shard set across cells (3 cells x 6 shards, minus any empty-slice cells). Assert
    // at least one serve per distinct shard name so no shard was under-shipped.
    let observables = dataset_serve_observables(&cellular);
    assert!(
        !observables.is_empty(),
        "no dataset-serve observable — the directory trace did not go over HTTP. Log tail:\n{}",
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
            "directory dataset-serve observable is not zstd-encoded: {line}"
        );
    }
    for i in 0..SHARDS {
        let shard = format!("shard-{i:03}.json");
        assert!(
            observables.iter().any(|l| l.contains(&shard)),
            "shard {shard} was never served over HTTP (shipped set != loader read set). Serves:\n{}",
            observables.join("\n")
        );
    }
    eprintln!(
        "multi-file HTTP+zstd weka directory shipping observed: {} serve(s):\n{}",
        observables.len(),
        observables.join("\n")
    );

    assert_eq!(
        cellular.artifacts.request_count() as u64,
        baseline.artifacts.request_count() as u64,
        "HTTP-shipped 3-cell weka directory run must dispatch the same total record count as 1-cell"
    );
    let cell_isl = &cellular.artifacts.json()["input_sequence_length"];
    let base_isl = &baseline.artifacts.json()["input_sequence_length"];
    assert!(
        !base_isl.is_null(),
        "baseline must report input_sequence_length"
    );
    assert_eq!(
        cell_isl, base_isl,
        "HTTP-shipped 3-cell weka directory run must reproduce the 1-cell input-token \
         distribution: 1-cell={base_isl} 3-cell={cell_isl}"
    );
    let base_osl = &baseline.artifacts.json()["output_sequence_length"];
    let cell_osl = &cellular.artifacts.json()["output_sequence_length"];
    assert_eq!(
        cell_osl, base_osl,
        "HTTP-shipped 3-cell weka directory run must reproduce the 1-cell output-token \
         distribution: 1-cell={base_osl} 3-cell={cell_osl}"
    );
}

/// The full text of the run's `logs/aiperf.log` (empty if absent).
fn aiperf_log(r: &RunResult) -> String {
    match r.artifacts.find_file("**/aiperf.log") {
        Some(path) => std::fs::read_to_string(&path).unwrap_or_default(),
        None => String::new(),
    }
}

/// The HTTP+zstd dataset-serve observable lines: one per served source, naming the
/// dataset and encoding.
fn dataset_serve_observables(r: &RunResult) -> Vec<String> {
    aiperf_log(r)
        .lines()
        .filter(|l| l.contains("served dataset source over HTTP"))
        .map(str::to_string)
        .collect()
}
