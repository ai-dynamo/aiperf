// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A cellular run ships its per-record
//! artifacts between real processes over **HTTP + streaming zstd** and the
//! controller reassembles them correctly.
//!
//! `AIPERF_CELL_ARTIFACT_HTTP_FORCE` exercises the transport on localhost with
//! real `aiperf --cell` subprocesses, the
//! real controller HTTP artifact upload server bound on loopback, real zstd over a
//! TCP socket, and controller artifact concatenation.
//!
//! The cross-host HTTP artifact path is normally gated to k8s (a `tcp://` velo
//! coordinate). The test/dev force seam
//! `AIPERF_CELL_ARTIFACT_HTTP_FORCE` makes a SAME-HOST `--cells N`
//! run drive it over loopback: the controller binds its upload server on
//! `127.0.0.1:0`, injects that authority into each locally-launched cell, the cells
//! POST their per-record artifact files (+ `inputs.json`) back with
//! `Content-Encoding: zstd`, and the controller concatenates the SHIPPED copies —
//! not any shared-FS write — into the run artifact dir. Default local `--cells N`
//! (flag unset) is byte-unchanged (shared-FS concat), which
//! `test_cellular::test_cellular_matches_single_cell` continues to guard.
//!
//! Merged per-record artifacts must match the single-cell deterministic row set,
//! and `inputs.json` must be byte-identical. The upload handler logs
//! `received artifact upload over HTTP … content_encoding=zstd` for each transfer.

mod common;
use std::collections::BTreeSet;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use common::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};

/// Full coverage: every synthetic conversation dispatches exactly once, so the
/// up-front `inputs.json` (full dataset) equals the dispatched record set.
const ENTRIES: u32 = 18;
/// Fixed seed so the baseline and cellular runs synthesize the identical dataset.
const SEED: u32 = 20260715;
/// Cells the forced multi-process run partitions across (>= 2; 3 exercises an uneven
/// round-robin split). `ENTRIES`/`CONCURRENCY` are both >= this so every phase budget
/// and concurrency cap slices cleanly (cellular requires `>= cell_count`).
const CELLS: u32 = 3;
/// Concurrency cap (>= `CELLS` so it splits per cell without flooring to 1).
const CONCURRENCY: u32 = 6;

/// A single-turn synthetic config with ALL per-record artifacts enabled, pointed at
/// `url`, seeded, and partitioned across `cells`. `cells = 1` is the single-process
/// baseline; `cells >= 2` becomes a controller + cell subprocesses.
fn full_coverage_config(url: &str, cells: u32) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         \n\
         benchmark:\n\
        \x20 model: {DEFAULT_MODEL}\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: synthetic\n\
        \x20   entries: {ENTRIES}\n\
        \x20   random_seed: {SEED}\n\
        \x20   prompts:\n\
        \x20     isl: 32\n\
        \x20     osl: 16\n\
        \x20 phases:\n\
        \x20   type: concurrency\n\
        \x20   requests: {ENTRIES}\n\
        \x20   concurrency: {CONCURRENCY}\n\
        \x20 artifacts:\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20     - csv\n\
        \x20     - parquet\n\
        \x20   raw: true\n\
        \x20   export_outputs_json: true\n\
         \n\
         runtime:\n\
        \x20 cells: {cells}\n"
    )
}

/// Run the full-coverage config against `h`'s mock at `cells` cells. When `force_http`
/// the run additionally sets the HTTP-force seam + the `info`-level artifact-upload
/// filter, so a multi-cell run ships its artifacts over loopback HTTP+zstd and logs one
/// observable per cell × file into `logs/aiperf.log`.
fn run_full_coverage(h: &AIPerfHarness, cells: u32, force_http: bool) -> RunResult {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = tmp.path().join("full_coverage.yaml");
    std::fs::write(&cfg, full_coverage_config(&h.mock.url, cells)).unwrap();
    // Surface just the cellular artifact-upload observable at `info`; everything else
    // stays at the runner's default `warn`. `aiperf=info` is required alongside it: the
    // upload event fires in the `--execute` child, and the parent re-emits forwarded child
    // lines under target `aiperf` (`cli/src/execute.rs`) into the `logs/aiperf.log` it owns.
    let mut env: Vec<(&str, &str)> =
        vec![("AIPERF_LOG", "warn,aiperf=info,aiperf_cellular_artifact=info")];
    if force_http {
        env.push(("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"));
    }
    h.run_env(&format!("--config {} --ui simple", cfg.display()), &env)
}

/// Deterministic, run-independent projection of one `profile_export.jsonl` record:
/// the GLOBAL dataset identity (`conversation_id`) + the two dataset-deterministic
/// metrics + error. Excludes wall-clock timestamps/latencies, per-request UUIDs, and
/// `session_num` — the latter is a per-CELL local counter under cellular mode (cell 0
/// and cell 1 both start it at 0), an accepted cellular characteristic the existing
/// cellular tests likewise exclude; `conversation_id` (`session_0000NN`) is the stable
/// global key that must match across topologies.
fn record_projection(r: &Value) -> String {
    let m = &r["metadata"];
    let met = &r["metrics"];
    json!({
        "conversation_id": m["conversation_id"],
        "turn_index": m["turn_index"],
        "benchmark_phase": m["benchmark_phase"],
        "input_sequence_length": met["input_sequence_length"],
        "output_sequence_length": met["output_sequence_length"],
        "error": r["error"],
    })
    .to_string()
}

/// Deterministic projection of one `outputs.json` row: GLOBAL identity
/// (`conversation_id`) + the mock's deterministic generated text. Excludes
/// `x_request_id`, the timing block, and the per-cell-local `session_num`.
fn output_projection(row: &Value) -> String {
    json!({
        "conversation_id": row["conversation_id"],
        "turn_index": row["turn_index"],
        "response_text": row["response_text"],
        "reasoning_text": row["reasoning_text"],
    })
    .to_string()
}

/// Sorted multiset of a projection over a slice.
fn sorted<T, F: Fn(&T) -> String>(items: &[T], f: F) -> Vec<String> {
    let mut v: Vec<String> = items.iter().map(f).collect();
    v.sort();
    v
}

/// Read a Parquet file's (schema column names, row count, deterministic-column SET).
fn read_parquet_projection(path: &Path) -> (Vec<String>, usize, BTreeSet<String>) {
    let file = std::fs::File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader builder");
    let reader = builder.build().expect("parquet reader");
    let batches: Vec<_> = reader.map(|b| b.expect("parquet batch")).collect();
    let schema = batches[0].schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    let conv_idx = schema
        .index_of("conversation_id")
        .expect("conversation_id column");
    let turn_idx = schema.index_of("turn_index").expect("turn_index column");
    let isl_idx = schema
        .index_of("input_sequence_length")
        .expect("input_sequence_length column");
    let osl_idx = schema
        .index_of("output_sequence_length")
        .expect("output_sequence_length column");
    let reasoning_idx = schema
        .index_of("reasoning_token_count")
        .expect("reasoning_token_count column");

    let opt_f64 = |a: &Float64Array, i: usize| -> String {
        if a.is_null(i) {
            "null".to_string()
        } else {
            a.value(i).to_string()
        }
    };

    let mut set = BTreeSet::new();
    for b in &batches {
        let conv = b
            .column(conv_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("conversation_id Utf8");
        let turn = b
            .column(turn_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("turn_index Int64");
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
        let reasoning = b
            .column(reasoning_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("reasoning_token_count Float64");
        for i in 0..b.num_rows() {
            let conv_cell = if conv.is_null(i) {
                "null".to_string()
            } else {
                conv.value(i).to_string()
            };
            set.insert(format!(
                "{}|{}|{}|{}|{}",
                conv_cell,
                turn.value(i),
                isl.value(i),
                osl.value(i),
                opt_f64(reasoning, i),
            ));
        }
    }
    (names, rows, set)
}

/// Split one RFC4180 CSV line into fields, honoring the runner's `csv_escape` quoting.
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

/// The dataset-deterministic records-CSV columns compared across runs. Excludes
/// `session_num` (per-cell-local under cellular mode); `conversation_id` is the stable
/// global key.
const CSV_DETERMINISTIC_COLUMNS: &[&str] = &[
    "conversation_id",
    "turn_index",
    "benchmark_phase",
    "was_cancelled",
    "Input Sequence Length (tokens)",
    "Output Sequence Length (tokens)",
    "Reasoning Token Count (tokens)",
    "error_code",
    "error_type",
    "error_message",
];

/// Read the records CSV as (header line, data-row count, sorted deterministic SET).
fn read_records_csv_projection(r: &RunResult) -> (String, usize, Vec<String>) {
    let p = r
        .artifacts
        .find_file("**/profile_export_records.csv")
        .expect("records csv");
    let text = std::fs::read_to_string(&p).unwrap();
    let mut it = text.lines();
    let header_line = it.next().unwrap_or_default().to_string();
    let header = parse_csv_line(&header_line);
    let col_idx: Vec<usize> = CSV_DETERMINISTIC_COLUMNS
        .iter()
        .map(|name| {
            header
                .iter()
                .position(|c| c == name)
                .unwrap_or_else(|| panic!("records CSV missing column {name}: {header:?}"))
        })
        .collect();

    let mut set = Vec::new();
    let mut rows = 0usize;
    for line in it {
        if line.trim().is_empty() {
            continue;
        }
        rows += 1;
        let fields = parse_csv_line(line);
        let projected: Vec<&str> = col_idx
            .iter()
            .map(|&i| fields.get(i).map(String::as_str).unwrap_or(""))
            .collect();
        set.push(projected.join("\u{1f}"));
    }
    set.sort();
    (header_line, rows, set)
}

/// Read `outputs.json`'s `data` array.
fn outputs(r: &RunResult) -> Vec<Value> {
    let p = r
        .artifacts
        .find_file("**/outputs.json")
        .expect("outputs.json");
    let v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    v["data"].as_array().cloned().unwrap_or_default()
}

/// Every forwarded-runner log line in `logs/aiperf.log`.
fn aiperf_log(r: &RunResult) -> String {
    let path = r
        .artifacts
        .find_file("**/aiperf.log")
        .expect("logs/aiperf.log should exist");
    std::fs::read_to_string(&path).unwrap_or_default()
}

/// The HTTP+zstd artifact-upload observable lines: one per received cell × file, each
/// naming the cell, encoding, and on-wire byte count.
fn upload_observables(r: &RunResult) -> Vec<String> {
    aiperf_log(r)
        .lines()
        .filter(|l| l.contains("received artifact upload over HTTP"))
        .map(str::to_string)
        .collect()
}

/// A same-host multi-process `--cells N` run with the HTTP-force seam ships
/// every per-record artifact over real HTTP+zstd between real cell subprocesses and the
/// controller, and the merged result matches a single-cell exact-fold baseline.
#[tokio::test]
async fn test_cellular_http_shipping_matches_single_cell() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    let h_base = AIPerfHarness::new().await;
    let baseline = run_full_coverage(&h_base, 1, false);
    assert!(
        baseline.success(),
        "1-cell baseline run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );

    // Forced multi-process HTTP shipping: N cell subprocesses POST their artifacts to
    // the controller's loopback upload server over zstd; the controller concatenates
    // the SHIPPED copies.
    let h_cell = AIPerfHarness::new().await;
    let cellular = run_full_coverage(&h_cell, CELLS, true);
    assert!(
        cellular.success(),
        "forced-HTTP {CELLS}-cell run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );

    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "{CELLS}-cell run must go through the controller (cellular-heartbeat.json sidecar)"
    );
    assert!(
        baseline
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_none(),
        "1-cell baseline must be single-process (no cellular sidecar)"
    );

    let observables = upload_observables(&cellular);
    assert!(
        !observables.is_empty(),
        "no HTTP artifact-upload observable found in logs/aiperf.log — the bytes did \
         not go over HTTP (or the force seam did not engage). Log tail:\n{}",
        aiperf_log(&cellular)
            .lines()
            .rev()
            .take(40)
            .collect::<Vec<_>>()
            .join("\n")
    );
    // The baseline (single-process, no shipping) must NOT have produced any.
    assert!(
        upload_observables(&baseline).is_empty(),
        "single-process baseline must not ship artifacts over HTTP, but observed: {:?}",
        upload_observables(&baseline)
    );
    // Every zstd observable really carries the zstd encoding.
    for line in &observables {
        assert!(
            line.contains("content_encoding=\"zstd\"") || line.contains("content_encoding=zstd"),
            "artifact-upload observable is not zstd-encoded: {line}"
        );
    }
    // Every cell shipped at least one file over HTTP (the records.jsonl each cell
    // always writes), and the always-on inputs.json crossed the wire for each cell.
    for cell_id in 0..CELLS {
        let cell_lines: Vec<&String> = observables
            .iter()
            .filter(|l| l.contains(&format!("cell_id={cell_id}")))
            .collect();
        assert!(
            !cell_lines.is_empty(),
            "cell {cell_id} shipped no artifacts over HTTP; observables:\n{}",
            observables.join("\n")
        );
        assert!(
            cell_lines
                .iter()
                .any(|l| l.contains("inputs.json") && l.contains("zstd")),
            "cell {cell_id} did not ship inputs.json over HTTP+zstd; its uploads:\n{}",
            cell_lines
                .iter()
                .map(|l| l.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
    eprintln!(
        "HTTP+zstd shipping observed: {} artifact uploads across {} cells:\n{}",
        observables.len(),
        CELLS,
        observables.join("\n")
    );

    // The controller merges the per-cell inputs slices back into the full seeded document:
    // re-interleaved round-robin (the inverse of the partition) with each payload body's
    // bytes carried through verbatim, so it is byte-identical to the single-process file.
    let inputs_base = std::fs::read(
        baseline
            .artifacts
            .find_file("**/inputs.json")
            .expect("baseline inputs.json"),
    )
    .unwrap();
    let inputs_cell = std::fs::read(
        cellular
            .artifacts
            .find_file("**/inputs.json")
            .expect("cellular inputs.json"),
    )
    .unwrap();
    assert_eq!(
        inputs_base, inputs_cell,
        "inputs.json must be byte-identical between the baseline and the HTTP-shipped \
         cellular run (seeded, timing-free)"
    );

    let recs_base = baseline.artifacts.jsonl();
    let recs_cell = cellular.artifacts.jsonl();
    assert_eq!(
        recs_base.len(),
        ENTRIES as usize,
        "full-coverage baseline must emit one record per conversation"
    );
    assert_eq!(
        recs_base.len(),
        recs_cell.len(),
        "baseline and HTTP-shipped cellular must emit the same records.jsonl count"
    );
    assert_eq!(
        sorted(&recs_base, record_projection),
        sorted(&recs_cell, record_projection),
        "records.jsonl deterministic row SET diverged after HTTP shipping"
    );

    let raw_base = baseline.artifacts.raw_records();
    let raw_cell = cellular.artifacts.raw_records();
    assert_eq!(
        raw_base.len(),
        ENTRIES as usize,
        "raw.jsonl must have one record per conversation"
    );
    let raw_key = |r: &Value| r["payload"]["messages"].to_string();
    assert_eq!(
        sorted(&raw_base, raw_key),
        sorted(&raw_cell, raw_key),
        "raw.jsonl request-payload SET diverged after HTTP shipping"
    );

    let (bh, br, bs) = read_records_csv_projection(&baseline);
    let (ch, cr, cs) = read_records_csv_projection(&cellular);
    assert_eq!(bh, ch, "records CSV header diverged");
    assert_eq!(
        br, ENTRIES as usize,
        "records CSV must have one row per record"
    );
    assert_eq!(br, cr, "records CSV row count diverged after HTTP shipping");
    assert_eq!(
        bs, cs,
        "records CSV deterministic content SET diverged after HTTP shipping"
    );

    let ob = outputs(&baseline);
    let oc = outputs(&cellular);
    assert_eq!(ob.len(), oc.len(), "outputs.json row count diverged");
    assert_eq!(
        sorted(&ob, |r| output_projection(r)),
        sorted(&oc, |r| output_projection(r)),
        "outputs.json deterministic (text) SET diverged after HTTP shipping"
    );

    let pb = read_parquet_projection(
        &baseline
            .artifacts
            .find_file("**/profile_export.parquet")
            .expect("baseline parquet"),
    );
    let pc = read_parquet_projection(
        &cellular
            .artifacts
            .find_file("**/profile_export.parquet")
            .expect("cellular parquet"),
    );
    assert_eq!(pb.0, pc.0, "parquet schema diverged after HTTP shipping");
    assert_eq!(
        pb.1, ENTRIES as usize,
        "baseline parquet must have one row per record"
    );
    assert_eq!(pb.1, pc.1, "parquet row count diverged after HTTP shipping");
    assert_eq!(
        pb.2, pc.2,
        "parquet deterministic-column SET diverged after HTTP shipping"
    );
}
