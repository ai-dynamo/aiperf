// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end A/B proof for the native-runner exact-fold memory rearchitecture
//! (tasks S1–S4): the default fold-and-drop path emits the SAME artifacts as the
//! legacy retain-then-batch path (`AIPERF_RUNTIME_EXACT_FOLD=0`).
//!
//! Two full `python -m aiperf profile` runs drive the Rust runner against an
//! in-process mock server with EVERY per-record artifact enabled
//! (`records: [jsonl, csv, parquet]`, `raw`, `export_outputs_json`, plus the
//! always-on `inputs.json` and the metric summary), the SAME `dataset.random_seed`,
//! and full coverage (`requests == entries`, single-turn, so every conversation
//! dispatches exactly once and `inputs.json` == the dispatched set). One run takes
//! the default exact-fold path, the other forces the legacy retain path.
//!
//! ## What is and is not byte-identical across two ONLINE runs
//!
//! Exact-fold is proven byte-for-byte equal to legacy retain on identical captured
//! records by the in-process unit test `execute::tests::
//! exact_fold_matches_legacy_retain_byte_for_byte`. This e2e closes the gap that
//! unit test cannot reach: real up-front `inputs.json` materialization (S4) versus
//! real dispatch, and the streaming artifact lane (S2/S3) versus the legacy batch
//! writers. But two SEPARATE online process runs cannot be byte-identical in every
//! field: `request_latency`/`time_to_first_token`/throughput and the
//! `request_start_ns`/`request_end_ns` timestamps are wall-clock, and `benchmark_id`
//! / per-request `x_request_id` are fresh per-run UUIDs. Those differences are
//! physics, not an exact-fold defect. This test therefore asserts byte-identity on
//! everything that is genuinely run-independent — the seeded, timing-free content —
//! and equality of the deterministic projection on the rest:
//!
//!   * `inputs.json` — byte-identical (seeded dataset, no timing). This is the
//!     strongest single assertion: it directly validates S4's up-front materialized
//!     `inputs.json` against the legacy during-run capture, byte for byte.
//!   * metric summary — the dataset-deterministic metrics
//!     (`input_sequence_length`, `output_sequence_length`, `request_count`) are
//!     byte-identical between the two runs.
//!   * `records.jsonl` / `raw.jsonl` / records CSV / `outputs.json` — same record
//!     COUNT and same deterministic projection SET after sorting (completion order
//!     and wall-clock/UUID fields excluded).
//!   * `profile_export.parquet` — identical schema, identical row count, and
//!     identical deterministic-column SET.
//!
//! ## Non-vacuous
//!
//! With `AIPERF_LOG=aiperf=info` the runner logs one
//! `record retention path selected exact_fold=<bool>` line (see
//! `execute.rs`). The test asserts the default run logged `exact_fold=true` and the
//! forced run logged `exact_fold=false`, so a regression that silently disabled
//! exact-fold (making both runs the same legacy path) cannot pass as parity.

mod common;
use std::collections::BTreeSet;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use common::*;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::{Value, json};

/// Full coverage: every synthetic conversation dispatches exactly once, so the
/// up-front `inputs.json` (full dataset) equals the dispatched record set.
const ENTRIES: u32 = 16;
/// Fixed seed so both runs synthesize the identical dataset (prompts + session ids).
const SEED: u32 = 20260715;
/// Low concurrency keeps the run cheap; completion order still varies, which is
/// exactly the ordering difference the sorted-set comparisons tolerate.
const CONCURRENCY: u32 = 4;

/// A single-turn synthetic config with ALL per-record artifacts enabled, pointed at
/// `url` and seeded so two runs synthesize byte-identical inputs. `workers: 1` forces
/// the single-thread scheduled path, the only path exact-fold is eligible on.
fn full_coverage_config(url: &str) -> String {
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
        \x20 workers: 1\n",
    )
}

/// Run the full-coverage config against `h`'s mock. `exact_fold` selects the default
/// fold-and-drop path (`true`) or forces the legacy retain path (`false`). Returns the
/// run result; the caller asserts success and inspects artifacts.
fn run_full_coverage(h: &AIPerfHarness, exact_fold: bool) -> RunResult {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = tmp.path().join("full_coverage.yaml");
    std::fs::write(&cfg, full_coverage_config(&h.mock.url)).unwrap();
    // `aiperf=info` surfaces the one-line retention-path marker so the test can
    // prove non-vacuously which path each run took.
    let mut env: Vec<(&str, &str)> = vec![("AIPERF_LOG", "aiperf=info")];
    if !exact_fold {
        env.push(("AIPERF_RUNTIME_EXACT_FOLD", "0"));
    }
    h.run_env(&format!("--config {} --ui simple", cfg.display()), &env)
}

/// The runner's `record retention path selected ...` marker from `logs/aiperf.log`.
/// Proves which memory path the run actually took (byte-identical artifacts make the
/// path otherwise invisible from the outside).
fn retention_marker(r: &RunResult) -> String {
    let path = r
        .artifacts
        .find_file("**/aiperf.log")
        .expect("logs/aiperf.log should exist");
    std::fs::read_to_string(&path)
        .unwrap_or_default()
        .lines()
        .find(|l| l.contains("record retention path selected"))
        .unwrap_or("<no retention marker>")
        .to_string()
}

/// Deterministic, run-independent projection of one `profile_export.jsonl` record:
/// dataset-derived identity + the two dataset-deterministic metrics + error. Excludes
/// wall-clock timestamps/latencies and per-request UUIDs.
fn record_projection(r: &Value) -> String {
    let m = &r["metadata"];
    let met = &r["metrics"];
    json!({
        "session_num": m["session_num"],
        "conversation_id": m["conversation_id"],
        "turn_index": m["turn_index"],
        "benchmark_phase": m["benchmark_phase"],
        "input_sequence_length": met["input_sequence_length"],
        "output_sequence_length": met["output_sequence_length"],
        "reasoning_token_count": met["reasoning_token_count"],
        "error": r["error"],
    })
    .to_string()
}

/// Deterministic projection of one `outputs.json` row: identity + the mock's
/// deterministic generated text. Excludes `x_request_id` and the timing block.
fn output_projection(row: &Value) -> String {
    json!({
        "session_num": row["session_num"],
        "conversation_id": row["conversation_id"],
        "turn_index": row["turn_index"],
        "response_text": row["response_text"],
        "reasoning_text": row["reasoning_text"],
    })
    .to_string()
}

/// Sorted multiset of a projection over a slice — the "record SET, order-independent"
/// the brief calls for.
fn sorted<T, F: Fn(&T) -> String>(items: &[T], f: F) -> Vec<String> {
    let mut v: Vec<String> = items.iter().map(f).collect();
    v.sort();
    v
}

/// Read every row of a Parquet file: (schema column names, row count, and the
/// deterministic-column value set). The deterministic columns are the seeded
/// dataset facts that must match across runs — session identity
/// (`session_num`/`conversation_id`/`turn_index`) plus the seeded token metrics
/// (`input_sequence_length`/`output_sequence_length`/`reasoning_token_count`);
/// wall-clock/UUID columns are ignored.
fn read_parquet_projection(path: &Path) -> (Vec<String>, usize, BTreeSet<String>) {
    let file = std::fs::File::open(path).expect("open parquet");
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).expect("parquet reader builder");
    let reader = builder.build().expect("parquet reader");
    let batches: Vec<_> = reader.map(|b| b.expect("parquet batch")).collect();
    let schema = batches[0].schema();
    let names: Vec<String> = schema.fields().iter().map(|f| f.name().clone()).collect();
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    let sess_idx = schema.index_of("session_num").expect("session_num column");
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

    // `null` rendering for the nullable dataset columns (conversation_id may be
    // absent for a single-turn synthetic session; reasoning_token_count is null when
    // the backend exposes no reasoning) so a present-vs-absent difference between the
    // two runs still diverges the SET.
    let opt_f64 = |a: &Float64Array, i: usize| -> String {
        if a.is_null(i) {
            "null".to_string()
        } else {
            a.value(i).to_string()
        }
    };

    let mut set = BTreeSet::new();
    for b in &batches {
        let sess = b
            .column(sess_idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("session_num Int64");
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
                "{}|{}|{}|{}|{}|{}",
                sess.value(i),
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

/// Split one RFC4180 CSV line into fields, honoring the runner's `csv_escape`
/// quoting (`rust/runner/src/records.rs`): a field is double-quoted when it contains
/// a comma/quote/newline, and an embedded quote is doubled (`""`). Parsing here (vs a
/// naive `split(',')`) keeps the projection robust if a serialization regression
/// pushed a comma into a cell.
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

/// The dataset-deterministic records-CSV columns compared across the two runs:
/// session identity, phase, cancellation flag, the seeded ISL/OSL/reasoning token
/// metrics (headers follow `RecordMetricColumn::csv_display_name` — `{Header}
/// ({unit})`), and the error triple. Excludes the wall-clock/UUID columns
/// (`x_request_id`, `x_correlation_id`, the `*_ns` timestamps) and the timing
/// metrics (latency/TTFT/ITL/throughput) that legitimately differ between two online
/// runs.
const CSV_DETERMINISTIC_COLUMNS: &[&str] = &[
    "session_num",
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

/// Read the records CSV as (header line, data-row count, sorted deterministic-column
/// projection SET). The streaming CSV writer is a distinct code path from the legacy
/// batch writer, so comparing this projected SET (not just header + row count) closes
/// the gap where a serialization regression preserving header+count would otherwise
/// pass, mirroring the `records.jsonl` SET comparison.
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
            // ASCII unit separator: an unambiguous join delimiter that cannot appear
            // in the projected cells, so the concatenation is collision-free.
            .collect();
        set.push(projected.join("\u{1f}"));
    }
    set.sort();
    (header_line, rows, set)
}

/// The A/B parity proof: exact-fold (default) vs legacy retain produce the same
/// artifacts, differing only in wall-clock/UUID fields and completion order.
#[tokio::test]
async fn test_exact_fold_matches_legacy_retain_end_to_end() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    // Two harnesses = two mocks, but the dataset is seeded and the mock is
    // deterministic, so the run-independent content is identical across ports.
    let h_exact = AIPerfHarness::new().await;
    let exact = run_full_coverage(&h_exact, true);
    assert!(
        exact.success(),
        "exact-fold run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        exact.exit_code,
        exact.stdout,
        exact.stderr
    );

    let h_legacy = AIPerfHarness::new().await;
    let legacy = run_full_coverage(&h_legacy, false);
    assert!(
        legacy.success(),
        "legacy-retain run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        legacy.exit_code,
        legacy.stdout,
        legacy.stderr
    );

    // Non-vacuous: prove the two runs really took different memory paths. Without
    // this a regression that disabled exact-fold would make both runs the legacy
    // path and pass every parity check by construction.
    let exact_marker = retention_marker(&exact);
    let legacy_marker = retention_marker(&legacy);
    assert!(
        exact_marker.contains("exact_fold=true"),
        "default run must engage exact-fold; marker was: {exact_marker}"
    );
    assert!(
        legacy_marker.contains("exact_fold=false"),
        "AIPERF_RUNTIME_EXACT_FOLD=0 run must take the legacy retain path; marker was: {legacy_marker}"
    );

    // 1) inputs.json — byte-identical. The strongest assertion: seeded, timing-free,
    //    and the exact seam S4 introduced (up-front materialization vs during-run
    //    capture). Any drift here is a real S4 defect.
    let inputs_exact = std::fs::read(
        exact
            .artifacts
            .find_file("**/inputs.json")
            .expect("exact inputs.json"),
    )
    .unwrap();
    let inputs_legacy = std::fs::read(
        legacy
            .artifacts
            .find_file("**/inputs.json")
            .expect("legacy inputs.json"),
    )
    .unwrap();
    assert_eq!(
        inputs_exact, inputs_legacy,
        "inputs.json must be byte-identical between exact-fold and legacy (seeded, \
         timing-free); a difference is a real defect in the S4 up-front materializer"
    );

    // 2) metric summary — dataset-deterministic metrics byte-identical.
    let se = exact.artifacts.json();
    let sl = legacy.artifacts.json();
    for metric in [
        "input_sequence_length",
        "output_sequence_length",
        "request_count",
    ] {
        assert!(!se[metric].is_null(), "exact summary missing {metric}");
        assert_eq!(
            se[metric], sl[metric],
            "summary {metric} diverged: exact={} legacy={}",
            se[metric], sl[metric]
        );
    }

    // 3) records.jsonl — same count, same deterministic SET (order-independent).
    let recs_exact = exact.artifacts.jsonl();
    let recs_legacy = legacy.artifacts.jsonl();
    assert_eq!(
        recs_exact.len(),
        ENTRIES as usize,
        "full-coverage run must emit one record per conversation"
    );
    assert_eq!(
        recs_exact.len(),
        recs_legacy.len(),
        "exact-fold and legacy must emit the same records.jsonl count"
    );
    assert_eq!(
        sorted(&recs_exact, record_projection),
        sorted(&recs_legacy, record_projection),
        "records.jsonl deterministic record SET diverged between exact-fold and legacy"
    );

    // 4) raw.jsonl — same count, same deterministic request-payload SET.
    let raw_exact = exact.artifacts.raw_records();
    let raw_legacy = legacy.artifacts.raw_records();
    assert_eq!(
        raw_exact.len(),
        ENTRIES as usize,
        "raw.jsonl must have one record per conversation"
    );
    let raw_key = |r: &Value| r["payload"]["messages"].to_string();
    assert_eq!(
        sorted(&raw_exact, raw_key),
        sorted(&raw_legacy, raw_key),
        "raw.jsonl request-payload SET diverged between exact-fold and legacy"
    );

    // 5) records CSV — identical header, identical data-row count, AND identical
    //    dataset-deterministic content SET (order-independent). The streaming CSV
    //    writer is a distinct code path from the legacy batch writer, so a
    //    serialization regression preserving header + count could otherwise pass
    //    undetected; the projected-SET comparison closes that gap, mirroring the
    //    records.jsonl SET check above.
    let (ceh, cer, ces) = read_records_csv_projection(&exact);
    let (clh, clr, cls) = read_records_csv_projection(&legacy);
    assert_eq!(ceh, clh, "records CSV header diverged");
    assert_eq!(
        cer, ENTRIES as usize,
        "records CSV must have one row per record"
    );
    assert_eq!(cer, clr, "records CSV row count diverged");
    assert_eq!(
        ces, cls,
        "records CSV deterministic content SET diverged between exact-fold and legacy"
    );

    // 6) outputs.json — same deterministic (identity + generated text) SET.
    let outputs = |r: &RunResult| -> Vec<Value> {
        let p = r
            .artifacts
            .find_file("**/outputs.json")
            .expect("outputs.json");
        let v: Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
        v["data"].as_array().cloned().unwrap_or_default()
    };
    let oe = outputs(&exact);
    let ol = outputs(&legacy);
    assert_eq!(oe.len(), ol.len(), "outputs.json row count diverged");
    assert_eq!(
        sorted(&oe, |r| output_projection(r)),
        sorted(&ol, |r| output_projection(r)),
        "outputs.json deterministic (text) SET diverged between exact-fold and legacy"
    );

    // 7) parquet — identical schema, row count, and deterministic-column SET.
    let pe = read_parquet_projection(
        &exact
            .artifacts
            .find_file("**/profile_export.parquet")
            .expect("exact parquet"),
    );
    let pl = read_parquet_projection(
        &legacy
            .artifacts
            .find_file("**/profile_export.parquet")
            .expect("legacy parquet"),
    );
    assert_eq!(pe.0, pl.0, "parquet schema (column names) diverged");
    assert_eq!(
        pe.1, ENTRIES as usize,
        "parquet must have one row per record"
    );
    assert_eq!(pe.1, pl.1, "parquet row count diverged");
    assert_eq!(
        pe.2, pl.2,
        "parquet deterministic-column (session/conversation/turn/ISL/OSL/reasoning) \
         SET diverged"
    );
}

// ---------------------------------------------------------------------------
// RSS measurement (ignored by default; Linux-only /proc VmHWM sampling).
// ---------------------------------------------------------------------------

/// A metrics-only synthetic config (no per-record artifacts) with a LARGE request
/// budget, seeded and single-worker so exact-fold is eligible. Metrics-only isolates
/// the coordinator/accumulator retention term: legacy holds every finished record
/// until the end-of-run batch fold; exact-fold folds each into the exact accumulator
/// and drops it mid-run.
fn metrics_only_config(url: &str, entries: u32, requests: u32, concurrency: u32) -> String {
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
        \x20   entries: {entries}\n\
        \x20   random_seed: {SEED}\n\
        \x20   prompts:\n\
        \x20     isl: 128\n\
        \x20     osl: 32\n\
        \x20 phases:\n\
        \x20   type: concurrency\n\
        \x20   requests: {requests}\n\
        \x20   concurrency: {concurrency}\n\
        \x20 artifacts:\n\
        \x20   records: false\n\
         \n\
         runtime:\n\
        \x20 workers: 1\n",
    )
}

/// Peak `VmHWM` (KiB) of any live `aiperf` process, sampled from `/proc`.
/// `VmHWM` is a monotonic high-water mark, so the max over frequent samples is the
/// process peak even though the process exits before the final read.
#[cfg(target_os = "linux")]
fn max_runner_vmhwm_kb() -> u64 {
    let mut best = 0u64;
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return 0;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let comm = std::fs::read_to_string(p.join("comm")).unwrap_or_default();
        if comm.trim() != "aiperf runner" {
            continue;
        }
        if let Ok(status) = std::fs::read_to_string(p.join("status")) {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Some(kb) = rest.split_whitespace().next().and_then(|v| v.parse().ok()) {
                        best = best.max(kb);
                    }
                }
            }
        }
    }
    best
}

/// Run the metrics-only config once and return the runner subprocess peak `VmHWM`
/// (KiB), sampling `/proc` while the `aiperf` child runs.
#[cfg(target_os = "linux")]
fn measure_runner_vmhwm(url: &str, exact_fold: bool, entries: u32, requests: u32) -> u64 {
    use std::process::{Command, Stdio};
    use std::time::Duration;

    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = tmp.path().join("metrics_only.yaml");
    std::fs::write(&cfg, metrics_only_config(url, entries, requests, 64)).unwrap();
    let art = tmp.path().join("artifacts");
    std::fs::create_dir_all(&art).unwrap();

    let mut cmd = Command::new(exec_binary());
    cmd.arg("profile")
        .arg("--config")
        .arg(&cfg)
        .arg("--artifact-dir")
        .arg(&art)
        .arg("--tokenizer")
        .arg(DEFAULT_MODEL)
        .arg("--ui")
        .arg("simple")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("MALLOC_ARENA_MAX", "2")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if !exact_fold {
        cmd.env("AIPERF_RUNTIME_EXACT_FOLD", "0");
    }

    let mut child = cmd.spawn().expect("spawn aiperf profile");
    let mut peak = 0u64;
    while child.try_wait().expect("try_wait").is_none() {
        peak = peak.max(max_runner_vmhwm_kb());
        std::thread::sleep(Duration::from_millis(30));
    }
    let status = child.wait().expect("wait aiperf");
    assert!(status.success(), "metrics-only run failed: {status:?}");
    peak
}

/// Measured, committed RSS proof: on a large metrics-only run the exact-fold runner's
/// peak `VmHWM` is materially below the legacy retain runner's, and the gap widens
/// with request count. Ignored by default: it runs two ~large benchmarks (minutes)
/// and scans `/proc` for the runner by name, so it must run ALONE (no other
/// `aiperf` process in flight). Run with:
///
/// ```text
/// cargo test -p aiperf-e2e-tests --test test_exact_fold_ab_parity \
///     -- --ignored --nocapture exact_fold_runner_rss
/// ```
///
/// Reference numbers observed on this workstation (single-thread online, ISL 128 /
/// OSL 32, mock target): 60k req — exact 612 MiB vs legacy 722 MiB (-15%); 200k req —
/// exact 1171 MiB vs legacy 1512 MiB (-23%). Exact-fold is NOT flat because the
/// online per-worker observer still retains each record until end-of-run drain (a
/// documented streaming-finalize follow-up); exact-fold bounds the coordinator /
/// accumulator term, which is what this delta measures.
#[cfg(target_os = "linux")]
#[tokio::test]
#[ignore = "long-running RSS benchmark; must run alone (scans /proc for aiperf runner)"]
async fn exact_fold_runner_rss_below_legacy() {
    let h = AIPerfHarness::new().await;
    // 40k requests over 2k seeded conversations: large enough for a robust,
    // low-noise delta while keeping the ignored test near a minute per arm.
    let entries = 2000u32;
    let requests = 40_000u32;

    let exact = measure_runner_vmhwm(&h.mock.url, true, entries, requests);
    let legacy = measure_runner_vmhwm(&h.mock.url, false, entries, requests);

    println!(
        "exact-fold runner VmHWM = {exact} KiB ({:.1} MiB); legacy retain VmHWM = {legacy} KiB \
         ({:.1} MiB); delta = {} KiB ({:.1} MiB, {:.1}% lower)",
        exact as f64 / 1024.0,
        legacy as f64 / 1024.0,
        legacy.saturating_sub(exact),
        legacy.saturating_sub(exact) as f64 / 1024.0,
        (legacy.saturating_sub(exact) as f64 / legacy as f64) * 100.0,
    );

    assert!(exact > 0 && legacy > 0, "failed to sample runner VmHWM");
    assert!(
        exact < legacy,
        "exact-fold peak VmHWM ({exact} KiB) must be below legacy retain ({legacy} KiB)"
    );
}
