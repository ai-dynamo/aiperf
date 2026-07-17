// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A cellular run over a non-synthetic `file`/`path` dataset ships the
//! dataset source from controller to cells over HTTP with streaming zstd, and each cell
//! recompiles it deterministically.
//!
//! A synthetic cellular run regenerates the identical dataset in every cell from
//! the shared seed, so nothing needs shipping. A `file`/`path` dataset cannot be
//! regenerated and its path is controller-local (unreachable by a k8s cell), so the
//! controller serves the source over the same HTTP+zstd plane as per-record
//! artifact uploads, and the cell downloads and recompiles it before
//! `build_file_dataset` runs.
//!
//! [`AIPERF_CELL_ARTIFACT_HTTP_FORCE`] exercises the transport over loopback: the
//! controller binds its artifact server on loopback, registers the dataset source,
//! injects the authority into each locally-launched cell, and the cells `GET
//! /dataset/{name}` it back over real TCP + zstd.
//!
//! The controller logs `served dataset source over HTTP … content_encoding=zstd`
//! for the transfer. The merged records must match the single-cell
//! dataset-deterministic conversation set.

mod common;
use common::*;
use serde_json::{Value, json};

/// One single-turn conversation per row; a full-coverage `request-count` dispatches
/// each exactly once (baseline == cellular record count).
const ROWS: u32 = 18;
/// Fixed seed so the baseline and cellular runs compile the identical dataset.
const SEED: u32 = 20260715;
/// Cells the forced multi-process run partitions across (>= 2; 3 exercises an uneven
/// round-robin split). `ROWS`/`CONCURRENCY` are both >= this so every phase budget and
/// concurrency cap slices cleanly (cellular requires `>= cell_count`).
const CELLS: u32 = 3;
/// Concurrency cap (>= `CELLS` so it splits per cell without flooring to 1).
const CONCURRENCY: u32 = 6;

/// Write a single-turn dataset file: `ROWS` deterministic `{"text", "output_length"}`
/// rows, one conversation each.
fn write_single_turn_file(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("prompts.jsonl");
    let mut body = String::new();
    for row in 0..ROWS {
        body.push_str(
            &json!({"text": format!("Stage G prompt number {row} over a file dataset"),
                    "output_length": 16})
            .to_string(),
        );
        body.push('\n');
    }
    std::fs::write(&path, body).unwrap();
    path
}

/// Run the single-turn file dataset against `h`'s mock at `cells` cells. When
/// `force_http`, additionally set the HTTP-force seam + the `info`-level artifact
/// filter so a multi-cell run ships the dataset over loopback HTTP+zstd and logs the
/// serve observable.
fn run_file_dataset(
    h: &AIPerfHarness,
    file: &std::path::Path,
    cells: u32,
    force_http: bool,
) -> RunResult {
    let mut env: Vec<(&str, &str)> = vec![("AIPERF_LOG", "warn,aiperf_cellular_artifact=info")];
    if force_http {
        env.push(("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"));
    }
    h.run_env(
        &format!(
            "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
             --input-file {} --custom-dataset-type single_turn \
             --concurrency {CONCURRENCY} --request-count {ROWS} \
             --random-seed {SEED} --cells {cells} --ui simple",
            h.mock.url,
            file.display(),
        ),
        &env,
    )
}

/// Deterministic, run-independent projection of one `profile_export.jsonl` record:
/// the GLOBAL dataset identity + the two dataset-deterministic metrics + error.
fn record_projection(r: &Value) -> String {
    let m = &r["metadata"];
    let met = &r["metrics"];
    json!({
        "conversation_id": m["conversation_id"],
        "turn_index": m["turn_index"],
        "input_sequence_length": met["input_sequence_length"],
        "output_sequence_length": met["output_sequence_length"],
        "error": r["error"],
    })
    .to_string()
}

fn sorted<T, F: Fn(&T) -> String>(items: &[T], f: F) -> Vec<String> {
    let mut v: Vec<String> = items.iter().map(f).collect();
    v.sort();
    v
}

fn aiperf_log(r: &RunResult) -> String {
    let path = r
        .artifacts
        .find_file("**/aiperf.log")
        .expect("logs/aiperf.log should exist");
    std::fs::read_to_string(&path).unwrap_or_default()
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

#[tokio::test]
async fn test_cellular_dataset_shipping_matches_single_cell() {
    // Flaky on macOS CI like the other artifact e2es; skip there.
    if cfg!(target_os = "macos") {
        return;
    }

    let tmp = tempfile::TempDir::new().unwrap();
    let file = write_single_turn_file(tmp.path());

    let h_base = AIPerfHarness::new().await;
    let baseline = run_file_dataset(&h_base, &file, 1, false);
    assert!(
        baseline.success(),
        "1-cell baseline run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );

    // Forced multi-process HTTP shipping: N cell subprocesses GET the dataset source
    // from the controller's loopback server over zstd, then recompile it.
    let h_cell = AIPerfHarness::new().await;
    let cellular = run_file_dataset(&h_cell, &file, CELLS, true);
    assert!(
        cellular.success(),
        "forced-HTTP {CELLS}-cell file-dataset run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
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

    let observables = dataset_serve_observables(&cellular);
    assert!(
        !observables.is_empty(),
        "no dataset-serve observable found in logs/aiperf.log — the source did not go \
         over HTTP (or the force seam did not engage). Log tail:\n{}",
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
            "dataset-serve observable is not zstd-encoded: {line}"
        );
    }
    // The baseline (single-process, no controller) must NOT have served any dataset.
    assert!(
        dataset_serve_observables(&baseline).is_empty(),
        "single-process baseline must not serve a dataset over HTTP, but observed: {:?}",
        dataset_serve_observables(&baseline)
    );
    eprintln!(
        "HTTP+zstd dataset shipping observed: {} serve(s):\n{}",
        observables.len(),
        observables.join("\n")
    );

    let recs_base = baseline.artifacts.jsonl();
    let recs_cell = cellular.artifacts.jsonl();
    assert_eq!(
        recs_base.len(),
        ROWS as usize,
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
        "records.jsonl deterministic row SET diverged after dataset shipping"
    );
}
