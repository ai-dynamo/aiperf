// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for MULTI-TURN cellular (`--cells N`) runs, reached through the
//! ordinary Python frontend.
//!
//! Multi-turn cellular budgets by CONVERSATION (session), not per turn: the controller
//! slices the phase `sessions` (`--num-conversations`) budget per cell with the same
//! `owned_positions` round-robin the sampler uses for its per-conversation stride, so
//! each cell single-passes its OWNED conversation slice. It is sound only on the
//! exact-fold merge (order-independent store concatenation), where a multi-turn
//! conversation's variable per-turn dispatch ordinal no longer matters. These tests
//! prove both halves from `aiperf profile`:
//!
//! 1. [`test_cellular_multi_turn_exact_fold_matches_single_cell`] — an UP-FRONT-capable
//!    multi-turn dataset (`inputs_json`, whose per-turn bodies are authored, so its
//!    `inputs.json` is generated without dispatching and the run stays exact-fold) runs
//!    `--cells N` and reproduces the 1-cell run's dataset-deterministic metrics + the
//!    per-record row SET.
//! 2. [`test_cellular_multi_turn_retain_is_rejected`] — a live-reply multi-turn synthetic
//!    dataset (whose `inputs.json` must be captured DURING the run, forcing the retain
//!    path) is REJECTED with a clear message, rather than silently mis-merging.
//!
//! Requires the launched `aiperf` to include the `cellular` cell transport
//! (the default build).

mod common;
use common::*;
use serde_json::json;

/// Distinct conversations (sessions). `>= CELLS` so every cell owns at least one.
const SESSIONS: u32 = 12;
/// Authored turns per conversation (fixed, so the per-record count is deterministic).
const TURNS: u32 = 3;
/// Cells the multi-process run partitions across (3 exercises an uneven round-robin).
const CELLS: u32 = 3;
/// Concurrency cap (>= `CELLS` so it splits per cell without flooring to 1).
const CONCURRENCY: u32 = 6;
/// Fixed seed so the baseline and cellular runs compose the identical dataset.
const SEED: u32 = 20260715;

/// Write an `inputs_json` multi-turn dataset: `SESSIONS` sessions, each `TURNS`
/// self-contained authored chat payloads. `inputs_json` compiles each session into a
/// `MessageArrayWithResponses` conversation whose per-turn request bodies are known up
/// front, so the run generates `inputs.json` without dispatching and stays on the
/// exact-fold path (unlike a live-reply synthetic multi-turn dataset).
fn write_inputs_json_dataset(dir: &std::path::Path) -> std::path::PathBuf {
    let sessions: Vec<serde_json::Value> = (0..SESSIONS)
        .map(|session| {
            let payloads: Vec<serde_json::Value> = (0..TURNS)
                .map(|turn| {
                    // `inputs_json` sends each payload verbatim (raw wire body), so it must
                    // be a complete request the endpoint accepts — the mock requires `model`.
                    json!({
                        "model": DEFAULT_MODEL,
                        "messages": [{
                            "role": "user",
                            "content": format!("session {session} turn {turn}: describe topic {session}-{turn}"),
                        }],
                        "max_tokens": 16,
                    })
                })
                .collect();
            json!({"session_id": format!("s{session}"), "payloads": payloads})
        })
        .collect();
    let path = dir.join("inputs_multi_turn.json");
    std::fs::write(&path, json!({"data": sessions}).to_string()).unwrap();
    path
}

/// A YAML config: the `inputs_json` multi-turn `file` dataset at `file`, a session-bounded
/// (`sessions: SESSIONS`) profiling phase, partitioned across `cells`. `cells = 1` is the
/// single-process baseline; `cells >= 2` becomes a controller + cell subprocesses.
fn inputs_json_config(url: &str, file: &std::path::Path, cells: u32) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         randomSeed: {SEED}\n\
         \n\
         benchmark:\n\
        \x20 model: {DEFAULT_MODEL}\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   format: inputs_json\n\
        \x20   path: {}\n\
        \x20 profiling:\n\
        \x20   type: concurrency\n\
        \x20   sessions: {SESSIONS}\n\
        \x20   concurrency: {CONCURRENCY}\n\
        \x20 artifacts:\n\
        \x20   records:\n\
        \x20     - jsonl\n\
         \n\
         runtime:\n\
        \x20 cells: {cells}\n",
        file.display(),
    )
}

fn run_config(h: &AIPerfHarness, cfg_body: &str) -> RunResult {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg = tmp.path().join("multi_turn.yaml");
    std::fs::write(&cfg, cfg_body).unwrap();
    // The dataset file lives in `tmp` too; keep `tmp` alive until the run returns.
    let r = h.run(&format!("--config {} --ui simple", cfg.display()));
    drop(tmp);
    r
}

/// The sorted multiset of each profiling record's dataset-deterministic
/// `(conversation_id, turn_index, input_sequence_length, output_sequence_length)`
/// projection — a stable per-record key independent of wall-clock timing / UUIDs.
fn record_multiset(r: &RunResult) -> Vec<String> {
    let mut rows: Vec<String> = r
        .artifacts
        .jsonl()
        .iter()
        .filter(|record| record["metadata"]["benchmark_phase"] == "profiling")
        .map(|record| {
            json!({
                "conversation_id": record["metadata"]["conversation_id"],
                "turn_index": record["metadata"]["turn_index"],
                "isl": record["metrics"]["input_sequence_length"]["value"],
                "osl": record["metrics"]["output_sequence_length"]["value"],
            })
            .to_string()
        })
        .collect();
    rows.sort();
    rows
}

/// A multi-turn `--cells N` exact-fold run reproduces the 1-cell run's
/// dataset-deterministic metrics and per-record row SET.
#[tokio::test]
async fn test_cellular_multi_turn_exact_fold_matches_single_cell() {
    if cfg!(target_os = "macos") {
        return; // artifact e2es are flaky on macOS CI
    }
    let files = tempfile::TempDir::new().unwrap();
    let dataset = write_inputs_json_dataset(files.path());

    let h1 = AIPerfHarness::new().await;
    let baseline = run_config(&h1, &inputs_json_config(&h1.mock.url, &dataset, 1));
    assert!(
        baseline.success(),
        "1-cell multi-turn baseline failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );

    let h3 = AIPerfHarness::new().await;
    let cellular = run_config(&h3, &inputs_json_config(&h3.mock.url, &dataset, CELLS));
    assert!(
        cellular.success(),
        "{CELLS}-cell multi-turn run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );

    // Topology guard: the multi-cell run went through the controller (which alone writes
    // cellular-heartbeat.json); the baseline did not.
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

    // Every conversation's every turn dispatched exactly once in both runs.
    let base_records = record_multiset(&baseline);
    let cell_records = record_multiset(&cellular);
    assert_eq!(
        base_records.len(),
        (SESSIONS * TURNS) as usize,
        "baseline must dispatch every turn of every conversation once"
    );
    assert_eq!(
        base_records, cell_records,
        "multi-turn {CELLS}-cell merged per-record row SET must equal the 1-cell run's"
    );

    // Dataset-deterministic summary metrics must match within tolerance (counts/min/max
    // exact; the integer sequence-length averages are exact under the order-independent
    // concat merge).
    for metric in ["input_sequence_length", "output_sequence_length"] {
        let base = &baseline.artifacts.json()[metric];
        let cell = &cellular.artifacts.json()[metric];
        assert!(!base.is_null(), "baseline missing metric {metric}");
        for stat in ["min", "max", "avg", "p50", "p99"] {
            let (b, c) = (&base[stat], &cell[stat]);
            if let (Some(b), Some(c)) = (b.as_f64(), c.as_f64()) {
                assert!(
                    (b - c).abs() <= 1e-9 * b.abs().max(1.0),
                    "multi-turn cellular {metric}.{stat} diverged: 1-cell={b} {CELLS}-cell={c}"
                );
            }
        }
    }
}

/// A live-reply multi-turn SYNTHETIC dataset forces the retain path (its `inputs.json`
/// must be captured during the run), on which multi-turn cellular cannot merge correctly.
/// The run must be REJECTED with a clear message — never silently mis-merged.
#[tokio::test]
async fn test_cellular_multi_turn_retain_is_rejected() {
    if cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    // Synthetic multi-turn (turns > 1) with num-conversations: the gate admits it
    // (exact-fold predicted), but the cell falls to retain to capture the live-reply
    // inputs.json, and the controller's merge-time backstop bails.
    let r = h.run(&format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-conversations {SESSIONS} --session-turns-mean {TURNS} --session-turns-stddev 0 \
         --concurrency {CONCURRENCY} --cells {CELLS} --random-seed {SEED} \
         --synthetic-input-tokens-mean 32 --output-tokens-mean 8 --ui simple",
        h.mock.url
    ));
    assert!(
        !r.success(),
        "live-reply multi-turn synthetic cellular must be rejected (retain path), but it succeeded"
    );
    let combined = format!("{}\n{}", r.stdout, r.stderr).to_lowercase();
    assert!(
        combined.contains("multi-turn") && combined.contains("exact-fold"),
        "rejection must name the multi-turn / exact-fold limitation; got:\nstdout:\n{}\nstderr:\n{}",
        r.stdout,
        r.stderr
    );
}
