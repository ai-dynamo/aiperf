// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for multi-turn cellular (`--cells N`) runs.
//!
//! Multi-turn cellular budgets by CONVERSATION (session), not per turn: the controller
//! slices the phase `sessions` (`--num-conversations`) budget per cell with the same
//! `owned_positions` round-robin the sampler uses for its per-conversation stride, so
//! each cell single-passes its OWNED conversation slice. It is sound only on the
//! exact-fold merge (order-independent store concatenation), where a multi-turn
//! conversation's variable per-turn dispatch ordinal no longer matters.
//! Authored `inputs_json` runs use exact folding and must match the single-cell
//! deterministic records. Live-reply datasets require retained records and are rejected.
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

    // Integer sequence-length aggregates are order-independent.
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

/// Multi-turn cellular must never SILENTLY mis-merge: on the retain path its cells
/// ship partitions the controller cannot combine, and the run must be rejected with a
/// clear message.
///
/// This used to be reachable by a live-reply multi-turn synthetic dataset, because
/// `inputs.json` was captured DURING the run and that capture forced retain. It is not
/// anymore — `inputs.json` is a projection of the resident dataset, generated up front,
/// so nothing about a live-reply dataset forces retain and such a run now takes
/// exact-fold and merges correctly (proved by
/// `test_cellular_multi_turn_exact_fold_matches_single_cell`, which asserts 1-cell vs
/// N-cell metric parity on exactly that path).
///
/// So the backstop is exercised through the env force-switch that routes every path to
/// retain, which is what that switch exists for. Both directions are asserted: the
/// ordinary run must SUCCEED, and the forced-retain run must be REJECTED. Asserting
/// only the rejection would let a change that made retain unreachable pass while
/// silently deleting the coverage.
#[tokio::test]
async fn test_cellular_multi_turn_retain_is_rejected() {
    if cfg!(target_os = "macos") {
        return;
    }
    let h = AIPerfHarness::new().await;
    let args = format!(
        "--model {DEFAULT_MODEL} --url {} --endpoint-type chat \
         --num-conversations {SESSIONS} --session-turns-mean {TURNS} --session-turns-stddev 0 \
         --concurrency {CONCURRENCY} --cells {CELLS} --random-seed {SEED} \
         --synthetic-input-tokens-mean 32 --output-tokens-mean 8 --ui simple",
        h.mock.url
    );

    // Exact-fold: multi-turn cellular merges correctly and the run completes.
    let folded = h.run(&args);
    assert!(
        folded.success(),
        "multi-turn cellular must succeed on the exact-fold path; \
         inputs.json no longer forces retain:\nstdout:\n{}\nstderr:\n{}",
        folded.stdout,
        folded.stderr
    );

    // Forced retain: the cells ship partitions the controller cannot merge, so the
    // run must bail rather than emit a silently wrong merge.
    let retained = h.run_env(&args, &[("AIPERF_RUNTIME_EXACT_FOLD", "0")]);
    assert!(
        !retained.success(),
        "multi-turn cellular on the retain path must be rejected, but it succeeded:\nstdout:\n{}\nstderr:\n{}",
        retained.stdout,
        retained.stderr
    );
    let combined = format!("{}\n{}", retained.stdout, retained.stderr).to_lowercase();
    assert!(
        combined.contains("multi-turn") && combined.contains("exact-fold"),
        "rejection must name the multi-turn / exact-fold limitation; got:\nstdout:\n{}\nstderr:\n{}",
        retained.stdout,
        retained.stderr
    );
}
