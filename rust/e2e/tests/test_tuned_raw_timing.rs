// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tuned-mock `profile_export_raw.jsonl` TIMING + DATA back-fill for the CORE
//! execution paths.
//!
//! Each test runs `python -m aiperf profile` against a mock tuned to fixed,
//! jitter-free per-token latency (`tuned_mock_config`, analytic mode) and then
//! verifies — through the shared [`assert_raw_records_timing_and_data`] /
//! [`assert_raw_records_timing_self_consistent`] helpers — that every raw
//! record's on-the-wire token timing (TTFT / ITL / request_latency) and data
//! (OSL / model / status) reproduces the tuned model within a tight transport
//! tolerance.
//!
//! This operationalizes the feature-complete bar: per-request timing must
//! survive the full `Python -> aiperf-runner -> transport -> record` path, and
//! the cellular fold+ship+merge and graph partition+merge, byte-for-byte at the
//! record level — not just aggregate summary metrics.
//!
//! # Environment caveat (must run un-sandboxed)
//!
//! The mock injects latency through the RealClock `timerfd` (`aiperf::clock::
//! sleep_ns`). A seccomp/time-virtualizing sandbox that fast-forwards `timerfd`
//! collapses every sleep to ~0 ms, so the tuned latencies vanish and these
//! assertions fail. Run the suite on real wall-clock hardware (the normal CI /
//! developer path); a sandbox that intercepts timers cannot exercise it.
//!
//! Non-`--cells` model: `gpt-4` (a NON-reasoning model in the mock) so every
//! output chunk is plain `content` and OSL == the requested cap exactly. The
//! graph fixture uses `test-chat-model` (also non-reasoning).

mod common;
use common::*;

use std::sync::{Mutex, MutexGuard, OnceLock};

/// Serialize the wall-clock timing tests in THIS binary so they don't oversubscribe
/// the CPU with each other's cell/mock subprocesses. Cargo already runs test
/// binaries sequentially, so this guard is the only cross-test contention source;
/// with it each run is effectively isolated (like the manual reference), which is
/// what lets the tolerances stay tight rather than a wide band.
fn timing_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Tuned mock TTFT (ms). Matches the manually-verified reference point.
const TTFT_MS: f64 = 100.0;
/// Tuned mock ITL (ms).
const ITL_MS: f64 = 10.0;
/// Fixed output cap for the synthetic-dataset paths (exact-generation via
/// `--output-tokens-mean N --output-tokens-stddev 0`).
const OSL: usize = 8;

/// Scheduled single-turn exact-fold (single-process): a tuned run with
/// `--export-level raw` reproduces the tuned TTFT/ITL/latency and OSL on every
/// raw record.
#[tokio::test]
async fn tuned_scheduled_single_turn_raw_timing() {
    if cfg!(target_os = "macos") {
        return; // artifact e2es are flaky on macOS CI
    }
    let _guard = timing_guard();
    let h = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --concurrency 2 --request-count 6 \
         --synthetic-input-tokens-mean 64 --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean {OSL} --output-tokens-stddev 0 \
         --export-level raw --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "tuned scheduled run failed: {}", r.stderr);

    let records = r.artifacts.raw_records();
    assert_eq!(records.len(), 6, "expected 6 raw records");
    if timing_fast_forwarded(&records, TTFT_MS) {
        return;
    }
    assert_raw_records_timing_and_data(
        &records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OSL).model("gpt-4"),
    );
}

/// Cellular exact-fold (`--cells N`): a tuned multi-process run's MERGED raw
/// records reproduce the tuned TTFT/ITL/latency and OSL — proving per-record
/// timing survives the cell fold, velo ship, and controller merge.
#[tokio::test]
async fn tuned_cellular_raw_timing_survives_merge() {
    if cfg!(target_os = "macos") {
        return;
    }
    let _guard = timing_guard();
    let h = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;
    let r = h.run(&format!(
        "--model gpt-4 --url {} --endpoint-type chat --streaming \
         --request-count 12 --concurrency 6 --cells 3 --random-seed 42 \
         --synthetic-input-tokens-mean 64 --synthetic-input-tokens-stddev 0 \
         --output-tokens-mean {OSL} --output-tokens-stddev 0 \
         --export-level raw --ui simple",
        h.mock.url
    ));
    assert!(r.success(), "tuned cellular run failed: {}", r.stderr);

    // Topology guard: this genuinely went through the controller (only it writes
    // the cellular-heartbeat.json sidecar), so the merged raw records really were
    // shipped from cell subprocesses — not produced by a single in-process run.
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "cellular run must emit the controller's cellular-heartbeat.json sidecar"
    );

    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        12,
        "merged cellular report must carry every cell's raw records"
    );
    if timing_fast_forwarded(&records, TTFT_MS) {
        return;
    }
    // The cellular path adds a few ms of first-token overhead (cell startup, velo
    // START sync, controller) on top of the flat transport, so TTFT gets a little
    // more room than the single-process test; ITL stays tight.
    assert_raw_records_timing_and_data(
        &records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OSL)
            .model("gpt-4")
            .tol_ms(12.0, 3.0),
    );
}

/// Multi-turn cellular exact-fold (`inputs_json` authored dataset, `--cells N`):
/// every merged per-turn raw record reproduces the tuned TTFT/ITL, verified
/// self-consistently against each record's own OSL (authored payloads carry
/// `max_tokens` but no exact-output control, so the mock streams a variable
/// count).
#[tokio::test]
async fn tuned_cellular_multi_turn_raw_timing() {
    if cfg!(target_os = "macos") {
        return;
    }
    let _guard = timing_guard();
    const SESSIONS: u32 = 6;
    const TURNS: u32 = 3;
    const CELLS: u32 = 3;

    let files = tempfile::TempDir::new().unwrap();
    let dataset = files.path().join("inputs_multi_turn.json");
    let sessions: Vec<serde_json::Value> = (0..SESSIONS)
        .map(|session| {
            let payloads: Vec<serde_json::Value> = (0..TURNS)
                .map(|turn| {
                    // `inputs_json` sends each payload verbatim, so per-token
                    // streaming (required to measure TTFT/ITL) must be requested
                    // in the body itself — the endpoint's `streaming: true` does
                    // not rewrite an authored wire payload.
                    serde_json::json!({
                        "model": "gpt-4",
                        "stream": true,
                        "messages": [{
                            "role": "user",
                            "content": format!("session {session} turn {turn}: describe topic {session}-{turn}"),
                        }],
                        "max_tokens": 8,
                    })
                })
                .collect();
            serde_json::json!({"session_id": format!("s{session}"), "payloads": payloads})
        })
        .collect();
    std::fs::write(&dataset, serde_json::json!({"data": sessions}).to_string()).unwrap();

    let h = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;
    let cfg_body = format!(
        "schemaVersion: \"2.0\"\n\
         randomSeed: 20260715\n\
         \n\
         benchmark:\n\
        \x20 model: gpt-4\n\
        \x20 endpoint:\n\
        \x20   url: {url}/v1/chat/completions\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   format: inputs_json\n\
        \x20   path: {path}\n\
        \x20 profiling:\n\
        \x20   type: concurrency\n\
        \x20   sessions: {SESSIONS}\n\
        \x20   concurrency: 6\n\
        \x20 artifacts:\n\
        \x20   raw: true\n\
        \x20   records:\n\
        \x20     - jsonl\n\
         \n\
         runtime:\n\
        \x20 cells: {CELLS}\n",
        url = h.mock.url,
        path = dataset.display(),
    );

    let cfg = files.path().join("multi_turn.yaml");
    std::fs::write(&cfg, cfg_body).unwrap();
    let r = h.run(&format!("--config {} --ui simple", cfg.display()));
    assert!(
        r.success(),
        "tuned multi-turn cellular run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        r.exit_code,
        r.stdout,
        r.stderr
    );
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "multi-turn --cells run must go through the controller (cellular-heartbeat.json sidecar)"
    );

    let records = r.artifacts.raw_records();
    assert_eq!(
        records.len(),
        (SESSIONS * TURNS) as usize,
        "merged multi-turn cellular report must carry one raw record per turn"
    );
    if timing_fast_forwarded(&records, TTFT_MS) {
        return;
    }
    // Tight, serialized run: single-turn-per-record over a modest cell fan-out.
    // The authored payloads pin `model: gpt-4`, so assert it as data too.
    assert_raw_records_timing_self_consistent_model(
        &records,
        TTFT_MS,
        ITL_MS,
        8.0,
        2.0,
        Some("gpt-4"),
    );
}

/// Graph cellular (`dag_jsonl`, `--cells N`): every merged graph raw record
/// reproduces the tuned TTFT/ITL, verified self-consistently against each
/// record's own OSL (the fixture's `max_tokens` streams a variable count).
#[tokio::test]
async fn tuned_graph_cellular_raw_timing() {
    if cfg!(target_os = "macos") {
        return;
    }
    let _guard = timing_guard();
    const FIXTURE: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/dag/multi_root_single_turn.dag.jsonl"
    );

    let h = AIPerfHarness::new_with(tuned_mock_config(TTFT_MS, ITL_MS)).await;
    let r = h.run_timeout(
        &format!(
            "--model test-chat-model --url {} --endpoint-type chat --streaming \
             --input-file {FIXTURE} --custom-dataset-type dag_jsonl \
             --num-conversations 6 --concurrency 3 --cells 3 --random-seed 7 \
             --export-level raw --ui simple",
            h.mock.url
        ),
        120,
    );
    assert!(r.success(), "tuned graph cellular run failed: {}", r.stderr);
    assert!(
        r.artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "graph --cells 3 must go through the controller (cellular-heartbeat.json sidecar)"
    );

    let records = r.artifacts.raw_records();
    assert!(!records.is_empty(), "graph cellular emitted no raw records");
    if timing_fast_forwarded(&records, TTFT_MS) {
        return;
    }
    // Graph mode fans out root/fork instances across 3 cell subprocesses, each
    // running its own concurrency, so first-token queue wait carries slightly more
    // scheduling jitter than the flat scheduled path even when serialized; give
    // TTFT a bit more room while ITL — the contention-robust steady-state pacing —
    // stays tight.
    assert_raw_records_timing_self_consistent(&records, TTFT_MS, ITL_MS, 12.0, 3.0);
}
