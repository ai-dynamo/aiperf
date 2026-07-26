// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::*;

use std::sync::{Mutex, MutexGuard, OnceLock};

// Serial execution keeps wall-clock tolerances meaningful under CPU contention.
fn timing_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

const TTFT_MS: f64 = 100.0;
const ITL_MS: f64 = 10.0;
const OSL: usize = 8;

#[tokio::test]
async fn tuned_scheduled_single_turn_raw_timing() {
    if cfg!(target_os = "macos") {
        return;
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
    assert_raw_records_timing_and_data(
        &records,
        &TunedExpectations::new(TTFT_MS, ITL_MS, OSL)
            .model("gpt-4")
            .tol_ms(12.0, 3.0),
    );
}

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
                    // Authored payloads must request streaming explicitly.
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
    assert_raw_records_timing_self_consistent_model(
        &records,
        TTFT_MS,
        ITL_MS,
        8.0,
        2.0,
        Some("gpt-4"),
    );
}

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
    // Graph fan-out adds first-token scheduling jitter; ITL remains tightly paced.
    assert_raw_records_timing_self_consistent(&records, TTFT_MS, ITL_MS, 12.0, 3.0);
}
