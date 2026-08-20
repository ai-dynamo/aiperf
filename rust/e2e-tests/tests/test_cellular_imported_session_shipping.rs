// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Imported Codex session sets ship over the cellular HTTP artifact plane.
//!
//! The multi-cell run uses the real `aiperf` controller and cells, while the
//! in-process mock server receives every reconstructed imported request. The raw
//! record set must match a one-cell run over the same exact source set.

mod common;
use std::fs;
use std::path::{Path, PathBuf};

use common::*;
use serde_json::{Value, json};

const SESSIONS: u32 = 3;
const CELLS: u32 = 3;
const CONCURRENCY: u32 = 3;

fn write_codex_session_set(root: &Path) -> PathBuf {
    let sessions = root.join("sessions");
    fs::create_dir_all(&sessions).expect("create imported-session directory");
    for session in 0..SESSIONS {
        let session_id = format!("cellular-import-{session}");
        let body = [
            json!({"type": "session_meta", "payload": {"id": session_id}}),
            json!({"type": "response_item", "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": format!("imported cellular prompt {session}")}],
            }}),
            json!({"type": "response_item", "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": format!("recorded reply {session}")}],
            }}),
        ]
        .into_iter()
        .map(|record| record.to_string())
        .collect::<Vec<_>>()
        .join("\n")
            + "\n";
        fs::write(sessions.join(format!("session-{session}.jsonl")), body)
            .expect("write imported Codex session");
    }
    sessions
}

fn config(url: &str, source: &Path, replay_root: &Path, cells: u32) -> String {
    format!(
        "schemaVersion: \"2.0\"\n\
         benchmark:\n\
        \x20 model: {DEFAULT_MODEL}\n\
        \x20 endpoint:\n\
        \x20   url: {url}\n\
        \x20   type: chat\n\
        \x20   streaming: true\n\
        \x20 dataset:\n\
        \x20   type: file\n\
        \x20   path: {}\n\
        \x20   format: agent_recording\n\
        \x20   graph:\n\
        \x20     source_format: codex\n\
        \x20     replay_root: {}\n\
        \x20 profiling:\n\
        \x20   type: concurrency\n\
        \x20   sessions: {SESSIONS}\n\
        \x20   concurrency: {CONCURRENCY}\n\
        \x20 artifacts:\n\
        \x20   records:\n\
        \x20     - jsonl\n\
        \x20   raw: true\n\
         runtime:\n\
        \x20 cells: {cells}\n",
        source.display(),
        replay_root.display(),
    )
}

fn run_imported_sessions(
    harness: &AIPerfHarness,
    source: &Path,
    replay_root: &Path,
    cells: u32,
    force_http: bool,
) -> RunResult {
    let temporary = tempfile::tempdir().expect("config temporary directory");
    let path = temporary.path().join("imported-sessions.yaml");
    fs::write(&path, config(&harness.mock.url, source, replay_root, cells))
        .expect("write imported-session config");
    let mut env = vec![(
        "AIPERF_LOG",
        "warn,aiperf=info,aiperf_cellular_artifact=info",
    )];
    if force_http {
        env.push(("AIPERF_CELL_ARTIFACT_HTTP_FORCE", "1"));
    }
    harness.run_env(&format!("--config {} --ui simple", path.display()), &env)
}

fn raw_projection(record: &Value) -> String {
    json!({
        "payload": record["payload"],
        "error": record["error"],
        "status": record["status"],
    })
    .to_string()
}

fn raw_response_text(record: &Value) -> String {
    record["responses"]
        .as_array()
        .expect("raw record responses must be an array")
        .iter()
        .flat_map(|response| {
            response["packets"]
                .as_array()
                .expect("raw response packets must be an array")
        })
        .filter(|packet| packet["name"] == "data")
        .filter_map(|packet| packet["value"].as_str())
        .filter_map(|chunk| serde_json::from_str::<Value>(chunk).ok())
        .filter_map(|chunk| {
            chunk
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect()
}

fn sorted_raw_records(result: &RunResult) -> Vec<String> {
    let mut records: Vec<_> = result
        .artifacts
        .raw_records()
        .iter()
        .map(raw_projection)
        .collect();
    records.sort();
    records
}

fn dataset_serve_observables(result: &RunResult) -> Vec<String> {
    let log = result
        .artifacts
        .find_file("**/aiperf.log")
        .expect("logs/aiperf.log should exist");
    fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .filter(|line| line.contains("served dataset source over HTTP"))
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn test_cellular_imported_session_exact_set_shipping_matches_single_cell_raw_records() {
    if cfg!(target_os = "macos") {
        return;
    }

    let temporary = tempfile::tempdir().expect("session fixture root");
    let source = write_codex_session_set(temporary.path());

    let baseline_harness = AIPerfHarness::new().await;
    let baseline = run_imported_sessions(&baseline_harness, &source, temporary.path(), 1, false);
    assert!(
        baseline.success(),
        "single-cell imported-session run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        baseline.exit_code,
        baseline.stdout,
        baseline.stderr
    );

    let cellular_harness = AIPerfHarness::new().await;
    let cellular = run_imported_sessions(&cellular_harness, &source, temporary.path(), CELLS, true);
    assert!(
        cellular.success(),
        "{CELLS}-cell imported-session run failed (exit {}):\nstdout:\n{}\nstderr:\n{}",
        cellular.exit_code,
        cellular.stdout,
        cellular.stderr
    );
    assert!(
        cellular
            .artifacts
            .find_file("**/cellular-heartbeat.json")
            .is_some(),
        "multi-cell imported-session run must go through the cellular controller"
    );

    let observables = dataset_serve_observables(&cellular);
    assert_eq!(
        observables.len(),
        (SESSIONS * CELLS) as usize,
        "each cell must fetch every exact-set source once: {observables:?}"
    );
    assert!(
        observables.iter().all(|line| {
            line.contains("content_encoding=\"zstd\"") || line.contains("content_encoding=zstd")
        }),
        "imported-session sources must be served over zstd: {observables:?}"
    );
    assert!(
        dataset_serve_observables(&baseline).is_empty(),
        "single-process run must not expose an artifact server"
    );

    let baseline_raw = sorted_raw_records(&baseline);
    let cellular_raw = sorted_raw_records(&cellular);
    assert_eq!(baseline_raw.len(), SESSIONS as usize);
    for record in cellular.artifacts.raw_records() {
        assert_eq!(
            record["status"], 200,
            "raw imported-session response: {record}"
        );
        assert!(
            !raw_response_text(&record).is_empty(),
            "raw imported-session response has no generated content: {record}"
        );
    }
    assert_eq!(
        baseline_raw, cellular_raw,
        "raw imported request/response set diverged"
    );
}
