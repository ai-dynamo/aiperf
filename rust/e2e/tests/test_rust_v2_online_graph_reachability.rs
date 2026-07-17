// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
mod common;
use common::*;

use std::io::Write;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn graph_rows() -> Vec<serde_json::Value> {
    vec![
        json!({
            "session_id": "root",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "root-0"}],
                    "forks": [{"child": "fork", "background": true}],
                    "spawns": [{"children": ["spawn"], "join_at": 1}],
                    "max_tokens": 1,
                },
                {
                    "messages": [{"role": "user", "content": "root-1"}],
                    "max_tokens": 1,
                },
            ],
        }),
        json!({
            "session_id": "fork",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "fork-0"}],
                    "max_tokens": 1,
                }
            ],
        }),
        json!({
            "session_id": "spawn",
            "turns": [
                {
                    "messages": [{"role": "user", "content": "spawn-0"}],
                    "max_tokens": 1,
                }
            ],
        }),
    ]
}

#[tokio::test]
async fn test_online_graph_terminal_reports_run_metadata() {
    let h = AIPerfHarness::new().await;
    let request = json!({
        "protocol_version": 2,
        "operation": "execute",
        "run": {
            "benchmark_id": "e2e-online-graph",
            "artifact_dir": h.artifact_path(),
            "random_seed": 7,
            "cfg": {
                "models": {"strategy": "round_robin", "items": [{"name": DEFAULT_MODEL}]},
                "endpoint": {
                    "type": "chat",
                    "urls": [format!("{}/v1/chat/completions", h.mock.url)],
                    "streaming": true,
                    "use_server_token_count": true,
                    "wait_for_model_timeout": 0.0,
                    "wait_for_model_interval": 5.0,
                    "wait_for_model_mode": "inference"
                },
                "datasets": [{
                    "type": "file",
                    "format": "dag_jsonl",
                    "sampling": "sequential",
                    "records": graph_rows()
                }],
                "tokenizer": {
                    "name": "cl100k_base",
                    "revision": "main",
                    "trust_remote_code": false,
                    "apply_chat_template": false
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "sessions": 1,
                    "concurrency": 2
                }],
                "transport": {"type": "http"},
                "runtime": {"workers": 2}
            }
        }
    });
    let debug_binary =
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/debug/aiperf");
    let binary = if debug_binary.exists() {
        debug_binary.display().to_string()
    } else {
        exec_binary()
    };
    let mut child = Command::new(binary)
        .arg("--execute")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn aiperf execute");
    child
        .stdin
        .take()
        .expect("aiperf stdin")
        .write_all(&serde_json::to_vec(&request["run"]).unwrap())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let terminal: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(terminal["event"], "run_terminal");
    assert_eq!(terminal["success"], true);
    assert_eq!(
        terminal["run_metadata"]["transport"], "http",
        "terminal={terminal}"
    );
    assert_eq!(
        terminal["run_metadata"]["workload"], "graph",
        "terminal={terminal}"
    );
}
