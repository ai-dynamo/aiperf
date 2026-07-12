// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in proof that canonical MCPMark tools and verification use Rust inference.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{Json, Router, extract::State, http::header, response::IntoResponse, routing::post};
use serde_json::{Value, json};

const MCPMARK_COMMIT: &str = "cd45b7f57923b9b3985467f5139927575f83141c";
const MCPMARK_LOCK_SHA256: &str =
    "85aed9ad589093de161c8ed00c2dbf64ffea1d06685a96a254c72fa4cf189a59";
const MCPMARK_SOURCE_SHA256: &str =
    "55bc1d0e43043101d4eed5b76d97c2efb14c3415e9a4c7e7b74cdc8f81fb21f2";

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<Value>>>);

async fn mcpmark_completion(
    State(captured): State<Captured>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let request_index = {
        let mut requests = captured.0.lock().unwrap();
        let request_index = requests.len();
        requests.push(body.clone());
        request_index
    };
    let (tool_calls, content, finish_reason) = match request_index {
        0 => (
            vec![tool_call(
                0,
                "allowed",
                "list_allowed_directories",
                json!({}),
            )],
            None,
            "tool_calls",
        ),
        1 => {
            let root = allowed_directory(&body)
                .expect("filesystem MCP result omitted its canonical allowed directory");
            (classification_calls(&root), None, "tool_calls")
        }
        _ => (Vec::new(), Some("Task completed"), "stop"),
    };
    let response_id = format!("mcpmark-e2e-{request_index}");
    let delta = if tool_calls.is_empty() {
        json!({"role": "assistant", "content": content})
    } else {
        json!({"role": "assistant", "content": content, "tool_calls": tool_calls})
    };
    let chunk = json!({
        "id": response_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    });
    let usage = json!({
        "id": response_id,
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": 211,
            "completion_tokens": 37,
            "prompt_tokens_details": {"cached_tokens": 11},
        },
    });
    let stream = format!("data: {chunk}\n\ndata: {usage}\n\ndata: [DONE]\n\n");
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

fn tool_call(index: usize, id: &str, name: &str, arguments: Value) -> Value {
    json!({
        "index": index,
        "id": format!("mcpmark-{id}"),
        "type": "function",
        "function": {"name": name, "arguments": arguments.to_string()},
    })
}

fn classification_calls(root: &Path) -> Vec<Value> {
    let mut calls = Vec::new();
    for directory in ["small_files", "medium_files", "large_files"] {
        calls.push(tool_call(
            calls.len(),
            &format!("mkdir-{directory}"),
            "create_directory",
            json!({"path": root.join(directory)}),
        ));
    }
    for (directory, files) in [
        (
            "small_files",
            ["random_file_1.txt", "random_file_3.txt"].as_slice(),
        ),
        ("medium_files", ["random_file_2.txt"].as_slice()),
        (
            "large_files",
            ["bear.jpg", "sg.jpg", "road.MOV", "bus.MOV", "bridge.jpg"].as_slice(),
        ),
    ] {
        for filename in files {
            calls.push(tool_call(
                calls.len(),
                &format!("move-{filename}"),
                "move_file",
                json!({
                    "source": root.join(filename),
                    "destination": root.join(directory).join(filename),
                }),
            ));
        }
    }
    calls
}

fn allowed_directory(body: &Value) -> Option<PathBuf> {
    let messages = body.get("messages")?.as_array()?;
    for message in messages.iter().rev() {
        if message.get("role").and_then(Value::as_str) != Some("tool") {
            continue;
        }
        let content = message.get("content")?.as_str()?;
        let decoded = serde_json::from_str::<Value>(content).ok()?;
        if let Some(path) = find_allowed_directory(&decoded) {
            return Some(PathBuf::from(path));
        }
    }
    None
}

fn find_allowed_directory(value: &Value) -> Option<&str> {
    match value {
        Value::String(text) => text
            .strip_prefix("Allowed directories:")
            .and_then(|rest| rest.lines().map(str::trim).find(|line| !line.is_empty())),
        Value::Array(items) => items.iter().find_map(find_allowed_directory),
        Value::Object(fields) => fields.values().find_map(find_allowed_directory),
        Value::Null | Value::Bool(_) | Value::Number(_) => None,
    }
}

#[tokio::test]
#[ignore = "requires the pinned MCPMark worker, canonical environment download, and Node MCP server"]
async fn real_mcpmark_filesystem_episode_uses_rust_transport_and_canonical_verifier() {
    let python = std::env::var_os("AIPERF_MCPMARK_AGENTIC_PYTHON")
        .expect("set AIPERF_MCPMARK_AGENTIC_PYTHON to the hash-pinned worker Python");
    let timeout_seconds = std::env::var("AIPERF_MCPMARK_E2E_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(300);
    let environment_root = std::env::var_os("AIPERF_MCPMARK_FILESYSTEM_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("aiperf-mcpmark-test-environments"));
    std::fs::create_dir_all(&environment_root).unwrap();

    let captured = Captured::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(mcpmark_completion))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let root = std::env::temp_dir().join(format!(
        "aiperf_real_mcpmark_{}_{}",
        std::process::id(),
        address.port()
    ));
    let artifacts = root.join("episodes");
    let report_path = root.join("report.json");
    let stdout_path = root.join("aiperf.stdout.log");
    let stderr_path = root.join("aiperf.stderr.log");
    std::fs::create_dir_all(&artifacts).unwrap();

    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"));
    command
        .arg(format!("http://{address}"))
        .arg("mcpmark-e2e-model")
        .arg("--agentic-benchmark")
        .arg(format!("mcpmark/filesystem/standard@{MCPMARK_COMMIT}"))
        .arg("--agentic-tasks")
        .arg("file_property/size_classification")
        .arg("--agentic-max-episodes")
        .arg("1")
        .arg("--agentic-task-concurrency")
        .arg("1")
        .arg("--agentic-environment")
        .arg("filesystem")
        .arg("--agentic-output-dir")
        .arg(&artifacts)
        .arg("--agentic-context-window")
        .arg("131072")
        .arg("--concurrency")
        .arg("1")
        .arg("--json")
        .arg(&report_path)
        .env("AIPERF_ACCURACY_PYTHON", python)
        .env("FILESYSTEM_TEST_ROOT", &environment_root)
        .stdout(Stdio::from(std::fs::File::create(&stdout_path).unwrap()))
        .stderr(Stdio::from(std::fs::File::create(&stderr_path).unwrap()))
        .kill_on_drop(true);
    let status =
        match tokio::time::timeout(Duration::from_secs(timeout_seconds), command.status()).await {
            Ok(result) => result.unwrap(),
            Err(error) => panic!(
                "real MCPMark acceptance run timed out: {error}\nstdout:\n{}\nstderr:\n{}",
                std::fs::read_to_string(&stdout_path).unwrap_or_default(),
                std::fs::read_to_string(&stderr_path).unwrap_or_default(),
            ),
        };
    server.abort();
    assert!(
        status.success(),
        "stdout:\n{}\nstderr:\n{}",
        std::fs::read_to_string(&stdout_path).unwrap_or_default(),
        std::fs::read_to_string(&stderr_path).unwrap_or_default(),
    );

    let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "agentic_accuracy");
    assert_eq!(report["evaluator"]["packages"]["MCPMark"], "0.0.1");
    assert_eq!(report["evaluator"]["packages"]["litellm"], "1.80.0");
    assert_eq!(
        report["evaluator"]["dependency_lock_sha256"],
        MCPMARK_LOCK_SHA256
    );
    assert_eq!(
        report["agentic"]["evaluator"]["harness"],
        "mcpmark-verified"
    );
    assert_eq!(
        report["agentic"]["evaluator"]["harness_source_sha256"],
        MCPMARK_SOURCE_SHA256
    );
    assert_eq!(report["agentic"]["evaluator"]["environment"], "filesystem");
    assert_eq!(
        report["agentic"]["evaluator"]["canonical_agent_config"]["max_turns"],
        100
    );
    assert_eq!(
        report["agentic"]["evaluator"]["canonical_agent_config"]["max_tokens"],
        32768
    );
    assert_eq!(
        report["agentic"]["evaluator"]["canonical_agent_config"]["mcp_server"]["artifact"],
        "@modelcontextprotocol/server-filesystem@2025.12.18"
    );
    assert_eq!(report["agentic"]["config"]["max_turns"], 100);
    assert_eq!(report["agentic"]["config"]["max_tokens"], 32768);
    assert_eq!(report["agentic"]["config"]["parser"], "openai_tool_calls");
    assert_eq!(report["agentic"]["config"]["enable_summarize"], false);
    assert_eq!(
        report["evaluator"]["dataset"]["benchmark"],
        format!("mcpmark/filesystem@{MCPMARK_COMMIT}")
    );
    assert!(
        report["evaluator"]["dataset"]["revision"]
            .as_str()
            .unwrap()
            .starts_with(&format!("git:{MCPMARK_COMMIT}+selection-sha256:"))
    );
    let dataset_revision = report["evaluator"]["dataset"]["revision"].as_str().unwrap();
    let environment_digest = dataset_revision
        .split("+environment-sha256:")
        .nth(1)
        .expect("MCPMark report omitted concrete filesystem-state provenance");
    assert_eq!(environment_digest.len(), 64);
    assert!(
        environment_digest
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    );
    assert_eq!(report["agentic"]["summary"]["episode_count"], 1);
    assert_eq!(report["agentic"]["summary"]["completed_count"], 1);
    assert_eq!(
        report["agentic"]["summary"]["infrastructure_error_count"],
        0
    );
    assert_eq!(report["agentic"]["summary"]["primary_score"], 1.0);
    let record = &report["agentic"]["records"][0];
    assert_eq!(record["task"], "file_property/size_classification");
    assert_eq!(record["outcome"], "completed");
    assert_eq!(record["rewards"]["pass"], 1.0);
    let artifact_path = Path::new(record["artifact_path"].as_str().unwrap());
    assert!(artifact_path.join("messages.json").is_file());
    assert!(artifact_path.join("meta.json").is_file());
    assert!(artifact_path.join("execution.log").is_file());

    let requests = captured.0.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(record["model_calls"], requests.len());
    assert!(requests.iter().all(|body| body["stream"] == true));
    assert!(
        requests
            .iter()
            .all(|body| body["stream_options"]["include_usage"] == true)
    );
    assert!(
        requests
            .iter()
            .all(|body| body["model"] == "mcpmark-e2e-model")
    );
    assert!(requests.iter().all(|body| body["max_tokens"] == 32768));
    assert!(requests.iter().all(|body| body["temperature"] == 1.0));
    assert!(requests.iter().all(|body| body["top_p"] == 1.0));
    assert!(requests.iter().all(|body| body["enforcer_mode"] == "on"));
    assert!(requests.iter().all(|body| body["think_mode"] == "on"));
    assert!(requests.iter().all(|body| {
        body["tools"]
            .as_array()
            .is_some_and(|tools| !tools.is_empty())
    }));
    drop(requests);

    eprintln!(
        "AIPERF_MCPMARK_E2E_PROOF={}",
        json!({
            "dataset": report["evaluator"]["dataset"],
            "task": record["task"],
            "reward": record["rewards"]["pass"],
            "model_calls": record["model_calls"],
            "mcpmark": report["evaluator"]["packages"]["MCPMark"],
            "litellm": report["evaluator"]["packages"]["litellm"],
            "dependency_lock_sha256": report["evaluator"]["dependency_lock_sha256"],
            "environment_sha256": environment_digest,
        })
    );

    std::fs::remove_dir_all(root).unwrap();
}
