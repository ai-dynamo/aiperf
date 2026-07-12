// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI acceptance: stateful evaluator -> normal Rust HTTP/SSE -> canonical verifier report.

use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, http::header, response::IntoResponse, routing::post};
use serde_json::Value;

const FAKE_AGENTIC_EVALUATOR: &str = r#"
import json
import sys

episodes = [
    {"episode_id": "opaque-episode-alpha", "task": "swebench.alpha", "source": "harbor/swebench"},
    {"episode_id": "opaque-episode-beta", "task": "swebench.beta", "source": "harbor/swebench"},
]
events = []
results = {}

def model_call(episode_id, turn_index, messages):
    return {
        "kind": "model_call",
        "call": {
            "episode_id": episode_id,
            "call_id": f"{episode_id}:call:{turn_index:08d}",
            "turn_index": turn_index,
            "prompt": messages[-1]["content"],
            "messages": messages,
            "generation": {
                "max_tokens": 73,
                "temperature": 0.2,
                "top_p": 0.9,
                "stop": ["</tool>"],
            },
            "tools": [{
                "type": "function",
                "function": {"name": "terminal", "parameters": {"type": "object"}},
            }],
            "tool_choice": "auto",
            "response_format": {"type": "json_object"},
        },
    }

def complete_alpha():
    return {
        "episode_id": "opaque-episode-alpha",
        "task": "swebench.alpha",
        "outcome": "completed",
        "rewards": {"reward": 1.0, "tests_passed": 0.75},
        "primary_reward": "reward",
        "duration_seconds": 2.5,
        "model_calls": 2,
        "prompt_tokens": 22,
        "completion_tokens": 8,
        "cached_tokens": 3,
        "artifact_path": "artifacts/alpha",
    }

def fail_beta():
    return {
        "episode_id": "opaque-episode-beta",
        "task": "swebench.beta",
        "outcome": "infrastructure_error",
        "rewards": {},
        "primary_reward": None,
        "duration_seconds": 0.5,
        "model_calls": 1,
        "error_kind": "SandboxStartup",
        "error_message": "fixture sandbox unavailable",
        "artifact_path": "artifacts/beta",
    }

for line in sys.stdin:
    request = json.loads(line)
    operation = request["op"]
    if operation == "hello":
        result = {
            "protocol": 1,
            "worker_version": "agentic-cli-fixture",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"aiperf": "fixture", "harbor": "0.18.0"},
            "worker_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "container_digest": "sha256:" + "c" * 64,
            "capabilities": [
                "load", "next_problems", "grade_batch", "shutdown", "agentic_harbor"
            ],
        }
    elif operation == "load_agentic":
        assert request["dataset"] == "harbor/swebench@fixture-lock"
        assert request["model"] == "fixture-model"
        assert request["config"]["task_concurrency"] == 2
        assert request["config"]["environment"] == "docker"
        result = {
            "harness": "harbor",
            "harness_version": "0.18.0",
            "harness_source_sha256": "d" * 64,
            "dataset": {
                "provider": "harbor package registry",
                "benchmark": "harbor/swebench",
                "repository": "harbor/swebench",
                "revision": "e" * 64,
                "evaluation_splits": ["tasks"],
            },
            "agent": "aiperf-terminus-2",
            "agent_version": "1.0.0+terminus-2.0.0",
            "environment": "docker",
            "verifier": "harbor packaged task verifier",
            "episode_count": len(episodes),
            "primary_reward": "reward",
        }
    elif operation == "next_episodes":
        offset = request["offset"]
        page = episodes[offset : offset + request["limit"]]
        result = {
            "items": page,
            "next_offset": offset + len(page),
            "done": offset + len(page) >= len(episodes),
        }
    elif operation == "start_episodes":
        for episode_id in request["episode_ids"]:
            events.append(model_call(
                episode_id,
                0,
                [{"role": "user", "content": f"repair {episode_id}"}],
            ))
        result = {"started": request["episode_ids"]}
    elif operation == "poll_agentic":
        page = events[: request["limit"]]
        del events[: len(page)]
        result = {"events": page}
    elif operation == "submit_model_results":
        accepted = []
        for item in request["items"]:
            assert item["status"] == "completed"
            assert item["response_id"].startswith("response-")
            assert item["finish_reason"] == "stop"
            assert item["prompt_tokens"] == 11
            assert item["completion_tokens"] == 4
            assert item["cached_tokens"] == 3
            accepted.append(item["call_id"])
            if item["episode_id"] == "opaque-episode-alpha" and item["call_id"].endswith("00000000"):
                events.append(model_call(
                    item["episode_id"],
                    1,
                    [
                        {"role": "user", "content": "repair opaque-episode-alpha"},
                        {"role": "assistant", "content": item["response"]},
                        {"role": "user", "content": "run the verifier"},
                    ],
                ))
            elif item["episode_id"] == "opaque-episode-alpha":
                terminal = complete_alpha()
                results[item["episode_id"]] = terminal
                events.append({"kind": "episode_completed", "result": terminal})
            else:
                terminal = fail_beta()
                results[item["episode_id"]] = terminal
                events.append({"kind": "episode_completed", "result": terminal})
        result = {"accepted": accepted}
    elif operation == "cancel_episodes":
        result = {"cancelled": request["episode_ids"]}
    elif operation == "finish_agentic":
        result = {"items": [results[episode["episode_id"]] for episode in episodes]}
    elif operation == "shutdown":
        result = {"shutdown": True}
    else:
        raise RuntimeError(operation)
    print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
    if operation == "shutdown":
        break
"#;

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<Value>>>);

async fn chat_handler(
    State(captured): State<Captured>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let request_index = captured.0.lock().unwrap().len();
    let answer = format!("{{\"command\":\"fixture-{request_index}\"}}");
    captured.0.lock().unwrap().push(body);
    let response_id = format!("response-{request_index}");
    let stream = format!(
        "data: {{\"id\":{response_id:?},\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"reasoning_content\":\"think\",\"content\":{answer:?}}},\"finish_reason\":\"stop\"}}]}}\n\n\
         data: {{\"id\":{response_id:?},\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{{\"prompt_tokens\":11,\"completion_tokens\":4,\"prompt_tokens_details\":{{\"cached_tokens\":3}}}}}}\n\n\
         data: [DONE]\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

#[tokio::test]
async fn cli_runs_stateful_harbor_calls_through_normal_rust_transport() {
    let captured = Captured::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let fixture_root = std::env::temp_dir().join(format!(
        "aiperf_agentic_worker_{}_{}",
        std::process::id(),
        address.port()
    ));
    let module_dir = fixture_root.join("aiperf/accuracy");
    std::fs::create_dir_all(&module_dir).unwrap();
    std::fs::write(fixture_root.join("aiperf/__init__.py"), "").unwrap();
    std::fs::write(module_dir.join("__init__.py"), "").unwrap();
    std::fs::write(module_dir.join("worker.py"), FAKE_AGENTIC_EVALUATOR).unwrap();

    let output_path = fixture_root.join("agentic-report.json");
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--agentic-benchmark")
        .arg("harbor/swebench@fixture-lock")
        .arg("--agentic-task-concurrency")
        .arg("2")
        .arg("--agentic-max-episodes")
        .arg("2")
        .arg("--agentic-primary-reward")
        .arg("reward")
        .arg("--concurrency")
        .arg("1")
        .arg("--json")
        .arg(&output_path)
        .env("PYTHONPATH", &fixture_root)
        .env(
            "AIPERF_ACCURACY_PYTHON",
            std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()),
        )
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: Value = serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "agentic_accuracy");
    assert_eq!(report["evaluator"]["worker_version"], "agentic-cli-fixture");
    assert_eq!(report["evaluator"]["packages"]["harbor"], "0.18.0");
    assert_eq!(
        report["evaluator"]["container_digest"],
        format!("sha256:{}", "c".repeat(64))
    );
    assert_eq!(report["evaluator"]["dataset"]["revision"], "e".repeat(64));
    assert_eq!(report["agentic"]["evaluator"]["harness"], "harbor");
    assert_eq!(report["agentic"]["evaluator"]["harness_version"], "0.18.0");
    assert_eq!(
        report["agentic"]["evaluator"]["harness_source_sha256"],
        "d".repeat(64)
    );
    assert_eq!(report["agentic"]["config"]["model_concurrency"], 1);
    assert_eq!(report["agentic"]["config"]["task_concurrency"], 2);
    assert_eq!(report["agentic"]["summary"]["episode_count"], 2);
    assert_eq!(report["agentic"]["summary"]["completed_count"], 1);
    assert_eq!(
        report["agentic"]["summary"]["infrastructure_error_count"],
        1
    );
    assert_eq!(report["agentic"]["summary"]["cancelled_count"], 0);
    assert_eq!(report["agentic"]["summary"]["primary_reward"], "reward");
    assert_eq!(report["agentic"]["summary"]["primary_score"], 1.0);
    assert_eq!(report["agentic"]["summary"]["rewards"]["reward"]["n"], 1);
    assert_eq!(report["agentic"]["records"].as_array().unwrap().len(), 2);
    assert_eq!(
        report["agentic"]["records"][1]["outcome"],
        "infrastructure_error"
    );
    assert_eq!(
        report["agentic"]["records"][1]["rewards"],
        serde_json::json!({})
    );
    assert_eq!(
        report["errors"][0]["type"],
        "AgenticInfrastructure:SandboxStartup"
    );
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        3.0
    );

    let requests = captured.0.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(requests.iter().all(|body| body["stream"] == true));
    assert!(
        requests
            .iter()
            .all(|body| body["stream_options"]["include_usage"] == true)
    );
    assert!(requests.iter().all(|body| body["temperature"] == 0.2));
    assert!(requests.iter().all(|body| body["top_p"] == 0.9));
    assert!(requests.iter().all(|body| body["max_tokens"] == 73));
    assert!(
        requests
            .iter()
            .all(|body| body["stop"] == serde_json::json!(["</tool>"]))
    );
    assert!(
        requests
            .iter()
            .all(|body| body["tools"][0]["function"]["name"] == "terminal")
    );
    assert!(
        requests
            .iter()
            .all(|body| body["response_format"]["type"] == "json_object")
    );
    assert!(
        requests
            .iter()
            .any(|body| body["messages"].as_array().unwrap().len() == 3)
    );
    drop(requests);
    std::fs::remove_dir_all(fixture_root).unwrap();
}
