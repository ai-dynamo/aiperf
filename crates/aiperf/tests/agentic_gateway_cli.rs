// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI proof that evaluator-side model calls return through Rust's normal pipeline.

use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, http::header, response::IntoResponse, routing::post};
use serde_json::Value;

const CALLBACK_EVALUATOR: &str = r#"
import json
import sys
import threading
import urllib.parse
import urllib.request

episode = {
    "episode_id": "opaque:aux:one",
    "task": "fixture.auxiliary-user-simulator",
    "source": "fixture/agentic-gateway",
}
events = []
results = {}
threads = []
lock = threading.Lock()
gateway = None

def run_environment_call():
    episode_id = episode["episode_id"]
    encoded = urllib.parse.quote(episode_id, safe="")
    url = f'{gateway["base_url"]}/episodes/{encoded}/environment/v1/chat/completions'
    body = {
        "model": "fixture-user-simulator",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "prior-call",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "prior-call", "content": "ready"},
            {"role": "user", "content": "Respond through the answer tool"},
        ],
        "tools": [{
            "type": "function",
            "function": {"name": "answer", "parameters": {"type": "object"}},
        }],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 41,
        "temperature": 0.1,
        "top_p": 0.8,
        "reasoning_effort": "low",
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f'Bearer {gateway["api_key"]}',
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
        assert payload["id"] == "provider-response"
        assert payload["model"] == "fixture-user-simulator"
        assert payload["choices"][0]["finish_reason"] == "tool_calls"
        tool_call = payload["choices"][0]["message"]["tool_calls"][0]
        assert tool_call["id"] == "answer-call"
        assert tool_call["function"]["name"] == "answer"
        assert tool_call["function"]["arguments"] == '{"value":1}'
        assert payload["usage"]["prompt_tokens"] == 7
        assert payload["usage"]["completion_tokens"] == 3

        verifier_url = f'{gateway["base_url"]}/episodes/{encoded}/verifier/v1/chat/completions'
        verifier_body = {
            "model": "fixture-judge",
            "messages": [{"role": "user", "content": "Judge the canonical trajectory"}],
            "tools": body["tools"],
            "tool_choice": "auto",
            "max_tokens": 23,
            "stream": False,
        }
        verifier_request = urllib.request.Request(
            verifier_url,
            data=json.dumps(verifier_body).encode(),
            headers={
                "Authorization": f'Bearer {gateway["api_key"]}',
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(verifier_request, timeout=30) as response:
            verifier_payload = json.load(response)
        assert verifier_payload["model"] == "fixture-judge"
        assert verifier_payload["choices"][0]["message"]["tool_calls"][0]["id"] == "answer-call"
        terminal = {
            "episode_id": episode_id,
            "task": episode["task"],
            "outcome": "completed",
            "rewards": {"reward": 1.0},
            "primary_reward": "reward",
            "duration_seconds": 0.25,
            "model_calls": 0,
            "artifact_path": "artifacts/auxiliary",
        }
    except Exception as error:
        terminal = {
            "episode_id": episode_id,
            "task": episode["task"],
            "outcome": "infrastructure_error",
            "rewards": {},
            "primary_reward": None,
            "duration_seconds": 0.25,
            "model_calls": 0,
            "error_kind": type(error).__name__,
            "error_message": str(error),
        }
    with lock:
        results[episode_id] = terminal
        events.append({"kind": "episode_completed", "result": terminal})

for line in sys.stdin:
    request = json.loads(line)
    operation = request["op"]
    if operation == "hello":
        result = {
            "protocol": 1,
            "worker_version": "agentic-gateway-fixture",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"aiperf": "fixture", "harbor": "0.18.0"},
            "worker_source_sha256": "a" * 64,
            "dependency_lock_sha256": "b" * 64,
            "container_digest": None,
            "capabilities": [
                "load", "next_problems", "grade_batch", "agentic_harbor",
                "agentic_inference_gateway", "shutdown"
            ],
        }
    elif operation == "load_agentic":
        gateway = request["config"]["inference_gateway"]
        assert gateway["base_url"].startswith("http://127.0.0.1:")
        assert gateway["api_key"].startswith("aiperf-")
        result = {
            "harness": "fixture-callback",
            "harness_version": "1.0.0",
            "harness_source_sha256": "c" * 64,
            "dataset": {
                "provider": "fixture",
                "benchmark": "fixture/agentic-gateway",
                "repository": "fixture/agentic-gateway",
                "revision": "d" * 64,
                "evaluation_splits": ["tasks"],
            },
            "agent": "fixture-agent",
            "agent_version": "1.0.0",
            "environment": "docker",
            "verifier": "fixture verifier",
            "episode_count": 1,
            "primary_reward": "reward",
        }
    elif operation == "next_episodes":
        items = [episode] if request["offset"] == 0 else []
        result = {"items": items, "next_offset": 1, "done": True}
    elif operation == "start_episodes":
        assert request["episode_ids"] == [episode["episode_id"]]
        thread = threading.Thread(target=run_environment_call, daemon=True)
        threads.append(thread)
        thread.start()
        result = {"started": request["episode_ids"]}
    elif operation == "poll_agentic":
        with lock:
            page = events[: request["limit"]]
            del events[: len(page)]
        result = {"events": page}
    elif operation == "submit_model_results":
        raise AssertionError("the fixture emits no primary agent calls")
    elif operation == "cancel_episodes":
        result = {"cancelled": request["episode_ids"]}
    elif operation == "finish_agentic":
        result = {"items": [results[episode["episode_id"]]]}
    elif operation == "shutdown":
        for thread in threads:
            thread.join(timeout=5)
        result = {"shutdown": True}
    else:
        raise RuntimeError(operation)
    print(json.dumps({"id": request["id"], "ok": True, "result": result}), flush=True)
    if operation == "shutdown":
        break
"#;

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<Value>>>);

async fn tool_call_completion(
    State(captured): State<Captured>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    captured.0.lock().unwrap().push(body);
    let stream = concat!(
        "data: {\"id\":\"provider-response\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"index\":0,\"id\":\"answer-call\",\"type\":\"function\",\"function\":{\"name\":\"answer\",\"arguments\":\"{\\\"value\\\":\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"provider-response\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        "data: {\"id\":\"provider-response\",\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":3,\"prompt_tokens_details\":{\"cached_tokens\":2}}}\n\n",
        "data: [DONE]\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

#[tokio::test]
async fn evaluator_http_callback_uses_rust_scheduler_transport_and_report() {
    let captured = Captured::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(tool_call_completion))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let fixture_root = std::env::temp_dir().join(format!(
        "aiperf_agentic_gateway_{}_{}",
        std::process::id(),
        address.port()
    ));
    let module_dir = fixture_root.join("aiperf/accuracy");
    std::fs::create_dir_all(&module_dir).unwrap();
    std::fs::write(fixture_root.join("aiperf/__init__.py"), "").unwrap();
    std::fs::write(module_dir.join("__init__.py"), "").unwrap();
    std::fs::write(module_dir.join("worker.py"), CALLBACK_EVALUATOR).unwrap();

    let output_path = fixture_root.join("report.json");
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("primary-model")
        .arg("--agentic-benchmark")
        .arg("fixture/agentic-gateway@locked")
        .arg("--agentic-inference-gateway-host")
        .arg("127.0.0.1")
        .arg("--agentic-max-episodes")
        .arg("1")
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
    server.abort();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report_bytes = std::fs::read(&output_path).unwrap();
    let report: Value = serde_json::from_slice(&report_bytes).unwrap();
    assert_eq!(report["agentic"]["summary"]["completed_count"], 1);
    assert_eq!(report["agentic"]["summary"]["model_calls"], 2);
    assert_eq!(report["agentic"]["summary"]["primary_model_calls"], 0);
    assert_eq!(report["agentic"]["summary"]["auxiliary_model_calls"], 2);
    assert_eq!(report["agentic"]["summary"]["environment_model_calls"], 1);
    assert_eq!(report["agentic"]["summary"]["verifier_model_calls"], 1);
    assert_eq!(report["agentic"]["summary"]["prompt_tokens"], 14);
    assert_eq!(report["agentic"]["summary"]["completion_tokens"], 6);
    assert_eq!(report["agentic"]["summary"]["cached_tokens"], 4);
    assert_eq!(report["agentic"]["records"][0]["model_calls"], 2);
    assert_eq!(
        report["agentic"]["records"][0]["environment_model_calls"],
        1
    );
    assert_eq!(report["agentic"]["records"][0]["verifier_model_calls"], 1);
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"],
        2.0
    );
    assert!(
        report["agentic"]["config"]["inference_gateway_base_url"]
            .as_str()
            .unwrap()
            .starts_with("http://127.0.0.1:")
    );
    assert!(!String::from_utf8_lossy(&report_bytes).contains("api_key"));

    let requests = captured.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let body = &requests[0];
    assert_eq!(body["model"], "fixture-user-simulator");
    assert_eq!(body["messages"][0]["tool_calls"][0]["id"], "prior-call");
    assert_eq!(body["messages"][1]["tool_call_id"], "prior-call");
    assert_eq!(body["tools"][0]["function"]["name"], "answer");
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["response_format"]["type"], "json_object");
    assert_eq!(body["max_tokens"], 41);
    assert_eq!(body["temperature"], 0.1);
    assert_eq!(body["top_p"], 0.8);
    assert_eq!(body["reasoning_effort"], "low");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    let verifier = &requests[1];
    assert_eq!(verifier["model"], "fixture-judge");
    assert_eq!(
        verifier["messages"][0]["content"],
        "Judge the canonical trajectory"
    );
    assert_eq!(verifier["max_tokens"], 23);
    drop(requests);

    std::fs::remove_dir_all(fixture_root).unwrap();
}
