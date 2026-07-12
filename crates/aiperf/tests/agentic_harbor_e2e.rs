// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in acceptance proof against a real pinned Harbor package and sandbox.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::{Json, Router, extract::State, http::header, response::IntoResponse, routing::post};
use serde_json::Value;

const AGENTIC_LOCK_SHA256: &str =
    "5ab314ec28af774ed9edf4a6baf5216f8831ecf06eb9bf3b62418bef275b57ef";
const TAU3_START_CONVERSATION: &str = r#"python - <<'PY'
import json
import urllib.request

url = "http://tau3-runtime:8000/mcp"
headers = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}

def post(payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        session = response.headers.get("Mcp-Session-Id")
        text = response.read().decode()
        content_type = response.headers.get("Content-Type", "")
    if session:
        headers["Mcp-Session-Id"] = session
    if not text:
        return None
    if "text/event-stream" in content_type:
        for line in text.splitlines():
            if line.startswith("data:"):
                return json.loads(line.removeprefix("data:").strip())
        raise RuntimeError("MCP response contained no data event")
    return json.loads(text)

post({
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "aiperf-canary", "version": "1.0"},
    },
})
post({"jsonrpc": "2.0", "method": "notifications/initialized"})
print(post({
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "start_conversation", "arguments": {}},
}))
PY"#;

#[derive(Clone, Default)]
struct Captured(Arc<Mutex<Vec<Value>>>);

async fn terminus_completion(
    State(captured): State<Captured>,
    Json(body): Json<Value>,
) -> impl IntoResponse {
    let request_index = captured.0.lock().unwrap().len();
    let auxiliary_call = body["model"] != "harbor-e2e-model";
    captured.0.lock().unwrap().push(body);
    let response_id = format!("harbor-e2e-{request_index}");
    let authored_command = std::env::var("AIPERF_AGENTIC_FIRST_COMMAND")
        .ok()
        .or_else(|| {
            std::env::var_os("AIPERF_AGENTIC_EXERCISE_TAU3")
                .is_some()
                .then(|| TAU3_START_CONVERSATION.to_string())
        });
    let answer = if auxiliary_call {
        "My mobile data is not working and I am currently abroad in France.".to_string()
    } else if request_index == 0
        && let Some(command) = authored_command
    {
        serde_json::json!({
            "analysis": "Exercise the packaged task's canonical MCP environment.",
            "plan": "Start the simulated-user conversation through the advertised MCP server.",
            "commands": [{"keystrokes": format!("{command}\n"), "duration": 30}],
            "task_complete": false,
        })
        .to_string()
    } else {
        serde_json::json!({
            "analysis": "The acceptance model intentionally leaves the remaining task environment unchanged.",
            "plan": "Submit the current environment to the packaged verifier.",
            "commands": [],
            "task_complete": true,
        })
        .to_string()
    };
    let stream = format!(
        "data: {{\"id\":{response_id:?},\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"content\":{answer:?}}},\"finish_reason\":\"stop\"}}]}}\n\n\
         data: {{\"id\":{response_id:?},\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{{\"prompt_tokens\":37,\"completion_tokens\":17,\"prompt_tokens_details\":{{\"cached_tokens\":5}}}}}}\n\n\
         data: [DONE]\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

#[tokio::test]
#[ignore = "requires pinned Harbor worker, registry access, and Docker"]
async fn real_harbor_package_runs_rust_inference_and_packaged_verifier() {
    let python = std::env::var_os("AIPERF_AGENTIC_PYTHON")
        .expect("set AIPERF_AGENTIC_PYTHON to the hash-pinned Harbor worker Python");
    let dataset = std::env::var("AIPERF_AGENTIC_DATASET")
        .unwrap_or_else(|_| "harbor/hello-world".to_string());
    let expected_task = std::env::var("AIPERF_AGENTIC_EXPECTED_TASK").ok();
    let expected_revision = std::env::var("AIPERF_AGENTIC_EXPECTED_REVISION").ok();
    let expected_benchmark = std::env::var("AIPERF_AGENTIC_EXPECTED_BENCHMARK").ok();
    let expected_provider = std::env::var("AIPERF_AGENTIC_EXPECTED_PROVIDER").ok();
    let gateway_host = std::env::var("AIPERF_AGENTIC_GATEWAY_HOST").ok();
    let expected_auxiliary_calls = std::env::var("AIPERF_AGENTIC_EXPECTED_AUXILIARY_CALLS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok());
    let timeout_seconds = std::env::var("AIPERF_AGENTIC_E2E_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(900);
    let max_turns = std::env::var("AIPERF_AGENTIC_MAX_TURNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);

    let captured = Captured::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(terminus_completion))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let root = std::env::temp_dir().join(format!(
        "aiperf_real_harbor_{}_{}",
        std::process::id(),
        address.port()
    ));
    let trials = root.join("trials");
    let report_path = root.join("report.json");
    std::fs::create_dir_all(&trials).unwrap();

    let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"));
    command
        .arg(format!("http://{address}"))
        .arg("harbor-e2e-model")
        .arg("--agentic-benchmark")
        .arg(&dataset)
        .arg("--agentic-max-episodes")
        .arg("1")
        .arg("--agentic-task-concurrency")
        .arg("1")
        .arg("--agentic-output-dir")
        .arg(&trials)
        .arg("--agentic-max-turns")
        .arg(max_turns.to_string())
        .arg("--agentic-max-tokens")
        .arg("512")
        .arg("--agentic-context-window")
        .arg("32768")
        .arg("--agentic-no-summarize")
        .arg("--concurrency")
        .arg("1")
        .arg("--json")
        .arg(&report_path)
        .env("AIPERF_ACCURACY_PYTHON", python)
        .kill_on_drop(true);
    if let Some(task) = &expected_task {
        command.arg("--agentic-tasks").arg(task);
    }
    if let Some(host) = &gateway_host {
        command.arg("--agentic-inference-gateway-host").arg(host);
    }
    let output = tokio::time::timeout(Duration::from_secs(timeout_seconds), command.output())
        .await
        .expect("real Harbor acceptance run timed out")
        .unwrap();
    server.abort();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: Value = serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "agentic_accuracy");
    assert_eq!(report["agentic"]["config"]["dataset"], dataset);
    assert_eq!(report["evaluator"]["packages"]["harbor"], "0.18.0");
    assert_eq!(
        report["evaluator"]["dependency_lock_sha256"],
        AGENTIC_LOCK_SHA256
    );
    assert_eq!(report["agentic"]["evaluator"]["harness"], "harbor");
    assert_eq!(report["agentic"]["evaluator"]["agent"], "aiperf-terminus-2");
    assert_eq!(report["agentic"]["evaluator"]["environment"], "docker");
    assert_eq!(
        report["agentic"]["evaluator"]["verifier"],
        "harbor packaged task verifier"
    );
    if let Some(expected) = &expected_benchmark {
        assert_eq!(
            report["evaluator"]["dataset"]["benchmark"],
            expected.as_str()
        );
    }
    if let Some(expected) = &expected_revision {
        assert_eq!(
            report["evaluator"]["dataset"]["revision"],
            expected.as_str()
        );
    }
    if let Some(expected) = &expected_provider {
        assert_eq!(
            report["evaluator"]["dataset"]["provider"],
            expected.as_str()
        );
    }
    assert_eq!(report["agentic"]["summary"]["episode_count"], 1);
    assert_eq!(report["agentic"]["summary"]["completed_count"], 1);
    assert_eq!(
        report["agentic"]["summary"]["infrastructure_error_count"],
        0
    );
    if let Some(expected) = expected_auxiliary_calls {
        assert!(
            report["agentic"]["summary"]["auxiliary_model_calls"]
                .as_u64()
                .unwrap()
                >= expected
        );
    }
    let records = report["agentic"]["records"].as_array().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0]["outcome"], "completed");
    if let Some(expected) = &expected_task {
        assert_eq!(records[0]["task"], expected.as_str());
    }
    assert!(!records[0]["rewards"].as_object().unwrap().is_empty());
    assert!(records[0]["model_calls"].as_u64().unwrap() >= 1);
    let artifact_path = records[0]["artifact_path"].as_str().unwrap();
    assert!(
        std::path::Path::new(artifact_path)
            .join("result.json")
            .is_file()
    );
    assert!(
        std::path::Path::new(artifact_path)
            .join("verifier")
            .is_dir()
    );

    let requests = captured.0.lock().unwrap();
    assert_eq!(
        records[0]["model_calls"].as_u64().unwrap(),
        requests.len() as u64
    );
    assert!(requests.iter().all(|body| body["stream"] == true));
    assert!(
        requests
            .iter()
            .all(|body| body["stream_options"]["include_usage"] == true)
    );
    assert!(
        requests
            .iter()
            .filter(|body| body["model"] == "harbor-e2e-model")
            .count() as u64
            >= records[0]["primary_model_calls"].as_u64().unwrap()
    );
    assert!(
        requests
            .iter()
            .all(|body| !body["messages"].as_array().unwrap().is_empty())
    );
    let model_calls = requests.len();
    drop(requests);

    eprintln!(
        "AIPERF_AGENTIC_E2E_PROOF={}",
        serde_json::json!({
            "dataset": dataset,
            "benchmark": report["evaluator"]["dataset"]["benchmark"],
            "provider": report["evaluator"]["dataset"]["provider"],
            "revision": report["evaluator"]["dataset"]["revision"],
            "task": records[0]["task"],
            "outcome": records[0]["outcome"],
            "rewards": records[0]["rewards"],
            "model_calls": model_calls,
            "primary_model_calls": records[0]["primary_model_calls"],
            "auxiliary_model_calls": records[0]["auxiliary_model_calls"],
            "environment_model_calls": records[0]["environment_model_calls"],
            "verifier_model_calls": records[0]["verifier_model_calls"],
            "inference_gateway_base_url": report["agentic"]["config"]["inference_gateway_base_url"],
            "artifact_path": artifact_path,
            "harbor": report["evaluator"]["packages"]["harbor"],
            "dependency_lock_sha256": report["evaluator"]["dependency_lock_sha256"],
        })
    );

    std::fs::remove_dir_all(root).unwrap();
}
