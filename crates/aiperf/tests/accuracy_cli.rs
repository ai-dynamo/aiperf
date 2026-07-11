// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI acceptance: stdio evaluator -> normal HTTP/SSE -> batch grade -> report.

use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, http::header, response::IntoResponse, routing::post};
use serde_json::Value;

const FAKE_EVALUATOR: &str = r#"
import json
import sys

problems = [
    {
        "problem_id": "opaque-alpha",
        "task": "mmlu_pro.math",
        "prompt": "scored-alpha",
        "messages": [{"role": "user", "content": "scored-alpha"}],
        "generation": {"max_tokens": 37, "temperature": 0.0, "top_p": 1.0, "stop": ["Question:"]},
    },
    {
        "problem_id": "opaque-beta",
        "task": "mmlu_pro.math",
        "prompt": "scored-beta",
        "messages": [{"role": "user", "content": "scored-beta"}],
        "generation": {"max_tokens": 37, "temperature": 0.0, "top_p": 1.0, "stop": ["Question:"]},
    },
]

for line in sys.stdin:
    request = json.loads(line)
    operation = request["op"]
    if operation == "hello":
        result = {
            "protocol": 1,
            "worker_version": "cli-fixture",
            "python_version": sys.version.split()[0],
            "python_executable": sys.executable,
            "packages": {"aiperf": "fixture", "lighteval": "fixture"},
            "worker_source_sha256": "fixture-source-digest",
            "dependency_lock_sha256": "fixture-lock-digest",
            "container_digest": "sha256:fixture-image",
            "capabilities": ["load", "next_problems", "grade_batch", "shutdown"],
        }
    elif operation == "load":
        result = {
            "benchmark": "mmlu-pro",
            "problem_count": len(problems),
            "dataset": {
                "provider": "lighteval",
                "repository": "TIGER-Lab/MMLU-Pro",
                "subset": "default",
                "revision": "fixture-dataset-revision",
                "evaluation_splits": ["test"],
                "task_version": 1,
            },
            "grader": "lighteval task metrics",
        }
    elif operation == "next_problems":
        offset = request["offset"]
        page = problems[offset : offset + request["limit"]]
        result = {
            "items": page,
            "next_offset": offset + len(page),
            "done": offset + len(page) >= len(problems),
        }
    elif operation == "grade_batch":
        assert len(request["items"]) == 2
        grades = []
        for item in request["items"]:
            correct = item["problem_id"] == "opaque-alpha" and "(B)" in item["response"]
            grades.append({
                "problem_id": item["problem_id"],
                "task": "mmlu_pro.math",
                "correct": correct,
                "unparsed": False,
                "confidence": 1.0 if correct else 0.0,
                "reasoning": "fixture canonical grade",
                "extracted_answer": item["response"],
            })
        result = {"items": grades}
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
    let prompt = body["messages"][0]["content"].as_str().unwrap_or_default();
    let answer = if prompt.contains("scored-alpha") {
        "The answer is (B)"
    } else {
        "The answer is (A)"
    };
    captured.0.lock().unwrap().push(body);
    let stream = format!(
        "data: {{\"object\":\"chat.completion.chunk\",\"choices\":[{{\"delta\":{{\"content\":{answer:?}}},\"finish_reason\":null}}]}}\n\n\
         data: {{\"object\":\"chat.completion.chunk\",\"choices\":[],\"usage\":{{\"prompt_tokens\":128,\"completion_tokens\":6}}}}\n\n\
         data: [DONE]\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

#[tokio::test]
async fn cli_keeps_inference_in_rust_and_grading_in_stdio_worker() {
    let captured = Captured::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let fixture_root = std::env::temp_dir().join(format!(
        "aiperf_accuracy_worker_{}_{}",
        std::process::id(),
        address.port()
    ));
    let module_dir = fixture_root.join("aiperf/accuracy");
    std::fs::create_dir_all(&module_dir).unwrap();
    std::fs::write(fixture_root.join("aiperf/__init__.py"), "").unwrap();
    std::fs::write(module_dir.join("__init__.py"), "").unwrap();
    std::fs::write(module_dir.join("worker.py"), FAKE_EVALUATOR).unwrap();

    let output_path = fixture_root.join("report.json");
    let csv_path = fixture_root.join("accuracy.csv");
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--accuracy-benchmark")
        .arg("mmlu-pro")
        .arg("--accuracy-tasks")
        .arg("math")
        .arg("--concurrency")
        .arg("2")
        .arg("--json")
        .arg(&output_path)
        .arg("--accuracy-csv")
        .arg(&csv_path)
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
    assert_eq!(report["run"]["mode"], "accuracy");
    assert_eq!(report["evaluator"]["worker_version"], "cli-fixture");
    assert_eq!(report["evaluator"]["packages"]["lighteval"], "fixture");
    assert_eq!(
        report["evaluator"]["dependency_lock_sha256"],
        "fixture-lock-digest"
    );
    assert_eq!(
        report["evaluator"]["dataset"]["revision"],
        "fixture-dataset-revision"
    );
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"].as_f64(),
        Some(2.0)
    );
    assert_eq!(report["accuracy"]["summary"]["overall"]["n"], 2);
    assert_eq!(report["accuracy"]["summary"]["overall"]["correct_count"], 1);
    assert_eq!(report["accuracy"]["summary"]["overall"]["accuracy"], 0.5);
    assert_eq!(report["accuracy_records"].as_array().unwrap().len(), 2);
    assert_eq!(report["accuracy_records"][0]["task"], "mmlu_pro.math");
    assert!(
        report["accuracy_records"][0]["result"]
            .get("ground_truth")
            .is_none()
    );
    let csv = std::fs::read_to_string(&csv_path).unwrap();
    assert!(csv.starts_with("task,correct,total,unparsed,accuracy\n"));
    assert!(csv.contains("mmlu_pro.math,1,2,0,0.5000"));
    assert!(csv.contains("OVERALL,1,2,0,0.5000"));

    let requests = captured.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|body| body["temperature"] == 0.0));
    assert!(requests.iter().all(|body| body["top_p"] == 1.0));
    assert!(requests.iter().all(|body| body["stream"] == true));
    assert!(
        requests
            .iter()
            .all(|body| body["stream_options"]["include_usage"] == true)
    );
    assert!(
        requests
            .iter()
            .all(|body| body["stop"] == serde_json::json!(["Question:"]))
    );
    assert!(requests.iter().all(|body| body["max_tokens"] == 37));
    drop(requests);
    std::fs::remove_dir_all(fixture_root).unwrap();
}
