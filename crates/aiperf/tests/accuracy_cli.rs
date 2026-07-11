// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native MMLU-Pro CLI acceptance test: dataset -> prompt -> HTTP/SSE -> grader -> report.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::{Json, Router, extract::State, http::header, response::IntoResponse, routing::post};
use serde_json::Value;

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
        "data: {{\"choices\":[{{\"delta\":{{\"content\":{answer:?}}},\"finish_reason\":null}}]}}\n\n\
         data: {{\"choices\":[],\"usage\":{{\"prompt_tokens\":128,\"completion_tokens\":6}}}}\n\n\
         data: [DONE]\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

#[tokio::test]
async fn cli_runs_mmlu_pro_end_to_end() {
    let captured = Captured::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_handler))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mmlu_pro");
    let output_path = std::env::temp_dir().join(format!(
        "aiperf_mmlu_pro_report_{}_{}.json",
        std::process::id(),
        address.port()
    ));
    let csv_path = std::env::temp_dir().join(format!(
        "aiperf_mmlu_pro_accuracy_{}_{}.csv",
        std::process::id(),
        address.port()
    ));
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--accuracy-benchmark")
        .arg("mmlu-pro")
        .arg("--accuracy-dataset")
        .arg(&fixture)
        .arg("--accuracy-tasks")
        .arg("math")
        .arg("--concurrency")
        .arg("2")
        .arg("--json")
        .arg(&output_path)
        .arg("--accuracy-csv")
        .arg(&csv_path)
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
    assert_eq!(report["metrics"]["request_count"]["type"], "counter");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"].as_f64(),
        Some(2.0)
    );
    assert_eq!(report["accuracy"]["summary"]["overall"]["n"], 2);
    assert_eq!(report["accuracy"]["summary"]["overall"]["correct_count"], 1);
    assert_eq!(report["accuracy"]["summary"]["overall"]["accuracy"], 0.5);
    assert_eq!(report["accuracy_records"].as_array().unwrap().len(), 2);
    assert_eq!(report["accuracy_records"][0]["task"], "mmlu_pro.math");
    let csv = std::fs::read_to_string(&csv_path).unwrap();
    assert!(csv.starts_with("task,correct,total,unparsed,accuracy\n"));
    assert!(csv.contains("mmlu_pro.math,1,2,0,0.5000"));
    assert!(csv.contains("OVERALL,1,2,0,0.5000"));

    let requests = captured.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|body| body["temperature"] == 0.0));
    assert!(requests.iter().all(|body| body["top_p"] == 1.0));
    assert!(
        requests
            .iter()
            .all(|body| body["stop"] == serde_json::json!(["Question:"]))
    );
    assert!(requests.iter().all(|body| body["max_tokens"] == 2_048));
    drop(requests);
    std::fs::remove_file(output_path).unwrap();
    std::fs::remove_file(csv_path).unwrap();
}
