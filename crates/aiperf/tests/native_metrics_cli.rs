// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI proof: HTTP/SSE events flow through the native accumulator and v2 reporter.

use axum::{Router, http::header, response::IntoResponse, routing::post};

async fn chat_handler() -> impl IntoResponse {
    let stream = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"one\"}}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\" two\"}}]}\n\n",
        "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2}}\n\n",
        "data: [DONE]\n\n"
    );
    ([(header::CONTENT_TYPE, "text/event-stream")], stream)
}

#[tokio::test]
async fn cli_writes_native_metrics_sweeps_and_usage_end_to_end() {
    let app = Router::new().route("/v1/chat/completions", post(chat_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let output_path = std::env::temp_dir().join(format!(
        "aiperf_native_metrics_{}_{}.json",
        std::process::id(),
        address.port()
    ));
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--requests")
        .arg("3")
        .arg("--concurrency")
        .arg("2")
        .arg("--isl")
        .arg("8")
        .arg("--osl")
        .arg("2")
        .arg("--json")
        .arg(&output_path)
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "online");
    assert_eq!(report["run"]["model"], "fixture-model");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"].as_f64(),
        Some(3.0)
    );
    assert_eq!(report["metrics"]["request_latency"]["type"], "distribution");
    assert_eq!(
        report["metrics"]["request_latency"]["series"][0]["stats"]["count"],
        3
    );
    assert_eq!(
        report["metrics"]["http_req_duration"]["type"],
        "distribution"
    );
    assert_eq!(
        report["metrics"]["stream_setup_latency"]["type"],
        "distribution"
    );
    assert_eq!(
        report["metrics"]["stream_prefill_latency"]["type"],
        "distribution"
    );
    assert!(
        report["metrics"]["http_req_data_received"]["series"][0]["stats"]["avg"]
            .as_f64()
            .is_some_and(|bytes| bytes > 0.0)
    );
    assert_eq!(
        report["metrics"]["effective_concurrency"]["type"],
        "distribution"
    );
    assert_eq!(
        report["metrics"]["total_usage_prompt_tokens"]["series"][0]["stats"]["value"].as_f64(),
        Some(21.0)
    );
    assert_eq!(
        report["metrics"]["total_output_tokens"]["series"][0]["stats"]["value"].as_f64(),
        Some(6.0)
    );
    assert!(report["summary"]["duration_s"].as_f64().is_some());

    std::fs::remove_file(output_path).unwrap();
}

#[tokio::test]
async fn graph_cli_merges_worker_metrics_into_the_same_native_report() {
    let app = Router::new().route("/v1/chat/completions", post(chat_handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let output_path = std::env::temp_dir().join(format!(
        "aiperf_graph_native_metrics_{}_{}.json",
        std::process::id(),
        address.port()
    ));
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg("--mode")
        .arg("graph")
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--turns")
        .arg("1")
        .arg("--instances")
        .arg("2")
        .arg("--workers")
        .arg("1")
        .arg("--concurrency")
        .arg("1")
        .arg("--osl")
        .arg("2")
        .arg("--conns")
        .arg("1")
        .arg("--json")
        .arg(&output_path)
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();
    assert_eq!(report["schema_version"], "2.0");
    assert_eq!(report["run"]["mode"], "graph");
    assert_eq!(
        report["metrics"]["request_count"]["series"][0]["stats"]["total"].as_f64(),
        Some(2.0)
    );
    assert_eq!(
        report["metrics"]["request_latency"]["series"][0]["stats"]["count"],
        2
    );
    assert_eq!(
        report["metrics"]["effective_concurrency"]["type"],
        "distribution"
    );
    assert_eq!(
        report["metrics"]["total_usage_prompt_tokens"]["series"][0]["stats"]["value"].as_f64(),
        Some(14.0)
    );

    std::fs::remove_file(output_path).unwrap();
}
