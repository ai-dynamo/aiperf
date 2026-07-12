// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-HTTP CLI proof for dataset-backed request-rate multi-turn scheduling.

use std::sync::{Arc, Mutex};

use axum::{
    Router, body::Bytes, extract::State, http::header, response::IntoResponse, routing::post,
};

const SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"live answer\"},\"finish_reason\":null}]}\n\n",
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2}}\n\n",
    "data: [DONE]\n\n",
);

#[derive(Clone, Default)]
struct CapturedBodies(Arc<Mutex<Vec<serde_json::Value>>>);

async fn capture(State(captured): State<CapturedBodies>, body: Bytes) -> impl IntoResponse {
    captured
        .0
        .lock()
        .unwrap()
        .push(serde_json::from_slice(&body).unwrap());
    ([(header::CONTENT_TYPE, "text/event-stream")], SSE)
}

#[tokio::test]
async fn cli_request_rate_accepts_dataset_and_drains_all_session_turns() {
    let captured = CapturedBodies::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(capture))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let unique = format!("{}-{}", std::process::id(), address.port());
    let dataset_path = std::env::temp_dir().join(format!("aiperf-rate-{unique}.json"));
    let timing_path = std::env::temp_dir().join(format!("aiperf-rate-{unique}-timing.json"));
    std::fs::write(
        &dataset_path,
        serde_json::to_vec(&serde_json::json!({
            "session_id":"proof",
            "turns":[
                {"text":"first question","output_length":2},
                {"text":"second question","delay":1,"output_length":2}
            ]
        }))
        .unwrap(),
    )
    .unwrap();

    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--request-rate")
        .arg("1000")
        .arg("--arrival")
        .arg("constant")
        .arg("--sessions")
        .arg("1")
        .arg("--concurrency")
        .arg("1")
        .arg("--prefill-concurrency")
        .arg("1")
        .arg("--input-file")
        .arg(&dataset_path)
        .arg("--input-format")
        .arg("multi_turn")
        .arg("--osl")
        .arg("2")
        .arg("--timing-json")
        .arg(&timing_path)
        .output()
        .await
        .unwrap();
    let _ = std::fs::remove_file(&dataset_path);
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let bodies = captured.0.lock().unwrap();
    assert_eq!(bodies.len(), 2);
    assert_eq!(bodies[0]["messages"][0]["content"], "first question");
    assert_eq!(
        bodies[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|message| message["role"].as_str().unwrap())
            .collect::<Vec<_>>(),
        vec!["user", "assistant", "user"]
    );
    assert_eq!(bodies[1]["messages"][1]["content"], "live answer");
    assert_eq!(bodies[1]["messages"][2]["content"], "second question");
    drop(bodies);

    let timing: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&timing_path).unwrap()).unwrap();
    assert_eq!(timing["strategy"], "request_rate");
    assert_eq!(timing["turns"].as_array().unwrap().len(), 2);
    assert_eq!(timing["turns"][0]["turn_index"], 0);
    assert_eq!(timing["turns"][1]["turn_index"], 1);
    assert_eq!(timing["schedule_timing"]["early_turns"], 0);
    std::fs::remove_file(timing_path).unwrap();
}

#[tokio::test]
async fn cli_request_rate_honors_synthetic_turn_count() {
    let captured = CapturedBodies::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(capture))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let timing_path = std::env::temp_dir().join(format!(
        "aiperf-rate-synthetic-{}-{}.json",
        std::process::id(),
        address.port()
    ));
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--request-rate")
        .arg("1000")
        .arg("--arrival")
        .arg("constant")
        .arg("--sessions")
        .arg("1")
        .arg("--turns")
        .arg("3")
        .arg("--think-time-ms")
        .arg("1")
        .arg("--isl")
        .arg("1")
        .arg("--osl")
        .arg("1")
        .arg("--timing-json")
        .arg(&timing_path)
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(captured.0.lock().unwrap().len(), 3);
    let timing: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&timing_path).unwrap()).unwrap();
    assert_eq!(timing["strategy"], "request_rate");
    assert_eq!(
        timing["turns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|turn| turn["turn_index"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    std::fs::remove_file(timing_path).unwrap();
}

#[tokio::test]
async fn cli_request_rate_adaptive_control_mutates_the_scheduled_generator() {
    let captured = CapturedBodies::default();
    let app = Router::new()
        .route("/v1/chat/completions", post(capture))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let artifact_dir = std::env::temp_dir().join(format!(
        "aiperf-rate-adaptive-{}-{}",
        std::process::id(),
        address.port()
    ));
    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("fixture-model")
        .arg("--request-rate")
        .arg("20")
        .arg("--arrival")
        .arg("constant")
        .arg("--duration")
        .arg("3")
        .arg("--turns")
        .arg("2")
        .arg("--concurrency")
        .arg("2")
        .arg("--isl")
        .arg("1")
        .arg("--osl")
        .arg("1")
        .arg("--adaptive-scale")
        .arg("--adaptive-control-variable")
        .arg("request_rate")
        .arg("--adaptive-control-min")
        .arg("5")
        .arg("--adaptive-control-max")
        .arg("20")
        .arg("--adaptive-assessment-period")
        .arg("1")
        .arg("--adaptive-sustain-duration")
        .arg("1")
        .arg("--adaptive-scale-sla")
        .arg("request_latency:p95:le:0")
        .arg("--adaptive-base-step")
        .arg("1")
        .arg("--adaptive-max-step-multiplier")
        .arg("1")
        .arg("--adaptive-artifact-dir")
        .arg(&artifact_dir)
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!captured.0.lock().unwrap().is_empty());
    let summary: serde_json::Value = serde_json::from_slice(
        &std::fs::read(artifact_dir.join("adaptive_scale_summary.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(summary["schema_version"], 2);
    assert_eq!(summary["control_variable"], "request_rate");
    assert_eq!(summary["status"], "failed");
    std::fs::remove_dir_all(artifact_dir).unwrap();
}
