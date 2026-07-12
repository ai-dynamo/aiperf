// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native CLI proof for Anthropic Messages wire compatibility.

use std::sync::{Arc, Mutex};

use axum::{
    Router,
    body::Bytes,
    extract::State,
    http::{HeaderMap, header},
    response::IntoResponse,
    routing::post,
    Json,
};

const SSE: &str = concat!(
    "event: message_start\n",
    "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":4,\"output_tokens\":1}}}\n\n",
    "event: content_block_delta\n",
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
    "event: message_delta\n",
    "data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":2}}\n\n",
    "event: message_stop\n",
    "data: {\"type\":\"message_stop\"}\n\n",
);

#[derive(Clone, Default)]
struct CapturedWire(Arc<Mutex<Option<(HeaderMap, Bytes)>>>);

async fn messages(
    State(captured): State<CapturedWire>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    *captured.0.lock().unwrap() = Some((headers, body));
    ([(header::CONTENT_TYPE, "text/event-stream")], SSE)
}

async fn messages_json(
    State(captured): State<CapturedWire>,
    headers: HeaderMap,
    body: Bytes,
) -> impl IntoResponse {
    *captured.0.lock().unwrap() = Some((headers, body));
    Json(serde_json::json!({
        "id":"msg_2",
        "type":"message",
        "role":"assistant",
        "content":[
            {"type":"thinking","thinking":"why"},
            {"type":"text","text":"answer"}
        ],
        "model":"claude-sonnet-4-20250514",
        "stop_reason":"end_turn",
        "usage":{"input_tokens":4,"output_tokens":2}
    }))
}

#[tokio::test]
async fn cli_selects_messages_and_preserves_python_pr_wire_bytes() {
    let captured = CapturedWire::default();
    let app = Router::new()
        .route("/v1/messages", post(messages))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let unique = format!("{}-{}", std::process::id(), address.port());
    let dataset_path = std::env::temp_dir().join(format!("aiperf-messages-{unique}.json"));
    let report_path = std::env::temp_dir().join(format!("aiperf-messages-{unique}-report.json"));
    std::fs::write(&dataset_path, br#"{"text":"Hello!","output_length":2}"#).unwrap();

    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("claude-sonnet-4-20250514")
        .arg("--endpoint-type")
        .arg("messages")
        .arg("--api-key")
        .arg("sk-ant-cli-secret")
        .arg("--header")
        .arg("anthropic-beta:extended-thinking-test")
        .arg("--extra-inputs")
        .arg("temperature:0.2")
        .arg("--request-rate")
        .arg("1000")
        .arg("--arrival")
        .arg("constant")
        .arg("--requests")
        .arg("1")
        .arg("--concurrency")
        .arg("1")
        .arg("--input-file")
        .arg(&dataset_path)
        .arg("--input-format")
        .arg("single_turn")
        .arg("--tokenizer")
        .arg("builtin")
        .arg("--osl")
        .arg("2")
        .arg("--json")
        .arg(&report_path)
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

    let (headers, body) = captured.0.lock().unwrap().take().unwrap();
    assert_eq!(
        body.as_ref(),
        br#"{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}],"max_tokens":2,"stream":true,"temperature":0.2}"#
    );
    assert_eq!(headers["x-api-key"], "sk-ant-cli-secret");
    assert_eq!(headers["anthropic-version"], "2023-06-01");
    assert_eq!(headers["anthropic-beta"], "extended-thinking-test");
    assert!(headers.get(header::AUTHORIZATION).is_none());

    let report: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&report_path).unwrap()).unwrap();
    assert_eq!(
        report["metrics"]["total_usage_prompt_tokens"]["series"][0]["stats"]["value"],
        4.0
    );
    assert_eq!(
        report["metrics"]["total_usage_completion_tokens"]["series"][0]["stats"]["value"],
        2.0
    );
    std::fs::remove_file(report_path).unwrap();
}

#[tokio::test]
async fn cli_messages_non_streaming_omits_stream_and_parses_message_json() {
    let captured = CapturedWire::default();
    let app = Router::new()
        .route("/v1/messages", post(messages_json))
        .with_state(captured.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let output = tokio::process::Command::new(env!("CARGO_BIN_EXE_aiperf"))
        .arg(format!("http://{address}"))
        .arg("claude-sonnet-4-20250514")
        .arg("--endpoint-type")
        .arg("messages")
        .arg("--streaming=false")
        .arg("--api-key")
        .arg("sk-ant-json-secret")
        .arg("--requests")
        .arg("1")
        .arg("--concurrency")
        .arg("1")
        .arg("--isl")
        .arg("1")
        .arg("--osl")
        .arg("2")
        .output()
        .await
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let (headers, body) = captured.0.lock().unwrap().take().unwrap();
    let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["model"], "claude-sonnet-4-20250514");
    assert_eq!(body["max_tokens"], 2);
    assert!(body.get("stream").is_none());
    assert_eq!(headers["x-api-key"], "sk-ant-json-secret");
    assert!(headers.get(header::AUTHORIZATION).is_none());
}
