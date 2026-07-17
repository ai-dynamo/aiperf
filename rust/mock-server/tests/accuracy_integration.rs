// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wire tests for accuracy responses, telemetry, and adversarial SSE frames.

use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::time::Duration;

use aiperf_mock_server::accuracy::AccuracyFormat;
use aiperf_mock_server::{MockServerConfig, build_router};
use serde_json::{Value, json};

async fn spawn_server(cfg: MockServerConfig) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let cfg = cfg.apply_flags();
    let std_listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr: SocketAddr = std_listener.local_addr().unwrap();
    drop(std_listener);
    let state = aiperf_mock_server::app::build_state(cfg);
    let app = build_router(state);
    let tcp = tokio::net::TcpListener::bind(addr).await.unwrap();
    let bound = tcp.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        axum::serve(tcp, app.into_make_service()).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    (bound, handle)
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

fn write_dataset(name: &str, lines: &[Value]) -> PathBuf {
    let mut body = String::new();
    for l in lines {
        body.push_str(&serde_json::to_string(l).unwrap());
        body.push('\n');
    }
    let path = std::env::temp_dir().join(format!(
        "aiperf-mock-accuracy-{}-{}.jsonl",
        name,
        std::process::id()
    ));
    std::fs::write(&path, body).unwrap();
    path
}

fn base_cfg(dataset: &Path) -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        random_seed: Some(7),
        accuracy_dataset: Some(dataset.to_string_lossy().into_owned()),
        accuracy_format: AccuracyFormat::Mmlu,
        ..MockServerConfig::default()
    }
}

async fn chat_content(addr: SocketAddr, prompt: &str) -> Value {
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": prompt}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    body["choices"][0]["message"].clone()
}

#[tokio::test]
async fn correct_answer_is_wired_through() {
    let ds = write_dataset(
        "correct",
        &[json!({"text": "What is 2+2?", "ground_truth": "B"})],
    );
    let mut cfg = base_cfg(&ds);
    cfg.accuracy_correct_rate = 1.0;
    let (addr, _h) = spawn_server(cfg).await;
    let msg = chat_content(addr, "What is 2+2?").await;
    assert_eq!(msg["content"], "The answer is (B)");
    assert!(msg.get("reasoning_content").is_none());
}

#[tokio::test]
async fn wrong_answer_at_zero_rate() {
    let ds = write_dataset(
        "wrong",
        &[json!({"text": "What is 2+2?", "ground_truth": "B"})],
    );
    let mut cfg = base_cfg(&ds);
    cfg.accuracy_correct_rate = 0.0;
    let (addr, _h) = spawn_server(cfg).await;
    let msg = chat_content(addr, "What is 2+2?").await;
    let content = msg["content"].as_str().unwrap();
    assert!(content.starts_with("The answer is ("), "content={content}");
    assert_ne!(content, "The answer is (B)");
}

#[tokio::test]
async fn cot_populates_reasoning_content() {
    let ds = write_dataset("cot", &[json!({"text": "Q", "ground_truth": "B"})]);
    let mut cfg = base_cfg(&ds);
    cfg.accuracy_correct_rate = 1.0;
    cfg.accuracy_cot_rate = 1.0;
    cfg.accuracy_reasoning_field = true;
    let (addr, _h) = spawn_server(cfg).await;
    let msg = chat_content(addr, "Q").await;
    assert_eq!(msg["content"], "The answer is (B)");
    let reasoning = msg["reasoning_content"].as_str().unwrap();
    assert!(
        reasoning.contains("The answer is (B)"),
        "reasoning={reasoning}"
    );
}

#[tokio::test]
async fn unmatched_prompt_falls_through_to_corpus() {
    let ds = write_dataset(
        "unmatched",
        &[json!({"text": "known prompt", "ground_truth": "B"})],
    );
    let mut cfg = base_cfg(&ds);
    cfg.accuracy_correct_rate = 1.0;
    let (addr, _h) = spawn_server(cfg).await;
    let msg = chat_content(addr, "an entirely different prompt string").await;
    assert_ne!(msg["content"], "The answer is (B)");
}

#[tokio::test]
async fn live_accuracy_endpoint_and_prometheus_reflect_served_requests() {
    let ds = write_dataset(
        "live",
        &[
            json!({"text": "p one", "ground_truth": "B", "task": "demo"}),
            json!({"text": "p two", "ground_truth": "B", "task": "demo"}),
            json!({"text": "p three", "ground_truth": "B", "task": "demo"}),
        ],
    );
    let mut cfg = base_cfg(&ds);
    cfg.accuracy_correct_rate = 1.0;
    let (addr, _h) = spawn_server(cfg).await;

    for p in ["p one", "p two", "p three", "totally unknown prompt"] {
        let _ = chat_content(addr, p).await;
    }

    let acc: Value = client()
        .get(format!("http://{addr}/accuracy"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(acc["enabled"], true);
    assert_eq!(acc["matched"], 3);
    assert_eq!(acc["correct"], 3);
    assert_eq!(acc["incorrect"], 0);
    assert_eq!(acc["accuracy"], 1.0);
    assert_eq!(acc["unmatched"], 1);
    assert_eq!(acc["tasks"]["demo"]["matched"], 3);
    assert_eq!(acc["tasks"]["demo"]["correct"], 3);

    let prom = client()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(
        prom.contains("aiperf_mock_accuracy_matched_total 3"),
        "missing matched counter:\n{prom}"
    );
    assert!(prom.contains("aiperf_mock_accuracy_correct_total 3"));
    assert!(prom.contains("aiperf_mock_accuracy_ratio 1.000000"));
    assert!(prom.contains("aiperf_mock_accuracy_task_correct_total{task=\"demo\"} 3"));
}

#[tokio::test]
async fn accuracy_endpoint_disabled_without_dataset() {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        ..MockServerConfig::default()
    };
    let (addr, _h) = spawn_server(cfg).await;
    let acc: Value = client()
        .get(format!("http://{addr}/accuracy"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(acc["enabled"], false);
}

#[tokio::test]
async fn adversarial_null_object_frame_is_served_in_stream() {
    let lines: Vec<Value> = (0..40)
        .map(|i| json!({"text": format!("prompt number {i}"), "ground_truth": "B"}))
        .collect();
    let ds = write_dataset("adversarial", &lines);
    let mut cfg = base_cfg(&ds);
    cfg.accuracy_correct_rate = 1.0;
    cfg.accuracy_adversarial_rate = 1.0;
    let (addr, _h) = spawn_server(cfg).await;

    let mut saw_null_object = false;
    for i in 0..40 {
        let resp = client()
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": format!("prompt number {i}")}],
                "stream": true,
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "request {i} failed");
        let body = resp.text().await.unwrap();
        assert!(body.contains("[DONE]"), "request {i} missing [DONE]");
        if body.contains("\"object\":null") {
            saw_null_object = true;
        }
    }
    assert!(
        saw_null_object,
        "no NullObjectChunk adversarial frame appeared across 40 prompts"
    );
}
