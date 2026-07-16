// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end integration tests for aiperf-mock-server.
//!
//! These spin up a real axum server on a random port, hit it via reqwest, and
//! validate responses, streaming behaviour, and Prometheus exposition.

use std::net::{SocketAddr, TcpListener};
use std::sync::Arc;
use std::time::Duration;

use aiperf_mock_server::{MockServerConfig, build_router};
use futures::StreamExt;
use serde_json::{Value, json};

async fn spawn_server(cfg: MockServerConfig) -> (SocketAddr, tokio::task::JoinHandle<()>) {
    let cfg = cfg.apply_flags();
    // Pick a free port by opening a 0 port, reading, then releasing.
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
    // Give the server a moment to accept connections.
    tokio::time::sleep(Duration::from_millis(50)).await;
    (bound, handle)
}

fn fast_cfg() -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        ..MockServerConfig::default()
    }
    .apply_flags()
}

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

// ============================================================================

#[tokio::test]
async fn health_returns_healthy() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "healthy");
    assert!(body["config"].is_object());
}

#[tokio::test]
async fn root_returns_info() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .get(format!("http://{addr}/"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["version"], "2.0.0");
    assert_eq!(body["message"], "AIPerf Mock Server");
}

#[tokio::test]
async fn chat_completions_non_streaming() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "gpt-4");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(usage["completion_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["choices"][0]["message"]["role"].as_str().unwrap(),
        "assistant"
    );
}

#[tokio::test]
async fn chat_completions_streaming() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Stream please"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.starts_with("text/event-stream"));

    let mut stream = resp.bytes_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        buf.extend_from_slice(&chunk.unwrap());
    }
    let text = String::from_utf8(buf).unwrap();
    assert!(text.contains("data: "));
    assert!(text.contains("[DONE]"));
    assert!(text.contains("chat.completion.chunk"));
    assert!(text.contains("\"usage\""));
}

#[tokio::test]
async fn messages_non_streaming_returns_anthropic_message() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/messages"))
        .header("x-api-key", "test")
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["model"], "mock-model");
    assert!(body["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(body["usage"]["output_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn messages_streaming_returns_anthropic_events() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/messages"))
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Stream please"}],
            "max_tokens": 8,
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get(reqwest::header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("text/event-stream")
    );

    let text = String::from_utf8(resp.bytes().await.unwrap().to_vec()).unwrap();
    assert!(text.contains(r#"event: message_start"#));
    assert!(text.contains(r#""type":"content_block_delta""#));
    assert!(text.contains(r#""type":"message_delta""#));
    assert!(text.contains(r#"event: message_stop"#));
}

#[tokio::test]
async fn text_completions_non_streaming() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/completions"))
        .json(&json!({ "model": "gpt-4", "prompt": "Hello world" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "text_completion");
    assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
    assert!(body["choices"][0]["text"].is_string());
}

#[tokio::test]
async fn text_completions_streaming_with_usage() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/completions"))
        .json(&json!({
            "model": "gpt-4",
            "prompt": "Streaming text please",
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("data: "));
    assert!(text.contains("[DONE]"));
    assert!(text.contains("\"usage\""));
}

#[tokio::test]
async fn embeddings_returns_768_dim() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/embeddings"))
        .json(&json!({ "model": "emb", "input": ["hello", "world"] }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["embedding"].as_array().unwrap().len(), 768);
    // Deterministic embeddings: same text => same embedding
    let resp2 = client()
        .post(format!("http://{addr}/v1/embeddings"))
        .json(&json!({ "model": "emb", "input": ["hello"] }))
        .send()
        .await
        .unwrap();
    let body2: Value = resp2.json().await.unwrap();
    assert_eq!(body2["data"][0]["embedding"], data[0]["embedding"]);
}

#[tokio::test]
async fn nim_ranking_sorts_by_score() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/ranking"))
        .json(&json!({
            "model": "reranker",
            "query": {"text": "q"},
            "passages": [{"text": "p1"}, {"text": "p2"}, {"text": "p3"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let rankings = body["rankings"].as_array().unwrap();
    assert_eq!(rankings.len(), 3);
    let scores: Vec<f64> = rankings
        .iter()
        .map(|r| r["relevance_score"].as_f64().unwrap())
        .collect();
    // Non-increasing
    assert!(scores.windows(2).all(|w| w[0] >= w[1]));
}

#[tokio::test]
async fn hf_tei_rerank() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/rerank"))
        .json(&json!({
            "query": "q",
            "texts": ["a", "b"],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["results"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn cohere_rerank() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v2/rerank"))
        .json(&json!({
            "query": "q",
            "documents": ["a", "b", "c"],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["results"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn tgi_generate_non_streaming() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/generate"))
        .json(&json!({
            "inputs": "Hello",
            "parameters": {"max_new_tokens": 10}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["generated_text"].is_string());
}

#[tokio::test]
async fn tgi_generate_streaming() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/generate_stream"))
        .json(&json!({
            "inputs": "Hello world",
            "parameters": {"max_new_tokens": 10}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    assert!(text.contains("token"));
    assert!(text.contains("generated_text"));
}

#[tokio::test]
async fn image_generation_b64() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/images/generations"))
        .json(&json!({
            "prompt": "a cat",
            "model": "flux",
            "n": 2,
            "response_format": "b64_json",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert!(data[0]["b64_json"].is_string());
}

#[tokio::test]
async fn image_retrieval() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/image/infer"))
        .json(&json!({
            "input": [{"type": "image_url", "url": "https://x/y.jpg"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert!(data[0]["bounding_boxes"].is_object());
}

#[tokio::test]
async fn custom_multimodal() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/custom-multimodal"))
        .json(&json!({
            "modality_bundle": {
                "text_fragments": ["hello"],
                "visual_assets": {"images": [], "videos": []},
                "audio_streams": []
            },
            "inference_params": {"model_id": "mm"}
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["text"].is_string());
    assert!(body["completion"]["metadata"]["tokens_used"].is_object());
}

#[tokio::test]
async fn solido_rag() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/rag/api/prompt"))
        .json(&json!({
            "query": ["what is ai"],
            "inference_model": "solido-model",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert!(body["content"].is_string());
    assert!(body["sources"].is_array());
}

#[tokio::test]
async fn prometheus_metrics_endpoints_all_return_text() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    // Fire one chat request so counters are non-zero.
    let _ = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({"model": "m1", "messages": [{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();

    for (path, marker) in [
        ("/metrics", "aiperf_mock_requests_total"),
        ("/vllm/metrics", "vllm:e2e_request_latency_seconds"),
        ("/sglang/metrics", "sglang:e2e_request_latency_seconds"),
        ("/trtllm/metrics", "trtllm:e2e_request_latency_seconds"),
        (
            "/dynamo_frontend/metrics",
            "dynamo_frontend_request_duration_seconds",
        ),
        (
            "/dynamo_component/prefill/metrics",
            "dynamo_component_request_duration_seconds",
        ),
        (
            "/dynamo_component/decode/metrics",
            "dynamo_component_request_duration_seconds",
        ),
    ] {
        let r = client()
            .get(format!("http://{addr}{path}"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "path={path}");
        let body = r.text().await.unwrap();
        assert!(
            body.contains(marker),
            "expected marker {marker} in {path}, got:\n{body}"
        );
    }
}

#[tokio::test]
async fn dcgm_metrics_endpoints() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    for i in 1..=2 {
        let r = client()
            .get(format!("http://{addr}/dcgm{i}/metrics"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "i={i}");
        let body = r.text().await.unwrap();
        assert!(body.contains("DCGM_FI_DEV_GPU_UTIL"));
        assert!(body.contains("DCGM_FI_DEV_POWER_USAGE"));
    }
    // 3 is out-of-range - only 2 fakers.
    let r = client()
        .get(format!("http://{addr}/dcgm3/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn error_injection() {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        error_rate: 100.0,
        random_seed: Some(42),
        ..MockServerConfig::default()
    };
    let (addr, _h) = spawn_server(cfg).await;
    let r = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({ "model": "m", "messages": [{"role":"user","content":"x"}] }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 500);
    let body: Value = r.json().await.unwrap();
    // Default menu is the single historical `500` code, echoed in the detail.
    assert_eq!(body["detail"], "Simulated error (status 500)");
}

/// The status-code menu is honored: with `--error-status-codes 429` every
/// injected error is a 429 carrying a `Retry-After` backoff header, not the
/// hardcoded 500.
#[tokio::test]
async fn error_injection_status_code_menu_and_retry_after() {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        error_rate: 100.0,
        error_status_codes: vec![429],
        error_retry_after: 7,
        random_seed: Some(42),
        ..MockServerConfig::default()
    };
    let (addr, _h) = spawn_server(cfg).await;
    let r = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({ "model": "m", "messages": [{"role":"user","content":"x"}] }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 429);
    assert_eq!(
        r.headers().get("retry-after").and_then(|v| v.to_str().ok()),
        Some("7"),
        "429 must carry the configured Retry-After header"
    );
    let body: Value = r.json().await.unwrap();
    assert_eq!(body["detail"], "Simulated error (status 429)");
}

/// A mid-stream SSE error emits a few normal token frames and then a terminal
/// `event: error` frame, with no `[DONE]` sentinel — the shape the runner
/// classifies as a transport SSE error.
#[tokio::test]
async fn error_injection_midstream_sse() {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        error_midstream_rate: 1.0,
        random_seed: Some(42),
        ..MockServerConfig::default()
    };
    let (addr, _h) = spawn_server(cfg).await;
    let body = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "m",
            "stream": true,
            "messages": [{"role":"user","content":"a longer prompt to force several output tokens"}],
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    // The stream terminates with an `event: error` frame and never emits [DONE].
    assert!(
        body.contains("event: error"),
        "mid-stream body must carry an SSE error frame, got: {body:?}"
    );
    assert!(
        !body.contains("[DONE]"),
        "mid-stream error must not send the [DONE] sentinel, got: {body:?}"
    );
    // Some normal content frames precede the error (partial content).
    assert!(
        body.contains("chat.completion.chunk"),
        "mid-stream body should include partial token frames, got: {body:?}"
    );
}

#[tokio::test]
async fn reasoning_model_includes_reasoning_content() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let r = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-oss-120b",
            "messages": [{"role":"user","content":"solve this problem carefully"}],
            "max_completion_tokens": 600,
            "reasoning_effort": "low",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let msg = &body["choices"][0]["message"];
    assert!(msg["reasoning_content"].is_string());
    let details = &body["usage"]["completion_tokens_details"];
    assert!(details.is_object());
    assert!(details["reasoning_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn deterministic_embeddings_across_requests() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let a = client()
        .post(format!("http://{addr}/v1/embeddings"))
        .json(&json!({ "model": "e", "input": "deterministic-text" }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    let b = client()
        .post(format!("http://{addr}/v1/embeddings"))
        .json(&json!({ "model": "e", "input": "deterministic-text" }))
        .send()
        .await
        .unwrap()
        .json::<Value>()
        .await
        .unwrap();
    assert_eq!(a["data"][0]["embedding"], b["data"][0]["embedding"]);
}

#[tokio::test]
async fn ignore_eos_produces_exact_max_tokens() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let r = client()
        .post(format!("http://{addr}/v1/completions"))
        .json(&json!({
            "model": "m",
            "prompt": "hello world",
            "max_tokens": 42,
            "ignore_eos": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let completion_tokens = body["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(completion_tokens, 42);
    assert_eq!(body["choices"][0]["finish_reason"], "length");
}

#[tokio::test]
async fn fast_mode_has_near_zero_ttft() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let start = std::time::Instant::now();
    let r = client()
        .post(format!("http://{addr}/v1/completions"))
        .json(&json!({
            "model": "m",
            "prompt": "short",
            "max_tokens": 10,
            "ignore_eos": true,
        }))
        .send()
        .await
        .unwrap();
    let elapsed = start.elapsed();
    assert_eq!(r.status(), 200);
    // Fast mode should keep total latency below 100 ms in local runs.
    assert!(elapsed < Duration::from_millis(500), "elapsed={elapsed:?}");
}

#[tokio::test]
async fn dcgm_load_updates_on_completions() {
    // Ensure scraping DCGM after chat requests doesn't fail.
    let (addr, _h) = spawn_server(fast_cfg()).await;
    for _ in 0..3 {
        let _ = client()
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&json!({"model": "m", "messages": [{"role":"user","content":"abc"}]}))
            .send()
            .await
            .unwrap();
    }
    let r = client()
        .get(format!("http://{addr}/dcgm1/metrics"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body = r.text().await.unwrap();
    assert!(body.contains("DCGM_FI_DEV_GPU_UTIL"));
}

#[tokio::test]
async fn empty_prompt_yields_zero_completion() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let r = client()
        .post(format!("http://{addr}/v1/completions"))
        .json(&json!({ "model": "m", "prompt": "" }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body: Value = r.json().await.unwrap();
    let u = &body["usage"];
    assert_eq!(u["prompt_tokens"].as_u64().unwrap(), 0);
    assert_eq!(u["completion_tokens"].as_u64().unwrap(), 0);
}

#[tokio::test]
async fn chat_streaming_records_streaming_metric() {
    // spawn, stream, then scrape /metrics and confirm streaming counter incremented.
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "m",
            "messages": [{"role":"user","content":"stream"}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    let _ = resp.text().await.unwrap();
    let m = client()
        .get(format!("http://{addr}/metrics"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(m.contains("aiperf_mock_streaming_requests_total"));
}

#[tokio::test]
async fn fast_cfg_zeroes_latency() {
    let cfg = fast_cfg();
    assert_eq!(cfg.ttft, 0.0);
    assert_eq!(cfg.itl, 0.0);
    let _state = Arc::new(cfg);
}

// ============================================================================
// /v1/models
// ============================================================================

#[tokio::test]
async fn list_models_returns_defaults() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    let data = body["data"].as_array().unwrap();
    assert!(!data.is_empty(), "default model list should be non-empty");
    // Shape check on the first entry.
    let first = &data[0];
    assert_eq!(first["object"], "model");
    assert!(first["id"].is_string());
    assert!(first["created"].is_number());
    assert_eq!(first["owned_by"], "aiperf-mock");
    // Sorted (BTreeSet) — alphabetical order.
    let ids: Vec<&str> = data.iter().map(|m| m["id"].as_str().unwrap()).collect();
    let mut sorted = ids.clone();
    sorted.sort();
    assert_eq!(ids, sorted, "model ids should be returned sorted");
}

#[tokio::test]
async fn list_models_honors_explicit_config() {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        models: vec![
            "my-custom-model".to_string(),
            "another/model-v2".to_string(),
        ],
        ..MockServerConfig::default()
    };
    let (addr, _h) = spawn_server(cfg).await;
    let body: Value = client()
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<String> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert!(ids.contains(&"my-custom-model".to_string()));
    assert!(ids.contains(&"another/model-v2".to_string()));
    // Defaults should NOT be present when the caller specifies an explicit list.
    assert!(!ids.contains(&"gpt-4".to_string()));
}

#[tokio::test]
async fn list_models_includes_models_seen_via_traffic() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    // Fire a request with a custom model name; init_model_config fires inside
    // the chat handler, which should add it to the seen set.
    let _ = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({"model": "dynamic-model-1", "messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    let body: Value = client()
        .get(format!("http://{addr}/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(
        ids.contains(&"dynamic-model-1"),
        "expected dynamically-seen model in /v1/models listing, got {ids:?}"
    );
}

#[tokio::test]
async fn get_model_returns_single_entry() {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        models: vec!["gpt-4".to_string()],
        ..MockServerConfig::default()
    };
    let (addr, _h) = spawn_server(cfg).await;
    let resp = client()
        .get(format!("http://{addr}/v1/models/gpt-4"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["id"], "gpt-4");
    assert_eq!(body["object"], "model");
    assert_eq!(body["owned_by"], "aiperf-mock");
    assert!(body["created"].is_number());
}

#[tokio::test]
async fn get_model_404_for_unknown() {
    let cfg = MockServerConfig {
        fast: true,
        no_tokenizer: true,
        models: vec!["only-one".to_string()],
        ..MockServerConfig::default()
    };
    let (addr, _h) = spawn_server(cfg).await;
    let resp = client()
        .get(format!("http://{addr}/v1/models/never-registered"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["detail"]
            .as_str()
            .unwrap()
            .to_ascii_lowercase()
            .contains("not found")
    );
}

// ============================================================================
// Extended usage-accounting fields (--usage-* knobs)
// ============================================================================

/// A `--fast` config with every extended usage knob pinned to a distinct
/// nonzero value, so each emitted `usage` sub-field is individually assertable.
fn usage_fields_cfg() -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        usage_cache_write_tokens: 11,
        usage_cache_miss_tokens: 22,
        usage_cache_read_tokens: 33,
        usage_prompt_audio_tokens: 44,
        usage_completion_audio_tokens: 55,
        usage_prompt_audio_seconds: 6.5,
        usage_accepted_prediction_tokens: 77,
        usage_rejected_prediction_tokens: 88,
        usage_tool_use_prompt_tokens: 99,
        ..MockServerConfig::default()
    }
}

#[tokio::test]
async fn chat_non_streaming_emits_extended_usage_fields() {
    let (addr, _h) = spawn_server(usage_fields_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let u = &body["usage"];
    // Top-level OpenAI keys read by aiperf::endpoints::usage.
    assert_eq!(u["cache_creation_input_tokens"], 11);
    assert_eq!(u["prompt_cache_miss_tokens"], 22);
    assert_eq!(u["toolUsePromptTokenCount"], 99);
    assert_eq!(u["prompt_audio_seconds"], 6.5);
    // Nested detail keys.
    assert_eq!(u["prompt_tokens_details"]["audio_tokens"], 44);
    assert_eq!(u["completion_tokens_details"]["audio_tokens"], 55);
    assert_eq!(
        u["completion_tokens_details"]["accepted_prediction_tokens"],
        77
    );
    assert_eq!(
        u["completion_tokens_details"]["rejected_prediction_tokens"],
        88
    );
    // Anthropic-only cache-read is NOT serialized on the OpenAI usage.
    assert!(u.get("cache_read_input_tokens").is_none());
}

#[tokio::test]
async fn chat_streaming_usage_chunk_carries_extended_fields() {
    let (addr, _h) = spawn_server(usage_fields_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Stream please"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.unwrap();
    // Locate the terminal usage frame (the only SSE data line with a "usage" key).
    let usage_frame = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| l.trim() != "[DONE]")
        .filter_map(|l| serde_json::from_str::<Value>(l.trim()).ok())
        .find(|v| v.get("usage").map(|u| !u.is_null()).unwrap_or(false))
        .expect("a streamed usage chunk");
    let u = &usage_frame["usage"];
    assert_eq!(u["cache_creation_input_tokens"], 11);
    assert_eq!(u["prompt_cache_miss_tokens"], 22);
    assert_eq!(u["toolUsePromptTokenCount"], 99);
    assert_eq!(u["prompt_audio_seconds"], 6.5);
    assert_eq!(u["prompt_tokens_details"]["audio_tokens"], 44);
    assert_eq!(u["completion_tokens_details"]["audio_tokens"], 55);
    assert_eq!(
        u["completion_tokens_details"]["accepted_prediction_tokens"],
        77
    );
    assert_eq!(
        u["completion_tokens_details"]["rejected_prediction_tokens"],
        88
    );
}

#[tokio::test]
async fn messages_usage_carries_anthropic_cache_fields() {
    let (addr, _h) = spawn_server(usage_fields_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/messages"))
        .header("x-api-key", "test")
        .header("anthropic-version", "2023-06-01")
        .json(&json!({
            "model": "mock-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 8,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let u = &body["usage"];
    assert!(u["input_tokens"].as_u64().unwrap() > 0);
    assert!(u["output_tokens"].as_u64().unwrap() > 0);
    // Disjoint cache accounting the runner re-totals into prompt_tokens.
    assert_eq!(u["cache_read_input_tokens"], 33);
    assert_eq!(u["cache_creation_input_tokens"], 11);
}

/// Without the knobs, none of the extended sub-fields appear — a normal run's
/// usage payload is unchanged.
#[tokio::test]
async fn default_usage_omits_extended_fields() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let u = &body["usage"];
    assert!(u.get("cache_creation_input_tokens").is_none());
    assert!(u.get("prompt_cache_miss_tokens").is_none());
    assert!(u.get("toolUsePromptTokenCount").is_none());
    assert!(u.get("prompt_audio_seconds").is_none());
    assert!(u["prompt_tokens_details"].get("audio_tokens").is_none());
    // completion_tokens_details is absent entirely for a non-reasoning model.
    assert!(u.get("completion_tokens_details").is_none());
}

// ============================================================================
// Tool-call / function-call emission (`--tool-call-rate`).
// ============================================================================

fn tool_call_cfg() -> MockServerConfig {
    MockServerConfig {
        fast: true,
        no_tokenizer: true,
        tool_call_rate: 1.0,
        random_seed: Some(7),
        ..MockServerConfig::default()
    }
}

/// Non-streaming: at rate 1.0 the assistant message carries a single
/// `tool_calls` entry with the configured function name and argument string,
/// the finish reason is `tool_calls`, and the usage reports
/// `toolUsePromptTokenCount`.
#[tokio::test]
async fn tool_call_non_streaming_emits_message_tool_calls() {
    let (addr, _h) = spawn_server(tool_call_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the weather?"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    let tc = &choice["message"]["tool_calls"][0];
    assert_eq!(tc["type"], "function");
    assert!(tc["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(tc["function"]["name"], "get_weather");
    assert_eq!(tc["function"]["arguments"], r#"{"location":"NYC"}"#);
    // Tool-definition prompt tokens are reported under the exact key AIPerf reads.
    assert!(body["usage"]["toolUsePromptTokenCount"].as_u64().unwrap() > 0);
}

/// Streaming: the argument string is split across two `delta.tool_calls`
/// frames; merging them by `index` reconstructs the full function name and
/// arguments, and the terminal frame carries `finish_reason: "tool_calls"`.
#[tokio::test]
async fn tool_call_streaming_emits_delta_tool_calls() {
    let (addr, _h) = spawn_server(tool_call_cfg()).await;
    let text = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();

    let mut name = String::new();
    let mut arguments = String::new();
    let mut saw_finish = false;
    let mut frames_with_tool_calls = 0usize;
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload.trim() == "[DONE]" {
            continue;
        }
        let obj: Value = match serde_json::from_str(payload.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let choice = &obj["choices"][0];
        if choice["finish_reason"] == "tool_calls" {
            saw_finish = true;
        }
        if let Some(tcs) = choice["delta"]["tool_calls"].as_array() {
            frames_with_tool_calls += 1;
            for tc in tcs {
                if let Some(n) = tc["function"]["name"].as_str() {
                    name.push_str(n);
                }
                if let Some(a) = tc["function"]["arguments"].as_str() {
                    arguments.push_str(a);
                }
            }
        }
    }
    assert!(
        frames_with_tool_calls >= 2,
        "arguments should stream across >=2 frames, got {frames_with_tool_calls}"
    );
    assert_eq!(name, "get_weather");
    assert_eq!(arguments, r#"{"location":"NYC"}"#);
    assert!(saw_finish, "a frame must carry finish_reason=tool_calls");
    assert!(text.contains("toolUsePromptTokenCount"));
}

/// Rate 0.0 (the default) never emits tool calls: a normal assistant turn.
#[tokio::test]
async fn tool_call_disabled_by_default() {
    let (addr, _h) = spawn_server(fast_cfg()).await;
    let resp = client()
        .post(format!("http://{addr}/v1/chat/completions"))
        .json(&json!({
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hello"}],
        }))
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    assert!(body["choices"][0]["message"].get("tool_calls").is_none());
    assert_ne!(body["choices"][0]["finish_reason"], "tool_calls");
    assert!(body["usage"].get("toolUsePromptTokenCount").is_none());
}
