// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online HTTP e2e for the legacy AgentX path: reconstruct a trace, build the
//! transport-ready dispatch plan, and fire each turn's wire request body over
//! **real HTTP** (in dispatch order) at an in-process server, verifying the
//! server receives the exact byte-exact request bodies. This exercises the full
//! path from trace bytes through the wire to an actual inference endpoint —
//! reconstruction content + dispatch timing landing on the wire unchanged.
#![cfg(feature = "engine")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use aiperf_runtime::agentx::config::WekaConfig;
use aiperf_runtime::agentx::loader::{MainReconstructOptions, convert_trace_to_conversations};
use aiperf_runtime::agentx::replay::build_dispatch_plan;
use aiperf_runtime::agentx::synth::TokenSynth;
use aiperf_runtime::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest, WekaTrace};
use aiperf_runtime::agentx::wire::ChatRequestOptions;

use axum::response::sse::{Event, Sse};
use axum::{Router, routing::post};
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

struct StubSynth;
impl TokenSynth for StubSynth {
    fn decode_block_tokens(&mut self, h: &[i64]) -> Vec<u32> {
        h.iter()
            .flat_map(|&x| (0..4).map(move |i| x as u32 * 1000 + i))
            .collect()
    }
    fn sample_partial_tail_tokens(&mut self, n: usize, _s: &str) -> Vec<u32> {
        (0..n as u32).map(|i| 900_000 + i).collect()
    }
    fn decode_tokens_to_text(&self, t: &[u32]) -> String {
        t.iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn legacy_dispatch_plan_lands_byte_exact_over_real_http() {
    // In-process inference server recording every received request body.
    let received: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let app = {
        let received = received.clone();
        Router::new().route(
            "/v1/chat/completions",
            post(move |body: String| {
                let received = received.clone();
                async move {
                    received.lock().unwrap().push(body);
                    "{\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}"
                }
            }),
        )
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Reconstruct a trace and build the transport-ready dispatch plan.
    let norm = |t: f64, hs: &[i64]| {
        WekaRequest::Normal(WekaNormalRequest {
            t,
            model: "m".into(),
            input_length: hs.len() as i64 * 4,
            output_length: 4,
            hash_ids: hs.to_vec(),
            input_types: vec![],
            output_types: vec![],
            stop: String::new(),
            api_time: Some(0.1),
            think_time: None,
        })
    };
    let trace = WekaTrace {
        id: "t".into(),
        models: vec!["m".into()],
        block_size: 4,
        hash_id_scope: HashIdScope::Local,
        tool_tokens: 0,
        system_tokens: 0,
        requests: vec![
            norm(0.0, &[1, 2]),
            norm(1.0, &[1, 2, 3]),
            norm(2.0, &[1, 2, 3, 4]),
        ],
        totals: None,
    };
    let mut synth = StubSynth;
    let convs = convert_trace_to_conversations(
        "t",
        &trace,
        &mut synth,
        &HashMap::new(),
        &WekaConfig {
            split_flattened_agents: false,
            ..WekaConfig::default()
        },
        &MainReconstructOptions::default(),
    )
    .unwrap();
    let plan = build_dispatch_plan(
        &convs,
        500.0,
        false,
        None,
        &ChatRequestOptions {
            streaming: true,
            ignore_eos: true,
            cache_bust_marker: None,
            cache_bust_first_user_turn: false,
        },
    );
    assert_eq!(plan.len(), 3);

    // Fire each request over REAL HTTP in dispatch order (sorted by dispatch_ns).
    let mut items = plan.clone();
    items.sort_by_key(|i| i.dispatch_ns);
    for item in &items {
        let body = item.request_body.to_string();
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(req.as_bytes()).await.unwrap();
        // Read the response fully (guarantees the server recorded the body).
        let mut resp = Vec::new();
        stream.read_to_end(&mut resp).await.unwrap();
        assert!(!resp.is_empty());
    }

    // The server received the exact byte-exact wire bodies, in dispatch order.
    let got = received.lock().unwrap().clone();
    assert_eq!(got.len(), 3, "server received all three requests");
    for (sent, item) in got.iter().zip(&items) {
        let parsed: serde_json::Value = serde_json::from_str(sent).unwrap();
        assert_eq!(parsed, item.request_body, "on-wire body byte-exact");
        // Content + scenario shape survived the round trip to the endpoint.
        assert_eq!(parsed["stream"], serde_json::json!(true));
        assert_eq!(parsed["ignore_eos"], serde_json::json!(true));
        assert!(!parsed["messages"].as_array().unwrap().is_empty());
    }
}

/// An observed per-request raw export record captured from an actual streaming
/// round-trip: request content (byte-exact) + measured response timing
/// (TTFT/ITL, within transport-overhead tolerance) + response content.
#[derive(Debug)]
struct ObservedRecord {
    dispatch_ns: i64,
    request_body: serde_json::Value,
    ttft_ms: f64,
    itls_ms: Vec<f64>,
    output_tokens: usize,
    response_text: String,
}

/// Streaming online e2e: fire the dispatch plan at a streaming SSE endpoint that
/// emits generated tokens with a fixed TTFT + ITL, capture the observed response
/// timing + content per request, and assemble the export-level raw records.
/// Content is byte-exact; TTFT/ITL are asserted within a transport-overhead
/// tolerance (matching the project's generated-token timing test contract).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_run_captures_observed_timing_and_content_into_export_records() {
    const SERVER_TTFT_MS: u64 = 40;
    const SERVER_ITL_MS: u64 = 25;
    const OUT_TOKENS: usize = 4;

    // Streaming server: TTFT delay, then OUT_TOKENS tokens each after ITL.
    async fn handler() -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>>
    {
        // i in 0..OUT_TOKENS emit a token (i==0 after TTFT, i>0 after ITL);
        // i==OUT_TOKENS emits [DONE].
        let stream = futures::stream::unfold(0usize, |i| async move {
            if i > OUT_TOKENS {
                return None;
            }
            if i == 0 {
                tokio::time::sleep(Duration::from_millis(SERVER_TTFT_MS)).await;
            } else {
                tokio::time::sleep(Duration::from_millis(SERVER_ITL_MS)).await;
            }
            let ev = if i < OUT_TOKENS {
                let chunk = serde_json::json!({
                    "choices": [{"delta": {"content": format!("tok{i} ")}}]
                });
                Event::default().data(chunk.to_string())
            } else {
                Event::default().data("[DONE]")
            };
            Some((Ok::<_, std::convert::Infallible>(ev), i + 1))
        });
        Sse::new(stream)
    }
    let app = Router::new().route("/v1/chat/completions", post(handler));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    // Reconstruct + plan.
    let norm = |t: f64, hs: &[i64]| {
        WekaRequest::Normal(WekaNormalRequest {
            t,
            model: "m".into(),
            input_length: hs.len() as i64 * 4,
            output_length: 4,
            hash_ids: hs.to_vec(),
            input_types: vec![],
            output_types: vec![],
            stop: String::new(),
            api_time: Some(0.1),
            think_time: None,
        })
    };
    let trace = WekaTrace {
        id: "t".into(),
        models: vec!["m".into()],
        block_size: 4,
        hash_id_scope: HashIdScope::Local,
        tool_tokens: 0,
        system_tokens: 0,
        requests: vec![norm(0.0, &[1, 2]), norm(1.0, &[1, 2, 3])],
        totals: None,
    };
    let mut synth = StubSynth;
    let convs = convert_trace_to_conversations(
        "t",
        &trace,
        &mut synth,
        &HashMap::new(),
        &WekaConfig {
            split_flattened_agents: false,
            ..WekaConfig::default()
        },
        &MainReconstructOptions::default(),
    )
    .unwrap();
    let mut plan = build_dispatch_plan(
        &convs,
        500.0,
        false,
        None,
        &ChatRequestOptions {
            streaming: true,
            ignore_eos: true,
            cache_bust_marker: None,
            cache_bust_first_user_turn: false,
        },
    );
    plan.sort_by_key(|i| i.dispatch_ns);

    // Fire each request over real HTTP, capturing observed streaming timing.
    let mut observed: Vec<ObservedRecord> = Vec::new();
    for item in &plan {
        let body = item.request_body.to_string();
        let req = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let mut stream = TcpStream::connect(addr).await.unwrap();
        let t0 = Instant::now();
        stream.write_all(req.as_bytes()).await.unwrap();

        // Read the SSE stream token-by-token, timestamping each data: line.
        let mut token_times: Vec<f64> = Vec::new();
        let mut text = String::new();
        let mut buf = [0u8; 512];
        let mut acc = String::new();
        loop {
            let n = stream.read(&mut buf).await.unwrap();
            if n == 0 {
                break;
            }
            acc.push_str(&String::from_utf8_lossy(&buf[..n]));
            while let Some(pos) = acc.find('\n') {
                let line: String = acc.drain(..=pos).collect();
                let line = line.trim();
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        continue;
                    }
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                            token_times.push(t0.elapsed().as_secs_f64() * 1000.0);
                            text.push_str(c);
                        }
                    }
                }
            }
        }
        let ttft = *token_times.first().unwrap();
        let itls: Vec<f64> = token_times.windows(2).map(|w| w[1] - w[0]).collect();
        observed.push(ObservedRecord {
            dispatch_ns: item.dispatch_ns,
            request_body: item.request_body.clone(),
            ttft_ms: ttft,
            itls_ms: itls,
            output_tokens: token_times.len(),
            response_text: text,
        });
    }

    // Assemble + assert the export-level raw records.
    assert_eq!(observed.len(), 2);
    for rec in &observed {
        // Content: byte-exact request messages survived to the endpoint.
        assert!(!rec.request_body["messages"].as_array().unwrap().is_empty());
        assert_eq!(rec.request_body["stream"], serde_json::json!(true));
        // Output content exact.
        assert_eq!(rec.output_tokens, OUT_TOKENS);
        assert_eq!(rec.response_text, "tok0 tok1 tok2 tok3 ");
        // Timing within transport-overhead tolerance (generous for CI).
        assert!(
            rec.ttft_ms >= SERVER_TTFT_MS as f64 - 5.0,
            "ttft {} too low",
            rec.ttft_ms
        );
        assert!(
            rec.ttft_ms < SERVER_TTFT_MS as f64 + 120.0,
            "ttft {} too high",
            rec.ttft_ms
        );
        for itl in &rec.itls_ms {
            assert!(
                *itl >= SERVER_ITL_MS as f64 - 5.0 && *itl < SERVER_ITL_MS as f64 + 120.0,
                "itl {itl}"
            );
        }
        // The export record combines byte-exact request timing (dispatch) + content
        // with the observed response timing — the raw export shape.
        let _export = serde_json::json!({
            "dispatch_ns": rec.dispatch_ns,
            "request": rec.request_body,
            "ttft_ms": rec.ttft_ms,
            "itls_ms": rec.itls_ms,
            "output_tokens": rec.output_tokens,
            "response": rec.response_text,
        });
    }
}
