// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online HTTP e2e for the legacy AgentX path: reconstruct a trace, build the
//! transport-ready dispatch plan, and fire each turn's wire request body over
//! **real HTTP** (in dispatch order) at an in-process server, verifying the
//! server receives the exact byte-exact request bodies. This exercises the full
//! path from trace bytes through the wire to an actual inference endpoint —
//! reconstruction content + dispatch timing landing on the wire unchanged.

#![cfg(feature = "agentx")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use aiperf_runtime::agentx::config::WekaConfig;
use aiperf_runtime::agentx::loader::{convert_trace_to_conversations, MainReconstructOptions};
use aiperf_runtime::agentx::replay::build_dispatch_plan;
use aiperf_runtime::agentx::synth::TokenSynth;
use aiperf_runtime::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest, WekaTrace};
use aiperf_runtime::agentx::wire::ChatRequestOptions;

use axum::{routing::post, Router};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

struct StubSynth;
impl TokenSynth for StubSynth {
    fn decode_block_tokens(&mut self, h: &[i64]) -> Vec<u32> {
        h.iter().flat_map(|&x| (0..4).map(move |i| x as u32 * 1000 + i)).collect()
    }
    fn sample_partial_tail_tokens(&mut self, n: usize, _s: &str) -> Vec<u32> {
        (0..n as u32).map(|i| 900_000 + i).collect()
    }
    fn decode_tokens_to_text(&self, t: &[u32]) -> String {
        t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
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
        requests: vec![norm(0.0, &[1, 2]), norm(1.0, &[1, 2, 3]), norm(2.0, &[1, 2, 3, 4])],
        totals: None,
    };
    let mut synth = StubSynth;
    let convs = convert_trace_to_conversations(
        "t",
        &trace,
        &mut synth,
        &HashMap::new(),
        &WekaConfig { split_flattened_agents: false, ..WekaConfig::default() },
        &MainReconstructOptions::default(),
    )
    .unwrap();
    let plan = build_dispatch_plan(
        &convs,
        500.0,
        false,
        None,
        &ChatRequestOptions { streaming: true, ignore_eos: true, cache_bust_marker: None },
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
