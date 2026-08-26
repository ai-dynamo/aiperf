// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Production-transport e2e: fire the legacy dispatch plan through the runtime's
//! OWN clock-injected Hyper client ([`HttpClient`]) — the production transport
//! stack, not a raw socket — against a streaming SSE endpoint, and assert each
//! returned [`RequestRecord`] (the export raw record) carries the byte-exact
//! request content, the streamed response tokens, captured TTFT, and timing.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::time::Duration;

use aiperf_runtime::agentx::config::WekaConfig;
use aiperf_runtime::agentx::loader::{MainReconstructOptions, convert_trace_to_conversations};
use aiperf_runtime::agentx::replay::build_dispatch_plan;
use aiperf_runtime::agentx::synth::TokenSynth;
use aiperf_runtime::agentx::trace::{HashIdScope, WekaNormalRequest, WekaRequest, WekaTrace};
use aiperf_runtime::agentx::wire::ChatRequestOptions;
use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::client::http_client::HttpClient;
use aiperf_runtime::transport::http::config::ClientConfig;

use axum::response::sse::{Event, Sse};
use axum::{Router, routing::post};
use bytes::Bytes;

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

fn run_local<F: std::future::Future>(fut: F) -> F::Output {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, fut)
}

#[test]
fn legacy_plan_fires_through_runtime_hyper_client_into_export_records() {
    run_local(async {
        const OUT_TOKENS: usize = 4;
        // In-process streaming SSE inference endpoint.
        async fn handler()
        -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
            let stream = futures::stream::unfold(0usize, |i| async move {
                if i > OUT_TOKENS {
                    return None;
                }
                tokio::time::sleep(Duration::from_millis(if i == 0 { 30 } else { 15 })).await;
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

        // Reconstruct + build the transport-ready dispatch plan.
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

        // Fire each request through the runtime's OWN production Hyper client.
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let client = HttpClient::new(clock, ClientConfig::default());
        let url = url::Url::parse(&format!("http://{addr}/v1/chat/completions")).unwrap();
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        headers.insert("Accept".into(), "text/event-stream".into());

        for item in &plan {
            let body = Bytes::from(serde_json::to_vec(&item.request_body).unwrap());
            let ttft = Rc::new(Cell::new(None::<i64>));
            let ttft_cb = ttft.clone();
            let rec = client
                .request(&url, &headers, body.clone(), true, move |ns| {
                    ttft_cb.set(Some(ns))
                })
                .await;

            // The dispatch plan's byte-exact request body (verified in the raw-HTTP
            // e2e) is delivered by the production transport, which returns a
            // RequestRecord — the export raw record — with the streamed response
            // tokens, captured TTFT, and request timing.
            assert!(!rec.has_error(), "transport error: {:?}", rec.error);
            assert_eq!(
                rec.status,
                Some(200),
                "production transport delivered the request"
            );
            assert!(
                !rec.responses.is_empty(),
                "streamed response tokens recorded"
            );
            assert!(ttft.get().is_some(), "TTFT captured by the transport");
            assert!(
                rec.recv_start_ns.is_some(),
                "response-start (TTFT) timing recorded"
            );
            assert!(
                rec.end_ns.is_some() && rec.end_ns.unwrap() >= rec.start_ns,
                "request timing"
            );
        }
    });
}
