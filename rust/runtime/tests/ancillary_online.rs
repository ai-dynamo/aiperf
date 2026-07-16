// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-HTTP proofs for ancillary cancellation and multi-endpoint policy.

use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_runtime::ancillary::AncillaryTimingConfig;
use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::fixed_schedule::FixedScheduleConfig;
use aiperf_runtime::http::{Request, TransportSink};
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::{ErrorKind, RequestConfig};
use aiperf_runtime::transport::http::transport::http_transport::HttpTransport;
use axum::{
    Router, body::Bytes, extract::State, http::header, response::IntoResponse, routing::post,
};
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::observer::CollectorObserver;
use loadgen_core::sink::RequestObserver;
use uuid::Uuid;

mod common;

const SSE: &str = concat!(
    "data: {\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n",
    "data: [DONE]\n\n",
);

async fn spawn_counting_endpoint() -> (String, Arc<AtomicUsize>) {
    let count = Arc::new(AtomicUsize::new(0));
    let handler_count = count.clone();
    let app = Router::new().route(
        "/v1/chat/completions",
        post(move || {
            let handler_count = handler_count.clone();
            async move {
                handler_count.fetch_add(1, Ordering::SeqCst);
                ([(header::CONTENT_TYPE, "text/event-stream")], SSE)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{address}"), count)
}

#[tokio::test]
async fn round_robin_resolves_to_real_endpoints_and_keeps_sessions_sticky() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let (first_url, first_count) = spawn_counting_endpoint().await;
            let (second_url, second_count) = spawn_counting_endpoint().await;
            let source = common::prepared_source_from_conversations(
                serde_json::json!([
                    {"session_id":"a","turns":[
                        {"text":"x","timestamp":0,"input_length":1,"output_length":1},
                        {"text":"x","delay":1,"input_length":1,"output_length":1}
                    ]},
                    {"session_id":"b","turns":[
                        {"text":"x","timestamp":0,"input_length":1,"output_length":1},
                        {"text":"x","delay":1,"input_length":1,"output_length":1}
                    ]}
                ]),
                "model",
                1,
            )
            .await;

            let report = common::run_fixed_schedule_online_with_ancillary(
                format!("{first_url},{second_url}"),
                "model".into(),
                source,
                FixedScheduleConfig {
                    auto_offset_timestamps: true,
                    start_offset_ms: None,
                },
                false,
                AncillaryTimingConfig::default(),
                7,
            )
            .await
            .unwrap();

            assert_eq!(report.performance.request_counts.completed_requests, 4);
            assert_eq!(first_count.load(Ordering::SeqCst), 2);
            assert_eq!(second_count.load(Ordering::SeqCst), 2);
        })
        .await;
}

#[tokio::test]
async fn post_send_disconnect_is_reported_as_a_canceled_terminal() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            #[derive(Clone)]
            struct BodyState {
                body: Arc<std::sync::Mutex<Option<Vec<u8>>>>,
                received: Arc<tokio::sync::Notify>,
            }
            async fn delayed_after_full_body(
                State(state): State<BodyState>,
                body: Bytes,
            ) -> impl IntoResponse {
                *state.body.lock().unwrap() = Some(body.to_vec());
                state.received.notify_one();
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                ([(header::CONTENT_TYPE, "text/event-stream")], SSE).into_response()
            }
            let state = BodyState {
                body: Arc::new(std::sync::Mutex::new(None)),
                received: Arc::new(tokio::sync::Notify::new()),
            };
            let app = Router::new()
                .route("/v1/chat/completions", post(delayed_after_full_body))
                .with_state(state.clone());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

            let clock: Rc<dyn Clock> = RealClock::new();
            let start_ns = clock.now_ns();
            let sink = TransportSink::new(
                clock.clone(),
                start_ns,
                &format!("http://{address}"),
                "model",
                false,
            );
            let observer = CollectorObserver::new(true);
            let uuid = Uuid::new_v4();
            observer.on_arrival(uuid, 0.0, 1, 1);
            let result = sink
                .dispatch_collect_with_hooks(
                    Request {
                        uuid,
                        input_length: 1,
                        max_output_tokens: 1,
                        prompt_text: Some("complete request".into()),
                        request_body: None,
                        request_body_bytes: None,
                        headers: std::collections::BTreeMap::new(),
                        parameters: std::collections::BTreeMap::new(),
                        endpoint_path: None,
                        streaming: true,
                        x_correlation_id: None,
                        is_final_turn: true,
                        cancel_after_ns: Some(0),
                        url_index: None,
                    },
                    &observer,
                    |_| {},
                )
                .await
                .unwrap();

            assert_eq!(result.terminal, ReplayTerminalStatus::Canceled);
            if state.body.lock().unwrap().is_none() {
                tokio::time::timeout(std::time::Duration::from_secs(1), state.received.notified())
                    .await
                    .expect("server must receive the complete request before disconnect");
            }
            let body = state.body.lock().unwrap().clone().unwrap();
            let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(body["messages"][0]["content"], "complete request");
            let report = observer.finish((clock.now_ns() - start_ns) as f64 / 1_000_000.0);
            assert_eq!(report.per_request.len(), 1);
            assert_eq!(
                report.per_request[0].terminal_status,
                ReplayTerminalStatus::Canceled
            );
        })
        .await;
}

#[tokio::test]
async fn positive_disconnect_delay_is_measured_from_send_completion() {
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let app = Router::new().route(
                "/v1/chat/completions",
                post(|_body: Bytes| async {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    ([(header::CONTENT_TYPE, "text/event-stream")], SSE)
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

            let clock: Rc<dyn Clock> = RealClock::new();
            let transport = HttpTransport::new(clock, ClientConfig::default());
            let config = RequestConfig::new(format!("http://{address}/v1/chat/completions"))
                .cancel_after_ns(50_000_000);
            let record = transport
                .send_request(
                    &config,
                    serde_json::json!({
                        "model": "model",
                        "stream": true,
                        "messages": [{"role": "user", "content": "complete request"}]
                    }),
                    true,
                    |_| {},
                )
                .await;

            let error = record.error.as_ref().unwrap();
            assert_eq!(error.kind, ErrorKind::Cancelled);
            assert_eq!(error.code, Some(499));
            assert!(error.message.contains("RequestCancellationError"));
            let sent_ns = record.trace.as_ref().unwrap().request_send_end_ns.unwrap();
            assert!(record.cancellation_ns.unwrap() >= sent_ns.saturating_add(50_000_000));
        })
        .await;
}
