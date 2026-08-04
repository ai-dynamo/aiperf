// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-HTTP tests for ancillary cancellation and multi-endpoint policy.

use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use aiperf_runtime::ancillary::AncillaryTimingConfig;
use aiperf_runtime::clock::{Clock, RealClock};
use aiperf_runtime::fixed_schedule::FixedScheduleConfig;
use aiperf_runtime::transport::core::ErrorKind;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::RequestConfig;
use aiperf_runtime::transport::http::transport::http_transport::HttpTransport;
use axum::{Router, body::Bytes, http::header, routing::post};

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
