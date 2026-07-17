// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod common;
use common::{MockServer, run_local};
use std::rc::Rc;

use aiperf_runtime::transport::core::{ConnectionReuseStrategy, ErrorKind};
use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::RequestConfig;
use aiperf_runtime::transport::http::transport::http_transport::HttpTransport;

fn payload() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt2", "stream": true,
        "max_tokens": 8, "messages": [{"role":"user","content":"hi"}]
    })
}

fn non_streaming_payload() -> serde_json::Value {
    serde_json::json!({
        "model": "gpt2", "stream": false,
        "max_tokens": 8, "messages": [{"role":"user","content":"hi"}]
    })
}

#[test]
fn facade_streams_a_chat_completion() {
    run_local(async {
        let Some(mock) = MockServer::spawn(&[]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let t = HttpTransport::new(clock, ClientConfig::default());
        let cfg = RequestConfig::new(format!("{}/v1/chat/completions", mock.base_url));
        let mut ttft = None;
        let rec = t
            .send_request(&cfg, payload(), true, |x| ttft = Some(x))
            .await;
        assert!(!rec.has_error(), "unexpected error: {:?}", rec.error);
        assert_eq!(rec.status, Some(200));
        assert!(!rec.responses.is_empty());
        assert!(ttft.is_some());
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&rec.request_body).unwrap(),
            payload()
        );
        assert_eq!(
            rec.request_headers.get("Accept").map(String::as_str),
            Some("text/event-stream")
        );
        assert_eq!(
            rec.request_headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
    });
}

#[test]
fn facade_sticky_reuse_reuses_connection_across_turns() {
    run_local(async {
        let Some(mock) = MockServer::spawn(&[]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let t = HttpTransport::new(clock, ClientConfig::default());
        let url = format!("{}/v1/chat/completions", mock.base_url);

        let cfg1 = RequestConfig::new(&url)
            .reuse(ConnectionReuseStrategy::StickyUserSessions)
            .correlation_id("session-A")
            .final_turn(false);
        let rec1 = t
            .send_request(&cfg1, non_streaming_payload(), false, |_| {})
            .await;
        assert_eq!(rec1.status, Some(200));
        let tr1 = rec1.trace.unwrap();
        assert!(
            tr1.connection_reused_ns.is_none(),
            "first turn is a fresh connect"
        );
        assert!(tr1.tcp_connect_start_ns.is_some());

        let cfg2 = RequestConfig::new(&url)
            .reuse(ConnectionReuseStrategy::StickyUserSessions)
            .correlation_id("session-A")
            .final_turn(true);
        let rec2 = t
            .send_request(&cfg2, non_streaming_payload(), false, |_| {})
            .await;
        assert_eq!(rec2.status, Some(200));
        let tr2 = rec2.trace.unwrap();
        assert!(
            tr2.connection_reused_ns.is_some(),
            "second turn should reuse the session conn"
        );
        assert!(
            tr2.tcp_connect_start_ns.is_none(),
            "reuse must not open a new socket"
        );

        let cfg3 = RequestConfig::new(&url)
            .reuse(ConnectionReuseStrategy::StickyUserSessions)
            .correlation_id("session-A")
            .final_turn(false);
        let rec3 = t
            .send_request(&cfg3, non_streaming_payload(), false, |_| {})
            .await;
        assert_eq!(rec3.status, Some(200));
        let tr3 = rec3.trace.unwrap();
        assert!(
            tr3.connection_reused_ns.is_none(),
            "post-release turn reconnects"
        );
        assert!(tr3.tcp_connect_start_ns.is_some());
    });
}

#[test]
fn total_timeout_bounds_connect_send_and_response_as_one_request() {
    run_local(async {
        let Some(mock) = MockServer::spawn(&["--ttft", "500"]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let transport = HttpTransport::new(
            clock,
            ClientConfig {
                total_timeout_ns: Some(20_000_000),
                ..ClientConfig::default()
            },
        );
        let config = RequestConfig::new(format!("{}/v1/chat/completions", mock.base_url));

        let record = transport
            .send_request(&config, payload(), true, |_| {})
            .await;

        let error = record
            .error
            .expect("slow response must hit the total timeout");
        assert_eq!(error.kind, ErrorKind::Timeout);
        assert_eq!(error.message, "request timeout after 20000000ns");
        assert_eq!(record.status, Some(200));
    });
}
