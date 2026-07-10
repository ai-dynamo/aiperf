// crates/aiperf-transport/tests/facade.rs
mod common;
use common::MockServer;
use std::rc::Rc;

use aiperf_transport::RealClock;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::{ConnectionReuseStrategy, RequestConfig};
use aiperf_transport::transport::http_transport::HttpTransport;

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
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        let Some(mock) = MockServer::spawn(&[]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_transport::Clock> = RealClock::new();
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
    });
}

#[test]
fn facade_sticky_reuse_reuses_connection_across_turns() {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        let Some(mock) = MockServer::spawn(&[]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_transport::Clock> = RealClock::new();
        let t = HttpTransport::new(clock, ClientConfig::default());
        let url = format!("{}/v1/chat/completions", mock.base_url);

        // Turn 1 (non-final): sticky session opens a fresh connection.
        let cfg1 = RequestConfig::new(&url)
            .reuse(ConnectionReuseStrategy::StickyUserSessions)
            .correlation_id("session-A")
            .final_turn(false);
        let rec1 = t.send_request(&cfg1, non_streaming_payload(), false, |_| {}).await;
        assert_eq!(rec1.status, Some(200));
        let tr1 = rec1.trace.unwrap();
        assert!(tr1.connection_reused_ns.is_none(), "first turn is a fresh connect");
        assert!(tr1.tcp_connect_start_ns.is_some());

        // Turn 2 (final): same session reuses the pooled connection, then releases.
        let cfg2 = RequestConfig::new(&url)
            .reuse(ConnectionReuseStrategy::StickyUserSessions)
            .correlation_id("session-A")
            .final_turn(true);
        let rec2 = t.send_request(&cfg2, non_streaming_payload(), false, |_| {}).await;
        assert_eq!(rec2.status, Some(200));
        let tr2 = rec2.trace.unwrap();
        assert!(tr2.connection_reused_ns.is_some(), "second turn should reuse the session conn");
        assert!(tr2.tcp_connect_start_ns.is_none(), "reuse must not open a new socket");

        // Turn 3: after the final turn released the lease, a new sticky request
        // for the same session opens a fresh connection again.
        let cfg3 = RequestConfig::new(&url)
            .reuse(ConnectionReuseStrategy::StickyUserSessions)
            .correlation_id("session-A")
            .final_turn(false);
        let rec3 = t.send_request(&cfg3, non_streaming_payload(), false, |_| {}).await;
        assert_eq!(rec3.status, Some(200));
        let tr3 = rec3.trace.unwrap();
        assert!(tr3.connection_reused_ns.is_none(), "post-release turn reconnects");
        assert!(tr3.tcp_connect_start_ns.is_some());
    });
}
