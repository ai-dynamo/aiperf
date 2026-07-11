// crates/aiperf-transport/tests/reuse.rs
mod common;
use common::MockServer;

use std::rc::Rc;

use aiperf_transport::RealClock;
use aiperf_transport::client::connection::{Sender, TimedBody};
use aiperf_transport::client::pool::ConnectionPool;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::{ConnectionReuseStrategy, TraceData};
use bytes::Bytes;
use http_body_util::BodyExt;

/// Send a real non-streaming chat request over `sender`, drain the body, and
/// return the HTTP status. Proves the connection actually carries traffic.
async fn send_chat(sender: &mut Sender, base: &str) -> u16 {
    let url = url::Url::parse(base).unwrap();
    let authority = url.authority();
    let payload = serde_json::json!({
        "model": "gpt2", "stream": false, "max_tokens": 4,
        "messages": [{"role":"user","content":"hi"}]
    });
    let bytes = Bytes::from(serde_json::to_vec(&payload).unwrap());
    let clock: Rc<dyn aiperf_transport::Clock> = RealClock::new();
    let sent = Rc::new(std::cell::Cell::new(None));
    let req = hyper::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(hyper::header::HOST, authority)
        .header("Content-Type", "application/json")
        .header("Accept", "application/json")
        .body(TimedBody::new(bytes, clock, sent))
        .unwrap();
    let resp = sender.send(req).await.expect("send over pooled connection");
    let status = resp.status().as_u16();
    let _ = resp.into_body().collect().await; // drain so the connection is reusable
    status
}

#[test]
fn never_uses_new_port_each_time() {
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
        let url = url::Url::parse(&mock.base_url).unwrap();
        let cfg = ClientConfig::default();
        let pool = ConnectionPool::new();

        let mut t1 = TraceData::default();
        let _ = pool
            .acquire(
                &url,
                &cfg,
                clock.clone(),
                ConnectionReuseStrategy::Never,
                None,
                &mut t1,
            )
            .await
            .unwrap();
        let mut t2 = TraceData::default();
        let _ = pool
            .acquire(
                &url,
                &cfg,
                clock.clone(),
                ConnectionReuseStrategy::Never,
                None,
                &mut t2,
            )
            .await
            .unwrap();

        assert!(t1.local_port.is_some() && t2.local_port.is_some());
        assert_ne!(
            t1.local_port, t2.local_port,
            "Never should use a fresh port each time"
        );
    });
}

#[test]
fn pooled_reuses_connection_and_records_reuse() {
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
        let url = url::Url::parse(&mock.base_url).unwrap();
        let cfg = ClientConfig::default();
        let pool = ConnectionPool::new();

        // First acquire establishes a real connection; send a real request over it.
        let mut t1 = TraceData::default();
        let mut s1 = pool
            .acquire(
                &url,
                &cfg,
                clock.clone(),
                ConnectionReuseStrategy::Pooled,
                None,
                &mut t1,
            )
            .await
            .unwrap();
        let first_port = t1.local_port;
        assert!(first_port.is_some());
        assert!(
            t1.connection_reused_ns.is_none(),
            "first acquire is a fresh connect"
        );
        assert_eq!(send_chat(&mut s1, &mock.base_url).await, 200);

        // Return it to the pool, then re-acquire: should reuse (no new connect).
        pool.put(&url, None, ConnectionReuseStrategy::Pooled, s1);
        let mut t2 = TraceData::default();
        let mut s2 = pool
            .acquire(
                &url,
                &cfg,
                clock.clone(),
                ConnectionReuseStrategy::Pooled,
                None,
                &mut t2,
            )
            .await
            .unwrap();
        assert!(
            t2.connection_reused_ns.is_some(),
            "second acquire should reuse the pooled conn"
        );
        assert!(
            t2.tcp_connect_start_ns.is_none(),
            "reuse must not open a new socket"
        );
        // The reused connection still carries traffic to the same server.
        assert_eq!(send_chat(&mut s2, &mock.base_url).await, 200);
    });
}
