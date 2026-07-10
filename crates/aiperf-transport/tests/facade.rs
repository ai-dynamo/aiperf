// crates/aiperf-transport/tests/facade.rs
mod common;
use common::MockServer;
use std::rc::Rc;

use aiperf_transport::RealClock;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::RequestConfig;
use aiperf_transport::transport::http_transport::HttpTransport;

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
        let payload = serde_json::json!({
            "model": "gpt2", "stream": true,
            "max_tokens": 8, "messages": [{"role":"user","content":"hi"}]
        });
        let mut ttft = None;
        let rec = t
            .send_request(&cfg, payload, true, |x| ttft = Some(x))
            .await;
        assert!(!rec.has_error(), "unexpected error: {:?}", rec.error);
        assert_eq!(rec.status, Some(200));
        assert!(!rec.responses.is_empty());
        assert!(ttft.is_some());
    });
}
