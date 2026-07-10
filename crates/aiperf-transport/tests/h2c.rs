// crates/aiperf-transport/tests/h2c.rs
mod common;
use common::MockServer;
use std::collections::BTreeMap;
use std::rc::Rc;

use aiperf_transport::RealClock;
use aiperf_transport::client::http_client::HttpClient;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::HttpVersion;
use bytes::Bytes;

fn body() -> Bytes {
    Bytes::from(
        serde_json::to_vec(&serde_json::json!({
            "model": "gpt2", "stream": true, "max_tokens": 8,
            "messages": [{"role":"user","content":"hi"}]
        }))
        .unwrap(),
    )
}

#[test]
fn h2c_prior_knowledge_completes_streaming() {
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
        let cfg = ClientConfig {
            http_version: HttpVersion::Http2PriorKnowledge,
            ..ClientConfig::default()
        };
        let client = HttpClient::new(clock, cfg);
        let url = url::Url::parse(&format!("{}/v1/chat/completions", mock.base_url)).unwrap();
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        headers.insert("Accept".into(), "text/event-stream".into());

        let rec = client.request(&url, &headers, body(), true, |_| {}).await;
        assert!(!rec.has_error(), "h2c error: {:?}", rec.error);
        assert_eq!(rec.status, Some(200));
        assert!(!rec.responses.is_empty());
    });
}
