// crates/aiperf-transport/tests/cancel.rs
mod common;
use common::MockServer;
use std::collections::BTreeMap;
use std::rc::Rc;

use aiperf_transport::RealClock;
use aiperf_transport::client::http_client::HttpClient;
use aiperf_transport::config::ClientConfig;
use bytes::Bytes;

#[test]
fn cancel_after_send_marks_record_cancelled() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        // High ITL so the stream stays open long enough to cancel mid-flight.
        let Some(mock) = MockServer::spawn(&["--ttft", "10", "--itl", "200"]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_transport::Clock> = RealClock::new();
        let client = HttpClient::new(clock, ClientConfig::default());
        let url = url::Url::parse(&format!("{}/v1/chat/completions", mock.base_url)).unwrap();
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        headers.insert("Accept".into(), "text/event-stream".into());
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt2", "stream": true, "max_tokens": 200,
                "messages": [{"role":"user","content":"hi"}]
            }))
            .unwrap(),
        );

        // Cancel 50ms after send; with 200ms ITL the stream is still open.
        let rec = client
            .request_cancellable(&url, &headers, body, true, 50_000_000, |_| {})
            .await;
        assert!(rec.was_cancelled(), "record should be cancelled");
        assert_eq!(rec.error.as_ref().unwrap().code, Some(499));
    });
}
