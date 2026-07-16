// rust/transport-http/tests/request.rs
mod common;
use common::{MockServer, chat_body, run_local};

use std::collections::BTreeMap;
use std::rc::Rc;

use aiperf_runtime::transport::core::Response;
use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::client::http_client::HttpClient;
use aiperf_runtime::transport::http::config::ClientConfig;
use bytes::Bytes;

#[test]
fn streaming_chat_records_tokens_ttft_and_usage() {
    run_local(async {
        let Some(mock) = MockServer::spawn(&[]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let client = HttpClient::new(clock, ClientConfig::default());
        let url = url::Url::parse(&format!("{}/v1/chat/completions", mock.base_url)).unwrap();
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        headers.insert("Accept".into(), "text/event-stream".into());

        let mut ttft: Option<i64> = None;
        let rec = client
            .request(&url, &headers, chat_body("gpt2"), true, |t| ttft = Some(t))
            .await;

        assert!(!rec.has_error(), "unexpected error: {:?}", rec.error);
        assert_eq!(rec.status, Some(200));
        assert!(rec.responses.iter().any(|r| matches!(r, Response::Sse(_))));
        assert!(ttft.is_some(), "first-token callback should fire");
        let t = rec.trace.unwrap();
        assert!(t.response_receive_start_ns.is_some());
        assert!(t.waiting().is_some());
        assert!(t.response_chunks_count > 0);
        assert!(t.response_bytes_total > 0);
        assert!(t.request_send_start_ns <= t.dns_cache_miss_ns);
        assert!(t.request_headers_sent_ns.is_some());
        assert!(t.request_headers_sent_ns <= t.request_send_end_ns);
        assert!(t.request_send_end_ns <= t.response_headers_received_ns);
    });
}

#[test]
fn non_streaming_models_endpoint_returns_text_json() {
    run_local(async {
        let Some(mock) = MockServer::spawn(&[]).await else {
            return;
        };
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let client = HttpClient::new(clock, ClientConfig::default());
        // A non-streaming completion request returns a single JSON body.
        let url = url::Url::parse(&format!("{}/v1/chat/completions", mock.base_url)).unwrap();
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".into(), "application/json".into());
        headers.insert("Accept".into(), "application/json".into());
        let body = Bytes::from(
            serde_json::to_vec(&serde_json::json!({
                "model": "gpt2",
                "stream": false,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "hello"}],
            }))
            .unwrap(),
        );
        let rec = client.request(&url, &headers, body, false, |_| {}).await;
        assert!(!rec.has_error(), "unexpected error: {:?}", rec.error);
        assert_eq!(rec.status, Some(200));
        match rec.responses.first() {
            Some(Response::Text(t)) => {
                assert!(t.json().is_some(), "body should be JSON: {}", t.text);
            }
            other => panic!("expected a Text response, got {other:?}"),
        }
    });
}
