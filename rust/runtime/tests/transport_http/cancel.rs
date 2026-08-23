// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "engine")]

mod common;

use std::rc::Rc;

use aiperf_runtime::transport::core::{ErrorKind, RequestRecord};
use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::RequestConfig;
use aiperf_runtime::transport::http::transport::http_transport::HttpTransport;
use common::{MockServer, run_local};

async fn cancelled_request(cancel_after_ns: i64) -> Option<RequestRecord> {
    // Keep the SSE stream open so both the zero- and positive-delay timers win.
    let mock = MockServer::spawn(&["--ttft", "500", "--itl", "500"]).await?;
    let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
    let transport = HttpTransport::new(clock, ClientConfig::default());
    let config = RequestConfig::new(format!("{}/v1/chat/completions", mock.base_url))
        .cancel_after_ns(cancel_after_ns);
    let payload = serde_json::json!({
        "model": "gpt2",
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "complete body must arrive"}]
    });
    Some(transport.send_request(&config, payload, true, |_| {}).await)
}

fn assert_post_send_cancelled(record: &RequestRecord, minimum_delay_ns: i64) {
    assert!(
        record.was_cancelled(),
        "record should be cancelled: {record:?}"
    );
    let error = record.error.as_ref().expect("cancellation error");
    assert_eq!(error.kind, ErrorKind::Cancelled);
    assert_eq!(error.code, Some(499));
    assert!(error.message.contains("RequestCancellationError"));

    let sent_ns = record
        .trace
        .as_ref()
        .and_then(|trace| trace.request_send_end_ns)
        .expect("send-complete timestamp must survive cancellation");
    let cancelled_ns = record.cancellation_ns.unwrap();
    assert!(cancelled_ns >= sent_ns.saturating_add(minimum_delay_ns));
}

#[test]
fn positive_delay_is_measured_after_send_completion() {
    run_local(async {
        let Some(record) = cancelled_request(50_000_000).await else {
            return;
        };
        assert_post_send_cancelled(&record, 50_000_000);
    });
}

#[test]
fn zero_delay_still_waits_until_the_complete_request_is_sent() {
    run_local(async {
        let Some(record) = cancelled_request(0).await else {
            return;
        };
        assert_post_send_cancelled(&record, 0);
    });
}
