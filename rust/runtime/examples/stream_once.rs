// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal driver: one streaming chat request against a server, printing the
//! record. Usage: aiperf-mock-server --no-tokenizer & then run with BASE_URL set.

use std::rc::Rc;

use aiperf_runtime::transport::http::RealClock;
use aiperf_runtime::transport::http::config::ClientConfig;
use aiperf_runtime::transport::http::models::RequestConfig;
use aiperf_runtime::transport::http::transport::http_transport::HttpTransport;

fn main() {
    let base = std::env::var("BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let local = tokio::task::LocalSet::new();
    local.block_on(&rt, async {
        let clock: Rc<dyn aiperf_runtime::transport::http::Clock> = RealClock::new();
        let t = HttpTransport::new(clock, ClientConfig::default());
        let cfg = RequestConfig::new(format!("{base}/v1/chat/completions"));
        let payload = serde_json::json!({
            "model": "gpt2", "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        });
        let mut ttft = None;
        let rec = t
            .send_request(&cfg, payload, true, |x| ttft = Some(x))
            .await;
        println!(
            "status={:?} responses={} ttft_ns={:?} error={:?}",
            rec.status,
            rec.responses.len(),
            ttft,
            rec.error
        );
    });
}
