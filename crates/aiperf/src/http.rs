// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online HTTP dispatch over the `aiperf-transport` (hyper) client.
//!
//! [`TransportSink`] implements `loadgen_core`'s [`RequestSink`] using the
//! Rust-native `aiperf-transport` client (hyper + the `aiperf-clock` `Clock`),
//! replacing the former reqwest sink. It is single-threaded (`!Send`, `Rc`-based)
//! and driven on a `LocalSet`; admit/token times are stamped from the same clock
//! origin the run loop uses for arrival, so all events share one timeline.

use std::rc::Rc;

use anyhow::Result;
use async_trait::async_trait;
use uuid::Uuid;

use aiperf_clock::Clock;
use aiperf_core::chat::chat_request_body;
use aiperf_core::sse::ChatChunk;
use aiperf_transport::config::ClientConfig;
use aiperf_transport::models::{HttpVersion, RequestConfig, Response};
use aiperf_transport::transport::http_transport::HttpTransport;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{Dispatchable, RequestObserver, RequestSink};

/// A slim online HTTP request carrying prompt text. This is the load
/// generator's own request type; implementing [`Dispatchable`] is all the
/// dispatch seam requires.
#[derive(Debug, Clone)]
pub struct HttpRequest {
    /// Stable per-request identifier used to correlate observer events.
    pub uuid: Uuid,
    /// Prompt length in tokens, for measurement accounting.
    pub input_length: usize,
    /// Maximum number of output tokens to request.
    pub max_output_tokens: usize,
    /// Prompt text placed on the wire.
    pub prompt_text: Option<String>,
}

impl Dispatchable for HttpRequest {
    fn uuid(&self) -> Uuid {
        self.uuid
    }
    fn input_length(&self) -> usize {
        self.input_length
    }
    fn max_output_tokens(&self) -> usize {
        self.max_output_tokens
    }
}

/// Live OpenAI-chat sink over [`aiperf_transport`]. Shares the caller's clock and
/// origin (`start_ns`) so admit/token timestamps sit on the same timeline as the
/// run loop's arrival events.
pub struct TransportSink {
    transport: HttpTransport,
    clock: Rc<dyn Clock>,
    url: String,
    model: String,
    start_ns: i64,
}

impl TransportSink {
    /// Build a sink targeting `base_url` for `model`. When `http2` is set,
    /// cleartext HTTP/2 (h2c prior-knowledge) multiplexes many streams over one
    /// connection.
    pub fn new(
        clock: Rc<dyn Clock>,
        start_ns: i64,
        base_url: &str,
        model: impl Into<String>,
        http2: bool,
    ) -> Self {
        let cfg = ClientConfig {
            http_version: if http2 {
                HttpVersion::Http2PriorKnowledge
            } else {
                HttpVersion::Auto
            },
            ..ClientConfig::default()
        };
        let transport = HttpTransport::new(clock.clone(), cfg);
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));
        Self {
            transport,
            clock,
            url,
            model: model.into(),
            start_ns,
        }
    }

    fn ms(&self, ns: i64) -> f64 {
        (ns - self.start_ns) as f64 / 1_000_000.0
    }
}

#[async_trait(?Send)]
impl RequestSink<HttpRequest> for TransportSink {
    async fn dispatch(&self, req: HttpRequest, obs: &dyn RequestObserver) -> Result<()> {
        let uuid = req.uuid;
        // No scheduler admission on the HTTP path; admit == dispatch time.
        let admit_ms = self.ms(self.clock.now_ns());

        let prompt = req.prompt_text.unwrap_or_default();
        let payload = chat_request_body(
            &self.model,
            &[("user", prompt.as_str())],
            req.max_output_tokens,
        );

        let cfg = RequestConfig::new(&self.url);
        let rec = self
            .transport
            .send_request(&cfg, payload, true, |_ttft_ns| {})
            .await;

        // A transport-level failure is surfaced to the caller (which records the
        // terminal Failed status); a completed-but-error response is handled below.
        if let Some(err) = &rec.error {
            return Err(anyhow::anyhow!("transport dispatch error: {err:?}"));
        }

        obs.on_admit(uuid, admit_ms, 0);

        // Parse the collected SSE messages into per-token arrival times, stamped
        // from the transport clock (real inter-token timing).
        let mut done = false;
        for resp in &rec.responses {
            let Response::Sse(msg) = resp else { continue };
            if msg.is_done() {
                done = true;
                continue;
            }
            let Some(data) = msg.data() else { continue };
            let Ok(chunk) = serde_json::from_str::<ChatChunk>(data) else {
                continue;
            };
            if !chunk.delta_text().is_empty() {
                obs.on_token(uuid, self.ms(msg.perf_ns));
            }
        }

        let terminal = if done && rec.status == Some(200) {
            ReplayTerminalStatus::Completed
        } else {
            ReplayTerminalStatus::Failed
        };
        obs.on_terminal(uuid, terminal);
        Ok(())
    }
}
