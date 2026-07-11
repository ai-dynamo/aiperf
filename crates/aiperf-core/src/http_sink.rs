// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! [`HttpSink`]: streams real chat completions over HTTP, reports measurement
//! events into the shared `TraceCollector` via [`RequestObserver`], and returns
//! the assistant response text (for multi-turn conversation assembly).
//!
//! Dispatches a slim [`HttpRequest`] (which implements `loadgen_core`'s
//! [`Dispatchable`]) through the transport-neutral [`RequestSink`] trait — no
//! engine/sim types cross this boundary.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use uuid::Uuid;

use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{Dispatchable, RequestObserver, RequestSink};

use crate::sse::{SseEvent, parse_sse_line};
use crate::wire::{WireEntry, WireTraceSink};

/// A slim, transport-native request for the HTTP path.
///
/// This is the load generator's own request type — implementing
/// [`Dispatchable`] is all the seam requires. The simulated engine uses a
/// different type (the mocker's `DirectRequest`); neither leaks into the other.
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

/// One chat message in an OpenAI `messages` array.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Role, e.g. `"user"` or `"assistant"`.
    pub role: String,
    /// Message content text.
    pub content: String,
}

impl ChatMessage {
    /// A `user` message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
        }
    }
    /// An `assistant` message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
        }
    }
}

/// Transport configuration for the HTTP client.
#[derive(Debug, Clone, Default)]
pub struct HttpConfig {
    /// Per-request timeout.
    pub timeout: Option<Duration>,
    /// Bearer API key, sent as `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// Extra request headers as `(name, value)` pairs.
    pub headers: Vec<(String, String)>,
    /// Send `ignore_eos` (vLLM extension) so the model generates up to
    /// `max_tokens` instead of stopping at EOS — deterministic output length.
    pub ignore_eos: bool,
}

/// Per-request measurements captured during streaming with no shared lock. The
/// caller submits these to the collector in one batch, so hot-path token events
/// never contend on the global collector mutex.
#[derive(Debug, Clone)]
pub struct StreamOutcome {
    /// Assistant response text (for multi-turn assembly).
    pub content: String,
    /// Dispatch/admit time in milliseconds.
    pub admit_ms: f64,
    /// Per-output-token arrival times in milliseconds.
    pub token_times: Vec<f64>,
    /// Terminal status.
    pub terminal: ReplayTerminalStatus,
    /// Server-reported input tokens (`usage.prompt_tokens`), if `include_usage`.
    /// Authoritative ISL (includes chat-template overhead).
    pub prompt_tokens: Option<u32>,
    /// Server-reported output tokens (`usage.completion_tokens`) — authoritative
    /// output count, independent of how tokens were chunked into SSE events.
    pub completion_tokens: Option<u32>,
    /// Finish reason of the last choice, e.g. `"stop"` (good) or `"length"`
    /// (truncated at `max_tokens`).
    pub finish_reason: Option<String>,
}

/// Streams `/v1/chat/completions` requests and emits measurement events.
///
/// The `start` instant is shared with the run harness so all observer
/// timestamps (arrival, admit, tokens) are on one monotonic clock.
pub struct HttpSink {
    base_url: String,
    model: String,
    client: reqwest::Client,
    start: Instant,
    wire: Option<Arc<dyn WireTraceSink>>,
    ignore_eos: bool,
}

/// True when `GRAPH_HTTP2` selects cleartext HTTP/2 (h2c prior knowledge).
/// HTTP/1.1 pins one connection (one ephemeral source port) per in-flight
/// request, so concurrency is capped at ~64k per source IP -> endpoint. HTTP/2
/// multiplexes many streams over each connection, decoupling concurrency from
/// the source-port space.
fn http2_enabled() -> bool {
    std::env::var("GRAPH_HTTP2")
        .map(|v| v != "0" && !v.is_empty())
        .unwrap_or(false)
}

/// Build the default reqwest client, opting into h2c prior knowledge when
/// `GRAPH_HTTP2` is set.
fn build_client() -> reqwest::Client {
    // `no_proxy()` so localhost/loopback benchmarking is never routed through an
    // ambient HTTP_PROXY (which would 405/refuse and tank throughput).
    let mut b = reqwest::Client::builder().no_proxy();
    if http2_enabled() {
        b = b.http2_prior_knowledge();
    }
    b.build().unwrap_or_else(|_| reqwest::Client::new())
}

impl HttpSink {
    /// Create a sink targeting `base_url` for `model`, timestamping against `start`.
    pub fn new(base_url: String, model: String, start: Instant) -> Self {
        Self {
            base_url,
            model,
            client: build_client(),
            start,
            wire: None,
            ignore_eos: false,
        }
    }

    /// Attach a wire-trace sink that captures each raw request/response pair.
    pub fn with_wire(mut self, sink: Arc<dyn WireTraceSink>) -> Self {
        self.wire = Some(sink);
        self
    }

    /// Rebuild the HTTP client with transport config (timeout, auth, headers).
    pub fn with_http_config(mut self, cfg: &HttpConfig) -> Result<Self> {
        // Match `build_client()`: `no_proxy()` so loopback benchmarking is never
        // routed through an ambient HTTP_PROXY.
        let mut builder = reqwest::Client::builder().no_proxy();
        if http2_enabled() {
            builder = builder.http2_prior_knowledge();
        }
        if let Some(timeout) = cfg.timeout {
            builder = builder.timeout(timeout);
        }
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = &cfg.api_key {
            let value = format!("Bearer {key}");
            headers.insert(
                reqwest::header::AUTHORIZATION,
                value
                    .parse()
                    .context("invalid api key for Authorization header")?,
            );
        }
        for (name, value) in &cfg.headers {
            let header_name = reqwest::header::HeaderName::from_bytes(name.as_bytes())
                .with_context(|| format!("invalid header name {name}"))?;
            let header_value = value
                .parse()
                .with_context(|| format!("invalid header value for {name}"))?;
            headers.insert(header_name, header_value);
        }
        if !headers.is_empty() {
            builder = builder.default_headers(headers);
        }
        self.client = builder.build().context("building HTTP client")?;
        self.ignore_eos = cfg.ignore_eos;
        Ok(self)
    }

    fn now_ms(&self) -> f64 {
        crate::elapsed_ms(self.start)
    }

    /// Stream one chat request built from `messages`, accumulating measurements
    /// locally (no shared lock) and returning them as a [`StreamOutcome`].
    pub async fn stream_chat(
        &self,
        uuid: Uuid,
        messages: &[ChatMessage],
        max_tokens: usize,
    ) -> Result<StreamOutcome> {
        self.stream_chat_cb(uuid, messages, max_tokens, || {}).await
    }

    /// Like [`stream_chat`](Self::stream_chat), but invokes `on_first_token` the
    /// moment the first content token arrives — before the reply completes.
    pub async fn stream_chat_cb(
        &self,
        uuid: Uuid,
        messages: &[ChatMessage],
        max_tokens: usize,
        mut on_first_token: impl FnMut(),
    ) -> Result<StreamOutcome> {
        // No scheduler admission on the HTTP path; admit == dispatch time.
        let admit_ms = self.now_ms();

        let mut body = serde_json::json!({
            "model": self.model,
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": max_tokens,
            "messages": messages
                .iter()
                .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                .collect::<Vec<_>>(),
        });
        if self.ignore_eos {
            // vLLM extension: don't stop at EOS, so the model generates up to
            // `max_tokens` for deterministic output length in controlled runs.
            body["ignore_eos"] = serde_json::json!(true);
        }

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&body)
            .send()
            .await?;
        let status = resp.status().as_u16();
        // On a non-2xx status, `error_for_status()` would drop the body — read it
        // and attach it so the caller sees the server's error message.
        if let Err(status_err) = resp.error_for_status_ref() {
            let body = resp.text().await.unwrap_or_default();
            return Err(anyhow::Error::new(status_err)
                .context(format!("HTTP {status} response body: {body}")));
        }

        let capture_wire = self.wire.is_some();
        let mut stream = resp.bytes_stream();
        // Raw network chunks are buffered as bytes and split into SSE lines on the
        // byte buffer; only COMPLETE lines are UTF-8 decoded, so a multibyte
        // sequence straddling a TCP chunk boundary is never corrupted.
        let mut buf: Vec<u8> = Vec::new();
        let mut raw: Vec<u8> = Vec::new();
        let mut content = String::new();
        let mut token_times: Vec<f64> = Vec::new();
        let mut terminal = false;
        let mut prompt_tokens: Option<u32> = None;
        let mut completion_tokens: Option<u32> = None;
        let mut finish_reason: Option<String> = None;
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            if capture_wire {
                raw.extend_from_slice(&bytes);
            }
            buf.extend_from_slice(&bytes);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=nl).collect();
                let line = String::from_utf8_lossy(&line_bytes);
                match parse_sse_line(&line) {
                    SseEvent::Data(chunk) => {
                        let delta = chunk.delta_text();
                        if !delta.is_empty() {
                            if token_times.is_empty() {
                                on_first_token();
                            }
                            token_times.push(self.now_ms());
                            content.push_str(&delta);
                        }
                        // Authoritative token counts from the usage chunk
                        // (stream_options.include_usage); the final chunk carries
                        // usage with empty choices.
                        if let Some(usage) = &chunk.usage {
                            prompt_tokens = Some(usage.prompt_tokens);
                            completion_tokens = Some(usage.completion_tokens);
                        }
                        if let Some(reason) =
                            chunk.choices.first().and_then(|c| c.finish_reason.clone())
                        {
                            finish_reason = Some(reason);
                        }
                    }
                    SseEvent::Done => terminal = true,
                    SseEvent::Other => {}
                }
            }
        }
        if let Some(wire) = &self.wire {
            wire.record(WireEntry {
                uuid: uuid.to_string(),
                status,
                request: body,
                response: String::from_utf8_lossy(&raw).into_owned(),
            });
        }
        Ok(StreamOutcome {
            content,
            admit_ms,
            token_times,
            terminal: if terminal {
                ReplayTerminalStatus::Completed
            } else {
                ReplayTerminalStatus::Failed
            },
            prompt_tokens,
            completion_tokens,
            finish_reason,
        })
    }
}

#[async_trait]
impl RequestSink<HttpRequest> for HttpSink {
    async fn dispatch(&self, req: HttpRequest, obs: &dyn RequestObserver) -> Result<()> {
        let uuid = req.uuid;
        let messages = [ChatMessage::user(req.prompt_text.unwrap_or_default())];
        let outcome = self
            .stream_chat(uuid, &messages, req.max_output_tokens)
            .await?;
        obs.on_admit(uuid, outcome.admit_ms, 0);
        for at in &outcome.token_times {
            obs.on_token(uuid, *at);
        }
        obs.on_terminal(uuid, outcome.terminal);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Counter {
        tokens: Mutex<usize>,
        done: Mutex<bool>,
    }
    impl RequestObserver for Counter {
        fn on_arrival(&self, _: Uuid, _: f64, _: usize, _: usize) {}
        fn on_admit(&self, _: Uuid, _: f64, _: usize) {}
        fn on_token(&self, _: Uuid, _: f64) {
            *self.tokens.lock().unwrap() += 1;
        }
        fn on_terminal(&self, _: Uuid, s: ReplayTerminalStatus) {
            if matches!(s, ReplayTerminalStatus::Completed) {
                *self.done.lock().unwrap() = true;
            }
        }
    }

    #[tokio::test]
    async fn http_sink_streams_two_tokens_then_done() {
        let base = crate::test_util::spawn_mock().await;
        let sink = HttpSink::new(base, "m".into(), Instant::now());
        let obs = Counter::default();
        let req = HttpRequest {
            uuid: Uuid::new_v4(),
            input_length: 4,
            max_output_tokens: 8,
            prompt_text: Some("hi there".into()),
        };
        sink.dispatch(req, &obs).await.unwrap();
        assert_eq!(*obs.tokens.lock().unwrap(), 2);
        assert!(*obs.done.lock().unwrap());
    }

    #[tokio::test]
    async fn stream_chat_returns_assistant_content() {
        let base = crate::test_util::spawn_mock().await;
        let sink = HttpSink::new(base, "m".into(), Instant::now());
        let outcome = sink
            .stream_chat(Uuid::new_v4(), &[ChatMessage::user("hi")], 8)
            .await
            .unwrap();
        // mock streams content deltas "a" then "b".
        assert_eq!(outcome.content, "ab");
        assert_eq!(outcome.token_times.len(), 2);
    }

    #[tokio::test]
    async fn stream_chat_cb_fires_first_token_once() {
        let base = crate::test_util::spawn_mock().await;
        let sink = HttpSink::new(base, "m".into(), Instant::now());
        let mut fires = 0usize;
        let outcome = sink
            .stream_chat_cb(Uuid::new_v4(), &[ChatMessage::user("hi")], 8, || fires += 1)
            .await
            .unwrap();
        // Mock streams a role-only chunk, then content "a"/"b", then finish.
        // The callback must fire exactly once, at the first content token.
        assert_eq!(outcome.content, "ab");
        assert_eq!(fires, 1);
    }
}
