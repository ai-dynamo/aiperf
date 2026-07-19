// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! A [`GraphSink`] backed by the Rust-native [`crate::transport::http`] HTTP client
//! (hyper + the `aiperf-clock` `Clock`).
//!
//! This is the graph dataflow's live dispatch path: it streams real OpenAI
//! chat-completions over HTTP to the target server (Dynamo frontend / the
//! `aiperf-mock-server` mock / a real inference server), parses the SSE deltas into
//! assistant text + per-token arrival times, and feeds the shared
//! [`RequestObserver`] the measurement events.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use uuid::Uuid;

use crate::clock::Clock;
use crate::dataset::{Overrides, build_message_body_from_wire_parts};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::RequestObserver;
use crate::transport::core::Response;
use crate::transport::http::config::ClientConfig;
use crate::transport::http::models::{HttpVersion, RequestConfig};
use crate::transport::http::sse::ChatChunk;
use crate::transport::http::transport::http_transport::HttpTransport;

use crate::graph::sink::{GraphDispatchOptions, GraphReply, GraphSink};
use crate::graph::wire::OpenAiChatMessage;

/// Live OpenAI-chat sink over [`crate::transport::http`]. Single-threaded per trace
/// (`Rc`/`!Send`), matching the executor's `?Send` dispatch seam.
pub struct TransportChatSink {
    transport: HttpTransport,
    clock: Rc<dyn Clock>,
    url: String,
    model: String,
    start_ns: i64,
    observer: Rc<dyn RequestObserver>,
    default_max_tokens: usize,
    /// Pre-serialized override tails (the inner bytes of
    /// `{model, stream, stream_options, max_tokens}`, braces stripped), keyed by
    /// the resolved `max_tokens`. Model/stream/include_usage are constant per
    /// sink, so a whole run usually has one distinct tail; each is serialized
    /// once here and cloned per dispatch instead of rebuilt through serde on the
    /// hot path. Interior-mutable because the sink is shared (`Rc`, single
    /// worker thread) across every trace's fire tasks.
    override_tails: RefCell<BTreeMap<u32, Bytes>>,
}

impl TransportChatSink {
    /// Build a sink targeting `base_url` for `model`, timestamping all observer
    /// events against the shared clock origin captured at construction. When
    /// `http2` is set, cleartext HTTP/2 (h2c prior-knowledge) is used so many
    /// streams multiplex over one connection.
    pub fn new(
        clock: Rc<dyn Clock>,
        base_url: &str,
        model: impl Into<String>,
        observer: Rc<dyn RequestObserver>,
        default_max_tokens: usize,
        http2: bool,
    ) -> Self {
        let start_ns = clock.now_ns();
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
            observer,
            default_max_tokens,
            override_tails: RefCell::new(BTreeMap::new()),
        }
    }

    /// Return the pre-serialized override tail for `max_tokens`, building and
    /// caching it on first use. Byte-identical to serializing a fresh
    /// `{model, stream, stream_options: {include_usage}, max_tokens}` object and
    /// stripping its braces.
    fn override_tail(&self, max_tokens: u32) -> Result<Bytes> {
        if let Some(tail) = self.override_tails.borrow().get(&max_tokens) {
            return Ok(tail.clone());
        }
        let mut overrides = Overrides::new();
        overrides.set_model(&self.model);
        overrides.set_stream(true);
        overrides.set_include_usage(true);
        overrides.set_max_tokens("max_tokens", max_tokens);
        let tail = Bytes::from(overrides.inner_bytes()?);
        self.override_tails
            .borrow_mut()
            .insert(max_tokens, tail.clone());
        Ok(tail)
    }

    fn ms(&self, ns: i64) -> f64 {
        (ns - self.start_ns) as f64 / 1_000_000.0
    }
}

#[async_trait(?Send)]
impl GraphSink<OpenAiChatMessage> for TransportChatSink {
    async fn dispatch(
        &self,
        _node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        self.dispatch_with_options(
            _node_id,
            messages,
            max_tokens,
            GraphDispatchOptions::default(),
            on_first_token,
        )
        .await
    }

    async fn dispatch_with_options(
        &self,
        _node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        options: GraphDispatchOptions,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        let uuid = Uuid::new_v4();
        // No scheduler admission on the HTTP path; admit == dispatch time.
        let admit_ms = self.ms(self.clock.now_ns());

        let resolved_max_tokens =
            u32::try_from(max_tokens.unwrap_or(self.default_max_tokens)).unwrap_or(u32::MAX);
        let override_tail = self.override_tail(resolved_max_tokens)?;
        let payload = build_message_body_from_wire_parts(&messages, &override_tail);

        let mut cfg = RequestConfig::new(&self.url);
        cfg.cancel_after_ns = options.cancel_after_ns;
        // The transport fires this at the first SSE message (first observed
        // token). Gate first-token-anchored successors the moment it arrives,
        // before the reply completes.
        let rec = self
            .transport
            .send_request_bytes(&cfg, payload, true, |_ttft_ns| on_first_token())
            .await;

        self.observer.on_admit(uuid, admit_ms, 0);

        if let Some(err) = &rec.error {
            tracing::debug!("transport dispatch error: {:?}", err);
            let terminal = if rec.was_cancelled() {
                ReplayTerminalStatus::Canceled
            } else {
                ReplayTerminalStatus::Failed
            };
            self.observer.on_terminal(uuid, terminal);
            return Ok(if terminal == ReplayTerminalStatus::Canceled {
                GraphReply::cancelled()
            } else {
                GraphReply::failed()
            });
        }

        // Parse the collected SSE messages into assistant text + token times.
        let mut content = String::new();
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
            let delta = chunk.delta_text();
            if !delta.is_empty() {
                // Real per-token arrival time from the transport's clock stamp.
                self.observer.on_token(uuid, self.ms(msg.perf_ns));
                content.push_str(&delta);
            }
        }

        let terminal = if done && rec.status == Some(200) {
            ReplayTerminalStatus::Completed
        } else {
            ReplayTerminalStatus::Failed
        };
        self.observer.on_terminal(uuid, terminal);

        Ok(match terminal {
            ReplayTerminalStatus::Completed => GraphReply::from_text(content),
            _ => GraphReply::failed(),
        })
    }
}
