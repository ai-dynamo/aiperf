// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The dispatch seam: an **extensible trait** (generic over the dialect message
//! `M`) the graph executor fires each node through. A `GraphSink<M>` impl owns
//! that dialect's body encoding + reply parsing:
//!
//! * `HttpChatSink` — OpenAI chat over HTTP via `dynamo_aiperf::HttpSink`
//!   (→ Dynamo frontend → mocker/GPUs), feeding the shared `TraceCollector`.
//! * `EchoSink<M>` — serverless test double for any dialect.
//!
//! A future Anthropic / Responses endpoint is a new `WireMessage` + a new
//! `GraphSink` impl; nothing else in the graph changes.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use uuid::Uuid;

use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;

use crate::wire::{OpenAiChatMessage, WireMessage};
use aiperf_core::http_sink::{ChatMessage, HttpSink};

/// A captured reply — the value a node writes onto its output channel for its
/// successors to splice. Generic over the dialect message `M`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphReply<M> {
    /// The assistant message to splice downstream, or `None` on empty/failed.
    pub message: Option<M>,
    pub status: ReplyStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplyStatus {
    Ok,
    Empty,
    Failed,
}

impl<M: WireMessage> GraphReply<M> {
    /// Build a reply from assistant text (empty text -> `Empty`).
    pub fn from_text(text: String) -> Self {
        if text.is_empty() {
            GraphReply {
                message: None,
                status: ReplyStatus::Empty,
            }
        } else {
            GraphReply {
                message: Some(M::assistant(text)),
                status: ReplyStatus::Ok,
            }
        }
    }
    pub fn failed() -> Self {
        GraphReply {
            message: None,
            status: ReplyStatus::Failed,
        }
    }
}

/// One dispatch of a materialized node prompt. `?Send`: the executor is
/// single-threaded per trace (`Rc`/`RefCell`), so sinks need not be `Send`.
#[async_trait(?Send)]
pub trait GraphSink<M: WireMessage> {
    /// Dispatch a node's materialized prompt. `on_first_token` must be invoked
    /// the moment the reply's first output token is observed (before the reply
    /// completes), so the executor can gate first-token-anchored successors.
    async fn dispatch(
        &self,
        node_id: &str,
        messages: Vec<M>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>>;
}

/// Live OpenAI-chat sink: streams chat-completions over HTTP, feeds the collector.
pub struct HttpChatSink {
    http: Arc<HttpSink>,
    observer: Arc<dyn RequestObserver>,
    default_max_tokens: usize,
}

impl HttpChatSink {
    pub fn new(
        http: Arc<HttpSink>,
        observer: Arc<dyn RequestObserver>,
        default_max_tokens: usize,
    ) -> Self {
        HttpChatSink {
            http,
            observer,
            default_max_tokens,
        }
    }
}

#[async_trait(?Send)]
impl GraphSink<OpenAiChatMessage> for HttpChatSink {
    async fn dispatch(
        &self,
        _node_id: &str,
        messages: Vec<OpenAiChatMessage>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        let chat: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| ChatMessage {
                role: m.role,
                content: m.content,
            })
            .collect();
        let uuid = Uuid::new_v4();
        let outcome = self
            .http
            .stream_chat_cb(
                uuid,
                &chat,
                max_tokens.unwrap_or(self.default_max_tokens),
                on_first_token,
            )
            .await?;
        // Feed the shared collector (same events RequestSink::dispatch emits).
        self.observer.on_admit(uuid, outcome.admit_ms, 0);
        for at in &outcome.token_times {
            self.observer.on_token(uuid, *at);
        }
        self.observer.on_terminal(uuid, outcome.terminal);
        Ok(match outcome.terminal {
            ReplayTerminalStatus::Completed => GraphReply::from_text(outcome.content),
            _ => GraphReply::failed(),
        })
    }
}

/// Serverless test double for any dialect: echoes the last message's debug form.
pub struct EchoSink;

#[async_trait(?Send)]
impl<M: WireMessage> GraphSink<M> for EchoSink {
    async fn dispatch(
        &self,
        node_id: &str,
        messages: Vec<M>,
        _max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>> {
        // The echo reply is produced instantly, so its first token is now.
        on_first_token();
        let last = messages
            .last()
            .map(|m| format!("{m:?}"))
            .unwrap_or_default();
        Ok(GraphReply::from_text(format!("[{node_id}] <= {last}")))
    }
}
