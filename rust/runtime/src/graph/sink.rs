// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The dispatch seam: an **extensible trait** (generic over the dialect message
//! `M`) the graph executor fires each node through. A `GraphSink<M>` impl owns
//! that dialect's body encoding + reply parsing:
//!
//! * `EchoSink<M>` — serverless test double for any dialect.
//!
//! The live over-the-wire sink is [`crate::graph::transport_sink::TransportChatSink`].

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

use crate::graph::wire::WireMessage;

/// Terminal classification returned by a graph sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphReplyStatus {
    /// The backend completed the request normally.
    Completed,
    /// The backend or endpoint failed the request.
    Failed,
    /// A configured client cancellation ended the request.
    Cancelled,
}

/// Per-node dispatch directives produced by injected graph policies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphDispatchOptions {
    /// Post-send cancellation delay in nanoseconds.
    pub cancel_after_ns: Option<i64>,
}

/// A captured reply — the value a node writes onto its output channel for its
/// successors to splice. Generic over the dialect message `M`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphReply<M> {
    /// The assistant message to splice downstream, or `None` on empty/failed.
    pub message: Option<M>,
    /// Pre-serialized assistant message retained for zero-reserialize splices.
    pub wire: Option<Bytes>,
    /// Terminal outcome used by the injected node-failure policy.
    pub status: GraphReplyStatus,
}

impl<M: WireMessage> GraphReply<M> {
    /// Build a reply from assistant text (empty text -> no message).
    pub fn from_text(text: String) -> Self {
        if text.is_empty() {
            GraphReply {
                message: None,
                wire: None,
                status: GraphReplyStatus::Completed,
            }
        } else {
            let message = M::assistant(text);
            let wire = Bytes::from(
                serde_json::to_vec(&message).expect("WireMessage serialization must succeed"),
            );
            GraphReply {
                message: Some(message),
                wire: Some(wire),
                status: GraphReplyStatus::Completed,
            }
        }
    }
    /// Build a reply classified as backend/endpoint failure.
    pub fn failed() -> Self {
        GraphReply {
            message: None,
            wire: None,
            status: GraphReplyStatus::Failed,
        }
    }

    /// Build a reply classified as client-cancelled.
    pub fn cancelled() -> Self {
        GraphReply {
            message: None,
            wire: None,
            status: GraphReplyStatus::Cancelled,
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
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>>;

    /// Dispatch with policy-produced directives.
    ///
    /// Existing endpoint implementations remain valid through this default;
    /// sinks with cancellation support override it and consume `options`.
    async fn dispatch_with_options(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        options: GraphDispatchOptions,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>> {
        let _ = options;
        self.dispatch(node_id, messages, max_tokens, on_first_token)
            .await
    }
}

/// Serverless test double for any dialect: echoes the last message's debug form.
pub struct EchoSink;

#[async_trait(?Send)]
impl<M: WireMessage> GraphSink<M> for EchoSink {
    async fn dispatch(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        _max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>> {
        // The echo reply is produced instantly, so its first token is now.
        on_first_token();
        let last = messages
            .last()
            .map(|message| String::from_utf8_lossy(message).into_owned())
            .unwrap_or_default();
        Ok(GraphReply::from_text(format!("[{node_id}] <= {last}")))
    }
}
