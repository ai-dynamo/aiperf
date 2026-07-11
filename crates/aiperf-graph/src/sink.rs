// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! The dispatch seam: an **extensible trait** (generic over the dialect message
//! `M`) the graph executor fires each node through. A `GraphSink<M>` impl owns
//! that dialect's body encoding + reply parsing:
//!
//! * `EchoSink<M>` — serverless test double for any dialect.
//!
//! The live over-the-wire sink is [`crate::transport_sink::TransportChatSink`].
//! A future Anthropic / Responses endpoint is a new `WireMessage` + a new
//! `GraphSink` impl; nothing else in the graph changes.

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;

use crate::wire::WireMessage;

/// A captured reply — the value a node writes onto its output channel for its
/// successors to splice. Generic over the dialect message `M`.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphReply<M> {
    /// The assistant message to splice downstream, or `None` on empty/failed.
    pub message: Option<M>,
    /// Pre-serialized assistant message retained for zero-reserialize splices.
    pub wire: Option<Bytes>,
}

impl<M: WireMessage> GraphReply<M> {
    /// Build a reply from assistant text (empty text -> no message).
    pub fn from_text(text: String) -> Self {
        if text.is_empty() {
            GraphReply {
                message: None,
                wire: None,
            }
        } else {
            let message = M::assistant(text);
            let wire = Bytes::from(
                serde_json::to_vec(&message).expect("WireMessage serialization must succeed"),
            );
            GraphReply {
                message: Some(message),
                wire: Some(wire),
            }
        }
    }
    pub fn failed() -> Self {
        GraphReply {
            message: None,
            wire: None,
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
