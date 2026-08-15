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

use crate::graph::materialize::MaterializedGraphRequest;
use crate::graph::model::ToolNode;
use crate::graph::wire::WireMessage;
use crate::metrics_core::Phase;

/// Stable identity for one placed trace invocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TraceInstanceId(String);

impl TraceInstanceId {
    /// Construct an identity from the workload-assigned trace instance id.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Borrow the workload-assigned identity.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Lifecycle section within one indivisible trace program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceSubphase {
    /// Trace-local setup warmup, excluded from profiling folds.
    Warmup,
    /// The measured trace program.
    Profiling,
}

impl TraceSubphase {
    /// Whether a node in this section must produce a native inference record.
    pub fn requires_native_request_record(self, is_llm_node: bool) -> bool {
        let _ = self;
        is_llm_node
    }
}

/// Typed provenance carried with every graph request dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphDispatchContext {
    /// Metric phase used for request credits and record partitioning.
    pub phase: Phase,
    /// Trace-local lifecycle section.
    pub trace_subphase: TraceSubphase,
    /// Workload-assigned invocation identity.
    pub trace_instance: TraceInstanceId,
}

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
    /// Execute one predetermined tool node without creating an inference record.
    async fn dispatch_tool_node(
        &self,
        node_id: &str,
        _node: &ToolNode,
        _context: &GraphDispatchContext,
    ) -> Result<GraphReply<M>> {
        anyhow::bail!("graph sink does not support tool node {node_id:?}")
    }
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

    /// Dispatch a request whose messages and core fields have already been
    /// materialized. Existing sinks may retain their message-only override
    /// until their endpoint dialect consumes the optional request fields.
    async fn dispatch_request(
        &self,
        node_id: &str,
        request: MaterializedGraphRequest,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>> {
        self.dispatch(
            node_id,
            request.messages,
            request.max_tokens,
            on_first_token,
        )
        .await
    }

    /// Dispatch one fully materialized request with trace lifecycle provenance.
    ///
    /// The default preserves the existing generic sink contract. Native sinks
    /// override it to retain the typed request fields and metric partition.
    async fn dispatch_request_with_context(
        &self,
        node_id: &str,
        request: MaterializedGraphRequest,
        _context: &GraphDispatchContext,
        options: GraphDispatchOptions,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<M>> {
        let _ = options;
        self.dispatch_request(node_id, request, on_first_token)
            .await
    }

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
