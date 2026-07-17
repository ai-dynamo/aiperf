// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Flat-graph fast path: a straight-line executor for an eligible single-LLM-node
//! trace that reuses the graph dispatch/measure seam without the general
//! scheduler, channel store, or trace context.
//!
//! A behavior graph needs channel versioning, fan-in readiness, and successor
//! scheduling. A degenerate single-request trace needs none of that: it admits,
//! dispatches one node, and finishes. [`FlatGraphActor`] executes exactly that
//! line over the same [`GraphSink`], [`PromptMaterializer`], and
//! [`NodeDispatchPolicy`] the general executor uses, so the emitted
//! `RequestRecord` is identical by construction while the per-trace fixed
//! overhead collapses to one materialize + one dispatch.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;

use tokio::sync::Notify;

use crate::graph::errors::TraceError;
use crate::graph::materialize::PromptMaterializer;
use crate::graph::model::{GraphRecord, GraphTracePlan};
use crate::graph::policy::{
    NodeDispatchInfo, NodeDispatchPolicy, NodeFailure, NodeFailureDisposition, NodeFailureKind,
    NodeFailurePolicy,
};
use crate::graph::reducers::ChanVal;
use crate::graph::sink::{GraphReplyStatus, GraphSink};
use crate::graph::wire::WireMessage;

/// Returns `true` when `graph` is an eligible flat trace: exactly one `LlmNode`
/// and that node declares no channel-requirement inputs (no fan-in). Every other
/// shape — zero nodes, more than one node, or a node awaiting a produced channel —
/// is ineligible and falls to the general `TraceExecutor`. The runtime graph model
/// has only one executable node type (`LlmNode`); spawn/fork/subgraph/loop/barrier/
/// tool behavior is lowered into multiple `LlmNode`s + edges before a `GraphRecord`
/// exists, so a multi-node or fan-in shape is exactly what this rejects (fails
/// closed).
pub fn is_flat_graph(graph: &GraphRecord) -> bool {
    let mut nodes = graph.nodes.values();
    match (nodes.next(), nodes.next()) {
        (Some(only), None) => only.inputs.is_empty(),
        _ => false,
    }
}

/// A `TraceContext`-free cancellation latch for the flat path. A placement backend
/// holds a `Weak` to it and trips it from `cancel_inflight`; the actor selects on
/// [`FlatAbort::tripped`] around admission and dispatch. Single-threaded and
/// worker-local — no cross-thread synchronization.
pub struct FlatAbort {
    flag: Cell<bool>,
    notify: Notify,
}

impl FlatAbort {
    /// Create an untripped latch.
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            flag: Cell::new(false),
            notify: Notify::new(),
        })
    }

    /// Mark the trace cancelled and wake any waiter.
    pub fn trip(&self) {
        self.flag.set(true);
        self.notify.notify_waiters();
    }

    /// `true` once tripped.
    pub fn is_tripped(&self) -> bool {
        self.flag.get()
    }

    /// Resolve once the latch is tripped (immediately if already tripped). Safe
    /// under a current-thread runtime: the flag check and waiter registration
    /// happen with no intervening await, so a concurrent `trip` cannot be lost.
    pub async fn tripped(&self) {
        loop {
            if self.flag.get() {
                return;
            }
            self.notify.notified().await;
        }
    }
}

/// Straight-line executor for an eligible single-LLM-node trace. Reuses the graph
/// `sink`/materializer/policy; allocates no scheduler, channel store, or context.
pub struct FlatGraphActor<M: WireMessage> {
    sink: Rc<dyn GraphSink<M>>,
    materializer: Rc<dyn PromptMaterializer>,
    node_policy: Rc<dyn NodeDispatchPolicy>,
    node_failure: Rc<dyn NodeFailurePolicy>,
}

impl<M: WireMessage + 'static> FlatGraphActor<M> {
    /// Build a flat actor over the same graph seams the general executor uses.
    pub fn new(
        sink: Rc<dyn GraphSink<M>>,
        materializer: Rc<dyn PromptMaterializer>,
        node_policy: Rc<dyn NodeDispatchPolicy>,
        node_failure: Rc<dyn NodeFailurePolicy>,
    ) -> Self {
        Self {
            sink,
            materializer,
            node_policy,
            node_failure,
        }
    }

    /// Execute the eligible trace as one admitted, measured dispatch. Emits no
    /// channel write (a terminal 1-node trace has no downstream reader) and
    /// schedules no successor. Returns `Ok` on completion, cancellation, or a
    /// dispatch the sink already recorded as failed.
    pub async fn run(&self, plan: GraphTracePlan, abort: &FlatAbort) -> Result<(), TraceError> {
        let trace_id: Rc<str> = Rc::from(plan.trace.id.as_str());
        let (node_id, node) = plan.graph.nodes.iter().next().ok_or_else(|| {
            TraceError::Other(format!("flat trace {trace_id:?} has no executable node"))
        })?;

        // No fan-in: inputs are empty; the prompt is materialized from node.items.
        let empty: BTreeMap<String, ChanVal> = BTreeMap::new();
        let messages = self
            .materializer
            .build(node, &empty)
            .map_err(|error| TraceError::Other(error.to_string()))?;

        let info = NodeDispatchInfo {
            trace_id: trace_id.clone(),
            node_id: node_id.clone(),
            max_tokens: node.max_tokens,
        };
        let admit = self.node_policy.admit(&info);
        tokio::pin!(admit);
        let permit = tokio::select! {
            biased;
            () = abort.tripped() => return Ok(()),
            result = &mut admit => result.map_err(|error| TraceError::Other(error.to_string()))?,
        };

        let options = permit.options();
        let first_token_seen = Cell::new(false);
        let on_first_token = || {
            if !first_token_seen.replace(true) {
                permit.on_first_token();
            }
        };
        let dispatch =
            self.sink
                .dispatch_with_options(node_id, messages, node.max_tokens, options, &on_first_token);
        tokio::pin!(dispatch);
        let reply = tokio::select! {
            biased;
            () = abort.tripped() => return Ok(()),
            result = &mut dispatch => result,
        };

        // Classify the terminal exactly as the general executor does: a Completed
        // reply is Ok; a Failed/Cancelled reply or a sink error becomes a
        // `NodeFailure` whose disposition (from the same `node_failure` policy)
        // decides Continue (Ok) vs Abort (Err). The sink already emitted the
        // RequestRecord in every case.
        let (kind, message) = match reply {
            Ok(reply) => {
                permit.on_terminal(reply.status);
                match reply.status {
                    GraphReplyStatus::Completed => return Ok(()),
                    GraphReplyStatus::Failed => {
                        (NodeFailureKind::FailedReply, "backend returned a failed reply".to_string())
                    }
                    GraphReplyStatus::Cancelled => (
                        NodeFailureKind::CancelledReply,
                        "backend returned a cancelled reply".to_string(),
                    ),
                }
            }
            Err(error) => {
                permit.on_terminal(GraphReplyStatus::Failed);
                (NodeFailureKind::Sink, error.to_string())
            }
        };

        let failure = NodeFailure {
            trace_id,
            node_id: node_id.clone(),
            kind,
            message,
        };
        match self.node_failure.on_failure(&failure) {
            NodeFailureDisposition::ContinueWithEmpty => Ok(()),
            NodeFailureDisposition::AbortTrace => {
                let message = format!(
                    "graph node {:?} failed ({:?}): {}",
                    failure.node_id, failure.kind, failure.message
                );
                if failure.kind == NodeFailureKind::CancelledReply {
                    Err(TraceError::Cancelled(message))
                } else {
                    Err(TraceError::Other(message))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::model::{ChannelRequirement, Count, LlmNode, TraceRecord};
    use crate::graph::sink::{GraphDispatchOptions, GraphReply};
    use crate::graph::wire::OpenAiChatMessage;
    use anyhow::Result;
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::cell::RefCell;

    fn llm_node(inputs: Vec<ChannelRequirement>) -> LlmNode {
        LlmNode {
            output: "out".into(),
            streaming: true,
            inputs,
            min_start_delay_us: None,
            max_tokens: Some(4),
            items: Vec::new(),
            metadata: BTreeMap::new(),
        }
    }

    fn graph(nodes: Vec<(&str, LlmNode)>) -> GraphRecord {
        GraphRecord {
            nodes: nodes
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            ..GraphRecord::default()
        }
    }

    fn one_node_plan(trace_id: &str) -> GraphTracePlan {
        GraphTracePlan {
            graph: graph(vec![("n0", llm_node(vec![]))]),
            trace: TraceRecord {
                id: trace_id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        }
    }

    #[test]
    fn single_node_no_inputs_is_flat() {
        assert!(is_flat_graph(&graph(vec![("a", llm_node(vec![]))])));
    }

    #[test]
    fn zero_nodes_is_not_flat() {
        assert!(!is_flat_graph(&graph(vec![])));
    }

    #[test]
    fn two_nodes_is_not_flat() {
        assert!(!is_flat_graph(&graph(vec![
            ("a", llm_node(vec![])),
            ("b", llm_node(vec![])),
        ])));
    }

    #[test]
    fn single_node_with_channel_input_is_not_flat() {
        let req = ChannelRequirement {
            channel: "dep".into(),
            count: Count::default(),
        };
        assert!(!is_flat_graph(&graph(vec![("a", llm_node(vec![req]))])));
    }

    /// Records every dispatch and returns a Completed reply.
    #[derive(Default)]
    struct RecordingSink {
        calls: RefCell<Vec<(String, usize, Option<usize>)>>,
    }

    #[async_trait(?Send)]
    impl GraphSink<OpenAiChatMessage> for RecordingSink {
        async fn dispatch(
            &self,
            node_id: &str,
            messages: Vec<Bytes>,
            max_tokens: Option<usize>,
            on_first_token: &dyn Fn(),
        ) -> Result<GraphReply<OpenAiChatMessage>> {
            on_first_token();
            self.calls
                .borrow_mut()
                .push((node_id.to_string(), messages.len(), max_tokens));
            Ok(GraphReply {
                message: None,
                wire: Some(Bytes::from_static(b"ok")),
                status: GraphReplyStatus::Completed,
            })
        }
    }

    /// Yields one message regardless of node, so `run` has something to dispatch.
    struct StubMaterializer;

    impl PromptMaterializer for StubMaterializer {
        fn build(
            &self,
            _node: &LlmNode,
            _inputs: &BTreeMap<String, ChanVal>,
        ) -> std::result::Result<Vec<Bytes>, crate::dataset::DatasetError> {
            Ok(vec![Bytes::from_static(b"[]")])
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn flat_actor_dispatches_the_single_node_once() {
        let sink = Rc::new(RecordingSink::default());
        let actor: FlatGraphActor<OpenAiChatMessage> = FlatGraphActor::new(
            sink.clone(),
            Rc::new(StubMaterializer),
            Rc::new(crate::graph::policy::NoopNodeDispatchPolicy),
            Rc::new(crate::graph::policy::ResilientNodeFailurePolicy),
        );
        let abort = FlatAbort::new();
        actor.run(one_node_plan("t-1"), &abort).await.unwrap();

        let calls = sink.calls.borrow();
        assert_eq!(calls.len(), 1, "exactly one dispatch");
        assert_eq!(calls[0].0, "n0");
        assert_eq!(calls[0].1, 1, "one materialized message");
        assert_eq!(calls[0].2, Some(4), "node.max_tokens forwarded");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn pre_tripped_abort_skips_dispatch() {
        let sink = Rc::new(RecordingSink::default());
        let actor: FlatGraphActor<OpenAiChatMessage> = FlatGraphActor::new(
            sink.clone(),
            Rc::new(StubMaterializer),
            Rc::new(crate::graph::policy::NoopNodeDispatchPolicy),
            Rc::new(crate::graph::policy::ResilientNodeFailurePolicy),
        );
        let abort = FlatAbort::new();
        abort.trip();
        actor.run(one_node_plan("t-1"), &abort).await.unwrap();
        assert_eq!(sink.calls.borrow().len(), 0, "tripped abort dispatches nothing");
    }
}
