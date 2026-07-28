// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Async-dataflow per-trace executor, wired to the extensible [`GraphSink`] +
//! [`PromptMaterializer`] traits.
//!
//! A node fires when its input channels are ready; its prompt is materialized
//! from the segment store + upstream channel content, dispatched through the
//! sink (HTTP → Dynamo/mocker), and the reply is written back onto its output
//! channel so successors splice it. The firing gate honors every edge-delay kind
//! (completion, start-anchored, first-token, min-start) plus the node-level
//! min-start delay and the compress/ignore overrides.

use std::cell::Cell;
use std::collections::BTreeMap;
use std::rc::Rc;

use bytes::Bytes;
use serde_json::Value;

use crate::graph::channel_store::{StoreError, VersionedChannelStore};
use crate::graph::channels::producers_per_channel;
use crate::graph::context::{NodeExecutionResult, TraceContext};
use crate::graph::errors::TraceError;
use crate::graph::materialize::PromptMaterializer;
use crate::graph::model::{ChannelSpec, ChannelType, GraphRecord, LlmNode, TraceRecord};
use crate::graph::policy::{
    NodeDispatchInfo, NodeDispatchPolicy, NodeFailure, NodeFailureDisposition, NodeFailureKind,
    NodeFailurePolicy, NoopNodeDispatchPolicy, ResilientNodeFailurePolicy,
};
use crate::graph::reducers::ChanVal;
use crate::graph::runtime::Handle;
use crate::graph::scheduler::Scheduler;
use crate::graph::sink::{GraphReply, GraphReplyStatus, GraphSink};
use crate::graph::wire::WireMessage;

/// The post-run channel snapshot for one trace.
#[derive(Debug, Clone)]
pub struct TraceResult {
    pub trace_id: String,
    pub channels: BTreeMap<String, ChanVal>,
}

/// Edge-timing overrides for a [`TraceExecutor`]. Defaults are all `false` (the
/// live/replay path honors every edge delay against the loop clock).
#[derive(Debug, Clone, Copy, Default)]
pub struct ExecutorFlags {
    /// Collapse all edge delays to zero (fire successors as soon as ready).
    pub compress_edge_delays: bool,
    /// Ignore edge delays entirely (same firing effect as `compress`).
    pub ignore_edge_delays: bool,
    /// Anchor node `min_start` offsets to an absolute wall origin captured on
    /// the first `build_context`, rather than to each node's firable instant.
    pub absolute_start_offsets: bool,
    /// Maximum global idle interval before a graph node dispatch, in milliseconds.
    pub system_idle_gap_cap_ms: Option<f64>,
}

/// Async-dataflow trace executor for a single resolved graph, generic over the
/// dialect message `M`.
pub struct TraceExecutor<M: WireMessage> {
    graph: Rc<GraphRecord>,
    /// `Rc`-shared handle to each node, built once at construction. `fire` clones
    /// the `Rc` (a pointer bump) instead of deep-cloning the whole `LlmNode`
    /// (its `items`/`inputs` vectors and channel strings) on every firing.
    node_index: BTreeMap<String, Rc<LlmNode>>,
    scheduler: Rc<Scheduler>,
    /// Run-immutable channel specs and declared-producer counts, allocated once
    /// and `Rc`-shared into every per-trace store instead of deep-cloned per
    /// trace (see [`VersionedChannelStore::new`]).
    channel_specs: Rc<BTreeMap<String, ChannelSpec>>,
    producers: Rc<BTreeMap<String, i64>>,
    materializer: Rc<dyn PromptMaterializer>,
    sink: Rc<dyn GraphSink<M>>,
    node_policy: Rc<dyn NodeDispatchPolicy>,
    failure_policy: Rc<dyn NodeFailurePolicy>,
    handle: Handle,
    compress_edge_delays: bool,
    ignore_edge_delays: bool,
    absolute_start_offsets: bool,
    system_idle_gap_cap_ms: Option<f64>,
    anchor_wall_us: Cell<Option<f64>>,
}

fn cap_system_idle_wait_us(wait_us: f64, cap_ms: Option<f64>) -> f64 {
    cap_ms
        .filter(|cap| cap.is_finite() && *cap >= 0.0)
        .map_or(wait_us, |cap| wait_us.min(cap * 1_000.0))
}

impl<M: WireMessage> TraceExecutor<M> {
    pub fn new(
        graph: Rc<GraphRecord>,
        materializer: Rc<dyn PromptMaterializer>,
        sink: Rc<dyn GraphSink<M>>,
        handle: Handle,
        flags: ExecutorFlags,
    ) -> Result<Rc<Self>, TraceError> {
        Self::new_with_policies(
            graph,
            materializer,
            sink,
            Rc::new(NoopNodeDispatchPolicy),
            Rc::new(ResilientNodeFailurePolicy),
            handle,
            flags,
        )
    }

    /// Construct with injected node admission/timing and failure semantics.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_policies(
        graph: Rc<GraphRecord>,
        materializer: Rc<dyn PromptMaterializer>,
        sink: Rc<dyn GraphSink<M>>,
        node_policy: Rc<dyn NodeDispatchPolicy>,
        failure_policy: Rc<dyn NodeFailurePolicy>,
        handle: Handle,
        flags: ExecutorFlags,
    ) -> Result<Rc<Self>, TraceError> {
        let scheduler =
            Rc::new(Scheduler::new(&graph).map_err(|e| TraceError::Other(e.to_string()))?);
        let producers = Rc::new(producers_per_channel(&graph));
        let channel_specs = Rc::new(graph.state.clone());
        let node_index = graph
            .nodes
            .iter()
            .map(|(id, node)| (id.clone(), Rc::new(node.clone())))
            .collect();
        Ok(Rc::new(TraceExecutor {
            graph,
            node_index,
            scheduler,
            channel_specs,
            producers,
            materializer,
            sink,
            node_policy,
            failure_policy,
            handle,
            compress_edge_delays: flags.compress_edge_delays,
            ignore_edge_delays: flags.ignore_edge_delays,
            absolute_start_offsets: flags.absolute_start_offsets,
            system_idle_gap_cap_ms: flags.system_idle_gap_cap_ms,
            anchor_wall_us: Cell::new(None),
        }))
    }

    pub fn build_context(
        self: &Rc<Self>,
        trace: TraceRecord,
    ) -> Result<Rc<TraceContext>, TraceError> {
        if self.absolute_start_offsets && self.anchor_wall_us.get().is_none() {
            self.anchor_wall_us.set(Some(self.loop_wall_us()));
        }
        let store = Rc::new(VersionedChannelStore::new(
            &trace.initial_state,
            self.channel_specs.clone(),
            self.producers.clone(),
        )?);
        Ok(TraceContext::new(trace, store))
    }

    pub fn schedule_entries(self: &Rc<Self>, ctx: &Rc<TraceContext>) {
        for entry_id in self.scheduler.entry_nodes() {
            self.clone().schedule(&entry_id, ctx);
        }
    }

    pub fn result(ctx: &Rc<TraceContext>) -> Result<TraceResult, TraceError> {
        Ok(TraceResult {
            trace_id: ctx.trace.id.clone(),
            channels: ctx.store.snapshot()?,
        })
    }

    fn loop_wall_us(&self) -> f64 {
        self.handle.now_ns() as f64 / 1_000.0
    }

    fn schedule(self: Rc<Self>, node_id: &str, ctx: &Rc<TraceContext>) {
        {
            let scheduled = ctx.scheduled_node_ids.borrow();
            if scheduled.contains(node_id) {
                if ctx.completed_node_ids.borrow().contains(node_id) {
                    ctx.set_abort(TraceError::Other(format!(
                        "cycle detected: node {node_id:?} re-scheduled after completing"
                    )));
                }
                return;
            }
        }
        ctx.scheduled_node_ids
            .borrow_mut()
            .insert(node_id.to_string());
        let node_id = node_id.to_string();
        let ctx = ctx.clone();
        let this = self.clone();
        self.handle.spawn(async move {
            this.fire(node_id, ctx).await;
        });
    }

    async fn fire(self: Rc<Self>, node_id: String, ctx: Rc<TraceContext>) {
        // An edge (or START) can target an id that isn't a declared node; treat
        // that as a clean trace error rather than panicking on a missing key.
        // Cloning the `Rc` here is a pointer bump; the owned handle lives across
        // the `run_node` await without borrowing `self` (which would make this a
        // self-referential future).
        let Some(node) = self.node_index.get(&node_id).cloned() else {
            ctx.set_abort(TraceError::Other(format!(
                "edge targets undeclared node {node_id:?}"
            )));
            return;
        };
        let outcome = self.clone().run_node(&node_id, &node, &ctx).await;
        let (result, err) = match outcome {
            Ok(result) => (result, None),
            Err(e) => (None, Some(e)),
        };
        let success = result.is_some();
        self.finalize_node(&node_id, &node, &ctx, success);
        if let Some(e) = err {
            ctx.set_abort(e);
            return;
        }
        if result.is_none() {
            return;
        }
        self.schedule_successors(&node_id, &ctx);
    }

    async fn run_node(
        self: Rc<Self>,
        node_id: &str,
        node: &LlmNode,
        ctx: &Rc<TraceContext>,
    ) -> Result<Option<NodeExecutionResult>, TraceError> {
        if ctx.is_aborted() {
            return Ok(None);
        }
        let gate_seq = match self.prepare_node_inputs(node_id, node, ctx).await {
            Ok(seq) => seq,
            Err(e) => return Err(TraceError::Store(e)),
        };
        if ctx.is_aborted() {
            return Ok(None);
        }

        // The materializer only reads this node's splice channels; reducing the
        // whole store per fire is O(channels × history) on the allocation hot
        // path. `snapshot_selected_at_seq` is byte-identical over these keys.
        let splice_channels = node.splice_channels();
        let inputs = ctx
            .store
            .snapshot_selected_at_seq(&splice_channels, gate_seq)?;
        let messages: Vec<Bytes> = self
            .materializer
            .build(node, &inputs)
            .map_err(|error| TraceError::Other(error.to_string()))?;

        let info = NodeDispatchInfo {
            trace_id: ctx.trace_id.clone(),
            node_id: node_id.to_string(),
            max_tokens: node.max_tokens,
        };
        let admission = self.node_policy.admit(&info);
        tokio::pin!(admission);
        let permit = match tokio::select! {
            biased;
            () = ctx.await_abort() => None,
            permit = &mut admission => Some(permit),
        } {
            None => return Ok(None),
            Some(Ok(permit)) => permit,
            Some(Err(error)) => {
                self.mark_dispatch_start(node_id, ctx);
                let failure = NodeFailure {
                    trace_id: ctx.trace_id.clone(),
                    node_id: node_id.to_string(),
                    kind: NodeFailureKind::Admission,
                    message: error.to_string(),
                };
                let value = self.value_after_failure(node, &failure)?;
                self.publish_write(node_id, &node.output, value, ctx)?;
                return Ok(Some(NodeExecutionResult));
            }
        };
        self.mark_dispatch_start(node_id, ctx);
        let options = permit.options();
        let first_token_seen = Cell::new(false);
        // Signal both the firing gate and injected prefill policy on the exact
        // same first-token edge from the one sink dispatch.
        let on_first_token = || {
            if !first_token_seen.replace(true) {
                permit.on_first_token();
                ctx.set_first_token(node_id, self.loop_wall_us());
            }
        };
        let reply = self
            .sink
            .dispatch_with_options(node_id, messages, node.max_tokens, options, &on_first_token)
            .await;
        let value = match reply {
            Ok(reply) if reply.status == GraphReplyStatus::Completed => {
                permit.on_terminal(GraphReplyStatus::Completed);
                self.reply_value(node, reply)
            }
            Ok(reply) => {
                permit.on_terminal(reply.status);
                let (kind, message) = match reply.status {
                    GraphReplyStatus::Failed => (
                        NodeFailureKind::FailedReply,
                        "backend returned a failed reply",
                    ),
                    GraphReplyStatus::Cancelled => (
                        NodeFailureKind::CancelledReply,
                        "backend returned a cancelled reply",
                    ),
                    GraphReplyStatus::Completed => unreachable!("guarded above"),
                };
                self.value_after_failure(
                    node,
                    &NodeFailure {
                        trace_id: ctx.trace_id.clone(),
                        node_id: node_id.to_string(),
                        kind,
                        message: message.to_string(),
                    },
                )?
            }
            Err(error) => {
                permit.on_terminal(GraphReplyStatus::Failed);
                self.value_after_failure(
                    node,
                    &NodeFailure {
                        trace_id: ctx.trace_id.clone(),
                        node_id: node_id.to_string(),
                        kind: NodeFailureKind::Sink,
                        message: error.to_string(),
                    },
                )?
            }
        };
        self.publish_write(node_id, &node.output, value, ctx)?;
        Ok(Some(NodeExecutionResult))
    }

    fn value_after_failure(
        &self,
        node: &LlmNode,
        failure: &NodeFailure,
    ) -> Result<ChanVal, TraceError> {
        match self.failure_policy.on_failure(failure) {
            NodeFailureDisposition::ContinueWithEmpty => Ok(self.empty_value(node)),
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

    /// The value written to a node's output channel for a reply. A messages-typed
    /// channel gets a one-element `[message]` list (empty on no content); a
    /// value-typed channel gets the serialized message (or `null`).
    fn reply_value(&self, node: &LlmNode, reply: GraphReply<M>) -> ChanVal {
        let messages = self.channel_is_messages(&node.output);
        match reply.message {
            Some(m) => {
                let mv = serde_json::to_value(m).unwrap_or(Value::Null);
                if messages {
                    let wire = reply.wire.unwrap_or_else(|| {
                        Bytes::from(
                            serde_json::to_vec(&mv)
                                .expect("serde_json::Value serialization is infallible"),
                        )
                    });
                    ChanVal::encoded_messages(vec![(mv, wire)])
                } else {
                    ChanVal::Val(mv)
                }
            }
            None => self.empty_value(node),
        }
    }

    fn empty_value(&self, node: &LlmNode) -> ChanVal {
        if self.channel_is_messages(&node.output) {
            ChanVal::encoded_messages(Vec::new())
        } else {
            ChanVal::Val(Value::Null)
        }
    }

    fn channel_is_messages(&self, channel: &str) -> bool {
        self.graph
            .state
            .get(channel)
            .map(|s| s.channel_type == ChannelType::Messages)
            .unwrap_or(false)
    }

    fn publish_write(
        &self,
        node_id: &str,
        channel: &str,
        value: ChanVal,
        ctx: &Rc<TraceContext>,
    ) -> Result<(), TraceError> {
        ctx.store.write_channel_value(&[channel], &value, node_id)?;
        Ok(())
    }

    fn mark_dispatch_start(self: &Rc<Self>, node_id: &str, ctx: &Rc<TraceContext>) {
        ctx.node_dispatch_wall_us
            .borrow_mut()
            .insert(node_id.to_string(), self.loop_wall_us());
        for successor in self.scheduler.start_anchored_successors(node_id) {
            self.clone().schedule(&successor, ctx);
        }
    }

    fn finalize_node(&self, node_id: &str, node: &LlmNode, ctx: &Rc<TraceContext>, success: bool) {
        ctx.node_finish_wall_us
            .borrow_mut()
            .insert(node_id.to_string(), self.loop_wall_us());
        // Mark completed BEFORE notifying so a successor gated on this node's
        // first token sees the resolved state on wake (it finished without one).
        ctx.completed_node_ids
            .borrow_mut()
            .insert(node_id.to_string());
        ctx.notify_first_token();
        for channel in node.write_channels() {
            let _ = ctx.store.mark_producer_done(channel, success);
        }
    }

    fn schedule_successors(self: &Rc<Self>, node_id: &str, ctx: &Rc<TraceContext>) {
        for succ_id in self.scheduler.successors_after(node_id) {
            self.clone().schedule(&succ_id, ctx);
        }
    }

    async fn prepare_node_inputs(
        self: &Rc<Self>,
        node_id: &str,
        node: &LlmNode,
        ctx: &Rc<TraceContext>,
    ) -> Result<i64, StoreError> {
        let _capture = ctx
            .store
            .await_inputs(node.inputs.iter().map(|r| (r.channel.as_str(), &r.count)))
            .await?;
        let gate_seq = ctx.store.current_seq();
        let node_firable_wall_us = self.loop_wall_us();
        self.apply_firing_delay(node_id, ctx, node_firable_wall_us)
            .await;
        Ok(gate_seq)
    }

    async fn apply_firing_delay(
        &self,
        node_id: &str,
        ctx: &Rc<TraceContext>,
        node_firable_wall_us: f64,
    ) {
        if self.ignore_edge_delays || self.compress_edge_delays {
            return;
        }
        for edge in self.scheduler.incoming_static_edges(node_id) {
            if edge.delay_after_predecessor_first_token_us.is_some() {
                ctx.await_first_token(&edge.source).await;
            }
        }
        let gate_us = self.compute_firing_gate_us(node_id, ctx, node_firable_wall_us);
        if gate_us <= 0.0 {
            return;
        }
        let wait_us = gate_us - self.loop_wall_us();
        if wait_us <= 0.0 {
            return;
        }
        let wait_us = cap_system_idle_wait_us(wait_us, self.system_idle_gap_cap_ms);
        self.handle.sleep_ns((wait_us * 1_000.0) as i64).await;
    }

    fn compute_firing_gate_us(
        &self,
        node_id: &str,
        ctx: &Rc<TraceContext>,
        node_firable_wall_us: f64,
    ) -> f64 {
        let mut gate_us = 0.0_f64;
        let finishes = ctx.node_finish_wall_us.borrow();
        let dispatches = ctx.node_dispatch_wall_us.borrow();
        let first_tokens = ctx.node_first_token_wall_us.borrow();
        for edge in self.scheduler.incoming_static_edges(node_id) {
            if let Some(delay) = edge.delay_after_predecessor_first_token_us
                && let Some(ft) = first_tokens.get(&edge.source)
            {
                gate_us = gate_us.max(ft + delay);
                continue;
            }
            if let Some(delay) = edge.delay_after_predecessor_us
                && let Some(finish) = finishes.get(&edge.source)
            {
                gate_us = gate_us.max(finish + delay);
            }
            if let Some(delay) = edge.min_start_delay_us {
                gate_us = gate_us.max(node_firable_wall_us + delay);
            }
            if let Some(delay) = edge.delay_after_predecessor_start_us
                && let Some(dispatch) = dispatches.get(&edge.source)
            {
                gate_us = gate_us.max(dispatch + delay);
            }
        }
        if let Some(node) = self.graph.nodes.get(node_id)
            && let Some(node_min_start) = node.min_start_delay_us
        {
            let anchor = match (self.absolute_start_offsets, self.anchor_wall_us.get()) {
                (true, Some(a)) => a,
                _ => node_firable_wall_us,
            };
            gate_us = gate_us.max(anchor + node_min_start);
        }
        gate_us
    }
}

#[cfg(test)]
mod tests {
    use super::cap_system_idle_wait_us;

    #[test]
    fn caps_oversized_graph_idle_wait() {
        assert_eq!(cap_system_idle_wait_us(100_000.0, Some(10.0)), 10_000.0);
    }

    #[test]
    fn preserves_disabled_and_short_waits() {
        assert_eq!(cap_system_idle_wait_us(5_000.0, Some(10.0)), 5_000.0);
        assert_eq!(cap_system_idle_wait_us(100_000.0, None), 100_000.0);
    }
}
