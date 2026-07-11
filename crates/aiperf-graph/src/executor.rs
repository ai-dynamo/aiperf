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

use serde_json::{Value, json};

use crate::channel_store::{StoreError, VersionedChannelStore};
use crate::channels::producers_per_channel;
use crate::context::{NodeExecutionResult, TraceContext};
use crate::errors::TraceError;
use crate::materialize::PromptMaterializer;
use crate::model::{ChannelType, Count, GraphRecord, LlmNode, TraceRecord};
use crate::reducers::ChanVal;
use crate::runtime::Handle;
use crate::scheduler::Scheduler;
use crate::sink::{GraphReply, GraphSink};
use crate::wire::WireMessage;

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
}

/// Async-dataflow trace executor for a single resolved graph, generic over the
/// dialect message `M`.
pub struct TraceExecutor<M: WireMessage> {
    graph: Rc<GraphRecord>,
    scheduler: Rc<Scheduler>,
    producers: BTreeMap<String, i64>,
    materializer: Rc<dyn PromptMaterializer<M>>,
    sink: Rc<dyn GraphSink<M>>,
    handle: Handle,
    compress_edge_delays: bool,
    ignore_edge_delays: bool,
    absolute_start_offsets: bool,
    anchor_wall_us: Cell<Option<f64>>,
}

impl<M: WireMessage> TraceExecutor<M> {
    pub fn new(
        graph: Rc<GraphRecord>,
        materializer: Rc<dyn PromptMaterializer<M>>,
        sink: Rc<dyn GraphSink<M>>,
        handle: Handle,
        flags: ExecutorFlags,
    ) -> Result<Rc<Self>, TraceError> {
        let scheduler =
            Rc::new(Scheduler::new(&graph).map_err(|e| TraceError::Other(e.to_string()))?);
        let producers = producers_per_channel(&graph);
        Ok(Rc::new(TraceExecutor {
            graph,
            scheduler,
            producers,
            materializer,
            sink,
            handle,
            compress_edge_delays: flags.compress_edge_delays,
            ignore_edge_delays: flags.ignore_edge_delays,
            absolute_start_offsets: flags.absolute_start_offsets,
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
            &self.graph.state,
            &self.producers,
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
        let Some(node) = self.graph.nodes.get(&node_id).cloned() else {
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

        let inputs = ctx.store.snapshot_at_seq(gate_seq)?;
        let messages: Vec<M> = self.materializer.build(node, &inputs);

        // Signal the node's first token the moment it streams in, so
        // first-token-anchored successors gate on it (not on completion).
        let on_first_token = || ctx.set_first_token(node_id, self.loop_wall_us());
        let reply = self
            .sink
            .dispatch(node_id, messages, node.max_tokens, &on_first_token)
            .await;
        let value = match reply {
            Ok(reply) => self.reply_value(node, reply),
            // Mid-conversation resilience: a transport/dispatch failure is
            // contained — write a type-correct empty so successors that splice
            // it get omission instead of orphaning, and the trace continues.
            Err(_e) => self.empty_value(node),
        };
        self.publish_write(node_id, &node.output, value, ctx)?;
        Ok(Some(NodeExecutionResult))
    }

    /// The value written to a node's output channel for a reply. A messages-typed
    /// channel gets a one-element `[message]` list (empty on no content); a
    /// value-typed channel gets the serialized message (or `null`).
    fn reply_value(&self, node: &LlmNode, reply: GraphReply<M>) -> Value {
        let messages = self.channel_is_messages(&node.output);
        match reply.message {
            Some(m) => {
                let mv = serde_json::to_value(m).unwrap_or(Value::Null);
                if messages { Value::Array(vec![mv]) } else { mv }
            }
            None => self.empty_value(node),
        }
    }

    fn empty_value(&self, node: &LlmNode) -> Value {
        if self.channel_is_messages(&node.output) {
            json!([])
        } else {
            Value::Null
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
        value: Value,
        ctx: &Rc<TraceContext>,
    ) -> Result<(), TraceError> {
        ctx.store
            .write(std::slice::from_ref(&channel.to_string()), &value, node_id)?;
        Ok(())
    }

    fn finalize_node(
        &self,
        node_id: &str,
        node: &LlmNode,
        ctx: &Rc<TraceContext>,
        success: bool,
    ) {
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
        let requirements: Vec<(String, Count)> = node
            .inputs
            .iter()
            .map(|r| (r.channel.clone(), r.count.clone()))
            .collect();
        let _capture = ctx.store.await_inputs(&requirements).await?;
        let gate_seq = ctx.store.current_seq();
        let node_firable_wall_us = self.loop_wall_us();
        self.apply_firing_delay(node_id, ctx, node_firable_wall_us)
            .await;
        ctx.node_dispatch_wall_us
            .borrow_mut()
            .insert(node_id.to_string(), self.loop_wall_us());
        for succ_id in self.scheduler.start_anchored_successors(node_id) {
            self.clone().schedule(&succ_id, ctx);
        }
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
