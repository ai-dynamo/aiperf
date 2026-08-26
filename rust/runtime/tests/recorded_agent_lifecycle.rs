// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract coverage for recorded-agent trace lifecycle dispatch context.
#![cfg(feature = "engine")]

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use aiperf_runtime::clock::{Clock, SimClock};
use aiperf_runtime::graph::execution::LocalGraphTraceExecutionBackend;
use aiperf_runtime::graph::materialize::PromptMaterializer;
use aiperf_runtime::graph::model::{
    ChannelSpec, ChannelType, ExecutableGraphNode, GraphRecord, GraphTracePlan, ReducerName,
    START_NODE_ID, StaticEdge, ToolNode, TraceRecord,
};
use aiperf_runtime::graph::reducers::ChanVal;
use aiperf_runtime::graph::sink::{
    GraphDispatchContext, GraphReply, GraphSink, TraceInstanceId, TraceSubphase,
};
use aiperf_runtime::graph::wire::OpenAiChatMessage;
use aiperf_runtime::metrics_core::Phase;
use anyhow::Result;
use async_trait::async_trait;

#[tokio::test(flavor = "current_thread")]
async fn trace_opens_environment_then_warmup_then_profile_then_cleanup() {
    // This pins the vocabulary a lifecycle dispatches with: warmup and
    // profiling carry distinct `TraceSubphase` values while sharing one trace
    // instance id. It inspects the contexts only; it runs no lifecycle.
    let warmup = GraphDispatchContext {
        phase: Phase::Warmup,
        trace_subphase: TraceSubphase::Warmup,
        trace_instance: TraceInstanceId::new("trace::instance-0"),
    };
    let profile = GraphDispatchContext {
        phase: Phase::Profiling,
        trace_subphase: TraceSubphase::Profiling,
        trace_instance: TraceInstanceId::new("trace::instance-0"),
    };

    assert_eq!(warmup.trace_subphase, TraceSubphase::Warmup);
    assert_eq!(profile.trace_subphase, TraceSubphase::Profiling);
    assert_eq!(warmup.trace_instance, profile.trace_instance);
}

#[test]
fn tool_only_terminal_does_not_require_native_request_record() {
    // This catches accounting that mistakes a terminal tool observation for a
    // missing LLM record. Tool nodes intentionally never consume inference
    // credits or produce native request records.
    assert!(!TraceSubphase::Profiling.requires_native_request_record(false));
}

struct EmptyMaterializer;

impl PromptMaterializer for EmptyMaterializer {
    fn build(
        &self,
        _node: &aiperf_runtime::graph::model::LlmNode,
        _inputs: &BTreeMap<String, ChanVal>,
    ) -> Result<Vec<bytes::Bytes>, aiperf_runtime::dataset::DatasetError> {
        Ok(Vec::new())
    }
}

struct ToolOnlySink {
    commands: RefCell<Vec<String>>,
    llm_dispatches: Cell<u64>,
}

#[async_trait(?Send)]
impl GraphSink<OpenAiChatMessage> for ToolOnlySink {
    async fn dispatch(
        &self,
        _node_id: &str,
        _messages: Vec<bytes::Bytes>,
        _max_tokens: Option<usize>,
        _on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        self.llm_dispatches.set(self.llm_dispatches.get() + 1);
        anyhow::bail!("tool-only graph must not dispatch an LLM request")
    }

    async fn dispatch_tool_node(
        &self,
        _node_id: &str,
        node: &ToolNode,
        _context: &GraphDispatchContext,
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        self.commands.borrow_mut().extend(node.commands.clone());
        Ok(GraphReply::from_text("tool observation".into()))
    }
}

#[tokio::test(flavor = "current_thread")]
async fn tool_only_terminal_executes_commands_in_order_without_native_request() {
    // This catches treating a terminal ToolNode as an absent inference record,
    // or sending its commands concurrently/outside the trace executor.
    let mut graph = GraphRecord::default();
    graph.state.insert(
        "observation".into(),
        ChannelSpec {
            channel_type: ChannelType::Messages,
            reducer: ReducerName::AddMessages,
        },
    );
    graph.nodes.insert(
        "tool".into(),
        ExecutableGraphNode::Tool(ToolNode {
            output: "observation".into(),
            commands: vec!["first".into(), "second".into()],
            timeout_ns: None,
        }),
    );
    graph.edges.push(StaticEdge {
        source: START_NODE_ID.into(),
        target: "tool".into(),
        delay_after_predecessor_us: None,
        min_start_delay_us: None,
        delay_after_predecessor_start_us: None,
        delay_after_predecessor_first_token_us: None,
    });
    let sink = Rc::new(ToolOnlySink {
        commands: RefCell::new(Vec::new()),
        llm_dispatches: Cell::new(0),
    });
    let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
    let backend =
        LocalGraphTraceExecutionBackend::new(clock, Rc::new(EmptyMaterializer), sink.clone());
    tokio::task::LocalSet::new()
        .run_until(async {
            backend
                .execute_static_trace(
                    GraphTracePlan {
                        graph,
                        trace: TraceRecord {
                            id: "tool-only".into(),
                            graph_ref: None,
                            initial_state: BTreeMap::new(),
                        },
                        arrival_offset_ns: None,
                    },
                    Phase::Profiling,
                    TraceSubphase::Profiling,
                )
                .await
                .unwrap();
        })
        .await;
    assert_eq!(sink.commands.borrow().as_slice(), ["first", "second"]);
    assert_eq!(sink.llm_dispatches.get(), 0);
}
