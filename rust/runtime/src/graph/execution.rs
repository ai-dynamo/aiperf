// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-trace execution and placement boundaries.
//!
//! The coordinator submits one complete [`GraphTraceProgram`] through
//! [`TracePlacement`]. A backend may execute locally, place traces
//! on thread-per-core workers, or serialize commands to remote workers. Node
//! turns never cross this boundary: one selected worker owns every firing gate,
//! branch, join, channel version, and dynamic reply splice for the trace.

use std::cell::{Cell, RefCell};
use std::rc::{Rc, Weak};

use crate::clock::Clock;
use async_trait::async_trait;

use crate::graph::context::TraceContext;
use crate::graph::errors::TraceError;
use crate::graph::executor::{ExecutorFlags, TraceExecutor};
use crate::graph::materialize::PromptMaterializer;
use crate::graph::model::GraphTraceProgram;
use crate::graph::policy::{
    NodeDispatchPolicy, NodeFailurePolicy, NoopNodeDispatchPolicy, ResilientNodeFailurePolicy,
};
use crate::graph::runtime::Handle;
use crate::graph::sink::{GraphSink, TraceSubphase};
use crate::graph::wire::WireMessage;
use crate::metrics_core::Phase;

/// Object-safe backend for one complete root trace.
///
/// Remote implementations can serialize [`GraphTraceProgram`] and return the
/// terminal result without changing workload scheduling. Dense segment handles
/// name an immutable catalog that the backend must provision before execution.
#[async_trait(?Send)]
pub trait TracePlacement {
    /// Execute one complete trace on one placement target.
    async fn execute_trace(&self, program: GraphTraceProgram) -> Result<(), TraceError>;

    /// Gracefully cancel every trace currently executing through this backend.
    ///
    /// Implementations latch cancellation before returning, arrange for active
    /// [`execute_trace`](Self::execute_trace) futures to reach their normal
    /// terminal cleanup and return [`TraceError::Cancelled`], and reject future
    /// traces. The hook must be idempotent. Placement queues are drained by the
    /// placement implementation rather than by worker-local backends.
    fn cancel_inflight(&self) -> Result<(), TraceError> {
        Err(TraceError::Other(
            "graph execution backend does not expose graceful cancellation".into(),
        ))
    }

    /// Update the placement-wide node-prefill limit.
    ///
    /// The default fails closed because silently accepting an actuator update
    /// would make graph ramps and adaptive control inert. Placements that own
    /// worker-local admission pools receive deterministic shards of the global
    /// limit. A worker-local zero is valid and disables new prefill admission
    /// on that shard; the public placement boundary still rejects a global zero.
    fn set_prefill_limit(&self, _limit: usize) -> Result<(), TraceError> {
        Err(TraceError::Other(
            "graph execution placement does not expose prefill control".into(),
        ))
    }
}

/// Local implementation backed by the canonical [`TraceExecutor`].
pub struct LocalGraphTraceExecutionBackend<M: WireMessage> {
    clock: Rc<dyn Clock>,
    materializer: Rc<dyn PromptMaterializer>,
    sink: Rc<dyn GraphSink<M>>,
    node_policy: Rc<dyn NodeDispatchPolicy>,
    node_failure: Rc<dyn NodeFailurePolicy>,
    flags: ExecutorFlags,
    cancelled: Cell<bool>,
    active: RefCell<Vec<Weak<TraceContext>>>,
    // Forces the general executor when enabled.
    force_full: bool,
    flat_aborts: RefCell<Vec<Weak<crate::graph::flat::FlatAbort>>>,
    /// Test probe: set true when the general executor arm runs.
    #[cfg(test)]
    executor_built: Cell<bool>,
}

impl<M: WireMessage> LocalGraphTraceExecutionBackend<M> {
    /// Construct with no-op node admission and resilient node failures.
    pub fn new(
        clock: Rc<dyn Clock>,
        materializer: Rc<dyn PromptMaterializer>,
        sink: Rc<dyn GraphSink<M>>,
    ) -> Self {
        Self {
            clock,
            materializer,
            sink,
            node_policy: Rc::new(NoopNodeDispatchPolicy),
            node_failure: Rc::new(ResilientNodeFailurePolicy),
            flags: ExecutorFlags::default(),
            cancelled: Cell::new(false),
            active: RefCell::new(Vec::new()),
            force_full: false,
            flat_aborts: RefCell::new(Vec::new()),
            #[cfg(test)]
            executor_built: Cell::new(false),
        }
    }

    /// Force the general executor for every plan.
    #[cfg(test)]
    pub(crate) fn with_force_full(mut self, force_full: bool) -> Self {
        self.force_full = force_full;
        self
    }

    /// Inject node prefill/cancellation policy.
    pub fn with_node_policy(mut self, policy: Rc<dyn NodeDispatchPolicy>) -> Self {
        self.node_policy = policy;
        self
    }

    /// Inject node failure handling.
    pub fn with_node_failure(mut self, policy: Rc<dyn NodeFailurePolicy>) -> Self {
        self.node_failure = policy;
        self
    }

    /// Inject edge-timing flags.
    pub fn with_executor_flags(mut self, flags: ExecutorFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Execute a static trace in one explicit lifecycle partition.
    pub async fn execute_static_trace(
        &self,
        plan: crate::graph::model::GraphTracePlan,
        phase: Phase,
        trace_subphase: TraceSubphase,
    ) -> Result<(), TraceError> {
        if self.cancelled.get() {
            return Err(local_cancellation(&plan.trace.id));
        }

        // The flat path predates typed dispatch context, so retain it only for
        // ordinary profiling. Trace-local warmup always takes the canonical
        // executor path and consequently preserves its record partition.
        if trace_subphase == TraceSubphase::Profiling
            && !self.force_full
            && !crate::graph::flat::flatgraph_disabled()
            && crate::graph::flat::is_flat_graph(&plan.graph)
        {
            let trace_id = plan.trace.id.clone();
            tracing::debug!(trace_id = %trace_id, "graph flat fast path");
            let abort = crate::graph::flat::FlatAbort::new();
            self.flat_aborts.borrow_mut().push(Rc::downgrade(&abort));
            let actor = crate::graph::flat::FlatGraphActor::new(
                self.sink.clone(),
                self.materializer.clone(),
                self.node_policy.clone(),
                self.node_failure.clone(),
            );
            let result = actor.run(plan, &abort).await;
            self.flat_aborts
                .borrow_mut()
                .retain(|weak| !weak.ptr_eq(&Rc::downgrade(&abort)));
            return match result {
                Ok(()) if abort.is_tripped() => Err(local_cancellation(&trace_id)),
                other => other,
            };
        }

        #[cfg(test)]
        self.executor_built.set(true);
        let handle = Handle::new(self.clock.clone());
        let executor = TraceExecutor::new_with_policies_and_dispatch_context(
            Rc::new(plan.graph),
            self.materializer.clone(),
            self.sink.clone(),
            self.node_policy.clone(),
            self.node_failure.clone(),
            handle.clone(),
            self.flags,
            phase,
            trace_subphase,
        )?;
        let context = executor.build_context(plan.trace)?;
        self.active.borrow_mut().push(Rc::downgrade(&context));
        executor.schedule_entries(&context);
        handle.wait_idle().await;
        self.active
            .borrow_mut()
            .retain(|active| active.as_ptr() != Rc::as_ptr(&context));
        context.abort.borrow().clone().map_or_else(|| Ok(()), Err)
    }

    /// Execute one static trace and retain its reduced channel snapshot.
    ///
    /// A staged trace driver needs immutable channel facts for its next bounded
    /// decision. This deliberately uses the canonical executor rather than the
    /// flat fast path because the latter predates channel-result ownership.
    pub async fn execute_static_trace_result(
        &self,
        plan: crate::graph::model::GraphTracePlan,
        phase: Phase,
        trace_subphase: TraceSubphase,
    ) -> Result<crate::graph::executor::TraceResult, TraceError> {
        if self.cancelled.get() {
            return Err(local_cancellation(&plan.trace.id));
        }
        let handle = Handle::new(self.clock.clone());
        let executor = TraceExecutor::new_with_policies_and_dispatch_context(
            Rc::new(plan.graph),
            self.materializer.clone(),
            self.sink.clone(),
            self.node_policy.clone(),
            self.node_failure.clone(),
            handle.clone(),
            self.flags,
            phase,
            trace_subphase,
        )?;
        let context = executor.build_context(plan.trace)?;
        self.active.borrow_mut().push(Rc::downgrade(&context));
        executor.schedule_entries(&context);
        handle.wait_idle().await;
        self.active
            .borrow_mut()
            .retain(|active| active.as_ptr() != Rc::as_ptr(&context));
        context
            .abort
            .borrow()
            .clone()
            .map_or_else(|| TraceExecutor::<M>::result(&context), Err)
    }
}

#[async_trait(?Send)]
impl<M: WireMessage + 'static> TracePlacement for LocalGraphTraceExecutionBackend<M> {
    async fn execute_trace(&self, program: GraphTraceProgram) -> Result<(), TraceError> {
        if !program.is_static_graph_program() {
            return Err(TraceError::UnsupportedDriver(program.driver.kind));
        }
        if let Some((node_id, _)) = program
            .profiling
            .graph
            .nodes
            .iter()
            .find(|(_, node)| matches!(node, crate::graph::model::ExecutableGraphNode::Tool(_)))
        {
            return Err(TraceError::UnsupportedNode {
                node_id: node_id.clone(),
                kind: "tool",
            });
        }
        self.execute_static_trace(
            program.profiling,
            Phase::Profiling,
            TraceSubphase::Profiling,
        )
        .await
    }

    fn cancel_inflight(&self) -> Result<(), TraceError> {
        self.cancelled.set(true);
        let active = self
            .active
            .borrow()
            .iter()
            .filter_map(Weak::upgrade)
            .collect::<Vec<_>>();
        for context in active {
            context.set_abort(local_cancellation(&context.trace.id));
        }
        for abort in self.flat_aborts.borrow().iter().filter_map(Weak::upgrade) {
            abort.trip();
        }
        Ok(())
    }
}

fn local_cancellation(trace_id: &str) -> TraceError {
    TraceError::Cancelled(format!(
        "graph trace {trace_id:?} was cancelled by its execution backend"
    ))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use crate::timing::SlotPool;

    use super::*;
    use crate::graph::model::{
        ChannelSpec, ChannelType, ExecutableGraphNode, GraphRecord, GraphTracePlan,
        GraphTraceProgram, LlmNode, ReducerName, START_NODE_ID, StaticEdge, ToolNode, TraceRecord,
    };
    use crate::graph::policy::PrefillSlotNodePolicy;
    use crate::graph::sink::{EchoSink, GraphReply, GraphSink};
    use crate::graph::wire::OpenAiChatMessage;

    struct EmptyMaterializer;

    impl PromptMaterializer for EmptyMaterializer {
        fn build(
            &self,
            _node: &LlmNode,
            _inputs: &BTreeMap<String, crate::graph::reducers::ChanVal>,
        ) -> Result<Vec<bytes::Bytes>, crate::dataset::DatasetError> {
            Ok(Vec::new())
        }
    }

    fn blocked_plan(id: &str) -> GraphTraceProgram {
        let mut graph = GraphRecord::default();
        graph.state.insert(
            "output".into(),
            ChannelSpec {
                channel_type: ChannelType::Messages,
                reducer: ReducerName::AddMessages,
            },
        );
        graph.nodes.insert(
            "blocked".into(),
            ExecutableGraphNode::Llm(LlmNode {
                output: "output".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: Vec::new(),
                request: None,
                metadata: BTreeMap::new(),
            }),
        );
        graph.edges.push(StaticEdge {
            source: START_NODE_ID.into(),
            target: "blocked".into(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        });
        GraphTraceProgram::static_graph(GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        })
    }

    struct FirstTokenParkingSink {
        calls: Rc<RefCell<Vec<String>>>,
        parent_first_token: Rc<tokio::sync::Notify>,
        parent_release: Rc<tokio::sync::Notify>,
    }

    struct NoTokenFailureSink {
        calls: Rc<RefCell<Vec<String>>>,
    }

    #[async_trait(?Send)]
    impl GraphSink<OpenAiChatMessage> for NoTokenFailureSink {
        async fn dispatch(
            &self,
            node_id: &str,
            _messages: Vec<bytes::Bytes>,
            _max_tokens: Option<usize>,
            _on_first_token: &dyn Fn(),
        ) -> anyhow::Result<GraphReply<OpenAiChatMessage>> {
            self.calls.borrow_mut().push(node_id.to_owned());
            if node_id == "parent" {
                Ok(GraphReply::failed())
            } else {
                Ok(GraphReply::from_text(node_id.to_owned()))
            }
        }
    }

    #[async_trait(?Send)]
    impl GraphSink<OpenAiChatMessage> for FirstTokenParkingSink {
        async fn dispatch(
            &self,
            node_id: &str,
            _messages: Vec<bytes::Bytes>,
            _max_tokens: Option<usize>,
            on_first_token: &dyn Fn(),
        ) -> anyhow::Result<GraphReply<OpenAiChatMessage>> {
            self.calls.borrow_mut().push(node_id.to_owned());
            on_first_token();
            if node_id == "parent" {
                on_first_token();
                self.parent_first_token.notify_one();
                self.parent_release.notified().await;
            }
            Ok(GraphReply::from_text(node_id.to_owned()))
        }
    }

    #[test]
    fn first_token_successor_dispatches_once_before_parent_terminal_release() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(tokio::task::LocalSet::new().run_until(async {
            let mut graph = GraphRecord::default();
            for node_id in ["parent", "child"] {
                graph.state.insert(
                    format!("{node_id}_output"),
                    ChannelSpec {
                        channel_type: ChannelType::Messages,
                        reducer: ReducerName::AddMessages,
                    },
                );
                graph.nodes.insert(
                    node_id.to_owned(),
                    ExecutableGraphNode::Llm(LlmNode {
                        output: format!("{node_id}_output"),
                        streaming: true,
                        inputs: Vec::new(),
                        min_start_delay_us: None,
                        max_tokens: Some(1),
                        items: Vec::new(),
                        request: None,
                        metadata: BTreeMap::new(),
                    }),
                );
            }
            graph.edges = vec![
                StaticEdge {
                    source: START_NODE_ID.into(),
                    target: "parent".into(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: None,
                },
                StaticEdge {
                    source: "parent".into(),
                    target: "child".into(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: Some(0.0),
                },
            ];
            let calls = Rc::new(RefCell::new(Vec::new()));
            let parent_first_token = Rc::new(tokio::sync::Notify::new());
            let parent_release = Rc::new(tokio::sync::Notify::new());
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(FirstTokenParkingSink {
                calls: calls.clone(),
                parent_first_token: parent_first_token.clone(),
                parent_release: parent_release.clone(),
            });
            let backend = Rc::new(
                LocalGraphTraceExecutionBackend::new(
                    Rc::new(crate::clock::SimClock::new()),
                    Rc::new(EmptyMaterializer),
                    sink,
                )
                .with_force_full(true),
            );
            let executing = backend.clone();
            let task = tokio::task::spawn_local(async move {
                executing
                    .execute_static_trace(
                        GraphTracePlan {
                            graph,
                            trace: TraceRecord {
                                id: "first-token".into(),
                                graph_ref: None,
                                initial_state: BTreeMap::new(),
                            },
                            arrival_offset_ns: None,
                        },
                        Phase::Profiling,
                        TraceSubphase::Profiling,
                    )
                    .await
            });
            parent_first_token.notified().await;
            tokio::task::yield_now().await;
            assert_eq!(calls.borrow().as_slice(), ["parent", "child"]);
            parent_release.notify_one();
            task.await.unwrap().unwrap();
        }));
    }

    #[test]
    fn first_token_successor_falls_back_to_resilient_parent_terminal() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(tokio::task::LocalSet::new().run_until(async {
            let mut graph = GraphRecord::default();
            for node_id in ["parent", "child"] {
                graph.state.insert(
                    format!("{node_id}_output"),
                    ChannelSpec {
                        channel_type: ChannelType::Messages,
                        reducer: ReducerName::AddMessages,
                    },
                );
                graph.nodes.insert(
                    node_id.to_owned(),
                    ExecutableGraphNode::Llm(LlmNode {
                        output: format!("{node_id}_output"),
                        streaming: true,
                        inputs: Vec::new(),
                        min_start_delay_us: None,
                        max_tokens: Some(1),
                        items: Vec::new(),
                        request: None,
                        metadata: BTreeMap::new(),
                    }),
                );
            }
            graph.edges = vec![
                StaticEdge {
                    source: START_NODE_ID.into(),
                    target: "parent".into(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: None,
                },
                StaticEdge {
                    source: "parent".into(),
                    target: "child".into(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: Some(0.0),
                },
            ];
            let calls = Rc::new(RefCell::new(Vec::new()));
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(NoTokenFailureSink {
                calls: calls.clone(),
            });
            LocalGraphTraceExecutionBackend::new(
                Rc::new(crate::clock::SimClock::new()),
                Rc::new(EmptyMaterializer),
                sink,
            )
            .with_force_full(true)
            .execute_static_trace(
                GraphTracePlan {
                    graph,
                    trace: TraceRecord {
                        id: "first-token-terminal".into(),
                        graph_ref: None,
                        initial_state: BTreeMap::new(),
                    },
                    arrival_offset_ns: None,
                },
                Phase::Profiling,
                TraceSubphase::Profiling,
            )
            .await
            .expect("resilient failure must drain successors");
            assert_eq!(calls.borrow().as_slice(), ["parent", "child"]);
        }));
    }

    #[test]
    fn abort_interrupts_graph_firing_gate() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(tokio::task::LocalSet::new().run_until(async {
            let mut graph = GraphRecord::default();
            for node_id in ["parent", "child"] {
                graph.state.insert(
                    format!("{node_id}_output"),
                    ChannelSpec {
                        channel_type: ChannelType::Messages,
                        reducer: ReducerName::AddMessages,
                    },
                );
                graph.nodes.insert(
                    node_id.to_owned(),
                    ExecutableGraphNode::Llm(LlmNode {
                        output: format!("{node_id}_output"),
                        streaming: true,
                        inputs: Vec::new(),
                        min_start_delay_us: None,
                        max_tokens: Some(1),
                        items: Vec::new(),
                        request: None,
                        metadata: BTreeMap::new(),
                    }),
                );
            }
            graph.edges = vec![
                StaticEdge {
                    source: START_NODE_ID.into(),
                    target: "parent".into(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: None,
                },
                StaticEdge {
                    source: "parent".into(),
                    target: "child".into(),
                    delay_after_predecessor_us: None,
                    min_start_delay_us: None,
                    delay_after_predecessor_start_us: None,
                    delay_after_predecessor_first_token_us: Some(1_000_000.0),
                },
            ];
            let clock = Rc::new(crate::clock::SimClock::new());
            let calls = Rc::new(RefCell::new(Vec::new()));
            let parent_first_token = Rc::new(tokio::sync::Notify::new());
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(FirstTokenParkingSink {
                calls: calls.clone(),
                parent_first_token: parent_first_token.clone(),
                parent_release: Rc::new(tokio::sync::Notify::new()),
            });
            let backend = Rc::new(
                LocalGraphTraceExecutionBackend::new(
                    clock.clone(),
                    Rc::new(EmptyMaterializer),
                    sink,
                )
                .with_force_full(true),
            );
            let executing = backend.clone();
            let task = tokio::task::spawn_local(async move {
                executing
                    .execute_static_trace(
                        GraphTracePlan {
                            graph,
                            trace: TraceRecord {
                                id: "abort-firing-gate".into(),
                                graph_ref: None,
                                initial_state: BTreeMap::new(),
                            },
                            arrival_offset_ns: None,
                        },
                        Phase::Profiling,
                        TraceSubphase::Profiling,
                    )
                    .await
            });
            parent_first_token.notified().await;
            for _ in 0..8 {
                if clock.scheduled_count() == 1 {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(calls.borrow().as_slice(), ["parent"]);
            assert_eq!(clock.scheduled_count(), 1);
            assert_eq!(clock.next_event_time(), Some(1_000_000_000));

            backend.cancel_inflight().unwrap();
            tokio::task::yield_now().await;

            assert!(task.is_finished(), "abort must interrupt the firing delay");
            let error = task.await.unwrap().unwrap_err();
            assert!(matches!(error, TraceError::Cancelled(_)));
            assert_eq!(calls.borrow().as_slice(), ["parent"]);
            assert!(clock.now_ns() < 1_000_000_000);
        }));
    }

    #[test]
    fn local_backend_latches_cancellation_and_drains_active_context() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(tokio::task::LocalSet::new().run_until(async {
            let clock: Rc<dyn Clock> = Rc::new(crate::clock::SimClock::new());
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(EchoSink);
            let backend = Rc::new(
                LocalGraphTraceExecutionBackend::new(clock, Rc::new(EmptyMaterializer), sink)
                    .with_node_policy(Rc::new(PrefillSlotNodePolicy::new(Rc::new(SlotPool::new(
                        0,
                    )))))
                    // Pin to the general executor so this test keeps exercising the
                    // active-context drain path its name describes.
                    .with_force_full(true),
            );
            let executing = backend.clone();
            let task = tokio::task::spawn_local(async move {
                executing.execute_trace(blocked_plan("active")).await
            });
            tokio::task::yield_now().await;

            backend.cancel_inflight().unwrap();
            let error = task.await.unwrap().unwrap_err();
            assert!(matches!(error, TraceError::Cancelled(_)));

            let error = backend
                .execute_trace(blocked_plan("after-cancel"))
                .await
                .unwrap_err();
            assert!(matches!(error, TraceError::Cancelled(_)));
        }));
    }

    #[test]
    fn local_backend_refuses_tool_nodes_before_flat_or_inference_dispatch() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(tokio::task::LocalSet::new().run_until(async {
            let clock: Rc<dyn Clock> = Rc::new(crate::clock::SimClock::new());
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(EchoSink);
            let backend =
                LocalGraphTraceExecutionBackend::new(clock, Rc::new(EmptyMaterializer), sink);
            let mut graph = GraphRecord::default();
            graph.nodes.insert(
                "tool".into(),
                ExecutableGraphNode::Tool(ToolNode {
                    output: "observation".into(),
                    commands: vec!["pwd".into()],
                    timeout_ns: None,
                }),
            );
            let error = backend
                .execute_trace(GraphTraceProgram::static_graph(GraphTracePlan {
                    graph,
                    trace: TraceRecord {
                        id: "tool-trace".into(),
                        graph_ref: None,
                        initial_state: BTreeMap::new(),
                    },
                    arrival_offset_ns: None,
                }))
                .await
                .unwrap_err();
            assert!(matches!(
                error,
                TraceError::UnsupportedNode { ref node_id, kind: "tool" } if node_id == "tool"
            ));
        }));
    }

    #[test]
    fn flat_arm_selected_for_single_node_and_force_full_uses_executor() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(tokio::task::LocalSet::new().run_until(async {
            let clock: Rc<dyn Clock> = Rc::new(crate::clock::SimClock::new());
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(EchoSink);

            // Default policy admits immediately; a 1-node no-input plan completes
            // through the flat arm without building a TraceExecutor.
            let flat = Rc::new(LocalGraphTraceExecutionBackend::new(
                clock.clone(),
                Rc::new(EmptyMaterializer),
                sink.clone(),
            ));
            flat.execute_trace(blocked_plan("flat")).await.unwrap();
            assert!(
                !flat.executor_built.get(),
                "single-node plan takes the flat arm"
            );

            let full = Rc::new(
                LocalGraphTraceExecutionBackend::new(clock, Rc::new(EmptyMaterializer), sink)
                    .with_force_full(true),
            );
            full.execute_trace(blocked_plan("full")).await.unwrap();
            assert!(
                full.executor_built.get(),
                "force_full routes to the executor"
            );
        }));
    }

    #[test]
    fn flat_arm_latches_cancellation() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(tokio::task::LocalSet::new().run_until(async {
            let clock: Rc<dyn Clock> = Rc::new(crate::clock::SimClock::new());
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(EchoSink);
            // 0 prefill slots block admission; the only exit is the flat abort latch.
            let backend = Rc::new(
                LocalGraphTraceExecutionBackend::new(clock, Rc::new(EmptyMaterializer), sink)
                    .with_node_policy(Rc::new(PrefillSlotNodePolicy::new(Rc::new(SlotPool::new(
                        0,
                    ))))),
            );
            let executing = backend.clone();
            let task = tokio::task::spawn_local(async move {
                executing.execute_trace(blocked_plan("flat-cancel")).await
            });
            tokio::task::yield_now().await;
            backend.cancel_inflight().unwrap();
            let error = task.await.unwrap().unwrap_err();
            assert!(
                matches!(error, TraceError::Cancelled(_)),
                "flat arm reports Cancelled like the executor arm"
            );
            assert!(
                !backend.executor_built.get(),
                "cancellation stayed on the flat arm"
            );
        }));
    }
}
