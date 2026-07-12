// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whole-trace execution and placement boundaries.
//!
//! The coordinator submits one complete [`GraphTracePlan`] through
//! [`GraphTraceExecutionBackend`]. A backend may execute locally, place traces
//! on thread-per-core workers, or serialize commands to remote workers. Node
//! turns never cross this boundary: one selected worker owns every firing gate,
//! branch, join, channel version, and dynamic reply splice for the trace.

use std::rc::Rc;

use aiperf_clock::Clock;
use async_trait::async_trait;

use crate::errors::TraceError;
use crate::executor::{ExecutorFlags, TraceExecutor};
use crate::materialize::PromptMaterializer;
use crate::model::GraphTracePlan;
use crate::policy::{
    NodeDispatchPolicy, NodeFailurePolicy, NoopNodeDispatchPolicy, ResilientNodeFailurePolicy,
};
use crate::runtime::Handle;
use crate::sink::GraphSink;
use crate::wire::WireMessage;

/// Object-safe backend for one complete root trace.
///
/// Remote implementations can serialize [`GraphTracePlan`] and return the
/// terminal result without changing workload scheduling. Dense segment handles
/// name an immutable catalog that the backend must provision before execution.
#[async_trait(?Send)]
pub trait GraphTraceExecutionBackend {
    /// Execute one complete trace on one placement target.
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError>;
}

/// Local implementation backed by the canonical [`TraceExecutor`].
pub struct LocalGraphTraceExecutionBackend<M: WireMessage> {
    clock: Rc<dyn Clock>,
    materializer: Rc<dyn PromptMaterializer>,
    sink: Rc<dyn GraphSink<M>>,
    node_policy: Rc<dyn NodeDispatchPolicy>,
    node_failure: Rc<dyn NodeFailurePolicy>,
    flags: ExecutorFlags,
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
        }
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
}

#[async_trait(?Send)]
impl<M: WireMessage + 'static> GraphTraceExecutionBackend for LocalGraphTraceExecutionBackend<M> {
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
        let handle = Handle::new(self.clock.clone());
        let executor = TraceExecutor::new_with_policies(
            Rc::new(plan.graph),
            self.materializer.clone(),
            self.sink.clone(),
            self.node_policy.clone(),
            self.node_failure.clone(),
            handle.clone(),
            self.flags,
        )?;
        let context = executor.build_context(plan.trace)?;
        executor.schedule_entries(&context);
        handle.wait_idle().await;
        context.abort.borrow().clone().map_or_else(|| Ok(()), Err)
    }
}
