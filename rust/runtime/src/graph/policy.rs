// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trait-injected graph admission, ancillary timing, and failure policy.
//!
//! Resilient mode treats an
//! errored child as completed for join accounting; fail-fast aborts the trace
//! and tells the phase to stop admitting unrelated roots. Here those choices
//! are independent traits around the one [`crate::graph::executor::TraceExecutor`]
//! dispatch path—never a second scheduler.

use std::cell::{Cell, RefCell};
use std::fmt::{self, Display};
use std::rc::Rc;

use crate::timing::{CancellationPolicy, Phase, SlotGuard, SlotPool};
use async_trait::async_trait;
use tokio::sync::Notify;

use crate::graph::errors::TraceError;
use crate::graph::sink::{GraphDispatchOptions, GraphReplyStatus};

/// Immutable context supplied before one node enters backend admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeDispatchInfo {
    /// Trace identifier. `Rc<str>`-shared from the per-trace context so the
    /// fire path pointer-bumps rather than reallocating the id per node.
    pub trace_id: Rc<str>,
    /// Graph node identifier.
    pub node_id: String,
    /// Authored output-token limit.
    pub max_tokens: Option<usize>,
}

/// A permit held from node admission through terminal completion.
pub trait NodeDispatchPermit {
    /// Directives consumed by supporting sinks.
    fn options(&self) -> GraphDispatchOptions {
        GraphDispatchOptions::default()
    }

    /// First output token was observed. Prefill permits normally release here.
    fn on_first_token(&self) {}

    /// Dispatch reached a terminal classification. Implementations must provide
    /// first-token fallback release when no first token was observed.
    fn on_terminal(&self, _status: GraphReplyStatus) {}
}

/// Object-safe node admission and ancillary-policy seam.
#[async_trait(?Send)]
pub trait NodeDispatchPolicy {
    /// Acquire policy resources and derive per-request directives.
    async fn admit(
        &self,
        info: &NodeDispatchInfo,
    ) -> Result<Box<dyn NodeDispatchPermit>, GraphPolicyError>;
}

/// Policy that adds no admission or timing behavior.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopNodeDispatchPolicy;

struct NoopNodeDispatchPermit;

impl NodeDispatchPermit for NoopNodeDispatchPermit {}

#[async_trait(?Send)]
impl NodeDispatchPolicy for NoopNodeDispatchPolicy {
    async fn admit(
        &self,
        _info: &NodeDispatchInfo,
    ) -> Result<Box<dyn NodeDispatchPermit>, GraphPolicyError> {
        Ok(Box::new(NoopNodeDispatchPermit))
    }
}

/// Prefill admission over the shared dynamic [`SlotPool`].
///
/// The pool is intentionally exposed so the existing ramp and adaptive
/// actuators mutate the same live capacity consumed by Graph-IR nodes.
pub struct PrefillSlotNodePolicy {
    pool: Rc<SlotPool>,
}

impl PrefillSlotNodePolicy {
    /// Bind node prefill admission to a live slot pool.
    pub fn new(pool: Rc<SlotPool>) -> Self {
        Self { pool }
    }

    /// Clone the pool handle used by ramp/adaptive controllers.
    pub fn pool(&self) -> Rc<SlotPool> {
        self.pool.clone()
    }
}

struct PrefillSlotPermit {
    guard: RefCell<Option<SlotGuard>>,
}

impl NodeDispatchPermit for PrefillSlotPermit {
    fn on_first_token(&self) {
        self.guard.borrow_mut().take();
    }

    fn on_terminal(&self, _status: GraphReplyStatus) {
        self.guard.borrow_mut().take();
    }
}

#[async_trait(?Send)]
impl NodeDispatchPolicy for PrefillSlotNodePolicy {
    async fn admit(
        &self,
        _info: &NodeDispatchInfo,
    ) -> Result<Box<dyn NodeDispatchPermit>, GraphPolicyError> {
        Ok(Box::new(PrefillSlotPermit {
            guard: RefCell::new(Some(self.pool.acquire().await)),
        }))
    }
}

/// Cancellation decisions delegated to the shared timing policy.
pub struct CancellationNodePolicy {
    policy: RefCell<Box<dyn CancellationPolicy>>,
    phase: Cell<Phase>,
}

impl CancellationNodePolicy {
    /// Construct from one stateful cancellation policy and current phase.
    pub fn new(policy: Box<dyn CancellationPolicy>, phase: Phase) -> Self {
        Self {
            policy: RefCell::new(policy),
            phase: Cell::new(phase),
        }
    }

    /// Switch warmup/profiling behavior without rebuilding the graph executor.
    pub fn set_phase(&self, phase: Phase) {
        self.phase.set(phase);
    }
}

struct CancellationPermit {
    delay_ns: Option<i64>,
}

impl NodeDispatchPermit for CancellationPermit {
    fn options(&self) -> GraphDispatchOptions {
        GraphDispatchOptions {
            cancel_after_ns: self.delay_ns,
        }
    }
}

#[async_trait(?Send)]
impl NodeDispatchPolicy for CancellationNodePolicy {
    async fn admit(
        &self,
        _info: &NodeDispatchInfo,
    ) -> Result<Box<dyn NodeDispatchPermit>, GraphPolicyError> {
        Ok(Box::new(CancellationPermit {
            delay_ns: self
                .policy
                .borrow_mut()
                .next_cancel_delay_ns(self.phase.get()),
        }))
    }
}

/// Ordered composition of independent node policies.
pub struct CompositeNodeDispatchPolicy {
    policies: Vec<Rc<dyn NodeDispatchPolicy>>,
}

impl CompositeNodeDispatchPolicy {
    /// Construct an ordered policy fanout.
    pub fn new(policies: Vec<Rc<dyn NodeDispatchPolicy>>) -> Self {
        Self { policies }
    }
}

struct CompositeNodeDispatchPermit {
    permits: Vec<Box<dyn NodeDispatchPermit>>,
}

impl NodeDispatchPermit for CompositeNodeDispatchPermit {
    fn options(&self) -> GraphDispatchOptions {
        GraphDispatchOptions {
            cancel_after_ns: self
                .permits
                .iter()
                .filter_map(|permit| permit.options().cancel_after_ns)
                .min(),
        }
    }

    fn on_first_token(&self) {
        for permit in &self.permits {
            permit.on_first_token();
        }
    }

    fn on_terminal(&self, status: GraphReplyStatus) {
        for permit in &self.permits {
            permit.on_terminal(status);
        }
    }
}

#[async_trait(?Send)]
impl NodeDispatchPolicy for CompositeNodeDispatchPolicy {
    async fn admit(
        &self,
        info: &NodeDispatchInfo,
    ) -> Result<Box<dyn NodeDispatchPermit>, GraphPolicyError> {
        let mut permits = Vec::with_capacity(self.policies.len());
        for policy in &self.policies {
            permits.push(policy.admit(info).await?);
        }
        Ok(Box::new(CompositeNodeDispatchPermit { permits }))
    }
}

/// Failure observed at the node/backend boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeFailure {
    /// Trace identifier. `Rc<str>`-shared from the per-trace context.
    pub trace_id: Rc<str>,
    /// Node identifier.
    pub node_id: String,
    /// Stable failure classification.
    pub kind: NodeFailureKind,
    /// Backend or policy detail.
    pub message: String,
}

/// Stable node failure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeFailureKind {
    /// The sink returned an error rather than a terminal reply.
    Sink,
    /// The sink returned a failed terminal reply.
    FailedReply,
    /// The sink returned a cancelled terminal reply.
    CancelledReply,
    /// Node admission policy rejected the dispatch.
    Admission,
}

/// Per-node decision after a dispatch failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeFailureDisposition {
    /// Publish a type-correct empty value and let successors/join gates drain.
    ContinueWithEmpty,
    /// Abort the trace; sibling tasks observe the trace abort latch.
    AbortTrace,
}

/// Injectable per-node failure semantics.
pub trait NodeFailurePolicy {
    /// Choose how the executor handles one observed failure.
    fn on_failure(&self, failure: &NodeFailure) -> NodeFailureDisposition;
}

/// Failed children count as done and the DAG drains.
#[derive(Debug, Clone, Copy, Default)]
pub struct ResilientNodeFailurePolicy;

impl NodeFailurePolicy for ResilientNodeFailurePolicy {
    fn on_failure(&self, _failure: &NodeFailure) -> NodeFailureDisposition {
        NodeFailureDisposition::ContinueWithEmpty
    }
}

/// Trace-aborting node policy used by whole-run fail-fast execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct AbortTraceNodeFailurePolicy;

impl NodeFailurePolicy for AbortTraceNodeFailurePolicy {
    fn on_failure(&self, _failure: &NodeFailure) -> NodeFailureDisposition {
        NodeFailureDisposition::AbortTrace
    }
}

/// Run-level admission policy updated after every trace result.
#[async_trait(?Send)]
pub trait RunFailurePolicy {
    /// Whether a new root trace may be admitted.
    fn may_admit(&self) -> bool;

    /// Observe a drained trace result before the next admission decision.
    fn on_trace_result(&self, trace_id: &str, result: &Result<(), TraceError>);

    /// Wait until new-root admission becomes forbidden.
    async fn wait_blocked(&self) {
        std::future::pending::<()>().await;
    }
}

/// Keep admitting unrelated roots after trace failures.
#[derive(Debug, Clone, Copy, Default)]
pub struct ContinueRunFailurePolicy;

impl RunFailurePolicy for ContinueRunFailurePolicy {
    fn may_admit(&self) -> bool {
        true
    }

    fn on_trace_result(&self, _trace_id: &str, _result: &Result<(), TraceError>) {}
}

/// Stop admitting new roots after the first trace failure.
#[derive(Debug, Default)]
pub struct FailFastRunFailurePolicy {
    failed: Cell<bool>,
    first_failure: RefCell<Option<(String, TraceError)>>,
    blocked_notify: Rc<Notify>,
}

impl FailFastRunFailurePolicy {
    /// Whether the policy has latched a failure.
    pub fn failed(&self) -> bool {
        self.failed.get()
    }

    /// First failing trace and error, if any.
    pub fn first_failure(&self) -> Option<(String, TraceError)> {
        self.first_failure.borrow().clone()
    }
}

#[async_trait(?Send)]
impl RunFailurePolicy for FailFastRunFailurePolicy {
    fn may_admit(&self) -> bool {
        !self.failed.get()
    }

    fn on_trace_result(&self, trace_id: &str, result: &Result<(), TraceError>) {
        let Err(error) = result else { return };
        if matches!(error, TraceError::Cancelled(_)) {
            return;
        }
        if !self.failed.replace(true) {
            *self.first_failure.borrow_mut() = Some((trace_id.to_string(), error.clone()));
            self.blocked_notify.notify_waiters();
        }
    }

    async fn wait_blocked(&self) {
        loop {
            if !self.may_admit() {
                return;
            }
            self.blocked_notify.notified().await;
        }
    }
}

/// Graph policy construction/admission failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphPolicyError(pub String);

impl Display for GraphPolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GraphPolicyError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::RngRoot;
    use crate::timing::BernoulliFixedDelay;

    #[tokio::test]
    async fn composite_consumes_prefill_and_cancellation_on_the_same_node() {
        let slots = Rc::new(SlotPool::new(1));
        let prefill = Rc::new(PrefillSlotNodePolicy::new(slots.clone()));
        let cancellation = Rc::new(CancellationNodePolicy::new(
            Box::new(BernoulliFixedDelay::new(Some(100.0), 0.25, RngRoot::new(Some(1))).unwrap()),
            Phase::Profiling,
        ));
        let composite = CompositeNodeDispatchPolicy::new(vec![prefill, cancellation]);
        let permit = composite
            .admit(&NodeDispatchInfo {
                trace_id: "t".into(),
                node_id: "n".into(),
                max_tokens: Some(4),
            })
            .await
            .unwrap();

        assert!(slots.locked());
        assert_eq!(permit.options().cancel_after_ns, Some(250_000_000));
        permit.on_first_token();
        assert!(!slots.locked());
        permit.on_terminal(GraphReplyStatus::Completed);
        assert_eq!(
            slots.stats().release_count,
            1,
            "terminal fallback is idempotent"
        );
    }

    #[test]
    fn fail_fast_latches_only_the_first_trace_failure() {
        let policy = FailFastRunFailurePolicy::default();
        let first = Err(TraceError::Other("first".into()));
        policy.on_trace_result("a", &first);
        policy.on_trace_result("b", &Err(TraceError::Other("second".into())));

        assert!(!policy.may_admit());
        assert_eq!(
            policy.first_failure(),
            Some(("a".into(), TraceError::Other("first".into())))
        );
    }

    #[test]
    fn fail_fast_does_not_turn_configured_cancellation_into_run_failure() {
        let policy = FailFastRunFailurePolicy::default();
        policy.on_trace_result(
            "cancelled",
            &Err(TraceError::Cancelled("configured cancellation".into())),
        );

        assert!(policy.may_admit());
        assert_eq!(policy.first_failure(), None);
    }
}
