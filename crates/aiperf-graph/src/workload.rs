// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Phase-ready multi-trace Graph-IR workload policy around the one executor.
//!
//! Arrival pacing, root-session admission, phase observation, node policy, and
//! run-failure behavior are independent traits. Every admitted trace still
//! dispatches exclusively through [`crate::executor::TraceExecutor`]; this
//! module does not implement a second benchmark or backend path.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::error::Error;
use std::fmt::{self, Display};
use std::rc::Rc;

use aiperf_clock::Clock;
use aiperf_timing::{IntervalGenerator, SlotGuard, SlotPool};
use async_trait::async_trait;

use crate::errors::TraceError;
use crate::execution::GraphTraceExecutionBackend;
pub use crate::model::GraphTracePlan;
use crate::policy::{ContinueRunFailurePolicy, RunFailurePolicy};

/// Stateful root-trace selection seam.
pub trait GraphTraceSource {
    /// Return the next trace plan, or `None` when sending is complete.
    fn next_trace(&self) -> Result<Option<GraphTracePlan>, GraphWorkloadError>;
}

/// Authored-order finite trace source.
pub struct VecGraphTraceSource {
    plans: RefCell<VecDeque<GraphTracePlan>>,
}

impl VecGraphTraceSource {
    /// Construct from plans in desired admission order.
    pub fn new(plans: impl IntoIterator<Item = GraphTracePlan>) -> Self {
        Self {
            plans: RefCell::new(plans.into_iter().collect()),
        }
    }
}

impl GraphTraceSource for VecGraphTraceSource {
    fn next_trace(&self) -> Result<Option<GraphTracePlan>, GraphWorkloadError> {
        Ok(self.plans.borrow_mut().pop_front())
    }
}

/// Build a finite source from dataset-lowered graph references.
pub fn lowered_trace_source(
    lowered: &crate::dataset_lowering::LoweredDatasetGraph,
) -> VecGraphTraceSource {
    VecGraphTraceSource::new(lowered.parsed.traces.iter().map(|trace| GraphTracePlan {
        graph: lowered.parsed.resolve_trace_graph(trace).clone(),
        trace: trace.clone(),
        arrival_offset_ns: None,
    }))
}

/// Arrival-pacing extension point.
#[async_trait(?Send)]
pub trait GraphArrivalPolicy {
    /// Wait until `plan` may arrive. `run_start_ns` anchors authored offsets.
    async fn wait_for_arrival(
        &self,
        clock: Rc<dyn Clock>,
        run_start_ns: i64,
        ordinal: u64,
        plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError>;
}

/// Immediate arrivals; session capacity governs throughput.
#[derive(Debug, Clone, Copy, Default)]
pub struct ImmediateGraphArrival;

#[async_trait(?Send)]
impl GraphArrivalPolicy for ImmediateGraphArrival {
    async fn wait_for_arrival(
        &self,
        _clock: Rc<dyn Clock>,
        _run_start_ns: i64,
        _ordinal: u64,
        _plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError> {
        Ok(())
    }
}

/// Clock-native authored-offset arrivals.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScheduledGraphArrival;

#[async_trait(?Send)]
impl GraphArrivalPolicy for ScheduledGraphArrival {
    async fn wait_for_arrival(
        &self,
        clock: Rc<dyn Clock>,
        run_start_ns: i64,
        _ordinal: u64,
        plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError> {
        let Some(offset) = plan.arrival_offset_ns else {
            return Ok(());
        };
        if offset < 0 {
            return Err(GraphWorkloadError(format!(
                "trace {:?} has negative arrival offset {offset}ns",
                plan.trace.id
            )));
        }
        let target = run_start_ns.saturating_add(offset);
        let delay_ns = target.saturating_sub(clock.now_ns());
        clock.sleep(delay_ns).await;
        Ok(())
    }
}

/// Interval-generator arrivals whose live rate is shared with ramp/adaptive controls.
pub struct IntervalGraphArrival {
    generator: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    next_at_ns: Cell<Option<i64>>,
}

impl IntervalGraphArrival {
    /// Bind to a live interval generator.
    pub fn new(generator: Rc<RefCell<Box<dyn IntervalGenerator>>>) -> Self {
        Self {
            generator,
            next_at_ns: Cell::new(None),
        }
    }

    /// Clone the generator handle used by request-rate ramp/adaptive actuators.
    pub fn generator(&self) -> Rc<RefCell<Box<dyn IntervalGenerator>>> {
        self.generator.clone()
    }
}

#[async_trait(?Send)]
impl GraphArrivalPolicy for IntervalGraphArrival {
    async fn wait_for_arrival(
        &self,
        clock: Rc<dyn Clock>,
        run_start_ns: i64,
        ordinal: u64,
        _plan: &GraphTracePlan,
    ) -> Result<(), GraphWorkloadError> {
        let target = if ordinal == 0 {
            run_start_ns
        } else {
            self.next_at_ns
                .get()
                .unwrap_or(run_start_ns)
                .saturating_add(self.generator.borrow_mut().next_interval_ns().max(0))
        };
        self.next_at_ns.set(Some(target));
        let delay_ns = target.saturating_sub(clock.now_ns());
        clock.sleep(delay_ns).await;
        Ok(())
    }
}

/// Immutable context supplied to root-session admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceAdmissionInfo {
    /// Trace identifier.
    pub trace_id: String,
    /// Static node/request count in the trace.
    pub node_count: usize,
    /// Clock time at which arrival pacing completed.
    pub arrival_ns: i64,
}

/// Permit held for the complete root trace, including every child node.
pub trait TraceAdmissionPermit {}

/// Root-session admission seam. DAG children never reacquire this permit.
#[async_trait(?Send)]
pub trait TraceAdmissionPolicy {
    /// Acquire capacity for one whole trace.
    async fn acquire(
        &self,
        info: &TraceAdmissionInfo,
    ) -> Result<Box<dyn TraceAdmissionPermit>, GraphWorkloadError>;
}

/// Admission policy with no cap.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnlimitedTraceAdmission;

struct UnlimitedTracePermit;

impl TraceAdmissionPermit for UnlimitedTracePermit {}

#[async_trait(?Send)]
impl TraceAdmissionPolicy for UnlimitedTraceAdmission {
    async fn acquire(
        &self,
        _info: &TraceAdmissionInfo,
    ) -> Result<Box<dyn TraceAdmissionPermit>, GraphWorkloadError> {
        Ok(Box::new(UnlimitedTracePermit))
    }
}

/// Dynamic root-session cap over the shared timing [`SlotPool`].
pub struct SlotPoolTraceAdmission {
    pool: Rc<SlotPool>,
}

impl SlotPoolTraceAdmission {
    /// Bind whole-trace admission to a live pool.
    pub fn new(pool: Rc<SlotPool>) -> Self {
        Self { pool }
    }

    /// Clone the pool used by phase resources, ramps, and adaptive control.
    pub fn pool(&self) -> Rc<SlotPool> {
        self.pool.clone()
    }
}

struct SlotPoolTracePermit {
    _guard: SlotGuard,
}

impl TraceAdmissionPermit for SlotPoolTracePermit {}

#[async_trait(?Send)]
impl TraceAdmissionPolicy for SlotPoolTraceAdmission {
    async fn acquire(
        &self,
        _info: &TraceAdmissionInfo,
    ) -> Result<Box<dyn TraceAdmissionPermit>, GraphWorkloadError> {
        Ok(Box::new(SlotPoolTracePermit {
            _guard: self.pool.acquire().await,
        }))
    }
}

/// Phase/run hooks emitted by the graph workload.
pub trait GraphWorkloadObserver {
    /// Arrival pacing completed.
    fn on_trace_arrival(&self, _info: &TraceAdmissionInfo) {}
    /// Root-session admission completed.
    fn on_trace_admit(&self, _info: &TraceAdmissionInfo, _admit_ns: i64) {}
    /// One trace drained through the executor.
    fn on_trace_complete(&self, _result: &GraphTraceRunResult) {}
    /// The finite source stopped producing new roots.
    fn on_sending_complete(&self, _at_ns: i64) {}
}

/// No-op workload observer.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopGraphWorkloadObserver;

impl GraphWorkloadObserver for NoopGraphWorkloadObserver {}

/// One drained trace outcome.
#[derive(Debug, Clone)]
pub struct GraphTraceRunResult {
    /// Trace identifier.
    pub trace_id: String,
    /// Success or trace-aborting failure.
    pub result: Result<(), TraceError>,
}

/// Aggregate workload outcome.
#[derive(Debug, Clone, Default)]
pub struct GraphWorkloadReport {
    /// Root traces that acquired session admission.
    pub admitted: u64,
    /// Successfully drained traces.
    pub completed: u64,
    /// Traces that aborted.
    pub failed: u64,
    /// Results in completion order.
    pub traces: Vec<GraphTraceRunResult>,
}

/// Policy-composed coordinator delegating complete traces to one backend.
pub struct GraphWorkload {
    clock: Rc<dyn Clock>,
    source: Rc<dyn GraphTraceSource>,
    arrival: Rc<dyn GraphArrivalPolicy>,
    admission: Rc<dyn TraceAdmissionPolicy>,
    backend: Rc<dyn GraphTraceExecutionBackend>,
    run_failure: Rc<dyn RunFailurePolicy>,
    observer: Rc<dyn GraphWorkloadObserver>,
    cancelled: Rc<Cell<bool>>,
}

impl GraphWorkload {
    /// Construct the default immediate/unlimited/resilient workload.
    pub fn new(
        clock: Rc<dyn Clock>,
        source: Rc<dyn GraphTraceSource>,
        backend: Rc<dyn GraphTraceExecutionBackend>,
    ) -> Self {
        Self {
            clock,
            source,
            arrival: Rc::new(ImmediateGraphArrival),
            admission: Rc::new(UnlimitedTraceAdmission),
            backend,
            run_failure: Rc::new(ContinueRunFailurePolicy),
            observer: Rc::new(NoopGraphWorkloadObserver),
            cancelled: Rc::new(Cell::new(false)),
        }
    }

    /// Inject arrival pacing.
    pub fn with_arrival(mut self, arrival: Rc<dyn GraphArrivalPolicy>) -> Self {
        self.arrival = arrival;
        self
    }

    /// Inject root-session admission.
    pub fn with_admission(mut self, admission: Rc<dyn TraceAdmissionPolicy>) -> Self {
        self.admission = admission;
        self
    }

    /// Inject run-level admission-after-failure behavior.
    pub fn with_run_failure(mut self, policy: Rc<dyn RunFailurePolicy>) -> Self {
        self.run_failure = policy;
        self
    }

    /// Inject phase/report observation.
    pub fn with_observer(mut self, observer: Rc<dyn GraphWorkloadObserver>) -> Self {
        self.observer = observer;
        self
    }

    /// Stop admitting new traces. Existing traces drain through their sink.
    pub fn cancel(&self) {
        self.cancelled.set(true);
    }

    /// Whether external cancellation has latched.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.get()
    }

    /// Execute all admitted traces on the caller's current-thread `LocalSet`.
    pub async fn execute(&self) -> Result<GraphWorkloadReport, GraphWorkloadError> {
        let run_start_ns = self.clock.now_ns();
        let (completed_tx, mut completed_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut active = 0_u64;
        let mut admitted = 0_u64;
        let mut ordinal = 0_u64;
        let mut report = GraphWorkloadReport::default();

        loop {
            self.drain_ready_results(&mut completed_rx, &mut active, &mut report);
            if self.cancelled.get() || !self.run_failure.may_admit() {
                break;
            }
            let Some(plan) = self.source.next_trace()? else {
                break;
            };
            self.arrival
                .wait_for_arrival(self.clock.clone(), run_start_ns, ordinal, &plan)
                .await?;
            ordinal = ordinal.saturating_add(1);
            self.drain_ready_results(&mut completed_rx, &mut active, &mut report);
            if self.cancelled.get() || !self.run_failure.may_admit() {
                break;
            }

            let info = TraceAdmissionInfo {
                trace_id: plan.trace.id.clone(),
                node_count: plan.graph.nodes.len(),
                arrival_ns: self.clock.now_ns(),
            };
            self.observer.on_trace_arrival(&info);
            let permit = self.admission.acquire(&info).await?;
            self.drain_ready_results(&mut completed_rx, &mut active, &mut report);
            if self.cancelled.get() || !self.run_failure.may_admit() {
                drop(permit);
                break;
            }
            self.observer.on_trace_admit(&info, self.clock.now_ns());
            admitted = admitted.saturating_add(1);
            active = active.saturating_add(1);

            let trace_id = plan.trace.id.clone();
            let backend = self.backend.clone();
            let observer = self.observer.clone();
            let run_failure = self.run_failure.clone();
            let completed_tx = completed_tx.clone();
            tokio::task::spawn_local(async move {
                let result = backend.execute_trace(plan).await;
                run_failure.on_trace_result(&trace_id, &result);
                let outcome = GraphTraceRunResult { trace_id, result };
                observer.on_trace_complete(&outcome);
                let _ = completed_tx.send(outcome);
                drop(permit);
            });

            // Let a same-instant failure latch before another burst admission.
            tokio::task::yield_now().await;
        }

        self.observer.on_sending_complete(self.clock.now_ns());
        while active > 0 {
            let outcome = completed_rx.recv().await.ok_or_else(|| {
                GraphWorkloadError("trace completion channel closed with active work".into())
            })?;
            active -= 1;
            push_result(&mut report, outcome);
        }
        report.admitted = admitted;
        Ok(report)
    }

    fn drain_ready_results(
        &self,
        completed_rx: &mut tokio::sync::mpsc::UnboundedReceiver<GraphTraceRunResult>,
        active: &mut u64,
        report: &mut GraphWorkloadReport,
    ) {
        while let Ok(outcome) = completed_rx.try_recv() {
            *active = active.saturating_sub(1);
            push_result(report, outcome);
        }
    }
}

fn push_result(report: &mut GraphWorkloadReport, outcome: GraphTraceRunResult) {
    if outcome.result.is_ok() {
        report.completed = report.completed.saturating_add(1);
    } else {
        report.failed = report.failed.saturating_add(1);
    }
    report.traces.push(outcome);
}

/// Workload/source/admission error outside an individual trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphWorkloadError(pub String);

impl Display for GraphWorkloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for GraphWorkloadError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use anyhow::anyhow;
    use bytes::Bytes;

    use super::*;
    use crate::materialize::{PromptMaterializer, SegmentItemsMaterializer};
    use crate::model::{
        ChannelSpec, ChannelType, GraphRecord, LlmNode, PromptItem, ReducerName, StaticEdge,
        TraceRecord,
    };
    use crate::policy::{AbortTraceNodeFailurePolicy, FailFastRunFailurePolicy};
    use crate::segment::{SegmentPool, intern_message};
    use crate::sink::{GraphReply, GraphSink};
    use crate::wire::OpenAiChatMessage;
    use aiperf_clock::sim_clock::SimClock;
    use aiperf_dataset::TiktokenTokenizer;

    fn one_node_plan(id: &str, handle: aiperf_dataset::Handle) -> GraphTracePlan {
        let output = format!("out-{id}");
        let mut graph = GraphRecord::default();
        graph.state.insert(
            output.clone(),
            ChannelSpec {
                channel_type: ChannelType::Messages,
                reducer: ReducerName::AddMessages,
            },
        );
        graph.nodes.insert(
            id.to_string(),
            LlmNode {
                output,
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: handle }],
                metadata: BTreeMap::new(),
            },
        );
        graph.edges.push(StaticEdge {
            source: crate::model::START_NODE_ID.into(),
            target: id.into(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        });
        GraphTracePlan {
            graph,
            trace: TraceRecord {
                id: id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        }
    }

    struct SelectiveSink;

    #[async_trait(?Send)]
    impl GraphSink<OpenAiChatMessage> for SelectiveSink {
        async fn dispatch(
            &self,
            node_id: &str,
            _messages: Vec<Bytes>,
            _max_tokens: Option<usize>,
            on_first_token: &dyn Fn(),
        ) -> anyhow::Result<GraphReply<OpenAiChatMessage>> {
            if node_id == "fail" {
                return Err(anyhow!("selected failure"));
            }
            on_first_token();
            Ok(GraphReply::from_text("ok".into()))
        }
    }

    struct RecordingBackend {
        plans: Rc<RefCell<Vec<GraphTracePlan>>>,
    }

    #[async_trait(?Send)]
    impl crate::execution::GraphTraceExecutionBackend for RecordingBackend {
        async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
            self.plans.borrow_mut().push(plan);
            Ok(())
        }
    }

    #[test]
    fn coordinator_delegates_a_complete_trace_through_the_backend_trait() {
        let clock = Rc::new(SimClock::new());
        let mut graph = GraphRecord::default();
        graph.nodes.insert(
            "left".into(),
            LlmNode {
                output: "left-output".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(7),
                items: Vec::new(),
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "right".into(),
            LlmNode {
                output: "right-output".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(11),
                items: Vec::new(),
                metadata: BTreeMap::new(),
            },
        );
        let source: Rc<dyn GraphTraceSource> =
            Rc::new(VecGraphTraceSource::new([GraphTracePlan {
                graph,
                trace: TraceRecord {
                    id: "whole-trace".into(),
                    graph_ref: Some("resolved-before-placement".into()),
                    initial_state: BTreeMap::from([(
                        "seed".into(),
                        serde_json::Value::String("value".into()),
                    )]),
                },
                arrival_offset_ns: Some(123),
            }]));
        let received = Rc::new(RefCell::new(Vec::new()));
        let backend: Rc<dyn crate::execution::GraphTraceExecutionBackend> =
            Rc::new(RecordingBackend {
                plans: received.clone(),
            });
        let workload = GraphWorkload::new(clock.clone(), source, backend);
        let report = Rc::new(RefCell::new(None));
        let report_slot = report.clone();
        let outcome = crate::runtime::drive_sim(clock, move |_handle| async move {
            *report_slot.borrow_mut() = Some(workload.execute().await.unwrap());
        });

        assert!(!outcome.deadlocked);
        assert_eq!(report.borrow().as_ref().unwrap().completed, 1);
        let received = received.borrow();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].trace.id, "whole-trace");
        assert_eq!(received[0].trace.initial_state["seed"], "value");
        assert_eq!(received[0].arrival_offset_ns, Some(123));
        assert_eq!(
            received[0].graph.nodes.keys().cloned().collect::<Vec<_>>(),
            vec!["left", "right"]
        );
    }

    #[test]
    fn fail_fast_stops_new_trace_admission_while_resilient_runs_all() {
        fn run(fail_fast: bool) -> GraphWorkloadReport {
            let clock = Rc::new(SimClock::new());
            let tokenizer = TiktokenTokenizer::builtin();
            let mut pool = SegmentPool::new();
            let message = intern_message(
                &mut pool,
                &OpenAiChatMessage::new("user", "u"),
                None,
                &tokenizer,
            )
            .unwrap();
            let source: Rc<dyn GraphTraceSource> = Rc::new(VecGraphTraceSource::new([
                one_node_plan("fail", message),
                one_node_plan("after", message),
            ]));
            let materializer: Rc<dyn PromptMaterializer> =
                Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));
            let sink: Rc<dyn GraphSink<OpenAiChatMessage>> = Rc::new(SelectiveSink);
            let slots = Rc::new(SlotPool::new(1));
            let mut backend = crate::execution::LocalGraphTraceExecutionBackend::new(
                clock.clone(),
                materializer,
                sink,
            );
            if fail_fast {
                backend = backend.with_node_failure(Rc::new(AbortTraceNodeFailurePolicy));
            }
            let mut workload = GraphWorkload::new(clock.clone(), source, Rc::new(backend))
                .with_admission(Rc::new(SlotPoolTraceAdmission::new(slots)));
            if fail_fast {
                workload = workload.with_run_failure(Rc::new(FailFastRunFailurePolicy::default()));
            }
            let result = Rc::new(RefCell::new(None));
            let result_slot = result.clone();
            let outcome = crate::runtime::drive_sim(clock, move |_handle| async move {
                *result_slot.borrow_mut() = Some(workload.execute().await.unwrap());
            });
            assert!(!outcome.deadlocked);
            result.borrow_mut().take().unwrap()
        }

        let resilient = run(false);
        assert_eq!(resilient.admitted, 2);
        assert_eq!(resilient.completed, 2);
        assert_eq!(resilient.failed, 0);

        let fail_fast = run(true);
        assert_eq!(fail_fast.admitted, 1);
        assert_eq!(fail_fast.completed, 0);
        assert_eq!(fail_fast.failed, 1);
        assert_eq!(fail_fast.traces[0].trace_id, "fail");
    }

    #[test]
    fn interval_arrival_and_session_pool_share_simclock_policy_path() {
        let clock = Rc::new(SimClock::new());
        let generator: Rc<RefCell<Box<dyn IntervalGenerator>>> = Rc::new(RefCell::new(Box::new(
            aiperf_timing::intervals::Constant::new(10.0),
        )));
        let arrival = IntervalGraphArrival::new(generator.clone());
        let plan = GraphTracePlan {
            graph: GraphRecord::default(),
            trace: TraceRecord {
                id: "t".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        };
        let observed = Rc::new(RefCell::new(Vec::new()));
        let observed_slot = observed.clone();
        let drive_clock = clock.clone();
        let outcome = crate::runtime::drive_sim(clock, move |_handle| async move {
            let start = drive_clock.now_ns();
            arrival
                .wait_for_arrival(drive_clock.clone(), start, 0, &plan)
                .await
                .unwrap();
            observed_slot.borrow_mut().push(drive_clock.now_ns());
            arrival
                .wait_for_arrival(drive_clock.clone(), start, 1, &plan)
                .await
                .unwrap();
            observed_slot.borrow_mut().push(drive_clock.now_ns());
            generator.borrow_mut().set_rate(20.0);
            arrival
                .wait_for_arrival(drive_clock.clone(), start, 2, &plan)
                .await
                .unwrap();
            observed_slot.borrow_mut().push(drive_clock.now_ns());
        });
        assert!(!outcome.deadlocked);
        assert_eq!(*observed.borrow(), vec![0, 100_000_000, 150_000_000]);
    }

    #[test]
    fn scheduled_arrival_honors_exact_virtual_offset() {
        let clock = Rc::new(SimClock::new());
        let plan = GraphTracePlan {
            graph: GraphRecord::default(),
            trace: TraceRecord {
                id: "scheduled".into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: Some(42_000),
        };
        let observed = Rc::new(Cell::new(-1));
        let observed_slot = observed.clone();
        let drive_clock = clock.clone();
        let outcome = crate::runtime::drive_sim(clock, move |_handle| async move {
            ScheduledGraphArrival
                .wait_for_arrival(drive_clock.clone(), 0, 0, &plan)
                .await
                .unwrap();
            observed_slot.set(drive_clock.now_ns());
        });
        assert!(!outcome.deadlocked);
        assert_eq!(observed.get(), 42_000);
    }

    #[test]
    fn fail_fast_wakes_fan_in_waiting_on_a_never_scheduled_producer() {
        let clock = Rc::new(SimClock::new());
        let tokenizer = TiktokenTokenizer::builtin();
        let mut pool = SegmentPool::new();
        let message = intern_message(
            &mut pool,
            &OpenAiChatMessage::new("user", "u"),
            None,
            &tokenizer,
        )
        .unwrap();
        let mut graph = GraphRecord::default();
        for channel in ["a", "b", "gate"] {
            graph.state.insert(
                channel.into(),
                ChannelSpec {
                    channel_type: ChannelType::Messages,
                    reducer: ReducerName::AddMessages,
                },
            );
        }
        graph.nodes.insert(
            "fail".into(),
            LlmNode {
                output: "a".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "never".into(),
            LlmNode {
                output: "b".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "waiting".into(),
            LlmNode {
                output: "gate".into(),
                streaming: true,
                inputs: vec![crate::model::ChannelRequirement {
                    channel: "b".into(),
                    count: crate::model::Count::N(1),
                }],
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.nodes.insert(
            "first-token-waiting".into(),
            LlmNode {
                output: "gate".into(),
                streaming: true,
                inputs: Vec::new(),
                min_start_delay_us: None,
                max_tokens: Some(1),
                items: vec![PromptItem::Seg { seg: message }],
                metadata: BTreeMap::new(),
            },
        );
        graph.edges.extend([
            StaticEdge {
                source: crate::model::START_NODE_ID.into(),
                target: "fail".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
            StaticEdge {
                source: crate::model::START_NODE_ID.into(),
                target: "waiting".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
            StaticEdge {
                source: "fail".into(),
                target: "never".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
            StaticEdge {
                source: "never".into(),
                target: "first-token-waiting".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: Some(1.0),
            },
            StaticEdge {
                source: crate::model::START_NODE_ID.into(),
                target: "first-token-waiting".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            },
        ]);
        let source: Rc<dyn GraphTraceSource> =
            Rc::new(VecGraphTraceSource::new([GraphTracePlan {
                graph,
                trace: TraceRecord {
                    id: "stranded".into(),
                    graph_ref: None,
                    initial_state: BTreeMap::new(),
                },
                arrival_offset_ns: None,
            }]));
        let materializer: Rc<dyn PromptMaterializer> =
            Rc::new(SegmentItemsMaterializer::new(Arc::new(pool.freeze())));
        let backend = crate::execution::LocalGraphTraceExecutionBackend::new(
            clock.clone(),
            materializer,
            Rc::new(SelectiveSink),
        )
        .with_node_failure(Rc::new(AbortTraceNodeFailurePolicy));
        let workload = GraphWorkload::new(clock.clone(), source, Rc::new(backend))
            .with_run_failure(Rc::new(FailFastRunFailurePolicy::default()));
        let report = Rc::new(RefCell::new(None));
        let report_slot = report.clone();
        let outcome = crate::runtime::drive_sim(clock, move |_handle| async move {
            *report_slot.borrow_mut() = Some(workload.execute().await.unwrap());
        });

        assert!(!outcome.deadlocked);
        let report = report.borrow_mut().take().unwrap();
        assert_eq!(report.failed, 1);
        assert_eq!(report.admitted, 1);
    }
}
