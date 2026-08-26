// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-owned HTTP execution for complete Graph-IR traces.
//!
//! Each worker owns its clock, transport, materializer, observer, and node policies.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::body_plan::RequestBody;
use crate::cellular::{CellPartition, ModuloCellPartition};
use crate::clock::{Clock, RealClock, RealClockAnchor};
use crate::dataset::{Handle, InMemorySegmentStore, Payload, SegmentPool, SegmentStore};
use crate::dispatch::collector::ReplayTerminalStatus;
use crate::dispatch::sink::RequestObserver;
use crate::endpoints::{
    CreditPhase, EndpointId, EndpointKey, EndpointRegistry, PreparedEndpointTable, PreparedRequest,
    Turn,
};
use crate::eval::NATIVE_GRAPH_LIVE_DRIVER_KIND;
use crate::failure::OnFailure;
use crate::graph::driver::{
    TraceAgentInvocationContext, TraceDriverContext, TraceIdentity, TraceProgramDriver,
    TraceProgramDriverFactory, TraceStageDirective, TraceStageResult, WorkerIdentity,
};
use crate::graph::errors::TraceError;
use crate::graph::execution::{LocalGraphTraceExecutionBackend, TracePlacement};
use crate::graph::executor::ExecutorFlags;
use crate::graph::materialize::SegmentItemsMaterializer;
use crate::graph::model::{GraphTracePlan, GraphTraceProgram, LlmNode, PromptItem};
use crate::graph::placement::{
    GraphPlacementError, LocalTracePlacement, ThreadPerCoreTracePlacement, TracePlacementFactory,
};
use crate::graph::policy::{
    AbortTraceNodeFailurePolicy, CancellationNodePolicy, CompositeNodeDispatchPolicy,
    NodeDispatchPolicy, NodeFailurePolicy, PrefillSlotNodePolicy, ResilientNodeFailurePolicy,
};
use crate::graph::reducers::ChanVal;
use crate::graph::replay::{ReplayCallMeasurement, ToolCallMeasurement};
use crate::graph::sink::{
    GraphDispatchContext, GraphDispatchOptions, GraphReply, GraphReplyStatus, GraphSink,
    TraceInstanceId, TraceSubphase,
};
use crate::graph::supplement::{PlannedReplayTraceInstance, TraceTerminalSupplement};
use crate::graph::tools::{ToolDispatchContext, ToolDispatchRequest, ToolDispatcher};
use crate::graph::wire::OpenAiChatMessage;
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use crate::metrics_core::{InferenceDimensions, MetricsConfig, Phase};
use crate::multiturn::{InputTokenCounter, TurnDataPolicy};
use crate::rng::{RngRoot, namespace};
use crate::timing::{BernoulliFixedDelay, SlotPool};
use crate::transport::core::{
    BoundedDecisionMode, BoundedDecisionReader, Dispatcher, PreparedEndpointBinding, PreparedTurn,
    Request,
};
use crate::transport::http::{PreparedEndpointReference, TransportSinkConfig};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use futures::future::{AbortHandle, Abortable};
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::engine::execute::DEFAULT_ENDPOINT_PROFILE_ID;
use crate::engine::record_lane::EvalNodeRecordArtifact;
use crate::engine::records::{CapturedHttpExchange, CapturedModelOutput, CapturedRecord};
use crate::engine::registry::{NativeTransportExecution, RunContext, ValidatedEndpointProfileV2};
use crate::eval::{
    GENERATION_METADATA_KEY, NativeGraphLivePolicyCallEvidence, validate_native_graph_trace_plan,
};

/// Composition seam for whole-trace graph execution placement.
///
/// The coordinator owns admission policy; this factory owns trace placement.
pub trait GraphPlacementFactory: Send + Sync {
    /// Build one placement backend from a worker-local execution factory.
    fn build(
        &self,
        worker_count: usize,
        worker_factory: Arc<dyn TracePlacementFactory>,
        clock: Rc<dyn Clock>,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError>;

    /// Whether successful traces must emit one native record per static node.
    ///
    /// Product placements return `true`. Lightweight injected placements used
    /// for coordinator tests may return no transport records and retain the
    /// default aggregate progress fallback.
    fn requires_node_records(&self) -> bool {
        false
    }
}

/// Worker-to-coordinator facts emitted by a graph execution placement.
pub(crate) enum GraphExecutionEvent {
    /// Bounded successful profiling facts emitted before the trace terminal.
    TraceSupplement {
        /// The completed trace's replay measurements in local completion order.
        supplement: TraceTerminalSupplement,
    },
    /// One node crossed its first meaningful token edge.
    FirstToken {
        /// Unique root execution-instance identity.
        trace_id: String,
        /// Native request identity used to reject duplicate edges.
        uuid: Uuid,
    },
    /// One node reached a transport terminal and has a complete native record.
    Record {
        /// Unique root execution-instance identity used by phase accounting.
        trace_id: String,
        /// Complete native metric record for the terminal node request.
        record: Box<CapturedRecord>,
        /// Static graph node id (e.g. `"n_1"`) this terminal record belongs to.
        ///
        /// Node identity for lane return accounting; absent when the backend
        /// cannot participate in warmup handoff.
        node_id: Option<String>,
    },
    /// One admitted root trace reached its placement terminal.
    TraceComplete {
        /// Unique execution-instance identity.
        trace_id: String,
        /// Static request count reserved at root admission.
        node_count: usize,
        /// Whether missing per-node records are an infrastructure failure.
        requires_node_records: bool,
        /// Explicit success, cancellation, or failure classification.
        result: Result<(), TraceError>,
    },
}

/// Object-safe worker-event delivery seam.
///
/// Implementations deliver ordered terminal facts to phase accounting.
pub(crate) trait GraphExecutionEventSink: Send + Sync {
    /// Deliver one ordered execution event or fail the run if delivery closed.
    fn emit(&self, event: GraphExecutionEvent) -> Result<(), TraceError>;
}

/// Channel-backed execution-event sink used by native placement.
pub(crate) struct ChannelRunnerGraphExecutionEventSink {
    sender: mpsc::UnboundedSender<GraphExecutionEvent>,
}

impl ChannelRunnerGraphExecutionEventSink {
    /// Bind the worker side to one phase-local coordinator receiver.
    pub(crate) fn new(sender: mpsc::UnboundedSender<GraphExecutionEvent>) -> Self {
        Self { sender }
    }
}

impl GraphExecutionEventSink for ChannelRunnerGraphExecutionEventSink {
    fn emit(&self, event: GraphExecutionEvent) -> Result<(), TraceError> {
        self.sender
            .send(event)
            .map_err(|_| TraceError::Other("graph execution event receiver closed".into()))
    }
}

/// Coordinator-side wrapper that observes every placement terminal.
///
/// This wrapper observes both placement rejection and worker terminals.
pub(crate) struct ObservedRunnerGraphPlacement {
    delegate: Rc<dyn TracePlacement>,
    events: Arc<dyn GraphExecutionEventSink>,
    requires_node_records: bool,
}

impl ObservedRunnerGraphPlacement {
    /// Decorate one placement without changing its dispatch/control semantics.
    pub(crate) fn new(
        delegate: Rc<dyn TracePlacement>,
        events: Arc<dyn GraphExecutionEventSink>,
        requires_node_records: bool,
    ) -> Self {
        Self {
            delegate,
            events,
            requires_node_records,
        }
    }
}

#[async_trait(?Send)]
impl TracePlacement for ObservedRunnerGraphPlacement {
    async fn execute_trace(&self, program: GraphTraceProgram) -> Result<(), TraceError> {
        let trace_id = program.profiling.trace.id.clone();
        let node_count = program.profiling.graph.llm_node_count();
        let result = self.delegate.execute_trace(program).await;
        self.events.emit(GraphExecutionEvent::TraceComplete {
            trace_id,
            node_count,
            requires_node_records: self.requires_node_records,
            result: result.clone(),
        })?;
        result
    }

    fn cancel_inflight(&self) -> Result<(), TraceError> {
        self.delegate.cancel_inflight()
    }

    fn set_prefill_limit(&self, limit: usize) -> Result<(), TraceError> {
        self.delegate.set_prefill_limit(limit)
    }
}

/// Stock thread-per-core whole-trace placement.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeRunnerGraphPlacementFactory;

impl GraphPlacementFactory for NativeRunnerGraphPlacementFactory {
    fn build(
        &self,
        worker_count: usize,
        worker_factory: Arc<dyn TracePlacementFactory>,
        _clock: Rc<dyn Clock>,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
        // Thread-per-core workers each reconstruct a `RealClock` bound to their own
        // reactor (a `!Send` clock cannot cross the thread boundary); the injected
        // clock is only meaningful to the single-reactor inline placement.
        Ok(Rc::new(ThreadPerCoreTracePlacement::new(
            worker_count,
            worker_factory,
        )?))
    }

    fn requires_node_records(&self) -> bool {
        true
    }
}

/// Single-reactor whole-trace placement factory: builds a [`LocalTracePlacement`]
/// that runs every trace inline on the caller's reactor with no worker threads.
///
/// This is the placement the virtual-clock (`dry_run` sim) path requires: a
/// `SimClock` driven by `drive_sim` can only advance the one reactor it pumps, so
/// the thread-per-core default — whose workers each own a private reactor — stalls
/// the replay after the root node under sim. `worker_count` is ignored (a single
/// reactor hosts every concurrently-`spawn_local`ed trace); node-record capture is
/// identical to the thread-per-core placement.
#[derive(Clone, Copy, Debug, Default)]
pub struct InlineGraphPlacementFactory;

impl GraphPlacementFactory for InlineGraphPlacementFactory {
    fn build(
        &self,
        _worker_count: usize,
        worker_factory: Arc<dyn TracePlacementFactory>,
        clock: Rc<dyn Clock>,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
        Ok(Rc::new(LocalTracePlacement::new(worker_factory, clock)?))
    }

    fn requires_node_records(&self) -> bool {
        true
    }
}

/// Startup seam that prepares one endpoint dispatcher per graph worker.
///
/// Prepares worker-local dispatchers and endpoint bindings.
pub(crate) trait GraphEndpointRuntimeFactory: Send + Sync {
    /// Prepare one worker-local dispatcher before any trace is admitted.
    fn prepare_worker(
        &self,
        clock: Rc<dyn Clock>,
        run_origin_ns: i64,
        model: &str,
    ) -> Result<Rc<dyn GraphEndpointRuntime>>;
}

pub(crate) trait GraphEndpointRuntime {
    fn materialize(&self, input: GraphEndpointRequest) -> Result<GraphEndpointDispatch>;

    /// Human-readable transport label the worker-local dispatchers were built
    /// over (for tracing/diagnostics), without downcasting the erased
    /// `Rc<dyn Dispatcher>`.
    fn transport_label(&self) -> &'static str;
}

pub(crate) struct GraphEndpointRequest {
    selector: Option<String>,
    model: String,
    turn: Turn,
    streaming: bool,
    authored_input_tokens: u64,
    max_output_tokens: usize,
    extra_headers: BTreeMap<String, String>,
    parameters: BTreeMap<String, String>,
    trace_id: String,
    is_final_turn: bool,
    cancel_after_ns: Option<i64>,
    session_num: u64,
    phase: Phase,
    /// Exact wire image count established by the node's authored shape, or `None`
    /// to let dispatch derive it by parsing the serialized body.
    known_image_count: Option<u32>,
}

pub(crate) struct GraphEndpointDispatch {
    transport: Rc<dyn Dispatcher>,
    request: Request,
    endpoint: PreparedEndpointBinding,
    input_tokens: u64,
}

/// Protocol-v2 factory that prepares open endpoint profiles directly through
/// the frozen endpoint registry on every graph worker.
pub(crate) struct PreparedRunnerGraphEndpointRuntimeFactory {
    registry: EndpointRegistry,
    profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    input_token_counters: BTreeMap<String, Arc<dyn InputTokenCounter>>,
    /// Resolved bindings own profile-local dispatcher construction. This keeps
    /// imported NativeGraph selectors authoritative without a transport kind
    /// switch in the shared graph runtime.
    transports: BTreeMap<String, Arc<dyn NativeTransportExecution>>,
    default_profile_id: Option<String>,
    /// Content-server origin for media-URL tagging at dispatch; `None` disables.
    content_server_base: Option<Arc<str>>,
    /// Whether raw HTTP-exchange artifacts are retained this run; threaded into
    /// each transport's `build_graph_dispatcher` so the gRPC sink skips building
    /// the discarded compatibility record when raw capture is off.
    raw_enabled: bool,
}

impl PreparedRunnerGraphEndpointRuntimeFactory {
    /// Bind one normalized profile plan to the process's frozen registry over
    /// the run's resolved transport.
    pub(crate) fn new(
        registry: EndpointRegistry,
        profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
        input_token_counter: Arc<dyn InputTokenCounter>,
        transport: Arc<dyn NativeTransportExecution>,
        content_server_base: Option<Arc<str>>,
        raw_enabled: bool,
    ) -> Result<Self> {
        let input_token_counters = profiles
            .iter()
            .map(|profile| (profile.profile_id.clone(), input_token_counter.clone()))
            .collect();
        let transports = profiles
            .iter()
            .map(|profile| (profile.profile_id.clone(), transport.clone()))
            .collect();
        Self::new_with_profile_bindings(
            registry,
            profiles,
            input_token_counters,
            transports,
            Some(DEFAULT_ENDPOINT_PROFILE_ID.into()),
            content_server_base,
            raw_enabled,
        )
    }

    /// Binds imported profile selectors to their independently resolved model
    /// transport and tokenizer authorities.
    pub(crate) fn new_with_profile_bindings(
        registry: EndpointRegistry,
        profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
        input_token_counters: BTreeMap<String, Arc<dyn InputTokenCounter>>,
        transports: BTreeMap<String, Arc<dyn NativeTransportExecution>>,
        default_profile_id: Option<String>,
        content_server_base: Option<Arc<str>>,
        raw_enabled: bool,
    ) -> Result<Self> {
        for profile in profiles.iter() {
            let descriptor = registry.resolve_factory(&profile.endpoint_id)?.descriptor();
            ensure!(
                !descriptor.requires_raw_token_ids,
                "graph endpoint profile {:?} selects {:?}, which requires raw token IDs unavailable to direct Graph-IR nodes",
                profile.profile_id,
                descriptor.id
            );
            ensure!(
                input_token_counters.contains_key(&profile.profile_id),
                "graph endpoint profile {:?} has no resolved input-token counter",
                profile.profile_id
            );
            ensure!(
                transports.contains_key(&profile.profile_id),
                "graph endpoint profile {:?} has no resolved native transport",
                profile.profile_id
            );
        }
        if let Some(profile_id) = default_profile_id.as_deref() {
            ensure!(
                profiles
                    .iter()
                    .any(|profile| profile.profile_id == profile_id),
                "default endpoint profile {profile_id:?} was not prepared"
            );
        }
        Ok(Self {
            registry,
            profiles,
            input_token_counters,
            transports,
            default_profile_id,
            content_server_base,
            raw_enabled,
        })
    }
}

impl GraphEndpointRuntimeFactory for PreparedRunnerGraphEndpointRuntimeFactory {
    fn prepare_worker(
        &self,
        clock: Rc<dyn Clock>,
        run_origin_ns: i64,
        model: &str,
    ) -> Result<Rc<dyn GraphEndpointRuntime>> {
        let mut table = PreparedEndpointTable::new();
        let mut staged = Vec::with_capacity(self.profiles.len());
        for profile in self.profiles.iter() {
            let input_token_counter = self
                .input_token_counters
                .get(&profile.profile_id)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "graph endpoint profile {:?} has no resolved input-token counter",
                        profile.profile_id
                    )
                })?;
            let transport = self
                .transports
                .get(&profile.profile_id)
                .cloned()
                .ok_or_else(|| {
                    anyhow!(
                        "graph endpoint profile {:?} has no resolved native transport",
                        profile.profile_id
                    )
                })?;
            let mut keys = [EndpointKey::from_index(0); 2];
            for streaming in [false, true] {
                let mut config = profile.config.clone();
                config.streaming = streaming;
                let endpoint = self
                    .registry
                    .prepare(&profile.endpoint_id, config)
                    .with_context(|| {
                        format!("preparing endpoint profile {:?}", profile.profile_id)
                    })?;
                keys[usize::from(streaming)] = table.push(endpoint)?;
            }
            staged.push(StagedGraphProfile {
                profile_id: profile.profile_id.clone(),
                endpoint_id: profile.endpoint_id.clone(),
                keys,
                urls: profile.config.urls.clone(),
                transport_config: TransportSinkConfig {
                    client: profile.client.clone(),
                    connection_reuse: profile.connection_reuse,
                    session_header: profile.session_header.clone(),
                    content_server_base: self.content_server_base.clone(),
                    // `build_graph_dispatcher` stamps the request-payload
                    // capture flag from this run's raw-artifact selection.
                    ..TransportSinkConfig::default()
                },
                url_count: profile.config.urls.len(),
                input_token_counter,
                transport,
            });
        }
        let table = Rc::new(table);
        // The injected dispatcher factory owns all transport-specific
        // construction (including any gRPC binding table); the graph runtime
        // stays transport-blind — no `match` on a transport kind here.
        let profiles = staged
            .into_iter()
            .map(|profile| {
                let transport = profile.transport.build_graph_dispatcher(
                    clock.clone(),
                    run_origin_ns,
                    &profile.urls,
                    model,
                    profile.transport_config,
                    table.clone(),
                    self.raw_enabled,
                )?;
                Ok((
                    profile.profile_id,
                    PreparedGraphProfileRuntime {
                        endpoint_id: profile.endpoint_id,
                        keys: profile.keys,
                        transport,
                        url_count: profile.url_count,
                        input_token_counter: profile.input_token_counter,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let transport_label = self
            .transports
            .values()
            .map(|transport| transport.graph_transport_label())
            .all(|label| {
                label
                    == self
                        .transports
                        .values()
                        .next()
                        .map_or("unknown", |transport| transport.graph_transport_label())
            })
            .then(|| {
                self.transports
                    .values()
                    .next()
                    .map_or("unknown", |transport| transport.graph_transport_label())
            })
            .unwrap_or("native_graph");
        Ok(Rc::new(PreparedRunnerGraphEndpointRuntime {
            table,
            profiles,
            default_profile_id: self.default_profile_id.clone(),
            transport_label,
        }))
    }
}

/// Profile inputs retained until the shared endpoint table is finalized.
struct StagedGraphProfile {
    profile_id: String,
    endpoint_id: EndpointId,
    keys: [EndpointKey; 2],
    urls: Vec<String>,
    transport_config: TransportSinkConfig,
    url_count: usize,
    input_token_counter: Arc<dyn InputTokenCounter>,
    transport: Arc<dyn NativeTransportExecution>,
}

struct PreparedGraphProfileRuntime {
    endpoint_id: EndpointId,
    keys: [EndpointKey; 2],
    transport: Rc<dyn Dispatcher>,
    url_count: usize,
    input_token_counter: Arc<dyn InputTokenCounter>,
}

struct PreparedRunnerGraphEndpointRuntime {
    table: Rc<PreparedEndpointTable>,
    profiles: BTreeMap<String, PreparedGraphProfileRuntime>,
    default_profile_id: Option<String>,
    transport_label: &'static str,
}

/// Worker-local prepared-profile transport for one selected NativeGraph policy model.
///
/// This is deliberately narrower than [`EngineGraphSink`]: a rollout policy decision uses the
/// same prepared endpoint and transport construction, but can expose only the host-owned bounded
/// reader. It has no graph record, event sink, or raw-response retention path.
pub(crate) struct NativeGraphPolicyEndpointRuntime {
    endpoint_runtime: Rc<dyn GraphEndpointRuntime>,
    profile_id: String,
    model: String,
    is_streaming: bool,
    next_session_num: u64,
    clock: Rc<dyn Clock>,
    live_policy_summary: Rc<RefCell<NativeGraphLivePolicyCallSummary>>,
}

impl NativeGraphPolicyEndpointRuntime {
    /// Prepares exactly one imported endpoint profile through the graph worker seam.
    pub(crate) fn new(
        registry: EndpointRegistry,
        profile: ValidatedEndpointProfileV2,
        input_token_counter: Arc<dyn InputTokenCounter>,
        transport: Arc<dyn NativeTransportExecution>,
        model: String,
        raw_enabled: bool,
        live_policy_summary: Rc<RefCell<NativeGraphLivePolicyCallSummary>>,
    ) -> Result<Self> {
        let profile_id = profile.profile_id.clone();
        let is_streaming = profile.config.streaming;
        let mut token_counters = BTreeMap::new();
        token_counters.insert(profile_id.clone(), input_token_counter);
        let mut transports = BTreeMap::new();
        transports.insert(profile_id.clone(), transport);
        let factory = PreparedRunnerGraphEndpointRuntimeFactory::new_with_profile_bindings(
            registry,
            Arc::new(vec![profile]),
            token_counters,
            transports,
            Some(profile_id.clone()),
            None,
            raw_enabled,
        )?;
        let clock: Rc<dyn Clock> = RealClock::new();
        let run_origin_ns = clock.now_ns();
        let endpoint_runtime = factory.prepare_worker(clock.clone(), run_origin_ns, &model)?;
        Ok(Self {
            endpoint_runtime,
            profile_id,
            model,
            is_streaming,
            next_session_num: 0,
            clock,
            live_policy_summary,
        })
    }

    /// Dispatches one restricted, bounded policy decision without creating a graph record.
    pub(crate) async fn open_bounded_decision(
        &mut self,
        turn: Turn,
        max_output_tokens: usize,
        max_decision_bytes: usize,
    ) -> Result<BoundedDecisionReader> {
        let session_num = self.next_session_num;
        self.next_session_num = session_num
            .checked_add(1)
            .ok_or_else(|| anyhow!("NativeGraph policy session ordinal overflow"))?;
        let trace_id = format!("native-graph-policy:{}:{session_num}", self.profile_id);
        let dispatch = self.endpoint_runtime.materialize(GraphEndpointRequest {
            selector: Some(self.profile_id.clone()),
            model: self.model.clone(),
            turn,
            streaming: self.is_streaming,
            authored_input_tokens: 0,
            max_output_tokens,
            extra_headers: BTreeMap::new(),
            parameters: BTreeMap::new(),
            trace_id: trace_id.clone(),
            is_final_turn: true,
            cancel_after_ns: None,
            session_num,
            phase: Phase::Profiling,
            known_image_count: Some(0),
        })?;
        let GraphEndpointDispatch {
            transport,
            request,
            endpoint,
            ..
        } = dispatch;
        let observer = NativeGraphPolicyDecisionObserver::new(
            self.clock.clone(),
            Rc::clone(&self.live_policy_summary),
        );
        transport
            .dispatch_bounded_decision(
                PreparedTurn {
                    runtime_session_id: trace_id,
                    request,
                    model: self.model.clone(),
                    endpoint,
                    endpoint_aware: true,
                    data_policy: TurnDataPolicy::restricted_transient(),
                    deferred: None,
                },
                &observer,
                &|_| {},
                BoundedDecisionMode::new(max_decision_bytes)?,
            )
            .await
    }
}

/// Bounded, non-retaining timing facts collected for one live policy rollout.
///
/// Each selected policy runtime dispatches decisions serially, so this retains one active request
/// identity and scalar timings only. It cannot retain prompts, observations, decision bytes, or
/// transport records.
pub(crate) struct NativeGraphLivePolicyCallSummary {
    call_count: u64,
    first_token_count: u64,
    total_first_token_ns: u64,
    max_first_token_ns: u64,
    active_request: Option<(Uuid, i64, bool)>,
    is_invalid: bool,
}

impl NativeGraphLivePolicyCallSummary {
    pub(crate) const fn new() -> Self {
        Self {
            call_count: 0,
            first_token_count: 0,
            total_first_token_ns: 0,
            max_first_token_ns: 0,
            active_request: None,
            is_invalid: false,
        }
    }

    fn on_arrival(&mut self, request: Uuid, now_ns: i64) {
        if self.active_request.is_some() {
            self.is_invalid = true;
            return;
        }
        let Some(call_count) = self.call_count.checked_add(1) else {
            self.is_invalid = true;
            return;
        };
        self.call_count = call_count;
        self.active_request = Some((request, now_ns, false));
    }

    fn on_admit(&mut self, request: Uuid, now_ns: i64) {
        match self.active_request {
            None => self.on_arrival(request, now_ns),
            Some((active, _, _)) if active == request => {}
            Some(_) => self.is_invalid = true,
        }
    }

    fn on_token(&mut self, request: Uuid, now_ns: i64) {
        let Some((active, arrival_ns, has_first_token)) = self.active_request else {
            self.is_invalid = true;
            return;
        };
        if active != request {
            self.is_invalid = true;
            return;
        }
        if has_first_token {
            return;
        }
        let Some(first_token_ns) = now_ns.checked_sub(arrival_ns) else {
            self.is_invalid = true;
            return;
        };
        let Ok(first_token_ns) = u64::try_from(first_token_ns) else {
            self.is_invalid = true;
            return;
        };
        let Some(first_token_count) = self.first_token_count.checked_add(1) else {
            self.is_invalid = true;
            return;
        };
        let Some(total_first_token_ns) = self.total_first_token_ns.checked_add(first_token_ns)
        else {
            self.is_invalid = true;
            return;
        };
        self.first_token_count = first_token_count;
        self.total_first_token_ns = total_first_token_ns;
        self.max_first_token_ns = self.max_first_token_ns.max(first_token_ns);
        self.active_request = Some((active, arrival_ns, true));
    }

    fn on_terminal(&mut self, request: Uuid) {
        let Some((active, _, _)) = self.active_request else {
            self.is_invalid = true;
            return;
        };
        if active != request {
            self.is_invalid = true;
            return;
        }
        self.active_request = None;
    }

    pub(crate) fn snapshot(&self) -> Result<NativeGraphLivePolicyCallEvidence> {
        ensure!(
            !self.is_invalid && self.active_request.is_none(),
            "NativeGraph live policy timing observer became inconsistent"
        );
        Ok(NativeGraphLivePolicyCallEvidence::new(
            self.call_count,
            self.first_token_count,
            self.total_first_token_ns,
            self.max_first_token_ns,
        ))
    }
}

/// Deliberately non-retaining request observer for one live policy decision.
///
/// The selected binding still configures the prepared transport, but the decision path must not
/// create a response record or let prompt, observation, or raw output escape into evidence.
struct NativeGraphPolicyDecisionObserver {
    clock: Rc<dyn Clock>,
    summary: Rc<RefCell<NativeGraphLivePolicyCallSummary>>,
}

impl NativeGraphPolicyDecisionObserver {
    fn new(clock: Rc<dyn Clock>, summary: Rc<RefCell<NativeGraphLivePolicyCallSummary>>) -> Self {
        Self { clock, summary }
    }

    fn observe(&self, operation: impl FnOnce(&mut NativeGraphLivePolicyCallSummary)) {
        if let Ok(mut summary) = self.summary.try_borrow_mut() {
            operation(&mut summary);
        }
    }
}

impl RequestObserver for NativeGraphPolicyDecisionObserver {
    fn on_arrival(&self, request: Uuid, _: f64, _: usize, _: usize) {
        let now_ns = self.clock.now_ns();
        self.observe(|summary| summary.on_arrival(request, now_ns));
    }

    fn on_admit(&self, request: Uuid, _: f64, _: usize) {
        let now_ns = self.clock.now_ns();
        self.observe(|summary| summary.on_admit(request, now_ns));
    }

    fn on_token(&self, request: Uuid, _: f64) {
        let now_ns = self.clock.now_ns();
        self.observe(|summary| summary.on_token(request, now_ns));
    }

    fn on_terminal(&self, request: Uuid, _: ReplayTerminalStatus) {
        self.observe(|summary| summary.on_terminal(request));
    }
}

/// Bounded transport facts emitted by one imported NativeGraph trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct NativeGraphTransportEvidence {
    pub(crate) model_records: usize,
    pub(crate) completed_traces: usize,
    pub(crate) live_policy_calls: u64,
}

const NATIVE_GRAPH_EVIDENCE_CHANNEL_CAPACITY: usize = 64;

/// Bounded event admission dedicated to the one-shot NativeGraph evidence path.
///
/// NativeGraph scoring retains each completed model record until the coordinator
/// consumes it, together with the trace terminal result. This deliberately stays
/// separate from the general phase event transport.
struct NativeGraphEvidenceSink {
    sender: mpsc::Sender<NativeGraphEvidenceEvent>,
}

impl NativeGraphEvidenceSink {
    fn bounded() -> (Self, mpsc::Receiver<NativeGraphEvidenceEvent>) {
        let (sender, receiver) = mpsc::channel(NATIVE_GRAPH_EVIDENCE_CHANNEL_CAPACITY);
        (Self { sender }, receiver)
    }

    fn emit_summary(&self, event: NativeGraphEvidenceEvent) -> Result<(), TraceError> {
        self.sender.try_send(event).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => {
                TraceError::Other("native graph evidence queue reached its bounded capacity".into())
            }
            mpsc::error::TrySendError::Closed(_) => {
                TraceError::Other("native graph evidence receiver closed".into())
            }
        })
    }
}

impl GraphExecutionEventSink for NativeGraphEvidenceSink {
    fn emit(&self, event: GraphExecutionEvent) -> Result<(), TraceError> {
        match event {
            GraphExecutionEvent::Record { record, .. } => {
                self.emit_summary(NativeGraphEvidenceEvent::Record(record))
            }
            GraphExecutionEvent::TraceComplete { result, .. } => self.emit_summary(
                NativeGraphEvidenceEvent::TraceComplete(result.map_err(|error| error.to_string())),
            ),
            GraphExecutionEvent::TraceSupplement { .. }
            | GraphExecutionEvent::FirstToken { .. } => Ok(()),
        }
    }
}

/// NativeGraph facts retained until the one-shot trace returns.
enum NativeGraphEvidenceEvent {
    Record(Box<CapturedRecord>),
    TraceComplete(Result<(), String>),
}

/// Executes one immutable NativeGraph trace through the application's graph path.
///
/// The caller supplies only already-resolved binding facts. Endpoint preparation,
/// dispatcher construction, token accounting, and request observation remain in
/// the current Graph-IR worker implementation.
pub(crate) async fn execute_native_graph_trace(
    context: &RunContext,
    profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    input_token_counters: BTreeMap<String, Arc<dyn InputTokenCounter>>,
    transports: BTreeMap<String, Arc<dyn NativeTransportExecution>>,
    default_model: String,
    raw_enabled: bool,
    program: GraphTraceProgram,
    record_artifact: Option<EvalNodeRecordArtifact>,
) -> Result<NativeGraphTransportEvidence> {
    let clock_anchor = RealClockAnchor::now();
    let clock: Rc<dyn Clock> = RealClock::from_anchor(clock_anchor);
    let run_origin_ns = clock.now_ns();
    let (events, mut receiver) = NativeGraphEvidenceSink::bounded();
    let event_sink: Arc<dyn GraphExecutionEventSink> = Arc::new(events);
    let endpoint_runtime_factory = Arc::new(
        PreparedRunnerGraphEndpointRuntimeFactory::new_with_profile_bindings(
            context.product_registry().endpoints().clone(),
            profiles,
            input_token_counters,
            transports,
            None,
            None,
            raw_enabled,
        )?,
    );
    let backend = Arc::new(GraphBackendFactory::new(GraphBackendFactoryConfig {
        real_clock_anchor: clock_anchor,
        run_origin_ns,
        model: default_model,
        default_max_tokens: 256,
        endpoint_runtime_factory,
        segments: Arc::new(InMemorySegmentStore::default()),
        replay_run_identity: None,
        metrics: MetricsConfig::default(),
        phase: Phase::Profiling,
        prefill_concurrency: None,
        cancellation: None,
        raw_enabled,
        events: event_sink.clone(),
        on_failure: OnFailure::Abort,
        cache_bust: None,
        ignore_trace_delays: false,
        system_idle_gap_cap_seconds: None,
        trace_driver: context.execution_factories().trace_driver_handle(),
    }));
    let placement = context
        .execution_factories()
        .graph()
        .build(1, backend, clock)
        .map_err(|error| anyhow!(error.to_string()))?;
    let placement = ObservedRunnerGraphPlacement::new(
        placement,
        event_sink,
        context
            .execution_factories()
            .graph()
            .requires_node_records(),
    );
    let mut evidence = NativeGraphTransportEvidence {
        model_records: 0,
        completed_traces: 0,
        live_policy_calls: 0,
    };
    let execution_result = {
        let execution = placement.execute_trace(program);
        tokio::pin!(execution);
        loop {
            tokio::select! {
                result = &mut execution => break result,
                event = receiver.recv() => {
                    let event = event.ok_or_else(|| {
                        anyhow!("native graph evidence receiver closed before trace completion")
                    })?;
                    observe_native_graph_evidence(&mut evidence, event, record_artifact.as_ref())?;
                }
            }
        }
    };
    execution_result.map_err(|error| anyhow!(error.to_string()))?;
    while let Ok(event) = receiver.try_recv() {
        observe_native_graph_evidence(&mut evidence, event, record_artifact.as_ref())?;
    }
    ensure!(
        evidence.completed_traces == 1,
        "native graph trace produced {} completion events, expected one",
        evidence.completed_traces
    );
    Ok(evidence)
}

fn observe_native_graph_evidence(
    evidence: &mut NativeGraphTransportEvidence,
    event: NativeGraphEvidenceEvent,
    record_artifact: Option<&EvalNodeRecordArtifact>,
) -> Result<()> {
    match event {
        NativeGraphEvidenceEvent::Record(record) => {
            if let Some(artifact) = record_artifact {
                artifact.append(&record).with_context(|| {
                    format!(
                        "appending completed native graph node record to {}",
                        artifact.destination()
                    )
                })?;
            }
            evidence.model_records = evidence
                .model_records
                .checked_add(1)
                .ok_or_else(|| anyhow!("native graph model-record evidence overflow"))?;
        }
        NativeGraphEvidenceEvent::TraceComplete(result) => {
            result.map_err(|error| anyhow!(error))?;
            evidence.completed_traces = evidence
                .completed_traces
                .checked_add(1)
                .ok_or_else(|| anyhow!("native graph completed-trace evidence overflow"))?;
        }
    }
    Ok(())
}

impl GraphEndpointRuntime for PreparedRunnerGraphEndpointRuntime {
    fn transport_label(&self) -> &'static str {
        self.transport_label
    }

    fn materialize(&self, input: GraphEndpointRequest) -> Result<GraphEndpointDispatch> {
        let profile_id = input.selector.as_deref().or(self.default_profile_id.as_deref()).ok_or_else(|| {
            anyhow!("graph node has no endpoint profile selector and this graph runtime has no default endpoint profile")
        })?;
        let profile = self.profiles.get(profile_id).ok_or_else(|| {
            anyhow!(
                "graph node selects unknown endpoint profile {profile_id:?}; prepared profiles: {}",
                self.profiles
                    .keys()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        let key = profile.keys[usize::from(input.streaming)];
        let endpoint = self.table.get(key)?;
        ensure!(
            endpoint.descriptor().id == profile.endpoint_id.as_str(),
            "prepared endpoint key {} resolved to {:?}, expected {:?}",
            key.index(),
            endpoint.descriptor().id,
            profile.endpoint_id.as_str()
        );
        let turns = [input.turn];
        let request_info = PreparedRequest::new(
            &input.model,
            &turns,
            None,
            None,
            credit_phase(input.phase),
            None,
            Some(&input.trace_id),
            Some(&input.trace_id),
        );
        let payload = endpoint
            .format_payload(&request_info)?
            .materialize_standalone()?;
        let input_tokens = profile.input_token_counter.count_prepared_input_tokens(
            endpoint,
            &payload,
            input.authored_input_tokens,
        )?;
        let mut headers = endpoint.headers().clone();
        headers.extend(input.extra_headers);
        let endpoint_path = endpoint.config().as_raw().path.clone().or_else(|| {
            if endpoint.config().streaming() {
                endpoint.descriptor().streaming_path
            } else {
                None
            }
            .or(endpoint.descriptor().endpoint_path)
            .map(str::to_owned)
        });
        let request = Request {
            uuid: Uuid::new_v4(),
            input_length: usize::try_from(input_tokens).unwrap_or(usize::MAX),
            max_output_tokens: input.max_output_tokens,
            prompt_text: None,
            body: Some(RequestBody::wire(payload)),
            headers,
            parameters: input.parameters,
            endpoint_path,
            streaming: endpoint.config().streaming(),
            x_correlation_id: Some(input.trace_id),
            is_final_turn: input.is_final_turn,
            cancel_after_ns: input.cancel_after_ns,
            url_index: Some(session_url_index(input.session_num, profile.url_count)),
            image_count: input.known_image_count,
            recorded_api_time_ns: None,
            recorded_ttft_ns: None,
        };
        Ok(GraphEndpointDispatch {
            transport: profile.transport.clone(),
            request,
            endpoint: PreparedEndpointBinding::Prepared(PreparedEndpointReference {
                key,
                endpoint_id: profile.endpoint_id.clone(),
            }),
            input_tokens,
        })
    }
}

fn session_url_index(session_num: u64, url_count: usize) -> u32 {
    debug_assert!(url_count > 0);
    u32::try_from(session_num % url_count as u64).unwrap_or(0)
}

fn credit_phase(phase: Phase) -> CreditPhase {
    match phase {
        Phase::Warmup => CreditPhase::Warmup,
        Phase::Profiling => CreditPhase::Profiling,
    }
}

/// Immutable inputs shared by every worker-local graph backend.
pub(crate) struct GraphBackendFactoryConfig {
    pub(crate) real_clock_anchor: RealClockAnchor,
    pub(crate) run_origin_ns: i64,
    pub(crate) model: String,
    pub(crate) default_max_tokens: usize,
    pub(crate) endpoint_runtime_factory: Arc<dyn GraphEndpointRuntimeFactory>,
    pub(crate) segments: Arc<dyn SegmentStore>,
    pub(crate) replay_run_identity: Option<crate::graph::replay::ReplayRunIdentity>,
    pub(crate) metrics: MetricsConfig,
    pub(crate) phase: Phase,
    pub(crate) prefill_concurrency: Option<usize>,
    pub(crate) cancellation: Option<GraphCancellationConfig>,
    pub(crate) raw_enabled: bool,
    pub(crate) events: Arc<dyn GraphExecutionEventSink>,
    /// Config-selected per-node failure discipline. `Abort` aborts the trace on
    /// the first node failure (default); `Continue` treats a failed node as
    /// empty and drains the DAG.
    pub(crate) on_failure: OnFailure,
    /// Scenario-locked cache-bust marker configuration. `None` when the run has
    /// no `cache_bust_target`; `Some` prepends the target's marker to the first
    /// message of the selected role in every request (a per-conversation nonce
    /// on the first user message for `FirstTurnPrefix`).
    pub(crate) cache_bust: Option<GraphCacheBust>,
    /// Ignore recorded trace inter-message/inter-request delays: fire every node
    /// as soon as its inputs are ready via `ExecutorFlags::ignore_edge_delays`.
    pub(crate) ignore_trace_delays: bool,
    pub(crate) system_idle_gap_cap_seconds: Option<f64>,
    /// Frozen driver registry used for preflight before any trace-local sink exists.
    pub(crate) trace_driver: Arc<dyn TraceProgramDriverFactory>,
}

/// First-turn cache-bust inputs shared by graph workers.
///
/// The marker is `[rid:<12 hex chars>]\n\n`; fixed shape preserves token-count
/// accounting while the trace instance id makes each conversation distinct.
#[derive(Clone)]
pub(crate) struct GraphCacheBust {
    /// Per-run benchmark identity digested into the marker.
    pub(crate) benchmark_id: String,
    /// Resolved marker placement target: `FirstTurnPrefix` mints the digest
    /// marker, the warmup-isolation targets mint a fixed `[warmup]` marker, and
    /// `None` mints nothing.
    pub(crate) target: crate::engine::graph_input::CacheBustTarget,
}

impl GraphCacheBust {
    /// Mint the per-conversation `[rid:<digest>]\n\n` marker for one trace
    /// instance, or `None` when the target is disabled. The digest is taken over
    /// the whole instance `trace_id` (base template + `::` nonce) so every
    /// instance and recycle draws a distinct marker while every node of one
    /// instance shares it. The warmup-isolation targets instead mint the fixed
    /// `[warmup]\n\n` marker, which is shared by every instance.
    fn marker(&self, trace_id: &str) -> Option<String> {
        use sha2::{Digest, Sha256};
        if !self.target.is_enabled() {
            return None;
        }
        if matches!(
            self.target,
            crate::engine::graph_input::CacheBustTarget::WarmupIsolationSystem
                | crate::engine::graph_input::CacheBustTarget::WarmupIsolationFirstTurn
        ) {
            return Some("[warmup]\n\n".to_string());
        }
        // The trace instance nonce supplies uniqueness; fixed zeros keep marker
        // length and token accounting stable.
        let unique = format!("{}:0:0:{trace_id}", self.benchmark_id);
        let digest = Sha256::digest(unique.as_bytes());
        let hex = digest
            .iter()
            .take(6)
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Some(format!("[rid:{hex}]\n\n"))
    }
}

/// Idempotently prepend `marker` to the first message the target selects: the
/// first `system` message for `WarmupIsolationSystem`, the first `user` message
/// otherwise.
fn prepend_cache_bust_marker(
    messages: &mut [Value],
    marker: &str,
    target: crate::engine::graph_input::CacheBustTarget,
) {
    let role = if target == crate::engine::graph_input::CacheBustTarget::WarmupIsolationSystem {
        "system"
    } else {
        "user"
    };
    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) != Some(role) {
            continue;
        }
        match object.get_mut("content") {
            Some(Value::String(content)) if !content.starts_with(marker) => {
                content.insert_str(0, marker);
            }
            Some(Value::Array(parts)) => {
                let marker_part = serde_json::json!({"type": "text", "text": marker.trim_end()});
                if parts.first() != Some(&marker_part) {
                    parts.insert(0, marker_part);
                }
            }
            _ => {}
        }
        return;
    }
}

/// Worker-local cancellation construction inputs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GraphCancellationConfig {
    pub(crate) rate: f64,
    pub(crate) delay_seconds: f64,
    /// Phase-local cancellation root; workers derive independent indexed roots.
    pub(crate) rng_root: RngRoot,
    pub(crate) phase: crate::timing::Phase,
}

/// Native runner factory installed into whole-trace placement.
pub(crate) struct GraphBackendFactory {
    config: GraphBackendFactoryConfig,
}

impl GraphBackendFactory {
    pub(crate) fn new(config: GraphBackendFactoryConfig) -> Self {
        Self { config }
    }
}

impl TracePlacementFactory for GraphBackendFactory {
    fn create_backend(
        &self,
        worker_id: usize,
        clock: Option<Rc<dyn Clock>>,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
        // A thread-per-core worker gets `None` and reconstructs a `RealClock`
        // bound to its own reactor; the single-reactor inline placement injects
        // the run's clock (the `SimClock` under a virtual run) so node sleeps are
        // virtual-time events, not real timerfd sleeps.
        let clock: Rc<dyn Clock> =
            clock.unwrap_or_else(|| RealClock::from_anchor(self.config.real_clock_anchor));
        let endpoint_runtime = self
            .config
            .endpoint_runtime_factory
            .prepare_worker(clock.clone(), self.config.run_origin_ns, &self.config.model)
            .map_err(|error| GraphPlacementError(error.to_string()))?;
        tracing::debug!(
            worker_id,
            transport = endpoint_runtime.transport_label(),
            "prepared graph worker endpoint runtime"
        );

        let mut policies = Vec::<Rc<dyn NodeDispatchPolicy>>::new();
        let prefill_slots = if let Some(limit) = self.config.prefill_concurrency {
            if limit == 0 {
                return Err(GraphPlacementError(
                    "graph prefill_concurrency must be positive".into(),
                ));
            }
            let slots = Rc::new(SlotPool::new(limit));
            policies.push(Rc::new(PrefillSlotNodePolicy::new(slots.clone())));
            Some(slots)
        } else {
            None
        };
        if let Some(cancellation) = self.config.cancellation {
            let worker_index = u64::try_from(worker_id)
                .map_err(|_| GraphPlacementError("graph worker index exceeds u64".to_string()))?;
            let worker_rng = cancellation
                .rng_root
                .derive_indexed_root(namespace::GRAPH_NODE_CANCELLATION_WORKER, worker_index);
            let policy = BernoulliFixedDelay::new(
                Some(cancellation.rate),
                cancellation.delay_seconds,
                worker_rng,
            )
            .map_err(|error| GraphPlacementError(error.to_string()))?;
            policies.push(Rc::new(CancellationNodePolicy::new(
                Box::new(policy),
                cancellation.phase,
            )));
        }
        let node_policy = (!policies.is_empty())
            .then(|| Rc::new(CompositeNodeDispatchPolicy::new(policies)) as Rc<_>);

        let observer = Rc::new(NativeMetricsObserver::new(
            clock.clone(),
            self.config.run_origin_ns,
            self.config.metrics.clone(),
        ));
        Ok(Rc::new(GraphWorkerBackend {
            clock,
            endpoint_runtime,
            materializer: Rc::new(SegmentItemsMaterializer::new(self.config.segments.clone())),
            segments: self.config.segments.clone(),
            replay_run_identity: self.config.replay_run_identity.clone(),
            observer,
            phase: self.config.phase,
            worker_id,
            model: self.config.model.clone(),
            default_max_tokens: self.config.default_max_tokens,
            run_origin_ns: self.config.run_origin_ns,
            raw_enabled: self.config.raw_enabled,
            events: self.config.events.clone(),
            node_policy,
            on_failure: self.config.on_failure,
            cache_bust: self.config.cache_bust.clone(),
            ignore_trace_delays: self.config.ignore_trace_delays,
            system_idle_gap_cap_seconds: self.config.system_idle_gap_cap_seconds,
            trace_driver: self.config.trace_driver.clone(),
            prefill_slots,
            next_session: Cell::new(0),
            next_execution: Cell::new(0),
            cancelled: Cell::new(false),
            active: RefCell::new(HashMap::new()),
        }))
    }
}

struct GraphWorkerBackend {
    clock: Rc<dyn Clock>,
    endpoint_runtime: Rc<dyn GraphEndpointRuntime>,
    materializer: Rc<SegmentItemsMaterializer>,
    segments: Arc<dyn SegmentStore>,
    replay_run_identity: Option<crate::graph::replay::ReplayRunIdentity>,
    /// One measurement observer for the worker's whole run, shared by every
    /// trace's sink. Records are uuid-keyed and drained on terminal, so concurrent
    /// traces stay isolated. This avoids constructing a fresh `MetricsAccumulator`
    /// and `ColumnStore` (and cloning `MetricsConfig`) per trace, which dominated
    /// per-trace allocation on the graph hot path. It mirrors the scheduled path's
    /// per-worker `WorkerMeasurement`.
    observer: Rc<NativeMetricsObserver>,
    phase: Phase,
    worker_id: usize,
    model: String,
    default_max_tokens: usize,
    run_origin_ns: i64,
    raw_enabled: bool,
    events: Arc<dyn GraphExecutionEventSink>,
    node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
    on_failure: OnFailure,
    cache_bust: Option<GraphCacheBust>,
    ignore_trace_delays: bool,
    system_idle_gap_cap_seconds: Option<f64>,
    trace_driver: Arc<dyn TraceProgramDriverFactory>,
    prefill_slots: Option<Rc<SlotPool>>,
    next_session: Cell<u64>,
    next_execution: Cell<u64>,
    cancelled: Cell<bool>,
    active: RefCell<HashMap<u64, ActiveGraphTrace>>,
}

#[derive(Clone)]
enum ActiveGraphTrace {
    Driver(AbortHandle),
    Graph(Rc<LocalGraphTraceExecutionBackend<OpenAiChatMessage>>),
}

struct ActiveGraphTraceLifecycle<'a> {
    active: &'a RefCell<HashMap<u64, ActiveGraphTrace>>,
    lifecycle_id: u64,
}

impl Drop for ActiveGraphTraceLifecycle<'_> {
    fn drop(&mut self) {
        self.active.borrow_mut().remove(&self.lifecycle_id);
    }
}

#[async_trait(?Send)]
impl TracePlacement for GraphWorkerBackend {
    async fn execute_trace(&self, program: GraphTraceProgram) -> Result<(), TraceError> {
        if self.cancelled.get() {
            return Err(TraceError::Cancelled(
                "graph worker was cancelled before opening a trace driver".into(),
            ));
        }
        // Driver capability preflight precedes per-trace sink construction. Future
        // environment provisioning uses this same frozen registry before any
        // workspace or sandbox exists.
        self.trace_driver
            .capabilities(&program.driver)
            .map_err(|error| TraceError::Other(error.to_string()))?;
        let execution_id = self.next_execution.get();
        let lifecycle_id = execution_id | (1_u64 << 63);
        self.next_execution.set(execution_id.saturating_add(1));
        let trace_warmup = (!program.is_static_graph_program())
            .then(|| program.warmup.clone())
            .flatten();
        let is_recorded_program = program.driver.kind == "recorded_replay";
        let mut opened_driver = None;
        let mut is_staged_driver = false;
        let mut driver_trace = None;
        let mut driver_invocation = None;
        let program = if program.is_static_graph_program() {
            program
        } else {
            driver_trace = Some(TraceIdentity {
                run_id: self.run_origin_ns.to_string(),
                trajectory_id: format!("{}::trajectory", program.profiling.trace.id),
                trace_id: program.profiling.trace.id.clone(),
            });
            let trace = driver_trace.as_ref().ok_or_else(|| {
                TraceError::Other("graph trace lost its driver identity before open".into())
            })?;
            driver_invocation = if program.driver.kind == NATIVE_GRAPH_LIVE_DRIVER_KIND {
                Some(TraceAgentInvocationContext::native_graph(
                    trace,
                    execution_id,
                    program
                        .driver
                        .source_provenance()
                        .ok_or_else(|| {
                            TraceError::Other(
                                "native graph trace is missing immutable task provenance".into(),
                            )
                        })?
                        .source_digest()
                        .to_owned(),
                ))
            } else {
                self.replay_run_identity.as_ref().map(|run_identity| {
                    TraceAgentInvocationContext::from_replay(run_identity, trace, execution_id)
                })
            };
            let mut driver = self
                .trace_driver
                .create(
                    WorkerIdentity {
                        worker_id: self.worker_id,
                    },
                    &trace,
                    &program.driver,
                )
                .map_err(|error| TraceError::Other(error.to_string()))?;
            is_staged_driver = driver.stage_bound().is_some();
            let (abort, registration) = AbortHandle::new_pair();
            self.active
                .borrow_mut()
                .insert(lifecycle_id, ActiveGraphTrace::Driver(abort));
            let driver_open_lifecycle = ActiveGraphTraceLifecycle {
                active: &self.active,
                lifecycle_id,
            };
            let driver_context = driver_invocation.as_ref().map_or_else(
                || TraceDriverContext::metadata_only(&trace),
                |invocation| {
                    TraceDriverContext::for_execution(
                        &trace,
                        &self.clock,
                        self.segments.as_ref(),
                        invocation,
                    )
                },
            );
            let opened = Abortable::new(driver.open(&program, &driver_context), registration).await;
            match opened {
                Ok(Ok(())) => {}
                Ok(Err(error)) => {
                    let _ = driver.abort_open().await;
                    return Err(TraceError::Other(error.to_string()));
                }
                Err(_) => {
                    let cleanup = driver.abort_open().await;
                    let message = match cleanup {
                        Ok(()) => "graph trace driver was cancelled during open".into(),
                        Err(error) => format!(
                            "graph trace driver was cancelled during open; rollback failed: {error}"
                        ),
                    };
                    return Err(TraceError::Cancelled(message));
                }
            }
            drop(driver_open_lifecycle);
            opened_driver = Some(driver);
            GraphTraceProgram::static_graph(program.profiling.clone())
        };
        let plan = &program.profiling;
        let profiling_trace_id = plan.trace.id.clone();
        let profiling_trajectory_id = format!("{}::trajectory", profiling_trace_id);
        let tool_dispatcher = opened_driver
            .as_ref()
            .and_then(|driver| driver.tool_dispatcher());
        if self.cancelled.get() {
            if let Some(mut driver) = opened_driver {
                let _ = self.close_driver(driver.as_mut()).await;
            }
            return Err(TraceError::Cancelled(format!(
                "graph trace {:?} was rejected after worker cancellation",
                plan.trace.id
            )));
        }
        let local_session = self.next_session.get();
        self.next_session.set(local_session.saturating_add(1));
        let session_num = (self.worker_id as u64)
            .checked_shl(48)
            .unwrap_or(0)
            .saturating_add(local_session);
        let terminal_nodes = terminal_graph_nodes(plan);
        // Mint the per-conversation cache-bust marker once for this trace
        // instance; every node dispatched for it shares the marker, so the
        // injected prefix is byte-stable across the conversation's turns. The
        // warmup-isolation targets only apply inside the warmup phase.
        let cache_bust_marker = self
            .cache_bust
            .as_ref()
            .filter(|cache_bust| {
                !matches!(
                    cache_bust.target,
                    crate::engine::graph_input::CacheBustTarget::WarmupIsolationSystem
                        | crate::engine::graph_input::CacheBustTarget::WarmupIsolationFirstTurn
                ) || self.phase == Phase::Warmup
            })
            .and_then(|cache_bust| cache_bust.marker(&plan.trace.id));
        let observer = self.observer.clone();
        let sink = Rc::new(EngineGraphSink {
            clock: self.clock.clone(),
            endpoint_runtime: self.endpoint_runtime.clone(),
            trace_id: plan.trace.id.clone(),
            nodes: RefCell::new(
                trace_warmup
                    .iter()
                    .map(|warmup| &warmup.graph)
                    .chain(std::iter::once(&plan.graph))
                    .flat_map(|graph| graph.nodes.iter())
                    .filter_map(|(node_id, node)| {
                        node.as_llm().map(|node| (node_id.clone(), node.clone()))
                    })
                    .collect(),
            ),
            prepared_metadata: RefCell::new(HashMap::new()),
            segments: self.segments.clone(),
            observer,
            phase: self.phase,
            worker_id: self.worker_id,
            session_num,
            model: self.model.clone(),
            default_max_tokens: self.default_max_tokens,
            run_origin_ns: self.run_origin_ns,
            raw_enabled: self.raw_enabled,
            terminal_nodes: RefCell::new(terminal_nodes),
            events: self.events.clone(),
            cache_bust_marker,
            cache_bust_target: self
                .cache_bust
                .as_ref()
                .map(|cache_bust| cache_bust.target)
                .unwrap_or(crate::engine::graph_input::CacheBustTarget::FirstTurnPrefix),
            tool_dispatcher: RefCell::new(tool_dispatcher),
            arrivals: Cell::new(0),
            terminal_records: Cell::new(0),
            profiling_records: Cell::new(0),
            replay_calls: Rc::new(RefCell::new(Vec::new())),
            replay_tools: Rc::new(RefCell::new(Vec::new())),
        });
        let mut local = LocalGraphTraceExecutionBackend::new(
            self.clock.clone(),
            self.materializer.clone(),
            sink.clone(),
        );
        local = local.with_executor_flags(ExecutorFlags {
            ignore_edge_delays: self.ignore_trace_delays,
            system_idle_gap_cap_ms: self.system_idle_gap_cap_seconds.map(|s| s * 1000.0),
            ..Default::default()
        });
        if let Some(policy) = &self.node_policy {
            local = local.with_node_policy(policy.clone());
        }
        // `Continue` treats a failed node as empty so the DAG can drain.
        let node_failure: Rc<dyn NodeFailurePolicy> = match self.on_failure {
            OnFailure::Abort => Rc::new(AbortTraceNodeFailurePolicy),
            OnFailure::Continue => Rc::new(ResilientNodeFailurePolicy),
        };
        local = local.with_node_failure(node_failure);
        let local = Rc::new(local);
        self.active
            .borrow_mut()
            .insert(execution_id, ActiveGraphTrace::Graph(local.clone()));
        let profile_started_ns = Cell::new(None);
        let mut staged_supplement = None;
        let result = async {
            if let Some(warmup) = trace_warmup {
                local
                    .execute_static_trace(warmup, Phase::Warmup, TraceSubphase::Warmup)
                    .await?;
            }
            profile_started_ns.set(Some(self.clock.now_ns()));
            if is_staged_driver {
                let driver = opened_driver.as_mut().ok_or_else(|| {
                    TraceError::Other("staged graph trace lost its opened driver".into())
                })?;
                let trace = driver_trace.as_ref().ok_or_else(|| {
                    TraceError::Other("staged graph trace lost its driver identity".into())
                })?;
                staged_supplement = Some(
                    self.execute_staged_driver(
                        local.as_ref(),
                        sink.as_ref(),
                        driver.as_mut(),
                        trace,
                        driver_invocation.as_ref(),
                        lifecycle_id,
                    )
                    .await?,
                );
                Ok(())
            } else {
                local
                    .execute_static_trace(program.profiling, self.phase, TraceSubphase::Profiling)
                    .await
            }
        }
        .await;
        let finalized = sink
            .verify_finalized_records()
            .map_err(|error| TraceError::Other(error.to_string()));
        self.active.borrow_mut().remove(&execution_id);
        let result = match (result, finalized) {
            (Err(primary), _) => Err(primary),
            (Ok(()), result) => result,
        };
        let replay_supplement =
            if should_emit_replay_supplement(self.phase, is_recorded_program, result.is_ok()) {
                let started = profile_started_ns.get().ok_or_else(|| {
                    TraceError::Other(
                        "recorded graph trace completed without profiling start".into(),
                    )
                })?;
                Some(
                    TraceTerminalSupplement::new(
                        self.run_origin_ns.to_string(),
                        profiling_trajectory_id.clone(),
                        profiling_trace_id.clone(),
                        self.worker_id,
                        "recorded_replay",
                    )
                    .with_planned_identity(PlannedReplayTraceInstance::from_cellular_instance(
                        ModuloCellPartition::from_env().map_or(0, |partition| partition.cell_id()),
                        profiling_trajectory_id,
                        profiling_trace_id,
                    ))
                    .with_profiling_measurements(
                        self.clock.now_ns().saturating_sub(started) as f64 / 1_000_000.0,
                        sink.replay_calls.borrow().clone(),
                        sink.replay_tools.borrow().clone(),
                    ),
                )
            } else {
                None
            };
        let terminal_supplement = result
            .is_ok()
            .then_some(staged_supplement)
            .flatten()
            .or(replay_supplement);
        if let Some(mut driver) = opened_driver {
            let close = self.close_driver(driver.as_mut()).await;
            let result = match (result, close) {
                (Err(primary), _) => Err(primary),
                (Ok(()), close) => close,
            };
            if result.is_ok()
                && let Some(supplement) = terminal_supplement
            {
                self.events
                    .emit(GraphExecutionEvent::TraceSupplement { supplement })?;
            }
            return result;
        }
        if result.is_ok()
            && let Some(supplement) = terminal_supplement
        {
            self.events
                .emit(GraphExecutionEvent::TraceSupplement { supplement })?;
        }
        result
    }

    fn cancel_inflight(&self) -> Result<(), TraceError> {
        self.cancelled.set(true);
        let active = self.active.borrow().values().cloned().collect::<Vec<_>>();
        let errors = active
            .iter()
            .filter_map(|trace| match trace {
                ActiveGraphTrace::Driver(abort) => {
                    abort.abort();
                    None
                }
                ActiveGraphTrace::Graph(backend) => backend.cancel_inflight().err(),
            })
            .map(|error| error.to_string())
            .collect::<Vec<_>>();
        if errors.is_empty() {
            Ok(())
        } else {
            Err(TraceError::Other(format!(
                "cancelling worker-local graph traces: {}",
                errors.join("; ")
            )))
        }
    }

    fn set_prefill_limit(&self, limit: usize) -> Result<(), TraceError> {
        let slots = self.prefill_slots.as_ref().ok_or_else(|| {
            TraceError::Other("graph worker has no configured prefill admission pool".into())
        })?;
        slots.set_limit(limit);
        Ok(())
    }
}

/// Replay terminal facts belong exclusively to successful profiling work. A warmup
/// trace can prime the same driver session but must never affect replay metrics.
fn should_emit_replay_supplement(phase: Phase, is_recorded_program: bool, succeeded: bool) -> bool {
    phase == Phase::Profiling && is_recorded_program && succeeded
}

struct FrozenStageOutputs {
    handles: BTreeMap<String, Handle>,
    store: Option<Arc<InMemorySegmentStore>>,
}

fn freeze_terminal_outputs(
    channels: &BTreeMap<String, ChanVal>,
    selected: &[String],
    base: &dyn SegmentStore,
) -> Result<FrozenStageOutputs, TraceError> {
    let concrete = selected
        .iter()
        .filter_map(|channel| {
            channels
                .get(channel)
                .and_then(ChanVal::as_value)
                .map(|value| (channel, value))
        })
        .collect::<Vec<_>>();
    if concrete.is_empty() {
        return Ok(FrozenStageOutputs {
            handles: BTreeMap::new(),
            store: None,
        });
    }

    let mut pool = SegmentPool::thaw(base);
    let handles = concrete
        .into_iter()
        .map(|(channel, value)| {
            serde_json::to_vec(value)
                .map_err(|error| {
                    TraceError::Other(format!(
                        "serializing declared terminal output {channel:?}: {error}"
                    ))
                })
                .and_then(|wire| {
                    pool.intern_raw(None, wire).map_err(|error| {
                        TraceError::Other(format!(
                            "freezing declared terminal output {channel:?}: {error}"
                        ))
                    })
                })
                .map(|handle| (channel.clone(), handle))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    Ok(FrozenStageOutputs {
        handles,
        store: Some(Arc::new(pool.freeze())),
    })
}

impl GraphWorkerBackend {
    /// Drive one finite sequence of driver-authored graph stages.
    ///
    /// The driver may choose the next graph only after receiving the immutable
    /// result of its prior stage. Placement independently enforces the driver's
    /// declared bound, so a faulty implementation cannot create an unbounded
    /// worker-local loop.
    async fn execute_staged_driver(
        &self,
        local: &LocalGraphTraceExecutionBackend<OpenAiChatMessage>,
        sink: &EngineGraphSink,
        driver: &mut dyn TraceProgramDriver,
        trace: &TraceIdentity,
        invocation: Option<&TraceAgentInvocationContext>,
        lifecycle_id: u64,
    ) -> Result<TraceTerminalSupplement, TraceError> {
        let _lifecycle = ActiveGraphTraceLifecycle {
            active: &self.active,
            lifecycle_id,
        };
        let bound = driver.stage_bound().ok_or_else(|| {
            TraceError::Other("staged graph driver omitted its finite stage bound".into())
        })?;
        let context = invocation.map_or_else(
            || TraceDriverContext::metadata_only(trace),
            |invocation| {
                TraceDriverContext::for_execution(
                    trace,
                    &self.clock,
                    self.segments.as_ref(),
                    invocation,
                )
            },
        );
        let mut executed_stages = 0;
        let mut terminal_output_store = None;
        loop {
            let (abort, registration) = AbortHandle::new_pair();
            self.active
                .borrow_mut()
                .insert(lifecycle_id, ActiveGraphTrace::Driver(abort));
            let directive = Abortable::new(driver.next_stage(&context), registration)
                .await
                .map_err(|_| {
                    TraceError::Cancelled(
                        "graph trace driver was cancelled during stage selection".into(),
                    )
                })?
                .map_err(|error| TraceError::Other(error.to_string()))?
                .ok_or_else(|| {
                    TraceError::Other(
                        "staged graph driver ended without a terminal supplement".into(),
                    )
                })?;
            match directive {
                TraceStageDirective::Complete(supplement) => {
                    return Ok(match terminal_output_store {
                        Some(store) => supplement.with_terminal_output_store(store),
                        None => supplement,
                    });
                }
                TraceStageDirective::Execute(plan) => {
                    if executed_stages >= bound.get() {
                        return Err(TraceError::Other(format!(
                            "staged graph driver exceeded its declared {}-stage bound",
                            bound
                        )));
                    }
                    validate_native_graph_trace_plan(&plan)
                        .map_err(|error| TraceError::Other(error.to_string()))?;
                    sink.configure_stage(&plan, driver.tool_dispatcher());
                    let result = local
                        .execute_static_trace_result(plan, self.phase, TraceSubphase::Profiling)
                        .await?;
                    let base = terminal_output_store
                        .as_deref()
                        .map_or(self.segments.as_ref(), |store| store as &dyn SegmentStore);
                    let frozen_outputs = freeze_terminal_outputs(
                        &result.channels,
                        driver.terminal_output_channels(),
                        base,
                    )?;
                    if let Some(store) = frozen_outputs.store {
                        terminal_output_store = Some(store);
                    }
                    let channels = result
                        .channels
                        .into_iter()
                        .map(|(channel, value)| {
                            serde_json::to_value(value)
                                .map(|value| (channel, value))
                                .map_err(|error| {
                                    TraceError::Other(format!(
                                        "serializing staged graph channel: {error}"
                                    ))
                                })
                        })
                        .collect::<Result<BTreeMap<_, _>, _>>()?;
                    let (abort, registration) = AbortHandle::new_pair();
                    self.active
                        .borrow_mut()
                        .insert(lifecycle_id, ActiveGraphTrace::Driver(abort));
                    Abortable::new(
                        driver.observe_stage(TraceStageResult {
                            plan_identity: format!("{}::stage-{executed_stages}", trace.trace_id),
                            terminal_status: GraphReplyStatus::Completed,
                            channels,
                            output_handles: frozen_outputs.handles,
                        }),
                        registration,
                    )
                    .await
                    .map_err(|_| {
                        TraceError::Cancelled(
                            "graph trace driver was cancelled during stage observation".into(),
                        )
                    })?
                    .map_err(|error| TraceError::Other(error.to_string()))?;
                    executed_stages += 1;
                }
            }
        }
    }

    /// Close an opened driver to completion before returning cancellation.
    ///
    /// A driver open remains cancellable to the caller, while its explicit
    /// `abort_open` rollback releases any partially provisioned state before
    /// placement returns cancellation. Once ownership transfers to placement,
    /// dispatcher and sandbox teardown is asynchronous and owns external resources.
    /// The worker therefore awaits close to completion. Resource-owning drivers
    /// place Clock-driven bounds inside their teardown layers so a generic deadline
    /// cannot drop an in-flight cleanup future.
    async fn close_driver(
        &self,
        driver: &mut dyn crate::graph::driver::TraceProgramDriver,
    ) -> Result<(), TraceError> {
        let close = driver
            .close()
            .await
            .map_err(|error| TraceError::Other(error.to_string()));
        if self.cancelled.get() {
            let message = match close {
                Ok(()) => "graph trace driver finished teardown after cancellation".into(),
                Err(error) => {
                    format!("graph trace driver finished teardown after cancellation: {error}")
                }
            };
            Err(TraceError::Cancelled(message))
        } else {
            close
        }
    }
}

/// Authored node dispatch metadata decoded once per node from the immutable
/// segment catalog. All fields are node-invariant, so a sink caches one instance
/// per node id and clones the parsed values into each dispatch rather than
/// re-parsing the backing segment bytes on every fire.
struct PreparedNodeMetadata {
    extra_headers: BTreeMap<String, String>,
    raw_tools: Option<Vec<Value>>,
    raw_system: Option<Vec<Value>>,
    extra_body: Option<Map<String, Value>>,
    parameters: BTreeMap<String, String>,
    /// Wire image count this node's body is known to carry, or `None` when it
    /// cannot be established without parsing the serialized body. See
    /// [`node_known_image_count`].
    known_image_count: Option<u32>,
}

/// The exact wire image count of a graph node's request body, when the node's
/// authored prompt program makes it knowable without parsing the serialized body.
///
/// Two of the four [`PromptItem`] variants are text-only, and the invariant is
/// held by their *producers*, not by any check on this path:
///
/// - `Seg` resolves to a `Payload::Message` wire. That payload is explicitly
///   allowed to be multimodal in general (see `dataset::segment`, which folds
///   the wire into the message identity for exactly that reason), so the
///   guarantee here is only that every graph `Seg` handle is interned from a
///   string-content message: `graph::recorded::trie` serializes `MessageWire {
///   role, content: &str }` and `graph::conditional::fold` serializes
///   `OpenAiChatMessage`, whose `content` is a `String`.
///
/// - `Text` renders `{"role", "content": <utf8>}` from a `Payload::Text`.
///
/// A new graph adapter that mints a `Seg` from a lowered multimodal turn (the
/// shape `dataset::dataset` interns for the scheduled path) would break this and
/// must extend the fallback below. Note that `graph::lowering`'s `MediaKind`
/// rejection does *not* cover it: that guards `turn.content`, while the
/// `turn.messages` loop that emits `Seg` items only checks the payload variant
/// and never inspects the wire.
///
/// The other two variants carry authored JSON that nothing on the graph path
/// inspects for content kind:
///
/// - `RawMessages`: `graph::materialize::raw_message_wires` splits an authored
///   `dag_jsonl` array into per-object wires and validates only object-ness.
///
/// - `Splice`: resolves against channel state, which a trace's authored
///   `initial_state`/`replay_outputs` seeds verbatim
///   (`graph::channel_store::channel_value` builds `EncodedMessages` straight
///   from the authored array). A splice of a *reply* is text-only, but the node
///   program cannot tell a seeded channel from a produced one without the
///   trace's state, which is not available here — so any splice falls back.
///
/// `Some(0)` lets the dispatch path skip a full body reparse whose only consumer
/// is `num_images`; `None` preserves that reparse.
///
/// Endpoint field assembly does not reintroduce images for the dialects a graph
/// node can reach: `extra_body`/`extra` merge into the payload map, but the
/// message-array dialects (chat, responses, anthropic) reserve the content root,
/// so the merged value is discarded and refilled from the rendered wires
/// (`BodyPlan::fill_reserved`). `raw_system` renders to Anthropic's top-level
/// `system` key, which the payload extractor does not walk.
fn node_known_image_count(node: &LlmNode) -> Option<u32> {
    let carries_authored_json = node.items.iter().any(|item| {
        matches!(
            item,
            PromptItem::RawMessages { .. } | PromptItem::Splice { .. }
        )
    });
    (!carries_authored_json).then_some(0)
}

struct EngineGraphSink {
    clock: Rc<dyn Clock>,
    endpoint_runtime: Rc<dyn GraphEndpointRuntime>,
    trace_id: String,
    nodes: RefCell<HashMap<String, LlmNode>>,
    /// Authored per-node metadata decoded from immutable segment bytes, cached on
    /// first dispatch of each node. These handles (`tools`, raw system, extra
    /// body/headers, request parameters) never change for a node, so bounded and
    /// loop cycling that re-fires the same node reuses the parsed values instead
    /// of re-decoding all five segments from bytes on every dispatch.
    prepared_metadata: RefCell<HashMap<String, Rc<PreparedNodeMetadata>>>,
    segments: Arc<dyn SegmentStore>,
    observer: Rc<NativeMetricsObserver>,
    phase: Phase,
    worker_id: usize,
    session_num: u64,
    model: String,
    default_max_tokens: usize,
    run_origin_ns: i64,
    raw_enabled: bool,
    terminal_nodes: RefCell<HashSet<String>>,
    events: Arc<dyn GraphExecutionEventSink>,
    /// Cache-bust marker for this trace, minted once per trace instance. `None`
    /// when cache-bust is disabled, and also `None` for a warmup-isolation
    /// target outside the warmup phase.
    cache_bust_marker: Option<String>,
    cache_bust_target: crate::engine::graph_input::CacheBustTarget,
    /// Requests this trace registered with the shared observer (one per dispatched
    /// node). Compared against `terminal_records` at finalization; a shared
    /// worker observer makes its global `arrival_count` cumulative, so the
    /// per-trace invariant is tracked on the sink instead.
    arrivals: Cell<u64>,
    terminal_records: Cell<u64>,
    profiling_records: Cell<u64>,
    tool_dispatcher: RefCell<Option<Rc<dyn ToolDispatcher>>>,
    /// Completed profiling-only LLM timing facts, retained until the trace terminal.
    replay_calls: Rc<RefCell<Vec<ReplayCallMeasurement>>>,
    /// Attempted profiling-only tool timings, retained without command/output bytes.
    replay_tools: Rc<RefCell<Vec<ToolCallMeasurement>>>,
}

#[async_trait(?Send)]
impl GraphSink<OpenAiChatMessage> for EngineGraphSink {
    async fn dispatch_tool_node(
        &self,
        node_id: &str,
        node: &crate::graph::model::ToolNode,
        context: &GraphDispatchContext,
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        let dispatcher = self
            .tool_dispatcher
            .borrow()
            .clone()
            .ok_or_else(|| anyhow!("graph tool node {node_id:?} has no trace dispatcher"))?;
        let mut output = String::new();
        for (index, command) in node.commands.iter().enumerate() {
            let result = dispatcher
                .dispatch(
                    ToolDispatchRequest::new(format!("{node_id}:{index}"), command),
                    &ToolDispatchContext {
                        timeout_ns: node.timeout_ns,
                    },
                )
                .await?;
            if context.trace_subphase == TraceSubphase::Profiling {
                let call_index = self.replay_tools.borrow().len();
                self.replay_tools.borrow_mut().push(
                    ToolCallMeasurement::new(
                        result.duration_ns as f64 / 1_000_000_000.0,
                        dispatcher.backend_identity().artifact_label(),
                    )
                    .with_call_index(call_index),
                );
            }
            output.push_str(&String::from_utf8_lossy(&result.output));
        }
        Ok(GraphReply::from_text(output))
    }
    async fn dispatch(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        self.dispatch_request_with_context(
            node_id,
            crate::graph::materialize::MaterializedGraphRequest {
                messages,
                tools: None,
                model: None,
                additional_body: None,
                max_tokens,
                streaming: true,
            },
            &GraphDispatchContext {
                phase: self.phase,
                trace_subphase: TraceSubphase::Profiling,
                trace_instance: TraceInstanceId::new(self.trace_id.clone()),
            },
            GraphDispatchOptions::default(),
            on_first_token,
        )
        .await
    }

    async fn dispatch_request_with_context(
        &self,
        node_id: &str,
        request: crate::graph::materialize::MaterializedGraphRequest,
        context: &GraphDispatchContext,
        options: GraphDispatchOptions,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        let crate::graph::materialize::MaterializedGraphRequest {
            messages,
            tools,
            model: model_override,
            additional_body,
            max_tokens,
            streaming,
        } = request;
        let node = self
            .nodes
            .borrow()
            .get(node_id)
            .cloned()
            .ok_or_else(|| anyhow!("graph node {node_id:?} has no execution metadata"))?;
        let model = model_override
            .or_else(|| metadata_string(&node, "model").map(str::to_string))
            .unwrap_or_else(|| self.model.clone());
        // Fast path: splice the already-serialized message wires verbatim through
        // `Turn.lowered` instead of decoding each into a serde_json::Value and
        // letting the endpoint re-serialize it. The materializer already produced
        // these bytes, and the endpoint's `RenderedMessage::Wire` path consumes
        // them directly, so this removes a per-request decode + Value clone +
        // re-encode round-trip (the dominant worker-side allocation). It matches
        // how the scheduled path carries load-time message bytes. Restricted to
        // profiling with no cache-bust marker: a cache-bust marker mutates message
        // content (needs a Value), and warmup makes the endpoint render the first
        // turn to a mutable value for the system prepend (`render_first`).
        let wire_messages = self.cache_bust_marker.is_none() && context.phase == Phase::Profiling;
        let (raw_messages, lowered) = if wire_messages {
            (None, Some(messages.into_iter().collect()))
        } else {
            let mut raw_messages = messages
                .iter()
                .enumerate()
                .map(|(index, wire)| {
                    serde_json::from_slice(wire).with_context(|| {
                        format!("decoding materialized graph message {index} for node {node_id:?}")
                    })
                })
                .collect::<Result<Vec<Value>>>()?;
            // Idempotent injection keeps accumulated requests on one cache-bust prefix.
            if let Some(marker) = self.cache_bust_marker.as_deref() {
                prepend_cache_bust_marker(&mut raw_messages, marker, self.cache_bust_target);
            }
            (Some(raw_messages), None)
        };
        let prepared = self.prepared_metadata(node_id, &node)?;
        let extra_headers = uniquify_dynamo_session_headers(
            prepared.extra_headers.clone(),
            context.phase,
            &self.trace_id,
        );
        let mut extra_body = additional_body
            .as_deref()
            .map(serde_json::from_slice)
            .transpose()
            .context("decoding typed graph additional body")?
            .or(prepared.extra_body.clone())
            .unwrap_or_default();
        if let Some(generation) = native_graph_generation_body(&node)? {
            for (field, value) in generation {
                extra_body.entry(field).or_insert(value);
            }
        }
        let turn = Turn {
            model: Some(model.clone()),
            max_tokens: max_tokens
                .map(u32::try_from)
                .transpose()
                .context("graph max_tokens exceeds u32")?,
            raw_messages,
            lowered,
            raw_tools: tools
                .as_deref()
                .map(serde_json::from_slice)
                .transpose()
                .context("decoding typed graph tools")?
                .or(prepared.raw_tools.clone()),
            raw_system: prepared.raw_system.clone(),
            extra_body: (!extra_body.is_empty()).then_some(extra_body),
            ..Turn::default()
        };
        let max_output_tokens = max_tokens.unwrap_or(self.default_max_tokens);
        let dispatch = self.endpoint_runtime.materialize(GraphEndpointRequest {
            selector: metadata_string(&node, "endpoint").map(str::to_owned),
            model: model.clone(),
            turn,
            streaming,
            authored_input_tokens: metadata_u64(&node, "input_tokens").unwrap_or(0),
            max_output_tokens,
            extra_headers,
            parameters: prepared.parameters.clone(),
            trace_id: self.trace_id.clone(),
            is_final_turn: self.terminal_nodes.borrow().contains(node_id),
            cancel_after_ns: options.cancel_after_ns,
            session_num: self.session_num,
            phase: context.phase,
            known_image_count: prepared.known_image_count,
        })?;
        let GraphEndpointDispatch {
            transport,
            request,
            endpoint,
            input_tokens,
        } = dispatch;
        // Lower the recorded api_time (and TTFT when present) onto the request so
        // the `dry_run` transport can reproduce the recorded timeline exactly under
        // the `recorded` latency model. Absent on non-recorded graphs → analytic.
        let mut request = request;
        request.recorded_api_time_ns =
            metadata_u64(&node, "recorded_api_time_us").map(|us| (us as i64).saturating_mul(1_000));
        request.recorded_ttft_ns =
            metadata_u64(&node, "recorded_ttft_us").map(|us| (us as i64).saturating_mul(1_000));
        let uuid = request.uuid;
        let dimensions: InferenceDimensions =
            Dispatcher::inference_dimensions(transport.as_ref(), &request);
        // `register_metadata` discards `worker_id`/`conversation_id` and blanks
        // `correlation_id` unless per-record dimensions are retained, so only pay
        // for these owned strings when the run will actually keep them. Blank
        // `correlation_id` matches the value `register_metadata` would substitute.
        let retain_dimensions = self.observer.retains_record_dimensions();
        self.observer.register_metadata(
            uuid,
            RequestMetricMetadata {
                phase: context.phase,
                session_num: Some(self.session_num),
                turn_index: metadata_u64(&node, "turn_index")
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or(0),
                worker_id: retain_dimensions.then(|| self.worker_id.to_string().into()),
                // One authored root may be cycled many times by a bounded or
                // duration phase. Metrics identity follows the unique trace
                // instance; the authored conversation stays in node metadata.
                conversation_id: retain_dimensions.then(|| self.trace_id.clone()),
                dimensions,
                correlation_id: Some(if retain_dimensions {
                    format!("{}:{node_id}", self.trace_id)
                } else {
                    String::new()
                }),
                has_credit_timestamp: false,
                ..RequestMetricMetadata::default()
            },
        );
        let arrival_ms =
            self.clock.now_ns().saturating_sub(self.run_origin_ns) as f64 / 1_000_000.0;
        self.observer.on_arrival(
            uuid,
            arrival_ms,
            usize::try_from(input_tokens).unwrap_or(usize::MAX),
            max_output_tokens,
        );
        let first_token_emitted = Cell::new(false);
        let first_token_error = RefCell::new(None);
        let emits_profile_events = context.trace_subphase == TraceSubphase::Profiling;
        let collected = transport
            .dispatch_collect(
                PreparedTurn {
                    runtime_session_id: self.trace_id.clone(),
                    request,
                    model,
                    endpoint,
                    endpoint_aware: true,
                    data_policy: crate::multiturn::TurnDataPolicy::ordinary(),
                    // Graph traces are dispatched directly by their placement and
                    // never routed as credits.
                    deferred: None,
                },
                self.observer.as_ref(),
                &|_| {
                    if emits_profile_events
                        && !first_token_emitted.replace(true)
                        && let Err(error) = self.events.emit(GraphExecutionEvent::FirstToken {
                            trace_id: self.trace_id.clone(),
                            uuid,
                        })
                    {
                        *first_token_error.borrow_mut() = Some(error);
                    }
                    on_first_token();
                },
            )
            .await?;
        self.arrivals.set(self.arrivals.get().saturating_add(1));
        if let Some(error) = first_token_error.into_inner() {
            return Err(anyhow!("emitting graph first-token event: {error}"));
        }
        let outcome = collected.outcome;
        self.observer.record_response(
            uuid,
            NativeResponseMetadata {
                start_ns: Some(outcome.start_ns),
                end_ns: Some(outcome.end_ns),
                prompt_tokens: outcome.prompt_tokens,
                completion_tokens: outcome.completion_tokens,
                http: outcome.http,
            },
        );
        let terminal_ordinal = self.terminal_records.get();
        let profiling_ordinal = self.record_ordinal(emits_profile_events);
        let ingest = self
            .observer
            .drain_terminal_record(
                uuid,
                if emits_profile_events {
                    profiling_ordinal
                } else {
                    terminal_ordinal
                },
            )
            .ok_or_else(|| {
                anyhow!(
                    "graph trace {:?} could not snapshot terminal node {node_id:?}",
                    self.trace_id
                )
            })?;
        let record = CapturedRecord {
            uuid,
            x_correlation_id: ingest.correlation_id.clone(),
            output: CapturedModelOutput::from_parts(
                &outcome.response_text,
                outcome.model_response.content.as_deref(),
                outcome.model_response.reasoning.as_deref(),
            ),
            raw: self.raw_enabled.then_some(CapturedHttpExchange {
                request_payload: collected.request_payload.clone(),
                record: collected.record,
            }),
            ingest,
        };
        if emits_profile_events {
            let call_index = self.replay_calls.borrow().len();
            self.replay_calls.borrow_mut().push(ReplayCallMeasurement {
                trace_id: self.trace_id.clone(),
                call_index,
                raw_end_to_end_ms: (outcome.end_ns - outcome.start_ns).max(0) as f64 / 1_000_000.0,
                raw_inference_ms: (outcome.end_ns - outcome.start_ns).max(0) as f64 / 1_000_000.0,
                raw_generation_ms: record
                    .ingest
                    .first_token_ns
                    .map(|first| outcome.end_ns.saturating_sub(first).max(0) as f64 / 1_000_000.0)
                    .unwrap_or_default(),
                ttft_ms: record.ingest.first_token_ns.map(|first| {
                    first.saturating_sub(outcome.start_ns).max(0) as f64 / 1_000_000.0
                }),
                stream_total_ms: Some(
                    (outcome.end_ns - outcome.start_ns).max(0) as f64 / 1_000_000.0,
                ),
                observed_isl: record.ingest.tokens.input.unwrap_or_default(),
                observed_osl: record.ingest.tokens.output.unwrap_or_default(),
                target_osl: max_output_tokens as u64,
                recorded_prompt_isl: Some(input_tokens),
                sse_event_count: record.ingest.token_arrival_ns.len() as u64,
                has_meaningful_output: record.output.response_text.is_some()
                    || record.output.reasoning_text.is_some(),
                has_done: matches!(outcome.terminal, ReplayTerminalStatus::Completed),
                has_required_usage: record.ingest.usage.completion_tokens.unwrap_or_default() > 0,
            });
        }
        if context.trace_subphase == TraceSubphase::Profiling {
            self.events.emit(GraphExecutionEvent::Record {
                trace_id: self.trace_id.clone(),
                record: Box::new(record),
                node_id: Some(node_id.to_owned()),
            })?;
        }
        self.did_drain_terminal_record(emits_profile_events);

        Ok(match outcome.terminal {
            ReplayTerminalStatus::Completed => GraphReply::from_text(outcome.response_text),
            ReplayTerminalStatus::Canceled => GraphReply::cancelled(),
            ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed => GraphReply::failed(),
        })
    }
}

fn native_graph_generation_body(node: &LlmNode) -> Result<Option<Map<String, Value>>> {
    let Some(generation) = node.metadata.get(GENERATION_METADATA_KEY) else {
        return Ok(None);
    };
    let body = generation.as_object().ok_or_else(|| {
        anyhow!(
            "graph node generation metadata {:?} must be an object",
            GENERATION_METADATA_KEY
        )
    })?;
    const FIELDS: &[&str] = &[
        "min_tokens",
        "temperature",
        "top_p",
        "top_k",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
    ];
    if let Some(field) = body.keys().find(|field| !FIELDS.contains(&field.as_str())) {
        anyhow::bail!("graph node generation metadata has unsupported field {field:?}");
    }
    Ok(Some(body.clone()))
}

impl EngineGraphSink {
    /// Install one driver-authorized stage before the shared executor runs it.
    ///
    /// Stage identity remains trace-local while node metadata and terminal-node
    /// classification are replaced atomically on this worker thread. Parsed
    /// metadata cannot carry across an authored rewrite of the same node id.
    fn configure_stage(&self, plan: &GraphTracePlan, dispatcher: Option<Rc<dyn ToolDispatcher>>) {
        *self.nodes.borrow_mut() = plan
            .graph
            .nodes
            .iter()
            .filter_map(|(node_id, node)| node.as_llm().map(|node| (node_id.clone(), node.clone())))
            .collect();
        self.prepared_metadata.borrow_mut().clear();
        *self.terminal_nodes.borrow_mut() = terminal_graph_nodes(plan);
        *self.tool_dispatcher.borrow_mut() = dispatcher;
    }

    /// Return the dense ordinal exported with this terminal record. Warmup
    /// terminals are drained to keep the shared observer bounded, but they do
    /// not reserve a profiling record row.
    fn record_ordinal(&self, is_profiling: bool) -> u64 {
        if is_profiling {
            self.profiling_records.get()
        } else {
            self.terminal_records.get()
        }
    }

    fn did_drain_terminal_record(&self, is_profiling: bool) {
        self.terminal_records
            .set(self.terminal_records.get().saturating_add(1));
        if is_profiling {
            self.profiling_records
                .set(self.profiling_records.get().saturating_add(1));
        }
    }

    fn verify_finalized_records(&self) -> Result<()> {
        // The observer is shared across the worker's traces, so its global
        // `arrival_count`/retained counts are cumulative; the per-trace invariant
        // is that every request this trace registered emitted exactly one record.
        // Each emitted record was drained (`take_terminal` removed its slot), so
        // `arrivals == emitted` also implies this trace left nothing retained.
        let arrivals = self.arrivals.get();
        let emitted = self.terminal_records.get();
        ensure!(
            arrivals == emitted,
            "graph trace {:?} registered {arrivals} requests but emitted {emitted} metric records",
            self.trace_id
        );
        Ok(())
    }

    fn metadata_handle(&self, node: &LlmNode, name: &str) -> Result<Option<Handle>> {
        metadata_u64(node, name)
            .map(|index| {
                u32::try_from(index)
                    .map(Handle::new)
                    .map_err(|_| anyhow!("graph metadata {name:?} handle exceeds u32"))
            })
            .transpose()
    }

    /// Decode (once per node) and cache the authored dispatch metadata backing
    /// this node's segment handles, so repeated dispatches of the same node reuse
    /// the parsed values instead of re-reading segment bytes each fire. Errors
    /// surface on the first dispatch of the offending node, exactly as when the
    /// handles were decoded inline per dispatch.
    fn prepared_metadata(&self, node_id: &str, node: &LlmNode) -> Result<Rc<PreparedNodeMetadata>> {
        if let Some(prepared) = self.prepared_metadata.borrow().get(node_id).cloned() {
            return Ok(prepared);
        }
        let prepared = Rc::new(PreparedNodeMetadata {
            extra_headers: self.raw_string_map(node, "extra_headers_handle")?,
            raw_tools: self.raw_array(node, "tools_handle")?,
            raw_system: self.raw_array(node, "raw_system_handle")?,
            extra_body: self.raw_object(node, "extra_body_handle")?,
            parameters: self.raw_string_map(node, "request_parameters_handle")?,
            known_image_count: node_known_image_count(node),
        });
        self.prepared_metadata
            .borrow_mut()
            .insert(node_id.to_string(), prepared.clone());
        Ok(prepared)
    }

    fn raw_value(&self, node: &LlmNode, name: &str) -> Result<Option<Value>> {
        let Some(handle) = self.metadata_handle(node, name)? else {
            return Ok(None);
        };
        let wire = match self.segments.get(handle)? {
            Payload::Raw { wire } | Payload::Message { wire, .. } => wire,
            payload => {
                return Err(anyhow!(
                    "graph metadata {name:?} handle {handle} contains {}, expected raw JSON",
                    payload.kind_name()
                ));
            }
        };
        serde_json::from_slice(wire)
            .map(Some)
            .with_context(|| format!("decoding graph metadata {name:?} handle {handle}"))
    }

    fn raw_array(&self, node: &LlmNode, name: &str) -> Result<Option<Vec<Value>>> {
        match self.raw_value(node, name)? {
            None => Ok(None),
            Some(Value::Array(values)) => Ok(Some(values)),
            Some(_) => Err(anyhow!("graph metadata {name:?} must be a JSON array")),
        }
    }

    fn raw_object(&self, node: &LlmNode, name: &str) -> Result<Option<Map<String, Value>>> {
        match self.raw_value(node, name)? {
            None => Ok(None),
            Some(Value::Object(values)) => Ok(Some(values)),
            Some(_) => Err(anyhow!("graph metadata {name:?} must be a JSON object")),
        }
    }

    fn raw_string_map(&self, node: &LlmNode, name: &str) -> Result<BTreeMap<String, String>> {
        let Some(values) = self.raw_object(node, name)? else {
            return Ok(BTreeMap::new());
        };
        values
            .into_iter()
            .map(|(key, value)| match value {
                Value::String(value) => Ok((key, value)),
                _ => Err(anyhow!(
                    "graph metadata {name:?} field {key:?} must be a string"
                )),
            })
            .collect()
    }
}

fn terminal_graph_nodes(plan: &GraphTracePlan) -> HashSet<String> {
    let nonterminal = plan
        .graph
        .edges
        .iter()
        .filter(|edge| edge.target != crate::graph::model::END_NODE_ID)
        .map(|edge| edge.source.as_str())
        .collect::<HashSet<_>>();
    plan.graph
        .nodes
        .keys()
        .filter(|node| !nonterminal.contains(node.as_str()))
        .cloned()
        .collect()
}

fn metadata_string<'a>(node: &'a LlmNode, name: &str) -> Option<&'a str> {
    node.metadata.get(name).and_then(Value::as_str)
}

fn metadata_u64(node: &LlmNode, name: &str) -> Option<u64> {
    node.metadata.get(name).and_then(Value::as_u64)
}

fn uniquify_dynamo_session_headers(
    mut headers: BTreeMap<String, String>,
    phase: Phase,
    trace_id: &str,
) -> BTreeMap<String, String> {
    if !headers.contains_key("x-dynamo-session-id") {
        return headers;
    }
    let Some((_, instance_nonce)) = trace_id.split_once("::") else {
        return headers;
    };
    let variant = match phase {
        Phase::Warmup => "warmup",
        Phase::Profiling => "profiling",
    };
    let suffix = format!("::{variant}-{instance_nonce}");
    for name in ["x-dynamo-session-id", "x-dynamo-parent-session-id"] {
        if let Some(value) = headers.get_mut(name) {
            value.push_str(&suffix);
        }
    }
    headers
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use crate::clock::SimClock;
    use crate::dataset::InMemorySegmentStore;
    use crate::endpoints::{
        ChatEndpoint, EffectiveEndpointConfig, Endpoint, EndpointDescriptor, EndpointFactory,
        EndpointRegistry, EndpointRegistryBuilder, EndpointResult, PreparedEndpoint,
        RawEndpointConfig, StatelessEndpointFactory,
    };
    use crate::engine::graph_input::CacheBustTarget;
    use crate::graph::driver::{
        RecordedReplayAgentLoopFactories, RecordedReplayTraceProgramDriverFactory,
        TraceDriverCapabilities, TraceDriverContext, TraceDriverError, TraceDriverSpec,
        TraceIdentity, TraceProgramDriver, TraceProgramDriverFactory, TraceStageDirective,
        TraceStageResult, WorkerIdentity,
    };
    use crate::graph::model::ExecutableGraphNode;
    use crate::graph::supplement::TraceTerminalSupplement;
    use crate::graph::{agent, tools};
    use crate::multiturn::AuthoredInputTokenCounter;
    use crate::transport::core::ConnectionReuseStrategy;

    fn node_with_items(items: Vec<PromptItem>) -> LlmNode {
        LlmNode {
            output: "out".into(),
            streaming: true,
            inputs: Vec::new(),
            min_start_delay_us: None,
            max_tokens: None,
            items,
            request: None,
            metadata: BTreeMap::new(),
        }
    }

    struct RecordingTraceDriverFactory {
        created: Arc<AtomicUsize>,
        ran: Arc<AtomicUsize>,
    }

    impl TraceProgramDriverFactory for RecordingTraceDriverFactory {
        fn capabilities(
            &self,
            _spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            Ok(TraceDriverCapabilities::default())
        }

        fn create(
            &self,
            _worker: WorkerIdentity,
            _trace: &TraceIdentity,
            _spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            self.created.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(RecordingTraceDriver {
                ran: self.ran.clone(),
            }))
        }
    }

    struct RecordingTraceDriver {
        ran: Arc<AtomicUsize>,
    }

    struct StagedTraceDriverFactory {
        observations: Arc<Mutex<Vec<String>>>,
        dispatches: Arc<Mutex<Vec<String>>>,
        blocks_open: bool,
        blocks_stage: bool,
        feedback_stage: bool,
    }

    impl TraceProgramDriverFactory for StagedTraceDriverFactory {
        fn capabilities(
            &self,
            _spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            Ok(TraceDriverCapabilities {
                has_live_turns: true,
                ..TraceDriverCapabilities::default()
            })
        }

        fn create(
            &self,
            _worker: WorkerIdentity,
            trace: &TraceIdentity,
            _spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            Ok(Box::new(StagedTraceDriver {
                trace: trace.clone(),
                observations: self.observations.clone(),
                dispatcher: Rc::new(OrderedToolDispatcher(self.dispatches.clone())),
                blocks_open: self.blocks_open,
                blocks_stage: self.blocks_stage,
                feedback_stage: self.feedback_stage,
                state: 0,
            }))
        }
    }

    struct StagedTraceDriver {
        trace: TraceIdentity,
        observations: Arc<Mutex<Vec<String>>>,
        dispatcher: Rc<dyn tools::ToolDispatcher>,
        blocks_open: bool,
        blocks_stage: bool,
        feedback_stage: bool,
        state: u8,
    }

    struct TerminalOutputStagedTraceDriverFactory {
        observations: Arc<Mutex<Vec<TraceStageResult>>>,
    }

    impl TraceProgramDriverFactory for TerminalOutputStagedTraceDriverFactory {
        fn capabilities(
            &self,
            _spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            Ok(TraceDriverCapabilities {
                has_live_turns: true,
                ..TraceDriverCapabilities::default()
            })
        }

        fn create(
            &self,
            _worker: WorkerIdentity,
            trace: &TraceIdentity,
            _spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            Ok(Box::new(TerminalOutputStagedTraceDriver {
                trace: trace.clone(),
                observations: self.observations.clone(),
                dispatcher: Rc::new(TerminalOutputToolDispatcher),
                terminal_outputs: vec!["tool-output".into()],
                selected_outputs: BTreeMap::new(),
                state: 0,
            }))
        }
    }

    struct TerminalOutputStagedTraceDriver {
        trace: TraceIdentity,
        observations: Arc<Mutex<Vec<TraceStageResult>>>,
        dispatcher: Rc<dyn tools::ToolDispatcher>,
        terminal_outputs: Vec<String>,
        selected_outputs: BTreeMap<String, Handle>,
        state: u8,
    }

    #[async_trait(?Send)]
    impl TraceProgramDriver for TerminalOutputStagedTraceDriver {
        fn tool_dispatcher(&self) -> Option<Rc<dyn tools::ToolDispatcher>> {
            Some(self.dispatcher.clone())
        }

        fn stage_bound(&self) -> Option<std::num::NonZeroU32> {
            std::num::NonZeroU32::new(2)
        }

        fn terminal_output_channels(&self) -> &[String] {
            &self.terminal_outputs
        }

        async fn next_stage(
            &mut self,
            _context: &TraceDriverContext<'_>,
        ) -> Result<Option<TraceStageDirective>, TraceDriverError> {
            match self.state {
                0 => {
                    self.state = 1;
                    Ok(Some(TraceStageDirective::Execute(tool_only_plan(
                        &self.trace.trace_id,
                        &["static-terminal"],
                    ))))
                }
                2 => {
                    self.state = 3;
                    Ok(Some(TraceStageDirective::Execute(tool_only_plan(
                        &self.trace.trace_id,
                        &["dynamic-terminal"],
                    ))))
                }
                4 => {
                    self.state = 5;
                    Ok(Some(TraceStageDirective::Complete(
                        TraceTerminalSupplement::new(
                            self.trace.run_id.clone(),
                            self.trace.trajectory_id.clone(),
                            self.trace.trace_id.clone(),
                            7,
                            "terminal-output-staged-test",
                        )
                        .with_terminal_outputs(self.selected_outputs.clone()),
                    )))
                }
                _ => Err(TraceDriverError::new(
                    "terminal-output staged test driver advanced out of order",
                )),
            }
        }

        async fn observe_stage(
            &mut self,
            result: TraceStageResult,
        ) -> Result<(), TraceDriverError> {
            if !matches!(self.state, 1 | 3) || result.terminal_status != GraphReplyStatus::Completed
            {
                return Err(TraceDriverError::new(
                    "terminal-output staged test driver observed an invalid result",
                ));
            }
            let handle = result
                .output_handles
                .get("tool-output")
                .copied()
                .ok_or_else(|| {
                    TraceDriverError::new("executor did not provide the declared terminal output")
                })?;
            self.selected_outputs.insert("tool-output".into(), handle);
            self.observations.lock().unwrap().push(result);
            self.state += 1;
            Ok(())
        }

        async fn run(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            unreachable!("terminal-output staged driver must execute through the stage loop")
        }
    }

    struct TerminalOutputToolDispatcher;

    #[async_trait(?Send)]
    impl tools::ToolDispatcher for TerminalOutputToolDispatcher {
        async fn open_trace(
            &self,
            _context: tools::TraceOpenContext<'_>,
        ) -> Result<(), tools::ToolDispatchError> {
            Ok(())
        }

        async fn dispatch(
            &self,
            request: tools::ToolDispatchRequest,
            _context: &tools::ToolDispatchContext,
        ) -> Result<tools::ToolDispatchResult, tools::ToolDispatchError> {
            Ok(tools::ToolDispatchResult::completed(
                request.call_id,
                0,
                Bytes::from(request.command),
            ))
        }

        async fn close_trace(
            &self,
            _trace: &TraceIdentity,
        ) -> Result<(), tools::ToolDispatchError> {
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl TraceProgramDriver for StagedTraceDriver {
        async fn open(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<(), TraceDriverError> {
            if self.blocks_open {
                std::future::pending::<()>().await;
            }
            Ok(())
        }

        fn tool_dispatcher(&self) -> Option<Rc<dyn tools::ToolDispatcher>> {
            Some(self.dispatcher.clone())
        }

        fn stage_bound(&self) -> Option<std::num::NonZeroU32> {
            Some(std::num::NonZeroU32::MIN)
        }

        async fn next_stage(
            &mut self,
            _context: &TraceDriverContext<'_>,
        ) -> Result<Option<TraceStageDirective>, TraceDriverError> {
            if self.blocks_stage {
                std::future::pending::<()>().await;
            }
            match self.state {
                0 => {
                    self.state = 1;
                    let mut plan = tool_only_plan(&self.trace.trace_id, &["inspect"]);
                    if self.feedback_stage {
                        plan.graph.edges[0].source = "tool".into();
                    }
                    Ok(Some(TraceStageDirective::Execute(plan)))
                }
                2 => {
                    self.state = 3;
                    Ok(Some(TraceStageDirective::Complete(
                        TraceTerminalSupplement::new(
                            self.trace.run_id.clone(),
                            self.trace.trajectory_id.clone(),
                            self.trace.trace_id.clone(),
                            7,
                            "staged-test",
                        ),
                    )))
                }
                _ => Err(TraceDriverError::new(
                    "staged test driver advanced out of order",
                )),
            }
        }

        async fn observe_stage(
            &mut self,
            result: TraceStageResult,
        ) -> Result<(), TraceDriverError> {
            if self.state != 1 || result.terminal_status != GraphReplyStatus::Completed {
                return Err(TraceDriverError::new(
                    "staged test driver observed an invalid result",
                ));
            }
            self.observations.lock().unwrap().push(result.plan_identity);
            self.state = 2;
            Ok(())
        }

        async fn run(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            unreachable!("staged driver must execute through the stage loop")
        }
    }

    struct PendingOpenDriverFactory {
        dropped: Arc<AtomicUsize>,
    }

    impl TraceProgramDriverFactory for PendingOpenDriverFactory {
        fn capabilities(
            &self,
            _spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            Ok(TraceDriverCapabilities::default())
        }

        fn create(
            &self,
            _worker: WorkerIdentity,
            _trace: &TraceIdentity,
            _spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            Ok(Box::new(PendingOpenDriver {
                dropped: self.dropped.clone(),
            }))
        }
    }

    struct PendingOpenDriver {
        dropped: Arc<AtomicUsize>,
    }

    impl Drop for PendingOpenDriver {
        fn drop(&mut self) {
            self.dropped.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[async_trait(?Send)]
    impl TraceProgramDriver for PendingOpenDriver {
        async fn open(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<(), TraceDriverError> {
            std::future::pending::<()>().await;
            Ok(())
        }

        async fn run(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            unreachable!("permanently pending driver never reaches graph execution")
        }
    }

    struct DelayedDispatcherOpenTraceDriverFactory {
        events: Arc<Mutex<Vec<String>>>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    impl TraceProgramDriverFactory for DelayedDispatcherOpenTraceDriverFactory {
        fn capabilities(
            &self,
            _spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            Ok(TraceDriverCapabilities::default())
        }

        fn create(
            &self,
            _worker: WorkerIdentity,
            trace: &TraceIdentity,
            _spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            Ok(Box::new(DelayedDispatcherOpenTraceDriver {
                trace: trace.clone(),
                dispatcher: Rc::new(DelayedOpenToolDispatcher {
                    events: self.events.clone(),
                    started: self.started.clone(),
                    release: self.release.clone(),
                }),
            }))
        }
    }

    struct DelayedDispatcherOpenTraceDriver {
        trace: TraceIdentity,
        dispatcher: Rc<dyn tools::ToolDispatcher>,
    }

    #[async_trait(?Send)]
    impl TraceProgramDriver for DelayedDispatcherOpenTraceDriver {
        async fn open(
            &mut self,
            _program: &GraphTraceProgram,
            context: &TraceDriverContext<'_>,
        ) -> Result<(), TraceDriverError> {
            let execution = context.execution.as_ref().ok_or_else(|| {
                TraceDriverError::new("delayed dispatcher test requires worker execution context")
            })?;
            self.dispatcher
                .open_trace(tools::TraceOpenContext {
                    trace: &self.trace,
                    environment: None,
                    workspace: None,
                    clock: execution.clock,
                    segments: execution.segments,
                    invocation: execution.invocation,
                })
                .await
                .map_err(|error| TraceDriverError::new(error.to_string()))
        }

        async fn close(&mut self) -> Result<(), TraceDriverError> {
            self.dispatcher
                .close_trace(&self.trace)
                .await
                .map_err(|error| TraceDriverError::new(error.to_string()))
        }

        async fn abort_open(&mut self) -> Result<(), TraceDriverError> {
            self.close().await
        }

        async fn run(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            unreachable!("aborted delayed dispatcher test never runs the graph")
        }
    }

    struct DelayedOpenToolDispatcher {
        events: Arc<Mutex<Vec<String>>>,
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait(?Send)]
    impl tools::ToolDispatcher for DelayedOpenToolDispatcher {
        async fn open_trace(
            &self,
            _context: tools::TraceOpenContext<'_>,
        ) -> Result<(), tools::ToolDispatchError> {
            self.events
                .lock()
                .expect("dispatcher events are available")
                .push("dispatcher:open".into());
            self.started.notify_one();
            self.release.notified().await;
            self.events
                .lock()
                .expect("dispatcher events are available")
                .push("dispatcher:opened".into());
            Ok(())
        }

        async fn dispatch(
            &self,
            _request: tools::ToolDispatchRequest,
            _context: &tools::ToolDispatchContext,
        ) -> Result<tools::ToolDispatchResult, tools::ToolDispatchError> {
            Err(tools::ToolDispatchError::new(
                "delayed dispatcher test does not execute tools",
            ))
        }

        async fn close_trace(
            &self,
            _trace: &TraceIdentity,
        ) -> Result<(), tools::ToolDispatchError> {
            self.events
                .lock()
                .expect("dispatcher events are available")
                .push("dispatcher:close".into());
            Ok(())
        }
    }

    struct OrderedTraceDriverFactory(Arc<Mutex<Vec<&'static str>>>);

    impl TraceProgramDriverFactory for OrderedTraceDriverFactory {
        fn capabilities(
            &self,
            _spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            Ok(TraceDriverCapabilities::default())
        }

        fn create(
            &self,
            _worker: WorkerIdentity,
            _trace: &TraceIdentity,
            _spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            Ok(Box::new(OrderedTraceDriver(self.0.clone())))
        }
    }

    struct OrderedTraceDriver(Arc<Mutex<Vec<&'static str>>>);

    #[async_trait(?Send)]
    impl TraceProgramDriver for OrderedTraceDriver {
        async fn open(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<(), TraceDriverError> {
            self.0.lock().unwrap().push("open");
            Ok(())
        }

        async fn close(&mut self) -> Result<(), TraceDriverError> {
            self.0.lock().unwrap().push("close");
            Ok(())
        }

        async fn run(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            unreachable!("placement must use the owned-session lifecycle, not the prepass")
        }
    }

    struct OrderedToolDispatcherFactory(Arc<Mutex<Vec<String>>>);

    impl tools::ToolDispatcherFactory for OrderedToolDispatcherFactory {
        fn create(
            &self,
            _trace_id: &str,
        ) -> Result<Rc<dyn tools::ToolDispatcher>, tools::ToolDispatchError> {
            Ok(Rc::new(OrderedToolDispatcher(self.0.clone())))
        }
    }

    struct OrderedToolDispatcher(Arc<Mutex<Vec<String>>>);

    #[async_trait(?Send)]
    impl tools::ToolDispatcher for OrderedToolDispatcher {
        async fn open_trace(
            &self,
            _context: tools::TraceOpenContext<'_>,
        ) -> Result<(), tools::ToolDispatchError> {
            self.0.lock().unwrap().push("open".into());
            Ok(())
        }

        async fn dispatch(
            &self,
            request: tools::ToolDispatchRequest,
            _context: &tools::ToolDispatchContext,
        ) -> Result<tools::ToolDispatchResult, tools::ToolDispatchError> {
            self.0.lock().unwrap().push(request.command);
            Ok(tools::ToolDispatchResult::completed(
                request.call_id,
                0,
                Bytes::new(),
            ))
        }

        async fn close_trace(
            &self,
            _trace: &TraceIdentity,
        ) -> Result<(), tools::ToolDispatchError> {
            self.0.lock().unwrap().push("close".into());
            Ok(())
        }
    }

    struct FailingCloseTraceDriverFactory(Arc<AtomicUsize>);

    impl TraceProgramDriverFactory for FailingCloseTraceDriverFactory {
        fn capabilities(
            &self,
            _spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            Ok(TraceDriverCapabilities::default())
        }

        fn create(
            &self,
            _worker: WorkerIdentity,
            _trace: &TraceIdentity,
            _spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            Ok(Box::new(FailingCloseTraceDriver(self.0.clone())))
        }
    }

    struct FailingCloseTraceDriver(Arc<AtomicUsize>);

    #[async_trait(?Send)]
    impl TraceProgramDriver for FailingCloseTraceDriver {
        async fn close(&mut self) -> Result<(), TraceDriverError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Err(TraceDriverError::new("cleanup failure"))
        }

        async fn run(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            unreachable!("placement must not invoke the retired driver prepass")
        }
    }

    #[async_trait(?Send)]
    impl TraceProgramDriver for RecordingTraceDriver {
        async fn open(
            &mut self,
            _program: &GraphTraceProgram,
            _context: &TraceDriverContext<'_>,
        ) -> Result<(), TraceDriverError> {
            self.ran.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn run(
            &mut self,
            _program: &GraphTraceProgram,
            context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            self.ran.fetch_add(1, Ordering::SeqCst);
            Ok(TraceTerminalSupplement::new(
                context.trace.run_id.clone(),
                context.trace.trajectory_id.clone(),
                context.trace.trace_id.clone(),
                7,
                "recording",
            ))
        }
    }

    /// Test-only observer that delegates to the frozen native driver registry.
    struct ObservedStockTraceDriverFactory {
        inner: Arc<dyn TraceProgramDriverFactory>,
        created: Arc<AtomicUsize>,
        ran: Arc<AtomicUsize>,
    }

    impl TraceProgramDriverFactory for ObservedStockTraceDriverFactory {
        fn capabilities(
            &self,
            spec: &TraceDriverSpec,
        ) -> Result<TraceDriverCapabilities, TraceDriverError> {
            self.inner.capabilities(spec)
        }

        fn create(
            &self,
            worker: WorkerIdentity,
            trace: &TraceIdentity,
            spec: &TraceDriverSpec,
        ) -> Result<Box<dyn TraceProgramDriver>, TraceDriverError> {
            self.created.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(ObservedStockTraceDriver {
                inner: self.inner.create(worker, trace, spec)?,
                ran: self.ran.clone(),
            }))
        }
    }

    struct ObservedStockTraceDriver {
        inner: Box<dyn TraceProgramDriver>,
        ran: Arc<AtomicUsize>,
    }

    #[async_trait(?Send)]
    impl TraceProgramDriver for ObservedStockTraceDriver {
        async fn open(
            &mut self,
            program: &GraphTraceProgram,
            context: &TraceDriverContext<'_>,
        ) -> Result<(), TraceDriverError> {
            self.ran.fetch_add(1, Ordering::SeqCst);
            self.inner.open(program, context).await
        }

        fn tool_dispatcher(&self) -> Option<Rc<dyn tools::ToolDispatcher>> {
            self.inner.tool_dispatcher()
        }

        fn stage_bound(&self) -> Option<std::num::NonZeroU32> {
            self.inner.stage_bound()
        }

        async fn next_stage(
            &mut self,
            context: &TraceDriverContext<'_>,
        ) -> Result<Option<TraceStageDirective>, TraceDriverError> {
            self.inner.next_stage(context).await
        }

        async fn observe_stage(
            &mut self,
            result: TraceStageResult,
        ) -> Result<(), TraceDriverError> {
            self.inner.observe_stage(result).await
        }

        async fn close(&mut self) -> Result<(), TraceDriverError> {
            self.inner.close().await
        }

        async fn run(
            &mut self,
            program: &GraphTraceProgram,
            context: &TraceDriverContext<'_>,
        ) -> Result<TraceTerminalSupplement, TraceDriverError> {
            self.ran.fetch_add(1, Ordering::SeqCst);
            self.inner.run(program, context).await
        }
    }

    struct UnusedGraphEndpointRuntime;

    impl GraphEndpointRuntime for UnusedGraphEndpointRuntime {
        fn materialize(&self, _input: GraphEndpointRequest) -> Result<GraphEndpointDispatch> {
            Err(anyhow!(
                "empty graph test must not materialize an endpoint request"
            ))
        }

        fn transport_label(&self) -> &'static str {
            "test"
        }
    }

    #[derive(Default)]
    struct PlacementLifecycleCounts {
        opened: AtomicUsize,
        close_started: AtomicUsize,
        closed: AtomicUsize,
    }

    struct CleanupRecordingContainerRuntime {
        removed: Arc<AtomicUsize>,
    }

    struct SuspendingCleanupContainerRuntime {
        remove_started: Arc<AtomicUsize>,
        fallback_removes: Arc<AtomicUsize>,
    }

    #[async_trait(?Send)]
    impl tools::ContainerRuntime for CleanupRecordingContainerRuntime {
        async fn inspect_image(&self, _image: &str) -> Result<(), tools::ToolSandboxError> {
            Ok(())
        }

        async fn create(
            &self,
            _spec: &tools::ContainerCreateSpec,
        ) -> Result<tools::ContainerId, tools::ToolSandboxError> {
            Ok(tools::ContainerId::new("cancellation-cleanup"))
        }

        async fn start(&self, _id: &tools::ContainerId) -> Result<(), tools::ToolSandboxError> {
            Ok(())
        }

        async fn open_exec(
            &self,
            _id: &tools::ContainerId,
            _argv: &[std::ffi::OsString],
        ) -> Result<Box<dyn tools::FramedCommandIo>, tools::ToolSandboxError> {
            Err(tools::ToolSandboxError::new(
                "cleanup test must not execute a Docker command",
            ))
        }

        async fn force_remove(
            &self,
            _id: &tools::ContainerId,
            _timeout_ns: u64,
        ) -> Result<(), tools::ToolSandboxError> {
            self.removed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn list_by_label(
            &self,
            _key: &str,
            _value: &str,
        ) -> Result<Vec<tools::ContainerId>, tools::ToolSandboxError> {
            Ok(Vec::new())
        }
    }

    #[async_trait(?Send)]
    impl tools::ContainerRuntime for SuspendingCleanupContainerRuntime {
        async fn inspect_image(&self, _image: &str) -> Result<(), tools::ToolSandboxError> {
            Ok(())
        }

        async fn create(
            &self,
            _spec: &tools::ContainerCreateSpec,
        ) -> Result<tools::ContainerId, tools::ToolSandboxError> {
            Ok(tools::ContainerId::new("suspended-cleanup"))
        }

        async fn start(&self, _id: &tools::ContainerId) -> Result<(), tools::ToolSandboxError> {
            Ok(())
        }

        async fn open_exec(
            &self,
            _id: &tools::ContainerId,
            _argv: &[std::ffi::OsString],
        ) -> Result<Box<dyn tools::FramedCommandIo>, tools::ToolSandboxError> {
            Err(tools::ToolSandboxError::new(
                "cleanup test must not execute a Docker command",
            ))
        }

        async fn force_remove(
            &self,
            _id: &tools::ContainerId,
            _timeout_ns: u64,
        ) -> Result<(), tools::ToolSandboxError> {
            self.remove_started.fetch_add(1, Ordering::SeqCst);
            std::future::pending().await
        }

        fn force_remove_on_drop(&self, _id: &tools::ContainerId) {
            self.fallback_removes.fetch_add(1, Ordering::SeqCst);
        }

        async fn list_by_label(
            &self,
            _key: &str,
            _value: &str,
        ) -> Result<Vec<tools::ContainerId>, tools::ToolSandboxError> {
            Ok(Vec::new())
        }
    }

    struct SuspendingOpenLifecycleFactoryFactory(Arc<PlacementLifecycleCounts>);

    impl agent::AgentInvocationLeaseFactoryFactory for SuspendingOpenLifecycleFactoryFactory {
        fn create(
            &self,
            _trace_id: &str,
            _root_dispatcher: Rc<dyn tools::ToolDispatcher>,
        ) -> Result<Box<dyn agent::AgentInvocationLeaseFactory>, agent::AgentLoopError> {
            Ok(Box::new(SuspendingOpenLifecycleFactory(self.0.clone())))
        }
    }

    struct SuspendingOpenLifecycleFactory(Arc<PlacementLifecycleCounts>);

    impl agent::AgentInvocationLeaseFactory for SuspendingOpenLifecycleFactory {
        fn begin_open(
            &self,
            _request: &agent::AgentInvocationRequest,
            _parent: Option<&dyn agent::AgentInvocationLease>,
        ) -> Result<Box<dyn agent::AgentInvocationLeaseOpening>, agent::AgentLoopError> {
            self.0.opened.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SuspendingOpenLifecycleOpening(self.0.clone())))
        }
    }

    struct SuspendingOpenLifecycleOpening(Arc<PlacementLifecycleCounts>);

    #[async_trait(?Send)]
    impl agent::AgentInvocationLeaseOpening for SuspendingOpenLifecycleOpening {
        async fn open(
            &mut self,
        ) -> Result<Box<dyn agent::AgentInvocationLease>, agent::AgentLoopError> {
            std::future::pending().await
        }

        fn cancel_on_drop(&mut self) {
            self.0.closed.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct SuspendingCloseLifecycleFactoryFactory {
        counts: Arc<PlacementLifecycleCounts>,
        release: Arc<tokio::sync::Notify>,
    }

    impl agent::AgentInvocationLeaseFactoryFactory for SuspendingCloseLifecycleFactoryFactory {
        fn create(
            &self,
            _trace_id: &str,
            _root_dispatcher: Rc<dyn tools::ToolDispatcher>,
        ) -> Result<Box<dyn agent::AgentInvocationLeaseFactory>, agent::AgentLoopError> {
            Ok(Box::new(SuspendingCloseLifecycleFactory {
                counts: self.counts.clone(),
                release: self.release.clone(),
            }))
        }
    }

    struct SuspendingCloseLifecycleFactory {
        counts: Arc<PlacementLifecycleCounts>,
        release: Arc<tokio::sync::Notify>,
    }

    impl agent::AgentInvocationLeaseFactory for SuspendingCloseLifecycleFactory {
        fn begin_open(
            &self,
            _request: &agent::AgentInvocationRequest,
            _parent: Option<&dyn agent::AgentInvocationLease>,
        ) -> Result<Box<dyn agent::AgentInvocationLeaseOpening>, agent::AgentLoopError> {
            self.counts.opened.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(SuspendingCloseLifecycleOpening {
                counts: self.counts.clone(),
                release: self.release.clone(),
            }))
        }
    }

    struct SuspendingCloseLifecycleOpening {
        counts: Arc<PlacementLifecycleCounts>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait(?Send)]
    impl agent::AgentInvocationLeaseOpening for SuspendingCloseLifecycleOpening {
        async fn open(
            &mut self,
        ) -> Result<Box<dyn agent::AgentInvocationLease>, agent::AgentLoopError> {
            Ok(Box::new(SuspendingCloseLifecycleLease {
                counts: self.counts.clone(),
                release: self.release.clone(),
            }))
        }

        fn cancel_on_drop(&mut self) {
            self.counts.closed.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct SuspendingCloseLifecycleLease {
        counts: Arc<PlacementLifecycleCounts>,
        release: Arc<tokio::sync::Notify>,
    }

    #[async_trait(?Send)]
    impl agent::AgentInvocationLease for SuspendingCloseLifecycleLease {
        fn dispatcher(&self) -> Rc<dyn tools::ToolDispatcher> {
            Rc::new(tools::InMemoryToolDispatcher::default())
        }

        fn close_on_drop(&mut self) {
            self.counts.closed.fetch_add(1, Ordering::SeqCst);
        }

        async fn close(&mut self) -> Result<(), agent::AgentLoopError> {
            self.counts.close_started.fetch_add(1, Ordering::SeqCst);
            self.release.notified().await;
            self.counts.closed.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    fn empty_recorded_program(trace_id: &str) -> GraphTraceProgram {
        let mut program = GraphTraceProgram::static_graph(GraphTracePlan {
            graph: crate::graph::model::GraphRecord::default(),
            trace: crate::graph::model::TraceRecord {
                id: trace_id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        });
        program.driver = TraceDriverSpec::recorded_replay();
        program
    }

    fn tool_only_plan(trace_id: &str, commands: &[&str]) -> GraphTracePlan {
        let mut graph = crate::graph::model::GraphRecord::default();
        graph.state.insert(
            "tool-output".into(),
            crate::graph::model::ChannelSpec {
                channel_type: crate::graph::model::ChannelType::Messages,
                reducer: crate::graph::model::ReducerName::AddMessages,
            },
        );
        graph.nodes.insert(
            "tool".into(),
            ExecutableGraphNode::Tool(crate::graph::model::ToolNode {
                output: "tool-output".into(),
                commands: commands.iter().map(|command| (*command).into()).collect(),
                timeout_ns: None,
            }),
        );
        graph.edges.push(crate::graph::model::StaticEdge {
            source: crate::graph::model::START_NODE_ID.into(),
            target: "tool".into(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        });
        graph.edges.push(crate::graph::model::StaticEdge {
            source: "tool".into(),
            target: crate::graph::model::END_NODE_ID.into(),
            delay_after_predecessor_us: None,
            min_start_delay_us: None,
            delay_after_predecessor_start_us: None,
            delay_after_predecessor_first_token_us: None,
        });
        GraphTracePlan {
            graph,
            trace: crate::graph::model::TraceRecord {
                id: trace_id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        }
    }

    fn native_graph_task_fixture(program: &[u8]) -> tempfile::TempDir {
        let task = tempfile::tempdir().expect("temporary task root");
        fs::create_dir_all(task.path().join("environment")).expect("task environment directory");
        fs::create_dir_all(task.path().join("tests")).expect("task tests directory");
        fs::create_dir_all(task.path().join("tools")).expect("adapter directory");
        fs::write(
            task.path().join("environment/Dockerfile"),
            b"FROM scratch\n",
        )
        .expect("task Dockerfile");
        fs::write(task.path().join("instruction.md"), b"Do work.\n").expect("task instruction");
        fs::write(task.path().join("tests/test.sh"), b"exit 0\n").expect("task verifier");
        fs::write(
            task.path().join("task.toml"),
            r#"schema_version = "1.1"

[task]
name = "example/native-graph-worker"

[native_graph]
profile = "native_graph"
program = "agent_graph.json"
model_bindings = "models.toml"
adapter_manifest = "adapters.toml"
"#,
        )
        .expect("task manifest");
        fs::write(task.path().join("agent_graph.json"), program).expect("graph source");
        fs::write(
            task.path().join("models.toml"),
            r#"[[model_bindings]]
id = "primary"
endpoint_profile_id = "provider-default"
endpoint_factory_id = "chat"
transport_factory_id = "http"
model = "example-model"
urls = ["https://provider.example/v1"]
streaming = true
request_timeout_ms = 30000
capture = "metadata"

[model_bindings.tokenizer]
type = "local"
name = "builtin"
revision = "main"
apply_chat_template = true

[model_bindings.generation]
"#,
        )
        .expect("model bindings");
        fs::write(
            task.path().join("adapters.toml"),
            r#"[[adapters]]
id = "tool-adapter"
role = "tool"
argv = ["tools/adapter.sh"]
executable = "tools/adapter.sh"
"#,
        )
        .expect("adapter manifest");
        fs::write(task.path().join("tools/adapter.sh"), b"#!/bin/sh\nexit 0\n")
            .expect("adapter executable");
        task
    }

    fn empty_graph_worker(trace_driver: Arc<dyn TraceProgramDriverFactory>) -> GraphWorkerBackend {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        empty_graph_worker_with_clock(trace_driver, clock)
    }

    fn empty_graph_worker_with_clock(
        trace_driver: Arc<dyn TraceProgramDriverFactory>,
        clock: Rc<dyn Clock>,
    ) -> GraphWorkerBackend {
        let segments: Arc<dyn SegmentStore> = Arc::new(InMemorySegmentStore::default());
        GraphWorkerBackend {
            clock: clock.clone(),
            endpoint_runtime: Rc::new(UnusedGraphEndpointRuntime),
            materializer: Rc::new(SegmentItemsMaterializer::new(segments.clone())),
            segments,
            replay_run_identity: Some(crate::graph::replay::ReplayRunIdentity::mint(
                RngRoot::new(Some(1)),
                "test-run",
            )),
            observer: Rc::new(NativeMetricsObserver::new(
                clock,
                123,
                MetricsConfig::default(),
            )),
            phase: Phase::Profiling,
            worker_id: 7,
            model: "test-model".into(),
            default_max_tokens: 1,
            run_origin_ns: 123,
            raw_enabled: false,
            events: Arc::new(DiscardGraphExecutionEvents),
            node_policy: None,
            on_failure: OnFailure::Abort,
            cache_bust: None,
            ignore_trace_delays: false,
            system_idle_gap_cap_seconds: None,
            trace_driver,
            prefill_slots: None,
            next_session: Cell::new(0),
            next_execution: Cell::new(0),
            cancelled: Cell::new(false),
            active: RefCell::new(HashMap::new()),
        }
    }

    struct DiscardGraphExecutionEvents;

    impl GraphExecutionEventSink for DiscardGraphExecutionEvents {
        fn emit(&self, _event: GraphExecutionEvent) -> Result<(), TraceError> {
            Ok(())
        }
    }

    #[derive(Default)]
    struct CapturingTerminalSupplements(Mutex<Vec<TraceTerminalSupplement>>);

    impl GraphExecutionEventSink for CapturingTerminalSupplements {
        fn emit(&self, event: GraphExecutionEvent) -> Result<(), TraceError> {
            if let GraphExecutionEvent::TraceSupplement { supplement } = event {
                self.0.lock().unwrap().push(supplement);
            }
            Ok(())
        }
    }

    fn completed_evidence_record() -> CapturedRecord {
        let mut ingest =
            crate::metrics_core::RecordIngest::minimal(1_000_000, 9_000_000, Phase::Profiling);
        ingest.session_num = 17;
        ingest.first_token_ns = Some(3_000_000);
        ingest.tokens.input = Some(11);
        ingest.tokens.output = Some(5);
        CapturedRecord {
            uuid: Uuid::from_u128(17),
            x_correlation_id: "trace-a".into(),
            output: CapturedModelOutput::from_parts("answer", Some("answer"), None),
            raw: None,
            ingest,
        }
    }

    #[test]
    fn native_graph_evidence_sink_preserves_completed_record() {
        let (sink, mut receiver) = NativeGraphEvidenceSink::bounded();
        let expected_uuid = Uuid::from_u128(17);

        sink.emit(GraphExecutionEvent::Record {
            trace_id: "trace-a".into(),
            record: Box::new(completed_evidence_record()),
            node_id: Some("node-a".into()),
        })
        .expect("completed record enters the evidence lane");

        match receiver
            .try_recv()
            .expect("evidence receiver gets one event")
        {
            NativeGraphEvidenceEvent::Record(actual) => {
                assert_eq!(actual.uuid, expected_uuid);
                assert_eq!(actual.x_correlation_id, "trace-a");
                assert_eq!(actual.ingest.session_num, 17);
                assert_eq!(actual.ingest.first_token_ns, Some(3_000_000));
                assert_eq!(actual.ingest.tokens.input, Some(11));
                assert_eq!(actual.ingest.tokens.output, Some(5));
            }
            NativeGraphEvidenceEvent::TraceComplete(_) => {
                panic!("expected record evidence, received trace completion")
            }
        }
    }

    #[test]
    fn native_graph_record_export_disabled_preserves_completed_evidence() {
        let mut evidence = NativeGraphTransportEvidence {
            model_records: 0,
            completed_traces: 0,
            live_policy_calls: 0,
        };

        observe_native_graph_evidence(
            &mut evidence,
            NativeGraphEvidenceEvent::Record(Box::new(completed_evidence_record())),
            None,
        )
        .expect("disabled record export leaves evidence collection unchanged");

        assert_eq!(evidence.model_records, 1);
        assert_eq!(evidence.completed_traces, 0);
    }

    #[test]
    fn live_policy_summary_counts_one_first_token_without_retaining_response_data() {
        let clock = Rc::new(SimClock::new());
        let summary = Rc::new(RefCell::new(NativeGraphLivePolicyCallSummary::new()));
        let observer = NativeGraphPolicyDecisionObserver::new(clock.clone(), Rc::clone(&summary));
        let request = Uuid::from_u128(41);

        observer.on_arrival(request, 0.0, 9, 7);
        observer.on_admit(request, 0.0, 0);
        clock.advance_to(17);
        observer.on_token(request, 17.0);
        clock.advance_to(23);
        observer.on_token(request, 23.0);
        observer.on_terminal(request, ReplayTerminalStatus::Completed);

        let observed = summary
            .borrow()
            .snapshot()
            .expect("finite policy timing facts freeze");
        assert_eq!(observed.call_count(), 1);
        assert_eq!(observed.first_token_count(), 1);
        assert_eq!(observed.total_first_token_ns(), 17);
        assert_eq!(observed.max_first_token_ns(), 17);
    }

    #[test]
    fn suite_owned_native_graph_record_artifact_appends_across_episodes() {
        let dir = tempfile::tempdir().expect("temporary record artifact directory");
        let path = dir.path().join("records.jsonl");
        let artifact =
            EvalNodeRecordArtifact::open(&path).expect("open suite-owned record artifact");

        for _ in 0..2 {
            let mut episode_evidence = NativeGraphTransportEvidence {
                model_records: 0,
                completed_traces: 0,
                live_policy_calls: 0,
            };
            observe_native_graph_evidence(
                &mut episode_evidence,
                NativeGraphEvidenceEvent::Record(Box::new(completed_evidence_record())),
                Some(&artifact),
            )
            .expect("append one completed record for the episode");
            assert_eq!(episode_evidence.model_records, 1);
        }
        artifact
            .finish()
            .expect("flush suite-owned record artifact");

        let rows = fs::read_to_string(path).expect("read completed record artifact");
        assert_eq!(rows.lines().count(), 2);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn native_graph_record_append_failure_does_not_complete_evidence() {
        let artifact = EvalNodeRecordArtifact::open(std::path::Path::new("/dev/full"))
            .expect("open deterministic full-device writer");
        let mut evidence = NativeGraphTransportEvidence {
            model_records: 0,
            completed_traces: 0,
            live_policy_calls: 0,
        };

        let error = observe_native_graph_evidence(
            &mut evidence,
            NativeGraphEvidenceEvent::Record(Box::new(completed_evidence_record())),
            Some(&artifact),
        )
        .expect_err("record append flush failure must fail the episode evidence drain");
        let message = error.to_string();

        assert!(message.contains("appending completed native graph node record to full"));
        assert!(!message.contains("/dev/full"));
        assert_eq!(evidence.model_records, 0);
        assert_eq!(evidence.completed_traces, 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn native_graph_evidence_sink_requires_drain_to_exceed_its_bounded_capacity() {
        let (sink, mut receiver) = NativeGraphEvidenceSink::bounded();
        for _ in 0..NATIVE_GRAPH_EVIDENCE_CHANNEL_CAPACITY {
            sink.emit_summary(NativeGraphEvidenceEvent::TraceComplete(Ok(())))
                .expect("bounded evidence capacity accepts its exact number of events");
        }
        assert!(
            sink.emit_summary(NativeGraphEvidenceEvent::TraceComplete(Ok(())))
                .is_err(),
            "a producer without a draining receiver must observe bounded backpressure"
        );

        let total = NATIVE_GRAPH_EVIDENCE_CHANNEL_CAPACITY * 4;
        let producer = async {
            for _ in 0..total {
                loop {
                    if sink
                        .emit_summary(NativeGraphEvidenceEvent::TraceComplete(Ok(())))
                        .is_ok()
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            }
        };
        let drain = async {
            for _ in 0..NATIVE_GRAPH_EVIDENCE_CHANNEL_CAPACITY + total {
                receiver
                    .recv()
                    .await
                    .expect("concurrent drain receives every bounded event");
            }
        };
        tokio::join!(producer, drain);
    }

    fn empty_engine_graph_sink() -> EngineGraphSink {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let segments: Arc<dyn SegmentStore> = Arc::new(InMemorySegmentStore::default());
        let (sender, _receiver) = mpsc::unbounded_channel();
        EngineGraphSink {
            clock: clock.clone(),
            endpoint_runtime: Rc::new(UnusedGraphEndpointRuntime),
            trace_id: "trace-record-ordinal".into(),
            nodes: RefCell::new(HashMap::new()),
            prepared_metadata: RefCell::new(HashMap::new()),
            segments,
            observer: Rc::new(NativeMetricsObserver::new(
                clock,
                123,
                MetricsConfig::default(),
            )),
            phase: Phase::Profiling,
            worker_id: 7,
            session_num: 0,
            model: "test-model".into(),
            default_max_tokens: 1,
            run_origin_ns: 123,
            raw_enabled: false,
            terminal_nodes: RefCell::new(HashSet::new()),
            events: Arc::new(ChannelRunnerGraphExecutionEventSink::new(sender)),
            cache_bust_marker: None,
            cache_bust_target: crate::engine::graph_input::CacheBustTarget::None,
            arrivals: Cell::new(0),
            terminal_records: Cell::new(0),
            profiling_records: Cell::new(0),
            tool_dispatcher: RefCell::new(None),
            replay_calls: Rc::new(RefCell::new(Vec::new())),
            replay_tools: Rc::new(RefCell::new(Vec::new())),
        }
    }

    #[test]
    fn staged_terminal_outputs_freeze_only_selected_static_channels() {
        let channels = BTreeMap::from([
            (
                "output".into(),
                ChanVal::Val(serde_json::json!({"answer": 7})),
            ),
            (
                "undeclared".into(),
                ChanVal::Val(serde_json::json!("ignored")),
            ),
        ]);
        let base = InMemorySegmentStore::default();

        let frozen = freeze_terminal_outputs(&channels, &["output".into()], &base)
            .expect("selected static output freezes");
        assert_eq!(frozen.handles.len(), 1);
        let handle = frozen.handles["output"];
        let store = frozen.store.expect("concrete output owns a frozen catalog");
        assert!(matches!(
            store.get(handle).expect("frozen handle resolves"),
            Payload::Raw { wire } if wire.as_ref() == br#"{"answer":7}"#
        ));
    }

    #[test]
    fn staged_terminal_outputs_extend_the_prior_dynamic_catalog() {
        let base = InMemorySegmentStore::default();
        let first = freeze_terminal_outputs(
            &BTreeMap::from([("output".into(), ChanVal::Val(serde_json::json!(1)))]),
            &["output".into()],
            &base,
        )
        .expect("first dynamic stage freezes its selected output");
        let first_handle = first.handles["output"];
        let first_store = first.store.expect("first stage has a catalog");

        let second = freeze_terminal_outputs(
            &BTreeMap::from([("output".into(), ChanVal::Val(serde_json::json!(2)))]),
            &["output".into()],
            first_store.as_ref(),
        )
        .expect("later dynamic stage extends the catalog");
        let second_handle = second.handles["output"];
        let second_store = second.store.expect("second stage has an extended catalog");
        assert!(matches!(
            second_store.get(first_handle).expect("prior handle remains resolvable"),
            Payload::Raw { wire } if wire.as_ref() == b"1"
        ));
        assert!(matches!(
            second_store.get(second_handle).expect("latest handle resolves"),
            Payload::Raw { wire } if wire.as_ref() == b"2"
        ));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_agent_non_static_trace_runs_registered_driver_before_node_dispatch() {
        let created = Arc::new(AtomicUsize::new(0));
        let ran = Arc::new(AtomicUsize::new(0));
        let factory = RecordingTraceDriverFactory {
            created: created.clone(),
            ran: ran.clone(),
        };
        let program = empty_recorded_program("trace-1");
        let backend = empty_graph_worker(Arc::new(factory));
        backend
            .execute_trace(program)
            .await
            .expect("registered driver runs before empty graph dispatch");

        assert_eq!(created.load(Ordering::SeqCst), 1);
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn staged_driver_executes_through_the_existing_graph_backend_before_completion() {
        let observations = Arc::new(Mutex::new(Vec::new()));
        let dispatches = Arc::new(Mutex::new(Vec::new()));
        let backend = empty_graph_worker(Arc::new(StagedTraceDriverFactory {
            observations: observations.clone(),
            dispatches: dispatches.clone(),
            blocks_open: false,
            blocks_stage: false,
            feedback_stage: false,
        }));
        let mut program = empty_recorded_program("staged-trace");
        program.driver.kind = "staged_test".into();

        tokio::task::LocalSet::new()
            .run_until(async {
                backend
                    .execute_trace(program)
                    .await
                    .expect("bounded staged driver completes through the existing graph executor");
            })
            .await;

        assert_eq!(
            observations.lock().unwrap().as_slice(),
            ["staged-trace::stage-0"]
        );
        assert_eq!(dispatches.lock().unwrap().as_slice(), ["inspect"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn staged_driver_cancellation_aborts_pending_stage_selection() {
        let backend = Rc::new(empty_graph_worker(Arc::new(StagedTraceDriverFactory {
            observations: Arc::new(Mutex::new(Vec::new())),
            dispatches: Arc::new(Mutex::new(Vec::new())),
            blocks_open: false,
            blocks_stage: true,
            feedback_stage: false,
        })));
        let task_backend = backend.clone();
        tokio::task::LocalSet::new()
            .run_until(async move {
                let task = tokio::task::spawn_local(async move {
                    let mut program = empty_recorded_program("staged-cancel");
                    program.driver.kind = "staged_test".into();
                    task_backend.execute_trace(program).await
                });
                tokio::task::yield_now().await;
                backend
                    .cancel_inflight()
                    .expect("cancels pending stage selection");
                assert!(matches!(task.await.unwrap(), Err(TraceError::Cancelled(_))));
                assert!(
                    backend.active.borrow().is_empty(),
                    "cancelled staged selection must release its lifecycle abort handle"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn staged_driver_failures_do_not_retain_lifecycle_handles() {
        let backend = empty_graph_worker(Arc::new(StagedTraceDriverFactory {
            observations: Arc::new(Mutex::new(Vec::new())),
            dispatches: Arc::new(Mutex::new(Vec::new())),
            blocks_open: false,
            blocks_stage: false,
            feedback_stage: true,
        }));

        tokio::task::LocalSet::new()
            .run_until(async {
                for trace_id in ["staged-failure-one", "staged-failure-two"] {
                    let mut program = empty_recorded_program(trace_id);
                    program.driver.kind = "staged_test".into();
                    assert!(backend.execute_trace(program).await.is_err());
                    assert!(
                        backend.active.borrow().is_empty(),
                        "failed staged trace {trace_id:?} must release its lifecycle abort handle"
                    );
                }
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn staged_terminal_outputs_propagate_static_and_dynamic_catalogs() {
        let observations = Arc::new(Mutex::new(Vec::new()));
        let mut backend = empty_graph_worker(Arc::new(TerminalOutputStagedTraceDriverFactory {
            observations: observations.clone(),
        }));
        let supplements = Arc::new(CapturingTerminalSupplements::default());
        backend.events = supplements.clone();
        let mut program = empty_recorded_program("terminal-output-staged");
        program.driver.kind = "terminal_output_staged".into();

        tokio::task::LocalSet::new()
            .run_until(async { backend.execute_trace(program).await })
            .await
            .expect("executor completes both terminal-output stages");
        let supplement = supplements
            .0
            .lock()
            .unwrap()
            .pop()
            .expect("execute_trace emits the staged terminal supplement");
        let observations = observations.lock().unwrap();
        assert_eq!(observations.len(), 2);
        let static_handle = observations[0].output_handles["tool-output"];
        let dynamic_handle = observations[1].output_handles["tool-output"];
        assert_ne!(static_handle, dynamic_handle);
        let static_wire = serde_json::to_vec(&observations[0].channels["tool-output"])
            .expect("static completed channel serializes");
        let dynamic_wire = serde_json::to_vec(&observations[1].channels["tool-output"])
            .expect("dynamic completed channel serializes");

        assert_eq!(
            supplement.terminal_outputs,
            BTreeMap::from([("tool-output".into(), dynamic_handle)])
        );
        assert!(matches!(
            supplement
                .terminal_output(static_handle)
                .expect("executor catalog retains the static stage handle"),
            Payload::Raw { wire } if wire.as_ref() == static_wire
        ));
        assert!(matches!(
            supplement
                .terminal_output(dynamic_handle)
                .expect("executor catalog resolves the selected dynamic handle"),
            Payload::Raw { wire } if wire.as_ref() == dynamic_wire
        ));
        assert!(backend.active.borrow().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn aborting_suspended_driver_open_releases_its_lifecycle_entry() {
        let backend = Rc::new(empty_graph_worker(Arc::new(StagedTraceDriverFactory {
            observations: Arc::new(Mutex::new(Vec::new())),
            dispatches: Arc::new(Mutex::new(Vec::new())),
            blocks_open: true,
            blocks_stage: false,
            feedback_stage: false,
        })));
        let task_backend = backend.clone();

        tokio::task::LocalSet::new()
            .run_until(async move {
                let task = tokio::task::spawn_local(async move {
                    let mut program = empty_recorded_program("suspended-open");
                    program.driver.kind = "staged_test".into();
                    task_backend.execute_trace(program).await
                });
                tokio::task::yield_now().await;
                task.abort();
                assert!(
                    task.await
                        .expect_err("suspended open task was aborted")
                        .is_cancelled()
                );
                assert!(
                    backend.active.borrow().is_empty(),
                    "aborting a suspended driver open must drop its lifecycle handle"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn cancelling_a_permanently_pending_driver_open_drops_its_lifecycle() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let backend = Rc::new(empty_graph_worker(Arc::new(PendingOpenDriverFactory {
            dropped: dropped.clone(),
        })));
        let task_backend = backend.clone();

        tokio::task::LocalSet::new()
            .run_until(async move {
                let task = tokio::task::spawn_local(async move {
                    let mut program = empty_recorded_program("permanently-pending-open");
                    program.driver.kind = "pending_open".into();
                    task_backend.execute_trace(program).await
                });
                tokio::task::yield_now().await;
                backend
                    .cancel_inflight()
                    .expect("cancels the registered pending driver open");
                assert!(matches!(task.await.unwrap(), Err(TraceError::Cancelled(_))));
                for _ in 0..8 {
                    if dropped.load(Ordering::SeqCst) == 1 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert_eq!(
                    dropped.load(Ordering::SeqCst),
                    1,
                    "cancellation cannot leave a permanently pending driver owned by a detached task"
                );
                assert!(
                    backend.active.borrow().is_empty(),
                    "a cancelled pending open has no retained lifecycle entry"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn aborting_pending_driver_open_rolls_back_delayed_dispatcher_cleanup() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let backend = Rc::new(empty_graph_worker(Arc::new(
            DelayedDispatcherOpenTraceDriverFactory {
                events: events.clone(),
                started: started.clone(),
                release: release.clone(),
            },
        )));
        let task_backend = backend.clone();

        tokio::task::LocalSet::new()
            .run_until(async move {
                let started_wait = started.notified();
                let task = tokio::task::spawn_local(async move {
                    let mut program = empty_recorded_program("delayed-dispatcher-open");
                    program.driver.kind = "delayed_dispatcher_open".into();
                    task_backend.execute_trace(program).await
                });
                started_wait.await;
                backend
                    .cancel_inflight()
                    .expect("cancels the registered delayed driver open");
                assert!(matches!(task.await.unwrap(), Err(TraceError::Cancelled(_))));
                assert!(
                    backend.active.borrow().is_empty(),
                    "the aborted trace releases its active lifecycle entry"
                );

                for _ in 0..8 {
                    if events
                        .lock()
                        .expect("dispatcher events are available")
                        .iter()
                        .any(|event| event == "dispatcher:close")
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert_eq!(
                    events
                        .lock()
                        .expect("dispatcher events are available")
                        .as_slice(),
                    ["dispatcher:open", "dispatcher:close"],
                    "cancellation rolls back the partially opened dispatcher exactly once"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn staged_driver_feedback_rewrite_is_rejected_before_tool_dispatch() {
        let dispatches = Arc::new(Mutex::new(Vec::new()));
        let backend = empty_graph_worker(Arc::new(StagedTraceDriverFactory {
            observations: Arc::new(Mutex::new(Vec::new())),
            dispatches: dispatches.clone(),
            blocks_open: false,
            blocks_stage: false,
            feedback_stage: true,
        }));
        let mut program = empty_recorded_program("staged-feedback");
        program.driver.kind = "staged_test".into();
        let error = tokio::task::LocalSet::new()
            .run_until(async { backend.execute_trace(program).await.unwrap_err() })
            .await;
        assert_eq!(error.kind(), "other");
        assert!(dispatches.lock().unwrap().is_empty());
    }

    #[test]
    fn warmup_terminal_does_not_shift_first_profiling_record_index() {
        let sink = empty_engine_graph_sink();
        assert_eq!(sink.record_ordinal(false), 0);
        sink.did_drain_terminal_record(false);

        // This ordinal becomes `RecordIngest::request_index`; retaining the
        // warmup ordinal here would make profile output begin at row one and
        // leave an empty metrics row at zero.
        assert_eq!(sink.record_ordinal(true), 0);
        sink.did_drain_terminal_record(true);
        assert_eq!(sink.profiling_records.get(), 1);
        assert_eq!(sink.terminal_records.get(), 2);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_worker_keeps_recorded_session_across_trace_warmup_and_profile() {
        // This catches the historical driver prepass, which closed its session
        // before the placement could execute either graph portion.
        let order = Arc::new(Mutex::new(Vec::new()));
        let backend = empty_graph_worker(Arc::new(OrderedTraceDriverFactory(order.clone())));
        let mut program = empty_recorded_program("trace-session-order");
        program.warmup = Some(program.profiling.clone());
        backend.execute_trace(program).await.unwrap();
        assert_eq!(order.lock().unwrap().as_slice(), ["open", "close"]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_worker_keeps_recorded_dispatcher_open_for_warmup_profile_and_tool_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let trace_driver = RecordedReplayTraceProgramDriverFactory::default()
            .with_agent_loop_factories(RecordedReplayAgentLoopFactories::new(
                Arc::new(agent::StaticAgentTurnCoordinatorFactory::default()),
                Arc::new(agent::InMemoryAgentResponseStoreFactory),
                Arc::new(agent::InMemoryAgentTrajectorySinkFactory),
                Arc::new(agent::InMemoryInvocationLeaseFactoryFactory),
                Arc::new(agent::InMemoryAgentInvocationLeaseFactoryFactory),
                Arc::new(OrderedToolDispatcherFactory(order.clone())),
                Arc::new(tools::InMemoryAgentToolCallDecoderFactory),
                Arc::new(tools::InMemoryAgentObservationFormatterFactory),
            ));
        let backend = empty_graph_worker(Arc::new(trace_driver));
        let mut program = empty_recorded_program("trace-tool-order");
        program.warmup = Some(tool_only_plan(
            "trace-tool-order-warmup",
            &["warmup-1", "warmup-2"],
        ));
        program.profiling = tool_only_plan("trace-tool-order", &["profile-1", "profile-2"]);

        tokio::task::LocalSet::new()
            .run_until(async { backend.execute_trace(program).await.unwrap() })
            .await;

        assert_eq!(
            order.lock().unwrap().as_slice(),
            [
                "open",
                "warmup-1",
                "warmup-2",
                "profile-1",
                "profile-2",
                "close",
            ],
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_worker_preserves_profile_error_when_recorded_cleanup_fails() {
        // This catches cleanup masking the useful profile failure; the tool node
        // has no dispatcher, so profiling fails before the driver's close error.
        let closes = Arc::new(AtomicUsize::new(0));
        let backend = empty_graph_worker(Arc::new(FailingCloseTraceDriverFactory(closes.clone())));
        let mut program = empty_recorded_program("trace-cleanup-precedence");
        program.profiling.graph.state.insert(
            "tool-output".into(),
            crate::graph::model::ChannelSpec {
                channel_type: crate::graph::model::ChannelType::Messages,
                reducer: crate::graph::model::ReducerName::AddMessages,
            },
        );
        program.profiling.graph.nodes.insert(
            "tool".into(),
            ExecutableGraphNode::Tool(crate::graph::model::ToolNode {
                output: "tool-output".into(),
                commands: vec!["false".into()],
                timeout_ns: None,
            }),
        );
        program
            .profiling
            .graph
            .edges
            .push(crate::graph::model::StaticEdge {
                source: crate::graph::model::START_NODE_ID.into(),
                target: "tool".into(),
                delay_after_predecessor_us: None,
                min_start_delay_us: None,
                delay_after_predecessor_start_us: None,
                delay_after_predecessor_first_token_us: None,
            });
        let error = tokio::task::LocalSet::new()
            .run_until(async { backend.execute_trace(program).await.unwrap_err() })
            .await;
        assert!(!error.to_string().contains("cleanup failure"));
        assert_eq!(closes.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_agent_stock_execution_factories_run_recorded_replay_driver_through_placement() {
        let trace_driver =
            crate::engine::execution_factories::native_execution_factories().trace_driver_handle();
        let trace = TraceIdentity {
            run_id: "run-stock".into(),
            trajectory_id: "trajectory-stock".into(),
            trace_id: "trace-stock".into(),
        };
        let program = empty_recorded_program(&trace.trace_id);
        let mut driver = trace_driver
            .create(WorkerIdentity { worker_id: 7 }, &trace, &program.driver)
            .expect("stock execution factories select recorded replay");
        let supplement = driver
            .run(&program, &TraceDriverContext::metadata_only(&trace))
            .await
            .expect("stock recorded replay driver runs its agent-loop composition");
        assert_eq!(supplement.driver_kind, "recorded_replay");
        assert_eq!(supplement.trace_id, trace.trace_id);

        let created = Arc::new(AtomicUsize::new(0));
        let ran = Arc::new(AtomicUsize::new(0));
        let backend = empty_graph_worker(Arc::new(ObservedStockTraceDriverFactory {
            inner: trace_driver,
            created: created.clone(),
            ran: ran.clone(),
        }));
        backend
            .execute_trace(program)
            .await
            .expect("graph worker placement runs the stock recorded replay driver");
        assert_eq!(created.load(Ordering::SeqCst), 1);
        assert_eq!(ran.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_agent_cancellation_aborts_registered_recorded_driver_and_fences_lease() {
        let lifecycle = Arc::new(PlacementLifecycleCounts::default());
        let trace_driver = RecordedReplayTraceProgramDriverFactory::default()
            .with_agent_loop_factories(RecordedReplayAgentLoopFactories::new(
                Arc::new(agent::StaticAgentTurnCoordinatorFactory::default()),
                Arc::new(agent::InMemoryAgentResponseStoreFactory),
                Arc::new(agent::InMemoryAgentTrajectorySinkFactory),
                Arc::new(agent::InMemoryInvocationLeaseFactoryFactory),
                Arc::new(SuspendingOpenLifecycleFactoryFactory(lifecycle.clone())),
                Arc::new(tools::InMemoryToolDispatcherFactory),
                Arc::new(tools::InMemoryAgentToolCallDecoderFactory),
                Arc::new(tools::InMemoryAgentObservationFormatterFactory),
            ));
        let backend = Rc::new(empty_graph_worker(Arc::new(trace_driver)));
        let task_backend = backend.clone();
        tokio::task::LocalSet::new()
            .run_until(async move {
                let task = tokio::task::spawn_local(async move {
                    task_backend
                        .execute_trace(empty_recorded_program("trace-cancel-placement"))
                        .await
                });
                tokio::task::yield_now().await;
                assert_eq!(lifecycle.opened.load(Ordering::SeqCst), 1);
                backend
                    .cancel_inflight()
                    .expect("cancellation aborts the registered lifecycle open");
                let error = task
                    .await
                    .expect("placement task completes after cancellation")
                    .expect_err("suspended recorded lifecycle open is cancelled");
                assert!(matches!(error, TraceError::Cancelled(_)));
                assert_eq!(lifecycle.opened.load(Ordering::SeqCst), 1);
                assert_eq!(lifecycle.closed.load(Ordering::SeqCst), 1);
                assert!(
                    backend.active.borrow().is_empty(),
                    "cancelling lifecycle open must unregister the active trace"
                );
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_agent_cancellation_awaits_async_close_before_returning_cancelled() {
        let lifecycle = Arc::new(PlacementLifecycleCounts::default());
        let release = Arc::new(tokio::sync::Notify::new());
        let removed = Arc::new(AtomicUsize::new(0));
        let runtime_removed = removed.clone();
        let trace_driver = RecordedReplayTraceProgramDriverFactory::default()
            .with_agent_loop_factories(RecordedReplayAgentLoopFactories::new(
                Arc::new(agent::StaticAgentTurnCoordinatorFactory::default()),
                Arc::new(agent::InMemoryAgentResponseStoreFactory),
                Arc::new(agent::InMemoryAgentTrajectorySinkFactory),
                Arc::new(agent::InMemoryInvocationLeaseFactoryFactory),
                Arc::new(SuspendingCloseLifecycleFactoryFactory {
                    counts: lifecycle.clone(),
                    release: release.clone(),
                }),
                Arc::new(tools::DockerToolDispatcherFactory::with_runtime_factory(
                    1024,
                    move |_| {
                        Rc::new(CleanupRecordingContainerRuntime {
                            removed: runtime_removed.clone(),
                        })
                    },
                )),
                Arc::new(tools::InMemoryAgentToolCallDecoderFactory),
                Arc::new(tools::InMemoryAgentObservationFormatterFactory),
            ));
        let backend = Rc::new(empty_graph_worker(Arc::new(trace_driver)));
        let task_backend = backend.clone();
        tokio::task::LocalSet::new()
            .run_until(async move {
                let mut program = empty_recorded_program("trace-close-cancel");
                let environment = tools::ResolvedTraceEnvironment {
                    kind: tools::EnvironmentRecipe::SweBench,
                    backend: tools::ToolExecutionBackend::Docker,
                    image: "recorded-agent-test:latest".into(),
                    workspace: tools::WorkspaceSpec::image_native(
                        "/testbed",
                        vec!["bash".into(), "-c".into()],
                        5_000_000_000,
                    ),
                };
                program.environment = Some(
                    crate::graph::driver::TraceEnvironmentSpec::from_resolved(&environment)
                        .unwrap(),
                );
                let task =
                    tokio::task::spawn_local(
                        async move { task_backend.execute_trace(program).await },
                    );
                tokio::task::yield_now().await;
                backend
                    .cancel_inflight()
                    .expect("cancellation marks the trace terminal");
                tokio::task::yield_now().await;
                assert_eq!(lifecycle.close_started.load(Ordering::SeqCst), 1);
                assert_eq!(
                    removed.load(Ordering::SeqCst),
                    1,
                    "stock Docker dispatcher removes its container before teardown waits"
                );
                assert!(
                    !task.is_finished(),
                    "post-open cancellation must await bounded driver teardown"
                );
                release.notify_one();
                let error = task
                    .await
                    .expect("placement task completes after teardown")
                    .expect_err("cancellation remains the terminal result");
                assert!(matches!(error, TraceError::Cancelled(_)));
                assert_eq!(lifecycle.opened.load(Ordering::SeqCst), 1);
                assert_eq!(lifecycle.close_started.load(Ordering::SeqCst), 1);
                assert_eq!(lifecycle.closed.load(Ordering::SeqCst), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_agent_cancellation_bounds_never_resolving_lifecycle_close_with_execution_clock()
    {
        let lifecycle = Arc::new(PlacementLifecycleCounts::default());
        let release = Arc::new(tokio::sync::Notify::new());
        let removed = Arc::new(AtomicUsize::new(0));
        let runtime_removed = removed.clone();
        let trace_driver = RecordedReplayTraceProgramDriverFactory::default()
            .with_agent_loop_factories(RecordedReplayAgentLoopFactories::new(
                Arc::new(agent::StaticAgentTurnCoordinatorFactory::default()),
                Arc::new(agent::InMemoryAgentResponseStoreFactory),
                Arc::new(agent::InMemoryAgentTrajectorySinkFactory),
                Arc::new(agent::InMemoryInvocationLeaseFactoryFactory),
                Arc::new(SuspendingCloseLifecycleFactoryFactory {
                    counts: lifecycle.clone(),
                    release,
                }),
                Arc::new(tools::DockerToolDispatcherFactory::with_runtime_factory(
                    1024,
                    move |_| {
                        Rc::new(CleanupRecordingContainerRuntime {
                            removed: runtime_removed.clone(),
                        })
                    },
                )),
                Arc::new(tools::InMemoryAgentToolCallDecoderFactory),
                Arc::new(tools::InMemoryAgentObservationFormatterFactory),
            ));
        let clock = Rc::new(SimClock::new());
        let backend = Rc::new(empty_graph_worker_with_clock(
            Arc::new(trace_driver),
            clock.clone(),
        ));
        let task_backend = backend.clone();

        tokio::task::LocalSet::new()
            .run_until(async move {
                let mut program = empty_recorded_program("trace-close-timeout");
                let environment = tools::ResolvedTraceEnvironment {
                    kind: tools::EnvironmentRecipe::SweBench,
                    backend: tools::ToolExecutionBackend::Docker,
                    image: "recorded-agent-test:latest".into(),
                    workspace: tools::WorkspaceSpec::image_native(
                        "/testbed",
                        vec!["bash".into(), "-c".into()],
                        5_000_000_000,
                    ),
                };
                program.environment = Some(
                    crate::graph::driver::TraceEnvironmentSpec::from_resolved(&environment)
                        .unwrap(),
                );
                let task =
                    tokio::task::spawn_local(
                        async move { task_backend.execute_trace(program).await },
                    );
                for _ in 0..8 {
                    if lifecycle.close_started.load(Ordering::SeqCst) == 1 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert_eq!(lifecycle.close_started.load(Ordering::SeqCst), 1);
                backend.cancel_inflight().unwrap();
                assert_eq!(removed.load(Ordering::SeqCst), 1);

                clock.advance_to(9_999_999_999);
                tokio::task::yield_now().await;
                assert!(!task.is_finished(), "driver close expired before its bound");
                clock.advance_to(10_000_000_000);
                for _ in 0..8 {
                    if task.is_finished() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert!(
                    task.is_finished(),
                    "Clock deadline must bound a never-resolving driver close"
                );
                let error = task
                    .await
                    .unwrap()
                    .expect_err("cancellation remains primary after close timeout");
                assert!(matches!(error, TraceError::Cancelled(_)));
                assert!(error.to_string().contains("timed out"), "{error}");
                assert_eq!(removed.load(Ordering::SeqCst), 1);
                assert_eq!(lifecycle.closed.load(Ordering::SeqCst), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_agent_cancellation_fences_never_resolving_docker_remove_before_lifecycle_close()
    {
        let lifecycle = Arc::new(PlacementLifecycleCounts::default());
        let release = Arc::new(tokio::sync::Notify::new());
        release.notify_one();
        let remove_started = Arc::new(AtomicUsize::new(0));
        let fallback_removes = Arc::new(AtomicUsize::new(0));
        let runtime_remove_started = remove_started.clone();
        let runtime_fallback_removes = fallback_removes.clone();
        let trace_driver = RecordedReplayTraceProgramDriverFactory::default()
            .with_agent_loop_factories(RecordedReplayAgentLoopFactories::new(
                Arc::new(agent::StaticAgentTurnCoordinatorFactory::default()),
                Arc::new(agent::InMemoryAgentResponseStoreFactory),
                Arc::new(agent::InMemoryAgentTrajectorySinkFactory),
                Arc::new(agent::InMemoryInvocationLeaseFactoryFactory),
                Arc::new(SuspendingCloseLifecycleFactoryFactory {
                    counts: lifecycle.clone(),
                    release,
                }),
                Arc::new(tools::DockerToolDispatcherFactory::with_runtime_factory(
                    1024,
                    move |_| {
                        Rc::new(SuspendingCleanupContainerRuntime {
                            remove_started: runtime_remove_started.clone(),
                            fallback_removes: runtime_fallback_removes.clone(),
                        })
                    },
                )),
                Arc::new(tools::InMemoryAgentToolCallDecoderFactory),
                Arc::new(tools::InMemoryAgentObservationFormatterFactory),
            ));
        let clock = Rc::new(SimClock::new());
        let backend = Rc::new(empty_graph_worker_with_clock(
            Arc::new(trace_driver),
            clock.clone(),
        ));
        let task_backend = backend.clone();

        tokio::task::LocalSet::new()
            .run_until(async move {
                let mut program = empty_recorded_program("trace-remove-timeout");
                let environment = tools::ResolvedTraceEnvironment {
                    kind: tools::EnvironmentRecipe::SweBench,
                    backend: tools::ToolExecutionBackend::Docker,
                    image: "recorded-agent-test:latest".into(),
                    workspace: tools::WorkspaceSpec::image_native(
                        "/testbed",
                        vec!["bash".into(), "-c".into()],
                        5_000_000_000,
                    ),
                };
                program.environment = Some(
                    crate::graph::driver::TraceEnvironmentSpec::from_resolved(&environment)
                        .unwrap(),
                );
                let task =
                    tokio::task::spawn_local(
                        async move { task_backend.execute_trace(program).await },
                    );
                for _ in 0..8 {
                    if remove_started.load(Ordering::SeqCst) == 1 {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert_eq!(remove_started.load(Ordering::SeqCst), 1);
                backend.cancel_inflight().unwrap();

                clock.advance_to(19_999_999_999);
                tokio::task::yield_now().await;
                assert!(
                    !task.is_finished(),
                    "Docker cleanup must retain its command and reap budget"
                );
                assert_eq!(fallback_removes.load(Ordering::SeqCst), 0);

                clock.advance_to(20_000_000_000);
                for _ in 0..8 {
                    if task.is_finished() {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
                assert!(task.is_finished(), "Docker cleanup must be Clock-bounded");
                let error = task
                    .await
                    .unwrap()
                    .expect_err("cancellation remains primary after Docker cleanup timeout");
                assert!(matches!(error, TraceError::Cancelled(_)));
                assert_eq!(remove_started.load(Ordering::SeqCst), 1);
                assert_eq!(fallback_removes.load(Ordering::SeqCst), 1);
                assert_eq!(lifecycle.close_started.load(Ordering::SeqCst), 1);
                assert_eq!(lifecycle.closed.load(Ordering::SeqCst), 1);
            })
            .await;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn graph_agent_pre_cancelled_trace_never_opens_or_waits_for_blocking_close() {
        // The close implementation deliberately never resolves. Before the
        // early cancellation guard this path synchronously opened the lease,
        // reached the post-open cancellation check, and waited here forever.
        let lifecycle = Arc::new(PlacementLifecycleCounts::default());
        let release = Arc::new(tokio::sync::Notify::new());
        let trace_driver = RecordedReplayTraceProgramDriverFactory::default()
            .with_agent_loop_factories(RecordedReplayAgentLoopFactories::new(
                Arc::new(agent::StaticAgentTurnCoordinatorFactory::default()),
                Arc::new(agent::InMemoryAgentResponseStoreFactory),
                Arc::new(agent::InMemoryAgentTrajectorySinkFactory),
                Arc::new(agent::InMemoryInvocationLeaseFactoryFactory),
                Arc::new(SuspendingCloseLifecycleFactoryFactory {
                    counts: lifecycle.clone(),
                    release,
                }),
                Arc::new(tools::InMemoryToolDispatcherFactory),
                Arc::new(tools::InMemoryAgentToolCallDecoderFactory),
                Arc::new(tools::InMemoryAgentObservationFormatterFactory),
            ));
        let backend = Rc::new(empty_graph_worker(Arc::new(trace_driver)));
        backend.cancel_inflight().unwrap();

        tokio::task::LocalSet::new()
            .run_until(async move {
                let mut execution =
                    Box::pin(backend.execute_trace(empty_recorded_program("trace-pre-cancelled")));
                let error = tokio::select! {
                    result = &mut execution => result.expect_err("pre-cancelled trace is rejected"),
                    _ = async {
                        for _ in 0..8 {
                            tokio::task::yield_now().await;
                        }
                    } => panic!("pre-cancelled trace waited for lifecycle close"),
                };
                assert!(matches!(error, TraceError::Cancelled(_)));
                assert_eq!(lifecycle.opened.load(Ordering::SeqCst), 0);
                assert_eq!(lifecycle.closed.load(Ordering::SeqCst), 0);
                assert!(backend.active.borrow().is_empty());
            })
            .await;
    }

    #[test]
    fn a_text_only_prompt_program_reports_a_known_zero_image_count() {
        let node = node_with_items(vec![
            PromptItem::Seg {
                seg: Handle::new(0),
            },
            PromptItem::Text {
                text: Handle::new(1),
                role: "user".into(),
            },
        ]);
        assert_eq!(
            node_known_image_count(&node),
            Some(0),
            "lowering rejects non-text media, so a Seg/Text program carries no images"
        );
    }

    #[test]
    fn an_authored_raw_message_array_falls_back_to_parsing_the_body() {
        // `raw_message_wires` validates only object-ness, so an authored
        // `image_url` part rides through unseen: the count stays unknown here.
        let node = node_with_items(vec![PromptItem::RawMessages {
            raw_messages: Handle::new(0),
        }]);
        assert_eq!(node_known_image_count(&node), None);
    }

    #[test]
    fn a_splice_falls_back_because_a_seeded_channel_can_carry_media() {
        // `channel_value` seeds a messages-typed channel from a trace's authored
        // `initial_state` verbatim, so `@messages` can splice in `image_url`
        // parts. A splice of a reply would be text-only, but the prompt program
        // alone cannot tell the two apart.
        let node = node_with_items(vec![PromptItem::Splice {
            splice: "messages".into(),
        }]);
        assert_eq!(node_known_image_count(&node), None);
    }

    #[derive(Debug)]
    struct PreparedOnlyChatFactory;

    impl EndpointFactory for PreparedOnlyChatFactory {
        fn descriptor(&self) -> &'static EndpointDescriptor {
            Endpoint::descriptor(&ChatEndpoint)
        }

        fn prepare(
            &self,
            config: EffectiveEndpointConfig,
        ) -> EndpointResult<Box<dyn PreparedEndpoint>> {
            EndpointFactory::prepare(&StatelessEndpointFactory::new(ChatEndpoint), config)
        }
    }

    #[test]
    fn prepared_graph_runtime_does_not_require_endpoint_adapter() {
        let mut registry = EndpointRegistryBuilder::new();
        registry.register_factory(PreparedOnlyChatFactory).unwrap();
        let registry = registry.freeze();
        assert!(
            registry
                .legacy_endpoint(&EndpointId::new("chat").unwrap())
                .is_err()
        );

        let factory = PreparedRunnerGraphEndpointRuntimeFactory::new(
            registry,
            Arc::new(vec![crate::engine::registry::ValidatedEndpointProfileV2 {
                profile_id: "default".into(),
                endpoint_id: EndpointId::new("chat").unwrap(),
                config: RawEndpointConfig {
                    urls: vec!["http://127.0.0.1:1".into()],
                    streaming: true,
                    ..RawEndpointConfig::default()
                },
                connection_reuse: ConnectionReuseStrategy::Pooled,
                client: Default::default(),
                session_header: None,
            }]),
            Arc::new(AuthoredInputTokenCounter),
            Arc::new(crate::engine::online_execution::HttpNativeExecution::new(
                Arc::new(crate::engine::turn_execution::HttpExecutionFactory),
            )),
            None,
            false,
        )
        .unwrap();
        let runtime = factory
            .prepare_worker(Rc::new(SimClock::new()), 0, "fixture-model")
            .unwrap();
        assert_eq!(runtime.transport_label(), "http");
        let dispatch = runtime
            .materialize(GraphEndpointRequest {
                selector: None,
                model: "fixture-model".into(),
                turn: Turn {
                    raw_messages: Some(vec![serde_json::json!({
                        "role": "user",
                        "content": "hello"
                    })]),
                    max_tokens: Some(1),
                    ..Turn::default()
                },
                streaming: true,
                authored_input_tokens: 4,
                max_output_tokens: 1,
                extra_headers: BTreeMap::new(),
                parameters: BTreeMap::new(),
                trace_id: "trace-1".into(),
                is_final_turn: true,
                cancel_after_ns: None,
                session_num: 0,
                phase: Phase::Profiling,
                known_image_count: Some(0),
            })
            .unwrap();
        assert_eq!(dispatch.input_tokens, 4);
        assert_eq!(dispatch.request.image_count, Some(0));
        let crate::transport::core::PreparedEndpointBinding::Prepared(reference) =
            dispatch.endpoint;
        assert_eq!(reference.endpoint_id.as_str(), "chat");
        assert_eq!(reference.key.index(), 1);
    }

    #[cfg(feature = "grpc")]
    #[test]
    fn prepared_graph_runtime_builds_a_grpc_dispatcher_when_selected() {
        // A gRPC-bindable endpoint (kserve v2 infer) prepares over the gRPC arm:
        // `prepare_worker` succeeds only because a real GrpcTransportSink with
        // dense gRPC bindings was assembled — the HTTP arm never touches the
        // gRPC binding registry. The `kind()` probe confirms it downcast-free.
        let factory = PreparedRunnerGraphEndpointRuntimeFactory::new(
            EndpointRegistry::builtin().unwrap(),
            Arc::new(vec![crate::engine::registry::ValidatedEndpointProfileV2 {
                profile_id: "default".into(),
                endpoint_id: EndpointId::new("kserve_v2_infer").unwrap(),
                config: RawEndpointConfig {
                    urls: vec!["grpc://127.0.0.1:1".into()],
                    ..RawEndpointConfig::default()
                },
                connection_reuse: ConnectionReuseStrategy::Pooled,
                client: Default::default(),
                session_header: None,
            }]),
            Arc::new(AuthoredInputTokenCounter),
            Arc::new(crate::engine::grpc_execution::GrpcNativeExecution::new()),
            None,
            false,
        )
        .unwrap();
        let runtime = factory
            .prepare_worker(Rc::new(SimClock::new()), 0, "fixture-model")
            .unwrap();
        assert_eq!(runtime.transport_label(), "grpc");
    }

    #[cfg(feature = "grpc")]
    #[test]
    fn prepared_graph_grpc_runtime_rejects_endpoints_without_a_grpc_binding() {
        // Negative control proving the gRPC arm genuinely prepares bindings: the
        // standard `chat` endpoint has no gRPC binding, so the gRPC arm must fail
        // at binding preparation (the HTTP arm would accept it).
        let factory = PreparedRunnerGraphEndpointRuntimeFactory::new(
            EndpointRegistry::builtin().unwrap(),
            Arc::new(vec![crate::engine::registry::ValidatedEndpointProfileV2 {
                profile_id: "default".into(),
                endpoint_id: EndpointId::new("chat").unwrap(),
                config: RawEndpointConfig {
                    urls: vec!["grpc://127.0.0.1:1".into()],
                    ..RawEndpointConfig::default()
                },
                connection_reuse: ConnectionReuseStrategy::Pooled,
                client: Default::default(),
                session_header: None,
            }]),
            Arc::new(AuthoredInputTokenCounter),
            Arc::new(crate::engine::grpc_execution::GrpcNativeExecution::new()),
            None,
            false,
        )
        .unwrap();
        let Err(error) = factory.prepare_worker(Rc::new(SimClock::new()), 0, "fixture-model")
        else {
            panic!("gRPC graph runtime accepted an endpoint with no gRPC binding")
        };
        let rendered = format!("{error:#}").to_ascii_lowercase();
        assert!(
            rendered.contains("grpc") || rendered.contains("binding"),
            "unexpected error building gRPC graph runtime: {error:#}"
        );
    }

    #[test]
    fn prepared_graph_runtime_rejects_dataset_only_raw_token_requirements() {
        let result = PreparedRunnerGraphEndpointRuntimeFactory::new(
            EndpointRegistry::builtin().unwrap(),
            Arc::new(vec![crate::engine::registry::ValidatedEndpointProfileV2 {
                profile_id: "default".into(),
                endpoint_id: EndpointId::new("vllm_generate").unwrap(),
                config: RawEndpointConfig {
                    urls: vec!["http://127.0.0.1:1".into()],
                    ..RawEndpointConfig::default()
                },
                connection_reuse: ConnectionReuseStrategy::Pooled,
                client: Default::default(),
                session_header: None,
            }]),
            Arc::new(AuthoredInputTokenCounter),
            Arc::new(crate::engine::online_execution::HttpNativeExecution::new(
                Arc::new(crate::engine::turn_execution::HttpExecutionFactory),
            )),
            None,
            false,
        );

        let Err(error) = result else {
            panic!("raw-token-only endpoint was accepted by direct Graph-IR")
        };
        assert!(format!("{error:#}").contains("requires raw token IDs"));
    }

    #[test]
    fn dynamo_session_and_parent_headers_share_the_instance_suffix() {
        let headers = BTreeMap::from([
            ("x-dynamo-session-id".into(), "child".into()),
            ("x-dynamo-parent-session-id".into(), "root::stale".into()),
            ("x-dynamo-session-final".into(), "true".into()),
        ]);
        let unique =
            uniquify_dynamo_session_headers(headers, Phase::Profiling, "root::instance-27");
        assert_eq!(
            unique["x-dynamo-session-id"],
            "child::profiling-instance-27"
        );
        assert_eq!(
            unique["x-dynamo-parent-session-id"],
            "root::stale::profiling-instance-27"
        );
        assert_eq!(unique["x-dynamo-session-final"], "true");
    }

    #[test]
    fn cache_bust_marker_is_disabled_for_none_target() {
        let cache_bust = GraphCacheBust {
            benchmark_id: "run-1".into(),
            target: crate::engine::graph_input::CacheBustTarget::None,
        };
        assert!(cache_bust.marker("trace::abc").is_none());
    }

    #[test]
    fn replay_supplement_requires_successful_profiling_work() {
        assert!(should_emit_replay_supplement(Phase::Profiling, true, true,));
        assert!(!should_emit_replay_supplement(Phase::Warmup, true, true));
        assert!(!should_emit_replay_supplement(
            Phase::Profiling,
            true,
            false,
        ));
    }

    #[test]
    fn cache_bust_marker_structure_is_stable_and_instance_unique() {
        let cache_bust = GraphCacheBust {
            benchmark_id: "run-1".into(),
            target: crate::engine::graph_input::CacheBustTarget::FirstTurnPrefix,
        };
        let first = cache_bust.marker("trace::inst-a").expect("marker minted");
        let second = cache_bust.marker("trace::inst-b").expect("marker minted");
        // `[rid:` + 12 hex + `]` + `\n\n` -> distinct nonce per instance, but a
        // fixed 20-byte shape so the mock's char tokenizer counts it identically.
        assert!(first.starts_with("[rid:") && first.ends_with("]\n\n"));
        assert_eq!(first.len(), "[rid:0123456789ab]\n\n".len());
        assert_eq!(first.len(), second.len());
        assert_ne!(first, second, "instance nonce must vary the marker");
        // Idempotent within one instance (all nodes of a trace share it).
        assert_eq!(cache_bust.marker("trace::inst-a").unwrap(), first);
    }

    #[test]
    fn prepend_marker_targets_first_user_message_idempotently() {
        let marker = "[rid:0123456789ab]\n\n";
        let mut messages = vec![
            serde_json::json!({"role": "user", "content": "hello"}),
            serde_json::json!({"role": "assistant", "content": "hi"}),
            serde_json::json!({"role": "user", "content": "again"}),
        ];
        prepend_cache_bust_marker(&mut messages, marker, CacheBustTarget::FirstTurnPrefix);
        assert_eq!(messages[0]["content"], format!("{marker}hello"));
        assert_eq!(messages[1]["content"], "hi");
        assert_eq!(messages[2]["content"], "again");
        // Re-applying the same marker is a no-op (shared prefix, every credit).
        prepend_cache_bust_marker(&mut messages, marker, CacheBustTarget::FirstTurnPrefix);
        assert_eq!(messages[0]["content"], format!("{marker}hello"));
    }

    #[test]
    fn prepend_marker_handles_multimodal_parts_content() {
        let marker = "[rid:0123456789ab]\n\n";
        let mut messages = vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        })];
        prepend_cache_bust_marker(&mut messages, marker, CacheBustTarget::FirstTurnPrefix);
        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(
            parts[0],
            serde_json::json!({"type": "text", "text": "[rid:0123456789ab]"})
        );
        assert_eq!(
            parts[1],
            serde_json::json!({"type": "text", "text": "hello"})
        );
        prepend_cache_bust_marker(&mut messages, marker, CacheBustTarget::FirstTurnPrefix);
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn plain_dynamo_trace_keeps_recorded_session_headers_verbatim() {
        let headers = BTreeMap::from([
            ("x-dynamo-session-id".into(), "child".into()),
            ("x-dynamo-session-final".into(), "true".into()),
        ]);
        assert_eq!(
            uniquify_dynamo_session_headers(headers.clone(), Phase::Profiling, "root"),
            headers
        );
    }
}
