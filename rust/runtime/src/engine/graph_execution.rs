// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-owned HTTP execution for complete Graph-IR traces.
//!
//! The graph coordinator never owns transport state. This factory is installed
//! behind `aiperf-graph`'s whole-trace placement seam, so each native worker
//! builds its own clock, transport, materializer, metric observer, and node
//! policies. Replacing native placement with ZMQ leaves graph scheduling and
//! this worker-side execution contract unchanged.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use crate::clock::{Clock, RealClock, RealClockAnchor};
use crate::dataset::{Handle, Payload, SegmentStore};
use crate::endpoints::{
    CreditPhase, EndpointId, EndpointKey, EndpointRegistry, PreparedEndpointTable, PreparedRequest,
    Turn,
};
use crate::failure::OnFailure;
use crate::graph::errors::TraceError;
use crate::graph::execution::{LocalGraphTraceExecutionBackend, TracePlacement};
use crate::graph::materialize::SegmentItemsMaterializer;
use crate::graph::model::{GraphTracePlan, LlmNode};
use crate::graph::placement::{
    GraphPlacementError, ThreadPerCoreTracePlacement, TracePlacementFactory,
};
use crate::graph::policy::{
    AbortTraceNodeFailurePolicy, CancellationNodePolicy, CompositeNodeDispatchPolicy,
    NodeDispatchPolicy, NodeFailurePolicy, PrefillSlotNodePolicy, ResilientNodeFailurePolicy,
};
use crate::graph::sink::{GraphDispatchOptions, GraphReply, GraphSink};
use crate::graph::wire::OpenAiChatMessage;
use crate::http::{
    Dispatcher, HttpRequest, PreparedEndpointReference, PreparedHttpEndpoint, PreparedTurn,
    TransportSink, TransportSinkConfig,
};
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use crate::metrics_core::{InferenceDimensions, MetricsConfig, Phase};
use crate::multiturn::InputTokenCounter;
use crate::rng::{RngRoot, namespace};
use crate::timing::{BernoulliFixedDelay, SlotPool};
use crate::transport_grpc::GrpcBindingRegistry;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::engine::execute::DEFAULT_ENDPOINT_PROFILE_ID;
use crate::engine::grpc_turn_execution::grpc_sink_with_endpoints;
use crate::engine::records::{CapturedHttpExchange, CapturedModelOutput, CapturedRecord};
use crate::engine::registry::ValidatedEndpointProfileV2;

/// Transport that a graph endpoint runtime dispatches its prepared turns over.
///
/// Selected from the run's resolved transport (`cfg.transport.type`). Graph
/// scheduling, body materialization, and metrics are transport-blind; only the
/// worker-local dispatcher construction in
/// [`PreparedRunnerGraphEndpointRuntimeFactory::prepare_worker`] branches on it.
/// A future `PreparedTurn` transport (e.g. WebSocket) adds one variant plus one
/// construction arm and nothing else in the graph runtime changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GraphTransportKind {
    /// Clock-injected hyper HTTP/SSE dispatch (`TransportSink`).
    Http,
    /// Clock-injected Tonic gRPC dispatch (`GrpcTransportSink`).
    ///
    /// Production routing selects this arm when the graph workload resolves a
    /// gRPC transport by id/type (`online_execution::lower_graph`), so `grpc +
    /// graph` dispatches graph nodes over Tonic.
    Grpc,
}

/// Composition seam for whole-trace graph execution placement.
///
/// The graph coordinator owns root selection, arrival, admission, and failure
/// policy. This factory owns only where a complete prepared trace executes. A
/// future ZMQ or RPC implementation can therefore replace native placement
/// without changing graph workload or runner orchestration code.
pub trait RunnerGraphPlacementFactory: Send + Sync {
    /// Build one placement backend from a worker-local execution factory.
    fn build(
        &self,
        worker_count: usize,
        worker_factory: Arc<dyn TracePlacementFactory>,
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
pub(crate) enum RunnerGraphExecutionEvent {
    /// One node crossed its first meaningful token edge.
    FirstToken {
        /// Unique root execution-instance identity.
        trace_id: String,
        /// Native request identity used to reject duplicate edges.
        uuid: Uuid,
    },
    /// One node reached a transport terminal and has a complete native record.
    Record {
        /// Complete native metric record for the terminal node request.
        record: Box<CapturedRecord>,
        /// Static graph node id (e.g. `"n_1"`) this terminal record belongs to.
        ///
        /// Carried so the phase observer can attribute the return to a specific
        /// node in its per-lane executed-node/return-wall ledgers (the E3c
        /// cache-pressure handoff). `None` for backends with no node identity
        /// (the offline dynosim adapter), which never feed the warmup handoff.
        /// Ports the `credit.node_ordinal` -> `node_id` inversion Python does in
        /// `graph_ir_replay.py:_record_return_wall`
        /// (`src/aiperf/timing/strategies/graph_ir_replay.py:884`); the Rust
        /// worker already holds the node id, so no ordinal round-trip is needed.
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
/// The stock implementation uses an in-process JSON-free channel. A future
/// remote placement can decode the same terminal facts into this seam without
/// changing phase accounting, adaptive sampling, or artifact collection.
pub(crate) trait RunnerGraphExecutionEventSink: Send + Sync {
    /// Deliver one ordered execution event or fail the run if delivery closed.
    fn emit(&self, event: RunnerGraphExecutionEvent) -> Result<(), TraceError>;
}

/// Channel-backed execution-event sink used by native placement.
pub(crate) struct ChannelRunnerGraphExecutionEventSink {
    sender: mpsc::UnboundedSender<RunnerGraphExecutionEvent>,
}

impl ChannelRunnerGraphExecutionEventSink {
    /// Bind the worker side to one phase-local coordinator receiver.
    pub(crate) fn new(sender: mpsc::UnboundedSender<RunnerGraphExecutionEvent>) -> Self {
        Self { sender }
    }
}

impl RunnerGraphExecutionEventSink for ChannelRunnerGraphExecutionEventSink {
    fn emit(&self, event: RunnerGraphExecutionEvent) -> Result<(), TraceError> {
        self.sender
            .send(event)
            .map_err(|_| TraceError::Other("graph execution event receiver closed".into()))
    }
}

/// Coordinator-side wrapper that observes every placement terminal.
///
/// Keeping this outside worker implementations covers native queue rejection
/// and future remote transports as well as traces that reached a worker.
pub(crate) struct ObservedRunnerGraphPlacement {
    delegate: Rc<dyn TracePlacement>,
    events: Arc<dyn RunnerGraphExecutionEventSink>,
    requires_node_records: bool,
}

impl ObservedRunnerGraphPlacement {
    /// Decorate one placement without changing its dispatch/control semantics.
    pub(crate) fn new(
        delegate: Rc<dyn TracePlacement>,
        events: Arc<dyn RunnerGraphExecutionEventSink>,
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
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
        let trace_id = plan.trace.id.clone();
        let node_count = plan.graph.nodes.len();
        let result = self.delegate.execute_trace(plan).await;
        self.events.emit(RunnerGraphExecutionEvent::TraceComplete {
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

impl RunnerGraphPlacementFactory for NativeRunnerGraphPlacementFactory {
    fn build(
        &self,
        worker_count: usize,
        worker_factory: Arc<dyn TracePlacementFactory>,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
        Ok(Rc::new(ThreadPerCoreTracePlacement::new(
            worker_count,
            worker_factory,
        )?))
    }

    fn requires_node_records(&self) -> bool {
        true
    }
}

/// Startup seam that prepares one endpoint dispatcher per graph worker.
///
/// Native HTTP prepares local transports and endpoint bindings. A future
/// remote placement can implement this contract with the same data-only
/// [`PreparedTurn`] identity without changing graph scheduling.
pub(crate) trait RunnerGraphEndpointRuntimeFactory: Send + Sync {
    /// Prepare one worker-local dispatcher before any trace is admitted.
    fn prepare_worker(
        &self,
        clock: Rc<dyn Clock>,
        run_origin_ns: i64,
        model: &str,
    ) -> Result<Rc<dyn RunnerGraphEndpointRuntime>>;
}

pub(crate) trait RunnerGraphEndpointRuntime {
    fn materialize(&self, input: GraphEndpointRequest) -> Result<GraphEndpointDispatch>;

    /// Transport the worker-local dispatchers were built over. Used to prove the
    /// gRPC arm was taken without downcasting the erased `Rc<dyn Dispatcher>`.
    fn kind(&self) -> GraphTransportKind;
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
}

pub(crate) struct GraphEndpointDispatch {
    transport: Rc<dyn Dispatcher>,
    request: HttpRequest,
    endpoint: PreparedHttpEndpoint,
    input_tokens: u64,
}

/// Protocol-v2 factory that prepares open endpoint profiles directly through
/// the frozen endpoint registry on every graph worker.
pub(crate) struct PreparedRunnerGraphEndpointRuntimeFactory {
    registry: EndpointRegistry,
    profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    transport_kind: GraphTransportKind,
}

impl PreparedRunnerGraphEndpointRuntimeFactory {
    /// Bind one normalized profile plan to the process's frozen registry over
    /// the run's resolved transport (`transport_kind`).
    pub(crate) fn new(
        registry: EndpointRegistry,
        profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
        input_token_counter: Arc<dyn InputTokenCounter>,
        transport_kind: GraphTransportKind,
    ) -> Result<Self> {
        for profile in profiles.iter() {
            let descriptor = registry.resolve_factory(&profile.endpoint_id)?.descriptor();
            ensure!(
                !descriptor.requires_raw_token_ids,
                "graph endpoint profile {:?} selects {:?}, which requires raw token IDs; direct Graph-IR nodes do not yet carry the dataset raw-token handle",
                profile.profile_id,
                descriptor.id
            );
        }
        Ok(Self {
            registry,
            profiles,
            input_token_counter,
            transport_kind,
        })
    }
}

impl RunnerGraphEndpointRuntimeFactory for PreparedRunnerGraphEndpointRuntimeFactory {
    fn prepare_worker(
        &self,
        clock: Rc<dyn Clock>,
        run_origin_ns: i64,
        model: &str,
    ) -> Result<Rc<dyn RunnerGraphEndpointRuntime>> {
        let mut table = PreparedEndpointTable::new();
        let mut staged = Vec::with_capacity(self.profiles.len());
        for profile in self.profiles.iter() {
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
                },
                url_count: profile.config.urls.len(),
            });
        }
        let table = Rc::new(table);
        // gRPC needs one dense binding table shared across every profile's sink;
        // HTTP needs no bindings at all, so only build it on the gRPC arm.
        let bindings = match self.transport_kind {
            GraphTransportKind::Http => None,
            GraphTransportKind::Grpc => Some(GrpcBindingRegistry::builtin()?),
        };
        let profiles = staged
            .into_iter()
            .map(|profile| {
                let transport: Rc<dyn Dispatcher> = match self.transport_kind {
                    GraphTransportKind::Http => Rc::new(
                        TransportSink::new_multi_configured(
                            clock.clone(),
                            run_origin_ns,
                            &profile.urls,
                            model,
                            profile.transport_config,
                        )?
                        .with_prepared_endpoints(table.clone()),
                    ),
                    GraphTransportKind::Grpc => Rc::new(grpc_sink_with_endpoints(
                        clock.clone(),
                        run_origin_ns,
                        &profile.urls,
                        model.to_string(),
                        profile.transport_config,
                        bindings
                            .clone()
                            .expect("gRPC bindings prepared for the gRPC transport arm"),
                        table.clone(),
                    )?),
                };
                Ok((
                    profile.profile_id,
                    PreparedGraphProfileRuntime {
                        endpoint_id: profile.endpoint_id,
                        keys: profile.keys,
                        transport,
                        url_count: profile.url_count,
                    },
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(
            profiles.contains_key(DEFAULT_ENDPOINT_PROFILE_ID),
            "default endpoint profile {DEFAULT_ENDPOINT_PROFILE_ID:?} was not prepared"
        );
        Ok(Rc::new(PreparedRunnerGraphEndpointRuntime {
            table,
            profiles,
            default_profile_id: DEFAULT_ENDPOINT_PROFILE_ID.into(),
            input_token_counter: self.input_token_counter.clone(),
            transport_kind: self.transport_kind,
        }))
    }
}

/// Per-profile inputs staged after endpoint preparation but before the
/// worker-local dispatcher is built (the dispatcher construction is deferred so
/// both transport arms share the finalized `Rc<PreparedEndpointTable>`).
struct StagedGraphProfile {
    profile_id: String,
    endpoint_id: EndpointId,
    keys: [EndpointKey; 2],
    urls: Vec<String>,
    transport_config: TransportSinkConfig,
    url_count: usize,
}

struct PreparedGraphProfileRuntime {
    endpoint_id: EndpointId,
    keys: [EndpointKey; 2],
    transport: Rc<dyn Dispatcher>,
    url_count: usize,
}

struct PreparedRunnerGraphEndpointRuntime {
    table: Rc<PreparedEndpointTable>,
    profiles: BTreeMap<String, PreparedGraphProfileRuntime>,
    default_profile_id: String,
    input_token_counter: Arc<dyn InputTokenCounter>,
    transport_kind: GraphTransportKind,
}

impl RunnerGraphEndpointRuntime for PreparedRunnerGraphEndpointRuntime {
    fn kind(&self) -> GraphTransportKind {
        self.transport_kind
    }

    fn materialize(&self, input: GraphEndpointRequest) -> Result<GraphEndpointDispatch> {
        let profile_id = input
            .selector
            .as_deref()
            .unwrap_or(&self.default_profile_id);
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
        let input_tokens = self.input_token_counter.count_prepared_input_tokens(
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
        let request = HttpRequest {
            uuid: Uuid::new_v4(),
            input_length: usize::try_from(input_tokens).unwrap_or(usize::MAX),
            max_output_tokens: input.max_output_tokens,
            prompt_text: None,
            request_body: None,
            request_body_bytes: Some(payload),
            headers,
            parameters: input.parameters,
            endpoint_path,
            streaming: endpoint.config().streaming(),
            x_correlation_id: Some(input.trace_id),
            is_final_turn: input.is_final_turn,
            cancel_after_ns: input.cancel_after_ns,
            url_index: Some(session_url_index(input.session_num, profile.url_count)),
        };
        Ok(GraphEndpointDispatch {
            transport: profile.transport.clone(),
            request,
            endpoint: PreparedHttpEndpoint::Prepared(PreparedEndpointReference {
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
pub(crate) struct RunnerGraphBackendFactoryConfig {
    pub(crate) real_clock_anchor: RealClockAnchor,
    pub(crate) run_origin_ns: i64,
    pub(crate) model: String,
    pub(crate) default_max_tokens: usize,
    pub(crate) endpoint_runtime_factory: Arc<dyn RunnerGraphEndpointRuntimeFactory>,
    pub(crate) segments: Arc<dyn SegmentStore>,
    pub(crate) metrics: MetricsConfig,
    pub(crate) phase: Phase,
    pub(crate) prefill_concurrency: Option<usize>,
    pub(crate) cancellation: Option<GraphCancellationConfig>,
    pub(crate) raw_enabled: bool,
    pub(crate) events: Arc<dyn RunnerGraphExecutionEventSink>,
    /// Config-selected per-node failure discipline. `Abort` aborts the trace on
    /// the first node failure (default); `Continue` treats a failed node as
    /// empty and drains the DAG.
    pub(crate) on_failure: OnFailure,
    /// Scenario-locked first-turn cache-bust marker configuration. `None` when
    /// the run has no `cache_bust_target`; `Some` prepends a per-conversation
    /// nonce marker to the first user message of every request.
    pub(crate) cache_bust: Option<GraphCacheBust>,
}

/// Resolved first-turn cache-bust marker inputs shared by every graph worker.
///
/// Ports the marker minting agentx does in
/// `ajc/agentx:src/aiperf/timing/strategies/cache_bust.py::build_cache_bust_marker`
/// (FIRST_TURN_PREFIX arm) onto the native recorded-graph path. The marker text
/// is `[rid:<sha256(f"{benchmark_id}:{recycle_pass}:{trajectory_index}:{trace_id}")[:12]>]\n\n`;
/// on the native path the per-trace instance nonce (the suffix of
/// [`RunnerGraphSink::trace_id`] after `"::"`) supplies the cross-instance and
/// cross-recycle uniqueness agentx gets from its `(recycle_pass, trajectory_index)`
/// tuple, so `recycle_pass`/`trajectory_index` are folded into the instance id
/// and pinned at `0` in the digest tuple. Because the marker is a per-run nonce,
/// its BYTES differ from agentx run to run; only its TOKEN COUNT matters for ISL
/// parity, and that is invariant (a fixed-length hex digest wrapped in
/// `[rid:...]\n\n` always tokenizes to the same count, with a clean whitespace
/// break before the message body). Sharing the same marker across a continued
/// warmup->profiling session (agentx's KV-cache lineage nicety) is a documented
/// lane-0-baseline future refinement, consistent with the graph-fidelity caveats
/// in the module map.
#[derive(Clone)]
pub(crate) struct GraphCacheBust {
    /// Per-run benchmark identity digested into the marker.
    pub(crate) benchmark_id: String,
    /// Resolved marker placement target (only `FirstTurnPrefix` mints a marker).
    pub(crate) target: crate::engine::graph_input::CacheBustTarget,
}

impl GraphCacheBust {
    /// Mint the per-conversation `[rid:<digest>]\n\n` marker for one trace
    /// instance, or `None` when the target is disabled. The digest is taken over
    /// the whole instance `trace_id` (base template + `::` nonce) so every
    /// instance and recycle draws a distinct marker while every node of one
    /// instance shares it.
    fn marker(&self, trace_id: &str) -> Option<String> {
        use sha2::{Digest, Sha256};
        if !self.target.is_enabled() {
            return None;
        }
        // Pin recycle_pass/trajectory_index to 0: the instance nonce inside
        // `trace_id` already carries the per-instance/per-recycle uniqueness
        // agentx encodes in that tuple, and the resulting marker is a nonce that
        // only needs a stable token length (invariant for any digest).
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

/// Prepend `marker` to the content of the first `role == "user"` message.
///
/// Port of `ajc/agentx:src/aiperf/workers/worker.py::_inject_marker_into_first_user_turn`
/// (string-content and multimodal-parts arms) for the FIRST_TURN_PREFIX target.
/// Idempotent: re-prepending the same constant marker is a no-op, matching
/// agentx's every-credit injection over a shared prefix. The marker's trailing
/// `\n\n` guarantees a clean whitespace token break before the message body, so
/// the recorded content's own tokenization is unchanged.
fn prepend_first_turn_marker(messages: &mut [Value], marker: &str) {
    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) != Some("user") {
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
pub(crate) struct RunnerGraphBackendFactory {
    config: RunnerGraphBackendFactoryConfig,
}

impl RunnerGraphBackendFactory {
    pub(crate) fn new(config: RunnerGraphBackendFactoryConfig) -> Self {
        Self { config }
    }
}

impl TracePlacementFactory for RunnerGraphBackendFactory {
    fn create_backend(
        &self,
        worker_id: usize,
    ) -> Result<Rc<dyn TracePlacement>, GraphPlacementError> {
        let clock: Rc<dyn Clock> = RealClock::from_anchor(self.config.real_clock_anchor);
        let endpoint_runtime = self
            .config
            .endpoint_runtime_factory
            .prepare_worker(clock.clone(), self.config.run_origin_ns, &self.config.model)
            .map_err(|error| GraphPlacementError(error.to_string()))?;
        tracing::debug!(
            worker_id,
            transport = ?endpoint_runtime.kind(),
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

        Ok(Rc::new(RunnerGraphWorkerBackend {
            clock,
            endpoint_runtime,
            materializer: Rc::new(SegmentItemsMaterializer::new(self.config.segments.clone())),
            segments: self.config.segments.clone(),
            metrics: self.config.metrics.clone(),
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
            prefill_slots,
            next_session: Cell::new(0),
            next_execution: Cell::new(0),
            cancelled: Cell::new(false),
            active: RefCell::new(HashMap::new()),
        }))
    }
}

struct RunnerGraphWorkerBackend {
    clock: Rc<dyn Clock>,
    endpoint_runtime: Rc<dyn RunnerGraphEndpointRuntime>,
    materializer: Rc<SegmentItemsMaterializer>,
    segments: Arc<dyn SegmentStore>,
    metrics: MetricsConfig,
    phase: Phase,
    worker_id: usize,
    model: String,
    default_max_tokens: usize,
    run_origin_ns: i64,
    raw_enabled: bool,
    events: Arc<dyn RunnerGraphExecutionEventSink>,
    node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
    on_failure: OnFailure,
    cache_bust: Option<GraphCacheBust>,
    prefill_slots: Option<Rc<SlotPool>>,
    next_session: Cell<u64>,
    next_execution: Cell<u64>,
    cancelled: Cell<bool>,
    active: RefCell<HashMap<u64, Rc<LocalGraphTraceExecutionBackend<OpenAiChatMessage>>>>,
}

#[async_trait(?Send)]
impl TracePlacement for RunnerGraphWorkerBackend {
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
        if self.cancelled.get() {
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
        let terminal_nodes = terminal_graph_nodes(&plan);
        // Mint the per-conversation cache-bust marker once for this trace
        // instance; every node dispatched for it shares the marker, so the
        // first-turn user prefix is byte-stable across the conversation's turns.
        let cache_bust_marker = self
            .cache_bust
            .as_ref()
            .and_then(|cache_bust| cache_bust.marker(&plan.trace.id));
        let observer = Rc::new(NativeMetricsObserver::new(
            self.clock.clone(),
            self.run_origin_ns,
            self.metrics.clone(),
        ));
        let sink = Rc::new(RunnerGraphSink {
            clock: self.clock.clone(),
            endpoint_runtime: self.endpoint_runtime.clone(),
            trace_id: plan.trace.id.clone(),
            nodes: plan.graph.nodes.clone().into_iter().collect(),
            segments: self.segments.clone(),
            observer,
            phase: self.phase,
            worker_id: self.worker_id,
            session_num,
            model: self.model.clone(),
            default_max_tokens: self.default_max_tokens,
            run_origin_ns: self.run_origin_ns,
            raw_enabled: self.raw_enabled,
            terminal_nodes,
            events: self.events.clone(),
            cache_bust_marker,
            emitted_records: Cell::new(0),
        });
        let mut local = LocalGraphTraceExecutionBackend::new(
            self.clock.clone(),
            self.materializer.clone(),
            sink.clone(),
        );
        if let Some(policy) = &self.node_policy {
            local = local.with_node_policy(policy.clone());
        }
        // Node failure discipline mirrors the run-level policy: `Abort` aborts
        // the trace on a node failure (default); `Continue` treats it as empty
        // and lets the DAG drain (the dormant Python-parity resilient default).
        let node_failure: Rc<dyn NodeFailurePolicy> = match self.on_failure {
            OnFailure::Abort => Rc::new(AbortTraceNodeFailurePolicy),
            OnFailure::Continue => Rc::new(ResilientNodeFailurePolicy),
        };
        local = local.with_node_failure(node_failure);
        let local = Rc::new(local);
        let execution_id = self.next_execution.get();
        self.next_execution.set(execution_id.saturating_add(1));
        self.active.borrow_mut().insert(execution_id, local.clone());
        let result = local.execute_trace(plan).await;
        let finalized = sink
            .verify_finalized_records()
            .map_err(|error| TraceError::Other(error.to_string()));
        self.active.borrow_mut().remove(&execution_id);
        finalized?;
        result
    }

    fn cancel_inflight(&self) -> Result<(), TraceError> {
        self.cancelled.set(true);
        let active = self.active.borrow().values().cloned().collect::<Vec<_>>();
        let errors = active
            .iter()
            .filter_map(|backend| backend.cancel_inflight().err())
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

struct RunnerGraphSink {
    clock: Rc<dyn Clock>,
    endpoint_runtime: Rc<dyn RunnerGraphEndpointRuntime>,
    trace_id: String,
    nodes: HashMap<String, LlmNode>,
    segments: Arc<dyn SegmentStore>,
    observer: Rc<NativeMetricsObserver>,
    phase: Phase,
    worker_id: usize,
    session_num: u64,
    model: String,
    default_max_tokens: usize,
    run_origin_ns: i64,
    raw_enabled: bool,
    terminal_nodes: HashSet<String>,
    events: Arc<dyn RunnerGraphExecutionEventSink>,
    /// Per-conversation first-turn cache-bust marker, minted once per trace
    /// instance. `None` when cache-bust is disabled.
    cache_bust_marker: Option<String>,
    emitted_records: Cell<u64>,
}

#[async_trait(?Send)]
impl GraphSink<OpenAiChatMessage> for RunnerGraphSink {
    async fn dispatch(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        self.dispatch_with_options(
            node_id,
            messages,
            max_tokens,
            GraphDispatchOptions::default(),
            on_first_token,
        )
        .await
    }

    async fn dispatch_with_options(
        &self,
        node_id: &str,
        messages: Vec<Bytes>,
        max_tokens: Option<usize>,
        options: GraphDispatchOptions,
        on_first_token: &dyn Fn(),
    ) -> Result<GraphReply<OpenAiChatMessage>> {
        let node = self
            .nodes
            .get(node_id)
            .ok_or_else(|| anyhow!("graph node {node_id:?} has no execution metadata"))?;
        let model = metadata_string(node, "model")
            .map(str::to_string)
            .unwrap_or_else(|| self.model.clone());
        let mut raw_messages = messages
            .iter()
            .enumerate()
            .map(|(index, wire)| {
                serde_json::from_slice(wire).with_context(|| {
                    format!("decoding materialized graph message {index} for node {node_id:?}")
                })
            })
            .collect::<Result<Vec<Value>>>()?;
        // Scenario-locked FIRST_TURN_PREFIX cache-bust: prepend the trace
        // instance's marker to the first user message. Applied on every request
        // (idempotent over the shared prefix), so the marker rides at the head
        // of the conversation's first user turn in every accumulated request the
        // way agentx's worker-side injection does.
        if let Some(marker) = self.cache_bust_marker.as_deref() {
            prepend_first_turn_marker(&mut raw_messages, marker);
        }
        let extra_headers = uniquify_dynamo_session_headers(
            self.raw_string_map(node, "extra_headers_handle")?,
            self.phase,
            &self.trace_id,
        );
        let turn = Turn {
            model: Some(model.clone()),
            max_tokens: max_tokens
                .map(u32::try_from)
                .transpose()
                .context("graph max_tokens exceeds u32")?,
            raw_messages: Some(raw_messages),
            raw_tools: self.raw_array(node, "tools_handle")?,
            raw_system: self.raw_array(node, "raw_system_handle")?,
            extra_body: self.raw_object(node, "extra_body_handle")?,
            ..Turn::default()
        };
        let max_output_tokens = max_tokens.unwrap_or(self.default_max_tokens);
        let dispatch = self.endpoint_runtime.materialize(GraphEndpointRequest {
            selector: metadata_string(node, "endpoint").map(str::to_owned),
            model: model.clone(),
            turn,
            streaming: node.streaming,
            authored_input_tokens: metadata_u64(node, "input_tokens").unwrap_or(0),
            max_output_tokens,
            extra_headers,
            parameters: self.raw_string_map(node, "request_parameters_handle")?,
            trace_id: self.trace_id.clone(),
            is_final_turn: self.terminal_nodes.contains(node_id),
            cancel_after_ns: options.cancel_after_ns,
            session_num: self.session_num,
            phase: self.phase,
        })?;
        let GraphEndpointDispatch {
            transport,
            request,
            endpoint,
            input_tokens,
        } = dispatch;
        let uuid = request.uuid;
        let dimensions: InferenceDimensions =
            Dispatcher::inference_dimensions(transport.as_ref(), &request);
        self.observer.register_metadata(
            uuid,
            RequestMetricMetadata {
                phase: self.phase,
                session_num: Some(self.session_num),
                turn_index: metadata_u64(node, "turn_index")
                    .and_then(|value| u32::try_from(value).ok())
                    .unwrap_or(0),
                worker_id: Some(self.worker_id.to_string()),
                // One authored root may be cycled many times by a bounded or
                // duration phase. Metrics identity follows the unique trace
                // instance; the authored conversation stays in node metadata.
                conversation_id: Some(self.trace_id.clone()),
                dimensions,
                correlation_id: Some(format!("{}:{node_id}", self.trace_id)),
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
        let collected = transport
            .dispatch_collect(
                PreparedTurn {
                    request,
                    model,
                    endpoint,
                    endpoint_aware: true,
                    data_policy: crate::multiturn::TurnDataPolicy::ordinary(),
                },
                self.observer.as_ref(),
                &|_| {
                    if !first_token_emitted.replace(true)
                        && let Err(error) =
                            self.events.emit(RunnerGraphExecutionEvent::FirstToken {
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
        let ordinal = self.emitted_records.get();
        let ingest = self
            .observer
            .drain_terminal_record(uuid, ordinal)
            .ok_or_else(|| {
                anyhow!(
                    "graph trace {:?} could not snapshot terminal node {node_id:?}",
                    self.trace_id
                )
            })?;
        self.events.emit(RunnerGraphExecutionEvent::Record {
            record: Box::new(CapturedRecord {
                uuid,
                x_correlation_id: self.trace_id.clone(),
                output: CapturedModelOutput::from_parts(
                    &outcome.response_text,
                    outcome.model_response.content.as_deref(),
                    outcome.model_response.reasoning.as_deref(),
                ),
                raw: self.raw_enabled.then_some(CapturedHttpExchange {
                    request_payload: collected.request_payload.to_vec(),
                    record: collected.record,
                }),
                ingest,
            }),
            node_id: Some(node_id.to_owned()),
        })?;
        self.emitted_records.set(ordinal.saturating_add(1));

        Ok(match outcome.terminal {
            ReplayTerminalStatus::Completed => GraphReply::from_text(outcome.response_text),
            ReplayTerminalStatus::Canceled => GraphReply::cancelled(),
            ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed => GraphReply::failed(),
        })
    }
}

impl RunnerGraphSink {
    fn verify_finalized_records(&self) -> Result<()> {
        let (arrivals, retained) = self.observer.record_counts();
        let emitted = usize::try_from(self.emitted_records.get()).unwrap_or(usize::MAX);
        ensure!(
            retained == 0 && arrivals == emitted,
            "graph trace {:?} retained {retained} of {arrivals} metric records after emitting {emitted}",
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
    use super::*;
    use crate::clock::SimClock;
    use crate::endpoints::{
        ChatEndpoint, EffectiveEndpointConfig, Endpoint, EndpointDescriptor, EndpointFactory,
        EndpointRegistry, EndpointRegistryBuilder, EndpointResult, PreparedEndpoint,
        RawEndpointConfig, StatelessEndpointFactory,
    };
    use crate::multiturn::AuthoredInputTokenCounter;
    use crate::transport_http::models::ConnectionReuseStrategy;

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
    fn prepared_graph_runtime_does_not_require_a_legacy_endpoint_adapter() {
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
            GraphTransportKind::Http,
        )
        .unwrap();
        let runtime = factory
            .prepare_worker(Rc::new(SimClock::new()), 0, "fixture-model")
            .unwrap();
        assert_eq!(runtime.kind(), GraphTransportKind::Http);
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
            })
            .unwrap();
        assert_eq!(dispatch.input_tokens, 4);
        let PreparedHttpEndpoint::Prepared(reference) = dispatch.endpoint;
        assert_eq!(reference.endpoint_id.as_str(), "chat");
        assert_eq!(reference.key.index(), 1);
    }

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
            GraphTransportKind::Grpc,
        )
        .unwrap();
        let runtime = factory
            .prepare_worker(Rc::new(SimClock::new()), 0, "fixture-model")
            .unwrap();
        assert_eq!(runtime.kind(), GraphTransportKind::Grpc);
    }

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
            GraphTransportKind::Grpc,
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
            GraphTransportKind::Http,
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
        prepend_first_turn_marker(&mut messages, marker);
        assert_eq!(messages[0]["content"], format!("{marker}hello"));
        // Later user turns and assistant turns are untouched.
        assert_eq!(messages[1]["content"], "hi");
        assert_eq!(messages[2]["content"], "again");
        // Re-applying the same marker is a no-op (shared prefix, every credit).
        prepend_first_turn_marker(&mut messages, marker);
        assert_eq!(messages[0]["content"], format!("{marker}hello"));
    }

    #[test]
    fn prepend_marker_handles_multimodal_parts_content() {
        let marker = "[rid:0123456789ab]\n\n";
        let mut messages = vec![serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "hello"}],
        })];
        prepend_first_turn_marker(&mut messages, marker);
        let parts = messages[0]["content"].as_array().unwrap();
        assert_eq!(
            parts[0],
            serde_json::json!({"type": "text", "text": "[rid:0123456789ab]"})
        );
        assert_eq!(
            parts[1],
            serde_json::json!({"type": "text", "text": "hello"})
        );
        prepend_first_turn_marker(&mut messages, marker);
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
