// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runner-owned HTTP execution for complete Graph-IR traces.
//!
//! The graph coordinator never owns transport state. This factory is installed
//! behind `aiperf-graph`'s whole-trace placement seam, so each native worker
//! builds its own clock, transport, materializer, metric observer, and node
//! policies. Replacing native placement with ZMQ leaves graph scheduling and
//! this worker-side execution contract unchanged.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;

use aiperf::http::{
    HttpRequest, HttpRequestDispatcher, PreparedEndpointReference, PreparedHttpEndpoint,
    PreparedHttpTurn, TransportSink, TransportSinkConfig,
};
use aiperf::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use aiperf::multiturn::InputTokenCounter;
use aiperf::multiturn::LegacyTurnEndpointBinding;
use aiperf_clock::{Clock, RealClock, RealClockAnchor};
use aiperf_dataset::{EndpointResolver, Handle, Payload, SegmentStore};
use aiperf_endpoints::{
    CreditPhase, EndpointConfig, EndpointId, EndpointKey, EndpointRegistry, ModelEndpoint,
    PreparedEndpointTable, PreparedRequest, RequestInfo, Turn,
};
use aiperf_graph::errors::TraceError;
use aiperf_graph::execution::{GraphTraceExecutionBackend, LocalGraphTraceExecutionBackend};
use aiperf_graph::materialize::SegmentItemsMaterializer;
use aiperf_graph::model::{GraphTracePlan, LlmNode};
use aiperf_graph::placement::{
    GraphPlacementError, GraphTraceExecutionBackendFactory, ThreadPerCoreGraphTraceExecutionBackend,
};
use aiperf_graph::policy::{
    AbortTraceNodeFailurePolicy, CancellationNodePolicy, CompositeNodeDispatchPolicy,
    NodeDispatchPolicy, PrefillSlotNodePolicy,
};
use aiperf_graph::sink::{GraphDispatchOptions, GraphReply, GraphSink};
use aiperf_graph::wire::OpenAiChatMessage;
use aiperf_metrics::{InferenceDimensions, MetricsConfig, Phase};
use aiperf_rng::{RngRoot, namespace};
use aiperf_timing::{BernoulliFixedDelay, SlotPool};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use bytes::Bytes;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use serde_json::{Map, Value};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::execute::DEFAULT_ENDPOINT_PROFILE_ID;
use crate::records::{CapturedHttpExchange, CapturedModelOutput, CapturedRecord};
use crate::registry::ValidatedEndpointProfileV2;

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
        worker_factory: Arc<dyn GraphTraceExecutionBackendFactory>,
    ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError>;
}

/// Stock thread-per-core whole-trace placement.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeRunnerGraphPlacementFactory;

impl RunnerGraphPlacementFactory for NativeRunnerGraphPlacementFactory {
    fn build(
        &self,
        worker_count: usize,
        worker_factory: Arc<dyn GraphTraceExecutionBackendFactory>,
    ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError> {
        Ok(Rc::new(ThreadPerCoreGraphTraceExecutionBackend::new(
            worker_count,
            worker_factory,
        )?))
    }
}

/// Startup seam that prepares one endpoint dispatcher per graph worker.
///
/// Native HTTP prepares local transports and endpoint bindings. A future
/// remote placement can implement this contract with the same data-only
/// [`PreparedHttpTurn`] identity without changing graph scheduling.
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
    transport: Rc<TransportSink>,
    request: HttpRequest,
    endpoint: PreparedHttpEndpoint,
    input_tokens: u64,
}

/// Protocol-v1 compatibility factory for graph endpoint dispatch.
pub(crate) struct LegacyRunnerGraphEndpointRuntimeFactory {
    base_urls: Vec<String>,
    transport: TransportSinkConfig,
    endpoint: EndpointConfig,
    endpoint_resolver: Arc<dyn EndpointResolver>,
    input_token_counter: Arc<dyn InputTokenCounter>,
}

impl LegacyRunnerGraphEndpointRuntimeFactory {
    /// Retain the historical dialect-selection behavior for protocol v1.
    pub(crate) fn new(
        base_urls: Vec<String>,
        transport: TransportSinkConfig,
        endpoint: EndpointConfig,
        endpoint_resolver: Arc<dyn EndpointResolver>,
        input_token_counter: Arc<dyn InputTokenCounter>,
    ) -> Self {
        Self {
            base_urls,
            transport,
            endpoint,
            endpoint_resolver,
            input_token_counter,
        }
    }
}

impl RunnerGraphEndpointRuntimeFactory for LegacyRunnerGraphEndpointRuntimeFactory {
    fn prepare_worker(
        &self,
        clock: Rc<dyn Clock>,
        run_origin_ns: i64,
        model: &str,
    ) -> Result<Rc<dyn RunnerGraphEndpointRuntime>> {
        let transport = TransportSink::new_multi_configured(
            clock,
            run_origin_ns,
            &self.base_urls,
            model,
            self.transport.clone(),
        )?;
        Ok(Rc::new(LegacyRunnerGraphEndpointRuntime {
            transport: Rc::new(transport),
            base_url_count: self.base_urls.len(),
            endpoint: self.endpoint.clone(),
            endpoint_resolver: self.endpoint_resolver.clone(),
            input_token_counter: self.input_token_counter.clone(),
        }))
    }
}

struct LegacyRunnerGraphEndpointRuntime {
    transport: Rc<TransportSink>,
    base_url_count: usize,
    endpoint: EndpointConfig,
    endpoint_resolver: Arc<dyn EndpointResolver>,
    input_token_counter: Arc<dyn InputTokenCounter>,
}

impl RunnerGraphEndpointRuntime for LegacyRunnerGraphEndpointRuntime {
    fn materialize(&self, input: GraphEndpointRequest) -> Result<GraphEndpointDispatch> {
        let endpoint = self.endpoint_resolver.resolve(input.selector.as_deref())?;
        let mut endpoint_config = self.endpoint.clone();
        endpoint_config.endpoint_type = endpoint.metadata().endpoint_type;
        endpoint_config.streaming = input.streaming && endpoint.metadata().supports_streaming;
        let request_info = RequestInfo {
            model_endpoint: ModelEndpoint {
                primary_model_name: input.model.clone(),
                endpoint: endpoint_config.clone(),
            },
            turns: vec![input.turn],
            system_message: None,
            user_context_message: None,
            credit_phase: credit_phase(input.phase),
            x_request_id: None,
            x_correlation_id: Some(input.trace_id.clone()),
            conversation_id: Some(input.trace_id.clone()),
        };
        let payload = Bytes::from(serde_json::to_vec(
            &endpoint.format_payload(&request_info)?,
        )?);
        let input_tokens = self.input_token_counter.count_input_tokens(
            endpoint.as_ref(),
            &payload,
            input.authored_input_tokens,
        )?;
        let mut headers = endpoint.format_headers(&endpoint_config);
        headers.extend(input.extra_headers);
        let endpoint_path = endpoint_config.path.clone().or_else(|| {
            if endpoint_config.streaming {
                endpoint.metadata().streaming_path
            } else {
                None
            }
            .or(endpoint.metadata().endpoint_path)
            .map(str::to_string)
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
            streaming: endpoint_config.streaming,
            x_correlation_id: Some(input.trace_id),
            is_final_turn: input.is_final_turn,
            cancel_after_ns: input.cancel_after_ns,
            url_index: Some(session_url_index(input.session_num, self.base_url_count)),
        };
        Ok(GraphEndpointDispatch {
            transport: self.transport.clone(),
            request,
            endpoint: PreparedHttpEndpoint::Legacy(Arc::new(LegacyTurnEndpointBinding {
                endpoint,
                config: endpoint_config,
            })),
            input_tokens,
        })
    }
}

/// Protocol-v2 factory that prepares open endpoint profiles directly through
/// the frozen endpoint registry on every graph worker.
pub(crate) struct PreparedRunnerGraphEndpointRuntimeFactory {
    registry: EndpointRegistry,
    profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    input_token_counter: Arc<dyn InputTokenCounter>,
}

impl PreparedRunnerGraphEndpointRuntimeFactory {
    /// Bind one normalized profile plan to the process's frozen registry.
    pub(crate) fn new(
        registry: EndpointRegistry,
        profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
        input_token_counter: Arc<dyn InputTokenCounter>,
    ) -> Self {
        Self {
            registry,
            profiles,
            input_token_counter,
        }
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
            let transport = TransportSink::new_multi_configured(
                clock.clone(),
                run_origin_ns,
                &profile.config.urls,
                model,
                TransportSinkConfig {
                    client: profile.client.clone(),
                    connection_reuse: profile.connection_reuse,
                    session_header: profile.session_header.clone(),
                },
            )?;
            staged.push((
                profile.profile_id.clone(),
                PreparedGraphProfile {
                    endpoint_id: profile.endpoint_id.clone(),
                    keys,
                    transport,
                    url_count: profile.config.urls.len(),
                },
            ));
        }
        let table = Rc::new(table);
        let profiles = staged
            .into_iter()
            .map(|(profile_id, profile)| {
                (
                    profile_id,
                    PreparedGraphProfileRuntime {
                        endpoint_id: profile.endpoint_id,
                        keys: profile.keys,
                        transport: Rc::new(
                            profile.transport.with_prepared_endpoints(table.clone()),
                        ),
                        url_count: profile.url_count,
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        ensure!(
            profiles.contains_key(DEFAULT_ENDPOINT_PROFILE_ID),
            "default endpoint profile {DEFAULT_ENDPOINT_PROFILE_ID:?} was not prepared"
        );
        Ok(Rc::new(PreparedRunnerGraphEndpointRuntime {
            table,
            profiles,
            default_profile_id: DEFAULT_ENDPOINT_PROFILE_ID.into(),
            input_token_counter: self.input_token_counter.clone(),
        }))
    }
}

struct PreparedGraphProfile {
    endpoint_id: EndpointId,
    keys: [EndpointKey; 2],
    transport: TransportSink,
    url_count: usize,
}

struct PreparedGraphProfileRuntime {
    endpoint_id: EndpointId,
    keys: [EndpointKey; 2],
    transport: Rc<TransportSink>,
    url_count: usize,
}

struct PreparedRunnerGraphEndpointRuntime {
    table: Rc<PreparedEndpointTable>,
    profiles: BTreeMap<String, PreparedGraphProfileRuntime>,
    default_profile_id: String,
    input_token_counter: Arc<dyn InputTokenCounter>,
}

impl RunnerGraphEndpointRuntime for PreparedRunnerGraphEndpointRuntime {
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
        let payload = Bytes::from(serde_json::to_vec(
            &endpoint.format_payload(&request_info)?,
        )?);
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
    pub(crate) captured: mpsc::UnboundedSender<Vec<CapturedRecord>>,
}

/// Worker-local cancellation construction inputs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GraphCancellationConfig {
    pub(crate) rate: f64,
    pub(crate) delay_seconds: f64,
    /// Phase-local cancellation root; workers derive independent indexed roots.
    pub(crate) rng_root: RngRoot,
    pub(crate) phase: aiperf_timing::Phase,
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

impl GraphTraceExecutionBackendFactory for RunnerGraphBackendFactory {
    fn create_backend(
        &self,
        worker_id: usize,
    ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError> {
        let clock: Rc<dyn Clock> = RealClock::from_anchor(self.config.real_clock_anchor);
        let endpoint_runtime = self
            .config
            .endpoint_runtime_factory
            .prepare_worker(clock.clone(), self.config.run_origin_ns, &self.config.model)
            .map_err(|error| GraphPlacementError(error.to_string()))?;

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
            captured: self.config.captured.clone(),
            node_policy,
            prefill_slots,
            next_session: Cell::new(0),
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
    captured: mpsc::UnboundedSender<Vec<CapturedRecord>>,
    node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
    prefill_slots: Option<Rc<SlotPool>>,
    next_session: Cell<u64>,
}

#[async_trait(?Send)]
impl GraphTraceExecutionBackend for RunnerGraphWorkerBackend {
    async fn execute_trace(&self, plan: GraphTracePlan) -> Result<(), TraceError> {
        let local_session = self.next_session.get();
        self.next_session.set(local_session.saturating_add(1));
        let session_num = (self.worker_id as u64)
            .checked_shl(48)
            .unwrap_or(0)
            .saturating_add(local_session);
        let terminal_nodes = terminal_graph_nodes(&plan);
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
            captured: self.captured.clone(),
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
        local = local.with_node_failure(Rc::new(AbortTraceNodeFailurePolicy));
        let result = local.execute_trace(plan).await;
        sink.verify_finalized_records()
            .map_err(|error| TraceError::Other(error.to_string()))?;
        result
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
    captured: mpsc::UnboundedSender<Vec<CapturedRecord>>,
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
        let raw_messages = messages
            .iter()
            .enumerate()
            .map(|(index, wire)| {
                serde_json::from_slice(wire).with_context(|| {
                    format!("decoding materialized graph message {index} for node {node_id:?}")
                })
            })
            .collect::<Result<Vec<Value>>>()?;
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
            extra_headers: self.raw_string_map(node, "extra_headers_handle")?,
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
            HttpRequestDispatcher::inference_dimensions(transport.as_ref(), &request);
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
        let collected = transport
            .dispatch_prepared_turn_collect_record(
                PreparedHttpTurn {
                    request,
                    model,
                    endpoint,
                    endpoint_aware: true,
                },
                self.observer.as_ref(),
                &|_| on_first_token(),
            )
            .await?;
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
            .snapshot_record(uuid, ordinal)
            .ok_or_else(|| {
                anyhow!(
                    "graph trace {:?} could not snapshot terminal node {node_id:?}",
                    self.trace_id
                )
            })?;
        self.captured
            .send(vec![CapturedRecord {
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
            }])
            .map_err(|_| anyhow!("graph metric capture receiver closed"))?;
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
        let finalized = self.observer.finish_with_records().records.len();
        let emitted = usize::try_from(self.emitted_records.get()).unwrap_or(usize::MAX);
        ensure!(
            finalized == emitted,
            "graph trace {:?} finalized {finalized} metric records after emitting {emitted}",
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
        .filter(|edge| edge.target != aiperf_graph::model::END_NODE_ID)
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

#[cfg(test)]
mod tests {
    use super::*;
    use aiperf::multiturn::AuthoredInputTokenCounter;
    use aiperf_clock::SimClock;
    use aiperf_endpoints::{
        ChatEndpoint, EffectiveEndpointConfig, Endpoint, EndpointDescriptor, EndpointFactory,
        EndpointRegistryBuilder, EndpointResult, PreparedEndpoint, RawEndpointConfig,
        StatelessEndpointFactory,
    };
    use aiperf_transport_http::models::ConnectionReuseStrategy;

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
            Arc::new(vec![crate::registry::ValidatedEndpointProfileV2 {
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
        );
        let runtime = factory
            .prepare_worker(Rc::new(SimClock::new()), 0, "fixture-model")
            .unwrap();
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
        let PreparedHttpEndpoint::Prepared(reference) = dispatch.endpoint else {
            panic!("protocol-v2 graph dispatch must retain a prepared endpoint reference")
        };
        assert_eq!(reference.endpoint_id.as_str(), "chat");
        assert_eq!(reference.key.index(), 1);
    }
}
