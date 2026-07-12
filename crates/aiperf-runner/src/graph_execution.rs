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
use std::sync::{Arc, mpsc};

use aiperf::http::{
    HttpRequest, HttpRequestDispatcher, PreparedHttpTurn, TransportSink, TransportSinkConfig,
};
use aiperf::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use aiperf::multiturn::InputTokenCounter;
use aiperf_clock::{Clock, RealClock, RealClockAnchor};
use aiperf_dataset::{EndpointResolver, Handle, Payload, SegmentStore};
use aiperf_endpoints::{CreditPhase, EndpointConfig, ModelEndpoint, RequestInfo, Turn};
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
use aiperf_rng::RngRoot;
use aiperf_timing::{BernoulliFixedDelay, SlotPool};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use bytes::Bytes;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::records::{CapturedHttpExchange, CapturedRecord};

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

/// Immutable inputs shared by every worker-local graph backend.
pub(crate) struct RunnerGraphBackendFactoryConfig {
    pub(crate) real_clock_anchor: RealClockAnchor,
    pub(crate) run_origin_ns: i64,
    pub(crate) base_urls: Vec<String>,
    pub(crate) model: String,
    pub(crate) default_max_tokens: usize,
    pub(crate) transport: TransportSinkConfig,
    pub(crate) endpoint: EndpointConfig,
    pub(crate) endpoint_resolver: Arc<dyn EndpointResolver>,
    pub(crate) segments: Arc<dyn SegmentStore>,
    pub(crate) input_token_counter: Arc<dyn InputTokenCounter>,
    pub(crate) metrics: MetricsConfig,
    pub(crate) phase: Phase,
    pub(crate) prefill_concurrency: Option<usize>,
    pub(crate) cancellation: Option<GraphCancellationConfig>,
    pub(crate) raw_enabled: bool,
    pub(crate) captured: mpsc::Sender<Vec<CapturedRecord>>,
}

/// Worker-local cancellation construction inputs.
#[derive(Debug, Clone, Copy)]
pub(crate) struct GraphCancellationConfig {
    pub(crate) rate: f64,
    pub(crate) delay_seconds: f64,
    pub(crate) seed: u64,
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
        let transport = TransportSink::new_multi_configured(
            clock.clone(),
            self.config.run_origin_ns,
            &self.config.base_urls,
            self.config.model.clone(),
            self.config.transport.clone(),
        )
        .map_err(|error| GraphPlacementError(error.to_string()))?;

        let mut policies = Vec::<Rc<dyn NodeDispatchPolicy>>::new();
        if let Some(limit) = self.config.prefill_concurrency {
            if limit == 0 {
                return Err(GraphPlacementError(
                    "graph prefill_concurrency must be positive".into(),
                ));
            }
            policies.push(Rc::new(PrefillSlotNodePolicy::new(Rc::new(SlotPool::new(
                limit,
            )))));
        }
        if let Some(cancellation) = self.config.cancellation {
            let seed = cancellation.seed ^ (worker_id as u64).rotate_left(17);
            let policy = BernoulliFixedDelay::new(
                Some(cancellation.rate),
                cancellation.delay_seconds,
                RngRoot::new(Some(seed)),
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
            transport: Rc::new(transport),
            materializer: Rc::new(SegmentItemsMaterializer::new(self.config.segments.clone())),
            endpoint: self.config.endpoint.clone(),
            endpoint_resolver: self.config.endpoint_resolver.clone(),
            segments: self.config.segments.clone(),
            input_token_counter: self.config.input_token_counter.clone(),
            metrics: self.config.metrics.clone(),
            phase: self.config.phase,
            worker_id,
            base_url_count: self.config.base_urls.len(),
            model: self.config.model.clone(),
            default_max_tokens: self.config.default_max_tokens,
            run_origin_ns: self.config.run_origin_ns,
            raw_enabled: self.config.raw_enabled,
            captured: self.config.captured.clone(),
            node_policy,
            next_session: Cell::new(0),
        }))
    }
}

struct RunnerGraphWorkerBackend {
    clock: Rc<dyn Clock>,
    transport: Rc<TransportSink>,
    materializer: Rc<SegmentItemsMaterializer>,
    endpoint: EndpointConfig,
    endpoint_resolver: Arc<dyn EndpointResolver>,
    segments: Arc<dyn SegmentStore>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    metrics: MetricsConfig,
    phase: Phase,
    worker_id: usize,
    base_url_count: usize,
    model: String,
    default_max_tokens: usize,
    run_origin_ns: i64,
    raw_enabled: bool,
    captured: mpsc::Sender<Vec<CapturedRecord>>,
    node_policy: Option<Rc<dyn NodeDispatchPolicy>>,
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
        let url_index = u32::try_from(session_num % self.base_url_count as u64).unwrap_or(0);
        let terminal_nodes = terminal_graph_nodes(&plan);
        let observer = Rc::new(NativeMetricsObserver::new(
            self.clock.clone(),
            self.run_origin_ns,
            self.metrics.clone(),
        ));
        let sink = Rc::new(RunnerGraphSink {
            clock: self.clock.clone(),
            transport: self.transport.clone(),
            trace_id: plan.trace.id.clone(),
            nodes: plan.graph.nodes.clone().into_iter().collect(),
            endpoint: self.endpoint.clone(),
            endpoint_resolver: self.endpoint_resolver.clone(),
            segments: self.segments.clone(),
            input_token_counter: self.input_token_counter.clone(),
            observer,
            phase: self.phase,
            worker_id: self.worker_id,
            session_num,
            url_index,
            model: self.model.clone(),
            default_max_tokens: self.default_max_tokens,
            run_origin_ns: self.run_origin_ns,
            raw_enabled: self.raw_enabled,
            terminal_nodes,
            order: RefCell::new(Vec::new()),
            captures: RefCell::new(HashMap::new()),
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
        let batch = sink
            .finish_records()
            .map_err(|error| TraceError::Other(error.to_string()))?;
        self.captured
            .send(batch)
            .map_err(|_| TraceError::Other("graph metric capture receiver closed".into()))?;
        result
    }
}

struct RunnerGraphSink {
    clock: Rc<dyn Clock>,
    transport: Rc<TransportSink>,
    trace_id: String,
    nodes: HashMap<String, LlmNode>,
    endpoint: EndpointConfig,
    endpoint_resolver: Arc<dyn EndpointResolver>,
    segments: Arc<dyn SegmentStore>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    observer: Rc<NativeMetricsObserver>,
    phase: Phase,
    worker_id: usize,
    session_num: u64,
    url_index: u32,
    model: String,
    default_max_tokens: usize,
    run_origin_ns: i64,
    raw_enabled: bool,
    terminal_nodes: HashSet<String>,
    order: RefCell<Vec<Uuid>>,
    captures: RefCell<HashMap<Uuid, PendingGraphCapture>>,
}

struct PendingGraphCapture {
    response_text: Option<String>,
    raw: Option<CapturedHttpExchange>,
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
        let endpoint_name = metadata_string(node, "endpoint");
        let endpoint = self.endpoint_resolver.resolve(endpoint_name)?;
        let mut endpoint_config = self.endpoint.clone();
        endpoint_config.endpoint_type = endpoint.metadata().endpoint_type;
        endpoint_config.streaming = node.streaming && endpoint.metadata().supports_streaming;
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
        let request_info = RequestInfo {
            model_endpoint: ModelEndpoint {
                primary_model_name: model.clone(),
                endpoint: endpoint_config.clone(),
            },
            turns: vec![turn],
            system_message: None,
            user_context_message: None,
            credit_phase: match self.phase {
                Phase::Warmup => CreditPhase::Warmup,
                Phase::Profiling => CreditPhase::Profiling,
            },
            x_request_id: None,
            x_correlation_id: Some(self.trace_id.clone()),
            conversation_id: Some(self.trace_id.clone()),
        };
        let payload = endpoint.format_payload(&request_info)?;
        let payload = Bytes::from(serde_json::to_vec(&payload)?);
        let authored_input_tokens = metadata_u64(node, "input_tokens").unwrap_or(0);
        let input_tokens = self.input_token_counter.count_input_tokens(
            endpoint.as_ref(),
            &payload,
            authored_input_tokens,
        )?;
        let max_output_tokens = max_tokens.unwrap_or(self.default_max_tokens);
        let uuid = Uuid::new_v4();
        let mut headers = endpoint.format_headers(&endpoint_config);
        headers.extend(self.raw_string_map(node, "extra_headers_handle")?);
        let parameters = self.raw_string_map(node, "request_parameters_handle")?;
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
            uuid,
            input_length: usize::try_from(input_tokens).unwrap_or(usize::MAX),
            max_output_tokens,
            prompt_text: None,
            request_body: None,
            request_body_bytes: Some(payload.clone()),
            headers,
            parameters,
            endpoint_path,
            streaming: endpoint_config.streaming,
            x_correlation_id: Some(self.trace_id.clone()),
            is_final_turn: self.terminal_nodes.contains(node_id),
            cancel_after_ns: options.cancel_after_ns,
            url_index: Some(self.url_index),
        };
        let dimensions: InferenceDimensions =
            HttpRequestDispatcher::inference_dimensions(self.transport.as_ref(), &request);
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
        self.order.borrow_mut().push(uuid);
        self.observer.on_arrival(
            uuid,
            arrival_ms,
            usize::try_from(input_tokens).unwrap_or(usize::MAX),
            max_output_tokens,
        );
        let collected = self
            .transport
            .dispatch_prepared_turn_collect_record(
                PreparedHttpTurn {
                    request,
                    endpoint,
                    endpoint_config,
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
        self.captures.borrow_mut().insert(
            uuid,
            PendingGraphCapture {
                response_text: (!outcome.response_text.is_empty())
                    .then(|| outcome.response_text.clone()),
                raw: self.raw_enabled.then_some(CapturedHttpExchange {
                    request_payload: collected.request_payload.to_vec(),
                    record: collected.record,
                }),
            },
        );

        Ok(match outcome.terminal {
            ReplayTerminalStatus::Completed => GraphReply::from_text(outcome.response_text),
            ReplayTerminalStatus::Canceled => GraphReply::cancelled(),
            ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed => GraphReply::failed(),
        })
    }
}

impl RunnerGraphSink {
    fn finish_records(&self) -> Result<Vec<CapturedRecord>> {
        let records = self.observer.finish_with_records().records;
        let order = self.order.take();
        if records.len() != order.len() {
            return Err(anyhow!(
                "graph trace {:?} finalized {} metric records for {} arrivals",
                self.trace_id,
                records.len(),
                order.len()
            ));
        }
        let mut captures = self.captures.take();
        Ok(records
            .into_iter()
            .zip(order)
            .map(|(ingest, uuid)| {
                let capture = captures.remove(&uuid).unwrap_or(PendingGraphCapture {
                    response_text: None,
                    raw: None,
                });
                CapturedRecord {
                    uuid,
                    x_correlation_id: self.trace_id.clone(),
                    response_text: capture.response_text,
                    raw: capture.raw,
                    ingest,
                }
            })
            .collect())
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
