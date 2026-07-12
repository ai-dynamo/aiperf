// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful agent-harness workload over the ordinary scheduled Rust pipeline.
//!
//! The pinned Python evaluator owns canonical task packages, the agent scaffold,
//! environment/tool execution, trajectories, and verification. It publishes
//! model-call events through [`AgenticEvaluator`]. [`AgenticWorkload`] admits
//! those events, lowers them through [`AgenticTurnBuilder`], and calls
//! [`ScheduledRuntime::issue_turn`]. Consequently model traffic still traverses
//! the same endpoint materializer, [`TurnDispatcher`](crate::scheduled::TurnDispatcher),
//! observer, credit, timing, metrics, and transport path as every other workload.
//! This module contains no HTTP client and makes no benchmark scoring decision.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::rc::Rc;
use std::sync::Arc;

use aiperf_accuracy::{
    AgenticEpisode, AgenticEpisodePage, AgenticEpisodeResult, AgenticEvaluator,
    AgenticEvaluatorEvent, AgenticEvaluatorIdentity, AgenticEvaluatorLoadConfig,
    AgenticInferenceStatus, AgenticModelCall, AgenticModelResult, EpisodeId, ModelCallId,
};
use aiperf_dataset::{
    AccuracyComposer, BuiltinEndpointResolver, ComposeConfig, Composer, ConversationContextMode,
    Dataset, EndpointResolver, RawRow, RowOrigin, SegmentPool, TextTokenizer,
};
use aiperf_endpoints::EndpointConfig;
use aiperf_metrics::{
    AgenticEpisodeReport, AgenticEpisodeReportOutcome, AgenticEvaluationReport,
    AgenticEvaluationSummary, AgenticEvaluatorReportInfo, AgenticRewardSummary,
    AgenticRunConfigReport, EvaluatorDatasetReportInfo, EvaluatorReportInfo, NativeReport,
    ReportError, ReportRunInfo, RunOutcome,
};
use aiperf_rng::RngRoot;
use aiperf_timing::SlotPool;
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use loadgen_core::collector::{ReplayTerminalStatus, TraceSimulationReport};
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::agentic_gateway::{
    AgenticAuxiliaryInferenceRequest, AgenticInferenceGateway, AgenticInferencePurpose,
};
use crate::multiturn::{ConversationSource, NativeDatasetConversationSource, TurnToSend};
use crate::scheduled::{ScheduledRuntime, TurnDispatchOutcome, Workload};

const EPISODE_PAGE_SIZE: usize = 256;
const EVENT_BATCH_SIZE: usize = 256;
const EVENT_WAIT_MS: u64 = 25;

/// Lower one evaluator-authored model call into an ordinary scheduled turn.
///
/// A future harness may choose another dataset format or endpoint dialect by
/// implementing this trait. Builders prepare requests only; dispatch remains
/// exclusively owned by [`ScheduledRuntime`].
pub trait AgenticTurnBuilder {
    /// Build one normal transport-ready turn without sending it.
    fn build_turn(&self, call: &AgenticModelCall) -> Result<TurnToSend>;
}

/// Unified-dataset implementation of [`AgenticTurnBuilder`].
///
/// Each dynamic model call becomes a one-turn conversation with the opaque call
/// ID as its request correlation and the opaque episode ID as its runtime
/// session. Generation controls and tool schemas are carried in the normal
/// endpoint `extra_body` merge path.
pub struct DatasetAgenticTurnBuilder {
    model: String,
    tokenizer: Arc<dyn TextTokenizer>,
    endpoint_config: EndpointConfig,
    endpoint_resolver: Arc<dyn EndpointResolver>,
}

impl DatasetAgenticTurnBuilder {
    /// Build the standard streaming Chat Completions lowerer.
    pub fn chat(model: impl Into<String>, tokenizer: Arc<dyn TextTokenizer>) -> Result<Self> {
        Self::new(
            model,
            tokenizer,
            EndpointConfig {
                streaming: true,
                use_server_token_count: true,
                ..EndpointConfig::default()
            },
            Arc::new(BuiltinEndpointResolver::default()),
        )
    }

    /// Build a lowerer with injected endpoint policy and registry.
    pub fn new(
        model: impl Into<String>,
        tokenizer: Arc<dyn TextTokenizer>,
        endpoint_config: EndpointConfig,
        endpoint_resolver: Arc<dyn EndpointResolver>,
    ) -> Result<Self> {
        let model = model.into();
        ensure!(!model.trim().is_empty(), "agentic model must not be empty");
        Ok(Self {
            model,
            tokenizer,
            endpoint_config: endpoint_config.validate()?,
            endpoint_resolver,
        })
    }
}

impl AgenticTurnBuilder for DatasetAgenticTurnBuilder {
    fn build_turn(&self, call: &AgenticModelCall) -> Result<TurnToSend> {
        let mut segments = SegmentPool::new();
        let row = agentic_call_row(call)?;
        let model = call.model.as_deref().unwrap_or(self.model.as_str());
        let config = ComposeConfig::new(model, RngRoot::new(Some(0)));
        let conversations =
            AccuracyComposer.compose(vec![row], &config, self.tokenizer.as_ref(), &mut segments)?;
        let dataset = Dataset::new(
            conversations,
            Arc::new(segments.freeze()),
            "sequential",
            ConversationContextMode::MessageArrayWithResponses,
        )?;
        let source = NativeDatasetConversationSource::sequential_with_endpoint_config_and_resolver(
            dataset,
            &self.model,
            call.generation.max_tokens,
            self.endpoint_config.clone(),
            self.endpoint_resolver.clone(),
        )?;
        let session =
            source.session_for(call.call_id.as_str(), call.episode_id.as_str().to_string())?;
        let turn = session.build_first_turn(Some(1))?;
        ensure!(
            turn.x_correlation_id == call.episode_id.as_str(),
            "agentic turn lost episode correlation {:?}",
            call.episode_id.as_str()
        );
        ensure!(
            turn.request_correlation_id == call.call_id.as_str(),
            "agentic turn lost model-call correlation {:?}",
            call.call_id.as_str()
        );
        Ok(turn)
    }
}

fn agentic_call_row(call: &AgenticModelCall) -> Result<RawRow> {
    ensure!(
        !call.messages.is_empty() || !call.prompt.trim().is_empty(),
        "agentic call {:?} has neither prompt nor messages",
        call.call_id.as_str()
    );
    ensure!(
        call.generation.max_tokens > 0,
        "agentic call {:?} max_tokens must be positive",
        call.call_id.as_str()
    );
    ensure!(
        call.generation.temperature.is_finite() && call.generation.temperature >= 0.0,
        "agentic call {:?} temperature must be finite and non-negative",
        call.call_id.as_str()
    );
    ensure!(
        call.generation.top_p.is_finite() && (0.0..=1.0).contains(&call.generation.top_p),
        "agentic call {:?} top_p must be in [0, 1]",
        call.call_id.as_str()
    );
    if let Some(model) = &call.model {
        ensure!(
            !model.trim().is_empty(),
            "agentic call {:?} model must not be empty",
            call.call_id.as_str()
        );
    }
    let messages = serde_json::to_value(&call.messages)
        .context("serializing evaluator-authored agent messages")?;
    let prompt = if call.prompt.trim().is_empty() {
        serde_json::to_string(&messages).context("serializing agent messages for accounting")?
    } else {
        call.prompt.clone()
    };
    let mut extra = call.extra_body.clone();
    for reserved in [
        "messages",
        "model",
        "stream",
        "stream_options",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
    ] {
        ensure!(
            !extra.contains_key(reserved),
            "agentic call {:?} extra_body must not override reserved field {reserved:?}",
            call.call_id.as_str()
        );
    }
    extra.insert("temperature".into(), json!(call.generation.temperature));
    extra.insert("top_p".into(), json!(call.generation.top_p));
    extra.insert("stop".into(), json!(call.generation.stop));
    if !call.tools.is_empty() {
        extra.insert("tools".into(), Value::Array(call.tools.clone()));
    }
    if let Some(tool_choice) = &call.tool_choice {
        extra.insert("tool_choice".into(), tool_choice.clone());
    }
    if let Some(response_format) = &call.response_format {
        extra.insert("response_format".into(), response_format.clone());
    }
    Ok(RawRow {
        value: json!({
            "prompt": prompt,
            "task": call.episode_id.as_str(),
            "session_id": call.call_id.as_str(),
            "correlation_id": call.call_id.as_str(),
            "raw_messages": messages,
            "metadata": {"generation_size": call.generation.max_tokens},
            "extra_body": extra,
        }),
        wire: None,
        session_id: None,
        group_key: None,
        origin: RowOrigin::Inline {
            index: call.turn_index,
        },
    })
}

#[derive(Debug, Default)]
struct OptionalTokenTotal {
    total: u64,
    missing: bool,
}

impl OptionalTokenTotal {
    fn record(&mut self, value: Option<u64>, field: &str) -> Result<()> {
        let Some(value) = value else {
            self.missing = true;
            return Ok(());
        };
        self.total = self
            .total
            .checked_add(value)
            .ok_or_else(|| anyhow!("agentic {field} token total overflowed u64"))?;
        Ok(())
    }

    fn value(&self, calls: usize) -> Option<u64> {
        (calls > 0 && !self.missing).then_some(self.total)
    }
}

#[derive(Debug, Default)]
struct InferenceClassStats {
    calls: usize,
    prompt_tokens: OptionalTokenTotal,
    completion_tokens: OptionalTokenTotal,
    cached_tokens: OptionalTokenTotal,
}

impl InferenceClassStats {
    fn record(&mut self, result: &AgenticModelResult) -> Result<()> {
        self.calls = self
            .calls
            .checked_add(1)
            .ok_or_else(|| anyhow!("agentic model-call count overflowed usize"))?;
        self.prompt_tokens.record(result.prompt_tokens, "prompt")?;
        self.completion_tokens
            .record(result.completion_tokens, "completion")?;
        self.cached_tokens.record(result.cached_tokens, "cached")?;
        Ok(())
    }
}

#[derive(Debug, Default)]
struct EpisodeInferenceStats {
    primary: InferenceClassStats,
    auxiliary: InferenceClassStats,
    environment_calls: usize,
    verifier_calls: usize,
}

impl EpisodeInferenceStats {
    fn record_primary(&mut self, result: &AgenticModelResult) -> Result<()> {
        self.primary.record(result)
    }

    fn record_auxiliary(
        &mut self,
        purpose: AgenticInferencePurpose,
        result: &AgenticModelResult,
    ) -> Result<()> {
        self.auxiliary.record(result)?;
        let count = match purpose {
            AgenticInferencePurpose::Environment => &mut self.environment_calls,
            AgenticInferencePurpose::Verifier => &mut self.verifier_calls,
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| anyhow!("agentic auxiliary call count overflowed usize"))?;
        Ok(())
    }
}

enum OutstandingAgenticCall {
    Primary { episode_id: EpisodeId },
    Auxiliary(Box<AgenticAuxiliaryInferenceRequest>),
}

impl OutstandingAgenticCall {
    fn episode_id(&self) -> &EpisodeId {
        match self {
            Self::Primary { episode_id } => episode_id,
            Self::Auxiliary(request) => &request.call.episode_id,
        }
    }
}

/// Prepared stateful evaluator workload.
///
/// One instance is single-use. Task admission and inference admission are
/// separate Rust-owned limits: `task_concurrency` controls active evaluator
/// environments, while `model_concurrency` gates ordinary model turns.
pub struct AgenticWorkload {
    evaluator: RefCell<Option<Box<dyn AgenticEvaluator>>>,
    identity: AgenticEvaluatorIdentity,
    config: AgenticEvaluatorLoadConfig,
    episodes: Vec<AgenticEpisode>,
    task_concurrency: usize,
    model_slots: Rc<SlotPool>,
    turn_builder: Rc<dyn AgenticTurnBuilder>,
    inference_gateway: RefCell<Option<Box<dyn AgenticInferenceGateway>>>,
    auxiliary_requests: RefCell<Option<mpsc::UnboundedReceiver<AgenticAuxiliaryInferenceRequest>>>,
    active_episode_ids: RefCell<BTreeSet<EpisodeId>>,
    results: RefCell<Option<Vec<AgenticEpisodeResult>>>,
    executed: Cell<bool>,
}

impl AgenticWorkload {
    /// Resolve and freeze the canonical dataset before the measurement clock starts.
    #[allow(clippy::too_many_arguments)]
    pub async fn prepare(
        evaluator: Box<dyn AgenticEvaluator>,
        dataset: &str,
        model: &str,
        config: &AgenticEvaluatorLoadConfig,
        model_concurrency: usize,
        turn_builder: Rc<dyn AgenticTurnBuilder>,
    ) -> Result<Rc<Self>> {
        Self::prepare_with_gateway(
            evaluator,
            dataset,
            model,
            config,
            model_concurrency,
            turn_builder,
            None,
        )
        .await
    }

    /// Resolve tasks with an optional Rust-owned auxiliary inference ingress.
    #[allow(clippy::too_many_arguments)]
    pub async fn prepare_with_gateway(
        mut evaluator: Box<dyn AgenticEvaluator>,
        dataset: &str,
        model: &str,
        config: &AgenticEvaluatorLoadConfig,
        model_concurrency: usize,
        turn_builder: Rc<dyn AgenticTurnBuilder>,
        mut inference_gateway: Option<Box<dyn AgenticInferenceGateway>>,
    ) -> Result<Rc<Self>> {
        ensure!(
            model_concurrency > 0,
            "agentic model concurrency must be positive"
        );
        ensure!(
            config.task_concurrency > 0,
            "agentic task concurrency must be positive"
        );
        ensure!(
            inference_gateway.is_none() || evaluator.supports_agentic_inference_gateway(),
            "canonical evaluator does not advertise agentic_inference_gateway support"
        );
        let mut effective_config = config.clone();
        let auxiliary_requests = if let Some(gateway) = inference_gateway.as_mut() {
            effective_config.inference_gateway = Some(gateway.evaluator_config().clone());
            Some(gateway.take_requests()?)
        } else {
            None
        };
        let identity = evaluator
            .load_agentic(dataset, model, &effective_config)
            .await
            .with_context(|| format!("canonical agentic evaluator failed to load {dataset:?}"))?;
        let episodes = load_agentic_episodes(evaluator.as_mut(), identity.episode_count).await?;
        Ok(Rc::new(Self {
            evaluator: RefCell::new(Some(evaluator)),
            identity,
            config: effective_config,
            episodes,
            task_concurrency: config.task_concurrency,
            model_slots: Rc::new(SlotPool::new(model_concurrency)),
            turn_builder,
            inference_gateway: RefCell::new(inference_gateway),
            auxiliary_requests: RefCell::new(auxiliary_requests),
            active_episode_ids: RefCell::new(BTreeSet::new()),
            results: RefCell::new(None),
            executed: Cell::new(false),
        }))
    }

    /// Frozen harness, package, dataset, agent, environment, and verifier identity.
    pub fn identity(&self) -> &AgenticEvaluatorIdentity {
        &self.identity
    }

    /// Effective evaluator configuration, including the Rust callback ingress.
    pub fn config(&self) -> &AgenticEvaluatorLoadConfig {
        &self.config
    }

    /// Frozen ordered task descriptors selected by the evaluator.
    pub fn episodes(&self) -> &[AgenticEpisode] {
        &self.episodes
    }

    /// Return canonical results after successful workload execution.
    pub fn results(&self) -> Result<Vec<AgenticEpisodeResult>> {
        self.results
            .borrow()
            .clone()
            .ok_or_else(|| anyhow!("agentic workload has not completed successfully"))
    }

    /// Shut down the supervised evaluator worker after report construction.
    pub async fn shutdown(&self) -> Result<()> {
        let mut evaluator = self
            .evaluator
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("agentic evaluator is unavailable or already shut down"))?;
        let evaluator_result = evaluator
            .shutdown()
            .await
            .context("shutting down canonical agentic evaluator");
        let mut gateway = self.inference_gateway.borrow_mut().take();
        if let Some(gateway) = gateway.as_mut() {
            gateway.shutdown().await?;
        }
        evaluator_result
    }

    async fn execute_inner(
        &self,
        runtime: Rc<ScheduledRuntime>,
        evaluator: &mut dyn AgenticEvaluator,
    ) -> Result<Vec<AgenticEpisodeResult>> {
        let episode_by_id = self
            .episodes
            .iter()
            .cloned()
            .map(|episode| (episode.episode_id.clone(), episode))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            episode_by_id.len() == self.episodes.len(),
            "canonical agentic evaluator returned duplicate episode IDs"
        );
        let mut pending = self
            .episodes
            .iter()
            .map(|episode| episode.episode_id.clone())
            .collect::<VecDeque<_>>();
        let mut results = BTreeMap::<EpisodeId, AgenticEpisodeResult>::new();
        let mut inference_stats = episode_by_id
            .keys()
            .cloned()
            .map(|episode_id| (episode_id, EpisodeInferenceStats::default()))
            .collect::<BTreeMap<_, _>>();
        let mut outstanding_calls = BTreeMap::<ModelCallId, OutstandingAgenticCall>::new();
        let mut next_turn = BTreeMap::<EpisodeId, usize>::new();
        let mut cancelled = BTreeSet::<EpisodeId>::new();
        let mut auxiliary_requests = self.auxiliary_requests.borrow_mut().take();
        let (completion_tx, mut completion_rx) = mpsc::unbounded_channel::<AgenticModelResult>();

        self.start_available(evaluator, &mut pending).await?;
        while results.len() < self.episodes.len() {
            let mut completed_calls = Vec::new();
            while let Ok(item) = completion_rx.try_recv() {
                let route = outstanding_calls.remove(&item.call_id).ok_or_else(|| {
                    anyhow!(
                        "normal pipeline completed unknown agentic call {:?}",
                        item.call_id.as_str()
                    )
                })?;
                ensure!(
                    route.episode_id() == &item.episode_id,
                    "normal pipeline call {:?} changed episode {:?} to {:?}",
                    item.call_id.as_str(),
                    route.episode_id().as_str(),
                    item.episode_id.as_str()
                );
                let episode_label = item.episode_id.as_str().to_string();
                let stats = inference_stats.get_mut(&item.episode_id).ok_or_else(|| {
                    anyhow!(
                        "normal pipeline completed call for unknown episode {:?}",
                        item.episode_id.as_str()
                    )
                })?;
                match route {
                    OutstandingAgenticCall::Primary { .. } => {
                        stats.record_primary(&item)?;
                        completed_calls.push(item);
                    }
                    OutstandingAgenticCall::Auxiliary(request) => {
                        stats.record_auxiliary(request.purpose, &item)?;
                        (*request).respond(item).with_context(|| {
                            format!(
                                "returning auxiliary inference result for episode {:?}",
                                episode_label
                            )
                        })?;
                    }
                }
            }
            if !completed_calls.is_empty() {
                evaluator
                    .submit_model_results(&completed_calls)
                    .await
                    .context("submitting normal Rust inference results to agent harness")?;
            }

            loop {
                let request = match auxiliary_requests.as_mut() {
                    None => break,
                    Some(receiver) => match receiver.try_recv() {
                        Ok(request) => request,
                        Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                        Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                            return Err(anyhow!(
                                "Rust agentic inference gateway stopped during an active run"
                            ));
                        }
                    },
                };
                ensure!(
                    self.active_episode_ids
                        .borrow()
                        .contains(&request.call.episode_id),
                    "auxiliary call {:?} belongs to inactive episode {:?}",
                    request.call.call_id.as_str(),
                    request.call.episode_id.as_str()
                );
                ensure!(
                    !cancelled.contains(&request.call.episode_id),
                    "cancelled episode {:?} requested auxiliary inference",
                    request.call.episode_id.as_str()
                );
                let call = request.call.clone();
                let rejected = self
                    .issue_model_call(
                        &runtime,
                        &call,
                        OutstandingAgenticCall::Auxiliary(Box::new(request)),
                        &mut outstanding_calls,
                        &completion_tx,
                    )
                    .await?;
                if let Some(OutstandingAgenticCall::Auxiliary(request)) = rejected {
                    let episode_id = request.call.episode_id.clone();
                    (*request).respond(rejected_model_result(&call))?;
                    cancelled.insert(episode_id.clone());
                    evaluator
                        .cancel_episodes(std::slice::from_ref(&episode_id))
                        .await
                        .with_context(|| {
                            format!(
                                "cancelling episode {:?} after auxiliary issuance rejection",
                                episode_id.as_str()
                            )
                        })?;
                }
            }

            let events = evaluator
                .poll_agentic(EVENT_BATCH_SIZE, EVENT_WAIT_MS)
                .await
                .context("polling canonical agentic evaluator")?;
            if events.events.is_empty() {
                ensure!(
                    !self.active_episode_ids.borrow().is_empty()
                        || !pending.is_empty()
                        || !outstanding_calls.is_empty(),
                    "agentic evaluator made no progress with no active work"
                );
                // A fixture or in-process harness may complete its bounded poll
                // without touching the reactor. Preserve LocalSet fairness so
                // ordinary dispatch tasks can advance and publish completions.
                tokio::task::yield_now().await;
                continue;
            }
            for event in events.events {
                match event {
                    AgenticEvaluatorEvent::ModelCall { call } => {
                        ensure!(
                            self.active_episode_ids.borrow().contains(&call.episode_id),
                            "agentic call {:?} belongs to inactive episode {:?}",
                            call.call_id.as_str(),
                            call.episode_id.as_str()
                        );
                        ensure!(
                            !cancelled.contains(&call.episode_id),
                            "cancelled episode {:?} emitted another model call",
                            call.episode_id.as_str()
                        );
                        let expected_turn = next_turn.entry(call.episode_id.clone()).or_default();
                        ensure!(
                            call.turn_index == *expected_turn,
                            "agentic episode {:?} emitted turn {} but {} was expected",
                            call.episode_id.as_str(),
                            call.turn_index,
                            *expected_turn
                        );
                        *expected_turn += 1;
                        ensure!(
                            !outstanding_calls.contains_key(&call.call_id),
                            "duplicate outstanding agentic call {:?}",
                            call.call_id.as_str()
                        );

                        let rejected = self
                            .issue_model_call(
                                &runtime,
                                &call,
                                OutstandingAgenticCall::Primary {
                                    episode_id: call.episode_id.clone(),
                                },
                                &mut outstanding_calls,
                                &completion_tx,
                            )
                            .await?;
                        if rejected.is_some() {
                            cancelled.insert(call.episode_id.clone());
                            evaluator
                                .cancel_episodes(std::slice::from_ref(&call.episode_id))
                                .await
                                .with_context(|| {
                                    format!(
                                        "cancelling episode {:?} after Rust issuance rejection",
                                        call.episode_id.as_str()
                                    )
                                })?;
                        }
                    }
                    AgenticEvaluatorEvent::EpisodeCompleted { result } => {
                        let descriptor =
                            episode_by_id.get(&result.episode_id).ok_or_else(|| {
                                anyhow!(
                                    "agentic evaluator completed unknown episode {:?}",
                                    result.episode_id.as_str()
                                )
                            })?;
                        ensure!(
                            descriptor.task == result.task,
                            "agentic episode {:?} changed task {:?} to {:?}",
                            result.episode_id.as_str(),
                            descriptor.task,
                            result.task
                        );
                        ensure!(
                            self.active_episode_ids
                                .borrow_mut()
                                .remove(&result.episode_id),
                            "agentic episode {:?} completed while inactive",
                            result.episode_id.as_str()
                        );
                        ensure!(
                            !outstanding_calls
                                .values()
                                .any(|call| call.episode_id() == &result.episode_id),
                            "agentic episode {:?} completed with an inference call outstanding",
                            result.episode_id.as_str()
                        );
                        ensure!(
                            results.insert(result.episode_id.clone(), result).is_none(),
                            "agentic evaluator completed an episode more than once"
                        );
                    }
                }
            }
            self.start_available(evaluator, &mut pending).await?;
            tokio::task::yield_now().await;
        }

        ensure!(
            pending.is_empty(),
            "agentic run ended with unstarted episodes"
        );
        ensure!(
            self.active_episode_ids.borrow().is_empty(),
            "agentic run ended with active episodes"
        );
        ensure!(
            outstanding_calls.is_empty(),
            "agentic run ended with model calls outstanding"
        );
        if let Some(receiver) = auxiliary_requests.as_mut() {
            match receiver.try_recv() {
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    return Err(anyhow!(
                        "Rust agentic inference gateway stopped before run finalization"
                    ));
                }
                Ok(request) => {
                    let call_id = request.call.call_id.clone();
                    let rejection = rejected_model_result(&request.call);
                    request.respond(rejection)?;
                    return Err(anyhow!(
                        "auxiliary call {:?} arrived after all episodes completed",
                        call_id.as_str()
                    ));
                }
            }
        }
        let mut ordered = self
            .episodes
            .iter()
            .map(|episode| {
                results.remove(&episode.episode_id).ok_or_else(|| {
                    anyhow!(
                        "missing canonical result for episode {:?}",
                        episode.episode_id.as_str()
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let finished = evaluator
            .finish_agentic()
            .await
            .context("finalizing canonical agentic evaluator")?;
        ensure!(
            finished.items == ordered,
            "finish_agentic results differed from terminal event results"
        );
        reconcile_inference_stats(&mut ordered, &inference_stats)?;
        Ok(ordered)
    }

    async fn start_available(
        &self,
        evaluator: &mut dyn AgenticEvaluator,
        pending: &mut VecDeque<EpisodeId>,
    ) -> Result<()> {
        let available = self
            .task_concurrency
            .saturating_sub(self.active_episode_ids.borrow().len());
        let mut starting = Vec::with_capacity(available.min(pending.len()));
        for _ in 0..available {
            let Some(episode_id) = pending.pop_front() else {
                break;
            };
            starting.push(episode_id);
        }
        if starting.is_empty() {
            return Ok(());
        }
        evaluator
            .start_episodes(&starting)
            .await
            .context("starting canonical agentic episodes")?;
        let mut active = self.active_episode_ids.borrow_mut();
        for episode_id in starting {
            ensure!(
                active.insert(episode_id.clone()),
                "agentic episode {:?} was started twice",
                episode_id.as_str()
            );
        }
        Ok(())
    }

    async fn issue_model_call(
        &self,
        runtime: &Rc<ScheduledRuntime>,
        call: &AgenticModelCall,
        route: OutstandingAgenticCall,
        outstanding: &mut BTreeMap<ModelCallId, OutstandingAgenticCall>,
        completion_tx: &mpsc::UnboundedSender<AgenticModelResult>,
    ) -> Result<Option<OutstandingAgenticCall>> {
        ensure!(
            route.episode_id() == &call.episode_id,
            "agentic completion route changed call {:?} episode identity",
            call.call_id.as_str()
        );
        ensure!(
            !outstanding.contains_key(&call.call_id),
            "duplicate outstanding agentic call {:?}",
            call.call_id.as_str()
        );
        let model_guard = self.model_slots.acquire().await;
        let turn = self.turn_builder.build_turn(call).with_context(|| {
            format!(
                "lowering evaluator model call {:?} into normal dataset turn",
                call.call_id.as_str()
            )
        })?;
        let call_id = call.call_id.clone();
        let episode_id = call.episode_id.clone();
        ensure!(
            outstanding.insert(call_id.clone(), route).is_none(),
            "duplicate outstanding agentic call {:?}",
            call_id.as_str()
        );
        let completion_tx = completion_tx.clone();
        let issued = runtime.issue_turn(
            turn,
            runtime.now_ns(),
            None,
            Box::new(move |_credit, outcome| {
                Box::pin(async move {
                    drop(model_guard);
                    let item = model_result(episode_id, call_id, outcome);
                    if completion_tx.send(item).is_err() {
                        tracing::error!(
                            "agentic workload dropped its inference completion channel"
                        );
                    }
                })
            }),
        );
        if issued {
            Ok(None)
        } else {
            Ok(outstanding.remove(&call.call_id))
        }
    }
}

#[async_trait(?Send)]
impl Workload for AgenticWorkload {
    fn name(&self) -> &'static str {
        "agentic"
    }

    async fn execute(&self, runtime: Rc<ScheduledRuntime>) -> Result<()> {
        ensure!(
            !self.executed.replace(true),
            "agentic workload instances are single-use"
        );
        let mut evaluator = self
            .evaluator
            .borrow_mut()
            .take()
            .ok_or_else(|| anyhow!("agentic evaluator is unavailable"))?;
        let result = self.execute_inner(runtime, evaluator.as_mut()).await;
        if result.is_err() {
            let active = self
                .active_episode_ids
                .borrow()
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            if !active.is_empty()
                && let Err(error) = evaluator.cancel_episodes(&active).await
            {
                tracing::warn!(error = %error, "failed to cancel agentic episodes after run error");
            }
            self.active_episode_ids.borrow_mut().clear();
        }
        *self.evaluator.borrow_mut() = Some(evaluator);
        let results = result?;
        *self.results.borrow_mut() = Some(results);
        Ok(())
    }
}

/// Strictly retrieve every opaque episode in evaluator order.
pub async fn load_agentic_episodes(
    evaluator: &mut dyn AgenticEvaluator,
    episode_count: usize,
) -> Result<Vec<AgenticEpisode>> {
    ensure!(
        episode_count > 0,
        "agentic evaluator selected zero episodes"
    );
    let mut episodes = Vec::with_capacity(episode_count);
    let mut ids = BTreeSet::new();
    let mut offset = 0;
    loop {
        let AgenticEpisodePage {
            items,
            next_offset,
            done,
        } = evaluator
            .next_episodes(offset, EPISODE_PAGE_SIZE)
            .await
            .with_context(|| format!("canonical agentic episode page at offset {offset}"))?;
        ensure!(
            !items.is_empty() || done,
            "canonical agentic evaluator returned an empty non-terminal page at offset {offset}"
        );
        let expected_next = offset
            .checked_add(items.len())
            .ok_or_else(|| anyhow!("agentic episode offset overflow"))?;
        ensure!(
            next_offset == expected_next,
            "canonical agentic evaluator advanced offset {offset} to {next_offset}, expected {expected_next}"
        );
        for episode in items {
            ensure!(
                ids.insert(episode.episode_id.clone()),
                "canonical agentic evaluator returned duplicate episode_id {:?}",
                episode.episode_id.as_str()
            );
            episodes.push(episode);
        }
        ensure!(
            episodes.len() <= episode_count,
            "canonical agentic evaluator returned more than declared {episode_count} episodes"
        );
        offset = next_offset;
        if done {
            break;
        }
        ensure!(
            episodes.len() < episode_count,
            "canonical agentic evaluator did not terminate after its declared episode count"
        );
    }
    ensure!(
        episodes.len() == episode_count,
        "canonical agentic evaluator declared {episode_count} episodes but returned {}",
        episodes.len()
    );
    Ok(episodes)
}

fn model_result(
    episode_id: EpisodeId,
    call_id: ModelCallId,
    outcome: TurnDispatchOutcome,
) -> AgenticModelResult {
    let status = match outcome.terminal {
        ReplayTerminalStatus::Completed => AgenticInferenceStatus::Completed,
        ReplayTerminalStatus::Canceled => AgenticInferenceStatus::Cancelled,
        ReplayTerminalStatus::Rejected | ReplayTerminalStatus::Failed => {
            AgenticInferenceStatus::Failed
        }
    };
    let completed = status == AgenticInferenceStatus::Completed;
    let response = outcome
        .model_response
        .content
        .clone()
        .unwrap_or_else(|| outcome.response_text.clone());
    let error_kind = (!completed).then(|| {
        outcome
            .model_response
            .error_kind
            .clone()
            .unwrap_or_else(|| {
                match outcome.terminal {
                    ReplayTerminalStatus::Canceled => "cancelled",
                    ReplayTerminalStatus::Rejected => "dispatch_rejected",
                    ReplayTerminalStatus::Failed => "transport_failure",
                    ReplayTerminalStatus::Completed => unreachable!("completed handled above"),
                }
                .to_string()
            })
    });
    let error_message = (!completed)
        .then(|| outcome.model_response.error_message.clone())
        .flatten();
    AgenticModelResult {
        episode_id,
        call_id,
        status,
        response,
        reasoning: outcome.model_response.reasoning,
        prompt_tokens: outcome.prompt_tokens,
        completion_tokens: outcome.completion_tokens,
        cached_tokens: outcome.model_response.cached_prompt_tokens,
        response_id: outcome.model_response.response_id,
        finish_reason: outcome.model_response.finish_reason,
        assistant_message: outcome.model_response.assistant_message,
        error_kind,
        error_message,
    }
}

fn rejected_model_result(call: &AgenticModelCall) -> AgenticModelResult {
    AgenticModelResult {
        episode_id: call.episode_id.clone(),
        call_id: call.call_id.clone(),
        status: AgenticInferenceStatus::Failed,
        response: String::new(),
        reasoning: None,
        prompt_tokens: None,
        completion_tokens: None,
        cached_tokens: None,
        response_id: None,
        finish_reason: None,
        assistant_message: None,
        error_kind: Some("dispatch_rejected".to_string()),
        error_message: Some("Rust scheduling policy rejected the model call".to_string()),
    }
}

fn reconcile_inference_stats(
    results: &mut [AgenticEpisodeResult],
    stats_by_episode: &BTreeMap<EpisodeId, EpisodeInferenceStats>,
) -> Result<()> {
    for result in results {
        let stats = stats_by_episode.get(&result.episode_id).ok_or_else(|| {
            anyhow!(
                "missing Rust inference statistics for episode {:?}",
                result.episode_id.as_str()
            )
        })?;
        ensure!(
            result.primary_model_calls == 0
                && result.auxiliary_model_calls == 0
                && result.environment_model_calls == 0
                && result.verifier_model_calls == 0
                && result.primary_prompt_tokens.is_none()
                && result.primary_completion_tokens.is_none()
                && result.primary_cached_tokens.is_none()
                && result.auxiliary_prompt_tokens.is_none()
                && result.auxiliary_completion_tokens.is_none()
                && result.auxiliary_cached_tokens.is_none(),
            "canonical evaluator attempted to author Rust-owned inference accounting for episode {:?}",
            result.episode_id.as_str()
        );
        ensure!(
            result.model_calls == stats.primary.calls,
            "canonical evaluator reported {} primary calls for episode {:?}, but Rust dispatched {}",
            result.model_calls,
            result.episode_id.as_str(),
            stats.primary.calls
        );

        let primary_prompt = stats.primary.prompt_tokens.value(stats.primary.calls);
        let primary_completion = stats.primary.completion_tokens.value(stats.primary.calls);
        let primary_cached = stats.primary.cached_tokens.value(stats.primary.calls);
        validate_canonical_token_total(result, "prompt", result.prompt_tokens, primary_prompt)?;
        validate_canonical_token_total(
            result,
            "completion",
            result.completion_tokens,
            primary_completion,
        )?;
        validate_canonical_token_total(result, "cached", result.cached_tokens, primary_cached)?;

        let auxiliary_prompt = stats.auxiliary.prompt_tokens.value(stats.auxiliary.calls);
        let auxiliary_completion = stats
            .auxiliary
            .completion_tokens
            .value(stats.auxiliary.calls);
        let auxiliary_cached = stats.auxiliary.cached_tokens.value(stats.auxiliary.calls);
        ensure!(
            stats
                .environment_calls
                .checked_add(stats.verifier_calls)
                .is_some_and(|total| total == stats.auxiliary.calls),
            "auxiliary purpose accounting diverged for episode {:?}",
            result.episode_id.as_str()
        );
        let model_calls = stats
            .primary
            .calls
            .checked_add(stats.auxiliary.calls)
            .ok_or_else(|| anyhow!("agentic model-call count overflowed usize"))?;
        result.model_calls = model_calls;
        result.primary_model_calls = stats.primary.calls;
        result.auxiliary_model_calls = stats.auxiliary.calls;
        result.environment_model_calls = stats.environment_calls;
        result.verifier_model_calls = stats.verifier_calls;
        result.primary_prompt_tokens = primary_prompt;
        result.primary_completion_tokens = primary_completion;
        result.primary_cached_tokens = primary_cached;
        result.auxiliary_prompt_tokens = auxiliary_prompt;
        result.auxiliary_completion_tokens = auxiliary_completion;
        result.auxiliary_cached_tokens = auxiliary_cached;
        result.prompt_tokens = combine_token_totals(
            stats.primary.calls,
            primary_prompt,
            stats.auxiliary.calls,
            auxiliary_prompt,
            "prompt",
        )?;
        result.completion_tokens = combine_token_totals(
            stats.primary.calls,
            primary_completion,
            stats.auxiliary.calls,
            auxiliary_completion,
            "completion",
        )?;
        result.cached_tokens = combine_token_totals(
            stats.primary.calls,
            primary_cached,
            stats.auxiliary.calls,
            auxiliary_cached,
            "cached",
        )?;
    }
    Ok(())
}

fn validate_canonical_token_total(
    result: &AgenticEpisodeResult,
    name: &str,
    canonical: Option<u64>,
    observed: Option<u64>,
) -> Result<()> {
    match (canonical, observed) {
        (None, _) | (Some(0), None) => Ok(()),
        (Some(canonical), Some(observed)) => {
            ensure!(
                canonical == observed,
                "canonical evaluator reported {canonical} primary {name} tokens for episode {:?}, but Rust observed {observed}",
                result.episode_id.as_str()
            );
            Ok(())
        }
        (Some(canonical), None) => Err(anyhow!(
            "canonical evaluator reported {canonical} primary {name} tokens for episode {:?}, but Rust received incomplete usage",
            result.episode_id.as_str()
        )),
    }
}

fn combine_token_totals(
    primary_calls: usize,
    primary: Option<u64>,
    auxiliary_calls: usize,
    auxiliary: Option<u64>,
    name: &str,
) -> Result<Option<u64>> {
    if primary_calls == 0 && auxiliary_calls == 0 {
        return Ok(None);
    }
    let primary = if primary_calls == 0 {
        0
    } else if let Some(value) = primary {
        value
    } else {
        return Ok(None);
    };
    let auxiliary = if auxiliary_calls == 0 {
        0
    } else if let Some(value) = auxiliary {
        value
    } else {
        return Ok(None);
    };
    primary
        .checked_add(auxiliary)
        .map(Some)
        .ok_or_else(|| anyhow!("agentic aggregate {name} token total overflowed u64"))
}

/// Combined performance and canonical agentic-evaluation result.
#[derive(Debug)]
pub struct AgenticRunReport {
    /// Requested Harbor Hub package, legacy dataset, or local task directory.
    pub dataset: String,
    /// Target model name.
    pub model: String,
    /// Exact Python worker and dependency identity.
    pub worker: aiperf_accuracy::EvaluatorIdentity,
    /// Frozen Harbor, dataset, agent, environment, and verifier identity.
    pub evaluator: AgenticEvaluatorIdentity,
    /// Standard performance report over every Rust-owned model call.
    pub performance: TraceSimulationReport,
    /// Typed agentic identity, configuration, aggregates, and episode records.
    pub evaluation: AgenticEvaluationReport,
    /// Unified native-v2 report.
    pub native_report: NativeReport,
    /// Canonical episode results in frozen evaluator order.
    pub results: Vec<AgenticEpisodeResult>,
}

/// Join a drained normal-pipeline run with canonical harness results.
///
/// Rust only aggregates finite verifier-owned reward values. It never decides
/// benchmark correctness, and infrastructure/cancelled episodes are reported
/// separately rather than converted into zero-valued model scores.
#[allow(clippy::too_many_arguments)]
pub fn finalize_agentic_report(
    requested_dataset: &str,
    model: &str,
    model_concurrency: usize,
    scheduled: crate::scheduled::ScheduledRunReport,
    worker: aiperf_accuracy::EvaluatorIdentity,
    evaluator: AgenticEvaluatorIdentity,
    config: &AgenticEvaluatorLoadConfig,
    results: Vec<AgenticEpisodeResult>,
) -> Result<AgenticRunReport> {
    ensure!(
        evaluator.episode_count == results.len(),
        "agentic evaluator declared {} episodes but finalized {}",
        evaluator.episode_count,
        results.len()
    );
    ensure!(
        evaluator.environment == config.environment,
        "agentic evaluator changed environment {:?} to {:?}",
        config.environment,
        evaluator.environment
    );

    let mut reward_values = BTreeMap::<String, Vec<f64>>::new();
    let mut completed_count = 0usize;
    let mut infrastructure_error_count = 0usize;
    let mut cancelled_count = 0usize;
    let mut model_calls = 0usize;
    let mut primary_model_calls = 0usize;
    let mut auxiliary_model_calls = 0usize;
    let mut environment_model_calls = 0usize;
    let mut verifier_model_calls = 0usize;
    let mut result_primary_rewards = BTreeSet::new();
    let mut report_records = Vec::with_capacity(results.len());
    for result in &results {
        ensure!(
            result.duration_seconds.is_finite() && result.duration_seconds >= 0.0,
            "agentic episode {:?} has invalid duration {}",
            result.episode_id.as_str(),
            result.duration_seconds
        );
        let outcome = match result.outcome {
            aiperf_accuracy::AgenticEpisodeOutcome::Completed => {
                completed_count += 1;
                for (name, value) in &result.rewards {
                    ensure!(
                        value.is_finite(),
                        "agentic episode {:?} has non-finite reward {name:?}",
                        result.episode_id.as_str()
                    );
                    reward_values.entry(name.clone()).or_default().push(*value);
                }
                if let Some(primary) = &result.primary_reward {
                    result_primary_rewards.insert(primary.clone());
                }
                AgenticEpisodeReportOutcome::Completed
            }
            aiperf_accuracy::AgenticEpisodeOutcome::InfrastructureError => {
                infrastructure_error_count += 1;
                AgenticEpisodeReportOutcome::InfrastructureError
            }
            aiperf_accuracy::AgenticEpisodeOutcome::Cancelled => {
                cancelled_count += 1;
                AgenticEpisodeReportOutcome::Cancelled
            }
        };
        model_calls = checked_call_count(model_calls, result.model_calls)?;
        primary_model_calls = checked_call_count(primary_model_calls, result.primary_model_calls)?;
        auxiliary_model_calls =
            checked_call_count(auxiliary_model_calls, result.auxiliary_model_calls)?;
        environment_model_calls =
            checked_call_count(environment_model_calls, result.environment_model_calls)?;
        verifier_model_calls =
            checked_call_count(verifier_model_calls, result.verifier_model_calls)?;
        report_records.push(AgenticEpisodeReport {
            episode_id: result.episode_id.as_str().to_string(),
            task: result.task.clone(),
            outcome,
            rewards: result.rewards.clone(),
            primary_reward: result.primary_reward.clone(),
            duration_seconds: result.duration_seconds,
            model_calls: result.model_calls,
            primary_model_calls: result.primary_model_calls,
            auxiliary_model_calls: result.auxiliary_model_calls,
            environment_model_calls: result.environment_model_calls,
            verifier_model_calls: result.verifier_model_calls,
            prompt_tokens: result.prompt_tokens,
            completion_tokens: result.completion_tokens,
            cached_tokens: result.cached_tokens,
            primary_prompt_tokens: result.primary_prompt_tokens,
            primary_completion_tokens: result.primary_completion_tokens,
            primary_cached_tokens: result.primary_cached_tokens,
            auxiliary_prompt_tokens: result.auxiliary_prompt_tokens,
            auxiliary_completion_tokens: result.auxiliary_completion_tokens,
            auxiliary_cached_tokens: result.auxiliary_cached_tokens,
            error_kind: result.error_kind.clone(),
            error_message: result.error_message.clone(),
            artifact_path: result.artifact_path.clone(),
        });
    }

    let mut rewards = BTreeMap::new();
    for (name, values) in reward_values {
        let n = values.len();
        let sum = values.iter().sum::<f64>();
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let avg = sum / n as f64;
        ensure!(
            avg.is_finite() && min.is_finite() && max.is_finite(),
            "canonical reward {name:?} overflowed finite report aggregation"
        );
        rewards.insert(name, AgenticRewardSummary { n, avg, min, max });
    }
    let primary_reward = evaluator.primary_reward.clone().or_else(|| {
        (result_primary_rewards.len() == 1)
            .then(|| result_primary_rewards.iter().next().cloned())
            .flatten()
    });
    let primary_score = primary_reward
        .as_ref()
        .and_then(|reward| rewards.get(reward))
        .map(|summary| summary.avg);
    if let Some(configured) = &evaluator.primary_reward {
        ensure!(
            primary_score.is_some(),
            "agentic evaluator selected primary reward {configured:?} but no completed episode reported it"
        );
    }
    ensure!(
        primary_model_calls
            .checked_add(auxiliary_model_calls)
            .is_some_and(|total| total == model_calls),
        "agentic run-level primary and auxiliary call counts diverged from the total"
    );
    ensure!(
        environment_model_calls
            .checked_add(verifier_model_calls)
            .is_some_and(|total| total == auxiliary_model_calls),
        "agentic run-level environment and verifier counts diverged from auxiliary calls"
    );
    let prompt_tokens = sum_episode_tokens(
        &results,
        |result| result.model_calls,
        |result| result.prompt_tokens,
        "prompt",
    )?;
    let completion_tokens = sum_episode_tokens(
        &results,
        |result| result.model_calls,
        |result| result.completion_tokens,
        "completion",
    )?;
    let cached_tokens = sum_episode_tokens(
        &results,
        |result| result.model_calls,
        |result| result.cached_tokens,
        "cached",
    )?;
    let primary_prompt_tokens = sum_episode_tokens(
        &results,
        |result| result.primary_model_calls,
        |result| result.primary_prompt_tokens,
        "primary prompt",
    )?;
    let primary_completion_tokens = sum_episode_tokens(
        &results,
        |result| result.primary_model_calls,
        |result| result.primary_completion_tokens,
        "primary completion",
    )?;
    let primary_cached_tokens = sum_episode_tokens(
        &results,
        |result| result.primary_model_calls,
        |result| result.primary_cached_tokens,
        "primary cached",
    )?;
    let auxiliary_prompt_tokens = sum_episode_tokens(
        &results,
        |result| result.auxiliary_model_calls,
        |result| result.auxiliary_prompt_tokens,
        "auxiliary prompt",
    )?;
    let auxiliary_completion_tokens = sum_episode_tokens(
        &results,
        |result| result.auxiliary_model_calls,
        |result| result.auxiliary_completion_tokens,
        "auxiliary completion",
    )?;
    let auxiliary_cached_tokens = sum_episode_tokens(
        &results,
        |result| result.auxiliary_model_calls,
        |result| result.auxiliary_cached_tokens,
        "auxiliary cached",
    )?;

    let evaluation = AgenticEvaluationReport {
        evaluator: AgenticEvaluatorReportInfo {
            harness: evaluator.harness.clone(),
            harness_version: evaluator.harness_version.clone(),
            harness_source_sha256: evaluator.harness_source_sha256.clone(),
            agent: evaluator.agent.clone(),
            agent_version: evaluator.agent_version.clone(),
            environment: evaluator.environment.clone(),
            verifier: evaluator.verifier.clone(),
        },
        config: AgenticRunConfigReport {
            dataset: requested_dataset.to_string(),
            task_names: config.task_names.clone(),
            max_episodes: config.max_episodes,
            task_concurrency: config.task_concurrency,
            model_concurrency,
            output_dir: config.output_dir.clone(),
            max_turns: config.max_turns,
            max_tokens: config.max_tokens,
            context_window: config.context_window,
            parser: config.parser.clone(),
            enable_summarize: config.enable_summarize,
            primary_reward: config.primary_reward.clone(),
            overwrite: config.overwrite,
            inference_gateway_base_url: config
                .inference_gateway
                .as_ref()
                .map(|gateway| gateway.base_url.clone()),
        },
        summary: AgenticEvaluationSummary {
            episode_count: results.len(),
            completed_count,
            infrastructure_error_count,
            cancelled_count,
            model_calls,
            primary_model_calls,
            auxiliary_model_calls,
            environment_model_calls,
            verifier_model_calls,
            prompt_tokens,
            completion_tokens,
            cached_tokens,
            primary_prompt_tokens,
            primary_completion_tokens,
            primary_cached_tokens,
            auxiliary_prompt_tokens,
            auxiliary_completion_tokens,
            auxiliary_cached_tokens,
            primary_reward,
            primary_score,
            rewards,
        },
        records: report_records,
    };
    let evaluator_report = agentic_worker_report_info(&worker, requested_dataset, &evaluator);
    let errors = agentic_report_errors(&results);
    let native_report = NativeReport::from_outcome(
        &scheduled.native_metrics,
        &RunOutcome {
            run: ReportRunInfo {
                mode: Some("agentic_accuracy".to_string()),
                model: Some(model.to_string()),
            },
            evaluator: Some(evaluator_report),
            agentic: Some(evaluation.clone()),
            errors,
            ..RunOutcome::default()
        },
    );
    Ok(AgenticRunReport {
        dataset: requested_dataset.to_string(),
        model: model.to_string(),
        worker,
        evaluator,
        performance: scheduled.performance,
        evaluation,
        native_report,
        results,
    })
}

fn checked_call_count(total: usize, value: usize) -> Result<usize> {
    total
        .checked_add(value)
        .ok_or_else(|| anyhow!("agentic report model-call count overflowed usize"))
}

fn sum_episode_tokens<F, G>(
    results: &[AgenticEpisodeResult],
    calls: F,
    tokens: G,
    name: &str,
) -> Result<Option<u64>>
where
    F: Fn(&AgenticEpisodeResult) -> usize,
    G: Fn(&AgenticEpisodeResult) -> Option<u64>,
{
    let mut total = 0u64;
    let mut observed = false;
    for result in results {
        if calls(result) == 0 {
            continue;
        }
        observed = true;
        let Some(value) = tokens(result) else {
            return Ok(None);
        };
        total = total
            .checked_add(value)
            .ok_or_else(|| anyhow!("agentic report {name} token total overflowed u64"))?;
    }
    Ok(observed.then_some(total))
}

fn agentic_worker_report_info(
    worker: &aiperf_accuracy::EvaluatorIdentity,
    requested_dataset: &str,
    evaluator: &AgenticEvaluatorIdentity,
) -> EvaluatorReportInfo {
    EvaluatorReportInfo {
        protocol: worker.protocol,
        worker_version: worker.worker_version.clone(),
        python_version: worker.python_version.clone(),
        python_executable: worker.python_executable.clone(),
        packages: worker.packages.clone(),
        worker_source_sha256: worker.worker_source_sha256.clone(),
        dependency_lock_sha256: worker.dependency_lock_sha256.clone(),
        container_digest: worker.container_digest.clone(),
        capabilities: worker.capabilities.clone(),
        benchmark: evaluator
            .dataset
            .benchmark
            .clone()
            .unwrap_or_else(|| requested_dataset.to_string()),
        grader: evaluator.verifier.clone(),
        dataset: EvaluatorDatasetReportInfo {
            provider: evaluator.dataset.provider.clone(),
            benchmark: evaluator.dataset.benchmark.clone(),
            repository: evaluator.dataset.repository.clone(),
            subset: evaluator.dataset.subset.clone(),
            revision: evaluator.dataset.revision.clone(),
            evaluation_splits: evaluator.dataset.evaluation_splits.clone(),
            task_version: evaluator.dataset.task_version,
        },
    }
}

fn agentic_report_errors(results: &[AgenticEpisodeResult]) -> Vec<ReportError> {
    let mut groups = BTreeMap::<(String, String), usize>::new();
    for result in results {
        let prefix = match result.outcome {
            aiperf_accuracy::AgenticEpisodeOutcome::Completed => continue,
            aiperf_accuracy::AgenticEpisodeOutcome::InfrastructureError => "AgenticInfrastructure",
            aiperf_accuracy::AgenticEpisodeOutcome::Cancelled => "AgenticCancelled",
        };
        let kind = result.error_kind.as_deref().unwrap_or("Unknown");
        let error_type = format!("{prefix}:{kind}");
        let message = result
            .error_message
            .clone()
            .unwrap_or_else(|| format!("agentic episode ended {prefix}"));
        *groups.entry((error_type, message)).or_default() += 1;
    }
    groups
        .into_iter()
        .map(|((error_type, message), count)| ReportError {
            code: None,
            error_type,
            message,
            count,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use aiperf_accuracy::{
        AccuracyEvaluator, AgenticEventBatch, AgenticInferenceGatewayConfig, AgenticMessage,
        AgenticResultBatch, EvaluatorDatasetIdentity, EvaluatorGradeBatch, EvaluatorGradeItem,
        EvaluatorIdentity, EvaluatorLoadConfig, EvaluatorLoadResult, EvaluatorProblemPage,
        EvaluatorWorkerError,
    };
    use aiperf_clock::{Clock, RealClock};
    use aiperf_dataset::TiktokenTokenizer;
    use aiperf_timing::StopConfig;
    use serde_json::Map;

    use super::*;
    use crate::http::TransportSink;
    use crate::scheduled::{TurnDispatcher, run_scheduled_workload};

    #[derive(Default)]
    struct FixtureState {
        submitted: Vec<AgenticModelResult>,
    }

    struct FixtureEvaluator {
        static_identity: EvaluatorIdentity,
        agentic_identity: AgenticEvaluatorIdentity,
        episode: AgenticEpisode,
        call: AgenticModelCall,
        events: VecDeque<AgenticEvaluatorEvent>,
        result: AgenticEpisodeResult,
        state: Rc<RefCell<FixtureState>>,
    }

    struct FixtureGateway {
        config: AgenticInferenceGatewayConfig,
        requests: Option<mpsc::UnboundedReceiver<AgenticAuxiliaryInferenceRequest>>,
    }

    impl FixtureGateway {
        fn new() -> (
            Self,
            mpsc::UnboundedSender<AgenticAuxiliaryInferenceRequest>,
        ) {
            let (sender, receiver) = mpsc::unbounded_channel();
            (
                Self {
                    config: AgenticInferenceGatewayConfig {
                        base_url: "http://fixture-gateway:4321".to_string(),
                        api_key: "fixture-secret".to_string(),
                    },
                    requests: Some(receiver),
                },
                sender,
            )
        }
    }

    #[async_trait(?Send)]
    impl AgenticInferenceGateway for FixtureGateway {
        fn evaluator_config(&self) -> &AgenticInferenceGatewayConfig {
            &self.config
        }

        fn take_requests(
            &mut self,
        ) -> Result<mpsc::UnboundedReceiver<AgenticAuxiliaryInferenceRequest>> {
            self.requests
                .take()
                .ok_or_else(|| anyhow!("fixture gateway receiver already taken"))
        }

        async fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    impl FixtureEvaluator {
        fn new(state: Rc<RefCell<FixtureState>>) -> Self {
            let episode_id = EpisodeId::new("episode-1").unwrap();
            let call_id = ModelCallId::new("call-1").unwrap();
            let episode = AgenticEpisode {
                episode_id: episode_id.clone(),
                task: "fixture-task".to_string(),
                source: "fixture/agentic".to_string(),
            };
            let call = AgenticModelCall {
                episode_id: episode_id.clone(),
                call_id,
                turn_index: 0,
                prompt: "Use the terminal".to_string(),
                model: Some("fixture-model".to_string()),
                messages: vec![AgenticMessage {
                    role: "user".to_string(),
                    content: Value::String("Use the terminal".to_string()),
                    extra: BTreeMap::new(),
                }],
                generation: aiperf_accuracy::EvaluatorGenerationConfig {
                    max_tokens: 2,
                    temperature: 0.2,
                    top_p: 0.9,
                    stop: vec!["</tool>".to_string()],
                },
                tools: vec![json!({
                    "type": "function",
                    "function": {"name": "terminal", "parameters": {"type": "object"}}
                })],
                tool_choice: Some(Value::String("auto".to_string())),
                response_format: Some(json!({"type": "json_object"})),
                extra_body: Map::from_iter([(
                    "reasoning_effort".to_string(),
                    Value::String("low".to_string()),
                )]),
            };
            let result = AgenticEpisodeResult {
                episode_id: episode_id.clone(),
                task: episode.task.clone(),
                outcome: aiperf_accuracy::AgenticEpisodeOutcome::Completed,
                rewards: BTreeMap::from([("reward".to_string(), 1.0)]),
                primary_reward: Some("reward".to_string()),
                duration_seconds: 1.0,
                model_calls: 1,
                primary_model_calls: 0,
                auxiliary_model_calls: 0,
                environment_model_calls: 0,
                verifier_model_calls: 0,
                prompt_tokens: Some(3),
                completion_tokens: Some(2),
                cached_tokens: None,
                primary_prompt_tokens: None,
                primary_completion_tokens: None,
                primary_cached_tokens: None,
                auxiliary_prompt_tokens: None,
                auxiliary_completion_tokens: None,
                auxiliary_cached_tokens: None,
                error_kind: None,
                error_message: None,
                artifact_path: Some("fixture-artifact".to_string()),
            };
            Self {
                static_identity: EvaluatorIdentity {
                    protocol: 1,
                    worker_version: "fixture".to_string(),
                    python_version: "3".to_string(),
                    python_executable: "/fixture/python".to_string(),
                    packages: BTreeMap::new(),
                    worker_source_sha256: "a".repeat(64),
                    dependency_lock_sha256: Some("b".repeat(64)),
                    container_digest: None,
                    capabilities: vec!["agentic_harbor".to_string()],
                },
                agentic_identity: AgenticEvaluatorIdentity {
                    harness: "harbor".to_string(),
                    harness_version: "0.18.0".to_string(),
                    harness_source_sha256: "c".repeat(64),
                    dataset: EvaluatorDatasetIdentity {
                        provider: "fixture".to_string(),
                        benchmark: Some("fixture/agentic".to_string()),
                        repository: Some("fixture/agentic".to_string()),
                        subset: None,
                        revision: Some("d".repeat(64)),
                        evaluation_splits: vec!["tasks".to_string()],
                        task_version: None,
                    },
                    agent: "aiperf-terminus-2".to_string(),
                    agent_version: "fixture".to_string(),
                    environment: "fixture".to_string(),
                    verifier: "fixture verifier".to_string(),
                    episode_count: 1,
                    primary_reward: Some("reward".to_string()),
                },
                episode,
                call,
                events: VecDeque::new(),
                result,
                state,
            }
        }
    }

    #[async_trait(?Send)]
    impl AccuracyEvaluator for FixtureEvaluator {
        fn identity(&self) -> &EvaluatorIdentity {
            &self.static_identity
        }

        async fn load(
            &mut self,
            _benchmark: &str,
            _config: &EvaluatorLoadConfig,
        ) -> std::result::Result<EvaluatorLoadResult, EvaluatorWorkerError> {
            Err(EvaluatorWorkerError::Protocol(
                "static load is not used by fixture".to_string(),
            ))
        }

        async fn next_problems(
            &mut self,
            _offset: usize,
            _limit: usize,
        ) -> std::result::Result<EvaluatorProblemPage, EvaluatorWorkerError> {
            Err(EvaluatorWorkerError::Protocol(
                "static problems are not used by fixture".to_string(),
            ))
        }

        async fn grade_batch(
            &mut self,
            _items: &[EvaluatorGradeItem],
        ) -> std::result::Result<EvaluatorGradeBatch, EvaluatorWorkerError> {
            Err(EvaluatorWorkerError::Protocol(
                "static grades are not used by fixture".to_string(),
            ))
        }

        async fn shutdown(&mut self) -> std::result::Result<(), EvaluatorWorkerError> {
            Ok(())
        }
    }

    #[async_trait(?Send)]
    impl AgenticEvaluator for FixtureEvaluator {
        fn supports_agentic(&self) -> bool {
            true
        }

        fn supports_agentic_inference_gateway(&self) -> bool {
            true
        }

        async fn load_agentic(
            &mut self,
            _dataset: &str,
            _model: &str,
            _config: &AgenticEvaluatorLoadConfig,
        ) -> std::result::Result<AgenticEvaluatorIdentity, EvaluatorWorkerError> {
            Ok(self.agentic_identity.clone())
        }

        async fn next_episodes(
            &mut self,
            offset: usize,
            _limit: usize,
        ) -> std::result::Result<AgenticEpisodePage, EvaluatorWorkerError> {
            Ok(AgenticEpisodePage {
                items: if offset == 0 {
                    vec![self.episode.clone()]
                } else {
                    Vec::new()
                },
                next_offset: 1,
                done: true,
            })
        }

        async fn start_episodes(
            &mut self,
            episode_ids: &[EpisodeId],
        ) -> std::result::Result<(), EvaluatorWorkerError> {
            assert_eq!(episode_ids, std::slice::from_ref(&self.episode.episode_id));
            self.events.push_back(AgenticEvaluatorEvent::ModelCall {
                call: self.call.clone(),
            });
            Ok(())
        }

        async fn poll_agentic(
            &mut self,
            limit: usize,
            _wait_ms: u64,
        ) -> std::result::Result<AgenticEventBatch, EvaluatorWorkerError> {
            let mut events = Vec::new();
            while events.len() < limit {
                let Some(event) = self.events.pop_front() else {
                    break;
                };
                events.push(event);
            }
            Ok(AgenticEventBatch { events })
        }

        async fn submit_model_results(
            &mut self,
            items: &[AgenticModelResult],
        ) -> std::result::Result<(), EvaluatorWorkerError> {
            self.state.borrow_mut().submitted.extend_from_slice(items);
            self.events
                .push_back(AgenticEvaluatorEvent::EpisodeCompleted {
                    result: self.result.clone(),
                });
            Ok(())
        }

        async fn cancel_episodes(
            &mut self,
            _episode_ids: &[EpisodeId],
        ) -> std::result::Result<(), EvaluatorWorkerError> {
            Ok(())
        }

        async fn finish_agentic(
            &mut self,
        ) -> std::result::Result<AgenticResultBatch, EvaluatorWorkerError> {
            Ok(AgenticResultBatch {
                items: vec![self.result.clone()],
            })
        }
    }

    fn builder() -> Rc<dyn AgenticTurnBuilder> {
        Rc::new(
            DatasetAgenticTurnBuilder::chat(
                "fixture-model",
                Arc::new(TiktokenTokenizer::builtin()),
            )
            .unwrap(),
        )
    }

    #[test]
    fn dataset_builder_preserves_messages_generation_and_tools() {
        let state = Rc::new(RefCell::new(FixtureState::default()));
        let evaluator = FixtureEvaluator::new(state);
        let turn = builder().build_turn(&evaluator.call).unwrap();
        assert_eq!(turn.x_correlation_id, "episode-1");
        assert_eq!(turn.request_correlation_id, "call-1");
        let body: Value = serde_json::from_slice(turn.request_body.as_ref().unwrap()).unwrap();
        assert_eq!(body["model"], "fixture-model");
        assert_eq!(body["messages"][0]["content"], "Use the terminal");
        assert_eq!(body["max_completion_tokens"], 2);
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["top_p"], 0.9);
        assert_eq!(body["stop"], json!(["</tool>"]));
        assert_eq!(body["tools"][0]["function"]["name"], "terminal");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["response_format"]["type"], "json_object");
        assert_eq!(body["reasoning_effort"], "low");
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn failed_normal_dispatch_remains_infrastructure_not_a_model_score() {
        let item = model_result(
            EpisodeId::new("episode-1").unwrap(),
            ModelCallId::new("call-1").unwrap(),
            TurnDispatchOutcome {
                start_ns: 1,
                end_ns: 2,
                terminal: ReplayTerminalStatus::Failed,
                response_text: "partial".to_string(),
                model_response: crate::scheduled::ModelResponseMetadata {
                    content: Some("partial".to_string()),
                    reasoning: Some("thinking".to_string()),
                    cached_prompt_tokens: Some(4),
                    error_kind: Some("timeout".to_string()),
                    error_message: Some("deadline elapsed".to_string()),
                    ..crate::scheduled::ModelResponseMetadata::default()
                },
                prompt_tokens: Some(8),
                completion_tokens: Some(1),
                http: aiperf_metrics::HttpTrace::default(),
            },
        );
        assert_eq!(item.status, AgenticInferenceStatus::Failed);
        assert_eq!(item.response, "partial");
        assert_eq!(item.reasoning.as_deref(), Some("thinking"));
        assert_eq!(item.cached_tokens, Some(4));
        assert_eq!(item.error_kind.as_deref(), Some("timeout"));
        assert_eq!(item.error_message.as_deref(), Some("deadline elapsed"));
    }

    #[tokio::test]
    async fn model_calls_use_normal_scheduled_http_and_return_parsed_metadata() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let state = Rc::new(RefCell::new(FixtureState::default()));
                let evaluator: Box<dyn AgenticEvaluator> =
                    Box::new(FixtureEvaluator::new(state.clone()));
                let config = AgenticEvaluatorLoadConfig::default();
                let workload = AgenticWorkload::prepare(
                    evaluator,
                    "fixture/agentic@locked",
                    "fixture-model",
                    &config,
                    1,
                    builder(),
                )
                .await
                .unwrap();
                let base_url = crate::test_util::spawn_mock().await;
                let clock: Rc<dyn Clock> = RealClock::new();
                let start_ns = clock.now_ns();
                let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(TransportSink::new(
                    clock.clone(),
                    start_ns,
                    &base_url,
                    "fixture-model",
                    false,
                ));
                let report = run_scheduled_workload(
                    workload.clone(),
                    clock,
                    start_ns,
                    dispatcher,
                    StopConfig::default(),
                    false,
                )
                .await
                .unwrap();

                assert_eq!(report.strategy, "agentic");
                assert_eq!(report.performance.request_counts.completed_requests, 1);
                assert_eq!(report.turns[0].x_correlation_id, "episode-1");
                let submitted = state.borrow().submitted.clone();
                assert_eq!(submitted.len(), 1);
                assert_eq!(submitted[0].status, AgenticInferenceStatus::Completed);
                assert_eq!(submitted[0].response, "ab");
                assert_eq!(submitted[0].response_id.as_deref(), Some("x"));
                assert_eq!(submitted[0].finish_reason.as_deref(), Some("stop"));
                assert_eq!(submitted[0].prompt_tokens, Some(3));
                assert_eq!(submitted[0].completion_tokens, Some(2));
                assert_eq!(workload.results().unwrap()[0].rewards["reward"], 1.0);
                workload.shutdown().await.unwrap();
            })
            .await;
    }

    #[tokio::test]
    async fn auxiliary_calls_share_normal_scheduling_transport_and_accounting() {
        let local = tokio::task::LocalSet::new();
        local
            .run_until(async {
                let state = Rc::new(RefCell::new(FixtureState::default()));
                let evaluator: Box<dyn AgenticEvaluator> =
                    Box::new(FixtureEvaluator::new(state.clone()));
                let (gateway, gateway_tx) = FixtureGateway::new();
                let workload = AgenticWorkload::prepare_with_gateway(
                    evaluator,
                    "fixture/agentic@locked",
                    "fixture-model",
                    &AgenticEvaluatorLoadConfig::default(),
                    1,
                    builder(),
                    Some(Box::new(gateway)),
                )
                .await
                .unwrap();
                assert_eq!(
                    workload
                        .config()
                        .inference_gateway
                        .as_ref()
                        .unwrap()
                        .base_url,
                    "http://fixture-gateway:4321"
                );

                let call = AgenticModelCall {
                    episode_id: EpisodeId::new("episode-1").unwrap(),
                    call_id: ModelCallId::new("episode-1:aux:environment:0000").unwrap(),
                    turn_index: 0,
                    model: Some("simulator-model".to_string()),
                    prompt: String::new(),
                    messages: vec![AgenticMessage {
                        role: "user".to_string(),
                        content: Value::String("Act as the simulated user".to_string()),
                        extra: BTreeMap::new(),
                    }],
                    generation: aiperf_accuracy::EvaluatorGenerationConfig {
                        max_tokens: 2,
                        temperature: 0.0,
                        top_p: 1.0,
                        stop: Vec::new(),
                    },
                    tools: Vec::new(),
                    tool_choice: None,
                    response_format: None,
                    extra_body: Map::new(),
                };
                let (request, response_rx) = AgenticAuxiliaryInferenceRequest::new(
                    AgenticInferencePurpose::Environment,
                    call,
                    false,
                );
                gateway_tx.send(request).unwrap();

                let base_url = crate::test_util::spawn_mock().await;
                let clock: Rc<dyn Clock> = RealClock::new();
                let start_ns = clock.now_ns();
                let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(TransportSink::new(
                    clock.clone(),
                    start_ns,
                    &base_url,
                    "fixture-model",
                    false,
                ));
                let report = run_scheduled_workload(
                    workload.clone(),
                    clock,
                    start_ns,
                    dispatcher,
                    StopConfig::default(),
                    false,
                )
                .await
                .unwrap();

                let auxiliary = response_rx.await.unwrap();
                assert_eq!(auxiliary.status, AgenticInferenceStatus::Completed);
                assert_eq!(auxiliary.response, "ab");
                assert_eq!(
                    auxiliary.assistant_message.as_ref().unwrap()["content"],
                    "ab"
                );
                assert_eq!(report.performance.request_counts.completed_requests, 2);
                assert_eq!(report.turns.len(), 2);
                assert!(
                    report
                        .turns
                        .iter()
                        .all(|turn| turn.x_correlation_id == "episode-1")
                );
                let result = &workload.results().unwrap()[0];
                assert_eq!(result.model_calls, 2);
                assert_eq!(result.primary_model_calls, 1);
                assert_eq!(result.auxiliary_model_calls, 1);
                assert_eq!(result.environment_model_calls, 1);
                assert_eq!(result.verifier_model_calls, 0);
                assert_eq!(result.prompt_tokens, Some(6));
                assert_eq!(result.completion_tokens, Some(4));
                assert_eq!(result.primary_prompt_tokens, Some(3));
                assert_eq!(result.auxiliary_prompt_tokens, Some(3));
                assert_eq!(state.borrow().submitted.len(), 1);
                workload.shutdown().await.unwrap();
            })
            .await;
    }
}
