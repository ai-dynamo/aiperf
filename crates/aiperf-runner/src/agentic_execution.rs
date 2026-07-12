// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stateful agentic execution through the runner's ordinary online dispatcher.
//!
//! The canonical Python worker owns benchmark packages, environments, agent
//! loops, hidden verifier state, and scoring. This module owns only process
//! supervision and Rust composition: every primary, environment, and verifier
//! model call enters one [`AgenticWorkload`], one [`TurnDispatcher`], and an
//! injected [`HttpExecutionBackendFactory`]. No Python model URL is exposed.
//!
//! The evaluator behavior is grounded in the complete current implementations
//! at `src/aiperf/accuracy/worker.py:1-1229`,
//! `src/aiperf/accuracy/agentic.py:1-410`,
//! `src/aiperf/accuracy/model_broker.py:1-190`, and the canonical provider
//! adapters in `harbor.py`, `browsergym.py`, and `mcpmark.py`. This runner
//! composition replaces only the removed native-CLI assembly formerly covered
//! by `crates/aiperf/tests/agentic_cli.rs` and
//! `crates/aiperf/tests/agentic_gateway_cli.rs` before commit `697b66f9f`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use aiperf::agentic::{
    AgenticRunReport, AgenticTurnBuilder, AgenticWorkload, DatasetAgenticTurnBuilder,
    finalize_agentic_report,
};
use aiperf::agentic_gateway::{
    AgenticInferenceGateway, HttpAgenticInferenceGateway, resolve_advertised_host,
};
use aiperf::http::{
    HttpTurnExecutionBackend, PreparedEndpointReference, PreparedHttpTurn, TransportSinkConfig,
};
use aiperf::multiturn::TurnToSend;
use aiperf::report::write_native_report_json;
use aiperf::scheduled::{TurnDispatchOutcome, TurnDispatcher, Workload, run_scheduled_workload};
use aiperf_accuracy::{
    AgenticEvaluator, AgenticEvaluatorLoadConfig, PythonEvaluator, WorkerProcessConfig,
};
use aiperf_clock::{Clock, RealClock, RealClockAnchor};
use aiperf_dataset::{HuggingFaceTokenizer, TextTokenizer, TiktokenEncoding, TiktokenTokenizer};
use aiperf_endpoints::{
    EndpointId, EndpointKey, EndpointRegistry, Modality, PreparedEndpointTable, RawEndpointConfig,
};
use aiperf_extensions::{AiperfRegistry, AiperfRegistryFactory, BuiltinAiperfRegistryFactory};
use aiperf_metrics::ReportPairRunFacts;
use aiperf_timing::StopConfig;
use aiperf_transport_http::config::ClientConfig;
use aiperf_transport_http::models::{ConnectionReuseStrategy, HttpVersion};
use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;
use loadgen_core::sink::RequestObserver;
use serde::Deserialize;
use serde_json::value::RawValue;

use crate::online_execution::{
    NativeOnlineTokenizerSourceResolver, OnlineTokenizerSourceResolver, lower_authored_tokenizer,
    validate_authored_tokenizer,
};
use crate::protocol::{PhaseSpec, TokenizerSpec};
use crate::protocol_v2::AuthoredRunSpecV2;
use crate::registry::{
    OnlineHttpBackendConfigV2, PreparedRunOutcome, PreparedRunnerOperation, RunnerClockKind,
    RunnerPairFactory, RunnerRegistryBuilder, RunnerRunContext, RunnerWorkloadDescriptor,
    RunnerWorkloadFactory, ValidatedBackendConfig, ValidatedWorkloadConfig, WorkloadRequirements,
};
use crate::turn_execution::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, HttpPreparedEndpointTableFactory,
    NativeHttpExecutionBackendFactory,
};

static AGENTIC_WORKLOAD_DESCRIPTOR: RunnerWorkloadDescriptor = RunnerWorkloadDescriptor {
    id: "agentic",
    description: "canonical stateful Python harness over Rust-owned inference",
    requires_semantic_responses: true,
    clock_kinds: &[RunnerClockKind::Real],
    required_backend_features: &["http"],
};

/// Process command for the supervised canonical evaluator worker.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticEvaluatorProcessSpec {
    /// Absolute interpreter selected by the Python orchestrator.
    pub python_executable: PathBuf,
    /// Importable worker module.
    #[serde(default = "default_evaluator_module")]
    pub worker_module: String,
    /// Explicit child environment. Values are never written to reports.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// Optional worker current directory.
    #[serde(default)]
    pub current_dir: Option<PathBuf>,
}

impl fmt::Debug for AgenticEvaluatorProcessSpec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgenticEvaluatorProcessSpec")
            .field("python_executable", &self.python_executable)
            .field("worker_module", &self.worker_module)
            .field(
                "environment_keys",
                &self.environment.keys().collect::<Vec<_>>(),
            )
            .field("current_dir", &self.current_dir)
            .finish()
    }
}

fn default_evaluator_module() -> String {
    "aiperf.accuracy.worker".to_string()
}

const fn default_one() -> usize {
    1
}

const fn default_max_tokens() -> usize {
    4_096
}

const fn default_context_window() -> usize {
    131_072
}

fn default_environment() -> String {
    "docker".to_string()
}

fn default_output_dir() -> PathBuf {
    PathBuf::from("agentic")
}

fn default_parser() -> String {
    "json".to_string()
}

const fn default_true() -> bool {
    true
}

/// Strict protocol-v2 config owned by the `agentic` workload factory.
///
/// `worker_count`, `dataset`, `tokenizer`, and `phases` intentionally use the
/// same names Python's authored v2 projection fills for every workload. For
/// agentic runs, the one profiling concurrency phase supplies model-call
/// admission only; the canonical harness owns episode ordering and cadence.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgenticWorkloadConfigV2 {
    /// Number of execution-placement workers below the single dispatcher.
    pub worker_count: usize,
    /// Opaque canonical provider dataset, including its immutable revision.
    pub dataset: String,
    /// Authored tokenizer acquisition policy resolved during pair preparation.
    pub tokenizer: Box<RawValue>,
    /// Exactly one profiling concurrency phase; no outer timing policy applies.
    pub phases: Vec<PhaseSpec>,
    /// Supervised evaluator process command.
    pub evaluator: AgenticEvaluatorProcessSpec,
    /// Exact task names or provider-owned patterns.
    #[serde(default)]
    pub task_names: Option<Vec<String>>,
    /// Deterministic episode cap after provider selection.
    #[serde(default)]
    pub max_episodes: Option<usize>,
    /// Evaluator environment concurrency.
    #[serde(default = "default_one")]
    pub task_concurrency: usize,
    /// Canonical harness environment/provider selector.
    #[serde(default = "default_environment")]
    pub environment: String,
    /// Relative directory below the exclusive run artifact target.
    #[serde(default = "default_output_dir")]
    pub output_dir: PathBuf,
    /// Optional provider-owned primary-agent call limit.
    #[serde(default)]
    pub max_turns: Option<usize>,
    /// Maximum output tokens on each Rust-dispatched model call.
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Explicit context window consumed by the canonical agent scaffold.
    #[serde(default = "default_context_window")]
    pub context_window: usize,
    /// Canonical provider parser selection.
    #[serde(default = "default_parser")]
    pub parser: String,
    /// Whether a provider that supports summarization may enable it.
    #[serde(default = "default_true")]
    pub enable_summarize: bool,
    /// Optional verifier reward chosen as primary.
    #[serde(default)]
    pub primary_reward: Option<String>,
    /// Whether the canonical provider may replace cached task packages.
    #[serde(default)]
    pub overwrite: bool,
    /// Explicit sandbox-reachable host; absent uses the kernel route probe.
    #[serde(default)]
    pub inference_gateway_host: Option<String>,
}

impl fmt::Debug for AgenticWorkloadConfigV2 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgenticWorkloadConfigV2")
            .field("worker_count", &self.worker_count)
            .field("dataset", &self.dataset)
            .field("tokenizer", &self.tokenizer.get())
            .field("phase_count", &self.phases.len())
            .field("model_concurrency", &self.model_concurrency().ok())
            .field("evaluator", &self.evaluator)
            .field("task_names", &self.task_names)
            .field("max_episodes", &self.max_episodes)
            .field("task_concurrency", &self.task_concurrency)
            .field("environment", &self.environment)
            .field("output_dir", &self.output_dir)
            .field("max_turns", &self.max_turns)
            .field("max_tokens", &self.max_tokens)
            .field("context_window", &self.context_window)
            .field("parser", &self.parser)
            .field("enable_summarize", &self.enable_summarize)
            .field("primary_reward", &self.primary_reward)
            .field("overwrite", &self.overwrite)
            .field("inference_gateway_host", &self.inference_gateway_host)
            .finish()
    }
}

impl AgenticWorkloadConfigV2 {
    /// Perform side-effect-free factory validation.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.worker_count > 0,
            "agentic worker_count must be positive"
        );
        ensure!(
            !self.dataset.trim().is_empty() && self.dataset.trim() == self.dataset,
            "agentic dataset must be non-empty and contain no surrounding whitespace"
        );
        ensure!(
            self.evaluator.python_executable.is_absolute(),
            "agentic evaluator.python_executable must be an absolute path"
        );
        ensure!(
            !self.evaluator.worker_module.trim().is_empty(),
            "agentic evaluator.worker_module must not be empty"
        );
        for key in self.evaluator.environment.keys() {
            ensure!(
                !key.is_empty() && !key.contains('='),
                "agentic evaluator environment key {key:?} is invalid"
            );
        }
        validate_authored_tokenizer(&self.tokenizer)?;
        ensure!(
            self.task_concurrency > 0,
            "agentic task_concurrency must be positive"
        );
        ensure!(self.max_tokens > 0, "agentic max_tokens must be positive");
        ensure!(
            self.context_window > 0 && self.max_tokens <= self.context_window,
            "agentic context_window must be positive and at least max_tokens"
        );
        ensure!(
            !self.environment.trim().is_empty(),
            "agentic environment must not be empty"
        );
        ensure!(
            !self.parser.trim().is_empty(),
            "agentic parser must not be empty"
        );
        validate_positive_optional(self.max_episodes, "max_episodes")?;
        validate_positive_optional(self.max_turns, "max_turns")?;
        validate_optional_names(self.task_names.as_deref(), "task_names")?;
        if let Some(reward) = &self.primary_reward {
            ensure!(
                !reward.trim().is_empty(),
                "agentic primary_reward must not be empty"
            );
        }
        validate_relative_artifact_path(&self.output_dir, "agentic output_dir")?;
        if let Some(host) = &self.inference_gateway_host {
            resolve_advertised_host(Some(host))?;
        }
        self.model_concurrency()?;
        Ok(())
    }

    /// Return the one profiling phase's model-call concurrency.
    pub fn model_concurrency(&self) -> Result<usize> {
        ensure!(
            self.phases.len() == 1,
            "agentic workloads require exactly one profiling concurrency phase"
        );
        let phase = &self.phases[0];
        let common = phase.common();
        ensure!(
            common.name == "profiling" && !common.exclude_from_results,
            "agentic phase must be profiling with exclude_from_results=false"
        );
        ensure!(
            common.requests.is_none()
                && common.sessions.is_none()
                && common.duration.is_none()
                && common.prefill_concurrency.is_none()
                && common.grace_period.is_none()
                && !common.seamless
                && common.concurrency_ramp.is_none()
                && common.prefill_ramp.is_none()
                && common.rate_ramp.is_none()
                && common.cancellation.is_none()
                && common.adaptive_scale.is_none(),
            "agentic harnesses own episode scheduling; outer stop, prefill, ramp, cancellation, adaptive, grace, and seamless policies are unsupported"
        );
        let PhaseSpec::Concurrency { concurrency, .. } = phase else {
            return Err(anyhow!(
                "agentic workloads require a concurrency phase; request-rate, user-centric, and fixed schedules would duplicate harness scheduling"
            ));
        };
        ensure!(
            *concurrency > 0,
            "agentic model concurrency must be positive"
        );
        Ok(*concurrency)
    }

    fn evaluator_config(&self, artifact_target: &Path) -> Result<AgenticEvaluatorLoadConfig> {
        self.validate()?;
        let output_dir = artifact_target.join(&self.output_dir);
        let output_dir = output_dir
            .to_str()
            .ok_or_else(|| anyhow!("agentic output directory must be valid UTF-8"))?
            .to_string();
        Ok(AgenticEvaluatorLoadConfig {
            task_names: self.task_names.clone(),
            max_episodes: self.max_episodes,
            task_concurrency: self.task_concurrency,
            environment: self.environment.clone(),
            output_dir,
            max_turns: self.max_turns,
            max_tokens: self.max_tokens,
            context_window: self.context_window,
            parser: self.parser.clone(),
            enable_summarize: self.enable_summarize,
            primary_reward: self.primary_reward.clone(),
            overwrite: self.overwrite,
            inference_gateway: None,
        })
    }
}

fn validate_positive_optional(value: Option<usize>, name: &str) -> Result<()> {
    ensure!(value != Some(0), "agentic {name} must be positive when set");
    Ok(())
}

fn validate_optional_names(values: Option<&[String]>, name: &str) -> Result<()> {
    let Some(values) = values else {
        return Ok(());
    };
    ensure!(
        !values.is_empty(),
        "agentic {name} must not be empty when set"
    );
    let mut unique = BTreeSet::new();
    for value in values {
        ensure!(
            !value.trim().is_empty(),
            "agentic {name} entries must not be empty"
        );
        ensure!(
            unique.insert(value),
            "agentic {name} contains duplicate entry {value:?}"
        );
    }
    Ok(())
}

fn validate_relative_artifact_path(path: &Path, name: &str) -> Result<()> {
    ensure!(!path.as_os_str().is_empty(), "{name} must not be empty");
    for component in path.components() {
        ensure!(
            matches!(component, Component::Normal(_)),
            "{name} must be a normalized relative path below the run artifact target"
        );
    }
    Ok(())
}

/// Registered protocol-v2 workload factory for stateful agentic evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct AgenticRunnerWorkloadFactory;

impl RunnerWorkloadFactory for AgenticRunnerWorkloadFactory {
    fn descriptor(&self) -> &'static RunnerWorkloadDescriptor {
        &AGENTIC_WORKLOAD_DESCRIPTOR
    }

    fn validate(&self, authored: &RawValue) -> Result<Box<dyn ValidatedWorkloadConfig>> {
        let config: AgenticWorkloadConfigV2 = serde_json::from_str(authored.get())?;
        config.validate()?;
        Ok(Box::new(config))
    }

    fn requirements(&self, config: &dyn ValidatedWorkloadConfig) -> Result<WorkloadRequirements> {
        let config = ValidatedWorkloadConfig::as_any(config)
            .downcast_ref::<AgenticWorkloadConfigV2>()
            .ok_or_else(|| anyhow!("agentic workload factory received the wrong config type"))?;
        config.validate()?;
        Ok(WorkloadRequirements {
            semantic_responses: true,
            clock_kinds: BTreeSet::from([RunnerClockKind::Real]),
            backend_features: BTreeSet::from(["http".to_string()]),
        })
    }
}

/// Register the open `agentic` workload without registering a backend pair.
///
/// The protocol-v2 composition root links this factory to its shared
/// `online_http` backend marker. Keeping pair registration outside this module
/// prevents a second online-backend config type from emerging.
pub fn register_agentic_workload(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    builder.register_workload(Arc::new(AgenticRunnerWorkloadFactory))
}

/// Register the executable `online_http + agentic` pair adapter.
pub fn register_agentic_online_pair(builder: &mut RunnerRegistryBuilder) -> Result<()> {
    builder.register_pair(Arc::new(AgenticOnlinePairFactory {
        tokenizers: Arc::new(NativeOnlineTokenizerSourceResolver::default()),
    }))
}

#[derive(Clone)]
struct AgenticOnlinePairFactory {
    tokenizers: Arc<dyn OnlineTokenizerSourceResolver>,
}

impl fmt::Debug for AgenticOnlinePairFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("AgenticOnlinePairFactory")
    }
}

impl RunnerPairFactory for AgenticOnlinePairFactory {
    fn backend_id(&self) -> &'static str {
        "online_http"
    }

    fn workload_id(&self) -> &'static str {
        AGENTIC_WORKLOAD_DESCRIPTOR.id
    }

    fn validate_pair(
        &self,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        validated_online_http_backend(backend)?;
        validated_agentic_workload(workload)?.validate()
    }

    fn validate_run(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: &dyn ValidatedBackendConfig,
        workload: &dyn ValidatedWorkloadConfig,
    ) -> Result<()> {
        self.validate_pair(backend, workload)?;
        validate_agentic_authored_run(run, context)
    }

    fn prepare(
        &self,
        _run: &AuthoredRunSpecV2,
        _backend: Box<dyn ValidatedBackendConfig>,
        _workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        Err(anyhow!(
            "online_http + agentic preparation requires the coordinator-owned runner context"
        ))
    }

    fn prepare_with_context(
        &self,
        run: &AuthoredRunSpecV2,
        context: &RunnerRunContext,
        backend: Box<dyn ValidatedBackendConfig>,
        workload: Box<dyn ValidatedWorkloadConfig>,
    ) -> Result<Box<dyn PreparedRunnerOperation>> {
        self.validate_run(run, context, backend.as_ref(), workload.as_ref())?;
        let workload = validated_agentic_workload(workload.as_ref())?.clone();
        let tokenizer = lower_authored_tokenizer(&workload.tokenizer, self.tokenizers.as_ref())?;
        let endpoint = context.default_endpoint_profile()?;
        let spec = AgenticOnlineExecutionSpec {
            benchmark_id: run.identity.benchmark_id.clone(),
            artifact_target: run.artifact_target.clone(),
            model: run.models.items[0].name.clone(),
            workload,
            tokenizer,
            endpoint: AgenticOnlineEndpointSpec {
                endpoint_id: endpoint.endpoint_id.clone(),
                config: endpoint.config.clone(),
                connection_reuse: endpoint.connection_reuse,
                session_header: endpoint.session_header.clone(),
                http2: endpoint.http2,
            },
        };
        spec.validate()?;
        Ok(Box::new(PreparedAgenticOnlineOperation {
            spec,
            registry: context.product_registry_handle(),
        }))
    }
}

fn validated_online_http_backend(
    config: &dyn ValidatedBackendConfig,
) -> Result<&OnlineHttpBackendConfigV2> {
    ValidatedBackendConfig::as_any(config)
        .downcast_ref::<OnlineHttpBackendConfigV2>()
        .ok_or_else(|| anyhow!("agentic pair received the wrong online_http backend config type"))
}

fn validated_agentic_workload(
    config: &dyn ValidatedWorkloadConfig,
) -> Result<&AgenticWorkloadConfigV2> {
    ValidatedWorkloadConfig::as_any(config)
        .downcast_ref::<AgenticWorkloadConfigV2>()
        .ok_or_else(|| anyhow!("agentic pair received the wrong workload config type"))
}

fn validate_agentic_authored_run(
    run: &AuthoredRunSpecV2,
    context: &RunnerRunContext,
) -> Result<()> {
    ensure!(
        run.models.items.len() == 1,
        "agentic workloads require exactly one model"
    );
    ensure!(
        context.endpoint_profiles().len() == 1,
        "agentic workloads require exactly one endpoint profile named \"default\""
    );
    let endpoint = context.default_endpoint_profile()?;
    let factory = context
        .product_registry()
        .endpoints()
        .resolve_factory(&endpoint.endpoint_id)
        .context("resolving validated agentic endpoint")?;
    let descriptor = factory.descriptor();
    ensure!(
        descriptor.produces_tokens
            && descriptor.supports_streaming
            && descriptor.input_modalities.contains(&Modality::Text),
        "agentic endpoint {:?} must consume text and produce streaming tokens",
        descriptor.id
    );
    ensure!(
        endpoint.config.wait_for_model_timeout <= 0.0,
        "agentic protocol-v2 execution does not yet implement endpoint readiness probes"
    );
    ensure!(
        run.metrics.slice_duration_seconds.is_none() && run.metrics.slos.is_empty(),
        "agentic protocol-v2 execution does not consume run-level metric slicing or SLOs"
    );
    ensure!(
        run.artifacts.records_path.is_none()
            && run.artifacts.raw_path.is_none()
            && run.artifacts.outputs_path.is_none()
            && !run.artifacts.trace
            && run.artifacts.user_files.is_empty(),
        "agentic protocol-v2 execution currently writes only its canonical native-v2 and provider artifacts"
    );
    Ok(())
}

struct PreparedAgenticOnlineOperation {
    spec: AgenticOnlineExecutionSpec,
    registry: Arc<AiperfRegistry>,
}

impl fmt::Debug for PreparedAgenticOnlineOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedAgenticOnlineOperation")
            .field("benchmark_id", &self.spec.benchmark_id)
            .field("artifact_target", &self.spec.artifact_target)
            .field("model", &self.spec.model)
            .finish_non_exhaustive()
    }
}

impl PreparedRunnerOperation for PreparedAgenticOnlineOperation {
    fn execute(self: Box<Self>) -> Result<PreparedRunOutcome> {
        let report =
            execute_agentic_online_report_with_registry(&self.spec, self.registry.as_ref())?;
        Ok(PreparedRunOutcome {
            native_report: report.native_report,
            report_facts: ReportPairRunFacts::new(),
            provenance: BTreeMap::from([
                ("backend".to_owned(), "online_http".to_owned()),
                ("workload".to_owned(), "agentic".to_owned()),
            ]),
        })
    }
}

/// Open endpoint binding plus transport policy for one agentic online run.
#[derive(Clone, Debug)]
pub struct AgenticOnlineEndpointSpec {
    /// Open endpoint factory ID or registered alias.
    pub endpoint_id: EndpointId,
    /// Identity-free endpoint policy. Agentic execution enables streaming usage.
    pub config: RawEndpointConfig,
    /// HTTP connection reuse/affinity policy.
    pub connection_reuse: ConnectionReuseStrategy,
    /// Optional replacement session-affinity header.
    pub session_header: Option<String>,
    /// Whether cleartext endpoints use HTTP/2 prior knowledge.
    pub http2: bool,
}

/// Completely authored input to the online agentic execution composition.
#[derive(Debug)]
pub struct AgenticOnlineExecutionSpec {
    /// Stable run identifier used by the subprocess terminal.
    pub benchmark_id: String,
    /// Exclusive artifact directory selected by Python.
    pub artifact_target: PathBuf,
    /// Primary model supplied to canonical agent providers.
    pub model: String,
    /// Strict factory-owned workload config.
    pub workload: AgenticWorkloadConfigV2,
    /// Fully resolved local or built-in tokenizer source.
    pub tokenizer: TokenizerSpec,
    /// Selected endpoint profile and HTTP policy.
    pub endpoint: AgenticOnlineEndpointSpec,
}

impl AgenticOnlineExecutionSpec {
    /// Validate common fields without filesystem or process side effects.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.benchmark_id.trim().is_empty(),
            "agentic benchmark_id must not be empty"
        );
        ensure!(
            !self.artifact_target.as_os_str().is_empty(),
            "agentic artifact_target must not be empty"
        );
        ensure!(
            !self.model.trim().is_empty(),
            "agentic model must not be empty"
        );
        ensure!(
            !self.tokenizer.name.trim().is_empty(),
            "agentic resolved tokenizer name must not be empty"
        );
        ensure!(
            !self.tokenizer.apply_chat_template,
            "agentic apply_chat_template is not yet represented by the dynamic turn builder"
        );
        ensure!(
            !self.endpoint.config.urls.is_empty(),
            "agentic endpoint must contain at least one URL"
        );
        self.workload.validate()
    }
}

/// Successful report commit plus the in-memory canonical result.
#[derive(Debug)]
pub struct AgenticExecutionOutcome {
    /// Authoritative native-v2 report path.
    pub report_path: PathBuf,
    /// Full canonical agentic report retained for embedded callers/tests.
    pub report: AgenticRunReport,
}

/// Process-construction seam for the canonical evaluator.
#[async_trait(?Send)]
pub trait AgenticEvaluatorProcessFactory {
    /// Spawn, supervise, and negotiate one evaluator worker.
    async fn spawn(&self, spec: &AgenticEvaluatorProcessSpec) -> Result<Box<dyn AgenticEvaluator>>;
}

/// Native stdio evaluator factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAgenticEvaluatorProcessFactory;

#[async_trait(?Send)]
impl AgenticEvaluatorProcessFactory for NativeAgenticEvaluatorProcessFactory {
    async fn spawn(&self, spec: &AgenticEvaluatorProcessSpec) -> Result<Box<dyn AgenticEvaluator>> {
        let mut process = WorkerProcessConfig::new(spec.python_executable.as_os_str())
            .arg("-u")
            .arg("-m")
            .arg(&spec.worker_module);
        for (key, value) in &spec.environment {
            process = process.env(key, value);
        }
        if let Some(current_dir) = &spec.current_dir {
            process = process.current_dir(current_dir);
        }
        let evaluator = PythonEvaluator::spawn(process)
            .await
            .context("starting canonical Python agentic evaluator")?;
        Ok(Box::new(evaluator))
    }
}

/// Construction seam for sandbox-facing auxiliary inference ingress.
#[async_trait(?Send)]
pub trait AgenticGatewayFactory {
    /// Bind one per-run gateway and return exclusive ownership to the workload.
    async fn bind(
        &self,
        advertised_host: &str,
        max_tokens: usize,
    ) -> Result<Box<dyn AgenticInferenceGateway>>;
}

/// Native authenticated HTTP callback ingress.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAgenticGatewayFactory;

#[async_trait(?Send)]
impl AgenticGatewayFactory for NativeAgenticGatewayFactory {
    async fn bind(
        &self,
        advertised_host: &str,
        max_tokens: usize,
    ) -> Result<Box<dyn AgenticInferenceGateway>> {
        Ok(Box::new(
            HttpAgenticInferenceGateway::bind(advertised_host, max_tokens)
                .await
                .context("starting Rust agentic auxiliary-inference ingress")?,
        ))
    }
}

/// Tokenizer construction seam for dynamic evaluator-authored calls.
pub trait AgenticTokenizerFactory {
    /// Resolve one built-in or local tokenizer before the worker is loaded.
    fn build(&self, name: &str) -> Result<Arc<dyn TextTokenizer>>;
}

/// Native tiktoken/Hugging Face tokenizer factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeAgenticTokenizerFactory;

impl AgenticTokenizerFactory for NativeAgenticTokenizerFactory {
    fn build(&self, name: &str) -> Result<Arc<dyn TextTokenizer>> {
        let path = Path::new(name);
        if path.is_dir() {
            return Ok(Arc::new(HuggingFaceTokenizer::from_directory(path)?));
        }
        if path.is_file() {
            return Ok(Arc::new(HuggingFaceTokenizer::from_file(path)?));
        }
        let encoding = name.parse::<TiktokenEncoding>()?;
        Ok(Arc::new(TiktokenTokenizer::new(encoding)))
    }
}

/// Execute and atomically commit native-v2 with stock in-tree factories.
pub fn execute_agentic_online(spec: AgenticOnlineExecutionSpec) -> Result<AgenticExecutionOutcome> {
    execute_agentic_online_with_factories(
        spec,
        &BuiltinAiperfRegistryFactory,
        &NativeHttpExecutionBackendFactory,
        &NativeAgenticEvaluatorProcessFactory,
        &NativeAgenticGatewayFactory,
        &NativeAgenticTokenizerFactory,
    )
}

/// Execute and commit native-v2 with every replaceable composition seam injected.
#[allow(clippy::too_many_arguments)]
pub fn execute_agentic_online_with_factories(
    spec: AgenticOnlineExecutionSpec,
    registry_factory: &dyn AiperfRegistryFactory,
    backend_factory: &dyn HttpExecutionBackendFactory,
    evaluator_factory: &dyn AgenticEvaluatorProcessFactory,
    gateway_factory: &dyn AgenticGatewayFactory,
    tokenizer_factory: &dyn AgenticTokenizerFactory,
) -> Result<AgenticExecutionOutcome> {
    let registry = registry_factory
        .build()
        .context("building injected AIPerf registry for agentic run")?;
    execute_agentic_online_with_registry_and_factories(
        spec,
        &registry,
        backend_factory,
        evaluator_factory,
        gateway_factory,
        tokenizer_factory,
    )
}

/// Execute and commit native-v2 through one already frozen product registry.
pub fn execute_agentic_online_with_registry(
    spec: AgenticOnlineExecutionSpec,
    registry: &AiperfRegistry,
) -> Result<AgenticExecutionOutcome> {
    execute_agentic_online_with_registry_and_factories(
        spec,
        registry,
        &NativeHttpExecutionBackendFactory,
        &NativeAgenticEvaluatorProcessFactory,
        &NativeAgenticGatewayFactory,
        &NativeAgenticTokenizerFactory,
    )
}

/// Execute and commit native-v2 with a coordinator-owned product registry and
/// every remaining process seam injected.
#[allow(clippy::too_many_arguments)]
pub fn execute_agentic_online_with_registry_and_factories(
    spec: AgenticOnlineExecutionSpec,
    registry: &AiperfRegistry,
    backend_factory: &dyn HttpExecutionBackendFactory,
    evaluator_factory: &dyn AgenticEvaluatorProcessFactory,
    gateway_factory: &dyn AgenticGatewayFactory,
    tokenizer_factory: &dyn AgenticTokenizerFactory,
) -> Result<AgenticExecutionOutcome> {
    let report = execute_agentic_online_report_with_registry_and_factories(
        &spec,
        registry,
        backend_factory,
        evaluator_factory,
        gateway_factory,
        tokenizer_factory,
    )?;
    let report_path = spec.artifact_target.join("native-v2.json");
    write_native_report_json(&report.native_report, &report_path)?;
    Ok(AgenticExecutionOutcome {
        report_path,
        report,
    })
}

fn execute_agentic_online_report_with_registry(
    spec: &AgenticOnlineExecutionSpec,
    registry: &AiperfRegistry,
) -> Result<AgenticRunReport> {
    execute_agentic_online_report_with_registry_and_factories(
        spec,
        registry,
        &NativeHttpExecutionBackendFactory,
        &NativeAgenticEvaluatorProcessFactory,
        &NativeAgenticGatewayFactory,
        &NativeAgenticTokenizerFactory,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_agentic_online_report_with_registry_and_factories(
    spec: &AgenticOnlineExecutionSpec,
    registry: &AiperfRegistry,
    backend_factory: &dyn HttpExecutionBackendFactory,
    evaluator_factory: &dyn AgenticEvaluatorProcessFactory,
    gateway_factory: &dyn AgenticGatewayFactory,
    tokenizer_factory: &dyn AgenticTokenizerFactory,
) -> Result<AgenticRunReport> {
    spec.validate()?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating agentic runner Tokio runtime")?;
    let local = tokio::task::LocalSet::new();
    local.block_on(
        &runtime,
        execute_agentic_online_async_with_registry_and_factories(
            spec,
            registry,
            backend_factory,
            evaluator_factory,
            gateway_factory,
            tokenizer_factory,
        ),
    )
}

/// Async composition used by an already-created current-thread runner runtime.
#[allow(clippy::too_many_arguments)]
pub async fn execute_agentic_online_async_with_factories(
    spec: &AgenticOnlineExecutionSpec,
    registry_factory: &dyn AiperfRegistryFactory,
    backend_factory: &dyn HttpExecutionBackendFactory,
    evaluator_factory: &dyn AgenticEvaluatorProcessFactory,
    gateway_factory: &dyn AgenticGatewayFactory,
    tokenizer_factory: &dyn AgenticTokenizerFactory,
) -> Result<AgenticRunReport> {
    let registry = registry_factory
        .build()
        .context("building injected AIPerf registry for agentic run")?;
    execute_agentic_online_async_with_registry_and_factories(
        spec,
        &registry,
        backend_factory,
        evaluator_factory,
        gateway_factory,
        tokenizer_factory,
    )
    .await
}

/// Async agentic composition through one already frozen product registry.
#[allow(clippy::too_many_arguments)]
pub async fn execute_agentic_online_async_with_registry_and_factories(
    spec: &AgenticOnlineExecutionSpec,
    registry: &AiperfRegistry,
    backend_factory: &dyn HttpExecutionBackendFactory,
    evaluator_factory: &dyn AgenticEvaluatorProcessFactory,
    gateway_factory: &dyn AgenticGatewayFactory,
    tokenizer_factory: &dyn AgenticTokenizerFactory,
) -> Result<AgenticRunReport> {
    spec.validate()?;
    let tokenizer = tokenizer_factory
        .build(&spec.tokenizer.name)
        .context("loading agentic tokenizer")?;
    let endpoint = prepare_agentic_endpoint(registry, &spec.endpoint)?;
    let model_concurrency = spec.workload.model_concurrency()?;
    let evaluator_config = spec.workload.evaluator_config(&spec.artifact_target)?;

    let mut evaluator = evaluator_factory.spawn(&spec.workload.evaluator).await?;
    if !evaluator.supports_agentic() {
        let shutdown = evaluator.shutdown().await;
        let error =
            anyhow!("canonical evaluator worker does not advertise a pinned agentic provider");
        return match shutdown {
            Ok(()) => Err(error),
            Err(shutdown) => Err(error.context(format!(
                "evaluator also failed during capability-error shutdown: {shutdown}"
            ))),
        };
    }
    if !evaluator.supports_agentic_inference_gateway() {
        let shutdown = evaluator.shutdown().await;
        let error =
            anyhow!("canonical evaluator worker does not advertise agentic_inference_gateway");
        return match shutdown {
            Ok(()) => Err(error),
            Err(shutdown) => Err(error.context(format!(
                "evaluator also failed during gateway-capability shutdown: {shutdown}"
            ))),
        };
    }
    let worker_identity = evaluator.identity().clone();
    let gateway_host = resolve_advertised_host(spec.workload.inference_gateway_host.as_deref())?;
    let gateway = match gateway_factory
        .bind(&gateway_host, spec.workload.max_tokens)
        .await
    {
        Ok(gateway) => gateway,
        Err(error) => {
            let shutdown = evaluator.shutdown().await;
            return match shutdown {
                Ok(()) => Err(error),
                Err(shutdown) => Err(error.context(format!(
                    "evaluator also failed while unwinding gateway setup: {shutdown}"
                ))),
            };
        }
    };
    let turn_builder: Rc<dyn AgenticTurnBuilder> = Rc::new(DatasetAgenticTurnBuilder::prepared(
        spec.model.clone(),
        tokenizer,
        endpoint.table.clone(),
        endpoint.reference.clone(),
    )?);
    let workload = AgenticWorkload::prepare_with_gateway(
        evaluator,
        &spec.workload.dataset,
        &spec.model,
        &evaluator_config,
        model_concurrency,
        turn_builder,
        Some(gateway),
    )
    .await?;
    let evaluator_identity = workload.identity().clone();

    // The measurement origin begins only after the canonical task set and all
    // endpoint/backend state are frozen.
    let real_clock_anchor = RealClockAnchor::now();
    let clock: Rc<dyn Clock> = RealClock::from_anchor(real_clock_anchor);
    let execution_backend = match backend_factory.build(HttpExecutionBackendConfig {
        workers: spec.workload.worker_count,
        coordinator_clock: clock.clone(),
        real_clock_anchor,
        base_urls: endpoint.urls,
        model: spec.model.clone(),
        transport: endpoint.transport,
        prepared_endpoints: Some(endpoint.table_factory),
    }) {
        Ok(backend) => backend,
        Err(error) => {
            let worker_shutdown = workload.shutdown().await;
            return combine_run_shutdown(Err(error), Ok(()), worker_shutdown);
        }
    };

    // Evaluator negotiation, provider/task loading, gateway binding, endpoint
    // preparation, tokenizer acquisition, and every execution worker are now
    // frozen. Only a runnable benchmark is allowed to create its artifact root.
    if let Err(error) = std::fs::create_dir_all(&spec.artifact_target).with_context(|| {
        format!(
            "creating agentic artifact target {}",
            spec.artifact_target.display()
        )
    }) {
        let backend_shutdown = execution_backend.shutdown();
        let worker_shutdown = workload.shutdown().await;
        return combine_run_shutdown(Err(error), backend_shutdown, worker_shutdown);
    }
    let start_ns = clock.now_ns();
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(AgenticDispatcher {
        execution_backend: execution_backend.clone(),
        model: spec.model.clone(),
    });
    let scheduled_workload: Rc<dyn Workload> = workload.clone();
    let run_result = async {
        execution_backend.set_run_origin(start_ns)?;
        let scheduled = run_scheduled_workload(
            scheduled_workload,
            clock,
            start_ns,
            dispatcher,
            StopConfig::default(),
            false,
        )
        .await?;
        let results = workload.results()?;
        finalize_agentic_report(
            &spec.workload.dataset,
            &spec.model,
            model_concurrency,
            scheduled,
            worker_identity,
            evaluator_identity,
            workload.config(),
            results,
        )
    }
    .await;
    let backend_shutdown = execution_backend.shutdown();
    let worker_shutdown = workload.shutdown().await;
    combine_run_shutdown(run_result, backend_shutdown, worker_shutdown)
}

fn combine_run_shutdown<T>(run: Result<T>, backend: Result<()>, worker: Result<()>) -> Result<T> {
    match (run, backend, worker) {
        (Ok(value), Ok(()), Ok(())) => Ok(value),
        (Err(error), Ok(()), Ok(())) => Err(error),
        (Ok(_), Err(error), Ok(())) => Err(error.context("shutting down agentic HTTP backend")),
        (Ok(_), Ok(()), Err(error)) => {
            Err(error.context("shutting down agentic evaluator/gateway"))
        }
        (Ok(_), Err(backend), Err(worker)) => Err(backend.context(format!(
            "agentic evaluator/gateway also failed during shutdown: {worker:#}"
        ))),
        (Err(error), backend, worker) => {
            let mut details = Vec::new();
            if let Err(backend) = backend {
                details.push(format!("HTTP backend shutdown failed: {backend:#}"));
            }
            if let Err(worker) = worker {
                details.push(format!("evaluator/gateway shutdown failed: {worker:#}"));
            }
            if details.is_empty() {
                Err(error)
            } else {
                Err(error.context(details.join("; ")))
            }
        }
    }
}

struct PreparedAgenticEndpoint {
    table: Rc<PreparedEndpointTable>,
    reference: PreparedEndpointReference,
    table_factory: Arc<dyn HttpPreparedEndpointTableFactory>,
    urls: Vec<String>,
    transport: TransportSinkConfig,
}

#[derive(Clone)]
struct AgenticPreparedEndpointTableFactory {
    registry: EndpointRegistry,
    endpoint_id: EndpointId,
    config: RawEndpointConfig,
}

impl fmt::Debug for AgenticPreparedEndpointTableFactory {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AgenticPreparedEndpointTableFactory")
            .field("endpoint_id", &self.endpoint_id)
            .finish_non_exhaustive()
    }
}

impl HttpPreparedEndpointTableFactory for AgenticPreparedEndpointTableFactory {
    fn prepare_worker(&self) -> Result<PreparedEndpointTable> {
        let endpoint = self
            .registry
            .prepare(&self.endpoint_id, self.config.clone())
            .context("preparing worker-local agentic endpoint profile")?;
        ensure_agentic_endpoint(endpoint.descriptor())?;
        let mut table = PreparedEndpointTable::new();
        let key = table.push(endpoint)?;
        ensure!(
            key == EndpointKey::from_index(0),
            "single agentic endpoint must occupy dense key zero"
        );
        Ok(table)
    }
}

fn prepare_agentic_endpoint(
    registry: &AiperfRegistry,
    spec: &AgenticOnlineEndpointSpec,
) -> Result<PreparedAgenticEndpoint> {
    let factory = registry
        .endpoints()
        .resolve_factory(&spec.endpoint_id)
        .context("resolving agentic endpoint factory")?;
    let descriptor = factory.descriptor();
    ensure_agentic_endpoint(descriptor)?;
    let canonical_endpoint_id = EndpointId::new(descriptor.id)
        .context("validating canonical agentic endpoint descriptor ID")?;
    let mut raw = spec.config.clone();
    raw.streaming = true;
    raw.use_server_token_count = true;
    raw.use_legacy_max_tokens = true;
    let table_factory = Arc::new(AgenticPreparedEndpointTableFactory {
        registry: registry.endpoints().clone(),
        endpoint_id: spec.endpoint_id.clone(),
        config: raw,
    });
    let table = Rc::new(table_factory.prepare_worker()?);
    let reference = PreparedEndpointReference {
        key: EndpointKey::from_index(0),
        endpoint_id: canonical_endpoint_id,
    };
    let prepared = table.get(reference.key)?;
    ensure!(
        prepared.descriptor().id == reference.endpoint_id.as_str(),
        "agentic prepared endpoint key resolved to {:?}, expected {:?}",
        prepared.descriptor().id,
        reference.endpoint_id.as_str()
    );
    let effective_raw = prepared.config().to_raw();
    ensure!(
        effective_raw.streaming,
        "agentic endpoint {:?} does not support streaming semantic responses",
        descriptor.id
    );
    let urls = effective_raw.urls.clone();
    ensure!(!urls.is_empty(), "agentic endpoint has no URL");
    let timeout_ns = seconds_to_ns(effective_raw.timeout_seconds)?;
    let transport = TransportSinkConfig {
        client: ClientConfig {
            http_version: if spec.http2 {
                HttpVersion::Http2PriorKnowledge
            } else {
                HttpVersion::Auto
            },
            total_timeout_ns: (timeout_ns > 0).then_some(timeout_ns),
            ..ClientConfig::default()
        },
        connection_reuse: spec.connection_reuse,
        session_header: spec.session_header.clone(),
    };
    Ok(PreparedAgenticEndpoint {
        table,
        reference,
        table_factory,
        urls,
        transport,
    })
}

fn ensure_agentic_endpoint(descriptor: &aiperf_endpoints::EndpointDescriptor) -> Result<()> {
    ensure!(
        descriptor.produces_tokens
            && descriptor.supports_streaming
            && descriptor.input_modalities.contains(&Modality::Text),
        "agentic endpoint {:?} must consume text and produce streaming tokens",
        descriptor.id
    );
    Ok(())
}

fn seconds_to_ns(seconds: f64) -> Result<i64> {
    ensure!(
        seconds.is_finite() && seconds >= 0.0 && seconds * 1_000_000_000.0 <= i64::MAX as f64,
        "agentic endpoint timeout must be finite, non-negative, and representable in nanoseconds"
    );
    Ok((seconds * 1_000_000_000.0).round_ties_even() as i64)
}

struct AgenticDispatcher {
    execution_backend: Rc<dyn HttpTurnExecutionBackend>,
    model: String,
}

#[async_trait(?Send)]
impl TurnDispatcher for AgenticDispatcher {
    fn inference_dimensions(&self, turn: &TurnToSend) -> aiperf_metrics::InferenceDimensions {
        self.execution_backend.inference_dimensions(turn)
    }

    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        Ok(self
            .execution_backend
            .execute_turn(
                PreparedHttpTurn::from_turn(turn, &self.model),
                observer,
                on_first_token,
            )
            .await?
            .outcome)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use serde_json::json;

    use super::*;

    fn valid_config() -> AgenticWorkloadConfigV2 {
        serde_json::from_value(json!({
            "worker_count": 2,
            "dataset": "harbor/swebench@sha256:locked",
            "tokenizer": {"name": "builtin"},
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "concurrency": 4
            }],
            "evaluator": {"python_executable": "/usr/bin/python3"},
            "task_concurrency": 2,
            "inference_gateway_host": "127.0.0.1"
        }))
        .unwrap()
    }

    #[test]
    fn v2_config_consumes_projection_fields_and_derives_model_admission() {
        let config = valid_config();
        config.validate().unwrap();
        assert_eq!(config.worker_count, 2);
        assert_eq!(config.model_concurrency().unwrap(), 4);
        assert_eq!(config.max_tokens, 4_096);
        assert_eq!(config.context_window, 131_072);
    }

    #[test]
    fn v2_config_rejects_outer_schedule_policy_and_unknown_fields() {
        let mut value = serde_json::to_value(json!({
            "worker_count": 1,
            "dataset": "fixture",
            "tokenizer": {},
            "phases": [{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "concurrency": 1,
                "duration": 1.0
            }],
            "evaluator": {"python_executable": "/usr/bin/python3"}
        }))
        .unwrap();
        let config: AgenticWorkloadConfigV2 = serde_json::from_value(value.clone()).unwrap();
        assert!(
            config
                .validate()
                .unwrap_err()
                .to_string()
                .contains("harnesses own")
        );
        value["unknown"] = json!(true);
        assert!(serde_json::from_value::<AgenticWorkloadConfigV2>(value).is_err());
    }

    #[test]
    fn workload_factory_advertises_semantic_real_http_requirements() {
        let raw = RawValue::from_string(
            serde_json::to_string(&json!({
                "worker_count": 1,
                "dataset": "fixture",
                "tokenizer": {},
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "concurrency": 1
                }],
                "evaluator": {"python_executable": "/usr/bin/python3"}
            }))
            .unwrap(),
        )
        .unwrap();
        let factory = AgenticRunnerWorkloadFactory;
        let validated = factory.validate(&raw).unwrap();
        let requirements = factory.requirements(validated.as_ref()).unwrap();
        assert!(requirements.semantic_responses);
        assert_eq!(
            requirements.clock_kinds,
            BTreeSet::from([RunnerClockKind::Real])
        );
        assert_eq!(
            requirements.backend_features,
            BTreeSet::from(["http".into()])
        );
    }

    #[derive(Default)]
    struct RecordingTokenizerResolver {
        request: Mutex<Option<(String, String, bool)>>,
    }

    impl OnlineTokenizerSourceResolver for RecordingTokenizerResolver {
        fn resolve(&self, name: &str, revision: &str, trust_remote_code: bool) -> Result<String> {
            *self.request.lock().unwrap() =
                Some((name.to_owned(), revision.to_owned(), trust_remote_code));
            Ok("/cache/tokenizers/resolved-commit".into())
        }
    }

    #[test]
    fn agentic_tokenizer_is_resolved_during_pair_preparation() {
        let mut config = valid_config();
        config.tokenizer = RawValue::from_string(
            serde_json::to_string(&json!({
                "name": "fixture/remote-tokenizer",
                "revision": "0123456789abcdef0123456789abcdef01234567",
                "trust_remote_code": false,
                "apply_chat_template": false
            }))
            .unwrap(),
        )
        .unwrap();
        config.validate().unwrap();
        let resolver = RecordingTokenizerResolver::default();
        let resolved = lower_authored_tokenizer(&config.tokenizer, &resolver).unwrap();
        assert_eq!(resolved.name, "/cache/tokenizers/resolved-commit");
        assert_eq!(
            resolver.request.into_inner().unwrap(),
            Some((
                "fixture/remote-tokenizer".into(),
                "0123456789abcdef0123456789abcdef01234567".into(),
                false
            ))
        );
    }
}
