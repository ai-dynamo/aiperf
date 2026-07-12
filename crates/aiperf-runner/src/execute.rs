// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native construction and execution of one resolved benchmark run.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use aiperf::accuracy::{
    AccuracyDataset, AccuracyRecordProcessor, accuracy_report_errors, grade_accuracy_responses,
    load_evaluator_problems_with_grader,
};
use aiperf::adaptive::{
    AdaptiveControlVariable, AdaptiveRunConfig, AdaptiveStepConfig, build_adaptive_with_origins,
    positive_seconds_to_ns,
};
use aiperf::ancillary::RATE_RAMP_UPDATE_INTERVAL_NS;
use aiperf::fixed_schedule::{
    DatasetFixedScheduleSource, FixedScheduleConfig, FixedScheduleWorkload,
};
use aiperf::http::{HttpTurnExecutionBackend, PreparedHttpTurn, TransportSinkConfig};
use aiperf::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use aiperf::multiturn::{
    ConversationSource, EndpointInputTokenCounter, InputTokenCounter, IssuedCredit,
    NativeDatasetConversationSource, PreparedEndpointReference, PreparedEndpointTableResolver,
    PreparedTurnEndpointResolver, TurnToSend,
};
use aiperf::phase_runtime::{
    RampScheduledPhaseController, ScheduledPhaseController, ScheduledPhasePlan,
    ScheduledPhaseResources, ScheduledRuntimeExtension, ScheduledRuntimeExtensionParts,
    SlotPoolPhaseResources, run_scheduled_phases,
};
use aiperf::report::write_native_report_json;
use aiperf::request_rate::RequestRateWorkload;
use aiperf::scheduled::{
    IssuanceGate, ScheduledAncillaryPolicies, TurnDispatchOutcome, TurnDispatcher,
    TurnRecordProcessor, Workload,
};
use aiperf::user_centric::{UserCentricConfig, UserCentricWorkload};
use aiperf_accuracy::{
    AccuracyEvaluator, EvaluatorLoadConfig, EvaluatorLoadResult, PythonEvaluator,
    WorkerProcessConfig,
};
use aiperf_adaptive::{AdaptiveScale, CorrelationContext, SlaFilter, UserTarget};
use aiperf_clock::{Clock, RealClock, RealClockAnchor};
use aiperf_dataset::{
    ComposeConfig, Dataset, DatasetSource, HuggingFaceTokenizer, LoadConfig,
    MaterializedTracePromptStorage, ModelId, ModelSelector, ModelSelectorFactory,
    RandomModelSelectorFactory, RoundRobinModelSelectorFactory, SourceImageSampling,
    SyntheticAudioConfig, SyntheticAudioFormat, SyntheticDatasetConfig, SyntheticImageConfig,
    SyntheticImageFormat, SyntheticImageSource, SyntheticPrefixConfig, SyntheticPromptConfig,
    SyntheticRankingsConfig, SyntheticVideoAudioConfig, SyntheticVideoConfig, SyntheticVideoFormat,
    SyntheticVideoPattern, TextTokenizer, TiktokenEncoding, TiktokenTokenizer,
    TracePromptStoragePolicy, TraceSynthesisConfig,
};
use aiperf_endpoints::{
    EndpointConfig, EndpointKey, EndpointRegistry, EndpointType, PreparedEndpointTable,
};
use aiperf_extensions::{AiperfRegistry, AiperfRegistryFactory, BuiltinAiperfRegistryFactory};
use aiperf_graph::input::{
    GraphInputAdapterRegistry, GraphInputAdapterResolver, GraphInputBundle, GraphInputConfig,
};
use aiperf_metrics::{
    CATALOG, ExportContext, InferenceDimensions, MetricTag, MetricsAccumulator, MetricsConfig,
    NativeReport, Phase as MetricsPhase, ReportRunInfo, ReportSummary, RunOutcome, SloThreshold,
};
use aiperf_rng::{
    EmpiricalPoint, PeakEntry, RandomGenerator, RngRoot, SamplingDistribution,
    SequenceLengthDistribution, SequenceLengthPair, namespace,
};
use aiperf_timing::{
    BernoulliFixedDelay, CancellationPolicy, ExponentialRamp, GracePeriod, LinearRamp,
    NoopPhaseObserver, PhaseConfig, PhaseKind, PhaseObserver, PoissonRamp, RampDriver,
    RampStrategy, RamperConfig, RoundRobinUrlSelector, SlotPool, StopConfig, UrlSelector,
    make_interval_generator,
};
use aiperf_transport_http::config::ClientConfig;
use aiperf_transport_http::models::HttpVersion;
use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use uuid::Uuid;

use crate::dataset_input::PreparedDatasetInput;
use crate::execution_factories::RunnerExecutionFactories;
use crate::gpu_telemetry::GpuTelemetryRun;
use crate::graph_execution::{
    LegacyRunnerGraphEndpointRuntimeFactory, NativeRunnerGraphPlacementFactory,
    PreparedRunnerGraphEndpointRuntimeFactory, RunnerGraphBackendFactory,
    RunnerGraphBackendFactoryConfig, RunnerGraphEndpointRuntimeFactory,
    RunnerGraphPlacementFactory,
};
use crate::graph_phase_runtime::{
    GraphPhaseBackendConfig, PreparedGraphPhaseBackend, RunnerGraphPhaseBackendFactory,
    run_graph_phases, validate_graph_phases,
};
use crate::live_streaming::{LiveResultsSink, PythonLiveStreamingRun, live_phase_observer};
use crate::network_latency::NetworkLatencyRun;
use crate::protocol::{
    AccuracySpec, AdaptiveControlVariableSpec, AdaptiveScaleSpec, AdaptiveStepPolicySpec,
    DatasetSpec, DistributionSpec, EndpointSpec, FileDatasetSpec, MetricsSpec,
    ModelSelectionStrategy, ModelsSpec, PhaseSpec, PublicDatasetSourceSpec, PublicDatasetSpec,
    RampSpec, RampStrategySpec, RunRequest, RunTerminal, SequenceDistributionEntrySpec,
    SourceImageSamplingSpec, SyntheticAudioFormatSpec, SyntheticAudioSpec, SyntheticDatasetSpec,
    SyntheticImageFormatSpec, SyntheticImageSpec, SyntheticPrefixPromptsSpec,
    SyntheticVideoFormatSpec, SyntheticVideoPatternSpec, SyntheticVideoSpec,
};
use crate::readiness::{PreparedOnlineReadiness, ReadinessTransportFactory};
use crate::records::{
    CapturedHttpExchange, CapturedModelOutput, CapturedRecord, write_outputs_json,
    write_raw_records_jsonl, write_records_jsonl,
};
use crate::registry::ValidatedEndpointProfileV2;
use crate::server_metrics::ServerMetricsRun;
use crate::sidecar_input::{
    GPU_TELEMETRY_SIDECAR_ID, GpuTelemetrySpec, LIVE_STREAMING_SIDECAR_ID, LiveStreamingSpec,
    NETWORK_LATENCY_SIDECAR_ID, NetworkLatencySpec, PreparedSidecarInputs,
    SERVER_METRICS_SIDECAR_ID, ServerMetricsSpec,
};
use crate::turn_execution::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, HttpPreparedEndpointTableFactory,
    NativeHttpExecutionBackendFactory,
};

type PhaseRuntimeParts = (
    Rc<dyn Workload>,
    Rc<RefCell<Box<dyn aiperf_timing::IntervalGenerator>>>,
    Option<Rc<SlotPool>>,
    Option<Rc<SlotPool>>,
    bool,
    Rc<dyn ScheduledPhaseResources>,
    Option<Rc<dyn UserTarget>>,
);

/// Admission resources shared by every phase in one scheduled run.
///
/// The same value is consumed by online HTTP and in-process offline adapters,
/// keeping cross-phase slot debt and adaptive actuator ownership above the
/// backend/clock seam.
pub(crate) struct NativeScheduledResources {
    session: Option<Rc<SlotPool>>,
    prefill: Option<Rc<SlotPool>>,
    phase: Rc<dyn ScheduledPhaseResources>,
}

/// Construct run-wide scheduled admission resources from authored phase
/// policy without selecting a transport or clock implementation.
pub(crate) fn native_scheduled_resources(phases: &[PhaseSpec]) -> NativeScheduledResources {
    let session = phases
        .iter()
        .any(|phase| {
            phase.request_arrival().is_some()
                && (phase.concurrency().is_some()
                    || phase
                        .common()
                        .adaptive_scale
                        .as_ref()
                        .is_some_and(|adaptive| {
                            matches!(
                                adaptive.control_variable,
                                AdaptiveControlVariableSpec::Concurrency
                            )
                        }))
        })
        .then(|| Rc::new(SlotPool::new(1)));
    let prefill = phases
        .iter()
        .any(|phase| {
            phase.request_arrival().is_some()
                && (phase.common().prefill_concurrency.is_some()
                    || phase
                        .common()
                        .adaptive_scale
                        .as_ref()
                        .is_some_and(|adaptive| {
                            matches!(
                                adaptive.control_variable,
                                AdaptiveControlVariableSpec::PrefillConcurrency
                            )
                        }))
        })
        .then(|| Rc::new(SlotPool::new(1)));
    let phase: Rc<dyn ScheduledPhaseResources> = Rc::new(SlotPoolPhaseResources::new(
        session.clone(),
        prefill.clone(),
    ));
    NativeScheduledResources {
        session,
        prefill,
        phase,
    }
}

/// Protocol-neutral execution plan consumed by the one native coordinator.
///
/// Protocol v1 and protocol v2 lower into this structure without serializing
/// through one another's wire DTOs. The nested policy values are shared Rust
/// value types; the process protocol discriminator is deliberately absent.
pub(crate) struct NativeRunPlan {
    pub(crate) run: NativeRunSpec,
}

/// Fully typed inputs required after a protocol implementation has validated
/// and lowered its authored request.
pub(crate) struct NativeRunSpec {
    pub(crate) benchmark_id: String,
    pub(crate) random_seed: Option<u64>,
    pub(crate) workers: usize,
    pub(crate) artifact_dir: PathBuf,
    pub(crate) models: ModelsSpec,
    pub(crate) endpoint: NativeEndpointPlan,
    pub(crate) dataset: NativeDatasetPlan,
    pub(crate) tokenizer: crate::protocol::TokenizerSpec,
    pub(crate) phases: Vec<PhaseSpec>,
    pub(crate) metrics: MetricsSpec,
    pub(crate) artifacts: crate::protocol::ArtifactSpec,
    pub(crate) sidecars: NativeSidecarPlan,
    pub(crate) user_files: Vec<crate::protocol_v2::UserFileSpecV2>,
}

/// Protocol-neutral retention of one run's already decoded sidecar inputs.
///
/// Protocol v1 keeps its compatibility values directly. Protocol v2 retains
/// the exact adapter-produced bundle from [`RunnerRunContext`](crate::registry::RunnerRunContext)
/// without projecting through v1 or decoding any body a second time.
pub(crate) enum NativeSidecarPlan {
    /// Protocol-v1 compatibility values decoded by its outer request.
    Legacy(Box<LegacyNativeSidecarInputs>),
    /// Protocol-v2 direct adapter outputs retained through execution.
    Prepared(Arc<PreparedSidecarInputs>),
}

/// Protocol-v1 compatibility sidecar values kept behind one cold-path box.
pub(crate) struct LegacyNativeSidecarInputs {
    gpu_telemetry: Option<GpuTelemetrySpec>,
    network_latency: Option<NetworkLatencySpec>,
    server_metrics: Option<ServerMetricsSpec>,
    live_streaming: Option<LiveStreamingSpec>,
}

impl NativeSidecarPlan {
    fn gpu_telemetry(&self) -> Result<Option<&GpuTelemetrySpec>> {
        match self {
            Self::Legacy(inputs) => Ok(inputs.gpu_telemetry.as_ref()),
            Self::Prepared(inputs) => inputs.get(GPU_TELEMETRY_SIDECAR_ID),
        }
    }

    fn network_latency(&self) -> Result<Option<&NetworkLatencySpec>> {
        match self {
            Self::Legacy(inputs) => Ok(inputs.network_latency.as_ref()),
            Self::Prepared(inputs) => inputs.get(NETWORK_LATENCY_SIDECAR_ID),
        }
    }

    fn server_metrics(&self) -> Result<Option<&ServerMetricsSpec>> {
        match self {
            Self::Legacy(inputs) => Ok(inputs.server_metrics.as_ref()),
            Self::Prepared(inputs) => inputs.get(SERVER_METRICS_SIDECAR_ID),
        }
    }

    pub(crate) fn live_streaming(&self) -> Result<Option<&LiveStreamingSpec>> {
        match self {
            Self::Legacy(inputs) => Ok(inputs.live_streaming.as_ref()),
            Self::Prepared(inputs) => inputs.get(LIVE_STREAMING_SIDECAR_ID),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Legacy(inputs) => {
                inputs.gpu_telemetry.is_none()
                    && inputs.network_latency.is_none()
                    && inputs.server_metrics.is_none()
                    && inputs.live_streaming.is_none()
            }
            Self::Prepared(inputs) => inputs.is_empty(),
        }
    }
}

/// Endpoint preparation selected by the source protocol.
///
/// Protocol v1 retains its closed compatibility value. Protocol v2 carries
/// normalized open endpoint profiles directly into worker-local preparation;
/// it is never projected through [`EndpointSpec`] or an [`EndpointType`].
#[derive(Clone)]
pub(crate) enum NativeEndpointPlan {
    /// Protocol-v1 compatibility policy.
    Legacy(Box<EndpointSpec>),
    /// Protocol-v2 open endpoint profiles.
    Prepared(Arc<Vec<ValidatedEndpointProfileV2>>),
}

impl NativeEndpointPlan {
    pub(crate) fn legacy(&self) -> Result<&EndpointSpec> {
        match self {
            Self::Legacy(spec) => Ok(spec),
            Self::Prepared(_) => {
                bail!("this workload has not converged on worker-local prepared endpoint bindings")
            }
        }
    }

    fn default_urls(&self) -> Result<&[String]> {
        match self {
            Self::Legacy(spec) => Ok(&spec.urls),
            Self::Prepared(profiles) => {
                Ok(&default_prepared_endpoint_profile(profiles)?.config.urls)
            }
        }
    }
}

/// Conventional profile selected when a workload does not name one explicitly.
pub(crate) const DEFAULT_ENDPOINT_PROFILE_ID: &str = "default";

/// Resolve one exact coordinator-validated profile retained by an adapter.
pub(crate) fn prepared_endpoint_profile<'a>(
    profiles: &'a [ValidatedEndpointProfileV2],
    profile_id: &str,
) -> Result<&'a ValidatedEndpointProfileV2> {
    profiles
        .iter()
        .find(|profile| profile.profile_id == profile_id)
        .ok_or_else(|| anyhow!("endpoint profile {profile_id:?} was not prepared"))
}

/// Resolve the conventional profile from the retained validated collection.
pub(crate) fn default_prepared_endpoint_profile(
    profiles: &[ValidatedEndpointProfileV2],
) -> Result<&ValidatedEndpointProfileV2> {
    prepared_endpoint_profile(profiles, DEFAULT_ENDPOINT_PROFILE_ID)
}

#[derive(Clone)]
struct NativePreparedEndpointTableFactory {
    registry: EndpointRegistry,
    profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
}

impl NativePreparedEndpointTableFactory {
    fn new(registry: EndpointRegistry, profiles: Arc<Vec<ValidatedEndpointProfileV2>>) -> Self {
        Self { registry, profiles }
    }

    fn prepare_table(&self) -> Result<PreparedEndpointTable> {
        let mut table = PreparedEndpointTable::new();
        for profile in self.profiles.iter() {
            let endpoint = self
                .registry
                .prepare(&profile.endpoint_id, profile.config.clone())
                .with_context(|| format!("preparing endpoint profile {:?}", profile.profile_id))?;
            table.push(endpoint)?;
        }
        Ok(table)
    }

    fn reference(&self, profile_id: &str) -> Result<PreparedEndpointReference> {
        let index = self
            .profiles
            .iter()
            .position(|profile| profile.profile_id == profile_id)
            .ok_or_else(|| anyhow!("endpoint profile {profile_id:?} was not prepared"))?;
        let index = u32::try_from(index).context("prepared endpoint profile index exceeds u32")?;
        Ok(PreparedEndpointReference {
            key: EndpointKey::from_index(index),
            endpoint_id: self.profiles[index as usize].endpoint_id.clone(),
        })
    }

    fn coordinator_resolver(&self) -> Result<Rc<dyn PreparedTurnEndpointResolver>> {
        let table = Rc::new(self.prepare_table()?);
        let default = self.reference(DEFAULT_ENDPOINT_PROFILE_ID)?;
        Ok(Rc::new(PreparedEndpointTableResolver::single(
            table, default,
        )?))
    }
}

impl HttpPreparedEndpointTableFactory for NativePreparedEndpointTableFactory {
    fn prepare_worker(&self) -> Result<PreparedEndpointTable> {
        self.prepare_table()
    }
}

/// Protocol-neutral dataset selection.
pub(crate) enum NativeDatasetPlan {
    /// Ordinary linear dataset composition.
    Linear(DatasetSpec),
    /// Canonical linear dataset loaded once during protocol-v2 preparation.
    PreparedLinear(PreparedDatasetInput),
    /// Canonical evaluator selection and dataset-load policy.
    StaticAccuracy(NativeStaticAccuracyPlan),
    /// Canonical Graph-IR bundle returned directly by the selected adapter.
    Graph(Box<NativeGraphDatasetPlan>),
    /// Protocol-v1 compatibility source awaiting its direct adapter load.
    ///
    /// Protocol v2 never constructs this value. Keeping the compatibility
    /// source outside [`NativeGraphDatasetPlan`] makes the prepared execution
    /// contract structurally incapable of carrying a half-lowered graph.
    AuthoredGraph(Box<AuthoredGraphDatasetPlan>),
}

/// Process coordinates selected for one static-accuracy evaluator.
///
/// This protocol-neutral value keeps protocol-v2 adapters from projecting
/// through the protocol-v1 [`AccuracySpec`] wire DTO. A future remote or
/// embedded evaluator may ignore these local-process coordinates behind
/// [`StaticAccuracyEvaluatorFactory`].
#[derive(Clone, Debug)]
pub struct StaticAccuracyEvaluatorProcessSpec {
    /// Absolute Python executable selected by the Python orchestrator.
    pub python_executable: PathBuf,
    /// Importable evaluator worker module.
    pub worker_module: String,
}

impl StaticAccuracyEvaluatorProcessSpec {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.python_executable.is_absolute(),
            "accuracy python_executable must be an absolute path"
        );
        ensure!(
            !self.worker_module.trim().is_empty(),
            "accuracy worker_module cannot be empty"
        );
        Ok(())
    }
}

/// Construction seam for the canonical static-accuracy evaluator.
///
/// The stock implementation supervises one Python JSONL worker. Alternate
/// distributions can inject a remote or embedded evaluator without changing
/// benchmark loading, scheduling, grading, or report composition.
#[async_trait(?Send)]
pub trait StaticAccuracyEvaluatorFactory: Send + Sync {
    /// Start and negotiate exactly one evaluator instance.
    async fn spawn(
        &self,
        process: &StaticAccuracyEvaluatorProcessSpec,
    ) -> Result<Box<dyn AccuracyEvaluator>>;
}

/// Native supervised-Python static evaluator factory.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeStaticAccuracyEvaluatorFactory;

#[async_trait(?Send)]
impl StaticAccuracyEvaluatorFactory for NativeStaticAccuracyEvaluatorFactory {
    async fn spawn(
        &self,
        process: &StaticAccuracyEvaluatorProcessSpec,
    ) -> Result<Box<dyn AccuracyEvaluator>> {
        let worker = WorkerProcessConfig::new(process.python_executable.as_os_str())
            .arg("-u")
            .arg("-m")
            .arg(&process.worker_module);
        let evaluator = PythonEvaluator::spawn(worker)
            .await
            .context("starting canonical Python accuracy evaluator")?;
        Ok(Box::new(evaluator))
    }
}

/// Protocol-neutral static evaluator selection and benchmark-load policy.
pub(crate) struct NativeStaticAccuracyPlan {
    pub(crate) benchmark: String,
    pub(crate) tasks: Option<Vec<String>>,
    pub(crate) n_shots: Option<usize>,
    pub(crate) enable_cot: Option<bool>,
    pub(crate) grader: Option<String>,
    pub(crate) system_prompt: Option<String>,
    pub(crate) process: StaticAccuracyEvaluatorProcessSpec,
    pub(crate) evaluator_factory: Arc<dyn StaticAccuracyEvaluatorFactory>,
}

impl NativeStaticAccuracyPlan {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.benchmark.trim().is_empty(),
            "accuracy benchmark cannot be empty"
        );
        self.process.validate()
    }
}

/// Fully prepared Graph-IR execution input.
pub(crate) struct NativeGraphDatasetPlan {
    pub(crate) input: Arc<GraphInputBundle>,
    pub(crate) random_seed: Option<u64>,
    pub(crate) default_output_tokens: usize,
}

/// Deprecated protocol-v1 graph source kept outside the prepared run shape.
pub(crate) struct AuthoredGraphDatasetPlan {
    adapter_name: String,
    input: GraphInputConfig,
    random_seed: Option<u64>,
    default_output_tokens: usize,
}

impl TryFrom<RunRequest> for NativeRunPlan {
    type Error = anyhow::Error;

    fn try_from(request: RunRequest) -> Result<Self> {
        let run = request.run;
        let mut dataset = lower_v1_dataset(run.dataset)?;
        if let Some(accuracy) = run.accuracy {
            ensure!(
                !matches!(
                    dataset,
                    NativeDatasetPlan::Graph(_) | NativeDatasetPlan::AuthoredGraph(_)
                ),
                "authored Graph-IR datasets cannot be combined with an accuracy evaluator"
            );
            dataset = NativeDatasetPlan::StaticAccuracy(lower_v1_static_accuracy(accuracy));
        }
        Ok(Self {
            run: NativeRunSpec {
                benchmark_id: run.benchmark_id,
                random_seed: run.random_seed,
                workers: run.workers,
                artifact_dir: run.artifact_dir,
                models: run.models,
                endpoint: NativeEndpointPlan::Legacy(Box::new(run.endpoint)),
                dataset,
                tokenizer: run.tokenizer,
                phases: run.phases,
                metrics: run.metrics,
                artifacts: run.artifacts,
                sidecars: NativeSidecarPlan::Legacy(Box::new(LegacyNativeSidecarInputs {
                    gpu_telemetry: run.gpu_telemetry,
                    network_latency: run.network_latency,
                    server_metrics: run.server_metrics,
                    live_streaming: run.live_streaming,
                })),
                user_files: Vec::new(),
            },
        })
    }
}

fn lower_v1_static_accuracy(spec: AccuracySpec) -> NativeStaticAccuracyPlan {
    NativeStaticAccuracyPlan {
        benchmark: spec.benchmark,
        tasks: spec.tasks,
        n_shots: spec.n_shots,
        enable_cot: spec.enable_cot,
        grader: spec.grader,
        system_prompt: spec.system_prompt,
        process: StaticAccuracyEvaluatorProcessSpec {
            python_executable: spec.python_executable,
            worker_module: spec.worker_module,
        },
        evaluator_factory: Arc::new(NativeStaticAccuracyEvaluatorFactory),
    }
}

fn lower_v1_dataset(dataset: DatasetSpec) -> Result<NativeDatasetPlan> {
    let adapter_name = match &dataset {
        DatasetSpec::File(spec) if spec.format == "dag_jsonl" => Some(spec.format.clone()),
        DatasetSpec::Public(spec) if spec.format == "dag_jsonl" => Some(spec.format.clone()),
        DatasetSpec::Synthetic(_) | DatasetSpec::File(_) | DatasetSpec::Public(_) => None,
    };
    let Some(adapter_name) = adapter_name else {
        return Ok(NativeDatasetPlan::Linear(dataset));
    };
    let (sampling, entries, synthesis) = match &dataset {
        DatasetSpec::File(spec) => (
            spec.sampling.as_str(),
            spec.entries,
            spec.synthesis.as_ref(),
        ),
        DatasetSpec::Public(spec) => (spec.sampling.as_str(), spec.entries, None),
        DatasetSpec::Synthetic(_) => unreachable!("synthetic datasets have no graph adapter"),
    };
    ensure!(
        sampling.trim().eq_ignore_ascii_case("sequential"),
        "authored Graph-IR runs currently require sequential dataset sampling; {sampling:?} would need an explicit GraphTraceSource implementation"
    );
    ensure!(
        entries != Some(0),
        "graph dataset entries must be positive when configured"
    );
    ensure!(
        synthesis.is_none(),
        "trace synthesis is not supported for authored Graph-IR datasets"
    );
    let random_seed = dataset_random_seed(&dataset);
    let default_output_tokens = default_output_tokens(&dataset)?;
    let input = graph_input_config(&dataset)?;
    Ok(NativeDatasetPlan::AuthoredGraph(Box::new(
        AuthoredGraphDatasetPlan {
            adapter_name,
            input,
            random_seed,
            default_output_tokens,
        },
    )))
}

/// Execute exactly one request with the native local execution backend.
pub fn execute_run(request: RunRequest) -> Result<RunTerminal> {
    let graph_inputs = GraphInputAdapterRegistry::with_builtin_adapters();
    execute_run_with_all_factories(
        request,
        &NativeHttpExecutionBackendFactory,
        &graph_inputs,
        &NativeRunnerGraphPlacementFactory,
        &BuiltinAiperfRegistryFactory,
    )
}

/// Execute one request with an injected HTTP execution-placement factory.
///
/// The benchmark scheduler and logical dispatcher are unchanged by this
/// choice. Distributions that need a remote data plane can inject a ZMQ, RPC,
/// or other backend here while retaining the Config-v2 wire, phases, admission,
/// adaptive control, capture, and report pipeline.
pub fn execute_run_with_backend_factory(
    request: RunRequest,
    backend_factory: &dyn HttpExecutionBackendFactory,
) -> Result<RunTerminal> {
    let graph_inputs = GraphInputAdapterRegistry::with_builtin_adapters();
    execute_run_with_all_factories(
        request,
        backend_factory,
        &graph_inputs,
        &NativeRunnerGraphPlacementFactory,
        &BuiltinAiperfRegistryFactory,
    )
}

/// Execute one request with injected HTTP placement and graph-input adapters.
pub fn execute_run_with_factories(
    request: RunRequest,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_inputs: &dyn GraphInputAdapterResolver,
) -> Result<RunTerminal> {
    execute_run_with_all_factories(
        request,
        backend_factory,
        graph_inputs,
        &NativeRunnerGraphPlacementFactory,
        &BuiltinAiperfRegistryFactory,
    )
}

/// Execute one request with every runner composition choice injected.
///
/// Scheduling, admission, dispatch, observation, and reporting remain inside
/// the single coordinator path. Factories choose only HTTP placement, direct
/// graph-input adapters, whole-trace placement, and the statically linked
/// registry universe.
pub fn execute_run_with_all_factories(
    request: RunRequest,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_inputs: &dyn GraphInputAdapterResolver,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry_factory: &dyn AiperfRegistryFactory,
) -> Result<RunTerminal> {
    let benchmark_id = request.run.benchmark_id.clone();
    let plan = NativeRunPlan::try_from(request)?;
    let registry = registry_factory
        .build()
        .context("constructing frozen runner registry")?;
    let report_path = execute_native_plan_with_factories(
        plan,
        backend_factory,
        graph_inputs,
        graph_placement,
        &registry,
    )?;
    Ok(RunTerminal::succeeded(benchmark_id, report_path))
}

/// Execute one protocol-neutral plan through the single native coordinator.
///
/// The caller supplies the already frozen product registry used while
/// preparing protocol-v2 endpoint profiles. This prevents execution from
/// silently composing a second registry universe after validation.
pub(crate) fn execute_native_plan_with_factories(
    plan: NativeRunPlan,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_inputs: &dyn GraphInputAdapterResolver,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
) -> Result<PathBuf> {
    let artifact_dir = plan.run.artifact_dir.clone();
    let native = execute_native_plan_uncommitted_with_factories(
        plan,
        backend_factory,
        graph_inputs,
        graph_placement,
        registry,
    )?;
    let report_path = artifact_dir.join("native-v2.json");
    write_native_report_json(&native, &report_path)?;
    Ok(report_path)
}

/// Execute one native plan and return its in-memory report without serializing it.
///
/// Protocol v2 uses this entry point so the process coordinator can stamp its
/// frozen registry identity and perform the sole authoritative report write.
/// Protocol v1 retains [`execute_native_plan_with_factories`] as a compatibility
/// wrapper around this same execution path.
pub(crate) fn execute_native_plan_uncommitted_with_factories(
    plan: NativeRunPlan,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_inputs: &dyn GraphInputAdapterResolver,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    validate_plan(&plan)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating native single-run Tokio runtime")?;
    let local = tokio::task::LocalSet::new();
    let sidecar_factory = BuiltinNativeSidecarResourceFactory;
    local.block_on(&runtime, async move {
        let plan = prepare_protocol_v1_graph(plan, graph_inputs).await?;
        prepare_and_execute_native(
            plan,
            backend_factory,
            graph_placement,
            registry,
            &sidecar_factory,
            None,
        )
        .await
    })
}

/// Execute a plan whose graph input, if present, is already fully prepared.
///
/// Protocol-v2 pair preparation uses this entry point. Its signature omits a
/// graph-input resolver on purpose: once the selected adapter has returned a
/// canonical [`GraphInputBundle`], the execution harness cannot load or
/// reinterpret the authored source a second time.
pub(crate) fn execute_prepared_native_plan_uncommitted_with_factories(
    plan: NativeRunPlan,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    execute_prepared_native_plan_uncommitted_with_all_factories(
        plan,
        backend_factory,
        graph_placement,
        registry,
        &BuiltinNativeSidecarResourceFactory,
    )
}

/// Execute a protocol-v2 plan through the exact coordinator-frozen factories.
///
/// Readiness was already expanded into an immutable endpoint-owned plan during
/// pair preparation. Activation happens on the run-owned Clock before the
/// exclusive artifact target is created.
pub(crate) fn execute_prepared_native_plan_uncommitted_with_execution_factories(
    plan: NativeRunPlan,
    factories: &RunnerExecutionFactories,
    registry: &AiperfRegistry,
    readiness: Box<dyn PreparedOnlineReadiness>,
) -> Result<NativeReport> {
    execute_prepared_native_plan_uncommitted_with_runtime_factories(
        plan,
        factories.http(),
        factories.graph(),
        registry,
        &BuiltinNativeSidecarResourceFactory,
        Some((readiness, factories.readiness_transport())),
    )
}

/// Execute one fully prepared plan with sidecar resource construction injected.
pub(crate) fn execute_prepared_native_plan_uncommitted_with_all_factories(
    plan: NativeRunPlan,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
    sidecar_factory: &dyn NativeSidecarResourceFactory,
) -> Result<NativeReport> {
    execute_prepared_native_plan_uncommitted_with_runtime_factories(
        plan,
        backend_factory,
        graph_placement,
        registry,
        sidecar_factory,
        None,
    )
}

fn execute_prepared_native_plan_uncommitted_with_runtime_factories(
    plan: NativeRunPlan,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
    sidecar_factory: &dyn NativeSidecarResourceFactory,
    readiness: Option<(
        Box<dyn PreparedOnlineReadiness>,
        &dyn ReadinessTransportFactory,
    )>,
) -> Result<NativeReport> {
    ensure!(
        !matches!(plan.run.dataset, NativeDatasetPlan::AuthoredGraph(_)),
        "prepared native execution cannot accept an authored Graph-IR source"
    );
    validate_plan(&plan)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating prepared native single-run Tokio runtime")?;
    let local = tokio::task::LocalSet::new();
    local.block_on(
        &runtime,
        prepare_and_execute_native(
            plan,
            backend_factory,
            graph_placement,
            registry,
            sidecar_factory,
            readiness,
        ),
    )
}

async fn prepare_protocol_v1_graph(
    mut plan: NativeRunPlan,
    graph_inputs: &dyn GraphInputAdapterResolver,
) -> Result<NativeRunPlan> {
    let NativeDatasetPlan::AuthoredGraph(source) = &plan.run.dataset else {
        return Ok(plan);
    };
    let adapter_name = source.adapter_name.clone();
    let input = GraphInputConfig {
        load: source.input.load.clone(),
        root_limit: source.input.root_limit,
    };
    let random_seed = source.random_seed;
    let default_output_tokens = source.default_output_tokens;
    let adapter = graph_inputs
        .find(&adapter_name)
        .ok_or_else(|| anyhow!("no Graph-IR input adapter is registered for {adapter_name:?}"))?;
    let tokenizer = load_tokenizer(Some(&plan.run.tokenizer.name))?;
    let input = Arc::new(
        adapter
            .load(input, tokenizer.as_ref())
            .await
            .context("loading direct protocol-v1 Graph-IR input")?,
    );
    ensure!(
        !input.plans.is_empty(),
        "authored Graph-IR input contains no root traces after root limiting"
    );
    ensure!(
        input.metadata.format == adapter_name,
        "Graph-IR adapter {:?} returned bundle format {:?}",
        adapter_name,
        input.metadata.format
    );
    plan.run.dataset = NativeDatasetPlan::Graph(Box::new(NativeGraphDatasetPlan {
        input,
        random_seed,
        default_output_tokens,
    }));
    Ok(plan)
}

fn materialize_user_files(
    artifact_dir: &Path,
    files: &[crate::protocol_v2::UserFileSpecV2],
) -> Result<()> {
    if files.is_empty() {
        return Ok(());
    }
    let root = artifact_dir
        .canonicalize()
        .with_context(|| format!("canonicalizing artifact root {}", artifact_dir.display()))?;
    for file in files {
        let relative = Path::new(&file.path);
        ensure!(
            relative
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
            "user file path {:?} must contain only normal relative components",
            file.path
        );
        let parent = relative.parent().unwrap_or_else(|| Path::new(""));
        let mut safe_parent = root.clone();
        for component in parent.components() {
            let Component::Normal(component) = component else {
                unreachable!("normal components validated above")
            };
            safe_parent.push(component);
            match std::fs::symlink_metadata(&safe_parent) {
                Ok(metadata) => ensure!(
                    metadata.is_dir() && !metadata.file_type().is_symlink(),
                    "user file path {:?} traverses non-directory or symlink {}",
                    file.path,
                    safe_parent.display()
                ),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    std::fs::create_dir(&safe_parent).with_context(|| {
                        format!(
                            "creating parent {} for user file {:?}",
                            safe_parent.display(),
                            file.path
                        )
                    })?;
                }
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!(
                            "inspecting parent {} for user file {:?}",
                            safe_parent.display(),
                            file.path
                        )
                    });
                }
            }
            let canonical = safe_parent.canonicalize().with_context(|| {
                format!(
                    "canonicalizing parent {} for user file {:?}",
                    safe_parent.display(),
                    file.path
                )
            })?;
            ensure!(
                canonical.starts_with(&root),
                "user file path {:?} escapes artifact root {}",
                file.path,
                root.display()
            );
        }
        let target = root.join(relative);
        match std::fs::symlink_metadata(&target) {
            Ok(metadata) => ensure!(
                metadata.is_file() && !metadata.file_type().is_symlink(),
                "user file target {:?} is not a regular file",
                file.path
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("inspecting user file target {:?}", file.path));
            }
        }
        std::fs::write(&target, file.content.as_bytes())
            .with_context(|| format!("writing user file {:?}", file.path))?;
    }
    Ok(())
}

fn validate_plan(request: &NativeRunPlan) -> Result<()> {
    let gpu_telemetry = request.run.sidecars.gpu_telemetry()?;
    let network_latency = request.run.sidecars.network_latency()?;
    let server_metrics = request.run.sidecars.server_metrics()?;
    let live_streaming = request.run.sidecars.live_streaming()?;
    ensure!(
        !request.run.benchmark_id.trim().is_empty(),
        "benchmark_id cannot be empty"
    );
    ensure!(
        !request.run.models.items.is_empty(),
        "at least one model is required"
    );
    ensure!(
        !request.run.endpoint.default_urls()?.is_empty(),
        "at least one endpoint URL is required"
    );
    ensure!(
        !request.run.phases.is_empty(),
        "at least one phase is required"
    );
    ensure!(request.run.workers > 0, "workers must be greater than zero");
    ensure!(
        request
            .run
            .phases
            .iter()
            .any(|phase| phase.common().name == "profiling"),
        "a profiling phase is required"
    );
    for (index, phase) in request.run.phases.iter().enumerate() {
        let common = phase.common();
        ensure!(
            matches!(common.name.as_str(), "warmup" | "profiling"),
            "phase {index} name must be warmup or profiling"
        );
        ensure!(
            common.exclude_from_results == (common.name == "warmup"),
            "phase {:?} exclude_from_results disagrees with its semantic kind",
            common.name
        );
    }
    if gpu_telemetry.is_some() {
        ensure!(
            request
                .run
                .phases
                .iter()
                .filter(|phase| phase.common().name == "profiling")
                .count()
                == 1,
            "GPU telemetry requires exactly one profiling phase"
        );
    }
    if network_latency.is_some() {
        ensure!(
            request
                .run
                .phases
                .iter()
                .filter(|phase| phase.common().name == "profiling")
                .count()
                == 1,
            "network latency calibration requires exactly one profiling phase"
        );
    }
    if let Some(spec) = server_metrics {
        ensure!(
            request
                .run
                .phases
                .iter()
                .filter(|phase| phase.common().name == "profiling")
                .count()
                == 1,
            "server metrics requires exactly one profiling phase"
        );
        ensure!(
            !spec.urls.is_empty(),
            "server metrics requires at least one endpoint URL"
        );
        ensure!(
            !spec.formats.is_empty(),
            "server metrics requires at least one export format"
        );
        let has_jsonl = spec
            .formats
            .contains(&crate::protocol::ServerMetricsFormatSpec::Jsonl);
        let has_parquet = spec
            .formats
            .contains(&crate::protocol::ServerMetricsFormatSpec::Parquet);
        ensure!(
            has_jsonl == spec.jsonl_path.is_some(),
            "server metrics jsonl_path must be present exactly when JSONL is selected"
        );
        ensure!(
            has_parquet == spec.parquet_wire_path.is_some(),
            "server metrics parquet_wire_path must be present exactly when Parquet is selected"
        );
    }
    if let Some(spec) = live_streaming {
        ensure!(
            spec.python_executable.is_absolute(),
            "live streaming python_executable must be absolute"
        );
        ensure!(
            !spec.worker_module.trim().is_empty(),
            "live streaming worker_module cannot be empty"
        );
        ensure!(
            spec.buffer_capacity > 0,
            "live streaming buffer_capacity must be positive"
        );
        ensure!(
            spec.otel.metrics_url.is_some() || spec.mlflow.tracking_uri.is_some(),
            "live streaming requires an OTel or MLflow destination"
        );
    }
    if let NativeDatasetPlan::StaticAccuracy(accuracy) = &request.run.dataset {
        accuracy.validate()?;
        for phase in &request.run.phases {
            ensure!(
                !matches!(
                    phase,
                    PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. }
                ),
                "accuracy evaluator datasets are single-turn and require a concurrency or request-rate phase"
            );
        }
    }
    Ok(())
}

struct PreparedAccuracy {
    evaluator: Box<dyn AccuracyEvaluator>,
    loaded: EvaluatorLoadResult,
    dataset: AccuracyDataset,
    processor: Rc<AccuracyRecordProcessor>,
    tokenizer: Arc<dyn TextTokenizer>,
}

/// Startup seam for native sidecar resources.
///
/// Preparation runs on the coordinator's `LocalSet`, may supervise extension
/// workers, and must return the exact Clock/anchor later given to scheduling
/// and HTTP execution. A distribution can replace resource construction
/// without changing artifact ownership or phase execution.
#[async_trait(?Send)]
pub(crate) trait NativeSidecarResourceFactory: std::fmt::Debug + Send + Sync {
    /// Prepare the complete run-owned bundle without creating local artifacts.
    async fn prepare(&self, run: &NativeRunSpec) -> Result<PreparedNativeSidecarResources>;
}

/// Built-in native sidecar resource composition.
#[derive(Debug)]
pub(crate) struct BuiltinNativeSidecarResourceFactory;

/// Resources prepared before the exclusive artifact target is created.
///
/// The bundle owns cleanup order and retains every path/fact derived during
/// preparation so execution never reopens the authored sidecar configuration.
pub(crate) struct PreparedNativeSidecarResources {
    real_clock_anchor: RealClockAnchor,
    clock: Rc<dyn Clock>,
    gpu_telemetry: Option<GpuTelemetryRun>,
    network_latency: Option<NetworkLatencyRun>,
    server_metrics: Option<ServerMetricsRun>,
    live_streaming: Option<PythonLiveStreamingRun>,
    gpu_records_path: Option<PathBuf>,
    network_latency_records_path: Option<PathBuf>,
    server_metrics_jsonl_path: Option<PathBuf>,
    server_metrics_parquet_wire_path: Option<PathBuf>,
}

#[async_trait(?Send)]
impl NativeSidecarResourceFactory for BuiltinNativeSidecarResourceFactory {
    async fn prepare(&self, run: &NativeRunSpec) -> Result<PreparedNativeSidecarResources> {
        let real_clock_anchor = RealClockAnchor::now();
        let clock: Rc<dyn Clock> = RealClock::from_anchor(real_clock_anchor);
        let endpoint_urls = run.endpoint.default_urls()?;
        let gpu_spec = run.sidecars.gpu_telemetry()?;
        let network_spec = run.sidecars.network_latency()?;
        let server_spec = run.sidecars.server_metrics()?;
        let live_spec = run.sidecars.live_streaming()?;

        // These constructors and path checks cannot start phase tasks. Finish
        // every fallible local step before supervising a GPU/live child.
        let network_latency = network_spec
            .map(|spec| {
                NetworkLatencyRun::new(&run.benchmark_id, spec, endpoint_urls, clock.clone())
            })
            .transpose()?;
        let server_metrics = server_spec
            .map(|spec| ServerMetricsRun::new(spec, clock.clone()))
            .transpose()?;
        let gpu_records_path = gpu_spec
            .map(|spec| {
                artifact_path(
                    &run.artifact_dir,
                    &spec.records_path,
                    "gpu_telemetry.records_path",
                )
            })
            .transpose()?;
        let network_latency_records_path = network_spec
            .and_then(|spec| spec.probe.as_ref())
            .map(|probe| {
                artifact_path(
                    &run.artifact_dir,
                    &probe.records_path,
                    "network_latency.probe.records_path",
                )
            })
            .transpose()?;
        let server_metrics_jsonl_path = server_spec
            .and_then(|spec| spec.jsonl_path.as_ref())
            .map(|path| artifact_path(&run.artifact_dir, path, "server_metrics.jsonl_path"))
            .transpose()?;
        let server_metrics_parquet_wire_path = server_spec
            .and_then(|spec| spec.parquet_wire_path.as_ref())
            .map(|path| artifact_path(&run.artifact_dir, path, "server_metrics.parquet_wire_path"))
            .transpose()?;
        let live_metrics_config = live_spec
            .is_some()
            .then(|| metrics_config(&run.metrics))
            .transpose()?;

        let gpu_telemetry = match gpu_spec {
            Some(spec) => Some(GpuTelemetryRun::new(spec, clock.clone()).await?),
            None => None,
        };
        let live_streaming = if live_spec.is_some() {
            match PythonLiveStreamingRun::spawn(
                run,
                live_metrics_config.expect("present live spec prepared its metrics config"),
            )
            .await
            {
                Ok(worker) => Some(worker),
                Err(error) => {
                    eprintln!("live telemetry extension failed to start: {error:#}");
                    None
                }
            }
        } else {
            None
        };

        Ok(PreparedNativeSidecarResources {
            real_clock_anchor,
            clock,
            gpu_telemetry,
            network_latency,
            server_metrics,
            live_streaming,
            gpu_records_path,
            network_latency_records_path,
            server_metrics_jsonl_path,
            server_metrics_parquet_wire_path,
        })
    }
}

impl PreparedNativeSidecarResources {
    fn live_sink(&self) -> Option<Rc<dyn LiveResultsSink>> {
        self.live_streaming
            .as_ref()
            .map(PythonLiveStreamingRun::sink)
    }

    async fn activate_live_streaming(&mut self) {
        let activation = match self.live_streaming.as_mut() {
            Some(worker) => worker.activate().await,
            None => return,
        };
        if let Err(error) = activation {
            eprintln!("live telemetry extension failed to activate: {error:#}");
            self.live_streaming.take();
        }
    }

    async fn shutdown_run_resources(&mut self) {
        if let Some(worker) = self.live_streaming.take()
            && let Err(error) = worker.shutdown().await
        {
            eprintln!("live telemetry extension failed to shut down cleanly: {error:#}");
        }

        // Server-metrics tasks belong to phase sidecars and have already
        // drained. Drop that retained source graph before supervised GPU
        // workers, matching the explicit run-owned cleanup order.
        self.server_metrics.take();
        if let Some(gpu_telemetry) = self.gpu_telemetry.take() {
            gpu_telemetry.shutdown().await;
        }
        self.network_latency.take();
    }
}

async fn prepare_and_execute_native(
    request: NativeRunPlan,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
    sidecar_factory: &dyn NativeSidecarResourceFactory,
    readiness: Option<(
        Box<dyn PreparedOnlineReadiness>,
        &dyn ReadinessTransportFactory,
    )>,
) -> Result<NativeReport> {
    if matches!(request.run.dataset, NativeDatasetPlan::Graph(_)) {
        validate_graph_request(&request)?;
    }
    let mut accuracy = prepare_static_accuracy(&request).await?;
    let mut sidecars = match sidecar_factory.prepare(&request.run).await {
        Ok(sidecars) => sidecars,
        Err(error) => {
            return finish_accuracy_lifecycle(
                Err(error.context("preparing native sidecar resources")),
                accuracy.as_mut(),
            )
            .await;
        }
    };
    if let Some((readiness, transport_factory)) = readiness
        && !readiness.is_empty()
    {
        let clock = sidecars.clock.clone();
        let transport = transport_factory.build(clock.clone());
        if let Err(error) = readiness.wait(clock, transport).await {
            sidecars.shutdown_run_resources().await;
            return finish_accuracy_lifecycle(
                Err(error.context("waiting for endpoint readiness")),
                accuracy.as_mut(),
            )
            .await;
        }
    }
    let result = execute_native(
        request,
        accuracy.as_mut(),
        &mut sidecars,
        backend_factory,
        graph_placement,
        registry,
    )
    .await;
    sidecars.shutdown_run_resources().await;
    finish_accuracy_lifecycle(result, accuracy.as_mut()).await
}

fn create_run_artifacts(run: &NativeRunSpec) -> Result<()> {
    std::fs::create_dir_all(&run.artifact_dir).with_context(|| {
        format!(
            "creating run artifact directory {}",
            run.artifact_dir.display()
        )
    })?;
    materialize_user_files(&run.artifact_dir, &run.user_files)
}

async fn execute_native(
    request: NativeRunPlan,
    accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    if matches!(request.run.dataset, NativeDatasetPlan::Graph(_)) {
        ensure!(
            accuracy.is_none(),
            "graph execution received prepared static-accuracy state"
        );
        return execute_graph_native(request, sidecars, graph_placement, registry).await;
    }
    execute_scheduled_native(request, accuracy, sidecars, backend_factory, registry).await
}

async fn execute_scheduled_native(
    request: NativeRunPlan,
    accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    backend_factory: &dyn HttpExecutionBackendFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    execute_native_inner(request, accuracy, sidecars, backend_factory, registry).await
}

fn validate_graph_request(request: &NativeRunPlan) -> Result<()> {
    ensure!(
        request.run.sidecars.is_empty(),
        "authored Graph-IR runs do not yet support GPU, network, server, or live-streaming telemetry"
    );
    ensure!(
        request.run.models.items.len() == 1,
        "authored Graph-IR runs currently require exactly one configured default model; per-node model overrides remain supported"
    );
    ensure!(
        matches!(
            request.run.models.strategy,
            ModelSelectionStrategy::RoundRobin
        ),
        "authored Graph-IR runs currently require round_robin model selection; other policies need a graph model-selection trait implementation"
    );
    ensure!(
        matches!(request.run.dataset, NativeDatasetPlan::Graph(_)),
        "graph execution requires a direct graph input plan"
    );
    validate_graph_phases(&request.run.phases)
}

fn graph_input_config(dataset: &DatasetSpec) -> Result<GraphInputConfig> {
    match dataset {
        DatasetSpec::File(spec) => {
            ensure!(
                spec.path.is_some() ^ spec.records.is_some(),
                "file dataset requires exactly one of path or records"
            );
            let source = match (&spec.path, &spec.records) {
                (Some(path), None) => DatasetSource::Path(path.clone()),
                (None, Some(records)) => DatasetSource::Inline(records.clone()),
                _ => unreachable!("file graph source exclusivity validated above"),
            };
            let mut load = LoadConfig::new(source);
            load.options = spec.options.clone();
            Ok(GraphInputConfig {
                load,
                root_limit: spec.entries,
            })
        }
        DatasetSpec::Public(spec) => {
            ensure!(
                !spec.name.trim().is_empty(),
                "public dataset name cannot be empty"
            );
            let option_limit = match spec.options.get("max_conversations") {
                None => None,
                Some(value) => Some(
                    value
                        .as_u64()
                        .and_then(|value| usize::try_from(value).ok())
                        .filter(|value| *value > 0)
                        .ok_or_else(|| {
                            anyhow!(
                                "public graph option max_conversations must be a positive usize"
                            )
                        })?,
                ),
            };
            let root_limit = spec.entries.or(option_limit);
            let source = match &spec.source {
                PublicDatasetSourceSpec::Url { url } => {
                    ensure!(!url.trim().is_empty(), "public dataset URL cannot be empty");
                    DatasetSource::Url(url.clone())
                }
                PublicDatasetSourceSpec::HuggingFace {
                    dataset,
                    subset,
                    split,
                    revision,
                } => DatasetSource::HuggingFace {
                    dataset: dataset.clone(),
                    config: subset.clone(),
                    split: split.clone(),
                    // DAG vertices must be acquired as one complete program.
                    max_rows: None,
                    revision: revision.clone(),
                },
            };
            let mut load = LoadConfig::new(source);
            load.options = spec.options.clone();
            load.options.remove("max_conversations");
            Ok(GraphInputConfig { load, root_limit })
        }
        DatasetSpec::Synthetic(_) => bail!("synthetic datasets do not author Graph-IR"),
    }
}

struct OnlineGraphPhaseBackendFactory<'a> {
    placement: &'a dyn RunnerGraphPlacementFactory,
    worker_count: usize,
    real_clock_anchor: RealClockAnchor,
    run_origin_ns: i64,
    model: String,
    default_max_tokens: usize,
    endpoint_runtime_factory: Arc<dyn RunnerGraphEndpointRuntimeFactory>,
    segments: Arc<dyn aiperf_dataset::SegmentStore>,
    metrics: MetricsConfig,
    raw_enabled: bool,
}

impl RunnerGraphPhaseBackendFactory for OnlineGraphPhaseBackendFactory<'_> {
    fn prepare_backend(
        &self,
        config: GraphPhaseBackendConfig,
    ) -> Result<PreparedGraphPhaseBackend> {
        let worker_factory = Arc::new(RunnerGraphBackendFactory::new(
            RunnerGraphBackendFactoryConfig {
                real_clock_anchor: self.real_clock_anchor,
                run_origin_ns: self.run_origin_ns,
                model: self.model.clone(),
                default_max_tokens: self.default_max_tokens,
                endpoint_runtime_factory: self.endpoint_runtime_factory.clone(),
                segments: self.segments.clone(),
                metrics: self.metrics.clone(),
                phase: config.metrics_phase,
                prefill_concurrency: config.prefill_concurrency,
                cancellation: config.cancellation,
                raw_enabled: self.raw_enabled,
                events: config.events,
            },
        ));
        let requires_node_records = self.placement.requires_node_records();
        let placement = self.placement.build(self.worker_count, worker_factory)?;
        Ok(PreparedGraphPhaseBackend {
            placement,
            requires_node_records,
        })
    }
}

async fn execute_graph_native(
    request: NativeRunPlan,
    sidecars: &PreparedNativeSidecarResources,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    let graph = match &request.run.dataset {
        NativeDatasetPlan::Graph(graph) => graph,
        NativeDatasetPlan::Linear(_)
        | NativeDatasetPlan::PreparedLinear(_)
        | NativeDatasetPlan::StaticAccuracy(_)
        | NativeDatasetPlan::AuthoredGraph(_) => {
            bail!("graph execution received a non-graph dataset plan")
        }
    };
    let graph_random_seed = graph.random_seed;
    let graph_default_output_tokens = graph.default_output_tokens;
    let metrics_config = metrics_config(&request.run.metrics)?;
    let tokenizer = load_tokenizer(Some(&request.run.tokenizer.name))?;
    let input_token_counter: Arc<dyn InputTokenCounter> = Arc::new(EndpointInputTokenCounter::new(
        tokenizer.clone(),
        request.run.tokenizer.apply_chat_template,
    ));
    let input = graph.input.clone();
    ensure!(
        !input.plans.is_empty(),
        "authored Graph-IR input contains no root traces after root limiting"
    );
    let primary_model = request.run.models.items[0].name.clone();
    let default_output_tokens = graph_default_output_tokens;
    let endpoints_configured = match &request.run.endpoint {
        NativeEndpointPlan::Legacy(spec) => spec.urls.clone(),
        NativeEndpointPlan::Prepared(profiles) => profiles
            .iter()
            .flat_map(|profile| profile.config.urls.iter().cloned())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
    };
    let endpoint_runtime_factory: Arc<dyn RunnerGraphEndpointRuntimeFactory> =
        match &request.run.endpoint {
            NativeEndpointPlan::Legacy(spec) => {
                let request_timeout_ns = seconds_to_ns(spec.timeout_seconds)?;
                Arc::new(LegacyRunnerGraphEndpointRuntimeFactory::new(
                    spec.urls.clone(),
                    TransportSinkConfig {
                        client: ClientConfig {
                            http_version: if spec.http2 {
                                HttpVersion::Http2PriorKnowledge
                            } else {
                                HttpVersion::Auto
                            },
                            total_timeout_ns: (request_timeout_ns > 0)
                                .then_some(request_timeout_ns),
                            ..ClientConfig::default()
                        },
                        connection_reuse: spec.connection_reuse,
                        session_header: spec.session_header.clone(),
                    },
                    endpoint_config(spec)?,
                    registry.endpoint_resolver(),
                    input_token_counter.clone(),
                ))
            }
            NativeEndpointPlan::Prepared(profiles) => {
                Arc::new(PreparedRunnerGraphEndpointRuntimeFactory::new(
                    registry.endpoints().clone(),
                    profiles.clone(),
                    input_token_counter.clone(),
                ))
            }
        };
    let real_clock_anchor = sidecars.real_clock_anchor;
    let clock = sidecars.clock.clone();
    let start_ns = clock.now_ns();
    let rng_root = RngRoot::new(graph_random_seed.or(request.run.random_seed));
    let backends = OnlineGraphPhaseBackendFactory {
        placement: graph_placement,
        worker_count: request.run.workers,
        real_clock_anchor,
        run_origin_ns: start_ns,
        model: primary_model.clone(),
        default_max_tokens: default_output_tokens,
        endpoint_runtime_factory,
        segments: input.segments.clone(),
        metrics: metrics_config.clone(),
        raw_enabled: request.run.artifacts.raw_path.is_some(),
    };
    create_run_artifacts(&request.run)?;
    let phased = run_graph_phases(
        &request.run.phases,
        &request.run.benchmark_id,
        &request.run.artifact_dir,
        input.as_ref(),
        clock.clone(),
        rng_root,
        &backends,
    )
    .await?;
    ensure!(
        phased.workload.failed == 0,
        "graph phase runtime returned failed traces without failing execution"
    );
    let phase_stats = phased.phases;
    let captured = phased.captured;

    let mut accumulator = MetricsAccumulator::with_config(metrics_config.clone());
    for record in &captured {
        accumulator.process_record(&record.ingest);
    }
    let profiling_metrics =
        accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
    let warmup = captured
        .iter()
        .any(|record| record.ingest.phase == MetricsPhase::Warmup)
        .then(|| accumulator.export_results(&ExportContext::phase(MetricsPhase::Warmup)));
    write_graph_artifacts(&request, &captured, &metrics_config)?;

    let profiling = captured
        .iter()
        .filter(|record| record.ingest.phase == MetricsPhase::Profiling)
        .collect::<Vec<_>>();
    let start_time = profiling.iter().map(|record| record.ingest.start_ns).min();
    let end_time = profiling.iter().map(|record| record.ingest.end_ns).max();
    let endpoints_successful = profiling
        .iter()
        .filter(|record| !record.ingest.errored && !record.ingest.canceled)
        .filter_map(|record| record.ingest.dimensions.endpoint_url.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let summary = ReportSummary {
        start_time,
        end_time,
        duration_s: start_time
            .zip(end_time)
            .map(|(start, end)| end.saturating_sub(start) as f64 / 1_000_000_000.0),
        was_cancelled: phase_stats.iter().any(|phase| phase.was_cancelled),
        endpoints_configured,
        endpoints_successful,
        server_metrics: None,
    };
    let outcome = RunOutcome {
        run: ReportRunInfo {
            mode: Some("graph".into()),
            model: Some(primary_model),
        },
        summary,
        warmup,
        ..RunOutcome::default()
    };
    Ok(NativeReport::from_outcome(&profiling_metrics, &outcome))
}

fn write_graph_artifacts(
    request: &NativeRunPlan,
    captured: &[CapturedRecord],
    metrics_config: &MetricsConfig,
) -> Result<()> {
    if let Some(records_path) = &request.run.artifacts.records_path {
        let path = artifact_path(&request.run.artifact_dir, records_path, "records_path")?;
        write_records_jsonl(&path, captured, metrics_config, request.run.artifacts.trace)?;
    }
    if let Some(raw_path) = &request.run.artifacts.raw_path {
        let path = artifact_path(&request.run.artifact_dir, raw_path, "raw_path")?;
        write_raw_records_jsonl(&path, captured)?;
    }
    if let Some(outputs_path) = &request.run.artifacts.outputs_path {
        let path = artifact_path(&request.run.artifact_dir, outputs_path, "outputs_path")?;
        write_outputs_json(&path, captured, metrics_config)?;
    }
    Ok(())
}

async fn prepare_static_accuracy(request: &NativeRunPlan) -> Result<Option<PreparedAccuracy>> {
    let NativeDatasetPlan::StaticAccuracy(spec) = &request.run.dataset else {
        return Ok(None);
    };
    let model = request
        .run
        .models
        .items
        .first()
        .map(|item| item.name.as_str())
        .ok_or_else(|| anyhow!("at least one model is required"))?;
    let tokenizer = load_tokenizer(Some(&request.run.tokenizer.name))?;
    let mut evaluator = spec.evaluator_factory.spawn(&spec.process).await?;
    let preparation = async {
        let evaluator_config = EvaluatorLoadConfig {
            tasks: spec.tasks.clone(),
            n_shots: spec.n_shots,
            enable_cot: spec.enable_cot,
            system_prompt: spec.system_prompt.clone(),
            max_problems: None,
            max_tokens: None,
            seed: request.run.random_seed.unwrap_or(0),
        };
        let (loaded, problems) = load_evaluator_problems_with_grader(
            evaluator.as_mut(),
            &spec.benchmark,
            &evaluator_config,
            spec.grader.as_deref(),
        )
        .await?;
        let dataset =
            AccuracyDataset::from_evaluator_problems(model, problems, tokenizer.as_ref())?;
        let processor = Rc::new(dataset.record_processor());
        Ok::<_, anyhow::Error>((loaded, dataset, processor))
    }
    .await;
    match preparation {
        Ok((loaded, dataset, processor)) => Ok(Some(PreparedAccuracy {
            evaluator,
            loaded,
            dataset,
            processor,
            tokenizer,
        })),
        Err(error) => {
            let shutdown = evaluator.shutdown().await.map_err(anyhow::Error::from);
            finish_accuracy_shutdown(Err(error), shutdown)
        }
    }
}

async fn finish_accuracy_lifecycle<T>(
    result: Result<T>,
    accuracy: Option<&mut PreparedAccuracy>,
) -> Result<T> {
    let shutdown = match accuracy {
        Some(accuracy) => accuracy
            .evaluator
            .shutdown()
            .await
            .map_err(anyhow::Error::from),
        None => Ok(()),
    };
    finish_accuracy_shutdown(result, shutdown)
}

fn finish_accuracy_shutdown<T>(result: Result<T>, shutdown: Result<()>) -> Result<T> {
    match (result, shutdown) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error.context("shutting down accuracy evaluator")),
        (Err(error), Err(shutdown)) => Err(error.context(format!(
            "accuracy evaluator also failed during shutdown: {shutdown:#}"
        ))),
    }
}

fn finish_execution_backend_shutdown<T>(result: Result<T>, shutdown: Result<()>) -> Result<T> {
    match (result, shutdown) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error.context("shutting down execution backend")),
        (Err(error), Err(shutdown)) => Err(error.context(format!(
            "execution backend also failed during shutdown: {shutdown:#}"
        ))),
    }
}

async fn execute_native_inner(
    request: NativeRunPlan,
    mut accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    backend_factory: &dyn HttpExecutionBackendFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    let live_sink = sidecars.live_sink();
    let rng_root = RngRoot::new(request.run.random_seed);
    let dataset_spec = match &request.run.dataset {
        NativeDatasetPlan::Linear(dataset) => Some(dataset),
        NativeDatasetPlan::PreparedLinear(_) | NativeDatasetPlan::StaticAccuracy(_) => None,
        NativeDatasetPlan::Graph(_) | NativeDatasetPlan::AuthoredGraph(_) => {
            bail!("scheduled execution received a direct graph dataset plan")
        }
    };
    let dataset_rng_root = match &request.run.dataset {
        NativeDatasetPlan::Linear(dataset) => dataset_rng_root(dataset, rng_root),
        NativeDatasetPlan::PreparedLinear(dataset) => dataset
            .random_seed
            .map_or(rng_root, |seed| RngRoot::new(Some(seed))),
        NativeDatasetPlan::StaticAccuracy(_) => rng_root,
        NativeDatasetPlan::Graph(_) | NativeDatasetPlan::AuthoredGraph(_) => {
            unreachable!("graph rejected above")
        }
    };
    let metrics_config = metrics_config(&request.run.metrics)?;
    let model_names = request
        .run
        .models
        .items
        .iter()
        .map(|item| item.name.clone())
        .collect::<Vec<_>>();
    let primary_model = model_names
        .first()
        .cloned()
        .ok_or_else(|| anyhow!("at least one model is required"))?;
    let tokenizer = match accuracy.as_ref() {
        Some(accuracy) => accuracy.tokenizer.clone(),
        None => load_tokenizer(Some(&request.run.tokenizer.name))?,
    };
    let input_token_counter: Arc<dyn InputTokenCounter> = Arc::new(EndpointInputTokenCounter::new(
        tokenizer.clone(),
        request.run.tokenizer.apply_chat_template,
    ));
    let (
        endpoint_urls,
        transport_config,
        prepared_endpoints,
        source_factory,
        legacy_endpoint_type,
    ): NativeEndpointExecutionParts<'_> = match &request.run.endpoint {
        NativeEndpointPlan::Legacy(spec) => {
            let endpoint = endpoint_config(spec)?;
            let request_timeout_ns = seconds_to_ns(spec.timeout_seconds)?;
            (
                spec.urls.clone(),
                TransportSinkConfig {
                    client: ClientConfig {
                        http_version: if spec.http2 {
                            HttpVersion::Http2PriorKnowledge
                        } else {
                            HttpVersion::Auto
                        },
                        total_timeout_ns: (request_timeout_ns > 0)
                            .then_some(request_timeout_ns),
                        ..ClientConfig::default()
                    },
                    connection_reuse: spec.connection_reuse,
                    session_header: spec.session_header.clone(),
                },
                None,
                Box::new(LegacyNativeConversationSourceFactory {
                    endpoint,
                    registry,
                }),
                Some(spec.endpoint_type),
            )
        }
        NativeEndpointPlan::Prepared(profiles) => {
            let profile = default_prepared_endpoint_profile(profiles)?;
            let table_factory = Arc::new(NativePreparedEndpointTableFactory::new(
                registry.endpoints().clone(),
                profiles.clone(),
            ));
            let endpoint_resolver = table_factory.coordinator_resolver()?;
            (
                profile.config.urls.clone(),
                TransportSinkConfig {
                    client: profile.client.clone(),
                    connection_reuse: profile.connection_reuse,
                    session_header: profile.session_header.clone(),
                },
                Some(table_factory),
                Box::new(PreparedNativeConversationSourceFactory {
                    endpoint_resolver,
                    samplers: registry.samplers(),
                }),
                None,
            )
        }
    };
    let dataset = if let Some(accuracy) = accuracy.as_ref() {
        accuracy.dataset.dataset().as_ref().clone()
    } else {
        match &request.run.dataset {
            NativeDatasetPlan::Linear(dataset) => {
                let endpoint_type = legacy_endpoint_type.ok_or_else(|| {
                    anyhow!(
                        "protocol-v2 prepared endpoints require a directly prepared linear dataset"
                    )
                })?;
                build_dataset(
                    registry,
                    dataset,
                    &request.run.models,
                    dataset_rng_root,
                    tokenizer.as_ref(),
                    endpoint_type,
                )
                .await?
            }
            NativeDatasetPlan::PreparedLinear(dataset) => dataset.dataset.clone(),
            NativeDatasetPlan::StaticAccuracy(_) => {
                bail!("evaluator dataset plan requires an accuracy evaluator")
            }
            NativeDatasetPlan::Graph(_) | NativeDatasetPlan::AuthoredGraph(_) => {
                unreachable!("graph rejected above")
            }
        }
    };
    let default_output_tokens = if accuracy.is_some() {
        dataset_default_output_tokens(&dataset)?
    } else {
        match (&request.run.dataset, dataset_spec) {
            (NativeDatasetPlan::Linear(_), Some(dataset)) => default_output_tokens(dataset)?,
            (NativeDatasetPlan::PreparedLinear(dataset), None) => dataset.default_output_tokens,
            (NativeDatasetPlan::StaticAccuracy(_), None) => {
                unreachable!("evaluator without accuracy rejected above")
            }
            _ => unreachable!("dataset plan/spec pairing is exhaustive"),
        }
    };

    let real_clock_anchor = sidecars.real_clock_anchor;
    let clock = sidecars.clock.clone();
    let execution_backend = backend_factory.build(HttpExecutionBackendConfig {
        workers: request.run.workers,
        coordinator_clock: clock.clone(),
        real_clock_anchor,
        base_urls: endpoint_urls.clone(),
        model: primary_model.clone(),
        transport: transport_config,
        prepared_endpoints,
    })?;
    let start_ns = clock.now_ns();
    let capture = Rc::new(RunCapture::new(
        clock.clone(),
        start_ns,
        metrics_config.clone(),
        request.run.artifacts.raw_path.is_some(),
    ));
    let execution_result = async {
        execution_backend.set_run_origin(start_ns)?;
        let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(ConfiguredDispatcher {
            execution_backend: execution_backend.clone(),
            model: primary_model.clone(),
            capture: capture.clone(),
        });

        let shared_resources = native_scheduled_resources(&request.run.phases);

        let mut plans = Vec::with_capacity(request.run.phases.len());
        for (phase_index, phase) in request.run.phases.iter().enumerate() {
            let mut plan = build_native_scheduled_phase_plan_with_source_factory(
                phase_index,
                phase,
                phase_seamless_to_next(&request.run.phases, phase_index),
                &dataset,
                &primary_model,
                default_output_tokens,
                dataset_rng_root,
                rng_root,
                source_factory.as_ref(),
                tokenizer.clone(),
                input_token_counter.clone(),
                clock.clone(),
                start_ns,
                &request.run.benchmark_id,
                &request.run.artifact_dir,
                &endpoint_urls,
                &shared_resources,
            )?;
            let record_processor: Rc<dyn TurnRecordProcessor> = Rc::new(CapturePhaseProcessor {
                capture: capture.clone(),
                phase: metrics_phase(phase)?,
                has_credit_timestamp: !matches!(phase, PhaseSpec::FixedSchedule { .. }),
                live_sink: live_sink.clone(),
            });
            let mut record_processors = vec![record_processor];
            if phase.common().name == "profiling"
                && let Some(accuracy) = accuracy.as_ref()
            {
                let processor: Rc<dyn TurnRecordProcessor> = accuracy.processor.clone();
                record_processors.push(processor);
            }
            plan = plan.with_record_processors(record_processors);
            let mut phase_sidecars = Vec::new();
            if let Some(server_metrics) = sidecars.server_metrics.as_ref() {
                phase_sidecars.push(server_metrics.sidecar(metrics_phase(phase)?));
            }
            if phase.common().name == "profiling" {
                if let Some(gpu_telemetry) = sidecars.gpu_telemetry.as_ref() {
                    phase_sidecars.push(gpu_telemetry.sidecar());
                }
                if let Some(network_latency) = sidecars.network_latency.as_ref()
                    && let Some(sidecar) = network_latency.sidecar()
                {
                    phase_sidecars.push(sidecar);
                }
            }
            if !phase_sidecars.is_empty() {
                plan = plan.with_sidecars(phase_sidecars);
            }
            plans.push(plan);
        }

        create_run_artifacts(&request.run)?;
        sidecars.activate_live_streaming().await;

        let observer: Rc<dyn PhaseObserver> = if let Some(sink) = live_sink {
            live_phase_observer(sink, clock.clone())
        } else {
            Rc::new(NoopPhaseObserver)
        };
        let phased = run_scheduled_phases(plans, clock, dispatcher, observer).await?;
        phased
            .reports
            .iter()
            .find(|report| report.kind == PhaseKind::Profiling)
            .ok_or_else(|| anyhow!("phase runtime completed without a profiling report"))?;
        Ok(phased)
    }
    .await;
    let shutdown = execution_backend.shutdown();
    let phased = finish_execution_backend_shutdown(execution_result, shutdown)?;
    let issued_times = phased
        .reports
        .iter()
        .flat_map(|report| report.report.turns.iter())
        .map(|turn| (turn.uuid, turn.issued_offset_ns))
        .collect::<HashMap<_, _>>();
    let captured = capture.finish(&issued_times)?;
    let gpu_telemetry = sidecars.gpu_telemetry.as_ref();
    let network_latency = sidecars.network_latency.as_ref();
    let server_metrics = sidecars.server_metrics.as_ref();
    let gpu_records_path = sidecars.gpu_records_path.as_ref();
    let network_latency_records_path = sidecars.network_latency_records_path.as_ref();
    let server_metrics_jsonl_path = sidecars.server_metrics_jsonl_path.as_ref();
    let server_metrics_parquet_wire_path = sidecars.server_metrics_parquet_wire_path.as_ref();
    let mut accumulator = MetricsAccumulator::with_config(metrics_config.clone());
    for record in &captured {
        accumulator.process_record(&record.ingest);
    }
    if let Some(network_latency) = network_latency {
        let mean_rtt_ns = network_latency.mean_rtt_ns();
        if network_latency.is_active_probe() && mean_rtt_ns.is_none() {
            eprintln!(
                "network latency calibration collected no successful probes; adjusted metrics are omitted"
            );
        }
        accumulator.set_network_rtt_ns(mean_rtt_ns);
    }
    let mut profiling_metrics =
        accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
    if let Some(gpu_telemetry) = gpu_telemetry {
        let total_output_tokens = profiling_metrics.finite_value(MetricTag::TotalOutputTokens);
        let concurrency = request
            .run
            .phases
            .iter()
            .find(|phase| phase.common().name == "profiling")
            .and_then(PhaseSpec::concurrency)
            .map(|value| value as u64);
        gpu_telemetry
            .summarize(total_output_tokens, concurrency)
            .attach_to(&mut profiling_metrics);
    }
    let profiling_server_summary = server_metrics.map(|server_metrics| {
        server_metrics.summarize(MetricsPhase::Profiling, metrics_config.slice_duration_ns)
    });
    let warmup = phased
        .reports
        .iter()
        .any(|report| report.kind == PhaseKind::Warmup)
        .then(|| accumulator.export_results(&ExportContext::phase(MetricsPhase::Warmup)));
    let warmup_server_summary = server_metrics
        .filter(|_| warmup.is_some())
        .map(|server_metrics| {
            server_metrics.summarize(MetricsPhase::Warmup, metrics_config.slice_duration_ns)
        });
    if let Some(records_path) = &request.run.artifacts.records_path {
        let records_path = artifact_path(&request.run.artifact_dir, records_path, "records_path")?;
        write_records_jsonl(
            &records_path,
            &captured,
            &metrics_config,
            request.run.artifacts.trace,
        )?;
    }
    if let Some(raw_path) = &request.run.artifacts.raw_path {
        let raw_path = artifact_path(&request.run.artifact_dir, raw_path, "raw_path")?;
        write_raw_records_jsonl(&raw_path, &captured)?;
    }
    if let Some(outputs_path) = &request.run.artifacts.outputs_path {
        let outputs_path = artifact_path(&request.run.artifact_dir, outputs_path, "outputs_path")?;
        write_outputs_json(&outputs_path, &captured, &metrics_config)?;
    }
    if let (Some(gpu_telemetry), Some(gpu_records_path)) = (gpu_telemetry, gpu_records_path) {
        gpu_telemetry.write_records_jsonl(gpu_records_path)?;
    }
    if let (Some(network_latency), Some(records_path)) =
        (network_latency, network_latency_records_path)
    {
        network_latency.write_records_jsonl(records_path)?;
    }
    if let (Some(server_metrics), Some(path)) = (server_metrics, server_metrics_jsonl_path) {
        server_metrics.write_slim_jsonl(path)?;
    }
    if let (Some(server_metrics), Some(path)) = (server_metrics, server_metrics_parquet_wire_path) {
        server_metrics.write_parquet_wire_jsonl(path)?;
    }
    let server_metrics_report = server_metrics.and_then(|server_metrics| {
        profiling_server_summary.as_ref().map(|profiling| {
            server_metrics.report_metadata(profiling, warmup_server_summary.as_ref())
        })
    });
    let mut outcome = RunOutcome {
        run: ReportRunInfo {
            mode: Some("online".into()),
            model: Some(primary_model),
        },
        summary: ReportSummary {
            endpoints_configured: endpoint_urls,
            server_metrics: server_metrics_report,
            ..ReportSummary::default()
        },
        warmup,
        server_metrics: profiling_server_summary
            .as_ref()
            .map(|summary| summary.sidecar_metrics().clone())
            .unwrap_or_default(),
        warmup_server_metrics: warmup_server_summary
            .as_ref()
            .map(|summary| summary.sidecar_metrics().clone())
            .unwrap_or_default(),
        ..RunOutcome::default()
    };
    if let Some(accuracy) = accuracy.as_mut() {
        let evaluation = grade_accuracy_responses(
            accuracy.processor.as_ref(),
            accuracy.evaluator.as_mut(),
            &accuracy.loaded,
            &profiling_metrics,
        )
        .await?;
        outcome.run.mode = Some("accuracy".to_string());
        outcome.accuracy = Some(evaluation.accuracy);
        outcome.accuracy_records = evaluation.records;
        outcome.evaluator = Some(evaluation.evaluator_report);
        outcome.errors = accuracy_report_errors(&evaluation.failures);
    }
    Ok(NativeReport::from_outcome(&profiling_metrics, &outcome))
}

fn dataset_default_output_tokens(dataset: &Dataset) -> Result<usize> {
    dataset
        .conversations()
        .iter()
        .flat_map(|conversation| conversation.turns.iter())
        .filter_map(|turn| turn.max_tokens)
        .map(|value| usize::try_from(value).map_err(Into::into))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .ok_or_else(|| anyhow!("accuracy evaluator dataset has no output-token limit"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn native_conversation_source(
    dataset: Dataset,
    model: String,
    default_output_tokens: usize,
    rng_root: RngRoot,
    endpoint: EndpointConfig,
    registry: &AiperfRegistry,
    tokenizer: Arc<dyn TextTokenizer>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    sequential: bool,
) -> Result<Box<dyn ConversationSource>> {
    let source = if sequential {
        NativeDatasetConversationSource::sequential_with_endpoint_config_and_resolver(
            dataset,
            model,
            default_output_tokens,
            endpoint,
            registry.endpoint_resolver(),
        )?
    } else {
        NativeDatasetConversationSource::preferred_with_endpoint_config_and_registries(
            dataset,
            model,
            default_output_tokens,
            rng_root,
            endpoint,
            registry.samplers(),
            registry.endpoint_resolver(),
        )?
    };
    Ok(Box::new(
        source
            .with_response_tokenizer(tokenizer)
            .with_input_token_counter(input_token_counter),
    ))
}

/// Prepared conversation-source construction behind the shared scheduled
/// workload. Legacy protocol-v1 and open protocol-v2 endpoint bindings
/// implement this seam without branching inside phase/scheduler policy.
pub(crate) trait NativeConversationSourceFactory {
    #[allow(clippy::too_many_arguments)]
    fn build(
        &self,
        dataset: Dataset,
        model: String,
        default_output_tokens: usize,
        rng_root: RngRoot,
        tokenizer: Arc<dyn TextTokenizer>,
        input_token_counter: Arc<dyn InputTokenCounter>,
        sequential: bool,
    ) -> Result<Box<dyn ConversationSource>>;
}

type NativeEndpointExecutionParts<'a> = (
    Vec<String>,
    TransportSinkConfig,
    Option<Arc<dyn HttpPreparedEndpointTableFactory>>,
    Box<dyn NativeConversationSourceFactory + 'a>,
    Option<EndpointType>,
);

struct LegacyNativeConversationSourceFactory<'a> {
    endpoint: EndpointConfig,
    registry: &'a AiperfRegistry,
}

impl NativeConversationSourceFactory for LegacyNativeConversationSourceFactory<'_> {
    fn build(
        &self,
        dataset: Dataset,
        model: String,
        default_output_tokens: usize,
        rng_root: RngRoot,
        tokenizer: Arc<dyn TextTokenizer>,
        input_token_counter: Arc<dyn InputTokenCounter>,
        sequential: bool,
    ) -> Result<Box<dyn ConversationSource>> {
        native_conversation_source(
            dataset,
            model,
            default_output_tokens,
            rng_root,
            self.endpoint.clone(),
            self.registry,
            tokenizer,
            input_token_counter,
            sequential,
        )
    }
}

struct PreparedNativeConversationSourceFactory<'a> {
    endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    samplers: &'a aiperf_dataset::SamplerRegistry,
}

impl NativeConversationSourceFactory for PreparedNativeConversationSourceFactory<'_> {
    fn build(
        &self,
        dataset: Dataset,
        model: String,
        default_output_tokens: usize,
        rng_root: RngRoot,
        tokenizer: Arc<dyn TextTokenizer>,
        input_token_counter: Arc<dyn InputTokenCounter>,
        sequential: bool,
    ) -> Result<Box<dyn ConversationSource>> {
        let source = if sequential {
            NativeDatasetConversationSource::sequential_with_prepared_resolver(
                dataset,
                model,
                default_output_tokens,
                self.endpoint_resolver.clone(),
            )?
        } else {
            NativeDatasetConversationSource::preferred_with_prepared_resolver(
                dataset,
                model,
                default_output_tokens,
                rng_root,
                self.samplers,
                self.endpoint_resolver.clone(),
            )?
        };
        Ok(Box::new(
            source
                .with_response_tokenizer(tokenizer)
                .with_input_token_counter(input_token_counter),
        ))
    }
}

/// Lower one authored phase into the shared scheduled runtime above the
/// injected `{transport, clock}` seams.
///
/// Dataset filtering/materialization, arrival policy, session/prefill
/// admission, fixed/user-centric scheduling, ramps, cancellation, adaptive
/// control, and phase lifecycle are deliberately composed here once. Backend
/// adapters may decorate the returned plan with observers or sidecars, but do
/// not reproduce its scheduler logic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_native_scheduled_phase_plan_with_source_factory(
    phase_index: usize,
    phase: &PhaseSpec,
    seamless_to_next: bool,
    dataset: &Dataset,
    primary_model: &str,
    default_output_tokens: usize,
    dataset_rng_root: RngRoot,
    rng_root: RngRoot,
    source_factory: &dyn NativeConversationSourceFactory,
    tokenizer: Arc<dyn TextTokenizer>,
    input_token_counter: Arc<dyn InputTokenCounter>,
    clock: Rc<dyn Clock>,
    start_ns: i64,
    benchmark_id: &str,
    artifact_dir: &Path,
    endpoint_names: &[String],
    shared: &NativeScheduledResources,
) -> Result<ScheduledPhasePlan> {
    let phase_rng =
        RngRoot::new(dataset_rng_root.derive_seed(&format!("runner.phase.{phase_index}.dataset")));
    let phase_dataset = match phase {
        PhaseSpec::FixedSchedule {
            start_offset,
            end_offset,
            ..
        } => dataset.filter_first_turn_window(*start_offset, *end_offset)?,
        _ => dataset.clone(),
    };
    let source = source_factory.build(
        phase_dataset,
        primary_model.to_owned(),
        default_output_tokens,
        phase_rng,
        tokenizer,
        input_token_counter,
        matches!(phase, PhaseSpec::FixedSchedule { .. }),
    )?;
    let arrival_seed = rng_root
        .derive_seed(&format!("runner.phase.{phase_index}.arrival"))
        .unwrap_or(phase_index as u64);
    let (
        workload,
        intervals,
        phase_session,
        phase_prefill,
        enforce_stop,
        resources,
        user_target,
    ): PhaseRuntimeParts = match phase {
        PhaseSpec::Concurrency { .. }
        | PhaseSpec::Poisson { .. }
        | PhaseSpec::Gamma { .. }
        | PhaseSpec::Constant { .. } => {
            let (arrival, rate, smoothness) = phase
                .request_arrival()
                .expect("request-rate phase variants have an arrival policy");
            let intervals = Rc::new(RefCell::new(make_interval_generator(
                arrival,
                rate,
                smoothness,
                arrival_seed,
            )));
            let workload = Rc::new(RequestRateWorkload::with_components(
                source,
                intervals.clone(),
                shared.session.clone(),
                shared.prefill.clone(),
            )?) as Rc<dyn Workload>;
            (
                workload,
                intervals,
                shared.session.clone(),
                shared.prefill.clone(),
                true,
                shared.phase.clone(),
                None,
            )
        }
        PhaseSpec::UserCentric {
            rate,
            users,
            concurrency,
            ..
        } => {
            ensure!(
                phase.common().prefill_concurrency.is_none(),
                "user_centric phases do not own a prefill admission pool"
            );
            ensure!(
                phase.common().rate_ramp.is_none(),
                "user_centric cadence is authored and does not accept rate_ramp"
            );
            let adaptive = phase.common().adaptive_scale.as_ref();
            let initial_users = adaptive
                .filter(|adaptive| {
                    matches!(
                        adaptive.control_variable,
                        AdaptiveControlVariableSpec::Users
                    )
                })
                .map(|adaptive| integer_adaptive_bound(adaptive.minimum, "users minimum"))
                .transpose()?
                .unwrap_or(*users);
            let session_concurrency = match (adaptive, concurrency) {
                (
                    Some(AdaptiveScaleSpec {
                        control_variable: AdaptiveControlVariableSpec::Concurrency,
                        maximum,
                        ..
                    }),
                    None,
                ) => Some(integer_adaptive_bound(*maximum, "concurrency maximum")?),
                _ => *concurrency,
            };
            let concrete = Rc::new(UserCentricWorkload::new(
                UserCentricConfig {
                    num_users: initial_users,
                    request_rate: *rate,
                    concurrency: session_concurrency,
                },
                source,
            )?);
            let phase_session = concrete.session_slots();
            let user_target: Rc<dyn UserTarget> = Rc::new(concrete.control());
            let resources: Rc<dyn ScheduledPhaseResources> =
                Rc::new(SlotPoolPhaseResources::new(phase_session.clone(), None));
            let intervals = Rc::new(RefCell::new(make_interval_generator(
                aiperf_timing::ArrivalPattern::ConcurrencyBurst,
                None,
                None,
                arrival_seed,
            )));
            (
                concrete,
                intervals,
                phase_session,
                None,
                true,
                resources,
                Some(user_target),
            )
        }
        PhaseSpec::FixedSchedule {
            auto_offset,
            start_offset,
            ..
        } => {
            ensure!(
                phase.common().concurrency_ramp.is_none()
                    && phase.common().prefill_ramp.is_none()
                    && phase.common().rate_ramp.is_none(),
                "fixed_schedule phases have authored timestamps and do not accept ramps"
            );
            ensure!(
                phase.common().prefill_concurrency.is_none(),
                "fixed_schedule prefill admission is not configured by the native scheduler"
            );
            let schedule_source = Rc::new(DatasetFixedScheduleSource::new(FixedScheduleConfig {
                auto_offset_timestamps: *auto_offset,
                start_offset_ms: *start_offset,
            })?);
            let workload =
                Rc::new(FixedScheduleWorkload::new(source, schedule_source)?) as Rc<dyn Workload>;
            let intervals = Rc::new(RefCell::new(make_interval_generator(
                aiperf_timing::ArrivalPattern::ConcurrencyBurst,
                None,
                None,
                arrival_seed,
            )));
            (
                workload,
                intervals,
                None,
                None,
                false,
                Rc::new(aiperf::phase_runtime::NoopScheduledPhaseResources),
                None,
            )
        }
    };
    let policies = ancillary_policies(
        phase,
        endpoint_names,
        RngRoot::new(rng_root.derive_seed(&format!("runner.phase.{phase_index}.cancellation"))),
    )?;
    let controller = ramp_controller(
        phase,
        clock,
        intervals.clone(),
        phase_session.clone(),
        phase_prefill.clone(),
        RngRoot::new(rng_root.derive_seed(&format!("runner.phase.{phase_index}.ramp"))),
    )?;
    let runtime_extension = adaptive_runtime_extension(
        phase,
        benchmark_id,
        artifact_dir,
        intervals,
        phase_session,
        phase_prefill,
        user_target,
    )?;
    let mut plan =
        ScheduledPhasePlan::new(phase_config(phase, seamless_to_next)?, workload, policies)
            .with_enforce_stop(enforce_stop)
            .with_start_ns(start_ns)
            .with_resources(resources)
            .with_controller(controller);
    if let Some(extension) = runtime_extension {
        plan = plan.with_runtime_extension(extension);
    }
    Ok(plan)
}

pub(crate) async fn build_dataset(
    registry: &AiperfRegistry,
    dataset: &DatasetSpec,
    models: &ModelsSpec,
    rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
    endpoint_type: EndpointType,
) -> Result<Dataset> {
    match dataset {
        DatasetSpec::Synthetic(spec) => {
            build_synthetic_dataset(
                registry,
                spec,
                models,
                rng_root,
                tokenizer,
                is_rankings_endpoint(endpoint_type),
            )
            .await
        }
        DatasetSpec::File(spec) => {
            build_file_dataset(
                registry,
                spec,
                models,
                rng_root,
                tokenizer,
                Arc::new(MaterializedTracePromptStorage),
            )
            .await
        }
        DatasetSpec::Public(spec) => {
            build_public_dataset(registry, spec, models, rng_root, tokenizer).await
        }
    }
}

pub(crate) async fn build_synthetic_dataset(
    registry: &AiperfRegistry,
    spec: &SyntheticDatasetSpec,
    models: &ModelsSpec,
    rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
    rankings: bool,
) -> Result<Dataset> {
    let mut compose = compose_config(models, rng_root)?;
    if let Some(prompts) = &spec.prompts {
        compose.output_length_distribution = prompts
            .osl
            .as_ref()
            .map(distribution)
            .transpose()?
            .filter(|value| value.expected_value() > 0.0);
        compose.sequence_length_distribution = prompts
            .sequence_distribution
            .as_deref()
            .map(sequence_length_distribution)
            .transpose()?;
    }
    compose.synthetic_config = Some(synthetic_config(spec)?);
    let mut load = LoadConfig::new(DatasetSource::Inline(if rankings {
        serde_json::json!({"__aiperf_synthetic_rankings": true})
    } else {
        serde_json::json!({"__aiperf_synthetic": true})
    }));
    load.sampling_strategy = Some(spec.sampling.clone());
    registry
        .dataset_formats()
        .build_dataset(
            Some(if rankings {
                "synthetic_rankings"
            } else {
                "synthetic"
            }),
            &load,
            &compose,
            tokenizer,
        )
        .await
        .map_err(Into::into)
}

pub(crate) fn dataset_rng_root(dataset: &DatasetSpec, run_rng_root: RngRoot) -> RngRoot {
    dataset_random_seed(dataset).map_or(run_rng_root, |seed| RngRoot::new(Some(seed)))
}

fn dataset_random_seed(dataset: &DatasetSpec) -> Option<u64> {
    match dataset {
        DatasetSpec::Synthetic(spec) => spec.random_seed,
        DatasetSpec::File(spec) => spec.random_seed,
        DatasetSpec::Public(spec) => spec.random_seed,
    }
}

const fn is_rankings_endpoint(endpoint_type: EndpointType) -> bool {
    matches!(
        endpoint_type,
        EndpointType::CohereRankings | EndpointType::HfTeiRankings | EndpointType::NimRankings
    )
}

fn compose_config(models: &ModelsSpec, rng_root: RngRoot) -> Result<ComposeConfig> {
    let mut compose = ComposeConfig::new(models.items[0].name.clone(), rng_root);
    compose.models = models
        .items
        .iter()
        .map(|item| ModelId::from(item.name.as_str()))
        .collect();
    compose.model_selector = model_selector(models, rng_root)?;
    Ok(compose)
}

pub(crate) async fn build_file_dataset(
    registry: &AiperfRegistry,
    spec: &FileDatasetSpec,
    models: &ModelsSpec,
    run_rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
    trace_prompt_storage: Arc<dyn TracePromptStoragePolicy>,
) -> Result<Dataset> {
    ensure!(
        spec.path.is_some() ^ spec.records.is_some(),
        "file dataset requires exactly one of path or records"
    );
    let rng_root = spec
        .random_seed
        .map(|seed| RngRoot::new(Some(seed)))
        .unwrap_or(run_rng_root);
    let mut compose = compose_config(models, rng_root)?;
    compose.output_length_distribution = spec.osl.as_ref().map(distribution).transpose()?;
    compose.format_options = spec.options.clone();
    compose.trace_prompt_storage = trace_prompt_storage;
    if let Some(synthesis) = &spec.synthesis {
        ensure!(
            matches!(
                spec.format.as_str(),
                "mooncake_trace" | "bailian_trace" | "burst_gpt"
            ),
            "trace synthesis is not supported by file format {:?}",
            spec.format
        );
        let block_size = spec
            .options
            .get("block_size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .unwrap_or_else(|| {
                if spec.format == "bailian_trace" {
                    16
                } else {
                    512
                }
            });
        let native_synthesis = TraceSynthesisConfig {
            speedup_ratio: synthesis.speedup_ratio,
            prefix_len_multiplier: synthesis.prefix_len_multiplier,
            prefix_root_multiplier: synthesis.prefix_root_multiplier,
            prompt_len_multiplier: synthesis.prompt_len_multiplier,
            output_len_multiplier: synthesis.output_len_multiplier,
            max_isl: synthesis.max_isl,
            max_osl: synthesis.max_osl,
            block_size,
        };
        native_synthesis.validate()?;
        compose.max_output_tokens = synthesis.max_osl;
        compose.trace_synthesis = Some(native_synthesis);
    }
    let source = match (&spec.path, &spec.records) {
        (Some(path), None) => DatasetSource::Path(path.clone()),
        (None, Some(records)) => DatasetSource::Inline(records.clone()),
        _ => unreachable!("source exclusivity validated above"),
    };
    let mut load = LoadConfig::new(source);
    load.max_rows = spec.entries;
    load.sampling_strategy = Some(spec.sampling.clone());
    if let Some(synthesis) = &spec.synthesis {
        load.max_input_tokens = synthesis.max_isl;
        load.max_output_tokens = synthesis.max_osl;
    }
    registry
        .dataset_formats()
        .build_dataset(Some(&spec.format), &load, &compose, tokenizer)
        .await
        .map_err(Into::into)
}

pub(crate) async fn build_public_dataset(
    registry: &AiperfRegistry,
    spec: &PublicDatasetSpec,
    models: &ModelsSpec,
    rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
) -> Result<Dataset> {
    ensure!(
        !spec.name.trim().is_empty(),
        "public dataset name cannot be empty"
    );
    ensure!(
        !spec.format.trim().is_empty(),
        "public dataset format cannot be empty"
    );
    let option_cap = spec
        .options
        .get("max_conversations")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok());
    let max_rows = spec.entries.or(option_cap);
    let source = match &spec.source {
        PublicDatasetSourceSpec::Url { url } => {
            ensure!(!url.trim().is_empty(), "public dataset URL cannot be empty");
            DatasetSource::Url(url.clone())
        }
        PublicDatasetSourceSpec::HuggingFace {
            dataset,
            subset,
            split,
            revision,
        } => DatasetSource::HuggingFace {
            dataset: dataset.clone(),
            config: subset.clone(),
            split: split.clone(),
            max_rows,
            revision: revision.clone(),
        },
    };
    let mut compose = compose_config(models, rng_root)?;
    compose.format_options = spec.options.clone();
    let mut load = LoadConfig::new(source);
    load.max_rows = max_rows;
    load.sampling_strategy = Some(spec.sampling.clone());
    load.options = spec.options.clone();
    registry
        .dataset_formats()
        .build_dataset(Some(&spec.format), &load, &compose, tokenizer)
        .await
        .map_err(Into::into)
}

fn synthetic_config(spec: &SyntheticDatasetSpec) -> Result<SyntheticDatasetConfig> {
    ensure!(
        spec.entries > 0,
        "synthetic dataset entries must be positive"
    );
    if let Some(prompts) = &spec.prompts {
        ensure!(
            prompts.batch_size > 0,
            "synthetic prompt batch_size must be positive"
        );
        ensure!(
            prompts.block_size.is_none_or(|value| value > 0),
            "synthetic prompt block_size must be positive when configured"
        );
    }
    ensure!(
        spec.turn_delay_ratio.is_finite() && spec.turn_delay_ratio >= 0.0,
        "synthetic turn_delay_ratio must be finite and non-negative"
    );
    let prompts = spec
        .prompts
        .as_ref()
        .and_then(|prompts| {
            prompts
                .isl
                .as_ref()
                .or_else(|| {
                    prompts
                        .sequence_distribution
                        .as_ref()
                        .and_then(|entries| entries.first())
                        .map(|entry| &entry.isl)
                })
                .map(|isl| (prompts, isl))
        })
        .map(|(prompts, isl)| -> Result<Option<SyntheticPromptConfig>> {
            let input_tokens = distribution(isl)?;
            Ok(
                (input_tokens.expected_value() > 0.0).then_some(SyntheticPromptConfig {
                    input_tokens,
                    batch_size: prompts.batch_size,
                }),
            )
        })
        .transpose()?
        .flatten();
    Ok(SyntheticDatasetConfig {
        entries: spec.entries,
        turns: distribution(&spec.turns)?,
        turn_delay_ms: distribution(&spec.turn_delay_ms)?,
        turn_delay_ratio: spec.turn_delay_ratio,
        prompts,
        prefixes: synthetic_prefixes(spec.prefix_prompts.as_ref()),
        images: spec.images.as_ref().map(synthetic_image).transpose()?,
        audio: spec.audio.as_ref().map(synthetic_audio).transpose()?,
        video: spec.video.as_ref().map(synthetic_video).transpose()?,
        rankings: spec
            .rankings
            .as_ref()
            .map(|rankings| -> Result<SyntheticRankingsConfig> {
                Ok(SyntheticRankingsConfig {
                    passages: distribution(&rankings.passages)?,
                    passage_tokens: distribution(&rankings.passage_tokens)?,
                    query_tokens: distribution(&rankings.query_tokens)?,
                })
            })
            .transpose()?,
    })
}

fn synthetic_prefixes(spec: Option<&SyntheticPrefixPromptsSpec>) -> SyntheticPrefixConfig {
    spec.map_or_else(SyntheticPrefixConfig::default, |prefixes| {
        SyntheticPrefixConfig {
            pool_size: prefixes.pool_size,
            prefix_tokens: prefixes.length,
            shared_system_tokens: prefixes.shared_system_length,
            user_context_tokens: prefixes.user_context_length,
        }
    })
}

fn synthetic_image(spec: &SyntheticImageSpec) -> Result<SyntheticImageConfig> {
    let width = distribution(&spec.width)?;
    let height = distribution(&spec.height)?;
    let dimensions_enabled = width.expected_value() > 0.0 && height.expected_value() > 0.0;
    let source = match spec.source.as_str() {
        "noise" => SyntheticImageSource::Noise,
        "assets" => SyntheticImageSource::BundledAssets,
        value => SyntheticImageSource::Directory(PathBuf::from(value)),
    };
    let format = match spec.format {
        SyntheticImageFormatSpec::Png => SyntheticImageFormat::Png,
        SyntheticImageFormatSpec::Jpeg => SyntheticImageFormat::Jpeg,
        SyntheticImageFormatSpec::Random => SyntheticImageFormat::Random,
    };
    let source_sampling = match spec.source_sampling {
        SourceImageSamplingSpec::RandomWithReplacement => {
            SourceImageSampling::RandomWithReplacement
        }
        SourceImageSamplingSpec::ShuffleCycle => SourceImageSampling::ShuffleCycle,
        SourceImageSamplingSpec::SequentialCycle => SourceImageSampling::SequentialCycle,
    };
    Ok(SyntheticImageConfig {
        batch_size: if dimensions_enabled {
            spec.batch_size
        } else {
            0
        },
        width,
        height,
        format,
        source,
        source_sampling,
    })
}

fn synthetic_audio(spec: &SyntheticAudioSpec) -> Result<SyntheticAudioConfig> {
    let duration_seconds = distribution(&spec.length)?;
    let enabled = duration_seconds.expected_value() > 0.0;
    let sample_rates_hz = spec
        .sample_rates
        .iter()
        .map(|value| khz_to_hz(*value, "audio sample rate"))
        .collect::<Result<Vec<_>>>()?;
    Ok(SyntheticAudioConfig {
        batch_size: if enabled { spec.batch_size } else { 0 },
        duration_seconds,
        format: match spec.format {
            SyntheticAudioFormatSpec::Wav => SyntheticAudioFormat::Wav,
            SyntheticAudioFormatSpec::Mp3 => SyntheticAudioFormat::Mp3,
        },
        sample_rates_hz,
        bit_depths: spec.depths.clone(),
        channels: spec.channels,
    })
}

fn synthetic_video(spec: &SyntheticVideoSpec) -> Result<SyntheticVideoConfig> {
    ensure!(
        spec.duration.is_finite() && spec.duration > 0.0,
        "synthetic video duration must be finite and positive"
    );
    Ok(SyntheticVideoConfig {
        batch_size: spec.batch_size,
        width: spec.width.unwrap_or(640),
        height: spec.height.unwrap_or(480),
        duration_seconds: spec.duration,
        frames_per_second: spec.fps,
        format: match spec.format {
            SyntheticVideoFormatSpec::Mp4 => SyntheticVideoFormat::Mp4,
            SyntheticVideoFormatSpec::Webm => SyntheticVideoFormat::WebM,
        },
        codec: spec.codec.clone(),
        pattern: match spec.synth_type {
            SyntheticVideoPatternSpec::MovingShapes => SyntheticVideoPattern::MovingShapes,
            SyntheticVideoPatternSpec::GridClock => SyntheticVideoPattern::GridClock,
            SyntheticVideoPatternSpec::Noise => SyntheticVideoPattern::Noise,
        },
        audio: SyntheticVideoAudioConfig {
            channels: spec.audio.channels,
            sample_rate_hz: khz_to_hz(spec.audio.sample_rate, "video audio sample rate")?,
            bit_depth: spec.audio.depth,
            codec: spec.audio.codec.clone(),
        },
    })
}

fn khz_to_hz(value: f64, field: &str) -> Result<u32> {
    let hz = value * 1_000.0;
    ensure!(
        value.is_finite() && value > 0.0 && hz <= f64::from(u32::MAX),
        "{field} must be finite, positive, and representable in hertz"
    );
    Ok(hz.round_ties_even() as u32)
}

fn sequence_length_distribution(
    entries: &[SequenceDistributionEntrySpec],
) -> Result<SequenceLengthDistribution> {
    let pairs = entries
        .iter()
        .map(|entry| {
            SequenceLengthPair::new_with_stddev(
                distribution_expected_i64(&entry.isl, "sequence-distribution ISL")?,
                distribution_normal_stddev(&entry.isl),
                distribution_expected_i64(&entry.osl, "sequence-distribution OSL")?,
                distribution_normal_stddev(&entry.osl),
                entry.probability,
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    SequenceLengthDistribution::new(pairs).map_err(Into::into)
}

fn distribution_expected_i64(spec: &DistributionSpec, field: &str) -> Result<i64> {
    let expected = distribution(spec)?.expected_value();
    ensure!(
        expected.is_finite() && expected > 0.0 && expected <= i64::MAX as f64,
        "{field} expected value must be positive and representable as i64"
    );
    Ok(expected as i64)
}

const fn distribution_normal_stddev(spec: &DistributionSpec) -> f64 {
    match spec {
        DistributionSpec::Normal(value) => value.stddev,
        _ => 0.0,
    }
}

pub(crate) fn default_output_tokens(dataset: &DatasetSpec) -> Result<usize> {
    let expected = match dataset {
        DatasetSpec::Synthetic(spec) => spec
            .prompts
            .as_ref()
            .and_then(|prompts| prompts.osl.as_ref())
            .map(distribution)
            .transpose()?
            .map(|distribution| distribution.expected_value().ceil())
            .filter(|value| *value > 0.0)
            .unwrap_or(1.0),
        DatasetSpec::File(spec) => spec
            .osl
            .as_ref()
            .map(distribution)
            .transpose()?
            .map(|distribution| distribution.expected_value().ceil())
            // The materialized request body preserves an absent max-token
            // field. This fallback exists only for the observer's requested
            // OSL dimension when a file row omits it.
            .unwrap_or(1.0),
        DatasetSpec::Public(_) => 1.0,
    };
    ensure!(
        expected.is_finite() && expected > 0.0 && expected <= usize::MAX as f64,
        "default OSL expected value is outside the native usize range"
    );
    Ok(expected as usize)
}

pub(crate) fn distribution(spec: &DistributionSpec) -> Result<SamplingDistribution> {
    let (distribution, min, max) = match spec {
        DistributionSpec::Fixed(value) => (
            SamplingDistribution::fixed(value.value)?,
            value.min,
            value.max,
        ),
        DistributionSpec::Normal(value) => (
            SamplingDistribution::normal(value.mean, value.stddev)?,
            value.min,
            value.max,
        ),
        DistributionSpec::LogNormal(value) => (
            SamplingDistribution::lognormal(value.mean, value.median)?,
            value.min,
            value.max,
        ),
        DistributionSpec::Multimodal(value) => (
            SamplingDistribution::multimodal(
                value
                    .peaks
                    .iter()
                    .map(|peak| {
                        Ok(PeakEntry::new(
                            distribution(&peak.distribution)?,
                            peak.weight,
                        )?)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )?,
            value.min,
            value.max,
        ),
        DistributionSpec::Empirical(value) => (
            SamplingDistribution::empirical(
                value
                    .points
                    .iter()
                    .map(|point| EmpiricalPoint::new(point.value, point.weight).map_err(Into::into))
                    .collect::<Result<Vec<_>>>()?,
            )?,
            value.min,
            value.max,
        ),
    };
    distribution.with_bounds(min, max).map_err(Into::into)
}

fn endpoint_config(spec: &EndpointSpec) -> Result<EndpointConfig> {
    EndpointConfig {
        endpoint_type: spec.endpoint_type,
        urls: spec.urls.clone(),
        path: spec.path.clone(),
        streaming: spec.streaming,
        template: spec.template.clone(),
        response_field: spec.response_field.clone(),
        request_content_type: spec.request_content_type,
        timeout_seconds: spec.timeout_seconds,
        download_video_content: spec.download_video_content,
        use_legacy_max_tokens: spec.use_legacy_max_tokens,
        use_server_token_count: spec.use_server_token_count,
        headers: spec.headers.clone(),
        api_key: spec.api_key.clone(),
        extra: (!spec.extra.is_empty()).then(|| spec.extra.clone()),
        ..EndpointConfig::default()
    }
    .validate()
    .map_err(Into::into)
}

pub(crate) fn metrics_config(spec: &MetricsSpec) -> Result<MetricsConfig> {
    let slice_duration_ns = spec
        .slice_duration_seconds
        .map(|seconds| {
            ensure!(seconds > 0.0, "metrics slice duration must be positive");
            seconds_to_ns(seconds)
        })
        .transpose()?;
    let mut slos = Vec::with_capacity(spec.slos.len());
    for (name, value) in &spec.slos {
        ensure!(value.is_finite(), "SLO {name:?} threshold must be finite");
        let metric = CATALOG
            .iter()
            .find(|metric| metric.tag.as_str() == name)
            .ok_or_else(|| anyhow!("SLO metric {name:?} is not in the native metric catalog"))?;
        ensure!(
            metric.kind == aiperf_metrics::MetricType::Record
                && !metric
                    .flags
                    .contains(aiperf_metrics::MetricFlags::NO_INDIVIDUAL_RECORDS),
            "SLO metric {name:?} does not produce one value per request"
        );
        slos.push(SloThreshold::from_display(metric.tag, *value)?);
    }
    Ok(MetricsConfig {
        slice_duration_ns,
        slos,
        ..MetricsConfig::default()
    })
}

pub(crate) fn metrics_phase(spec: &PhaseSpec) -> Result<MetricsPhase> {
    match spec.common().name.as_str() {
        "warmup" => Ok(MetricsPhase::Warmup),
        "profiling" => Ok(MetricsPhase::Profiling),
        name => bail!("unsupported phase name {name:?}"),
    }
}

fn artifact_path(root: &Path, relative: &Path, field: &str) -> Result<PathBuf> {
    ensure!(
        !relative.as_os_str().is_empty() && !relative.is_absolute(),
        "artifact {field} must be a non-empty relative path"
    );
    ensure!(
        relative
            .components()
            .all(|component| matches!(component, Component::Normal(_))),
        "artifact {field} cannot contain parent, root, or current-directory components"
    );
    Ok(root.join(relative))
}

/// Resolve the current phase's outbound handoff from authored phase order.
// Config v2 authors `seamless` on the subsequent phase, while the native
// PhaseConfig owns the current -> next handoff. Preserve that direction once
// at the adapter seam. Source: `src/aiperf/config/config.py:522-530`.
pub(crate) fn phase_seamless_to_next(phases: &[PhaseSpec], phase_index: usize) -> bool {
    phases
        .get(phase_index + 1)
        .is_some_and(|next| next.common().seamless)
}

pub(crate) fn phase_config(spec: &PhaseSpec, seamless_to_next: bool) -> Result<PhaseConfig> {
    let common = spec.common();
    let kind = match common.name.as_str() {
        "warmup" => PhaseKind::Warmup,
        "profiling" => PhaseKind::Profiling,
        _ => bail!("unsupported phase name {:?}", common.name),
    };
    let stop = StopConfig {
        total_expected_requests: common.requests,
        expected_num_sessions: common.sessions,
        expected_duration_ns: common.duration.map(seconds_to_ns).transpose()?,
    };
    let mut phase = PhaseConfig::new(&common.name, kind, stop)
        .with_seamless(seamless_to_next)
        .with_concurrency(spec.concurrency(), common.prefill_concurrency);
    if let Some(grace) = common.grace_period {
        phase = phase.with_grace_period(GracePeriod::Finite(seconds_to_ns(grace)?));
    }
    phase.validate()?;
    Ok(phase)
}

fn ancillary_policies(
    spec: &PhaseSpec,
    urls: &[String],
    rng_root: RngRoot,
) -> Result<ScheduledAncillaryPolicies> {
    let cancellation_policy = spec
        .common()
        .cancellation
        .map(|cancellation| -> Result<Box<dyn CancellationPolicy>> {
            let policy =
                BernoulliFixedDelay::new(Some(cancellation.rate), cancellation.delay, rng_root)?;
            Ok(Box::new(policy) as Box<dyn CancellationPolicy>)
        })
        .transpose()?;
    let url_selector = (urls.len() > 1)
        .then(|| {
            RoundRobinUrlSelector::new(urls.to_vec())
                .map(|selector| Box::new(selector) as Box<dyn UrlSelector>)
        })
        .transpose()?;
    Ok(ScheduledAncillaryPolicies {
        cancellation_policy,
        url_selector,
        phase: if spec.common().name == "warmup" {
            aiperf_timing::Phase::Warmup
        } else {
            aiperf_timing::Phase::Profiling
        },
    })
}

/// Phase-local roots for independently randomized ramp actuators.
///
/// The controller derives this layer before constructing a strategy. A
/// stochastic strategy such as `PoissonRamp` then derives its curve-local
/// `timing.ramp.poisson` stream, producing the stable hierarchy
/// `phase -> actuator -> curve` without coupling simultaneous actuators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RampActuatorRngRoots {
    concurrency: RngRoot,
    prefill_concurrency: RngRoot,
    request_rate: RngRoot,
}

impl RampActuatorRngRoots {
    /// Derive every actuator root as a pure function of one phase-local root.
    pub(crate) fn from_phase_root(root: RngRoot) -> Self {
        Self {
            concurrency: root.derive_root(namespace::TIMING_RAMP_CONCURRENCY),
            prefill_concurrency: root.derive_root(namespace::TIMING_RAMP_PREFILL_CONCURRENCY),
            request_rate: root.derive_root(namespace::TIMING_RAMP_REQUEST_RATE),
        }
    }

    /// Root for session-concurrency ramps, including user-centric admission.
    pub(crate) const fn concurrency(self) -> RngRoot {
        self.concurrency
    }

    /// Root for prefill-concurrency ramps.
    pub(crate) const fn prefill_concurrency(self) -> RngRoot {
        self.prefill_concurrency
    }

    /// Root for request-rate ramps.
    pub(crate) const fn request_rate(self) -> RngRoot {
        self.request_rate
    }
}

fn ramp_controller(
    spec: &PhaseSpec,
    clock: Rc<dyn Clock>,
    intervals: Rc<RefCell<Box<dyn aiperf_timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    rng_root: RngRoot,
) -> Result<Rc<dyn ScheduledPhaseController>> {
    let common = spec.common();
    let rng_roots = RampActuatorRngRoots::from_phase_root(rng_root);
    let target_rate = spec
        .request_arrival()
        .and_then(|(_, target_rate, _)| target_rate);
    let mut drivers = Vec::new();
    if let Some(ramp) = &common.concurrency_ramp {
        let target = spec
            .concurrency()
            .ok_or_else(|| anyhow!("concurrency_ramp requires a concurrency target"))?;
        let slots = session_slots
            .clone()
            .ok_or_else(|| anyhow!("concurrency_ramp requires session admission"))?;
        let strategy = ramp_strategy(ramp, 1.0, target as f64, false, rng_roots.concurrency())?;
        drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
            slots.set_limit(value.round() as usize)
        }));
    }
    if let Some(ramp) = &common.prefill_ramp {
        let target = common
            .prefill_concurrency
            .ok_or_else(|| anyhow!("prefill_ramp requires prefill_concurrency"))?;
        let slots = prefill_slots
            .clone()
            .ok_or_else(|| anyhow!("prefill_ramp requires prefill admission"))?;
        let strategy = ramp_strategy(
            ramp,
            1.0,
            target as f64,
            false,
            rng_roots.prefill_concurrency(),
        )?;
        drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
            slots.set_limit(value.round() as usize)
        }));
    }
    if let Some(ramp) = &common.rate_ramp {
        let target = target_rate.ok_or_else(|| anyhow!("rate_ramp requires a rate phase"))?;
        let duration_ns = seconds_to_u64_ns(ramp.duration)?;
        let start = target * RATE_RAMP_UPDATE_INTERVAL_NS as f64 / duration_ns as f64;
        let strategy = ramp_strategy(ramp, start, target, true, rng_roots.request_rate())?;
        drivers.push(RampDriver::new(clock, strategy, move |value| {
            intervals.borrow_mut().set_rate(value)
        }));
    }
    if drivers.is_empty() {
        Ok(Rc::new(aiperf::phase_runtime::NoopScheduledPhaseController))
    } else {
        Ok(Rc::new(RampScheduledPhaseController::new(drivers)))
    }
}

#[allow(clippy::too_many_arguments)]
fn adaptive_runtime_extension(
    phase: &PhaseSpec,
    benchmark_id: &str,
    artifact_dir: &Path,
    intervals: Rc<RefCell<Box<dyn aiperf_timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    user_target: Option<Rc<dyn UserTarget>>,
) -> Result<Option<Rc<dyn ScheduledRuntimeExtension>>> {
    let Some(config) = adaptive_run_config(phase, benchmark_id, artifact_dir)? else {
        return Ok(None);
    };
    match config.control_variable {
        AdaptiveControlVariable::Concurrency => {
            ensure!(
                session_slots.is_some(),
                "adaptive concurrency requires session admission"
            );
            ensure!(
                phase.common().concurrency_ramp.is_none(),
                "adaptive concurrency cannot be combined with concurrency_ramp"
            );
        }
        AdaptiveControlVariable::PrefillConcurrency => {
            ensure!(
                !matches!(phase, PhaseSpec::UserCentric { .. }),
                "user_centric phases do not expose prefill admission"
            );
            ensure!(
                prefill_slots.is_some(),
                "adaptive prefill_concurrency requires prefill admission"
            );
            ensure!(
                phase.common().prefill_ramp.is_none(),
                "adaptive prefill_concurrency cannot be combined with prefill_ramp"
            );
            let session_target = phase.concurrency().ok_or_else(|| {
                anyhow!("adaptive prefill_concurrency requires a session concurrency cap")
            })?;
            ensure!(
                config.maximum <= session_target as f64,
                "adaptive prefill_concurrency maximum must be <= concurrency"
            );
        }
        AdaptiveControlVariable::RequestRate => {
            ensure!(
                matches!(
                    phase,
                    PhaseSpec::Poisson { .. }
                        | PhaseSpec::Gamma { .. }
                        | PhaseSpec::Constant { .. }
                ),
                "adaptive request_rate requires a rate-controlled phase"
            );
            ensure!(
                phase.common().rate_ramp.is_none(),
                "adaptive request_rate cannot be combined with rate_ramp"
            );
        }
        AdaptiveControlVariable::Users => {
            ensure!(
                matches!(phase, PhaseSpec::UserCentric { .. }) && user_target.is_some(),
                "adaptive users requires a user_centric phase"
            );
        }
    }
    Ok(Some(Rc::new(AdaptiveRuntimeExtension {
        config,
        intervals,
        session_slots,
        prefill_slots,
        user_target,
        session_target: phase.concurrency(),
        prefill_target: phase.common().prefill_concurrency,
    })))
}

pub(crate) fn adaptive_run_config(
    phase: &PhaseSpec,
    benchmark_id: &str,
    artifact_dir: &Path,
) -> Result<Option<AdaptiveRunConfig>> {
    let Some(spec) = phase.common().adaptive_scale.as_ref() else {
        return Ok(None);
    };
    ensure!(
        phase.common().name == "profiling",
        "adaptive_scale is supported only on profiling phases"
    );
    ensure!(
        phase.common().duration.is_some(),
        "adaptive_scale requires a phase duration"
    );
    ensure!(
        !matches!(phase, PhaseSpec::FixedSchedule { .. }),
        "adaptive_scale is not defined for fixed_schedule phases"
    );
    let control_variable = match spec.control_variable {
        AdaptiveControlVariableSpec::Concurrency => AdaptiveControlVariable::Concurrency,
        AdaptiveControlVariableSpec::PrefillConcurrency => {
            AdaptiveControlVariable::PrefillConcurrency
        }
        AdaptiveControlVariableSpec::RequestRate => AdaptiveControlVariable::RequestRate,
        AdaptiveControlVariableSpec::Users => AdaptiveControlVariable::Users,
    };
    let step = match spec.step_policy {
        AdaptiveStepPolicySpec::SlaMargin => AdaptiveStepConfig::SlaMargin {
            base_step: spec.base_step,
            max_step_multiplier: spec.max_step_multiplier,
        },
        AdaptiveStepPolicySpec::FixedPercentStep => AdaptiveStepConfig::FixedPercent {
            percent: spec.step_percent,
        },
    };
    let sla_filters = spec
        .sla_filters
        .iter()
        .map(|filter| {
            SlaFilter::new(
                filter.metric_tag.clone(),
                filter.stat.parse()?,
                filter.op.parse()?,
                filter.threshold,
            )
            .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Some(AdaptiveRunConfig {
        control_variable,
        minimum: spec.minimum,
        maximum: spec.maximum,
        assessment_period_ns: positive_seconds_to_ns(
            spec.assessment_period_seconds,
            "adaptive assessment period",
        )?,
        sustain_duration_ns: positive_seconds_to_ns(
            spec.sustain_duration_seconds,
            "adaptive sustain duration",
        )?,
        min_completed_requests: spec.min_completed_requests,
        sla_filters,
        step,
        artifact_dir: artifact_dir.to_path_buf(),
        correlation: CorrelationContext {
            run_id: Some(benchmark_id.to_string()),
            phase_id: phase.common().name.clone(),
            phase_name: Some(phase.common().name.clone()),
            ..CorrelationContext::default()
        },
    }))
}

pub(crate) fn integer_adaptive_bound(value: f64, label: &str) -> Result<usize> {
    ensure!(
        value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value <= usize::MAX as f64,
        "adaptive {label} must be an integer in the usize range"
    );
    Ok(value as usize)
}

struct AdaptiveRuntimeExtension {
    config: AdaptiveRunConfig,
    intervals: Rc<RefCell<Box<dyn aiperf_timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    user_target: Option<Rc<dyn UserTarget>>,
    session_target: Option<usize>,
    prefill_target: Option<usize>,
}

impl ScheduledRuntimeExtension for AdaptiveRuntimeExtension {
    fn build(
        &self,
        clock: Rc<dyn Clock>,
        observer_origin_ns: i64,
        phase_start_ns: i64,
        delegate: Rc<dyn RequestObserver>,
        controller: Rc<dyn ScheduledPhaseController>,
    ) -> Result<ScheduledRuntimeExtensionParts> {
        if self.config.control_variable != AdaptiveControlVariable::Concurrency
            && let (Some(slots), Some(target)) = (&self.session_slots, self.session_target)
        {
            slots.set_limit(target);
        }
        if self.config.control_variable != AdaptiveControlVariable::PrefillConcurrency
            && let (Some(slots), Some(target)) = (&self.prefill_slots, self.prefill_target)
        {
            slots.set_limit(target);
        }
        let built = build_adaptive_with_origins(
            self.config.clone(),
            clock,
            observer_origin_ns,
            phase_start_ns,
            delegate,
            self.intervals.clone(),
            self.session_slots.clone(),
            self.prefill_slots.clone(),
            self.user_target.clone(),
        )?;
        let gate: Rc<dyn IssuanceGate> = built.scale.clone();
        let controller: Rc<dyn ScheduledPhaseController> = Rc::new(
            AdaptiveScheduledPhaseController::new(built.scale, controller),
        );
        Ok(ScheduledRuntimeExtensionParts {
            observer: built.observer,
            issuance_gate: Some(gate),
            controller,
        })
    }
}

pub(crate) struct AdaptiveScheduledPhaseController {
    scale: Rc<AdaptiveScale>,
    delegate: Rc<dyn ScheduledPhaseController>,
    assessment: RefCell<Option<tokio::task::JoinHandle<()>>>,
}

impl AdaptiveScheduledPhaseController {
    pub(crate) fn new(
        scale: Rc<AdaptiveScale>,
        delegate: Rc<dyn ScheduledPhaseController>,
    ) -> Self {
        Self {
            scale,
            delegate,
            assessment: RefCell::new(None),
        }
    }
}

impl ScheduledPhaseController for AdaptiveScheduledPhaseController {
    fn start(&self) -> Result<()> {
        ensure!(
            self.assessment.borrow().is_none(),
            "adaptive phase controller was already started"
        );
        self.delegate.start()?;
        self.scale.start()?;
        let scale = self.scale.clone();
        *self.assessment.borrow_mut() = Some(tokio::task::spawn_local(scale.assessment_loop()));
        Ok(())
    }

    fn stop(&self) -> aiperf_timing::LocalPhaseFuture<Result<()>> {
        self.scale.deactivate();
        let assessment = self.assessment.borrow_mut().take();
        let scale = self.scale.clone();
        let delegate = self.delegate.clone();
        Box::pin(async move {
            let mut errors = Vec::new();
            if let Some(assessment) = assessment {
                assessment.abort();
                if let Err(error) = assessment.await
                    && !error.is_cancelled()
                {
                    errors.push(format!("adaptive assessment task: {error}"));
                }
            }
            if let Err(error) = scale.complete_phase() {
                errors.push(format!("completing adaptive phase: {error}"));
            }
            if let Some(error) = scale.last_error() {
                errors.push(format!("adaptive assessment failed: {error}"));
            }
            if let Err(error) = delegate.stop().await {
                errors.push(format!("stopping delegated phase controller: {error:#}"));
            }
            if errors.is_empty() {
                Ok(())
            } else {
                bail!(errors.join("; "))
            }
        })
    }

    fn wait_until_stop(&self) -> aiperf_timing::LocalPhaseFuture<()> {
        let scale = self.scale.clone();
        Box::pin(async move { scale.wait_until_stop_sending().await })
    }
}

pub(crate) fn ramp_strategy(
    ramp: &RampSpec,
    start: f64,
    target: f64,
    continuous: bool,
    rng_root: RngRoot,
) -> Result<Box<dyn RampStrategy>> {
    let mut config = RamperConfig::from_seconds(start, target, ramp.duration)?;
    if continuous {
        config = config.with_update_interval_ns(RATE_RAMP_UPDATE_INTERVAL_NS)?;
    }
    Ok(match ramp.strategy {
        RampStrategySpec::Linear => Box::new(LinearRamp::new(config)),
        RampStrategySpec::Exponential => Box::new(ExponentialRamp::new(config)),
        RampStrategySpec::Poisson => Box::new(PoissonRamp::new(config, rng_root)?),
    })
}

fn model_selector(models: &ModelsSpec, rng_root: RngRoot) -> Result<Arc<dyn ModelSelectorFactory>> {
    match models.strategy {
        ModelSelectionStrategy::RoundRobin => Ok(Arc::new(RoundRobinModelSelectorFactory)),
        ModelSelectionStrategy::Random => Ok(Arc::new(RandomModelSelectorFactory)),
        ModelSelectionStrategy::Weighted => {
            let weights = models
                .items
                .iter()
                .map(|item| {
                    item.weight.ok_or_else(|| {
                        anyhow!("weighted model selection requires every model weight")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Arc::new(WeightedModelSelectorFactory { weights, rng_root }))
        }
    }
}

struct WeightedModelSelectorFactory {
    weights: Vec<f64>,
    rng_root: RngRoot,
}

impl ModelSelectorFactory for WeightedModelSelectorFactory {
    fn create(
        &self,
        models: &[ModelId],
        _root: RngRoot,
    ) -> aiperf_dataset::Result<Box<dyn ModelSelector>> {
        if models.len() != self.weights.len() || models.is_empty() {
            return Err(aiperf_dataset::DatasetError::Validation(
                "weighted model values and weights must have the same non-zero length".into(),
            ));
        }
        let total = self.weights.iter().sum::<f64>();
        if !self
            .weights
            .iter()
            .all(|weight| weight.is_finite() && *weight >= 0.0)
            || !(0.99..=1.01).contains(&total)
        {
            return Err(aiperf_dataset::DatasetError::Validation(
                "weighted model weights must be finite, non-negative, and sum to 1.0 (+/-0.01)"
                    .into(),
            ));
        }
        Ok(Box::new(WeightedModelSelector {
            models: models.to_vec(),
            weights: self.weights.clone(),
            rng: RandomGenerator::from_seed(
                self.rng_root.derive_seed("runner.model.weighted_selection"),
            ),
        }))
    }
}

struct WeightedModelSelector {
    models: Vec<ModelId>,
    weights: Vec<f64>,
    rng: RandomGenerator,
}

impl ModelSelector for WeightedModelSelector {
    fn next(&mut self) -> ModelId {
        self.rng
            .weighted_choice(&self.models, Some(&self.weights))
            .expect("factory validates weighted model selection")
    }
}

pub(crate) fn load_tokenizer(spec: Option<&str>) -> Result<Arc<dyn TextTokenizer>> {
    let spec = spec.unwrap_or("builtin");
    let path = Path::new(spec);
    if path.is_dir() {
        return Ok(Arc::new(HuggingFaceTokenizer::from_directory(path)?));
    }
    if path.is_file() {
        return Ok(Arc::new(HuggingFaceTokenizer::from_file(path)?));
    }
    let encoding = spec.parse::<TiktokenEncoding>()?;
    Ok(Arc::new(TiktokenTokenizer::new(encoding)))
}

fn seconds_to_ns(value: f64) -> Result<i64> {
    let nanos = seconds_to_u64_ns(value)?;
    i64::try_from(nanos).map_err(|_| anyhow!("duration is outside the i64 nanosecond range"))
}

pub(crate) fn seconds_to_u64_ns(value: f64) -> Result<u64> {
    ensure!(
        value.is_finite() && value >= 0.0 && value * 1_000_000_000.0 <= i64::MAX as f64,
        "duration must be finite, non-negative, and representable in nanoseconds"
    );
    Ok((value * 1_000_000_000.0).round_ties_even() as u64)
}

struct CaptureIdentity {
    uuid: Uuid,
    x_correlation_id: String,
}

struct RunCapture {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
    observer: Rc<NativeMetricsObserver>,
    identities: RefCell<Vec<CaptureIdentity>>,
    outputs: RefCell<HashMap<Uuid, CapturedModelOutput>>,
    raw_enabled: bool,
    raw_exchanges: RefCell<HashMap<Uuid, CapturedHttpExchange>>,
}

impl RunCapture {
    fn new(clock: Rc<dyn Clock>, origin_ns: i64, config: MetricsConfig, raw_enabled: bool) -> Self {
        Self {
            observer: Rc::new(NativeMetricsObserver::new(clock.clone(), origin_ns, config)),
            clock,
            origin_ns,
            identities: RefCell::new(Vec::new()),
            outputs: RefCell::new(HashMap::new()),
            raw_enabled,
            raw_exchanges: RefCell::new(HashMap::new()),
        }
    }

    fn begin(&self, turn: &TurnToSend) {
        self.identities.borrow_mut().push(CaptureIdentity {
            uuid: turn.uuid,
            x_correlation_id: turn.x_correlation_id.clone(),
        });
        self.observer.register_metadata(
            turn.uuid,
            RequestMetricMetadata {
                turn_index: u32::try_from(turn.turn_index).unwrap_or(u32::MAX),
                conversation_id: Some(turn.conversation_id.clone()),
                audio_duration_s: turn.audio_duration_seconds,
                ..RequestMetricMetadata::default()
            },
        );
        let arrival_ms = self.clock.now_ns().saturating_sub(self.origin_ns) as f64 / 1_000_000.0;
        self.observer.on_arrival(
            turn.uuid,
            arrival_ms,
            turn.input_length,
            turn.max_output_tokens,
        );
    }

    fn label(&self, credit: &IssuedCredit, phase: MetricsPhase, has_credit_timestamp: bool) {
        self.observer.register_metadata(
            credit.turn.uuid,
            RequestMetricMetadata {
                phase,
                session_num: Some(credit.id),
                turn_index: u32::try_from(credit.turn.turn_index).unwrap_or(u32::MAX),
                conversation_id: Some(credit.turn.conversation_id.clone()),
                audio_duration_s: credit.turn.audio_duration_seconds,
                has_credit_timestamp,
                ..RequestMetricMetadata::default()
            },
        );
    }

    fn record_model_output(
        &self,
        uuid: Uuid,
        flattened_text: &str,
        visible_text: Option<&str>,
        reasoning_text: Option<&str>,
    ) -> Result<()> {
        ensure!(
            self.outputs
                .borrow_mut()
                .insert(
                    uuid,
                    CapturedModelOutput::from_parts(flattened_text, visible_text, reasoning_text),
                )
                .is_none(),
            "native model output was recorded more than once for request {uuid}"
        );
        Ok(())
    }

    fn record_http_exchange(
        &self,
        uuid: Uuid,
        request_payload: Vec<u8>,
        record: aiperf_transport_http::models::RequestRecord,
    ) -> Result<()> {
        if !self.raw_enabled {
            return Ok(());
        }
        ensure!(
            self.raw_exchanges
                .borrow_mut()
                .insert(
                    uuid,
                    CapturedHttpExchange {
                        request_payload,
                        record,
                    },
                )
                .is_none(),
            "native HTTP exchange was recorded more than once for request {uuid}"
        );
        Ok(())
    }

    fn snapshot(&self, credit: &IssuedCredit) -> Result<CapturedRecord> {
        let mut ingest = self
            .observer
            .snapshot_record(credit.turn.uuid, credit.id)
            .ok_or_else(|| {
                anyhow!(
                    "terminal request {} was absent from native metric capture",
                    credit.turn.uuid
                )
            })?;
        if ingest.admit_ns.is_some() {
            ingest.admit_ns = Some(credit.issued_ns.saturating_sub(self.origin_ns));
        }
        Ok(CapturedRecord {
            uuid: credit.turn.uuid,
            x_correlation_id: credit.turn.x_correlation_id.clone(),
            output: self
                .outputs
                .borrow()
                .get(&credit.turn.uuid)
                .cloned()
                .unwrap_or_default(),
            raw: None,
            ingest,
        })
    }

    fn finish(&self, issued_times: &HashMap<Uuid, i64>) -> Result<Vec<CapturedRecord>> {
        let collection = self.observer.finish_with_records();
        let identities = self.identities.borrow();
        let outputs = self.outputs.borrow();
        let mut raw_exchanges = self.raw_exchanges.take();
        ensure!(
            collection.records.len() == identities.len(),
            "native record capture finalized {} records for {} dispatched identities",
            collection.records.len(),
            identities.len()
        );
        collection
            .records
            .into_iter()
            .zip(identities.iter())
            .map(|(mut ingest, identity)| {
                ensure!(
                    ingest.correlation_id == identity.uuid.to_string(),
                    "native record arrival order diverged from dispatch identity order"
                );
                if ingest.admit_ns.is_some() {
                    ingest.admit_ns = Some(*issued_times.get(&identity.uuid).ok_or_else(|| {
                        anyhow!("captured request {} has no issuer timestamp", identity.uuid)
                    })?);
                }
                Ok(CapturedRecord {
                    uuid: identity.uuid,
                    x_correlation_id: identity.x_correlation_id.clone(),
                    output: outputs.get(&identity.uuid).cloned().unwrap_or_default(),
                    raw: raw_exchanges.remove(&identity.uuid),
                    ingest,
                })
            })
            .collect()
    }
}

struct CapturePhaseProcessor {
    capture: Rc<RunCapture>,
    phase: MetricsPhase,
    has_credit_timestamp: bool,
    live_sink: Option<Rc<dyn LiveResultsSink>>,
}

#[async_trait(?Send)]
impl TurnRecordProcessor for CapturePhaseProcessor {
    async fn process(&self, credit: &IssuedCredit, _outcome: &TurnDispatchOutcome) -> Result<()> {
        self.capture
            .label(credit, self.phase, self.has_credit_timestamp);
        if let Some(sink) = &self.live_sink {
            sink.emit_record(&self.capture.snapshot(credit)?);
        }
        Ok(())
    }
}

struct DualObserver<'a> {
    runtime: &'a dyn RequestObserver,
    capture: &'a dyn RequestObserver,
}

impl RequestObserver for DualObserver<'_> {
    fn on_arrival(&self, uuid: Uuid, at_ms: f64, input: usize, output: usize) {
        self.runtime.on_arrival(uuid, at_ms, input, output);
        self.capture.on_arrival(uuid, at_ms, input, output);
    }

    fn on_admit(&self, uuid: Uuid, at_ms: f64, reused_input_tokens: usize) {
        self.runtime.on_admit(uuid, at_ms, reused_input_tokens);
        self.capture.on_admit(uuid, at_ms, reused_input_tokens);
    }

    fn on_token(&self, uuid: Uuid, at_ms: f64) {
        self.runtime.on_token(uuid, at_ms);
        self.capture.on_token(uuid, at_ms);
    }

    fn on_classified_token(&self, uuid: Uuid, at_ms: f64, kind: ObservedTokenKind) {
        self.runtime.on_classified_token(uuid, at_ms, kind);
        self.capture.on_classified_token(uuid, at_ms, kind);
    }

    fn on_usage(&self, uuid: Uuid, usage: ObservedUsage) {
        self.runtime.on_usage(uuid, usage);
        self.capture.on_usage(uuid, usage);
    }

    fn on_endpoint_metrics(&self, uuid: Uuid, metrics: ObservedEndpointMetrics) {
        self.runtime.on_endpoint_metrics(uuid, metrics);
        self.capture.on_endpoint_metrics(uuid, metrics);
    }

    fn on_terminal(&self, uuid: Uuid, status: ReplayTerminalStatus) {
        self.runtime.on_terminal(uuid, status);
        self.capture.on_terminal(uuid, status);
    }
}

struct ConfiguredDispatcher {
    execution_backend: Rc<dyn HttpTurnExecutionBackend>,
    model: String,
    capture: Rc<RunCapture>,
}

#[async_trait(?Send)]
impl TurnDispatcher for ConfiguredDispatcher {
    fn inference_dimensions(&self, turn: &TurnToSend) -> InferenceDimensions {
        self.execution_backend.inference_dimensions(turn)
    }

    async fn dispatch_turn(
        &self,
        turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        let uuid = turn.uuid;
        self.capture.begin(&turn);
        let tee = DualObserver {
            runtime: observer,
            capture: self.capture.observer.as_ref(),
        };
        let turn = PreparedHttpTurn::from_turn(turn, &self.model);
        let collected = self
            .execution_backend
            .execute_turn(turn, &tee, on_first_token)
            .await;
        match collected {
            Ok(collected) => {
                let outcome = collected.outcome;
                self.capture.record_http_exchange(
                    uuid,
                    collected.request_payload.to_vec(),
                    collected.record,
                )?;
                self.capture.record_model_output(
                    uuid,
                    &outcome.response_text,
                    outcome.model_response.content.as_deref(),
                    outcome.model_response.reasoning.as_deref(),
                )?;
                self.capture.observer.record_response(
                    uuid,
                    NativeResponseMetadata {
                        start_ns: Some(outcome.start_ns),
                        end_ns: Some(outcome.end_ns),
                        prompt_tokens: outcome.prompt_tokens,
                        completion_tokens: outcome.completion_tokens,
                        http: outcome.http,
                    },
                );
                Ok(outcome)
            }
            Err(error) => {
                let now = self.capture.clock.now_ns();
                self.capture
                    .observer
                    .on_terminal(uuid, ReplayTerminalStatus::Failed);
                self.capture.observer.record_response(
                    uuid,
                    NativeResponseMetadata {
                        start_ns: Some(now),
                        end_ns: Some(now),
                        ..NativeResponseMetadata::default()
                    },
                );
                Err(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use aiperf_graph::errors::TraceError;
    use aiperf_graph::execution::GraphTraceExecutionBackend;
    use aiperf_graph::placement::{GraphPlacementError, GraphTraceExecutionBackendFactory};
    use serde_json::json;
    use tokio::sync::Notify;

    use super::*;

    fn synthetic(value: serde_json::Value) -> SyntheticDatasetSpec {
        serde_json::from_value(value).unwrap()
    }

    fn models() -> ModelsSpec {
        serde_json::from_value(json!({
            "strategy": "round_robin",
            "items": [{"name": "mock-model"}]
        }))
        .unwrap()
    }

    #[derive(Debug)]
    struct UnusedHttpPlacement;

    impl HttpExecutionBackendFactory for UnusedHttpPlacement {
        fn build(
            &self,
            _config: HttpExecutionBackendConfig,
        ) -> Result<Rc<dyn HttpTurnExecutionBackend>> {
            panic!("direct graph execution must not construct HTTP turn placement")
        }
    }

    #[derive(Debug)]
    struct RejectingSidecarFactory {
        artifact_target: PathBuf,
        preparations: Arc<AtomicUsize>,
    }

    #[async_trait(?Send)]
    impl NativeSidecarResourceFactory for RejectingSidecarFactory {
        async fn prepare(&self, _run: &NativeRunSpec) -> Result<PreparedNativeSidecarResources> {
            assert!(
                !self.artifact_target.exists(),
                "sidecar preparation must precede artifact creation"
            );
            self.preparations.fetch_add(1, Ordering::SeqCst);
            bail!("intentional sidecar preparation failure")
        }
    }

    struct RecordingGraphPlacement {
        builds: Arc<AtomicUsize>,
        traces: Arc<AtomicUsize>,
        prefill_updates: Arc<AtomicUsize>,
    }

    impl RunnerGraphPlacementFactory for RecordingGraphPlacement {
        fn build(
            &self,
            worker_count: usize,
            _worker_factory: Arc<dyn GraphTraceExecutionBackendFactory>,
        ) -> Result<Rc<dyn aiperf_graph::execution::GraphTraceExecutionBackend>, GraphPlacementError>
        {
            assert_eq!(worker_count, 3);
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(Rc::new(RecordingGraphBackend {
                traces: self.traces.clone(),
                prefill_updates: self.prefill_updates.clone(),
            }))
        }
    }

    struct RecordingGraphBackend {
        traces: Arc<AtomicUsize>,
        prefill_updates: Arc<AtomicUsize>,
    }

    struct CancellingGraphPlacement {
        builds: Arc<AtomicUsize>,
        executions: Arc<AtomicUsize>,
        cancellations: Arc<AtomicUsize>,
        cancelled: Arc<AtomicBool>,
        wake: Arc<Notify>,
    }

    impl RunnerGraphPlacementFactory for CancellingGraphPlacement {
        fn build(
            &self,
            worker_count: usize,
            _worker_factory: Arc<dyn GraphTraceExecutionBackendFactory>,
        ) -> Result<Rc<dyn GraphTraceExecutionBackend>, GraphPlacementError> {
            assert_eq!(worker_count, 1);
            self.builds.fetch_add(1, Ordering::SeqCst);
            Ok(Rc::new(BlockingUntilCancelledGraphBackend {
                executions: self.executions.clone(),
                cancellations: self.cancellations.clone(),
                cancelled: self.cancelled.clone(),
                wake: self.wake.clone(),
            }))
        }
    }

    struct BlockingUntilCancelledGraphBackend {
        executions: Arc<AtomicUsize>,
        cancellations: Arc<AtomicUsize>,
        cancelled: Arc<AtomicBool>,
        wake: Arc<Notify>,
    }

    #[test]
    fn sidecar_resource_factory_finishes_before_artifact_creation() {
        let root = tempfile::tempdir().unwrap();
        let artifact_target = root.path().join("not-created");
        let request: RunRequest = serde_json::from_value(json!({
            "protocol_version": 1,
            "run": {
                "benchmark_id": "sidecar-preparation-order",
                "artifact_dir": artifact_target.clone(),
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "urls": ["http://must-not-be-contacted.invalid"],
                    "type": "chat",
                    "streaming": true
                },
                "dataset": {
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {
                        "isl": {"value": 1.0},
                        "osl": {"value": 1.0}
                    }
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "requests": 1,
                    "concurrency": 1
                }],
                "metrics": {},
                "artifacts": {}
            }
        }))
        .unwrap();
        let plan = NativeRunPlan::try_from(request).unwrap();
        let preparations = Arc::new(AtomicUsize::new(0));
        let factory = RejectingSidecarFactory {
            artifact_target: artifact_target.clone(),
            preparations: preparations.clone(),
        };
        let registry = AiperfRegistry::builtin().unwrap();

        let error = execute_prepared_native_plan_uncommitted_with_all_factories(
            plan,
            &UnusedHttpPlacement,
            &NativeRunnerGraphPlacementFactory,
            &registry,
            &factory,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("preparing native sidecar resources")
        );
        assert_eq!(preparations.load(Ordering::SeqCst), 1);
        assert!(!artifact_target.exists());
    }

    #[async_trait::async_trait(?Send)]
    impl aiperf_graph::execution::GraphTraceExecutionBackend for RecordingGraphBackend {
        async fn execute_trace(
            &self,
            _plan: aiperf_graph::model::GraphTracePlan,
        ) -> Result<(), TraceError> {
            self.traces.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn set_prefill_limit(&self, limit: usize) -> Result<(), TraceError> {
            self.prefill_updates.store(limit, Ordering::SeqCst);
            Ok(())
        }
    }

    #[async_trait::async_trait(?Send)]
    impl GraphTraceExecutionBackend for BlockingUntilCancelledGraphBackend {
        async fn execute_trace(
            &self,
            _plan: aiperf_graph::model::GraphTracePlan,
        ) -> Result<(), TraceError> {
            self.executions.fetch_add(1, Ordering::SeqCst);
            loop {
                let notified = self.wake.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                if self.cancelled.load(Ordering::SeqCst) {
                    return Err(TraceError::Cancelled(
                        "cancelled by the graph phase lifecycle".into(),
                    ));
                }
                notified.await;
            }
        }

        fn cancel_inflight(&self) -> Result<(), TraceError> {
            self.cancellations.fetch_add(1, Ordering::SeqCst);
            self.cancelled.store(true, Ordering::SeqCst);
            self.wake.notify_waiters();
            Ok(())
        }
    }

    #[test]
    fn graph_coordinator_accepts_injected_whole_trace_placement() {
        let artifacts = tempfile::tempdir().unwrap();
        let request: RunRequest = serde_json::from_value(json!({
            "protocol_version": 1,
            "run": {
                "benchmark_id": "injected-graph-placement",
                "workers": 3,
                "artifact_dir": artifacts.path(),
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "urls": ["http://127.0.0.1:1"],
                    "type": "chat",
                    "streaming": true
                },
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "records": [
                        {
                            "session_id": "root",
                            "turns": [{
                                "messages": [{"role": "user", "content": "hello"}],
                                "forks": ["child"]
                            }]
                        },
                        {
                            "session_id": "child",
                            "turns": [{
                                "messages": [{"role": "user", "content": "child"}]
                            }]
                        }
                    ]
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "sessions": 1,
                    "concurrency": 1
                }],
                "metrics": {},
                "artifacts": {}
            }
        }))
        .unwrap();
        let builds = Arc::new(AtomicUsize::new(0));
        let traces = Arc::new(AtomicUsize::new(0));
        let prefill_updates = Arc::new(AtomicUsize::new(0));
        let placement = RecordingGraphPlacement {
            builds: builds.clone(),
            traces: traces.clone(),
            prefill_updates,
        };
        let graph_inputs = GraphInputAdapterRegistry::with_builtin_adapters();

        let terminal = execute_run_with_all_factories(
            request,
            &UnusedHttpPlacement,
            &graph_inputs,
            &placement,
            &BuiltinAiperfRegistryFactory,
        )
        .unwrap();

        assert!(terminal.success);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(traces.load(Ordering::SeqCst), 1);
        assert!(artifacts.path().join("native-v2.json").is_file());
    }

    #[test]
    fn authored_seamless_lowers_to_the_previous_phase_outbound_handoff() {
        let phases: Vec<PhaseSpec> = serde_json::from_value(json!([{
            "type": "concurrency",
            "name": "warmup",
            "exclude_from_results": true,
            "requests": 1,
            "concurrency": 1
        }, {
            "type": "concurrency",
            "name": "profiling",
            "exclude_from_results": false,
            "requests": 1,
            "concurrency": 1,
            "seamless": true
        }]))
        .unwrap();

        let lowered = phases
            .iter()
            .enumerate()
            .map(|(index, phase)| {
                phase_config(phase, phase_seamless_to_next(&phases, index)).unwrap()
            })
            .collect::<Vec<_>>();

        assert!(lowered[0].seamless);
        assert!(!lowered[1].seamless);
    }

    #[test]
    fn ramp_actuator_roots_follow_phase_actuator_curve_hierarchy() {
        let phase_root = RngRoot::new(Some(73));
        let roots = RampActuatorRngRoots::from_phase_root(phase_root);

        assert_eq!(
            roots.concurrency(),
            phase_root.derive_root(aiperf_rng::namespace::TIMING_RAMP_CONCURRENCY)
        );
        assert_eq!(
            roots.prefill_concurrency(),
            phase_root.derive_root(aiperf_rng::namespace::TIMING_RAMP_PREFILL_CONCURRENCY)
        );
        assert_eq!(
            roots.request_rate(),
            phase_root.derive_root(aiperf_rng::namespace::TIMING_RAMP_REQUEST_RATE)
        );

        let curve_seeds = [
            roots
                .request_rate()
                .derive_seed(aiperf_rng::namespace::TIMING_RAMP_POISSON),
            roots
                .prefill_concurrency()
                .derive_seed(aiperf_rng::namespace::TIMING_RAMP_POISSON),
            roots
                .concurrency()
                .derive_seed(aiperf_rng::namespace::TIMING_RAMP_POISSON),
        ];
        assert!(curve_seeds.iter().all(Option::is_some));
        assert_ne!(curve_seeds[0], curve_seeds[1]);
        assert_ne!(curve_seeds[0], curve_seeds[2]);
        assert_ne!(curve_seeds[1], curve_seeds[2]);
        assert_ne!(
            roots.concurrency(),
            phase_root.derive_root(aiperf_rng::namespace::TIMING_RAMP_POISSON),
            "the phase must not pre-derive the curve-local Poisson namespace"
        );
    }

    #[test]
    fn graph_duration_grace_cancels_placement_and_drains_as_policy() {
        let artifacts = tempfile::tempdir().unwrap();
        let request: RunRequest = serde_json::from_value(json!({
            "protocol_version": 1,
            "run": {
                "benchmark_id": "graph-lifecycle-cancellation",
                "workers": 1,
                "artifact_dir": artifacts.path(),
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "urls": ["http://127.0.0.1:1"],
                    "type": "chat",
                    "streaming": true
                },
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "records": [{
                        "session_id": "root",
                        "turns": [{"messages": [{"role": "user", "content": "hello"}]}]
                    }]
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "duration": 0.02,
                    "grace_period": 0.005,
                    "concurrency": 1
                }],
                "metrics": {},
                "artifacts": {}
            }
        }))
        .unwrap();
        let builds = Arc::new(AtomicUsize::new(0));
        let executions = Arc::new(AtomicUsize::new(0));
        let cancellations = Arc::new(AtomicUsize::new(0));
        let placement = CancellingGraphPlacement {
            builds: builds.clone(),
            executions: executions.clone(),
            cancellations: cancellations.clone(),
            cancelled: Arc::new(AtomicBool::new(false)),
            wake: Arc::new(Notify::new()),
        };

        let terminal = execute_run_with_all_factories(
            request,
            &UnusedHttpPlacement,
            &GraphInputAdapterRegistry::with_builtin_adapters(),
            &placement,
            &BuiltinAiperfRegistryFactory,
        )
        .unwrap();

        assert!(terminal.success, "{:?}", terminal.error);
        assert_eq!(builds.load(Ordering::SeqCst), 1);
        assert_eq!(executions.load(Ordering::SeqCst), 1);
        assert_eq!(cancellations.load(Ordering::SeqCst), 1);
        assert!(artifacts.path().join("native-v2.json").is_file());
    }

    #[test]
    fn graph_uses_shared_phase_lifecycle_for_seamless_ramps_and_prefill_control() {
        let artifacts = tempfile::tempdir().unwrap();
        let request: RunRequest = serde_json::from_value(json!({
            "protocol_version": 1,
            "run": {
                "benchmark_id": "graph-phase-lifecycle",
                "workers": 3,
                "artifact_dir": artifacts.path(),
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "urls": ["http://127.0.0.1:1"],
                    "type": "chat",
                    "streaming": true
                },
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "records": [{
                        "session_id": "root",
                        "turns": [{"messages": [{"role": "user", "content": "hello"}]}]
                    }]
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "warmup",
                    "exclude_from_results": true,
                    "sessions": 1,
                    "concurrency": 2,
                    "concurrency_ramp": {"duration": 0.001, "strategy": "linear"}
                }, {
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "sessions": 1,
                    "concurrency": 2,
                    "seamless": true,
                    "prefill_concurrency": 2,
                    "prefill_ramp": {"duration": 0.001, "strategy": "linear"},
                    "grace_period": 0.01
                }],
                "metrics": {},
                "artifacts": {}
            }
        }))
        .unwrap();
        let builds = Arc::new(AtomicUsize::new(0));
        let traces = Arc::new(AtomicUsize::new(0));
        let prefill_updates = Arc::new(AtomicUsize::new(0));
        let placement = RecordingGraphPlacement {
            builds: builds.clone(),
            traces: traces.clone(),
            prefill_updates: prefill_updates.clone(),
        };
        let graph_inputs = GraphInputAdapterRegistry::with_builtin_adapters();

        let terminal = execute_run_with_all_factories(
            request,
            &UnusedHttpPlacement,
            &graph_inputs,
            &placement,
            &BuiltinAiperfRegistryFactory,
        )
        .unwrap();

        assert!(terminal.success, "{:?}", terminal.error);
        assert_eq!(builds.load(Ordering::SeqCst), 2);
        assert_eq!(traces.load(Ordering::SeqCst), 2);
        assert!(prefill_updates.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn graph_adaptive_prefill_uses_the_shared_controller_and_artifact_contract() {
        let artifacts = tempfile::tempdir().unwrap();
        let request: RunRequest = serde_json::from_value(json!({
            "protocol_version": 1,
            "run": {
                "benchmark_id": "graph-adaptive-prefill",
                "workers": 3,
                "artifact_dir": artifacts.path(),
                "models": {"items": [{"name": "fixture-model"}]},
                "endpoint": {
                    "urls": ["http://127.0.0.1:1"],
                    "type": "chat",
                    "streaming": true
                },
                "dataset": {
                    "type": "file",
                    "format": "dag_jsonl",
                    "records": [{
                        "session_id": "root",
                        "turns": [{"messages": [{"role": "user", "content": "hello"}]}]
                    }]
                },
                "phases": [{
                    "type": "concurrency",
                    "name": "profiling",
                    "exclude_from_results": false,
                    "sessions": 1,
                    "duration": 0.01,
                    "concurrency": 2,
                    "adaptive_scale": {
                        "control_variable": "prefill_concurrency",
                        "minimum": 1.0,
                        "maximum": 2.0,
                        "assessment_period_seconds": 1.0,
                        "sustain_duration_seconds": 1.0,
                        "min_completed_requests": 1,
                        "strategy_type": "ramp_until_fail",
                        "step_policy": "fixed_percent_step",
                        "base_step": 1,
                        "max_step_multiplier": 1,
                        "step_percent": 100.0,
                        "sla_filters": [{
                            "metric_tag": "request_latency",
                            "stat": "p95",
                            "op": "le",
                            "threshold": 1000.0
                        }]
                    }
                }],
                "metrics": {},
                "artifacts": {}
            }
        }))
        .unwrap();
        let builds = Arc::new(AtomicUsize::new(0));
        let traces = Arc::new(AtomicUsize::new(0));
        let prefill_updates = Arc::new(AtomicUsize::new(0));
        let placement = RecordingGraphPlacement {
            builds,
            traces,
            prefill_updates: prefill_updates.clone(),
        };

        let terminal = execute_run_with_all_factories(
            request,
            &UnusedHttpPlacement,
            &GraphInputAdapterRegistry::with_builtin_adapters(),
            &placement,
            &BuiltinAiperfRegistryFactory,
        )
        .unwrap();

        assert!(terminal.success, "{:?}", terminal.error);
        assert_eq!(prefill_updates.load(Ordering::SeqCst), 1);
        assert!(
            artifacts
                .path()
                .join("adaptive_scale_events.jsonl")
                .is_file()
        );
        assert!(
            artifacts
                .path()
                .join("adaptive_scale_summary.json")
                .is_file()
        );
    }

    #[test]
    fn complete_synthetic_shape_maps_to_native_generation_config() {
        let spec = synthetic(json!({
            "entries": 3,
            "random_seed": 41,
            "sampling": "shuffle",
            "prompts": {
                "isl": {"value": 12.0},
                "osl": {"value": 5.0},
                "block_size": 16,
                "batch_size": 2,
                "sequence_distribution": [
                    {
                        "isl": {"value": 12.0},
                        "osl": {"value": 5.0},
                        "probability": 40.0
                    },
                    {
                        "isl": {"mean": 24.0, "stddev": 2.0},
                        "osl": {"mean": 7.0, "stddev": 1.0},
                        "probability": 60.0
                    }
                ]
            },
            "prefix_prompts": {
                "shared_system_length": 4,
                "user_context_length": 3
            },
            "turns": {"value": 2.0},
            "turn_delay_ms": {"value": 7.0},
            "turn_delay_ratio": 0.5,
            "images": {
                "batch_size": 1,
                "width": {"value": 8.0},
                "height": {"value": 6.0},
                "format": "png",
                "source": "noise",
                "source_sampling": "random-with-replacement"
            },
            "audio": {
                "batch_size": 1,
                "length": {"value": 0.02},
                "format": "wav",
                "sample_rates": [16.0],
                "depths": [16],
                "channels": 1
            },
            "video": {
                "batch_size": 1,
                "duration": 0.25,
                "fps": 4,
                "width": 8,
                "height": 6,
                "format": "webm",
                "codec": "libvpx-vp9",
                "synth_type": "grid_clock",
                "audio": {
                    "sample_rate": 44.1,
                    "channels": 1,
                    "codec": "libvorbis",
                    "depth": 16
                }
            },
            "rankings": {
                "passages": {"value": 3.0},
                "passage_tokens": {"value": 9.0},
                "query_tokens": {"value": 4.0}
            }
        }));

        let native = synthetic_config(&spec).unwrap();

        assert_eq!(native.entries, 3);
        assert_eq!(native.prompts.unwrap().batch_size, 2);
        assert_eq!(native.prefixes.shared_system_tokens, Some(4));
        assert_eq!(native.prefixes.user_context_tokens, Some(3));
        assert_eq!(native.images.unwrap().format, SyntheticImageFormat::Png);
        assert_eq!(native.audio.unwrap().sample_rates_hz, vec![16_000]);
        let video = native.video.unwrap();
        assert_eq!((video.width, video.height), (8, 6));
        assert_eq!(video.pattern, SyntheticVideoPattern::GridClock);
        assert_eq!(video.audio.sample_rate_hz, 44_100);
        assert_eq!(native.rankings.unwrap().query_tokens.expected_value(), 4.0);
        let paired = sequence_length_distribution(
            spec.prompts
                .as_ref()
                .unwrap()
                .sequence_distribution
                .as_deref()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(paired.pairs()[1].input_seq_len, 24);
        assert_eq!(paired.pairs()[1].input_seq_len_stddev, 2.0);
        assert_eq!(paired.pairs()[1].output_seq_len_stddev, 1.0);
    }

    #[tokio::test]
    async fn paired_lengths_and_sampling_policy_reach_the_native_dataset() {
        let spec = synthetic(json!({
            "entries": 2,
            "random_seed": 73,
            "sampling": "shuffle",
            "prompts": {
                "batch_size": 1,
                "sequence_distribution": [{
                    "isl": {"value": 6.0},
                    "osl": {"value": 3.0},
                    "probability": 100.0
                }]
            }
        }));
        let registry = AiperfRegistry::builtin().unwrap();
        let dataset = build_dataset(
            &registry,
            &DatasetSpec::Synthetic(Box::new(spec)),
            &models(),
            RngRoot::new(Some(73)),
            &TiktokenTokenizer::builtin(),
            EndpointType::Chat,
        )
        .await
        .unwrap();

        assert_eq!(dataset.metadata().sampling_strategy, "shuffle");
        assert_eq!(dataset.conversations().len(), 2);
        for conversation in dataset.conversations() {
            assert_eq!(conversation.turns[0].max_tokens, Some(3));
            assert_eq!(conversation.turns[0].input_tokens, 6);
        }
    }

    #[tokio::test]
    async fn ranking_endpoint_selects_the_native_rankings_composer() {
        let spec = synthetic(json!({
            "entries": 1,
            "prompts": null,
            "rankings": {
                "passages": {"value": 2.0},
                "passage_tokens": {"value": 5.0},
                "query_tokens": {"value": 4.0}
            }
        }));
        let registry = AiperfRegistry::builtin().unwrap();
        let dataset = build_dataset(
            &registry,
            &DatasetSpec::Synthetic(Box::new(spec)),
            &models(),
            RngRoot::new(Some(3)),
            &TiktokenTokenizer::builtin(),
            EndpointType::NimRankings,
        )
        .await
        .unwrap();

        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.content[0].name, "query");
        assert_eq!(turn.content[1].name, "passages");
        assert_eq!(turn.content[1].handles.len(), 2);
        assert_eq!(turn.input_tokens, 14);
    }

    #[test]
    fn user_files_write_exact_pre_rendered_utf8_after_artifact_creation() {
        let artifact_dir = tempfile::tempdir().unwrap();
        let files = vec![crate::protocol_v2::UserFileSpecV2 {
            path: "nested/run.json".into(),
            format: crate::protocol_v2::UserFileFormatV2::Json,
            content: "{\n  \"count\": 7\n}".into(),
        }];

        materialize_user_files(artifact_dir.path(), &files).unwrap();

        assert_eq!(
            std::fs::read(artifact_dir.path().join("nested/run.json")).unwrap(),
            b"{\n  \"count\": 7\n}"
        );
    }

    #[cfg(unix)]
    #[test]
    fn user_files_reject_symlinked_parent_escape() {
        use std::os::unix::fs::symlink;

        let artifact_dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), artifact_dir.path().join("escape")).unwrap();
        let files = vec![crate::protocol_v2::UserFileSpecV2 {
            path: "escape/owned.txt".into(),
            format: crate::protocol_v2::UserFileFormatV2::Text,
            content: "must-not-write".into(),
        }];

        let error = materialize_user_files(artifact_dir.path(), &files)
            .unwrap_err()
            .to_string();

        assert!(error.contains("symlink"), "{error}");
        assert!(!outside.path().join("owned.txt").exists());
    }
}
