// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native construction and execution of one resolved benchmark run.

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, mpsc};

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
    NativeDatasetConversationSource, TurnToSend,
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
    ComposeConfig, Dataset, DatasetSource, HuggingFaceTokenizer, LoadConfig, ModelId,
    ModelSelector, ModelSelectorFactory, RandomModelSelectorFactory,
    RoundRobinModelSelectorFactory, SourceImageSampling, SyntheticAudioConfig,
    SyntheticAudioFormat, SyntheticDatasetConfig, SyntheticImageConfig, SyntheticImageFormat,
    SyntheticImageSource, SyntheticPrefixConfig, SyntheticPromptConfig, SyntheticRankingsConfig,
    SyntheticVideoAudioConfig, SyntheticVideoConfig, SyntheticVideoFormat, SyntheticVideoPattern,
    TextTokenizer, TiktokenEncoding, TiktokenTokenizer, TraceSynthesisConfig,
};
use aiperf_endpoints::{EndpointConfig, EndpointId, EndpointType, RawEndpointConfig};
use aiperf_extensions::{AiperfRegistry, AiperfRegistryFactory, BuiltinAiperfRegistryFactory};
use aiperf_graph::input::{
    GraphInputAdapterRegistry, GraphInputAdapterResolver, GraphInputBundle, GraphInputConfig,
};
use aiperf_graph::policy::FailFastRunFailurePolicy;
use aiperf_graph::workload::{
    CyclingGraphTraceSource, DurationGraphStop, GraphArrivalPolicy, GraphTraceInstanceSequence,
    GraphTraceSource, GraphWorkload, ImmediateGraphArrival, IntervalGraphArrival,
    SlotPoolTraceAdmission,
};
use aiperf_metrics::{
    CATALOG, ExportContext, InferenceDimensions, MetricTag, MetricsAccumulator, MetricsConfig,
    NativeReport, Phase as MetricsPhase, ReportRunInfo, ReportSummary, RunOutcome, SloThreshold,
};
use aiperf_rng::{
    EmpiricalPoint, PeakEntry, RandomGenerator, RngRoot, SamplingDistribution,
    SequenceLengthDistribution, SequenceLengthPair,
};
use aiperf_timing::{
    BernoulliFixedDelay, CancellationPolicy, ExponentialRamp, GracePeriod, LinearRamp,
    NoopPhaseObserver, PhaseConfig, PhaseKind, PhaseObserver, PoissonRamp, RampDriver,
    RampStrategy, RamperConfig, RoundRobinUrlSelector, SlotPool, StopConfig, UrlSelector,
    make_interval_generator,
};
use aiperf_transport_http::config::ClientConfig;
use aiperf_transport_http::models::{ConnectionReuseStrategy, HttpVersion};
use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use uuid::Uuid;

use crate::gpu_telemetry::GpuTelemetryRun;
use crate::graph_execution::{
    GraphCancellationConfig, LegacyRunnerGraphEndpointRuntimeFactory,
    NativeRunnerGraphPlacementFactory, PreparedRunnerGraphEndpointRuntimeFactory,
    RunnerGraphBackendFactory, RunnerGraphBackendFactoryConfig, RunnerGraphEndpointRuntimeFactory,
    RunnerGraphPlacementFactory,
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
use crate::records::{
    CapturedHttpExchange, CapturedModelOutput, CapturedRecord, write_outputs_json,
    write_raw_records_jsonl, write_records_jsonl,
};
use crate::server_metrics::ServerMetricsRun;
use crate::turn_execution::{
    HttpExecutionBackendConfig, HttpExecutionBackendFactory, NativeHttpExecutionBackendFactory,
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
    pub(crate) accuracy: Option<AccuracySpec>,
    pub(crate) gpu_telemetry: Option<crate::protocol::GpuTelemetrySpec>,
    pub(crate) network_latency: Option<crate::protocol::NetworkLatencySpec>,
    pub(crate) server_metrics: Option<crate::protocol::ServerMetricsSpec>,
    pub(crate) live_streaming: Option<crate::protocol::LiveStreamingSpec>,
    pub(crate) user_files: Vec<crate::protocol_v2::UserFileSpecV2>,
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
    Prepared(NativePreparedEndpointPlan),
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
            Self::Prepared(plan) => Ok(&plan.default_profile()?.config.urls),
        }
    }
}

/// Deterministically ordered protocol-v2 endpoint profiles retained until
/// each execution worker prepares its own dense binding table.
#[derive(Clone)]
pub(crate) struct NativePreparedEndpointPlan {
    pub(crate) default_profile_id: String,
    pub(crate) profiles: Vec<NativePreparedEndpointProfile>,
}

impl NativePreparedEndpointPlan {
    pub(crate) fn default_profile(&self) -> Result<&NativePreparedEndpointProfile> {
        self.profile(&self.default_profile_id)
    }

    pub(crate) fn profile(&self, profile_id: &str) -> Result<&NativePreparedEndpointProfile> {
        self.profiles
            .iter()
            .find(|profile| profile.profile_id == profile_id)
            .ok_or_else(|| anyhow!("endpoint profile {profile_id:?} was not prepared"))
    }
}

/// One normalized endpoint profile without a legacy closed-enum identity.
#[derive(Clone)]
pub(crate) struct NativePreparedEndpointProfile {
    pub(crate) profile_id: String,
    pub(crate) endpoint_id: EndpointId,
    pub(crate) config: RawEndpointConfig,
    pub(crate) connection_reuse: ConnectionReuseStrategy,
    pub(crate) http2: bool,
    pub(crate) session_header: Option<String>,
}

/// Protocol-neutral dataset selection.
pub(crate) enum NativeDatasetPlan {
    /// Ordinary linear dataset composition.
    Linear(DatasetSpec),
    /// Direct authored graph input consumed exactly once by its adapter.
    Graph(Box<NativeGraphDatasetPlan>),
}

/// Direct graph adapter selection and already typed load policy.
pub(crate) struct NativeGraphDatasetPlan {
    pub(crate) input: NativeGraphInputPlan,
    pub(crate) random_seed: Option<u64>,
    pub(crate) default_output_tokens: usize,
}

/// Direct Graph-IR input state.
///
/// V1 retains an authored source because its historical protocol performs
/// acquisition during execution. V2 stores the adapter's canonical bundle at
/// preparation time so topology validation and source loading happen once,
/// before run artifacts or HTTP resources exist.
pub(crate) enum NativeGraphInputPlan {
    /// Source still awaiting its single direct adapter load.
    Authored {
        adapter_name: String,
        input: Box<GraphInputConfig>,
    },
    /// Canonical Graph-IR bundle loaded and validated during preparation.
    Prepared(Arc<GraphInputBundle>),
}

impl TryFrom<RunRequest> for NativeRunPlan {
    type Error = anyhow::Error;

    fn try_from(request: RunRequest) -> Result<Self> {
        let run = request.run;
        let dataset = lower_v1_dataset(run.dataset)?;
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
                accuracy: run.accuracy,
                gpu_telemetry: run.gpu_telemetry,
                network_latency: run.network_latency,
                server_metrics: run.server_metrics,
                live_streaming: run.live_streaming,
                user_files: Vec::new(),
            },
        })
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
    Ok(NativeDatasetPlan::Graph(Box::new(NativeGraphDatasetPlan {
        input: NativeGraphInputPlan::Authored {
            adapter_name,
            input: Box::new(input),
        },
        random_seed,
        default_output_tokens,
    })))
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
    validate_plan(&plan)?;
    let artifact_dir = plan.run.artifact_dir.clone();
    std::fs::create_dir_all(&artifact_dir)
        .with_context(|| format!("creating run artifact directory {}", artifact_dir.display()))?;
    materialize_user_files(&artifact_dir, &plan.run.user_files)?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating native single-run Tokio runtime")?;
    let local = tokio::task::LocalSet::new();
    let native = local.block_on(
        &runtime,
        execute_native(
            plan,
            backend_factory,
            graph_inputs,
            graph_placement,
            registry,
        ),
    )?;
    let report_path = artifact_dir.join("native-v2.json");
    write_native_report_json(&native, &report_path)?;
    Ok(report_path)
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
    if request.run.gpu_telemetry.is_some() {
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
    if request.run.network_latency.is_some() {
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
    if let Some(spec) = &request.run.server_metrics {
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
    if let Some(spec) = &request.run.live_streaming {
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
    Ok(())
}

struct AccuracyWorkerRun<'a> {
    evaluator: &'a mut dyn AccuracyEvaluator,
    spec: AccuracySpec,
}

struct PreparedAccuracy<'a> {
    evaluator: &'a mut dyn AccuracyEvaluator,
    loaded: EvaluatorLoadResult,
    dataset: AccuracyDataset,
    processor: Rc<AccuracyRecordProcessor>,
}

async fn execute_native(
    request: NativeRunPlan,
    backend_factory: &dyn HttpExecutionBackendFactory,
    graph_inputs: &dyn GraphInputAdapterResolver,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    if matches!(request.run.dataset, NativeDatasetPlan::Graph(_)) {
        validate_graph_request(&request)?;
        return execute_graph_native(request, graph_inputs, graph_placement, registry).await;
    }
    let mut live_streaming = if request.run.live_streaming.is_some() {
        match PythonLiveStreamingRun::spawn(&request.run, metrics_config(&request.run.metrics)?)
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
    let live_sink = live_streaming.as_ref().map(PythonLiveStreamingRun::sink);
    let result = execute_native_with_accuracy(request, live_sink, backend_factory, registry).await;
    if let Some(worker) = live_streaming.take()
        && let Err(error) = worker.shutdown().await
    {
        eprintln!("live telemetry extension failed to shut down cleanly: {error:#}");
    }
    result
}

fn validate_graph_request(request: &NativeRunPlan) -> Result<()> {
    ensure!(
        request.run.accuracy.is_none(),
        "authored Graph-IR datasets cannot be combined with an accuracy evaluator"
    );
    ensure!(
        request.run.gpu_telemetry.is_none()
            && request.run.network_latency.is_none()
            && request.run.server_metrics.is_none()
            && request.run.live_streaming.is_none(),
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
    for (phase_index, phase) in request.run.phases.iter().enumerate() {
        ensure!(
            matches!(
                phase,
                PhaseSpec::Concurrency { .. }
                    | PhaseSpec::Poisson { .. }
                    | PhaseSpec::Gamma { .. }
                    | PhaseSpec::Constant { .. }
            ),
            "graph phase {phase_index} must use concurrency, poisson, gamma, or constant scheduling"
        );
        let common = phase.common();
        ensure!(
            common.concurrency_ramp.is_none()
                && common.prefill_ramp.is_none()
                && common.rate_ramp.is_none(),
            "graph phase {phase_index} does not yet support actuator ramps"
        );
        ensure!(
            common.adaptive_scale.is_none(),
            "graph phase {phase_index} does not yet support adaptive scale"
        );
        ensure!(
            !common.seamless,
            "graph phase {phase_index} does not yet support seamless handoff"
        );
        ensure!(
            common.grace_period.is_none(),
            "graph phase {phase_index} drains admitted traces and does not accept a separate grace_period"
        );
        ensure!(
            common.requests != Some(0) && common.sessions != Some(0),
            "graph phase {phase_index} request/session bounds must be positive when configured"
        );
        ensure!(
            phase.concurrency() != Some(0),
            "graph phase {phase_index} concurrency must be positive when configured"
        );
        ensure!(
            common.prefill_concurrency != Some(0),
            "graph phase {phase_index} prefill_concurrency must be positive when configured"
        );
        ensure!(
            request.run.workers == 1 || common.prefill_concurrency.is_none(),
            "graph prefill_concurrency is worker-local today; configure one worker or omit it until a distributed admission implementation is selected"
        );
        if common.duration.is_none() && common.requests.is_none() && common.sessions.is_none() {
            // The direct source performs exactly one authored pass in this case.
            continue;
        }
        if let Some(duration) = common.duration {
            seconds_to_ns(duration)
                .with_context(|| format!("validating graph phase {phase_index} duration"))?;
        }
    }
    Ok(())
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

struct PreparedGraphPhase {
    workload: GraphWorkload,
    records: mpsc::Receiver<Vec<CapturedRecord>>,
}

async fn execute_graph_native(
    request: NativeRunPlan,
    graph_inputs: &dyn GraphInputAdapterResolver,
    graph_placement: &dyn RunnerGraphPlacementFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    let graph = match &request.run.dataset {
        NativeDatasetPlan::Graph(graph) => graph,
        NativeDatasetPlan::Linear(_) => bail!("graph execution received a linear dataset plan"),
    };
    let graph_random_seed = graph.random_seed;
    let graph_default_output_tokens = graph.default_output_tokens;
    let metrics_config = metrics_config(&request.run.metrics)?;
    let tokenizer = load_tokenizer(Some(&request.run.tokenizer.name))?;
    let input_token_counter: Arc<dyn InputTokenCounter> = Arc::new(EndpointInputTokenCounter::new(
        tokenizer.clone(),
        request.run.tokenizer.apply_chat_template,
    ));
    let input = match &graph.input {
        NativeGraphInputPlan::Prepared(input) => input.clone(),
        NativeGraphInputPlan::Authored {
            adapter_name,
            input,
        } => {
            let adapter = graph_inputs.find(adapter_name).ok_or_else(|| {
                anyhow!("no Graph-IR input adapter is registered for {adapter_name:?}")
            })?;
            let input = GraphInputConfig {
                load: input.load.clone(),
                root_limit: input.root_limit,
            };
            Arc::new(
                adapter
                    .load(input, tokenizer.as_ref())
                    .await
                    .context("loading direct authored Graph-IR input")?,
            )
        }
    };
    ensure!(
        !input.plans.is_empty(),
        "authored Graph-IR input contains no root traces after root limiting"
    );
    let primary_model = request.run.models.items[0].name.clone();
    let default_output_tokens = graph_default_output_tokens;
    let endpoints_configured = match &request.run.endpoint {
        NativeEndpointPlan::Legacy(spec) => spec.urls.clone(),
        NativeEndpointPlan::Prepared(plan) => plan
            .profiles
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
            NativeEndpointPlan::Prepared(plan) => {
                Arc::new(PreparedRunnerGraphEndpointRuntimeFactory::new(
                    registry.endpoints().clone(),
                    plan.clone(),
                    input_token_counter.clone(),
                ))
            }
        };
    let real_clock_anchor = RealClockAnchor::now();
    let clock: Rc<dyn Clock> = RealClock::from_anchor(real_clock_anchor);
    let start_ns = clock.now_ns();
    let rng_root = RngRoot::new(graph_random_seed.or(request.run.random_seed));
    let trace_instances = GraphTraceInstanceSequence::default();

    // Construct every phase's placement workers before the first root can be
    // admitted. Any parser, policy, transport, or worker setup error therefore
    // fails the entire run before HTTP traffic.
    let mut phases = Vec::with_capacity(request.run.phases.len());
    for (phase_index, phase) in request.run.phases.iter().enumerate() {
        phases.push(prepare_graph_phase(
            phase_index,
            phase,
            &request,
            input.as_ref(),
            endpoint_runtime_factory.clone(),
            metrics_config.clone(),
            real_clock_anchor,
            clock.clone(),
            start_ns,
            &primary_model,
            default_output_tokens,
            rng_root,
            trace_instances.clone(),
            graph_placement,
        )?);
    }

    let mut captured = Vec::new();
    for prepared in phases {
        let report = prepared.workload.execute().await?;
        let failure = report.traces.iter().find_map(|trace| {
            trace
                .result
                .as_ref()
                .err()
                .map(|error| format!("graph trace {:?} failed: {error}", trace.trace_id))
        });
        drop(prepared.workload);
        captured.extend(prepared.records.into_iter().flatten());
        if report.failed > 0 {
            bail!(
                "graph phase aborted after {} failed trace(s): {}",
                report.failed,
                failure.unwrap_or_else(|| "unknown trace failure".into())
            );
        }
    }
    captured.sort_by(|left, right| {
        left.ingest
            .start_ns
            .cmp(&right.ingest.start_ns)
            .then_with(|| left.uuid.cmp(&right.uuid))
    });

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
        was_cancelled: false,
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

#[allow(clippy::too_many_arguments)]
fn prepare_graph_phase(
    phase_index: usize,
    phase: &PhaseSpec,
    request: &NativeRunPlan,
    input: &GraphInputBundle,
    endpoint_runtime_factory: Arc<dyn RunnerGraphEndpointRuntimeFactory>,
    metrics: MetricsConfig,
    real_clock_anchor: RealClockAnchor,
    clock: Rc<dyn Clock>,
    run_origin_ns: i64,
    primary_model: &str,
    default_output_tokens: usize,
    rng_root: RngRoot,
    trace_instances: GraphTraceInstanceSequence,
    graph_placement: &dyn RunnerGraphPlacementFactory,
) -> Result<PreparedGraphPhase> {
    let common = phase.common();
    let one_pass =
        common.sessions.is_none() && common.requests.is_none() && common.duration.is_none();
    let session_limit = if one_pass {
        Some(u64::try_from(input.plans.len()).context("graph root count exceeds u64")?)
    } else {
        common.sessions
    };
    let source: Rc<dyn GraphTraceSource> =
        Rc::new(CyclingGraphTraceSource::with_budgets_and_sequence(
            input.plans.clone(),
            session_limit,
            common.requests,
            trace_instances,
        )?);
    let arrival: Rc<dyn GraphArrivalPolicy> = match phase {
        PhaseSpec::Concurrency { .. } => Rc::new(ImmediateGraphArrival),
        PhaseSpec::Poisson { .. } | PhaseSpec::Gamma { .. } | PhaseSpec::Constant { .. } => {
            let (pattern, rate, smoothness) = phase
                .request_arrival()
                .expect("validated graph rate phase has an arrival policy");
            let seed = rng_root
                .derive_seed(&format!("runner.graph.phase.{phase_index}.arrival"))
                .unwrap_or(phase_index as u64);
            Rc::new(IntervalGraphArrival::new(Rc::new(RefCell::new(
                make_interval_generator(pattern, rate, smoothness, seed),
            ))))
        }
        PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } => {
            unreachable!("unsupported graph phase rejected before input acquisition")
        }
    };
    let (records_tx, records_rx) = mpsc::channel();
    let cancellation = common
        .cancellation
        .map(|cancellation| GraphCancellationConfig {
            rate: cancellation.rate,
            delay_seconds: cancellation.delay,
            seed: rng_root
                .derive_seed(&format!("runner.graph.phase.{phase_index}.cancellation"))
                .unwrap_or(phase_index as u64),
            phase: if common.name == "warmup" {
                aiperf_timing::Phase::Warmup
            } else {
                aiperf_timing::Phase::Profiling
            },
        });
    let worker_factory = Arc::new(RunnerGraphBackendFactory::new(
        RunnerGraphBackendFactoryConfig {
            real_clock_anchor,
            run_origin_ns,
            model: primary_model.to_string(),
            default_max_tokens: default_output_tokens,
            endpoint_runtime_factory,
            segments: input.segments.clone(),
            metrics,
            phase: metrics_phase(phase)?,
            prefill_concurrency: common.prefill_concurrency,
            cancellation,
            raw_enabled: request.run.artifacts.raw_path.is_some(),
            captured: records_tx,
        },
    ));
    let placement = graph_placement.build(request.run.workers, worker_factory)?;
    let mut workload = GraphWorkload::new(clock, source, placement)
        .with_arrival(arrival)
        .with_run_failure(Rc::new(FailFastRunFailurePolicy::default()));
    if let Some(concurrency) = phase.concurrency() {
        workload = workload.with_admission(Rc::new(SlotPoolTraceAdmission::new(Rc::new(
            SlotPool::new(concurrency),
        ))));
    }
    if let Some(duration) = common.duration {
        workload =
            workload.with_stop_policy(Rc::new(DurationGraphStop::new(seconds_to_ns(duration)?)?));
    }
    Ok(PreparedGraphPhase {
        workload,
        records: records_rx,
    })
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

async fn execute_native_with_accuracy(
    request: NativeRunPlan,
    live_sink: Option<Rc<dyn LiveResultsSink>>,
    backend_factory: &dyn HttpExecutionBackendFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    let Some(spec) = request.run.accuracy.clone() else {
        return execute_native_inner(request, None, live_sink, backend_factory, registry).await;
    };
    ensure!(
        spec.python_executable.is_absolute(),
        "accuracy python_executable must be an absolute path"
    );
    ensure!(
        !spec.worker_module.trim().is_empty(),
        "accuracy worker_module cannot be empty"
    );
    let worker = WorkerProcessConfig::new(spec.python_executable.as_os_str())
        .arg("-u")
        .arg("-m")
        .arg(&spec.worker_module);
    let mut evaluator = PythonEvaluator::spawn(worker)
        .await
        .context("starting canonical Python accuracy evaluator")?;
    let result = execute_native_inner(
        request,
        Some(AccuracyWorkerRun {
            evaluator: &mut evaluator,
            spec,
        }),
        live_sink,
        backend_factory,
        registry,
    )
    .await;
    let shutdown = evaluator.shutdown().await;
    match (result, shutdown) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(anyhow!(error).context("shutting down accuracy evaluator")),
        (Err(error), Err(shutdown)) => Err(error.context(format!(
            "accuracy evaluator also failed during shutdown: {shutdown}"
        ))),
    }
}

async fn execute_native_inner(
    request: NativeRunPlan,
    accuracy: Option<AccuracyWorkerRun<'_>>,
    live_sink: Option<Rc<dyn LiveResultsSink>>,
    backend_factory: &dyn HttpExecutionBackendFactory,
    registry: &AiperfRegistry,
) -> Result<NativeReport> {
    let endpoint_spec = request.run.endpoint.legacy()?;
    let rng_root = RngRoot::new(request.run.random_seed);
    let dataset_spec = match &request.run.dataset {
        NativeDatasetPlan::Linear(dataset) => dataset,
        NativeDatasetPlan::Graph(_) => {
            bail!("scheduled execution received a direct graph dataset plan")
        }
    };
    let dataset_rng_root = dataset_rng_root(dataset_spec, rng_root);
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
    let tokenizer = load_tokenizer(Some(&request.run.tokenizer.name))?;
    let input_token_counter: Arc<dyn InputTokenCounter> = Arc::new(EndpointInputTokenCounter::new(
        tokenizer.clone(),
        request.run.tokenizer.apply_chat_template,
    ));
    let mut prepared_accuracy = if let Some(accuracy) = accuracy {
        let evaluator_config = EvaluatorLoadConfig {
            tasks: accuracy.spec.tasks.clone(),
            n_shots: accuracy.spec.n_shots,
            enable_cot: accuracy.spec.enable_cot,
            system_prompt: accuracy.spec.system_prompt.clone(),
            max_problems: None,
            max_tokens: None,
            seed: request.run.random_seed.unwrap_or(0),
        };
        let (loaded, problems) = load_evaluator_problems_with_grader(
            accuracy.evaluator,
            &accuracy.spec.benchmark,
            &evaluator_config,
            accuracy.spec.grader.as_deref(),
        )
        .await?;
        let dataset =
            AccuracyDataset::from_evaluator_problems(&primary_model, problems, tokenizer.as_ref())?;
        let processor = Rc::new(dataset.record_processor());
        Some(PreparedAccuracy {
            evaluator: accuracy.evaluator,
            loaded,
            dataset,
            processor,
        })
    } else {
        None
    };
    let dataset = if let Some(accuracy) = &prepared_accuracy {
        accuracy.dataset.dataset().as_ref().clone()
    } else {
        build_dataset(
            registry,
            dataset_spec,
            &request.run.models,
            dataset_rng_root,
            tokenizer.as_ref(),
            endpoint_spec.endpoint_type,
        )
        .await?
    };
    let endpoint = endpoint_config(endpoint_spec)?;
    let default_output_tokens = if prepared_accuracy.is_some() {
        dataset_default_output_tokens(&dataset)?
    } else {
        default_output_tokens(dataset_spec)?
    };
    if prepared_accuracy.is_some() {
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

    let real_clock_anchor = RealClockAnchor::now();
    let clock: Rc<dyn Clock> = RealClock::from_anchor(real_clock_anchor);
    let gpu_telemetry = if let Some(spec) = request.run.gpu_telemetry.as_ref() {
        Some(GpuTelemetryRun::new(spec, clock.clone()).await?)
    } else {
        None
    };
    let network_latency = request
        .run
        .network_latency
        .as_ref()
        .map(|spec| {
            NetworkLatencyRun::new(
                &request.run.benchmark_id,
                spec,
                &endpoint_spec.urls,
                clock.clone(),
            )
        })
        .transpose()?;
    let server_metrics = request
        .run
        .server_metrics
        .as_ref()
        .map(|spec| ServerMetricsRun::new(spec, clock.clone()))
        .transpose()?;
    let gpu_records_path = request
        .run
        .gpu_telemetry
        .as_ref()
        .map(|spec| {
            artifact_path(
                &request.run.artifact_dir,
                &spec.records_path,
                "gpu_telemetry.records_path",
            )
        })
        .transpose()?;
    let network_latency_records_path = request
        .run
        .network_latency
        .as_ref()
        .and_then(|spec| spec.probe.as_ref())
        .map(|probe| {
            artifact_path(
                &request.run.artifact_dir,
                &probe.records_path,
                "network_latency.probe.records_path",
            )
        })
        .transpose()?;
    let server_metrics_jsonl_path = request
        .run
        .server_metrics
        .as_ref()
        .and_then(|spec| spec.jsonl_path.as_ref())
        .map(|path| artifact_path(&request.run.artifact_dir, path, "server_metrics.jsonl_path"))
        .transpose()?;
    let server_metrics_parquet_wire_path = request
        .run
        .server_metrics
        .as_ref()
        .and_then(|spec| spec.parquet_wire_path.as_ref())
        .map(|path| {
            artifact_path(
                &request.run.artifact_dir,
                path,
                "server_metrics.parquet_wire_path",
            )
        })
        .transpose()?;
    let request_timeout_ns = seconds_to_ns(endpoint_spec.timeout_seconds)?;
    let execution_backend = backend_factory.build(HttpExecutionBackendConfig {
        workers: request.run.workers,
        coordinator_clock: clock.clone(),
        real_clock_anchor,
        base_urls: endpoint_spec.urls.clone(),
        model: primary_model.clone(),
        transport: TransportSinkConfig {
            client: ClientConfig {
                http_version: if endpoint_spec.http2 {
                    HttpVersion::Http2PriorKnowledge
                } else {
                    HttpVersion::Auto
                },
                total_timeout_ns: (request_timeout_ns > 0).then_some(request_timeout_ns),
                ..ClientConfig::default()
            },
            connection_reuse: endpoint_spec.connection_reuse,
            session_header: endpoint_spec.session_header.clone(),
        },
        prepared_endpoints: None,
    })?;
    let start_ns = clock.now_ns();
    execution_backend.set_run_origin(start_ns)?;
    let capture = Rc::new(RunCapture::new(
        clock.clone(),
        start_ns,
        metrics_config.clone(),
        request.run.artifacts.raw_path.is_some(),
    ));
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(ConfiguredDispatcher {
        execution_backend: execution_backend.clone(),
        model: primary_model.clone(),
        capture: capture.clone(),
    });

    let shared_resources = native_scheduled_resources(&request.run.phases);

    let mut plans = Vec::with_capacity(request.run.phases.len());
    for (phase_index, phase) in request.run.phases.iter().enumerate() {
        let mut plan = build_native_scheduled_phase_plan(
            phase_index,
            phase,
            &dataset,
            &primary_model,
            default_output_tokens,
            dataset_rng_root,
            rng_root,
            &endpoint,
            registry,
            tokenizer.clone(),
            input_token_counter.clone(),
            clock.clone(),
            start_ns,
            &request.run.benchmark_id,
            &request.run.artifact_dir,
            &endpoint_spec.urls,
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
            && let Some(accuracy) = &prepared_accuracy
        {
            let processor: Rc<dyn TurnRecordProcessor> = accuracy.processor.clone();
            record_processors.push(processor);
        }
        plan = plan.with_record_processors(record_processors);
        let mut sidecars = Vec::new();
        if let Some(server_metrics) = &server_metrics {
            sidecars.push(server_metrics.sidecar(metrics_phase(phase)?));
        }
        if phase.common().name == "profiling" {
            if let Some(gpu_telemetry) = &gpu_telemetry {
                sidecars.push(gpu_telemetry.sidecar());
            }
            if let Some(network_latency) = &network_latency
                && let Some(sidecar) = network_latency.sidecar()
            {
                sidecars.push(sidecar);
            }
        }
        if !sidecars.is_empty() {
            plan = plan.with_sidecars(sidecars);
        }
        plans.push(plan);
    }

    let observer: Rc<dyn PhaseObserver> = if let Some(sink) = live_sink {
        live_phase_observer(sink, clock.clone())
    } else {
        Rc::new(NoopPhaseObserver)
    };
    let phased = run_scheduled_phases(plans, clock, dispatcher, observer).await?;
    execution_backend.shutdown()?;
    phased
        .reports
        .iter()
        .find(|report| report.kind == PhaseKind::Profiling)
        .ok_or_else(|| anyhow!("phase runtime completed without a profiling report"))?;
    let issued_times = phased
        .reports
        .iter()
        .flat_map(|report| report.report.turns.iter())
        .map(|turn| (turn.uuid, turn.issued_offset_ns))
        .collect::<HashMap<_, _>>();
    let captured = capture.finish(&issued_times)?;
    let mut accumulator = MetricsAccumulator::with_config(metrics_config.clone());
    for record in &captured {
        accumulator.process_record(&record.ingest);
    }
    if let Some(network_latency) = &network_latency {
        let mean_rtt_ns = network_latency.mean_rtt_ns();
        if request
            .run
            .network_latency
            .as_ref()
            .is_some_and(|spec| spec.probe.is_some())
            && mean_rtt_ns.is_none()
        {
            eprintln!(
                "network latency calibration collected no successful probes; adjusted metrics are omitted"
            );
        }
        accumulator.set_network_rtt_ns(mean_rtt_ns);
    }
    let mut profiling_metrics =
        accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
    if let Some(gpu_telemetry) = &gpu_telemetry {
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
    let profiling_server_summary = server_metrics.as_ref().map(|server_metrics| {
        server_metrics.summarize(MetricsPhase::Profiling, metrics_config.slice_duration_ns)
    });
    let warmup = phased
        .reports
        .iter()
        .any(|report| report.kind == PhaseKind::Warmup)
        .then(|| accumulator.export_results(&ExportContext::phase(MetricsPhase::Warmup)));
    let warmup_server_summary =
        server_metrics
            .as_ref()
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
    if let (Some(gpu_telemetry), Some(gpu_records_path)) = (&gpu_telemetry, &gpu_records_path) {
        gpu_telemetry.write_records_jsonl(gpu_records_path)?;
    }
    if let (Some(network_latency), Some(records_path)) =
        (&network_latency, &network_latency_records_path)
    {
        network_latency.write_records_jsonl(records_path)?;
    }
    if let (Some(server_metrics), Some(path)) = (&server_metrics, &server_metrics_jsonl_path) {
        server_metrics.write_slim_jsonl(path)?;
    }
    if let (Some(server_metrics), Some(path)) = (&server_metrics, &server_metrics_parquet_wire_path)
    {
        server_metrics.write_parquet_wire_jsonl(path)?;
    }
    let server_metrics_report = server_metrics.as_ref().and_then(|server_metrics| {
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
            endpoints_configured: endpoint_spec.urls.clone(),
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
    if let Some(accuracy) = prepared_accuracy.take() {
        let evaluation = grade_accuracy_responses(
            accuracy.processor.as_ref(),
            accuracy.evaluator,
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

/// Lower one authored phase into the shared scheduled runtime above the
/// injected `{transport, clock}` seams.
///
/// Dataset filtering/materialization, arrival policy, session/prefill
/// admission, fixed/user-centric scheduling, ramps, cancellation, adaptive
/// control, and phase lifecycle are deliberately composed here once. Backend
/// adapters may decorate the returned plan with observers or sidecars, but do
/// not reproduce its scheduler logic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_native_scheduled_phase_plan(
    phase_index: usize,
    phase: &PhaseSpec,
    dataset: &Dataset,
    primary_model: &str,
    default_output_tokens: usize,
    dataset_rng_root: RngRoot,
    rng_root: RngRoot,
    endpoint: &EndpointConfig,
    registry: &AiperfRegistry,
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
    let source = native_conversation_source(
        phase_dataset,
        primary_model.to_owned(),
        default_output_tokens,
        phase_rng,
        endpoint.clone(),
        registry,
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
    let mut plan = ScheduledPhasePlan::new(phase_config(phase)?, workload, policies)
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
            build_file_dataset(registry, spec, models, rng_root, tokenizer).await
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

fn metrics_phase(spec: &PhaseSpec) -> Result<MetricsPhase> {
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

fn phase_config(spec: &PhaseSpec) -> Result<PhaseConfig> {
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
        .with_seamless(common.seamless)
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

fn ramp_controller(
    spec: &PhaseSpec,
    clock: Rc<dyn Clock>,
    intervals: Rc<RefCell<Box<dyn aiperf_timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    rng_root: RngRoot,
) -> Result<Rc<dyn ScheduledPhaseController>> {
    let common = spec.common();
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
        let strategy = ramp_strategy(ramp, 1.0, target as f64, false, rng_root)?;
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
        let strategy = ramp_strategy(ramp, 1.0, target as f64, false, rng_root)?;
        drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
            slots.set_limit(value.round() as usize)
        }));
    }
    if let Some(ramp) = &common.rate_ramp {
        let target = target_rate.ok_or_else(|| anyhow!("rate_ramp requires a rate phase"))?;
        let duration_ns = seconds_to_u64_ns(ramp.duration)?;
        let start = target * RATE_RAMP_UPDATE_INTERVAL_NS as f64 / duration_ns as f64;
        let strategy = ramp_strategy(ramp, start, target, true, rng_root)?;
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
        AdaptiveControlVariableSpec::Concurrency => {
            ensure!(
                session_slots.is_some(),
                "adaptive concurrency requires session admission"
            );
            ensure!(
                phase.common().concurrency_ramp.is_none(),
                "adaptive concurrency cannot be combined with concurrency_ramp"
            );
            AdaptiveControlVariable::Concurrency
        }
        AdaptiveControlVariableSpec::PrefillConcurrency => {
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
                spec.maximum <= session_target as f64,
                "adaptive prefill_concurrency maximum must be <= concurrency"
            );
            AdaptiveControlVariable::PrefillConcurrency
        }
        AdaptiveControlVariableSpec::RequestRate => {
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
            AdaptiveControlVariable::RequestRate
        }
        AdaptiveControlVariableSpec::Users => {
            ensure!(
                matches!(phase, PhaseSpec::UserCentric { .. }) && user_target.is_some(),
                "adaptive users requires a user_centric phase"
            );
            AdaptiveControlVariable::Users
        }
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
    let config = AdaptiveRunConfig {
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
    };
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

fn integer_adaptive_bound(value: f64, label: &str) -> Result<usize> {
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

struct AdaptiveScheduledPhaseController {
    scale: Rc<AdaptiveScale>,
    delegate: Rc<dyn ScheduledPhaseController>,
    assessment: RefCell<Option<tokio::task::JoinHandle<()>>>,
}

impl AdaptiveScheduledPhaseController {
    fn new(scale: Rc<AdaptiveScale>, delegate: Rc<dyn ScheduledPhaseController>) -> Self {
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

fn ramp_strategy(
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

fn seconds_to_u64_ns(value: f64) -> Result<u64> {
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aiperf_graph::errors::TraceError;
    use aiperf_graph::placement::{GraphPlacementError, GraphTraceExecutionBackendFactory};
    use serde_json::json;

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

    struct RecordingGraphPlacement {
        builds: Arc<AtomicUsize>,
        traces: Arc<AtomicUsize>,
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
            }))
        }
    }

    struct RecordingGraphBackend {
        traces: Arc<AtomicUsize>,
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
                    "concurrency": 1
                }],
                "metrics": {},
                "artifacts": {}
            }
        }))
        .unwrap();
        let builds = Arc::new(AtomicUsize::new(0));
        let traces = Arc::new(AtomicUsize::new(0));
        let placement = RecordingGraphPlacement {
            builds: builds.clone(),
            traces: traces.clone(),
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
