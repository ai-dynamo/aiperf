// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native construction and execution of one resolved benchmark run.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use crate::accuracy::{
    AccuracyDataset, AccuracyRecordProcessor, CapturedResponse, ProblemAssociation,
    accuracy_report_errors, grade_accuracy_captures, load_evaluator_problems_with_grader,
};
use crate::accuracy_core::{
    AccuracyEvaluator, EvaluatorLoadConfig, EvaluatorLoadResult, PythonEvaluator,
    WorkerProcessConfig,
};
use crate::adaptive::{
    AdaptiveControlVariable, AdaptiveRunConfig, AdaptiveStepConfig, build_adaptive_with_origins,
    positive_seconds_to_ns,
};
use crate::adaptive_core::{
    AdaptiveScale, CorrelationContext, SharedWindowSampler, SlaFilter, UserTarget,
};
use crate::ancillary::RATE_RAMP_UPDATE_INTERVAL_NS;
use crate::cellular::{CellPartition, IssuanceAuthority, ModuloCellPartition};
use crate::clock::{Clock, RealClock, RealClockAnchor};
use crate::content_server::{
    ContentRequestRecord, ContentServerConfig, ContentServerFactory, ContentServerRuntime,
    MediaFetchAggregator, MediaMetricsSummary, MediaRecordWriter, NativeContentServerFactory,
};
use crate::dataset::{
    ComposeConfig, Dataset, DatasetSource, HuggingFaceTokenizer, LoadConfig, ModelId,
    ModelSelector, ModelSelectorFactory, RandomModelSelectorFactory,
    RoundRobinModelSelectorFactory, SourceImageSampling, SyntheticAudioConfig,
    SyntheticAudioFormat, SyntheticDatasetConfig, SyntheticImageConfig, SyntheticImageFormat,
    SyntheticImageSource, SyntheticMediaGeneratorFactory, SyntheticPrefixConfig,
    SyntheticPromptConfig, SyntheticRankingsConfig, SyntheticVideoAudioConfig,
    SyntheticVideoConfig, SyntheticVideoFormat, SyntheticVideoPattern, TextTokenizer,
    TiktokenEncoding, TiktokenTokenizer, TracePromptStoragePolicy, TraceSynthesisConfig,
};
use crate::endpoints::{EndpointKey, EndpointRegistry, PreparedEndpointTable};
use crate::export::otel::OtelRecordAccumulator;
use crate::extensions::AIPerfRegistry;
use crate::failure::OnFailure;
use crate::fixed_schedule::{
    DatasetFixedScheduleSource, FixedScheduleConfig, FixedScheduleWorkload,
};
use crate::graph::input::GraphInputBundle;
use crate::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use crate::metrics_core::{
    AccumulatorSummary, CATALOG, ExportContext, InferenceDimensions, MetricTag, MetricsAccumulator,
    MetricsConfig, NativeReport, Phase as MetricsPhase, RecordIngest, ReportRunInfo,
    ReportServerMetricsMetadata, ReportSummary, RunOutcome, SloThreshold,
};
use crate::multiturn::{
    AuthoredInputTokenCounter, ConversationSource, EndpointInputTokenCounter, InputTokenCounter,
    IssuedCredit, NativeDatasetConversationSource, PreparedEndpointReference,
    PreparedEndpointTableResolver, PreparedTurnEndpointResolver, TurnToSend,
};
use crate::phase_runtime::{
    RampScheduledPhaseController, ScheduledPhaseController, ScheduledPhasePlan,
    ScheduledPhaseResources, ScheduledPhaseSidecar, ScheduledRuntimeExtension,
    ScheduledRuntimeExtensionParts, SlotPoolPhaseResources, run_scheduled_phases,
};
use crate::request_rate::RequestRateWorkload;
use crate::rng::{
    EmpiricalPoint, PeakEntry, RandomGenerator, RngRoot, SamplingDistribution,
    SequenceLengthDistribution, SequenceLengthPair, namespace,
};
use crate::scheduled::{
    IssuanceGate, ScheduledAncillaryPolicies, TurnDispatchOutcome, TurnDispatcher,
    TurnRecordProcessor, Workload,
};
use crate::timing::{
    BernoulliFixedDelay, CancellationPolicy, ExponentialRamp, GracePeriod, LinearRamp,
    NoopPhaseObserver, PhaseConfig, PhaseKind, PhaseObserver, PoissonRamp, RampDriver,
    RampStrategy, RamperConfig, RoundRobinUrlSelector, SlotPool, StopConfig, UrlSelector,
    make_interval_generator,
};
use crate::transport::core::{MeasuredContext, MeasuredOutcome};
use crate::transport::core::{PreparedTurn, RequestExecutor};
use crate::transport::http::TransportSinkConfig;
use crate::user_centric::{UserCentricConfig, UserCentricWorkload};
use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::RequestObserver;
use uuid::Uuid;

use crate::engine::dataset_input::PreparedDatasetInput;
use crate::engine::execution_factories::ExecutionFactories;
use crate::engine::gpu_telemetry::GpuTelemetryRun;
use crate::engine::graph_execution::{
    GraphBackendFactory, GraphBackendFactoryConfig, GraphEndpointRuntimeFactory,
    GraphPlacementFactory, PreparedRunnerGraphEndpointRuntimeFactory,
};
use crate::engine::graph_phase_runtime::{
    GraphPhaseBackendConfig, GraphPhaseBackendFactory, PreparedGraphPhaseBackend, run_graph_phases,
    validate_graph_phases,
};
use crate::engine::heartbeat_lane::{
    CompositePhaseObserver, HeartbeatLane, HeartbeatPhaseObserver,
};
use crate::engine::live_streaming::{LiveResultsSink, PythonLiveStreamingRun, live_phase_observer};
use crate::engine::network_latency::NetworkLatencyRun;
use crate::engine::protocol::{
    AdaptiveControlVariableSpec, AdaptiveScaleSpec, AdaptiveStepPolicySpec, DistributionSpec,
    FileDatasetSpec, MetricsSpec, ModelSelectionStrategy, ModelsSpec, PhaseSpec,
    PublicDatasetSourceSpec, PublicDatasetSpec, RampSpec, RampStrategySpec,
    SequenceDistributionEntrySpec, SourceImageSamplingSpec, SyntheticAudioFormatSpec,
    SyntheticAudioSpec, SyntheticDatasetSpec, SyntheticImageFormatSpec, SyntheticImageSpec,
    SyntheticPrefixPromptsSpec, SyntheticVideoFormatSpec, SyntheticVideoPatternSpec,
    SyntheticVideoSpec,
};
use crate::engine::readiness::{PreparedOnlineReadiness, ReadinessTransportFactory};
use crate::engine::record_lane::RecordArtifactLane;
use crate::engine::records::{
    CapturedHttpExchange, CapturedModelOutput, CapturedRecord, InputSession, group_record_errors,
    observe_otel_record, write_inputs_json, write_outputs_json, write_raw_records_jsonl,
    write_records_csv, write_records_jsonl,
};
use crate::engine::registry::ValidatedEndpointProfileV2;
use crate::engine::server_metrics::ServerMetricsRun;
use crate::engine::sidecar_input::{
    CONTENT_SERVER_SIDECAR_ID, ContentServerSpec, GPU_TELEMETRY_SIDECAR_ID, GpuTelemetrySpec,
    LIVE_STREAMING_SIDECAR_ID, LiveStreamingSpec, NETWORK_LATENCY_SIDECAR_ID, NetworkLatencySpec,
    PreparedSidecarInputs, SERVER_METRICS_SIDECAR_ID, ServerMetricsSpec,
};
use crate::engine::turn_execution::{
    ExecutionBackendConfig, PreparedEndpointTableFactory, RequestExecutorFactory,
};
use crate::server_metrics::ServerMetricsSummary;

type PhaseRuntimeParts = (
    Rc<dyn Workload>,
    Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
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
    pub(crate) tokenizer: crate::engine::protocol::TokenizerSpec,
    pub(crate) phases: Vec<PhaseSpec>,
    pub(crate) metrics: MetricsSpec,
    pub(crate) artifacts: crate::engine::protocol::ArtifactSpec,
    pub(crate) sidecars: NativeSidecarPlan,
    pub(crate) user_files: Vec<crate::engine::protocol_v2::UserFileSpecV2>,
    /// Optional configured run-failure behavior. `None` lets each execution
    /// path select its default at the point of use
    /// ([`OnFailure::scheduled_or_default`] / [`OnFailure::graph_or_default`]).
    pub(crate) failure_policy: Option<OnFailure>,
    /// Whether the native OTLP metrics sink is enabled for this run. When set,
    /// the scheduled path accumulates per-record GenAI-semconv histograms so the
    /// post-report sink emits populated `bucket_counts`; otherwise that
    /// projection is skipped entirely (no per-record recompute cost).
    pub(crate) native_otel_enabled: bool,
    /// Resolved transport binding (`cfg.transport.type`) the graph execution path
    /// builds its worker-local dispatchers over (`build_graph_dispatcher`).
    /// `Some` only when `dataset` is [`NativeDatasetPlan::Graph`]; the scheduled
    /// path resolves its transport through the injected `RequestExecutorFactory`
    /// and leaves this `None`.
    pub(crate) transport: Option<Arc<dyn crate::engine::registry::NativeTransportExecution>>,
}

/// Protocol-neutral retention of one run's already decoded sidecar inputs.
pub(crate) enum NativeSidecarPlan {
    /// Protocol-v2 direct adapter outputs retained through execution.
    Prepared(Arc<PreparedSidecarInputs>),
}

impl NativeSidecarPlan {
    fn content_server(&self) -> Result<Option<&ContentServerSpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(CONTENT_SERVER_SIDECAR_ID)
    }

    fn gpu_telemetry(&self) -> Result<Option<&GpuTelemetrySpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(GPU_TELEMETRY_SIDECAR_ID)
    }

    fn network_latency(&self) -> Result<Option<&NetworkLatencySpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(NETWORK_LATENCY_SIDECAR_ID)
    }

    fn server_metrics(&self) -> Result<Option<&ServerMetricsSpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(SERVER_METRICS_SIDECAR_ID)
    }

    pub(crate) fn live_streaming(&self) -> Result<Option<&LiveStreamingSpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(LIVE_STREAMING_SIDECAR_ID)
    }
}

/// Endpoint preparation selected by the source protocol.
#[derive(Clone)]
pub(crate) enum NativeEndpointPlan {
    /// Protocol-v2 open endpoint profiles.
    Prepared(Arc<Vec<ValidatedEndpointProfileV2>>),
}

impl NativeEndpointPlan {
    fn default_urls(&self) -> Result<&[String]> {
        let Self::Prepared(profiles) = self;
        Ok(&default_prepared_endpoint_profile(profiles)?.config.urls)
    }

    /// Server-token-count policy of the primary (default) endpoint, used for
    /// run-level tokenizer-free input accounting. Uses the default endpoint's
    /// `use_server_token_count`; falls back to `false` when no
    /// default profile resolves.
    pub(crate) fn use_server_token_count(&self) -> bool {
        let Self::Prepared(profiles) = self;
        default_prepared_endpoint_profile(profiles)
            .map(|profile| profile.config.use_server_token_count)
            .unwrap_or(false)
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

/// Worker-local prepared endpoint table factory. `pub(crate)` so the
/// thread-per-core sharded runtime can retain the concrete `Arc` in its shared
/// bundle and build one coordinator resolver per sub-cell thread (the resolver is
/// `Rc`/`!Send`, so it cannot be shared — only rebuilt per thread from this
/// `Send + Sync` factory).
#[derive(Clone)]
pub(crate) struct NativePreparedEndpointTableFactory {
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

    pub(crate) fn coordinator_resolver(&self) -> Result<Rc<dyn PreparedTurnEndpointResolver>> {
        let table = Rc::new(self.prepare_table()?);
        let default = self.reference(DEFAULT_ENDPOINT_PROFILE_ID)?;
        Ok(Rc::new(PreparedEndpointTableResolver::single(
            table, default,
        )?))
    }
}

impl PreparedEndpointTableFactory for NativePreparedEndpointTableFactory {
    fn prepare_worker(&self) -> Result<PreparedEndpointTable> {
        self.prepare_table()
    }
}

/// Protocol-neutral dataset selection.
pub(crate) enum NativeDatasetPlan {
    /// Canonical linear dataset loaded once during protocol-v2 preparation.
    PreparedLinear(PreparedDatasetInput),
    /// Canonical evaluator selection and dataset-load policy.
    StaticAccuracy(NativeStaticAccuracyPlan),
    /// Canonical Graph-IR bundle returned directly by the selected adapter.
    Graph(Box<NativeGraphDatasetPlan>),
}

impl NativeDatasetPlan {
    /// Whether this run drives the graph (trace-replay) engine rather than the
    /// scheduled/linear one. The two are separate execution models over the shared
    /// `{transport, clock}` seam; this is the one predicate the shared driver glue
    /// (`execute_native`, the scheduled/graph entry guards) dispatches on.
    pub(crate) fn is_graph(&self) -> bool {
        matches!(self, Self::Graph(_))
    }
}

/// Process launch parameters for one static-accuracy evaluator.
#[derive(Clone, Debug)]
pub struct StaticAccuracyEvaluatorProcessSpec {
    /// Absolute Python executable used to launch the evaluator.
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
    pub(crate) allow_dataset_wrap: bool,
    pub(crate) t_star_window: crate::engine::graph_input::TStarWindow,
    pub(crate) cache_bust_target: crate::engine::graph_input::CacheBustTarget,
}

/// Execute a protocol-v2 plan through a transport-selected request executor.
///
/// The workload resolves the transport's turn-placement factory by
/// `transport_id` (HTTP vs gRPC `RequestExecutorFactory`) and passes it here
/// with an optional readiness plan (present only for transports that expose a
/// readiness control plane). The graph placement and its `transport_kind` arm
/// are resolved from the shared factories, so both scheduled and graph plans run
/// over either transport through one entry point. Graph input, if present, is
/// already fully prepared: once the workload has returned a canonical
/// `GraphInputBundle`, the harness cannot reinterpret the authored source again.
pub(crate) fn execute_prepared_native_plan_uncommitted_selected(
    plan: NativeRunSpec,
    request_executor: Arc<dyn RequestExecutorFactory>,
    factories: &ExecutionFactories,
    registry: &AIPerfRegistry,
    readiness: Option<Box<dyn PreparedOnlineReadiness>>,
) -> Result<NativeReport> {
    execute_prepared_native_plan_uncommitted_with_runtime_factories(
        plan,
        request_executor,
        factories.graph(),
        registry,
        &BuiltinNativeSidecarResourceFactory,
        readiness.map(|readiness| (readiness, factories.readiness_transport())),
    )
}

/// The single native driver layer: construct the run clock once, then let the
/// clock drive itself.
///
/// The `{clock}` seam is the whole difference between a real and a virtual run.
/// The transport binding declares which via
/// [`uses_virtual_clock`](crate::engine::registry::NativeTransportExecution::uses_virtual_clock)
/// (only `dry_run` with `clock: sim` says virtual); everything downstream —
/// reactor discipline ([`Clock::drive`]), graph placement, and worker count — is
/// a pure function of that one bit, chosen here and nowhere else. A virtual run
/// forces one worker and inline whole-trace placement because a `SimClock` can
/// only advance the single reactor its idle-pump drives; thread-per-core workers
/// own private reactors the pump cannot reach.
fn execute_prepared_native_plan_uncommitted_with_runtime_factories(
    mut plan: NativeRunSpec,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    graph_placement: &dyn GraphPlacementFactory,
    registry: &AIPerfRegistry,
    sidecar_factory: &dyn NativeSidecarResourceFactory,
    readiness: Option<(
        Box<dyn PreparedOnlineReadiness>,
        &dyn ReadinessTransportFactory,
    )>,
) -> Result<NativeReport> {
    validate_plan(&plan)?;
    // Thread-per-core sharding hands each worker a disjoint conversation subset via a
    // modulo partition (`two_level_partition`). When the scheduled dataset has fewer
    // conversations than worker threads, the surplus threads receive an empty subset:
    // a request-bounded phase then fails building its request-rate workload
    // ("conversation dataset cannot be empty"), and a rate phase later fails issuing a
    // new session ("... is not sampleable"). Cap the worker count to the conversation
    // count so every worker owns at least one conversation and recycles it to fill its
    // budget share (matching the Python frontend, which recycles a small dataset to
    // fill request_count). Graph and static-accuracy plans partition differently and
    // are left untouched.
    if let NativeDatasetPlan::PreparedLinear(prepared) = &plan.dataset {
        let conversations = prepared.dataset.conversations().len();
        if conversations > 0 && plan.workers > conversations {
            plan.workers = conversations;
        }
    }
    let virtual_clock = plan
        .transport
        .as_ref()
        .is_some_and(|transport| transport.uses_virtual_clock());
    let real_clock_anchor = RealClockAnchor::now();
    let clock: Rc<dyn Clock> = if virtual_clock {
        Rc::new(crate::clock::SimClock::new())
    } else {
        RealClock::from_anchor(real_clock_anchor)
    };
    let inline_placement = crate::engine::graph_execution::InlineGraphPlacementFactory;
    let placement: &dyn GraphPlacementFactory = if virtual_clock {
        plan.workers = 1;
        &inline_placement
    } else {
        graph_placement
    };
    let slot: Rc<RefCell<Option<Result<NativeReport>>>> = Rc::new(RefCell::new(None));
    let slot_for_body = slot.clone();
    let clock_for_body = clock.clone();
    let outcome = clock.drive(Box::pin(async move {
        let report = prepare_and_execute_native(
            plan,
            clock_for_body,
            real_clock_anchor,
            transport_factory,
            placement,
            registry,
            sidecar_factory,
            readiness,
        )
        .await;
        *slot_for_body.borrow_mut() = Some(report);
    }));
    ensure!(
        !outcome.deadlocked,
        "native virtual-clock run deadlocked: parked with no schedulable virtual-time event"
    );
    slot.borrow_mut()
        .take()
        .ok_or_else(|| anyhow!("native run driver produced no report"))?
}

fn materialize_user_files(
    artifact_dir: &Path,
    files: &[crate::engine::protocol_v2::UserFileSpecV2],
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

/// A side-channel subsystem that samples over the profiling window requires
/// exactly one profiling phase to anchor to.
fn require_single_profiling_phase(request: &NativeRunSpec, subsystem: &str) -> Result<()> {
    ensure!(
        request
            .phases
            .iter()
            .filter(|phase| phase.common().name == "profiling")
            .count()
            == 1,
        "{subsystem} requires exactly one profiling phase"
    );
    Ok(())
}

fn validate_plan(request: &NativeRunSpec) -> Result<()> {
    let _content_server = request.sidecars.content_server()?;
    let gpu_telemetry = request.sidecars.gpu_telemetry()?;
    let network_latency = request.sidecars.network_latency()?;
    let server_metrics = request.sidecars.server_metrics()?;
    let live_streaming = request.sidecars.live_streaming()?;
    ensure!(
        !request.benchmark_id.trim().is_empty(),
        "benchmark_id cannot be empty"
    );
    ensure!(
        !request.models.items.is_empty(),
        "at least one model is required"
    );
    ensure!(
        !request.endpoint.default_urls()?.is_empty(),
        "at least one endpoint URL is required"
    );
    ensure!(!request.phases.is_empty(), "at least one phase is required");
    ensure!(request.workers > 0, "workers must be greater than zero");
    // Sketch retention keeps no per-record values, so per-record artifacts and the
    // native per-record OTLP histograms are impossible. The frontend already drops
    // these from the projection; this fail-closed check guards a hand-authored plan.
    if request.metrics.sketch {
        ensure!(
            request.artifacts.records_path.is_none()
                && request.artifacts.raw_path.is_none()
                && request.artifacts.outputs_path.is_none(),
            "sketch metrics mode cannot emit per-record artifacts \
             (records_path/raw_path/outputs_path); disable them or the sketch flag"
        );
        ensure!(
            !request.native_otel_enabled,
            "sketch metrics mode cannot emit per-record OTLP histograms; \
             disable native OTLP or the sketch flag"
        );
    }
    ensure!(
        request
            .phases
            .iter()
            .any(|phase| phase.common().name == "profiling"),
        "a profiling phase is required"
    );
    for (index, phase) in request.phases.iter().enumerate() {
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
        require_single_profiling_phase(request, "GPU telemetry")?;
    }
    if network_latency.is_some() {
        require_single_profiling_phase(request, "network latency calibration")?;
    }
    if let Some(spec) = server_metrics {
        require_single_profiling_phase(request, "server metrics")?;
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
            .contains(&crate::engine::protocol::ServerMetricsFormatSpec::Jsonl);
        let has_parquet = spec
            .formats
            .contains(&crate::engine::protocol::ServerMetricsFormatSpec::Parquet);
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
    if let NativeDatasetPlan::StaticAccuracy(accuracy) = &request.dataset {
        accuracy.validate()?;
        for phase in &request.phases {
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

/// Whether the run requests a per-record artifact the fold path cannot yet stream,
/// which therefore disqualifies exact-fold.
///
/// The streaming [`RecordArtifactLane`] supports records JSONL, raw JSONL, CSV,
/// Parquet, and `outputs.json`. Per-record OTLP histograms fold into an
/// order-independent accumulator. Reproducible `inputs.json` payloads are generated
/// up front, so these outputs do not require record retention.
///
/// `inputs_need_retain` is the one dataset-dependent input: `inputs.json` still needs
/// the during-run capture path (and so still disqualifies exact-fold) when the dataset
/// is a live-reply multi-turn shape whose later-turn bodies cannot be reproduced up
/// front (see [`dataset_supports_up_front_inputs`]).
///
/// Parquet is only streamable under the `parquet` feature; a lite runner cannot emit
/// it, so a requested Parquet sidecar still disqualifies exact-fold on a lite build
/// (the run then falls to the retain path, which warns and skips the artifact).
fn wants_per_record_artifacts(
    artifacts: &crate::engine::protocol::ArtifactSpec,
    inputs_need_retain: bool,
) -> bool {
    // Per-record OTLP folds at completion and outputs.json streams through the lane,
    // so neither requires retained records.
    #[cfg(feature = "parquet")]
    let parquet_needs_retain = {
        // Parquet streams through the lane on a `parquet` build, so it never forces
        // retain — the sidecar path is not read here.
        let _ = artifacts;
        false
    };
    #[cfg(not(feature = "parquet"))]
    let parquet_needs_retain = artifacts.records_parquet_path.is_some();
    // Dataset-analysis (`--dry-run`) reads the full retained record set to derive its
    // per-turn / length / cache-reuse sections, so it disqualifies exact-fold on BOTH
    // the scheduled and graph paths (the graph path double-gates already — harmless).
    inputs_need_retain || parquet_needs_retain || artifacts.dataset_analysis_path.is_some()
}

/// Whether every conversation in `dataset` can have its `inputs.json` request bodies
/// generated up front, WITHOUT dispatching.
///
/// A conversation is reproducible up front unless it is BOTH multi-turn AND captures
/// live model replies into its later turns — context modes
/// [`DeltasWithoutResponses`](crate::dataset::ConversationContextMode::DeltasWithoutResponses)
/// or
/// [`MessageArrayWithoutResponses`](crate::dataset::ConversationContextMode::MessageArrayWithoutResponses)
/// with more than one turn — because a later turn's body then splices the live reply.
/// This uses the per-conversation rule in
/// [`NativeDatasetConversationSource::build_input_payloads`] so the cheap gate check
/// and the actual generation agree.
fn dataset_supports_up_front_inputs(dataset: &Dataset) -> bool {
    use crate::dataset::ConversationContextMode;
    dataset.conversations().iter().all(|conversation| {
        conversation.turns.len() <= 1
            || !matches!(
                dataset.context_mode(conversation),
                ConversationContextMode::DeltasWithoutResponses
                    | ConversationContextMode::MessageArrayWithoutResponses
            )
    })
}

/// Generate the `inputs.json` sessions up front from the resident `dataset`, reusing
/// the dispatch-side session materializer through a freshly built sequential source.
/// Returns the sessions ready for [`write_inputs_json`], or an error if the
/// source declines (which the caller only reaches after
/// [`dataset_supports_up_front_inputs`] already vouched for the shape).
#[allow(clippy::too_many_arguments)]
fn build_up_front_input_sessions(
    dataset: &Dataset,
    source_factory: &dyn NativeConversationSourceFactory,
    primary_model: &str,
    default_output_tokens: usize,
    rng_root: RngRoot,
    tokenizer: Arc<dyn TextTokenizer>,
    input_token_counter: Arc<dyn InputTokenCounter>,
) -> Result<Vec<InputSession>> {
    let source = source_factory.build(
        dataset.clone(),
        primary_model.to_owned(),
        default_output_tokens,
        rng_root,
        tokenizer,
        input_token_counter,
        true,
    )?;
    let sessions = source
        .materialize_input_payloads()?
        .ok_or_else(|| anyhow!("resident dataset cannot generate inputs.json up front"))?;
    sessions
        .into_iter()
        .map(|session| {
            let payloads = session
                .payloads
                .iter()
                .map(|payload| {
                    serde_json::from_slice::<Box<serde_json::value::RawValue>>(payload)
                        .with_context(|| {
                            format!(
                                "validating up-front inputs.json payload for conversation {:?}",
                                session.session_id
                            )
                        })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(InputSession {
                session_id: session.session_id,
                payloads,
            })
        })
        .collect()
}

/// Force-disable switch for exact-fold. Default on;
/// `AIPERF_RUNTIME_EXACT_FOLD` set to `0`/`false`/`off`/`no` retains records.
pub(crate) fn exact_fold_enabled_by_env() -> bool {
    match std::env::var("AIPERF_RUNTIME_EXACT_FOLD") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

// Exact-fold is eligible when no downstream consumer requires retained records.
// Sharded and cellular execution merge worker-local exact stores independently.
fn exact_fold_eligible(inputs: ExactFoldInputs) -> bool {
    !inputs.sketch_mode
        && !inputs.has_accuracy
        && !inputs.wants_adaptive_record
        && !inputs.has_live_sink
        && !inputs.has_heartbeat
        && !inputs.wants_per_record_artifacts
}

// Inputs considered by exact-fold eligibility.
#[derive(Clone, Copy, Debug)]
struct ExactFoldInputs {
    /// Sketch storage mode: has its own bounded t-digest fold path.
    sketch_mode: bool,
    /// Recorded for call-site clarity; does not affect eligibility.
    #[allow(dead_code)]
    shardable: bool,
    /// Recorded for call-site clarity; does not affect eligibility.
    #[allow(dead_code)]
    is_cellular: bool,
    /// A static/stateful accuracy run: retains records for post-run scoring.
    has_accuracy: bool,
    /// Adaptive scale: samples retained per-turn records per control window.
    wants_adaptive_record: bool,
    /// A Python live sink is attached: reads a per-record clone the fold drops.
    has_live_sink: bool,
    /// The single-process cellular heartbeat lane is enabled: also reads the
    /// per-record clone.
    has_heartbeat: bool,
    /// A per-record file artifact (records/raw/CSV/parquet on a lite build) or the
    /// during-run inputs.json capture still needs the retained records
    /// ([`wants_per_record_artifacts`]).
    wants_per_record_artifacts: bool,
}

/// Whether the graph exact-fold fold-and-drop pass is safe to run given the per-record
/// consumers the graph path actually WIRES.
///
/// Exact-fold folds each clean record into the accumulator and discards it, so it is
/// safe only when nothing downstream reads a per-record clone. The graph path
/// (`execute_graph_native`) STRUCTURALLY constructs no such consumer: it builds no live
/// sink, never accumulates per-record OTLP histograms (`report.otel_per_record` is set
/// ONLY on the scheduled path), and constructs no `HeartbeatLane`. The single per-record
/// consumer a graph run could carry — the live-streaming record extension — is rejected
/// upstream by [`validate_graph_request`], so `live_streaming_wired` is always false
/// here and its absence is a structural invariant we assert rather than a flag we read.
///
/// `native_otel_enabled` and `HeartbeatLane::enabled_by_env` do not identify
/// per-record consumers and therefore do not affect this decision.
fn graph_exact_fold_drop_is_safe(live_streaming_wired: bool) -> bool {
    !live_streaming_wired
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
    ///
    /// The `clock` and `real_clock_anchor` are constructed once at the native
    /// driver layer (real vs virtual chosen there) and threaded in, so the
    /// bundle returns the exact clock scheduling and HTTP execution will use;
    /// the factory does not create a separate clock.
    async fn prepare(
        &self,
        run: &NativeRunSpec,
        clock: Rc<dyn Clock>,
        real_clock_anchor: RealClockAnchor,
    ) -> Result<PreparedNativeSidecarResources>;
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
    content_server: Option<Box<dyn ContentServerRuntime>>,
    gpu_telemetry: Option<GpuTelemetryRun>,
    network_latency: Option<NetworkLatencyRun>,
    server_metrics: Option<ServerMetricsRun>,
    live_streaming: Option<PythonLiveStreamingRun>,
    gpu_records_path: Option<PathBuf>,
    network_latency_records_path: Option<PathBuf>,
    server_metrics_jsonl_path: Option<PathBuf>,
    server_metrics_parquet_wire_path: Option<PathBuf>,
    /// Signals the media-fetch drain task to finish and flush.
    media_finalize: Option<tokio::sync::oneshot::Sender<()>>,
    /// Background task folding content records into media-fetch metrics.
    media_handle: Option<tokio::task::JoinHandle<MediaMetricsSummary>>,
}

/// Artifact filename for per-fetch media records.
const MEDIA_RECORDS_FILENAME: &str = "media_records.jsonl";

/// Ingest one content record into the aggregator and stream its row. Ingestion
/// (and thus metric folding) always happens; the row is written only when the
/// artifact writer is available.
fn ingest_media_record(
    aggregator: &mut MediaFetchAggregator,
    writer: Option<&mut MediaRecordWriter>,
    record: &ContentRequestRecord,
) {
    if let Some(media_record) = aggregator.ingest(record)
        && let Some(writer) = writer
        && let Err(error) = writer.write(&media_record)
    {
        tracing::warn!(error = %error, "writing media_records line failed");
    }
}

#[async_trait(?Send)]
impl NativeSidecarResourceFactory for BuiltinNativeSidecarResourceFactory {
    async fn prepare(
        &self,
        run: &NativeRunSpec,
        clock: Rc<dyn Clock>,
        real_clock_anchor: RealClockAnchor,
    ) -> Result<PreparedNativeSidecarResources> {
        let endpoint_urls = run.endpoint.default_urls()?;
        let content_server_spec = run.sidecars.content_server()?;
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
            .then(|| metrics_config(&run.metrics, run.endpoint.use_server_token_count()))
            .transpose()?;

        let mut media_finalize = None;
        let mut media_handle = None;
        let content_server = match content_server_spec {
            Some(spec) => {
                // Wire media-fetch metrics only when the server publishes files
                // (a content dir is set); otherwise media stays inline, no URLs
                // are fetched, and there is nothing to correlate.
                let record_sink = if spec.content_dir.is_some() {
                    let path =
                        artifact_path(&run.artifact_dir, MEDIA_RECORDS_FILENAME, "media_records")?;
                    let (record_tx, mut record_rx) =
                        tokio::sync::mpsc::unbounded_channel::<ContentRequestRecord>();
                    let (finalize_tx, mut finalize_rx) = tokio::sync::oneshot::channel::<()>();
                    let handle = tokio::spawn(async move {
                        let mut aggregator = MediaFetchAggregator::new();
                        let mut writer = match MediaRecordWriter::create(&path) {
                            Ok(writer) => Some(writer),
                            Err(error) => {
                                tracing::warn!(error = %error, "media_records artifact unavailable");
                                None
                            }
                        };
                        loop {
                            tokio::select! {
                                received = record_rx.recv() => match received {
                                    Some(record) => ingest_media_record(
                                        &mut aggregator,
                                        writer.as_mut(),
                                        &record,
                                    ),
                                    None => break,
                                },
                                _ = &mut finalize_rx => {
                                    while let Ok(record) = record_rx.try_recv() {
                                        ingest_media_record(
                                            &mut aggregator,
                                            writer.as_mut(),
                                            &record,
                                        );
                                    }
                                    break;
                                }
                            }
                        }
                        if let Some(mut writer) = writer {
                            let _ = writer.flush();
                        }
                        aggregator.finish()
                    });
                    media_finalize = Some(finalize_tx);
                    media_handle = Some(handle);
                    Some(record_tx)
                } else {
                    None
                };
                Some(
                    NativeContentServerFactory::default()
                        .start(ContentServerConfig {
                            host: spec.host.clone(),
                            port: spec.port,
                            content_dir: spec.content_dir.clone(),
                            max_tracked_records: spec.max_tracked_records,
                            record_sink,
                        })
                        .await
                        .context("starting native content server")?,
                )
            }
            None => None,
        };

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
                    tracing::warn!(
                        error = format!("{error:#}"),
                        "live telemetry extension failed to start"
                    );
                    None
                }
            }
        } else {
            None
        };

        Ok(PreparedNativeSidecarResources {
            real_clock_anchor,
            clock,
            content_server,
            gpu_telemetry,
            network_latency,
            server_metrics,
            live_streaming,
            gpu_records_path,
            network_latency_records_path,
            server_metrics_jsonl_path,
            server_metrics_parquet_wire_path,
            media_finalize,
            media_handle,
        })
    }
}

impl PreparedNativeSidecarResources {
    /// Signal the media-fetch drain task to finish, then collect its finalized
    /// distributions. Returns an empty summary when the content server had no
    /// media wiring. Idempotent: a second call returns the empty default.
    async fn finalize_media_metrics(&mut self) -> MediaMetricsSummary {
        let (Some(finalize), Some(handle)) = (self.media_finalize.take(), self.media_handle.take())
        else {
            return MediaMetricsSummary::default();
        };
        // The receiver drains remaining records on this signal; a closed channel
        // (task already ended) is fine.
        let _ = finalize.send(());
        match handle.await {
            Ok(summary) => {
                tracing::info!(
                    total_fetches = summary.total_fetches,
                    unmatched = summary.unmatched,
                    negative_ttmf = summary.negative_ttmf,
                    "media-fetch metrics finalized"
                );
                summary
            }
            Err(error) => {
                tracing::warn!(error = %error, "media aggregator task failed to join");
                MediaMetricsSummary::default()
            }
        }
    }

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
            tracing::warn!(
                error = format!("{error:#}"),
                "live telemetry extension failed to activate"
            );
            self.live_streaming.take();
        }
    }

    async fn shutdown_run_resources(&mut self) {
        if let Some(worker) = self.live_streaming.take()
            && let Err(error) = worker.shutdown().await
        {
            tracing::warn!(
                error = format!("{error:#}"),
                "live telemetry extension failed to shut down cleanly"
            );
        }

        // Server-metrics tasks belong to phase sidecars and have already
        // drained. Drop that retained source graph before supervised GPU
        // workers, matching the explicit run-owned cleanup order.
        self.server_metrics.take();
        if let Some(gpu_telemetry) = self.gpu_telemetry.take() {
            gpu_telemetry.shutdown().await;
        }
        self.network_latency.take();
        if let Some(mut content_server) = self.content_server.take()
            && let Err(error) = content_server.shutdown().await
        {
            tracing::warn!(
                error = format!("{error:#}"),
                "content server failed to shut down cleanly"
            );
        }
    }
}

async fn prepare_and_execute_native(
    request: NativeRunSpec,
    clock: Rc<dyn Clock>,
    real_clock_anchor: RealClockAnchor,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    graph_placement: &dyn GraphPlacementFactory,
    registry: &AIPerfRegistry,
    sidecar_factory: &dyn NativeSidecarResourceFactory,
    readiness: Option<(
        Box<dyn PreparedOnlineReadiness>,
        &dyn ReadinessTransportFactory,
    )>,
) -> Result<NativeReport> {
    if request.dataset.is_graph() {
        validate_graph_request(&request)?;
    }
    let mut accuracy = prepare_static_accuracy(&request).await?;
    let mut sidecars = match sidecar_factory
        .prepare(&request, clock, real_clock_anchor)
        .await
    {
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
        transport_factory,
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
    request: NativeRunSpec,
    accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    graph_placement: &dyn GraphPlacementFactory,
    registry: &AIPerfRegistry,
) -> Result<NativeReport> {
    if request.dataset.is_graph() {
        ensure!(
            accuracy.is_none(),
            "graph execution received prepared static-accuracy state"
        );
        return execute_graph_native(request, sidecars, graph_placement, registry).await;
    }
    execute_scheduled_native(request, accuracy, sidecars, transport_factory, registry).await
}

async fn execute_scheduled_native(
    request: NativeRunSpec,
    accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    registry: &AIPerfRegistry,
) -> Result<NativeReport> {
    execute_native_inner(request, accuracy, sidecars, transport_factory, registry).await
}

fn validate_graph_request(request: &NativeRunSpec) -> Result<()> {
    ensure!(
        request.sidecars.live_streaming()?.is_none(),
        "authored Graph-IR runs support the content server and GPU/network/server telemetry side-channels but not the live-streaming record extension"
    );
    ensure!(
        request.models.items.len() == 1,
        "authored Graph-IR runs require exactly one configured default model; per-node model overrides remain supported"
    );
    ensure!(
        matches!(request.models.strategy, ModelSelectionStrategy::RoundRobin),
        "authored Graph-IR runs require round_robin model selection"
    );
    ensure!(
        request.dataset.is_graph(),
        "graph execution requires a direct graph input plan"
    );
    validate_graph_phases(&request.phases)
}

struct OnlineGraphPhaseBackendFactory<'a> {
    placement: &'a dyn GraphPlacementFactory,
    worker_count: usize,
    /// The run's injected clock, handed to the placement so a single-reactor
    /// (virtual) run drives its backend on the `SimClock` rather than a
    /// reconstructed `RealClock`.
    clock: Rc<dyn Clock>,
    real_clock_anchor: RealClockAnchor,
    run_origin_ns: i64,
    model: String,
    default_max_tokens: usize,
    endpoint_runtime_factory: Arc<dyn GraphEndpointRuntimeFactory>,
    segments: Arc<dyn crate::dataset::SegmentStore>,
    metrics: MetricsConfig,
    raw_enabled: bool,
    on_failure: OnFailure,
    cache_bust: Option<crate::engine::graph_execution::GraphCacheBust>,
}

impl GraphPhaseBackendFactory for OnlineGraphPhaseBackendFactory<'_> {
    fn prepare_backend(
        &self,
        config: GraphPhaseBackendConfig,
    ) -> Result<PreparedGraphPhaseBackend> {
        let worker_factory = Arc::new(GraphBackendFactory::new(GraphBackendFactoryConfig {
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
            on_failure: self.on_failure,
            cache_bust: self.cache_bust.clone(),
        }));
        let requires_node_records = self.placement.requires_node_records();
        let placement =
            self.placement
                .build(self.worker_count, worker_factory, self.clock.clone())?;
        Ok(PreparedGraphPhaseBackend {
            placement,
            requires_node_records,
        })
    }
}

async fn execute_graph_native(
    request: NativeRunSpec,
    sidecars: &PreparedNativeSidecarResources,
    graph_placement: &dyn GraphPlacementFactory,
    registry: &AIPerfRegistry,
) -> Result<NativeReport> {
    let graph = match &request.dataset {
        NativeDatasetPlan::Graph(graph) => graph,
        NativeDatasetPlan::PreparedLinear(_) | NativeDatasetPlan::StaticAccuracy(_) => {
            bail!("graph execution received a non-graph dataset plan")
        }
    };
    let graph_random_seed = graph.random_seed;
    let graph_default_output_tokens = graph.default_output_tokens;
    let allow_dataset_wrap = graph.allow_dataset_wrap;
    let t_star_window = graph.t_star_window;
    let metrics_config =
        metrics_config(&request.metrics, request.endpoint.use_server_token_count())?;
    let tokenizer = load_tokenizer(Some(&request.tokenizer.name))?;
    let input_token_counter =
        select_input_token_counter(tokenizer.clone(), request.tokenizer.apply_chat_template);
    let input = graph.input.clone();
    ensure!(
        !input.plans.is_empty(),
        "authored Graph-IR input contains no root traces after root limiting"
    );
    let primary_model = request.models.items[0].name.clone();
    let default_output_tokens = graph_default_output_tokens;
    let NativeEndpointPlan::Prepared(configured_profiles) = &request.endpoint;
    let endpoints_configured = configured_profiles
        .iter()
        .flat_map(|profile| profile.config.urls.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let endpoint_runtime_factory: Arc<dyn GraphEndpointRuntimeFactory> = {
        let NativeEndpointPlan::Prepared(profiles) = &request.endpoint;
        let transport = request
            .transport
            .clone()
            .ok_or_else(|| anyhow!("graph execution plan is missing its transport binding"))?;
        Arc::new(PreparedRunnerGraphEndpointRuntimeFactory::new(
            registry.endpoints().clone(),
            profiles.clone(),
            input_token_counter.clone(),
            transport,
            request
                .sidecars
                .content_server()?
                .filter(|spec| spec.content_dir.is_some())
                .map(|spec| Arc::from(spec.base_url())),
        )?)
    };
    let real_clock_anchor = sidecars.real_clock_anchor;
    let clock = sidecars.clock.clone();
    let start_ns = crate::engine::cell_origin::run_origin_now_ns(&clock);
    let rng_root = RngRoot::new(graph_random_seed.or(request.random_seed));
    let on_failure = OnFailure::graph_or_default(request.failure_policy);
    // Scenario-locked first-turn cache bust mints per-conversation markers only
    // when the run resolves a target.
    let cache_bust = graph.cache_bust_target.is_enabled().then(|| {
        crate::engine::graph_execution::GraphCacheBust {
            benchmark_id: request.benchmark_id.clone(),
            target: graph.cache_bust_target,
        }
    });
    let backends = OnlineGraphPhaseBackendFactory {
        placement: graph_placement,
        worker_count: request.workers,
        clock: clock.clone(),
        real_clock_anchor,
        run_origin_ns: start_ns,
        model: primary_model.clone(),
        default_max_tokens: default_output_tokens,
        endpoint_runtime_factory,
        segments: input.segments.clone(),
        metrics: metrics_config.clone(),
        raw_enabled: request.artifacts.raw_path.is_some(),
        on_failure,
        cache_bust,
    };
    // Telemetry sidecars are side-channel producers synchronized to phase
    // barriers, not to the workload, so the graph path attaches the same
    // phase-sidecar seam the scheduled path uses: server metrics run every
    // phase; GPU and network calibration run only during profiling.
    let phase_sidecars = request
        .phases
        .iter()
        .map(|phase| compose_phase_sidecars(phase, sidecars))
        .collect::<Result<Vec<_>>>()?;
    create_run_artifacts(&request)?;
    let phased = run_graph_phases(
        &request.phases,
        &request.benchmark_id,
        &request.artifact_dir,
        input.as_ref(),
        clock.clone(),
        rng_root,
        allow_dataset_wrap,
        t_star_window,
        phase_sidecars,
        &backends,
        on_failure,
    )
    .await?;
    // Under `Abort` (the graph default) the fail-fast policy latches the run on
    // the first non-cancellation failure, so a surviving failed count is a bug.
    // Under `Continue` failed traces are an expected, recorded outcome — the run
    // succeeds and the records carry the failures, so the assertion must not fire.
    if on_failure.is_abort() {
        ensure!(
            phased.workload.failed == 0,
            "graph phase runtime returned failed traces without failing execution"
        );
    }
    let phase_stats = phased.phases;
    // Graph exact-fold bounds memory independently of record count by folding each
    // record immediately. Per-record artifacts stream through `RecordArtifactLane`;
    // unstreamable Parquet and an explicit disabled exact-fold retain full records.
    // One gate for both executors: the graph path wires NONE of the retain-forcing
    // per-record consumers (live sink / heartbeat / adaptive / accuracy — all rejected
    // or never built by `execute_graph_native`; see `graph_exact_fold_drop_is_safe`),
    // so it passes them as `false` and the shared `exact_fold_eligible` reduces to the
    // graph-relevant `!sketch && !unstreamable_parquet` check. `inputs_need_retain` is
    // false because the graph path never writes `inputs.json`.
    // The dry-run dataset analysis is a per-record consumer: it reads the FULL
    // retained record set (clean + errored) to build the length, timeline, and
    // prefix-cache sections, so requesting it forces retain mode on the graph
    // path (exact-fold would drop the clean records mid-run). Threaded as a local
    // disqualifier on `graph_exact_fold` rather than a shared `ExactFoldInputs`
    // field to keep the change scoped to the graph path.
    let wants_dataset_analysis = request.artifacts.dataset_analysis_path.is_some();
    let graph_exact_fold = exact_fold_enabled_by_env()
        && !wants_dataset_analysis
        && exact_fold_eligible(ExactFoldInputs {
            sketch_mode: request.metrics.sketch,
            shardable: false,
            is_cellular: ModuloCellPartition::from_env().is_some(),
            has_accuracy: false,
            wants_adaptive_record: false,
            has_live_sink: false,
            has_heartbeat: false,
            wants_per_record_artifacts: wants_per_record_artifacts(&request.artifacts, false),
        });
    // Sketch retention (bounded memory) also folds each record into the accumulator's
    // per-(phase,tag) t-digest and drops it: `metrics_config` already carries the
    // `MetricsStorageMode::Sketch` mode (from `metrics_config()` → the sketch storage
    // branch), so `accumulator.process_record` folds-and-clears the row automatically.
    // Like exact-fold, a sketch cell ships its folded STORE (a t-digest store), which
    // the controller merges via `merge_store_partitions` (t-digest merge). Sketch cannot
    // emit per-record artifacts (validated up front at `request.metrics.sketch`), so it
    // never builds a record lane. It is independent of the exact-fold env switch — sketch
    // MUST ship a store (raw-record shipping would defeat its bounded memory and the
    // controller would rebuild an exact accumulator), so it is gated on the sketch flag
    // alone, not `exact_fold_enabled_by_env`.
    let graph_sketch = request.metrics.sketch;
    // The fold-and-drop + ship-store path covers BOTH exact-fold and sketch. Only the
    // retain path (`AIPERF_RUNTIME_EXACT_FOLD=0`, or a lite build's unstreamable Parquet
    // sidecar) keeps the full record `Vec` and runs the batch `write_graph_artifacts`
    // tail. `graph_exact_fold` alone still gates the streaming record lane (sketch has no
    // per-record artifacts to stream).
    let graph_fold = graph_exact_fold || graph_sketch;

    // Streaming per-record artifact lane: on the exact-fold path the fold-drop
    // pass writes each completed record's row(s) here BEFORE dropping the record, so an
    // artifact-enabled graph run streams records/raw/CSV/Parquet/outputs to disk without
    // retaining the full `Vec`. Pointed at the run/cell `artifact_dir` — the exact dir the
    // batch `write_graph_artifacts` targets, so cross-host shipping and same-host
    // concatenation consume the lane's files directly. `None` on the retain path
    // or when no per-record artifact is requested (a metrics-only run).
    let record_lane = if graph_exact_fold {
        RecordArtifactLane::new(
            request
                .artifacts
                .records_path
                .as_ref()
                .map(|path| artifact_path(&request.artifact_dir, path, "records_path"))
                .transpose()?,
            request
                .artifacts
                .raw_path
                .as_ref()
                .map(|path| artifact_path(&request.artifact_dir, path, "raw_path"))
                .transpose()?,
            request
                .artifacts
                .records_csv_path
                .as_ref()
                .map(|path| artifact_path(&request.artifact_dir, path, "records_csv_path"))
                .transpose()?,
            request
                .artifacts
                .records_parquet_path
                .as_ref()
                .map(|path| artifact_path(&request.artifact_dir, path, "records_parquet_path"))
                .transpose()?,
            request
                .artifacts
                .outputs_path
                .as_ref()
                .map(|path| artifact_path(&request.artifact_dir, path, "outputs_path"))
                .transpose()?,
            request.artifacts.trace,
        )?
    } else {
        None
    };

    // INVARIANT: the graph caller feeds the shared `exact_fold_eligible`
    // gate `false` for the live-sink / per-record-OTLP / heartbeat disqualifiers —
    // because the graph execution path wires NONE of them (see
    // [`graph_exact_fold_drop_is_safe`]): `execute_graph_native` builds no live
    // sink, never accumulates per-record OTLP histograms (`report.otel_per_record` is set
    // only on the scheduled path), and constructs no `HeartbeatLane`. The ONLY per-record
    // consumer a graph run could carry — the live-streaming record extension — is rejected
    // upstream by `validate_graph_request`, so this tripwire asserts on that structural
    // fact (no live-streaming consumer wired), NOT on the raw `native_otel_enabled` /
    // `HeartbeatLane::enabled_by_env()` config/env probes, which do not indicate
    // whether graph execution built a corresponding consumer.
    debug_assert!(
        !graph_fold
            || graph_exact_fold_drop_is_safe(
                request.sidecars.live_streaming().ok().flatten().is_some()
            ),
        "graph exact-fold drops per-record data, but a live-streaming record consumer \
         that reads it is wired on the graph path — validate_graph_request should have \
         rejected it before execution (thread any NEW per-record consumer's disqualifier \
         into the graph caller's ExactFoldInputs by setting the matching field true)"
    );

    let gpu_telemetry = sidecars.gpu_telemetry.as_ref();
    let network_latency = sidecars.network_latency.as_ref();
    let server_metrics = sidecars.server_metrics.as_ref();
    let mut accumulator = MetricsAccumulator::with_config(metrics_config.clone());

    // The fold-and-drop pass folds every record while computing profiling span,
    // successful endpoints, warmup presence, and full-run span. Exact-fold retains
    // errored and canceled records for report error details and drops clean records;
    // retained-record execution keeps the full vector for shipping and artifacts.
    let mut has_warmup = false;
    let mut profiling_start: Option<i64> = None;
    let mut profiling_end: Option<i64> = None;
    let mut endpoints_successful_set: BTreeSet<String> = BTreeSet::new();
    // The full-run span (across every phase) — the elapsed span the cell ships as
    // `epoch_ns`, matching the retain path's records min-start .. max-end derivation.
    let mut run_start: Option<i64> = None;
    let mut run_end: Option<i64> = None;
    let mut errored_count: u64 = 0;
    let captured: Vec<CapturedRecord> = {
        let mut retained: Vec<CapturedRecord> = Vec::new();
        for record in phased.captured {
            let ingest = &record.ingest;
            accumulator.process_record(ingest);
            let is_error = ingest.errored || ingest.canceled;
            if is_error {
                errored_count += 1;
            }
            if ingest.phase == MetricsPhase::Warmup {
                has_warmup = true;
            }
            run_start = Some(run_start.map_or(ingest.start_ns, |value| value.min(ingest.start_ns)));
            run_end = Some(run_end.map_or(ingest.end_ns, |value| value.max(ingest.end_ns)));
            if ingest.phase == MetricsPhase::Profiling {
                profiling_start = Some(
                    profiling_start.map_or(ingest.start_ns, |value| value.min(ingest.start_ns)),
                );
                profiling_end =
                    Some(profiling_end.map_or(ingest.end_ns, |value| value.max(ingest.end_ns)));
                if !is_error && let Some(url) = ingest.dimensions.endpoint_url.clone() {
                    endpoints_successful_set.insert(url);
                }
            }
            if graph_fold {
                // Stream every record's artifact row before the fold drops it. The lane
                // sees the full record set (clean + errored), so its files match the
                // batch writer's full-Vec output; only errored records are retained for
                // report error details. The accumulator has
                // already folded this record (exact columns or the sketch t-digest), so
                // dropping the clean ones bounds memory on both fold paths.
                if let Some(lane) = &record_lane {
                    lane.write(&record, &metrics_config)?;
                }
                if is_error {
                    retained.push(record);
                }
            } else {
                retained.push(record);
            }
        }
        retained
    };
    // Flush the streaming lane now that every record has been written. A no-op
    // on the retain path (`record_lane` is `None`); the batch `write_graph_artifacts`
    // tail below handles that path instead.
    if let Some(lane) = &record_lane {
        lane.finish()?;
    }

    // A graph cell ships its terminal partition to the controller. Absent the controller
    // address (the single-process path) this is inert. Two shapes, by mode:
    // - RETAIN: ship the full captured record `Vec` — each carries a LOCAL per-cell
    //   `request_index` — which the controller concatenation-merges (by cell_id) into the
    //   single authoritative report.
    // - FOLD (exact-fold, or sketch): the fold-and-drop pass folded every record
    //   into `accumulator` and dropped the clean ones (`captured` holds only the retained
    //   errored records), so there is no full record `Vec` — ship the folded STORE instead
    //   (exact columns, or the sketch t-digest store). The controller appends every cell's
    //   store (`merge_store_partitions`) into the merged report. The counters are exact:
    //   `issued` is the accumulator's ingested count (which survives sketch's fold-and-clear
    //   — `record_count()` is 0 for a sketch store), `errored` the retained errored count,
    //   so `completed = issued - errored`.
    #[cfg(feature = "cellular")]
    if let Some(shipper) = crate::engine::cellular_cell::CellRecordsShipper::from_env() {
        // No `capture`/wall clock is in scope on the graph path, so derive the run span
        // from the records themselves: last observed end minus first observed start,
        // matching the elapsed span the scheduled path passes.
        let epoch_ns: i64 = run_end
            .unwrap_or(0)
            .saturating_sub(run_start.unwrap_or(0))
            .max(0);
        if graph_fold {
            let issued = accumulator.ingested_count();
            let counters = crate::cellular::HeartbeatCounters {
                issued,
                completed: issued.saturating_sub(errored_count),
                errored: errored_count,
            };
            shipper.ship_store(accumulator.column_store().clone(), counters, epoch_ns)?;
        } else {
            let records: Vec<RecordIngest> = captured
                .iter()
                .map(|record| record.ingest.clone())
                .collect();
            shipper.ship_records(records, epoch_ns)?;
        }
    }
    let RunMetricsSummaries {
        profiling_metrics,
        profiling_server_summary,
        warmup,
        warmup_server_summary,
    } = summarize_run_metrics(
        &mut accumulator,
        gpu_telemetry,
        network_latency,
        server_metrics,
        &request,
        &metrics_config,
        has_warmup,
    );
    // Under either fold path `captured` holds only the retained errored records (the clean
    // ones were dropped mid-run). Exact-fold already STREAMED any requested per-record
    // artifact through `record_lane` (flushed above); sketch requests none (it is
    // metrics-only by validation). So the batch writers must NOT run on the fold path (they
    // would see only the errored subset). Only the retain path (lite Parquet /
    // `AIPERF_RUNTIME_EXACT_FOLD=0` on a non-sketch run) writes them from the full Vec here.
    if !graph_fold {
        write_graph_artifacts(&request, &captured, &metrics_config)?;
        // Dataset analysis reads the full retained record set. `wants_dataset_analysis`
        // forced the retain path above, so `captured` holds every clean + errored record.
        if let Some(relative) = &request.artifacts.dataset_analysis_path {
            let base = artifact_path(&request.artifact_dir, relative, "dataset_analysis_path")?;
            let analysis_request = crate::engine::dataset_analysis_writer::DatasetAnalysisRequest {
                path: base,
                options: crate::dataset::analysis::AnalysisOptions {
                    block_size: request.artifacts.dataset_analysis_block_size.unwrap_or(16),
                    explicit_cache_blocks: request.artifacts.dataset_analysis_cache_blocks,
                },
                per_conversation: request.artifacts.dataset_analysis_per_conversation,
            };
            crate::engine::dataset_analysis_writer::write_dataset_analysis(
                &analysis_request,
                &captured,
                &input,
            )?;
        }
    }
    let server_metrics_report = write_sidecar_records(
        gpu_telemetry,
        network_latency,
        server_metrics,
        sidecars.gpu_records_path.as_deref(),
        sidecars.network_latency_records_path.as_deref(),
        sidecars.server_metrics_jsonl_path.as_deref(),
        sidecars.server_metrics_parquet_wire_path.as_deref(),
        profiling_server_summary.as_ref(),
        warmup_server_summary.as_ref(),
    )?;

    // The profiling span + successful-endpoint set were accumulated over the FULL record
    // set in the fold-and-drop pass (before any drop), so they are correct on both the
    // retain and exact-fold paths — unlike a scan over `captured`, which under exact-fold
    // holds only the retained errored subset.
    let start_time = profiling_start;
    let end_time = profiling_end;
    let endpoints_successful = endpoints_successful_set.into_iter().collect();
    let media_metrics = sidecars.finalize_media_metrics().await.metrics;
    let summary = ReportSummary {
        start_time,
        end_time,
        duration_s: start_time
            .zip(end_time)
            .map(|(start, end)| end.saturating_sub(start) as f64 / 1_000_000_000.0),
        was_cancelled: phase_stats.iter().any(|phase| phase.was_cancelled),
        endpoints_configured,
        endpoints_successful,
        server_metrics: server_metrics_report,
    };
    let outcome = RunOutcome {
        run: ReportRunInfo {
            mode: Some("graph".into()),
            model: Some(primary_model),
        },
        summary,
        warmup,
        server_metrics: profiling_server_summary
            .as_ref()
            .map(|summary| summary.sidecar_metrics().clone())
            .unwrap_or_default(),
        warmup_server_metrics: warmup_server_summary
            .as_ref()
            .map(|summary| summary.sidecar_metrics().clone())
            .unwrap_or_default(),
        media_metrics,
        ..RunOutcome::default()
    };
    // Cross-host cells ship local per-record artifacts to the controller with
    // streaming zstd. Same-host and single-process runs need no upload.
    #[cfg(feature = "cellular")]
    crate::engine::cellular_cell::ship_http_artifacts_if_enabled(
        &request.artifact_dir,
        &request.artifacts,
    )?;
    Ok(NativeReport::from_outcome(&profiling_metrics, &outcome))
}

/// The post-capture metric summaries shared by both executors' finalize tails
/// ([`summarize_run_metrics`]): the profiling metrics export (with any GPU
/// telemetry already attached) plus the optional profiling/warmup server
/// summaries and the optional warmup metrics export.
struct RunMetricsSummaries {
    profiling_metrics: AccumulatorSummary,
    profiling_server_summary: Option<ServerMetricsSummary>,
    warmup: Option<AccumulatorSummary>,
    warmup_server_summary: Option<ServerMetricsSummary>,
}

/// Inject the calibrated network RTT into the accumulator, export the profiling
/// metrics (attaching GPU telemetry when present), and derive the profiling and
/// warmup server-metrics summaries plus the warmup metrics export. The two
/// callers differ only in cell shipping, artifact writing, and outcome assembly.
fn summarize_run_metrics(
    accumulator: &mut MetricsAccumulator,
    gpu_telemetry: Option<&GpuTelemetryRun>,
    network_latency: Option<&NetworkLatencyRun>,
    server_metrics: Option<&ServerMetricsRun>,
    request: &NativeRunSpec,
    metrics_config: &MetricsConfig,
    has_warmup: bool,
) -> RunMetricsSummaries {
    if let Some(network_latency) = network_latency {
        let mean_rtt_ns = network_latency.mean_rtt_ns();
        if network_latency.is_active_probe() && mean_rtt_ns.is_none() {
            tracing::warn!(
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
    // `has_warmup` was computed in the fold-and-drop pass over the full record set
    // (before any drop), so it is correct on both the retain and exact-fold paths.
    let warmup =
        has_warmup.then(|| accumulator.export_results(&ExportContext::phase(MetricsPhase::Warmup)));
    let warmup_server_summary = server_metrics
        .filter(|_| warmup.is_some())
        .map(|server_metrics| {
            server_metrics.summarize(MetricsPhase::Warmup, metrics_config.slice_duration_ns)
        });
    RunMetricsSummaries {
        profiling_metrics,
        profiling_server_summary,
        warmup,
        warmup_server_summary,
    }
}

/// Write the GPU / network-latency / server-metrics record sidecars (each a no-op
/// when its producer or destination path is absent) and build the additive
/// server-metrics report metadata. Byte-identical tail shared by the
/// scheduled/accuracy and graph executors; the record paths are passed in so both
/// callers (graph inlines `sidecars.*`, scheduled pre-binds the same as locals)
/// produce identical behavior.
#[allow(clippy::too_many_arguments)]
fn write_sidecar_records(
    gpu_telemetry: Option<&GpuTelemetryRun>,
    network_latency: Option<&NetworkLatencyRun>,
    server_metrics: Option<&ServerMetricsRun>,
    gpu_records_path: Option<&Path>,
    network_latency_records_path: Option<&Path>,
    server_metrics_jsonl_path: Option<&Path>,
    server_metrics_parquet_wire_path: Option<&Path>,
    profiling_server_summary: Option<&ServerMetricsSummary>,
    warmup_server_summary: Option<&ServerMetricsSummary>,
) -> Result<Option<ReportServerMetricsMetadata>> {
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
        profiling_server_summary
            .map(|profiling| server_metrics.report_metadata(profiling, warmup_server_summary))
    });
    Ok(server_metrics_report)
}

/// Emit the optional wide per-record Parquet sidecar beside the per-request
/// JSONL. Best-effort and gated on the `parquet` feature: a runner built without
/// it warns once and skips, so a lite build still decodes the wire field.
fn write_records_parquet_artifact(
    request: &NativeRunSpec,
    captured: &[CapturedRecord],
    metrics_config: &MetricsConfig,
) -> Result<()> {
    let Some(parquet_path) = &request.artifacts.records_parquet_path else {
        return Ok(());
    };
    let path = artifact_path(&request.artifact_dir, parquet_path, "records_parquet_path")?;
    #[cfg(feature = "parquet")]
    {
        crate::engine::records::write_records_parquet(
            &path,
            captured,
            metrics_config,
            request.artifacts.trace,
        )?;
    }
    #[cfg(not(feature = "parquet"))]
    {
        let _ = (captured, metrics_config);
        tracing::warn!(
            "records_parquet requested ({}) but this runner was built without the \
             `parquet` feature; skipping",
            path.display()
        );
    }
    Ok(())
}

/// Emit the optional per-record CSV sidecar beside the per-request JSONL. Unlike
/// the Parquet sidecar this needs no extra Cargo feature (CSV is stdlib), so it is
/// always available.
fn write_records_csv_artifact(
    request: &NativeRunSpec,
    captured: &[CapturedRecord],
    metrics_config: &MetricsConfig,
) -> Result<()> {
    let Some(csv_path) = &request.artifacts.records_csv_path else {
        return Ok(());
    };
    let path = artifact_path(&request.artifact_dir, csv_path, "records_csv_path")?;
    write_records_csv(&path, captured, metrics_config, request.artifacts.trace)
}

fn write_graph_artifacts(
    request: &NativeRunSpec,
    captured: &[CapturedRecord],
    metrics_config: &MetricsConfig,
) -> Result<()> {
    if let Some(records_path) = &request.artifacts.records_path {
        let path = artifact_path(&request.artifact_dir, records_path, "records_path")?;
        write_records_jsonl(&path, captured, metrics_config, request.artifacts.trace)?;
    }
    write_records_parquet_artifact(request, captured, metrics_config)?;
    write_records_csv_artifact(request, captured, metrics_config)?;
    if let Some(raw_path) = &request.artifacts.raw_path {
        let path = artifact_path(&request.artifact_dir, raw_path, "raw_path")?;
        write_raw_records_jsonl(&path, captured)?;
    }
    if let Some(outputs_path) = &request.artifacts.outputs_path {
        let path = artifact_path(&request.artifact_dir, outputs_path, "outputs_path")?;
        write_outputs_json(&path, captured, metrics_config)?;
    }
    Ok(())
}

async fn prepare_static_accuracy(request: &NativeRunSpec) -> Result<Option<PreparedAccuracy>> {
    let NativeDatasetPlan::StaticAccuracy(spec) = &request.dataset else {
        return Ok(None);
    };
    let model = request
        .models
        .items
        .first()
        .map(|item| item.name.as_str())
        .ok_or_else(|| anyhow!("at least one model is required"))?;
    let tokenizer = load_tokenizer(Some(&request.tokenizer.name))?;
    let mut evaluator = spec.evaluator_factory.spawn(&spec.process).await?;
    let preparation = async {
        let evaluator_config = EvaluatorLoadConfig {
            tasks: spec.tasks.clone(),
            n_shots: spec.n_shots,
            enable_cot: spec.enable_cot,
            system_prompt: spec.system_prompt.clone(),
            max_problems: None,
            max_tokens: None,
            seed: request.random_seed.unwrap_or(0),
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
            finish_with_shutdown(Err(error), shutdown, "accuracy evaluator")
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
    finish_with_shutdown(result, shutdown, "accuracy evaluator")
}

/// Reconcile a primary result with the outcome of a subsequent resource
/// shutdown, preserving the primary error while surfacing any shutdown failure.
///
/// `label` names the resource being torn down (for example `"accuracy
/// evaluator"` or `"execution backend"`) so both call sites share one match.
fn finish_with_shutdown<T>(result: Result<T>, shutdown: Result<()>, label: &str) -> Result<T> {
    match (result, shutdown) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error.context(format!("shutting down {label}"))),
        (Err(error), Err(shutdown)) => {
            Err(error.context(format!("{label} also failed during shutdown: {shutdown:#}")))
        }
    }
}

/// The `Send + Sync` inputs one thread-per-core sub-cell needs to build and run
/// its whole scheduled pipeline, shared read-only across the `W` worker threads
/// through an `Arc`.
///
/// Everything here is either owned+`Send` (config, dataset, phase specs, rng
/// roots), an `Arc` handle (the transport factory, the prepared-endpoint table
/// factory, the tokenizers), or `Copy` (the clock anchor). The `!Send` per-thread
/// stack — clock, transport sink, `RunCapture`, dispatcher, `SlotPool`s, plans —
/// is built *inside* each worker from these. Only the read-only dataset/registry
/// `Arc`s cross the spawn boundary, so no lock sits on the hot path.
pub(crate) struct ShardedShared {
    /// The injected transport factory (HTTP or gRPC). Each thread builds its own
    /// `workers == 1` sink from it, co-locating scheduler and transport.
    pub(crate) transport_factory: Arc<dyn RequestExecutorFactory>,
    /// Concrete prepared-endpoint table factory; each thread derives its own
    /// worker-local prepared table and (`Rc`) coordinator resolver from it.
    pub(crate) table_factory: Arc<NativePreparedEndpointTableFactory>,
    /// Cloned sampler registry (the source factory borrows it per thread).
    pub(crate) samplers: crate::dataset::SamplerRegistry,
    /// The composed dataset every thread partitions.
    pub(crate) dataset: Dataset,
    /// Effective primary model.
    pub(crate) primary_model: String,
    /// Resolved native metrics policy.
    pub(crate) metrics_config: MetricsConfig,
    /// Shared response tokenizer.
    pub(crate) tokenizer: Arc<dyn TextTokenizer>,
    /// Shared input-token counter.
    pub(crate) input_token_counter: Arc<dyn InputTokenCounter>,
    /// Ordered inference endpoint URLs.
    pub(crate) endpoint_urls: Vec<String>,
    /// Fully resolved transport policy.
    pub(crate) transport_config: TransportSinkConfig,
    /// Dataset default output-token bound.
    pub(crate) default_output_tokens: usize,
    /// Dataset RNG root (seeded per phase inside the plan builder).
    pub(crate) dataset_rng_root: RngRoot,
    /// Run RNG root (arrival/cancellation/ramp derivations).
    pub(crate) rng_root: RngRoot,
    /// The authored phases (unsliced; each thread slices its own copy by `W`).
    pub(crate) phases: Vec<PhaseSpec>,
    /// Stable benchmark id (adaptive artifact naming).
    pub(crate) benchmark_id: String,
    /// Run artifact directory (adaptive artifacts).
    pub(crate) artifact_dir: PathBuf,
    /// Whether raw HTTP exchanges are retained.
    pub(crate) raw_enabled: bool,
    /// Whether `inputs.json` canonical payloads are retained.
    pub(crate) inputs_enabled: bool,
    /// Final (relative) per-record artifact paths for the per-shard lanes.
    /// When this sharded run selected exact-fold AND any of these is requested, each
    /// worker opens its OWN [`RecordArtifactLane`] to a per-shard temp file derived
    /// from the artifact's file name (see
    /// [`crate::engine::shard_artifacts`]); the coordinator concatenates the
    /// per-shard files into the single final artifact at finalize. `None` on the
    /// retain path (the batch writers run over the merged retained records instead).
    pub(crate) records_path: Option<PathBuf>,
    pub(crate) raw_path: Option<PathBuf>,
    pub(crate) records_csv_path: Option<PathBuf>,
    pub(crate) records_parquet_path: Option<PathBuf>,
    pub(crate) outputs_path: Option<PathBuf>,
    /// Whether per-record artifacts include transport-timing trace columns.
    pub(crate) include_trace: bool,
    /// Whether an adaptive phase needs each completed turn's terminal record.
    pub(crate) wants_adaptive_record: bool,
    /// Static-accuracy response associations, shared read-only across the shards.
    ///
    /// `Some` only for a static-accuracy run: each shard clones this `Send + Sync`
    /// handle, builds its own capture [`AccuracyRecordProcessor`] over the SAME
    /// associations, and registers it on the profiling phase. The disjoint per-shard
    /// captures concatenate at the coordinator, which grades them once on the main
    /// thread (the single `!Send` Python evaluator never crosses the spawn boundary).
    pub(crate) accuracy_associations: Option<Arc<[ProblemAssociation]>>,
    /// Whether this sharded run selected exact-fold: each worker capture
    /// folds every completed record into its own EXACT accumulator (stamped with a
    /// LOCAL-dense fold ordinal) and drops the heavy per-record data mid-run, so the
    /// coordinator merges bounded per-shard accumulators instead of retaining every
    /// record. `false` retains records for finalization.
    pub(crate) exact_fold: bool,
    /// Run-failure discipline.
    pub(crate) on_failure: OnFailure,
    /// The shared monotonic real-clock origin; each thread builds a reactor-local
    /// clock from it so all timestamps sit on one timeline.
    pub(crate) real_clock_anchor: RealClockAnchor,
    /// The run origin (`now_ns` captured once on the main thread).
    pub(crate) start_ns: i64,
    /// This process's cell id (0 when not a controller child).
    pub(crate) cell_id: u32,
    /// This run's cell count (1 when not a controller child).
    pub(crate) cells: u32,
    /// This cell's thread-per-core sub-cell count.
    pub(crate) workers: u32,
    /// Each phase's global ordinal base (env for a controller child, computed for a
    /// lone process), injected identically into every thread's issuer.
    pub(crate) phase_ordinal_bases: HashMap<MetricsPhase, usize>,
}

/// A sub-cell thread's finished records: kept exactly (retained for the report
/// ingest + per-record artifacts) or folded into a bounded per-shard accumulator and
/// dropped (sketch or exact-fold). Folded storage uses
/// O(shards × accumulator) memory independently of record count.
pub(crate) enum ShardRecords {
    /// Exact mode: every record retained, each stamped with its global two-level
    /// dispatch ordinal. The report tail ingests them and per-record artifacts read
    /// them.
    Retained(Vec<CapturedRecord>),
    /// Fold-and-drop mode (sketch OR exact-fold): records streamed into this
    /// shard's bounded accumulator and dropped; only errored records survive for the
    /// report's error grouping. Shards merge accumulator-to-accumulator, never by
    /// concatenating records — sketch merges an associative t-digest partition, exact-
    /// fold concatenates the shard's dense LOCAL-ordinal store through `append_store`.
    Folded {
        accumulator: MetricsAccumulator,
        errored: Vec<CapturedRecord>,
    },
}

impl Default for ShardRecords {
    fn default() -> Self {
        ShardRecords::Retained(Vec::new())
    }
}

impl ShardRecords {
    /// Merge another shard's records into this one: concatenate retained records, or
    /// merge folded accumulators (append-only) + concatenate their errored records.
    /// Every shard runs the same storage mode, so the variants always match.
    fn absorb(&mut self, other: ShardRecords) -> Result<()> {
        match (self, other) {
            (ShardRecords::Retained(a), ShardRecords::Retained(b)) => a.extend(b),
            (
                ShardRecords::Folded {
                    accumulator: a,
                    errored: ea,
                },
                ShardRecords::Folded {
                    accumulator: b,
                    errored: eb,
                },
            ) => {
                a.merge(&b).map_err(|error| {
                    anyhow!("merging sharded fold-and-drop partitions: {error}")
                })?;
                ea.extend(eb);
            }
            _ => bail!(
                "sharded scheduled shards disagree on storage mode (retained vs folded) — \
                 every shard must run the same metrics storage mode"
            ),
        }
        Ok(())
    }
}

/// One sub-cell thread's finished records plus the phase facts the once-per-cell
/// report tail folds across threads. `Send` so it crosses the worker join.
#[derive(Default)]
pub(crate) struct ScheduledShardOutcome {
    /// This thread's records — retained (retain path) or folded-and-dropped (sketch
    /// or exact-fold).
    pub(crate) records: ShardRecords,
    /// This thread's `inputs.json` sessions (disjoint conversation ids across
    /// threads, so the union needs only a re-sort by session id).
    pub(crate) input_sessions: Vec<InputSession>,
    /// This thread's static-accuracy terminal captures (empty for a non-accuracy
    /// run). Disjoint across shards (each stamps globally-unique dispatch
    /// sequences), so the coordinator concatenates them and grades once.
    pub(crate) accuracy_captures: Vec<CapturedResponse>,
    /// Whether any of this thread's phases was externally cancelled.
    pub(crate) was_cancelled: bool,
    /// Whether this thread ran a warmup phase (gates the warmup metrics export).
    pub(crate) has_warmup: bool,
}

impl ScheduledShardOutcome {
    /// Fold another thread's shard into this one: merge records (mode-aware), union
    /// input sessions and accuracy captures, OR the phase flags. Record ordering is
    /// applied once after all shards are absorbed (retained records only).
    pub(crate) fn absorb(&mut self, other: ScheduledShardOutcome) -> Result<()> {
        self.records.absorb(other.records)?;
        self.input_sessions.extend(other.input_sessions);
        self.accuracy_captures.extend(other.accuracy_captures);
        self.was_cancelled |= other.was_cancelled;
        self.has_warmup |= other.has_warmup;
        Ok(())
    }
}

/// Build and run one sub-cell thread's complete scheduled pipeline over its `1/W`
/// nested partition, returning its record shard.
///
/// Each thread is a self-contained sub-cell with a fresh
/// reactor-local clock, a `workers == 1` co-located transport (no hop), an
/// injected two-level [`IssuanceAuthority`] + injected ordinal bases, its
/// per-thread cell partition threaded into both the sampler and the issuer, and
/// phases sliced to this thread's `1/W` share. It runs [`run_scheduled_phases`]
/// and never touches a sidecar, the live sink, the
/// heartbeat lane, or an artifact — those stay once-per-cell on the main thread.
///
/// Called on a worker OS thread inside that thread's own `current_thread` runtime
/// + `LocalSet`, so its entire `!Send` stack is thread-local.
pub(crate) async fn execute_scheduled_shard(
    shared: &ShardedShared,
    thread_id: usize,
) -> Result<ScheduledShardOutcome> {
    // A reactor-local clock on the shared real-clock timeline (the graph
    // thread-per-core model); never the coordinator's clock object.
    let clock: Rc<dyn Clock> = RealClock::from_anchor(shared.real_clock_anchor);
    let start_ns = shared.start_ns;
    // This thread's nested `(cell × thread)` partition — the same object feeds the
    // sampler (which instances it draws) and the issuer (which global ordinals it
    // stamps), so `within*(cells*W) + index == instance` holds and the ordinals
    // tile 0..total.
    let partition = crate::engine::sharded_scheduled::two_level_partition(
        shared.cell_id,
        shared.cells,
        thread_id,
        shared.workers,
    )?;

    let prepared_endpoints: Arc<dyn PreparedEndpointTableFactory> = shared.table_factory.clone();
    let execution_backend = shared.transport_factory.build(ExecutionBackendConfig {
        // One worker keeps the sink and scheduler on this thread's reactor.
        workers: 1,
        coordinator_clock: clock.clone(),
        real_clock_anchor: shared.real_clock_anchor,
        base_urls: shared.endpoint_urls.clone(),
        model: shared.primary_model.clone(),
        transport: shared.transport_config.clone(),
        prepared_endpoints: Some(prepared_endpoints),
    })?;
    // when this sharded run selected exact-fold AND a per-record artifact is
    // requested, this shard streams each completed record's rows into its OWN lane
    // writing to a per-shard temp file (`<artifact_dir>/.shard-<id>/<name>`), dropping
    // the record immediately after — exactly like the single-thread lane. The
    // coordinator concatenates the per-shard files into the single final artifact at
    // finalize (`shard_artifacts::concatenate_shard_artifacts`). `None` on the retain
    // path or when no lane artifact is requested.
    let record_lane = if shared.exact_fold {
        let per_shard = |relative: &Option<PathBuf>| -> Option<PathBuf> {
            relative.as_ref().map(|path| {
                crate::engine::shard_artifacts::shard_artifact_path(
                    &shared.artifact_dir,
                    thread_id,
                    path,
                )
            })
        };
        RecordArtifactLane::new(
            per_shard(&shared.records_path),
            per_shard(&shared.raw_path),
            per_shard(&shared.records_csv_path),
            per_shard(&shared.records_parquet_path),
            per_shard(&shared.outputs_path),
            shared.include_trace,
        )?
    } else {
        None
    };
    let capture = Rc::new(
        RunCapture::new_with_issuance_and_bases(
            clock.clone(),
            start_ns,
            shared.metrics_config.clone(),
            shared.raw_enabled,
            // Under exact-fold, inputs.json is generated once at the coordinator
            // from the resident dataset, so the
            // shard never captures it during dispatch (which would double-count across
            // shards). The retain path keeps the per-shard during-run capture.
            shared.inputs_enabled && !shared.exact_fold,
            // Worker threads never feed the live sink or heartbeat lane (D5); those are
            // driven once-per-cell on the main thread.
            false,
            shared.wants_adaptive_record,
            // when the run selected exact-fold, each worker capture folds every
            // completed record into its OWN exact accumulator and drops the heavy
            // per-record data mid-run. The fold ordinal comes from this capture's private
            // `fold_dispatch_next` counter (`assign_fold_ordinal` at `begin`), which is
            // LOCAL-dense `0..N_shard` — NOT the STRIDED global ordinal the
            // `CellularAutonomousIssuer` stamps on the retain path (`global_ordinal`, used
            // only in `patch_joined_ingest`). So each shard's store is dense and the shards
            // concatenate through `append_store` at merge, yielding a summary within
            // numerical tolerance of the retain path (counts/percentiles exact,
            // sums/means a few ULPs).
            shared.exact_fold,
            crate::engine::cellular_cell::issuance_authority_for(partition),
            shared.phase_ordinal_bases.clone(),
        )
        .with_record_lane(record_lane)
        // Stage each turn's model output text so this shard's streaming outputs.json
        // entry carries it, then drop it in the fold (exact-fold + outputs.json only).
        .with_outputs_capture(shared.exact_fold && shared.outputs_path.is_some()),
    );
    let resolver = shared.table_factory.coordinator_resolver()?;
    let source_factory = PreparedNativeConversationSourceFactory {
        endpoint_resolver: resolver,
        samplers: &shared.samplers,
        // Inject this thread's partition so its sampler draws only its nested
        // subset of the cell's instances.
        cell_partition: Some(partition),
    };
    let sliced_phases: Vec<PhaseSpec> = shared
        .phases
        .iter()
        .map(|phase| {
            crate::engine::sharded_scheduled::slice_phase_for_thread(
                phase,
                thread_id,
                shared.workers,
            )
        })
        .collect();

    // A static-accuracy run gives each shard its OWN capture processor over the
    // shared read-only associations; it is registered on the profiling phase below
    // and drained into the shard outcome after the run. The disjoint per-shard
    // captures concatenate at the coordinator, which grades once on the main thread.
    let shard_accuracy_processor: Option<Rc<AccuracyRecordProcessor>> = shared
        .accuracy_associations
        .as_ref()
        .map(|associations| Rc::new(AccuracyRecordProcessor::new(associations.clone())));

    let execution_result = async {
        execution_backend.set_run_origin(start_ns)?;
        execution_backend.configure_measurement(shared.metrics_config.clone(), start_ns)?;
        let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(ConfiguredDispatcher {
            execution_backend: execution_backend.clone(),
            model: shared.primary_model.clone(),
            capture: capture.clone(),
        });
        let shared_resources = native_scheduled_resources(&sliced_phases);

        let mut plans = Vec::with_capacity(sliced_phases.len());
        for (phase_index, phase) in sliced_phases.iter().enumerate() {
            let mut plan = build_native_scheduled_phase_plan_with_source_factory(
                phase_index,
                phase,
                phase_seamless_to_next(&sliced_phases, phase_index),
                &shared.dataset,
                &shared.primary_model,
                shared.default_output_tokens,
                shared.dataset_rng_root,
                shared.rng_root,
                &source_factory,
                shared.tokenizer.clone(),
                shared.input_token_counter.clone(),
                clock.clone(),
                start_ns,
                &shared.benchmark_id,
                &shared.artifact_dir,
                &shared.endpoint_urls,
                &shared_resources,
                shared
                    .wants_adaptive_record
                    .then(|| capture.clone() as Rc<dyn AdaptiveTerminalRecordSource>),
                shared.on_failure,
            )?;
            let record_processor: Rc<dyn TurnRecordProcessor> = Rc::new(CapturePhaseProcessor {
                capture: capture.clone(),
                phase: metrics_phase(phase)?,
                has_credit_timestamp: !matches!(phase, PhaseSpec::FixedSchedule { .. }),
                // Once-per-cell on the main thread; a worker never feeds them.
                live_sink: None,
                heartbeat: None,
            });
            let mut record_processors = vec![record_processor];
            // Static-accuracy captures its terminal responses on the profiling phase
            // only; each shard feeds its own
            // processor, drained into the outcome after the run.
            if phase.common().name == "profiling"
                && let Some(accuracy_processor) = &shard_accuracy_processor
            {
                record_processors.push(accuracy_processor.clone() as Rc<dyn TurnRecordProcessor>);
            }
            plan = plan
                .with_record_processors(record_processors)
                .with_performance_record_capture(false)
                .with_native_metric_record_dimensions(false);
            plans.push(plan);
        }

        // No live/heartbeat phase observers on a worker thread.
        let observer: Rc<dyn PhaseObserver> = Rc::new(NoopPhaseObserver);
        let phased = run_scheduled_phases(plans, clock.clone(), dispatcher, observer).await?;
        phased
            .reports
            .iter()
            .find(|report| report.kind == PhaseKind::Profiling)
            .ok_or_else(|| {
                anyhow!("sharded scheduled worker completed without a profiling report")
            })?;
        Ok(phased)
    }
    .await;

    // Fold-and-drop modes (sketch or exact-fold) already folded every completed
    // record on the fly (the worker moved each out of its observer as it streamed), so
    // there is nothing left to drain and materializing that Vec would reintroduce the
    // O(records) peak. Only the retain path drains.
    let drained = if execution_result.is_ok() && !capture.folds_records() {
        execution_backend.drain_records(clock.now_ns())
    } else {
        Ok(Vec::new())
    };
    let shutdown = execution_backend.shutdown();
    let phased = finish_with_shutdown(execution_result, shutdown, "sharded execution backend")?;
    let drained = drained?;
    let issued_times = phased
        .reports
        .iter()
        .flat_map(|report| report.report.turns.iter())
        .map(|turn| (turn.uuid, turn.issued_offset_ns))
        .collect::<HashMap<_, _>>();
    // A fold-and-drop mode folded each completed record into this shard's own bounded
    // accumulator as it streamed and dropped it (only errored records retained): sketch
    // keeps a t-digest partition, exact-fold keeps a dense EXACT accumulator
    // whose rows sit at their LOCAL-dense fold ordinals. The retain path keeps the full
    // record Vec. Shards merge accumulator-to-accumulator (`append_store` concatenates
    // the dense exact stores) downstream, so the coordinator never holds O(all records)
    // in either fold mode.
    let records = if capture.folds_records() {
        // this shard streamed each completed record's rows into its per-shard
        // artifact lane at completion; flush and close it now that every record has
        // folded. A no-op when no lane is attached (sketch, or no lane artifact). The
        // coordinator concatenates the per-shard files into the final artifact.
        capture.finish_record_lane()?;
        let (accumulator, errored) = capture.take_streamed();
        ShardRecords::Folded {
            accumulator,
            errored,
        }
    } else {
        ShardRecords::Retained(capture.finish(&issued_times, drained)?)
    };
    let input_sessions = capture.take_input_sessions();
    // Drain this shard's static-accuracy captures (empty for a non-accuracy run);
    // the coordinator concatenates every shard's captures and grades once.
    let accuracy_captures = shard_accuracy_processor
        .map(|processor| processor.take_captures())
        .unwrap_or_default();
    let was_cancelled = phased.phases.iter().any(|phase| phase.was_cancelled);
    let has_warmup = phased
        .reports
        .iter()
        .any(|report| report.kind == PhaseKind::Warmup);
    Ok(ScheduledShardOutcome {
        records,
        input_sessions,
        accuracy_captures,
        was_cancelled,
        has_warmup,
    })
}

/// Compose the phase-scoped side-channel sidecars for one scheduled/graph phase:
/// server metrics run every phase; GPU telemetry and network-latency calibration
/// run only during profiling. Shared verbatim by the graph phase-plan builder and
/// the single-thread scheduled arm of [`execute_native_inner`]. The sharded arm
/// builds a profiling-only set inline (it has no per-phase loop) and the per-shard
/// worker never touches a sidecar, so those two paths do not use this helper.
fn compose_phase_sidecars(
    phase: &PhaseSpec,
    sidecars: &PreparedNativeSidecarResources,
) -> Result<Vec<Rc<dyn ScheduledPhaseSidecar>>> {
    let mut phase_sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>> = Vec::new();
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
    Ok(phase_sidecars)
}

async fn execute_native_inner(
    request: NativeRunSpec,
    mut accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    registry: &AIPerfRegistry,
) -> Result<NativeReport> {
    let live_sink = sidecars.live_sink();
    let rng_root = RngRoot::new(request.random_seed);
    if request.dataset.is_graph() {
        bail!("scheduled execution received a direct graph dataset plan");
    }
    let dataset_rng_root = match &request.dataset {
        NativeDatasetPlan::PreparedLinear(dataset) => dataset
            .random_seed
            .map_or(rng_root, |seed| RngRoot::new(Some(seed))),
        NativeDatasetPlan::StaticAccuracy(_) => rng_root,
        NativeDatasetPlan::Graph(_) => unreachable!("graph rejected above"),
    };
    let metrics_config =
        metrics_config(&request.metrics, request.endpoint.use_server_token_count())?;
    let model_names = request
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
        None => load_tokenizer(Some(&request.tokenizer.name))?,
    };
    let input_token_counter =
        select_input_token_counter(tokenizer.clone(), request.tokenizer.apply_chat_template);
    let (endpoint_urls, transport_config, prepared_endpoints, source_factory): NativeEndpointExecutionParts<'_> = {
        let NativeEndpointPlan::Prepared(profiles) = &request.endpoint;
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
                // Tag content-server media URLs so served fetches correlate back
                // to the request; only when the server publishes files.
                content_server_base: request
                    .sidecars
                    .content_server()?
                    .filter(|spec| spec.content_dir.is_some())
                    .map(|spec| Arc::from(spec.base_url())),
            },
            Some(table_factory),
            Box::new(PreparedNativeConversationSourceFactory {
                endpoint_resolver,
                samplers: registry.samplers(),
                // Coordinator and cell-entry paths read the process-global partition.
                cell_partition: None,
            }),
        )
    };
    let dataset = if let Some(accuracy) = accuracy.as_ref() {
        accuracy.dataset.dataset().as_ref().clone()
    } else {
        match &request.dataset {
            NativeDatasetPlan::PreparedLinear(dataset) => dataset.dataset.clone(),
            NativeDatasetPlan::StaticAccuracy(_) => {
                bail!("evaluator dataset plan requires an accuracy evaluator")
            }
            NativeDatasetPlan::Graph(_) => {
                unreachable!("graph rejected above")
            }
        }
    };
    let default_output_tokens = if accuracy.is_some() {
        dataset_default_output_tokens(&dataset)?
    } else {
        match &request.dataset {
            NativeDatasetPlan::PreparedLinear(dataset) => dataset.default_output_tokens,
            NativeDatasetPlan::StaticAccuracy(_) => {
                unreachable!("evaluator without accuracy rejected above")
            }
            NativeDatasetPlan::Graph(_) => unreachable!("graph rejected above"),
        }
    };

    let real_clock_anchor = sidecars.real_clock_anchor;
    let clock = sidecars.clock.clone();
    // An adaptive phase needs each completed turn's finished worker record fed
    // into its window sampler; the online dispatcher records per-token facts
    // worker-locally, so the coordinator sampler is otherwise starved.
    let wants_adaptive_record = request
        .phases
        .iter()
        .any(|phase| phase.common().adaptive_scale.is_some());

    // `workers == 1` uses the coordinator reactor. Multiple workers partition every
    // scheduled phase shape: request-bounded phases by budget and trace-driven
    // `user_centric`/`fixed_schedule` phases per conversation — INCLUDING static
    // accuracy: its per-record capture is pure `Send` data (a `problem_id` lookup
    // pushing a `CapturedResponse`), so each shard owns a capture processor over the
    // shared read-only associations and the disjoint captures concatenate at the
    // coordinator, which grades once on the main thread (the `!Send` Python evaluator
    // never crosses the spawn boundary). Co-located transports do not cross a
    // per-request thread boundary.
    // Sketch storage mode streams each record into a bounded accumulator and drops
    // it. The thread-per-core sharded path folds per shard — each sub-cell owns its
    // own sketch accumulator, merged accumulator-to-accumulator at the join — so a
    // sketch run shards exactly like an exact run and coordinator memory stays
    // O(shards × sketch) rather than O(records). `sketch_mode` still gates the
    // finalize (fold vs ingest) and the cellular record-shipping guard below.
    let sketch_mode = matches!(
        metrics_config.storage_mode,
        crate::metrics_core::MetricsStorageMode::Sketch { .. }
    );
    // Every scheduled phase shape is shardable: rate-based phases partition the
    // request budget (`slice_phase_for_thread`), and the trace-driven
    // `user_centric`/`fixed_schedule` phases partition per conversation (each sub-cell
    // owns a disjoint conversation subset via the injected two-level partition — the
    // enumeration filter in `NativeDatasetConversationSource` plus the partitioned
    // sampler). Static accuracy shards too: the `!Send` evaluator/grader stays on the
    // main thread, but the per-record CAPTURE is pure `Send` data, so each shard owns
    // a capture `AccuracyRecordProcessor` over the shared associations and the
    // disjoint captures concatenate at the coordinator for a single main-thread grade.
    let shardable = request.workers > 1;
    // Both arms converge on the same `captured` records + phase facts the
    // once-per-cell report tail below folds, so that tail stays written exactly once.
    // The accumulator that tail exports is created once here and populated inside the
    // branch that runs: the single-thread finalize folds/ingests into it directly (so
    // sketch mode drops records as it goes), the sharded arm ingests its merged shard
    // records. Created before the branch so both arms share the one instance.
    let mut accumulator = MetricsAccumulator::with_config(metrics_config.clone());
    // Exact-fold folds each completed record into the exact accumulator and
    // drops the heavy per-record data mid-run, but only on the single-thread
    // `DirectIssuanceAuthority` path with no per-record artifacts. Computed once here
    // like `sketch_mode`: the single-thread arm reads it to build the capture and
    // pick the finalize, the sharded arm's gate rejects it (`shardable`), and the
    // cellular-shipping guard reads it below. Heartbeat presence is probed from the env
    // rather than the (file-truncating) lane so this stays a pre-branch decision.
    // inputs.json is generated up front from the resident dataset unless the
    // dataset is a live-reply multi-turn shape, or a fixed-schedule phase filters the
    // dispatched conversations to a first-turn window (an up-front full-dataset pass
    // would then over-include). Either case keeps inputs.json on the during-run capture
    // path, which still disqualifies exact-fold.
    let inputs_up_front_ok = dataset_supports_up_front_inputs(&dataset)
        && !request
            .phases
            .iter()
            .any(|phase| matches!(phase, PhaseSpec::FixedSchedule { .. }));
    let inputs_need_retain = request.artifacts.inputs_path.is_some() && !inputs_up_front_ok;
    let exact_fold = exact_fold_enabled_by_env()
        && exact_fold_eligible(ExactFoldInputs {
            sketch_mode,
            shardable,
            is_cellular: ModuloCellPartition::from_env().is_some(),
            has_accuracy: accuracy.is_some(),
            wants_adaptive_record,
            has_live_sink: live_sink.is_some(),
            has_heartbeat: HeartbeatLane::enabled_by_env(),
            wants_per_record_artifacts: wants_per_record_artifacts(
                &request.artifacts,
                inputs_need_retain,
            ),
        });
    // Expose the selected memory path for operational diagnostics. Artifact bytes do
    // not depend on this choice.
    tracing::info!(
        exact_fold,
        shardable,
        sketch_mode,
        "record retention path selected"
    );
    // Per-record OTLP folded at completion by the exact-fold capture; the
    // retain/sharded arms leave this `None` and fold their retained records post-run.
    let mut folded_otel: Option<OtelRecordAccumulator> = None;
    // Static-accuracy terminal captures, collected by whichever arm runs: the
    // single-thread arm drains its one processor; the sharded arm concatenates the
    // per-shard captures. Graded once at finalize (order-independent by problem id).
    // Empty for a non-accuracy run.
    let mut accuracy_captures: Vec<CapturedResponse> = Vec::new();
    let (captured, input_sessions, was_cancelled, has_warmup, start_ns) = if !shardable {
        // The `!shardable` branch is reached only for `workers == 1` (any dataset,
        // including static accuracy). All `workers > 1` runs — accuracy included —
        // shard above the transport, so this branch's transport is always co-located
        // on the coordinator reactor; there is no per-request cross-thread transport
        // hop.
        let execution_backend = transport_factory.build(ExecutionBackendConfig {
            workers: 1,
            coordinator_clock: clock.clone(),
            real_clock_anchor,
            base_urls: endpoint_urls.clone(),
            model: primary_model.clone(),
            transport: transport_config,
            prepared_endpoints,
        })?;
        let start_ns = crate::engine::cell_origin::run_origin_now_ns(&clock);
        // Env-gated single-process cellular heartbeat lane; the controller merges
        // the same accumulator/t-digest across cells. It consumes the per-record
        // live clone, so it forces record capture on even without the Python sink.
        let heartbeat_lane = HeartbeatLane::from_env(clock.clone(), start_ns)?;
        // Streaming per-record artifact lane: only the exact-fold path, which
        // drops each record mid-run, needs it — the retain path still uses the batch
        // writers in the tail. Built here so it truncates its files before dispatch;
        // returns `None` when exact-fold is off or no records/raw/CSV artifact is
        // requested. Its parent dirs are created eagerly (before `create_run_artifacts`
        // in the async block), matching the batch writers' own dir creation.
        let record_lane = if exact_fold {
            RecordArtifactLane::new(
                request
                    .artifacts
                    .records_path
                    .as_ref()
                    .map(|path| artifact_path(&request.artifact_dir, path, "records_path"))
                    .transpose()?,
                request
                    .artifacts
                    .raw_path
                    .as_ref()
                    .map(|path| artifact_path(&request.artifact_dir, path, "raw_path"))
                    .transpose()?,
                request
                    .artifacts
                    .records_csv_path
                    .as_ref()
                    .map(|path| artifact_path(&request.artifact_dir, path, "records_csv_path"))
                    .transpose()?,
                request
                    .artifacts
                    .records_parquet_path
                    .as_ref()
                    .map(|path| artifact_path(&request.artifact_dir, path, "records_parquet_path"))
                    .transpose()?,
                // outputs.json streams through the lane at completion.
                request
                    .artifacts
                    .outputs_path
                    .as_ref()
                    .map(|path| artifact_path(&request.artifact_dir, path, "outputs_path"))
                    .transpose()?,
                request.artifacts.trace,
            )?
        } else {
            None
        };
        // Under exact-fold, inputs.json is generated up front from the resident dataset
        // and the during-run capture is disabled; the retain path keeps the
        // during-run capture. outputs.json streams through the lane, so exact-fold folds
        // and drops the model output text at completion rather than retaining it.
        let capture = Rc::new(
            RunCapture::new(
                clock.clone(),
                start_ns,
                metrics_config.clone(),
                request.artifacts.raw_path.is_some(),
                request.artifacts.inputs_path.is_some() && !exact_fold,
                live_sink.is_some() || heartbeat_lane.is_some(),
                wants_adaptive_record,
                exact_fold,
            )
            .with_record_lane(record_lane)
            // Per-record OTLP folds at completion only on the exact-fold
            // path; the retain path still folds the retained records post-run.
            .with_otel(exact_fold && request.native_otel_enabled)
            // Stage each turn's model output text so the streaming outputs.json entry
            // carries it, then drop it in the fold (exact-fold + outputs.json only).
            .with_outputs_capture(exact_fold && request.artifacts.outputs_path.is_some()),
        );
        let execution_result = async {
            execution_backend.set_run_origin(start_ns)?;
            // Build one worker-local observer per execution worker from the single
            // resolved metrics configuration; token accumulation moves off the
            // coordinator onto each worker's core.
            execution_backend.configure_measurement(metrics_config.clone(), start_ns)?;
            let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(ConfiguredDispatcher {
                execution_backend: execution_backend.clone(),
                model: primary_model.clone(),
                capture: capture.clone(),
            });

            let shared_resources = native_scheduled_resources(&request.phases);
            let on_failure = OnFailure::scheduled_or_default(request.failure_policy);
            // User-centric and fixed-schedule workloads reject `abort` rather than
            // silently ignoring it.
            if on_failure.is_abort()
                && request.phases.iter().any(|phase| {
                    matches!(
                        phase,
                        PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. }
                    )
                })
            {
                tracing::warn!(
                    "failure_policy=abort is honored only for request-rate/concurrency scheduled \
                 phases; user_centric and fixed_schedule phases in this run stay resilient"
                );
            }

            let mut plans = Vec::with_capacity(request.phases.len());
            for (phase_index, phase) in request.phases.iter().enumerate() {
                let mut plan = build_native_scheduled_phase_plan_with_source_factory(
                    phase_index,
                    phase,
                    phase_seamless_to_next(&request.phases, phase_index),
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
                    &request.benchmark_id,
                    &request.artifact_dir,
                    &endpoint_urls,
                    &shared_resources,
                    wants_adaptive_record
                        .then(|| capture.clone() as Rc<dyn AdaptiveTerminalRecordSource>),
                    on_failure,
                )?;
                let record_processor: Rc<dyn TurnRecordProcessor> =
                    Rc::new(CapturePhaseProcessor {
                        capture: capture.clone(),
                        phase: metrics_phase(phase)?,
                        has_credit_timestamp: !matches!(phase, PhaseSpec::FixedSchedule { .. }),
                        live_sink: live_sink.clone(),
                        heartbeat: heartbeat_lane.clone(),
                    });
                let mut record_processors = vec![record_processor];
                if phase.common().name == "profiling"
                    && let Some(accuracy) = accuracy.as_ref()
                {
                    let processor: Rc<dyn TurnRecordProcessor> = accuracy.processor.clone();
                    record_processors.push(processor);
                }
                plan = plan.with_record_processors(record_processors);
                let phase_sidecars = compose_phase_sidecars(phase, sidecars)?;
                if !phase_sidecars.is_empty() {
                    plan = plan.with_sidecars(phase_sidecars);
                }
                // The coordinator's per-run `CollectorObserver`/`NativeMetricsObserver`
                // retention is dead work on the runner path: the native-v2 report is
                // rebuilt from the drained per-worker records, and the only value the
                // coordinator report supplies is per-turn `issued_offset_ns`, which
                // comes from the runtime's own `DetailedSchedule` (gated separately by
                // timing-record capture), not from these observers. Drop the discarded
                // full-record retention and keep the coordinator native metrics
                // aggregate-only so we do not accumulate a per-request record graph
                // that is never read.
                plan = plan
                    .with_performance_record_capture(false)
                    .with_native_metric_record_dimensions(false);
                plans.push(plan);
            }

            create_run_artifacts(&request)?;
            sidecars.activate_live_streaming().await;

            let mut observers: Vec<Rc<dyn PhaseObserver>> = Vec::new();
            if let Some(sink) = live_sink {
                observers.push(live_phase_observer(sink, clock.clone()));
            }
            if let Some(lane) = &heartbeat_lane {
                observers.push(Rc::new(HeartbeatPhaseObserver::new(lane.clone())));
            }
            let observer: Rc<dyn PhaseObserver> = if observers.is_empty() {
                Rc::new(NoopPhaseObserver)
            } else {
                CompositePhaseObserver::compose(observers)
            };
            let phased = run_scheduled_phases(plans, clock.clone(), dispatcher, observer).await?;
            phased
                .reports
                .iter()
                .find(|report| report.kind == PhaseKind::Profiling)
                .ok_or_else(|| anyhow!("phase runtime completed without a profiling report"))?;
            Ok(phased)
        }
        .await;
        // Drain each worker observer's records before shutting the workers down; on
        // the failure path the report is discarded, so an empty drain is fine.
        // Fold-and-drop modes (sketch or exact-fold) already folded and dropped every
        // completed record as it streamed (the worker moved each out of its observer),
        // so skip the drain — materializing that Vec would reintroduce the O(records)
        // peak.
        let folds_records = sketch_mode || exact_fold;
        let drained = if execution_result.is_ok() && !folds_records {
            execution_backend.drain_records(capture.clock.now_ns())
        } else {
            Ok(Vec::new())
        };
        let shutdown = execution_backend.shutdown();
        let phased = finish_with_shutdown(execution_result, shutdown, "execution backend")?;
        let drained = drained?;
        let issued_times = phased
            .reports
            .iter()
            .flat_map(|report| report.report.turns.iter())
            .map(|turn| (turn.uuid, turn.issued_offset_ns))
            .collect::<HashMap<_, _>>();
        // Single-thread finalize: a fold-and-drop mode folded each record into the
        // capture's streaming accumulator as the run streamed and dropped it (only
        // errored records retained for error grouping); merge that into the report
        // `accumulator`. Sketch merges a bounded t-digest partition; exact-fold merges
        // a dense EXACT accumulator whose rows already sit at their absolute
        // `request_index` slots, so the merged report is byte-identical to the retain
        // path's dispatch-order re-ingest. The retain path keeps the full record Vec
        // and ingests it.
        let captured = if folds_records {
            // Exact-fold streamed each record's records/raw/CSV rows into the artifact
            // lane at completion; flush and close it now that every record has
            // folded. A no-op when no lane is attached (sketch, or no lane artifact).
            capture.finish_record_lane()?;
            // Exact-fold accumulates profiling-record OTLP histograms at completion;
            // take that accumulator for the report tail. It is absent in sketch mode
            // and when native OTLP is disabled.
            folded_otel = capture.take_otel();
            let (streamed, errored) = capture.take_streamed();
            accumulator
                .merge(&streamed)
                .map_err(|error| anyhow!("merging streamed fold-and-drop accumulator: {error}"))?;
            errored
        } else {
            let captured = capture.finish(&issued_times, drained)?;
            for record in &captured {
                accumulator.process_record(&record.ingest);
            }
            captured
        };
        // Exact-fold generates inputs.json up front from the resident dataset:
        // a pure export the benchmark never read, formatted through the same session
        // materializer dispatch uses, so it is byte-identical to the disabled during-run
        // capture. The retain path keeps the during-run capture (`take_input_sessions`).
        let input_sessions = if exact_fold && request.artifacts.inputs_path.is_some() {
            // Exact-fold emits a FULL-dataset inputs.json (every conversation in the
            // resident dataset), whereas the during-run capture records ONLY the conversations a run
            // actually dispatched. The two agree for a full-coverage run, but a
            // partial-coverage run (a request-bounded phase that stops before touching
            // every conversation) yields a strictly larger inputs.json under exact-fold.
            // Surface that cross-mode difference once rather than leaving it silent — it
            // runs exactly once per run (single-thread finalize).
            tracing::info!(
                "exact-fold inputs.json is generated up front from the full resident \
                 dataset (Python-aligned), not the dispatched-only capture; a \
                 partial-coverage run therefore lists every conversation, not just the \
                 dispatched subset"
            );
            build_up_front_input_sessions(
                &dataset,
                source_factory.as_ref(),
                &primary_model,
                default_output_tokens,
                rng_root,
                tokenizer.clone(),
                input_token_counter.clone(),
            )?
        } else {
            capture.take_input_sessions()
        };
        let was_cancelled = phased.phases.iter().any(|phase| phase.was_cancelled);
        let has_warmup = phased
            .reports
            .iter()
            .any(|report| report.kind == PhaseKind::Warmup);
        // Drain the single-thread accuracy processor (the one registered on the
        // profiling phase above); `Vec::new` for a non-accuracy run. Graded at finalize.
        if let Some(accuracy) = accuracy.as_ref() {
            accuracy_captures = accuracy.processor.take_captures();
        }
        (
            captured,
            input_sessions,
            was_cancelled,
            has_warmup,
            start_ns,
        )
    } else {
        // `shardable` guarantees workers > 1. A static-accuracy run shards
        // too: each shard captures its own terminal responses over the shared
        // associations, concatenated into `accuracy_captures` for a single main-thread
        // grade.
        // `exact_fold` may be true here (a metrics-only sharded run selects it); each
        // worker capture then folds into its own exact accumulator with a LOCAL-dense
        // fold ordinal, and the coordinator merges those dense stores via `append_store`
        // below. Consumers requiring retained records keep `exact_fold` false.
        // Once-per-cell on the main thread, before the sub-cell threads spawn.
        create_run_artifacts(&request)?;
        sidecars.activate_live_streaming().await;
        if live_sink.is_some() {
            tracing::warn!(
                "live-streaming per-record updates are delivered only at end-of-run under \
                 thread-per-core scheduled execution (workers > 1); intermediate live progress is \
                 degraded"
            );
        }
        let start_ns = crate::engine::cell_origin::run_origin_now_ns(&clock);
        // The two-level grid: this process is cell `cell_id` of `cells`
        // (`AIPERF_CELL_ID`/`_COUNT`, default the lone `(0, 1)`), sub-divided into
        // `workers` thread-per-core sub-cells.
        let (cell_id, cells) = ModuloCellPartition::from_env()
            .map(|partition| (partition.cell_id(), partition.cell_count()))
            .unwrap_or((0, 1));
        // A controller child already carries the global phase ordinal bases in the
        // env; a lone process computes them from its (global == local) phase
        // budgets so profiling ordinals never collide with warmup's `[0, W)` block.
        let env_bases = crate::engine::cellular_cell::phase_ordinal_bases_from_env();
        let phase_ordinal_bases = if env_bases.is_empty() {
            crate::engine::sharded_scheduled::compute_phase_ordinal_bases(&request.phases)?
        } else {
            env_bases
        };
        // Sub-cell threads share a concrete prepared-endpoint table factory and each
        // derives its own `Rc` resolver.
        let table_factory = {
            let NativeEndpointPlan::Prepared(profiles) = &request.endpoint;
            Arc::new(NativePreparedEndpointTableFactory::new(
                registry.endpoints().clone(),
                profiles.clone(),
            ))
        };
        let shared = Arc::new(ShardedShared {
            transport_factory: transport_factory.clone(),
            table_factory,
            samplers: registry.samplers().clone(),
            dataset: dataset.clone(),
            primary_model: primary_model.clone(),
            metrics_config: metrics_config.clone(),
            tokenizer: tokenizer.clone(),
            input_token_counter: input_token_counter.clone(),
            endpoint_urls: endpoint_urls.clone(),
            transport_config: transport_config.clone(),
            default_output_tokens,
            dataset_rng_root,
            rng_root,
            phases: request.phases.clone(),
            benchmark_id: request.benchmark_id.clone(),
            artifact_dir: request.artifact_dir.clone(),
            raw_enabled: request.artifacts.raw_path.is_some(),
            inputs_enabled: request.artifacts.inputs_path.is_some(),
            // per-shard artifact lanes: the shards stream these when exact-fold
            // is selected; the coordinator concatenates them at finalize.
            records_path: request.artifacts.records_path.clone(),
            raw_path: request.artifacts.raw_path.clone(),
            records_csv_path: request.artifacts.records_csv_path.clone(),
            records_parquet_path: request.artifacts.records_parquet_path.clone(),
            outputs_path: request.artifacts.outputs_path.clone(),
            include_trace: request.artifacts.trace,
            wants_adaptive_record,
            // Static-accuracy associations, shared read-only across the shards; each
            // builds its own capture processor over them.
            accuracy_associations: accuracy
                .as_ref()
                .map(|accuracy| accuracy.dataset.associations()),
            exact_fold,
            on_failure: OnFailure::scheduled_or_default(request.failure_policy),
            real_clock_anchor,
            start_ns,
            cell_id,
            cells,
            workers: request.workers as u32,
            phase_ordinal_bases,
        });
        // Build the once-per-cell profiling-phase side-channel sidecars on the main
        // thread; the sharded runtime drives them over the run window while the
        // sub-cell threads execute (a worker thread never scrapes telemetry).
        let mut profiling_sidecars: Vec<Rc<dyn crate::phase_runtime::ScheduledPhaseSidecar>> =
            Vec::new();
        if let Some(server_metrics) = sidecars.server_metrics.as_ref() {
            profiling_sidecars.push(server_metrics.sidecar(MetricsPhase::Profiling));
        }
        if let Some(gpu_telemetry) = sidecars.gpu_telemetry.as_ref() {
            profiling_sidecars.push(gpu_telemetry.sidecar());
        }
        if let Some(network_latency) = sidecars.network_latency.as_ref()
            && let Some(sidecar) = network_latency.sidecar()
        {
            profiling_sidecars.push(sidecar);
        }
        let outcome = crate::engine::sharded_scheduled::run_sharded_scheduled(
            shared,
            profiling_sidecars,
            clock.clone(),
        )
        .await?;
        // Concatenate the per-shard static-accuracy captures (empty for a non-accuracy
        // run); graded once at finalize. Moved out before the record match below.
        accuracy_captures = outcome.accuracy_captures;
        // Retain-path shards return the full record Vec to ingest into the report
        // accumulator; fold-and-drop shards (sketch or exact-fold) already
        // folded into per-shard accumulators that merge_shards combined via
        // `append_store`, so merge that into the report accumulator and keep only the
        // errored records for error grouping. Peak coordinator memory is bounded by the
        // mode: O(records) on the retain path (per-record artifacts need them),
        // O(shards × accumulator) for both fold modes.
        let captured = match outcome.records {
            ShardRecords::Retained(records) => {
                for record in &records {
                    accumulator.process_record(&record.ingest);
                }
                records
            }
            ShardRecords::Folded {
                accumulator: shard_accumulator,
                errored,
            } => {
                accumulator.merge(&shard_accumulator).map_err(|error| {
                    anyhow!(
                        "merging sharded fold-and-drop partition into the report \
                         accumulator: {error}"
                    )
                })?;
                errored
            }
        };
        // an exact-fold sharded run streamed each shard's records/raw/CSV/
        // parquet/outputs into per-shard temp files; fuse them into the single final
        // artifact now that every shard has finished, and remove the temp dirs. The
        // finalize tail below skips the batch writers under `exact_fold`, so this is the
        // sole writer of those artifacts on the sharded exact-fold path.
        let input_sessions = if exact_fold {
            crate::engine::shard_artifacts::concatenate_shard_artifacts(
                &request.artifact_dir,
                &request.artifacts,
                request.workers,
            )?;
            // inputs.json is generated once at the coordinator from the resident
            // dataset; shards disable their during-run capture. Absent an
            // `inputs.json` request this yields an unused empty list.
            if request.artifacts.inputs_path.is_some() {
                build_up_front_input_sessions(
                    &dataset,
                    source_factory.as_ref(),
                    &primary_model,
                    default_output_tokens,
                    rng_root,
                    tokenizer.clone(),
                    input_token_counter.clone(),
                )?
            } else {
                outcome.input_sessions
            }
        } else {
            outcome.input_sessions
        };
        (
            captured,
            input_sessions,
            outcome.was_cancelled,
            outcome.has_warmup,
            start_ns,
        )
    };
    // A cell ships its terminal partition to the controller. Absent the controller
    // address (the single-process path) this is inert. Two shapes, by mode:
    // - RETAIN: ship the full captured record `Vec` — each record carries the dense
    //   global dispatch ordinal the autonomous issuer stamped — which the controller
    //   merges in global order into the single authoritative report (byte-exact).
    // - FOLD-AND-DROP (exact-fold OR sketch): the cell folded every record into
    //   `accumulator` and dropped it (`captured` holds only the retained errored
    //   records), so there is no full record `Vec` — ship the folded STORE instead. The
    //   controller appends every cell's store (`merge_store_partitions`), which for a
    //   sketch store merges the per-`(phase, tag)` t-digests associatively (the store
    //   retains no rows) and for an exact-fold store appends the dense rows — both a
    //   within-tolerance summary, the same bar as the in-process sharded merge.
    //   Counters: `errored` is the retained errored subset; `issued` is the folded
    //   record total. For exact-fold that is the store's `record_count()`, but a sketch
    //   store clears each row after harvesting (`record_count() == 0`), so the surviving
    //   `ingested_count()` is the true total. `completed = issued - errored`.
    #[cfg(feature = "cellular")]
    if let Some(shipper) = crate::engine::cellular_cell::CellRecordsShipper::from_env() {
        let epoch_ns: i64 = clock.now_ns().saturating_sub(start_ns);
        if exact_fold || sketch_mode {
            let issued = if sketch_mode {
                accumulator.ingested_count()
            } else {
                accumulator.record_count() as u64
            };
            let errored = captured.len() as u64;
            let counters = crate::cellular::HeartbeatCounters {
                issued,
                completed: issued.saturating_sub(errored),
                errored,
            };
            shipper.ship_store(accumulator.column_store().clone(), counters, epoch_ns)?;
        } else {
            let records: Vec<RecordIngest> = captured
                .iter()
                .map(|record| record.ingest.clone())
                .collect();
            shipper.ship_records(records, epoch_ns)?;
        }
    }
    let gpu_telemetry = sidecars.gpu_telemetry.as_ref();
    let network_latency = sidecars.network_latency.as_ref();
    let server_metrics = sidecars.server_metrics.as_ref();
    let RunMetricsSummaries {
        profiling_metrics,
        profiling_server_summary,
        warmup,
        warmup_server_summary,
    } = summarize_run_metrics(
        &mut accumulator,
        gpu_telemetry,
        network_latency,
        server_metrics,
        &request,
        &metrics_config,
        has_warmup,
    );
    // The exact-fold path already streamed records.jsonl / raw.jsonl / the per-record
    // CSV AND the columnar Parquet sidecar row-by-row through the artifact lane (and
    // flushed it at the finalize), so `captured` here holds only the retained errored
    // records — running the batch writers over it would truncate the streamed files to
    // the errored subset. Skip all four streamed batch writers under exact-fold; the
    // retain path (and every non-folding mode) still writes them here. outputs.json
    // also streamed through the lane under exact-fold, so its batch writer is
    // skipped too — `captured` holds only the retained errored records. inputs.json is
    // NOT skipped: under exact-fold `input_sessions` holds the up-front, dataset-derived
    // export, so the writer below emits the same bytes the capture path would.
    if !exact_fold {
        if let Some(records_path) = &request.artifacts.records_path {
            let records_path = artifact_path(&request.artifact_dir, records_path, "records_path")?;
            write_records_jsonl(
                &records_path,
                &captured,
                &metrics_config,
                request.artifacts.trace,
            )?;
        }
        write_records_csv_artifact(&request, &captured, &metrics_config)?;
        if let Some(raw_path) = &request.artifacts.raw_path {
            let raw_path = artifact_path(&request.artifact_dir, raw_path, "raw_path")?;
            write_raw_records_jsonl(&raw_path, &captured)?;
        }
        write_records_parquet_artifact(&request, &captured, &metrics_config)?;
        if let Some(outputs_path) = &request.artifacts.outputs_path {
            let outputs_path = artifact_path(&request.artifact_dir, outputs_path, "outputs_path")?;
            write_outputs_json(&outputs_path, &captured, &metrics_config)?;
        }
        // Dataset analysis (`--dry-run`) reads the full retained record set. Requesting
        // it forces the retain path via `wants_per_record_artifacts`, so `exact_fold` is
        // disabled and `captured` holds every clean + errored record here. Sketch mode
        // folds and drops records, leaving only the errored subset, so skip it there
        // (dry-run defaults to exact; T11 gating rejects the sketch combination).
        if !sketch_mode {
            if let Some(relative) = &request.artifacts.dataset_analysis_path {
                let base = artifact_path(&request.artifact_dir, relative, "dataset_analysis_path")?;
                let analysis_request =
                    crate::engine::dataset_analysis_writer::DatasetAnalysisRequest {
                        path: base,
                        options: crate::dataset::analysis::AnalysisOptions {
                            block_size: request.artifacts.dataset_analysis_block_size.unwrap_or(16),
                            explicit_cache_blocks: request.artifacts.dataset_analysis_cache_blocks,
                        },
                        per_conversation: request.artifacts.dataset_analysis_per_conversation,
                    };
                crate::engine::dataset_analysis_writer::write_dataset_analysis_from_records(
                    &analysis_request,
                    &captured,
                )?;
            }
        }
    }
    if let Some(inputs_path) = &request.artifacts.inputs_path {
        let inputs_path = artifact_path(&request.artifact_dir, inputs_path, "inputs_path")?;
        write_inputs_json(&inputs_path, &input_sessions)?;
    }
    // (cross-host k8s cell): all per-record artifacts (+ inputs.json) are now
    // on this cell's own filesystem; ship them to the controller's HTTP upload server
    // with streaming zstd. A no-op on the same-host launcher (concatenates the
    // local writes) or the single-process path.
    #[cfg(feature = "cellular")]
    crate::engine::cellular_cell::ship_http_artifacts_if_enabled(
        &request.artifact_dir,
        &request.artifacts,
    )?;
    let server_metrics_report = write_sidecar_records(
        gpu_telemetry,
        network_latency,
        server_metrics,
        sidecars.gpu_records_path.as_deref(),
        sidecars.network_latency_records_path.as_deref(),
        sidecars.server_metrics_jsonl_path.as_deref(),
        sidecars.server_metrics_parquet_wire_path.as_deref(),
        profiling_server_summary.as_ref(),
        warmup_server_summary.as_ref(),
    )?;
    let media_metrics = sidecars.finalize_media_metrics().await.metrics;
    let mut outcome = RunOutcome {
        run: ReportRunInfo {
            mode: Some("online".into()),
            model: Some(primary_model),
        },
        summary: ReportSummary {
            endpoints_configured: endpoint_urls,
            server_metrics: server_metrics_report,
            // Propagate external cancellation alongside partial results. Sharded
            // runs OR this across every sub-cell thread.
            was_cancelled,
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
        media_metrics,
        errors: group_record_errors(&captured),
        ..RunOutcome::default()
    };
    if let Some(accuracy) = accuracy.as_mut() {
        // Grade the collected captures once on the main thread: the single-thread
        // arm's drained processor OR the concatenated per-shard captures. Grading is
        // keyed by problem id, so the merged (order-independent) set gives the same
        // tally regardless of worker count.
        let evaluation = grade_accuracy_captures(
            std::mem::take(&mut accuracy_captures),
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
    let mut report = NativeReport::from_outcome(&profiling_metrics, &outcome);
    // Per-record OTLP histograms: accumulate the same profiling-record metric
    // projection the aggregate report is built from (and the live-streaming sink
    // forwards to Python) so the post-report OTLP sink emits populated
    // `bucket_counts` instead of zeros. Gated on the native OTLP sink being
    // enabled so no per-record recompute happens otherwise. Merged here (the
    // scheduled online path runs one current-thread worker set that already
    // joins its per-worker records into `captured`).
    if request.native_otel_enabled {
        // Exact-fold already folded each profiling record at completion;
        // reuse that order-independent accumulator. The retain/sharded arms retain the
        // full record set, so fold it here. Both produce identical histograms.
        let otel_records = match folded_otel.take() {
            Some(folded) => folded,
            None => {
                let mut otel_records = OtelRecordAccumulator::new();
                for record in &captured {
                    if record.ingest.phase == MetricsPhase::Profiling {
                        observe_otel_record(&mut otel_records, record, &metrics_config);
                    }
                }
                otel_records
            }
        };
        if !otel_records.is_empty() {
            report.otel_per_record = Some(otel_records);
        }
    }
    Ok(report)
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

/// Prepared conversation-source construction behind the shared scheduled workload.
/// Both protocol endpoint bindings implement this seam without branching inside
/// phase or scheduler policy.
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
    Option<Arc<dyn PreparedEndpointTableFactory>>,
    Box<dyn NativeConversationSourceFactory + 'a>,
);

struct PreparedNativeConversationSourceFactory<'a> {
    endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    samplers: &'a crate::dataset::SamplerRegistry,
    /// The dataset instance partition this source draws. `None` reads the
    /// process-global partition; thread-per-core execution injects a per-thread
    /// partition that `AIPERF_CELL_ID`/`_COUNT` cannot express.
    cell_partition: Option<ModuloCellPartition>,
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
            NativeDatasetConversationSource::sequential_with_prepared_resolver_for_partition(
                dataset,
                model,
                default_output_tokens,
                self.endpoint_resolver.clone(),
                self.cell_partition,
            )?
        } else {
            NativeDatasetConversationSource::preferred_with_prepared_resolver_for_partition(
                dataset,
                model,
                default_output_tokens,
                rng_root,
                self.samplers,
                self.endpoint_resolver.clone(),
                self.cell_partition,
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
    adaptive_record_source: Option<Rc<dyn AdaptiveTerminalRecordSource>>,
    on_failure: OnFailure,
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
            let workload = Rc::new(
                RequestRateWorkload::with_components(
                    source,
                    intervals.clone(),
                    shared.session.clone(),
                    shared.prefill.clone(),
                )?
                .with_failure_policy(on_failure),
            ) as Rc<dyn Workload>;
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
                crate::timing::ArrivalPattern::ConcurrencyBurst,
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
                crate::timing::ArrivalPattern::ConcurrencyBurst,
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
                Rc::new(crate::phase_runtime::NoopScheduledPhaseResources),
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
        adaptive_record_source,
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

pub(crate) async fn build_synthetic_dataset(
    registry: &AIPerfRegistry,
    spec: &SyntheticDatasetSpec,
    models: &ModelsSpec,
    rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
    rankings: bool,
    media_generator_factory: Arc<dyn SyntheticMediaGeneratorFactory>,
    requires_raw_token_ids: bool,
) -> Result<Dataset> {
    let mut compose = compose_config(models, rng_root)?;
    compose.media_generator_factory = media_generator_factory;
    compose.requires_raw_token_ids = requires_raw_token_ids;
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
    registry: &AIPerfRegistry,
    spec: &FileDatasetSpec,
    models: &ModelsSpec,
    run_rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
    trace_prompt_storage: Arc<dyn TracePromptStoragePolicy>,
    requires_raw_token_ids: bool,
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
    compose.requires_raw_token_ids = requires_raw_token_ids;
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
    // An empty format means `--custom-dataset-type` was not supplied; defer to
    // structural auto-detection. A non-empty
    // format is an explicit, honored loader selection.
    let explicit_format = (!spec.format.is_empty()).then_some(spec.format.as_str());
    registry
        .dataset_formats()
        .build_dataset(explicit_format, &load, &compose, tokenizer)
        .await
        .map_err(Into::into)
}

pub(crate) async fn build_public_dataset(
    registry: &AIPerfRegistry,
    spec: &PublicDatasetSpec,
    models: &ModelsSpec,
    rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
    requires_raw_token_ids: bool,
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
    compose.requires_raw_token_ids = requires_raw_token_ids;
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
        expected.is_finite() && expected > 0.0 && expected < i64::MAX as f64,
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

/// Resolve the optional metric-slice duration (seconds → i64 ns). `field` names the
/// duration in the representability error so scheduled (`"duration"`) and offline
/// (`"metrics slice duration"`) callers preserve their exact messages. Shared by the
/// scheduled and offline metrics-config builders.
pub(crate) fn resolve_slice_duration_ns(
    slice_duration_seconds: Option<f64>,
    field: &str,
) -> Result<Option<i64>> {
    slice_duration_seconds
        .map(|seconds| {
            ensure!(seconds > 0.0, "metrics slice duration must be positive");
            ensure!(
                seconds.is_finite()
                    && seconds >= 0.0
                    && seconds * 1_000_000_000.0 < i64::MAX as f64,
                "{field} must be finite, non-negative, and representable in nanoseconds"
            );
            Ok((seconds * 1_000_000_000.0).round_ties_even() as i64)
        })
        .transpose()
}

/// Resolve and validate the configured SLO thresholds against the native metric
/// catalog. Shared by the scheduled and offline metrics-config builders.
pub(crate) fn resolve_slos(spec: &MetricsSpec) -> Result<Vec<SloThreshold>> {
    let mut slos = Vec::with_capacity(spec.slos.len());
    for (name, value) in &spec.slos {
        ensure!(value.is_finite(), "SLO {name:?} threshold must be finite");
        let metric = CATALOG
            .iter()
            .find(|metric| metric.tag.as_str() == name)
            .ok_or_else(|| anyhow!("SLO metric {name:?} is not in the native metric catalog"))?;
        ensure!(
            metric.kind == crate::metrics_core::MetricType::Record
                && !metric
                    .flags
                    .contains(crate::metrics_core::MetricFlags::NO_INDIVIDUAL_RECORDS),
            "SLO metric {name:?} does not produce one value per request"
        );
        slos.push(SloThreshold::from_display(metric.tag, *value)?);
    }
    Ok(slos)
}

pub(crate) fn metrics_config(
    spec: &MetricsSpec,
    use_server_token_count: bool,
) -> Result<MetricsConfig> {
    let slice_duration_ns = resolve_slice_duration_ns(spec.slice_duration_seconds, "duration")?;
    let slos = resolve_slos(spec)?;
    let storage_mode = if spec.sketch {
        crate::metrics_core::MetricsStorageMode::Sketch {
            compression: crate::metrics_core::SKETCH_DEFAULT_COMPRESSION,
        }
    } else {
        crate::metrics_core::MetricsStorageMode::Exact
    };
    Ok(MetricsConfig {
        slice_duration_ns,
        slos,
        use_server_token_count,
        storage_mode,
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
// at the adapter seam.
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
            crate::timing::Phase::Warmup
        } else {
            crate::timing::Phase::Profiling
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

/// Push a session-concurrency ramp driver that paces `session_slots`' admission
/// limit from 1 up to the phase concurrency target. `admission_msg` names the
/// missing admission pool in the error (scheduled vs graph wording). Shared by the
/// scheduled and graph ramp controllers.
pub(crate) fn push_concurrency_ramp_driver(
    drivers: &mut Vec<RampDriver>,
    spec: &PhaseSpec,
    ramp: &RampSpec,
    clock: &Rc<dyn Clock>,
    session_slots: &Option<Rc<SlotPool>>,
    rng_root: RngRoot,
    admission_msg: &'static str,
) -> Result<()> {
    let target = spec
        .concurrency()
        .ok_or_else(|| anyhow!("concurrency_ramp requires a concurrency target"))?;
    let slots = session_slots
        .clone()
        .ok_or_else(|| anyhow!(admission_msg))?;
    let strategy = ramp_strategy(ramp, 1.0, target as f64, false, rng_root)?;
    drivers.push(RampDriver::new(clock.clone(), strategy, move |value| {
        slots.set_limit(value.round() as usize)
    }));
    Ok(())
}

/// Push a request-rate ramp driver that paces `intervals`' rate from a
/// duration-derived start value up to `target_rate`. Shared by the scheduled and
/// graph ramp controllers.
pub(crate) fn push_rate_ramp_driver(
    drivers: &mut Vec<RampDriver>,
    ramp: &RampSpec,
    clock: Rc<dyn Clock>,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    target_rate: Option<f64>,
    rng_root: RngRoot,
) -> Result<()> {
    let target = target_rate.ok_or_else(|| anyhow!("rate_ramp requires a rate phase"))?;
    let duration_ns = seconds_to_u64_ns(ramp.duration)?;
    let start = target * RATE_RAMP_UPDATE_INTERVAL_NS as f64 / duration_ns as f64;
    let strategy = ramp_strategy(ramp, start, target, true, rng_root)?;
    drivers.push(RampDriver::new(clock, strategy, move |value| {
        intervals.borrow_mut().set_rate(value)
    }));
    Ok(())
}

fn ramp_controller(
    spec: &PhaseSpec,
    clock: Rc<dyn Clock>,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
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
        push_concurrency_ramp_driver(
            &mut drivers,
            spec,
            ramp,
            &clock,
            &session_slots,
            rng_roots.concurrency(),
            "concurrency_ramp requires session admission",
        )?;
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
        push_rate_ramp_driver(
            &mut drivers,
            ramp,
            clock,
            intervals,
            target_rate,
            rng_roots.request_rate(),
        )?;
    }
    if drivers.is_empty() {
        Ok(Rc::new(crate::phase_runtime::NoopScheduledPhaseController))
    } else {
        Ok(Rc::new(RampScheduledPhaseController::new(drivers)))
    }
}

#[allow(clippy::too_many_arguments)]
fn adaptive_runtime_extension(
    phase: &PhaseSpec,
    benchmark_id: &str,
    artifact_dir: &Path,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    user_target: Option<Rc<dyn UserTarget>>,
    record_source: Option<Rc<dyn AdaptiveTerminalRecordSource>>,
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
        record_source,
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
        value.is_finite() && value >= 1.0 && value.fract() == 0.0 && value < usize::MAX as f64,
        "adaptive {label} must be an integer in the usize range"
    );
    Ok(value as usize)
}

struct AdaptiveRuntimeExtension {
    config: AdaptiveRunConfig,
    intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    user_target: Option<Rc<dyn UserTarget>>,
    session_target: Option<usize>,
    prefill_target: Option<usize>,
    /// Worker-record source feeding the sampler on the online path. `None` for
    /// dispatchers that feed the callback observer directly (offline), which
    /// keeps the sampler from being double-fed.
    record_source: Option<Rc<dyn AdaptiveTerminalRecordSource>>,
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
        // On the online path the dispatcher records token/usage/terminal facts
        // worker-locally and discards the callback observer's token feed, so the
        // sampler must be fed each completed turn's finished record explicitly,
        // exactly as the graph phase runtime does. Offline supplies no source and
        // keeps the callback-observer feed built into `built.observer`.
        let record_processors: Vec<Rc<dyn TurnRecordProcessor>> =
            if let Some(source) = self.record_source.clone() {
                vec![Rc::new(AdaptiveSamplerRecordProcessor {
                    source,
                    sampler: built.scale.sampler().clone(),
                })]
            } else {
                Vec::new()
            };
        let controller: Rc<dyn ScheduledPhaseController> = Rc::new(
            AdaptiveScheduledPhaseController::new(built.scale, controller),
        );
        Ok(ScheduledRuntimeExtensionParts {
            observer: built.observer,
            issuance_gate: Some(gate),
            controller,
            record_processors,
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

    fn stop(&self) -> crate::timing::LocalPhaseFuture<Result<()>> {
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

    fn wait_until_stop(&self) -> crate::timing::LocalPhaseFuture<()> {
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
    ) -> crate::dataset::Result<Box<dyn ModelSelector>> {
        if models.len() != self.weights.len() || models.is_empty() {
            return Err(crate::dataset::DatasetError::Validation(
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
            return Err(crate::dataset::DatasetError::Validation(
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

/// Select the input-token accounting policy for one native run.
///
/// AIPerf pre-tokenizes every dataset segment once at composition and stores the
/// exact per-segment token counts; the materializer sums them into each turn's
/// authored input length. When no chat template is applied, the wire body is
/// exactly those pre-tokenized segments, so the authored count is already exact
/// and re-encoding the assembled body on every request is pure redundant work —
/// the profiled online hot spot. Trust the pre-tokenized count verbatim
/// ([`AuthoredInputTokenCounter`]) so the benchmark loop stays tokenizer-free.
/// A chat template injects role/generation-prompt tokens composition did not
/// measure, so that case re-encodes the templated wire body per request.
fn select_input_token_counter(
    tokenizer: Arc<dyn TextTokenizer>,
    apply_chat_template: bool,
) -> Arc<dyn InputTokenCounter> {
    if apply_chat_template {
        Arc::new(EndpointInputTokenCounter::new(tokenizer, true))
    } else {
        Arc::new(AuthoredInputTokenCounter)
    }
}

fn seconds_to_ns(value: f64) -> Result<i64> {
    let nanos = seconds_to_u64_ns(value)?;
    i64::try_from(nanos).map_err(|_| anyhow!("duration is outside the i64 nanosecond range"))
}

pub(crate) fn seconds_to_u64_ns(value: f64) -> Result<u64> {
    ensure!(
        value.is_finite() && value >= 0.0 && value * 1_000_000_000.0 < i64::MAX as f64,
        "duration must be finite, non-negative, and representable in nanoseconds"
    );
    Ok((value * 1_000_000_000.0).round_ties_even() as u64)
}

struct CaptureIdentity {
    uuid: Uuid,
    x_correlation_id: String,
    /// Coordinator-known arrival facts used to synthesize a fallback record
    /// when an identity has no drained worker record.
    context: MeasuredContext,
}

/// Coordinator-owned finalization facts learned after dispatch.
///
/// Worker observers record only transport facts. The coordinator uuid join applies
/// phase, session number, credit-latency policy, and terminal outcome.
struct CaptureLabel {
    phase: MetricsPhase,
    session_num: u64,
    has_credit_timestamp: bool,
    terminal: ReplayTerminalStatus,
    start_ns: i64,
    end_ns: i64,
}

struct RunCapture {
    clock: Rc<dyn Clock>,
    origin_ns: i64,
    metrics_config: MetricsConfig,
    identities: RefCell<Vec<CaptureIdentity>>,
    labels: RefCell<HashMap<Uuid, CaptureLabel>>,
    outputs: RefCell<HashMap<Uuid, CapturedModelOutput>>,
    raw_enabled: bool,
    raw_exchanges: RefCell<HashMap<Uuid, CapturedHttpExchange>>,
    /// Whether `inputs.json` is requested; gates canonical-payload retention.
    inputs_enabled: bool,
    /// Per-conversation canonical request bodies keyed by turn index. Retained
    /// only when `inputs_enabled`; deduplicated per `(conversation_id, turn)`
    /// (first write wins) so dataset recycling under `--request-count` collapses
    /// to one payload per dataset turn, matching `inputs.json` semantics.
    /// Sessions are emitted in conversation-id order — a stable ordering that is
    /// independent of run-to-run worker dispatch races and, for the
    /// deterministic synthetic session ids (`session_000000`, …), reproduces
    /// dataset composition order.
    input_sessions: RefCell<BTreeMap<String, BTreeMap<usize, Box<serde_json::value::RawValue>>>>,
    /// Non-consuming cloned records for the live-results sink, keyed by uuid;
    /// the authoritative record stays in the worker observer for the drain.
    live_records: RefCell<HashMap<Uuid, RecordIngest>>,
    /// Finished worker records staged for the adaptive window sampler, keyed by
    /// uuid. The online dispatcher records per-token facts worker-locally, so the
    /// coordinator's adaptive sampler never sees them through the callback
    /// observer; the per-phase adaptive record processor drains this map through
    /// [`AdaptiveTerminalRecordSource`]. Consumed per completed turn, so it never
    /// accumulates beyond in-flight requests.
    adaptive_records: RefCell<HashMap<Uuid, RecordIngest>>,
    /// Whether the live-results sink consumes each completed record.
    wants_live_sink_record: bool,
    /// Whether an adaptive phase consumes each completed record.
    wants_adaptive_record: bool,
    /// Dispatch-ordinal authority. Cellular controllers inject an autonomous issuer
    /// that stamps global ordinals spanning every cell partition.
    issuance: Rc<dyn IssuanceAuthority>,
    /// Global cumulative dispatch count of the phases before each phase (0 for the
    /// first). A cell's per-phase sampler restarts at 0, so the autonomous issuer
    /// adds this base to a turn's phase-local slot to recover the single-cell
    /// absolute slot. Empty (all-zero) for the single-process path.
    phase_ordinal_bases: HashMap<MetricsPhase, usize>,
    /// Whether this capture runs in metrics-only (sketch) mode: each completed
    /// turn's record is folded into `accumulator` and dropped as the run streams,
    /// so peak coordinator memory is O(sketch) instead of O(records).
    metrics_only: bool,
    /// Bounded streaming accumulator that folds each metrics-only record on
    /// completion (see [`RunCapture::fold_streaming`]). Empty and unused in exact
    /// mode. `RefCell` because the fold runs from `&self` record processing.
    accumulator: RefCell<MetricsAccumulator>,
    /// Errored/canceled metrics-only records retained for the report's error
    /// grouping ([`group_record_errors`]); the fold drops every non-errored
    /// record, so this stays O(errors), not O(records).
    streaming_errored: RefCell<Vec<CapturedRecord>>,
    /// Whether this capture runs in exact-fold mode: like sketch, it folds each
    /// completed record into `accumulator` and drops the heavy per-record data as
    /// the run streams, BUT the accumulator stays in EXACT (non-sketch) storage and
    /// each record is stamped with its absolute dispatch `request_index` before the
    /// fold, so `export_results` yields exact percentiles/timeslices/series — not the
    /// sketch approximation. Distinct from `metrics_only` (sketch); the two are
    /// mutually exclusive. Selected only for the single-thread `DirectIssuanceAuthority`
    /// scheduled path with no per-record file artifacts (see [`exact_fold_eligible`]).
    exact_fold: bool,
    /// Monotonic dispatch-ordinal counter for exact-fold, incremented once per
    /// [`RunCapture::begin`]. Its value at `begin` is the turn's `flat_local` — the
    /// dense absolute record slot the [`DirectIssuanceAuthority`] would stamp — so
    /// exact-fold rows land at the same absolute ordinals as the retained-record path.
    /// Unused (and left at 0) in sketch/exact modes.
    fold_dispatch_next: Cell<usize>,
    /// Maps each dispatched turn's uuid to the dispatch ordinal assigned at `begin`,
    /// consumed once at completion by the phase processor's exact-fold branch. Drained
    /// per completed turn, so it never outgrows in-flight work. Only populated in
    /// exact-fold mode.
    fold_dispatch_ordinals: RefCell<HashMap<Uuid, usize>>,
    /// Streaming per-record artifact lane: when exact-fold runs a records/
    /// raw/CSV-artifact run, each completed record's rows are appended here before the
    /// fold drops it, so the artifacts are still emitted without retaining every record.
    /// `None` on the retained-record path (which uses the batch writers) and whenever no
    /// lane artifact is requested. Set once at construction via [`Self::with_record_lane`].
    record_lane: Option<Rc<RecordArtifactLane>>,
    /// Per-record OTLP histogram accumulator: when native OTLP is enabled on
    /// the exact-fold path, each completed profiling record is folded here at
    /// completion (an order-independent fold) and then dropped, instead of iterating
    /// the retained record set post-run. `None` when native OTLP is off. Set once at
    /// construction via [`Self::with_otel`].
    otel: Option<RefCell<OtelRecordAccumulator>>,
    /// Whether the fold-and-drop path must retain each turn's model output text long
    /// enough to stream its `outputs.json` entry: `record_model_output`
    /// stages the text in `outputs`, [`Self::fold_record`] attaches it to the streamed
    /// record and drops it. `false` on the retain path (which keeps `outputs` for the
    /// batch writer) and whenever no `outputs.json` artifact is requested. Set once via
    /// [`Self::with_outputs_capture`].
    capture_outputs_text: bool,
}

impl RunCapture {
    #[allow(clippy::too_many_arguments)]
    fn new(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
        raw_enabled: bool,
        inputs_enabled: bool,
        wants_live_sink_record: bool,
        wants_adaptive_record: bool,
        exact_fold: bool,
    ) -> Self {
        // Cell processes select the autonomous issuer from `AIPERF_CELL_ID` and
        // `AIPERF_CELL_COUNT`; single-process execution uses direct issuance.
        Self::new_with_issuance(
            clock,
            origin_ns,
            config,
            raw_enabled,
            inputs_enabled,
            wants_live_sink_record,
            wants_adaptive_record,
            exact_fold,
            crate::engine::cellular_cell::issuance_authority_from_env(),
        )
    }

    /// Construct with an explicitly injected dispatch-ordinal issuer. Thread-per-core
    /// execution builds one `RunCapture` per sub-cell thread with a per-thread issuer
    /// (see
    /// [`issuance_authority_for`](crate::engine::cellular_cell::issuance_authority_for))
    /// whose `(cell_id, cell_count)` partition the process-global env vars cannot
    /// express. Per-phase ordinal bases come from the environment and carry no
    /// partition.
    #[allow(clippy::too_many_arguments)]
    fn new_with_issuance(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
        raw_enabled: bool,
        inputs_enabled: bool,
        wants_live_sink_record: bool,
        wants_adaptive_record: bool,
        exact_fold: bool,
        issuance: Rc<dyn IssuanceAuthority>,
    ) -> Self {
        // Cell processes read global phase ordinal bases from
        // `AIPERF_CELL_PHASE_ORDINAL_BASES`; single-process execution uses zero bases.
        Self::new_with_issuance_and_bases(
            clock,
            origin_ns,
            config,
            raw_enabled,
            inputs_enabled,
            wants_live_sink_record,
            wants_adaptive_record,
            exact_fold,
            issuance,
            crate::engine::cellular_cell::phase_ordinal_bases_from_env(),
        )
    }

    /// Construct with explicitly injected per-phase global ordinal bases.
    ///
    /// A single-process thread-per-core scheduled run (cells == 1, no controller)
    /// has no `AIPERF_CELL_PHASE_ORDINAL_BASES` env var, yet its `W` sub-cell
    /// threads still need each phase's global base so profiling ordinals do not
    /// collide with warmup's `[0, W)` block. The sharded runtime computes the
    /// bases from the phase `requests` budgets (the same policy as
    /// [`crate::engine::cellular_controller::phase_ordinal_bases`]) and injects the same
    /// map into every thread's capture. Controller children use the global,
    /// partition-independent environment bases for every thread.
    #[allow(clippy::too_many_arguments)]
    fn new_with_issuance_and_bases(
        clock: Rc<dyn Clock>,
        origin_ns: i64,
        config: MetricsConfig,
        raw_enabled: bool,
        inputs_enabled: bool,
        wants_live_sink_record: bool,
        wants_adaptive_record: bool,
        exact_fold: bool,
        issuance: Rc<dyn IssuanceAuthority>,
        phase_ordinal_bases: HashMap<MetricsPhase, usize>,
    ) -> Self {
        // Sketch storage mode selects metrics-only fold-and-drop; exact mode does not
        // use the streaming fields.
        let metrics_only = matches!(
            config.storage_mode,
            crate::metrics_core::MetricsStorageMode::Sketch { .. }
        );
        // Sketch and exact-fold are mutually exclusive: sketch keeps the bounded
        // t-digest, exact-fold keeps exact NaN-sparse columns. The caller only sets
        // exact_fold on the exact (non-sketch) scheduled path, but guard it so a
        // sketch config can never accidentally run the exact-fold column path.
        let exact_fold = exact_fold && !metrics_only;
        let accumulator = RefCell::new(MetricsAccumulator::with_config(config.clone()));
        Self {
            clock,
            origin_ns,
            metrics_config: config,
            identities: RefCell::new(Vec::new()),
            labels: RefCell::new(HashMap::new()),
            outputs: RefCell::new(HashMap::new()),
            raw_enabled,
            raw_exchanges: RefCell::new(HashMap::new()),
            inputs_enabled,
            input_sessions: RefCell::new(BTreeMap::new()),
            live_records: RefCell::new(HashMap::new()),
            adaptive_records: RefCell::new(HashMap::new()),
            wants_live_sink_record,
            wants_adaptive_record,
            issuance,
            phase_ordinal_bases,
            metrics_only,
            accumulator,
            streaming_errored: RefCell::new(Vec::new()),
            exact_fold,
            fold_dispatch_next: Cell::new(0),
            fold_dispatch_ordinals: RefCell::new(HashMap::new()),
            record_lane: None,
            otel: None,
            capture_outputs_text: false,
        }
    }

    /// Attach the streaming per-record artifact lane, consumed once per completed
    /// record in the exact-fold [`Self::fold_record`] path before the record is
    /// dropped. Builder-style so only the single-thread exact-fold call site opts in;
    /// every other construction leaves it `None` and uses the batch writers.
    fn with_record_lane(mut self, lane: Option<Rc<RecordArtifactLane>>) -> Self {
        self.record_lane = lane;
        self
    }

    /// Enable per-record OTLP folding at completion. When `enabled`, each
    /// completed profiling record is folded into a bounded [`OtelRecordAccumulator`]
    /// in [`Self::fold_record`] and dropped, so the OTLP histograms need no retained
    /// record set. Builder-style so only the exact-fold call site with native OTLP
    /// opts in; every other construction leaves it `None` and the retain path folds
    /// the retained records post-run.
    fn with_otel(mut self, enabled: bool) -> Self {
        if enabled {
            self.otel = Some(RefCell::new(OtelRecordAccumulator::new()));
        }
        self
    }

    /// Retain each turn's model output text for streaming `outputs.json`.
    /// When `enabled`, `record_model_output` stages the text even on the fold-and-drop
    /// path so [`Self::fold_record`] can attach it to the streamed record before the
    /// fold drops it. Builder-style so only the exact-fold call site with an
    /// `outputs.json` artifact opts in.
    fn with_outputs_capture(mut self, enabled: bool) -> Self {
        self.capture_outputs_text = enabled;
        self
    }

    /// Move the folded per-record OTLP accumulator out for the finalize, if one was
    /// attached. Consumed once at run end (leaves an empty accumulator behind).
    fn take_otel(&self) -> Option<OtelRecordAccumulator> {
        self.otel
            .as_ref()
            .map(|cell| std::mem::take(&mut *cell.borrow_mut()))
    }

    /// Flush and close the streaming per-record artifact lane, if one is attached.
    /// Called once at run end after every record has been folded (and its rows
    /// streamed); a lazy CSV that saw no non-skipped row stays absent.
    fn finish_record_lane(&self) -> Result<()> {
        match &self.record_lane {
            Some(lane) => lane.finish(),
            None => Ok(()),
        }
    }

    /// Retain one turn's canonical request body for `inputs.json`.
    ///
    /// Called on the coordinator thread for every dispatched turn (independent
    /// of raw-artifact capture). Deduplicates per `(conversation_id, turn_index)`
    /// so recycled dataset turns collapse to a single payload, and remembers
    /// first-dispatch conversation order for deterministic session ordering.
    fn record_input_payload(
        &self,
        conversation_id: &str,
        turn_index: usize,
        payload: &[u8],
    ) -> Result<()> {
        if !self.inputs_enabled {
            return Ok(());
        }
        let mut sessions = self.input_sessions.borrow_mut();
        let turns = sessions.entry(conversation_id.to_string()).or_default();
        if let std::collections::btree_map::Entry::Vacant(slot) = turns.entry(turn_index) {
            let parsed: Box<serde_json::value::RawValue> = serde_json::from_slice(payload)
                .with_context(|| {
                    format!(
                        "validating canonical request payload for inputs.json \
                         (conversation {conversation_id}, turn {turn_index})"
                    )
                })?;
            slot.insert(parsed);
        }
        Ok(())
    }

    /// Consume the retained payloads into conversation-id-ordered
    /// `inputs.json` sessions. The `BTreeMap` iteration yields sorted keys, so
    /// ordering is identical across same-seed runs regardless of worker races.
    fn take_input_sessions(&self) -> Vec<InputSession> {
        self.input_sessions
            .take()
            .into_iter()
            .map(|(session_id, turns)| InputSession {
                session_id,
                payloads: turns.into_values().collect(),
            })
            .collect()
    }

    /// Whether the worker should return a per-turn record: a non-consuming
    /// snapshot for the live-results sink or the adaptive window sampler, or a
    /// consuming drain in metrics-only mode (see [`MeasuredContext::consume_record`]).
    fn wants_live_record(&self) -> bool {
        self.wants_live_sink_record || self.wants_adaptive_record || self.folds_records()
    }

    /// Whether this capture folds each completed record into `accumulator` and drops
    /// the heavy per-record data (worker `token_arrivals_ns`, identities/labels) as
    /// the run streams, rather than retaining every record until end-of-run. True for
    /// both sketch (`metrics_only`) and exact-fold; the two differ only in the
    /// accumulator's storage mode and whether `request_index` is stamped.
    fn folds_records(&self) -> bool {
        self.metrics_only || self.exact_fold
    }

    /// Assign and record the next dense dispatch ordinal for `uuid` (exact-fold only).
    /// Called once per turn at [`Self::begin`]; the value is the turn's `flat_local`,
    /// which the [`DirectIssuanceAuthority`] maps identically to its `request_index`.
    fn assign_fold_ordinal(&self, uuid: Uuid) -> usize {
        let ordinal = self.fold_dispatch_next.get();
        self.fold_dispatch_next.set(ordinal + 1);
        self.fold_dispatch_ordinals
            .borrow_mut()
            .insert(uuid, ordinal);
        ordinal
    }

    /// Consume the dispatch ordinal assigned to `uuid` at `begin`. `None` for a turn
    /// no `begin` recorded (never happens on the exact-fold path, where every
    /// dispatched turn passes through `begin`), in which case the fold appends.
    fn take_fold_ordinal(&self, uuid: Uuid) -> Option<usize> {
        self.fold_dispatch_ordinals.borrow_mut().remove(&uuid)
    }

    /// Record the dispatch identity plus coordinator-known arrival facts, and
    /// return the measured context the dispatcher forwards to the worker so it
    /// registers arrival locally. The identity push order is the global dispatch
    /// ordinal used at finish; it runs on the coordinator thread before backend
    /// dispatch, so it is independent of worker count.
    fn begin(&self, turn: &TurnToSend) -> MeasuredContext {
        let arrival_ms = self.clock.now_ns().saturating_sub(self.origin_ns) as f64 / 1_000_000.0;
        let context = MeasuredContext {
            arrival_ms,
            input_length: turn.input_length,
            requested_output_length: turn.max_output_tokens,
            metadata: RequestMetricMetadata {
                turn_index: u32::try_from(turn.turn_index).unwrap_or(u32::MAX),
                conversation_id: Some(turn.conversation_id.clone()),
                audio_duration_s: turn.audio_duration_seconds,
                ..RequestMetricMetadata::default()
            },
            wants_live_record: self.wants_live_record(),
            // Fold-and-drop modes (sketch + exact-fold) fold each record and drop it,
            // so the worker must move the record out of its observer to free token
            // storage as it goes.
            consume_record: self.folds_records(),
        };
        // Fold-and-drop modes never join records by dispatch identity at finish (they
        // fold each on completion), so skip the O(records) identity retention — the
        // fold source is the per-turn record staged by `record_live`, and a failed
        // turn's record is synthesized in `process` instead. Other modes retain the
        // identity for the finish-time UUID join.
        if !self.folds_records() {
            self.identities.borrow_mut().push(CaptureIdentity {
                uuid: turn.uuid,
                x_correlation_id: turn.x_correlation_id.clone(),
                context: context.clone(),
            });
        }
        // Exact-fold stamps each record's absolute dispatch `request_index` so its row
        // lands at the same ordinal the retained-record path would assign. The ordinal is
        // this turn's `begin` push order (dense `0, 1, 2, …`), matching the
        // `DirectIssuanceAuthority` `flat_local`; record it here for the completion-time
        // fold. (Sketch ignores `request_index`, so it needs no ordinal.)
        if self.exact_fold {
            self.assign_fold_ordinal(turn.uuid);
        }
        context
    }

    fn label(
        &self,
        credit: &IssuedCredit,
        phase: MetricsPhase,
        has_credit_timestamp: bool,
        outcome: &TurnDispatchOutcome,
    ) {
        // Labels feed the finish-time uuid join only; fold-and-drop modes fold each
        // record on completion and never join, so retaining them would be pure
        // O(records) waste. The fold applies phase/session/admit itself.
        if self.folds_records() {
            return;
        }
        self.labels.borrow_mut().insert(
            credit.turn.uuid,
            CaptureLabel {
                phase,
                session_num: credit.id,
                has_credit_timestamp,
                terminal: outcome.terminal,
                start_ns: outcome.start_ns,
                end_ns: outcome.end_ns,
            },
        );
    }

    fn record_live(&self, uuid: Uuid, record: RecordIngest) {
        // Fold-and-drop modes stage every completed turn's record for the phase
        // processor's fold-and-drop, regardless of any live/adaptive consumer. An
        // adaptive phase still needs its own copy (read-only window sampling), so
        // clone into `adaptive_records` when one is active. Both maps are drained
        // per completed turn, so neither outgrows in-flight work.
        if self.folds_records() {
            if self.wants_adaptive_record {
                self.adaptive_records
                    .borrow_mut()
                    .insert(uuid, record.clone());
            }
            self.live_records.borrow_mut().insert(uuid, record);
            return;
        }
        // Fan the worker's non-consuming snapshot out to each interested
        // consumer. Both drain their own map per completed turn, so neither
        // consumer starves the other and neither map outgrows in-flight work.
        match (self.wants_adaptive_record, self.wants_live_sink_record) {
            (true, true) => {
                self.adaptive_records
                    .borrow_mut()
                    .insert(uuid, record.clone());
                self.live_records.borrow_mut().insert(uuid, record);
            }
            (true, false) => {
                self.adaptive_records.borrow_mut().insert(uuid, record);
            }
            (false, true) => {
                self.live_records.borrow_mut().insert(uuid, record);
            }
            (false, false) => {}
        }
    }

    fn record_model_output(
        &self,
        uuid: Uuid,
        flattened_text: &str,
        visible_text: Option<&str>,
        reasoning_text: Option<&str>,
    ) -> Result<()> {
        // Fold-and-drop modes retain no per-record output artifact by default (sketch
        // forbids `outputs_path` in `validate_plan`), so keeping the text would be pure
        // O(records) waste; drop it before the map grows. The exception is exact-fold
        // with a streaming `outputs.json`: stage the text so `fold_record` can
        // attach it to the streamed entry and drop it per completion (bounded to
        // in-flight work).
        if self.folds_records() && !self.capture_outputs_text {
            return Ok(());
        }
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
        record: crate::transport::core::RequestRecord,
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

    /// Build a live-sink record from the worker's non-consuming cloned record.
    ///
    /// The clone is removed from the pending map (each request emits once); the
    /// authoritative record stays in the worker observer for the final drain, so
    /// live emission never undercounts the end-of-run aggregate. `admit_ns` and
    /// `session_num` are patched to the credit-issued values the live consumer
    /// expects.
    fn snapshot_live(&self, credit: &IssuedCredit) -> Option<CapturedRecord> {
        let mut ingest = self.live_records.borrow_mut().remove(&credit.turn.uuid)?;
        ingest.session_num = credit.id;
        if ingest.admit_ns.is_some() {
            ingest.admit_ns = Some(credit.issued_ns.saturating_sub(self.origin_ns));
        }
        Some(CapturedRecord {
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

    /// Join per-worker drained records to dispatch identities and produce the
    /// coordinator's captured records in dispatch order.
    ///
    /// Worker observers accumulate per-worker in dense-local order, so a record's
    /// drain slot is meaningless globally. Keyed on each record's true drain
    /// `Uuid` (never `correlation_id`, which aggregate-only mode blanks), this:
    ///
    /// 1. resolves each identity to its worker record, or synthesizes a fallback
    ///    record for an identity that failed before any worker touched it;
    /// 2. stamps `request_index` to the identity's global dispatch ordinal so the
    ///    downstream re-ingest lands each record at a unique, dense dispatch-ordered
    ///    slot;
    /// 3. patches `phase`/`session_num`/`admit_ns` from the coordinator-owned
    ///    label, preserving the credit-latency time base.
    fn finish(
        &self,
        issued_times: &HashMap<Uuid, i64>,
        drained: Vec<(Uuid, RecordIngest)>,
    ) -> Result<Vec<CapturedRecord>> {
        let identities = self.identities.borrow();
        let labels = self.labels.borrow();
        let outputs = self.outputs.borrow();
        let mut raw_exchanges = self.raw_exchanges.take();

        let mut records_by_uuid = self.resolve_records_by_uuid(&identities, &labels, drained)?;

        // Emit rows in dispatch (identity) order. `ordinal` (begin order) is the
        // cumulative flat dispatch index; `phase_counters` tracks the per-phase
        // dispatch index because a cell's sampler restarts each phase, so the
        // cellular issuer's ordinal must be phase-local. The identity issuer uses the
        // flat ordinal.
        let mut phase_counters: HashMap<_, usize> = HashMap::new();
        identities
            .iter()
            .enumerate()
            .map(|(ordinal, identity)| {
                let mut ingest = records_by_uuid.remove(&identity.uuid).ok_or_else(|| {
                    anyhow!(
                        "captured request {} produced no native metric record",
                        identity.uuid
                    )
                })?;
                self.patch_joined_ingest(
                    ordinal,
                    identity,
                    &labels,
                    issued_times,
                    &mut phase_counters,
                    &mut ingest,
                )?;
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

    /// Fold one completed fold-and-drop turn's record into the streaming
    /// accumulator and drop it, keeping peak memory O(sketch)/O(scalars) rather than
    /// O(full records).
    ///
    /// Applies the coordinator-owned fields the finish-time join would apply —
    /// `phase` (the worker defaults every record, warmup included, to Profiling),
    /// `session_num`, and the credit-issued `admit_ns` (bit-equal to the finish
    /// path's `issued_offset_ns` because the run origin equals every phase's start).
    /// In exact-fold `request_index` is `Some` (the turn's dense absolute dispatch
    /// ordinal), so the record's row lands at the same absolute slot as retained-record
    /// execution and exact percentiles/timeslices/series are byte-identical; in
    /// sketch it is `None` (the sketch store ignores it).
    ///
    /// Sketch approximate-memory contract: the sketch (t-digest percentiles, Welford
    /// mean/M2, and the float running sums) is order-independent only up to a few
    /// ULPs, and this folds in *completion* order rather than the finish path's
    /// *dispatch* order. So percentiles, means, and float sums drift a few ULPs —
    /// below display precision — and lose exact run-to-run reproducibility, while
    /// counts, min/max, and integer sums stay bit-identical. Exact-fold does NOT drift:
    /// `insert_record_at(request_index)` places each row at its absolute slot, so the
    /// dense NaN-sparse columns are byte-identical to dispatch-order ingestion. That
    /// sketch drift is the accepted price of bounded memory; a reorder buffer would be
    /// O(records) and defeat the goal.
    fn fold_streaming(
        &self,
        ingest: RecordIngest,
        phase: MetricsPhase,
        has_credit_timestamp: bool,
        request_index: Option<usize>,
        credit: &IssuedCredit,
    ) -> Result<()> {
        let admit_ns =
            has_credit_timestamp.then(|| credit.issued_ns.saturating_sub(self.origin_ns));
        self.fold_record(
            ingest,
            credit.turn.uuid,
            &credit.turn.x_correlation_id,
            phase,
            credit.id,
            admit_ns,
            request_index,
        )
    }

    /// Stamp the coordinator-owned fields onto one completed record, process it into
    /// the streaming accumulator, and drop it — retaining only errored/canceled
    /// records for [`group_record_errors`]. The primitive both the sketch and
    /// exact-fold fold paths share; taking the fields directly (not an
    /// [`IssuedCredit`]) keeps it unit-testable in isolation.
    ///
    /// `request_index` overwrites the worker's dense-local slot with the absolute
    /// dispatch ordinal in exact-fold; passing `None` (sketch) leaves the worker value
    /// untouched, which the sketch store ignores.
    #[allow(clippy::too_many_arguments)]
    fn fold_record(
        &self,
        mut ingest: RecordIngest,
        uuid: Uuid,
        x_correlation_id: &str,
        phase: MetricsPhase,
        session_num: u64,
        admit_ns: Option<i64>,
        request_index: Option<usize>,
    ) -> Result<()> {
        ingest.phase = phase;
        ingest.session_num = session_num;
        ingest.admit_ns = admit_ns;
        if let Some(row) = request_index {
            ingest.request_index = Some(row);
        }
        self.accumulator.borrow_mut().process_record(&ingest);
        let errored = ingest.errored || ingest.canceled;
        // Per-record OTLP folds every PROFILING record (success and error
        // alike, matching the retain path's post-run loop) into the order-independent
        // accumulator; warmup records never contribute.
        let wants_otel = self.otel.is_some() && phase == MetricsPhase::Profiling;
        // Materialize a CapturedRecord only when something consumes it: the streaming
        // artifact lane writes every record's rows, the per-record OTLP fold
        // observes each profiling record, and the error grouping retains
        // errored records. The raw HTTP exchange captured for this uuid is pulled out
        // here (present only when `raw_path` is enabled) so raw.jsonl and the error
        // classification see the same transport facts the retain path does; the drop
        // keeps `raw_exchanges` bounded to in-flight work.
        if self.record_lane.is_some() || errored || wants_otel {
            let raw = self.raw_exchanges.borrow_mut().remove(&uuid);
            // The streaming outputs.json entry reads the model output text;
            // drain the text staged by `record_model_output` so the entry carries it,
            // then the record (and its text) is dropped here. records/raw/CSV rows never
            // read it, so the default is byte-safe when outputs.json is not requested.
            let output = if self.capture_outputs_text {
                self.outputs.borrow_mut().remove(&uuid).unwrap_or_default()
            } else {
                CapturedModelOutput::default()
            };
            let captured = CapturedRecord {
                uuid,
                x_correlation_id: x_correlation_id.to_string(),
                output,
                raw,
                ingest,
            };
            if let Some(lane) = &self.record_lane {
                lane.write(&captured, &self.metrics_config)?;
            }
            if wants_otel && let Some(cell) = &self.otel {
                observe_otel_record(&mut cell.borrow_mut(), &captured, &self.metrics_config);
            }
            if errored {
                self.streaming_errored.borrow_mut().push(captured);
            }
        }
        Ok(())
    }

    /// Remove one metrics-only turn's staged worker record for the phase
    /// processor's fold. Absent for a turn that failed or was canceled before it
    /// completed (its `Err`/cancel path never called `record_live`), in which case
    /// the processor synthesizes the record instead.
    fn take_streaming_record(&self, uuid: Uuid) -> Option<RecordIngest> {
        self.live_records.borrow_mut().remove(&uuid)
    }

    /// Synthesize the record for a metrics-only turn the worker never staged — a
    /// dispatch that failed or was canceled before completion.
    ///
    /// Built exactly as [`resolve_records_by_uuid`]'s finish-time fallback: a
    /// one-shot [`NativeMetricsObserver`] fed the same arrival/terminal/response
    /// facts (from `credit.turn` and `outcome`), so the errored/canceled flag,
    /// `ErrorRequestCount`, and error grouping match the exact and retained paths.
    /// `fold_streaming` then applies phase/session/admit.
    fn synthesize_streaming_fallback(
        &self,
        credit: &IssuedCredit,
        outcome: &TurnDispatchOutcome,
    ) -> RecordIngest {
        let turn = &credit.turn;
        let fallback = NativeMetricsObserver::new(
            self.clock.clone(),
            self.origin_ns,
            self.metrics_config.clone(),
        );
        let arrival_ms = self.clock.now_ns().saturating_sub(self.origin_ns) as f64 / 1_000_000.0;
        fallback.register_metadata(
            turn.uuid,
            RequestMetricMetadata {
                turn_index: u32::try_from(turn.turn_index).unwrap_or(u32::MAX),
                conversation_id: Some(turn.conversation_id.clone()),
                audio_duration_s: turn.audio_duration_seconds,
                ..RequestMetricMetadata::default()
            },
        );
        fallback.on_arrival(
            turn.uuid,
            arrival_ms,
            turn.input_length,
            turn.max_output_tokens,
        );
        fallback.on_terminal(turn.uuid, outcome.terminal);
        fallback.record_response(
            turn.uuid,
            NativeResponseMetadata {
                start_ns: Some(outcome.start_ns),
                end_ns: Some(outcome.end_ns),
                ..NativeResponseMetadata::default()
            },
        );
        fallback
            .finish_with_records()
            .records
            .into_iter()
            .find_map(|(uuid, ingest)| (uuid == turn.uuid).then_some(ingest))
            .unwrap_or_else(|| {
                // Defensive: the observer always yields the record it was just fed;
                // fall back to a minimal terminal record rather than panicking.
                let mut ingest = RecordIngest::minimal(
                    outcome.start_ns,
                    outcome.end_ns,
                    MetricsPhase::Profiling,
                );
                ingest.errored = matches!(
                    outcome.terminal,
                    ReplayTerminalStatus::Failed | ReplayTerminalStatus::Rejected
                );
                ingest.canceled = outcome.terminal == ReplayTerminalStatus::Canceled;
                ingest
            })
    }

    /// Move the streaming accumulator and its retained errored records out for the
    /// finalize, leaving a fresh empty accumulator behind so the capture stays
    /// reusable. The sharded path ships the returned accumulator as its shard
    /// partition; the single-thread path merges it into the report accumulator.
    fn take_streamed(&self) -> (MetricsAccumulator, Vec<CapturedRecord>) {
        let accumulator = std::mem::replace(
            &mut *self.accumulator.borrow_mut(),
            MetricsAccumulator::with_config(self.metrics_config.clone()),
        );
        (accumulator, self.streaming_errored.take())
    }

    /// Builds the uuid→record map for the exact-mode finish, synthesizing fallback
    /// records for identities no worker observer produced. (Metrics-only mode folds
    /// each record on completion and synthesizes its own per-turn fallback in
    /// [`RunCapture::synthesize_streaming_fallback`], so it never joins here.)
    fn resolve_records_by_uuid(
        &self,
        identities: &[CaptureIdentity],
        labels: &HashMap<Uuid, CaptureLabel>,
        drained: Vec<(Uuid, RecordIngest)>,
    ) -> Result<HashMap<Uuid, RecordIngest>> {
        let mut records_by_uuid: HashMap<Uuid, RecordIngest> =
            HashMap::with_capacity(drained.len());
        for (uuid, ingest) in drained {
            ensure!(
                records_by_uuid.insert(uuid, ingest).is_none(),
                "worker drained request {uuid} more than once"
            );
        }
        let missing: Vec<&CaptureIdentity> = identities
            .iter()
            .filter(|identity| !records_by_uuid.contains_key(&identity.uuid))
            .collect();
        if !missing.is_empty() {
            let fallback = NativeMetricsObserver::new(
                self.clock.clone(),
                self.origin_ns,
                self.metrics_config.clone(),
            );
            for identity in &missing {
                let label = labels.get(&identity.uuid);
                let terminal = label
                    .map(|label| label.terminal)
                    .unwrap_or(ReplayTerminalStatus::Failed);
                fallback.register_metadata(identity.uuid, identity.context.metadata.clone());
                fallback.on_arrival(
                    identity.uuid,
                    identity.context.arrival_ms,
                    identity.context.input_length,
                    identity.context.requested_output_length,
                );
                fallback.on_terminal(identity.uuid, terminal);
                let (start_ns, end_ns) = label
                    .map(|label| (label.start_ns, label.end_ns))
                    .unwrap_or_else(|| {
                        let now = self.clock.now_ns();
                        (now, now)
                    });
                fallback.record_response(
                    identity.uuid,
                    NativeResponseMetadata {
                        start_ns: Some(start_ns),
                        end_ns: Some(end_ns),
                        ..NativeResponseMetadata::default()
                    },
                );
            }
            for (uuid, ingest) in fallback.finish_with_records().records {
                records_by_uuid.insert(uuid, ingest);
            }
        }
        ensure!(
            records_by_uuid.len() == identities.len(),
            "native record capture finalized {} records for {} dispatched identities",
            records_by_uuid.len(),
            identities.len()
        );
        Ok(records_by_uuid)
    }

    /// Patches one joined record's coordinator-owned fields (phase, session number,
    /// global dispatch ordinal, and admit timestamp) exactly as [`finish`] does.
    fn patch_joined_ingest(
        &self,
        ordinal: usize,
        identity: &CaptureIdentity,
        labels: &HashMap<Uuid, CaptureLabel>,
        issued_times: &HashMap<Uuid, i64>,
        phase_counters: &mut HashMap<MetricsPhase, usize>,
        ingest: &mut RecordIngest,
    ) -> Result<()> {
        let has_credit_timestamp = labels
            .get(&identity.uuid)
            .map(|label| label.has_credit_timestamp)
            .unwrap_or(true);
        if let Some(label) = labels.get(&identity.uuid) {
            ingest.phase = label.phase;
            ingest.session_num = label.session_num;
        }
        let within_phase = phase_counters.entry(ingest.phase).or_insert(0);
        let within = *within_phase;
        *within_phase += 1;
        let phase_base = self
            .phase_ordinal_bases
            .get(&ingest.phase)
            .copied()
            .unwrap_or(0);
        ingest.request_index = Some(self.issuance.global_ordinal(ordinal, phase_base, within));
        ingest.admit_ns = if has_credit_timestamp {
            Some(*issued_times.get(&identity.uuid).ok_or_else(|| {
                anyhow!("captured request {} has no issuer timestamp", identity.uuid)
            })?)
        } else {
            None
        };
        Ok(())
    }
}

/// Source of the authoritative worker-built terminal record for a completed
/// online turn.
///
/// The online scheduled dispatcher records per-token facts in worker-local
/// observers, so the coordinator's adaptive window sampler never sees them
/// through the callback observer. This seam lets the per-phase adaptive record
/// processor pull each completed turn's finished [`RecordIngest`] and feed it to
/// the sampler through `WindowSampler::on_record`, exactly as the graph phase
/// runtime does. Backends whose dispatcher feeds the callback observer directly
/// (offline co-simulation) supply no source and keep the observer feed, so the
/// sampler is never double-fed.
pub(crate) trait AdaptiveTerminalRecordSource {
    /// Consume the finished record for `uuid`, if the worker produced one.
    fn take_terminal_record(&self, uuid: Uuid) -> Option<RecordIngest>;
}

impl AdaptiveTerminalRecordSource for RunCapture {
    fn take_terminal_record(&self, uuid: Uuid) -> Option<RecordIngest> {
        self.adaptive_records.borrow_mut().remove(&uuid)
    }
}

/// Feeds each completed online turn's finished worker record into the adaptive
/// window sampler.
///
/// This uses `graph_phase_runtime`'s `sampler.on_record(&record.ingest)` feed
/// for the scheduled online path, where token/usage/terminal facts are recorded
/// worker-locally and the coordinator's callback observer only sees arrivals.
/// It runs after normal measurement and credit return, keeping token
/// accumulation off the coordinator.
struct AdaptiveSamplerRecordProcessor {
    source: Rc<dyn AdaptiveTerminalRecordSource>,
    sampler: SharedWindowSampler,
}

#[async_trait(?Send)]
impl TurnRecordProcessor for AdaptiveSamplerRecordProcessor {
    async fn process(&self, credit: &IssuedCredit, _outcome: &TurnDispatchOutcome) -> Result<()> {
        if let Some(ingest) = self.source.take_terminal_record(credit.turn.uuid) {
            self.sampler.borrow_mut().on_record(&ingest);
        }
        Ok(())
    }
}

struct CapturePhaseProcessor {
    capture: Rc<RunCapture>,
    phase: MetricsPhase,
    has_credit_timestamp: bool,
    live_sink: Option<Rc<dyn LiveResultsSink>>,
    heartbeat: Option<Rc<HeartbeatLane>>,
}

#[async_trait(?Send)]
impl TurnRecordProcessor for CapturePhaseProcessor {
    async fn process(&self, credit: &IssuedCredit, outcome: &TurnDispatchOutcome) -> Result<()> {
        if self.capture.folds_records() {
            // Fold-and-drop mode (sketch or exact-fold): fold this turn's record into
            // the streaming accumulator and drop it, so peak memory stays
            // O(sketch)/O(scalars). A successful turn staged its record via
            // `record_live`; a failed or canceled turn never did (its `Err`/cancel path
            // skips `record_live`), so synthesize the record — matching the exact
            // path's finish-time fallback — to keep error counts and grouping correct.
            // Exact-fold also stamps the turn's absolute dispatch `request_index`
            // (assigned at `begin`) so its row lands at the retained-record path's
            // ordinal; sketch ignores it. The per-record live sink and cellular
            // heartbeat are not driven here: fold-and-drop retains no per-record data to
            // stream, and the exact-fold gate/sharded workers run with neither attached.
            let ingest = match self.capture.take_streaming_record(credit.turn.uuid) {
                Some(ingest) => ingest,
                None => self.capture.synthesize_streaming_fallback(credit, outcome),
            };
            let request_index = if self.capture.exact_fold {
                self.capture.take_fold_ordinal(credit.turn.uuid)
            } else {
                None
            };
            self.capture.fold_streaming(
                ingest,
                self.phase,
                self.has_credit_timestamp,
                request_index,
                credit,
            )?;
            return Ok(());
        }
        self.capture
            .label(credit, self.phase, self.has_credit_timestamp, outcome);
        // The per-record clone is consumed once; feed both the Python live sink and
        // the cellular heartbeat lane from that single snapshot.
        if (self.live_sink.is_some() || self.heartbeat.is_some())
            && let Some(record) = self.capture.snapshot_live(credit)
        {
            if let Some(sink) = &self.live_sink {
                sink.emit_record(&record);
            }
            if let Some(heartbeat) = &self.heartbeat {
                heartbeat.observe_record(&record.ingest);
            }
        }
        Ok(())
    }
}

struct ConfiguredDispatcher {
    execution_backend: Rc<dyn RequestExecutor>,
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
        _observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        let uuid = turn.uuid;
        // Retain the conversation identity for `inputs.json` session grouping
        // before `turn` is consumed by request preparation below.
        let inputs_conversation_id = turn.conversation_id.clone();
        let inputs_turn_index = turn.turn_index;
        // `begin` runs on the coordinator thread before backend dispatch, so its
        // push order is the worker-count-independent global dispatch ordinal.
        // It returns the measured context the worker registers locally; the
        // runner's native-v2 report is then produced from the drained worker
        // records, not a single coordinator observer. The ScheduledRuntime's own
        // observer (`_observer`) is still computed and discarded by the runner.
        let context = self.capture.begin(&turn);
        let turn = PreparedTurn::from_turn(turn, &self.model);
        match self
            .execution_backend
            .execute_measured(turn, context, on_first_token)
            .await
        {
            Ok(MeasuredOutcome {
                result: collected,
                live_record,
            }) => {
                let outcome = collected.outcome;
                self.capture.record_input_payload(
                    &inputs_conversation_id,
                    inputs_turn_index,
                    &collected.request_payload,
                )?;
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
                if let Some(live_record) = live_record {
                    self.capture.record_live(uuid, live_record);
                }
                Ok(outcome)
            }
            // The worker (or, for a pre-dispatch failure, the coordinator
            // fallback at finish) owns finalizing the failed record; the
            // dispatcher only propagates the error.
            Err(error) => Err(error),
        }
    }

    async fn prewarm(&self, turn: TurnToSend) -> Result<()> {
        // Warm the execution backend (every worker) with the real prepared
        // request shape; the backend discards the round-trip and records
        // nothing, so timed issuance starts from a warmed transport.
        let turn = PreparedTurn::from_turn(turn, &self.model);
        self.execution_backend.prewarm(turn).await
    }
}

#[cfg(test)]
mod tests {
    use crate::clock::SimClock;
    use serde_json::json;

    use super::*;

    /// Streamable artifacts do not disqualify exact-fold. `inputs.json` still
    /// disqualifies ONLY when the dataset cannot be generated up front
    /// (`inputs_need_retain == true`). (Parquet only streams under the `parquet`
    /// feature; a lite build keeps it disqualifying — asserted below under the matching
    /// cfg.)
    #[test]
    fn exact_fold_gate_accepts_streamed_artifacts_and_rejects_retained_ones() {
        use crate::engine::protocol::ArtifactSpec;

        let eligible = |artifacts: &ArtifactSpec, inputs_need_retain: bool| {
            exact_fold_eligible(ExactFoldInputs {
                sketch_mode: false,
                shardable: false,
                is_cellular: false,
                has_accuracy: false,
                wants_adaptive_record: false,
                has_live_sink: false,
                has_heartbeat: false,
                wants_per_record_artifacts: wants_per_record_artifacts(
                    artifacts,
                    inputs_need_retain,
                ),
            })
        };

        // No artifacts require retention.
        assert!(eligible(&ArtifactSpec::default(), false));

        // The lane streams records, raw data, and CSV; per-record OTLP folds at
        // completion and does not participate in this gate.
        let streamed = ArtifactSpec {
            records_path: Some("profile_export.jsonl".into()),
            raw_path: Some("profile_export_raw.jsonl".into()),
            records_csv_path: Some("profile_export_records.csv".into()),
            trace: true,
            ..ArtifactSpec::default()
        };
        assert!(eligible(&streamed, false));

        // Parquet streams when the feature is enabled; a lite build retains records.
        let parquet = ArtifactSpec {
            records_parquet_path: Some("profile_export.parquet".into()),
            ..ArtifactSpec::default()
        };
        #[cfg(feature = "parquet")]
        assert!(
            eligible(&parquet, false),
            "parquet streams under the parquet feature"
        );
        #[cfg(not(feature = "parquet"))]
        assert!(
            !eligible(&parquet, false),
            "a lite build cannot stream parquet, so it disqualifies exact-fold"
        );

        // outputs.json streams through the lane.
        let outputs = ArtifactSpec {
            outputs_path: Some("outputs.json".into()),
            ..ArtifactSpec::default()
        };
        assert!(
            eligible(&outputs, false),
            "outputs.json streams at completion"
        );

        // inputs.json is eligible when it can be generated up front, and disqualifying
        // only when the dataset forces the during-run capture path.
        let inputs = ArtifactSpec {
            inputs_path: Some("inputs.json".into()),
            ..ArtifactSpec::default()
        };
        assert!(
            eligible(&inputs, false),
            "up-front-feasible inputs.json does not disqualify"
        );
        assert!(
            !eligible(&inputs, true),
            "a live-reply multi-turn dataset keeps inputs.json on the retain path"
        );
    }

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
            phase_root.derive_root(crate::rng::namespace::TIMING_RAMP_CONCURRENCY)
        );
        assert_eq!(
            roots.prefill_concurrency(),
            phase_root.derive_root(crate::rng::namespace::TIMING_RAMP_PREFILL_CONCURRENCY)
        );
        assert_eq!(
            roots.request_rate(),
            phase_root.derive_root(crate::rng::namespace::TIMING_RAMP_REQUEST_RATE)
        );

        let curve_seeds = [
            roots
                .request_rate()
                .derive_seed(crate::rng::namespace::TIMING_RAMP_POISSON),
            roots
                .prefill_concurrency()
                .derive_seed(crate::rng::namespace::TIMING_RAMP_POISSON),
            roots
                .concurrency()
                .derive_seed(crate::rng::namespace::TIMING_RAMP_POISSON),
        ];
        assert!(curve_seeds.iter().all(Option::is_some));
        assert_ne!(curve_seeds[0], curve_seeds[1]);
        assert_ne!(curve_seeds[0], curve_seeds[2]);
        assert_ne!(curve_seeds[1], curve_seeds[2]);
        assert_ne!(
            roots.concurrency(),
            phase_root.derive_root(crate::rng::namespace::TIMING_RAMP_POISSON),
            "the phase must not pre-derive the curve-local Poisson namespace"
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
        let registry = AIPerfRegistry::builtin().unwrap();
        let dataset = build_synthetic_dataset(
            &registry,
            &spec,
            &models(),
            RngRoot::new(Some(73)),
            &TiktokenTokenizer::builtin(),
            false,
            Arc::new(crate::dataset::NativeSyntheticMediaGeneratorFactory::default()),
            false,
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
        let registry = AIPerfRegistry::builtin().unwrap();
        let dataset = build_synthetic_dataset(
            &registry,
            &spec,
            &models(),
            RngRoot::new(Some(3)),
            &TiktokenTokenizer::builtin(),
            true,
            Arc::new(crate::dataset::NativeSyntheticMediaGeneratorFactory::default()),
            false,
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
        let files = vec![crate::engine::protocol_v2::UserFileSpecV2 {
            path: "nested/run.json".into(),
            format: crate::engine::protocol_v2::UserFileFormatV2::Json,
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
        let files = vec![crate::engine::protocol_v2::UserFileSpecV2 {
            path: "escape/owned.txt".into(),
            format: crate::engine::protocol_v2::UserFileFormatV2::Text,
            content: "must-not-write".into(),
        }];

        let error = materialize_user_files(artifact_dir.path(), &files)
            .unwrap_err()
            .to_string();

        assert!(error.contains("symlink"), "{error}");
        assert!(!outside.path().join("owned.txt").exists());
    }

    /// Distinct per-request facts for the `RunCapture::finish` join tests.
    struct RequestFacts {
        uuid: Uuid,
        arrival_ms: f64,
        token_times_ms: &'static [f64],
        prompt_tokens: u64,
        completion_tokens: u64,
        start_ns: i64,
        end_ns: i64,
    }

    /// Drive one request through a worker observer exactly as
    /// `TransportSink::dispatch_measured` does: register begin-known
    /// metadata (no `request_index` — the worker uses a dense-local arrival slot),
    /// arrival, admit, per-token arrivals, terminal, and the authoritative
    /// transport/usage response.
    fn drive_worker_request(observer: &NativeMetricsObserver, facts: &RequestFacts) {
        observer.register_metadata(facts.uuid, RequestMetricMetadata::default());
        observer.on_arrival(
            facts.uuid,
            facts.arrival_ms,
            facts.prompt_tokens as usize,
            8,
        );
        observer.on_admit(facts.uuid, facts.arrival_ms, 0);
        for &at in facts.token_times_ms {
            observer.on_token(facts.uuid, at);
        }
        observer.on_terminal(facts.uuid, ReplayTerminalStatus::Completed);
        observer.record_response(
            facts.uuid,
            NativeResponseMetadata {
                start_ns: Some(facts.start_ns),
                end_ns: Some(facts.end_ns),
                prompt_tokens: Some(facts.prompt_tokens),
                completion_tokens: Some(facts.completion_tokens),
                ..NativeResponseMetadata::default()
            },
        );
    }

    /// Register a dispatch identity + its coordinator label, exactly as `begin`
    /// and `label` do on the coordinator thread.
    fn register_identity(
        capture: &RunCapture,
        x_correlation_id: &str,
        session_num: u64,
        terminal: ReplayTerminalStatus,
        facts: &RequestFacts,
    ) {
        capture.identities.borrow_mut().push(CaptureIdentity {
            uuid: facts.uuid,
            x_correlation_id: x_correlation_id.to_string(),
            context: MeasuredContext {
                arrival_ms: facts.arrival_ms,
                input_length: facts.prompt_tokens as usize,
                requested_output_length: 8,
                metadata: RequestMetricMetadata::default(),
                wants_live_record: false,
                consume_record: false,
            },
        });
        capture.labels.borrow_mut().insert(
            facts.uuid,
            CaptureLabel {
                phase: MetricsPhase::Profiling,
                session_num,
                has_credit_timestamp: true,
                terminal,
                start_ns: facts.start_ns,
                end_ns: facts.end_ns,
            },
        );
    }

    fn facts() -> (RequestFacts, RequestFacts, RequestFacts) {
        (
            RequestFacts {
                uuid: Uuid::from_u128(0xA),
                arrival_ms: 1.0,
                token_times_ms: &[5.0, 8.0],
                prompt_tokens: 4,
                completion_tokens: 2,
                start_ns: 2_000_000,
                end_ns: 9_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0xB),
                arrival_ms: 2.0,
                token_times_ms: &[6.0, 10.0, 14.0],
                prompt_tokens: 5,
                completion_tokens: 3,
                start_ns: 3_000_000,
                end_ns: 15_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0xC),
                arrival_ms: 3.0,
                token_times_ms: &[7.0],
                prompt_tokens: 6,
                completion_tokens: 1,
                start_ns: 4_000_000,
                end_ns: 8_000_000,
            },
        )
    }

    /// inputs.json parity: the during-run capture path (fed in arbitrary
    /// dispatch order, with a recycled duplicate turn) and the up-front, dataset-ordered
    /// generation both funnel through the same `write_inputs_json`, so the two files are
    /// byte-identical — dedup per `(conversation_id, turn)` and the conversation-id sort
    /// are order-independent.
    #[test]
    fn inputs_json_up_front_matches_capture_regardless_of_dispatch_order() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let capture = RunCapture::new(
            clock,
            0,
            MetricsConfig::default(),
            false,
            true, // inputs_enabled
            false,
            false,
            false,
        );
        // Distinct canonical bodies per (conversation, turn). "conv-b" is a two-turn
        // conversation; "conv-a" is single-turn.
        let body = |tag: &str| format!(r#"{{"model":"m","tag":"{tag}"}}"#).into_bytes();
        // Feed the during-run capture out of conversation order, with a recycled
        // duplicate (conv-b turn 0 dispatched twice, e.g. via --request-count recycling);
        // the second write must be ignored (first-write-wins dedup).
        capture
            .record_input_payload("conv-b", 0, &body("b0"))
            .unwrap();
        capture
            .record_input_payload("conv-a", 0, &body("a0"))
            .unwrap();
        capture
            .record_input_payload("conv-b", 1, &body("b1"))
            .unwrap();
        capture
            .record_input_payload("conv-b", 0, &body("b0-recycled"))
            .unwrap();
        let capture_sessions = capture.take_input_sessions();

        // The up-front generator emits sessions conversation-id-sorted, each with its
        // turns in order — build the equivalent list directly.
        let parse = |bytes: Vec<u8>| {
            serde_json::from_slice::<Box<serde_json::value::RawValue>>(&bytes).unwrap()
        };
        let up_front = vec![
            InputSession {
                session_id: "conv-a".into(),
                payloads: vec![parse(body("a0"))],
            },
            InputSession {
                session_id: "conv-b".into(),
                payloads: vec![parse(body("b0")), parse(body("b1"))],
            },
        ];

        let dir = tempfile::tempdir().unwrap();
        let capture_path = dir.path().join("inputs_capture.json");
        let up_front_path = dir.path().join("inputs_up_front.json");
        write_inputs_json(&capture_path, &capture_sessions).unwrap();
        write_inputs_json(&up_front_path, &up_front).unwrap();

        assert_eq!(
            std::fs::read(&capture_path).unwrap(),
            std::fs::read(&up_front_path).unwrap(),
            "up-front inputs.json must be byte-identical to the capture-based output"
        );
    }

    /// Worker records arrive per worker, not in global dispatch
    /// order. `RunCapture::finish` must key each record to its identity by uuid,
    /// emit rows in dispatch order, and stamp each record's `request_index` to its
    /// global dispatch ordinal so the downstream re-ingest is collision-free
    /// because a per-worker-local `request_index=Some(0)` collision would
    /// otherwise panic `insert_record_at`.
    #[test]
    fn run_capture_finish_stamps_global_index_and_joins_worker_records() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let capture = RunCapture::new(
            clock.clone(),
            0,
            MetricsConfig::default(),
            false,
            false,
            false,
            false,
            false,
        );
        let (a, b, c) = facts();
        // Dispatch order A, B, C.
        register_identity(&capture, "corr-a", 0, ReplayTerminalStatus::Completed, &a);
        register_identity(&capture, "corr-b", 1, ReplayTerminalStatus::Completed, &b);
        register_identity(&capture, "corr-c", 2, ReplayTerminalStatus::Completed, &c);
        // Two workers: worker0 handled A then C (local slots 0, 1); worker1 handled
        // B (local slot 0). Concatenated drain order is [A, C, B] != dispatch order.
        let worker0 = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        let worker1 = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        drive_worker_request(&worker0, &a);
        drive_worker_request(&worker0, &c);
        drive_worker_request(&worker1, &b);
        let mut drained = worker0.finish_with_records().records;
        drained.extend(worker1.finish_with_records().records);
        assert_eq!(
            drained.iter().map(|(uuid, _)| *uuid).collect::<Vec<_>>(),
            vec![a.uuid, c.uuid, b.uuid],
        );
        // Both worker0 records carry local request_index Some(0)/Some(1) and
        // worker1's is Some(0): a raw re-ingest would collide.
        assert_eq!(drained[0].1.request_index, Some(0));
        assert_eq!(drained[2].1.request_index, Some(0));

        let issued: HashMap<Uuid, i64> = [(a.uuid, 111), (b.uuid, 222), (c.uuid, 333)]
            .into_iter()
            .collect();
        let captured = capture.finish(&issued, drained).unwrap();

        assert_eq!(
            captured.iter().map(|r| r.uuid).collect::<Vec<_>>(),
            vec![a.uuid, b.uuid, c.uuid],
        );
        assert_eq!(
            captured
                .iter()
                .map(|r| r.x_correlation_id.as_str())
                .collect::<Vec<_>>(),
            vec!["corr-a", "corr-b", "corr-c"],
        );
        // Global dispatch ordinal stamped dense 0..N-1 in dispatch order.
        assert_eq!(
            captured
                .iter()
                .map(|r| r.ingest.request_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1), Some(2)],
        );
        // uuid join: each row carries its own record; admit patched per uuid.
        for record in &captured {
            assert_eq!(record.ingest.correlation_id, record.uuid.to_string());
        }
        assert_eq!(captured[0].ingest.admit_ns, Some(111));
        assert_eq!(captured[1].ingest.admit_ns, Some(222));
        assert_eq!(captured[2].ingest.admit_ns, Some(333));
        // Re-ingest is collision-free (no insert_record_at panic) and counts all 3.
        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig::default());
        for record in &captured {
            accumulator.process_record(&record.ingest);
        }
        let summary = accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
        assert_eq!(summary.finite_value(MetricTag::RequestCount), Some(3.0));
    }

    /// The same request set produces a byte-identical re-ingested report whether it
    /// drains from one worker or is split across two. `finish` stamps the
    /// global dispatch ordinal in both cases, so the IEEE-754 fold order is
    /// identical and no worker-count reorder occurs on the runner path.
    #[test]
    fn run_capture_finish_worker_split_matches_single_worker_byte_for_byte() {
        let build = |split: bool| -> (Vec<u8>, Option<f64>) {
            let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
            let config = MetricsConfig::default();
            let capture = RunCapture::new(
                clock.clone(),
                0,
                config.clone(),
                false,
                false,
                false,
                false,
                false,
            );
            let (a, b, c) = facts();
            register_identity(&capture, "corr-a", 0, ReplayTerminalStatus::Completed, &a);
            register_identity(&capture, "corr-b", 1, ReplayTerminalStatus::Completed, &b);
            register_identity(&capture, "corr-c", 2, ReplayTerminalStatus::Completed, &c);
            let drained = if split {
                let worker0 = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                let worker1 = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                drive_worker_request(&worker0, &a);
                drive_worker_request(&worker0, &c);
                drive_worker_request(&worker1, &b);
                let mut drained = worker0.finish_with_records().records;
                drained.extend(worker1.finish_with_records().records);
                drained
            } else {
                let worker = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                drive_worker_request(&worker, &a);
                drive_worker_request(&worker, &b);
                drive_worker_request(&worker, &c);
                worker.finish_with_records().records
            };
            let issued: HashMap<Uuid, i64> = [
                (a.uuid, 1_500_000),
                (b.uuid, 2_500_000),
                (c.uuid, 3_500_000),
            ]
            .into_iter()
            .collect();
            let captured = capture.finish(&issued, drained).unwrap();
            let mut accumulator = MetricsAccumulator::with_config(config);
            for record in &captured {
                accumulator.process_record(&record.ingest);
            }
            let summary =
                accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
            (
                serde_json::to_vec(&summary).unwrap(),
                summary.finite_value(MetricTag::RequestCount),
            )
        };
        let (single_bytes, single_count) = build(false);
        let (split_bytes, split_count) = build(true);
        assert_eq!(single_count, Some(3.0));
        assert_eq!(split_count, Some(3.0));
        assert_eq!(
            single_bytes, split_bytes,
            "worker-split drain must re-ingest byte-identically to a single worker",
        );
    }

    /// The live-results sink must read a non-consuming clone. The
    /// worker returns `snapshot_record` (not `drain_terminal_record`), so the
    /// authoritative record stays in the worker observer and the end-of-run drain
    /// still counts every live-emitted request — a `--live` run cannot undercount.
    #[test]
    fn live_record_snapshot_does_not_consume_the_drain_record() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let observer = NativeMetricsObserver::new(clock, 0, MetricsConfig::default());
        let (a, _b, _c) = facts();
        drive_worker_request(&observer, &a);
        // The live sink emits from this non-consuming clone.
        let live = observer.snapshot_record(a.uuid, 0);
        assert!(live.is_some(), "a terminal request yields a live snapshot");
        // The authoritative record is still present for the end-of-run drain.
        let drained = observer.finish_with_records().records;
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].0, a.uuid);
    }

    /// An identity that fails before any worker observer registers it has no drained
    /// record. `finish` must synthesize an errored
    /// fallback record so `RequestCount`/`ErrorRequestCount` stay exact and the run
    /// does not abort fail-closed on the missing lookup.
    #[test]
    fn run_capture_finish_synthesizes_fallback_for_pre_worker_failures() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let capture = RunCapture::new(
            clock.clone(),
            0,
            MetricsConfig::default(),
            false,
            false,
            false,
            false,
            false,
        );
        let (a, b, _c) = facts();
        register_identity(&capture, "corr-a", 0, ReplayTerminalStatus::Completed, &a);
        // B is dispatched (identity + Failed label) but never reaches a worker.
        register_identity(&capture, "corr-b", 1, ReplayTerminalStatus::Failed, &b);
        let worker = NativeMetricsObserver::new(clock.clone(), 0, MetricsConfig::default());
        drive_worker_request(&worker, &a);
        let drained = worker.finish_with_records().records;
        assert_eq!(drained.len(), 1);

        let issued: HashMap<Uuid, i64> = [(a.uuid, 111), (b.uuid, 222)].into_iter().collect();
        let captured = capture.finish(&issued, drained).unwrap();

        assert_eq!(
            captured.iter().map(|r| r.uuid).collect::<Vec<_>>(),
            vec![a.uuid, b.uuid],
        );
        assert_eq!(
            captured
                .iter()
                .map(|r| r.ingest.request_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(1)],
        );
        assert!(!captured[0].ingest.errored);
        assert!(
            captured[1].ingest.errored,
            "the pre-worker failure is errored"
        );

        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig::default());
        for record in &captured {
            accumulator.process_record(&record.ingest);
        }
        let summary = accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
        // RequestCount counts successes only; the errored fallback lands in
        // ErrorRequestCount, so the total CompletedRequestCount is 2.
        assert_eq!(summary.finite_value(MetricTag::RequestCount), Some(1.0));
        assert_eq!(
            summary.finite_value(MetricTag::ErrorRequestCount),
            Some(1.0)
        );
        assert_eq!(
            summary.finite_value(MetricTag::CompletedRequestCount),
            Some(2.0)
        );
    }

    /// Folding each completed record into the exact accumulator in
    /// completion order, stamping the absolute dispatch `request_index` — and merging
    /// that accumulator into the report yields byte-identical exported results to the
    /// retained-record path's dispatch-order re-ingest, for BOTH the profiling and warmup
    /// windows and including an errored record's accounting. This is the core contract:
    /// exact-fold keeps exact NaN-sparse columns (not the sketch approximation), so the
    /// mid-run fold-and-drop is invisible in the summary.
    #[test]
    fn exact_fold_matches_compatibility_retain_byte_for_byte() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let config = MetricsConfig::default();

        // Four realistic worker-drained records; array index i is dispatch ordinal i.
        let source_facts = [
            RequestFacts {
                uuid: Uuid::from_u128(0x11),
                arrival_ms: 1.0,
                token_times_ms: &[5.0, 8.0],
                prompt_tokens: 4,
                completion_tokens: 2,
                start_ns: 2_000_000,
                end_ns: 9_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0x22),
                arrival_ms: 2.0,
                token_times_ms: &[6.0, 10.0, 14.0],
                prompt_tokens: 5,
                completion_tokens: 3,
                start_ns: 3_000_000,
                end_ns: 15_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0x33),
                arrival_ms: 3.0,
                token_times_ms: &[7.0, 9.0],
                prompt_tokens: 6,
                completion_tokens: 2,
                start_ns: 4_000_000,
                end_ns: 12_000_000,
            },
            RequestFacts {
                uuid: Uuid::from_u128(0x44),
                arrival_ms: 4.0,
                token_times_ms: &[8.0],
                prompt_tokens: 7,
                completion_tokens: 1,
                start_ns: 5_000_000,
                end_ns: 8_000_000,
            },
        ];
        // Per-record coordinator-owned facts: phase (mix of warmup + profiling),
        // session number, admit ns, and whether the record errored (record 3).
        let phases = [
            MetricsPhase::Profiling,
            MetricsPhase::Warmup,
            MetricsPhase::Profiling,
            MetricsPhase::Profiling,
        ];
        let admits = [1_500_000i64, 2_500_000, 3_500_000, 4_500_000];
        let is_errored = [false, false, false, true];

        // Build the drained ingests fresh (dispatch order), applying the errored flag.
        let build_records = || -> Vec<RecordIngest> {
            source_facts
                .iter()
                .enumerate()
                .map(|(i, facts)| {
                    let observer = NativeMetricsObserver::new(clock.clone(), 0, config.clone());
                    drive_worker_request(&observer, facts);
                    let mut ingest = observer
                        .finish_with_records()
                        .records
                        .into_iter()
                        .next()
                        .unwrap()
                        .1;
                    ingest.errored = is_errored[i];
                    ingest
                })
                .collect()
        };

        // Retained-record reference: patch coordinator-owned fields and process them
        // in dispatch order.
        let reference_summary = |phase: MetricsPhase| -> Vec<u8> {
            let mut accumulator = MetricsAccumulator::with_config(config.clone());
            for (i, mut ingest) in build_records().into_iter().enumerate() {
                ingest.phase = phases[i];
                ingest.session_num = i as u64;
                ingest.admit_ns = Some(admits[i]);
                ingest.request_index = Some(i);
                accumulator.process_record(&ingest);
            }
            serde_json::to_vec(&accumulator.export_results(&ExportContext::phase(phase))).unwrap()
        };

        // Fold each record in reverse completion order to validate order-independent
        // absolute-slot placement in the capture's exact
        // accumulator, then merge it into a fresh report accumulator.
        let subject_summary = |phase: MetricsPhase| -> Vec<u8> {
            let capture = RunCapture::new(
                clock.clone(),
                0,
                config.clone(),
                false,
                false,
                false,
                false,
                true,
            );
            assert!(
                capture.exact_fold && !capture.metrics_only,
                "exact-fold keeps EXACT storage, not sketch"
            );
            let records = build_records();
            for i in (0..records.len()).rev() {
                capture
                    .fold_record(
                        records[i].clone(),
                        source_facts[i].uuid,
                        "corr",
                        phases[i],
                        i as u64,
                        Some(admits[i]),
                        Some(i),
                    )
                    .unwrap();
            }
            let (streamed, errored_records) = capture.take_streamed();
            assert_eq!(
                errored_records.len(),
                1,
                "only the errored record is retained; the rest are dropped"
            );
            let mut accumulator = MetricsAccumulator::with_config(config.clone());
            accumulator.merge(&streamed).unwrap();
            serde_json::to_vec(&accumulator.export_results(&ExportContext::phase(phase))).unwrap()
        };

        assert_eq!(
            reference_summary(MetricsPhase::Profiling),
            subject_summary(MetricsPhase::Profiling),
            "profiling window must be byte-identical to the retain path",
        );
        assert_eq!(
            reference_summary(MetricsPhase::Warmup),
            subject_summary(MetricsPhase::Warmup),
            "warmup window must be byte-identical to the retain path",
        );
    }

    /// The exact-fold capture folds each profiling record's per-record OTLP
    /// histogram at completion (`with_otel` + `fold_record`), and `take_otel` yields
    /// the byte-identical accumulator the retain path builds by looping the retained
    /// records post-run — for the same record sequence. Warmup records never
    /// contribute, matching the post-run loop's `phase == Profiling` filter.
    #[test]
    fn fold_record_folds_otel_matching_post_run_loop() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let config = MetricsConfig::default();

        // (uuid, isl, osl, end_ns, phase, errored) — a mix of profiling successes, one
        // errored profiling record, and one warmup record the OTLP fold must ignore.
        let make = |isl: u64, osl: u64, end_ns: i64, phase: MetricsPhase| -> RecordIngest {
            let mut ingest = RecordIngest::minimal(1_000_000, end_ns, phase);
            ingest.first_token_ns = Some(3_000_000);
            ingest.token_arrival_ns = vec![3_000_000, 5_000_000, end_ns];
            ingest.tokens = crate::metrics_core::TokenCounts {
                input: Some(isl),
                output: Some(osl),
                requested_output: Some(osl),
                ..Default::default()
            };
            ingest
        };
        let specs: Vec<(Uuid, RecordIngest, bool)> = vec![
            (
                Uuid::from_u128(0x1),
                make(8, 3, 11_000_000, MetricsPhase::Profiling),
                false,
            ),
            (
                Uuid::from_u128(0x2),
                make(16, 5, 21_000_000, MetricsPhase::Profiling),
                false,
            ),
            (
                Uuid::from_u128(0x3),
                make(64, 1, 4_000_000, MetricsPhase::Profiling),
                true,
            ),
            (
                Uuid::from_u128(0x4),
                make(8, 3, 11_000_000, MetricsPhase::Warmup),
                false,
            ),
        ];

        // Exact-fold path: fold each record at completion with native OTLP enabled.
        let capture = RunCapture::new(
            clock.clone(),
            0,
            config.clone(),
            false,
            false,
            false,
            false,
            true,
        )
        .with_otel(true);
        for (i, (uuid, ingest, errored)) in specs.iter().enumerate() {
            let mut ingest = ingest.clone();
            ingest.errored = *errored;
            let phase = ingest.phase;
            capture
                .fold_record(ingest, *uuid, "corr", phase, i as u64, None, Some(i))
                .unwrap();
        }
        let folded = capture.take_otel().expect("otel enabled");

        // Retain path: build the equivalent stamped records and fold the profiling
        // subset via the post-run loop, in the same order.
        let mut post_run = OtelRecordAccumulator::new();
        for (i, (uuid, ingest, errored)) in specs.iter().enumerate() {
            let mut ingest = ingest.clone();
            ingest.errored = *errored;
            ingest.session_num = i as u64;
            ingest.request_index = Some(i);
            if ingest.phase == MetricsPhase::Profiling {
                let captured = CapturedRecord {
                    uuid: *uuid,
                    x_correlation_id: "corr".into(),
                    output: CapturedModelOutput::default(),
                    raw: None,
                    ingest,
                };
                observe_otel_record(&mut post_run, &captured, &config);
            }
        }

        assert!(
            !folded.is_empty(),
            "profiling records populate the histograms"
        );
        assert_eq!(
            folded, post_run,
            "fold-at-completion OTLP must equal the post-run-loop OTLP for the same sequence"
        );
    }

    /// Exact-fold sets the fold-and-drop flags (so the worker consumes each record out
    /// of its observer) and assigns dense `0..N` dispatch ordinals at begin, consumed
    /// once at completion. A plain exact (retain) capture folds nothing.
    #[test]
    fn exact_fold_flags_and_dense_dispatch_ordinals() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let config = MetricsConfig::default();
        let capture = RunCapture::new(
            clock.clone(),
            0,
            config.clone(),
            false,
            false,
            false,
            false,
            true,
        );
        assert!(
            capture.folds_records(),
            "exact-fold is a fold-and-drop mode"
        );
        assert!(
            capture.wants_live_record(),
            "the worker must return each record so the fold can consume it"
        );
        let a = Uuid::from_u128(0xA1);
        let b = Uuid::from_u128(0xB2);
        let c = Uuid::from_u128(0xC3);
        assert_eq!(capture.assign_fold_ordinal(a), 0);
        assert_eq!(capture.assign_fold_ordinal(b), 1);
        assert_eq!(capture.assign_fold_ordinal(c), 2);
        assert_eq!(capture.take_fold_ordinal(b), Some(1));
        assert_eq!(capture.take_fold_ordinal(b), None, "consumed exactly once");
        assert_eq!(capture.take_fold_ordinal(a), Some(0));
        assert_eq!(capture.take_fold_ordinal(c), Some(2));

        // The default exact (retain) capture folds nothing and needs no live record.
        let retain = RunCapture::new(clock, 0, config, false, false, false, false, false);
        assert!(!retain.folds_records());
        assert!(!retain.wants_live_record());
    }

    /// The exact-fold gate accepts a clean scheduled metrics run and rejects every
    /// remaining disqualifier independently. Since the thread-per-core sharded
    /// arm (`shardable`) is accepted, so a metrics-only sharded run selects
    /// exact-fold. A cellular child (`is_cellular`) is likewise accepted for
    /// a metrics-only run (it ships its folded store to the controller). A cellular run
    /// that also wants an unstreamable per-record artifact rejects via
    /// `wants_per_record_artifacts`. Graph datasets never reach this scheduled path.
    #[test]
    fn exact_fold_gate_accepts_clean_run_and_rejects_disqualifiers() {
        // A clean single-thread scheduled metrics-only run: every disqualifier false.
        let clean = ExactFoldInputs {
            sketch_mode: false,
            shardable: false,
            is_cellular: false,
            has_accuracy: false,
            wants_adaptive_record: false,
            has_live_sink: false,
            has_heartbeat: false,
            wants_per_record_artifacts: false,
        };
        assert!(
            exact_fold_eligible(clean),
            "a clean single-thread scheduled metrics-only run is eligible"
        );
        // Each disqualifier, toggled on in isolation, must reject the fold.
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                sketch_mode: true,
                ..clean
            }),
            "sketch mode has its own fold path"
        );
        // The thread-per-core sharded arm does not disqualify: a
        // metrics-only `workers > 1` run folds each shard into its own exact
        // accumulator with a LOCAL-dense fold ordinal and merges via `append_store`.
        assert!(
            exact_fold_eligible(ExactFoldInputs {
                shardable: true,
                ..clean
            }),
            "the sharded arm selects exact-fold (local-dense fold)"
        );
        // A metrics-only cellular child does not disqualify: it folds into
        // its own dense-LOCAL exact accumulator and ships the folded store to the
        // controller (`CellMessage::StorePartition`), which appends every cell's store.
        assert!(
            exact_fold_eligible(ExactFoldInputs {
                is_cellular: true,
                ..clean
            }),
            "a metrics-only cellular child selects exact-fold (store shipping)"
        );
        // An unstreamable per-record artifact keeps a cellular run on retention.
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                is_cellular: true,
                wants_per_record_artifacts: true,
                ..clean
            }),
            "cellular + per-record artifacts stays on retain (deferred)"
        );
        // Records, raw data, CSV, Parquet, and outputs stream and merge across shards.
        // `wants_per_record_artifacts` only flags
        // the residual unstreamable cases (live-reply-multiturn inputs.json retain, or a
        // lite-build Parquet request), which STILL reject even when sharded because
        // neither can be produced by the fold-and-drop path.
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                shardable: true,
                wants_per_record_artifacts: true,
                ..clean
            }),
            "an unstreamable per-record artifact (inputs-retain / lite parquet) stays on retain even sharded"
        );
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                has_accuracy: true,
                ..clean
            }),
            "accuracy stays on the retain path"
        );
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                wants_adaptive_record: true,
                ..clean
            }),
            "adaptive samples retained records"
        );
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                has_live_sink: true,
                ..clean
            }),
            "the live sink reads per-record clones"
        );
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                has_heartbeat: true,
                ..clean
            }),
            "the heartbeat lane reads per-record clones"
        );
        assert!(
            !exact_fold_eligible(ExactFoldInputs {
                wants_per_record_artifacts: true,
                ..clean
            }),
            "per-record artifacts still read the retained records"
        );
    }

    /// A metrics-only graph run and graph runs requesting streamable per-record
    /// artifacts select exact-fold because [`RecordArtifactLane`] writes each row before
    /// the fold drops it. Only sketch mode (its own bounded
    /// fold, no store-merge path) — and, on a LITE build without the `parquet` feature, a
    /// requested Parquet sidecar (unstreamable there) — keep the run on retain. A bare
    /// `inputs_path` — always projected by `rust_wire` but never written by the graph
    /// path — does NOT disqualify.
    #[test]
    fn graph_exact_fold_gate_streams_artifacts_and_rejects_sketch_and_lite_parquet() {
        use crate::engine::protocol::ArtifactSpec;

        // The production graph caller wires none of the retain-forcing consumers:
        // wires NONE of the retain-forcing per-record consumers, so it feeds the shared
        // `exact_fold_eligible` gate `false` for them and `wants_per_record_artifacts`
        // with `inputs_need_retain = false` (the graph path never writes inputs.json).
        let graph_gate = |artifacts: &ArtifactSpec, sketch: bool| {
            exact_fold_eligible(ExactFoldInputs {
                sketch_mode: sketch,
                shardable: false,
                is_cellular: false,
                has_accuracy: false,
                wants_adaptive_record: false,
                has_live_sink: false,
                has_heartbeat: false,
                wants_per_record_artifacts: wants_per_record_artifacts(artifacts, false),
            })
        };

        // Metrics-only (no per-record file artifacts), not sketch → eligible.
        let metrics_only = ArtifactSpec::default();
        assert!(
            graph_gate(&metrics_only, false),
            "a metrics-only graph run is exact-fold-eligible"
        );

        // `inputs_path` is a no-op on the graph path (never written), so it must not
        // disqualify — otherwise every graph run (rust_wire always sets it) would retain.
        assert!(
            graph_gate(
                &ArtifactSpec {
                    inputs_path: Some("inputs.json".into()),
                    ..ArtifactSpec::default()
                },
                false,
            ),
            "a bare inputs_path (never written on the graph path) does not disqualify"
        );

        // Sketch mode has its own bounded fold and no store-merge path → rejected.
        assert!(
            !graph_gate(&metrics_only, true),
            "sketch mode is not eligible for the graph exact-fold store path"
        );

        // Each streamable per-record artifact stays on exact-fold because the graph
        // fold-drop pass writes each row through the lane. Parquet is
        // streamable only under the `parquet` feature; on a lite build it disqualifies (see
        // the dedicated assertion below), so it is excluded from this always-eligible set.
        for (label, artifacts) in [
            (
                "records_path",
                ArtifactSpec {
                    records_path: Some("profile_export.jsonl".into()),
                    ..ArtifactSpec::default()
                },
            ),
            (
                "records_csv_path",
                ArtifactSpec {
                    records_csv_path: Some("profile_export_records.csv".into()),
                    ..ArtifactSpec::default()
                },
            ),
            (
                "raw_path",
                ArtifactSpec {
                    raw_path: Some("profile_export_raw.jsonl".into()),
                    ..ArtifactSpec::default()
                },
            ),
            (
                "outputs_path",
                ArtifactSpec {
                    outputs_path: Some("outputs.json".into()),
                    ..ArtifactSpec::default()
                },
            ),
        ] {
            assert!(
                graph_gate(&artifacts, false),
                "a requested {label} now streams through the lane and stays on exact-fold"
            );
        }

        // Parquet: streamable (and so eligible) under the `parquet` feature; on a lite
        // build it cannot stream, so it keeps the graph run on the retain path.
        let with_parquet = ArtifactSpec {
            records_parquet_path: Some("profile_export.parquet".into()),
            ..ArtifactSpec::default()
        };
        #[cfg(feature = "parquet")]
        assert!(
            graph_gate(&with_parquet, false),
            "a parquet build streams the Parquet sidecar through the lane (exact-fold)"
        );
        #[cfg(not(feature = "parquet"))]
        assert!(
            !graph_gate(&with_parquet, false),
            "a lite build cannot stream Parquet, so it disqualifies graph exact-fold"
        );
    }

    /// Driving a record slice through the [`RecordArtifactLane`]
    /// (the graph fold-drop pass's per-record streaming writer) produces the same rows as
    /// the batch `write_graph_artifacts` (which delegates to `write_records_jsonl` /
    /// `write_raw_records_jsonl` / `write_records_csv` / `write_outputs_json`). Completion-
    /// order streaming ⇒ the JSONL/raw row SET equals the batch row SET, the CSV data-row
    /// SET matches (shared header), and the `outputs.json` `data` arrays are set-equal.
    #[test]
    fn graph_lane_matches_batch_write_graph_artifacts_row_set() {
        use crate::metrics_core::{Phase, RecordIngest, TokenCounts};
        use std::collections::BTreeSet;

        let config = MetricsConfig::default();

        // A representative graph slice: profiling successes with output text (out of
        // completion order), one profiling cancel (HTTP 499), and a warmup record that
        // `outputs.json` must exclude.
        let make = |session: u64, turn: u32, text: &str, phase: Phase, canceled: bool| {
            let mut ingest = RecordIngest::minimal(1_000_000, 11_000_000, phase);
            ingest.session_num = session;
            ingest.turn_index = turn;
            ingest.conversation_id = Some(format!("conversation-{session}"));
            ingest.canceled = canceled;
            if !canceled {
                ingest.first_token_ns = Some(6_000_000);
                ingest.token_arrival_ns = vec![6_000_000, 8_000_000, 11_000_000];
                ingest.tokens = TokenCounts {
                    input: Some(8),
                    output: Some(3),
                    requested_output: Some(3),
                    ..TokenCounts::default()
                };
            }
            CapturedRecord {
                uuid: Uuid::from_u128(u128::from(session) * 10 + u128::from(turn)),
                x_correlation_id: format!("session-{session}"),
                output: CapturedModelOutput::from_parts(text, Some(text), None),
                raw: None,
                ingest,
            }
        };
        let records = vec![
            make(2, 1, "second answer", Phase::Profiling, false),
            make(1, 0, "first answer", Phase::Profiling, false),
            make(3, 0, "", Phase::Profiling, true),
            make(2, 0, "middle answer", Phase::Profiling, false),
            make(9, 0, "warmup answer", Phase::Warmup, false),
        ];

        // Lane path: one write per completed record then finish, writing into dir A.
        let lane_dir = tempfile::tempdir().unwrap();
        let lane = RecordArtifactLane::new(
            Some(lane_dir.path().join("profile_export.jsonl")),
            Some(lane_dir.path().join("profile_export_raw.jsonl")),
            Some(lane_dir.path().join("profile_export_records.csv")),
            None,
            Some(lane_dir.path().join("outputs.json")),
            false,
        )
        .unwrap()
        .expect("lane requested four artifacts");
        for record in &records {
            lane.write(record, &config).unwrap();
        }
        lane.finish().unwrap();

        // Batch path: the exact writers `write_graph_artifacts` invokes, into dir B.
        let batch_dir = tempfile::tempdir().unwrap();
        write_records_jsonl(
            &batch_dir.path().join("profile_export.jsonl"),
            &records,
            &config,
            false,
        )
        .unwrap();
        write_raw_records_jsonl(&batch_dir.path().join("profile_export_raw.jsonl"), &records)
            .unwrap();
        write_records_csv(
            &batch_dir.path().join("profile_export_records.csv"),
            &records,
            &config,
            false,
        )
        .unwrap();
        write_outputs_json(&batch_dir.path().join("outputs.json"), &records, &config).unwrap();

        let line_set = |dir: &Path, name: &str| -> BTreeSet<String> {
            std::fs::read_to_string(dir.join(name))
                .unwrap_or_default()
                .lines()
                .filter(|l| !l.is_empty())
                .map(str::to_string)
                .collect()
        };
        for name in ["profile_export.jsonl", "profile_export_raw.jsonl"] {
            assert_eq!(
                line_set(lane_dir.path(), name),
                line_set(batch_dir.path(), name),
                "lane vs batch JSONL row SET mismatch for {name}"
            );
        }
        // CSV: same header line, same data-row SET.
        assert_eq!(
            line_set(lane_dir.path(), "profile_export_records.csv"),
            line_set(batch_dir.path(), "profile_export_records.csv"),
            "lane vs batch CSV row SET mismatch"
        );
        // outputs.json: set-equal `data` arrays (sorted by session/turn).
        let outputs_data = |dir: &Path| -> serde_json::Value {
            let mut doc: serde_json::Value =
                serde_json::from_slice(&std::fs::read(dir.join("outputs.json")).unwrap()).unwrap();
            let data = doc["data"].as_array_mut().unwrap();
            data.sort_by_key(|row| {
                (
                    row["session_num"].as_u64().unwrap_or(0),
                    row["turn_index"].as_u64().unwrap_or(0),
                )
            });
            doc
        };
        assert_eq!(
            outputs_data(lane_dir.path()),
            outputs_data(batch_dir.path()),
            "lane vs batch outputs.json data SET mismatch"
        );
    }

    /// The graph exact-fold tripwire reflects the
    /// per-record consumers the graph path STRUCTURALLY wires, not the raw OTEL/heartbeat
    /// config flags. A legitimate summary-only graph run with an OTEL metrics URL sets
    /// `native_otel_enabled = true` (the Python frontend does not reject it on the graph
    /// path) and selects exact-fold. The graph path builds no OTLP accumulator, no
    /// heartbeat lane, and — via `validate_graph_request` — no live sink, so the drop is
    /// safe and the debug tripwire must not fire.
    #[test]
    fn graph_exact_fold_drop_safe_ignores_native_otel_and_heartbeat_flags() {
        // No live-streaming consumer wired (validate_graph_request guarantees this) →
        // the drop is safe regardless of the OTEL / heartbeat config, which the graph
        // path never turns into a per-record consumer.
        assert!(
            graph_exact_fold_drop_is_safe(false),
            "graph exact-fold drop is safe when no live-streaming consumer is wired"
        );

        // Reconstruct the exact boolean the debug_assert evaluates for an exact-fold graph
        // run that has an OTEL metrics URL (native_otel_enabled = true) and, per env,
        // heartbeat enabled — but no live sink. It must be true so the assert does not
        // panic.
        let graph_exact_fold = true;
        let native_otel_enabled = true;
        let heartbeat_enabled_by_env = HeartbeatLane::enabled_by_env();
        let live_streaming_wired = false;
        // These flags do not represent graph per-record consumers.
        let _ = (native_otel_enabled, heartbeat_enabled_by_env);
        assert!(
            !graph_exact_fold || graph_exact_fold_drop_is_safe(live_streaming_wired),
            "an exact-fold graph run with native_otel_enabled must not trip the tripwire"
        );

        // The real invariant still bites: a wired live-streaming consumer would trip it.
        assert!(
            !graph_exact_fold_drop_is_safe(true),
            "a wired live-streaming record consumer must still fail the drop-safety check"
        );
    }

    /// gate wiring: a run requesting records/raw/CSV/parquet/outputs (but no
    /// inputs.json retain) does NOT set `wants_per_record_artifacts` on a Parquet build,
    /// so a `workers > 1` run with every streamed artifact is exact-fold-eligible (the
    /// per-shard lanes + coordinator concat handle them). The residual retain triggers —
    /// live-reply inputs.json and (lite build) Parquet — still flag it.
    #[test]
    fn wants_per_record_artifacts_streamed_set_is_not_a_disqualifier() {
        use crate::engine::protocol::ArtifactSpec;

        let streamed = ArtifactSpec {
            records_path: Some(PathBuf::from("profile_export.jsonl")),
            raw_path: Some(PathBuf::from("profile_export_raw.jsonl")),
            records_csv_path: Some(PathBuf::from("profile_export_records.csv")),
            records_parquet_path: Some(PathBuf::from("profile_export.parquet")),
            outputs_path: Some(PathBuf::from("outputs.json")),
            inputs_path: Some(PathBuf::from("inputs.json")),
            trace: false,
            dataset_analysis_path: None,
            ..Default::default()
        };
        // `inputs_need_retain == false`: inputs.json is up-front-able for this shape.
        #[cfg(feature = "parquet")]
        {
            assert!(
                !wants_per_record_artifacts(&streamed, false),
                "records/raw/CSV/parquet/outputs stream+merge — not a disqualifier on a Parquet build"
            );
            // Sharded runs remain eligible with all streamable artifacts.
            assert!(exact_fold_eligible(ExactFoldInputs {
                sketch_mode: false,
                shardable: true,
                is_cellular: false,
                has_accuracy: false,
                wants_adaptive_record: false,
                has_live_sink: false,
                has_heartbeat: false,
                wants_per_record_artifacts: wants_per_record_artifacts(&streamed, false),
            }));
        }
        // Live-reply inputs.json that cannot be reproduced up front still disqualifies.
        assert!(wants_per_record_artifacts(&streamed, true));
        // A lite build cannot stream Parquet, so a requested sidecar still disqualifies.
        #[cfg(not(feature = "parquet"))]
        assert!(wants_per_record_artifacts(&streamed, false));
    }
}
