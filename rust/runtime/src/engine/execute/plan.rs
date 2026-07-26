// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native run specification, plan types, and validation for one resolved benchmark run.

use super::*;

/// Admission resources shared by every phase in one scheduled run.
///
/// The same value is consumed by online HTTP and in-process offline adapters,
/// keeping cross-phase slot debt and adaptive actuator ownership above the
/// backend/clock seam.
pub(crate) struct NativeScheduledResources {
    pub(crate) session: Option<Rc<SlotPool>>,
    pub(crate) prefill: Option<Rc<SlotPool>>,
    pub(crate) phase: Rc<dyn ScheduledPhaseResources>,
    /// Cell-shared request-rate gate for this phase under `global`/`global-hop`
    /// dispatch (`None` for `sharded` dispatch, non-rate phases, and the
    /// single-thread coordinator path). When present, a request-rate workload
    /// paces against it instead of its locally-sliced `intervals`, so aggregate
    /// arrival rate across all `W` worker threads matches the global rate
    /// exactly — the rate-pacing analogue of the `session` `GlobalSlotPool`.
    pub(crate) rate: Option<Arc<crate::timing::GlobalRateGate>>,
}

/// Select one phase's admission resources for one worker thread under
/// `global`/`global-hop` dispatch.
///
/// The ordinary cross-phase-persistent `shared_resources` (one `SlotPool`
/// reused, resized via `set_limit`, across every phase in the run so a
/// seamless warmup->profiling handoff keeps its session guards live) is a
/// `Sharded`-mode-only property: a `Local` `SlotPool`'s limit can be resized
/// in place, but a `Global`-backed one cannot swap the `Arc<GlobalSlotPool>`
/// it draws from without breaking in-flight guards. So when this phase both
/// (a) authors a `concurrency` cap and (b) the cell built a shared gate for
/// it (`ShardedShared::global_admission`), this thread gets its OWN
/// `SlotPool::new_global` over that gate's `Arc<GlobalSlotPool>` — the same
/// `Arc` every other worker thread in the cell holds — instead of continuing
/// to draw from `shared_resources`'s persistent local pool. Every other case
/// (Sharded dispatch, or a phase with no concurrency cap, e.g.
/// `fixed_schedule`) falls back to `shared_resources` unchanged.
///
/// A known approximation: under `Global`/`GlobalHop`, a seamless
/// warmup->profiling transition that carries session guards across the
/// boundary switches from one phase's `GlobalSlotPool` to the next phase's
/// (rather than resizing one persistent pool in place, as `Sharded` does).
pub(crate) fn phase_scheduled_resources(
    phase: &PhaseSpec,
    shared: &ShardedShared,
    shared_resources: &NativeScheduledResources,
) -> Result<NativeScheduledResources> {
    // A phase's shared rate gate is independent of its concurrency gate: a pure
    // request-rate phase authors a `rate` but no `concurrency` cap, and a pure
    // concurrency-burst phase authors the reverse. Resolve the rate gate first
    // so it attaches even when this phase falls through to the shared local
    // concurrency pool below.
    let global_rate = shared
        .global_admission
        .as_ref()
        .and_then(|admission| admission.rate.get(&metrics_phase(phase).ok()?).cloned());
    let global_pool = shared.global_admission.as_ref().and_then(|admission| {
        admission
            .concurrency
            .get(&metrics_phase(phase).ok()?)
            .cloned()
    });
    let Some(global_pool) = global_pool else {
        return Ok(NativeScheduledResources {
            session: shared_resources.session.clone(),
            prefill: shared_resources.prefill.clone(),
            phase: shared_resources.phase.clone(),
            rate: global_rate,
        });
    };
    let session = Some(Rc::new(SlotPool::new_global(global_pool)));
    let phase_resources: Rc<dyn ScheduledPhaseResources> = Rc::new(SlotPoolPhaseResources::new(
        session.clone(),
        shared_resources.prefill.clone(),
    ));
    Ok(NativeScheduledResources {
        session,
        prefill: shared_resources.prefill.clone(),
        phase: phase_resources,
        rate: global_rate,
    })
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
        // The single-thread coordinator path never builds a `GlobalAdmission`;
        // rate pacing there is always the local per-phase `intervals` grid.
        rate: None,
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
    /// Admission strategy for `workers>1` scheduled execution (`runtime.dispatch`).
    /// Consumed only by the sharded thread-per-core path
    /// ([`ShardedShared::dispatch_mode`]); the single-thread coordinator path
    /// has no cross-thread admission concern regardless of this value.
    pub(crate) dispatch_mode: DispatchMode,
    /// Worker-assignment policy for the [`DispatchMode::GlobalHop`] hop executor
    /// (`runtime.hop_routing`). `None` leaves the hop executor on its
    /// [`crate::engine::protocol::HopRouting::default`] (`RoundRobin`) placement.
    /// Inert under any other dispatch mode or `workers == 1`.
    pub(crate) hop_routing: Option<crate::engine::protocol::HopRouting>,
}

/// Protocol-neutral retention of one run's already decoded sidecar inputs.
pub(crate) enum NativeSidecarPlan {
    /// Protocol-v2 direct adapter outputs retained through execution.
    Prepared(Arc<PreparedSidecarInputs>),
}

impl NativeSidecarPlan {
    pub(crate) fn content_server(&self) -> Result<Option<&ContentServerSpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(CONTENT_SERVER_SIDECAR_ID)
    }

    pub(crate) fn gpu_telemetry(&self) -> Result<Option<&GpuTelemetrySpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(GPU_TELEMETRY_SIDECAR_ID)
    }

    pub(crate) fn network_latency(&self) -> Result<Option<&NetworkLatencySpec>> {
        let Self::Prepared(inputs) = self;
        inputs.get(NETWORK_LATENCY_SIDECAR_ID)
    }

    pub(crate) fn server_metrics(&self) -> Result<Option<&ServerMetricsSpec>> {
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
    pub(crate) fn default_urls(&self) -> Result<&[String]> {
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
    pub(crate) registry: EndpointRegistry,
    pub(crate) profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
}

impl NativePreparedEndpointTableFactory {
    pub(crate) fn new(
        registry: EndpointRegistry,
        profiles: Arc<Vec<ValidatedEndpointProfileV2>>,
    ) -> Self {
        Self { registry, profiles }
    }

    pub(crate) fn prepare_table(&self) -> Result<PreparedEndpointTable> {
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

    pub(crate) fn reference(&self, profile_id: &str) -> Result<PreparedEndpointReference> {
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
    pub(crate) fn validate(&self) -> Result<()> {
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
    pub(crate) fn validate(&self) -> Result<()> {
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
    /// Ignore recorded trace inter-message/inter-request delays: the graph
    /// executor fires every node as soon as its inputs are ready.
    pub(crate) ignore_trace_delays: bool,
}

/// A side-channel subsystem that samples over the profiling window requires
/// at least one profiling phase to anchor to.
pub(crate) fn require_single_profiling_phase(
    request: &NativeRunSpec,
    subsystem: &str,
) -> Result<()> {
    ensure!(
        request
            .phases
            .iter()
            .any(|phase| !phase.common().exclude_from_results),
        "{subsystem} requires at least one profiling phase"
    );
    Ok(())
}

pub(crate) fn validate_plan(request: &NativeRunSpec) -> Result<()> {
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
            .any(|phase| !phase.common().exclude_from_results),
        "a profiling phase is required"
    );
    for (index, phase) in request.phases.iter().enumerate() {
        let common = phase.common();
        ensure!(
            common.exclude_from_results == common.is_warmup(),
            "phase {index} ({:?}) exclude_from_results disagrees with its semantic kind",
            common.name
        );
        if let Some(series) = &common.rate_series {
            ensure!(
                series.points.len() >= 2,
                "phase {index} rate_series requires at least two points"
            );
            ensure!(
                !matches!(phase, PhaseSpec::UserCentric { .. }),
                "user-centric phases do not support rate_series"
            );
        }
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
                    PhaseSpec::UserCentric { .. } | PhaseSpec::FixedSchedule { .. } | PhaseSpec::AgenticReplay { .. }
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
pub(crate) fn wants_per_record_artifacts(
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
pub(crate) fn dataset_supports_up_front_inputs(dataset: &Dataset) -> bool {
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
pub(crate) fn build_up_front_input_sessions(
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
    // Move the shared body handles through verbatim; each is validated once into
    // a borrowed RawValue at write time (`write_inputs_json`), not copied here.
    Ok(sessions
        .into_iter()
        .map(|session| InputSession {
            session_id: session.session_id,
            payloads: session.payloads,
        })
        .collect())
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
pub(crate) fn exact_fold_eligible(inputs: ExactFoldInputs) -> bool {
    !inputs.sketch_mode
        && !inputs.has_accuracy
        && !inputs.wants_adaptive_record
        && !inputs.has_live_sink
        && !inputs.has_heartbeat
        && !inputs.wants_per_record_artifacts
}

// Inputs considered by exact-fold eligibility.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ExactFoldInputs {
    /// Sketch storage mode: has its own bounded t-digest fold path.
    pub(crate) sketch_mode: bool,
    /// Recorded for call-site clarity; does not affect eligibility.
    #[allow(dead_code)]
    pub(crate) shardable: bool,
    /// Recorded for call-site clarity; does not affect eligibility.
    #[allow(dead_code)]
    pub(crate) is_cellular: bool,
    /// A static/stateful accuracy run: retains records for post-run scoring.
    pub(crate) has_accuracy: bool,
    /// Adaptive scale: samples retained per-turn records per control window.
    pub(crate) wants_adaptive_record: bool,
    /// A Python live sink is attached: reads a per-record clone the fold drops.
    pub(crate) has_live_sink: bool,
    /// The single-process cellular heartbeat lane is enabled: also reads the
    /// per-record clone.
    pub(crate) has_heartbeat: bool,
    /// A per-record file artifact (records/raw/CSV/parquet on a lite build) or the
    /// during-run inputs.json capture still needs the retained records
    /// ([`wants_per_record_artifacts`]).
    pub(crate) wants_per_record_artifacts: bool,
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
pub(crate) fn graph_exact_fold_drop_is_safe(live_streaming_wired: bool) -> bool {
    !live_streaming_wired
}

#[cfg(test)]
mod tests {

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
