// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Top-level native execution entrypoints, run summaries, and artifact writers.

use std::collections::BTreeMap;

use super::*;
use crate::engine::graph_phase_runtime::GraphTraceTerminalCallback;
use crate::graph::driver::TraceProgramDriverFactory;
use crate::graph::replay::{
    CacheIsolationPolicy, CompletedReplayTask, ReplayCheckpoint, ReplayProvenance,
    ReplayRunIdentity, ReplayTaskClassification, redact_replay_provenance,
};

struct PreparedNativeExecution<'a> {
    clock: Rc<dyn Clock>,
    real_clock_anchor: RealClockAnchor,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    graph_placement: &'a dyn GraphPlacementFactory,
    trace_driver: Arc<dyn TraceProgramDriverFactory>,
    control_plane_http: Arc<dyn crate::engine::control_plane_http::ControlPlaneHttpProviderFactory>,
    registry: &'a AIPerfRegistry,
    sidecar_factory: &'a dyn NativeSidecarResourceFactory,
    readiness: Option<(
        Box<dyn PreparedOnlineReadiness>,
        &'a dyn ReadinessTransportFactory,
    )>,
}

pub(crate) struct NativeExecutionDependencies<'a> {
    pub(crate) transport_factory: Arc<dyn RequestExecutorFactory>,
    pub(crate) graph_placement: &'a dyn GraphPlacementFactory,
    pub(crate) trace_driver: Arc<dyn TraceProgramDriverFactory>,
    pub(crate) control_plane_http:
        Arc<dyn crate::engine::control_plane_http::ControlPlaneHttpProviderFactory>,
    pub(crate) registry: &'a AIPerfRegistry,
    pub(crate) sidecar_factory: &'a dyn NativeSidecarResourceFactory,
    pub(crate) readiness: Option<(
        Box<dyn PreparedOnlineReadiness>,
        &'a dyn ReadinessTransportFactory,
    )>,
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
pub(crate) fn execute_prepared_native_plan_uncommitted_with_runtime_factories(
    mut plan: NativeRunSpec,
    dependencies: NativeExecutionDependencies<'_>,
) -> Result<NativeReport> {
    let NativeExecutionDependencies {
        transport_factory,
        graph_placement,
        trace_driver,
        control_plane_http,
        registry,
        sidecar_factory,
        readiness,
    } = dependencies;
    validate_plan(&plan)?;
    // Thread-per-core sharding hands each worker a disjoint conversation subset via a
    // modulo partition (`two_level_partition`). Its modulus is the GLOBAL sub-cell
    // grid width `cells * workers` (a thread of cell `c` owns instances
    // `i % (cells*workers) == c + cells*thread`), so a thread's conversation source is
    // empty unless a cell-local conversation lands in its residue class. When the grid
    // is wider than the cell's conversation count, the surplus threads receive an empty
    // subset: a request-bounded phase then fails building its request-rate workload
    // ("conversation dataset cannot be empty"), and a rate phase later fails issuing a
    // new session ("... is not sampleable").
    //
    // Cap `workers` so the FULL grid `cells * workers` fits within the cell's
    // conversation count — i.e. `workers <= conversations / cells` — so every one of
    // the grid's threads owns at least one conversation and recycles it to fill its
    // budget share (matching the Python frontend, which recycles a small dataset to
    // fill request_count). A single-process run (`cells == 1`) reduces to the plain
    // `workers <= conversations` cap. Graph and static-accuracy plans partition
    // differently and are left untouched.
    if let NativeDatasetPlan::PreparedLinear(prepared) = &plan.dataset {
        let conversations = prepared.dataset.conversations().len();
        // The sub-cell grid width is `cells * workers`; `cells` comes from the same
        // `AIPERF_CELL_COUNT` env the two-level partition reads (default 1).
        let cells = ModuloCellPartition::from_env()
            .map(|partition| partition.cell_count())
            .unwrap_or(1)
            .max(1) as usize;
        let max_workers = (conversations / cells).max(1);
        if conversations > 0 && plan.workers > max_workers {
            plan.workers = max_workers;
        }
    }
    // Adaptive scale must observe the run's AGGREGATE load through ONE controller
    // on ONE control knob. Under thread-per-core sharding each of W workers would
    // run its own controller over a 1/W load slice — none reaching a coherent
    // saturation decision — and all W would race the same adaptive artifact files.
    // Pin adaptive runs to a single worker so the controller is global and its
    // artifact writer is unique.
    if plan
        .phases
        .iter()
        .any(|phase| phase.common().adaptive_scale.is_some())
    {
        plan.workers = 1;
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
            PreparedNativeExecution {
                clock: clock_for_body,
                real_clock_anchor,
                transport_factory,
                graph_placement: placement,
                trace_driver,
                control_plane_http: control_plane_http.clone(),
                registry,
                sidecar_factory,
                readiness,
            },
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

pub(crate) fn materialize_user_files(
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

pub(crate) struct PreparedAccuracy {
    pub(crate) evaluator: Box<dyn AccuracyEvaluator>,
    pub(crate) loaded: EvaluatorLoadResult,
    pub(crate) dataset: AccuracyDataset,
    pub(crate) processor: Rc<AccuracyRecordProcessor>,
    pub(crate) tokenizer: Arc<dyn TextTokenizer>,
}

async fn prepare_and_execute_native(
    request: NativeRunSpec,
    execution: PreparedNativeExecution<'_>,
) -> Result<NativeReport> {
    let PreparedNativeExecution {
        clock,
        real_clock_anchor,
        transport_factory,
        graph_placement,
        trace_driver,
        control_plane_http,
        registry,
        sidecar_factory,
        readiness,
    } = execution;
    if request.dataset.is_graph() {
        validate_graph_request(&request)?;
        if let NativeDatasetPlan::Graph(graph) = &request.dataset {
            for program in &graph.input.programs {
                trace_driver
                    .capabilities(&program.driver)
                    .map_err(|error| anyhow!(error.to_string()))?;
            }
        }
    }
    let mut accuracy = prepare_static_accuracy(&request).await?;
    let mut sidecars = match sidecar_factory
        .prepare(&request, clock.clone(), real_clock_anchor)
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
    if let Err(error) = attach_server_profiler_hook(
        &request,
        &mut sidecars,
        clock.clone(),
        control_plane_http.as_ref(),
    ) {
        sidecars.shutdown_run_resources().await;
        return finish_accuracy_lifecycle(
            Err(error.context("preparing endpoint-local server profiler hook")),
            accuracy.as_mut(),
        )
        .await;
    }
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
        trace_driver,
        registry,
    )
    .await;
    sidecars.shutdown_run_resources().await;
    finish_accuracy_lifecycle(result, accuracy.as_mut()).await
}

fn attach_server_profiler_hook(
    request: &NativeRunSpec,
    sidecars: &mut PreparedNativeSidecarResources,
    clock: Rc<dyn Clock>,
    control_plane_http: &dyn crate::engine::control_plane_http::ControlPlaneHttpProviderFactory,
) -> Result<()> {
    let NativeEndpointPlan::Prepared(profiles) = &request.endpoint;
    let profile = default_prepared_endpoint_profile(profiles)?;
    if profile.config.server_profiler.is_none() {
        return Ok(());
    }
    let provider = control_plane_http.prepare(
        clock.clone(),
        crate::engine::control_plane_http::ControlPlaneClientPolicy::default(),
    );
    let hooks = crate::engine::control_hooks::prepare_endpoint_control_hooks(
        clock,
        provider.as_ref(),
        profile,
    )?;
    sidecars.server_profiler = hooks
        .server_profiler
        .map(|hook| Rc::new(crate::engine::control_hooks::ServerProfilerCoordinator::new(hook)));
    Ok(())
}

pub(crate) async fn execute_native(
    request: NativeRunSpec,
    accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    graph_placement: &dyn GraphPlacementFactory,
    trace_driver: Arc<dyn TraceProgramDriverFactory>,
    registry: &AIPerfRegistry,
) -> Result<NativeReport> {
    if request.dataset.is_graph() {
        ensure!(
            accuracy.is_none(),
            "graph execution received prepared static-accuracy state"
        );
        return execute_graph_native(request, sidecars, graph_placement, trace_driver, registry)
            .await;
    }
    execute_scheduled_native(request, accuracy, sidecars, transport_factory, registry).await
}

pub(crate) async fn execute_scheduled_native(
    request: NativeRunSpec,
    accuracy: Option<&mut PreparedAccuracy>,
    sidecars: &mut PreparedNativeSidecarResources,
    transport_factory: Arc<dyn RequestExecutorFactory>,
    registry: &AIPerfRegistry,
) -> Result<NativeReport> {
    execute_native_inner(request, accuracy, sidecars, transport_factory, registry).await
}

pub(crate) fn validate_graph_request(request: &NativeRunSpec) -> Result<()> {
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

pub(crate) async fn execute_graph_native(
    request: NativeRunSpec,
    sidecars: &mut PreparedNativeSidecarResources,
    graph_placement: &dyn GraphPlacementFactory,
    trace_driver: Arc<dyn TraceProgramDriverFactory>,
    registry: &AIPerfRegistry,
) -> Result<NativeReport> {
    let planned_replay_traces = request.planned_replay_traces.clone();
    let graph = match &request.dataset {
        NativeDatasetPlan::Graph(graph) => graph,
        NativeDatasetPlan::PreparedLinear(_) | NativeDatasetPlan::StaticAccuracy(_) => {
            bail!("graph execution received a non-graph dataset plan")
        }
    };
    let graph_random_seed = graph.random_seed;
    let graph_default_output_tokens = graph.default_output_tokens;
    let allow_dataset_wrap = graph.allow_dataset_wrap;
    let cache_bust_enabled = graph.cache_bust_target.is_enabled();
    let t_star_window = graph.t_star_window;
    let ignore_trace_delays = graph.ignore_trace_delays;
    let system_idle_gap_cap_seconds = graph.system_idle_gap_cap_seconds;
    let metrics_config =
        metrics_config(&request.metrics, request.endpoint.use_server_token_count())?;
    let tokenizer = build_tokenizer(&request.tokenizer)?;
    let input_token_counter =
        select_input_token_counter(tokenizer.clone(), request.tokenizer.apply_chat_template);
    let mut input = graph.input.clone();
    ensure!(
        !input.programs.is_empty(),
        "authored Graph-IR input contains no root traces after root limiting"
    );
    let checkpoint_path = request.artifact_dir.join("replay-checkpoint.json");
    if graph.replay_resume
        && replay_resume_cell_partition_is_unsupported(ModuloCellPartition::from_env())
    {
        bail!("recorded-agent resume requires one controller-owned cell; cells > 1 is unsupported");
    }
    let replay_checkpoint = prepare_recorded_replay_checkpoint(
        &input,
        graph.replay_resume,
        &checkpoint_path,
        rng_root_for_checkpoint(graph.random_seed.or(request.random_seed)),
    )?
    .map(|checkpoint| Rc::new(RefCell::new(checkpoint)));
    let replay_run_identity = replay_checkpoint
        .as_ref()
        .map(|checkpoint| checkpoint.borrow().run.clone());
    if let Some(checkpoint) = replay_checkpoint.as_ref()
        && graph.replay_resume
    {
        let programs = input
            .programs
            .iter()
            .filter(|program| {
                let Some(replay) = program.replay.as_ref() else {
                    return true;
                };
                !checkpoint.borrow().should_skip(
                    &replay.identity,
                    replay.manifest_ordinal,
                    &replay.source_digest,
                    &replay.request_profile_identity,
                    replay.expected_llm_node_count,
                )
            })
            .cloned()
            .collect();
        input = Arc::new(crate::graph::input::GraphInputBundle {
            programs,
            segments: input.segments.clone(),
            metadata: input.metadata.clone(),
        });
    }
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
            content_server_media_base(&request)?,
            request.artifacts.raw_path.is_some(),
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
        trace_driver,
        worker_count: request.workers,
        clock: clock.clone(),
        real_clock_anchor,
        run_origin_ns: start_ns,
        model: primary_model.clone(),
        default_max_tokens: default_output_tokens,
        endpoint_runtime_factory,
        segments: input.segments.clone(),
        replay_run_identity,
        metrics: metrics_config.clone(),
        raw_enabled: request.artifacts.raw_path.is_some(),
        on_failure,
        cache_bust,
        ignore_trace_delays,
        system_idle_gap_cap_seconds,
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
    let terminal_callback = replay_checkpoint
        .as_ref()
        .filter(|_| ModuloCellPartition::from_env().is_none())
        .map(|checkpoint| {
            recorded_replay_terminal_callback(&input, checkpoint.clone(), checkpoint_path.clone())
        })
        .transpose()?;
    create_run_artifacts(&request)?;
    let phased = run_graph_phases(
        &request.phases,
        &request.benchmark_id,
        &request.artifact_dir,
        input.as_ref(),
        clock.clone(),
        rng_root,
        allow_dataset_wrap,
        cache_bust_enabled,
        t_star_window,
        phase_sidecars,
        &backends,
        on_failure,
        terminal_callback,
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
    // Recorded-agent replay artifacts are controller-owned and strict. They must
    // finish before the generic best-effort exporter tail, so a malformed timing
    // result fails the run instead of silently publishing a partial scenario.
    let replay_paths = crate::graph::replay::ReplayArtifactPaths {
        tool_time_path: request
            .artifacts
            .graph_tool_time_path
            .as_ref()
            .map(|path| artifact_path(&request.artifact_dir, path, "graph_tool_time_path"))
            .transpose()?,
        trace_summary_path: request
            .artifacts
            .graph_trace_summary_path
            .as_ref()
            .map(|path| artifact_path(&request.artifact_dir, path, "graph_trace_summary_path"))
            .transpose()?,
        metrics_json_path: request
            .artifacts
            .graph_replay_metrics_path
            .as_ref()
            .map(|path| artifact_path(&request.artifact_dir, path, "graph_replay_metrics_path"))
            .transpose()?,
        metrics_csv_path: request
            .artifacts
            .graph_replay_metrics_csv_path
            .as_ref()
            .map(|path| artifact_path(&request.artifact_dir, path, "graph_replay_metrics_csv_path"))
            .transpose()?,
        backend_metadata_path: request
            .artifacts
            .graph_replay_backend_metadata_path
            .as_ref()
            .map(|path| {
                artifact_path(
                    &request.artifact_dir,
                    path,
                    "graph_replay_backend_metadata_path",
                )
            })
            .transpose()?,
    };
    // A cellular child ships the foldable supplement in its terminal partition. The
    // controller is the sole final-artifact owner; writing here would publish one
    // scratch shard per cell and race the controller's merged report.
    if ModuloCellPartition::from_env().is_none() {
        let replay_traces = phased
            .supplement
            .traces
            .iter()
            .map(crate::graph::replay::ReplayTraceSupplement::from)
            .collect::<Vec<_>>();
        crate::graph::replay::write_replay_artifacts(&replay_paths, &replay_traces)
            .map_err(|error| anyhow!("writing recorded replay artifacts: {error}"))?;
        if let Some(checkpoint) = replay_checkpoint.as_ref() {
            finalize_recorded_replay_checkpoint(
                &mut checkpoint.borrow_mut(),
                &input,
                &phased.supplement.traces,
                &checkpoint_path,
                request
                    .artifacts
                    .graph_replay_failures_path
                    .as_ref()
                    .map(|path| request.artifact_dir.join(path)),
                request
                    .artifacts
                    .graph_replay_provenance_path
                    .as_ref()
                    .map(|path| request.artifact_dir.join(path)),
            )?;
        }
    }
    let phase_stats = phased.phases;
    // Graph exact-fold bounds memory independently of record count by folding each
    // record immediately. Per-record artifacts stream through `RecordArtifactLane`;
    // unstreamable Parquet and an explicit disabled exact-fold retain full records.
    // One gate for both executors: the graph path wires NONE of the retain-forcing
    // per-record consumers (live sink / heartbeat / adaptive / accuracy — all rejected
    // or never built by `execute_graph_native`; see `graph_exact_fold_drop_is_safe`),
    // so it passes them as `false` and the shared `exact_fold_eligible` reduces to the
    // graph-relevant `!sketch && !unstreamable_parquet` check.
    // The dry-run dataset analysis is a per-record consumer: it reads the FULL
    // retained record set (clean + errored) to build the length, timeline, and
    // prefix-cache sections, so requesting it forces retain mode on the graph
    // path (exact-fold would drop the clean records mid-run). The shared
    // `wants_per_record_artifacts` already disqualifies it; this local guard
    // states the graph path's own requirement rather than relying on that.
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
            wants_per_record_artifacts: wants_per_record_artifacts(&request.artifacts),
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
    //
    // As on the scheduled path, the partition is SNAPSHOT here and SHIPPED once this
    // cell's artifact files are written: the arriving partition is the controller's
    // same-host barrier for this cell, and `accumulator` is mutated by
    // `summarize_run_metrics` below, so its store must be cloned before that runs.
    #[cfg(feature = "cellular")]
    let cell_partition_ship = crate::engine::cellular_cell::CellRecordsShipper::from_env()?.map(
        |shipper| -> (_, crate::engine::cellular_cell::CellPartitionPayload) {
            use crate::engine::cellular_cell::CellPartitionPayload;
            // No `capture`/wall clock is in scope on the graph path, so derive the run span
            // from the records themselves: last observed end minus first observed start,
            // matching the elapsed span the scheduled path passes.
            let epoch_ns: i64 = run_end
                .unwrap_or(0)
                .saturating_sub(run_start.unwrap_or(0))
                .max(0);
            let payload = if graph_fold {
                let issued = accumulator.ingested_count();
                let counters = crate::cellular::HeartbeatCounters {
                    issued,
                    completed: issued.saturating_sub(errored_count),
                    errored: errored_count,
                };
                CellPartitionPayload::Store {
                    store: accumulator.column_store().clone(),
                    counters,
                    epoch_ns,
                    graph_supplement: Some(phased.supplement.clone()),
                    expected_replay_traces: planned_replay_traces.clone(),
                }
            } else {
                let records: Vec<RecordIngest> = captured
                    .iter()
                    .map(|record| record.ingest.clone())
                    .collect();
                CellPartitionPayload::Records {
                    records,
                    epoch_ns,
                    graph_supplement: Some(phased.supplement.clone()),
                    expected_replay_traces: planned_replay_traces.clone(),
                }
            };
            (shipper, payload)
        },
    );
    let RunMetricsSummaries {
        profiling_metrics,
        profiling_server_summary,
        warmup,
        warmup_server_summary,
        steady_state,
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
                    per_conversation: request.artifacts.dataset_analysis_per_conversation,
                },
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
        steady_state,
        ..RunOutcome::default()
    };
    // Every local artifact file now exists, so release the partition snapshotted above.
    // Ordering is load-bearing on the same-host path: the controller treats a cell's
    // arriving partition as that cell's completion barrier and then reads `cell-{id}/…`
    // from the shared scratch tree.
    #[cfg(feature = "cellular")]
    if let Some((shipper, payload)) = cell_partition_ship {
        payload.ship(&shipper)?;
    }
    // Cross-host cells ship local per-record artifacts to the controller with
    // streaming zstd. Same-host and single-process runs need no upload.
    #[cfg(feature = "cellular")]
    crate::engine::cellular_cell::ship_artifacts_if_enabled(
        &request.artifact_dir,
        &request.artifacts,
    )?;
    Ok(NativeReport::from_outcome(&profiling_metrics, &outcome))
}

fn rng_root_for_checkpoint(seed: Option<u64>) -> RngRoot {
    RngRoot::new(seed)
}

fn replay_resume_cell_partition_is_unsupported(partition: Option<ModuloCellPartition>) -> bool {
    partition.is_some_and(|partition| partition.cell_count() > 1)
}

fn prepare_recorded_replay_checkpoint(
    input: &Arc<crate::graph::input::GraphInputBundle>,
    resume: bool,
    path: &std::path::Path,
    rng_root: RngRoot,
) -> Result<Option<ReplayCheckpoint>> {
    let replay = input
        .programs
        .iter()
        .filter_map(|program| program.replay.as_ref())
        .collect::<Vec<_>>();
    if replay.is_empty() {
        return Ok(None);
    }
    let mut recordings = BTreeMap::new();
    let mut profiles = BTreeMap::new();
    let mut environments = BTreeMap::new();
    for program in &input.programs {
        let Some(trace) = program.replay.as_ref() else {
            continue;
        };
        let key = format!("{}:{}", trace.identity.adapter, trace.identity.task_id);
        recordings.insert(key.clone(), trace.source_digest.clone());
        profiles.insert(
            key.clone(),
            blake3::hash(
                serde_json::to_vec(&program.profiling)
                    .map_err(|error| {
                        anyhow!("serializing effective replay request profile: {error}")
                    })?
                    .as_slice(),
            )
            .to_hex()
            .to_string(),
        );
        environments.insert(
            key,
            blake3::hash(
                serde_json::to_string(&program.environment)
                    .map_err(|error| anyhow!("serializing replay environment identity: {error}"))?
                    .as_bytes(),
            )
            .to_hex()
            .to_string(),
        );
    }
    let manifest_digest = blake3::hash(
        replay
            .iter()
            .flat_map(|trace| {
                [
                    trace.source_digest.as_bytes(),
                    trace.request_profile_identity.as_bytes(),
                ]
            })
            .flatten()
            .copied()
            .collect::<Vec<_>>()
            .as_slice(),
    )
    .to_hex()
    .to_string();
    let root_digest = blake3::hash(input.metadata.format.as_bytes())
        .to_hex()
        .to_string();
    if resume && path.exists() {
        let run = ReplayCheckpoint::restore_run_identity(
            path,
            root_digest,
            manifest_digest,
            recordings,
            profiles,
            environments,
        )
        .map_err(|error| anyhow!("recovering recorded replay checkpoint namespace: {error}"))?;
        return ReplayCheckpoint::read_for_resume(path, &run)
            .map(Some)
            .map_err(|error| anyhow!("loading recorded replay checkpoint: {error}"));
    }
    let run_id = format!(
        "recorded-agent-replay:{}:{}",
        manifest_digest.chars().take(12).collect::<String>(),
        Uuid::new_v4().simple()
    );
    let namespace_identity = ReplayRunIdentity::mint(rng_root, &run_id);
    let namespace = CacheIsolationPolicy::first_message_prefix(namespace_identity)
        .namespace()
        .ok_or_else(|| anyhow!("recorded replay cache namespace was not created"))?
        .to_string();
    let run = ReplayRunIdentity::for_checkpoint_with_environment(
        run_id,
        root_digest,
        manifest_digest.clone(),
        recordings,
        profiles,
        environments,
        namespace,
    );
    let checkpoint = ReplayCheckpoint::new(run, manifest_digest);
    checkpoint
        .write_atomic(path)
        .map_err(|error| anyhow!("persisting recorded replay checkpoint before warmup: {error}"))?;
    Ok(Some(checkpoint))
}

fn recorded_replay_terminal_callback(
    input: &Arc<crate::graph::input::GraphInputBundle>,
    checkpoint: Rc<RefCell<ReplayCheckpoint>>,
    checkpoint_path: std::path::PathBuf,
) -> Result<GraphTraceTerminalCallback> {
    let mut tasks = BTreeMap::new();
    for program in &input.programs {
        let Some(replay) = program.replay.as_ref() else {
            continue;
        };
        let environment_digest = blake3::hash(
            serde_json::to_string(&program.environment)
                .map_err(|error| anyhow!("serializing replay environment identity: {error}"))?
                .as_bytes(),
        )
        .to_hex()
        .to_string();
        if tasks
            .insert(
                program.profiling.trace.id.clone(),
                (
                    replay.identity.clone(),
                    replay.manifest_ordinal,
                    replay.source_digest.clone(),
                    replay.request_profile_identity.clone(),
                    environment_digest,
                    replay.expected_llm_node_count,
                ),
            )
            .is_some()
        {
            bail!(
                "recorded replay checkpoint has duplicate trace id {:?}",
                program.profiling.trace.id
            );
        }
    }
    Ok(Rc::new(move |trace| {
        let Some((
            identity,
            manifest_ordinal,
            source_digest,
            profile_digest,
            environment_digest,
            expected,
        )) = tasks
            .get(trace.trace_id.as_str())
            .or_else(|| tasks.get(recorded_replay_template_trace_id(&trace.trace_id)))
        else {
            bail!(
                "recorded replay terminal trace {:?} has no controller checkpoint task",
                trace.trace_id
            );
        };
        let call_count = trace.calls.len() as u64;
        let mut checkpoint = checkpoint.borrow_mut();
        let merged = merge_terminal_replay_task(
            checkpoint.completed.get(identity),
            CompletedReplayTask {
                manifest_ordinal: *manifest_ordinal,
                source_digest: source_digest.clone(),
                request_profile_digest: profile_digest.clone(),
                environment_digest: environment_digest.clone(),
                successful_call_count: call_count,
                classification: if trace.completed && call_count == *expected {
                    ReplayTaskClassification::Successful
                } else {
                    ReplayTaskClassification::Partial
                },
                artifact_offset_start: 0,
                artifact_offset_end: call_count,
            },
            *expected,
        )?;
        checkpoint.completed.insert(identity.clone(), merged);
        checkpoint.write_atomic(&checkpoint_path).map_err(|error| {
            anyhow!("persisting recorded replay checkpoint after terminal task cleanup: {error}")
        })
    }))
}

fn finalize_recorded_replay_checkpoint(
    checkpoint: &mut ReplayCheckpoint,
    input: &Arc<crate::graph::input::GraphInputBundle>,
    traces: &[crate::graph::supplement::TraceTerminalSupplement],
    checkpoint_path: &std::path::Path,
    failures_path: Option<std::path::PathBuf>,
    provenance_path: Option<std::path::PathBuf>,
) -> Result<()> {
    let traces = traces
        .iter()
        .map(|trace| (trace.trace_id.as_str(), trace))
        .collect::<BTreeMap<_, _>>();
    let mut environment_digests = BTreeMap::new();
    for program in &input.programs {
        let Some(replay) = program.replay.as_ref() else {
            continue;
        };
        let key = format!("{}:{}", replay.identity.adapter, replay.identity.task_id);
        let environment_digest = blake3::hash(
            serde_json::to_string(&program.environment)
                .map_err(|error| anyhow!("serializing replay environment identity: {error}"))?
                .as_bytes(),
        )
        .to_hex()
        .to_string();
        environment_digests.insert(key, environment_digest.clone());
        let terminals = traces
            .values()
            .filter(|trace| {
                trace.trace_id == program.profiling.trace.id
                    || recorded_replay_template_trace_id(&trace.trace_id)
                        == program.profiling.trace.id.as_str()
            })
            .copied()
            .collect::<Vec<_>>();
        let successful = terminals.iter().find(|trace| {
            trace.completed && trace.calls.len() as u64 == replay.expected_llm_node_count
        });
        let call_count = successful.map_or_else(
            || {
                terminals
                    .iter()
                    .map(|trace| trace.calls.len() as u64)
                    .max()
                    .unwrap_or(0)
            },
            |trace| trace.calls.len() as u64,
        );
        let classification = if successful.is_some() {
            ReplayTaskClassification::Successful
        } else if terminals.iter().any(|trace| trace.completed) {
            ReplayTaskClassification::Partial
        } else {
            ReplayTaskClassification::Failed
        };
        checkpoint.completed.insert(
            replay.identity.clone(),
            CompletedReplayTask {
                manifest_ordinal: replay.manifest_ordinal,
                source_digest: replay.source_digest.clone(),
                request_profile_digest: replay.request_profile_identity.clone(),
                environment_digest,
                successful_call_count: call_count,
                classification: classification.clone(),
                artifact_offset_start: 0,
                artifact_offset_end: call_count,
            },
        );
        checkpoint.write_atomic(checkpoint_path).map_err(|error| {
            anyhow!("persisting recorded replay checkpoint after task cleanup: {error}")
        })?;
    }
    let failures = checkpoint
        .completed
        .iter()
        .filter(|(_, completed)| completed.classification != ReplayTaskClassification::Successful)
        .map(|(task, completed)| {
            format!(
                "{}\t{}\t{:?}",
                tsv_cell(&task.adapter),
                tsv_cell(&task.task_id),
                completed.classification
            )
        })
        .collect::<Vec<_>>();
    let complete = failures.is_empty()
        && checkpoint.completed.len() == checkpoint.run.recording_digests().len();
    checkpoint
        .write_atomic(checkpoint_path)
        .map_err(|error| anyhow!("persisting recorded replay checkpoint after cleanup: {error}"))?;
    if let Some(path) = failures_path {
        write_replay_text(
            &path,
            &format!("adapter\ttask_id\tclassification\n{}", failures.join("\n")),
        )?;
    }
    if let Some(path) = provenance_path {
        let provenance = ReplayProvenance {
            manifest_digest: checkpoint.manifest_digest.clone(),
            recording_digests: checkpoint.run.recording_digests().clone(),
            request_profile_digests: checkpoint.run.request_profile_digests().clone(),
            environment_digests,
            cache_isolation_mode: "first_message_prefix".to_string(),
            cache_namespace: None,
            cache_namespace_digest: checkpoint.run.cache_namespace_digest(),
            endpoint: None,
            hardware_description: None,
            debug_overrides: Vec::new(),
            comparable: complete,
        };
        let bytes = serde_json::to_vec_pretty(&redact_replay_provenance(&provenance))
            .map_err(|error| anyhow!("serializing replay provenance: {error}"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| anyhow!("creating replay provenance directory: {error}"))?;
        }
        std::fs::write(&path, bytes)
            .map_err(|error| anyhow!("writing replay provenance {}: {error}", path.display()))?;
    }
    Ok(())
}

fn merge_terminal_replay_task(
    existing: Option<&CompletedReplayTask>,
    incoming: CompletedReplayTask,
    expected_call_count: u64,
) -> Result<CompletedReplayTask> {
    let Some(existing) = existing else {
        return Ok(incoming);
    };
    ensure!(
        existing.manifest_ordinal == incoming.manifest_ordinal
            && existing.source_digest == incoming.source_digest
            && existing.request_profile_digest == incoming.request_profile_digest
            && existing.environment_digest == incoming.environment_digest,
        "recorded replay runtime instances resolved to inconsistent checkpoint task metadata"
    );
    if existing.classification == ReplayTaskClassification::Successful {
        return Ok(existing.clone());
    }
    if incoming.classification == ReplayTaskClassification::Successful {
        return Ok(incoming);
    }
    ensure!(
        expected_call_count > 0,
        "recorded replay expected call count must be positive for partial checkpoint merging"
    );
    Ok(incoming)
}

fn recorded_replay_template_trace_id(trace_id: &str) -> &str {
    trace_id
        .rsplit_once("::instance-")
        .filter(|(_, ordinal)| {
            !ordinal.is_empty() && ordinal.bytes().all(|byte| byte.is_ascii_digit())
        })
        .map_or(trace_id, |(template, _)| template)
}

fn write_replay_text(path: &std::path::Path, contents: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| anyhow!("creating replay artifact directory: {error}"))?;
    }
    std::fs::write(path, contents)
        .map_err(|error| anyhow!("writing replay artifact {}: {error}", path.display()))
}

fn tsv_cell(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// The post-capture metric summaries shared by both executors' finalize tails
/// ([`summarize_run_metrics`]): the profiling metrics export (with any GPU
/// telemetry already attached) plus the optional profiling/warmup server
/// summaries and the optional warmup metrics export.
pub(crate) struct RunMetricsSummaries {
    pub(crate) profiling_metrics: AccumulatorSummary,
    pub(crate) profiling_server_summary: Option<ServerMetricsSummary>,
    pub(crate) warmup: Option<AccumulatorSummary>,
    pub(crate) warmup_server_summary: Option<ServerMetricsSummary>,
    pub(crate) steady_state: Option<crate::metrics_core::SteadyStateOutcome>,
}

/// Inject the calibrated network RTT into the accumulator, export the profiling
/// metrics (attaching GPU telemetry when present), and derive the profiling and
/// warmup server-metrics summaries plus the warmup metrics export. The two
/// callers differ only in cell shipping, artifact writing, and outcome assembly.
pub(crate) fn summarize_run_metrics(
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
    // Profiling-phase concurrency target, shared by GPU-telemetry normalization
    // and steady-state windowing.
    let profiling_concurrency = request
        .phases
        .iter()
        .find(|phase| phase.common().name == "profiling")
        .and_then(PhaseSpec::concurrency);
    if let Some(gpu_telemetry) = gpu_telemetry {
        let total_output_tokens = profiling_metrics.finite_value(MetricTag::TotalOutputTokens);
        gpu_telemetry
            .summarize(
                total_output_tokens,
                profiling_concurrency.map(|value| value as u64),
            )
            .attach_to(&mut profiling_metrics);
    }
    // Closed-loop steady-state summary over the auto-detected saturated window.
    // Internally gated: yields None unless enabled with a positive concurrency
    // target, so disabled runs are unaffected.
    let steady_state = profiling_concurrency.and_then(|target| {
        crate::metrics_core::steady_state_summary(accumulator, &metrics_config.steady_state, target)
    });
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
        steady_state,
    }
}

/// Write the GPU / network-latency / server-metrics record sidecars (each a no-op
/// when its producer or destination path is absent) and build the additive
/// server-metrics report metadata. Byte-identical tail shared by the
/// scheduled/accuracy and graph executors; the record paths are passed in so both
/// callers (graph inlines `sidecars.*`, scheduled pre-binds the same as locals)
/// produce identical behavior.
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_sidecar_records(
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
pub(crate) fn write_records_parquet_artifact(
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
/// the Parquet sidecar this needs no extra Cargo feature (the writer is
/// hand-rolled over `std::io`), so it is always available.
pub(crate) fn write_records_csv_artifact(
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

pub(crate) fn write_graph_artifacts(
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

pub(crate) async fn prepare_static_accuracy(
    request: &NativeRunSpec,
) -> Result<Option<PreparedAccuracy>> {
    let NativeDatasetPlan::StaticAccuracy(spec) = &request.dataset else {
        return Ok(None);
    };
    let model = request
        .models
        .items
        .first()
        .map(|item| item.name.as_str())
        .ok_or_else(|| anyhow!("at least one model is required"))?;
    let tokenizer = build_tokenizer(&request.tokenizer)?;
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

pub(crate) async fn finish_accuracy_lifecycle<T>(
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
pub(crate) fn finish_with_shutdown<T>(
    result: Result<T>,
    shutdown: Result<()>,
    label: &str,
) -> Result<T> {
    match (result, shutdown) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error.context(format!("shutting down {label}"))),
        (Err(error), Err(shutdown)) => {
            Err(error.context(format!("{label} also failed during shutdown: {shutdown:#}")))
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn recorded_replay_input() -> Arc<crate::graph::input::GraphInputBundle> {
        recorded_replay_input_with_expected_calls(0)
    }

    fn recorded_replay_input_with_expected_calls(
        expected_llm_node_count: u64,
    ) -> Arc<crate::graph::input::GraphInputBundle> {
        recorded_replay_input_with_trace_id("trace-1", expected_llm_node_count)
    }

    fn recorded_replay_input_with_trace_id(
        trace_id: &str,
        expected_llm_node_count: u64,
    ) -> Arc<crate::graph::input::GraphInputBundle> {
        let plan = crate::graph::model::GraphTracePlan {
            graph: crate::graph::model::GraphRecord::default(),
            trace: crate::graph::model::TraceRecord {
                id: trace_id.into(),
                graph_ref: None,
                initial_state: BTreeMap::new(),
            },
            arrival_offset_ns: None,
        };
        let mut program = crate::graph::model::GraphTraceProgram::static_graph(plan);
        program.driver = crate::graph::driver::TraceDriverSpec::recorded_replay();
        program.replay = Some(crate::graph::driver::ReplayTraceMetadata {
            manifest_ordinal: 0,
            identity: crate::graph::driver::ReplayTaskIdentity {
                adapter: "pinchbench".into(),
                family: "pinchbench".into(),
                task_id: "task-1".into(),
                primary_role: None,
            },
            source_digest: "source-digest".into(),
            normalization_target_digest: None,
            target_output_tokens: Vec::new(),
            expected_llm_node_count,
            expected_tool_node_count: 0,
            request_profile_identity: "profile-id".into(),
            comparability_annotations: BTreeMap::new(),
        });
        Arc::new(crate::graph::input::GraphInputBundle {
            programs: vec![program],
            segments: Arc::new(crate::dataset::InMemorySegmentStore::default()),
            metadata: crate::graph::input::GraphInputMetadata {
                format: "agent_recording".into(),
                root_count: 1,
                node_count: 0,
                warning_facts: Vec::new(),
            },
        })
    }

    #[test]
    fn new_recorded_replay_checkpoints_get_distinct_resume_stable_run_ids() {
        let first_dir = tempfile::tempdir().unwrap();
        let second_dir = tempfile::tempdir().unwrap();
        let first_path = first_dir.path().join("replay-checkpoint.json");
        let second_path = second_dir.path().join("replay-checkpoint.json");
        let input = recorded_replay_input();

        let first =
            prepare_recorded_replay_checkpoint(&input, false, &first_path, RngRoot::new(Some(7)))
                .unwrap()
                .unwrap();
        let second =
            prepare_recorded_replay_checkpoint(&input, false, &second_path, RngRoot::new(Some(7)))
                .unwrap()
                .unwrap();

        assert_ne!(
            first.run.label(),
            second.run.label(),
            "concurrent same-manifest runs need disjoint Docker cleanup labels"
        );
        let resumed =
            prepare_recorded_replay_checkpoint(&input, true, &first_path, RngRoot::new(Some(7)))
                .unwrap()
                .unwrap();
        assert_eq!(resumed.run.label(), first.run.label());
        assert_eq!(
            resumed.run.cache_namespace_digest(),
            first.run.cache_namespace_digest()
        );
    }

    #[test]
    fn recorded_replay_terminal_callback_maps_runtime_instance_to_authored_task() {
        let output = tempfile::tempdir().unwrap();
        let checkpoint_path = output.path().join("replay-checkpoint.json");
        let input = recorded_replay_input();
        let checkpoint = Rc::new(RefCell::new(
            prepare_recorded_replay_checkpoint(
                &input,
                false,
                &checkpoint_path,
                RngRoot::new(Some(7)),
            )
            .unwrap()
            .unwrap(),
        ));
        let callback =
            recorded_replay_terminal_callback(&input, checkpoint.clone(), checkpoint_path).unwrap();

        callback(&crate::graph::supplement::TraceTerminalSupplement::new(
            "run".into(),
            "trace-1::instance-0::trajectory".into(),
            "trace-1::instance-0".into(),
            0,
            "recorded_replay",
        ))
        .expect("runtime instance maps back to the authored replay task");

        assert_eq!(checkpoint.borrow().completed.len(), 1);
    }

    #[test]
    fn recorded_replay_terminal_callback_keeps_repeated_runtime_partials_partial() {
        let output = tempfile::tempdir().unwrap();
        let checkpoint_path = output.path().join("replay-checkpoint.json");
        let input = recorded_replay_input_with_expected_calls(5);
        let checkpoint = Rc::new(RefCell::new(
            prepare_recorded_replay_checkpoint(
                &input,
                false,
                &checkpoint_path,
                RngRoot::new(Some(7)),
            )
            .unwrap()
            .unwrap(),
        ));
        let callback =
            recorded_replay_terminal_callback(&input, checkpoint.clone(), checkpoint_path).unwrap();

        let mut first = crate::graph::supplement::TraceTerminalSupplement::new(
            "run".into(),
            "trace-1::instance-0::trajectory".into(),
            "trace-1::instance-0".into(),
            0,
            "recorded_replay",
        );
        first.completed = true;
        first.calls = vec![
            crate::graph::replay::ReplayCallMeasurement::completed("trace-1::instance-0", 0),
            crate::graph::replay::ReplayCallMeasurement::completed("trace-1::instance-0", 1),
        ];
        callback(&first).expect("first runtime instance updates the checkpoint");

        let mut second = crate::graph::supplement::TraceTerminalSupplement::new(
            "run".into(),
            "trace-1::instance-1::trajectory".into(),
            "trace-1::instance-1".into(),
            0,
            "recorded_replay",
        );
        second.completed = true;
        second.calls = vec![
            crate::graph::replay::ReplayCallMeasurement::completed("trace-1::instance-1", 0),
            crate::graph::replay::ReplayCallMeasurement::completed("trace-1::instance-1", 1),
            crate::graph::replay::ReplayCallMeasurement::completed("trace-1::instance-1", 2),
        ];
        callback(&second)
            .expect("second runtime instance accumulates without promoting to success");

        let completed = checkpoint
            .borrow()
            .completed
            .values()
            .next()
            .cloned()
            .expect("task checkpoint entry exists");
        assert_eq!(completed.successful_call_count, 3);
        assert_eq!(completed.classification, ReplayTaskClassification::Partial);
    }

    #[test]
    fn recorded_replay_terminal_callback_keeps_any_completed_runtime_instance_successful() {
        let output = tempfile::tempdir().unwrap();
        let checkpoint_path = output.path().join("replay-checkpoint.json");
        let input = recorded_replay_input_with_expected_calls(2);
        let checkpoint = Rc::new(RefCell::new(
            prepare_recorded_replay_checkpoint(
                &input,
                false,
                &checkpoint_path,
                RngRoot::new(Some(7)),
            )
            .unwrap()
            .unwrap(),
        ));
        let callback =
            recorded_replay_terminal_callback(&input, checkpoint.clone(), checkpoint_path).unwrap();

        for ordinal in 0..2 {
            let trace_id = format!("trace-1::instance-{ordinal}");
            let mut terminal = crate::graph::supplement::TraceTerminalSupplement::new(
                "run".into(),
                format!("{trace_id}::trajectory"),
                trace_id.clone(),
                0,
                "recorded_replay",
            );
            terminal.completed = true;
            terminal.calls = vec![
                crate::graph::replay::ReplayCallMeasurement::completed(&trace_id, 0),
                crate::graph::replay::ReplayCallMeasurement::completed(&trace_id, 1),
            ];
            callback(&terminal).expect("completed runtime instance updates the checkpoint");
        }

        let completed = checkpoint
            .borrow()
            .completed
            .values()
            .next()
            .cloned()
            .expect("task checkpoint entry exists");
        assert_eq!(completed.successful_call_count, 2);
        assert_eq!(
            completed.classification,
            ReplayTaskClassification::Successful
        );
    }

    #[test]
    fn recorded_replay_terminal_callback_prefers_an_exact_authored_instance_suffix() {
        let output = tempfile::tempdir().unwrap();
        let checkpoint_path = output.path().join("replay-checkpoint.json");
        let input = recorded_replay_input_with_trace_id("trace-1::instance-0", 0);
        let checkpoint = Rc::new(RefCell::new(
            prepare_recorded_replay_checkpoint(
                &input,
                false,
                &checkpoint_path,
                RngRoot::new(Some(7)),
            )
            .unwrap()
            .unwrap(),
        ));
        let callback =
            recorded_replay_terminal_callback(&input, checkpoint.clone(), checkpoint_path).unwrap();

        callback(&crate::graph::supplement::TraceTerminalSupplement::new(
            "run".into(),
            "trace-1::instance-0::trajectory".into(),
            "trace-1::instance-0".into(),
            0,
            "recorded_replay",
        ))
        .expect("exact authored trace id must not be normalized away");

        assert_eq!(checkpoint.borrow().completed.len(), 1);
    }

    #[test]
    fn replay_resume_allows_identity_cell_partition_but_rejects_multi_cell() {
        assert!(!replay_resume_cell_partition_is_unsupported(None));
        assert!(!replay_resume_cell_partition_is_unsupported(Some(
            ModuloCellPartition::new(0, 1).expect("identity cell partition"),
        )));
        assert!(replay_resume_cell_partition_is_unsupported(Some(
            ModuloCellPartition::new(0, 2).expect("multi-cell partition"),
        )));
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
}
