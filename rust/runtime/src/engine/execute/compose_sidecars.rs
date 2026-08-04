// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-phase sidecar composition and the inner native execution driver.

use super::*;

/// Compose the phase-scoped side-channel sidecars for one scheduled/graph phase:
/// server metrics run every phase; GPU telemetry and network-latency calibration
/// run only during profiling. Shared verbatim by the graph phase-plan builder and
/// the single-thread scheduled arm of [`execute_native_inner`]. The sharded arm
/// builds a profiling-only set inline (it has no per-phase loop) and the per-shard
/// worker never touches a sidecar, so those two paths do not use this helper.
pub(crate) fn compose_phase_sidecars(
    phase: &PhaseSpec,
    sidecars: &PreparedNativeSidecarResources,
) -> Result<Vec<Rc<dyn ScheduledPhaseSidecar>>> {
    let mut phase_sidecars: Vec<Rc<dyn ScheduledPhaseSidecar>> = Vec::new();
    if let Some(server_metrics) = sidecars.server_metrics.as_ref() {
        phase_sidecars.push(server_metrics.sidecar(metrics_phase(phase)?));
    }
    if !phase.common().exclude_from_results {
        if let Some(gpu_telemetry) = sidecars.gpu_telemetry.as_ref() {
            phase_sidecars.push(gpu_telemetry.sidecar());
        }
        if let Some(network_latency) = sidecars.network_latency.as_ref()
            && let Some(sidecar) = network_latency.sidecar()
        {
            phase_sidecars.push(sidecar);
        }
        if let Some(server_profiler) = sidecars.server_profiler.as_ref() {
            phase_sidecars.push(crate::engine::server_profiler::sidecar(
                server_profiler.clone(),
                phase.common().name.clone(),
            ));
        }
    }
    Ok(phase_sidecars)
}

pub(crate) async fn execute_native_inner(
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
        None => build_tokenizer(&request.tokenizer)?,
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
                content_server_base: content_server_media_base(&request)?,
                // The request-payload capture flags are stamped per execution
                // backend from `ExecutionBackendConfig` (which knows whether
                // this run shards and exact-folds); the defaults here capture.
                ..TransportSinkConfig::default()
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
    // Side-channel subagent join-gate specs (empty for every non-agentic run),
    // carried alongside the composed dataset and threaded into the agentic phase
    // plan below.
    let agentic_trees = match &request.dataset {
        NativeDatasetPlan::PreparedLinear(dataset) => dataset.agentic_trees.clone(),
        _ => std::sync::Arc::default(),
    };
    // Cross-phase accelerated cache-warmup handoff carrier (empty for every
    // non-accelerated run), threaded into both agentic phase instances.
    let warmup_handoff = match &request.dataset {
        NativeDatasetPlan::PreparedLinear(dataset) => dataset.warmup_handoff.clone(),
        _ => crate::agentic_tree::empty_warmup_handoff_carrier(),
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
        && !request.phases.iter().any(|phase| {
            matches!(
                phase,
                PhaseSpec::FixedSchedule { .. } | PhaseSpec::AgenticReplay { .. }
            )
        });
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
            raw_enabled: request.artifacts.raw_path.is_some(),
            // Mirrors the `RunCapture` below: under exact-fold `inputs.json` comes
            // from the resident dataset at the coordinator, not from dispatch.
            inputs_enabled: request.artifacts.inputs_path.is_some() && !exact_fold,
            prepared_endpoints,
            hop_routing: if request.dispatch_mode
                == crate::engine::protocol::DispatchMode::GlobalHop
            {
                request.hop_routing.unwrap_or_default()
            } else {
                crate::engine::protocol::HopRouting::RoundRobin
            },
            virtual_worker_width: request.virtual_worker_width,
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
                        PhaseSpec::UserCentric { .. }
                            | PhaseSpec::FixedSchedule { .. }
                            | PhaseSpec::AgenticReplay { .. }
                    )
                })
            {
                tracing::warn!(
                    "failure_policy=abort is honored only for request-rate/concurrency scheduled \
                 phases; user_centric and fixed_schedule phases in this run stay resilient"
                );
            }

            let mut plans = Vec::with_capacity(request.phases.len());
            let mut profiling_index = 0usize;
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
                    agentic_trees.clone(),
                    warmup_handoff.clone(),
                )?;
                let profiling_idx = if phase.common().exclude_from_results {
                    None
                } else {
                    let idx = profiling_index;
                    profiling_index += 1;
                    Some(idx)
                };
                let identity = phase_identity_from_spec(phase, phase_index, profiling_idx);
                let record_processor: Rc<dyn TurnRecordProcessor> =
                    Rc::new(CapturePhaseProcessor {
                        capture: capture.clone(),
                        phase: metrics_phase(phase)?,
                        identity,
                        has_credit_timestamp: !matches!(
                            phase,
                            PhaseSpec::FixedSchedule { .. } | PhaseSpec::AgenticReplay { .. }
                        ),
                        live_sink: live_sink.clone(),
                        heartbeat: heartbeat_lane.clone(),
                    });
                let mut record_processors = vec![record_processor];
                if !phase.common().exclude_from_results
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
        // Built once per cell, on the main thread, before any worker thread
        // spawns (worker threads are spawned only after `run_sharded_scheduled`
        // receives this already-fully-built `Arc<ShardedShared>`), from the
        // cell-local (not-yet-thread-sliced) phase specs — see `GlobalAdmission`.
        // Only `Global` builds shared cross-thread admission gates, to make its
        // `W` independent scheduling loops jointly exact. `Sharded` slices
        // per-thread and needs none; `GlobalHop` runs a single coordinator loop
        // with the full local cap and needs none either (see
        // `global_hop::run_global_hop` — "one loop, one full-cap local pool").
        let global_admission = match request.dispatch_mode {
            DispatchMode::Global => Some(Arc::new(GlobalAdmission::build(&request.phases)?)),
            DispatchMode::Sharded | DispatchMode::GlobalHop => None,
        };
        let shared = Arc::new(ShardedShared {
            transport_factory: transport_factory.clone(),
            table_factory,
            samplers: registry.samplers().clone(),
            dataset: dataset.clone(),
            agentic_trees: agentic_trees.clone(),
            warmup_handoff: warmup_handoff.clone(),
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
            dispatch_mode: request.dispatch_mode,
            hop_routing: request.hop_routing.unwrap_or_default(),
            global_admission,
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
        if let Some(server_profiler) = sidecars.server_profiler.as_ref() {
            profiling_sidecars.push(crate::engine::server_profiler::sidecar(
                server_profiler.clone(),
                "profiling",
            ));
        }
        // `GlobalHop` runs one coordinator loop hopping turns to worker threads;
        // `Sharded`/`Global` run `W` independent per-thread scheduling loops.
        let outcome = match request.dispatch_mode {
            DispatchMode::GlobalHop => {
                crate::engine::global_hop::run_global_hop(shared, profiling_sidecars, clock.clone())
                    .await?
            }
            DispatchMode::Sharded | DispatchMode::Global => {
                crate::engine::sharded_scheduled::run_sharded_scheduled(
                    shared,
                    profiling_sidecars,
                    clock.clone(),
                )
                .await?
            }
        };
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
    //
    // The partition is SNAPSHOT here but SHIPPED after this cell's artifact files are
    // written (see the deferred ship below). The partition arriving at the controller is
    // the controller's only same-host barrier for this cell, so shipping it before the
    // local writes lets the controller concatenate/merge `cell-{id}` while a file is
    // still missing — observed as a cellular `inputs.json` short by one cell's slice.
    // Snapshotting (rather than moving the whole block) keeps the shipped bytes and
    // `epoch_ns` byte-identical to the pre-defer behavior: `epoch_ns` is stamped at this
    // point in the finalize, and `accumulator` is mutated by `summarize_run_metrics`
    // below, so its store must be cloned before that runs.
    #[cfg(feature = "cellular")]
    let cell_partition_ship = crate::engine::cellular_cell::CellRecordsShipper::from_env().map(
        |shipper| -> (_, crate::engine::cellular_cell::CellPartitionPayload) {
            use crate::engine::cellular_cell::CellPartitionPayload;
            let epoch_ns: i64 = clock.now_ns().saturating_sub(start_ns);
            let payload = if exact_fold || sketch_mode {
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
                CellPartitionPayload::Store {
                    store: accumulator.column_store().clone(),
                    counters,
                    epoch_ns,
                }
            } else {
                let records: Vec<RecordIngest> = captured
                    .iter()
                    .map(|record| record.ingest.clone())
                    .collect();
                CellPartitionPayload::Records { records, epoch_ns }
            };
            (shipper, payload)
        },
    );
    let gpu_telemetry = sidecars.gpu_telemetry.as_ref();
    let network_latency = sidecars.network_latency.as_ref();
    let server_metrics = sidecars.server_metrics.as_ref();
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
                            per_conversation: request.artifacts.dataset_analysis_per_conversation,
                        },
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
    // Every local artifact file now exists, so release the partition snapshotted above.
    // Ordering is load-bearing on the same-host path: the controller treats a cell's
    // arriving partition as that cell's completion barrier and then reads
    // `cell-{id}/…` from the shared scratch tree.
    #[cfg(feature = "cellular")]
    if let Some((shipper, payload)) = cell_partition_ship {
        payload.ship(&shipper)?;
    }
    // (cross-host k8s cell): all per-record artifacts (+ inputs.json) are now
    // on this cell's own filesystem; ship them to the controller's HTTP upload server
    // with streaming zstd. A no-op on the same-host launcher (concatenates the
    // local writes) or the single-process path.
    #[cfg(feature = "cellular")]
    crate::engine::cellular_cell::ship_artifacts_if_enabled(
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
        steady_state,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimClock;
    use crate::engine::network_latency::NetworkLatencyRun;
    use crate::engine::server_metrics::ServerMetricsRun;
    use crate::engine::sidecar_input::{NetworkLatencySpec, ServerMetricsSpec};

    fn phase(name: &str) -> PhaseSpec {
        serde_json::from_value(serde_json::json!({
            "type": "concurrency",
            "name": name,
            "exclude_from_results": name != "profiling",
            "requests": 4,
            "concurrency": 2,
        }))
        .unwrap()
    }

    fn server_metrics_run(clock: Rc<dyn Clock>) -> ServerMetricsRun {
        let spec: ServerMetricsSpec = serde_json::from_value(serde_json::json!({
            "collection_interval_ns": 1_000_000,
            "reachability_timeout_ns": 1_000_000,
            "urls": ["http://localhost:9400/metrics"],
            "formats": [],
        }))
        .unwrap();
        ServerMetricsRun::new(&spec, clock).unwrap()
    }

    fn network_latency_run(clock: Rc<dyn Clock>) -> NetworkLatencyRun {
        let spec: NetworkLatencySpec = serde_json::from_value(serde_json::json!({
            "probe": {
                "ping_interval_ns": 1_000_000,
                "connect_timeout_ns": 1_000_000,
                "complete_topup_timeout_ns": 1_000_000,
                "min_successful_samples": 1,
                "records_path": "network_latency.jsonl",
            }
        }))
        .unwrap();
        NetworkLatencyRun::new(
            "bench-id",
            &spec,
            std::slice::from_ref(&"http://localhost:8000".to_string()),
            clock,
        )
        .unwrap()
    }

    fn resources(
        server_metrics: Option<ServerMetricsRun>,
        network_latency: Option<NetworkLatencyRun>,
    ) -> PreparedNativeSidecarResources {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        PreparedNativeSidecarResources {
            real_clock_anchor: RealClockAnchor::now(),
            clock,
            content_server: None,
            gpu_telemetry: None,
            network_latency,
            server_metrics,
            server_profiler: None,
            live_streaming: None,
            gpu_records_path: None,
            network_latency_records_path: None,
            server_metrics_jsonl_path: None,
            server_metrics_parquet_wire_path: None,
            media_handle: None,
        }
    }

    #[test]
    fn compose_phase_sidecars_empty_when_no_resources() {
        let sidecars = resources(None, None);
        assert!(
            compose_phase_sidecars(&phase("warmup"), &sidecars)
                .unwrap()
                .is_empty(),
            "no configured resources composes to no warmup sidecars"
        );
        assert!(
            compose_phase_sidecars(&phase("profiling"), &sidecars)
                .unwrap()
                .is_empty(),
            "no configured resources composes to no profiling sidecars"
        );
    }

    #[test]
    fn compose_phase_sidecars_gates_profiling_only_resources_out_of_warmup() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        // Both server metrics (every phase) and network-latency calibration
        // (profiling only) are configured.
        let sidecars = resources(
            Some(server_metrics_run(clock.clone())),
            Some(network_latency_run(clock)),
        );

        // Warmup runs server metrics only — the profiling-only network probe is
        // excluded.
        let warmup = compose_phase_sidecars(&phase("warmup"), &sidecars).unwrap();
        assert_eq!(
            warmup.len(),
            1,
            "warmup composes server metrics only, never the profiling-only probe"
        );

        // Profiling adds the network-latency calibration on top of server metrics.
        let profiling = compose_phase_sidecars(&phase("profiling"), &sidecars).unwrap();
        assert_eq!(
            profiling.len(),
            2,
            "profiling composes server metrics plus the network-latency probe"
        );
    }

    #[test]
    fn compose_phase_sidecars_fixed_rtt_network_has_no_probe_sidecar() {
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        // A fixed mean-RTT network-latency config owns no probe, so it contributes
        // no phase sidecar even during profiling.
        let spec: NetworkLatencySpec = serde_json::from_value(serde_json::json!({
            "mean_rtt_ns": 500_000.0,
        }))
        .unwrap();
        let fixed = NetworkLatencyRun::new("bench-id", &spec, &[], clock.clone()).unwrap();
        let sidecars = resources(Some(server_metrics_run(clock)), Some(fixed));
        let profiling = compose_phase_sidecars(&phase("profiling"), &sidecars).unwrap();
        assert_eq!(
            profiling.len(),
            1,
            "a fixed-RTT network config adds no profiling probe sidecar"
        );
    }
}
