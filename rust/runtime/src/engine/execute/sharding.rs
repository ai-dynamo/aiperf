// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduled-shard admission primitives and shard/pipeline execution.

use super::*;

/// Per-cell shared admission gates for `Global` dispatch.
///
/// Built once per cell, on the main thread, before worker threads spawn, from
/// this cell's phase specs — the *cell-level* budgets (already narrowed from
/// the global run by `owned_positions(global, cell_id, cells)` upstream in the
/// cellular controller), NOT further sliced by `workers`. Every worker thread
/// in the cell shares the same `Arc<GlobalSlotPool>`/`Arc<GlobalRateGate>` per
/// phase, so aggregate concurrency and rate across all `W` threads combined
/// enforce the single cell-level target exactly, instead of `W` independent
/// `1/W`-sliced local limits.
///
/// Present only under [`DispatchMode::Global`]; `None` for
/// [`DispatchMode::Sharded`] (per-thread `1/W` slicing needs no shared gate) and
/// for [`DispatchMode::GlobalHop`]/[`DispatchMode::GlobalPush`] (their single
/// coordinator loop enforces the full cap through one local `SlotPool` — see
/// [`crate::engine::global_hop`]). The shared gate exists specifically to make
/// `Global`'s `W` independent scheduling loops jointly exact.
pub(crate) struct GlobalAdmission {
    /// One shared concurrency gate per phase that authors a `concurrency` cap.
    pub(crate) concurrency: HashMap<MetricsPhase, Arc<crate::timing::GlobalSlotPool>>,
    /// One shared prefill gate per phase that authors a `prefill_concurrency` cap.
    ///
    /// `prefill_concurrency` is an admission cap exactly as `concurrency` is, so without a shared
    /// gate it was statically partitioned per thread by `slice_common` — leaving `Global` able to
    /// strand prefill capacity by the same mechanism the shared concurrency gate exists to
    /// prevent: a thread that has finished its own work still owns its prefill share, and a thread
    /// that needs it cannot borrow it. It is also floored at one per thread by `owned_cap`, so a
    /// prefill cap below the thread count over-subscribed. One shared pool removes both.
    pub(crate) prefill: HashMap<MetricsPhase, Arc<crate::timing::GlobalSlotPool>>,
    /// One shared rate gate per phase that authors a `rate`.
    ///
    /// Consumed by `phase_scheduled_resources`, which hands the per-phase gate
    /// to that phase's [`RequestRateWorkload`] (via `with_rate_gate`) so every
    /// worker thread paces against one shared next-fire-time counter. The union
    /// of the base slots each thread claims is exactly the global rate grid, so
    /// aggregate arrival rate across all `W` threads matches the configured
    /// global rate.
    pub(crate) rate: HashMap<MetricsPhase, Arc<crate::timing::GlobalRateGate>>,
}

impl GlobalAdmission {
    /// Build one gate per phase from the cell-local (unsliced-by-thread) phase specs.
    pub(crate) fn build(phases: &[PhaseSpec]) -> Result<Self> {
        let mut concurrency = HashMap::new();
        let mut prefill = HashMap::new();
        let mut rate = HashMap::new();
        for phase in phases {
            let phase_key = metrics_phase(phase)?;
            if let Some(cap) = phase.concurrency() {
                concurrency.insert(phase_key, crate::timing::GlobalSlotPool::new(cap));
            }
            if let Some(cap) = phase.common().prefill_concurrency {
                prefill.insert(phase_key, crate::timing::GlobalSlotPool::new(cap));
            }
            if let Some(r) = phase.rate() {
                rate.insert(phase_key, crate::timing::GlobalRateGate::new(r));
            }
        }
        Ok(Self {
            concurrency,
            prefill,
            rate,
        })
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
    /// Side-channel subagent join-gate specs for the `agentic_replay` timing
    /// mode (empty for every non-agentic run). `Send + Sync` plain data, cloned
    /// per shard and threaded into the agentic phase plan.
    pub(crate) agentic_trees: std::sync::Arc<Vec<crate::agentic_tree::TreeSpec>>,
    /// Type-erased cross-phase accelerated cache-warmup handoff carrier for the
    /// `agentic_replay` timing mode (empty for every non-accelerated run).
    /// `Send + Sync`; cloned per shard and threaded into both agentic phase plans.
    pub(crate) warmup_handoff: crate::agentic_tree::WarmupHandoffCarrierAny,
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
    pub(crate) transport_config: crate::engine::turn_execution::ExecutionTransportPolicy,
    /// Transport-selected lowering used by every per-thread conversation source.
    pub(crate) request_materializer: Arc<dyn crate::dataset::RequestMaterializer>,
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
    /// Selected admission strategy for this cell's worker threads.
    pub(crate) dispatch_mode: DispatchMode,
    /// Resolved worker-assignment policy, read by the single-coordinator hop
    /// executor ([`DispatchMode::GlobalHop`] and [`DispatchMode::GlobalPush`],
    /// both through [`crate::engine::global_hop::run_single_coordinator`]) when
    /// `workers > 1`. Carried verbatim from the authored `runtime.hop_routing`
    /// (`RoundRobin` when absent) and inert under every other mode and for
    /// `workers == 1`.
    pub(crate) hop_routing: crate::engine::protocol::HopRouting,
    /// Per-phase shared admission gates for `Global` dispatch, built once on the
    /// main thread from this cell's (already cell-level-sliced, not
    /// thread-level-sliced) phase budgets. `None` under every other mode.
    pub(crate) global_admission: Option<Arc<GlobalAdmission>>,
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
        accumulator: Box<MetricsAccumulator>,
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
    pub(crate) fn absorb(&mut self, other: ShardRecords) -> Result<()> {
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
    /// accuracy captures, OR the phase flags. Record ordering is applied once after
    /// all shards are absorbed (retained records only).
    pub(crate) fn absorb(&mut self, other: ScheduledShardOutcome) -> Result<()> {
        self.records.absorb(other.records)?;
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
        credit_materializer: None,
        // `workers == 1` co-located sink: no hop, so routing is inert.
        hop_routing: crate::engine::protocol::HopRouting::RoundRobin,
        virtual_worker_width: None,
        // A `workers == 1` co-located sink runs on this shard's own thread; the
        // shard labels its records itself through the capture below.
        worker_labels: None,
    })?;
    // Each `Sharded`/`Global` worker thread slices the authored phases into its
    // own `1/W` share (see `slice_phase_for_thread`); the pipeline below is shared
    // with the `GlobalHop` coordinator, which passes the full un-thread-sliced
    // phases instead.
    let sliced_phases: Vec<PhaseSpec> = shared
        .phases
        .iter()
        .map(|phase| {
            crate::engine::sharded_scheduled::slice_phase_for_thread(
                phase,
                thread_id,
                shared.workers,
                shared.dispatch_mode,
            )
        })
        .collect();
    // This thread IS the executing worker for every request it issues (the
    // `workers == 1` backend above is a co-located sink, so there is no hop that
    // could re-attribute them). Label its records with the two-level partition
    // index, which is unique across the whole `(cell × thread)` grid and is the
    // same residue class that decides which corpus positions this thread draws.
    let worker_label = crate::engine::records::worker_label(partition);
    execute_scheduled_pipeline(
        shared,
        thread_id,
        partition,
        sliced_phases,
        clock,
        execution_backend,
        Some(worker_label),
    )
    .await
}

/// Build and run one complete scheduled pipeline over `partition` with
/// `sliced_phases`, dispatching every issued turn through `execution_backend`.
///
/// Shared by two callers:
/// - [`execute_scheduled_shard`] runs one pipeline per `Sharded`/`Global` worker
///   thread, over that thread's `1/W` nested partition and thread-sliced phases,
///   dispatching to a co-located (`workers == 1`) transport sink.
/// - [`crate::engine::global_hop::run_global_hop`] runs ONE coordinator-owned
///   pipeline over the full cell partition and un-thread-sliced phases,
///   dispatching each turn through the cross-thread thread-per-core hop executor.
///
/// `shard_id` names this pipeline's per-shard artifact temp directory (`0` for
/// the single-coordinator `GlobalHop` pipeline).
///
/// `worker_label` is this pipeline's executing-worker identity for the per-record
/// `worker_id`, and is `Some` exactly when the pipeline runs its own requests: a
/// per-thread shard labels its own records, while the single-coordinator pipeline
/// passes `None` because the hop worker loop knows which thread actually ran each
/// request.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn execute_scheduled_pipeline(
    shared: &ShardedShared,
    shard_id: usize,
    partition: crate::cellular::ModuloCellPartition,
    sliced_phases: Vec<PhaseSpec>,
    clock: Rc<dyn Clock>,
    execution_backend: Rc<dyn RequestExecutor>,
    worker_label: Option<Arc<str>>,
) -> Result<ScheduledShardOutcome> {
    let start_ns = shared.start_ns;
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
                    shard_id,
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
        .with_worker_label(worker_label)
        // Stage each turn's model output text so this shard's streaming outputs.json
        // entry carries it, then drop it in the fold (exact-fold + outputs.json only).
        .with_outputs_capture(shared.exact_fold && shared.outputs_path.is_some()),
    );
    let resolver = shared.table_factory.coordinator_resolver()?;
    let source_factory = PreparedNativeConversationSourceFactory {
        endpoint_resolver: resolver,
        samplers: &shared.samplers,
        materializer: shared.request_materializer.clone(),
        // Inject this thread's partition so its sampler draws only its nested
        // subset of the cell's instances.
        cell_partition: Some(partition),
        // `Global` promises parity with one global limiter; drawing absolute
        // positions extends that promise from admission to the dataset, so its
        // W shards sample the corpus exactly as a single issuer would.
        // `Sharded` keeps the per-shard residue walk — it is the documented
        // throughput opt-in "where byte-exact parity does not matter".
        position_addressed: matches!(shared.dispatch_mode, DispatchMode::Global),
    };

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

        // `GlobalPush` routes every turn as a credit the worker returns out of
        // band; every other mode awaits one dispatch future per request. A lone
        // worker has no cross-thread placement to route to -- `build_native`
        // gives it a co-located sink -- so it keeps the ordinary path, exactly
        // as a lone-worker `GlobalHop` run is just a single-thread run.
        let credit_dispatch =
            matches!(shared.dispatch_mode, DispatchMode::GlobalPush) && shared.workers > 1;
        let mut plans = Vec::with_capacity(sliced_phases.len());
        let mut profiling_index = 0usize;
        for (phase_index, phase) in sliced_phases.iter().enumerate() {
            // Under `global`, a phase with a shared cell-level
            // concurrency gate admits from that gate (this thread's OWN
            // SlotPool::new_global handle over it) instead of the persistent
            // local `shared_resources` pool. See `phase_scheduled_resources`.
            let phase_resources = phase_scheduled_resources(phase, shared, &shared_resources)?;
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
                &phase_resources,
                shared
                    .wants_adaptive_record
                    .then(|| capture.clone() as Rc<dyn AdaptiveTerminalRecordSource>),
                shared.on_failure,
                shared.agentic_trees.clone(),
                shared.warmup_handoff.clone(),
                credit_dispatch,
            )?;
            let profiling_idx = if phase.common().exclude_from_results {
                None
            } else {
                let idx = profiling_index;
                profiling_index += 1;
                Some(idx)
            };
            let identity = phase_identity_from_spec(phase, phase_index, profiling_idx);
            let record_processor: Rc<dyn TurnRecordProcessor> = Rc::new(CapturePhaseProcessor {
                capture: capture.clone(),
                phase: metrics_phase(phase)?,
                identity,
                has_credit_timestamp: !matches!(
                    phase,
                    PhaseSpec::FixedSchedule { .. } | PhaseSpec::AgenticReplay { .. }
                ),
                // Once-per-cell on the main thread; a worker never feeds them.
                live_sink: None,
                heartbeat: None,
            });
            let mut record_processors = vec![record_processor];
            // Static-accuracy captures its terminal responses on the profiling phase
            // only; each shard feeds its own
            // processor, drained into the outcome after the run.
            if !phase.common().exclude_from_results
                && let Some(accuracy_processor) = &shard_accuracy_processor
            {
                record_processors.push(accuracy_processor.clone() as Rc<dyn TurnRecordProcessor>);
            }
            plan = plan
                .with_record_processors(record_processors)
                .with_performance_record_capture(false)
                .with_native_metric_record_dimensions(false)
                // `GlobalPush` routes every turn as a credit the worker returns
                // out of band; every other mode awaits one dispatch future per
                // request. Selected per phase because the phase runtime owns the
                // credit-return loop's lifetime.
                .with_credit_dispatch(credit_dispatch)
                // This pipeline builds its report from the DRAINED WORKER records
                // and reads only the timing recorder off the phase report
                // (`issued_times` below), so the phase's own compatibility and
                // native-metric planes are computed per request and thrown away.
                // True for every dispatch mode here, not just credit dispatch.
                .with_discarded_local_measurement(true);
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
    let shutdown = execution_backend.shutdown().await;
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
            accumulator: Box::new(accumulator),
            errored,
        }
    } else {
        ShardRecords::Retained(capture.finish(&issued_times, drained)?)
    };
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
        accuracy_captures,
        was_cancelled,
        has_warmup,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::records::{CapturedModelOutput, CapturedRecord};
    use crate::metrics_core::{
        ExportContext, MetricTag, MetricsAccumulator, MetricsConfig, RecordIngest,
    };

    /// A retained record stamped with `request_index`, the field `merge_shards`
    /// later sorts on.
    fn retained_record(index: usize) -> CapturedRecord {
        let mut ingest = RecordIngest::minimal(0, 1_000, MetricsPhase::Profiling);
        ingest.request_index = Some(index);
        CapturedRecord {
            uuid: uuid::Uuid::from_u128(index as u128),
            x_correlation_id: format!("corr-{index}"),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest,
        }
    }

    /// A bounded accumulator carrying `count` completed profiling records, the
    /// fold-and-drop shard's per-shard partition.
    fn folded_accumulator(count: usize) -> MetricsAccumulator {
        let mut accumulator = MetricsAccumulator::with_config(MetricsConfig::default());
        for index in 0..count {
            accumulator.process_record(&RecordIngest::minimal(
                index as i64,
                index as i64 + 1_000,
                MetricsPhase::Profiling,
            ));
        }
        accumulator
    }

    fn request_count(accumulator: &MetricsAccumulator) -> Option<f64> {
        accumulator
            .export_results(&ExportContext::phase(MetricsPhase::Profiling))
            .finite_value(MetricTag::RequestCount)
    }

    #[test]
    fn shard_records_absorb_concatenates_retained_in_source_order() {
        let mut left = ShardRecords::Retained(vec![retained_record(0), retained_record(2)]);
        let right = ShardRecords::Retained(vec![retained_record(1)]);
        left.absorb(right).unwrap();
        let ShardRecords::Retained(records) = left else {
            panic!("absorbing two retained shards must stay retained");
        };
        // Concatenation preserves source order (self then other); the coordinator
        // re-sorts by `request_index` later, so absorb itself does not reorder.
        assert_eq!(
            records
                .iter()
                .map(|record| record.ingest.request_index)
                .collect::<Vec<_>>(),
            vec![Some(0), Some(2), Some(1)],
        );
    }

    #[test]
    fn shard_records_absorb_merges_folded_accumulators_and_errored() {
        let mut left = ShardRecords::Folded {
            accumulator: Box::new(folded_accumulator(2)),
            errored: vec![retained_record(10)],
        };
        let right = ShardRecords::Folded {
            accumulator: Box::new(folded_accumulator(3)),
            errored: vec![retained_record(11)],
        };
        left.absorb(right).unwrap();
        let ShardRecords::Folded {
            accumulator,
            errored,
        } = left
        else {
            panic!("absorbing two folded shards must stay folded");
        };
        // The merged accumulator sums both shards' record counts (2 + 3)...
        assert_eq!(request_count(&accumulator), Some(5.0));
        // ...and the errored-record lists concatenate.
        assert_eq!(errored.len(), 2);
    }

    #[test]
    fn shard_records_absorb_rejects_storage_mode_mismatch() {
        let mut retained = ShardRecords::Retained(vec![retained_record(0)]);
        let folded = ShardRecords::Folded {
            accumulator: Box::new(folded_accumulator(1)),
            errored: Vec::new(),
        };
        let error = retained.absorb(folded).unwrap_err().to_string();
        assert!(
            error.contains("storage mode"),
            "mixed retained/folded shards must fail closed, got: {error}"
        );
    }

    #[test]
    fn scheduled_shard_outcome_absorb_concatenates_records_and_ors_flags() {
        let mut left = ScheduledShardOutcome {
            records: ShardRecords::Retained(vec![retained_record(0)]),
            accuracy_captures: Vec::new(),
            was_cancelled: false,
            has_warmup: true,
        };
        let right = ScheduledShardOutcome {
            records: ShardRecords::Retained(vec![retained_record(1)]),
            accuracy_captures: Vec::new(),
            // Cancellation on any shard must propagate; warmup ORs to true.
            was_cancelled: true,
            has_warmup: false,
        };
        left.absorb(right).unwrap();
        assert!(left.was_cancelled, "cancellation ORs across shards");
        assert!(left.has_warmup, "warmup presence ORs across shards");
        let ShardRecords::Retained(records) = left.records else {
            panic!("retained shards stay retained through outcome absorb");
        };
        assert_eq!(
            records.len(),
            2,
            "records concatenate through outcome absorb"
        );
    }
}
