// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dataset, config, distribution, and phase-plan builders for native runs.

use super::*;

pub(crate) struct SyntheticDatasetBuildContext<'a> {
    pub(crate) registry: &'a AIPerfRegistry,
    pub(crate) models: &'a ModelsSpec,
    pub(crate) rng_root: RngRoot,
    pub(crate) tokenizer: &'a dyn TextTokenizer,
    pub(crate) rankings: bool,
    pub(crate) media_generator_factory: Arc<dyn SyntheticMediaGeneratorFactory>,
    pub(crate) requires_raw_token_ids: bool,
}

pub(crate) struct FileDatasetBuildContext<'a> {
    pub(crate) registry: &'a AIPerfRegistry,
    pub(crate) models: &'a ModelsSpec,
    pub(crate) run_rng_root: RngRoot,
    pub(crate) tokenizer: &'a dyn TextTokenizer,
    pub(crate) trace_prompt_storage: Arc<dyn TracePromptStoragePolicy>,
    pub(crate) requires_raw_token_ids: bool,
    pub(crate) consumes_system_message: bool,
}

pub(crate) fn dataset_default_output_tokens(dataset: &Dataset) -> Result<usize> {
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

pub(crate) struct PreparedNativeConversationSourceFactory<'a> {
    pub(crate) endpoint_resolver: Rc<dyn PreparedTurnEndpointResolver>,
    pub(crate) samplers: &'a crate::dataset::SamplerRegistry,
    /// Transport-selected request materialization target.
    pub(crate) materializer: Arc<dyn crate::dataset::RequestMaterializer>,
    /// The dataset instance partition this source draws. `None` reads the
    /// process-global partition; thread-per-core execution injects a per-thread
    /// partition that `AIPERF_CELL_ID`/`_COUNT` cannot express.
    pub(crate) cell_partition: Option<ModuloCellPartition>,
    /// Draw absolute corpus positions rather than recycling this shard's own
    /// residue class, so the cell's draw sequence matches a single issuer's.
    /// `Global` only: `Sharded` is the explicit throughput mode where per-shard
    /// partitioning is the point.
    pub(crate) position_addressed: bool,
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
                self.position_addressed,
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
                self.position_addressed,
            )?
        };
        Ok(Box::new(
            source
                .with_request_materializer(self.materializer.clone())
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
    // Side-channel subagent join-gate specs (empty for every non-agentic run).
    // Consumed only by the `agentic_replay` phase branch below.
    agentic_trees: std::sync::Arc<Vec<crate::agentic_tree::TreeSpec>>,
    // Cross-phase accelerated cache-warmup handoff carrier (empty for every
    // non-accelerated run). Consumed only by the `agentic_replay` phase branch.
    warmup_handoff: crate::agentic_tree::WarmupHandoffCarrierAny,
    // Whether this phase routes single-turn credits whose body the WORKER
    // builds (`--dispatch global-push`); see
    // `RequestRateWorkload::with_deferred_single_turn_bodies`.
    defer_single_turn_bodies: bool,
) -> Result<ScheduledPhasePlan> {
    // Agentic replay consumes both fields during normal phase construction below.
    let _ = &agentic_trees;
    let _ = &warmup_handoff;
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
                .with_failure_policy(on_failure)
                // Under `global`/`global-hop` dispatch this phase paces against
                // the cell-shared gate; `None` (sharded / single-thread) leaves
                // local `intervals` pacing intact.
                .with_rate_gate(shared.rate.clone())
                .with_deferred_single_turn_bodies(defer_single_turn_bodies),
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
        PhaseSpec::AgenticReplay {
            start_min_ratio,
            start_max_ratio,
            idle_gap_cap_seconds,
            system_idle_gap_cap_seconds,
            burst_phase_starts,
            ..
        } => {
            ensure!(
                phase.common().concurrency_ramp.is_none()
                    && phase.common().prefill_ramp.is_none()
                    && phase.common().rate_ramp.is_none(),
                "agentic_replay phases own their dispatch timing and do not accept ramps"
            );
            {
                use crate::agentic_replay::{
                    AgenticPhase, AgenticReplayConfig, AgenticReplayWorkload,
                };
                let agentic_phase = match phase.common().semantic_role() {
                    crate::engine::protocol::PhaseRoleSpec::Warmup => AgenticPhase::Warmup,
                    crate::engine::protocol::PhaseRoleSpec::Profiling => AgenticPhase::Profiling,
                };
                let config = AgenticReplayConfig {
                    phase: agentic_phase,
                    start_min_ratio: *start_min_ratio,
                    start_max_ratio: *start_max_ratio,
                    idle_gap_cap_ms: idle_gap_cap_seconds.map(|s| s * 1000.0),
                    system_idle_gap_cap_ms: system_idle_gap_cap_seconds.map(|s| s * 1000.0),
                    burst_phase_starts: *burst_phase_starts,
                    // Base t\* seed is dataset-level (phase-independent) so the
                    // WARMUP and PROFILING instances sample the SAME t\* per lane.
                    random_seed: dataset_rng_root
                        .derive_seed_or_entropy("agentic_replay.tstar_base"),
                    benchmark_id: benchmark_id.to_string(),
                    cache_bust_target: crate::agentx::cache_bust::CacheBustTarget::FirstTurnPrefix,
                    // Gating is a profiling concern; the warmup instance carries
                    // no trees (it primes turn n-1 and drains, never joins).
                    trees: match agentic_phase {
                        AgenticPhase::Profiling => Rc::new(agentic_trees.as_ref().clone()),
                        AgenticPhase::Warmup => Rc::new(Vec::new()),
                    },
                    // Accelerated cache-warmup is a WARMUP-phase concern; the
                    // authored `agentic_cache_warmup_duration` is threaded onto
                    // the warmup phase's common by `lower_legacy_agentic`. The
                    // PROFILING instance never drives the substage.
                    cache_warmup_duration_s: match agentic_phase {
                        AgenticPhase::Warmup => phase.common().agentic_cache_warmup_duration,
                        AgenticPhase::Profiling => None,
                    },
                    // Force `max_tokens=1` on the WARMUP instance when accelerated
                    // cache-warmup is armed (Python `_WARMUP_MAX_TOKENS=1`); the
                    // PROFILING instance and non-accelerated warmup keep recorded caps.
                    max_tokens_override: match agentic_phase {
                        AgenticPhase::Warmup
                            if phase.common().agentic_cache_warmup_duration.is_some() =>
                        {
                            Some(1)
                        }
                        _ => None,
                    },
                    // Shared cross-phase handoff carrier: both agentic instances
                    // downcast the SAME type-erased carrier so WARMUP's finalize is
                    // visible to PROFILING's resume. A non-agentic/empty carrier
                    // downcasts to `None` and leaves profiling as the non-accel path.
                    warmup_handoff: crate::agentic_replay::downcast_warmup_handoff_carrier(
                        &warmup_handoff,
                    )
                    .unwrap_or_else(crate::agentic_replay::new_warmup_handoff_carrier),
                };
                let workload =
                    Rc::new(AgenticReplayWorkload::new(source, config)?) as Rc<dyn Workload>;
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
                    // enforce_stop=false: the agentic mode owns its dispatch timing
                    // (no rate/concurrency gate). The phase runtime still enforces
                    // the duration budget and cancels pending recycles at expiry;
                    // recycle re-draws until then to sustain the run.
                    false,
                    Rc::new(crate::phase_runtime::NoopScheduledPhaseResources),
                    None,
                )
            }
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
    spec: &SyntheticDatasetSpec,
    context: SyntheticDatasetBuildContext<'_>,
) -> Result<Dataset> {
    let SyntheticDatasetBuildContext {
        registry,
        models,
        rng_root,
        tokenizer,
        rankings,
        media_generator_factory,
        requires_raw_token_ids,
    } = context;
    let mut compose = compose_config(models, rng_root)?;
    compose.media_generator_factory = media_generator_factory;
    compose.requires_raw_token_ids = requires_raw_token_ids;
    compose.prompt_generator = synthetic_prompt_generator(spec.prompts.as_ref())?;
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
        if let Some(authored_ratio) = prompts.random_range_ratio {
            ensure!(
                prompts.sequence_distribution.is_none(),
                "random_range_ratio cannot be combined with sequence_distribution"
            );
            let input_mean = fixed_range_mean(
                prompts.isl.as_ref(),
                "random_range_ratio requires an explicitly authored fixed ISL",
            )?;
            let output_mean = fixed_range_mean(
                prompts.osl.as_ref(),
                "random_range_ratio requires an explicitly authored fixed OSL",
            )?;
            let special_tokens = i64::try_from(tokenizer.num_special_tokens_to_add())?;
            let policy = crate::dataset::RandomRangePlan::new(
                prompts.random_corpus_style,
                input_mean,
                output_mean,
                authored_ratio.checked(prompts.random_corpus_style)?,
                special_tokens,
            )?;
            policy.validate_minimum_input(
                spec.prefix_prompts
                    .as_ref()
                    .and_then(|prefixes| prefixes.length)
                    .unwrap_or(0),
            )?;
            let vocab_size = tokenizer
                .vocab_size()
                .filter(|size| *size > 0)
                .ok_or_else(|| anyhow!("random_range_ratio requires a tokenizer vocabulary"))?;
            let seed = rng_root
                .seed()
                .unwrap_or_else(|| rng_root.derive_seed_or_entropy("dataset.random_range"));
            let seeded = policy.preseed(spec.entries, seed, vocab_size)?;
            if prompts.corpus.as_deref() == Some("random") {
                compose.prompt_generator = Arc::new(
                    CorpusPromptGeneratorFactory::random_reference_plan(seeded.clone()),
                );
            }
            compose.random_range_plan = Some(seeded);
            compose.sequence_length_distribution = None;
        }
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

fn fixed_range_mean(spec: Option<&DistributionSpec>, missing: &str) -> Result<i64> {
    let value = match spec.ok_or_else(|| anyhow!(missing.to_string()))? {
        DistributionSpec::Fixed(value) => value.value,
        DistributionSpec::Normal(value) if value.stddev == 0.0 => value.mean,
        DistributionSpec::Normal(_) => {
            bail!("random_range_ratio cannot be combined with non-zero sequence stddev")
        }
        _ => bail!("random_range_ratio requires fixed ISL and OSL distributions"),
    };
    ensure!(
        value.is_finite() && value >= 0.0 && value <= i64::MAX as f64,
        "random_range_ratio mean must be finite, non-negative, and representable"
    );
    Ok(value as i64)
}

fn prompt_generator_factory(corpus: &str) -> Result<Arc<dyn PromptGeneratorFactory>> {
    let factory: Arc<dyn PromptGeneratorFactory> = match corpus {
        "sonnet" => Arc::new(CorpusPromptGeneratorFactory::sonnet()),
        "coding" => Arc::new(CorpusPromptGeneratorFactory::coding()),
        "random" => Arc::new(CorpusPromptGeneratorFactory::random()),
        other => bail!("unknown prompt corpus {other:?}; expected sonnet, coding, or random"),
    };
    Ok(factory)
}

fn synthetic_prompt_generator(
    prompts: Option<&SyntheticPromptsSpec>,
) -> Result<Arc<dyn PromptGeneratorFactory>> {
    match prompts.and_then(|prompts| prompts.corpus.as_deref()) {
        Some("random") => Ok(Arc::new(CorpusPromptGeneratorFactory::random_with_style(
            prompts.map_or(crate::dataset::RandomCorpusStyle::default(), |prompts| {
                prompts.random_corpus_style
            }),
        ))),
        corpus => prompt_generator_factory(corpus.unwrap_or("sonnet")),
    }
}

fn authored_prompt_generator(
    prompts: Option<&PromptSelectionSpec>,
) -> Result<Option<Arc<dyn PromptGeneratorFactory>>> {
    prompts
        .and_then(|prompts| prompts.corpus.as_deref())
        .map(prompt_generator_factory)
        .transpose()
}

pub(crate) fn compose_config(models: &ModelsSpec, rng_root: RngRoot) -> Result<ComposeConfig> {
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
    spec: &FileDatasetSpec,
    context: FileDatasetBuildContext<'_>,
) -> Result<Dataset> {
    let FileDatasetBuildContext {
        registry,
        models,
        run_rng_root,
        tokenizer,
        trace_prompt_storage,
        requires_raw_token_ids,
        consumes_system_message,
    } = context;
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
    compose.hoist_leading_system_message = consumes_system_message;
    compose.output_length_distribution = spec.osl.as_ref().map(distribution).transpose()?;
    compose.format_options = spec.options.clone();
    compose.trace_prompt_storage = trace_prompt_storage;
    if spec.prefetch_media_urls {
        // Fetch remote image URLs once, now, before any credits are issued.
        compose.media_resolver = Arc::new(PrefetchMediaResolver::new());
    }
    if let Some(generator) = authored_prompt_generator(spec.prompts.as_ref())? {
        compose.prompt_generator = generator;
    }
    if let Some(synthesis) = &spec.synthesis {
        // baseten_trace replays recorded prompts verbatim over a distinct
        // composer that never consumes `compose.trace_synthesis`, so it
        // only honors the isl/osl caps, not the compounding/reshaping
        // fields. Mirrors baseten_trace.py's own rejection check, which
        // only guards speedup_ratio and the three prefix/prompt
        // multipliers (they'd desync the forwarded hash_ids KV hints);
        // max_isl/max_osl-only synthesis is accepted there.
        if spec.format == "baseten_trace" {
            ensure!(
                synthesis.speedup_ratio == 1.0
                    && synthesis.prefix_len_multiplier == 1.0
                    && synthesis.prefix_root_multiplier == 1
                    && synthesis.prompt_len_multiplier == 1.0,
                "trace synthesis is not supported by the baseten_trace loader \
                 beyond max_isl/max_osl: it replays recorded prompts verbatim, \
                 so hash-reshaping synthesis cannot change the sent prompt and \
                 would desync the forwarded hash_ids KV hints"
            );
            compose.max_output_tokens = synthesis.max_osl;
        } else {
            ensure!(
                matches!(
                    spec.format.as_str(),
                    "mooncake_trace" | "bailian_trace" | "burst_gpt_trace"
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
    }
    let source = match (&spec.path, &spec.records) {
        (Some(path), None) => DatasetSource::Path(path.clone()),
        (None, Some(records)) => DatasetSource::Inline(records.clone()),
        _ => unreachable!("source exclusivity validated above"),
    };
    let mut load = LoadConfig::new(source);
    load.max_rows = spec.entries;
    load.sampling_strategy = Some(spec.sampling.clone());
    load.options = spec.options.clone();
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
    if spec.prefetch_media_urls {
        // Fetch remote image URLs once, now, before any credits are issued.
        compose.media_resolver = Arc::new(PrefetchMediaResolver::new());
    }
    if let Some(generator) = authored_prompt_generator(spec.prompts.as_ref())? {
        compose.prompt_generator = generator;
    }
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

pub(crate) fn synthetic_config(spec: &SyntheticDatasetSpec) -> Result<SyntheticDatasetConfig> {
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
            ensure!(
                prompts.prefix_reuse_fraction.is_finite()
                    && (0.0..=1.0).contains(&prompts.prefix_reuse_fraction),
                "synthetic prompt prefix_reuse_fraction must be within [0, 1]"
            );
            ensure!(
                prompts.prefix_reuse_ratio.is_finite()
                    && (0.0..=1.0).contains(&prompts.prefix_reuse_ratio),
                "synthetic prompt prefix_reuse_ratio must be within [0, 1]"
            );
            Ok(
                (input_tokens.expected_value() > 0.0).then_some(SyntheticPromptConfig {
                    input_tokens,
                    batch_size: prompts.batch_size,
                    prefix_reuse_fraction: prompts.prefix_reuse_fraction,
                    prefix_reuse_ratio: prompts.prefix_reuse_ratio,
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

pub(crate) fn synthetic_prefixes(
    spec: Option<&SyntheticPrefixPromptsSpec>,
) -> SyntheticPrefixConfig {
    spec.map_or_else(SyntheticPrefixConfig::default, |prefixes| {
        SyntheticPrefixConfig {
            pool_size: prefixes.pool_size,
            prefix_tokens: prefixes.length,
            shared_system_tokens: prefixes.shared_system_length,
            user_context_tokens: prefixes.user_context_length,
        }
    })
}

pub(crate) fn synthetic_image(spec: &SyntheticImageSpec) -> Result<SyntheticImageConfig> {
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

pub(crate) fn synthetic_audio(spec: &SyntheticAudioSpec) -> Result<SyntheticAudioConfig> {
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

pub(crate) fn synthetic_video(spec: &SyntheticVideoSpec) -> Result<SyntheticVideoConfig> {
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

pub(crate) fn khz_to_hz(value: f64, field: &str) -> Result<u32> {
    let hz = value * 1_000.0;
    ensure!(
        value.is_finite() && value > 0.0 && hz <= f64::from(u32::MAX),
        "{field} must be finite, positive, and representable in hertz"
    );
    Ok(hz.round_ties_even() as u32)
}

pub(crate) fn sequence_length_distribution(
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

pub(crate) fn distribution_expected_i64(spec: &DistributionSpec, field: &str) -> Result<i64> {
    let expected = distribution(spec)?.expected_value();
    ensure!(
        expected.is_finite() && expected > 0.0 && expected < i64::MAX as f64,
        "{field} expected value must be positive and representable as i64"
    );
    Ok(expected as i64)
}

pub(crate) const fn distribution_normal_stddev(spec: &DistributionSpec) -> f64 {
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
    let steady_state = crate::metrics_core::SteadyStateConfig {
        enabled: spec.steady_state.enabled,
        fraction: spec
            .steady_state
            .fraction
            .unwrap_or(crate::metrics_core::DEFAULT_STEADY_STATE_FRACTION),
        hybrid_latency: spec.steady_state.hybrid_latency,
    };
    Ok(MetricsConfig {
        slice_duration_ns,
        slos,
        use_server_token_count,
        storage_mode,
        steady_state,
        ..MetricsConfig::default()
    })
}

pub(crate) fn metrics_phase(spec: &PhaseSpec) -> Result<MetricsPhase> {
    if spec.common().is_warmup() {
        Ok(MetricsPhase::Warmup)
    } else {
        Ok(MetricsPhase::Profiling)
    }
}

pub(crate) fn artifact_path(root: &Path, relative: &Path, field: &str) -> Result<PathBuf> {
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
    let kind = if common.is_warmup() {
        PhaseKind::Warmup
    } else {
        PhaseKind::Profiling
    };
    // The accelerated cache-warmup phase (`--agentic-cache-warmup-duration`) is
    // duration-driven: its own `execute` arms a Clock drain timer and self-drains,
    // issuing MANY pressure turns (live-trajectory replay under compression), not a
    // fixed count. Its `common.requests` carries the static-prime count SOLELY to
    // offset the following PROFILING phase's ordinal base (see
    // `compute_phase_ordinal_bases`), NOT as a send bound. Passing it through as
    // `total_expected_requests` would freeze the phase's progress tracker after that
    // many sends (independent of `enforce_stop`), and the next pressure turn would
    // hit `record a send after sending complete`. Drop the count bound for this phase
    // (the ordinal reservation still reads `common.requests`).
    let accelerated_warmup = matches!(spec, PhaseSpec::AgenticReplay { .. })
        && common.is_warmup()
        && common.agentic_cache_warmup_duration.is_some();
    let stop = StopConfig {
        total_expected_requests: if accelerated_warmup {
            None
        } else {
            common.requests
        },
        expected_num_sessions: common.sessions,
        expected_duration_ns: common.duration.map(seconds_to_ns).transpose()?,
    };
    let mut phase = PhaseConfig::new(&common.name, kind, stop)
        .with_seamless(seamless_to_next)
        .with_concurrency(spec.concurrency(), common.prefill_concurrency);
    if let Some(grace) = common.grace_period {
        phase = phase.with_grace_period(GracePeriod::Finite(seconds_to_ns(grace)?));
    }
    if let Some(threshold) = common.failed_request_threshold {
        ensure!(
            threshold.is_finite() && (0.0..=1.0).contains(&threshold),
            "failed_request_threshold must be finite and in [0.0, 1.0], got {threshold}"
        );
        phase = phase.with_failed_request_threshold(Some(threshold));
    }
    phase.validate()?;
    Ok(phase)
}

pub(crate) fn ancillary_policies(
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

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn synthetic(value: serde_json::Value) -> SyntheticDatasetSpec {
        serde_json::from_value(value).unwrap()
    }

    fn agentic_warmup_spec(cache_warmup_duration: Option<f64>) -> PhaseSpec {
        let common: crate::engine::protocol::PhaseCommonSpec = serde_json::from_value(json!({
            "name": "warmup",
            "kind": "warmup",
            "exclude_from_results": true,
            "requests": 8,
            "agentic_cache_warmup_duration": cache_warmup_duration,
        }))
        .unwrap();
        PhaseSpec::AgenticReplay {
            common,
            start_min_ratio: 0.0,
            start_max_ratio: 1.0,
            idle_gap_cap_seconds: Some(10.0),
            system_idle_gap_cap_seconds: None,
            burst_phase_starts: false,
        }
    }

    #[test]
    fn accelerated_warmup_phase_drops_the_count_stop_bound() {
        // Accelerated cache-warmup is duration-driven and emits many pressure turns;
        // its `requests` reserves ordinal space only, so `phase_config` must NOT turn
        // it into a send bound (which would freeze the tracker mid-pressure).
        let accel = phase_config(&agentic_warmup_spec(Some(3.0)), false).unwrap();
        assert_eq!(
            accel.stop.total_expected_requests, None,
            "accelerated warmup must not carry a request stop bound"
        );
        // A non-accelerated agentic warmup keeps its authored request bound.
        let plain = phase_config(&agentic_warmup_spec(None), false).unwrap();
        assert_eq!(plain.stop.total_expected_requests, Some(8));
    }

    fn models() -> ModelsSpec {
        serde_json::from_value(json!({
            "strategy": "round_robin",
            "items": [{"name": "mock-model"}]
        }))
        .unwrap()
    }

    async fn synthetic_dataset_for_prompt_corpus(
        corpus: &str,
        input_tokens: usize,
        tokenizer: &dyn TextTokenizer,
        requires_raw_token_ids: bool,
    ) -> Dataset {
        let spec = synthetic(json!({
            "entries": 1,
            "random_seed": 17,
            "prompts": {
                "isl": {"value": input_tokens as f64},
                "corpus": corpus
            }
        }));
        let registry = AIPerfRegistry::builtin().unwrap();
        build_synthetic_dataset(
            &spec,
            SyntheticDatasetBuildContext {
                registry: &registry,
                models: &models(),
                rng_root: RngRoot::new(Some(17)),
                tokenizer,
                rankings: false,
                media_generator_factory: Arc::new(
                    crate::dataset::NativeSyntheticMediaGeneratorFactory::default(),
                ),
                requires_raw_token_ids,
            },
        )
        .await
        .unwrap()
    }

    async fn file_trace_dataset_for_prompt_corpus(
        corpus: &str,
        hash_ids: serde_json::Value,
        tokenizer: &dyn TextTokenizer,
    ) -> Dataset {
        let spec: FileDatasetSpec = serde_json::from_value(json!({
            "format": "mooncake_trace",
            "records": [{
                "session_id": "root",
                "input_length": 16,
                "output_length": 3,
                "hash_ids": hash_ids,
            }],
            "prompts": {
                "corpus": corpus
            }
        }))
        .unwrap();
        let registry = AIPerfRegistry::builtin().unwrap();
        build_file_dataset(
            &spec,
            FileDatasetBuildContext {
                registry: &registry,
                models: &models(),
                run_rng_root: RngRoot::new(Some(23)),
                tokenizer,
                trace_prompt_storage: Arc::new(crate::dataset::MaterializedTracePromptStorage),
                requires_raw_token_ids: false,
                consumes_system_message: false,
            },
        )
        .await
        .unwrap()
    }

    fn first_text_tokens(dataset: &Dataset, tokenizer: &dyn TextTokenizer) -> Vec<u32> {
        let handle = dataset.conversations()[0].turns[0].content[0].handles[0];
        let crate::dataset::segment::Payload::Text {
            bytes, token_count, ..
        } = dataset.segments().get(handle).unwrap()
        else {
            panic!("synthetic prompt must be stored as text");
        };
        let text = std::str::from_utf8(bytes).unwrap();
        let tokens = tokenizer.encode(text).unwrap();
        assert_eq!(tokens.len(), *token_count as usize);
        tokens
    }

    #[cfg(feature = "parquet")]
    fn write_baseten_fixture(directory: &std::path::Path) -> PathBuf {
        use std::sync::Arc as StdArc;

        use parquet::data_type::{ByteArrayType, Int64Type};
        use parquet::file::writer::SerializedFileWriter;
        use parquet::schema::parser::parse_message_type;

        let schema = StdArc::new(
            parse_message_type(
                "message schema {
                    REQUIRED INT64 timestamp_start_unix_ms;
                    REQUIRED BYTE_ARRAY prompt (UTF8);
                    REQUIRED INT64 input_tokens;
                    REQUIRED INT64 output_tokens;
                    REQUIRED BYTE_ARRAY provided_session_id (UTF8);
                    REQUIRED INT64 duration_e2e_ms;
                }",
            )
            .unwrap(),
        );
        let path = directory.join("baseten.parquet");
        let file = std::fs::File::create(&path).unwrap();
        let mut writer = SerializedFileWriter::new(file, schema, Default::default()).unwrap();
        let mut row_group = writer.next_row_group().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(&[0_i64], None, None)
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<ByteArrayType>()
            .write_batch(
                &[parquet::data_type::ByteArray::from(b"hi".to_vec())],
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(&[3_i64], None, None)
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(&[100_i64], None, None)
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<ByteArrayType>()
            .write_batch(
                &[parquet::data_type::ByteArray::from(b"s1".to_vec())],
                None,
                None,
            )
            .unwrap();
        column.close().unwrap();

        let mut column = row_group.next_column().unwrap().unwrap();
        column
            .typed::<Int64Type>()
            .write_batch(&[0_i64], None, None)
            .unwrap();
        column.close().unwrap();

        row_group.close().unwrap();
        writer.close().unwrap();
        path
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
    async fn prompt_corpus_selection_reaches_native_text_dataset() {
        let tokenizer = TiktokenTokenizer::builtin();
        let sonnet = synthetic_dataset_for_prompt_corpus("sonnet", 16, &tokenizer, false).await;
        let coding = synthetic_dataset_for_prompt_corpus("coding", 16, &tokenizer, false).await;
        let random = synthetic_dataset_for_prompt_corpus("random", 16, &tokenizer, false).await;

        let sonnet_tokens = first_text_tokens(&sonnet, &tokenizer);
        let coding_tokens = first_text_tokens(&coding, &tokenizer);
        let random_tokens = first_text_tokens(&random, &tokenizer);

        assert_eq!(sonnet.conversations()[0].turns[0].input_tokens, Some(16));
        assert_eq!(coding.conversations()[0].turns[0].input_tokens, Some(16));
        assert_eq!(random.conversations()[0].turns[0].input_tokens, Some(16));
        assert_ne!(coding_tokens, sonnet_tokens);
        assert_ne!(random_tokens, sonnet_tokens);
    }

    #[tokio::test]
    async fn random_prompt_corpus_reaches_native_raw_token_dataset() {
        use crate::dataset::tokenizer::NoDecodeTokenizer;

        let dataset =
            synthetic_dataset_for_prompt_corpus("random", 8, &NoDecodeTokenizer, true).await;
        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.input_tokens, Some(8));
        let handle = *turn.body.first().expect("raw token handle");
        let crate::dataset::segment::Payload::TokenIds { token_ids } =
            dataset.segments().get(handle).unwrap()
        else {
            panic!("random raw-token prompt must be stored as token IDs");
        };
        assert_eq!(token_ids.len(), 8);
        assert!(!token_ids.contains(&9));
    }

    #[tokio::test]
    async fn random_range_ratio_reaches_composed_native_lengths_in_reference_order() {
        use crate::dataset::tokenizer::NoDecodeTokenizer;

        let spec = synthetic(json!({
            "entries": 4,
            "prompts": {
                "isl": {"value": 100.0},
                "osl": {"value": 20.0},
                "corpus": "random",
                "random_range_ratio": {"input": 0.3, "output": 0.5},
                "random_corpus_style": "vllm"
            }
        }));
        let registry = AIPerfRegistry::builtin().unwrap();
        let dataset = build_synthetic_dataset(
            &spec,
            SyntheticDatasetBuildContext {
                registry: &registry,
                models: &models(),
                rng_root: RngRoot::new(Some(42)),
                tokenizer: &NoDecodeTokenizer,
                rankings: false,
                media_generator_factory: Arc::new(
                    crate::dataset::NativeSyntheticMediaGeneratorFactory::default(),
                ),
                requires_raw_token_ids: true,
            },
        )
        .await
        .unwrap();

        let inputs: Vec<_> = dataset
            .conversations()
            .iter()
            .map(|conversation| conversation.turns[0].input_tokens.unwrap())
            .collect();
        let outputs: Vec<_> = dataset
            .conversations()
            .iter()
            .map(|conversation| conversation.turns[0].max_tokens.unwrap())
            .collect();
        assert_eq!(inputs, [75, 117, 109, 96]);
        assert_eq!(outputs, [19, 28, 11, 24]);
    }

    #[tokio::test]
    async fn prompt_corpus_selection_reaches_count_only_file_trace_dataset() {
        let tokenizer = TiktokenTokenizer::builtin();
        let sonnet = file_trace_dataset_for_prompt_corpus("sonnet", json!([]), &tokenizer).await;
        let coding = file_trace_dataset_for_prompt_corpus("coding", json!([]), &tokenizer).await;
        let random = file_trace_dataset_for_prompt_corpus("random", json!([]), &tokenizer).await;

        let sonnet_tokens = first_text_tokens(&sonnet, &tokenizer);
        let coding_tokens = first_text_tokens(&coding, &tokenizer);
        let random_tokens = first_text_tokens(&random, &tokenizer);

        assert_eq!(sonnet.conversations()[0].turns[0].input_tokens, Some(16));
        assert_eq!(coding.conversations()[0].turns[0].input_tokens, Some(16));
        assert_eq!(random.conversations()[0].turns[0].input_tokens, Some(16));
        assert_ne!(coding_tokens, sonnet_tokens);
        assert_ne!(random_tokens, sonnet_tokens);
    }

    #[tokio::test]
    async fn prompt_corpus_selection_reaches_hash_backed_file_trace_dataset() {
        let tokenizer = TiktokenTokenizer::builtin();
        let sonnet = file_trace_dataset_for_prompt_corpus("sonnet", json!([7]), &tokenizer).await;
        let coding = file_trace_dataset_for_prompt_corpus("coding", json!([7]), &tokenizer).await;
        let random = file_trace_dataset_for_prompt_corpus("random", json!([7]), &tokenizer).await;

        let sonnet_tokens = first_text_tokens(&sonnet, &tokenizer);
        let coding_tokens = first_text_tokens(&coding, &tokenizer);
        let random_tokens = first_text_tokens(&random, &tokenizer);

        assert_eq!(sonnet.conversations()[0].turns[0].input_tokens, Some(16));
        assert_eq!(coding.conversations()[0].turns[0].input_tokens, Some(16));
        assert_eq!(random.conversations()[0].turns[0].input_tokens, Some(16));
        assert_ne!(coding_tokens, sonnet_tokens);
        assert_ne!(random_tokens, sonnet_tokens);
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
            &spec,
            SyntheticDatasetBuildContext {
                registry: &registry,
                models: &models(),
                rng_root: RngRoot::new(Some(73)),
                tokenizer: &TiktokenTokenizer::builtin(),
                rankings: false,
                media_generator_factory: Arc::new(
                    crate::dataset::NativeSyntheticMediaGeneratorFactory::default(),
                ),
                requires_raw_token_ids: false,
            },
        )
        .await
        .unwrap();

        assert_eq!(dataset.metadata().sampling_strategy, "shuffle");
        assert_eq!(dataset.conversations().len(), 2);
        for conversation in dataset.conversations() {
            assert_eq!(conversation.turns[0].max_tokens, Some(3));
            assert_eq!(conversation.turns[0].input_tokens, Some(6));
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
            &spec,
            SyntheticDatasetBuildContext {
                registry: &registry,
                models: &models(),
                rng_root: RngRoot::new(Some(3)),
                tokenizer: &TiktokenTokenizer::builtin(),
                rankings: true,
                media_generator_factory: Arc::new(
                    crate::dataset::NativeSyntheticMediaGeneratorFactory::default(),
                ),
                requires_raw_token_ids: false,
            },
        )
        .await
        .unwrap();

        let turn = &dataset.conversations()[0].turns[0];
        assert_eq!(turn.content[0].name, "query");
        assert_eq!(turn.content[1].name, "passages");
        assert_eq!(turn.content[1].handles.len(), 2);
        assert_eq!(turn.input_tokens, Some(14));
    }

    #[cfg(feature = "parquet")]
    #[tokio::test]
    async fn baseten_trace_accepts_max_isl_max_osl_only_synthesis_and_rejects_reshaping() {
        // Mirrors baseten_trace.py's __init__ rejection check: only
        // speedup_ratio and the three prefix/prompt multipliers are
        // rejected (they'd desync the forwarded hash_ids KV hints);
        // max_isl/max_osl-only synthesis is accepted.
        use crate::dataset::compose::MaterializedTracePromptStorage;

        let directory = tempfile::tempdir().unwrap();
        let path = write_baseten_fixture(directory.path());
        let registry = AIPerfRegistry::builtin().unwrap();

        let accepted: FileDatasetSpec = serde_json::from_value(json!({
            "path": path,
            "format": "baseten_trace",
            "synthesis": {
                "speedup_ratio": 1.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0,
                "max_osl": 10
            }
        }))
        .unwrap();
        let dataset = build_file_dataset(
            &accepted,
            FileDatasetBuildContext {
                registry: &registry,
                models: &models(),
                run_rng_root: RngRoot::new(Some(1)),
                tokenizer: &TiktokenTokenizer::builtin(),
                trace_prompt_storage: Arc::new(MaterializedTracePromptStorage),
                requires_raw_token_ids: false,
                consumes_system_message: false,
            },
        )
        .await
        .unwrap();
        assert_eq!(dataset.conversations()[0].turns[0].max_tokens, Some(10));

        let rejected: FileDatasetSpec = serde_json::from_value(json!({
            "path": path,
            "format": "baseten_trace",
            "synthesis": {
                "speedup_ratio": 2.0,
                "prefix_len_multiplier": 1.0,
                "prefix_root_multiplier": 1,
                "prompt_len_multiplier": 1.0,
                "output_len_multiplier": 1.0
            }
        }))
        .unwrap();
        let error = build_file_dataset(
            &rejected,
            FileDatasetBuildContext {
                registry: &registry,
                models: &models(),
                run_rng_root: RngRoot::new(Some(1)),
                tokenizer: &TiktokenTokenizer::builtin(),
                trace_prompt_storage: Arc::new(MaterializedTracePromptStorage),
                requires_raw_token_ids: false,
                consumes_system_message: false,
            },
        )
        .await
        .unwrap_err();
        assert!(
            error.to_string().contains("baseten_trace loader"),
            "unexpected error: {error}"
        );
    }
}
