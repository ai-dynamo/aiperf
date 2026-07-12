// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native construction and execution of one resolved benchmark run.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
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
use aiperf::http::TransportSink;
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
use aiperf_clock::{Clock, RealClock};
use aiperf_dataset::{
    ComposeConfig, Dataset, DatasetSource, HuggingFaceTokenizer, LoadConfig, ModelId,
    ModelSelector, ModelSelectorFactory, RandomModelSelectorFactory,
    RoundRobinModelSelectorFactory, SourceImageSampling, SyntheticAudioConfig,
    SyntheticAudioFormat, SyntheticDatasetConfig, SyntheticImageConfig, SyntheticImageFormat,
    SyntheticImageSource, SyntheticPrefixConfig, SyntheticPromptConfig, SyntheticRankingsConfig,
    SyntheticVideoAudioConfig, SyntheticVideoConfig, SyntheticVideoFormat, SyntheticVideoPattern,
    TextTokenizer, TiktokenEncoding, TiktokenTokenizer, TraceSynthesisConfig,
};
use aiperf_endpoints::{EndpointConfig, EndpointType};
use aiperf_extensions::AiperfRegistry;
use aiperf_metrics::{
    CATALOG, ExportContext, MetricsAccumulator, MetricsConfig, NativeReport, Phase as MetricsPhase,
    ReportRunInfo, ReportSummary, RunOutcome, SloThreshold,
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
use anyhow::{Context, Result, anyhow, bail, ensure};
use async_trait::async_trait;
use loadgen_core::collector::ReplayTerminalStatus;
use loadgen_core::sink::{
    ObservedEndpointMetrics, ObservedTokenKind, ObservedUsage, RequestObserver,
};
use uuid::Uuid;

use crate::protocol::{
    AccuracySpec, AdaptiveControlVariableSpec, AdaptiveScaleSpec, AdaptiveStepPolicySpec,
    DatasetSpec, DistributionSpec, EndpointSpec, FileDatasetSpec, MetricsSpec,
    ModelSelectionStrategy, ModelsSpec, PhaseSpec, PublicDatasetSourceSpec, PublicDatasetSpec,
    RampSpec, RampStrategySpec, RunRequest, RunTerminal, SequenceDistributionEntrySpec,
    SourceImageSamplingSpec, SyntheticAudioFormatSpec, SyntheticAudioSpec, SyntheticDatasetSpec,
    SyntheticImageFormatSpec, SyntheticImageSpec, SyntheticPrefixPromptsSpec,
    SyntheticVideoFormatSpec, SyntheticVideoPatternSpec, SyntheticVideoSpec,
};
use crate::records::{CapturedRecord, write_records_jsonl};

type PhaseRuntimeParts = (
    Rc<dyn Workload>,
    Rc<RefCell<Box<dyn aiperf_timing::IntervalGenerator>>>,
    Option<Rc<SlotPool>>,
    Option<Rc<SlotPool>>,
    bool,
    Rc<dyn ScheduledPhaseResources>,
    Option<Rc<dyn UserTarget>>,
);

/// Execute exactly one request on a fresh current-thread Tokio runtime.
pub fn execute_run(request: RunRequest) -> Result<RunTerminal> {
    validate_request(&request)?;
    let benchmark_id = request.run.benchmark_id.clone();
    let artifact_dir = request.run.artifact_dir.clone();
    std::fs::create_dir_all(&artifact_dir)
        .with_context(|| format!("creating run artifact directory {}", artifact_dir.display()))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("creating native single-run Tokio runtime")?;
    let local = tokio::task::LocalSet::new();
    let native = local.block_on(&runtime, execute_native(request))?;
    let report_path = artifact_dir.join("native-v2.json");
    write_native_report_json(&native, &report_path)?;
    Ok(RunTerminal::succeeded(benchmark_id, report_path))
}

fn validate_request(request: &RunRequest) -> Result<()> {
    ensure!(
        !request.run.benchmark_id.trim().is_empty(),
        "benchmark_id cannot be empty"
    );
    ensure!(
        !request.run.models.items.is_empty(),
        "at least one model is required"
    );
    ensure!(
        !request.run.endpoint.urls.is_empty(),
        "at least one endpoint URL is required"
    );
    ensure!(
        !request.run.phases.is_empty(),
        "at least one phase is required"
    );
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

async fn execute_native(request: RunRequest) -> Result<NativeReport> {
    let Some(spec) = request.run.accuracy.clone() else {
        return execute_native_inner(request, None).await;
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
    request: RunRequest,
    accuracy: Option<AccuracyWorkerRun<'_>>,
) -> Result<NativeReport> {
    let registry = AiperfRegistry::builtin()?;
    let rng_root = RngRoot::new(request.run.random_seed);
    let dataset_rng_root = dataset_rng_root(&request.run.dataset, rng_root);
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
            &registry,
            &request.run.dataset,
            &request.run.models,
            dataset_rng_root,
            tokenizer.as_ref(),
            request.run.endpoint.endpoint_type,
        )
        .await?
    };
    let endpoint = endpoint_config(&request.run.endpoint)?;
    let default_output_tokens = if prepared_accuracy.is_some() {
        dataset_default_output_tokens(&dataset)?
    } else {
        default_output_tokens(&request.run.dataset)?
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

    let clock: Rc<dyn Clock> = RealClock::new();
    let start_ns = clock.now_ns();
    let capture = Rc::new(RunCapture::new(
        clock.clone(),
        start_ns,
        metrics_config.clone(),
    ));
    let transport = TransportSink::new_multi(
        clock.clone(),
        start_ns,
        &request.run.endpoint.urls,
        primary_model.clone(),
        request.run.endpoint.http2,
    )?;
    let dispatcher: Rc<dyn TurnDispatcher> = Rc::new(ConfiguredDispatcher {
        transport,
        headers: request.run.endpoint.headers.clone(),
        api_key: request.run.endpoint.api_key.clone(),
        session_header: request.run.endpoint.session_header.clone(),
        capture: capture.clone(),
    });

    let shared_session = request
        .run
        .phases
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
    let shared_prefill = request
        .run
        .phases
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
    let request_resources: Rc<dyn ScheduledPhaseResources> = Rc::new(SlotPoolPhaseResources::new(
        shared_session.clone(),
        shared_prefill.clone(),
    ));

    let mut plans = Vec::with_capacity(request.run.phases.len());
    for (phase_index, phase) in request.run.phases.iter().enumerate() {
        let phase_rng = RngRoot::new(
            dataset_rng_root.derive_seed(&format!("runner.phase.{phase_index}.dataset")),
        );
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
            primary_model.clone(),
            default_output_tokens,
            phase_rng,
            endpoint.clone(),
            &registry,
            tokenizer.clone(),
            input_token_counter.clone(),
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
                    shared_session.clone(),
                    shared_prefill.clone(),
                )?) as Rc<dyn Workload>;
                (
                    workload,
                    intervals,
                    shared_session.clone(),
                    shared_prefill.clone(),
                    true,
                    request_resources.clone(),
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
                    "fixed_schedule prefill admission is not configured by protocol v1"
                );
                let schedule_source =
                    Rc::new(DatasetFixedScheduleSource::new(FixedScheduleConfig {
                        auto_offset_timestamps: *auto_offset,
                        start_offset_ms: *start_offset,
                    })?);
                let workload = Rc::new(FixedScheduleWorkload::new(source, schedule_source)?)
                    as Rc<dyn Workload>;
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
        let phase_config = phase_config(phase)?;
        let policies = ancillary_policies(
            phase,
            &request.run.endpoint.urls,
            RngRoot::new(rng_root.derive_seed(&format!("runner.phase.{phase_index}.cancellation"))),
        )?;
        let controller = ramp_controller(
            phase,
            clock.clone(),
            intervals.clone(),
            phase_session.clone(),
            phase_prefill.clone(),
            RngRoot::new(rng_root.derive_seed(&format!("runner.phase.{phase_index}.ramp"))),
        )?;
        let runtime_extension = adaptive_runtime_extension(
            phase,
            &request.run.benchmark_id,
            &request.run.artifact_dir,
            intervals,
            phase_session,
            phase_prefill,
            user_target,
        )?;
        let record_processor: Rc<dyn TurnRecordProcessor> = Rc::new(CapturePhaseProcessor {
            capture: capture.clone(),
            phase: metrics_phase(phase)?,
            has_credit_timestamp: !matches!(phase, PhaseSpec::FixedSchedule { .. }),
        });
        let mut record_processors = vec![record_processor];
        if phase.common().name == "profiling"
            && let Some(accuracy) = &prepared_accuracy
        {
            let processor: Rc<dyn TurnRecordProcessor> = accuracy.processor.clone();
            record_processors.push(processor);
        }
        let mut plan = ScheduledPhasePlan::new(phase_config, workload, policies)
            .with_enforce_stop(enforce_stop)
            .with_start_ns(start_ns)
            .with_resources(resources)
            .with_record_processors(record_processors)
            .with_controller(controller);
        if let Some(extension) = runtime_extension {
            plan = plan.with_runtime_extension(extension);
        }
        plans.push(plan);
    }

    let observer: Rc<dyn PhaseObserver> = Rc::new(NoopPhaseObserver);
    let phased = run_scheduled_phases(plans, clock, dispatcher, observer).await?;
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
    let profiling_metrics =
        accumulator.export_results(&ExportContext::phase(MetricsPhase::Profiling));
    let warmup = phased
        .reports
        .iter()
        .any(|report| report.kind == PhaseKind::Warmup)
        .then(|| accumulator.export_results(&ExportContext::phase(MetricsPhase::Warmup)));
    if let Some(records_path) = &request.run.artifacts.records_path {
        let records_path = artifact_path(&request.run.artifact_dir, records_path, "records_path")?;
        write_records_jsonl(
            &records_path,
            &captured,
            &metrics_config,
            request.run.artifacts.trace,
        )?;
    }
    let mut outcome = RunOutcome {
        run: ReportRunInfo {
            mode: Some("online".into()),
            model: Some(primary_model),
        },
        summary: ReportSummary {
            endpoints_configured: request.run.endpoint.urls,
            ..ReportSummary::default()
        },
        warmup,
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
fn native_conversation_source(
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

async fn build_dataset(
    registry: &AiperfRegistry,
    dataset: &DatasetSpec,
    models: &ModelsSpec,
    rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
    endpoint_type: EndpointType,
) -> Result<Dataset> {
    match dataset {
        DatasetSpec::Synthetic(spec) => {
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
            let rankings = is_rankings_endpoint(endpoint_type);
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
        DatasetSpec::File(spec) => {
            build_file_dataset(registry, spec, models, rng_root, tokenizer).await
        }
        DatasetSpec::Public(spec) => {
            build_public_dataset(registry, spec, models, rng_root, tokenizer).await
        }
    }
}

fn dataset_rng_root(dataset: &DatasetSpec, run_rng_root: RngRoot) -> RngRoot {
    let override_seed = match dataset {
        DatasetSpec::Synthetic(spec) => spec.random_seed,
        DatasetSpec::File(spec) => spec.random_seed,
        DatasetSpec::Public(spec) => spec.random_seed,
    };
    override_seed.map_or(run_rng_root, |seed| RngRoot::new(Some(seed)))
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

async fn build_file_dataset(
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

async fn build_public_dataset(
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

fn default_output_tokens(dataset: &DatasetSpec) -> Result<usize> {
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

fn distribution(spec: &DistributionSpec) -> Result<SamplingDistribution> {
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
        timeout_seconds: spec.timeout_seconds,
        use_legacy_max_tokens: spec.use_legacy_max_tokens,
        use_server_token_count: spec.use_server_token_count,
        extra: (!spec.extra.is_empty()).then(|| spec.extra.clone()),
        ..EndpointConfig::default()
    }
    .validate()
    .map_err(Into::into)
}

fn metrics_config(spec: &MetricsSpec) -> Result<MetricsConfig> {
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

fn load_tokenizer(spec: Option<&str>) -> Result<Arc<dyn TextTokenizer>> {
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
}

impl RunCapture {
    fn new(clock: Rc<dyn Clock>, origin_ns: i64, config: MetricsConfig) -> Self {
        Self {
            observer: Rc::new(NativeMetricsObserver::new(clock.clone(), origin_ns, config)),
            clock,
            origin_ns,
            identities: RefCell::new(Vec::new()),
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

    fn finish(&self, issued_times: &HashMap<Uuid, i64>) -> Result<Vec<CapturedRecord>> {
        let collection = self.observer.finish_with_records();
        let identities = self.identities.borrow();
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
}

#[async_trait(?Send)]
impl TurnRecordProcessor for CapturePhaseProcessor {
    async fn process(&self, credit: &IssuedCredit, _outcome: &TurnDispatchOutcome) -> Result<()> {
        self.capture
            .label(credit, self.phase, self.has_credit_timestamp);
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
    transport: TransportSink,
    headers: BTreeMap<String, String>,
    api_key: Option<String>,
    session_header: Option<String>,
    capture: Rc<RunCapture>,
}

#[async_trait(?Send)]
impl TurnDispatcher for ConfiguredDispatcher {
    async fn dispatch_turn(
        &self,
        mut turn: TurnToSend,
        observer: &dyn RequestObserver,
        on_first_token: &dyn Fn(i64),
    ) -> Result<TurnDispatchOutcome> {
        for (name, value) in &self.headers {
            turn.request_headers
                .entry(name.clone())
                .or_insert_with(|| value.clone());
        }
        if let Some(api_key) = &self.api_key {
            turn.request_headers
                .entry("Authorization".into())
                .or_insert_with(|| format!("Bearer {api_key}"));
        }
        if let Some(header) = &self.session_header {
            turn.request_headers
                .insert(header.clone(), turn.request_correlation_id.clone());
        }
        let uuid = turn.uuid;
        self.capture.begin(&turn);
        let tee = DualObserver {
            runtime: observer,
            capture: self.capture.observer.as_ref(),
        };
        let result = self
            .transport
            .dispatch_turn(turn, &tee, on_first_token)
            .await;
        match &result {
            Ok(outcome) => self.capture.observer.record_response(
                uuid,
                NativeResponseMetadata {
                    start_ns: Some(outcome.start_ns),
                    end_ns: Some(outcome.end_ns),
                    prompt_tokens: outcome.prompt_tokens,
                    completion_tokens: outcome.completion_tokens,
                    http: outcome.http,
                },
            ),
            Err(_) => {
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
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
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
}
