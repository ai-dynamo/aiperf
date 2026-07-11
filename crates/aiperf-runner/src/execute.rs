// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native construction and execution of one resolved benchmark run.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::path::{Component, Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use aiperf::ancillary::RATE_RAMP_UPDATE_INTERVAL_NS;
use aiperf::fixed_schedule::{
    DatasetFixedScheduleSource, FixedScheduleConfig, FixedScheduleWorkload,
};
use aiperf::http::TransportSink;
use aiperf::metrics::{NativeMetricsObserver, NativeResponseMetadata, RequestMetricMetadata};
use aiperf::multiturn::{
    ConversationSource, IssuedCredit, NativeDatasetConversationSource, TurnToSend,
};
use aiperf::phase_runtime::{
    RampScheduledPhaseController, ScheduledPhaseController, ScheduledPhasePlan,
    ScheduledPhaseResources, SlotPoolPhaseResources, run_scheduled_phases,
};
use aiperf::report::write_native_report_json;
use aiperf::request_rate::RequestRateWorkload;
use aiperf::scheduled::{
    ScheduledAncillaryPolicies, TurnDispatchOutcome, TurnDispatcher, TurnRecordProcessor, Workload,
};
use aiperf::user_centric::{UserCentricConfig, UserCentricWorkload};
use aiperf_clock::{Clock, RealClock};
use aiperf_dataset::{
    ComposeConfig, Dataset, DatasetSource, HuggingFaceTokenizer, LoadConfig, ModelId,
    ModelSelector, ModelSelectorFactory, RandomModelSelectorFactory,
    RoundRobinModelSelectorFactory, SyntheticDatasetConfig, SyntheticPromptConfig, TextTokenizer,
    TiktokenEncoding, TiktokenTokenizer,
};
use aiperf_endpoints::EndpointConfig;
use aiperf_extensions::AiperfRegistry;
use aiperf_metrics::{
    CATALOG, ExportContext, MetricsAccumulator, MetricsConfig, NativeReport, Phase as MetricsPhase,
    ReportRunInfo, ReportSummary, RunOutcome, SloThreshold,
};
use aiperf_rng::{EmpiricalPoint, PeakEntry, RandomGenerator, RngRoot, SamplingDistribution};
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
    DatasetSpec, DistributionSpec, EndpointSpec, FileDatasetSpec, MetricsSpec,
    ModelSelectionStrategy, ModelsSpec, PhaseSpec, RampSpec, RampStrategySpec, RunRequest,
    RunTerminal, SyntheticDatasetSpec,
};
use crate::records::{CapturedRecord, write_records_jsonl};

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

async fn execute_native(request: RunRequest) -> Result<NativeReport> {
    let registry = AiperfRegistry::builtin()?;
    let rng_root = RngRoot::new(request.run.random_seed);
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
    let dataset = build_dataset(
        &registry,
        &request.run.dataset,
        &request.run.models,
        rng_root,
        tokenizer.as_ref(),
    )
    .await?;
    let endpoint = endpoint_config(&request.run.endpoint)?;
    let default_output_tokens = default_output_tokens(&request.run.dataset)?;

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
        .any(|phase| phase.request_arrival().is_some() && phase.concurrency().is_some())
        .then(|| Rc::new(SlotPool::new(1)));
    let shared_prefill = request
        .run
        .phases
        .iter()
        .any(|phase| {
            phase.request_arrival().is_some() && phase.common().prefill_concurrency.is_some()
        })
        .then(|| Rc::new(SlotPool::new(1)));
    let request_resources: Rc<dyn ScheduledPhaseResources> = Rc::new(SlotPoolPhaseResources::new(
        shared_session.clone(),
        shared_prefill.clone(),
    ));

    let mut plans = Vec::with_capacity(request.run.phases.len());
    for (phase_index, phase) in request.run.phases.iter().enumerate() {
        let phase_rng =
            RngRoot::new(rng_root.derive_seed(&format!("runner.phase.{phase_index}.dataset")));
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
            matches!(phase, PhaseSpec::FixedSchedule { .. }),
        )?;
        let arrival_seed = rng_root
            .derive_seed(&format!("runner.phase.{phase_index}.arrival"))
            .unwrap_or(phase_index as u64);
        let (workload, intervals, phase_session, phase_prefill, enforce_stop, resources): (
            Rc<dyn Workload>,
            Rc<RefCell<Box<dyn aiperf_timing::IntervalGenerator>>>,
            Option<Rc<SlotPool>>,
            Option<Rc<SlotPool>>,
            bool,
            Rc<dyn ScheduledPhaseResources>,
        ) = match phase {
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
                let concrete = Rc::new(UserCentricWorkload::new(
                    UserCentricConfig {
                        num_users: *users,
                        request_rate: *rate,
                        concurrency: *concurrency,
                    },
                    source,
                )?);
                let phase_session = concrete.session_slots();
                let resources: Rc<dyn ScheduledPhaseResources> =
                    Rc::new(SlotPoolPhaseResources::new(phase_session.clone(), None));
                let intervals = Rc::new(RefCell::new(make_interval_generator(
                    aiperf_timing::ArrivalPattern::ConcurrencyBurst,
                    None,
                    None,
                    arrival_seed,
                )));
                (concrete, intervals, phase_session, None, true, resources)
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
            intervals,
            phase_session,
            phase_prefill,
            RngRoot::new(rng_root.derive_seed(&format!("runner.phase.{phase_index}.ramp"))),
        )?;
        let record_processor: Rc<dyn TurnRecordProcessor> = Rc::new(CapturePhaseProcessor {
            capture: capture.clone(),
            phase: metrics_phase(phase)?,
            has_credit_timestamp: !matches!(phase, PhaseSpec::FixedSchedule { .. }),
        });
        plans.push(
            ScheduledPhasePlan::new(phase_config, workload, policies)
                .with_enforce_stop(enforce_stop)
                .with_start_ns(start_ns)
                .with_resources(resources)
                .with_record_processors(vec![record_processor])
                .with_controller(controller),
        );
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
    let outcome = RunOutcome {
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
    Ok(NativeReport::from_outcome(&profiling_metrics, &outcome))
}

#[allow(clippy::too_many_arguments)]
fn native_conversation_source(
    dataset: Dataset,
    model: String,
    default_output_tokens: usize,
    rng_root: RngRoot,
    endpoint: EndpointConfig,
    registry: &AiperfRegistry,
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
    Ok(Box::new(source))
}

async fn build_dataset(
    registry: &AiperfRegistry,
    dataset: &DatasetSpec,
    models: &ModelsSpec,
    rng_root: RngRoot,
    tokenizer: &dyn TextTokenizer,
) -> Result<Dataset> {
    match dataset {
        DatasetSpec::Synthetic(spec) => {
            let mut compose = compose_config(models, rng_root)?;
            compose.output_length_distribution = Some(distribution(&spec.prompts.osl)?);
            compose.synthetic_config = Some(synthetic_config(spec)?);
            let load = LoadConfig::new(DatasetSource::Inline(
                serde_json::json!({"__aiperf_synthetic": true}),
            ));
            registry
                .dataset_formats()
                .build_dataset(Some("synthetic"), &load, &compose, tokenizer)
                .await
                .map_err(Into::into)
        }
        DatasetSpec::File(spec) => {
            build_file_dataset(registry, spec, models, rng_root, tokenizer).await
        }
    }
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
    let source = match (&spec.path, &spec.records) {
        (Some(path), None) => DatasetSource::Path(path.clone()),
        (None, Some(records)) => DatasetSource::Inline(records.clone()),
        _ => unreachable!("source exclusivity validated above"),
    };
    let mut load = LoadConfig::new(source);
    load.max_rows = spec.entries;
    load.sampling_strategy = Some(spec.sampling.clone());
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
    ensure!(
        spec.prompts.batch_size > 0,
        "synthetic prompt batch_size must be positive"
    );
    ensure!(
        spec.turn_delay_ratio.is_finite() && spec.turn_delay_ratio >= 0.0,
        "synthetic turn_delay_ratio must be finite and non-negative"
    );
    Ok(SyntheticDatasetConfig {
        entries: spec.entries,
        turns: distribution(&spec.turns)?,
        turn_delay_ms: distribution(&spec.turn_delay_ms)?,
        turn_delay_ratio: spec.turn_delay_ratio,
        prompts: Some(SyntheticPromptConfig {
            input_tokens: distribution(&spec.prompts.isl)?,
            batch_size: spec.prompts.batch_size,
        }),
        ..SyntheticDatasetConfig::default()
    })
}

fn default_output_tokens(dataset: &DatasetSpec) -> Result<usize> {
    let expected = match dataset {
        DatasetSpec::Synthetic(spec) => distribution(&spec.prompts.osl)?.expected_value().ceil(),
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
    };
    ensure!(
        expected.is_finite() && expected > 0.0 && expected <= usize::MAX as f64,
        "synthetic OSL expected value is outside the native usize range"
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

fn load_tokenizer(spec: Option<&str>) -> Result<Box<dyn TextTokenizer>> {
    let spec = spec.unwrap_or("builtin");
    let path = Path::new(spec);
    if path.is_dir() {
        return Ok(Box::new(HuggingFaceTokenizer::from_directory(path)?));
    }
    if path.is_file() {
        return Ok(Box::new(HuggingFaceTokenizer::from_file(path)?));
    }
    let encoding = spec.parse::<TiktokenEncoding>()?;
    Ok(Box::new(TiktokenTokenizer::new(encoding)))
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
