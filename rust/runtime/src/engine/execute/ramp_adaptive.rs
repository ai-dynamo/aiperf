// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Ramp drivers, adaptive control, model selection, and tokenizer helpers.

use super::*;
use crate::rng::{ConfiguredRandomGenerator, RuntimeRandomGenerator};

/// Phase-local roots for independently randomized ramp actuators.
///
/// The controller derives this layer before constructing a strategy. A
/// stochastic strategy such as `PoissonRamp` then derives its curve-local
/// `timing.ramp.poisson` stream, producing the stable hierarchy
/// `phase -> actuator -> curve` without coupling simultaneous actuators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RampActuatorRngRoots {
    pub(crate) concurrency: RngRoot,
    pub(crate) prefill_concurrency: RngRoot,
    pub(crate) request_rate: RngRoot,
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

pub(crate) fn ramp_controller(
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
            clock.clone(),
            intervals.clone(),
            target_rate,
            rng_roots.request_rate(),
        )?;
    }
    let rate_series_controller: Option<Rc<dyn ScheduledPhaseController>> =
        if let Some(series) = common.rate_series.as_ref() {
            let start_delay_ns = common
                .rate_ramp
                .as_ref()
                .map(|r| seconds_to_u64_ns(r.duration))
                .transpose()?
                .unwrap_or(0);
            Some(Rc::new(
                crate::timing::rate_series::RateSeriesScheduledPhaseController::new(
                    crate::timing::rate_series::RateSeriesDriver::new(
                        clock.clone(),
                        crate::timing::rate_series::RateSeriesSchedule::from_points(
                            series.points.iter().map(|p| (p.time_s, p.qps)),
                        ),
                        intervals,
                        start_delay_ns,
                    ),
                ),
            ) as Rc<dyn ScheduledPhaseController>)
        } else {
            None
        };
    match (drivers.is_empty(), rate_series_controller) {
        (true, None) => Ok(Rc::new(crate::phase_runtime::NoopScheduledPhaseController)),
        (false, None) => Ok(Rc::new(RampScheduledPhaseController::new(drivers))),
        (true, Some(series)) => Ok(series),
        (false, Some(series)) => Ok(Rc::new(CombinedScheduledPhaseController {
            ramp: RampScheduledPhaseController::new(drivers),
            series,
        })),
    }
}

struct CombinedScheduledPhaseController {
    ramp: RampScheduledPhaseController,
    series: Rc<dyn ScheduledPhaseController>,
}

impl ScheduledPhaseController for CombinedScheduledPhaseController {
    fn start(&self) -> Result<()> {
        self.ramp.start()?;
        self.series.start()
    }

    fn stop(&self) -> crate::timing::LocalPhaseFuture<Result<()>> {
        let series = self.series.clone();
        let ramp_stop = self.ramp.stop();
        Box::pin(async move {
            ramp_stop.await?;
            series.stop().await
        })
    }

    fn wait_until_stop(&self) -> crate::timing::LocalPhaseFuture<()> {
        let series = self.series.clone();
        let ramp_wait = self.ramp.wait_until_stop();
        Box::pin(async move {
            ramp_wait.await;
            series.wait_until_stop().await;
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adaptive_runtime_extension(
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
                phase.common().rate_ramp.is_none() && phase.common().rate_series.is_none(),
                "adaptive request_rate cannot be combined with rate_ramp or rate_series"
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
        !phase.common().exclude_from_results,
        "adaptive_scale is supported only on profiling phases"
    );
    ensure!(
        phase.common().duration.is_some(),
        "adaptive_scale requires a phase duration"
    );
    ensure!(
        !matches!(
            phase,
            PhaseSpec::FixedSchedule { .. } | PhaseSpec::AgenticReplay { .. }
        ),
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

pub(crate) struct AdaptiveRuntimeExtension {
    pub(crate) config: AdaptiveRunConfig,
    pub(crate) intervals: Rc<RefCell<Box<dyn crate::timing::IntervalGenerator>>>,
    pub(crate) session_slots: Option<Rc<SlotPool>>,
    pub(crate) prefill_slots: Option<Rc<SlotPool>>,
    pub(crate) user_target: Option<Rc<dyn UserTarget>>,
    pub(crate) session_target: Option<usize>,
    pub(crate) prefill_target: Option<usize>,
    /// Worker-record source feeding the sampler on the online path. `None` for
    /// dispatchers that feed the callback observer directly (offline), which
    /// keeps the sampler from being double-fed.
    pub(crate) record_source: Option<Rc<dyn AdaptiveTerminalRecordSource>>,
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
    pub(crate) scale: Rc<AdaptiveScale>,
    pub(crate) delegate: Rc<dyn ScheduledPhaseController>,
    pub(crate) assessment: RefCell<Option<tokio::task::JoinHandle<()>>>,
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

pub(crate) fn model_selector(
    models: &ModelsSpec,
    rng_root: RngRoot,
) -> Result<Arc<dyn ModelSelectorFactory>> {
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

pub(crate) struct WeightedModelSelectorFactory {
    pub(crate) weights: Vec<f64>,
    pub(crate) rng_root: RngRoot,
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
            rng: self
                .rng_root
                .derive_generator("runner.model.weighted_selection"),
        }))
    }
}

pub(crate) struct WeightedModelSelector {
    pub(crate) models: Vec<ModelId>,
    pub(crate) weights: Vec<f64>,
    pub(crate) rng: ConfiguredRandomGenerator,
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
        // Resolution order mirrors vLLM's Rust frontend: the HuggingFace fast
        // tokenizer (`tokenizer.json`) first, then a native `tiktoken.model` /
        // `tokenizer.model` / `*.tiktoken` BPE vocab for Kimi/Qwen/DeepSeek-class
        // repositories that ship no `tokenizer.json`. Only when neither is
        // present do we fall through to the actionable error at the resolver.
        if path.join("tokenizer.json").is_file() {
            return Ok(Arc::new(HuggingFaceTokenizer::from_directory(path)?));
        }
        if find_tiktoken_model_file(path).is_some() {
            return Ok(Arc::new(NativeTiktokenTokenizer::from_directory(path)?));
        }
        return Ok(Arc::new(HuggingFaceTokenizer::from_directory(path)?));
    }
    if path.is_file() {
        return Ok(Arc::new(HuggingFaceTokenizer::from_file(path)?));
    }
    let encoding = spec.parse::<TiktokenEncoding>()?;
    Ok(Arc::new(TiktokenTokenizer::new(encoding)))
}

/// Build the tokenizer selected by a protocol-v2 [`TokenizerSpec`].
///
/// A populated `server_url` selects the [`ServerTokenizer`], which offloads
/// tokenization to the inference server; the spec `name` then carries only the
/// model selector forwarded to that server. Otherwise the local built-in /
/// Hugging Face resolution in [`load_tokenizer`] applies.
pub(crate) fn build_tokenizer(
    spec: &crate::engine::protocol::TokenizerSpec,
) -> Result<Arc<dyn TextTokenizer>> {
    if let Some(server_url) = spec.server_url.as_deref() {
        let model = (spec.name != "builtin").then(|| spec.name.clone());
        return Ok(Arc::new(ServerTokenizer::new(server_url, model)?));
    }
    load_tokenizer(Some(&spec.name))
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
pub(crate) fn select_input_token_counter(
    tokenizer: Arc<dyn TextTokenizer>,
    apply_chat_template: bool,
) -> Arc<dyn InputTokenCounter> {
    if apply_chat_template {
        Arc::new(EndpointInputTokenCounter::new(tokenizer, true))
    } else {
        Arc::new(AuthoredInputTokenCounter)
    }
}

pub(crate) fn seconds_to_ns(value: f64) -> Result<i64> {
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

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn load_tokenizer_loads_tiktoken_model_dir_natively() {
        // A resolved directory with a `tiktoken.model` (Kimi/Qwen-class) and no
        // `tokenizer.json` must load through the native tiktoken loader, not fail.
        use base64::Engine as _;
        let dir = tempfile::tempdir().unwrap();
        let engine = base64::engine::general_purpose::STANDARD;
        let mut model = String::new();
        for byte in 0u8..=255 {
            model.push_str(&format!("{} {}\n", engine.encode([byte]), byte as u32));
        }
        std::fs::write(dir.path().join("tiktoken.model"), model).unwrap();

        let tokenizer = load_tokenizer(dir.path().to_str()).expect("native tiktoken load");
        let text = "hello world";
        assert_eq!(
            tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(),
            text
        );
        // Deterministic, network-free token count.
        assert_eq!(tokenizer.count("hi").unwrap(), 2);
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
}
