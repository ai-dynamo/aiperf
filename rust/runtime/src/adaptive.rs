// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Application-layer adaptive configuration and runtime construction.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;
use std::sync::Arc;

use crate::adaptive_core::{
    AdaptiveObserver, AdaptiveScale, AdaptiveScaleOptions, ControlActuator, CorrelationContext,
    DefaultSlaEvaluator, FileArtifactSink, FixedPercentStep, PrefillConcurrencyActuator,
    RampUntilFailController, RampUntilFailOptions, RequestRateActuator, SessionConcurrencyActuator,
    SharedArtifactSink, SharedWindowSampler, SlaFilter, SlaMarginStep, SlaOp, SlaStat, StepPolicy,
    TumblingWindowSampler, UserTarget, UsersActuator,
};
use crate::clock::Clock;
use crate::extensions::TransactionalRegistry;
use crate::timing::{IntervalGenerator, SlotPool};
use anyhow::{Context, Result, anyhow, bail};
use loadgen_core::sink::RequestObserver;

/// Adaptive control variable selected by the CLI/config layer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AdaptiveControlVariable {
    /// Concurrent single-turn sessions.
    Concurrency,
    /// Requests awaiting first token.
    PrefillConcurrency,
    /// Mean request issue rate.
    RequestRate,
    /// Active target for a user-centric workload.
    Users,
}

impl FromStr for AdaptiveControlVariable {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "concurrency" => Ok(Self::Concurrency),
            "prefill_concurrency" | "prefill-concurrency" => Ok(Self::PrefillConcurrency),
            "request_rate" | "request-rate" => Ok(Self::RequestRate),
            "users" => Ok(Self::Users),
            other => bail!(
                "unknown adaptive control variable {other:?} (expected concurrency|prefill_concurrency|request_rate|users)"
            ),
        }
    }
}

/// Adaptive step-policy configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AdaptiveStepConfig {
    /// Tightest-SLA-margin-scaled increments.
    SlaMargin {
        /// Minimum increment.
        base_step: usize,
        /// Maximum base-step multiplier.
        max_step_multiplier: usize,
    },
    /// A fixed percentage of the current value.
    FixedPercent {
        /// Positive percentage.
        percent: f64,
    },
}

/// Fully lowered adaptive settings for one online phase.
#[derive(Clone, Debug)]
pub struct AdaptiveRunConfig {
    /// Controlled load variable.
    pub control_variable: AdaptiveControlVariable,
    /// Inclusive minimum control value.
    pub minimum: f64,
    /// Inclusive maximum control value.
    pub maximum: f64,
    /// Assessment window period in nanoseconds.
    pub assessment_period_ns: i64,
    /// Sustain hold duration in nanoseconds.
    pub sustain_duration_ns: i64,
    /// Minimum successful completions for a conclusive window.
    pub min_completed_requests: usize,
    /// Conjunctive SLA filters.
    pub sla_filters: Vec<SlaFilter>,
    /// Ramp step policy.
    pub step: AdaptiveStepConfig,
    /// Directory receiving schema-v2 event and summary artifacts.
    pub artifact_dir: PathBuf,
    /// Run/phase correlation fields.
    pub correlation: CorrelationContext,
}

/// Resources shared by the issuer and adaptive assessment task.
pub struct BuiltAdaptive {
    /// Clock-paced controller runtime.
    pub scale: Rc<AdaptiveScale>,
    /// Observer tee feeding both aggregate metrics and adaptive windows.
    pub observer: Rc<dyn RequestObserver>,
}

/// Injected resources available to an [`ActuatorFactory`] at construction time.
///
/// A factory consumes the context by value because it builds exactly one
/// actuator and moves the resources it owns (the interval generator, a slot
/// pool, or the user-target hook) into that actuator. The registry resolves and
/// invokes exactly one factory per adaptive phase, so single-use ownership is
/// sufficient and avoids cloning the injected seams.
pub struct ActuatorBuildContext {
    /// Inclusive minimum control value from the lowered config.
    pub minimum: f64,
    /// Inclusive maximum control value from the lowered config.
    pub maximum: f64,
    /// Run clock injected into time-stamping actuators.
    pub clock: Rc<dyn Clock>,
    /// Interval generator paced by the request-rate actuator.
    pub intervals: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    /// Session-concurrency slot pool, when the workload exposes one.
    pub session_slots: Option<Rc<SlotPool>>,
    /// Prefill-concurrency slot pool, when the workload exposes one.
    pub prefill_slots: Option<Rc<SlotPool>>,
    /// User-centric workload target hook, when the workload exposes one.
    pub user_target: Option<Rc<dyn UserTarget>>,
}

/// Stateless factory registered by control-variable identifier to construct one
/// adaptive [`ControlActuator`].
pub trait ActuatorFactory {
    /// Stable control-variable id keying this factory in the registry.
    ///
    /// The returned id is authoritative: it must equal the built actuator's
    /// [`ControlActuator::variable`], since the controller cross-checks the two.
    fn id(&self) -> &'static str;

    /// Construct the actuator, consuming the resources it owns from `context`.
    fn build(&self, context: ActuatorBuildContext) -> Result<Rc<dyn ControlActuator>>;
}

/// Session-concurrency control-variable factory.
struct ConcurrencyActuatorFactory;

impl ActuatorFactory for ConcurrencyActuatorFactory {
    fn id(&self) -> &'static str {
        "concurrency"
    }

    fn build(&self, context: ActuatorBuildContext) -> Result<Rc<dyn ControlActuator>> {
        let pool = context
            .session_slots
            .ok_or_else(|| anyhow!("adaptive concurrency requires a session slot pool"))?;
        Ok(Rc::new(SessionConcurrencyActuator::new(
            pool,
            integer_bound(context.minimum, "adaptive concurrency minimum")?,
            integer_bound(context.maximum, "adaptive concurrency maximum")?,
        )?))
    }
}

/// Prefill-concurrency control-variable factory.
struct PrefillConcurrencyActuatorFactory;

impl ActuatorFactory for PrefillConcurrencyActuatorFactory {
    fn id(&self) -> &'static str {
        "prefill_concurrency"
    }

    fn build(&self, context: ActuatorBuildContext) -> Result<Rc<dyn ControlActuator>> {
        let pool = context
            .prefill_slots
            .ok_or_else(|| anyhow!("adaptive prefill_concurrency requires a prefill slot pool"))?;
        let minimum = integer_bound(context.minimum, "adaptive prefill minimum")?;
        let maximum = integer_bound(context.maximum, "adaptive prefill maximum")?;
        if let Some(session) = &context.session_slots
            && maximum > session.current_limit()
        {
            bail!("adaptive prefill_concurrency maximum must be <= concurrency");
        }
        Ok(Rc::new(PrefillConcurrencyActuator::new(
            pool, minimum, maximum,
        )?))
    }
}

/// Request-rate control-variable factory.
struct RequestRateActuatorFactory;

impl ActuatorFactory for RequestRateActuatorFactory {
    fn id(&self) -> &'static str {
        "request_rate"
    }

    fn build(&self, context: ActuatorBuildContext) -> Result<Rc<dyn ControlActuator>> {
        Ok(Rc::new(RequestRateActuator::new(
            context.intervals,
            context.minimum,
            context.maximum,
        )?))
    }
}

/// Target-users control-variable factory.
struct UsersActuatorFactory;

impl ActuatorFactory for UsersActuatorFactory {
    fn id(&self) -> &'static str {
        "users"
    }

    fn build(&self, context: ActuatorBuildContext) -> Result<Rc<dyn ControlActuator>> {
        let target = context.user_target.ok_or_else(|| {
            anyhow!("adaptive users requires a user-centric workload target hook")
        })?;
        Ok(Rc::new(UsersActuator::new(
            target,
            context.clock.clone(),
            integer_bound(context.minimum, "adaptive users minimum")?,
            integer_bound(context.maximum, "adaptive users maximum")?,
        )?))
    }
}

/// Id-keyed registry of adaptive control-variable actuator factories.
///
/// The shared [`TransactionalRegistry`] rejects duplicate identifiers and commits
/// batch registration atomically.
#[derive(Clone)]
pub struct ActuatorRegistry {
    // Factories are stateless and thread-safe because AIPerfRegistry is Send +
    // Sync; built actuators remain worker-local Rc values.
    factories: TransactionalRegistry<Arc<dyn ActuatorFactory + Send + Sync>>,
}

impl Default for ActuatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ActuatorRegistry {
    pub fn new() -> Self {
        Self {
            factories: TransactionalRegistry::new(),
        }
    }

    /// Populate the four native control-variable actuator factories into this
    /// registry.
    ///
    /// This is the single source of truth for the built-in actuator set; both the
    /// [`Self::with_builtin_actuators`] convenience constructor and the
    /// `BuiltinActuatorsExtension` that folds these factories into the unified
    /// [`AIPerfRegistry`](crate::extensions::AIPerfRegistry) delegate here.
    pub fn register_builtins(&mut self) -> Result<()> {
        self.register(ConcurrencyActuatorFactory)?;
        self.register(PrefillConcurrencyActuatorFactory)?;
        self.register(RequestRateActuatorFactory)?;
        self.register(UsersActuatorFactory)?;
        Ok(())
    }

    pub fn with_builtin_actuators() -> Result<Self> {
        let mut registry = Self::new();
        registry.register_builtins()?;
        Ok(registry)
    }

    /// Register one factory, rejecting a duplicate control-variable id.
    pub fn register(
        &mut self,
        factory: impl ActuatorFactory + Send + Sync + 'static,
    ) -> Result<()> {
        let id = factory.id();
        self.factories
            .insert(id, Arc::new(factory))
            .map_err(|error| anyhow!("{error}"))
    }

    /// Build the actuator registered under `id`, consuming the build context.
    pub fn build(
        &self,
        id: &str,
        context: ActuatorBuildContext,
    ) -> Result<Rc<dyn ControlActuator>> {
        let factory = self.factories.get(id).ok_or_else(|| {
            anyhow!(
                "unknown adaptive control variable {id:?}; registered variables: {}",
                self.factories
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })?;
        factory.build(context)
    }
}

/// Parse `metric:stat:op:threshold` into a validated SLA filter.
pub fn parse_sla_filter(value: &str) -> Result<SlaFilter> {
    let parts: Vec<&str> = value.split(':').collect();
    if parts.len() != 4 {
        bail!("adaptive SLA must be metric:stat:op:threshold, got {value:?}");
    }
    let stat = parts[1].parse::<SlaStat>().map_err(anyhow::Error::new)?;
    let op = parts[2].parse::<SlaOp>().map_err(anyhow::Error::new)?;
    let threshold = parts[3]
        .parse::<f64>()
        .with_context(|| format!("invalid adaptive SLA threshold in {value:?}"))?;
    SlaFilter::new(parts[0], stat, op, threshold).map_err(anyhow::Error::new)
}

/// Convert positive seconds to a bounded integer-nanosecond duration.
pub fn positive_seconds_to_ns(value: f64, flag: &str) -> Result<i64> {
    if !value.is_finite() || value <= 0.0 {
        bail!("{flag} must be positive and finite, got {value}");
    }
    let nanoseconds = value * 1_000_000_000.0;
    if nanoseconds >= i64::MAX as f64 {
        bail!("{flag} is outside the i64 nanosecond range");
    }
    let nanoseconds = nanoseconds.round_ties_even() as i64;
    if nanoseconds == 0 {
        bail!("{flag} is below one nanosecond after rounding");
    }
    Ok(nanoseconds)
}

/// Construct the adaptive runtime and observer over already-created issuer
/// actuators. Exactly one of the supplied pools/generator/user hook is selected
/// by `config.control_variable`; the controller itself never branches on it.
#[allow(clippy::too_many_arguments)]
pub fn build_adaptive(
    config: AdaptiveRunConfig,
    clock: Rc<dyn Clock>,
    start_ns: i64,
    delegate: Rc<dyn RequestObserver>,
    intervals: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    user_target: Option<Rc<dyn UserTarget>>,
) -> Result<BuiltAdaptive> {
    build_adaptive_with_origins(
        config,
        clock,
        start_ns,
        start_ns,
        delegate,
        intervals,
        session_slots,
        prefill_slots,
        user_target,
    )
}

/// Construct adaptive policy when transport observations and phase-local
/// windows have distinct origins.
///
/// A phased run keeps all transport timestamps on one benchmark timeline, but
/// each adaptive sampler begins at its own phase boundary. The distinct origins
/// do not change the observer wire contract.
#[allow(clippy::too_many_arguments)]
pub fn build_adaptive_with_origins(
    config: AdaptiveRunConfig,
    clock: Rc<dyn Clock>,
    observer_origin_ns: i64,
    window_start_ns: i64,
    delegate: Rc<dyn RequestObserver>,
    intervals: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    session_slots: Option<Rc<SlotPool>>,
    prefill_slots: Option<Rc<SlotPool>>,
    user_target: Option<Rc<dyn UserTarget>>,
) -> Result<BuiltAdaptive> {
    let actuator = ActuatorRegistry::with_builtin_actuators()?.build(
        control_variable_name(config.control_variable),
        ActuatorBuildContext {
            minimum: config.minimum,
            maximum: config.maximum,
            clock: clock.clone(),
            intervals,
            session_slots,
            prefill_slots,
            user_target,
        },
    )?;
    let sampler: SharedWindowSampler = Rc::new(RefCell::new(Box::new(TumblingWindowSampler::new(
        window_start_ns,
    ))));
    let scale = build_adaptive_scale(config, clock.clone(), actuator, sampler.clone())?;
    let observer: Rc<dyn RequestObserver> = Rc::new(AdaptiveObserver::new(
        delegate,
        sampler,
        clock,
        observer_origin_ns,
    ));
    Ok(BuiltAdaptive { scale, observer })
}

/// Build the controller runtime over an explicitly injected actuator and
/// terminal-record sampler.
///
/// Callers may inject placement-backed actuators and feed the sampler from
/// completed native records without changing the controller or artifact contract.
pub fn build_adaptive_scale(
    config: AdaptiveRunConfig,
    clock: Rc<dyn Clock>,
    actuator: Rc<dyn ControlActuator>,
    sampler: SharedWindowSampler,
) -> Result<Rc<AdaptiveScale>> {
    if actuator.variable() != control_variable_name(config.control_variable) {
        bail!(
            "adaptive actuator variable {:?} does not match configured {:?}",
            actuator.variable(),
            control_variable_name(config.control_variable)
        );
    }
    let step: Box<dyn StepPolicy> = match config.step {
        AdaptiveStepConfig::SlaMargin {
            base_step,
            max_step_multiplier,
        } => Box::new(SlaMarginStep::new(base_step, max_step_multiplier)?),
        AdaptiveStepConfig::FixedPercent { percent } => Box::new(FixedPercentStep::new(percent)?),
    };
    let controller = RampUntilFailController::new(
        actuator,
        Box::new(DefaultSlaEvaluator),
        step,
        config.sla_filters,
        RampUntilFailOptions {
            min_completed_requests: config.min_completed_requests,
            sustain_duration_ns: config.sustain_duration_ns,
        },
    )?;
    let artifacts: SharedArtifactSink = Rc::new(RefCell::new(Box::new(FileArtifactSink::new(
        &config.artifact_dir,
    )?)));
    let scale = Rc::new(AdaptiveScale::new(
        clock.clone(),
        sampler.clone(),
        Box::new(controller),
        artifacts,
        AdaptiveScaleOptions {
            assessment_period_ns: config.assessment_period_ns,
            sustain_duration_ns: config.sustain_duration_ns,
            correlation: config.correlation,
        },
    )?);
    Ok(scale)
}

fn control_variable_name(variable: AdaptiveControlVariable) -> &'static str {
    match variable {
        AdaptiveControlVariable::Concurrency => "concurrency",
        AdaptiveControlVariable::PrefillConcurrency => "prefill_concurrency",
        AdaptiveControlVariable::RequestRate => "request_rate",
        AdaptiveControlVariable::Users => "users",
    }
}

fn integer_bound(value: f64, label: &str) -> Result<usize> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        bail!("{label} must be an integer >= 1, got {value}");
    }
    if value >= usize::MAX as f64 {
        bail!("{label} is outside the usize range");
    }
    Ok(value as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_core::ControlSnapshot;
    use crate::clock::SimClock;
    use crate::timing::intervals::Constant;

    struct NoopUserTarget;

    impl UserTarget for NoopUserTarget {
        fn set_target_users(
            &self,
            _value: usize,
            _now_ns: i64,
        ) -> std::result::Result<(), crate::adaptive_core::AdaptiveError> {
            Ok(())
        }

        fn user_control_snapshot(&self) -> ControlSnapshot {
            ControlSnapshot {
                target_value: 1.0,
                actual_value: 1.0,
                active_users: None,
                retiring_users: None,
                cancelled: None,
            }
        }
    }

    fn build_context() -> ActuatorBuildContext {
        ActuatorBuildContext {
            minimum: 1.0,
            maximum: 4.0,
            clock: Rc::new(SimClock::new()),
            intervals: Rc::new(RefCell::new(Box::new(Constant::new(2.0)))),
            session_slots: Some(Rc::new(SlotPool::new(4))),
            prefill_slots: Some(Rc::new(SlotPool::new(4))),
            user_target: Some(Rc::new(NoopUserTarget)),
        }
    }

    #[test]
    fn registry_resolves_each_control_variable_to_its_actuator() {
        let registry = ActuatorRegistry::with_builtin_actuators().unwrap();
        for variable in [
            AdaptiveControlVariable::Concurrency,
            AdaptiveControlVariable::PrefillConcurrency,
            AdaptiveControlVariable::RequestRate,
            AdaptiveControlVariable::Users,
        ] {
            let id = control_variable_name(variable);
            let actuator = registry
                .build(id, build_context())
                .unwrap_or_else(|error| panic!("building {id:?} actuator failed: {error}"));
            // Factory lookup and the built actuator must expose the same identifier.
            assert_eq!(actuator.variable(), id);
        }
    }

    #[test]
    fn registry_rejects_an_unknown_control_variable() {
        let registry = ActuatorRegistry::with_builtin_actuators().unwrap();
        assert!(registry.build("nonexistent", build_context()).is_err());
    }

    #[test]
    fn parses_compact_sla_filter() {
        let filter = parse_sla_filter("request_latency:p95:le:100").unwrap();
        assert_eq!(filter.metric_tag, "request_latency");
        assert_eq!(filter.stat, SlaStat::P95);
        assert_eq!(filter.op, SlaOp::Le);
        assert_eq!(filter.threshold, 100.0);
    }

    #[test]
    fn compact_sla_rejects_bad_shape_and_non_finite_threshold() {
        assert!(parse_sla_filter("bad").is_err());
        assert!(parse_sla_filter("ttft:p95:le:inf").is_err());
    }

    #[test]
    fn control_variable_aliases_are_accepted() {
        assert_eq!(
            "prefill-concurrency"
                .parse::<AdaptiveControlVariable>()
                .unwrap(),
            AdaptiveControlVariable::PrefillConcurrency
        );
        assert_eq!(
            "request_rate".parse::<AdaptiveControlVariable>().unwrap(),
            AdaptiveControlVariable::RequestRate
        );
    }

    #[test]
    fn positive_seconds_must_survive_nanosecond_rounding() {
        assert_eq!(
            positive_seconds_to_ns(1.5, "duration").unwrap(),
            1_500_000_000
        );
        assert!(positive_seconds_to_ns(0.1e-9, "duration").is_err());
    }
}
