// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Clock-driven value ramping with pluggable curves.
//!
//! Discrete mode asks a
//! strategy when to apply its next step; continuous mode samples `value_at` on
//! a fixed cadence. Both modes force the exact target on natural completion.
//! Dropping or aborting the spawned task freezes the last applied value.
//!
//! Every elapsed-time read and sleep goes through an injected
//! [`crate::clock::Clock`]. The driver therefore follows the same code path on
//! real and virtual clocks and never touches `tokio::time`.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::rc::Rc;

use crate::clock::Clock;
use crate::rng::namespace::TIMING_RAMP_POISSON;
use crate::rng::{RngRoot, RustRandomGenerator};
use tokio::task::JoinHandle;

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;
const DEFAULT_STEP_SIZE: f64 = 1.0;
const DEFAULT_EXPONENT: f64 = 2.0;

/// Invalid ramp configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RampConfigError {
    /// A floating-point field was non-finite or outside its valid range.
    InvalidFloat {
        /// Configuration field name.
        field: &'static str,
        /// Rejected value.
        value: f64,
        /// Required range or property.
        requirement: &'static str,
    },
    /// A nanosecond duration was zero or exceeded the clock's signed range.
    InvalidDurationNs {
        /// Configuration field name.
        field: &'static str,
        /// Rejected value.
        value: u64,
    },
    /// The Poisson rate derived from range and duration was not representable.
    InvalidPoissonRate(f64),
}

impl Display for RampConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidFloat {
                field,
                value,
                requirement,
            } => write!(f, "ramp {field} must be {requirement}, got {value}"),
            Self::InvalidDurationNs { field, value } => {
                write!(f, "ramp {field} must be in 1..={}, got {value}ns", i64::MAX)
            }
            Self::InvalidPoissonRate(value) => write!(
                f,
                "Poisson ramp requires a finite positive range/duration rate, got {value}"
            ),
        }
    }
}

impl Error for RampConfigError {}

/// Immutable parameters shared by ramp strategies and the driver.
///
/// Curve selection is intentionally not an enum field: callers inject a
/// concrete [`RampStrategy`], leaving new curves open without modifying a
/// central match statement.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RamperConfig {
    start: f64,
    target: f64,
    duration_ns: u64,
    update_interval_ns: Option<u64>,
    step_size: f64,
    exponent: f64,
}

impl RamperConfig {
    /// Construct a discrete ramp using the default step size (`1`) and
    /// exponential exponent (`2`).
    pub fn new(start: f64, target: f64, duration_ns: u64) -> Result<Self, RampConfigError> {
        validate_positive_finite("start", start)?;
        validate_positive_finite("target", target)?;
        validate_duration("duration", duration_ns)?;
        let range = (target - start).abs();
        if !range.is_finite() {
            return Err(RampConfigError::InvalidFloat {
                field: "range",
                value: range,
                requirement: "finite",
            });
        }
        Ok(Self {
            start,
            target,
            duration_ns,
            update_interval_ns: None,
            step_size: DEFAULT_STEP_SIZE,
            exponent: DEFAULT_EXPONENT,
        })
    }

    /// Construct a discrete ramp from a duration expressed in seconds.
    pub fn from_seconds(
        start: f64,
        target: f64,
        duration_seconds: f64,
    ) -> Result<Self, RampConfigError> {
        Self::new(start, target, seconds_to_ns("duration", duration_seconds)?)
    }

    /// Select continuous mode with a fixed update cadence in nanoseconds.
    pub fn with_update_interval_ns(
        mut self,
        update_interval_ns: u64,
    ) -> Result<Self, RampConfigError> {
        validate_duration("update_interval", update_interval_ns)?;
        self.update_interval_ns = Some(update_interval_ns);
        Ok(self)
    }

    /// Select continuous mode with a fixed update cadence in seconds.
    pub fn with_update_interval_seconds(
        self,
        update_interval_seconds: f64,
    ) -> Result<Self, RampConfigError> {
        self.with_update_interval_ns(seconds_to_ns("update_interval", update_interval_seconds)?)
    }

    /// Override the linear discrete step size.
    pub fn with_step_size(mut self, step_size: f64) -> Result<Self, RampConfigError> {
        validate_positive_finite("step_size", step_size)?;
        self.step_size = step_size;
        Ok(self)
    }

    /// Override the exponential ease-in exponent, which must be greater than one.
    pub fn with_exponent(mut self, exponent: f64) -> Result<Self, RampConfigError> {
        if !exponent.is_finite() || exponent <= 1.0 {
            return Err(RampConfigError::InvalidFloat {
                field: "exponent",
                value: exponent,
                requirement: "finite and greater than 1",
            });
        }
        self.exponent = exponent;
        Ok(self)
    }

    /// Initial value applied before the first sleep.
    pub const fn start(&self) -> f64 {
        self.start
    }

    /// Exact value forced on natural completion.
    pub const fn target(&self) -> f64 {
        self.target
    }

    /// Total ramp duration in nanoseconds.
    pub const fn duration_ns(&self) -> u64 {
        self.duration_ns
    }

    /// Continuous update cadence, or `None` for discrete stepping.
    pub const fn update_interval_ns(&self) -> Option<u64> {
        self.update_interval_ns
    }

    /// Linear discrete step size.
    pub const fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Exponential ease-in exponent.
    pub const fn exponent(&self) -> f64 {
        self.exponent
    }
}

fn validate_positive_finite(field: &'static str, value: f64) -> Result<(), RampConfigError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(RampConfigError::InvalidFloat {
            field,
            value,
            requirement: "finite and greater than zero",
        });
    }
    Ok(())
}

fn validate_duration(field: &'static str, value: u64) -> Result<(), RampConfigError> {
    if value == 0 || value > i64::MAX as u64 {
        return Err(RampConfigError::InvalidDurationNs { field, value });
    }
    Ok(())
}

fn seconds_to_ns(field: &'static str, value: f64) -> Result<u64, RampConfigError> {
    if !value.is_finite() || value <= 0.0 || value * NANOS_PER_SECOND >= i64::MAX as f64 {
        return Err(RampConfigError::InvalidFloat {
            field,
            value,
            requirement: "finite, greater than zero, and representable in nanoseconds",
        });
    }
    let ns = (value * NANOS_PER_SECOND).round() as u64;
    validate_duration(field, ns)?;
    Ok(ns)
}

/// Object-safe curve seam used by [`RampDriver`].
pub trait RampStrategy {
    /// Initial ramp value.
    fn start(&self) -> f64;

    /// Exact terminal ramp value.
    fn target(&self) -> f64;

    /// Configured continuous sampling cadence, or `None` for discrete mode.
    fn update_interval_ns(&self) -> Option<u64>;

    /// In discrete mode, return `(delay_ns, next_value)` or `None` when done.
    fn next_step(&mut self, current: f64, elapsed_ns: u64) -> Option<(u64, f64)>;

    /// In continuous mode, return the value at `elapsed_ns`, or `None` once done.
    fn value_at(&self, elapsed_ns: u64) -> Option<f64>;
}

#[derive(Clone, Copy)]
struct CommonRamp {
    start: f64,
    target: f64,
    duration_ns: u64,
    update_interval_ns: Option<u64>,
    range: f64,
    direction: f64,
}

impl CommonRamp {
    fn new(config: RamperConfig) -> Self {
        let range = (config.target - config.start).abs();
        let direction = if range == 0.0 {
            0.0
        } else if config.target > config.start {
            1.0
        } else {
            -1.0
        };
        Self {
            start: config.start,
            target: config.target,
            duration_ns: config.duration_ns,
            update_interval_ns: config.update_interval_ns,
            range,
            direction,
        }
    }

    fn next_step(
        self,
        current: f64,
        elapsed_ns: u64,
        next_value: impl FnOnce(f64, f64, f64) -> f64,
        apply_curve: impl FnOnce(f64) -> f64,
    ) -> Option<(u64, f64)> {
        if current == self.target || self.range == 0.0 {
            return None;
        }
        if (self.direction > 0.0 && current > self.target)
            || (self.direction < 0.0 && current < self.target)
        {
            return None;
        }

        let next = next_value(current, self.direction, self.target);
        let progress = ((next - self.start).abs() / self.range).clamp(0.0, 1.0);
        let at_ns = (self.duration_ns as f64 * apply_curve(progress)).round() as u64;
        Some((at_ns.saturating_sub(elapsed_ns), next))
    }

    fn value_at(
        self,
        elapsed_ns: u64,
        time_to_value_progress: impl FnOnce(f64) -> f64,
    ) -> Option<f64> {
        if self.range == 0.0 || elapsed_ns >= self.duration_ns {
            return None;
        }
        let time_progress = (elapsed_ns as f64 / self.duration_ns as f64).clamp(0.0, 1.0);
        let value_progress = time_to_value_progress(time_progress);
        Some(self.start + self.range * self.direction * value_progress)
    }
}

/// Linear curve with configurable discrete step size.
pub struct LinearRamp {
    common: CommonRamp,
    step_size: f64,
}

impl LinearRamp {
    /// Build a linear strategy from validated configuration.
    pub fn new(config: RamperConfig) -> Self {
        Self {
            common: CommonRamp::new(config),
            step_size: config.step_size,
        }
    }
}

impl RampStrategy for LinearRamp {
    fn start(&self) -> f64 {
        self.common.start
    }

    fn target(&self) -> f64 {
        self.common.target
    }

    fn update_interval_ns(&self) -> Option<u64> {
        self.common.update_interval_ns
    }

    fn next_step(&mut self, current: f64, elapsed_ns: u64) -> Option<(u64, f64)> {
        let step_size = self.step_size;
        self.common.next_step(
            current,
            elapsed_ns,
            move |current, direction, target| {
                let next = current + step_size * direction;
                if direction > 0.0 {
                    next.min(target)
                } else {
                    next.max(target)
                }
            },
            |progress| progress,
        )
    }

    fn value_at(&self, elapsed_ns: u64) -> Option<f64> {
        self.common.value_at(elapsed_ns, |progress| progress)
    }
}

/// Exponential ease-in curve (`value_progress = time_progress^exponent`).
pub struct ExponentialRamp {
    common: CommonRamp,
    exponent: f64,
    inverse_exponent: f64,
}

impl ExponentialRamp {
    /// Build an exponential strategy from validated configuration.
    pub fn new(config: RamperConfig) -> Self {
        Self {
            common: CommonRamp::new(config),
            exponent: config.exponent,
            inverse_exponent: 1.0 / config.exponent,
        }
    }

    /// Map value progress to the time fraction used by discrete scheduling.
    pub fn value_progress_to_time(&self, progress: f64) -> f64 {
        progress.powf(self.inverse_exponent)
    }

    /// Map time progress to value progress for continuous sampling.
    pub fn time_to_value_progress(&self, progress: f64) -> f64 {
        progress.powf(self.exponent)
    }
}

impl RampStrategy for ExponentialRamp {
    fn start(&self) -> f64 {
        self.common.start
    }

    fn target(&self) -> f64 {
        self.common.target
    }

    fn update_interval_ns(&self) -> Option<u64> {
        self.common.update_interval_ns
    }

    fn next_step(&mut self, current: f64, elapsed_ns: u64) -> Option<(u64, f64)> {
        let inverse_exponent = self.inverse_exponent;
        self.common.next_step(
            current,
            elapsed_ns,
            |current, direction, _target| current + direction,
            move |progress| progress.powf(inverse_exponent),
        )
    }

    fn value_at(&self, elapsed_ns: u64) -> Option<f64> {
        let exponent = self.exponent;
        self.common
            .value_at(elapsed_ns, move |progress| progress.powf(exponent))
    }
}

/// Precomputed Poisson-process step trajectory normalized to the exact duration
/// and target.
pub struct PoissonRamp {
    common: CommonRamp,
    event_times_ns: Vec<u64>,
    values: Vec<f64>,
    step_index: usize,
}

impl PoissonRamp {
    /// Generate a deterministic trajectory from the run RNG root.
    pub fn new(config: RamperConfig, root: RngRoot) -> Result<Self, RampConfigError> {
        let common = CommonRamp::new(config);
        if common.range == 0.0 {
            return Ok(Self {
                common,
                event_times_ns: Vec::new(),
                values: vec![common.start],
                step_index: 0,
            });
        }

        let duration_seconds = common.duration_ns as f64 / NANOS_PER_SECOND;
        let expected_rate = common.range / duration_seconds;
        if !expected_rate.is_finite() || expected_rate <= 0.0 {
            return Err(RampConfigError::InvalidPoissonRate(expected_rate));
        }
        let mut rng = RustRandomGenerator::from_seed(root.derive_seed(TIMING_RAMP_POISSON));
        let mut raw_intervals = Vec::new();
        let mut cumulative = 0.0;
        while cumulative < duration_seconds {
            let interval = rng
                .expovariate(expected_rate)
                .expect("validated finite positive Poisson rate");
            raw_intervals.push(interval);
            cumulative += interval;
        }

        let time_scale = duration_seconds / cumulative;
        let event_count = raw_intervals.len();
        let step_size = common.range / event_count as f64;
        let mut event_times_ns = Vec::with_capacity(event_count);
        let mut values = Vec::with_capacity(event_count + 1);
        values.push(common.start);
        cumulative = 0.0;
        for (index, interval) in raw_intervals.into_iter().enumerate() {
            cumulative += interval * time_scale;
            let event_ns = if index + 1 == event_count {
                common.duration_ns
            } else {
                (cumulative * NANOS_PER_SECOND)
                    .round()
                    .clamp(0.0, common.duration_ns as f64) as u64
            };
            event_times_ns.push(event_ns);
            values.push(if index + 1 == event_count {
                common.target
            } else {
                common.start + step_size * (index + 1) as f64 * common.direction
            });
        }

        Ok(Self {
            common,
            event_times_ns,
            values,
            step_index: 0,
        })
    }

    /// Precomputed event times in nanoseconds, including an exact final duration.
    pub fn event_times_ns(&self) -> &[u64] {
        &self.event_times_ns
    }

    /// Step-function values: initial value followed by one value per event.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

impl RampStrategy for PoissonRamp {
    fn start(&self) -> f64 {
        self.common.start
    }

    fn target(&self) -> f64 {
        self.common.target
    }

    fn update_interval_ns(&self) -> Option<u64> {
        self.common.update_interval_ns
    }

    fn next_step(&mut self, _current: f64, elapsed_ns: u64) -> Option<(u64, f64)> {
        let target_time = *self.event_times_ns.get(self.step_index)?;
        let next_value = self.values[self.step_index + 1];
        self.step_index += 1;
        Some((target_time.saturating_sub(elapsed_ns), next_value))
    }

    fn value_at(&self, elapsed_ns: u64) -> Option<f64> {
        if self.event_times_ns.is_empty() || elapsed_ns >= self.common.duration_ns {
            return None;
        }
        let index = self
            .event_times_ns
            .partition_point(|event_ns| *event_ns <= elapsed_ns);
        Some(self.values[index])
    }
}

/// Async driver that applies a strategy to an injected setter closure.
pub struct RampDriver {
    clock: Rc<dyn Clock>,
    strategy: Box<dyn RampStrategy>,
    setter: Box<dyn FnMut(f64)>,
    update_interval_ns: Option<u64>,
}

impl RampDriver {
    /// Build a driver. The strategy's immutable configuration selects discrete
    /// or continuous mode, leaving no second cadence that could disagree.
    pub fn new(
        clock: Rc<dyn Clock>,
        strategy: Box<dyn RampStrategy>,
        setter: impl FnMut(f64) + 'static,
    ) -> Self {
        let update_interval_ns = strategy.update_interval_ns();
        Self {
            clock,
            strategy,
            setter: Box::new(setter),
            update_interval_ns,
        }
    }

    /// Run to natural completion in the current task.
    pub async fn run(mut self) {
        let current = self.strategy.start();
        (self.setter)(current);
        self.run_after_start(current).await;
    }

    /// Apply the initial value synchronously, then spawn the remainder on the
    /// current [`tokio::task::LocalSet`]. Synchronous initial application keeps
    /// issuance from observing the steady-state actuator before the ramp task is
    /// first polled.
    pub fn spawn_local(mut self) -> RampHandle {
        let current = self.strategy.start();
        (self.setter)(current);
        let task = tokio::task::spawn_local(async move {
            self.run_after_start(current).await;
        });
        RampHandle { task }
    }

    async fn run_after_start(&mut self, mut current: f64) {
        let started_at_ns = self.clock.now_ns();
        match self.update_interval_ns {
            None => loop {
                let elapsed_ns = elapsed_since(self.clock.now_ns(), started_at_ns);
                let Some((delay_ns, next_value)) = self.strategy.next_step(current, elapsed_ns)
                else {
                    if current != self.strategy.target() {
                        (self.setter)(self.strategy.target());
                    }
                    break;
                };
                self.clock.clone().sleep(delay_ns as i64).await;
                current = next_value;
                (self.setter)(current);
            },
            Some(update_interval_ns) => loop {
                self.clock.clone().sleep(update_interval_ns as i64).await;
                let elapsed_ns = elapsed_since(self.clock.now_ns(), started_at_ns);
                match self.strategy.value_at(elapsed_ns) {
                    Some(value) => (self.setter)(value),
                    None => {
                        (self.setter)(self.strategy.target());
                        break;
                    }
                }
            },
        }
    }
}

fn elapsed_since(now_ns: i64, started_at_ns: i64) -> u64 {
    now_ns.saturating_sub(started_at_ns).max(0) as u64
}

/// Control handle for a locally spawned ramp.
pub struct RampHandle {
    task: JoinHandle<()>,
}

/// Terminal failure from a locally spawned ramp task.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RampTaskError {
    /// The caller stopped the ramp, intentionally freezing its current value.
    Cancelled,
    /// The ramp task panicked; the task runtime's diagnostic is retained.
    Panicked(String),
}

impl RampTaskError {
    /// Whether this is the expected result of [`RampHandle::stop`].
    pub const fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl Display for RampTaskError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cancelled => write!(f, "ramp task was cancelled"),
            Self::Panicked(message) => write!(f, "ramp task panicked: {message}"),
        }
    }
}

impl Error for RampTaskError {}

impl RampHandle {
    /// Whether the ramp task has not yet reached a terminal state.
    pub fn is_running(&self) -> bool {
        !self.task.is_finished()
    }

    /// Stop the ramp without applying the target. The last setter value remains
    /// in effect.
    pub fn stop(&self) {
        self.task.abort();
    }

    /// Wait for natural completion or task cancellation.
    pub async fn wait(self) -> Result<(), RampTaskError> {
        self.task.await.map_err(|error| {
            if error.is_cancelled() {
                RampTaskError::Cancelled
            } else {
                RampTaskError::Panicked(error.to_string())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{SimClock, drive_sim};
    use std::cell::RefCell;

    fn cfg(start: f64, target: f64, duration_ns: u64) -> RamperConfig {
        RamperConfig::new(start, target, duration_ns).unwrap()
    }

    #[test]
    fn config_validates_every_numeric_field() {
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(RamperConfig::new(value, 1.0, 1).is_err());
            assert!(RamperConfig::new(1.0, value, 1).is_err());
        }
        assert!(RamperConfig::new(1.0, 2.0, 0).is_err());
        assert!(cfg(1.0, 2.0, 10).with_step_size(0.0).is_err());
        assert!(cfg(1.0, 2.0, 10).with_exponent(1.0).is_err());
        assert!(cfg(1.0, 2.0, 10).with_update_interval_ns(0).is_err());
    }

    #[test]
    fn linear_discrete_steps_and_self_corrects_timing() {
        let mut up = LinearRamp::new(cfg(1.0, 5.0, 400));
        assert_eq!(up.next_step(1.0, 0), Some((100, 2.0)));
        assert_eq!(up.next_step(3.0, 250), Some((50, 4.0)));
        assert_eq!(up.next_step(5.0, 400), None);
        assert_eq!(up.next_step(6.0, 400), None);

        let custom = cfg(1.0, 100.0, 990).with_step_size(25.0).unwrap();
        let mut ramp = LinearRamp::new(custom);
        let mut current = 1.0;
        let mut values = vec![current];
        while let Some((_delay, next)) = ramp.next_step(current, 0) {
            current = next;
            values.push(current);
        }
        assert_eq!(values, vec![1.0, 26.0, 51.0, 76.0, 100.0]);

        let mut down = LinearRamp::new(cfg(5.0, 1.0, 400));
        assert_eq!(down.next_step(5.0, 0), Some((100, 4.0)));
    }

    #[test]
    fn continuous_linear_interpolates_both_directions() {
        let up = LinearRamp::new(cfg(1.0, 101.0, 1_000));
        assert_eq!(up.value_at(0), Some(1.0));
        assert_eq!(up.value_at(500), Some(51.0));
        assert_eq!(up.value_at(1_000), None);

        let down = LinearRamp::new(cfg(100.0, 1.0, 1_000));
        assert_eq!(down.value_at(500), Some(50.5));
    }

    #[test]
    fn exponential_curve_is_ease_in_and_the_pair_is_inverse() {
        let config = cfg(1.0, 101.0, 1_000).with_exponent(3.0).unwrap();
        let ramp = ExponentialRamp::new(config);
        assert_eq!(ramp.value_at(500), Some(13.5));
        for progress in [0.0_f64, 0.01, 0.25, 0.5, 0.9, 1.0] {
            let round_trip = ramp.time_to_value_progress(ramp.value_progress_to_time(progress));
            assert!(
                (round_trip - progress).abs() < 1e-12,
                "{progress} -> {round_trip}"
            );
        }
    }

    #[test]
    fn exponential_discrete_uses_unit_steps_and_decreasing_delays() {
        let mut ramp = ExponentialRamp::new(cfg(1.0, 10.0, 900));
        let mut current = 1.0;
        let mut elapsed = 0;
        let mut delays = Vec::new();
        while let Some((delay, next)) = ramp.next_step(current, elapsed) {
            delays.push(delay);
            elapsed += delay;
            current = next;
        }
        assert_eq!(current, 10.0);
        assert_eq!(elapsed, 900);
        assert!(delays.windows(2).all(|pair| pair[1] <= pair[0]));
    }

    #[test]
    fn poisson_trajectory_is_deterministic_normalized_and_pinned() {
        let root = RngRoot::new(Some(42));
        let a = PoissonRamp::new(cfg(1.0, 20.0, 10_000), root).unwrap();
        let b = PoissonRamp::new(cfg(1.0, 20.0, 10_000), root).unwrap();
        assert_eq!(a.event_times_ns(), b.event_times_ns());
        assert_eq!(a.values(), b.values());
        assert_eq!(a.event_times_ns().last(), Some(&10_000));
        assert_eq!(a.values().last(), Some(&20.0));
        assert!(a.event_times_ns().windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(a.values().windows(2).all(|pair| pair[0] <= pair[1]));
    }

    #[test]
    fn poisson_actuator_trajectories_are_distinct_and_component_order_independent() {
        let root = RngRoot::new(Some(42));
        let trajectory = |identifier| {
            PoissonRamp::new(cfg(1.0, 20.0, 1_000_000_000), root.derive_root(identifier))
                .unwrap()
                .event_times_ns()
                .to_vec()
        };
        let forward = [
            crate::rng::namespace::TIMING_RAMP_CONCURRENCY,
            crate::rng::namespace::TIMING_RAMP_PREFILL_CONCURRENCY,
            crate::rng::namespace::TIMING_RAMP_REQUEST_RATE,
        ]
        .map(trajectory);
        let reverse = [
            crate::rng::namespace::TIMING_RAMP_REQUEST_RATE,
            crate::rng::namespace::TIMING_RAMP_PREFILL_CONCURRENCY,
            crate::rng::namespace::TIMING_RAMP_CONCURRENCY,
        ]
        .map(trajectory);

        assert_eq!(forward[0], reverse[2]);
        assert_eq!(forward[1], reverse[1]);
        assert_eq!(forward[2], reverse[0]);
        assert_ne!(forward[0], forward[1]);
        assert_ne!(forward[0], forward[2]);
        assert_ne!(forward[1], forward[2]);
    }

    #[test]
    fn poisson_value_at_is_a_step_function_consistent_with_next_step() {
        let root = RngRoot::new(Some(9));
        let mut steps = PoissonRamp::new(cfg(10.0, 1.0, 1_000_000), root).unwrap();
        let events = steps.event_times_ns().to_vec();
        let values = steps.values().to_vec();
        let sampled = PoissonRamp::new(cfg(10.0, 1.0, 1_000_000), root).unwrap();
        assert_eq!(sampled.value_at(0), Some(10.0));
        for (index, event) in events.iter().enumerate() {
            if *event < 1_000_000 {
                assert_eq!(sampled.value_at(*event), Some(values[index + 1]));
            }
        }
        let mut elapsed = 0;
        let mut current = 10.0;
        while let Some((delay, next)) = steps.next_step(current, elapsed) {
            elapsed += delay;
            current = next;
        }
        assert_eq!(elapsed, 1_000_000);
        assert_eq!(current, 1.0);
    }

    #[test]
    fn discrete_driver_uses_sim_clock_and_reaches_exact_target() {
        let clock = Rc::new(SimClock::new());
        let values = Rc::new(RefCell::new(Vec::new()));
        let captured = values.clone();
        let setter_clock = clock.clone();
        let driver = RampDriver::new(
            clock.clone(),
            Box::new(LinearRamp::new(cfg(1.0, 5.0, 400))),
            move |value| captured.borrow_mut().push((setter_clock.now_ns(), value)),
        );
        drive_sim(clock.clone(), driver.run());
        assert_eq!(
            *values.borrow(),
            vec![(0, 1.0), (100, 2.0), (200, 3.0), (300, 4.0), (400, 5.0)]
        );
    }

    #[test]
    fn decreasing_slot_ramp_uses_debt_drain_until_live_guards_return() {
        let clock = Rc::new(SimClock::new());
        let slots = Rc::new(crate::timing::SlotPool::new(5));
        let slots_for_setter = slots.clone();
        let driver = RampDriver::new(
            clock.clone(),
            Box::new(LinearRamp::new(cfg(5.0, 1.0, 400))),
            move |value| slots_for_setter.set_limit(value as usize),
        );
        drive_sim(clock, async {
            let guards = vec![
                slots.acquire().await,
                slots.acquire().await,
                slots.acquire().await,
                slots.acquire().await,
            ];
            driver.run().await;
            assert_eq!(slots.current_limit(), 1);
            assert_eq!(slots.effective_slots(), 0);
            assert_eq!(slots.debt(), 3);
            drop(guards);
            assert_eq!(slots.debt(), 0);
            assert_eq!(slots.effective_slots(), 1);
        });
    }

    #[test]
    fn continuous_driver_samples_then_force_sets_target() {
        let clock = Rc::new(SimClock::new());
        let values = Rc::new(RefCell::new(Vec::new()));
        let captured = values.clone();
        let setter_clock = clock.clone();
        let driver = RampDriver::new(
            clock.clone(),
            Box::new(LinearRamp::new(
                cfg(1.0, 5.0, 1_000).with_update_interval_ns(200).unwrap(),
            )),
            move |value| captured.borrow_mut().push((setter_clock.now_ns(), value)),
        );
        drive_sim(clock.clone(), driver.run());
        let values = values.borrow();
        assert_eq!(values.first(), Some(&(0, 1.0)));
        assert_eq!(values.last(), Some(&(1_000, 5.0)));
        assert_eq!(values.len(), 6);
        assert!(values.windows(2).all(|pair| pair[1].1 >= pair[0].1));
    }

    #[test]
    fn stopping_a_spawned_driver_freezes_partial_progress() {
        let clock = Rc::new(SimClock::new());
        let values = Rc::new(RefCell::new(Vec::new()));
        let captured = values.clone();
        let driver = RampDriver::new(
            clock.clone(),
            Box::new(LinearRamp::new(cfg(1.0, 11.0, 1_000))),
            move |value| captured.borrow_mut().push(value),
        );
        drive_sim(clock.clone(), async {
            let handle = driver.spawn_local();
            assert!(handle.is_running());
            clock.clone().sleep(350).await;
            handle.stop();
            let error = handle.wait().await.expect_err("aborted ramp task");
            assert!(error.is_cancelled());
        });
        let values = values.borrow();
        assert_eq!(values.first(), Some(&1.0));
        assert!(*values.last().unwrap() < 11.0);
        assert!(!values.contains(&11.0));
    }

    #[test]
    fn start_equal_target_is_applied_once_in_discrete_mode() {
        let clock = Rc::new(SimClock::new());
        let values = Rc::new(RefCell::new(Vec::new()));
        let captured = values.clone();
        let driver = RampDriver::new(
            clock.clone(),
            Box::new(LinearRamp::new(cfg(5.0, 5.0, 100))),
            move |value| captured.borrow_mut().push(value),
        );
        drive_sim(clock, driver.run());
        assert_eq!(*values.borrow(), vec![5.0]);
    }

    #[test]
    fn strategy_trait_is_object_safe() {
        let mut strategy: Box<dyn RampStrategy> = Box::new(LinearRamp::new(cfg(1.0, 2.0, 100)));
        assert_eq!(strategy.next_step(1.0, 0), Some((100, 2.0)));
    }
}
