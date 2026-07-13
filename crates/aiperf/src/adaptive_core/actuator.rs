// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive control actuators.
//!
//! The controller depends only on [`ControlActuator`]; changing from session
//! concurrency to prefill concurrency, request rate, or target users does not
//! alter controller logic.

use std::cell::RefCell;
use std::rc::Rc;

use crate::clock::Clock;
use crate::timing::{IntervalGenerator, SlotPool};
use serde::Serialize;

use crate::adaptive_core::error::AdaptiveError;

/// Serializable target/actual state for one adaptive control variable.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ControlSnapshot {
    /// Requested control value.
    pub target_value: f64,
    /// Effective control value reported by the controlled subsystem.
    pub actual_value: f64,
    /// Active user count for a user-target actuator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active_users: Option<usize>,
    /// Users draining after a target reduction.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retiring_users: Option<usize>,
    /// User sessions cancelled while retiring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cancelled: Option<usize>,
}

impl ControlSnapshot {
    fn scalar(value: f64) -> Self {
        Self {
            target_value: value,
            actual_value: value,
            active_users: None,
            retiring_users: None,
            cancelled: None,
        }
    }
}

/// Object-safe setter interface for one adaptive load knob.
pub trait ControlActuator {
    /// Stable control-variable name used in artifacts.
    fn variable(&self) -> &'static str;
    /// Inclusive lower bound.
    fn minimum(&self) -> f64;
    /// Inclusive upper bound.
    fn maximum(&self) -> f64;
    /// Current effective value.
    fn current(&self) -> f64;
    /// Clamp and apply a requested value, returning the applied value.
    fn set(&self, value: f64) -> Result<f64, AdaptiveError>;
    /// Snapshot target and effective state for artifact emission.
    fn snapshot(&self) -> ControlSnapshot;
}

fn validate_integer_bounds(
    minimum: usize,
    maximum: usize,
    variable: &str,
) -> Result<(), AdaptiveError> {
    if minimum == 0 {
        return Err(AdaptiveError::InvalidConfig(format!(
            "adaptive {variable} minimum must be >= 1"
        )));
    }
    if maximum <= minimum {
        return Err(AdaptiveError::InvalidConfig(format!(
            "adaptive {variable} maximum ({maximum}) must be > minimum ({minimum})"
        )));
    }
    Ok(())
}

fn clamp_integer(value: f64, minimum: usize, maximum: usize) -> Result<usize, AdaptiveError> {
    if !value.is_finite() {
        return Err(AdaptiveError::Actuator(format!(
            "integer control value must be finite, got {value}"
        )));
    }
    Ok(value.clamp(minimum as f64, maximum as f64).trunc() as usize)
}

/// Session-concurrency adapter over a debt-draining [`SlotPool`].
pub struct SessionConcurrencyActuator {
    pool: Rc<SlotPool>,
    minimum: usize,
    maximum: usize,
}

impl SessionConcurrencyActuator {
    /// Build a session-concurrency actuator with validated integer bounds.
    pub fn new(pool: Rc<SlotPool>, minimum: usize, maximum: usize) -> Result<Self, AdaptiveError> {
        validate_integer_bounds(minimum, maximum, "concurrency")?;
        Ok(Self {
            pool,
            minimum,
            maximum,
        })
    }

    /// The controlled session slot pool.
    pub fn pool(&self) -> &Rc<SlotPool> {
        &self.pool
    }
}

impl ControlActuator for SessionConcurrencyActuator {
    fn variable(&self) -> &'static str {
        "concurrency"
    }

    fn minimum(&self) -> f64 {
        self.minimum as f64
    }

    fn maximum(&self) -> f64 {
        self.maximum as f64
    }

    fn current(&self) -> f64 {
        self.pool.current_limit() as f64
    }

    fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
        let value = clamp_integer(value, self.minimum, self.maximum)?;
        self.pool.set_limit(value);
        Ok(value as f64)
    }

    fn snapshot(&self) -> ControlSnapshot {
        ControlSnapshot::scalar(self.current())
    }
}

/// Prefill-concurrency adapter over a debt-draining [`SlotPool`].
pub struct PrefillConcurrencyActuator {
    pool: Rc<SlotPool>,
    minimum: usize,
    maximum: usize,
}

impl PrefillConcurrencyActuator {
    /// Build a prefill-concurrency actuator with validated integer bounds.
    pub fn new(pool: Rc<SlotPool>, minimum: usize, maximum: usize) -> Result<Self, AdaptiveError> {
        validate_integer_bounds(minimum, maximum, "prefill_concurrency")?;
        Ok(Self {
            pool,
            minimum,
            maximum,
        })
    }

    /// The controlled prefill slot pool.
    pub fn pool(&self) -> &Rc<SlotPool> {
        &self.pool
    }
}

impl ControlActuator for PrefillConcurrencyActuator {
    fn variable(&self) -> &'static str {
        "prefill_concurrency"
    }

    fn minimum(&self) -> f64 {
        self.minimum as f64
    }

    fn maximum(&self) -> f64 {
        self.maximum as f64
    }

    fn current(&self) -> f64 {
        self.pool.current_limit() as f64
    }

    fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
        let value = clamp_integer(value, self.minimum, self.maximum)?;
        self.pool.set_limit(value);
        Ok(value as f64)
    }

    fn snapshot(&self) -> ControlSnapshot {
        ControlSnapshot::scalar(self.current())
    }
}

/// Request-rate adapter over the active [`IntervalGenerator`].
pub struct RequestRateActuator {
    generator: Rc<RefCell<Box<dyn IntervalGenerator>>>,
    minimum: f64,
    maximum: f64,
}

impl RequestRateActuator {
    /// Build a request-rate actuator with positive finite bounds.
    pub fn new(
        generator: Rc<RefCell<Box<dyn IntervalGenerator>>>,
        minimum: f64,
        maximum: f64,
    ) -> Result<Self, AdaptiveError> {
        validate_rate_bounds(minimum, maximum)?;
        Ok(Self {
            generator,
            minimum,
            maximum,
        })
    }

    /// The controlled interval generator.
    pub fn generator(&self) -> &Rc<RefCell<Box<dyn IntervalGenerator>>> {
        &self.generator
    }
}

impl ControlActuator for RequestRateActuator {
    fn variable(&self) -> &'static str {
        "request_rate"
    }

    fn minimum(&self) -> f64 {
        self.minimum
    }

    fn maximum(&self) -> f64 {
        self.maximum
    }

    fn current(&self) -> f64 {
        self.generator.borrow().rate()
    }

    fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
        if !value.is_finite() {
            return Err(AdaptiveError::Actuator(format!(
                "request-rate control value must be finite, got {value}"
            )));
        }
        let value = value.clamp(self.minimum, self.maximum);
        self.generator.borrow_mut().set_rate(value);
        let actual = self.generator.borrow().rate();
        if !actual.is_finite() || (actual - value).abs() > f64::EPSILON * value.abs().max(1.0) {
            return Err(AdaptiveError::Actuator(
                "selected interval generator does not support request-rate control".to_string(),
            ));
        }
        Ok(actual)
    }

    fn snapshot(&self) -> ControlSnapshot {
        ControlSnapshot::scalar(self.current())
    }
}

fn validate_rate_bounds(minimum: f64, maximum: f64) -> Result<(), AdaptiveError> {
    if !minimum.is_finite() || minimum <= 0.0 {
        return Err(AdaptiveError::InvalidConfig(format!(
            "adaptive request_rate minimum must be positive and finite, got {minimum}"
        )));
    }
    if !maximum.is_finite() || maximum <= minimum {
        return Err(AdaptiveError::InvalidConfig(format!(
            "adaptive request_rate maximum ({maximum}) must be finite and > minimum ({minimum})"
        )));
    }
    Ok(())
}

/// Runtime hook implemented by a user-centric workload that can resize its
/// active target without rebuilding the adaptive controller.
pub trait UserTarget {
    /// Apply a new positive target-user count at `now_ns` on the run clock.
    fn set_target_users(&self, value: usize, now_ns: i64) -> Result<(), AdaptiveError>;
    /// Snapshot target, active, retiring, and cancellation counts.
    fn user_control_snapshot(&self) -> ControlSnapshot;
}

/// Target-user adapter over a [`UserTarget`] workload hook.
pub struct UsersActuator {
    target: Rc<dyn UserTarget>,
    clock: Rc<dyn Clock>,
    minimum: usize,
    maximum: usize,
    current: std::cell::Cell<usize>,
}

impl UsersActuator {
    /// Build a user-target actuator with validated integer bounds.
    pub fn new(
        target: Rc<dyn UserTarget>,
        clock: Rc<dyn Clock>,
        minimum: usize,
        maximum: usize,
    ) -> Result<Self, AdaptiveError> {
        validate_integer_bounds(minimum, maximum, "users")?;
        Ok(Self {
            target,
            clock,
            minimum,
            maximum,
            current: std::cell::Cell::new(minimum),
        })
    }
}

impl ControlActuator for UsersActuator {
    fn variable(&self) -> &'static str {
        "users"
    }

    fn minimum(&self) -> f64 {
        self.minimum as f64
    }

    fn maximum(&self) -> f64 {
        self.maximum as f64
    }

    fn current(&self) -> f64 {
        self.current.get() as f64
    }

    fn set(&self, value: f64) -> Result<f64, AdaptiveError> {
        let value = clamp_integer(value, self.minimum, self.maximum)?;
        self.target.set_target_users(value, self.clock.now_ns())?;
        self.current.set(value);
        Ok(value as f64)
    }

    fn snapshot(&self) -> ControlSnapshot {
        let mut snapshot = self.target.user_control_snapshot();
        snapshot.target_value = self.current();
        snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SimClock;
    use crate::timing::intervals::ConcurrencyBurst;
    use crate::timing::intervals::Constant;

    #[test]
    fn slot_actuator_clamps_and_uses_debt_drain() {
        let pool = Rc::new(SlotPool::new(4));
        let actuator = SessionConcurrencyActuator::new(pool.clone(), 2, 8).unwrap();
        let _held = [pool.try_acquire().unwrap(), pool.try_acquire().unwrap()];
        assert_eq!(actuator.set(99.0).unwrap(), 8.0);
        assert_eq!(pool.current_limit(), 8);
        assert_eq!(actuator.set(2.0).unwrap(), 2.0);
        assert_eq!(pool.current_limit(), 2);
        assert!(pool.debt() <= 2);
        assert_eq!(actuator.snapshot().target_value, 2.0);
    }

    #[test]
    fn rate_actuator_updates_the_live_generator() {
        let generator: Rc<RefCell<Box<dyn IntervalGenerator>>> =
            Rc::new(RefCell::new(Box::new(Constant::new(20.0))));
        let actuator = RequestRateActuator::new(generator.clone(), 2.0, 20.0).unwrap();
        actuator.set(12.5).unwrap();
        assert_eq!(generator.borrow().rate(), 12.5);
        assert_eq!(generator.borrow_mut().next_interval_ns(), 80_000_000);
    }

    #[test]
    fn invalid_bounds_are_rejected() {
        assert!(SessionConcurrencyActuator::new(Rc::new(SlotPool::new(1)), 0, 2).is_err());
        assert!(PrefillConcurrencyActuator::new(Rc::new(SlotPool::new(1)), 2, 2).is_err());
        let generator: Rc<RefCell<Box<dyn IntervalGenerator>>> =
            Rc::new(RefCell::new(Box::new(Constant::new(1.0))));
        assert!(RequestRateActuator::new(generator, 2.0, 1.0).is_err());
    }

    #[test]
    fn burst_generator_rejects_request_rate_control() {
        let generator: Rc<RefCell<Box<dyn IntervalGenerator>>> =
            Rc::new(RefCell::new(Box::new(ConcurrencyBurst)));
        let actuator = RequestRateActuator::new(generator, 1.0, 2.0).unwrap();
        assert!(actuator.set(1.5).is_err());
    }

    struct RecordingUserTarget {
        target: std::cell::Cell<usize>,
        set_at_ns: std::cell::Cell<i64>,
    }

    impl UserTarget for RecordingUserTarget {
        fn set_target_users(&self, value: usize, now_ns: i64) -> Result<(), AdaptiveError> {
            self.target.set(value);
            self.set_at_ns.set(now_ns);
            Ok(())
        }

        fn user_control_snapshot(&self) -> ControlSnapshot {
            ControlSnapshot {
                target_value: self.target.get() as f64,
                actual_value: 3.0,
                active_users: Some(3),
                retiring_users: Some(1),
                cancelled: Some(0),
            }
        }
    }

    #[test]
    fn users_actuator_clamps_and_stamps_the_injected_clock() {
        let clock = Rc::new(SimClock::new());
        clock.advance_to(123);
        let target = Rc::new(RecordingUserTarget {
            target: std::cell::Cell::new(1),
            set_at_ns: std::cell::Cell::new(0),
        });
        let actuator = UsersActuator::new(target.clone(), clock, 1, 5).unwrap();
        assert_eq!(actuator.set(99.0).unwrap(), 5.0);
        assert_eq!(target.target.get(), 5);
        assert_eq!(target.set_at_ns.get(), 123);
        assert_eq!(actuator.snapshot().active_users, Some(3));
        assert_eq!(actuator.snapshot().target_value, 5.0);
    }
}
