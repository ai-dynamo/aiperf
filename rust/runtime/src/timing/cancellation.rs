// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-request client-disconnect policy.
//!
//! The policy makes one Bernoulli decision at issuance time and returns a fixed
//! delay. The transport consumes that scalar only after the complete request body
//! has been sent. Warmup returns before drawing from the RNG, so warmup traffic
//! cannot perturb the reproducible profiling sequence.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::rng::namespace::TIMING_REQUEST_CANCELLATION;
use crate::rng::{ConfiguredRandomGenerator, RandomGenerator, RngRoot};
use crate::timing::NANOS_PER_SECOND;

/// Benchmark phase relevant to ancillary timing policies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Setup traffic for which simulated client disconnects are disabled.
    Warmup,
    /// Measured traffic for which the configured policy is active.
    Profiling,
}

/// A pluggable decision policy for arming a post-send cancellation timer.
///
/// The returned delay is relative to the transport's send-complete timestamp,
/// never relative to issuance or connection acquisition.
pub trait CancellationPolicy {
    /// Return the fixed delay for the next request, or `None` when it should not
    /// be cancelled.
    fn next_cancel_delay_ns(&mut self, phase: Phase) -> Option<i64>;
}

/// Invalid cancellation-policy configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CancellationPolicyError {
    /// The percentage was not finite or outside the inclusive `0..=100` range.
    InvalidRatePercent(f64),
    /// The delay in seconds was negative, non-finite, or too large for clock time.
    InvalidDelaySeconds(f64),
    /// A nanosecond delay supplied directly was negative.
    InvalidDelayNs(i64),
}

impl Display for CancellationPolicyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRatePercent(value) => write!(
                f,
                "cancellation rate must be a finite percentage in 0..=100, got {value}"
            ),
            Self::InvalidDelaySeconds(value) => write!(
                f,
                "cancellation delay must be finite, non-negative, and representable in nanoseconds, got {value} seconds"
            ),
            Self::InvalidDelayNs(value) => {
                write!(f, "cancellation delay must be non-negative, got {value}ns")
            }
        }
    }
}

impl Error for CancellationPolicyError {}

/// Bernoulli selection with one constant post-send delay.
///
/// `rate_percent=None` and `Some(0.0)` disable the policy. A configured stream
/// derives `timing.request.cancellation` from the run root so other component
/// draws cannot perturb cancellation decisions.
pub struct BernoulliFixedDelay {
    enabled: bool,
    cancellation_probability: f64,
    delay_ns: i64,
    rng: ConfiguredRandomGenerator,
}

impl BernoulliFixedDelay {
    /// Construct from a percentage, delay in seconds, and optional run seed.
    pub fn from_seed(
        rate_percent: Option<f64>,
        delay_seconds: f64,
        seed: Option<u64>,
    ) -> Result<Self, CancellationPolicyError> {
        Self::new(rate_percent, delay_seconds, RngRoot::new(seed))
    }

    /// Construct from a percentage and delay in seconds.
    ///
    /// Seconds are truncated to integer nanoseconds.
    pub fn new(
        rate_percent: Option<f64>,
        delay_seconds: f64,
        root: RngRoot,
    ) -> Result<Self, CancellationPolicyError> {
        if !delay_seconds.is_finite()
            || delay_seconds < 0.0
            || delay_seconds * NANOS_PER_SECOND >= i64::MAX as f64
        {
            return Err(CancellationPolicyError::InvalidDelaySeconds(delay_seconds));
        }
        let delay_ns = (delay_seconds * NANOS_PER_SECOND) as i64;
        Self::from_delay_ns(rate_percent, delay_ns, root)
    }

    /// Construct from a percentage and an already-converted nanosecond delay.
    pub fn from_delay_ns(
        rate_percent: Option<f64>,
        delay_ns: i64,
        root: RngRoot,
    ) -> Result<Self, CancellationPolicyError> {
        let rate_percent = rate_percent.unwrap_or(0.0);
        if !rate_percent.is_finite() || !(0.0..=100.0).contains(&rate_percent) {
            return Err(CancellationPolicyError::InvalidRatePercent(rate_percent));
        }
        if delay_ns < 0 {
            return Err(CancellationPolicyError::InvalidDelayNs(delay_ns));
        }
        Ok(Self {
            enabled: rate_percent != 0.0,
            cancellation_probability: rate_percent / 100.0,
            delay_ns,
            rng: root.derive_generator(TIMING_REQUEST_CANCELLATION),
        })
    }

    /// Construct from an integer-nanosecond delay and optional run seed.
    pub fn from_delay_ns_seed(
        rate_percent: Option<f64>,
        delay_ns: i64,
        seed: Option<u64>,
    ) -> Result<Self, CancellationPolicyError> {
        Self::from_delay_ns(rate_percent, delay_ns, RngRoot::new(seed))
    }

    /// Whether a non-zero cancellation percentage is configured.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// The constant delay returned for selected profiling requests.
    pub const fn delay_ns(&self) -> i64 {
        self.delay_ns
    }

    /// Cancellation probability in the inclusive `0.0..=1.0` range.
    pub const fn probability(&self) -> f64 {
        self.cancellation_probability
    }
}

impl CancellationPolicy for BernoulliFixedDelay {
    fn next_cancel_delay_ns(&mut self, phase: Phase) -> Option<i64> {
        if !self.enabled || phase == Phase::Warmup {
            return None;
        }
        (self.rng.random() < self.cancellation_probability).then_some(self.delay_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_rates_never_cancel() {
        for rate in [None, Some(0.0)] {
            let mut policy =
                BernoulliFixedDelay::new(rate, 1.0, RngRoot::new(Some(42))).expect("valid policy");
            assert!(!policy.is_enabled());
            assert!((0..100).all(|_| policy.next_cancel_delay_ns(Phase::Profiling).is_none()));
        }
    }

    #[test]
    fn full_rate_returns_the_fixed_delay_including_zero() {
        let mut delayed =
            BernoulliFixedDelay::new(Some(100.0), 2.5, RngRoot::new(Some(1))).unwrap();
        assert!(
            (0..100)
                .all(|_| { delayed.next_cancel_delay_ns(Phase::Profiling) == Some(2_500_000_000) })
        );

        let mut immediate =
            BernoulliFixedDelay::new(Some(100.0), 0.0, RngRoot::new(Some(1))).unwrap();
        assert_eq!(immediate.next_cancel_delay_ns(Phase::Profiling), Some(0));
    }

    #[test]
    fn warmup_is_disabled_without_consuming_rng() {
        let root = RngRoot::new(Some(42));
        let mut with_warmup = BernoulliFixedDelay::new(Some(50.0), 1.0, root).unwrap();
        for _ in 0..20 {
            assert_eq!(with_warmup.next_cancel_delay_ns(Phase::Warmup), None);
        }
        let after_warmup: Vec<_> = (0..100)
            .map(|_| with_warmup.next_cancel_delay_ns(Phase::Profiling))
            .collect();

        let mut profiling_only = BernoulliFixedDelay::new(Some(50.0), 1.0, root).unwrap();
        let direct: Vec<_> = (0..100)
            .map(|_| profiling_only.next_cancel_delay_ns(Phase::Profiling))
            .collect();
        assert_eq!(after_warmup, direct);
    }

    #[test]
    fn seeded_decisions_are_reproducible_and_statistical() {
        let root = RngRoot::new(Some(7));
        let mut a = BernoulliFixedDelay::new(Some(50.0), 0.25, root).unwrap();
        let mut b = BernoulliFixedDelay::new(Some(50.0), 0.25, root).unwrap();
        let seq_a: Vec<_> = (0..1_000)
            .map(|_| a.next_cancel_delay_ns(Phase::Profiling))
            .collect();
        let seq_b: Vec<_> = (0..1_000)
            .map(|_| b.next_cancel_delay_ns(Phase::Profiling))
            .collect();
        assert_eq!(seq_a, seq_b);
        let selected = seq_a.iter().filter(|value| value.is_some()).count();
        assert!((400..=600).contains(&selected), "selected {selected}/1000");
        assert!(seq_a.iter().flatten().all(|delay| *delay == 250_000_000));
    }

    #[test]
    fn validates_rate_and_delay() {
        let root = RngRoot::new(Some(0));
        for rate in [f64::NAN, f64::INFINITY, -0.1, 100.1] {
            assert!(BernoulliFixedDelay::new(Some(rate), 0.0, root).is_err());
        }
        for delay in [f64::NAN, f64::INFINITY, -0.1] {
            assert!(BernoulliFixedDelay::new(Some(1.0), delay, root).is_err());
        }
        assert!(BernoulliFixedDelay::from_delay_ns(Some(1.0), -1, root).is_err());
    }
}
