// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inter-arrival interval generators.
//!
//! Each generator yields successive inter-arrival intervals in **integer
//! nanoseconds** (the `Clock`'s native unit), drawn from a **seeded** RNG so a run
//! is bit-reproducible. `rate` is the average requests/second; actual intervals
//! vary except for [`Constant`] (deterministic) and [`ConcurrencyBurst`] (always 0,
//! throughput bounded by concurrency, not rate).
//!
//! The pacer converts a `next_interval_ns()` into an absolute target time and
//! `clock.sleep`s to it — identical code on real and virtual clocks.

use crate::rng::RandomGenerator;

const NANOS_PER_SECOND: f64 = 1_000_000_000.0;

/// Convert a non-negative interval in seconds to integer nanoseconds, rounding
/// to the nearest nanosecond with ties away from zero (`f64::round`; not Python
/// `round`'s half-to-even). Non-finite or negative inputs clamp to 0 (an
/// immediate arrival).
fn secs_to_ns(secs: f64) -> i64 {
    if !secs.is_finite() || secs <= 0.0 {
        return 0;
    }
    (secs * NANOS_PER_SECOND).round() as i64
}

/// Arrival pattern selecting the inter-arrival distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArrivalPattern {
    /// Fixed interval `1/rate` — deterministic, evenly spaced.
    Constant,
    /// Exponential inter-arrivals (a Poisson process); realistic bursty traffic.
    Poisson,
    /// Gamma inter-arrivals with tunable burstiness (`shape = smoothness`):
    /// `< 1` burstier, `= 1` Poisson-equivalent, `> 1` smoother.
    Gamma,
    /// Zero delay; throughput is bounded by the concurrency limit, not a rate.
    ConcurrencyBurst,
}

/// Produces successive inter-arrival intervals in nanoseconds. `set_rate`
/// supports rate ramping mid-run (takes effect on the next `next_interval_ns`).
pub trait IntervalGenerator {
    /// Next inter-arrival interval in nanoseconds (`>= 0`).
    fn next_interval_ns(&mut self) -> i64;
    /// Update the average request rate (requests/second). No-op for burst.
    fn set_rate(&mut self, rate: f64);
    /// Current average request rate (0.0 for burst).
    fn rate(&self) -> f64;
}

/// Poisson process: exponential inter-arrival times with mean `1/rate`.
pub struct Poisson {
    rate: f64,
    rng: RandomGenerator,
}

impl Poisson {
    /// Requires `rate > 0`.
    pub fn new(rate: f64, seed: u64) -> Self {
        assert!(rate > 0.0, "Poisson rate must be > 0, got {rate}");
        Self {
            rate,
            rng: RandomGenerator::from_seed(Some(seed)),
        }
    }
}

impl IntervalGenerator for Poisson {
    fn next_interval_ns(&mut self) -> i64 {
        // Exp(λ) has mean 1/λ; λ = rate, so mean interval = 1/rate seconds.
        secs_to_ns(
            self.rng
                .expovariate(self.rate)
                .expect("rate > 0 checked at construction/set_rate"),
        )
    }
    fn set_rate(&mut self, rate: f64) {
        assert!(rate > 0.0, "Poisson rate must be > 0, got {rate}");
        self.rate = rate;
    }
    fn rate(&self) -> f64 {
        self.rate
    }
}

/// Gamma inter-arrivals: generalizes Poisson with a smoothness (shape) knob while
/// holding the mean at `1/rate`.
pub struct GammaProcess {
    rate: f64,
    smoothness: f64,
    // Cached scale parameter (shape = smoothness), recomputed only on set_rate.
    // `RandomGenerator::gammavariate` reconstructs the Marsaglia-Tsang constants
    // internally per draw; caching the scale keeps the arithmetic off this hot
    // path, and arrival sampling is once-per-request so the rebuild is not a
    // measured bottleneck.
    scale: f64,
    rng: RandomGenerator,
}

impl GammaProcess {
    /// Requires `rate > 0` and `smoothness > 0`. `smoothness = 1` == Poisson.
    pub fn new(rate: f64, smoothness: f64, seed: u64) -> Self {
        assert!(rate > 0.0, "Gamma rate must be > 0, got {rate}");
        assert!(
            smoothness > 0.0,
            "Gamma smoothness must be > 0, got {smoothness}"
        );
        Self {
            rate,
            smoothness,
            scale: Self::scale(rate, smoothness),
            rng: RandomGenerator::from_seed(Some(seed)),
        }
    }

    /// scale = 1/(rate*smoothness) so mean = shape*scale = smoothness/(rate*smoothness) = 1/rate.
    fn scale(rate: f64, smoothness: f64) -> f64 {
        1.0 / (rate * smoothness)
    }
}

impl IntervalGenerator for GammaProcess {
    fn next_interval_ns(&mut self) -> i64 {
        secs_to_ns(
            self.rng
                .gammavariate(self.smoothness, self.scale)
                .expect("shape and scale > 0 checked at construction/set_rate"),
        )
    }
    fn set_rate(&mut self, rate: f64) {
        assert!(rate > 0.0, "Gamma rate must be > 0, got {rate}");
        self.rate = rate;
        self.scale = Self::scale(rate, self.smoothness);
    }
    fn rate(&self) -> f64 {
        self.rate
    }
}

/// Constant inter-arrivals: fixed period `1/rate`, evenly spaced.
pub struct Constant {
    rate: f64,
    period_ns: i64,
}

impl Constant {
    /// Requires `rate > 0`.
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "Constant rate must be > 0, got {rate}");
        Self {
            rate,
            period_ns: secs_to_ns(1.0 / rate),
        }
    }
}

impl IntervalGenerator for Constant {
    fn next_interval_ns(&mut self) -> i64 {
        self.period_ns
    }
    fn set_rate(&mut self, rate: f64) {
        assert!(rate > 0.0, "Constant rate must be > 0, got {rate}");
        self.rate = rate;
        self.period_ns = secs_to_ns(1.0 / rate);
    }
    fn rate(&self) -> f64 {
        self.rate
    }
}

/// Zero-delay burst: throughput is governed by the concurrency limit, not a rate.
pub struct ConcurrencyBurst;

impl IntervalGenerator for ConcurrencyBurst {
    fn next_interval_ns(&mut self) -> i64 {
        0
    }
    fn set_rate(&mut self, _rate: f64) {}
    fn rate(&self) -> f64 {
        0.0
    }
}

/// Build the interval generator for `pattern`. `rate` is required for every
/// pattern except [`ArrivalPattern::ConcurrencyBurst`]; `smoothness` defaults to
/// `1.0` (Poisson-equivalent) for [`ArrivalPattern::Gamma`] when `None`.
pub fn make_interval_generator(
    pattern: ArrivalPattern,
    rate: Option<f64>,
    smoothness: Option<f64>,
    seed: u64,
) -> Box<dyn IntervalGenerator> {
    match pattern {
        ArrivalPattern::ConcurrencyBurst => Box::new(ConcurrencyBurst),
        ArrivalPattern::Constant => Box::new(Constant::new(
            rate.expect("constant arrival requires a rate"),
        )),
        ArrivalPattern::Poisson => Box::new(Poisson::new(
            rate.expect("poisson arrival requires a rate"),
            seed,
        )),
        ArrivalPattern::Gamma => Box::new(GammaProcess::new(
            rate.expect("gamma arrival requires a rate"),
            smoothness.unwrap_or(1.0),
            seed,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean_ns(g: &mut dyn IntervalGenerator, n: usize) -> f64 {
        let sum: i128 = (0..n).map(|_| g.next_interval_ns() as i128).sum();
        sum as f64 / n as f64
    }

    #[test]
    fn constant_is_fixed_period() {
        let mut g = Constant::new(4.0); // 4 rps -> 250ms -> 250_000_000 ns
        assert_eq!(g.next_interval_ns(), 250_000_000);
        assert_eq!(g.next_interval_ns(), 250_000_000);
        g.set_rate(1000.0); // 1ms
        assert_eq!(g.next_interval_ns(), 1_000_000);
    }

    #[test]
    fn burst_is_always_zero_and_rate_is_a_noop() {
        let mut g = ConcurrencyBurst;
        assert_eq!(g.next_interval_ns(), 0);
        g.set_rate(999.0);
        assert_eq!(g.next_interval_ns(), 0);
        assert_eq!(g.rate(), 0.0);
    }

    #[test]
    fn poisson_mean_approximates_inverse_rate() {
        // rate=100 rps -> mean interval 10ms = 10_000_000 ns. Law of large numbers.
        let mut g = Poisson::new(100.0, 42);
        let mean = mean_ns(&mut g, 200_000);
        let expected = 10_000_000.0;
        assert!(
            (mean - expected).abs() / expected < 0.02,
            "poisson mean {mean} not within 2% of {expected}"
        );
    }

    #[test]
    fn gamma_mean_approximates_inverse_rate_regardless_of_smoothness() {
        for smoothness in [0.5, 1.0, 3.0] {
            let mut g = GammaProcess::new(50.0, smoothness, 7);
            let mean = mean_ns(&mut g, 200_000);
            let expected = 20_000_000.0; // 50 rps -> 20ms
            assert!(
                (mean - expected).abs() / expected < 0.02,
                "gamma(smoothness={smoothness}) mean {mean} not within 2% of {expected}"
            );
        }
    }

    #[test]
    fn poisson_is_deterministic_for_a_given_seed() {
        let mut a = Poisson::new(100.0, 123);
        let mut b = Poisson::new(100.0, 123);
        let seq_a: Vec<i64> = (0..1000).map(|_| a.next_interval_ns()).collect();
        let seq_b: Vec<i64> = (0..1000).map(|_| b.next_interval_ns()).collect();
        assert_eq!(
            seq_a, seq_b,
            "same seed must reproduce the interval sequence"
        );
    }

    #[test]
    fn factory_selects_the_right_generator() {
        assert_eq!(
            make_interval_generator(ArrivalPattern::ConcurrencyBurst, None, None, 0)
                .next_interval_ns(),
            0
        );
        assert_eq!(
            make_interval_generator(ArrivalPattern::Constant, Some(2.0), None, 0)
                .next_interval_ns(),
            500_000_000
        );
        // Gamma with default smoothness (1.0) behaves Poisson-like: positive, finite.
        let mut g = make_interval_generator(ArrivalPattern::Gamma, Some(10.0), None, 1);
        assert!(g.next_interval_ns() >= 0);
    }
}
