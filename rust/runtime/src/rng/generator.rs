// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Random-generator wrapper used by AIPerf workload components.
//!
//! One `rand_pcg::Pcg64` instance drives scalar and batch operations, while
//! wrapper code preserves bounds, replacement, and rejection-sampling behavior
//! that shapes benchmark workloads.

use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Exp, Gamma, Normal};
use rand_pcg::Pcg64;

use crate::rng::error::{Result, RngError};
use crate::rng::random_generator::{RandomGenerator, RuntimeRandomGenerator};

const NORMAL_REJECTION_LIMIT: usize = 10_000;
const U64_CARDINALITY: u128 = (u64::MAX as u128) + 1;
const I64_UPPER_EXCLUSIVE_AS_F64: f64 = 9_223_372_036_854_775_808.0;

/// One deterministic `Pcg64` PRNG plus AIPerf's sampling convenience methods.
///
/// A `Some(seed)` generator is bit-reproducible within this Rust implementation.
/// A `None` generator seeds from thread/OS entropy and records `None` until it is
/// explicitly reseeded.
#[derive(Clone, Debug)]
pub struct RustRandomGenerator {
    seed: Option<u64>,
    rng: Pcg64,
}

impl RustRandomGenerator {
    /// Construct a generator from a deterministic seed or entropy.
    pub fn from_seed(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(seed) => Pcg64::seed_from_u64(seed),
            None => {
                let mut entropy = rand::rng();
                Pcg64::from_rng(&mut entropy)
            }
        };
        Self { seed, rng }
    }

    /// Return the deterministic seed if this generator has one.
    pub const fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Replace the generator state with `seed`.
    pub fn reseed(&mut self, seed: u64) {
        self.seed = Some(seed);
        self.rng = Pcg64::seed_from_u64(seed);
    }

    /// Generate one uniformly distributed `u64`.
    pub fn random_u64(&mut self) -> u64 {
        self.rng.random()
    }

    /// Fill `dest` with random bytes.
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.rng.fill_bytes(dest);
    }

    /// Uniform float in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        self.rng.random()
    }

    /// Uniform integer from `range(start, stop, step)` semantics.
    pub fn randrange(&mut self, start: i64, stop: i64, step: i64) -> Result<i64> {
        if step == 0 {
            return Err(RngError::EmptyRange {
                what: "randrange step=0",
            });
        }

        // Widen before subtracting so the full i64 domain keeps range arithmetic
        // overflow-free in debug and release builds.
        let start = i128::from(start);
        let stop = i128::from(stop);
        let step = i128::from(step);
        let width = stop - start;
        let n = if step > 0 {
            if width <= 0 {
                0
            } else {
                ((width - 1) / step) + 1
            }
        } else if width >= 0 {
            0
        } else {
            ((width + 1) / step) + 1
        };

        if n <= 0 {
            return Err(RngError::EmptyRange { what: "randrange" });
        }
        let idx = self.uniform_index(n as u128) as i128;
        let sampled = start + idx * step;
        Ok(sampled
            .try_into()
            .expect("a member of an i64 range must remain representable"))
    }

    /// Uniform integer from `[0, stop)`.
    pub fn randbelow(&mut self, stop: i64) -> Result<i64> {
        self.randrange(0, stop, 1)
    }

    /// Uniform integer from `[lo, hi)`.
    pub fn randrange_u64(&mut self, lo: u64, hi: u64) -> Result<u64> {
        if lo >= hi {
            return Err(RngError::EmptyRange {
                what: "randrange_u64",
            });
        }
        Ok(self.rng.random_range(lo..hi))
    }

    /// Uniform integer `N` such that `a <= N <= b`.
    pub fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        if a > b {
            return Err(RngError::EmptyRange { what: "randint" });
        }
        let width = (i128::from(b) - i128::from(a) + 1) as u128;
        let sampled = i128::from(a) + self.uniform_index(width) as i128;
        Ok(sampled
            .try_into()
            .expect("an inclusive i64 sample must remain representable"))
    }

    /// Uniform float in `[a, b)` or `[b, a)` when `b < a`.
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// Select one element uniformly from a non-empty slice.
    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> Result<&'a T> {
        if seq.is_empty() {
            return Err(RngError::EmptySequence { what: "choice" });
        }
        let idx = self.rng.random_range(0..seq.len());
        Ok(&seq[idx])
    }

    /// Select `k` elements uniformly with replacement.
    pub fn choices<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        if population.is_empty() && k > 0 {
            return Err(RngError::EmptySequence { what: "choices" });
        }
        (0..k).map(|_| self.choice(population).cloned()).collect()
    }

    /// Select `k` unique elements uniformly without replacement.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        if k > population.len() {
            return Err(RngError::SampleTooLarge {
                k,
                len: population.len(),
            });
        }
        let mut values = population.to_vec();
        self.shuffle(&mut values);
        values.truncate(k);
        Ok(values)
    }

    /// Shuffle a slice in place with Fisher-Yates.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.rng.random_range(0..=i);
            values.swap(i, j);
        }
    }

    /// Select one element, uniformly when `weights` is `None` or by cumulative weights.
    pub fn weighted_choice<T: Clone>(
        &mut self,
        values: &[T],
        weights: Option<&[f64]>,
    ) -> Result<T> {
        let Some(weights) = weights else {
            return self.choice(values).cloned();
        };
        let idx = self.weighted_index(values.len(), weights)?;
        Ok(values[idx].clone())
    }

    /// NumPy-style `choice` over a slice, with optional weights and replacement.
    pub fn numpy_choice<T: Clone>(
        &mut self,
        values: &[T],
        size: usize,
        weights: Option<&[f64]>,
        replace: bool,
    ) -> Result<Vec<T>> {
        if values.is_empty() && size > 0 {
            return Err(RngError::EmptySequence {
                what: "numpy_choice",
            });
        }
        if !replace && size > values.len() {
            return Err(RngError::SampleTooLarge {
                k: size,
                len: values.len(),
            });
        }
        if let Some(weights) = weights {
            validated_weight_total(values.len(), weights)?;
            if !replace && weights.iter().filter(|weight| **weight > 0.0).count() < size {
                return Err(RngError::InvalidWeights {
                    reason: "fewer positive weights than requested samples",
                });
            }
        }

        if replace {
            let Some(weights) = weights else {
                return (0..size)
                    .map(|_| self.weighted_choice(values, None))
                    .collect();
            };
            // The replace path draws `size` samples from an unchanging, already
            // validated weight vector; caching the cumulative sums once keeps
            // sampling O(n + size*log n) instead of the O(n*size) that per-draw
            // `weighted_choice` (re-summing + re-validating each draw) incurs.
            let cumulative = cumulative_weights(weights);
            return Ok((0..size)
                .map(|_| values[self.weighted_index_cached(&cumulative)].clone())
                .collect());
        }
        if weights.is_none() {
            return self.sample(values, size);
        }

        let mut pool = values.to_vec();
        let mut pool_weights = weights.expect("checked above").to_vec();
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            let idx = self
                .weighted_index(pool.len(), &pool_weights)
                .expect("validated positive weights remain sampleable");
            out.push(pool.remove(idx));
            pool_weights.remove(idx);
        }
        Ok(out)
    }

    /// Exponential distribution with rate `lambda` and mean `1/lambda`.
    pub fn expovariate(&mut self, lambd: f64) -> Result<f64> {
        if lambd <= 0.0 || !lambd.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "lambda",
                value: lambd,
            });
        }
        Ok(Exp::new(lambd)
            .expect("validated lambda")
            .sample(&mut self.rng))
    }

    /// Gamma distribution with shape `alpha`, scale `beta`, and mean `alpha * beta`.
    pub fn gammavariate(&mut self, alpha: f64, beta: f64) -> Result<f64> {
        if alpha <= 0.0 || !alpha.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "alpha",
                value: alpha,
            });
        }
        if beta <= 0.0 || !beta.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "beta",
                value: beta,
            });
        }
        Ok(Gamma::new(alpha, beta)
            .expect("validated gamma parameters")
            .sample(&mut self.rng))
    }

    /// Normal distribution with mean `loc` and standard deviation `scale`.
    pub fn normal(&mut self, loc: f64, scale: f64) -> Result<f64> {
        if scale < 0.0 || !scale.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "scale",
                value: scale,
            });
        }
        if !loc.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "loc",
                value: loc,
            });
        }
        if scale == 0.0 {
            return Ok(loc);
        }
        Ok(Normal::new(loc, scale)
            .expect("validated normal parameters")
            .sample(&mut self.rng))
    }

    /// Sample a bounded normal using the configured rejection cap and clamp fallback.
    pub fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64> {
        if lower.is_nan() {
            return Err(RngError::InvalidParameter {
                what: "lower",
                value: lower,
            });
        }
        if upper.is_nan() {
            return Err(RngError::InvalidParameter {
                what: "upper",
                value: upper,
            });
        }
        if lower > upper {
            return Err(RngError::InvalidBounds { lower, upper });
        }
        if stddev < 0.0 || !stddev.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "stddev",
                value: stddev,
            });
        }
        if !mean.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "mean",
                value: mean,
            });
        }
        for _ in 0..NORMAL_REJECTION_LIMIT {
            let n = self
                .normal(mean, stddev)
                .expect("bounded-normal parameters were validated");
            if lower <= n && n <= upper {
                return Ok(n);
            }
        }
        let fallback = mean.clamp(lower, upper);
        tracing::warn!(
            mean,
            stddev,
            lower,
            upper,
            attempts = NORMAL_REJECTION_LIMIT,
            fallback,
            "bounded normal rejection limit exhausted; using clamped mean"
        );
        Ok(fallback)
    }

    /// Sample a normal value truncated at zero.
    pub fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> Result<f64> {
        if mean < 0.0 {
            return Err(RngError::InvalidParameter {
                what: "mean",
                value: mean,
            });
        }
        self.sample_normal(mean, stddev, 0.0, f64::INFINITY)
    }

    /// Sample a positive integer from a positive normal distribution.
    pub fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> Result<i64> {
        if !mean.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "mean",
                value: mean,
            });
        }
        if !stddev.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "stddev",
                value: stddev,
            });
        }
        if stddev <= 0.0 {
            return positive_integer_from_f64(mean.round_ties_even(), "rounded mean");
        }
        positive_integer_from_f64(
            self.sample_positive_normal(mean, stddev)?.ceil(),
            "normal integer sample",
        )
    }

    /// Generate `size` integers using NumPy's `[low, high)` calling convention.
    pub fn integers(&mut self, low: i64, high: Option<i64>, size: usize) -> Result<Vec<i64>> {
        let (lo, hi) = match high {
            Some(high) => (low, high),
            None => (0, low),
        };
        if lo >= hi {
            return Err(RngError::EmptyRange { what: "integers" });
        }
        (0..size).map(|_| self.randrange(lo, hi, 1)).collect()
    }

    /// Generate `size` normal samples.
    pub fn normal_batch(&mut self, loc: f64, scale: f64, size: usize) -> Result<Vec<f64>> {
        if scale < 0.0 || !scale.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "scale",
                value: scale,
            });
        }
        if !loc.is_finite() {
            return Err(RngError::InvalidParameter {
                what: "loc",
                value: loc,
            });
        }
        if scale == 0.0 {
            return Ok(vec![loc; size]);
        }
        let dist = Normal::new(loc, scale).expect("validated normal parameters");
        Ok((0..size).map(|_| dist.sample(&mut self.rng)).collect())
    }

    /// Generate `size` uniform floats in `[0, 1)`.
    pub fn random_batch(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.random()).collect()
    }

    fn weighted_index(&mut self, value_len: usize, weights: &[f64]) -> Result<usize> {
        let total = validated_weight_total(value_len, weights)?;
        let r = self.random() * total;
        Ok(cumulative_weight_index(weights, r))
    }

    /// Sample one index against precomputed cumulative weights, consuming a
    /// single uniform draw. `cumulative` must be the running sums produced by
    /// [`cumulative_weights`] over already-validated weights; its last element
    /// is the total. This uses `weighted_index`'s draw and boundary
    /// selected index) while skipping that method's per-draw O(n) validation and
    /// re-accumulation, so repeated draws over a fixed weight vector stay cheap.
    fn weighted_index_cached(&mut self, cumulative: &[f64]) -> usize {
        let total = *cumulative
            .last()
            .expect("cumulative weights are validated non-empty");
        let r = self.random() * total;
        // First index whose running total exceeds `r`, matching
        // `cumulative_weight_index`'s strict `r < cumulative` boundary; falls
        // back to the last index when floating-point drift leaves `r >= total`.
        cumulative
            .partition_point(|c| *c <= r)
            .min(cumulative.len() - 1)
    }

    fn uniform_index(&mut self, len: u128) -> u128 {
        debug_assert!((1..=U64_CARDINALITY).contains(&len));
        if len == U64_CARDINALITY {
            u128::from(self.random_u64())
        } else {
            u128::from(self.rng.random_range(0..len as u64))
        }
    }
}

impl RandomGenerator for RustRandomGenerator {
    fn random(&mut self) -> f64 {
        Self::random(self)
    }

    fn choice<'a, T>(&mut self, seq: &'a [T]) -> Result<&'a T> {
        Self::choice(self, seq)
    }

    fn randrange(&mut self, stop: i64) -> Result<i64> {
        Self::randbelow(self, stop)
    }

    fn randrange_step(&mut self, start: i64, stop: i64, step: i64) -> Result<i64> {
        Self::randrange(self, start, stop, step)
    }

    fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        Self::randint(self, a, b)
    }

    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        Self::uniform(self, a, b)
    }

    fn choices<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        Self::choices(self, population, k)
    }

    fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        Self::sample(self, population, k)
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        Self::shuffle(self, values);
    }

    fn random_batch(&mut self, size: usize) -> Vec<f64> {
        Self::random_batch(self, size)
    }

    fn integers(&mut self, low: i64, high: i64) -> i64 {
        Self::integers(self, low, Some(high), 1)
            .expect("RandomGenerator::integers requires a valid bounded range")[0]
    }

    fn integers_batch(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        Self::integers(self, low, Some(high), size)
            .expect("RandomGenerator::integers_batch requires a valid bounded range")
    }

    fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        Self::normal(self, loc, scale).expect("RandomGenerator::normal requires valid parameters")
    }

    fn normal_batch(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        Self::normal_batch(self, loc, scale, size)
            .expect("RandomGenerator::normal_batch requires valid parameters")
    }

    fn numpy_choice_uniform(&mut self, pop_size: i64, size: usize) -> Vec<i64> {
        Self::integers(self, 0, Some(pop_size), size)
            .expect("RandomGenerator::numpy_choice_uniform requires a valid population size")
    }

    fn numpy_choice_weighted(&mut self, weights: &[f64]) -> usize {
        self.weighted_index(weights.len(), weights)
            .expect("RandomGenerator::numpy_choice_weighted requires valid weights")
    }

    fn numpy_choice_weighted_batch(&mut self, weights: &[f64], size: usize) -> Vec<usize> {
        validated_weight_total(weights.len(), weights)
            .expect("RandomGenerator::numpy_choice_weighted_batch requires valid weights");
        let cumulative = cumulative_weights(weights);
        (0..size)
            .map(|_| self.weighted_index_cached(&cumulative))
            .collect()
    }

    fn numpy_choice_weighted_without_replacement(
        &mut self,
        weights: &[f64],
        size: usize,
    ) -> Result<Vec<usize>> {
        let values: Vec<_> = (0..weights.len()).collect();
        Self::numpy_choice(self, &values, size, Some(weights), false)
    }

    fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64> {
        Self::sample_normal(self, mean, stddev, lower, upper)
    }

    fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> Result<f64> {
        Self::sample_positive_normal(self, mean, stddev)
    }

    fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> Result<i64> {
        Self::sample_positive_normal_integer(self, mean, stddev)
    }

    fn expovariate(&mut self, lambd: f64) -> f64 {
        Self::expovariate(self, lambd)
            .expect("RandomGenerator::expovariate requires a finite positive rate")
    }

    fn gammavariate(&mut self, alpha: f64, beta: f64) -> Result<f64> {
        Self::gammavariate(self, alpha, beta)
    }
}

impl RuntimeRandomGenerator for RustRandomGenerator {
    fn seed(&self) -> Option<u64> {
        Self::seed(self)
    }

    fn reseed(&mut self, seed: u64) {
        Self::reseed(self, seed);
    }

    fn random_u64(&mut self) -> u64 {
        Self::random_u64(self)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        Self::fill_bytes(self, dest);
    }

    fn randrange_u64(&mut self, lo: u64, hi: u64) -> Result<u64> {
        Self::randrange_u64(self, lo, hi)
    }

    fn weighted_choice<T: Clone>(&mut self, values: &[T], weights: Option<&[f64]>) -> Result<T> {
        Self::weighted_choice(self, values, weights)
    }

    fn normal_checked(&mut self, loc: f64, scale: f64) -> Result<f64> {
        Self::normal(self, loc, scale)
    }

    fn normal_batch_checked(&mut self, loc: f64, scale: f64, size: usize) -> Result<Vec<f64>> {
        Self::normal_batch(self, loc, scale, size)
    }

    fn integers_checked(&mut self, low: i64, high: Option<i64>, size: usize) -> Result<Vec<i64>> {
        Self::integers(self, low, high, size)
    }
}

fn cumulative_weight_index(weights: &[f64], r: f64) -> usize {
    let mut cumulative = 0.0;
    for (idx, weight) in weights.iter().enumerate() {
        cumulative += *weight;
        if r < cumulative {
            return idx;
        }
    }
    weights.len() - 1
}

/// Left-to-right running sums of `weights`; the final element is the total.
/// The accumulation order matches `cumulative_weight_index` and
/// `validated_weight_total` so cached sampling stays bit-identical to the
/// per-draw path.
fn cumulative_weights(weights: &[f64]) -> Vec<f64> {
    let mut running = 0.0;
    weights
        .iter()
        .map(|weight| {
            running += *weight;
            running
        })
        .collect()
}

pub(crate) fn positive_integer_from_f64(value: f64, what: &'static str) -> Result<i64> {
    if !value.is_finite() || value >= I64_UPPER_EXCLUSIVE_AS_F64 {
        return Err(RngError::InvalidParameter { what, value });
    }
    Ok(i64::max(1, value as i64))
}

fn validated_weight_total(value_len: usize, weights: &[f64]) -> Result<f64> {
    if weights.len() != value_len {
        return Err(RngError::InvalidWeights {
            reason: "weights length must match values length",
        });
    }
    if weights.is_empty() {
        return Err(RngError::InvalidWeights {
            reason: "weights cannot be empty",
        });
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(RngError::InvalidWeights {
            reason: "weights must be finite and non-negative",
        });
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() {
        return Err(RngError::InvalidWeights {
            reason: "weights must have a finite sum",
        });
    }
    if total <= 0.0 {
        return Err(RngError::InvalidWeights {
            reason: "weights must sum to a positive value",
        });
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn variance(values: &[f64]) -> f64 {
        let mean = mean(values);
        values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64
    }

    #[test]
    fn same_seed_reproduces_draw_sequence() {
        let mut a = RustRandomGenerator::from_seed(Some(42));
        let mut b = RustRandomGenerator::from_seed(Some(42));
        let seq_a: Vec<_> = (0..100).map(|_| a.random_u64()).collect();
        let seq_b: Vec<_> = (0..100).map(|_| b.random_u64()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn different_seeds_change_draw_sequence() {
        let mut a = RustRandomGenerator::from_seed(Some(1));
        let mut b = RustRandomGenerator::from_seed(Some(2));
        let seq_a: Vec<_> = (0..20).map(|_| a.random_u64()).collect();
        let seq_b: Vec<_> = (0..20).map(|_| b.random_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn clone_fill_bytes_entropy_and_reseed_obey_state_contracts() {
        let mut original = RustRandomGenerator::from_seed(Some(123));
        let _ = original.random_u64();
        let mut cloned = original.clone();
        let mut original_bytes = [0_u8; 37];
        let mut cloned_bytes = [0_u8; 37];
        original.fill_bytes(&mut original_bytes);
        cloned.fill_bytes(&mut cloned_bytes);
        assert_eq!(original_bytes, cloned_bytes);
        assert!(original_bytes.iter().any(|byte| *byte != 0));

        original.reseed(9);
        let mut fresh = RustRandomGenerator::from_seed(Some(9));
        assert_eq!(original.seed(), Some(9));
        assert_eq!(original.random_u64(), fresh.random_u64());

        let mut entropy_a = RustRandomGenerator::from_seed(None);
        let mut entropy_b = RustRandomGenerator::from_seed(None);
        assert_eq!(entropy_a.seed(), None);
        assert_eq!(entropy_b.seed(), None);
        assert_ne!(
            [entropy_a.random_u64(), entropy_a.random_u64()],
            [entropy_b.random_u64(), entropy_b.random_u64()]
        );
    }

    #[test]
    fn integer_ranges_follow_bound_conventions() {
        let mut rng = RustRandomGenerator::from_seed(Some(7));
        for _ in 0..1000 {
            let x = rng.randrange(2, 10, 2).unwrap();
            assert!([2, 4, 6, 8].contains(&x));
            let y = rng.randrange(10, 2, -3).unwrap();
            assert!([10, 7, 4].contains(&y));
            let z = rng.randint(1, 3).unwrap();
            assert!((1..=3).contains(&z));
        }
        assert!(rng.randrange(1, 1, 1).is_err());
        assert!(rng.randrange(1, 3, 0).is_err());
        assert!(rng.randint(3, 1).is_err());
    }

    #[test]
    fn integer_ranges_cover_extreme_i64_and_u64_bounds_without_overflow() {
        let mut rng = RustRandomGenerator::from_seed(Some(70));
        for _ in 0..1_000 {
            let full_half_open = rng.randrange(i64::MIN, i64::MAX, 1).unwrap();
            assert!(full_half_open < i64::MAX);
            let full_inclusive = rng.randint(i64::MIN, i64::MAX).unwrap();
            assert!((i64::MIN..=i64::MAX).contains(&full_inclusive));
            assert!(
                [i64::MIN, -1, i64::MAX - 1]
                    .contains(&rng.randrange(i64::MIN, i64::MAX, i64::MAX).unwrap())
            );
            assert!([i64::MAX, -1].contains(&rng.randrange(i64::MAX, i64::MIN, i64::MIN).unwrap()));
        }
        assert_eq!(rng.randbelow(1).unwrap(), 0);
        assert!(rng.randbelow(0).is_err());
        assert_eq!(
            rng.randrange_u64(u64::MAX - 1, u64::MAX).unwrap(),
            u64::MAX - 1
        );
        assert!(rng.randrange_u64(1, 1).is_err());
        assert!(rng.randrange_u64(2, 1).is_err());
        assert!(rng.randrange(0, 10, -1).is_err());
        assert!(rng.randrange(10, 0, 1).is_err());
    }

    #[test]
    fn uniform_choice_sample_and_shuffle_cover_boundary_shapes() {
        let mut rng = RustRandomGenerator::from_seed(Some(71));
        assert_eq!(rng.uniform(4.0, 4.0), 4.0);
        for _ in 0..100 {
            assert!((0.0..1.0).contains(&rng.random()));
            assert!((-3.0..=2.0).contains(&rng.uniform(2.0, -3.0)));
            assert!([10, 20, 30].contains(rng.choice(&[10, 20, 30]).unwrap()));
        }

        assert_eq!(rng.choices(&[1, 2], 0).unwrap(), Vec::<i32>::new());
        assert!(
            rng.choices(&[1, 2], 20)
                .unwrap()
                .iter()
                .all(|x| [1, 2].contains(x))
        );
        assert_eq!(rng.sample(&[1, 2], 0).unwrap(), Vec::<i32>::new());
        let mut all = rng.sample(&[1, 2, 3], 3).unwrap();
        all.sort_unstable();
        assert_eq!(all, vec![1, 2, 3]);

        let mut empty: [i32; 0] = [];
        let mut singleton = [1];
        rng.shuffle(&mut empty);
        rng.shuffle(&mut singleton);
        assert_eq!(singleton, [1]);

        let mut first = [1, 2, 3, 4, 5, 6];
        let mut second = first;
        let mut a = RustRandomGenerator::from_seed(Some(72));
        let mut b = RustRandomGenerator::from_seed(Some(72));
        a.shuffle(&mut first);
        b.shuffle(&mut second);
        assert_eq!(first, second);
        assert_ne!(first, [1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn choice_and_sampling_validate_empty_or_oversized_inputs() {
        let mut rng = RustRandomGenerator::from_seed(Some(1));
        assert!(rng.choice::<i32>(&[]).is_err());
        assert!(rng.choices::<i32>(&[], 1).is_err());
        assert_eq!(rng.choices::<i32>(&[], 0).unwrap(), Vec::<i32>::new());
        assert!(rng.sample(&[1, 2], 3).is_err());
        let sample = rng.sample(&[1, 2, 3, 4], 3).unwrap();
        assert_eq!(sample.len(), 3);
        let mut sorted = sample;
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn weighted_choice_respects_validation_and_dominant_weight() {
        let mut rng = RustRandomGenerator::from_seed(Some(3));
        assert!(rng.weighted_choice(&[1, 2], Some(&[1.0])).is_err());
        assert!(rng.weighted_choice(&[1, 2], Some(&[0.0, 0.0])).is_err());
        assert!(
            rng.weighted_choice(&[1, 2], Some(&[1.0, f64::NAN]))
                .is_err()
        );
        for _ in 0..100 {
            assert_eq!(rng.weighted_choice(&[1, 2], Some(&[0.0, 5.0])).unwrap(), 2);
        }
        assert!(rng.weighted_choice::<i32>(&[], None).is_err());
        assert!(rng.weighted_choice::<i32>(&[], Some(&[])).is_err());
        assert!(rng.weighted_choice(&[1, 2], Some(&[-1.0, 2.0])).is_err());
        assert!(
            rng.weighted_choice(&[1, 2], Some(&[1.0, f64::INFINITY]))
                .is_err()
        );
        assert!(
            rng.weighted_choice(&[1, 2], Some(&[f64::MAX, f64::MAX]))
                .is_err()
        );

        let draws: Vec<_> = (0..40_000)
            .map(|_| rng.weighted_choice(&[0, 1], Some(&[1.0, 3.0])).unwrap())
            .collect();
        let fraction_one =
            draws.iter().filter(|value| **value == 1).count() as f64 / draws.len() as f64;
        assert!((fraction_one - 0.75).abs() < 0.02, "{fraction_one}");
        assert_eq!(cumulative_weight_index(&[1.0, 3.0], 0.0), 0);
        assert_eq!(cumulative_weight_index(&[1.0, 3.0], 1.0), 1);
        assert_eq!(cumulative_weight_index(&[1.0, 3.0], 4.0), 1);
    }

    #[test]
    fn numpy_choice_without_replacement_handles_weights() {
        let mut rng = RustRandomGenerator::from_seed(Some(4));
        let picked = rng
            .numpy_choice(&[10, 20, 30], 2, Some(&[0.0, 1.0, 1.0]), false)
            .unwrap();
        assert_eq!(picked.len(), 2);
        assert!(!picked.contains(&10));
        assert_ne!(picked[0], picked[1]);

        assert!(rng.numpy_choice::<i32>(&[], 1, None, true).is_err());
        assert!(rng.numpy_choice(&[1, 2], 3, None, false).is_err());
        assert!(rng.numpy_choice(&[1, 2], 1, Some(&[1.0]), true).is_err());
        assert!(
            rng.numpy_choice(&[1, 2], 2, Some(&[1.0, 0.0]), false)
                .is_err()
        );
        assert_eq!(
            rng.numpy_choice::<i32>(&[], 0, None, true).unwrap(),
            Vec::<i32>::new()
        );
        assert_eq!(rng.numpy_choice(&[9], 5, None, true).unwrap(), vec![9; 5]);
        let unweighted = rng.numpy_choice(&[1, 2, 3], 2, None, false).unwrap();
        assert_eq!(unweighted.len(), 2);
        assert_ne!(unweighted[0], unweighted[1]);
    }

    #[test]
    fn bounded_normal_validates_bounds_and_clamps_degenerate_case() {
        let mut rng = RustRandomGenerator::from_seed(Some(5));
        assert!(rng.sample_normal(0.0, 1.0, 2.0, 1.0).is_err());
        assert_eq!(rng.sample_normal(10.0, 0.0, 0.0, 5.0).unwrap(), 5.0);
        let n = rng.sample_normal(10.0, 2.0, 8.0, 12.0).unwrap();
        assert!((8.0..=12.0).contains(&n));
    }

    #[test]
    fn bounded_normal_rejects_non_finite_parameters_and_pins_rejection_limit() {
        let mut rng = RustRandomGenerator::from_seed(Some(51));
        assert!(rng.sample_normal(0.0, 1.0, f64::NAN, 1.0).is_err());
        assert!(rng.sample_normal(0.0, 1.0, 0.0, f64::NAN).is_err());
        assert!(rng.sample_normal(0.0, -1.0, -1.0, 1.0).is_err());
        assert!(rng.sample_normal(0.0, f64::NAN, -1.0, 1.0).is_err());
        assert!(rng.sample_normal(f64::NAN, 1.0, -1.0, 1.0).is_err());
        assert!(rng.sample_normal(f64::INFINITY, 1.0, -1.0, 1.0).is_err());

        let mut actual = RustRandomGenerator::from_seed(Some(52));
        assert_eq!(
            actual.sample_normal(0.0, 1.0, 1_000.0, 1_001.0).unwrap(),
            1_000.0
        );
        let actual_next = actual.random_u64();

        let mut expected = RustRandomGenerator::from_seed(Some(52));
        for _ in 0..NORMAL_REJECTION_LIMIT {
            let rejected = expected.normal(0.0, 1.0).unwrap();
            assert!(rejected < 1_000.0);
        }
        assert_eq!(actual_next, expected.random_u64());

        let mut zero_scale = RustRandomGenerator::from_seed(Some(53));
        let mut untouched = zero_scale.clone();
        assert_eq!(
            zero_scale
                .sample_normal(5.0, 0.0, f64::NEG_INFINITY, f64::INFINITY)
                .unwrap(),
            5.0
        );
        assert_eq!(zero_scale.random_u64(), untouched.random_u64());
    }

    #[test]
    fn positive_normal_integer_preserves_shortcut_semantics() {
        let mut rng = RustRandomGenerator::from_seed(Some(6));
        assert_eq!(rng.sample_positive_normal_integer(0.1, 0.0).unwrap(), 1);
        assert_eq!(rng.sample_positive_normal_integer(2.5, 0.0).unwrap(), 2);
        assert_eq!(rng.sample_positive_normal_integer(3.5, 0.0).unwrap(), 4);
        for _ in 0..100 {
            assert!(rng.sample_positive_normal_integer(10.0, 2.0).unwrap() >= 1);
        }
        assert_eq!(rng.sample_positive_normal_integer(-10.0, -1.0).unwrap(), 1);
        assert!(rng.sample_positive_normal(-1.0, 1.0).is_err());
        assert!(rng.sample_positive_normal_integer(-1.0, 1.0).is_err());
        assert!(rng.sample_positive_normal_integer(f64::NAN, 0.0).is_err());
        assert!(rng.sample_positive_normal_integer(1.0, f64::NAN).is_err());
        assert!(
            rng.sample_positive_normal_integer(I64_UPPER_EXCLUSIVE_AS_F64, 0.0)
                .is_err()
        );
    }

    #[test]
    fn exponential_and_gamma_sample_means_match_parameterization() {
        let mut rng = RustRandomGenerator::from_seed(Some(8));
        let exp: Vec<_> = (0..200_000)
            .map(|_| rng.expovariate(4.0).unwrap())
            .collect();
        let exp_mean = mean(&exp);
        assert!((exp_mean - 0.25).abs() / 0.25 < 0.02, "{exp_mean}");
        let exp_variance = variance(&exp);
        assert!(
            (exp_variance - 0.0625).abs() / 0.0625 < 0.04,
            "{exp_variance}"
        );

        let gamma: Vec<_> = (0..200_000)
            .map(|_| rng.gammavariate(2.0, 3.0).unwrap())
            .collect();
        let gamma_mean = mean(&gamma);
        assert!((gamma_mean - 6.0).abs() / 6.0 < 0.02, "{gamma_mean}");
        let gamma_variance = variance(&gamma);
        assert!(
            (gamma_variance - 18.0).abs() / 18.0 < 0.04,
            "{gamma_variance}"
        );

        let rate = 20.0;
        let smoothness = 3.0;
        let intervals: Vec<_> = (0..200_000)
            .map(|_| {
                rng.gammavariate(smoothness, 1.0 / (rate * smoothness))
                    .unwrap()
            })
            .collect();
        assert!((mean(&intervals) - 1.0 / rate).abs() / (1.0 / rate) < 0.02);
    }

    #[test]
    fn continuous_distributions_validate_parameters_and_normal_moments() {
        let mut rng = RustRandomGenerator::from_seed(Some(81));
        for lambda in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(rng.expovariate(lambda).is_err());
        }
        for alpha in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(rng.gammavariate(alpha, 1.0).is_err());
        }
        for beta in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(rng.gammavariate(1.0, beta).is_err());
        }
        for scale in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(rng.normal(0.0, scale).is_err());
        }
        for loc in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(rng.normal(loc, 1.0).is_err());
        }
        assert_eq!(rng.normal(3.0, 0.0).unwrap(), 3.0);

        let normal: Vec<_> = (0..200_000)
            .map(|_| rng.normal(4.0, 2.0).unwrap())
            .collect();
        assert!((mean(&normal) - 4.0).abs() < 0.02);
        assert!((variance(&normal) - 4.0).abs() / 4.0 < 0.03);
    }

    #[test]
    fn batch_methods_follow_scalar_bounds() {
        let mut rng = RustRandomGenerator::from_seed(Some(9));
        let ints = rng.integers(3, Some(6), 100).unwrap();
        assert!(ints.iter().all(|x| (3..6).contains(x)));
        let from_zero = rng.integers(4, None, 100).unwrap();
        assert!(from_zero.iter().all(|x| (0..4).contains(x)));
        assert!(rng.integers(4, Some(4), 1).is_err());
        assert_eq!(rng.normal_batch(3.0, 0.0, 5).unwrap(), vec![3.0; 5]);
        let uniforms = rng.random_batch(7);
        assert_eq!(uniforms.len(), 7);
        assert!(uniforms.iter().all(|value| (0.0..1.0).contains(value)));
        assert!(rng.random_batch(0).is_empty());
        assert!(rng.integers(4, None, 0).unwrap().is_empty());
        assert!(rng.integers(5, Some(4), 0).is_err());

        for scale in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(rng.normal_batch(0.0, scale, 1).is_err());
        }
        assert!(rng.normal_batch(f64::NAN, 1.0, 1).is_err());
        assert!(rng.normal_batch(f64::INFINITY, 1.0, 1).is_err());
        assert!(rng.normal_batch(0.0, 1.0, 0).unwrap().is_empty());
        let normals = rng.normal_batch(7.0, 1.5, 100_000).unwrap();
        assert!((mean(&normals) - 7.0).abs() < 0.03);
        assert!((variance(&normals) - 2.25).abs() / 2.25 < 0.04);

        let extremes = rng.integers(i64::MIN, Some(i64::MAX), 1_000).unwrap();
        assert!(extremes.iter().all(|value| *value < i64::MAX));
    }
}
