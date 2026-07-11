// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust-native random generator wrapper used by AIPerf workload components.
//!
//! The generator intentionally preserves Python AIPerf's *method semantics* rather
//! than Python's exact byte streams. One `rand_pcg::Pcg64` instance drives scalar
//! and batch operations, while wrapper code preserves bounds, replacement, and
//! rejection-sampling behavior that shapes benchmark workloads.

use rand::{Rng, RngCore, SeedableRng};
use rand_distr::{Distribution, Exp, Gamma, Normal};
use rand_pcg::Pcg64;

use crate::error::{Result, RngError};

const NORMAL_REJECTION_LIMIT: usize = 10_000;

/// One deterministic `Pcg64` PRNG plus AIPerf's sampling convenience methods.
///
/// A `Some(seed)` generator is bit-reproducible within this Rust implementation.
/// A `None` generator seeds from thread/OS entropy and records `None` until it is
/// explicitly reseeded.
#[derive(Clone, Debug)]
pub struct RandomGenerator {
    seed: Option<u64>,
    rng: Pcg64,
}

impl RandomGenerator {
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
        let idx = self.rng.random_range(0..n);
        Ok(start + idx * step)
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
        Ok(self.rng.random_range(a..=b))
    }

    /// Uniform float in `[a, b)` or `[b, a)` when `b < a`, matching Python's formula.
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
            validate_weights(values.len(), weights)?;
        }

        if replace {
            return (0..size)
                .map(|_| self.weighted_choice(values, weights))
                .collect();
        }
        if weights.is_none() {
            return self.sample(values, size);
        }

        let mut pool = values.to_vec();
        let mut pool_weights = weights.expect("checked above").to_vec();
        let mut out = Vec::with_capacity(size);
        for _ in 0..size {
            let idx = self.weighted_index(pool.len(), &pool_weights)?;
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

    /// Sample a bounded normal using Python AIPerf's rejection cap and clamp fallback.
    pub fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64> {
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
        if stddev == 0.0 {
            return Ok(mean.clamp(lower, upper));
        }
        for _ in 0..NORMAL_REJECTION_LIMIT {
            let n = self.normal(mean, stddev)?;
            if lower <= n && n <= upper {
                return Ok(n);
            }
        }
        Ok(mean.clamp(lower, upper))
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
        if stddev <= 0.0 {
            return Ok(i64::max(1, mean.round_ties_even() as i64));
        }
        Ok(i64::max(
            1,
            self.sample_positive_normal(mean, stddev)?.ceil() as i64,
        ))
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
        Ok((0..size).map(|_| self.rng.random_range(lo..hi)).collect())
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
        validate_weights(value_len, weights)?;
        let total: f64 = weights.iter().sum();
        let r = self.random() * total;
        let mut cumulative = 0.0;
        for (idx, weight) in weights.iter().enumerate() {
            cumulative += *weight;
            if r < cumulative {
                return Ok(idx);
            }
        }
        Ok(weights.len() - 1)
    }
}

fn validate_weights(value_len: usize, weights: &[f64]) -> Result<()> {
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
    if total <= 0.0 {
        return Err(RngError::InvalidWeights {
            reason: "weights must sum to a positive value",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    #[test]
    fn same_seed_reproduces_draw_sequence() {
        let mut a = RandomGenerator::from_seed(Some(42));
        let mut b = RandomGenerator::from_seed(Some(42));
        let seq_a: Vec<_> = (0..100).map(|_| a.random_u64()).collect();
        let seq_b: Vec<_> = (0..100).map(|_| b.random_u64()).collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn different_seeds_change_draw_sequence() {
        let mut a = RandomGenerator::from_seed(Some(1));
        let mut b = RandomGenerator::from_seed(Some(2));
        let seq_a: Vec<_> = (0..20).map(|_| a.random_u64()).collect();
        let seq_b: Vec<_> = (0..20).map(|_| b.random_u64()).collect();
        assert_ne!(seq_a, seq_b);
    }

    #[test]
    fn integer_ranges_match_python_bound_conventions() {
        let mut rng = RandomGenerator::from_seed(Some(7));
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
    fn choice_and_sampling_validate_empty_or_oversized_inputs() {
        let mut rng = RandomGenerator::from_seed(Some(1));
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
        let mut rng = RandomGenerator::from_seed(Some(3));
        assert!(rng.weighted_choice(&[1, 2], Some(&[1.0])).is_err());
        assert!(rng.weighted_choice(&[1, 2], Some(&[0.0, 0.0])).is_err());
        assert!(
            rng.weighted_choice(&[1, 2], Some(&[1.0, f64::NAN]))
                .is_err()
        );
        for _ in 0..100 {
            assert_eq!(rng.weighted_choice(&[1, 2], Some(&[0.0, 5.0])).unwrap(), 2);
        }
    }

    #[test]
    fn numpy_choice_without_replacement_handles_weights() {
        let mut rng = RandomGenerator::from_seed(Some(4));
        let picked = rng
            .numpy_choice(&['a', 'b', 'c'], 2, Some(&[0.0, 1.0, 1.0]), false)
            .unwrap();
        assert_eq!(picked.len(), 2);
        assert!(!picked.contains(&'a'));
        assert_ne!(picked[0], picked[1]);
    }

    #[test]
    fn bounded_normal_validates_bounds_and_clamps_degenerate_case() {
        let mut rng = RandomGenerator::from_seed(Some(5));
        assert!(rng.sample_normal(0.0, 1.0, 2.0, 1.0).is_err());
        assert_eq!(rng.sample_normal(10.0, 0.0, 0.0, 5.0).unwrap(), 5.0);
        let n = rng.sample_normal(10.0, 2.0, 8.0, 12.0).unwrap();
        assert!((8.0..=12.0).contains(&n));
    }

    #[test]
    fn positive_normal_integer_preserves_shortcut_semantics() {
        let mut rng = RandomGenerator::from_seed(Some(6));
        assert_eq!(rng.sample_positive_normal_integer(0.1, 0.0).unwrap(), 1);
        assert_eq!(rng.sample_positive_normal_integer(2.5, 0.0).unwrap(), 2);
        assert_eq!(rng.sample_positive_normal_integer(3.5, 0.0).unwrap(), 4);
        for _ in 0..100 {
            assert!(rng.sample_positive_normal_integer(10.0, 2.0).unwrap() >= 1);
        }
    }

    #[test]
    fn exponential_and_gamma_sample_means_match_parameterization() {
        let mut rng = RandomGenerator::from_seed(Some(8));
        let exp: Vec<_> = (0..200_000)
            .map(|_| rng.expovariate(4.0).unwrap())
            .collect();
        let exp_mean = mean(&exp);
        assert!((exp_mean - 0.25).abs() / 0.25 < 0.02, "{exp_mean}");

        let gamma: Vec<_> = (0..200_000)
            .map(|_| rng.gammavariate(2.0, 3.0).unwrap())
            .collect();
        let gamma_mean = mean(&gamma);
        assert!((gamma_mean - 6.0).abs() / 6.0 < 0.02, "{gamma_mean}");
    }

    #[test]
    fn batch_methods_follow_scalar_bounds() {
        let mut rng = RandomGenerator::from_seed(Some(9));
        let ints = rng.integers(3, Some(6), 100).unwrap();
        assert!(ints.iter().all(|x| (3..6).contains(x)));
        let from_zero = rng.integers(4, None, 100).unwrap();
        assert!(from_zero.iter().all(|x| (0..4).contains(x)));
        assert!(rng.integers(4, Some(4), 1).is_err());
        assert_eq!(rng.normal_batch(3.0, 0.0, 5).unwrap(), vec![3.0; 5]);
        assert_eq!(rng.random_batch(7).len(), 7);
    }
}
