// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `RandomGenerator` trait: AIPerf's Python-parity random-operation
//! contract, mirroring `src/aiperf/common/random_generator.py`'s
//! `RandomGenerator` class method-for-method.
//!
//! [`crate::rng::compat::python_random::PythonRandomGenerator`] is the sole
//! implementer: a composite of one CPython Mersenne Twister
//! ([`crate::rng::compat::python_mt::PythonMt19937`]) and one numpy-compatible
//! PCG64 ([`crate::rng::compat::numpy_generator::NumpyGenerator`]), seeded
//! together and dispatching operations exactly as the Python class dispatches
//! between `self._python_rng` and `self._numpy_rng`. This trait is
//! deliberately distinct from [`crate::rng::generator::RustRandomGenerator`]
//! (BLAKE3 + `rand_pcg`, AIPerf's native scheduling/sampling substrate) — the
//! two are unrelated backends serving different purposes, matching Python's
//! own `"python"` vs `"rust_parity"` `Environment.RNG.BACKEND` split.

use crate::rng::error::Result;

/// The full operation set of `random_generator.py`'s `RandomGenerator` class,
/// dispatched across a CPython-MT stream (scalar ops) and a numpy-PCG64 stream
/// (array ops) — see the module docs for the exact routing table.
pub trait RandomGenerator {
    /// `random.Random.random()`: uniform float in `[0.0, 1.0)` (MT `res53`).
    fn random(&mut self) -> f64;

    /// `random.Random.choice(seq)`: `seq[_randbelow(len(seq))]`.
    fn choice<'a, T>(&mut self, seq: &'a [T]) -> Result<&'a T>;

    /// `random.Random.randrange(stop)` (single-arg): `_randbelow(stop)`.
    fn randrange(&mut self, stop: i64) -> Result<i64>;

    /// `random.Random.randrange(start, stop, step)` (full form). Rust's
    /// natural split of Python's `*args`-overloaded `randrange`; see
    /// [`Self::randrange`] for the single-arg `(stop)` form.
    fn randrange_step(&mut self, start: i64, stop: i64, step: i64) -> Result<i64>;

    /// `random.Random.randint(a, b)`: `a + _randbelow(b - a + 1)`, inclusive.
    fn randint(&mut self, a: i64, b: i64) -> Result<i64>;

    /// `random.Random.uniform(a, b)`: `a + (b - a) * random()`.
    fn uniform(&mut self, a: f64, b: f64) -> f64;

    /// `random.Random.choices(population, k=k)` (unweighted): `k` draws with
    /// replacement, each `population[floor(random() * n)]`.
    fn choices<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>>;

    /// `random.Random.sample(population, k)`: CPython 3.12 `Lib/random.py` pool /
    /// selected-set algorithm, byte-exact in `_randbelow` call count and order.
    fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>>;

    /// `RandomGenerator.shuffle(x)`: numpy `default_rng(seed).shuffle(x)` in place.
    fn shuffle<T>(&mut self, values: &mut [T]);

    /// `RandomGenerator.random_batch(size)`: `size` numpy `random()` draws.
    fn random_batch(&mut self, size: usize) -> Vec<f64>;

    /// `RandomGenerator.integers(low, high)`: one numpy integer draw in
    /// `[low, high)`.
    fn integers(&mut self, low: i64, high: i64) -> i64;

    /// `RandomGenerator.integers(low, high, size)`: `size` numpy integer draws.
    fn integers_batch(&mut self, low: i64, high: i64, size: usize) -> Vec<i64>;

    /// `RandomGenerator.normal(loc, scale)`: one numpy normal draw.
    fn normal(&mut self, loc: f64, scale: f64) -> f64;

    /// `RandomGenerator.normal(loc, scale, size)`: `size` numpy normal draws.
    fn normal_batch(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64>;

    /// `RandomGenerator.numpy_choice(a, size, p=None, replace=True)` restricted
    /// to the uniform (unweighted, with-replacement) shape: `size` numpy
    /// integer draws in `[0, pop_size)`.
    fn numpy_choice_uniform(&mut self, pop_size: i64, size: usize) -> Vec<i64>;

    /// `RandomGenerator.numpy_choice(a, p=weights)` (with-replacement, one
    /// draw): one numpy weighted-index draw in `[0, weights.len())`.
    fn numpy_choice_weighted(&mut self, weights: &[f64]) -> usize;

    /// `RandomGenerator.numpy_choice(a, size, p=weights)` (with-replacement,
    /// batch): `size` numpy weighted-index draws in `[0, weights.len())`.
    fn numpy_choice_weighted_batch(&mut self, weights: &[f64], size: usize) -> Vec<usize>;

    /// `RandomGenerator.numpy_choice(a, size, p=weights, replace=False)`:
    /// `size` DISTINCT numpy weighted-index draws in `[0, weights.len())`.
    /// Errs if fewer than `size` weights are positive.
    fn numpy_choice_weighted_without_replacement(
        &mut self,
        weights: &[f64],
        size: usize,
    ) -> Result<Vec<usize>>;

    /// `RandomGenerator.sample_normal(mean, stddev, lower, upper)`: rejection
    /// sampling via `Random.gauss`, clamped to `[lower, upper]`.
    ///
    /// Mirrors `random_generator.py::sample_normal`: rejects `lower > upper`,
    /// retries up to 10,000 times, and falls back to the mean clamped to
    /// `[lower, upper]` if every draw is rejected (unreachable bounds).
    fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64>;

    /// `RandomGenerator.sample_positive_normal(mean, stddev)`:
    /// `sample_normal(mean, stddev, lower=0, upper=+inf)`. Rejects `mean < 0`.
    fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> Result<f64>;

    /// `RandomGenerator.sample_positive_normal_integer(mean, stddev)`: as
    /// [`Self::sample_positive_normal`], ceiling-rounded and floored at `1`.
    /// `stddev <= 0` short-circuits to `max(1, round(mean))` without drawing.
    fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> Result<i64>;

    /// `RandomGenerator.expovariate(lambd)`: `-ln(1 - random()) / lambd`.
    fn expovariate(&mut self, lambd: f64) -> f64;

    /// `RandomGenerator.gammavariate(alpha, beta)`. Rejects `alpha <= 0` or
    /// `beta <= 0`.
    fn gammavariate(&mut self, alpha: f64, beta: f64) -> Result<f64>;
}
