// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-selected RNG backend wrapper.
//!
//! The native Rust runtime keeps `rust` as its default backend so
//! existing deterministic behavior does not change unless the operator opts into
//! `AIPERF_RNG_BACKEND=python`. The wrapper delegates every shared RNG trait to
//! either the native Rust stream or the Python-compatible stream, letting
//! higher-level runtime code store one backend-agnostic concrete type.

use crate::rng::compat::python_random::PythonRandomGenerator;
use crate::rng::derive::{DerivedRandomGenerator, RngRoot};
use crate::rng::error::Result;
use crate::rng::generator::RustRandomGenerator;
use crate::rng::random_generator::{RandomGenerator, RuntimeRandomGenerator};

/// Runtime RNG backend selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeRngBackend {
    /// Native Rust BLAKE3 + PCG64 semantics.
    Rust,
    /// Python-compatible SHA-256 + CPython/NumPy semantics.
    Python,
}

impl RuntimeRngBackend {
    /// Canonical environment spelling for this backend.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
        }
    }
}

/// Return the RNG backend selected for this native Rust process.
///
/// Native Rust runs default to `rust` to preserve today's behavior. When
/// the environment contains an unknown value, the runtime logs a warning and
/// falls back to that same default.
pub fn configured_runtime_rng_backend() -> RuntimeRngBackend {
    match std::env::var("AIPERF_RNG_BACKEND").ok().as_deref() {
        Some("python") => RuntimeRngBackend::Python,
        Some("rust") | None => RuntimeRngBackend::Rust,
        Some(other) => {
            tracing::warn!(
                value = other,
                fallback = RuntimeRngBackend::Rust.as_str(),
                "unknown AIPERF_RNG_BACKEND for native runtime; using rust"
            );
            RuntimeRngBackend::Rust
        }
    }
}

/// Concrete runtime RNG chosen from [`configured_runtime_rng_backend`].
pub enum ConfiguredRandomGenerator {
    /// Native Rust BLAKE3 + PCG64 stream.
    Rust(RustRandomGenerator),
    /// Python-compatible SHA-256 + CPython/NumPy stream.
    Python(PythonRandomGenerator),
}

impl ConfiguredRandomGenerator {
    /// Construct the configured backend from an explicit seed or fresh entropy.
    pub fn from_seed_or_entropy(seed: Option<u64>) -> Self {
        match configured_runtime_rng_backend() {
            RuntimeRngBackend::Rust => Self::Rust(RustRandomGenerator::from_seed(seed)),
            RuntimeRngBackend::Python => {
                Self::Python(PythonRandomGenerator::from_seed_or_entropy(seed))
            }
        }
    }
}

impl DerivedRandomGenerator for ConfiguredRandomGenerator {
    fn from_rng_root(root: RngRoot, identifier: &str) -> Self {
        match configured_runtime_rng_backend() {
            RuntimeRngBackend::Rust => {
                Self::Rust(root.derive_generator::<RustRandomGenerator>(identifier))
            }
            RuntimeRngBackend::Python => {
                Self::Python(root.derive_generator::<PythonRandomGenerator>(identifier))
            }
        }
    }
}

impl RandomGenerator for ConfiguredRandomGenerator {
    fn random(&mut self) -> f64 {
        match self {
            Self::Rust(generator) => RandomGenerator::random(generator),
            Self::Python(generator) => RandomGenerator::random(generator),
        }
    }

    fn choice<'a, T>(&mut self, seq: &'a [T]) -> Result<&'a T> {
        match self {
            Self::Rust(generator) => RandomGenerator::choice(generator, seq),
            Self::Python(generator) => RandomGenerator::choice(generator, seq),
        }
    }

    fn randrange(&mut self, stop: i64) -> Result<i64> {
        match self {
            Self::Rust(generator) => RandomGenerator::randrange(generator, stop),
            Self::Python(generator) => RandomGenerator::randrange(generator, stop),
        }
    }

    fn randrange_step(&mut self, start: i64, stop: i64, step: i64) -> Result<i64> {
        match self {
            Self::Rust(generator) => RandomGenerator::randrange_step(generator, start, stop, step),
            Self::Python(generator) => {
                RandomGenerator::randrange_step(generator, start, stop, step)
            }
        }
    }

    fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        match self {
            Self::Rust(generator) => RandomGenerator::randint(generator, a, b),
            Self::Python(generator) => RandomGenerator::randint(generator, a, b),
        }
    }

    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        match self {
            Self::Rust(generator) => RandomGenerator::uniform(generator, a, b),
            Self::Python(generator) => RandomGenerator::uniform(generator, a, b),
        }
    }

    fn choices<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        match self {
            Self::Rust(generator) => RandomGenerator::choices(generator, population, k),
            Self::Python(generator) => RandomGenerator::choices(generator, population, k),
        }
    }

    fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        match self {
            Self::Rust(generator) => RandomGenerator::sample(generator, population, k),
            Self::Python(generator) => RandomGenerator::sample(generator, population, k),
        }
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        match self {
            Self::Rust(generator) => RandomGenerator::shuffle(generator, values),
            Self::Python(generator) => RandomGenerator::shuffle(generator, values),
        }
    }

    fn random_batch(&mut self, size: usize) -> Vec<f64> {
        match self {
            Self::Rust(generator) => RandomGenerator::random_batch(generator, size),
            Self::Python(generator) => RandomGenerator::random_batch(generator, size),
        }
    }

    fn integers(&mut self, low: i64, high: i64) -> i64 {
        match self {
            Self::Rust(generator) => RandomGenerator::integers(generator, low, high),
            Self::Python(generator) => RandomGenerator::integers(generator, low, high),
        }
    }

    fn integers_batch(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        match self {
            Self::Rust(generator) => RandomGenerator::integers_batch(generator, low, high, size),
            Self::Python(generator) => RandomGenerator::integers_batch(generator, low, high, size),
        }
    }

    fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        match self {
            Self::Rust(generator) => RandomGenerator::normal(generator, loc, scale),
            Self::Python(generator) => RandomGenerator::normal(generator, loc, scale),
        }
    }

    fn normal_batch(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        match self {
            Self::Rust(generator) => RandomGenerator::normal_batch(generator, loc, scale, size),
            Self::Python(generator) => RandomGenerator::normal_batch(generator, loc, scale, size),
        }
    }

    fn numpy_choice_uniform(&mut self, pop_size: i64, size: usize) -> Vec<i64> {
        match self {
            Self::Rust(generator) => {
                RandomGenerator::numpy_choice_uniform(generator, pop_size, size)
            }
            Self::Python(generator) => {
                RandomGenerator::numpy_choice_uniform(generator, pop_size, size)
            }
        }
    }

    fn numpy_choice_weighted(&mut self, weights: &[f64]) -> usize {
        match self {
            Self::Rust(generator) => RandomGenerator::numpy_choice_weighted(generator, weights),
            Self::Python(generator) => RandomGenerator::numpy_choice_weighted(generator, weights),
        }
    }

    fn numpy_choice_weighted_batch(&mut self, weights: &[f64], size: usize) -> Vec<usize> {
        match self {
            Self::Rust(generator) => {
                RandomGenerator::numpy_choice_weighted_batch(generator, weights, size)
            }
            Self::Python(generator) => {
                RandomGenerator::numpy_choice_weighted_batch(generator, weights, size)
            }
        }
    }

    fn numpy_choice_weighted_without_replacement(
        &mut self,
        weights: &[f64],
        size: usize,
    ) -> Result<Vec<usize>> {
        match self {
            Self::Rust(generator) => {
                RandomGenerator::numpy_choice_weighted_without_replacement(generator, weights, size)
            }
            Self::Python(generator) => {
                RandomGenerator::numpy_choice_weighted_without_replacement(generator, weights, size)
            }
        }
    }

    fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64> {
        match self {
            Self::Rust(generator) => {
                RandomGenerator::sample_normal(generator, mean, stddev, lower, upper)
            }
            Self::Python(generator) => {
                RandomGenerator::sample_normal(generator, mean, stddev, lower, upper)
            }
        }
    }

    fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> Result<f64> {
        match self {
            Self::Rust(generator) => {
                RandomGenerator::sample_positive_normal(generator, mean, stddev)
            }
            Self::Python(generator) => {
                RandomGenerator::sample_positive_normal(generator, mean, stddev)
            }
        }
    }

    fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> Result<i64> {
        match self {
            Self::Rust(generator) => {
                RandomGenerator::sample_positive_normal_integer(generator, mean, stddev)
            }
            Self::Python(generator) => {
                RandomGenerator::sample_positive_normal_integer(generator, mean, stddev)
            }
        }
    }

    fn expovariate(&mut self, lambd: f64) -> f64 {
        match self {
            Self::Rust(generator) => RandomGenerator::expovariate(generator, lambd),
            Self::Python(generator) => RandomGenerator::expovariate(generator, lambd),
        }
    }

    fn gammavariate(&mut self, alpha: f64, beta: f64) -> Result<f64> {
        match self {
            Self::Rust(generator) => RandomGenerator::gammavariate(generator, alpha, beta),
            Self::Python(generator) => RandomGenerator::gammavariate(generator, alpha, beta),
        }
    }
}

impl RuntimeRandomGenerator for ConfiguredRandomGenerator {
    fn seed(&self) -> Option<u64> {
        match self {
            Self::Rust(generator) => RuntimeRandomGenerator::seed(generator),
            Self::Python(generator) => RuntimeRandomGenerator::seed(generator),
        }
    }

    fn reseed(&mut self, seed: u64) {
        match self {
            Self::Rust(generator) => RuntimeRandomGenerator::reseed(generator, seed),
            Self::Python(generator) => RuntimeRandomGenerator::reseed(generator, seed),
        }
    }

    fn random_u64(&mut self) -> u64 {
        match self {
            Self::Rust(generator) => RuntimeRandomGenerator::random_u64(generator),
            Self::Python(generator) => RuntimeRandomGenerator::random_u64(generator),
        }
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        match self {
            Self::Rust(generator) => RuntimeRandomGenerator::fill_bytes(generator, dest),
            Self::Python(generator) => RuntimeRandomGenerator::fill_bytes(generator, dest),
        }
    }

    fn randrange_u64(&mut self, lo: u64, hi: u64) -> Result<u64> {
        match self {
            Self::Rust(generator) => RuntimeRandomGenerator::randrange_u64(generator, lo, hi),
            Self::Python(generator) => RuntimeRandomGenerator::randrange_u64(generator, lo, hi),
        }
    }

    fn weighted_choice<T: Clone>(&mut self, values: &[T], weights: Option<&[f64]>) -> Result<T> {
        match self {
            Self::Rust(generator) => {
                RuntimeRandomGenerator::weighted_choice(generator, values, weights)
            }
            Self::Python(generator) => {
                RuntimeRandomGenerator::weighted_choice(generator, values, weights)
            }
        }
    }

    fn normal_checked(&mut self, loc: f64, scale: f64) -> Result<f64> {
        match self {
            Self::Rust(generator) => RuntimeRandomGenerator::normal_checked(generator, loc, scale),
            Self::Python(generator) => {
                RuntimeRandomGenerator::normal_checked(generator, loc, scale)
            }
        }
    }

    fn normal_batch_checked(&mut self, loc: f64, scale: f64, size: usize) -> Result<Vec<f64>> {
        match self {
            Self::Rust(generator) => {
                RuntimeRandomGenerator::normal_batch_checked(generator, loc, scale, size)
            }
            Self::Python(generator) => {
                RuntimeRandomGenerator::normal_batch_checked(generator, loc, scale, size)
            }
        }
    }

    fn integers_checked(&mut self, low: i64, high: Option<i64>, size: usize) -> Result<Vec<i64>> {
        match self {
            Self::Rust(generator) => {
                RuntimeRandomGenerator::integers_checked(generator, low, high, size)
            }
            Self::Python(generator) => {
                RuntimeRandomGenerator::integers_checked(generator, low, high, size)
            }
        }
    }
}
