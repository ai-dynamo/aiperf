// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Portions derived from CPython (Lib/random.py), PSF License. See ATTRIBUTIONS.md.
// Portions derived from NumPy (numpy.random), BSD-3-Clause / NCSA. See ATTRIBUTIONS.md.

//! Byte-exact CPython/numpy composite implementing [`crate::rng::RandomGenerator`].
//!
//! Mirrors `src/aiperf/common/random_generator.py`'s `RandomGenerator` class:
//! one CPython Mersenne Twister and one numpy-compatible PCG64, seeded from the
//! same child seed, with operations routed exactly as the Python class routes
//! them between `self._python_rng` and `self._numpy_rng`:
//!
//! - CPython MT ([`crate::rng::compat::python_mt::PythonMt19937`]): `random`,
//!   `choice`, `sample`, `randrange`, `randint`, `uniform`, `choices`,
//!   `sample_normal` (via `gauss`), `expovariate`, `gammavariate`.
//! - numpy PCG64 ([`crate::rng::compat::numpy_generator::NumpyGenerator`]):
//!   `shuffle`, `integers`, `normal`, `random_batch`, `numpy_choice_uniform`,
//!   `numpy_choice_weighted`.
//!
//! Child seeds are derived as
//! `child = int.from_bytes(sha256(f"{root_seed}:{identifier}").digest()[:8],
//! "big")`, exactly as `_RNGManager.derive`.
//!
//! [`PythonRandomGenerator`] was originally scoped to the subset of methods the
//! procedural coding corpus (`crate::graph::recorded::coding`) needs; it now
//! implements the full [`crate::rng::RandomGenerator`] trait. The BLAKE3 plus
//! `rand_pcg` native stream (`crate::rng::generator::RustRandomGenerator`) is
//! intentionally a different, unrelated backend — this one exists purely for
//! byte-exact parity with `random_generator.py`. Committed golden vectors
//! (`tests/data/agentx_rng_golden.json`) pin the subset AgentX exercises; new
//! methods are pinned against a local CPython/numpy interpreter in this file's
//! tests.

use sha2::{Digest, Sha256};

use crate::rng::RandomGenerator;
use crate::rng::compat::numpy_generator::NumpyGenerator;
use crate::rng::compat::python_mt::PythonMt19937;
use crate::rng::error::{Result, RngError};

/// `random_generator.py`'s composite: a CPython-MT generator and a numpy-PCG64
/// generator, both seeded from one child seed with disjoint operation sets.
pub struct PythonRandomGenerator {
    /// The child seed both backends were constructed from.
    seed: u64,
    /// CPython `random.Random(seed)` — scalar operations.
    mt: PythonMt19937,
    /// numpy `default_rng(seed)` — array/numpy-flavored operations.
    np: NumpyGenerator,
}

impl PythonRandomGenerator {
    /// Construct both backends from one child seed.
    pub fn from_child_seed(seed: u64) -> Self {
        Self {
            seed,
            mt: PythonMt19937::from_u64_seed(seed),
            np: NumpyGenerator::from_seed(seed),
        }
    }

    /// Derive a child generator from a root seed and a dotted identifier, exactly
    /// as `_RNGManager.derive`: `child_seed = big-endian u64 of the first 8 bytes
    /// of sha256(f"{root_seed}:{identifier}")`.
    pub fn derive(root_seed: u64, identifier: &str) -> Self {
        Self::from_child_seed(Self::derive_child_seed(root_seed, identifier))
    }

    /// The `_RNGManager.derive` seed derivation, exposed for golden tests.
    pub fn derive_child_seed(root_seed: u64, identifier: &str) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(format!("{root_seed}:{identifier}").as_bytes());
        let digest = hasher.finalize();
        let mut low8 = [0u8; 8];
        low8.copy_from_slice(&digest[..8]);
        u64::from_be_bytes(low8)
    }

    /// The child seed this generator was constructed from.
    pub const fn seed(&self) -> u64 {
        self.seed
    }
}

impl RandomGenerator for PythonRandomGenerator {
    fn random(&mut self) -> f64 {
        self.mt.random()
    }

    fn choice<'a, T>(&mut self, seq: &'a [T]) -> Result<&'a T> {
        if seq.is_empty() {
            return Err(RngError::EmptySequence { what: "choice" });
        }
        let idx = self.mt.randbelow(seq.len() as u64) as usize;
        Ok(&seq[idx])
    }

    fn randrange(&mut self, stop: i64) -> Result<i64> {
        if stop <= 0 {
            return Err(RngError::EmptyRange { what: "randrange" });
        }
        Ok(self.mt.randbelow(stop as u64) as i64)
    }

    fn randrange_step(&mut self, start: i64, stop: i64, step: i64) -> Result<i64> {
        let width = stop - start;
        if step == 1 {
            return if width > 0 {
                Ok(start + self.mt.randbelow(width as u64) as i64)
            } else {
                Err(RngError::EmptyRange { what: "randrange" })
            };
        }
        if step == 0 {
            return Err(RngError::InvalidParameter {
                what: "randrange: zero step",
                value: 0.0,
            });
        }
        let n = if step > 0 {
            py_floordiv(width + step - 1, step)
        } else {
            py_floordiv(width + step + 1, step)
        };
        if n <= 0 {
            return Err(RngError::EmptyRange { what: "randrange" });
        }
        Ok(start + step * self.mt.randbelow(n as u64) as i64)
    }

    fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        if a > b {
            return Err(RngError::EmptyRange { what: "randint" });
        }
        let width = (i128::from(b) - i128::from(a) + 1) as u64;
        Ok(a + self.mt.randbelow(width) as i64)
    }

    fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    fn choices<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        let n = population.len();
        if n == 0 && k > 0 {
            return Err(RngError::EmptySequence { what: "choices" });
        }
        let mut out = Vec::with_capacity(k);
        let nf = n as f64;
        for _ in 0..k {
            let idx = (self.random() * nf).floor() as usize;
            out.push(population[idx].clone());
        }
        Ok(out)
    }

    fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
        let n = population.len();
        if k > n {
            return Err(RngError::SampleTooLarge { k, len: n });
        }
        // setsize = 21; if k > 5: setsize += 4 ** ceil(log(k*3, 4)).
        let mut setsize: usize = 21;
        if k > 5 {
            let exp = ((k as f64 * 3.0).ln() / 4.0_f64.ln()).ceil();
            setsize += 4usize.pow(exp as u32);
        }
        let mut result = Vec::with_capacity(k);
        if n <= setsize {
            // Pool method: an n-length list is smaller than a k-length set.
            let mut pool: Vec<T> = population.to_vec();
            for i in 0..k {
                let j = self.mt.randbelow((n - i) as u64) as usize;
                result.push(pool[j].clone());
                pool[j] = pool[n - i - 1].clone();
            }
        } else {
            // Selected-set method.
            let mut selected = std::collections::HashSet::new();
            for _ in 0..k {
                let mut j = self.mt.randbelow(n as u64) as usize;
                while selected.contains(&j) {
                    j = self.mt.randbelow(n as u64) as usize;
                }
                selected.insert(j);
                result.push(population[j].clone());
            }
        }
        Ok(result)
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        self.np.shuffle(values);
    }

    fn random_batch(&mut self, size: usize) -> Vec<f64> {
        self.np.random_batch(size)
    }

    fn integers(&mut self, low: i64, high: i64) -> i64 {
        self.np.integers(low, high)
    }

    fn integers_batch(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        self.np.integers_batch(low, high, size)
    }

    fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        self.np.normal(loc, scale)
    }

    fn normal_batch(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        self.np.normal_batch(loc, scale, size)
    }

    fn numpy_choice_uniform(&mut self, pop_size: i64, size: usize) -> Vec<i64> {
        self.np.choice_uniform(pop_size, size)
    }

    fn numpy_choice_weighted(&mut self, weights: &[f64]) -> usize {
        self.np.choice_weighted(weights)
    }

    fn numpy_choice_weighted_batch(&mut self, weights: &[f64], size: usize) -> Vec<usize> {
        self.np.choice_weighted_batch(weights, size)
    }

    fn numpy_choice_weighted_without_replacement(
        &mut self,
        weights: &[f64],
        size: usize,
    ) -> Result<Vec<usize>> {
        self.np.choice_weighted_without_replacement(weights, size)
    }

    fn sample_normal(&mut self, mean: f64, stddev: f64, lower: f64, upper: f64) -> Result<f64> {
        if lower > upper {
            return Err(RngError::InvalidBounds { lower, upper });
        }
        const MAX_ITERATIONS: usize = 10_000;
        for _ in 0..MAX_ITERATIONS {
            let n = self.mt.gauss(mean, stddev);
            if lower <= n && n <= upper {
                return Ok(n);
            }
        }
        Ok(mean.max(lower).min(upper))
    }

    fn sample_positive_normal(&mut self, mean: f64, stddev: f64) -> Result<f64> {
        if mean < 0.0 {
            return Err(RngError::InvalidParameter {
                what: "sample_positive_normal: mean should be greater than 0",
                value: mean,
            });
        }
        self.sample_normal(mean, stddev, 0.0, f64::INFINITY)
    }

    fn sample_positive_normal_integer(&mut self, mean: f64, stddev: f64) -> Result<i64> {
        if stddev <= 0.0 {
            return Ok((mean.round() as i64).max(1));
        }
        let sample = self.sample_positive_normal(mean, stddev)?;
        Ok((sample.ceil() as i64).max(1))
    }

    fn expovariate(&mut self, lambd: f64) -> f64 {
        self.mt.expovariate(lambd)
    }

    fn gammavariate(&mut self, alpha: f64, beta: f64) -> Result<f64> {
        self.mt.gammavariate(alpha, beta)
    }
}

/// Python's `//` floor division: rounds toward negative infinity regardless of
/// operand signs (unlike Rust's truncating `/`), matching `randrange`'s
/// `(width + step ± 1) // step` step-count computation.
fn py_floordiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && (r < 0) != (b < 0) {
        q - 1
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct Golden {
        #[serde(rename = "child_seed_42_coding.test")]
        child_seed: u64,
        #[serde(rename = "choice_range100_10x")]
        choice_range100: Vec<i64>,
        // Exact IEEE-754 bit patterns: decimal JSON text round-trips 1 ULP low
        // under serde_json, so floats are pinned by their bits, not their text.
        #[serde(rename = "random_8x_bits")]
        random8_bits: Vec<u64>,
        #[serde(rename = "randrange50_10x")]
        randrange50: Vec<i64>,
        #[serde(rename = "randint_3_17_10x")]
        randint_3_17: Vec<i64>,
        #[serde(rename = "uniform_1_30_5x_bits")]
        uniform_1_30_bits: Vec<u64>,
        sample_5of20: Vec<i64>,
        sample_30of200: Vec<i64>,
        choices_5of10: Vec<i64>,
        shuffle12: Vec<usize>,
        #[serde(rename = "seed_999_dataset.coding_content.template")]
        seed_999_template: u64,
        #[serde(rename = "seed_999_dataset.coding_content.corpus")]
        seed_999_corpus: u64,
        #[serde(rename = "seed_999_dataset.coding_content.length")]
        seed_999_length: u64,
    }

    fn load() -> Golden {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/agentx_rng_golden.json"
        );
        let raw = std::fs::read_to_string(path).expect("read agentx rng golden vectors");
        serde_json::from_str(&raw).expect("parse agentx rng golden vectors")
    }

    fn fresh(seed: u64) -> PythonRandomGenerator {
        PythonRandomGenerator::from_child_seed(seed)
    }

    #[test]
    fn derive_matches_agentx_rng_manager() {
        let g = load();
        assert_eq!(
            PythonRandomGenerator::derive_child_seed(42, "coding.test"),
            g.child_seed
        );
        assert_eq!(
            PythonRandomGenerator::derive_child_seed(999, "dataset.coding_content.template"),
            g.seed_999_template
        );
        assert_eq!(
            PythonRandomGenerator::derive_child_seed(999, "dataset.coding_content.corpus"),
            g.seed_999_corpus
        );
        assert_eq!(
            PythonRandomGenerator::derive_child_seed(999, "dataset.coding_content.length"),
            g.seed_999_length
        );
    }

    #[test]
    fn choice_matches_cpython() {
        let g = load();
        let seq: Vec<i64> = (0..100).collect();
        let mut r = fresh(g.child_seed);
        let got: Vec<i64> = (0..10).map(|_| *r.choice(&seq).unwrap()).collect();
        assert_eq!(got, g.choice_range100);
    }

    #[test]
    fn random_matches_cpython_res53() {
        let g = load();
        let mut r = fresh(g.child_seed);
        for &expected in &g.random8_bits {
            assert_eq!(r.random().to_bits(), expected);
        }
    }

    #[test]
    fn randrange_matches_cpython() {
        let g = load();
        let mut r = fresh(g.child_seed);
        let got: Vec<i64> = (0..10).map(|_| r.randrange(50).unwrap()).collect();
        assert_eq!(got, g.randrange50);
    }

    #[test]
    fn randint_matches_cpython() {
        let g = load();
        let mut r = fresh(g.child_seed);
        let got: Vec<i64> = (0..10).map(|_| r.randint(3, 17).unwrap()).collect();
        assert_eq!(got, g.randint_3_17);
    }

    #[test]
    fn uniform_matches_cpython() {
        let g = load();
        let mut r = fresh(g.child_seed);
        for &expected in &g.uniform_1_30_bits {
            assert_eq!(r.uniform(1.0, 30.0).to_bits(), expected);
        }
    }

    #[test]
    fn sample_pool_method_matches_cpython() {
        let g = load();
        let pop: Vec<i64> = (0..20).collect();
        let mut r = fresh(g.child_seed);
        assert_eq!(r.sample(&pop, 5).unwrap(), g.sample_5of20);
    }

    #[test]
    fn sample_large_k_matches_cpython() {
        let g = load();
        let pop: Vec<i64> = (0..200).collect();
        let mut r = fresh(g.child_seed);
        assert_eq!(r.sample(&pop, 30).unwrap(), g.sample_30of200);
    }

    #[test]
    fn choices_matches_cpython() {
        let g = load();
        let pop: Vec<i64> = (0..10).collect();
        let mut r = fresh(g.child_seed);
        assert_eq!(r.choices(&pop, 5).unwrap(), g.choices_5of10);
    }

    #[test]
    fn shuffle_matches_numpy() {
        let g = load();
        let mut r = fresh(g.child_seed);
        let mut x: Vec<usize> = (0..12).collect();
        r.shuffle(&mut x);
        assert_eq!(x, g.shuffle12);
    }
}
