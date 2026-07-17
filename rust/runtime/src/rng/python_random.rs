// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact AgentX `RandomGenerator` compatibility.
//!
//! The reference implementation in `aiperf/common/random_generator.py` wraps a
//! `random.Random(child_seed)` (CPython Mersenne Twister) and a
//! `numpy.random.default_rng(child_seed)` (PCG64), seeded from the SAME child
//! seed, and routes each operation to one backend:
//!
//! - CPython MT ([`crate::rng::python_mt::PythonMt19937`]): `random`, `choice`,
//!   `sample`, `randrange`, `randint`, `uniform`, `choices`.
//! - numpy PCG64 ([`crate::rng::numpy_pcg64::NumpyPcg64`]): `shuffle` (in place).
//!
//! Child-seed derivation mirrors `_RNGManager.derive(identifier)`:
//! `child = int.from_bytes(sha256(f"{root_seed}:{identifier}").digest()[:8],
//! "big")`.
//!
//! This is the substrate the procedural coding corpus
//! ([`crate::graph::recorded::coding`]) must use to reproduce agentx's generated
//! text byte-for-byte; the canonical BLAKE3 + `rand_pcg` [`crate::rng::generator`]
//! is a DIFFERENT stream and cannot match agentx.
//!
//! Python (`random.Random` / `numpy`) is the fixed reference; this Rust conforms.
//! Parity is pinned by committed golden vectors
//! (`tests/data/agentx_rng_golden.json`).

use sha2::{Digest, Sha256};

use crate::rng::error::{Result, RngError};
use crate::rng::numpy_pcg64::NumpyPcg64;
use crate::rng::python_mt::PythonMt19937;

/// agentx `RandomGenerator`: a CPython-MT generator and a numpy-PCG64 generator,
/// both seeded from one child seed, each owning the operations it backs in Python.
pub struct PythonRandomGenerator {
    /// The child seed both backends were constructed from.
    seed: u64,
    /// CPython `random.Random(seed)` — scalar operations.
    mt: PythonMt19937,
    /// numpy `default_rng(seed)` — array `shuffle`.
    np: NumpyPcg64,
}

impl PythonRandomGenerator {
    /// Construct both backends from one child seed (mirrors
    /// `RandomGenerator.__init__`: `random.Random(seed)` + `default_rng(seed)`).
    pub fn from_child_seed(seed: u64) -> Self {
        Self {
            seed,
            mt: PythonMt19937::from_u64_seed(seed),
            np: NumpyPcg64::from_u64_seed(seed),
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

    /// `random.Random.random()`: uniform float in `[0.0, 1.0)` (MT `res53`).
    pub fn random(&mut self) -> f64 {
        self.mt.random()
    }

    /// `random.Random.choice(seq)`: `seq[_randbelow(len(seq))]`.
    pub fn choice<'a, T>(&mut self, seq: &'a [T]) -> Result<&'a T> {
        if seq.is_empty() {
            return Err(RngError::EmptySequence { what: "choice" });
        }
        let idx = self.mt.randbelow(seq.len() as u64) as usize;
        Ok(&seq[idx])
    }

    /// `random.Random.randrange(stop)` (single-arg): `_randbelow(stop)`.
    pub fn randrange(&mut self, stop: i64) -> Result<i64> {
        if stop <= 0 {
            return Err(RngError::EmptyRange { what: "randrange" });
        }
        Ok(self.mt.randbelow(stop as u64) as i64)
    }

    /// `random.Random.randint(a, b)`: `a + _randbelow(b - a + 1)`, inclusive.
    pub fn randint(&mut self, a: i64, b: i64) -> Result<i64> {
        if a > b {
            return Err(RngError::EmptyRange { what: "randint" });
        }
        let width = (i128::from(b) - i128::from(a) + 1) as u64;
        Ok(a + self.mt.randbelow(width) as i64)
    }

    /// `random.Random.uniform(a, b)`: `a + (b - a) * random()`.
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.random()
    }

    /// `random.Random.choices(population, k=k)` (unweighted): `k` draws with
    /// replacement, each `population[floor(random() * n)]`.
    pub fn choices<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
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

    /// `random.Random.sample(population, k)`: CPython 3.12 `Lib/random.py` pool /
    /// selected-set algorithm, byte-exact in `_randbelow` call count and order.
    pub fn sample<T: Clone>(&mut self, population: &[T], k: usize) -> Result<Vec<T>> {
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

    /// `RandomGenerator.shuffle(x)`: numpy `default_rng(seed).shuffle(x)` in place.
    pub fn shuffle<T>(&mut self, values: &mut [T]) {
        self.np.shuffle(values);
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
