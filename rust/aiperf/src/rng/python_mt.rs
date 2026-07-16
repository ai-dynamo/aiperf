// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPython `random.Random` Mersenne Twister, ported byte-exact.
//!
//! This is the generator behind legacy agentx `RandomSampler`
//! (`dataset/dataset_samplers.py`), which samples the recorded-graph corpus WITH
//! replacement via `random.Random(seed).choice(ids)`. numpy's PCG64
//! ([`crate::rng::numpy_pcg64`]) drives the shuffle/`t*` draws; the with-replacement
//! `RandomSampler` uses CPython's stdlib `random` instead, a DIFFERENT algorithm,
//! so it needs its own byte-exact port.
//!
//! Ported from CPython 3.12 (stream-stable across 3.x):
//!   - `Modules/_randommodule.c`: `init_genrand`, `init_by_array`,
//!     `genrand_uint32`, and integer seeding in `random_seed` (absolute value
//!     split into little-endian 32-bit key words, then `init_by_array`).
//!   - `Lib/random.py`: `Random.getrandbits` (for `k <= 32`,
//!     `genrand_uint32() >> (32 - k)`), `Random._randbelow_with_getrandbits`, and
//!     `Random.choice` (`seq[_randbelow(len(seq))]`).
//!
//! Python is the fixed reference; this Rust conforms to it. Parity is pinned by
//! committed golden vectors (`tests/data/legacy_random_sampler_vectors.json`,
//! replayed in [`crate::graph::tstar`] and below).

/// Mersenne Twister state size (`N` in `_randommodule.c`).
const N: usize = 624;
/// Mersenne Twister recurrence offset (`M`).
const M: usize = 397;
/// Twist matrix constant `MATRIX_A`.
const MATRIX_A: u32 = 0x9908_b0df;
/// High-order bit mask `UPPER_MASK`.
const UPPER_MASK: u32 = 0x8000_0000;
/// Low-order bits mask `LOWER_MASK`.
const LOWER_MASK: u32 = 0x7fff_ffff;

/// CPython `random.Random`'s MT19937 generator, seeded from an integer.
///
/// Construct with [`PythonMt19937::from_u64_seed`] to match `random.Random(seed)`
/// for a non-negative integer seed, then draw with [`PythonMt19937::next_u32`],
/// [`PythonMt19937::getrandbits`], or [`PythonMt19937::randbelow`] (the
/// `choice`-index primitive).
pub struct PythonMt19937 {
    /// The 624-word state vector (`mt`).
    mt: [u32; N],
    /// Index of the next word to temper; `>= N` forces a fresh twist block.
    index: usize,
}

impl PythonMt19937 {
    /// Seed exactly as `random.Random(seed)` does for a non-negative integer:
    /// split the value into little-endian 32-bit key words (dropping trailing
    /// zero words but keeping at least one — seed `0` yields key `[0]`), then run
    /// `init_by_array` (`_randommodule.c` `random_seed`). The key length is the
    /// number of 32-bit words the value occupies, so `seed < 2^32` uses `[low]`
    /// and a larger `u64` uses `[low, high]`.
    pub fn from_u64_seed(seed: u64) -> Self {
        let key: Vec<u32> = if seed == 0 {
            vec![0]
        } else {
            let mut words = Vec::with_capacity(2);
            let mut value = seed;
            while value > 0 {
                words.push((value & 0xffff_ffff) as u32);
                value >>= 32;
            }
            words
        };
        let mut generator = Self {
            mt: [0u32; N],
            index: N,
        };
        generator.init_by_array(&key);
        generator
    }

    /// `init_genrand` (`_randommodule.c`): seed the state vector from one word.
    fn init_genrand(&mut self, seed: u32) {
        self.mt[0] = seed;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253_u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    /// `init_by_array` (`_randommodule.c`): mix the key words into the state
    /// after `init_genrand(19650218)`.
    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let key_length = key.len();
        let mut i = 1usize;
        let mut j = 0usize;
        // First loop: run max(N, key_length) mixing steps folding in key words.
        let mut k = if N > key_length { N } else { key_length };
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
            k -= 1;
        }
        // Second loop: N-1 further mixing steps that decorrelate the state.
        let mut k = N - 1;
        while k > 0 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ ((prev ^ (prev >> 30)).wrapping_mul(1_566_083_941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        // MSB is 1, assuring a non-zero initial array.
        self.mt[0] = 0x8000_0000;
    }

    /// Regenerate the 624-word block (the twist), resetting the temper index.
    fn generate_block(&mut self) {
        for kk in 0..(N - M) {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01(y);
        }
        for kk in (N - M)..(N - 1) {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
            self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01(y);
        }
        let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
        self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01(y);
        self.index = 0;
    }

    /// Draw the next tempered 32-bit word (`genrand_uint32`).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.generate_block();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        // Standard MT19937 tempering.
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `Random.random()` (`_randommodule.c` `random_random` / `genrand_res53`):
    /// draw two tempered words, take the high 27 and 26 bits respectively, and
    /// combine into a 53-bit-mantissa double in `[0.0, 1.0)`:
    /// `(a * 2^26 + b) / 2^53` with `a = next_u32() >> 5`, `b = next_u32() >> 6`.
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `Random.getrandbits(k)` for `0 < k <= 32`: `genrand_uint32() >> (32 - k)`.
    ///
    /// The recorded-graph `choice` never needs more than a corpus-size worth of
    /// bits (`k <= 32`), so the arbitrary-precision `k > 32` path CPython supports
    /// is intentionally unimplemented here; it asserts rather than silently
    /// diverging.
    pub fn getrandbits(&mut self, k: u32) -> u32 {
        assert!(
            (1..=32).contains(&k),
            "PythonMt19937::getrandbits supports 1..=32 bits (choice never needs more)"
        );
        self.next_u32() >> (32 - k)
    }

    /// `Random._randbelow_with_getrandbits(n)`: uniform int in `[0, n)`.
    ///
    /// `n == 0` returns `0` (CPython leaves it undefined, but `choice` guards the
    /// empty sequence, so this is the safe corpus-degenerate answer). Otherwise
    /// draws `getrandbits(bit_length(n))` and rejects values `>= n`. `bit_length`
    /// uses `n` itself (not `n - 1`), matching CPython's comment that `n` can be
    /// `1`.
    pub fn randbelow(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        let k = 64 - n.leading_zeros();
        let mut r = u64::from(self.getrandbits(k));
        while r >= n {
            r = u64::from(self.getrandbits(k));
        }
        r
    }
}

/// The `mag01` twist term: `MATRIX_A` when `y` is odd, else `0`.
#[inline]
fn mag01(y: u32) -> u32 {
    if y & 1 == 0 { 0 } else { MATRIX_A }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt_core_seed_zero_sanity() {
        // The seeding contract's canary: `random.Random(0)` first four
        // `getrandbits(32)` draws (also the first four `genrand_uint32`).
        let mut generator = PythonMt19937::from_u64_seed(0);
        assert_eq!(
            [
                generator.next_u32(),
                generator.next_u32(),
                generator.next_u32(),
                generator.next_u32(),
            ],
            [3_626_764_237, 1_654_615_998, 3_255_389_356, 3_823_568_514]
        );
    }

    #[derive(serde::Deserialize)]
    struct MtVector {
        seed: u64,
        vals: Vec<u32>,
    }

    #[derive(serde::Deserialize)]
    struct ChoiceVector {
        seed: u64,
        n: u64,
        seq: Vec<u64>,
    }

    #[derive(serde::Deserialize)]
    struct Fixtures {
        mt_getrandbits32: Vec<MtVector>,
        choice_stream: Vec<ChoiceVector>,
    }

    fn load_fixtures() -> Fixtures {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/legacy_random_sampler_vectors.json"
        );
        let raw = std::fs::read_to_string(path).expect("read legacy random sampler vectors");
        serde_json::from_str(&raw).expect("parse legacy random sampler vectors")
    }

    #[test]
    fn mt_getrandbits32_matches_cpython() {
        // Layer 1: MT seeding + genrand core against `getrandbits(32)` streams.
        let fixtures = load_fixtures();
        assert!(!fixtures.mt_getrandbits32.is_empty());
        for vector in &fixtures.mt_getrandbits32 {
            let mut generator = PythonMt19937::from_u64_seed(vector.seed);
            for (draw, &expected) in vector.vals.iter().enumerate() {
                assert_eq!(
                    generator.getrandbits(32),
                    expected,
                    "seed {} draw {draw}",
                    vector.seed
                );
            }
        }
    }

    #[test]
    fn choice_stream_matches_cpython_with_replacement() {
        // Layer 2: `_randbelow` + `choice(range(n))` (with replacement) streams.
        let fixtures = load_fixtures();
        assert!(!fixtures.choice_stream.is_empty());
        for vector in &fixtures.choice_stream {
            let mut generator = PythonMt19937::from_u64_seed(vector.seed);
            for (draw, &expected) in vector.seq.iter().enumerate() {
                assert_eq!(
                    generator.randbelow(vector.n),
                    expected,
                    "seed {} n {} draw {draw}",
                    vector.seed,
                    vector.n
                );
            }
        }
    }
}
