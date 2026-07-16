// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NumPy-compatible `SeedSequence` + PCG64 bit generator.
//!
//! This is a byte-exact port of the seeding and generation path that
//! `numpy.random.default_rng(int_seed)` uses, built solely to reproduce
//! Python's `np.random.default_rng(seed).uniform(lo, hi)` t* values in Rust.
//! It is deliberately independent of `crate::rng::generator` (BLAKE3 +
//! `rand_pcg`), which uses different seeding and a different output stream and
//! is NOT numpy-compatible.
//!
//! Ported from numpy 2.5.1 (constants verified against numpy 1.26.4 sdist,
//! which are stream-stable across these versions):
//!   - `numpy/random/bit_generator.pyx`: `_coerce_to_uint32_array` /
//!     `_int_to_uint32_array`, `SeedSequence.mix_entropy`,
//!     `SeedSequence.generate_state`, `hashmix`, `mix`, and the hashing
//!     constants `INIT_A`, `INIT_B`, `MULT_A`, `MULT_B`, `MIX_MULT_L`,
//!     `MIX_MULT_R`, `XSHIFT`.
//!   - `numpy/random/_pcg64.pyx` + `numpy/random/src/pcg64/pcg64.{c,h}`:
//!     `pcg64_set_seed`, `pcg_setseq_128_srandom_r`, `pcg_setseq_128_step_r`,
//!     `pcg_output_xsl_rr_128_64`, `pcg64_next32`, and
//!     `PCG_DEFAULT_MULTIPLIER_128`.
//!   - `numpy/random/_generator.pyx`: `Generator.permutation` (`arange` +
//!     `shuffle`) and `Generator.shuffle` -> `_shuffle_raw` Fisher-Yates.
//!   - `numpy/random/src/distributions/distributions.c`: `random_interval`
//!     masked-rejection bounded-integer draw (uint32 branch).
//!
//! `default_rng(int_seed)` == `SeedSequence(int_seed)` -> `generate_state(4,
//! uint64)` -> PCG64 128-bit state/increment split -> XSL-RR 128->64 output.

/// The four `SeedSequence` hashing constants and shift, from
/// `bit_generator.pyx` lines 58-64.
const INIT_A: u32 = 0x43b0_d7e5;
/// Multiplier stepped inside `hashmix`.
const MULT_A: u32 = 0x931e_8875;
/// Initial hash constant for `generate_state`.
const INIT_B: u32 = 0x8b51_f9dd;
/// Multiplier stepped inside `generate_state`.
const MULT_B: u32 = 0x58f3_8ded;
/// Left multiplier in `mix`.
const MIX_MULT_L: u32 = 0xca01_f9dd;
/// Right multiplier in `mix`.
const MIX_MULT_R: u32 = 0x4973_f715;
/// Half the 32-bit word width; the xorshift distance shared by `hashmix`,
/// `mix`, and `generate_state`.
const XSHIFT: u32 = 16;
/// The default entropy pool size (128 bits of pooled state).
const POOL_SIZE: usize = 4;

/// PCG64's 128-bit LCG multiplier (`PCG_DEFAULT_MULTIPLIER_128`), assembled
/// from its high/low 64-bit halves in `pcg64.h`.
const PCG_DEFAULT_MULTIPLIER_128: u128 =
    ((2_549_297_995_355_413_924_u64 as u128) << 64) | (4_865_540_595_714_422_341_u64 as u128);

/// `numpy`-compatible PCG64 bit generator seeded through `SeedSequence`.
///
/// Construct with [`NumpyPcg64::from_u64_seed`] to match
/// `np.random.default_rng(seed)`, then draw raw words with
/// [`NumpyPcg64::next_u64`] or doubles with [`NumpyPcg64::next_double`] /
/// [`NumpyPcg64::uniform`].
pub struct NumpyPcg64 {
    /// The 128-bit LCG state.
    state: u128,
    /// The 128-bit (odd) LCG increment.
    inc: u128,
    /// Whether `uinteger` holds a buffered high-32 word, mirroring
    /// `pcg64_state.has_uint32` (`pcg64.h`). Freshly seeded generators start
    /// with an empty buffer, matching `default_rng`.
    has_uint32: bool,
    /// The buffered high-32 word consumed by the next [`NumpyPcg64::next_u32`]
    /// call, mirroring `pcg64_state.uinteger` (`pcg64.h`).
    uinteger: u32,
}

impl NumpyPcg64 {
    /// Seed exactly as `np.random.default_rng(seed)` does for an integer seed:
    /// `SeedSequence(seed).generate_state(4, uint64)` split into a 128-bit
    /// initial state and 128-bit increment for PCG64.
    pub fn from_u64_seed(seed: u64) -> Self {
        // `_int_to_uint32_array`: little-endian uint32 words, lowest first;
        // zero-length inputs never occur here (seed 0 yields `[0]`).
        let mut entropy: Vec<u32> = Vec::with_capacity(2);
        if seed == 0 {
            entropy.push(0);
        } else {
            let mut n = seed;
            while n > 0 {
                entropy.push((n & 0xffff_ffff) as u32);
                n >>= 32;
            }
        }

        let pool = mix_entropy(&entropy);
        // generate_state(4, uint64) draws 8 uint32 words viewed little-endian
        // as four uint64 words.
        let words32 = generate_state(&pool, POOL_SIZE * 2);
        let mut words64 = [0u64; 4];
        for (j, w) in words64.iter_mut().enumerate() {
            *w = (words32[2 * j] as u64) | ((words32[2 * j + 1] as u64) << 32);
        }

        // pcg64_set_seed: seed = words64[0..2], inc = words64[2..4], each as
        // (hi<<64)|lo.
        let initstate = ((words64[0] as u128) << 64) | (words64[1] as u128);
        let initseq = ((words64[2] as u128) << 64) | (words64[3] as u128);

        // pcg_setseq_128_srandom_r.
        let mut g = NumpyPcg64 {
            state: 0,
            inc: (initseq << 1) | 1,
            has_uint32: false,
            uinteger: 0,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    /// One PCG64 LCG step: `state = state * MULT + inc` (mod 2^128).
    fn step(&mut self) {
        self.state = self
            .state
            .wrapping_mul(PCG_DEFAULT_MULTIPLIER_128)
            .wrapping_add(self.inc);
    }

    /// Draw the next raw 64-bit word, matching numpy's
    /// `bit_generator.random_raw` / `pcg_setseq_128_xsl_rr_64_random_r`:
    /// step, then XSL-RR 128->64 output.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        // pcg_output_xsl_rr_128_64: rotate-right of (high64 ^ low64) by the
        // top 6 bits (state >> 122).
        let rot = (self.state >> 122) as u32;
        let xored = ((self.state >> 64) as u64) ^ (self.state as u64);
        xored.rotate_right(rot)
    }

    /// Draw the next raw 32-bit word, matching numpy's `pcg64_next32`
    /// (`pcg64.h`): return the buffered high-32 word if present, otherwise draw
    /// a fresh 64-bit word, return its low 32 bits, and buffer the high 32 bits
    /// for the next call.
    pub(crate) fn next_u32(&mut self) -> u32 {
        if self.has_uint32 {
            self.has_uint32 = false;
            return self.uinteger;
        }
        let next = self.next_u64();
        self.has_uint32 = true;
        self.uinteger = (next >> 32) as u32;
        (next & 0xffff_ffff) as u32
    }

    /// Draw a uniform integer in `[0, max]` (inclusive) via numpy's masked
    /// rejection sampler `random_interval` (`distributions.c`): build the
    /// smallest `2^k - 1` mask `>= max`, then reject `next_u32() & mask` until
    /// it is `<= max`. Only the `max <= 0xffff_ffff` (uint32) branch is used,
    /// which is exactly the branch Fisher-Yates shuffle indices take.
    fn random_interval_u32(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        // Smallest bit mask >= max.
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= max {
                return value;
            }
        }
    }

    /// Reproduce `np.random.default_rng(seed).permutation(n)` exactly for an
    /// integer argument: `arange(n)` shuffled in place by numpy's Fisher-Yates
    /// (`_generator.pyx` `permutation` -> `shuffle` -> `_shuffle_raw`):
    /// `for i in reversed(1..n): j = random_interval(i); swap(x[i], x[j])`.
    ///
    /// Indices never exceed `n - 1`, so the draw stays on `random_interval`'s
    /// uint32 branch; `n` beyond `u32::MAX` is not supported (and never arises
    /// for dataset sampling).
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut x: Vec<usize> = (0..n).collect();
        self.shuffle(&mut x);
        x
    }

    /// In-place Fisher-Yates shuffle matching numpy's 1-d `shuffle` fast path
    /// (`_generator.pyx` `_shuffle_raw`): draws each swap partner with
    /// [`NumpyPcg64::random_interval_u32`]. `slice.len()` must fit in `u32`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        let n = x.len();
        for i in (1..n).rev() {
            let j = self.random_interval_u32(i as u32) as usize;
            x.swap(i, j);
        }
    }

    /// Draw the next double in [0, 1), matching numpy's `pcg64_double`:
    /// `(next_u64() >> 11) * 2^-53`.
    pub fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }

    /// Draw a uniform double in [lo, hi), matching
    /// `Generator.uniform(lo, hi)` for scalar finite bounds:
    /// `lo + (hi - lo) * next_double()`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_double()
    }
}

/// `hashmix` from `bit_generator.pyx`: mixes `value` into the running
/// `hash_const` (input-output) and returns the hashed word.
#[inline]
fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^= v >> XSHIFT;
    v
}

/// `mix` from `bit_generator.pyx`: combines two pool words.
#[inline]
fn mix(x: u32, y: u32) -> u32 {
    let mut result = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    result ^= result >> XSHIFT;
    result
}

/// `SeedSequence.mix_entropy`: fill and cross-mix a `POOL_SIZE` pool from the
/// assembled entropy words.
fn mix_entropy(entropy: &[u32]) -> [u32; POOL_SIZE] {
    let mut pool = [0u32; POOL_SIZE];
    let mut hash_const = INIT_A;

    // Fill the pool, running the hash out past the entropy length if needed.
    for (i, slot) in pool.iter_mut().enumerate() {
        let val = if i < entropy.len() { entropy[i] } else { 0 };
        *slot = hashmix(val, &mut hash_const);
    }

    // Cross-mix so late words affect earlier ones.
    for i_src in 0..POOL_SIZE {
        for i_dst in 0..POOL_SIZE {
            if i_src != i_dst {
                let h = hashmix(pool[i_src], &mut hash_const);
                pool[i_dst] = mix(pool[i_dst], h);
            }
        }
    }

    // Fold in any entropy beyond the pool size.
    for &word in entropy.iter().skip(POOL_SIZE) {
        for slot in pool.iter_mut() {
            let h = hashmix(word, &mut hash_const);
            *slot = mix(*slot, h);
        }
    }

    pool
}

/// `SeedSequence.generate_state`: produce `n_words` uint32 seed words by
/// cycling over the pool.
fn generate_state(pool: &[u32; POOL_SIZE], n_words: usize) -> Vec<u32> {
    let mut hash_const = INIT_B;
    let mut state = Vec::with_capacity(n_words);
    for i in 0..n_words {
        let mut data_val = pool[i % POOL_SIZE];
        data_val ^= hash_const;
        hash_const = hash_const.wrapping_mul(MULT_B);
        data_val = data_val.wrapping_mul(hash_const);
        data_val ^= data_val >> XSHIFT;
        state.push(data_val);
    }
    state
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_numpy_default_rng_uniform() {
        // Python (numpy 2.5.1):
        //   rng = np.random.default_rng(12345)
        //   [rng.uniform(0.0, 1.0) for _ in range(3)]
        //   -> 0.22733602246716966, 0.31675833970975287, 0.7973654573327341
        let mut g = NumpyPcg64::from_u64_seed(12345);
        let got = [
            g.uniform(0.0, 1.0),
            g.uniform(0.0, 1.0),
            g.uniform(0.0, 1.0),
        ];
        let expected: [u64; 3] = [
            0x3fcd_1958_c6d9_7fe0,
            0x3fd4_45c4_c572_7930,
            0x3fe9_8404_9046_889d,
        ];
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), *b);
        }
    }

    #[test]
    fn matches_numpy_random_raw_u64() {
        // Python (numpy 2.5.1):
        //   rng = np.random.default_rng(12345)
        //   rng.bit_generator.random_raw(3)
        //   -> 4193609425186963869, 5843160025838961886, 14708796524633321433
        let mut g = NumpyPcg64::from_u64_seed(12345);
        let got = [g.next_u64(), g.next_u64(), g.next_u64()];
        let expected: [u64; 3] = [
            4_193_609_425_186_963_869,
            5_843_160_025_838_961_886,
            14_708_796_524_633_321_433,
        ];
        assert_eq!(got, expected);
    }

    #[test]
    fn permutation_matches_numpy_default_rng() {
        // Python (numpy 2.5.1):
        //   np.random.default_rng(SEED).permutation(N)
        // Covers n=0,1,2, a prime (17), sizes >64 (100, 200) to exercise the
        // masked-rejection mask width, and multiple seeds.
        let cases: &[(u64, usize, &[usize])] = &[
            (12345, 0, &[]),
            (12345, 1, &[0]),
            (12345, 2, &[0, 1]),
            (12345, 5, &[4, 3, 0, 2, 1]),
            (12345, 8, &[4, 3, 0, 2, 1, 6, 7, 5]),
            (
                12345,
                17,
                &[11, 12, 10, 16, 6, 4, 1, 15, 7, 3, 8, 0, 2, 9, 5, 14, 13],
            ),
            (
                12345,
                100,
                &[
                    32, 6, 97, 84, 0, 44, 34, 60, 92, 19, 93, 73, 51, 47, 14, 59, 86, 83, 3, 58,
                    41, 54, 70, 69, 33, 7, 95, 30, 48, 12, 63, 23, 82, 72, 62, 28, 53, 87, 17, 80,
                    57, 26, 76, 31, 38, 1, 52, 96, 20, 81, 71, 11, 8, 74, 91, 43, 27, 77, 88, 40,
                    24, 56, 36, 85, 65, 35, 5, 9, 61, 55, 64, 50, 42, 78, 66, 90, 79, 22, 18, 4,
                    37, 75, 10, 15, 25, 45, 68, 49, 99, 39, 67, 98, 46, 16, 2, 89, 21, 94, 13, 29,
                ],
            ),
            (
                999,
                100,
                &[
                    47, 42, 57, 50, 97, 26, 49, 86, 48, 77, 35, 78, 19, 34, 63, 76, 68, 82, 10, 70,
                    55, 83, 17, 72, 12, 46, 94, 75, 43, 71, 9, 1, 33, 29, 99, 27, 74, 64, 31, 8,
                    20, 22, 66, 36, 5, 38, 52, 51, 53, 60, 37, 65, 93, 88, 32, 89, 84, 39, 23, 18,
                    41, 3, 81, 61, 85, 25, 7, 69, 15, 59, 24, 90, 30, 40, 73, 62, 54, 58, 2, 4, 45,
                    6, 56, 67, 87, 21, 16, 14, 79, 92, 0, 28, 13, 80, 96, 98, 44, 91, 95, 11,
                ],
            ),
            (
                7,
                200,
                &[
                    0, 43, 7, 144, 151, 21, 5, 2, 120, 71, 57, 100, 160, 156, 28, 159, 145, 192,
                    115, 164, 182, 20, 174, 55, 191, 103, 80, 54, 35, 173, 117, 4, 190, 65, 140,
                    108, 78, 41, 193, 163, 147, 36, 89, 45, 92, 118, 75, 93, 130, 49, 127, 58, 123,
                    9, 142, 198, 6, 170, 126, 40, 178, 102, 197, 12, 64, 152, 114, 96, 111, 70,
                    168, 172, 199, 195, 23, 106, 53, 32, 101, 177, 167, 51, 19, 83, 88, 194, 154,
                    166, 135, 62, 25, 10, 60, 74, 119, 125, 67, 1, 73, 39, 138, 16, 37, 87, 196,
                    15, 90, 68, 56, 155, 129, 61, 42, 176, 97, 46, 121, 91, 76, 44, 14, 50, 132,
                    24, 26, 150, 77, 48, 187, 18, 148, 104, 116, 22, 17, 134, 136, 175, 85, 29, 13,
                    105, 157, 180, 128, 34, 86, 185, 124, 84, 131, 79, 8, 158, 189, 113, 165, 47,
                    181, 11, 186, 27, 63, 109, 184, 137, 3, 110, 82, 59, 99, 183, 107, 161, 141,
                    179, 30, 81, 146, 31, 162, 133, 122, 72, 33, 94, 143, 38, 112, 52, 153, 69,
                    171, 66, 95, 98, 188, 149, 169, 139,
                ],
            ),
        ];
        for (seed, n, expected) in cases {
            let mut g = NumpyPcg64::from_u64_seed(*seed);
            assert_eq!(&g.permutation(*n), expected, "seed={seed} n={n}");
        }
    }

    #[test]
    fn shuffle_matches_permutation() {
        // `shuffle` on `arange(n)` is the fast path `permutation` uses.
        let mut a = NumpyPcg64::from_u64_seed(12345);
        let mut data: Vec<usize> = (0..8).collect();
        a.shuffle(&mut data);
        assert_eq!(data, vec![4, 3, 0, 2, 1, 6, 7, 5]);
    }

    #[test]
    fn seed_sequence_generate_state_matches_numpy() {
        // Python: np.random.SeedSequence(12345).generate_state(4)
        //   -> [0xa03d837c, 0xb5ae6482, 0xfa1f7a2f, 0xbbe2996f]
        let pool = mix_entropy(&[12345]);
        let state = generate_state(&pool, 4);
        assert_eq!(
            state,
            vec![0xa03d_837c, 0xb5ae_6482, 0xfa1f_7a2f, 0xbbe2_996f]
        );
    }
}
