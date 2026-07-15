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
//!     `pcg_output_xsl_rr_128_64`, and `PCG_DEFAULT_MULTIPLIER_128`.
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
