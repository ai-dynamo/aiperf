// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NumPy (numpy.random), BSD-3-Clause / NCSA. See ATTRIBUTIONS.md.
//! Byte-exact `numpy.random.Generator` compatibility atop
//! the [`NumpyPcg64`] bit generator.
//!
//! Reproduces `np.random.default_rng(seed).<method>()` bit-for-bit for the
//! methods AIPerf's seeded dataset generators use: `random`, `standard_normal`,
//! `normal`, `lognormal`, `integers`, `bytes`, and weighted/uniform `choice`.
//! Algorithms are transcribed from the NumPy 1.26.4 C source
//! (`src/distributions/distributions.c`, `_generator.pyx`), cited per method;
//! the normal-ziggurat tables are extracted verbatim in
//! `crate::rng::compat::ziggurat_constants`. The golden vectors in this file's
//! tests come from numpy 2.5.1, which produces the same streams for these
//! methods.

use crate::rng::compat::numpy_pcg64::NumpyPcg64;
use crate::rng::compat::ziggurat_constants::{
    FI_DOUBLE, KI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R,
};
use crate::rng::error::{Result, RngError};

/// A numpy `Generator` (PCG64) reproduced bit-for-bit.
pub struct NumpyGenerator {
    bit: NumpyPcg64,
}

impl NumpyGenerator {
    /// Construct from an integer seed, exactly as `np.random.default_rng(seed)`.
    pub fn from_seed(seed: u64) -> Self {
        Self {
            bit: NumpyPcg64::from_u64_seed(seed),
        }
    }

    /// `random()` — a float64 in `[0, 1)` (`next_double`).
    pub fn random(&mut self) -> f64 {
        self.bit.next_double()
    }

    /// `random(size)` — `size` draws.
    pub fn random_batch(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.bit.next_double()).collect()
    }

    /// `shuffle(x)` — Fisher-Yates in place, delegating to the wrapped
    /// [`NumpyPcg64`] bit generator.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        self.bit.shuffle(x);
    }

    /// `standard_normal()` — the byte-exact normal ziggurat from
    /// `random_standard_normal` (`distributions.c`). 99.3% fast path.
    pub fn standard_normal(&mut self) -> f64 {
        loop {
            let r = self.bit.next_u64();
            let idx = (r & 0xff) as usize;
            let r = r >> 8;
            let sign = r & 0x1;
            let rabs = (r >> 1) & 0x000f_ffff_ffff_ffff;
            let mut x = rabs as f64 * WI_DOUBLE[idx];
            if sign & 0x1 == 1 {
                x = -x;
            }
            if rabs < KI_DOUBLE[idx] {
                return x; // 99.3% of the time
            }
            if idx == 0 {
                // Tail: sample from the exponential tail beyond R.
                loop {
                    let xx = -ZIGGURAT_NOR_INV_R * (-self.bit.next_double()).ln_1p();
                    let yy = -(-self.bit.next_double()).ln_1p();
                    if yy + yy > xx * xx {
                        return if (rabs >> 8) & 0x1 == 1 {
                            -(ZIGGURAT_NOR_R + xx)
                        } else {
                            ZIGGURAT_NOR_R + xx
                        };
                    }
                }
            } else if (FI_DOUBLE[idx - 1] - FI_DOUBLE[idx]) * self.bit.next_double()
                + FI_DOUBLE[idx]
                < (-0.5 * x * x).exp()
            {
                return x;
            }
        }
    }

    /// `normal(loc, scale)` = `loc + scale * standard_normal()`
    /// (`random_normal`).
    pub fn normal(&mut self, loc: f64, scale: f64) -> f64 {
        loc + scale * self.standard_normal()
    }

    /// `normal(loc, scale, size)` — `size` independent draws.
    pub fn normal_batch(&mut self, loc: f64, scale: f64, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.normal(loc, scale)).collect()
    }

    /// `lognormal(mean, sigma)` = `exp(normal(mean, sigma))`
    /// (`random_lognormal`).
    pub fn lognormal(&mut self, mean: f64, sigma: f64) -> f64 {
        self.normal(mean, sigma).exp()
    }

    /// `lognormal(mean, sigma, size)`.
    pub fn lognormal_batch(&mut self, mean: f64, sigma: f64, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.lognormal(mean, sigma)).collect()
    }

    /// Bounded uint32 via numpy's Lemire algorithm
    /// (`buffered_bounded_lemire_uint32`). `rng` is the inclusive span
    /// (`high - 1 - low` for an exclusive `high`), and must be `!= 0xFFFFFFFF`.
    fn bounded_lemire_u32(&mut self, rng: u32) -> u32 {
        let rng_excl = (rng as u64) + 1;
        let mut m = (self.bit.next_u32() as u64) * rng_excl;
        let mut leftover = m & 0xFFFF_FFFF;
        if leftover < rng_excl {
            let threshold = (u32::MAX as u64 - rng as u64) % rng_excl;
            while leftover < threshold {
                m = (self.bit.next_u32() as u64) * rng_excl;
                leftover = m & 0xFFFF_FFFF;
            }
        }
        (m >> 32) as u32
    }

    /// `integers(low, high)` (endpoint exclusive) — one draw in `[low, high)`.
    /// Uses `random_bounded_uint64_fill`'s `rng <= 0xFFFFFFFF` Lemire path,
    /// which is the branch AIPerf's small-range integer draws hit.
    pub fn integers(&mut self, low: i64, high: i64) -> i64 {
        let rng = (high - 1 - low) as u64; // inclusive span
        if rng == 0 {
            return low;
        }
        assert!(
            rng <= 0xFFFF_FFFF,
            "integers range must fit in u32 for NumPy-compatible sampling"
        );
        if rng == 0xFFFF_FFFF {
            return low + self.bit.next_u32() as i64;
        }
        low + self.bounded_lemire_u32(rng as u32) as i64
    }

    /// `integers(low, high, size)` — `size` independent draws.
    pub fn integers_batch(&mut self, low: i64, high: i64, size: usize) -> Vec<i64> {
        (0..size).map(|_| self.integers(low, high)).collect()
    }

    /// `bytes(length)` — `ceil(length/4)` raw uint32 words written little-endian,
    /// truncated to `length` (`integers(0, 2**32, n_uint32, uint32).tobytes()`;
    /// that range hits the `rng == 0xFFFFFFFF` raw-`next_uint32` path).
    pub fn bytes(&mut self, length: usize) -> Vec<u8> {
        if length == 0 {
            return Vec::new();
        }
        let n_uint32 = (length - 1) / 4 + 1;
        let mut out = Vec::with_capacity(n_uint32 * 4);
        for _ in 0..n_uint32 {
            out.extend_from_slice(&self.bit.next_u32().to_le_bytes());
        }
        out.truncate(length);
        out
    }

    /// `choice(n, p=weights)` (replace=True, size=None) — one weighted index in
    /// `[0, n)` using `cdf = cumsum(p); cdf /= cdf[-1];
    /// idx = cdf.searchsorted(random(), side='right')`.
    pub fn choice_weighted(&mut self, weights: &[f64]) -> usize {
        let cdf = normalized_cdf(weights);
        searchsorted_right(&cdf, self.bit.next_double())
    }

    /// `choice(pop_size, size)` (replace=True, p=None) — uniform indices via
    /// `integers(0, pop_size, size)`.
    pub fn choice_uniform(&mut self, pop_size: i64, size: usize) -> Vec<i64> {
        (0..size).map(|_| self.integers(0, pop_size)).collect()
    }

    /// `choice(n, size, p=weights)` (replace=True) — `size` weighted indices,
    /// one shared cdf, one `next_double()` draw per index in order (matches
    /// `uniform_samples = self.random(shape); idx =
    /// cdf.searchsorted(uniform_samples, side='right')`).
    pub fn choice_weighted_batch(&mut self, weights: &[f64], size: usize) -> Vec<usize> {
        let cdf = normalized_cdf(weights);
        (0..size)
            .map(|_| searchsorted_right(&cdf, self.bit.next_double()))
            .collect()
    }

    /// `choice(n, size, replace=False, p=weights)` — `size` distinct weighted
    /// indices, `_generator.pyx`'s rejection-refill loop: each round draws
    /// `size - n_uniq` uniforms against the cdf of the still-unselected
    /// weights (previously selected entries zeroed), keeps only the values
    /// that are the first occurrence of their index within the round
    /// (`np.unique(..., return_index=True)` + position-sort), and repeats
    /// until `size` distinct indices are collected.
    pub fn choice_weighted_without_replacement(
        &mut self,
        weights: &[f64],
        size: usize,
    ) -> Result<Vec<usize>> {
        let nonzero = weights.iter().filter(|&&w| w > 0.0).count();
        if nonzero < size {
            return Err(RngError::InvalidWeights {
                reason: "fewer non-zero entries in p than size",
            });
        }
        let mut p = weights.to_vec();
        let mut found = Vec::with_capacity(size);
        while found.len() < size {
            let batch = size - found.len();
            let draws = self.random_batch(batch);
            for &idx in &found {
                p[idx] = 0.0;
            }
            let cdf = normalized_cdf(&p);
            let mut seen = std::collections::HashSet::new();
            for &x in &draws {
                let idx = searchsorted_right(&cdf, x);
                if seen.insert(idx) {
                    found.push(idx);
                    if found.len() == size {
                        break;
                    }
                }
            }
        }
        Ok(found)
    }
}

/// `cdf = cumsum(weights); cdf /= cdf[-1]`.
fn normalized_cdf(weights: &[f64]) -> Vec<f64> {
    let mut cdf: Vec<f64> = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        cdf.push(acc);
    }
    let total = *cdf.last().expect("non-empty weights");
    for c in &mut cdf {
        *c /= total;
    }
    cdf
}

/// `cdf.searchsorted(x, side='right')`: first index with `cdf[idx] > x`,
/// clamped to the last index (numpy's cdf always ends at exactly `1.0`).
fn searchsorted_right(cdf: &[f64], x: f64) -> usize {
    cdf.iter().position(|&c| c > x).unwrap_or(cdf.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden vectors captured from numpy 2.5.1: np.random.default_rng(42).
    #[test]
    fn random_matches_numpy() {
        let mut g = NumpyGenerator::from_seed(42);
        let got = [g.random(), g.random(), g.random()];
        assert_eq!(got[0].to_bits(), 0.773_956_048_555_963_3_f64.to_bits());
        assert_eq!(got[1].to_bits(), 0.438_878_439_752_052_3_f64.to_bits());
        assert_eq!(got[2].to_bits(), 0.858_597_919_911_382_5_f64.to_bits());
    }

    #[test]
    fn standard_normal_matches_numpy() {
        let mut g = NumpyGenerator::from_seed(42);
        let got = [
            g.standard_normal(),
            g.standard_normal(),
            g.standard_normal(),
        ];
        assert_eq!(got[0].to_bits(), 0.304_717_079_754_431_35_f64.to_bits());
        assert_eq!(got[1].to_bits(), (-1.039_984_106_240_495_5_f64).to_bits());
        assert_eq!(got[2].to_bits(), 0.750_451_195_806_457_2_f64.to_bits());
    }

    #[test]
    fn integers_matches_numpy() {
        let mut g = NumpyGenerator::from_seed(42);
        let got: Vec<i64> = (0..5).map(|_| g.integers(1, 100)).collect();
        assert_eq!(got, [9, 77, 65, 44, 43]);
    }

    #[test]
    fn bytes_matches_numpy() {
        let mut g = NumpyGenerator::from_seed(42);
        assert_eq!(hex(&g.bytes(8)), "8826d916cdfb21c6");
    }

    #[test]
    fn choice_uniform_matches_numpy() {
        // np.random.default_rng(42).choice([10,20,30,40],3) -> [10,40,30] (idx 0,3,2)
        let mut g = NumpyGenerator::from_seed(42);
        assert_eq!(g.choice_uniform(4, 3), [0, 3, 2]);
    }

    #[test]
    fn choice_weighted_matches_numpy() {
        // np.random.default_rng(42).choice(5, p=[.1,.2,.3,.25,.15]) x8
        let mut g = NumpyGenerator::from_seed(42);
        let w = [0.1, 0.2, 0.3, 0.25, 0.15];
        let got: Vec<usize> = (0..8).map(|_| g.choice_weighted(&w)).collect();
        assert_eq!(got, [3, 2, 4, 3, 0, 4, 3, 3]);
    }

    #[test]
    fn lognormal_matches_numpy() {
        let mut g = NumpyGenerator::from_seed(42);
        let got = [
            g.lognormal(0.0, 1.0),
            g.lognormal(0.0, 1.0),
            g.lognormal(0.0, 1.0),
        ];
        assert_eq!(got[0].to_bits(), 1.356_241_240_616_863_6_f64.to_bits());
        assert_eq!(got[1].to_bits(), 0.353_460_299_727_134_55_f64.to_bits());
        assert_eq!(got[2].to_bits(), 2.117_955_413_661_803_3_f64.to_bits());
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }
}
