// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Portions derived from CPython (Lib/random.py, Modules/_randommodule.c), PSF License. See ATTRIBUTIONS.md.

//! Byte-exact CPython `random.Random` Mersenne Twister.
//!
//! AgentX random sampling draws with replacement via
//! `random.Random(seed).choice(ids)`. NumPy-compatible PCG64
//! ([`crate::rng::compat::numpy_pcg64`]) drives the shuffle/`t*` draws; the with-replacement
//! `RandomSampler` uses CPython's stdlib `random` instead, a DIFFERENT algorithm,
//! so it needs its own byte-exact implementation.
//!
//! Source reference: CPython 3.12 (stream-stable across 3.x):
//!   - `Modules/_randommodule.c`: `init_genrand`, `init_by_array`,
//!     `genrand_uint32`, and integer seeding in `random_seed` (absolute value
//!     split into little-endian 32-bit key words, then `init_by_array`).
//!   - `Lib/random.py`: `Random.getrandbits` (for `k <= 32`,
//!     `genrand_uint32() >> (32 - k)`), `Random._randbelow_with_getrandbits`, and
//!     `Random.choice` (`seq[_randbelow(len(seq))]`).
//!
//! Committed golden vectors pin the stream.

use crate::rng::error::{Result, RngError};

/// `math.log(4.0)`, CPython `random.LOG4` — Cheng's-method constant in
/// [`PythonMt19937::gammavariate`]'s `alpha > 1.0` branch.
const LOG4: f64 = 1.386_294_361_119_890_6;
/// `1.0 + math.log(4.5)`, CPython `random.SG_MAGICCONST` — the same branch's
/// acceptance-test constant.
const SG_MAGICCONST: f64 = 2.504_077_396_776_274;
/// `2.0 * math.pi`, CPython `random.TWOPI` — [`PythonMt19937::gauss`]'s
/// Box-Muller angle scale.
const TWOPI: f64 = 6.283_185_307_179_586;

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
    /// `random.Random.gauss_next`: the cached second Box-Muller variate from the
    /// previous [`PythonMt19937::gauss`] call, consumed (and cleared) by the next
    /// one. `None` means the next call must draw a fresh pair.
    gauss_next: Option<f64>,
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
            gauss_next: None,
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

    /// `Random.gauss(mu, sigma)`: the cached two-variable Box-Muller transform.
    ///
    /// Each pair of draws yields two independent standard-normal variates; the
    /// second is cached in [`Self::gauss_next`] and returned (scaled) by the
    /// following call instead of drawing a fresh pair, so the two calls together
    /// consume exactly two [`Self::random`] draws.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let z = match self.gauss_next.take() {
            Some(z) => z,
            None => {
                let x2pi = self.random() * TWOPI;
                let g2rad = (-2.0 * (1.0 - self.random()).ln()).sqrt();
                let z = x2pi.cos() * g2rad;
                self.gauss_next = Some(x2pi.sin() * g2rad);
                z
            }
        };
        mu + z * sigma
    }

    /// `Random.expovariate(lambd)`: `-ln(1 - random()) / lambd`.
    pub fn expovariate(&mut self, lambd: f64) -> f64 {
        -(1.0 - self.random()).ln() / lambd
    }

    /// `Random.gammavariate(alpha, beta)`: Cheng's method (`alpha > 1.0`), the
    /// exponential shortcut (`alpha == 1.0`), or Ahrens-Dieter algorithm GS
    /// (`0 < alpha < 1.0`).
    pub fn gammavariate(&mut self, alpha: f64, beta: f64) -> Result<f64> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(RngError::InvalidParameter {
                what: "gammavariate: alpha and beta must be > 0.0",
                value: if alpha <= 0.0 { alpha } else { beta },
            });
        }
        if alpha > 1.0 {
            let ainv = (2.0 * alpha - 1.0).sqrt();
            let bbb = alpha - LOG4;
            let ccc = alpha + ainv;
            loop {
                let u1 = self.random();
                if !(1e-7 < u1 && u1 < 0.9999999) {
                    continue;
                }
                let u2 = 1.0 - self.random();
                let v = (u1 / (1.0 - u1)).ln() / ainv;
                let x = alpha * v.exp();
                let z = u1 * u1 * u2;
                let r = bbb + ccc * v - x;
                if r + SG_MAGICCONST - 4.5 * z >= 0.0 || r >= z.ln() {
                    return Ok(x * beta);
                }
            }
        } else if alpha == 1.0 {
            return Ok(-(1.0 - self.random()).ln() * beta);
        }
        // 0 < alpha < 1.0: Ahrens-Dieter algorithm GS.
        let e = std::f64::consts::E;
        loop {
            let u = self.random();
            let b = (e + alpha) / e;
            let p = b * u;
            let x = if p <= 1.0 {
                p.powf(1.0 / alpha)
            } else {
                -((b - p) / alpha).ln()
            };
            let u1 = self.random();
            if p > 1.0 {
                if u1 <= x.powf(alpha - 1.0) {
                    return Ok(x * beta);
                }
            } else if u1 <= (-x).exp() {
                return Ok(x * beta);
            }
        }
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
            "/tests/data/random_sampler_vectors.json"
        );
        let raw = std::fs::read_to_string(path).expect("read CPython random sampler vectors");
        serde_json::from_str(&raw).expect("parse CPython random sampler vectors")
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

    /// Parse a Rust hex-float literal (`0x1.35517978b5bf3p-2`,
    /// `-0x1.65eb880d86c3cp-2`) into its `f64` value via `f64::from_bits`,
    /// avoiding any decimal round-trip. Golden values were captured with
    /// `float.hex()` from a local CPython 3.12.10 interpreter
    /// (`python3 -c "import random; ..."`), which emits this same syntax.
    fn hexf(s: &str) -> f64 {
        let (neg, s) = match s.strip_prefix('-') {
            Some(rest) => (true, rest),
            None => (false, s),
        };
        let s = s.strip_prefix("0x").expect("hex float prefix");
        let (mantissa, exp) = s.split_once('p').expect("hex float exponent");
        let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
        let int_val = u64::from_str_radix(int_part, 16).expect("hex int part");
        let mut frac_val = 0.0f64;
        let mut scale = 1.0f64 / 16.0;
        for c in frac_part.chars() {
            frac_val += (c.to_digit(16).expect("hex frac digit") as f64) * scale;
            scale /= 16.0;
        }
        let exp_val: i32 = exp.parse().expect("hex exponent");
        let value = (int_val as f64 + frac_val) * 2f64.powi(exp_val);
        if neg { -value } else { value }
    }

    #[test]
    fn gauss_matches_cpython() {
        // python3 -c "import random; r = random.Random(999888777);
        //   print([r.gauss(0.0, 1.0).hex() for _ in range(8)])"
        let expected = [
            "0x1.35517978b5bf3p-2",
            "-0x1.65eb880d86c3cp-2",
            "0x1.c81db64ee9223p-1",
            "-0x1.4f8c93553eee2p+0",
            "0x1.cc41c0f5400b5p-1",
            "0x1.f5c4d0bc36ceep-3",
            "0x1.1e43d46feed4cp-1",
            "0x1.84209d0647111p-2",
        ];
        let mut r = PythonMt19937::from_u64_seed(999_888_777);
        for (i, hex) in expected.iter().enumerate() {
            assert_eq!(r.gauss(0.0, 1.0).to_bits(), hexf(hex).to_bits(), "draw {i}");
        }
    }

    #[test]
    fn expovariate_matches_cpython() {
        // python3 -c "import random; r = random.Random(12345);
        //   print([r.expovariate(2.0).hex() for _ in range(5)])"
        let expected = [
            "0x1.13ecd5daa2246p-2",
            "0x1.4eede17bc72fep-8",
            "0x1.be809e3475578p-1",
            "0x1.6b3f54020f85bp-3",
            "0x1.d68bc1b7451d2p-3",
        ];
        let mut r = PythonMt19937::from_u64_seed(12345);
        for (i, hex) in expected.iter().enumerate() {
            assert_eq!(
                r.expovariate(2.0).to_bits(),
                hexf(hex).to_bits(),
                "draw {i}"
            );
        }
    }

    #[test]
    fn gammavariate_alpha_gt_one_matches_cpython() {
        // python3 -c "import random; r = random.Random(42424242);
        //   print([r.gammavariate(3.5, 2.0).hex() for _ in range(5)])"
        let expected = [
            "0x1.87a957e064260p+3",
            "0x1.ad97651fe88c8p+3",
            "0x1.8e7971d23cea5p+0",
            "0x1.3355354c9df94p+3",
            "0x1.318a44c171fbbp+2",
        ];
        let mut r = PythonMt19937::from_u64_seed(42_424_242);
        for (i, hex) in expected.iter().enumerate() {
            assert_eq!(
                r.gammavariate(3.5, 2.0).unwrap().to_bits(),
                hexf(hex).to_bits(),
                "draw {i}"
            );
        }
    }

    #[test]
    fn gammavariate_alpha_eq_one_matches_cpython() {
        // python3 -c "import random; r = random.Random(7);
        //   print([r.gammavariate(1.0, 1.5).hex() for _ in range(5)])"
        let expected = [
            "0x1.2c87a0ff48909p-1",
            "0x1.f65425ad92286p-3",
            "0x1.9428877c32687p+0",
            "0x1.cdfd9c43ddb52p-4",
            "0x1.26c3c4ae9cb51p+0",
        ];
        let mut r = PythonMt19937::from_u64_seed(7);
        for (i, hex) in expected.iter().enumerate() {
            assert_eq!(
                r.gammavariate(1.0, 1.5).unwrap().to_bits(),
                hexf(hex).to_bits(),
                "draw {i}"
            );
        }
    }

    #[test]
    fn gammavariate_alpha_lt_one_matches_cpython() {
        // python3 -c "import random; r = random.Random(31415);
        //   print([r.gammavariate(0.5, 1.0).hex() for _ in range(5)])"
        let expected = [
            "0x1.f39ffb7f45de7p-2",
            "0x1.4aa3d348ae418p+0",
            "0x1.8cd4e298e4b9cp-2",
            "0x1.7dd94ed2820ebp-1",
            "0x1.4021cab812a56p-2",
        ];
        let mut r = PythonMt19937::from_u64_seed(31415);
        for (i, hex) in expected.iter().enumerate() {
            assert_eq!(
                r.gammavariate(0.5, 1.0).unwrap().to_bits(),
                hexf(hex).to_bits(),
                "draw {i}"
            );
        }
    }

    #[test]
    fn gammavariate_rejects_non_positive_parameters() {
        let mut r = PythonMt19937::from_u64_seed(0);
        assert!(r.gammavariate(0.0, 1.0).is_err());
        assert!(r.gammavariate(1.0, 0.0).is_err());
        assert!(r.gammavariate(-1.0, 1.0).is_err());
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
