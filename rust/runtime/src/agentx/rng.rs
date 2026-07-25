// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-exact port of Python `HashIdRandomGenerator`
//! (`src/aiperf/common/hash_id_random_generator.py`).
//!
//! Each `(trace_id, hash_id)` pair re-seeds a CPython Mersenne-Twister
//! deterministically, so parallel workers produce identical content for the same
//! hash id regardless of processing order. The Python derivation is
//! `sha256(f"{seed}:{trace_id}:{hash_id}")`, first 8 bytes big-endian, fed to
//! `random.Random.seed(int)`. That is exactly
//! [`PythonRandomGenerator::derive_child_seed`] with the identifier
//! `"{trace_id}:{hash_id}"`, so this type reuses the runtime's CPython-MT compat
//! rather than reimplementing the hash/seed path.

use crate::rng::compat::python_mt::PythonMt19937;
use crate::rng::compat::python_random::PythonRandomGenerator;

/// Deterministic per-`(trace_id, hash_id)` RNG, matching Python
/// `HashIdRandomGenerator`.
///
/// NOT thread-safe (mirrors the Python contract): each worker owns an instance.
/// The base `seed` and `trace_id` are fixed for a trace; `reseed_for_hash_id`
/// installs a fresh CPython-MT stream before generating that hash id's tokens.
// Not `Clone`/`Debug`: the underlying `PythonMt19937` is neither, and the Python
// generator is likewise a single-owner mutable stream.
pub struct HashIdRandomGenerator {
    seed: u64,
    trace_id: String,
    mt: PythonMt19937,
}

impl HashIdRandomGenerator {
    /// Construct from a base seed (typically `rng.derive()`'s child seed). The
    /// generator is unusable until [`Self::reseed_for_hash_id`] installs a
    /// per-hash-id stream, matching the Python object's post-init state.
    pub fn new(base_seed: u64) -> Self {
        Self {
            seed: base_seed,
            trace_id: String::new(),
            // Placeholder stream; replaced on the first reseed_for_hash_id.
            mt: PythonMt19937::from_u64_seed(base_seed),
        }
    }

    /// Scope hash ids to a specific trace file. Different trace files MUST use
    /// different `trace_id`s so overlapping `hash_id` values produce distinct
    /// content (Python `set_trace_id`).
    pub fn set_trace_id(&mut self, trace_id: impl Into<String>) {
        self.trace_id = trace_id.into();
    }

    /// Re-seed deterministically for `hash_id`. After this, all draws use the
    /// derived CPython-MT stream until the next call (Python
    /// `reseed_for_hash_id`).
    pub fn reseed_for_hash_id(&mut self, hash_id: i64) {
        let identifier = format!("{}:{}", self.trace_id, hash_id);
        let derived = PythonRandomGenerator::derive_child_seed(self.seed, &identifier);
        self.mt = PythonMt19937::from_u64_seed(derived);
    }

    /// `random.Random.randrange(stop)` — uniform int in `[0, stop)`.
    ///
    /// Panics if `stop <= 0`, matching `randrange`'s empty-range error (callers
    /// never pass a non-positive block/tail length).
    pub fn randrange(&mut self, stop: i64) -> i64 {
        assert!(stop > 0, "randrange stop must be positive, got {stop}");
        self.mt.randbelow(stop as u64) as i64
    }

    /// `random.Random.random()` — uniform float in `[0, 1)`.
    pub fn random(&mut self) -> f64 {
        self.mt.random()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values captured from CPython:
    /// `d = int.from_bytes(sha256(f"{seed}:{tid}:{hid}").digest()[:8], "big")`,
    /// `r = random.Random(); r.seed(d); [r.randrange(1000) for _ in range(5)]`.
    #[test]
    fn matches_cpython_hash_id_random_generator() {
        let cases: &[(u64, &str, i64, [i64; 5])] = &[
            (42, "t", 7, [366, 486, 241, 501, 226]),
            (1234567890, "trace_0012", 99999, [303, 268, 444, 404, 995]),
            (0, "", 0, [672, 999, 375, 166, 753]),
        ];
        for &(seed, tid, hid, expected) in cases {
            let mut g = HashIdRandomGenerator::new(seed);
            g.set_trace_id(tid);
            g.reseed_for_hash_id(hid);
            let got: [i64; 5] = std::array::from_fn(|_| g.randrange(1000));
            assert_eq!(got, expected, "seed={seed} tid={tid:?} hid={hid}");
        }
    }

    /// Re-seeding for the same hash id reproduces the same stream (parallel-safe).
    #[test]
    fn reseed_is_idempotent_per_hash_id() {
        let mut g = HashIdRandomGenerator::new(42);
        g.set_trace_id("t");
        g.reseed_for_hash_id(7);
        let first: [i64; 3] = std::array::from_fn(|_| g.randrange(1000));
        g.reseed_for_hash_id(7);
        let second: [i64; 3] = std::array::from_fn(|_| g.randrange(1000));
        assert_eq!(first, second);
    }
}
