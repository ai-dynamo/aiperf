// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hash-ID-scoped RNG for order-independent parallel trace synthesis.
//!
//! This ports the canonical Python
//! `aiperf-graph-ir/src/aiperf/common/hash_id_random_generator.py` semantics: a
//! base seed is preserved without consuming state when present, seed `0` is legal,
//! and each `(trace_id, hash_id)` pair deterministically reseeds the inner
//! generator so worker scheduling cannot perturb generated content.

use std::ops::{Deref, DerefMut};

use crate::derive::derive_seed_parts;
use crate::generator::RandomGenerator;

/// Random generator that reseeds per `(trace_id, hash_id)` scope.
#[derive(Clone, Debug)]
pub struct HashIdRandomGenerator {
    base_seed: u64,
    trace_id: String,
    generator: RandomGenerator,
}

impl HashIdRandomGenerator {
    /// Build a hash-id generator from a base generator.
    ///
    /// If `base` has a deterministic seed, that seed is read without consuming RNG
    /// state, including `Some(0)`. If `base` is seedless, this consumes one `u64`
    /// fallback seed and makes the hash-id generator deterministic from there.
    pub fn from_base(base: &mut RandomGenerator) -> Self {
        let base_seed = base.seed().unwrap_or_else(|| base.random_u64());
        Self {
            base_seed,
            trace_id: String::new(),
            generator: RandomGenerator::from_seed(Some(base_seed)),
        }
    }

    /// Return the base seed used in hash-id derivation.
    pub const fn base_seed(&self) -> u64 {
        self.base_seed
    }

    /// Set the instance trace scope used when `reseed_for_hash_id` gets no override.
    pub fn set_trace_id(&mut self, trace_id: impl Into<String>) {
        self.trace_id = trace_id.into();
    }

    /// Return the current instance trace scope.
    pub fn trace_id(&self) -> &str {
        &self.trace_id
    }

    /// Reseed the inner generator for `hash_id` in the selected trace scope.
    ///
    /// `trace_id_override` applies only to this call and does not mutate the
    /// instance scope. `None` uses the scope set by [`Self::set_trace_id`]; the
    /// default empty string is the global namespace used by content-global paths.
    pub fn reseed_for_hash_id(&mut self, hash_id: i64, trace_id_override: Option<&str>) {
        let scope = trace_id_override.unwrap_or(&self.trace_id);
        let mut seed_buf = itoa::Buffer::new();
        let mut hash_buf = itoa::Buffer::new();
        let seed = derive_seed_parts(&[
            seed_buf.format(self.base_seed).as_bytes(),
            b":",
            scope.as_bytes(),
            b":",
            hash_buf.format(hash_id).as_bytes(),
        ]);
        self.generator.reseed(seed);
    }

    /// Borrow the inner generator.
    pub const fn generator(&self) -> &RandomGenerator {
        &self.generator
    }

    /// Mutably borrow the inner generator.
    pub fn generator_mut(&mut self) -> &mut RandomGenerator {
        &mut self.generator
    }
}

impl Deref for HashIdRandomGenerator {
    type Target = RandomGenerator;

    fn deref(&self) -> &Self::Target {
        &self.generator
    }
}

impl DerefMut for HashIdRandomGenerator {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.generator
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::derive::derive_seed_u64;

    #[test]
    fn from_base_preserves_seed_zero() {
        let mut base = RandomGenerator::from_seed(Some(0));
        let derived = HashIdRandomGenerator::from_base(&mut base);
        assert_eq!(derived.base_seed(), 0);
        assert_eq!(derived.generator().seed(), Some(0));
    }

    #[test]
    fn from_base_seeded_does_not_consume_base_state() {
        let mut base = RandomGenerator::from_seed(Some(0));
        let mut fresh = RandomGenerator::from_seed(Some(0));
        let _derived = HashIdRandomGenerator::from_base(&mut base);
        assert_eq!(base.random_u64(), fresh.random_u64());
    }

    #[test]
    fn same_trace_and_hash_id_reproduce_sequences() {
        let mut base_a = RandomGenerator::from_seed(Some(42));
        let mut base_b = RandomGenerator::from_seed(Some(42));
        let mut a = HashIdRandomGenerator::from_base(&mut base_a);
        let mut b = HashIdRandomGenerator::from_base(&mut base_b);

        a.reseed_for_hash_id(7, Some("trace"));
        b.reseed_for_hash_id(7, Some("trace"));

        let seq_a: Vec<_> = (0..20)
            .map(|_| a.randrange(0, 1_000_000, 1).unwrap())
            .collect();
        let seq_b: Vec<_> = (0..20)
            .map(|_| b.randrange(0, 1_000_000, 1).unwrap())
            .collect();
        assert_eq!(seq_a, seq_b);
    }

    #[test]
    fn trace_scope_changes_hash_id_seed() {
        let mut base = RandomGenerator::from_seed(Some(11));
        let mut hash_rng = HashIdRandomGenerator::from_base(&mut base);

        hash_rng.set_trace_id("trace-a");
        hash_rng.reseed_for_hash_id(5, None);
        let a = hash_rng.random_u64();
        hash_rng.reseed_for_hash_id(5, Some("trace-b"));
        let b = hash_rng.random_u64();
        hash_rng.reseed_for_hash_id(5, None);
        let a_again = hash_rng.random_u64();

        assert_ne!(a, b);
        assert_eq!(a, a_again);
        assert_eq!(hash_rng.trace_id(), "trace-a");
    }

    #[test]
    fn seedless_base_draws_fallback_seed() {
        let mut base = RandomGenerator::from_seed(None);
        let derived = HashIdRandomGenerator::from_base(&mut base);
        assert_eq!(derived.generator().seed(), Some(derived.base_seed()));
    }

    #[test]
    fn reseed_uses_same_bytes_as_colon_joined_key() {
        let mut base = RandomGenerator::from_seed(Some(123));
        let mut hash_rng = HashIdRandomGenerator::from_base(&mut base);
        hash_rng.reseed_for_hash_id(-9, Some("scope"));
        assert_eq!(
            hash_rng.generator().seed(),
            Some(derive_seed_u64("123:scope:-9"))
        );
    }
}
