// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hash-derived seed algebra for order-independent random streams.
//!
//! This ports the derivation semantics from Python
//! `src/aiperf/common/random_generator.py`: components name their stream and the
//! child seed depends only on `(root_seed, identifier)`. The Rust port deliberately
//! uses BLAKE3 rather than Python's SHA-256 because the project does not require
//! cross-language byte parity; the stable contract is BLAKE3's first eight digest
//! bytes interpreted as a big-endian `u64`.

use crate::generator::RandomGenerator;

/// Root seed for a reproducible run.
///
/// `Some(seed)` produces deterministic child streams. `None` makes every derived
/// generator seed from OS/thread entropy, matching Python's seedless pass-through
/// semantics without a global singleton.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RngRoot(pub Option<u64>);

impl RngRoot {
    /// Create a root from an optional seed.
    pub const fn new(seed: Option<u64>) -> Self {
        Self(seed)
    }

    /// Return the underlying optional root seed.
    pub const fn seed(self) -> Option<u64> {
        self.0
    }

    /// Derive an owned generator for `identifier`.
    ///
    /// Seeded roots produce a fresh deterministic generator whose stream depends
    /// only on `(root, identifier)`. Seedless roots produce a fresh entropy-seeded
    /// generator. Creating or drawing from any other child cannot perturb it.
    pub fn derive(self, identifier: &str) -> RandomGenerator {
        RandomGenerator::from_seed(self.derive_seed(identifier))
    }

    /// Derive the deterministic child seed for `identifier`.
    ///
    /// Returns `None` when this root is seedless. For seeded roots, this is the
    /// BLAKE3 port of Python `_RNGManager.derive`: hash the UTF-8 bytes of
    /// `"{root}:{identifier}"` and read the first eight digest bytes as a
    /// big-endian `u64`.
    pub fn derive_seed(self, identifier: &str) -> Option<u64> {
        self.0.map(|root| {
            let mut root_buf = itoa::Buffer::new();
            derive_seed_parts(&[
                root_buf.format(root).as_bytes(),
                b":",
                identifier.as_bytes(),
            ])
        })
    }

    /// Derive an adaptive-sweep variation seed for `label`.
    ///
    /// Mirrors Python `derive_variation_seed(root, label)` with the BLAKE3 hash
    /// selected by the Rust RNG spec: `"{root}:variation:{label}"`.
    pub fn derive_variation_seed(self, label: &str) -> Option<u64> {
        self.0.map(|root| {
            let mut root_buf = itoa::Buffer::new();
            derive_seed_parts(&[
                root_buf.format(root).as_bytes(),
                b":variation:",
                label.as_bytes(),
            ])
        })
    }
}

/// Derive a `u64` seed from one UTF-8 key string.
///
/// The function is the shared primitive: BLAKE3 digest, first eight bytes,
/// big-endian. Changing this reshuffles every deterministic stream.
pub fn derive_seed_u64(key: &str) -> u64 {
    derive_seed_parts(&[key.as_bytes()])
}

/// Derive a `u64` seed from byte slices without allocating an intermediate key.
///
/// Concatenating the parts must produce exactly the same bytes as a single string
/// passed to [`derive_seed_u64`]. This is used on the hot hash-id reseed path.
pub fn derive_seed_parts(parts: &[&[u8]]) -> u64 {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut first = [0_u8; 8];
    first.copy_from_slice(&digest.as_bytes()[..8]);
    u64::from_be_bytes(first)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::namespace;

    #[test]
    fn derive_seed_matches_spec_vectors() {
        assert_eq!(
            RngRoot(Some(42)).derive_seed(namespace::DATASET_LOADER),
            Some(2_466_643_113_772_406_410)
        );
        assert_eq!(
            RngRoot(Some(42)).derive_seed(namespace::TIMING_REQUEST_RATE),
            Some(12_613_212_627_144_784_801)
        );
        assert_eq!(
            RngRoot(Some(42)).derive_seed(""),
            Some(1_788_878_741_536_589_501)
        );
        assert_eq!(
            RngRoot(Some(0)).derive_seed("a"),
            Some(10_339_543_760_652_402_899)
        );
        assert_eq!(
            RngRoot(Some(42)).derive_variation_seed("concurrency=4"),
            Some(10_717_291_070_465_836_476)
        );
    }

    #[test]
    fn seedless_root_keeps_derived_streams_seedless() {
        let root = RngRoot(None);
        assert_eq!(root.derive_seed(namespace::DATASET_LOADER), None);
        assert_eq!(root.derive_variation_seed("concurrency=4"), None);
    }

    #[test]
    fn root_accessors_and_derived_generators_preserve_isolation() {
        let root = RngRoot::new(Some(42));
        assert_eq!(root.seed(), Some(42));

        let mut first = root.derive(namespace::DATASET_PROMPT_LENGTH);
        let mut second = root.derive(namespace::DATASET_PROMPT_LENGTH);
        let mut unrelated = root.derive(namespace::DATASET_AUDIO_DURATION);
        assert_eq!(
            first.seed(),
            root.derive_seed(namespace::DATASET_PROMPT_LENGTH)
        );
        assert_eq!(first.random_u64(), second.random_u64());
        let first_next = first.random_u64();
        let _ = unrelated.random_u64();
        assert_eq!(first_next, second.random_u64());
    }

    #[test]
    fn seedless_derive_creates_independent_entropy_streams() {
        let mut first = RngRoot::new(None).derive("first");
        let mut second = RngRoot::new(None).derive("second");
        assert_eq!(first.seed(), None);
        assert_eq!(second.seed(), None);
        let first_draws = [first.random_u64(), first.random_u64()];
        let second_draws = [second.random_u64(), second.random_u64()];
        assert_ne!(first_draws, second_draws);
    }

    #[test]
    fn parts_match_single_key_hash() {
        assert_eq!(
            derive_seed_parts(&[b"42", b":", b"trace", b":", b"7"]),
            derive_seed_u64("42:trace:7")
        );
        assert_eq!(derive_seed_parts(&[]), derive_seed_u64(""));
        assert_eq!(
            derive_seed_parts(&[b"binary\0", &[0xff, 0x01]]),
            derive_seed_parts(&[b"binary\0\xff\x01"])
        );
    }

    #[test]
    fn derivation_is_order_independent_for_every_permutation() {
        let root = RngRoot(Some(99));
        let ids = [
            "dataset.audio.duration",
            "timing.request.poisson_interval",
            "dataset.sampler.shuffle",
        ];
        let expected = ids.map(|id| root.derive_seed(id));
        for order in [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ] {
            for idx in order {
                assert_eq!(root.derive_seed(ids[idx]), expected[idx]);
            }
        }
        assert_ne!(expected[0], expected[1]);
        assert_ne!(expected[1], expected[2]);
    }

    #[test]
    fn every_canonical_namespace_has_a_distinct_seed() {
        let root = RngRoot::new(Some(7));
        let mut seeds: Vec<_> = namespace::ALL
            .iter()
            .map(|identifier| root.derive_seed(identifier).unwrap())
            .collect();
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), namespace::ALL.len());
    }
}
