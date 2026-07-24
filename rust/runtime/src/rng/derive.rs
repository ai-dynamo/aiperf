// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hash-derived seed algebra for order-independent random streams.
//!
//! Components name their stream and the
//! child seed depends only on `(root_seed, identifier)`. The stable contract is
//! BLAKE3's first eight digest bytes interpreted as a big-endian `u64`.

use crate::rng::compat::python_random::PythonRandomGenerator;
use crate::rng::generator::RustRandomGenerator;

/// Construct one backend-specific generator from an [`RngRoot`] and identifier.
///
/// Native Rust streams preserve the BLAKE3 derivation contract and seedless
/// `None` semantics. Python-parity streams instead mirror Python's SHA-256
/// derivation and use a fresh entropy seed when no deterministic root exists.
pub trait DerivedRandomGenerator: Sized {
    /// Construct one named child stream from `root`.
    fn from_rng_root(root: RngRoot, identifier: &str) -> Self;
}

impl DerivedRandomGenerator for RustRandomGenerator {
    fn from_rng_root(root: RngRoot, identifier: &str) -> Self {
        Self::from_seed(root.derive_seed(identifier))
    }
}

impl DerivedRandomGenerator for PythonRandomGenerator {
    fn from_rng_root(root: RngRoot, identifier: &str) -> Self {
        match root.seed() {
            Some(seed) => Self::derive(seed, identifier),
            None => Self::from_seed_or_entropy(None),
        }
    }
}

/// Root seed for a reproducible run.
///
/// `Some(seed)` produces deterministic child streams. `None` makes every derived
/// generator seed from OS/thread entropy without a global singleton.
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
    pub fn derive(self, identifier: &str) -> RustRandomGenerator {
        self.derive_generator(identifier)
    }

    /// Derive one backend-specific child generator for `identifier`.
    pub fn derive_generator<R: DerivedRandomGenerator>(self, identifier: &str) -> R {
        R::from_rng_root(self, identifier)
    }

    /// Derive a child root for a named subsystem.
    ///
    /// Hierarchical roots let a coordinator isolate a component first and then
    /// derive its internal streams without constructing ad-hoc compound names.
    /// Seedless roots remain seedless, so callers preserve the ordinary entropy
    /// semantics rather than replacing them with a fallback constant.
    pub fn derive_root(self, identifier: &str) -> Self {
        Self(self.derive_seed(identifier))
    }

    /// Derive a child root for one indexed instance of a named subsystem.
    ///
    /// The index is encoded as canonical ASCII decimal after `identifier` and a
    /// colon. This is the standard split for phase- and worker-local streams;
    /// adding another worker cannot perturb any existing worker's sequence.
    pub fn derive_indexed_root(self, identifier: &str, index: u64) -> Self {
        Self(self.derive_indexed_seed(identifier, index))
    }

    /// Derive the deterministic child seed for `identifier`.
    ///
    /// Returns `None` when this root is seedless. For seeded roots, hashes the
    /// UTF-8 bytes of `"{root}:{identifier}"` and reads the first eight digest
    /// bytes as a big-endian `u64`.
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

    /// Return a deterministic derived seed or one fresh entropy-backed value.
    ///
    /// Use this only at boundaries whose downstream API requires a concrete
    /// `u64` seed. Consumers able to accept [`RngRoot`] should retain the root so
    /// seedless behavior remains explicit in their own type.
    pub fn derive_seed_or_entropy(self, identifier: &str) -> u64 {
        self.derive_seed(identifier)
            .unwrap_or_else(|| self.derive(identifier).random_u64())
    }

    /// Derive the deterministic child seed for one indexed component instance.
    ///
    /// Seeded roots hash `"{root}:{identifier}:{index}"`; seedless roots return
    /// `None`. The parts are streamed directly into BLAKE3 without allocating a
    /// formatted namespace string.
    pub fn derive_indexed_seed(self, identifier: &str, index: u64) -> Option<u64> {
        self.0.map(|root| {
            let mut root_buf = itoa::Buffer::new();
            let mut index_buf = itoa::Buffer::new();
            derive_seed_parts(&[
                root_buf.format(root).as_bytes(),
                b":",
                identifier.as_bytes(),
                b":",
                index_buf.format(index).as_bytes(),
            ])
        })
    }

    /// Derive an adaptive-sweep variation seed for `label`.
    ///
    /// For seeded roots, hashes `"{root}:variation:{label}"` with BLAKE3.
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

/// Derive a `u64` from the first eight bytes of a BLAKE3 digest in big-endian
/// order. Changing this reshuffles every deterministic stream.
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
    use crate::rng::namespace;

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
        assert_eq!(root.derive_root(namespace::GRAPH_PHASE), root);
        assert_eq!(root.derive_indexed_seed(namespace::GRAPH_PHASE, 7), None);
        assert_eq!(root.derive_indexed_root(namespace::GRAPH_PHASE, 7), root);
        assert_eq!(root.derive_variation_seed("concurrency=4"), None);
    }

    #[test]
    fn hierarchical_and_indexed_roots_are_canonical_and_isolated() {
        let root = RngRoot::new(Some(42));
        let phase_zero = root.derive_indexed_root(namespace::GRAPH_PHASE, 0);
        let phase_one = root.derive_indexed_root(namespace::GRAPH_PHASE, 1);

        assert_eq!(
            root.derive_indexed_seed(namespace::GRAPH_PHASE, 0),
            root.seed()
                .map(|seed| { derive_seed_u64(&format!("{seed}:{}:0", namespace::GRAPH_PHASE)) })
        );
        assert_ne!(phase_zero, phase_one);
        assert_ne!(
            phase_zero.derive_seed(namespace::GRAPH_ARRIVAL),
            phase_one.derive_seed(namespace::GRAPH_ARRIVAL)
        );
        assert_ne!(
            phase_zero
                .derive_root(namespace::GRAPH_NODE_CANCELLATION)
                .derive_indexed_seed(namespace::GRAPH_NODE_CANCELLATION_WORKER, 0),
            phase_zero
                .derive_root(namespace::GRAPH_NODE_CANCELLATION)
                .derive_indexed_seed(namespace::GRAPH_NODE_CANCELLATION_WORKER, 1)
        );
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
    fn generic_derivation_uses_each_backend_seed_algebra() {
        let root = RngRoot::new(Some(42));
        let native: RustRandomGenerator = root.derive_generator(namespace::DATASET_PROMPT_LENGTH);
        let python: PythonRandomGenerator = root.derive_generator(namespace::DATASET_PROMPT_LENGTH);
        assert_eq!(
            native.seed(),
            root.derive_seed(namespace::DATASET_PROMPT_LENGTH)
        );
        assert_eq!(
            python.seed(),
            PythonRandomGenerator::derive_child_seed(42, namespace::DATASET_PROMPT_LENGTH)
        );
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
    fn required_seed_boundary_preserves_seeded_derivation() {
        let root = RngRoot::new(Some(19));
        assert_eq!(
            root.derive_seed_or_entropy(namespace::GRAPH_ARRIVAL),
            root.derive_seed(namespace::GRAPH_ARRIVAL).unwrap()
        );
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
