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

    /// Derive the deterministic child seed for `identifier`.
    ///
    /// Returns `None` when this root is seedless. For seeded roots, this is the
    /// BLAKE3 port of Python `_RNGManager.derive`: hash the UTF-8 bytes of
    /// `"{root}:{identifier}"` and read the first eight digest bytes as a
    /// big-endian `u64`.
    pub fn derive_seed(self, identifier: &str) -> Option<u64> {
        self.0.map(|root| {
            derive_seed_parts(&[root.to_string().as_bytes(), b":", identifier.as_bytes()])
        })
    }

    /// Derive an adaptive-sweep variation seed for `label`.
    ///
    /// Mirrors Python `derive_variation_seed(root, label)` with the BLAKE3 hash
    /// selected by the Rust RNG spec: `"{root}:variation:{label}"`.
    pub fn derive_variation_seed(self, label: &str) -> Option<u64> {
        self.0.map(|root| {
            derive_seed_parts(&[
                root.to_string().as_bytes(),
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

    #[test]
    fn derive_seed_matches_spec_vectors() {
        assert_eq!(
            RngRoot(Some(42)).derive_seed("dataset.loader"),
            Some(2_466_643_113_772_406_410)
        );
        assert_eq!(
            RngRoot(Some(42)).derive_seed("timing.request_rate"),
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
        assert_eq!(root.derive_seed("dataset.loader"), None);
        assert_eq!(root.derive_variation_seed("concurrency=4"), None);
    }

    #[test]
    fn parts_match_single_key_hash() {
        assert_eq!(
            derive_seed_parts(&[b"42", b":", b"trace", b":", b"7"]),
            derive_seed_u64("42:trace:7")
        );
    }

    #[test]
    fn derivation_is_order_independent() {
        let root = RngRoot(Some(99));
        let ids = [
            "dataset.audio.duration",
            "timing.request.poisson_interval",
            "dataset.sampler.shuffle",
        ];
        let forward: Vec<_> = ids.iter().map(|id| root.derive_seed(id)).collect();
        let mut reverse_ids = ids;
        reverse_ids.reverse();
        let reverse: Vec<_> = reverse_ids.iter().map(|id| root.derive_seed(id)).collect();

        assert_eq!(forward[0], reverse[2]);
        assert_eq!(forward[1], reverse[1]);
        assert_eq!(forward[2], reverse[0]);
        assert_ne!(forward[0], forward[1]);
        assert_ne!(forward[1], forward[2]);
    }
}
