// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private session-host ownership of the checked quarantine tombstone proof.
//!
//! `super` declares this as `mod host;`, so the module path is private to the
//! session subtree. No sibling of `super`, and no other module in this crate,
//! can name the concrete proof or reach its production mint. The P1B session
//! host lands as a descendant — `streaming/session/host/<name>.rs` — and
//! inherits that mint authority.

use super::{SessionQuarantineTombstoneView, reliability_view_seal};
use crate::streaming::{checkpoint::StreamRunIdentity, identity::ContentDigest};

/// Durable budgeted quarantine tombstones minted inside this host subtree.
pub mod tombstones;

/// Session-host-owned sealed borrow of the retained quarantine tombstone map.
///
/// The borrowed entry slice prevents reliability preparation from moving or
/// cloning the retained map; the host-subtree-private mint keeps the proof
/// unavailable to adapters and to unrelated crate modules alike.
// The production mint lands with the P1B session host; until then the type is
// exercised only by the crate-private test fixture.
#[allow(dead_code)]
pub struct CheckedSessionQuarantineTombstoneView<'a> {
    run: StreamRunIdentity,
    tombstone_root: ContentDigest,
    revision: u64,
    canonical_encoded_entries: &'a [u8],
}

#[allow(dead_code)]
impl<'a> CheckedSessionQuarantineTombstoneView<'a> {
    /// Borrow the session host's retained tombstone map as a checked proof.
    ///
    /// Deliberately declared without a visibility modifier: production mint
    /// authority belongs to this host subtree and to nothing else in the crate.
    const fn new(
        run: StreamRunIdentity,
        tombstone_root: ContentDigest,
        revision: u64,
        canonical_encoded_entries: &'a [u8],
    ) -> Self {
        Self {
            run,
            tombstone_root,
            revision,
            canonical_encoded_entries,
        }
    }
}

#[cfg(test)]
impl<'a> CheckedSessionQuarantineTombstoneView<'a> {
    /// Mint a checked tombstone borrow for in-crate reliability unit tests.
    ///
    /// This fixture does not exist in any production build.
    pub(crate) const fn for_test(
        run: StreamRunIdentity,
        tombstone_root: ContentDigest,
        revision: u64,
        canonical_encoded_entries: &'a [u8],
    ) -> Self {
        Self::new(run, tombstone_root, revision, canonical_encoded_entries)
    }
}

impl reliability_view_seal::SessionQuarantineTombstoneView
    for CheckedSessionQuarantineTombstoneView<'_>
{
}

impl SessionQuarantineTombstoneView for CheckedSessionQuarantineTombstoneView<'_> {
    fn run(&self) -> &StreamRunIdentity {
        &self.run
    }

    fn tombstone_root(&self) -> ContentDigest {
        self.tombstone_root
    }

    fn revision(&self) -> u64 {
        self.revision
    }

    fn canonical_encoded_entries(&self) -> &[u8] {
        self.canonical_encoded_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::identity::LogicalReplayRunId;

    const PARENT_SOURCE: &str = include_str!("../session.rs");
    const HOST_SOURCE: &str = include_str!("host.rs");

    /// Return `host.rs` with its own test module removed.
    fn host_production_source() -> &'static str {
        HOST_SOURCE
            .split_once("\n#[cfg(test)]\nmod tests {")
            .map_or(HOST_SOURCE, |(head, _)| head)
    }

    #[test]
    fn session_host_module_is_not_reachable_from_siblings() {
        assert!(
            PARENT_SOURCE.contains("\nmod host;\n"),
            "session.rs must declare the session host as a private child module"
        );
        for widened in ["pub mod host", "pub(crate) mod host", "pub(super) mod host"] {
            assert!(
                !PARENT_SOURCE.contains(widened),
                "the session host module must stay private: {widened}"
            );
        }
    }

    #[test]
    fn session_proof_reexport_is_test_only() {
        let before = PARENT_SOURCE
            .split_once("pub(crate) use host::CheckedSessionQuarantineTombstoneView;")
            .map(|(head, _)| head)
            .unwrap_or_else(|| {
                panic!("session.rs must re-export the host proof for in-crate tests")
            });
        assert!(
            before.trim_end().ends_with("#[cfg(test)]"),
            "the host proof re-export must be gated on cfg(test)"
        );
    }

    #[test]
    fn session_proof_mint_is_host_subtree_private() {
        let source = host_production_source();
        assert_eq!(
            source.matches("    const fn new(").count(),
            1,
            "the tombstone mint must carry no visibility modifier"
        );
        for widened in [
            "pub const fn new(",
            "pub(crate) const fn new(",
            "pub(super) const fn new(",
        ] {
            assert!(
                !source.contains(widened),
                "session proof mint visibility was widened: {widened}"
            );
        }
    }

    #[test]
    fn host_minted_proofs_are_the_only_construction_path() {
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x41; 32]));
        let entries = b"canonical-tombstones";
        let root = ContentDigest::from_bytes(*blake3::hash(entries).as_bytes());
        let view = CheckedSessionQuarantineTombstoneView::for_test(run, root, 9, entries);
        assert_eq!(view.run(), &run);
        assert_eq!(view.tombstone_root(), root);
        assert_eq!(view.revision(), 9);
        assert_eq!(view.canonical_encoded_entries(), entries);
    }
}
