// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ABI closure record: the universe + all plugin builds sharing that universe.
//!
//! `AbiClosureRecordV1` bundles a `HostAbiUniverseRecordV1` with the set of
//! `PluginArtifactBuildRecordV1` values that were built against it.  The
//! loader validates the bundle before admitting any plugin.
//!
//! `AbiClosureRevocationRecordV1` signals that a previously issued universe
//! digest is no longer valid (e.g. because a private field was reclassified as
//! common, changing the universe identity and revoking all prior build IDs).

use serde::{Deserialize, Serialize};

use crate::identity::{HostAbiUniverseRecordV1, PluginArtifactBuildRecordV1};

/// A validated bundle of one host universe and its plugin builds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiClosureRecordV1 {
    pub universe: HostAbiUniverseRecordV1,
    pub builds: Vec<PluginArtifactBuildRecordV1>,
}

/// Error returned by [`AbiClosureRecordV1::validate`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbiClosureError {
    /// The universe record's `canonical_digest` does not match a freshly computed digest.
    UniverseDigestMismatch {
        stored: String,
        computed: String,
    },
    /// A build record's `universe_digest` does not match the universe's `canonical_digest`.
    BuildUniverseMismatch {
        build_index: usize,
        package: String,
        build_universe_digest: String,
        expected_universe_digest: String,
    },
    /// A build record's `canonical_digest` does not match a freshly computed digest.
    BuildDigestMismatch {
        build_index: usize,
        package: String,
        stored: String,
        computed: String,
    },
}

impl std::fmt::Display for AbiClosureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AbiClosureError::UniverseDigestMismatch { stored, computed } => {
                write!(
                    f,
                    "universe canonical_digest mismatch: stored={stored} computed={computed}"
                )
            }
            AbiClosureError::BuildUniverseMismatch {
                build_index,
                package,
                build_universe_digest,
                expected_universe_digest,
            } => write!(
                f,
                "build[{build_index}] ({package}) universe_digest={build_universe_digest} \
                 does not match universe canonical_digest={expected_universe_digest}"
            ),
            AbiClosureError::BuildDigestMismatch {
                build_index,
                package,
                stored,
                computed,
            } => write!(
                f,
                "build[{build_index}] ({package}) canonical_digest mismatch: \
                 stored={stored} computed={computed}"
            ),
        }
    }
}

impl std::error::Error for AbiClosureError {}

impl AbiClosureRecordV1 {
    /// Validates all digests in the closure.
    ///
    /// Returns the first error found, or `Ok(())` when all digests are consistent.
    pub fn validate(&self) -> Result<(), AbiClosureError> {
        if !self.universe.verify_digest() {
            let computed = HostAbiUniverseRecordV1::compute_digest(
                &self.universe.rustc_exe_digest,
                &self.universe.rustc_commit,
                &self.universe.rustc_full_version,
                &self.universe.sysroot_digest,
                &self.universe.target_triple,
                self.universe.pointer_width,
                &self.universe.endian,
                &self.universe.codegen_backend,
                &self.universe.cfg_flags,
                &self.universe.codegen_flags,
                &self.universe.abi_crates,
                &self.universe.allocator,
                &self.universe.panic_strategy,
                self.universe.target_policy_version,
                self.universe.linker_exe_digest.as_deref(),
            );
            return Err(AbiClosureError::UniverseDigestMismatch {
                stored: self.universe.canonical_digest.clone(),
                computed,
            });
        }
        for (i, build) in self.builds.iter().enumerate() {
            if build.universe_digest != self.universe.canonical_digest {
                return Err(AbiClosureError::BuildUniverseMismatch {
                    build_index: i,
                    package: build.package_name.clone(),
                    build_universe_digest: build.universe_digest.clone(),
                    expected_universe_digest: self.universe.canonical_digest.clone(),
                });
            }
            if !build.verify_digest() {
                let computed = PluginArtifactBuildRecordV1::compute_digest(
                    &build.universe_digest,
                    &build.package_name,
                    &build.package_version,
                    build.build_script_digest.as_deref(),
                    &build.common_sources_digest,
                    &build.private_sources_digest,
                    &build.native_deps,
                    &build.pre_embed_payload_digest,
                    &build.artifact_digest,
                );
                return Err(AbiClosureError::BuildDigestMismatch {
                    build_index: i,
                    package: build.package_name.clone(),
                    stored: build.canonical_digest.clone(),
                    computed,
                });
            }
        }
        Ok(())
    }
}

/// Signals that a previously issued universe digest is no longer valid.
///
/// Issued when a field is reclassified from private to common (crossing the
/// ABI boundary), which changes the `HostAbiUniverseRecordV1.canonical_digest`
/// and thereby revokes every `PluginArtifactBuildRecordV1` that referenced
/// the old universe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiClosureRevocationRecordV1 {
    /// The universe digest that is no longer valid.
    pub revoked_universe_digest: String,
    /// Human-readable reason (e.g. `"cfg_flags reclassified as common"`).
    pub reason: String,
    /// The digest of the replacement universe, when one exists.
    pub replacement_universe_digest: Option<String>,
}
