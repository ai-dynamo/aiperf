// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Host-universe and plugin-artifact build identity records.
//!
//! `HostAbiUniverseRecordV1` captures the compiler, toolchain, and ABI policy
//! that govern what plugins are binary-compatible with this host.
//!
//! `PluginArtifactBuildRecordV1` captures the exact build inputs for one plugin
//! cdylib, binding it to a host universe via `universe_digest`.
//!
//! Both records carry a `canonical_digest` (BLAKE3 over all other fields in
//! declaration order) which serves as the opaque identity used by the loader.

use serde::{Deserialize, Serialize};

use crate::canonical::{feed_json, feed_opt_str, feed_str, feed_u8, feed_u32, finalize_hex};

/// Identity of one ABI-contributing crate (sorted by `name` then `version`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CrateIdentity {
    pub name: String,
    pub version: String,
    /// BLAKE3 hex of the crate source tree.
    pub source_digest: String,
}

/// A native/system library dependency.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeDep {
    pub name: String,
    pub version: Option<String>,
    /// BLAKE3 hex of the library file, when available.
    pub library_digest: Option<String>,
}

/// Immutable record of the host compiler and ABI universe.
///
/// Two builds are ABI-compatible only when their `canonical_digest` values
/// match.  A common-field change (anything that affects the ABI boundary)
/// produces a new `canonical_digest`, revoking all prior plugin build IDs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostAbiUniverseRecordV1 {
    /// BLAKE3 hex of the `rustc` executable.
    pub rustc_exe_digest: String,
    /// Short Git commit hash embedded in the rustc version string.
    pub rustc_commit: String,
    /// Full `rustc --version --verbose` output, normalized.
    pub rustc_full_version: String,
    /// BLAKE3 hex of the Rust sysroot directory tree.
    pub sysroot_digest: String,
    /// Target triple (e.g. `"x86_64-unknown-linux-gnu"`).
    pub target_triple: String,
    /// Pointer width in bits (32 or 64).
    pub pointer_width: u8,
    /// Byte order (`"little"` or `"big"`).
    pub endian: String,
    /// Codegen backend name (e.g. `"llvm"`).
    pub codegen_backend: String,
    /// Sorted, normalized cfg flags that affect the ABI.
    pub cfg_flags: Vec<String>,
    /// Sorted codegen flags (e.g. `-C opt-level=3`).
    pub codegen_flags: Vec<String>,
    /// Sorted ABI-contributing crate identities.
    pub abi_crates: Vec<CrateIdentity>,
    /// Global allocator identity (e.g. `"aiperf-allocator-provider"`).
    pub allocator: String,
    /// Panic strategy (`"abort"` or `"unwind"`).
    pub panic_strategy: String,
    /// Monotonically increasing host target-policy schema version.
    pub target_policy_version: u32,
    /// BLAKE3 hex of the linker executable (absent when using the default toolchain linker).
    pub linker_exe_digest: Option<String>,
    /// BLAKE3 hex over all preceding fields in declaration order.
    pub canonical_digest: String,
}

impl HostAbiUniverseRecordV1 {
    /// Computes and returns the canonical digest for the given field values.
    ///
    /// All fields except `canonical_digest` contribute; their names are the
    /// exact serde field names, in declaration order.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_digest(
        rustc_exe_digest: &str,
        rustc_commit: &str,
        rustc_full_version: &str,
        sysroot_digest: &str,
        target_triple: &str,
        pointer_width: u8,
        endian: &str,
        codegen_backend: &str,
        cfg_flags: &[String],
        codegen_flags: &[String],
        abi_crates: &[CrateIdentity],
        allocator: &str,
        panic_strategy: &str,
        target_policy_version: u32,
        linker_exe_digest: Option<&str>,
    ) -> String {
        let mut h = blake3::Hasher::new();
        feed_str(&mut h, "rustc_exe_digest", rustc_exe_digest);
        feed_str(&mut h, "rustc_commit", rustc_commit);
        feed_str(&mut h, "rustc_full_version", rustc_full_version);
        feed_str(&mut h, "sysroot_digest", sysroot_digest);
        feed_str(&mut h, "target_triple", target_triple);
        feed_u8(&mut h, "pointer_width", pointer_width);
        feed_str(&mut h, "endian", endian);
        feed_str(&mut h, "codegen_backend", codegen_backend);
        feed_json(&mut h, "cfg_flags", cfg_flags);
        feed_json(&mut h, "codegen_flags", codegen_flags);
        feed_json(&mut h, "abi_crates", abi_crates);
        feed_str(&mut h, "allocator", allocator);
        feed_str(&mut h, "panic_strategy", panic_strategy);
        feed_u32(&mut h, "target_policy_version", target_policy_version);
        feed_opt_str(&mut h, "linker_exe_digest", linker_exe_digest);
        finalize_hex(h)
    }

    /// Verifies that `self.canonical_digest` matches a freshly computed digest.
    ///
    /// Comparison uses `blake3::Hash` constant-time `PartialEq` to avoid
    /// timing side-channels on the digest comparison.
    pub fn verify_digest(&self) -> bool {
        let expected = Self::compute_digest(
            &self.rustc_exe_digest,
            &self.rustc_commit,
            &self.rustc_full_version,
            &self.sysroot_digest,
            &self.target_triple,
            self.pointer_width,
            &self.endian,
            &self.codegen_backend,
            &self.cfg_flags,
            &self.codegen_flags,
            &self.abi_crates,
            &self.allocator,
            &self.panic_strategy,
            self.target_policy_version,
            self.linker_exe_digest.as_deref(),
        );
        // Constant-time comparison: parse both hex digests to blake3::Hash,
        // which implements constant-time PartialEq.
        match (
            blake3::Hash::from_hex(&expected),
            blake3::Hash::from_hex(&self.canonical_digest),
        ) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }

    /// Returns the `HostAbiUniverseId` for this record.
    pub fn universe_id(&self) -> HostAbiUniverseId {
        HostAbiUniverseId(self.canonical_digest.clone())
    }
}

/// Opaque identity token for a host ABI universe.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HostAbiUniverseId(pub String);

impl std::fmt::Display for HostAbiUniverseId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Immutable record of one plugin cdylib's exact build inputs.
///
/// `universe_digest` must equal the `canonical_digest` of the host's
/// `HostAbiUniverseRecordV1`.  A mismatch means the plugin was built against
/// a different ABI universe and must be rejected by the loader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PluginArtifactBuildRecordV1 {
    /// Must match the host's `HostAbiUniverseRecordV1.canonical_digest`.
    pub universe_digest: String,
    /// Cargo package name.
    pub package_name: String,
    /// Cargo package version.
    pub package_version: String,
    /// BLAKE3 hex of the build script source, when present.
    pub build_script_digest: Option<String>,
    /// BLAKE3 hex of the common (ABI-boundary-crossing) source tree.
    pub common_sources_digest: String,
    /// BLAKE3 hex of the private (plugin-internal) source tree.
    pub private_sources_digest: String,
    /// Sorted native/system dependencies.
    pub native_deps: Vec<NativeDep>,
    /// BLAKE3 hex of the link payload before record embedding.
    pub pre_embed_payload_digest: String,
    /// BLAKE3 hex of the final artifact (after record embedding).
    pub artifact_digest: String,
    /// BLAKE3 hex over all preceding fields in declaration order.
    pub canonical_digest: String,
}

impl PluginArtifactBuildRecordV1 {
    /// Computes and returns the canonical digest for the given field values.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_digest(
        universe_digest: &str,
        package_name: &str,
        package_version: &str,
        build_script_digest: Option<&str>,
        common_sources_digest: &str,
        private_sources_digest: &str,
        native_deps: &[NativeDep],
        pre_embed_payload_digest: &str,
        artifact_digest: &str,
    ) -> String {
        let mut h = blake3::Hasher::new();
        feed_str(&mut h, "universe_digest", universe_digest);
        feed_str(&mut h, "package_name", package_name);
        feed_str(&mut h, "package_version", package_version);
        feed_opt_str(&mut h, "build_script_digest", build_script_digest);
        feed_str(&mut h, "common_sources_digest", common_sources_digest);
        feed_str(&mut h, "private_sources_digest", private_sources_digest);
        feed_json(&mut h, "native_deps", native_deps);
        feed_str(&mut h, "pre_embed_payload_digest", pre_embed_payload_digest);
        feed_str(&mut h, "artifact_digest", artifact_digest);
        finalize_hex(h)
    }

    /// Verifies that `self.canonical_digest` matches a freshly computed digest.
    ///
    /// Comparison uses `blake3::Hash` constant-time `PartialEq` to avoid
    /// timing side-channels on the digest comparison.
    pub fn verify_digest(&self) -> bool {
        let expected = Self::compute_digest(
            &self.universe_digest,
            &self.package_name,
            &self.package_version,
            self.build_script_digest.as_deref(),
            &self.common_sources_digest,
            &self.private_sources_digest,
            &self.native_deps,
            &self.pre_embed_payload_digest,
            &self.artifact_digest,
        );
        match (
            blake3::Hash::from_hex(&expected),
            blake3::Hash::from_hex(&self.canonical_digest),
        ) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }

    /// Returns the `PluginArtifactBuildId` for this record.
    pub fn build_id(&self) -> PluginArtifactBuildId {
        PluginArtifactBuildId(self.canonical_digest.clone())
    }
}

/// Opaque identity token for one plugin artifact build.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PluginArtifactBuildId(pub String);

impl std::fmt::Display for PluginArtifactBuildId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
