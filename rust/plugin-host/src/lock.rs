// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical plugin lock — the stable artifact recording which packages are
//! active in a run's plugin universe.
//!
//! [`PluginLockV1`] is the full lock document.  Its [`PluginLockDigest`]
//! is a BLAKE3 hash over the canonical JSON serialization of the package list,
//! making any post-write mutation detectable by [`LockedCatalogBundle`].

use serde::{Deserialize, Serialize};

/// Status of a locked plugin package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageStatus {
    /// The package was loaded and registered successfully.
    Active,
    /// The package is present in the catalog but explicitly disabled.
    Disabled,
    /// The package failed to load; retained for diagnostics.
    Failed,
}

/// Digest authenticating the package list in a [`PluginLockV1`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginLockDigest {
    /// Hash algorithm.  Always `"blake3"`.
    pub algorithm: String,
    /// Lowercase hex digest string (64 characters for BLAKE3).
    pub hex: String,
}

impl PluginLockDigest {
    /// Compute the digest over the canonical JSON serialization of a package
    /// list.  The serialization is deterministic: packages are provided in
    /// caller-supplied order; the caller is responsible for stable ordering.
    pub fn compute(packages: &[LockedPackageV1]) -> Self {
        let canonical =
            serde_json::to_vec(packages).expect("LockedPackageV1 serializes infallibly");
        let digest = blake3::hash(&canonical);
        Self {
            algorithm: "blake3".to_string(),
            hex: digest.to_hex().to_string(),
        }
    }
}

/// One package entry in the canonical plugin lock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockedPackageV1 {
    /// Normalized package identifier.
    pub id: String,
    /// Semantic version string (`major.minor.patch`).
    pub version: String,
    /// Whether the package is active, disabled, or failed.
    pub status: PackageStatus,
    /// `"blake3:<hex>"` digest of the platform artifact (cdylib).
    pub artifact_digest: String,
    /// `"blake3:<hex>"` digest of the host ABI closure record.
    pub closure_digest: String,
}

/// The canonical plugin lock document (`schema_version: "1.0"`).
///
/// Constructed via [`PluginLockV1::new`], which computes the digest
/// immediately so the stored digest always matches the stored package list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginLockV1 {
    /// Always `"1.0"`.
    pub schema_version: String,
    /// Locked package entries in load order.
    pub packages: Vec<LockedPackageV1>,
    /// BLAKE3 digest of the canonical JSON serialization of [`Self::packages`].
    pub digest: PluginLockDigest,
}

impl PluginLockV1 {
    /// Construct a lock and immediately compute its digest.
    pub fn new(packages: Vec<LockedPackageV1>) -> Self {
        let digest = PluginLockDigest::compute(&packages);
        Self {
            schema_version: "1.0".to_string(),
            packages,
            digest,
        }
    }

    /// Verify that the stored digest matches the package list.
    pub fn verify(&self) -> bool {
        let expected = PluginLockDigest::compute(&self.packages);
        expected.hex == self.digest.hex && expected.algorithm == self.digest.algorithm
    }
}
