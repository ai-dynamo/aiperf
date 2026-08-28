// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Comparator identity validation for the distribution parity gate.
//!
//! The static comparator must be built from byte-for-byte identical source,
//! dependencies, and build configuration as the dynamic candidate whose
//! performance it is being compared against. Any identity difference means the
//! two builds measure different implementations, and the parity number is
//! meaningless.
//!
//! [`ComparatorIdentityCheck`] validates each identity dimension in isolation
//! so the error names exactly what drifted rather than only that something did.

/// All identity fields that must match between the dynamic candidate and the
/// static comparator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComparatorSpec {
    /// BLAKE3 digest of the workspace source tree snapshot.
    pub source_tree_digest: [u8; 32],
    /// BLAKE3 digest of the `Cargo.lock` file.
    pub cargo_lock_digest: [u8; 32],
    /// Sorted list of implementation-leaf crate ids linked into this build.
    pub implementation_leaf_census: Vec<String>,
    /// Sorted list of Cargo feature flags active for this build.
    pub feature_set: Vec<String>,
    /// Whether this build was compiled with fat LTO.
    ///
    /// Fat LTO is required so inlining opportunities across crate boundaries are
    /// identical between the static and dynamic builds.
    pub fat_lto: bool,
    /// Whether this build links mimalloc as a static library.
    pub static_mimalloc: bool,
    /// BLAKE3 digest of the canonical default configuration snapshot.
    pub config_default_digest: [u8; 32],
}

impl ComparatorSpec {
    /// Return a synthetic candidate fixture suitable for tests.
    pub fn synthetic_candidate_fixture() -> Self {
        Self {
            source_tree_digest: [0x11; 32],
            cargo_lock_digest: [0x22; 32],
            implementation_leaf_census: vec![
                "nvidia/export-basic".to_owned(),
                "nvidia/transport-http".to_owned(),
            ],
            feature_set: vec!["grpc".to_owned()],
            fat_lto: true,
            static_mimalloc: true,
            config_default_digest: [0x33; 32],
        }
    }

    /// Return a synthetic comparator fixture that exactly matches the candidate.
    pub fn synthetic_comparator_fixture() -> Self {
        // Identical to the candidate; tests mutate specific fields to trigger errors.
        Self::synthetic_candidate_fixture()
    }
}

/// Why a comparator identity check failed.
#[derive(Debug, PartialEq, Eq)]
pub enum ComparatorIdentityError {
    /// The source-tree digests differ.
    SourceTreeMismatch {
        /// Digest from the dynamic candidate.
        candidate: [u8; 32],
        /// Digest from the static comparator.
        comparator: [u8; 32],
    },
    /// The `Cargo.lock` digests differ.
    CargoLockMismatch {
        candidate: [u8; 32],
        comparator: [u8; 32],
    },
    /// The implementation-leaf census does not match.
    ImplementationLeafCensusMismatch {
        candidate: Vec<String>,
        comparator: Vec<String>,
    },
    /// The active Cargo feature sets differ.
    FeatureSetMismatch {
        candidate: Vec<String>,
        comparator: Vec<String>,
    },
    /// The comparator was not compiled with fat LTO.
    FatLtoRequired,
    /// The comparator does not use static mimalloc.
    StaticMimallocRequired,
    /// The default-configuration digests differ.
    ConfigDefaultDigestMismatch {
        candidate: [u8; 32],
        comparator: [u8; 32],
    },
}

impl std::fmt::Display for ComparatorIdentityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceTreeMismatch { .. } => write!(f, "source-tree digest mismatch"),
            Self::CargoLockMismatch { .. } => write!(f, "Cargo.lock digest mismatch"),
            Self::ImplementationLeafCensusMismatch { .. } => {
                write!(f, "implementation-leaf census mismatch")
            }
            Self::FeatureSetMismatch { .. } => write!(f, "feature-set mismatch"),
            Self::FatLtoRequired => write!(f, "comparator must be compiled with fat LTO"),
            Self::StaticMimallocRequired => {
                write!(f, "comparator must link mimalloc as a static library")
            }
            Self::ConfigDefaultDigestMismatch { .. } => {
                write!(f, "default-configuration digest mismatch")
            }
        }
    }
}

impl std::error::Error for ComparatorIdentityError {}

/// Validates that a static comparator build is identity-equivalent to the
/// dynamic candidate it will be paired against.
pub struct ComparatorIdentityCheck<'a> {
    candidate: &'a ComparatorSpec,
    comparator: &'a ComparatorSpec,
}

impl<'a> ComparatorIdentityCheck<'a> {
    /// Bind a candidate and comparator for validation.
    pub fn new(candidate: &'a ComparatorSpec, comparator: &'a ComparatorSpec) -> Self {
        Self {
            candidate,
            comparator,
        }
    }

    /// Validate all identity dimensions in order.
    ///
    /// Returns the first mismatch found; callers that want all mismatches must
    /// call validate repeatedly after each fix.
    pub fn validate(&self) -> Result<(), ComparatorIdentityError> {
        if self.candidate.source_tree_digest != self.comparator.source_tree_digest {
            return Err(ComparatorIdentityError::SourceTreeMismatch {
                candidate: self.candidate.source_tree_digest,
                comparator: self.comparator.source_tree_digest,
            });
        }
        if self.candidate.cargo_lock_digest != self.comparator.cargo_lock_digest {
            return Err(ComparatorIdentityError::CargoLockMismatch {
                candidate: self.candidate.cargo_lock_digest,
                comparator: self.comparator.cargo_lock_digest,
            });
        }
        if self.candidate.implementation_leaf_census
            != self.comparator.implementation_leaf_census
        {
            return Err(ComparatorIdentityError::ImplementationLeafCensusMismatch {
                candidate: self.candidate.implementation_leaf_census.clone(),
                comparator: self.comparator.implementation_leaf_census.clone(),
            });
        }
        if self.candidate.feature_set != self.comparator.feature_set {
            return Err(ComparatorIdentityError::FeatureSetMismatch {
                candidate: self.candidate.feature_set.clone(),
                comparator: self.comparator.feature_set.clone(),
            });
        }
        if !self.comparator.fat_lto {
            return Err(ComparatorIdentityError::FatLtoRequired);
        }
        if !self.comparator.static_mimalloc {
            return Err(ComparatorIdentityError::StaticMimallocRequired);
        }
        if self.candidate.config_default_digest != self.comparator.config_default_digest {
            return Err(ComparatorIdentityError::ConfigDefaultDigestMismatch {
                candidate: self.candidate.config_default_digest,
                comparator: self.comparator.config_default_digest,
            });
        }
        Ok(())
    }
}
