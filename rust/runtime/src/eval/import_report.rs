// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Source-normalization reports for Harbor-compatible imports.

use serde::{Deserialize, Serialize};

use super::ArtifactDigest;

/// How faithfully the importer represented its source package.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ImportDisposition {
    /// The package already maps directly to native evaluation semantics.
    Native,
    /// The package was normalized without losing executable meaning.
    LosslessNormalized,
    /// The package was imported with explicit, documented information loss.
    LossyNormalized,
    /// The package contains semantics the native importer must refuse.
    Unsupported,
}

impl ImportDisposition {
    /// Returns the stable wire spelling for this disposition.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::LosslessNormalized => "lossless_normalized",
            Self::LossyNormalized => "lossy_normalized",
            Self::Unsupported => "unsupported",
        }
    }
}

/// Immutable source and normalized artifact provenance for one import.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ImportReport {
    /// Preserved source package digest.
    pub source_digest: ArtifactDigest,
    /// Normalized native-task representation digest.
    pub normalized_digest: ArtifactDigest,
    /// Fidelity of the imported representation.
    pub disposition: ImportDisposition,
}
