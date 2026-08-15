// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Harbor-compatible importer that preserves source bytes before normalization.

use std::fmt::{self, Display, Formatter};

use crate::eval::{ArtifactDigest, EvalTaskRef, ImportDisposition, ImportReport};

use super::{normalize, HarborSource, SourceAcquirer};

/// Native normalized representation of one imported task package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedTask {
    /// Immutable normalized task reference.
    pub task: EvalTaskRef,
    /// Immutable import provenance report.
    pub report: ImportReport,
}

/// Typed failure of a Harbor-compatible import before environment provisioning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HarborImportError {
    /// Source could not be acquired.
    Unavailable(String),
    /// Source reference was malformed.
    InvalidSource(&'static str),
    /// Source bytes did not satisfy the supported native task contract.
    InvalidPackage(String),
    /// Package semantics are unsupported and must not proceed to provisioning.
    Unsupported(ImportReport),
}

impl HarborImportError {
    /// Returns the importer disposition when this error contains a report.
    pub const fn disposition(&self) -> Option<ImportDisposition> {
        match self {
            Self::Unsupported(report) => Some(report.disposition),
            Self::Unavailable(_) | Self::InvalidSource(_) | Self::InvalidPackage(_) => None,
        }
    }
}

impl Display for HarborImportError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable(location) => write!(formatter, "source {location:?} is unavailable"),
            Self::InvalidSource(field) => write!(formatter, "invalid Harbor source {field}"),
            Self::InvalidPackage(reason) => write!(formatter, "invalid Harbor package: {reason}"),
            Self::Unsupported(_) => formatter.write_str("unsupported Harbor package semantics"),
        }
    }
}

impl std::error::Error for HarborImportError {}

/// Imports Harbor-compatible source packages through a caller-owned acquirer.
pub struct HarborImporter<'a> {
    acquirer: &'a dyn SourceAcquirer,
}

impl<'a> HarborImporter<'a> {
    /// Creates an importer backed by an injected source acquirer.
    pub fn new(acquirer: &'a dyn SourceAcquirer) -> Self {
        Self { acquirer }
    }

    /// Preserves source bytes and normalizes only supported package semantics.
    pub fn import(&self, source: &HarborSource) -> Result<ImportedTask, HarborImportError> {
        let bytes = self.acquirer.acquire(source)?;
        let source_digest = ArtifactDigest::from_bytes(&bytes);
        if has_unsupported_semantics(&bytes) {
            return Err(HarborImportError::Unsupported(ImportReport {
                source_digest,
                normalized_digest: ArtifactDigest::from_bytes(&[]),
                disposition: ImportDisposition::Unsupported,
            }));
        }
        let (_, task) = normalize::normalize(&bytes)?;
        let report = ImportReport {
            source_digest,
            normalized_digest: task.digest.clone(),
            disposition: ImportDisposition::LosslessNormalized,
        };
        Ok(ImportedTask { task, report })
    }
}

fn has_unsupported_semantics(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| value.get("unsupported_semantics").cloned())
        .is_some()
}
