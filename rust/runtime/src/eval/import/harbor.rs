// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Harbor-compatible importer that preserves source bytes before normalization.

use std::fmt::{self, Display, Formatter};

use crate::eval::{
    ArtifactDigest, CanonicalPackagePlan, EvalTaskRef, ImportDisposition, ImportReport,
    append_identity_field,
};

use super::{HarborSource, HarborTaskPackage, SourceAcquirer, normalize};

/// Native normalized representation of one imported task package.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportedTask {
    /// Immutable normalized task reference.
    pub task: EvalTaskRef,
    /// Immutable import provenance report.
    pub report: ImportReport,
    /// Strict executable package material retained for native execution.
    pub package: HarborTaskPackage,
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
        let acquired = self.acquirer.acquire_artifact(source)?;
        let source_digest = acquired.source_digest();
        if has_unsupported_semantics(acquired.primary_bytes()) {
            return Err(HarborImportError::Unsupported(ImportReport {
                source_digest,
                normalized_digest: ArtifactDigest::from_bytes(&[]),
                disposition: ImportDisposition::Unsupported,
            }));
        }
        let (draft, executable_view) =
            if acquired.is_tree() && acquired.primary_path() == "task.toml" {
                normalize::normalize_standard_directory(&acquired)?
            } else {
                normalize::normalize(&acquired)?
            };
        let plan_digest = CanonicalPackagePlan::new(
            draft.id(),
            draft.agent_command(),
            draft.verifier_command(),
            draft.execution_plan(),
            draft.native_graph(),
        )
        .digest();
        let executable_source_digest = acquired.executable_source_digest(&executable_view)?;
        let package_digest = package_identity(&plan_digest, &executable_source_digest);
        let task = EvalTaskRef::new(draft.id().to_owned(), package_digest.clone())
            .map_err(|error| HarborImportError::InvalidPackage(error.to_string()))?;
        let package = draft.into_package(acquired, package_digest.clone());
        let report = ImportReport {
            source_digest,
            normalized_digest: package_digest,
            disposition: ImportDisposition::LosslessNormalized,
        };
        Ok(ImportedTask {
            task,
            report,
            package,
        })
    }
}

fn package_identity(
    plan_digest: &ArtifactDigest,
    executable_source_digest: &ArtifactDigest,
) -> ArtifactDigest {
    let mut material = Vec::new();
    append_identity_field(
        &mut material,
        "package-identity.domain",
        b"aiperf-eval-package-v2",
    );
    append_identity_field(
        &mut material,
        "package-identity.plan-digest",
        plan_digest.as_str().as_bytes(),
    );
    append_identity_field(
        &mut material,
        "package-identity.executable-source-digest",
        executable_source_digest.as_str().as_bytes(),
    );
    ArtifactDigest::from_bytes(&material)
}

fn has_unsupported_semantics(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| value.get("unsupported_semantics").cloned())
        .is_some()
}
