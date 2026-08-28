// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest normalization and validation.
//!
//! `normalize_manifest` validates a parsed `PluginManifestV2` and returns the
//! normalized form.  It rejects invalid schema versions, non-canonical SemVer,
//! unsafe artifact paths, duplicate artifact targets, and empty category lists.

use std::collections::HashSet;

use semver::Version;

use crate::{
    error::ManifestError,
    manifest::{PluginManifestV2, PluginPackageManifestV2},
};

/// A validated, normalized plugin ID.
///
/// Accepts lowercase alphanumeric characters, hyphens, and dots; rejects
/// leading/trailing hyphens, absolute path characters, parent-traversal
/// sequences, and Windows Alternate Data Streams.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NormalizedIdV1(pub String);

impl NormalizedIdV1 {
    pub fn parse(raw: &str) -> Result<Self, ManifestError> {
        if raw.is_empty() {
            return Err(ManifestError::MissingField("id".into()));
        }
        // Reject parent-traversal sequences (e.g. "..", "../", "..dangerous")
        if raw == ".." || raw.contains("..") {
            return Err(ManifestError::InvalidPath(format!(
                "id contains parent-traversal sequence: {raw}"
            )));
        }
        // Reject path-like characters
        if raw.starts_with('/') || raw.starts_with('\\') || raw.contains(':') {
            return Err(ManifestError::InvalidPath(format!(
                "id contains path characters: {raw}"
            )));
        }
        if raw.starts_with('-') || raw.ends_with('-') {
            return Err(ManifestError::InvalidSemVer(format!(
                "id has leading/trailing hyphen: {raw}"
            )));
        }
        let lower = raw.to_lowercase();
        for ch in lower.chars() {
            if !ch.is_ascii_alphanumeric() && ch != '-' && ch != '.' {
                return Err(ManifestError::InvalidPath(format!(
                    "id contains invalid character '{ch}': {raw}"
                )));
            }
        }
        Ok(NormalizedIdV1(lower))
    }
}

/// Validate and normalize `raw`, returning the canonical form.
pub fn normalize_manifest(raw: PluginManifestV2) -> Result<PluginManifestV2, ManifestError> {
    // Version gate: "1.0" → Python format (stable error code)
    if raw.schema_version == "1.0" {
        return Err(ManifestError::PythonManifest(raw.schema_version));
    }
    if raw.schema_version != "2.0" {
        return Err(ManifestError::UnsupportedSchemaVersion(raw.schema_version));
    }

    let packages = raw
        .packages
        .into_iter()
        .map(normalize_package)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PluginManifestV2 {
        schema_version: raw.schema_version,
        packages,
    })
}

fn normalize_package(
    mut pkg: PluginPackageManifestV2,
) -> Result<PluginPackageManifestV2, ManifestError> {
    // Validate id
    NormalizedIdV1::parse(&pkg.id)?;

    // Validate version as canonical SemVer (X.Y.Z)
    Version::parse(&pkg.version).map_err(|_| ManifestError::InvalidSemVer(pkg.version.clone()))?;

    // Validate artifact paths and check for duplicate targets
    let mut seen_targets: HashSet<String> = HashSet::new();
    for artifact in &pkg.artifacts {
        validate_artifact_path(&artifact.path)?;
        if !seen_targets.insert(artifact.target.clone()) {
            return Err(ManifestError::DuplicateBaselineArtifact(
                artifact.target.clone(),
            ));
        }
    }

    // At least one category required
    if pkg.categories.is_empty() {
        return Err(ManifestError::NoCategories);
    }

    // Sort and deduplicate aliases
    pkg.aliases.sort_unstable();
    pkg.aliases.dedup();

    // Sort dependency edges
    pkg.depends_on.sort_unstable();

    Ok(pkg)
}

/// Reject absolute paths, parent-traversal components, and Windows ADS.
fn validate_artifact_path(path: &str) -> Result<(), ManifestError> {
    // Reject Windows ADS: filename contains ':'
    if path.contains(':') {
        return Err(ManifestError::InvalidPath(format!(
            "path contains Windows ADS colon: {path}"
        )));
    }
    // Reject absolute paths
    if path.starts_with('/') || path.starts_with('\\') {
        return Err(ManifestError::InvalidPath(format!(
            "absolute path not allowed: {path}"
        )));
    }
    // Reject drive-letter paths (Windows): C:\...
    if path.len() >= 2 && path.as_bytes()[1] == b'\\' && path.as_bytes()[0].is_ascii_alphabetic() {
        return Err(ManifestError::InvalidPath(format!(
            "Windows drive-letter path not allowed: {path}"
        )));
    }
    // Reject parent traversal components
    for component in path.split('/').chain(path.split('\\')) {
        if component == ".." {
            return Err(ManifestError::InvalidPath(format!(
                "parent-traversal component in path: {path}"
            )));
        }
    }
    Ok(())
}
