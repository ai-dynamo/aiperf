// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Harbor-compatible importer that preserves source bytes before normalization.

use std::{
    fmt::{self, Display, Formatter},
    fs,
    path::{Path, PathBuf},
};

use crate::eval::{ArtifactDigest, EvalTaskRef, ImportDisposition, ImportReport};

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
        let bytes = self.acquirer.acquire(source)?;
        let source_digest = source_digest(source, &bytes)?;
        if has_unsupported_semantics(&bytes) {
            return Err(HarborImportError::Unsupported(ImportReport {
                source_digest,
                normalized_digest: ArtifactDigest::from_bytes(&[]),
                disposition: ImportDisposition::Unsupported,
            }));
        }
        let (mut package, task) = match source {
            HarborSource::Local(location) if std::path::Path::new(location).is_dir() => {
                let source_root = std::path::Path::new(location);
                if source_root.join("task.toml").is_file() {
                    normalize::normalize_standard_directory(source_root, &bytes)?
                } else {
                    normalize::normalize(&bytes)?
                }
            }
            _ => normalize::normalize(&bytes)?,
        };
        if let HarborSource::Local(location) = source {
            let path = std::path::Path::new(location);
            let source_root = path.is_dir().then(|| path.to_path_buf()).or_else(|| {
                path.is_file()
                    .then(|| path.parent().map(std::path::Path::to_path_buf))
                    .flatten()
            });
            if let Some(source_root) = source_root {
                package.set_source_root(source_root);
            }
        }
        package.set_source_digest(source_digest.clone());
        let report = ImportReport {
            source_digest,
            normalized_digest: task.digest.clone(),
            disposition: ImportDisposition::LosslessNormalized,
        };
        Ok(ImportedTask {
            task,
            report,
            package,
        })
    }
}

fn source_digest(source: &HarborSource, bytes: &[u8]) -> Result<ArtifactDigest, HarborImportError> {
    let HarborSource::Local(location) = source else {
        return Ok(ArtifactDigest::from_bytes(bytes));
    };
    let root = Path::new(location);
    if !root.is_dir() || !root.join("task.toml").is_file() {
        return Ok(ArtifactDigest::from_bytes(bytes));
    }
    let mut files = Vec::new();
    collect_source_files(root, root, &mut files)?;
    let mut material = Vec::new();
    for file in files {
        let relative = file
            .strip_prefix(root)
            .map_err(|error| HarborImportError::Unavailable(error.to_string()))?;
        let name = relative.to_string_lossy();
        let contents = fs::read(&file).map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", file.display()))
        })?;
        material.extend_from_slice(name.as_bytes());
        material.push(0);
        material.extend_from_slice(&(contents.len() as u64).to_le_bytes());
        material.extend_from_slice(&contents);
    }
    Ok(ArtifactDigest::from_bytes(&material))
}

fn collect_source_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<PathBuf>,
) -> Result<(), HarborImportError> {
    let mut entries = fs::read_dir(directory)
        .map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", directory.display()))
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", directory.display()))
        })?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let kind = entry.file_type().map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", path.display()))
        })?;
        if kind.is_dir() {
            collect_source_files(root, &path, files)?;
        } else if kind.is_file() {
            files.push(path);
        } else {
            return Err(HarborImportError::InvalidPackage(format!(
                "source entry must be a regular file or directory: {}",
                path.strip_prefix(root).unwrap_or(&path).display()
            )));
        }
    }
    Ok(())
}

fn has_unsupported_semantics(bytes: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(bytes)
        .ok()
        .and_then(|value| value.get("unsupported_semantics").cloned())
        .is_some()
}
