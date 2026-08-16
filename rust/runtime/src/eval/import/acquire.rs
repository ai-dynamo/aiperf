// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable source locations and native source acquisition.

use std::{fs, path::Path, process::Command};

use super::HarborImportError;

/// An immutable Harbor-compatible package source reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HarborSource {
    /// A package supplied from a caller-controlled local source location.
    Local(String),
    /// A package at a pinned Git revision.
    PinnedGit {
        /// Local Git repository containing the package.
        repository: String,
        /// Full immutable Git object identifier.
        revision: String,
        /// Repository-relative package file path.
        package_path: String,
    },
    /// An immutable registry package reference.
    Registry(String),
}

impl HarborSource {
    /// Creates a nonempty local source reference.
    pub fn local(location: impl Into<String>) -> Result<Self, HarborImportError> {
        let location = location.into();
        if location.trim().is_empty() {
            return Err(HarborImportError::InvalidSource("local location"));
        }
        Ok(Self::Local(location))
    }

    /// Creates a source reference pinned to one Git commit and package path.
    pub fn pinned_git(
        repository: impl Into<String>,
        revision: impl Into<String>,
        package_path: impl Into<String>,
    ) -> Result<Self, HarborImportError> {
        let repository = repository.into();
        let revision = revision.into();
        let package_path = package_path.into();
        if repository.trim().is_empty() {
            return Err(HarborImportError::InvalidSource("Git repository"));
        }
        if revision.len() != 40
            || !revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte <= b'f'))
        {
            return Err(HarborImportError::InvalidSource("Git revision"));
        }
        if package_path.trim().is_empty()
            || package_path.starts_with('/')
            || package_path
                .split('/')
                .any(|component| component.is_empty() || component == "." || component == "..")
        {
            return Err(HarborImportError::InvalidSource("Git package path"));
        }
        Ok(Self::PinnedGit {
            repository,
            revision,
            package_path,
        })
    }

    /// Returns the stable source location key passed to an acquirer.
    pub fn location(&self) -> &str {
        match self {
            Self::Local(location) | Self::Registry(location) => location,
            Self::PinnedGit { repository, .. } => repository,
        }
    }
}

/// Copies source package bytes into the native importer without provider coupling.
pub trait SourceAcquirer {
    /// Acquires the exact source bytes identified by a source reference.
    fn acquire(&self, source: &HarborSource) -> Result<Vec<u8>, HarborImportError>;
}

/// Native local-file and pinned-Git source acquisition.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeSourceAcquirer;

impl SourceAcquirer for NativeSourceAcquirer {
    fn acquire(&self, source: &HarborSource) -> Result<Vec<u8>, HarborImportError> {
        match source {
            HarborSource::Local(location) => {
                let path = Path::new(location);
                let package = if path.is_dir() {
                    path.join("task.json")
                } else {
                    path.to_path_buf()
                };
                fs::read(&package).map_err(|error| {
                    HarborImportError::Unavailable(format!("{}: {error}", package.display()))
                })
            }
            HarborSource::PinnedGit {
                repository,
                revision,
                package_path,
            } => acquire_git_file(repository, revision, package_path),
            HarborSource::Registry(reference) => Err(HarborImportError::Unavailable(format!(
                "offline native source acquirer cannot fetch registry reference {reference:?}"
            ))),
        }
    }
}

fn acquire_git_file(
    repository: &str,
    revision: &str,
    package_path: &str,
) -> Result<Vec<u8>, HarborImportError> {
    let object = format!("{revision}:{package_path}");
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .arg("show")
        .arg(object)
        .output()
        .map_err(|error| HarborImportError::Unavailable(format!("{repository}: {error}")))?;
    if output.status.success() {
        Ok(output.stdout)
    } else {
        Err(HarborImportError::Unavailable(format!(
            "{repository}@{revision}:{package_path}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )))
    }
}
