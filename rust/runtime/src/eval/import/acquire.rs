// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable source locations and native source acquisition.

use std::{
    fs::{self, OpenOptions},
    io::{self, Cursor},
    os::unix::fs::PermissionsExt,
    path::{Component, Path},
    process::Command,
};

use tar::Archive;
use tempfile::tempdir;

use super::{AcquiredSource, HarborImportError, source_snapshot::SourceTreeSnapshot};

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

    /// Acquires an owned artifact detached from its caller-controlled source.
    fn acquire_artifact(&self, source: &HarborSource) -> Result<AcquiredSource, HarborImportError> {
        self.acquire(source).map(AcquiredSource::file)
    }
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
                    let json = path.join("task.json");
                    if json.is_file() {
                        json
                    } else {
                        path.join("task.toml")
                    }
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

    fn acquire_artifact(&self, source: &HarborSource) -> Result<AcquiredSource, HarborImportError> {
        if let HarborSource::Local(location) = source {
            let root = Path::new(location);
            if root.is_dir() {
                let tree = SourceTreeSnapshot::capture(root)?;
                let primary_path = if tree.contains_file("task.json") {
                    "task.json"
                } else {
                    "task.toml"
                };
                return AcquiredSource::tree(primary_path, tree);
            }
        }
        if let HarborSource::PinnedGit {
            repository,
            revision,
            package_path,
        } = source
            && (package_path.ends_with("/task.toml") || package_path == "task.toml")
        {
            return acquire_git_task_tree(repository, revision, package_path);
        }
        self.acquire(source).map(AcquiredSource::file)
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

fn acquire_git_task_tree(
    repository: &str,
    revision: &str,
    package_path: &str,
) -> Result<AcquiredSource, HarborImportError> {
    let task_root = package_path.strip_suffix("/task.toml").unwrap_or_default();
    let object = if task_root.is_empty() {
        revision.to_owned()
    } else {
        format!("{revision}:{task_root}")
    };
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["archive", "--format=tar", &object])
        .output()
        .map_err(|error| HarborImportError::Unavailable(format!("{repository}: {error}")))?;
    if !output.status.success() {
        return Err(HarborImportError::Unavailable(format!(
            "{repository}@{revision}:{package_path}: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let directory = tempdir().map_err(|error| {
        HarborImportError::Unavailable(format!("could not retain pinned Git task tree: {error}"))
    })?;
    extract_git_tree(&output.stdout, directory.path())?;
    let tree = SourceTreeSnapshot::capture(directory.path())?;
    AcquiredSource::tree("task.toml", tree)
}

fn extract_git_tree(archive: &[u8], destination: &Path) -> Result<(), HarborImportError> {
    let mut archive = Archive::new(Cursor::new(archive));
    for entry in archive.entries().map_err(git_tree_error)? {
        let mut entry = entry.map_err(git_tree_error)?;
        let entry_type = entry.header().entry_type();
        if !entry_type.is_file() && !entry_type.is_dir() {
            return Err(HarborImportError::InvalidPackage(
                "pinned Git task tree contains a link or special entry".to_owned(),
            ));
        }
        let path = entry.path().map_err(git_tree_error)?;
        if path.as_os_str().is_empty()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(HarborImportError::InvalidPackage(
                "pinned Git task tree contains an invalid entry path".to_owned(),
            ));
        }
        let target = destination.join(path.as_ref());
        if entry_type.is_dir() {
            fs::create_dir_all(&target).map_err(git_tree_error)?;
            fs::set_permissions(&target, fs::Permissions::from_mode(0o755))
                .map_err(git_tree_error)?;
            continue;
        }
        let parent = target.parent().ok_or_else(|| {
            HarborImportError::InvalidPackage(
                "pinned Git task tree contains a file without a parent".to_owned(),
            )
        })?;
        fs::create_dir_all(parent).map_err(git_tree_error)?;
        let mut output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&target)
            .map_err(git_tree_error)?;
        io::copy(&mut entry, &mut output).map_err(git_tree_error)?;
        let mode = entry.header().mode().map_err(git_tree_error)? & 0o777;
        fs::set_permissions(&target, fs::Permissions::from_mode(mode)).map_err(git_tree_error)?;
    }
    Ok(())
}

fn git_tree_error(error: impl std::fmt::Display) -> HarborImportError {
    HarborImportError::Unavailable(format!("could not retain pinned Git task tree: {error}"))
}
