// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Capability-limited artifact access shared by every category SDK.
//!
//! [`ArtifactAccess`] is the only artifact I/O a plugin is given. It exposes a
//! scoped listing, scoped reads, and approved relative creates and appends. It
//! deliberately exposes **no** raw directory path and **no** unchecked join: a
//! plugin cannot reconstruct the host's artifact root, and cannot address a
//! path the host did not approve.
//!
//! ```compile_fail
//! use aiperf_core::artifact::ArtifactAccess;
//! fn escape(access: &dyn ArtifactAccess) -> &std::path::Path {
//!     // There is no `raw_path` on the capability, by design.
//!     access.raw_path()
//! }
//! ```

use std::fmt::{self, Display, Formatter};
use std::path::{Component, Path, PathBuf};

/// One artifact visible through a scoped listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactEntry {
    /// Path relative to the host's artifact scope, using `/` separators.
    pub relative_path: String,
    /// Exact byte length of the artifact.
    pub len: u64,
}

/// Why a scoped artifact operation did not happen.
#[derive(Debug)]
pub enum ArtifactError {
    /// The requested relative path is not one the host will approve.
    Rejected(String),
    /// The underlying filesystem operation failed.
    Io(std::io::Error),
}

impl Display for ArtifactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rejected(reason) => write!(formatter, "artifact path rejected: {reason}"),
            Self::Io(error) => write!(formatter, "artifact i/o failed: {error}"),
        }
    }
}

impl std::error::Error for ArtifactError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rejected(_) => None,
            Self::Io(error) => Some(error),
        }
    }
}

impl From<std::io::Error> for ArtifactError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

/// Approve one caller-supplied relative artifact path.
///
/// A path is approved only when it is non-empty, relative, and composed
/// entirely of ordinary named components: no root, no prefix, no `.`, and no
/// `..`. Interior NUL bytes are refused because they truncate at the syscall
/// boundary rather than at this check.
pub fn check_relative(relative_path: &str) -> Result<PathBuf, ArtifactError> {
    if relative_path.is_empty() {
        return Err(ArtifactError::Rejected("empty path".to_owned()));
    }
    if relative_path.contains('\0') {
        return Err(ArtifactError::Rejected(format!(
            "interior NUL in {relative_path:?}"
        )));
    }
    let candidate = Path::new(relative_path);
    let mut approved = PathBuf::new();
    for component in candidate.components() {
        match component {
            Component::Normal(part) => approved.push(part),
            Component::CurDir | Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(ArtifactError::Rejected(format!(
                    "non-relative component in {relative_path:?}"
                )));
            }
        }
    }
    if approved.as_os_str().is_empty() {
        return Err(ArtifactError::Rejected(format!(
            "no named component in {relative_path:?}"
        )));
    }
    Ok(approved)
}

/// Scoped artifact capability handed to a plugin.
///
/// Every method addresses artifacts by a relative path the implementation
/// approves through [`check_relative`]. No method returns, accepts, or derives
/// an absolute host path.
pub trait ArtifactAccess {
    /// List every artifact currently visible in this scope.
    fn list(&self) -> Result<Vec<ArtifactEntry>, ArtifactError>;

    /// Read one artifact's exact bytes.
    fn read(&self, relative_path: &str) -> Result<Vec<u8>, ArtifactError>;

    /// Create or replace one artifact with exactly `contents`.
    fn create(&self, relative_path: &str, contents: &[u8]) -> Result<(), ArtifactError>;

    /// Append `contents` to one artifact, creating it when absent.
    fn append(&self, relative_path: &str, contents: &[u8]) -> Result<(), ArtifactError>;
}

/// The host-side [`ArtifactAccess`] backed by one directory.
///
/// The root is private: it is never handed back through the trait, so a plugin
/// holding this as `&dyn ArtifactAccess` cannot recover it.
#[derive(Debug, Clone)]
pub struct DirectoryArtifacts {
    root: PathBuf,
}

impl DirectoryArtifacts {
    /// Scope artifact access to `root`.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    fn resolve(&self, relative_path: &str) -> Result<PathBuf, ArtifactError> {
        Ok(self.root.join(check_relative(relative_path)?))
    }

    fn walk(&self, directory: &Path, prefix: &str, into: &mut Vec<ArtifactEntry>) -> Result<(), ArtifactError> {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let relative_path = if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}/{name}")
            };
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                self.walk(&entry.path(), &relative_path, into)?;
            } else if metadata.is_file() {
                into.push(ArtifactEntry {
                    relative_path,
                    len: metadata.len(),
                });
            }
        }
        Ok(())
    }
}

impl ArtifactAccess for DirectoryArtifacts {
    fn list(&self) -> Result<Vec<ArtifactEntry>, ArtifactError> {
        let mut entries = Vec::new();
        if self.root.is_dir() {
            self.walk(&self.root.clone(), "", &mut entries)?;
        }
        entries.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
        Ok(entries)
    }

    fn read(&self, relative_path: &str) -> Result<Vec<u8>, ArtifactError> {
        Ok(std::fs::read(self.resolve(relative_path)?)?)
    }

    fn create(&self, relative_path: &str, contents: &[u8]) -> Result<(), ArtifactError> {
        let path = self.resolve(relative_path)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        Ok(std::fs::write(path, contents)?)
    }

    fn append(&self, relative_path: &str, contents: &[u8]) -> Result<(), ArtifactError> {
        use std::io::Write;
        let path = self.resolve(relative_path)?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        file.write_all(contents)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approved_paths_stay_inside_the_scope() {
        assert!(check_relative("a/b.json").is_ok());
        for rejected in ["", "/etc/passwd", "../escape", "./here", "a/../b", "a\0b"] {
            assert!(
                check_relative(rejected).is_err(),
                "approved {rejected:?} unexpectedly"
            );
        }
    }

    #[test]
    fn directory_artifacts_round_trip_within_the_scope() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let artifacts = DirectoryArtifacts::new(root.path());
        artifacts.create("nested/report.json", b"{}").expect("create");
        artifacts.append("nested/report.json", b"\n").expect("append");
        assert_eq!(artifacts.read("nested/report.json").expect("read"), b"{}\n");
        let listed = artifacts.list().expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].relative_path, "nested/report.json");
    }

    #[test]
    fn escapes_are_refused_before_any_filesystem_call() {
        let root = tempfile::tempdir().expect("temporary artifact root");
        let artifacts = DirectoryArtifacts::new(root.path());
        assert!(artifacts.read("../../etc/passwd").is_err());
        assert!(artifacts.create("/tmp/escaped", b"x").is_err());
    }
}
