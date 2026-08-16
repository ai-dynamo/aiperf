// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Owned canonical source-tree snapshots for native evaluation packages.

use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    os::unix::fs::PermissionsExt,
    path::{Component, Path},
    sync::Arc,
};

use crate::eval::ArtifactDigest;

use super::HarborImportError;

const SOURCE_TREE_DOMAIN: &[u8] = b"aiperf-eval-source-tree-v1";
const SOURCE_PROJECTION_DOMAIN: &[u8] = b"aiperf-eval-source-projection-v1";

/// One owned source artifact acquired before package normalization.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcquiredSource {
    primary_path: SourcePath,
    primary_bytes: Arc<[u8]>,
    artifact: SourceArtifact,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum SourceArtifact {
    File(Arc<[u8]>),
    Tree(Arc<SourceTreeSnapshot>),
}

impl AcquiredSource {
    /// Creates a detached single-file package artifact.
    pub fn file(bytes: Vec<u8>) -> Self {
        let bytes = Arc::from(bytes);
        Self {
            primary_path: SourcePath("task.json".to_owned()),
            primary_bytes: Arc::clone(&bytes),
            artifact: SourceArtifact::File(bytes),
        }
    }

    pub(super) fn tree(
        primary_path: &str,
        tree: SourceTreeSnapshot,
    ) -> Result<Self, HarborImportError> {
        let primary_path = SourcePath::parse(primary_path)?;
        let primary_bytes = Arc::clone(tree.file_bytes(&primary_path)?);
        Ok(Self {
            primary_path,
            primary_bytes,
            artifact: SourceArtifact::Tree(Arc::new(tree)),
        })
    }

    /// Returns the exact retained bytes of the package's primary manifest.
    pub fn primary_bytes(&self) -> &[u8] {
        &self.primary_bytes
    }

    /// Returns the provenance digest of the complete acquired artifact.
    pub fn source_digest(&self) -> ArtifactDigest {
        match &self.artifact {
            SourceArtifact::File(bytes) => ArtifactDigest::from_bytes(bytes),
            SourceArtifact::Tree(tree) => tree.digest(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct SourcePath(String);

impl SourcePath {
    pub(super) fn from_relative_path(path: &Path) -> Result<Self, HarborImportError> {
        if path.as_os_str().is_empty()
            || path
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(HarborImportError::InvalidPackage(format!(
                "source entry path must be nonempty and relative: {}",
                path.display()
            )));
        }
        let path = path.to_str().ok_or_else(|| {
            HarborImportError::InvalidPackage("source entry path must be valid UTF-8".to_owned())
        })?;
        Ok(Self(path.to_owned()))
    }

    pub(super) fn parse(path: &str) -> Result<Self, HarborImportError> {
        Self::from_relative_path(Path::new(path))
    }

    pub(super) fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SourceEntryKind {
    Directory,
    File,
}

impl SourceEntryKind {
    const fn tag(self) -> u8 {
        match self {
            Self::Directory => 0,
            Self::File => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SourceEntry {
    path: SourcePath,
    kind: SourceEntryKind,
    mode: u32,
    bytes: Arc<[u8]>,
}

impl SourceEntry {
    pub(super) fn path(&self) -> &SourcePath {
        &self.path
    }

    pub(super) const fn kind(&self) -> SourceEntryKind {
        self.kind
    }

    pub(super) const fn mode(&self) -> u32 {
        self.mode
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct SourceTreeSnapshot {
    entries: Vec<SourceEntry>,
}

impl SourceTreeSnapshot {
    pub(super) fn capture(root: &Path) -> Result<Self, HarborImportError> {
        let metadata = fs::symlink_metadata(root).map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", root.display()))
        })?;
        if !metadata.file_type().is_dir() {
            return Err(HarborImportError::InvalidPackage(format!(
                "source tree root must be a directory: {}",
                root.display()
            )));
        }

        let mut entries = Vec::new();
        Self::capture_directory(root, root, &mut entries)?;
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(Self { entries })
    }

    fn capture_directory(
        root: &Path,
        directory: &Path,
        entries: &mut Vec<SourceEntry>,
    ) -> Result<(), HarborImportError> {
        let children = fs::read_dir(directory)
            .map_err(|error| {
                HarborImportError::Unavailable(format!("{}: {error}", directory.display()))
            })?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| {
                HarborImportError::Unavailable(format!("{}: {error}", directory.display()))
            })?;
        for child in children {
            let path = child.path();
            let relative = path.strip_prefix(root).map_err(|error| {
                HarborImportError::Unavailable(format!("{}: {error}", path.display()))
            })?;
            let source_path = SourcePath::from_relative_path(relative)?;
            let file_type = child.file_type().map_err(|error| {
                HarborImportError::Unavailable(format!("{}: {error}", path.display()))
            })?;
            if file_type.is_dir() {
                entries.push(SourceEntry {
                    path: source_path,
                    kind: SourceEntryKind::Directory,
                    mode: 0o755,
                    bytes: Arc::from(Vec::<u8>::new()),
                });
                Self::capture_directory(root, &path, entries)?;
            } else if file_type.is_file() {
                let metadata = child.metadata().map_err(|error| {
                    HarborImportError::Unavailable(format!("{}: {error}", path.display()))
                })?;
                let mode = if metadata.permissions().mode() & 0o111 == 0 {
                    0o644
                } else {
                    0o755
                };
                entries.push(SourceEntry {
                    path: source_path,
                    kind: SourceEntryKind::File,
                    mode,
                    bytes: Arc::from(fs::read(&path).map_err(|error| {
                        HarborImportError::Unavailable(format!("{}: {error}", path.display()))
                    })?),
                });
            } else {
                return Err(HarborImportError::InvalidPackage(format!(
                    "source entry must be a regular file or directory: {}",
                    source_path.as_str()
                )));
            }
        }
        Ok(())
    }

    pub(super) fn entries(&self) -> &[SourceEntry] {
        &self.entries
    }

    pub(super) fn read(&self, relative_path: &str) -> Result<&[u8], HarborImportError> {
        let relative_path = SourcePath::parse(relative_path)?;
        self.file_bytes(&relative_path).map(AsRef::as_ref)
    }

    pub(super) fn contains_file(&self, relative_path: &str) -> bool {
        self.read(relative_path).is_ok()
    }

    fn file_bytes(&self, relative_path: &SourcePath) -> Result<&Arc<[u8]>, HarborImportError> {
        self.entries
            .binary_search_by(|entry| entry.path.cmp(relative_path))
            .ok()
            .and_then(|index| self.entries.get(index))
            .filter(|entry| entry.kind == SourceEntryKind::File)
            .map(|entry| &entry.bytes)
            .ok_or_else(|| {
                HarborImportError::InvalidPackage(format!(
                    "source file is missing: {:?}",
                    relative_path.as_str()
                ))
            })
    }

    pub(super) fn digest(&self) -> ArtifactDigest {
        digest_entries(SOURCE_TREE_DOMAIN, &self.entries)
    }

    pub(super) fn project_digest<I, S>(&self, roots: I) -> Result<ArtifactDigest, HarborImportError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut roots = roots
            .into_iter()
            .map(|root| SourcePath::parse(root.as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        roots.sort();
        roots.dedup();
        if roots.is_empty()
            || roots
                .iter()
                .any(|root| !self.entries.iter().any(|entry| entry.path == *root))
        {
            return Err(HarborImportError::InvalidPackage(
                "source projection root is missing".to_owned(),
            ));
        }
        let selected = self
            .entries
            .iter()
            .filter(|entry| {
                roots.iter().any(|root| {
                    Path::new(entry.path.as_str()).starts_with(Path::new(root.as_str()))
                })
            })
            .cloned()
            .collect::<Vec<_>>();
        Ok(digest_entries(SOURCE_PROJECTION_DOMAIN, &selected))
    }

    pub(super) fn materialize_into(&self, destination: &Path) -> io::Result<()> {
        let metadata = fs::symlink_metadata(destination)?;
        if !metadata.file_type().is_dir() || fs::read_dir(destination)?.next().is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "source materialization destination must be an empty directory",
            ));
        }
        for entry in self
            .entries
            .iter()
            .filter(|entry| entry.kind == SourceEntryKind::Directory)
        {
            let target = destination.join(entry.path.as_str());
            ensure_real_parent(destination, &target)?;
            fs::create_dir(&target)?;
            fs::set_permissions(&target, fs::Permissions::from_mode(entry.mode))?;
        }
        for entry in self
            .entries
            .iter()
            .filter(|entry| entry.kind == SourceEntryKind::File)
        {
            let target = destination.join(entry.path.as_str());
            ensure_real_parent(destination, &target)?;
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&target)?;
            file.write_all(&entry.bytes)?;
            file.sync_all()?;
            fs::set_permissions(&target, fs::Permissions::from_mode(entry.mode))?;
        }
        Ok(())
    }
}

fn ensure_real_parent(root: &Path, target: &Path) -> io::Result<()> {
    let parent = target.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "source entry has no materialization parent",
        )
    })?;
    if !parent.starts_with(root) || !fs::symlink_metadata(parent)?.file_type().is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source entry parent is not a real directory",
        ));
    }
    Ok(())
}

fn digest_entries(domain: &[u8], entries: &[SourceEntry]) -> ArtifactDigest {
    let mut material = Vec::new();
    append_bytes(&mut material, domain);
    material.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for entry in entries {
        append_bytes(&mut material, entry.path.as_str().as_bytes());
        material.push(entry.kind.tag());
        material.extend_from_slice(&entry.mode.to_le_bytes());
        append_bytes(&mut material, &entry.bytes);
    }
    ArtifactDigest::from_bytes(&material)
}

fn append_bytes(material: &mut Vec<u8>, value: &[u8]) {
    material.extend_from_slice(&(value.len() as u64).to_le_bytes());
    material.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use std::{
        ffi::OsString,
        fs,
        os::{
            unix::ffi::OsStringExt,
            unix::fs::{PermissionsExt, symlink},
        },
        path::Path,
        process::Command,
    };

    use super::{SourceEntryKind, SourcePath, SourceTreeSnapshot};

    #[test]
    fn capture_orders_entries_and_normalizes_modes_independent_of_creation_order() {
        let first = tempfile::tempdir().unwrap();
        fs::create_dir_all(first.path().join("environment/nested/empty")).unwrap();
        fs::create_dir_all(first.path().join("tests")).unwrap();
        fs::write(first.path().join("tests/helper.sh"), b"#!/bin/sh\nexit 0\n").unwrap();
        fs::write(first.path().join("environment/context.txt"), b"context\n").unwrap();
        fs::set_permissions(
            first.path().join("tests/helper.sh"),
            fs::Permissions::from_mode(0o711),
        )
        .unwrap();

        let second = tempfile::tempdir().unwrap();
        fs::create_dir_all(second.path().join("tests")).unwrap();
        fs::create_dir_all(second.path().join("environment/nested/empty")).unwrap();
        fs::write(second.path().join("environment/context.txt"), b"context\n").unwrap();
        fs::write(
            second.path().join("tests/helper.sh"),
            b"#!/bin/sh\nexit 0\n",
        )
        .unwrap();
        fs::set_permissions(
            second.path().join("tests/helper.sh"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();

        let first = SourceTreeSnapshot::capture(first.path()).unwrap();
        let second = SourceTreeSnapshot::capture(second.path()).unwrap();
        let entries = first
            .entries()
            .iter()
            .map(|entry| (entry.path().as_str(), entry.kind(), entry.mode()))
            .collect::<Vec<_>>();

        assert_eq!(
            entries,
            vec![
                ("environment", SourceEntryKind::Directory, 0o755),
                ("environment/context.txt", SourceEntryKind::File, 0o644),
                ("environment/nested", SourceEntryKind::Directory, 0o755),
                (
                    "environment/nested/empty",
                    SourceEntryKind::Directory,
                    0o755,
                ),
                ("tests", SourceEntryKind::Directory, 0o755),
                ("tests/helper.sh", SourceEntryKind::File, 0o755),
            ]
        );
        assert_eq!(first.digest(), second.digest());
    }

    #[test]
    fn empty_directories_and_executable_bits_independently_change_the_tree_digest() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("helper.sh"), b"exit 0\n").unwrap();
        let plain = SourceTreeSnapshot::capture(root.path()).unwrap().digest();

        fs::create_dir(root.path().join("empty")).unwrap();
        let with_empty = SourceTreeSnapshot::capture(root.path()).unwrap().digest();
        assert_ne!(plain, with_empty);

        fs::set_permissions(
            root.path().join("helper.sh"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        let executable = SourceTreeSnapshot::capture(root.path()).unwrap().digest();
        assert_ne!(with_empty, executable);
    }

    #[test]
    fn snapshot_reads_projects_and_materializes_after_origin_mutation_and_deletion() {
        let origin = tempfile::tempdir().unwrap();
        fs::create_dir_all(origin.path().join("environment/empty")).unwrap();
        fs::create_dir_all(origin.path().join("tests")).unwrap();
        fs::write(
            origin.path().join("environment/context.txt"),
            b"original context\n",
        )
        .unwrap();
        fs::write(origin.path().join("tests/test.sh"), b"#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(
            origin.path().join("tests/test.sh"),
            fs::Permissions::from_mode(0o755),
        )
        .unwrap();
        let snapshot = SourceTreeSnapshot::capture(origin.path()).unwrap();
        let digest = snapshot.digest();
        let environment_digest = snapshot.project_digest(["environment"]).unwrap();

        fs::write(origin.path().join("environment/context.txt"), b"mutated\n").unwrap();
        fs::remove_dir_all(origin.path().join("tests")).unwrap();
        fs::remove_dir_all(origin.path().join("environment")).unwrap();

        assert_eq!(
            snapshot.read("environment/context.txt").unwrap(),
            b"original context\n"
        );
        assert_eq!(snapshot.digest(), digest);
        assert_eq!(
            snapshot.project_digest(["environment"]).unwrap(),
            environment_digest
        );

        let materialized = tempfile::tempdir().unwrap();
        snapshot.materialize_into(materialized.path()).unwrap();
        assert_eq!(
            fs::read(materialized.path().join("environment/context.txt")).unwrap(),
            b"original context\n"
        );
        assert!(materialized.path().join("environment/empty").is_dir());
        assert_eq!(
            fs::metadata(materialized.path().join("tests/test.sh"))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o755
        );
    }

    #[test]
    fn snapshot_rejects_links_special_entries_non_utf8_and_escaping_paths() {
        let symlink_root = tempfile::tempdir().unwrap();
        fs::write(symlink_root.path().join("target"), b"target").unwrap();
        symlink("target", symlink_root.path().join("link")).unwrap();
        assert!(SourceTreeSnapshot::capture(symlink_root.path()).is_err());

        let special_root = tempfile::tempdir().unwrap();
        assert!(
            Command::new("mkfifo")
                .arg(special_root.path().join("fifo"))
                .status()
                .unwrap()
                .success()
        );
        assert!(SourceTreeSnapshot::capture(special_root.path()).is_err());

        let non_utf8_root = tempfile::tempdir().unwrap();
        let non_utf8 = OsString::from_vec(vec![b'n', b'o', b'n', 0xff]);
        fs::write(non_utf8_root.path().join(non_utf8), b"bytes").unwrap();
        assert!(SourceTreeSnapshot::capture(non_utf8_root.path()).is_err());

        assert!(SourcePath::from_relative_path(Path::new("../escape")).is_err());
        assert!(SourcePath::from_relative_path(Path::new("/absolute")).is_err());
        assert!(SourcePath::from_relative_path(Path::new("")).is_err());
    }
}
