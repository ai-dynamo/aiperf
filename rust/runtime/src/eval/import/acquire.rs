// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable source locations and native source acquisition.

use std::{
    fs::{self, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Component, Path},
    process::{Command, Stdio},
};

use tar::Archive;
use tempfile::{NamedTempFile, tempdir};

use super::{AcquiredSource, HarborImportError, source_snapshot::SourceTreeSnapshot};

const MAX_PINNED_GIT_ARCHIVE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_PINNED_GIT_TREE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_PINNED_GIT_TREE_ENTRIES: usize = 10_000;
const MAX_PINNED_GIT_FILE_BYTES: u64 = 128 * 1024 * 1024;

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
                read_local_file_capped(&package)
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
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repository)
        .arg("show")
        .arg(object)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| HarborImportError::Unavailable(format!("{repository}: {error}")))?;
    let stdout = child.stdout.take().ok_or_else(|| {
        HarborImportError::Unavailable("could not capture pinned Git package file".to_owned())
    })?;
    let mut output = NamedTempFile::new().map_err(git_file_error)?;
    if let Err(error) =
        copy_file_stream_capped(stdout, output.as_file_mut(), MAX_PINNED_GIT_FILE_BYTES)
    {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    if !child.wait().map_err(git_file_error)?.success() {
        return Err(HarborImportError::Unavailable(format!(
            "{repository}@{revision}:{package_path}: git show failed"
        )));
    }
    output
        .as_file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(git_file_error)?;
    let mut bytes = Vec::new();
    output
        .as_file_mut()
        .read_to_end(&mut bytes)
        .map_err(git_file_error)?;
    Ok(bytes)
}

fn read_local_file_capped(package: &Path) -> Result<Vec<u8>, HarborImportError> {
    let mut source = open_regular_local_file(package)?;
    let mut temporary = NamedTempFile::new().map_err(git_file_error)?;
    copy_file_stream_capped(
        &mut source,
        temporary.as_file_mut(),
        MAX_PINNED_GIT_FILE_BYTES,
    )?;
    temporary
        .as_file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(git_file_error)?;
    let mut bytes = Vec::new();
    temporary
        .as_file_mut()
        .read_to_end(&mut bytes)
        .map_err(git_file_error)?;
    Ok(bytes)
}

fn open_regular_local_file(package: &Path) -> Result<fs::File, HarborImportError> {
    let source = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(package)
        .map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", package.display()))
        })?;
    let metadata = source.metadata().map_err(|error| {
        HarborImportError::Unavailable(format!("{}: {error}", package.display()))
    })?;
    if !metadata.file_type().is_file() {
        return Err(HarborImportError::Unavailable(format!(
            "{}: package source is not a regular file",
            package.display()
        )));
    }
    Ok(source)
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
    let mut child = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(["archive", "--format=tar", &object])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|error| HarborImportError::Unavailable(format!("{repository}: {error}")))?;
    let mut archive = NamedTempFile::new().map_err(git_tree_error)?;
    let stdout = child.stdout.take().ok_or_else(|| {
        HarborImportError::Unavailable("could not capture pinned Git task tree".to_owned())
    })?;
    if let Err(error) =
        copy_stream_capped(stdout, archive.as_file_mut(), MAX_PINNED_GIT_ARCHIVE_BYTES)
    {
        let _ = child.kill();
        let _ = child.wait();
        return Err(error);
    }
    if !child.wait().map_err(git_tree_error)?.success() {
        return Err(HarborImportError::Unavailable(format!(
            "{repository}@{revision}:{package_path}: git archive failed"
        )));
    }
    let directory = tempdir().map_err(|error| {
        HarborImportError::Unavailable(format!("could not retain pinned Git task tree: {error}"))
    })?;
    archive
        .as_file_mut()
        .seek(SeekFrom::Start(0))
        .map_err(git_tree_error)?;
    extract_git_tree_with_limits(
        archive.as_file_mut(),
        directory.path(),
        MAX_PINNED_GIT_TREE_ENTRIES,
        MAX_PINNED_GIT_TREE_BYTES,
    )?;
    let tree = SourceTreeSnapshot::capture(directory.path())?;
    AcquiredSource::tree("task.toml", tree)
}

fn extract_git_tree_with_limits(
    source: impl Read,
    destination: &Path,
    max_entries: usize,
    max_bytes: u64,
) -> Result<(), HarborImportError> {
    let mut archive = Archive::new(source);
    let mut entries = 0_usize;
    let mut bytes = 0_u64;
    for entry in archive.entries().map_err(git_tree_error)? {
        entries = entries.checked_add(1).ok_or_else(|| {
            HarborImportError::InvalidPackage("pinned Git task tree exceeds entry limit".to_owned())
        })?;
        if entries > max_entries {
            return Err(HarborImportError::InvalidPackage(
                "pinned Git task tree exceeds entry limit".to_owned(),
            ));
        }
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
        copy_entry_capped(&mut entry, &mut output, &mut bytes, max_bytes)?;
        let mode = entry.header().mode().map_err(git_tree_error)? & 0o777;
        fs::set_permissions(&target, fs::Permissions::from_mode(mode)).map_err(git_tree_error)?;
    }
    Ok(())
}

fn copy_stream_capped(
    mut source: impl Read,
    destination: &mut fs::File,
    max_bytes: u64,
) -> Result<(), HarborImportError> {
    let mut bytes = 0_u64;
    copy_entry_capped(&mut source, destination, &mut bytes, max_bytes)
}

fn copy_file_stream_capped(
    mut source: impl Read,
    destination: &mut fs::File,
    max_bytes: u64,
) -> Result<(), HarborImportError> {
    let mut copied = 0_u64;
    let mut buffer = [0_u8; 8192];
    loop {
        let remaining = max_bytes.saturating_sub(copied);
        let read_size = if remaining == 0 {
            1
        } else {
            remaining.min(buffer.len() as u64) as usize
        };
        let read = source
            .read(&mut buffer[..read_size])
            .map_err(git_file_error)?;
        if read == 0 {
            return Ok(());
        }
        copied = copied.checked_add(read as u64).ok_or_else(|| {
            HarborImportError::InvalidPackage("package file exceeds byte limit".to_owned())
        })?;
        if copied > max_bytes {
            return Err(HarborImportError::InvalidPackage(
                "package file exceeds byte limit".to_owned(),
            ));
        }
        destination
            .write_all(&buffer[..read])
            .map_err(git_file_error)?;
    }
}

fn copy_entry_capped(
    source: &mut dyn Read,
    destination: &mut dyn Write,
    bytes: &mut u64,
    max_bytes: u64,
) -> Result<(), HarborImportError> {
    let mut buffer = [0_u8; 8192];
    loop {
        let read = source.read(&mut buffer).map_err(git_tree_error)?;
        if read == 0 {
            return Ok(());
        }
        *bytes = bytes.checked_add(read as u64).ok_or_else(|| {
            HarborImportError::InvalidPackage("pinned Git task tree exceeds byte limit".to_owned())
        })?;
        if *bytes > max_bytes {
            return Err(HarborImportError::InvalidPackage(
                "pinned Git task tree exceeds byte limit".to_owned(),
            ));
        }
        destination
            .write_all(&buffer[..read])
            .map_err(git_tree_error)?;
    }
}

fn git_tree_error(error: impl std::fmt::Display) -> HarborImportError {
    HarborImportError::Unavailable(format!("could not retain pinned Git task tree: {error}"))
}

fn git_file_error(error: impl std::fmt::Display) -> HarborImportError {
    HarborImportError::Unavailable(format!("could not retain package file: {error}"))
}

#[cfg(test)]
mod tests {
    use std::{
        io::{Cursor, Write},
        process::Command,
        sync::mpsc,
        thread,
        time::Duration,
    };

    use super::{copy_file_stream_capped, extract_git_tree_with_limits, read_local_file_capped};

    #[test]
    fn local_file_acquisition_rejects_a_fifo_without_blocking() {
        let temporary = tempfile::tempdir().unwrap();
        let fifo = temporary.path().join("task.json");
        assert!(
            Command::new("mkfifo")
                .arg(&fifo)
                .status()
                .unwrap()
                .success()
        );
        let (sender, receiver) = mpsc::channel();
        thread::spawn(move || {
            sender.send(read_local_file_capped(&fifo)).unwrap();
        });

        let result = receiver
            .recv_timeout(Duration::from_secs(1))
            .expect("local task acquisition must not block on a FIFO");
        assert!(matches!(
            result,
            Err(super::HarborImportError::Unavailable(_))
        ));
    }

    #[test]
    fn pinned_git_file_stream_rejects_bytes_before_the_temporary_file_exceeds_its_cap() {
        let mut output = tempfile::NamedTempFile::new().unwrap();
        let error = copy_file_stream_capped(Cursor::new(b"oversized"), output.as_file_mut(), 8)
            .expect_err("pinned Git files must be capped while streaming");

        assert!(
            matches!(error, super::HarborImportError::InvalidPackage(message) if message.contains("byte limit"))
        );
        output.as_file_mut().flush().unwrap();
        assert!(output.as_file().metadata().unwrap().len() <= 8);
    }

    #[test]
    fn pinned_git_extraction_rejects_more_entries_than_the_cap() {
        let temporary = tempfile::tempdir().unwrap();
        let mut archive = tar::Builder::new(Vec::new());
        for name in ["one", "two"] {
            let mut header = tar::Header::new_gnu();
            header.set_size(1);
            header.set_mode(0o644);
            header.set_cksum();
            archive.append_data(&mut header, name, &b"x"[..]).unwrap();
        }
        let bytes = archive.into_inner().unwrap();

        let error = extract_git_tree_with_limits(Cursor::new(bytes), temporary.path(), 1, 1024)
            .expect_err("a pinned Git tree must reject an entry count above its cap");

        assert!(
            matches!(error, super::HarborImportError::InvalidPackage(message) if message.contains("entry limit"))
        );
    }
}
