// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Owned canonical source-tree snapshots for native evaluation packages.

use std::{
    collections::BTreeSet,
    ffi::{CStr, CString, OsString},
    fs::{self, File, Metadata, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd, IntoRawFd},
        unix::{
            ffi::{OsStrExt, OsStringExt},
            fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        },
    },
    path::{Component, Path},
    sync::Arc,
};

use crate::eval::ArtifactDigest;

use super::HarborImportError;

const SOURCE_TREE_DOMAIN: &[u8] = b"aiperf-eval-source-tree-v1";
const EXECUTABLE_SOURCE_DOMAIN: &[u8] = b"aiperf-eval-executable-source-v1";
const MAX_SOURCE_TREE_BYTES: u64 = 128 * 1024 * 1024;
const MAX_SOURCE_TREE_ENTRIES: usize = 10_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) enum ExecutableSourceView {
    PrimaryFile,
    WholeTree,
    SelectedRoots(BTreeSet<SourcePath>),
}

impl ExecutableSourceView {
    pub(super) fn selected_roots<I, S>(roots: I) -> Result<Self, HarborImportError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let roots = roots
            .into_iter()
            .map(|root| SourcePath::parse(root.as_ref()))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if roots.is_empty() {
            return Err(HarborImportError::InvalidPackage(
                "executable source view must select at least one root".to_owned(),
            ));
        }
        Ok(Self::SelectedRoots(roots))
    }
}

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

    pub(super) fn primary_path(&self) -> &str {
        self.primary_path.as_str()
    }

    pub(super) const fn is_tree(&self) -> bool {
        matches!(&self.artifact, SourceArtifact::Tree(_))
    }

    pub(crate) fn read(&self, relative_path: &str) -> Result<&[u8], HarborImportError> {
        match &self.artifact {
            SourceArtifact::File(_) if relative_path == self.primary_path.as_str() => {
                Ok(&self.primary_bytes)
            }
            SourceArtifact::File(_) => Err(HarborImportError::InvalidPackage(format!(
                "source file is missing: {relative_path:?}"
            ))),
            SourceArtifact::Tree(tree) => tree.read(relative_path),
        }
    }

    pub(crate) fn read_owned(&self, relative_path: &str) -> Result<Arc<[u8]>, HarborImportError> {
        match &self.artifact {
            SourceArtifact::File(_) if relative_path == self.primary_path.as_str() => {
                Ok(Arc::clone(&self.primary_bytes))
            }
            SourceArtifact::File(_) => Err(HarborImportError::InvalidPackage(format!(
                "source file is missing: {relative_path:?}"
            ))),
            SourceArtifact::Tree(tree) => tree.read_owned(relative_path),
        }
    }

    pub(super) fn contains_file(&self, relative_path: &str) -> bool {
        self.read(relative_path).is_ok()
    }

    pub(super) fn contains_path(&self, relative_path: &str) -> bool {
        match &self.artifact {
            SourceArtifact::File(_) => relative_path == self.primary_path.as_str(),
            SourceArtifact::Tree(tree) => tree.contains_path(relative_path),
        }
    }

    pub(super) fn executable_source_digest(
        &self,
        view: &ExecutableSourceView,
    ) -> Result<ArtifactDigest, HarborImportError> {
        match (view, &self.artifact) {
            (ExecutableSourceView::PrimaryFile, SourceArtifact::File(_)) => Ok(digest_file_entry(
                EXECUTABLE_SOURCE_DOMAIN,
                &self.primary_path,
                &self.primary_bytes,
            )),
            (ExecutableSourceView::PrimaryFile, SourceArtifact::Tree(tree)) => {
                tree.project_digest(std::iter::once(self.primary_path.as_str()))
            }
            (ExecutableSourceView::WholeTree, SourceArtifact::Tree(tree)) => {
                Ok(digest_entries(EXECUTABLE_SOURCE_DOMAIN, &tree.entries))
            }
            (ExecutableSourceView::SelectedRoots(roots), SourceArtifact::Tree(tree)) => {
                tree.project_source_digest(roots)
            }
            (ExecutableSourceView::WholeTree | ExecutableSourceView::SelectedRoots(_), _) => {
                Err(HarborImportError::InvalidPackage(
                    "single-file source cannot expose a tree view".to_owned(),
                ))
            }
        }
    }

    pub(super) fn materialize_into(&self, destination: &Path) -> io::Result<()> {
        match &self.artifact {
            SourceArtifact::Tree(tree) => tree.materialize_into(destination),
            SourceArtifact::File(_) => {
                ensure_empty_directory(destination)?;
                let target = destination.join(self.primary_path.as_str());
                let mut file = OpenOptions::new()
                    .create_new(true)
                    .write(true)
                    .open(&target)?;
                file.write_all(&self.primary_bytes)?;
                fs::set_permissions(target, fs::Permissions::from_mode(0o644))
            }
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
    #[cfg(test)]
    pub(super) fn path(&self) -> &SourcePath {
        &self.path
    }

    #[cfg(test)]
    pub(super) const fn kind(&self) -> SourceEntryKind {
        self.kind
    }

    #[cfg(test)]
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
        Self::capture_with_hook(root, &mut |_relative_path: &Path| {})
    }

    #[cfg(test)]
    fn capture_with_before_open(
        root: &Path,
        mut before_open: impl FnMut(&Path),
    ) -> Result<Self, HarborImportError> {
        Self::capture_with_hook(root, &mut before_open)
    }

    fn capture_with_hook<F>(root: &Path, before_open: &mut F) -> Result<Self, HarborImportError>
    where
        F: FnMut(&Path),
    {
        let path_metadata = fs::symlink_metadata(root).map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", root.display()))
        })?;
        if !path_metadata.file_type().is_dir() {
            return Err(HarborImportError::InvalidPackage(format!(
                "source tree root must be a directory: {}",
                root.display()
            )));
        }
        let root_directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
            .open(root)
            .map_err(|error| {
                HarborImportError::Unavailable(format!("{}: {error}", root.display()))
            })?;
        let opened_metadata = root_directory.metadata().map_err(|error| {
            HarborImportError::Unavailable(format!("{}: {error}", root.display()))
        })?;
        if !same_opened_object(&path_metadata, &opened_metadata)
            || !opened_metadata.file_type().is_dir()
        {
            return Err(source_changed_error(Path::new(".")));
        }

        let mut entries = Vec::new();
        let mut budget = SourceTreeCaptureBudget::default();
        Self::capture_directory(
            &root_directory,
            Path::new(""),
            &mut entries,
            &mut budget,
            before_open,
        )?;
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(Self { entries })
    }

    fn capture_directory<F>(
        directory: &File,
        relative_directory: &Path,
        entries: &mut Vec<SourceEntry>,
        budget: &mut SourceTreeCaptureBudget,
        before_open: &mut F,
    ) -> Result<(), HarborImportError>
    where
        F: FnMut(&Path),
    {
        let before = directory
            .metadata()
            .map_err(|error| source_unavailable_error(relative_directory, error))?;
        let children = read_directory(directory)
            .map_err(|error| source_unavailable_error(relative_directory, error))?;
        for child in children {
            let relative = relative_directory.join(&child.name);
            let source_path = SourcePath::from_relative_path(&relative)?;
            if !is_supported_directory_entry_type(child.entry_type) {
                return Err(unsupported_source_entry_error(&source_path));
            }
            before_open(&relative);
            let mut opened = open_source_entry(directory, &child.name, &source_path)?;
            let metadata = opened
                .metadata()
                .map_err(|error| source_unavailable_error(&relative, error))?;
            if (child.inode != 0 && child.inode != metadata.ino())
                || !directory_entry_type_matches(child.entry_type, &metadata)
            {
                return Err(source_changed_error(&relative));
            }
            if metadata.file_type().is_dir() {
                budget.reserve_entry(&source_path)?;
                entries.push(SourceEntry {
                    path: source_path,
                    kind: SourceEntryKind::Directory,
                    mode: 0o755,
                    bytes: Arc::from(Vec::<u8>::new()),
                });
                Self::capture_directory(&opened, &relative, entries, budget, before_open)?;
            } else if metadata.file_type().is_file() {
                budget.reserve_entry(&source_path)?;
                let mode = if metadata.permissions().mode() & 0o111 == 0 {
                    0o644
                } else {
                    0o755
                };
                let mut bytes = Vec::new();
                read_source_file_bounded(&mut opened, &relative, &mut bytes, budget)?;
                let after = opened
                    .metadata()
                    .map_err(|error| source_unavailable_error(&relative, error))?;
                if metadata_fingerprint(&metadata) != metadata_fingerprint(&after) {
                    return Err(source_changed_error(&relative));
                }
                entries.push(SourceEntry {
                    path: source_path,
                    kind: SourceEntryKind::File,
                    mode,
                    bytes: Arc::from(bytes),
                });
            } else {
                return Err(unsupported_source_entry_error(&source_path));
            }
        }
        let after = directory
            .metadata()
            .map_err(|error| source_unavailable_error(relative_directory, error))?;
        if metadata_fingerprint(&before) != metadata_fingerprint(&after) {
            return Err(source_changed_error(relative_directory));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn entries(&self) -> &[SourceEntry] {
        &self.entries
    }

    pub(super) fn read(&self, relative_path: &str) -> Result<&[u8], HarborImportError> {
        let relative_path = SourcePath::parse(relative_path)?;
        self.file_bytes(&relative_path).map(AsRef::as_ref)
    }

    fn read_owned(&self, relative_path: &str) -> Result<Arc<[u8]>, HarborImportError> {
        let relative_path = SourcePath::parse(relative_path)?;
        self.file_bytes(&relative_path).map(Arc::clone)
    }

    pub(super) fn contains_file(&self, relative_path: &str) -> bool {
        self.read(relative_path).is_ok()
    }

    fn contains_path(&self, relative_path: &str) -> bool {
        SourcePath::parse(relative_path).is_ok_and(|relative_path| {
            self.entries
                .binary_search_by(|entry| entry.path.cmp(&relative_path))
                .is_ok()
        })
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
        let roots = roots
            .into_iter()
            .map(|root| SourcePath::parse(root.as_ref()))
            .collect::<Result<BTreeSet<_>, _>>()?;
        self.project_source_digest(&roots)
    }

    fn project_source_digest(
        &self,
        roots: &BTreeSet<SourcePath>,
    ) -> Result<ArtifactDigest, HarborImportError> {
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
        Ok(digest_entries(EXECUTABLE_SOURCE_DOMAIN, &selected))
    }

    pub(super) fn materialize_into(&self, destination: &Path) -> io::Result<()> {
        ensure_empty_directory(destination)?;
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

#[derive(Default)]
struct SourceTreeCaptureBudget {
    entries: usize,
    bytes: u64,
}

impl SourceTreeCaptureBudget {
    fn reserve_entry(&mut self, path: &SourcePath) -> Result<(), HarborImportError> {
        self.entries = self.entries.checked_add(1).ok_or_else(|| {
            HarborImportError::InvalidPackage("source tree exceeds entry limit".to_owned())
        })?;
        if self.entries > MAX_SOURCE_TREE_ENTRIES {
            return Err(HarborImportError::InvalidPackage(format!(
                "source tree exceeds the {MAX_SOURCE_TREE_ENTRIES} entry limit at {:?}",
                path.as_str()
            )));
        }
        Ok(())
    }
}

fn read_source_file_bounded(
    source: &mut File,
    relative: &Path,
    destination: &mut Vec<u8>,
    budget: &mut SourceTreeCaptureBudget,
) -> Result<(), HarborImportError> {
    let mut buffer = [0_u8; 8192];
    loop {
        let remaining = MAX_SOURCE_TREE_BYTES.saturating_sub(budget.bytes);
        let read_size = if remaining == 0 {
            1
        } else {
            remaining.min(buffer.len() as u64) as usize
        };
        let read = source
            .read(&mut buffer[..read_size])
            .map_err(|error| source_unavailable_error(relative, error))?;
        if read == 0 {
            return Ok(());
        }
        budget.bytes = budget.bytes.checked_add(read as u64).ok_or_else(|| {
            HarborImportError::InvalidPackage("source tree exceeds byte limit".to_owned())
        })?;
        if budget.bytes > MAX_SOURCE_TREE_BYTES {
            return Err(HarborImportError::InvalidPackage(format!(
                "source tree exceeds the {MAX_SOURCE_TREE_BYTES} byte limit at {}",
                relative.display()
            )));
        }
        destination.extend_from_slice(&buffer[..read]);
    }
}

#[derive(Debug)]
struct EnumeratedSourceEntry {
    name: OsString,
    inode: u64,
    entry_type: u8,
}

struct DirectoryStream(*mut libc::DIR);

impl DirectoryStream {
    fn open(directory: &File) -> io::Result<Self> {
        let descriptor = directory.try_clone()?.into_raw_fd();
        // SAFETY: `descriptor` is a valid owned directory descriptor. On success,
        // `fdopendir` takes ownership and `DirectoryStream::drop` closes it.
        let stream = unsafe { libc::fdopendir(descriptor) };
        if stream.is_null() {
            let error = io::Error::last_os_error();
            // SAFETY: `fdopendir` did not take ownership on failure, so rebuilding
            // the `File` transfers the still-owned descriptor back to RAII cleanup.
            drop(unsafe { File::from_raw_fd(descriptor) });
            return Err(error);
        }
        Ok(Self(stream))
    }

    fn read_entries(&mut self) -> io::Result<Vec<EnumeratedSourceEntry>> {
        let mut entries = Vec::new();
        loop {
            set_errno(0);
            // SAFETY: `self.0` remains a live `DIR*` owned by this stream, and the
            // returned pointer is consumed before the next `readdir` call.
            let entry = unsafe { libc::readdir(self.0) };
            if entry.is_null() {
                let error = current_errno();
                if error == 0 {
                    break;
                }
                return Err(io::Error::from_raw_os_error(error));
            }
            // SAFETY: POSIX guarantees that `d_name` is NUL-terminated for the
            // lifetime of the current directory entry.
            let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) }.to_bytes();
            if name == b"." || name == b".." {
                continue;
            }
            entries.push(EnumeratedSourceEntry {
                name: OsString::from_vec(name.to_vec()),
                // SAFETY: `entry` is non-null and valid until the next readdir.
                inode: unsafe { (*entry).d_ino as u64 },
                // SAFETY: same as for `d_ino`; copy the value before advancing.
                entry_type: unsafe { (*entry).d_type },
            });
        }
        entries.sort_by(|left, right| left.name.as_bytes().cmp(right.name.as_bytes()));
        Ok(entries)
    }
}

impl Drop for DirectoryStream {
    fn drop(&mut self) {
        // SAFETY: this stream uniquely owns the live `DIR*` returned by fdopendir.
        let _ = unsafe { libc::closedir(self.0) };
    }
}

fn read_directory(directory: &File) -> io::Result<Vec<EnumeratedSourceEntry>> {
    DirectoryStream::open(directory)?.read_entries()
}

fn open_source_entry(
    directory: &File,
    name: &std::ffi::OsStr,
    source_path: &SourcePath,
) -> Result<File, HarborImportError> {
    let name = CString::new(name.as_bytes()).map_err(|_| {
        HarborImportError::InvalidPackage(format!(
            "source entry name contains NUL: {:?}",
            source_path.as_str()
        ))
    })?;
    // SAFETY: `directory` is a live directory descriptor and `name` is a
    // NUL-terminated single directory-entry name. O_NOFOLLOW prevents the final
    // component from becoming a link traversal between enumeration and open.
    let descriptor = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK,
        )
    };
    if descriptor < 0 {
        let error = io::Error::last_os_error();
        return Err(match error.raw_os_error() {
            Some(libc::ELOOP | libc::ENOENT | libc::ENOTDIR | libc::ESTALE) => {
                source_changed_error(Path::new(source_path.as_str()))
            }
            _ => HarborImportError::Unavailable(format!("{}: {error}", source_path.as_str())),
        });
    }
    // SAFETY: a successful `openat` returns one new owned descriptor.
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

fn is_supported_directory_entry_type(entry_type: u8) -> bool {
    entry_type == libc::DT_UNKNOWN || entry_type == libc::DT_DIR || entry_type == libc::DT_REG
}

fn directory_entry_type_matches(entry_type: u8, metadata: &Metadata) -> bool {
    entry_type == libc::DT_UNKNOWN
        || (entry_type == libc::DT_DIR && metadata.file_type().is_dir())
        || (entry_type == libc::DT_REG && metadata.file_type().is_file())
}

fn same_opened_object(before: &Metadata, after: &Metadata) -> bool {
    before.dev() == after.dev()
        && before.ino() == after.ino()
        && before.file_type().is_dir() == after.file_type().is_dir()
        && before.file_type().is_file() == after.file_type().is_file()
}

fn metadata_fingerprint(metadata: &Metadata) -> (u64, u64, u32, u64, i64, i64, i64, i64) {
    (
        metadata.dev(),
        metadata.ino(),
        metadata.mode(),
        metadata.size(),
        metadata.mtime(),
        metadata.mtime_nsec(),
        metadata.ctime(),
        metadata.ctime_nsec(),
    )
}

fn unsupported_source_entry_error(source_path: &SourcePath) -> HarborImportError {
    HarborImportError::InvalidPackage(format!(
        "source entry must be a regular file or directory: {}",
        source_path.as_str()
    ))
}

fn source_changed_error(relative_path: &Path) -> HarborImportError {
    HarborImportError::InvalidPackage(format!(
        "source entry changed during acquisition or became a link: {}",
        relative_path.display()
    ))
}

fn source_unavailable_error(relative_path: &Path, error: io::Error) -> HarborImportError {
    HarborImportError::Unavailable(format!("{}: {error}", relative_path.display()))
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn errno_pointer() -> *mut libc::c_int {
    // SAFETY: libc exposes the calling thread's errno location on these targets.
    unsafe { libc::__errno_location() }
}

#[cfg(any(
    target_os = "macos",
    target_os = "ios",
    target_os = "freebsd",
    target_os = "dragonfly",
    target_os = "openbsd",
    target_os = "netbsd"
))]
fn errno_pointer() -> *mut libc::c_int {
    // SAFETY: libc exposes the calling thread's errno location on these targets.
    unsafe { libc::__error() }
}

fn set_errno(value: libc::c_int) {
    // SAFETY: `errno_pointer` returns this thread's writable errno cell.
    unsafe { *errno_pointer() = value };
}

fn current_errno() -> libc::c_int {
    // SAFETY: `errno_pointer` returns this thread's readable errno cell.
    unsafe { *errno_pointer() }
}

fn ensure_empty_directory(destination: &Path) -> io::Result<()> {
    let metadata = fs::symlink_metadata(destination)?;
    if !metadata.file_type().is_dir() || fs::read_dir(destination)?.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "source materialization destination must be an empty directory",
        ));
    }
    Ok(())
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

fn digest_file_entry(domain: &[u8], path: &SourcePath, bytes: &[u8]) -> ArtifactDigest {
    let entry = SourceEntry {
        path: path.clone(),
        kind: SourceEntryKind::File,
        mode: 0o644,
        bytes: Arc::from(bytes),
    };
    digest_entries(domain, &[entry])
}

fn append_bytes(material: &mut Vec<u8>, value: &[u8]) {
    material.extend_from_slice(&(value.len() as u64).to_le_bytes());
    material.extend_from_slice(value);
}

#[cfg(test)]
mod tests {
    use std::{
        cell::Cell,
        ffi::OsString,
        fs,
        os::{
            unix::ffi::OsStringExt,
            unix::fs::{PermissionsExt, symlink},
        },
        path::Path,
        process::Command,
    };

    use super::{
        HarborImportError, MAX_SOURCE_TREE_BYTES, SourceEntryKind, SourcePath,
        SourceTreeCaptureBudget, SourceTreeSnapshot, read_source_file_bounded,
    };

    #[test]
    fn capture_rejects_bytes_after_the_aggregate_source_budget_is_exhausted() {
        let temporary = tempfile::tempdir().unwrap();
        let file = temporary.path().join("task.json");
        fs::write(&file, b"x").unwrap();
        let mut source = fs::File::open(&file).unwrap();
        let mut bytes = Vec::new();
        let mut budget = SourceTreeCaptureBudget {
            entries: 0,
            bytes: MAX_SOURCE_TREE_BYTES,
        };

        let error =
            read_source_file_bounded(&mut source, Path::new("task.json"), &mut bytes, &mut budget)
                .expect_err("source capture must reject bytes above the aggregate cap");

        assert!(
            matches!(error, HarborImportError::InvalidPackage(message) if message.contains("byte limit"))
        );
        assert!(bytes.is_empty());
    }

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

    #[test]
    fn snapshot_rejects_file_swapped_to_an_outside_symlink_before_open() {
        let origin = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let source_file = origin.path().join("source.txt");
        let outside_file = outside.path().join("outside.txt");
        fs::write(&source_file, b"source bytes\n").unwrap();
        fs::write(&outside_file, b"outside bytes must never be captured\n").unwrap();
        let did_swap = Cell::new(false);

        let result = SourceTreeSnapshot::capture_with_before_open(origin.path(), |relative| {
            if relative == Path::new("source.txt") {
                fs::remove_file(&source_file).unwrap();
                symlink(&outside_file, &source_file).unwrap();
                did_swap.set(true);
            }
        });

        assert!(did_swap.get(), "the adversarial swap hook must run");
        assert!(matches!(result, Err(HarborImportError::InvalidPackage(_))));
    }
}
