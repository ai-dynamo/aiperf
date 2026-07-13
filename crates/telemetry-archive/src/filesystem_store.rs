// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authoritative local-filesystem archive object-store adapter.
//!
//! The adapter implements the same immutable-create and linearizable-head-CAS
//! contract as remote providers. A store-wide advisory lock serializes named
//! head decisions across cooperating processes; exact-body versions make CAS
//! comparisons stable across process restarts. Immutable objects still use an
//! atomic link-if-absent transition so an existing unequal object is never
//! replaced.

use std::ffi::CString;
use std::fmt::{self, Debug, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use bytes::Bytes;

use crate::{
    ArchiveObjectStore, ArchiveStoreCapabilities, ArchiveStoreError, CreateReceipt, Digest,
    HeadUpdateError, NamedObjectVisibility, ObjectVersionKind, StableObjectVersion, VersionedHead,
    archive_object_digest,
};

const LOCK_NAME: &str = ".aiperf-archive-store.lock";

/// Local directory implementing the authoritative archive object-store seam.
pub struct FileArchiveObjectStore {
    root: PathBuf,
    next_temporary: AtomicU64,
}

impl FileArchiveObjectStore {
    /// Create or open an absolute, non-symlinked store root.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, ArchiveStoreError> {
        let path = path.as_ref();
        if !path.is_absolute() {
            return Err(ArchiveStoreError::Transport(
                "filesystem archive store root must be absolute".to_owned(),
            ));
        }
        reject_symlink_components(path)?;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(path)
            .map_err(|error| file_error("create store root", error))?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| file_error("set store permissions", error))?;
        reject_symlink_components(path)?;
        let root =
            fs::canonicalize(path).map_err(|error| file_error("canonicalize store root", error))?;
        let store = Self {
            root,
            next_temporary: AtomicU64::new(0),
        };
        let _guard = store.lock()?;
        sync_directory(&store.root)?;
        Ok(store)
    }

    /// Canonical credential-free local store root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    fn lock(&self) -> Result<FileStoreLock, ArchiveStoreError> {
        let path = self.root.join(LOCK_NAME);
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(&path)
            .map_err(|error| file_error("open store lock", error))?;
        let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if result != 0 {
            return Err(file_error("lock archive store", io::Error::last_os_error()));
        }
        Ok(FileStoreLock { file })
    }

    fn object_path(&self, key: &str) -> Result<PathBuf, ArchiveStoreError> {
        validate_key(key)?;
        Ok(self.root.join(key))
    }

    fn ensure_parent(&self, path: &Path) -> Result<(), ArchiveStoreError> {
        let parent = path
            .parent()
            .ok_or_else(|| ArchiveStoreError::InvalidKey("object key has no parent".to_owned()))?;
        reject_symlink_components(parent)?;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(parent)
            .map_err(|error| file_error("create object parent", error))?;
        reject_symlink_components(parent)?;
        Ok(())
    }

    fn temporary_path(&self, final_path: &Path) -> Result<PathBuf, ArchiveStoreError> {
        let parent = final_path
            .parent()
            .ok_or_else(|| ArchiveStoreError::InvalidKey("object key has no parent".to_owned()))?;
        let name = final_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| ArchiveStoreError::InvalidKey("non-UTF-8 object key".to_owned()))?;
        let sequence = self.next_temporary.fetch_add(1, Ordering::Relaxed);
        Ok(parent.join(format!(".{name}.tmp-{}-{sequence}", std::process::id())))
    }

    fn write_temporary(
        &self,
        final_path: &Path,
        body: &[u8],
    ) -> Result<PathBuf, ArchiveStoreError> {
        self.ensure_parent(final_path)?;
        let temporary = self.temporary_path(final_path)?;
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&temporary)
            .map_err(|error| file_error("create temporary object", error))?;
        if let Err(error) = file
            .write_all(body)
            .and_then(|()| file.flush())
            .and_then(|()| file.sync_all())
        {
            let _ = fs::remove_file(&temporary);
            return Err(file_error("durably write temporary object", error));
        }
        Ok(temporary)
    }

    fn read_object(&self, path: &Path) -> Result<Bytes, ArchiveStoreError> {
        reject_symlink_components(path)?;
        let metadata = fs::symlink_metadata(path).map_err(|error| {
            if error.kind() == io::ErrorKind::NotFound {
                ArchiveStoreError::NotFound(relative_display(&self.root, path))
            } else {
                file_error("stat archive object", error)
            }
        })?;
        if !metadata.file_type().is_file() || metadata.nlink() != 1 {
            return Err(ArchiveStoreError::Transport(
                "archive object is not one regular single-link file".to_owned(),
            ));
        }
        fs::read(path)
            .map(Bytes::from)
            .map_err(|error| file_error("read archive object", error))
    }

    fn version(digest: Digest) -> StableObjectVersion {
        StableObjectVersion::new(
            "filesystem-archive-store-v1",
            ObjectVersionKind::Etag,
            digest.as_bytes().to_vec(),
        )
        .expect("a digest is a nonempty stable version")
    }

    fn verified_head(&self, path: &Path) -> Result<VersionedHead, ArchiveStoreError> {
        let body = self.read_object(path)?;
        let digest = archive_object_digest(&body);
        Ok(VersionedHead {
            body,
            digest,
            version: Self::version(digest),
        })
    }

    fn verify_supplied(key: &str, body: &[u8], expected: Digest) -> Result<(), ArchiveStoreError> {
        let actual = archive_object_digest(body);
        if actual != expected {
            return Err(ArchiveStoreError::DigestMismatch {
                key: key.to_owned(),
                expected,
                actual,
            });
        }
        Ok(())
    }

    fn put_immutable_locked(
        &self,
        key: &str,
        body: Bytes,
        digest: Digest,
    ) -> Result<CreateReceipt, ArchiveStoreError> {
        Self::verify_supplied(key, &body, digest)?;
        let final_path = self.object_path(key)?;
        if final_path.exists() {
            let existing = self.read_object(&final_path)?;
            if existing != body {
                return Err(ArchiveStoreError::AlreadyExistsDifferent(key.to_owned()));
            }
            Self::verify_supplied(key, &existing, digest)?;
            return Ok(CreateReceipt {
                created: false,
                version: Self::version(digest),
            });
        }

        let temporary = self.write_temporary(&final_path, &body)?;
        let installed = match rename_noreplace(&temporary, &final_path) {
            Ok(installed) => installed,
            Err(error) => {
                let _ = fs::remove_file(&temporary);
                return Err(error);
            }
        };
        if !installed {
            fs::remove_file(&temporary)
                .map_err(|error| file_error("remove temporary object", error))?;
        }
        sync_directory(final_path.parent().expect("object path has a parent"))?;
        if !installed {
            let existing = self.read_object(&final_path)?;
            if existing != body {
                return Err(ArchiveStoreError::AlreadyExistsDifferent(key.to_owned()));
            }
        }
        Ok(CreateReceipt {
            created: installed,
            version: Self::version(digest),
        })
    }

    fn replace_head_locked(
        &self,
        path: &Path,
        replacement: &[u8],
    ) -> Result<(), ArchiveStoreError> {
        let temporary = self.write_temporary(path, replacement)?;
        fs::rename(&temporary, path).map_err(|error| file_error("replace archive head", error))?;
        sync_directory(path.parent().expect("head path has a parent"))
    }
}

impl Debug for FileArchiveObjectStore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FileArchiveObjectStore")
            .field("root", &self.root)
            .finish()
    }
}

#[async_trait]
impl ArchiveObjectStore for FileArchiveObjectStore {
    fn capabilities(&self) -> ArchiveStoreCapabilities {
        ArchiveStoreCapabilities {
            immutable_create_if_absent: true,
            exact_byte_verification: true,
            linearizable_head_cas: true,
            named_object_visibility: NamedObjectVisibility::Immediate,
        }
    }

    async fn put_if_absent(
        &self,
        key: &str,
        body: Bytes,
        digest: Digest,
    ) -> Result<CreateReceipt, ArchiveStoreError> {
        let _guard = self.lock()?;
        self.put_immutable_locked(key, body, digest)
    }

    async fn get_verified(&self, key: &str, expected: Digest) -> Result<Bytes, ArchiveStoreError> {
        let _guard = self.lock()?;
        let path = self.object_path(key)?;
        let body = self.read_object(&path)?;
        Self::verify_supplied(key, &body, expected)?;
        Ok(body)
    }

    async fn read_head(&self, key: &str) -> Result<Option<VersionedHead>, ArchiveStoreError> {
        let _guard = self.lock()?;
        let path = self.object_path(key)?;
        if !path.exists() {
            return Ok(None);
        }
        self.verified_head(&path).map(Some)
    }

    async fn create_head_if_absent(
        &self,
        key: &str,
        replacement: Bytes,
        digest: Digest,
    ) -> Result<CreateReceipt, HeadUpdateError> {
        let _guard = self.lock().map_err(HeadUpdateError::Store)?;
        Self::verify_supplied(key, &replacement, digest).map_err(HeadUpdateError::Store)?;
        let path = self.object_path(key).map_err(HeadUpdateError::Store)?;
        if path.exists() {
            return Err(HeadUpdateError::Conflict {
                current: Some(self.verified_head(&path).map_err(HeadUpdateError::Store)?),
            });
        }
        self.put_immutable_locked(key, replacement, digest)
            .map_err(HeadUpdateError::Store)
    }

    async fn compare_and_swap_head(
        &self,
        key: &str,
        expected_version: &StableObjectVersion,
        replacement: Bytes,
        digest: Digest,
    ) -> Result<StableObjectVersion, HeadUpdateError> {
        let _guard = self.lock().map_err(HeadUpdateError::Store)?;
        Self::verify_supplied(key, &replacement, digest).map_err(HeadUpdateError::Store)?;
        let path = self.object_path(key).map_err(HeadUpdateError::Store)?;
        if !path.exists() {
            return Err(HeadUpdateError::Conflict { current: None });
        }
        let current = self.verified_head(&path).map_err(HeadUpdateError::Store)?;
        if &current.version != expected_version {
            return Err(HeadUpdateError::Conflict {
                current: Some(current),
            });
        }
        self.replace_head_locked(&path, &replacement)
            .map_err(HeadUpdateError::Store)?;
        Ok(Self::version(digest))
    }
}

struct FileStoreLock {
    file: File,
}

impl Drop for FileStoreLock {
    fn drop(&mut self) {
        let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
    }
}

fn validate_key(key: &str) -> Result<(), ArchiveStoreError> {
    if key.is_empty()
        || key.starts_with('/')
        || key.ends_with('/')
        || key.contains('\0')
        || key
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        return Err(ArchiveStoreError::InvalidKey(key.to_owned()));
    }
    for component in Path::new(key).components() {
        match component {
            Component::Normal(value) if !value.is_empty() => {}
            _ => return Err(ArchiveStoreError::InvalidKey(key.to_owned())),
        }
    }
    Ok(())
}

fn reject_symlink_components(path: &Path) -> Result<(), ArchiveStoreError> {
    let mut current = PathBuf::new();
    for component in path.components() {
        current.push(component.as_os_str());
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(ArchiveStoreError::Transport(
                    "filesystem archive store path contains a symbolic link".to_owned(),
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(file_error("inspect store path", error)),
        }
    }
    Ok(())
}

fn rename_noreplace(source: &Path, destination: &Path) -> Result<bool, ArchiveStoreError> {
    let source = CString::new(source.as_os_str().as_bytes()).map_err(|_| {
        ArchiveStoreError::Transport("temporary object path contains a NUL byte".to_owned())
    })?;
    let destination = CString::new(destination.as_os_str().as_bytes()).map_err(|_| {
        ArchiveStoreError::Transport("final object path contains a NUL byte".to_owned())
    })?;
    // SAFETY: both C strings remain live for the syscall and AT_FDCWD selects
    // their already validated absolute paths.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            destination.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::AlreadyExists {
        return Ok(false);
    }
    if matches!(error.raw_os_error(), Some(libc::ENOSYS | libc::EINVAL)) {
        return Err(ArchiveStoreError::MissingCapability(
            "atomic_rename_noreplace",
        ));
    }
    Err(file_error("install immutable object", error))
}

fn sync_directory(path: &Path) -> Result<(), ArchiveStoreError> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| file_error("fsync store directory", error))
}

fn relative_display(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn file_error(operation: &str, error: io::Error) -> ArchiveStoreError {
    ArchiveStoreError::Transport(format!("{operation}: {error}"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tempfile::TempDir;

    use super::*;

    fn store() -> (TempDir, Arc<FileArchiveObjectStore>) {
        let temporary = TempDir::new().unwrap();
        let store =
            Arc::new(FileArchiveObjectStore::open(temporary.path().join("remote")).unwrap());
        (temporary, store)
    }

    #[tokio::test]
    async fn immutable_create_is_exact_and_restart_stable() {
        let (temporary, store) = store();
        let body = Bytes::from_static(b"immutable");
        let digest = archive_object_digest(&body);
        let first = store
            .put_if_absent("objects/value", body.clone(), digest)
            .await
            .unwrap();
        assert!(first.created);
        let second = store
            .put_if_absent("objects/value", body.clone(), digest)
            .await
            .unwrap();
        assert!(!second.created);
        assert_eq!(first.version, second.version);

        drop(store);
        let reopened = FileArchiveObjectStore::open(temporary.path().join("remote")).unwrap();
        assert_eq!(
            reopened
                .get_verified("objects/value", digest)
                .await
                .unwrap(),
            body
        );
    }

    #[tokio::test]
    async fn unequal_immutable_reuse_never_replaces_existing_bytes() {
        let (_temporary, store) = store();
        let first = Bytes::from_static(b"first");
        let first_digest = archive_object_digest(&first);
        store
            .put_if_absent("objects/value", first.clone(), first_digest)
            .await
            .unwrap();
        let second = Bytes::from_static(b"second");
        let error = store
            .put_if_absent(
                "objects/value",
                second.clone(),
                archive_object_digest(&second),
            )
            .await
            .unwrap_err();
        assert_eq!(
            error,
            ArchiveStoreError::AlreadyExistsDifferent("objects/value".to_owned())
        );
        assert_eq!(
            store
                .get_verified("objects/value", first_digest)
                .await
                .unwrap(),
            first
        );
    }

    #[tokio::test]
    async fn head_compare_and_swap_rejects_stale_versions() {
        let (_temporary, store) = store();
        let first = Bytes::from_static(b"head-one");
        let first_digest = archive_object_digest(&first);
        let created = store
            .create_head_if_absent("REMOTE-LATEST", first, first_digest)
            .await
            .unwrap();
        let second = Bytes::from_static(b"head-two");
        let second_digest = archive_object_digest(&second);
        let second_version = store
            .compare_and_swap_head("REMOTE-LATEST", &created.version, second, second_digest)
            .await
            .unwrap();
        let stale = store
            .compare_and_swap_head(
                "REMOTE-LATEST",
                &created.version,
                Bytes::from_static(b"head-three"),
                archive_object_digest(b"head-three"),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            stale,
            HeadUpdateError::Conflict { current: Some(_) }
        ));
        assert_eq!(
            store
                .read_head("REMOTE-LATEST")
                .await
                .unwrap()
                .unwrap()
                .version,
            second_version
        );
    }

    #[test]
    fn relative_roots_and_traversal_keys_fail_closed() {
        assert!(FileArchiveObjectStore::open("relative").is_err());
        let (_temporary, store) = store();
        assert!(store.object_path("../escape").is_err());
        assert!(store.object_path("a//b").is_err());
        assert!(store.object_path("/absolute").is_err());
    }
}
