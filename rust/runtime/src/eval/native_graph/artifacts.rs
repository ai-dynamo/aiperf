// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Capability-scoped, descriptor-validated NativeGraph artifact storage.

use std::{
    collections::{BTreeMap, BTreeSet},
    ffi::CString,
    fmt::{self, Display, Formatter},
    fs::{self, DirBuilder, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::eval::ArtifactDigest;

/// Per-episode artifact limits enforced before an adapter receives a capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArtifactQuota {
    /// Maximum retained frozen artifacts plus live upload reservations.
    pub max_artifacts: usize,
    /// Maximum bytes retained or reserved by this episode.
    pub max_total_bytes: u64,
    /// Maximum bytes in one upload reservation.
    pub max_artifact_bytes: u64,
    /// Maximum live download capabilities.
    pub max_download_handles: usize,
}

/// Opaque one-shot capability that permits one bounded upload.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ArtifactUploadHandle(String);

impl ArtifactUploadHandle {
    /// Returns the opaque wire token for the protocol ledger.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Opaque one-shot capability that permits one validated download.
#[derive(Clone, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ArtifactDownloadHandle(String);

impl ArtifactDownloadHandle {
    /// Returns the opaque wire token for the protocol ledger.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Immutable artifact identity retained by the Rust-owned store.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenArtifact {
    digest: ArtifactDigest,
    length: u64,
}

impl FrozenArtifact {
    /// Reconstitutes a descriptor only after a bounded wire decoder validates its fields.
    pub(crate) fn from_descriptor(digest: ArtifactDigest, length: u64) -> Self {
        Self { digest, length }
    }

    /// Returns the canonical BLAKE3 identity.
    pub fn digest(&self) -> &ArtifactDigest {
        &self.digest
    }

    /// Returns the exact verified byte length.
    pub fn length(&self) -> u64 {
        self.length
    }
}

/// Immutable content identity paired with the only Rust-issued child read capability.
#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenArtifactReference {
    download: ArtifactDownloadHandle,
    artifact: FrozenArtifact,
}

impl FrozenArtifactReference {
    /// Binds one opaque download capability to its verified frozen-content descriptor.
    pub fn new(download: ArtifactDownloadHandle, artifact: FrozenArtifact) -> Self {
        Self { download, artifact }
    }

    /// Borrows the store-issued one-shot download capability.
    pub fn download(&self) -> &ArtifactDownloadHandle {
        &self.download
    }

    /// Borrows the verified frozen-content descriptor.
    pub fn artifact(&self) -> &FrozenArtifact {
        &self.artifact
    }
}

/// Canonical frozen-artifact selection passed between isolated phases.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenArtifactManifest {
    artifacts: Vec<FrozenArtifact>,
}

impl FrozenArtifactManifest {
    /// Returns canonical immutable artifacts in digest order.
    pub fn artifacts(&self) -> &[FrozenArtifact] {
        &self.artifacts
    }

    /// Builds a canonical manifest from a bounded decoder's sorted unique descriptors.
    pub(crate) fn from_canonical_artifacts(artifacts: Vec<FrozenArtifact>) -> Option<Self> {
        if artifacts.windows(2).all(|pair| pair[0] < pair[1]) {
            Some(Self { artifacts })
        } else {
            None
        }
    }
}

/// Rust-owned artifact state for one isolated episode.
pub struct EpisodeArtifactStore {
    root: PathBuf,
    staging_root: PathBuf,
    frozen_root: PathBuf,
    staging_dir: File,
    frozen_dir: File,
    quota: ArtifactQuota,
    uploads: BTreeMap<String, UploadEntry>,
    downloads: BTreeMap<String, ArtifactDigest>,
    frozen: BTreeMap<ArtifactDigest, FrozenArtifact>,
    reserved_bytes: u64,
    is_closed: bool,
}

struct UploadEntry {
    file: File,
    declared_bytes: u64,
    state: UploadState,
}

enum UploadState {
    Staged { written_bytes: u64 },
    Poisoned,
    PublishCleanupPending { published_name: String },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    length: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

impl EpisodeArtifactStore {
    /// Creates a private store below a caller-provided trusted episode directory.
    pub fn new(parent: &Path, quota: ArtifactQuota) -> Result<Self, ArtifactError> {
        validate_quota(quota)?;
        let parent_metadata = fs::symlink_metadata(parent).map_err(io_error)?;
        if parent_metadata.file_type().is_symlink() || !parent_metadata.is_dir() {
            return Err(ArtifactError::Io(
                "artifact parent directory validation failed".to_owned(),
            ));
        }

        let root = parent.join(format!("native-graph-artifacts-{}", new_capability()));
        create_private_directory(&root)?;
        let staging_root = root.join("staging");
        let frozen_root = root.join("frozen");
        let directories = (|| {
            create_private_directory(&staging_root)?;
            create_private_directory(&frozen_root)?;
            Ok((
                open_private_directory(&staging_root)?,
                open_private_directory(&frozen_root)?,
            ))
        })();
        let (staging_dir, frozen_dir) = match directories {
            Ok(directories) => directories,
            Err(error) => {
                cleanup_empty_directories([&frozen_root, &staging_root, &root]);
                return Err(error);
            }
        };

        Ok(Self {
            root,
            staging_root,
            frozen_root,
            staging_dir,
            frozen_dir,
            quota,
            uploads: BTreeMap::new(),
            downloads: BTreeMap::new(),
            frozen: BTreeMap::new(),
            reserved_bytes: 0,
            is_closed: false,
        })
    }

    /// Returns the immutable episode quota used to admit frozen evidence.
    pub const fn quota(&self) -> ArtifactQuota {
        self.quota
    }

    /// Reserves a byte-exact upload and returns its only write capability.
    pub fn begin_upload(
        &mut self,
        declared_bytes: u64,
    ) -> Result<ArtifactUploadHandle, ArtifactError> {
        self.ensure_open()?;
        if declared_bytes > self.quota.max_artifact_bytes {
            return Err(ArtifactError::ArtifactBytesQuotaExceeded {
                limit: self.quota.max_artifact_bytes,
                requested: declared_bytes,
            });
        }
        let requested_total = self
            .reserved_bytes
            .checked_add(retained_bytes(&self.frozen)?)
            .and_then(|total| total.checked_add(declared_bytes))
            .ok_or(ArtifactError::UploadLengthOverflow)?;
        if requested_total > self.quota.max_total_bytes {
            return Err(ArtifactError::TotalBytesQuotaExceeded {
                limit: self.quota.max_total_bytes,
                requested: requested_total,
            });
        }
        if self
            .frozen
            .len()
            .checked_add(self.uploads.len())
            .ok_or(ArtifactError::UploadLengthOverflow)?
            >= self.quota.max_artifacts
        {
            return Err(ArtifactError::ArtifactCountQuotaExceeded {
                limit: self.quota.max_artifacts,
            });
        }

        let handle = ArtifactUploadHandle(new_capability());
        let file = open_new_regular_file_at(&self.staging_dir, handle.as_str())?;
        if let Err(error) = validate_open_regular(&file, 0) {
            let _ = unlink_file_at(&self.staging_dir, handle.as_str());
            return Err(ArtifactError::StagingValidation(error));
        }
        self.reserved_bytes = self
            .reserved_bytes
            .checked_add(declared_bytes)
            .ok_or(ArtifactError::UploadLengthOverflow)?;
        self.uploads.insert(
            handle.0.clone(),
            UploadEntry {
                file,
                declared_bytes,
                state: UploadState::Staged { written_bytes: 0 },
            },
        );
        Ok(handle)
    }

    /// Appends bytes from a child stream and poisons the capability after any partial failure.
    pub fn write_upload(
        &mut self,
        upload: &ArtifactUploadHandle,
        reader: &mut dyn Read,
    ) -> Result<(), ArtifactError> {
        self.ensure_open()?;
        let write_result = {
            let entry = self.known_upload_mut(upload)?;
            let written_bytes = match entry.state {
                UploadState::Staged { written_bytes } => written_bytes,
                UploadState::Poisoned => return Err(ArtifactError::UploadPoisoned),
                UploadState::PublishCleanupPending { .. } => {
                    return Err(ArtifactError::UploadCleanupPending);
                }
            };
            write_staging_bytes(&mut entry.file, entry.declared_bytes, written_bytes, reader)
        };
        match write_result {
            Ok(written_bytes) => {
                let entry = self.known_upload_mut(upload)?;
                entry.state = UploadState::Staged { written_bytes };
                Ok(())
            }
            Err(error) => {
                self.poison_upload(upload)?;
                Err(error)
            }
        }
    }

    /// Rehashes one complete staging descriptor and atomically links it into frozen storage.
    pub fn commit_upload(
        &mut self,
        upload: &ArtifactUploadHandle,
    ) -> Result<FrozenArtifact, ArtifactError> {
        self.ensure_open()?;
        let (declared_bytes, written_bytes) = self.staged_upload(upload)?;
        if written_bytes != declared_bytes {
            self.poison_upload(upload)?;
            return Err(ArtifactError::LengthMismatch {
                expected: declared_bytes,
                actual: written_bytes,
            });
        }
        let digest = {
            let entry = self.known_upload_mut(upload)?;
            rehash_staging_file(&mut entry.file, declared_bytes)
        };
        let digest = match digest {
            Ok(digest) => digest,
            Err(error) => {
                self.poison_upload(upload)?;
                return Err(error);
            }
        };
        let name = digest_filename(&digest)?;
        let artifact = FrozenArtifact {
            digest: digest.clone(),
            length: declared_bytes,
        };

        let link_result = {
            let entry = self.known_upload(upload)?;
            link_file_at(&entry.file, &self.frozen_dir, name)
        };
        let is_new_link = match link_result {
            Ok(()) => {
                if let Err(error) = self.frozen_dir.sync_all().map_err(io_error) {
                    return Err(self.rollback_published_link(upload, name, error));
                }
                let source_identity = {
                    let entry = self.known_upload(upload)?;
                    validate_open_regular(&entry.file, declared_bytes)
                        .map_err(ArtifactError::StagingValidation)?
                };
                if let Err(error) = validate_frozen_file_at(
                    &self.frozen_dir,
                    name,
                    &artifact,
                    Some(source_identity),
                ) {
                    return Err(self.rollback_published_link(upload, name, error));
                }
                true
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                validate_frozen_file_at(&self.frozen_dir, name, &artifact, None)?;
                false
            }
            Err(error) => return Err(io_error(error)),
        };

        if let Err(error) = unlink_file_at(&self.staging_dir, upload.as_str())
            .and_then(|()| self.staging_dir.sync_all().map_err(io_error))
        {
            if is_new_link {
                return Err(self.rollback_published_link(upload, name, error));
            }
            return Err(error);
        }
        self.uploads.remove(upload.as_str());
        self.reserved_bytes = self.reserved_bytes.saturating_sub(declared_bytes);
        self.frozen
            .entry(digest)
            .or_insert_with(|| artifact.clone());
        Ok(artifact)
    }

    /// Revokes an upload and removes only directly named store entries before releasing quota.
    pub fn abort_upload(&mut self, upload: &ArtifactUploadHandle) -> Result<(), ArtifactError> {
        self.ensure_open()?;
        let (declared_bytes, published_name) = {
            let entry = self.known_upload(upload)?;
            let published_name = match &entry.state {
                UploadState::PublishCleanupPending { published_name } => {
                    Some(published_name.clone())
                }
                UploadState::Staged { .. } | UploadState::Poisoned => None,
            };
            (entry.declared_bytes, published_name)
        };
        if let Some(published_name) = published_name {
            unlink_file_at(&self.frozen_dir, &published_name)?;
            self.frozen_dir.sync_all().map_err(io_error)?;
        }
        unlink_file_at(&self.staging_dir, upload.as_str())?;
        self.staging_dir.sync_all().map_err(io_error)?;
        self.uploads.remove(upload.as_str());
        self.reserved_bytes = self.reserved_bytes.saturating_sub(declared_bytes);
        Ok(())
    }

    /// Issues one opaque, one-shot read capability for a known frozen artifact.
    pub fn issue_download(
        &mut self,
        artifact: &FrozenArtifact,
    ) -> Result<ArtifactDownloadHandle, ArtifactError> {
        self.ensure_open()?;
        self.ensure_known_frozen(artifact)?;
        if self.downloads.len() >= self.quota.max_download_handles {
            return Err(ArtifactError::DownloadHandleLimit {
                limit: self.quota.max_download_handles,
            });
        }
        let handle = ArtifactDownloadHandle(new_capability());
        self.downloads
            .insert(handle.0.clone(), artifact.digest.clone());
        Ok(handle)
    }

    /// Checks whether one more immutable reference can be issued without mutating store state.
    ///
    /// Callers that retain this store's exclusive mutable borrow through `issue_reference` can
    /// use this preflight before publishing a new artifact, avoiding a frozen artifact whose
    /// reference cannot be granted.
    pub(crate) fn preflight_reference(&self) -> Result<(), ArtifactError> {
        self.ensure_open()?;
        if self.downloads.len() >= self.quota.max_download_handles {
            return Err(ArtifactError::DownloadHandleLimit {
                limit: self.quota.max_download_handles,
            });
        }
        Ok(())
    }

    /// Issues one immutable reference whose capability and descriptor came from this store.
    pub fn issue_reference(
        &mut self,
        artifact: &FrozenArtifact,
    ) -> Result<FrozenArtifactReference, ArtifactError> {
        let download = self.issue_download(artifact)?;
        Ok(FrozenArtifactReference::new(download, artifact.clone()))
    }

    /// Validates the complete frozen descriptor before delivering any bytes and consumes its grant.
    pub fn copy_download(
        &mut self,
        download: &ArtifactDownloadHandle,
        writer: &mut dyn Write,
    ) -> Result<(), ArtifactError> {
        self.ensure_open()?;
        let digest = self
            .downloads
            .remove(download.as_str())
            .ok_or(ArtifactError::UnknownDownloadHandle)?;
        let artifact = self
            .frozen
            .get(&digest)
            .cloned()
            .ok_or(ArtifactError::UnknownFrozenArtifact)?;
        let bytes = read_frozen_file_at(&self.frozen_dir, digest_filename(&digest)?, &artifact)?;
        writer.write_all(&bytes).map_err(io_error)
    }

    /// Revokes an unused download capability.
    pub fn revoke_download(
        &mut self,
        download: &ArtifactDownloadHandle,
    ) -> Result<(), ArtifactError> {
        self.ensure_open()?;
        self.downloads
            .remove(download.as_str())
            .map(|_| ())
            .ok_or(ArtifactError::UnknownDownloadHandle)
    }

    /// Revokes one exact immutable reference before its child capability can be replayed.
    pub fn revoke_reference(
        &mut self,
        reference: &FrozenArtifactReference,
    ) -> Result<(), ArtifactError> {
        self.ensure_open()?;
        let digest = self
            .downloads
            .get(reference.download().as_str())
            .ok_or(ArtifactError::UnknownDownloadHandle)?;
        if digest != reference.artifact().digest() {
            return Err(ArtifactError::ArtifactReferenceMismatch);
        }
        self.downloads.remove(reference.download().as_str());
        Ok(())
    }

    /// Validates one live capability against its exact frozen descriptor without consuming it.
    pub fn validate_reference(
        &self,
        reference: &FrozenArtifactReference,
    ) -> Result<(), ArtifactError> {
        self.ensure_open()?;
        let digest = self
            .downloads
            .get(reference.download().as_str())
            .ok_or(ArtifactError::UnknownDownloadHandle)?;
        if digest == reference.artifact().digest() {
            Ok(())
        } else {
            Err(ArtifactError::ArtifactReferenceMismatch)
        }
    }

    /// Reads complete verified frozen bytes without issuing a child capability.
    pub fn read_frozen(&self, artifact: &FrozenArtifact) -> Result<Vec<u8>, ArtifactError> {
        self.ensure_known_frozen(artifact)?;
        read_frozen_file_at(
            &self.frozen_dir,
            digest_filename(&artifact.digest)?,
            artifact,
        )
    }

    /// Freezes a canonical, duplicate-free selection of known artifacts.
    pub fn freeze_manifest(
        &self,
        artifacts: impl IntoIterator<Item = FrozenArtifact>,
    ) -> Result<FrozenArtifactManifest, ArtifactError> {
        let mut selected = BTreeSet::new();
        for artifact in artifacts {
            self.ensure_known_frozen(&artifact)?;
            if selected.contains(&artifact) {
                return Err(ArtifactError::DuplicateFrozenArtifact);
            }
            if selected.len() >= self.quota.max_artifacts {
                return Err(ArtifactError::ArtifactCountQuotaExceeded {
                    limit: self.quota.max_artifacts,
                });
            }
            selected.insert(artifact);
        }
        Ok(FrozenArtifactManifest {
            artifacts: selected.into_iter().collect(),
        })
    }

    /// Removes direct store-owned entries without traversing replacement directories.
    pub fn close(&mut self) -> Result<(), ArtifactError> {
        if self.is_closed {
            return Ok(());
        }
        self.downloads.clear();
        let uploads = self
            .uploads
            .keys()
            .cloned()
            .map(ArtifactUploadHandle)
            .collect::<Vec<_>>();
        for upload in uploads {
            self.abort_upload(&upload)?;
        }
        let frozen = self.frozen.values().cloned().collect::<Vec<_>>();
        for artifact in frozen {
            unlink_file_at(&self.frozen_dir, digest_filename(&artifact.digest)?)?;
        }
        self.frozen_dir.sync_all().map_err(io_error)?;
        self.frozen.clear();
        fs::remove_dir(&self.staging_root).map_err(io_error)?;
        fs::remove_dir(&self.frozen_root).map_err(io_error)?;
        fs::remove_dir(&self.root).map_err(io_error)?;
        self.is_closed = true;
        Ok(())
    }

    fn ensure_open(&self) -> Result<(), ArtifactError> {
        if self.is_closed {
            Err(ArtifactError::StoreClosed)
        } else {
            Ok(())
        }
    }

    fn known_upload(&self, upload: &ArtifactUploadHandle) -> Result<&UploadEntry, ArtifactError> {
        self.uploads
            .get(upload.as_str())
            .ok_or(ArtifactError::UnknownUploadHandle)
    }

    fn known_upload_mut(
        &mut self,
        upload: &ArtifactUploadHandle,
    ) -> Result<&mut UploadEntry, ArtifactError> {
        self.uploads
            .get_mut(upload.as_str())
            .ok_or(ArtifactError::UnknownUploadHandle)
    }

    fn staged_upload(&self, upload: &ArtifactUploadHandle) -> Result<(u64, u64), ArtifactError> {
        let entry = self.known_upload(upload)?;
        match entry.state {
            UploadState::Staged { written_bytes } => Ok((entry.declared_bytes, written_bytes)),
            UploadState::Poisoned => Err(ArtifactError::UploadPoisoned),
            UploadState::PublishCleanupPending { .. } => Err(ArtifactError::UploadCleanupPending),
        }
    }

    fn poison_upload(&mut self, upload: &ArtifactUploadHandle) -> Result<(), ArtifactError> {
        self.known_upload_mut(upload)?.state = UploadState::Poisoned;
        Ok(())
    }

    fn rollback_published_link(
        &mut self,
        upload: &ArtifactUploadHandle,
        published_name: &str,
        publish_error: ArtifactError,
    ) -> ArtifactError {
        match unlink_file_at(&self.frozen_dir, published_name)
            .and_then(|()| self.frozen_dir.sync_all().map_err(io_error))
        {
            Ok(()) => publish_error,
            Err(rollback_error) => {
                if let Ok(entry) = self.known_upload_mut(upload) {
                    entry.state = UploadState::PublishCleanupPending {
                        published_name: published_name.to_owned(),
                    };
                }
                ArtifactError::PublishRollback {
                    publish: publish_error.to_string(),
                    rollback: rollback_error.to_string(),
                }
            }
        }
    }

    fn ensure_known_frozen(&self, artifact: &FrozenArtifact) -> Result<(), ArtifactError> {
        if self.frozen.get(&artifact.digest) == Some(artifact) {
            Ok(())
        } else {
            Err(ArtifactError::UnknownFrozenArtifact)
        }
    }
}

impl Drop for EpisodeArtifactStore {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(error = %error, component = "native_graph_artifacts", "artifact store cleanup failed during drop");
        }
    }
}

/// NativeGraph artifact-store failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    /// The configured resource quota was internally inconsistent.
    InvalidQuota,
    /// A requested upload was larger than the per-artifact maximum.
    ArtifactBytesQuotaExceeded { limit: u64, requested: u64 },
    /// An upload reservation would exceed the episode-wide byte maximum.
    TotalBytesQuotaExceeded { limit: u64, requested: u64 },
    /// An upload reservation or frozen manifest selection would exceed the entry maximum.
    ArtifactCountQuotaExceeded { limit: usize },
    /// A read capability would exceed the active-grant maximum.
    DownloadHandleLimit { limit: usize },
    /// The supplied upload capability is unknown or already revoked.
    UnknownUploadHandle,
    /// The supplied download capability is unknown or already revoked.
    UnknownDownloadHandle,
    /// The supplied frozen artifact was not created by this store.
    UnknownFrozenArtifact,
    /// A capability did not name the descriptor it was issued for.
    ArtifactReferenceMismatch,
    /// A commit observed an incomplete staged stream.
    LengthMismatch { expected: u64, actual: u64 },
    /// A streamed write exceeded its granted length.
    UploadLengthExceeded { expected: u64, actual: u64 },
    /// A byte accounting operation overflowed.
    UploadLengthOverflow,
    /// A failed source, destination, or descriptor check poisoned the upload.
    UploadPoisoned,
    /// Publication needs cleanup before the upload can be reused or aborted.
    UploadCleanupPending,
    /// A frozen manifest named one artifact twice.
    DuplicateFrozenArtifact,
    /// A closed store received another operation.
    StoreClosed,
    /// A staging or frozen descriptor failed identity, length, or digest validation.
    StagingValidation(String),
    /// Publication failed and the new frozen link could not be rolled back.
    PublishRollback { publish: String, rollback: String },
    /// A filesystem operation failed without disclosing a store path.
    Io(String),
}

impl Display for ArtifactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidQuota => formatter.write_str("artifact quota must be nonzero and bounded"),
            Self::ArtifactBytesQuotaExceeded { limit, requested } => {
                write!(formatter, "artifact bytes {requested} exceed limit {limit}")
            }
            Self::TotalBytesQuotaExceeded { limit, requested } => {
                write!(
                    formatter,
                    "artifact bytes {requested} exceed total limit {limit}"
                )
            }
            Self::ArtifactCountQuotaExceeded { limit } => {
                write!(formatter, "artifact count exceeds limit {limit}")
            }
            Self::DownloadHandleLimit { limit } => {
                write!(formatter, "download capability count exceeds limit {limit}")
            }
            Self::UnknownUploadHandle => formatter.write_str("unknown artifact upload handle"),
            Self::UnknownDownloadHandle => formatter.write_str("unknown artifact download handle"),
            Self::UnknownFrozenArtifact => formatter.write_str("unknown frozen artifact"),
            Self::ArtifactReferenceMismatch => {
                formatter.write_str("artifact capability does not match frozen descriptor")
            }
            Self::LengthMismatch { expected, actual } => {
                write!(
                    formatter,
                    "artifact length {actual} does not equal {expected}"
                )
            }
            Self::UploadLengthExceeded { expected, actual } => {
                write!(formatter, "artifact length {actual} exceeds {expected}")
            }
            Self::UploadLengthOverflow => {
                formatter.write_str("artifact length accounting overflow")
            }
            Self::UploadPoisoned => formatter.write_str("artifact upload was poisoned"),
            Self::UploadCleanupPending => formatter.write_str("artifact upload cleanup is pending"),
            Self::DuplicateFrozenArtifact => formatter.write_str("frozen artifact is duplicated"),
            Self::StoreClosed => formatter.write_str("artifact store is closed"),
            Self::StagingValidation(error) => {
                write!(formatter, "artifact descriptor validation failed: {error}")
            }
            Self::PublishRollback { publish, rollback } => {
                write!(
                    formatter,
                    "artifact publish failed ({publish}) and rollback failed ({rollback})"
                )
            }
            Self::Io(error) => write!(formatter, "artifact store I/O failed: {error}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

fn validate_quota(quota: ArtifactQuota) -> Result<(), ArtifactError> {
    if quota.max_artifacts == 0
        || quota.max_total_bytes == 0
        || quota.max_artifact_bytes == 0
        || quota.max_artifact_bytes > quota.max_total_bytes
        || quota.max_download_handles == 0
    {
        return Err(ArtifactError::InvalidQuota);
    }
    Ok(())
}

fn retained_bytes(frozen: &BTreeMap<ArtifactDigest, FrozenArtifact>) -> Result<u64, ArtifactError> {
    frozen.values().try_fold(0_u64, |total, artifact| {
        total
            .checked_add(artifact.length)
            .ok_or(ArtifactError::UploadLengthOverflow)
    })
}

fn new_capability() -> String {
    Uuid::new_v4().simple().to_string()
}

fn create_private_directory(path: &Path) -> Result<(), ArtifactError> {
    DirBuilder::new()
        .mode(0o700)
        .create(path)
        .map_err(io_error)?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(io_error)?;
    let metadata = fs::symlink_metadata(path).map_err(io_error)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ArtifactError::Io(
            "artifact directory validation failed".to_owned(),
        ));
    }
    Ok(())
}

fn cleanup_empty_directories(paths: impl IntoIterator<Item = impl AsRef<Path>>) {
    for path in paths {
        let _ = fs::remove_dir(path);
    }
}

fn open_private_directory(path: &Path) -> Result<File, ArtifactError> {
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(path)
        .map_err(io_error)
}

fn name_as_c_string(name: &str) -> Result<CString, ArtifactError> {
    if name.is_empty() || name.contains('/') {
        return Err(ArtifactError::StagingValidation(
            "artifact entry name is invalid".to_owned(),
        ));
    }
    CString::new(name).map_err(|_| {
        ArtifactError::StagingValidation("artifact entry name contains NUL".to_owned())
    })
}

fn open_new_regular_file_at(directory: &File, name: &str) -> Result<File, ArtifactError> {
    let name = name_as_c_string(name)?;
    // The directory descriptor is retained by the store; the returned descriptor owns this file.
    let descriptor = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_CREAT | libc::O_EXCL | libc::O_RDWR | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        )
    };
    if descriptor < 0 {
        return Err(io_error(io::Error::last_os_error()));
    }
    // `openat` returned a unique owned descriptor on the successful branch above.
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

fn open_existing_regular_file_at(directory: &File, name: &str) -> Result<File, ArtifactError> {
    let name = name_as_c_string(name)?;
    // The directory descriptor cannot be redirected by replacement of its path.
    let descriptor = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if descriptor < 0 {
        return Err(io_error(io::Error::last_os_error()));
    }
    // `openat` returned a unique owned descriptor on the successful branch above.
    Ok(unsafe { File::from_raw_fd(descriptor) })
}

fn link_file_at(source: &File, target_directory: &File, target_name: &str) -> io::Result<()> {
    let target_name = name_as_c_string(target_name).map_err(artifact_error_to_io)?;
    let source_name = CString::new(Vec::<u8>::new()).map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("empty source descriptor name is invalid: {error}"),
        )
    })?;
    // Linux `AT_EMPTY_PATH` links the retained source descriptor without resolving its old name.
    let result = unsafe {
        libc::linkat(
            source.as_raw_fd(),
            source_name.as_ptr(),
            target_directory.as_raw_fd(),
            target_name.as_ptr(),
            libc::AT_EMPTY_PATH,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn unlink_file_at(directory: &File, name: &str) -> Result<(), ArtifactError> {
    let name = name_as_c_string(name)?;
    // `unlinkat` names one direct child of the retained private directory and never recurses.
    let result = unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::NotFound {
        Ok(())
    } else {
        Err(io_error(error))
    }
}

fn write_staging_bytes(
    file: &mut File,
    declared_bytes: u64,
    written_bytes: u64,
    reader: &mut dyn Read,
) -> Result<u64, ArtifactError> {
    validate_open_regular(file, written_bytes).map_err(ArtifactError::StagingValidation)?;
    file.seek(SeekFrom::End(0)).map_err(io_error)?;
    let mut total = written_bytes;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            return Ok(total);
        }
        total = total
            .checked_add(u64::try_from(count).map_err(|_| ArtifactError::UploadLengthOverflow)?)
            .ok_or(ArtifactError::UploadLengthOverflow)?;
        if total > declared_bytes {
            return Err(ArtifactError::UploadLengthExceeded {
                expected: declared_bytes,
                actual: total,
            });
        }
        file.write_all(&buffer[..count]).map_err(io_error)?;
    }
}

fn rehash_staging_file(
    file: &mut File,
    expected_length: u64,
) -> Result<ArtifactDigest, ArtifactError> {
    let before =
        validate_open_regular(file, expected_length).map_err(ArtifactError::StagingValidation)?;
    file.sync_all().map_err(io_error)?;
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    let mut hasher = blake3::Hasher::new();
    let mut length = 0_u64;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(u64::try_from(count).map_err(|_| ArtifactError::UploadLengthOverflow)?)
            .ok_or(ArtifactError::UploadLengthOverflow)?;
        hasher.update(&buffer[..count]);
    }
    let after =
        validate_open_regular(file, expected_length).map_err(ArtifactError::StagingValidation)?;
    if before != after || length != expected_length {
        return Err(ArtifactError::StagingValidation(
            "staging descriptor changed during validation".to_owned(),
        ));
    }
    ArtifactDigest::parse(format!("blake3:{}", hasher.finalize().to_hex()))
        .map_err(|error| ArtifactError::StagingValidation(error.to_string()))
}

fn validate_frozen_file_at(
    directory: &File,
    name: &str,
    artifact: &FrozenArtifact,
    expected_identity: Option<FileIdentity>,
) -> Result<(), ArtifactError> {
    let mut file = open_existing_regular_file_at(directory, name)?;
    let identity =
        validate_open_regular(&file, artifact.length).map_err(ArtifactError::StagingValidation)?;
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(ArtifactError::StagingValidation(
            "published descriptor does not match the retained upload descriptor".to_owned(),
        ));
    }
    read_validated_bytes(&mut file, artifact).map(|_| ())
}

fn read_frozen_file_at(
    directory: &File,
    name: &str,
    artifact: &FrozenArtifact,
) -> Result<Vec<u8>, ArtifactError> {
    let mut file = open_existing_regular_file_at(directory, name)?;
    read_validated_bytes(&mut file, artifact)
}

fn read_validated_bytes(
    file: &mut File,
    artifact: &FrozenArtifact,
) -> Result<Vec<u8>, ArtifactError> {
    let before =
        validate_open_regular(file, artifact.length).map_err(ArtifactError::StagingValidation)?;
    let capacity = usize::try_from(artifact.length).map_err(|_| {
        ArtifactError::StagingValidation("artifact length cannot fit in memory".to_owned())
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    let mut hasher = blake3::Hasher::new();
    let mut length = 0_u64;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        length = length
            .checked_add(u64::try_from(count).map_err(|_| ArtifactError::UploadLengthOverflow)?)
            .ok_or(ArtifactError::UploadLengthOverflow)?;
        bytes.extend_from_slice(&buffer[..count]);
        hasher.update(&buffer[..count]);
    }
    let after =
        validate_open_regular(file, artifact.length).map_err(ArtifactError::StagingValidation)?;
    if before != after || length != artifact.length {
        return Err(ArtifactError::StagingValidation(
            "frozen descriptor changed during validation".to_owned(),
        ));
    }
    let actual = ArtifactDigest::parse(format!("blake3:{}", hasher.finalize().to_hex()))
        .map_err(|error| ArtifactError::StagingValidation(error.to_string()))?;
    if actual != artifact.digest {
        return Err(ArtifactError::StagingValidation(
            "frozen descriptor digest mismatch".to_owned(),
        ));
    }
    Ok(bytes)
}

fn validate_open_regular(file: &File, expected_length: u64) -> Result<FileIdentity, String> {
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.file_type().is_file() || metadata.nlink() == 0 || metadata.len() != expected_length
    {
        return Err("artifact descriptor is not the expected regular file".to_owned());
    }
    Ok(FileIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        length: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    })
}

fn digest_filename(digest: &ArtifactDigest) -> Result<&str, ArtifactError> {
    digest.as_str().strip_prefix("blake3:").ok_or_else(|| {
        ArtifactError::StagingValidation("artifact digest lacks blake3 prefix".to_owned())
    })
}

fn artifact_error_to_io(error: ArtifactError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error.to_string())
}

fn io_error(error: io::Error) -> ArtifactError {
    ArtifactError::Io(error.to_string())
}
