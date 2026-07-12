// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Quiescence-gated, FD-relative evaluator artifact verification and sealing.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CStr, CString, OsStr};
use std::fmt::{self, Display};
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::canonical::{CanonicalJson, CanonicalJsonLimits, is_sha256, sha256_hex};
use crate::isolation::IsolationQuiescenceProof;
use crate::provider_protocol::{
    ArtifactVisibility, EvaluationArtifactManifestEntry, EvaluationFinishCandidate,
};

static SEAL_SEQUENCE: AtomicU64 = AtomicU64::new(1);

/// Hard artifact count/byte limits enforced independently of provider claims.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSealLimits {
    /// Maximum declared regular files.
    pub max_artifacts: usize,
    /// Maximum aggregate raw staging bytes.
    pub max_total_bytes: u64,
    /// Maximum one public JSON projection bytes.
    pub max_public_projection_bytes: usize,
}

impl Default for ArtifactSealLimits {
    fn default() -> Self {
        Self {
            max_artifacts: 16_384,
            max_total_bytes: 64 * 1024 * 1024 * 1024,
            max_public_projection_bytes: 16 * 1024 * 1024,
        }
    }
}

/// Factory-owned validator for one reviewed public artifact projection schema.
pub trait PublicArtifactProjectionValidator: Send + Sync {
    /// Exact registered schema fingerprint.
    fn schema_sha256(&self) -> &str;

    /// Validate a strict canonical JSON projection.
    fn validate(&self, value: &CanonicalJson) -> Result<(), ArtifactSealError>;
}

/// One factory-owned public path/media/schema rule.
#[derive(Clone)]
pub struct PublicArtifactProjectionRule {
    media_type: String,
    max_bytes: usize,
    validator: Arc<dyn PublicArtifactProjectionValidator>,
}

impl fmt::Debug for PublicArtifactProjectionRule {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PublicArtifactProjectionRule")
            .field("media_type", &self.media_type)
            .field("max_bytes", &self.max_bytes)
            .field("schema_sha256", &self.validator.schema_sha256())
            .finish()
    }
}

/// Deterministic factory-owned artifact projection registry.
#[derive(Debug, Clone, Default)]
pub struct ArtifactProjectionPolicy {
    rules: BTreeMap<String, PublicArtifactProjectionRule>,
}

impl ArtifactProjectionPolicy {
    /// Policy that keeps every provider artifact restricted.
    pub fn restricted_only() -> Self {
        Self::default()
    }

    /// Register one exact path/media/schema public projection rule.
    pub fn register(
        &mut self,
        path: impl Into<String>,
        media_type: impl Into<String>,
        max_bytes: usize,
        validator: Arc<dyn PublicArtifactProjectionValidator>,
    ) -> Result<&mut Self, ArtifactSealError> {
        let path = path.into();
        validate_relative_path(&path)?;
        let media_type = media_type.into();
        if media_type.trim().is_empty()
            || max_bytes == 0
            || !is_sha256(validator.schema_sha256())
            || self.rules.contains_key(&path)
        {
            return Err(ArtifactSealError::Policy(
                "public artifact projection rule was empty, mutable, or duplicated".to_string(),
            ));
        }
        self.rules.insert(
            path,
            PublicArtifactProjectionRule {
                media_type,
                max_bytes,
                validator,
            },
        );
        Ok(self)
    }

    fn rule(&self, path: &str) -> Option<&PublicArtifactProjectionRule> {
        self.rules.get(path)
    }
}

/// Rust-verified immutable sealed artifact entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationArtifact {
    /// Stable provider artifact identity.
    pub artifact_id: crate::provider_protocol::EvaluationArtifactId,
    /// Relative immutable path.
    pub path: String,
    /// Verified media type.
    pub media_type: String,
    /// Rust-authorized final visibility.
    pub visibility: ArtifactVisibility,
    /// Final promoted byte length.
    pub size_bytes: u64,
    /// Rust-computed final raw byte digest.
    pub artifact_content_sha256: String,
    /// Public projection schema fingerprint when public.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub public_projection_schema_sha256: Option<String>,
}

/// Immutable Rust-owned artifact tree and canonical bundle digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SealedEvaluationArtifacts {
    /// Promoted immutable root.
    pub root: PathBuf,
    /// Verified entries in manifest order.
    pub entries: Vec<SealedEvaluationArtifact>,
    /// Rust-computed digest of the canonical provider bundle bytes.
    pub provider_bundle_sha256: String,
    /// Quiescence proof bound to this seal.
    pub quiescence_proof_sha256: String,
}

/// Artifact sealer using no-follow FD-relative traversal on Linux/Unix.
pub struct EvaluationArtifactSealer {
    limits: ArtifactSealLimits,
    policy: ArtifactProjectionPolicy,
}

impl EvaluationArtifactSealer {
    /// Construct a sealer from host limits and factory-owned projection rules.
    pub fn new(
        limits: ArtifactSealLimits,
        policy: ArtifactProjectionPolicy,
    ) -> Result<Self, ArtifactSealError> {
        if limits.max_artifacts == 0
            || limits.max_total_bytes == 0
            || limits.max_public_projection_bytes == 0
        {
            return Err(ArtifactSealError::Policy(
                "artifact seal limits must be positive".to_string(),
            ));
        }
        Ok(Self { limits, policy })
    }

    /// Verify the hostile staging tree, canonicalize reviewed public JSON, and
    /// atomically promote a new Rust-owned immutable tree.
    pub fn seal(
        &self,
        staging_root: &Path,
        promoted_root: &Path,
        candidate: &mut EvaluationFinishCandidate,
        quiescence: &IsolationQuiescenceProof,
    ) -> Result<SealedEvaluationArtifacts, ArtifactSealError> {
        candidate
            .validate()
            .map_err(|error| ArtifactSealError::Manifest(error.to_string()))?;
        if !staging_root.is_absolute() || !promoted_root.is_absolute() {
            return Err(ArtifactSealError::Manifest(
                "artifact staging and promoted roots must be absolute".to_string(),
            ));
        }
        if candidate.artifacts.len() > self.limits.max_artifacts {
            return Err(ArtifactSealError::Manifest(
                "artifact count exceeded host limit".to_string(),
            ));
        }
        let claimed_total = candidate
            .artifacts
            .iter()
            .try_fold(0_u64, |sum, artifact| {
                sum.checked_add(artifact.size_bytes).ok_or_else(|| {
                    ArtifactSealError::Manifest("artifact byte total overflow".to_string())
                })
            })?;
        if claimed_total > self.limits.max_total_bytes {
            return Err(ArtifactSealError::Manifest(
                "artifact bytes exceeded host limit".to_string(),
            ));
        }

        let root = open_directory(staging_root)?;
        let walked = walk_tree(root.as_raw_fd())?;
        let declared = candidate
            .artifacts
            .iter()
            .map(|entry| entry.path.clone())
            .collect::<BTreeSet<_>>();
        if walked != declared {
            let undeclared = walked.difference(&declared).next().cloned();
            let missing = declared.difference(&walked).next().cloned();
            return Err(ArtifactSealError::Manifest(format!(
                "artifact tree differed from manifest (undeclared={undeclared:?}, missing={missing:?})"
            )));
        }

        let parent = promoted_root.parent().ok_or_else(|| {
            ArtifactSealError::Promotion("promoted root had no parent".to_string())
        })?;
        std::fs::create_dir_all(parent).map_err(ArtifactSealError::io)?;
        if promoted_root.exists() {
            return Err(ArtifactSealError::Promotion(
                "promoted artifact root already existed".to_string(),
            ));
        }
        let temp_name = format!(
            ".aiperf-evaluation-seal-{}-{}",
            std::process::id(),
            SEAL_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        );
        let temp_root = parent.join(temp_name);
        std::fs::create_dir(&temp_root).map_err(ArtifactSealError::io)?;

        let result = self.copy_verified_entries(
            root.as_raw_fd(),
            &temp_root,
            &candidate.artifacts,
            &candidate.provider_bundle.path,
        );
        let (entries, provider_bundle_sha256) = match result {
            Ok(result) => result,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&temp_root);
                return Err(error);
            }
        };
        // Persist an immutable, owner-only tree. Restricted artifacts carry
        // provider targets and hidden verifier state, so the sealed tree must
        // not remain world-readable, world-traversable, or owner-writable.
        // Hardening happens before promotion so the tree at its final path is
        // never briefly world-readable.
        if let Err(error) = harden_sealed_tree(&temp_root) {
            make_tree_removable(&temp_root);
            let _ = std::fs::remove_dir_all(&temp_root);
            return Err(error);
        }
        std::fs::rename(&temp_root, promoted_root).map_err(|error| {
            make_tree_removable(&temp_root);
            let _ = std::fs::remove_dir_all(&temp_root);
            ArtifactSealError::Promotion(error.to_string())
        })?;
        // Durably persist the rename: per-file `sync_all` cannot survive a crash
        // that loses the parent directory entry created by the rename.
        let parent_dir = File::open(parent).map_err(ArtifactSealError::io)?;
        parent_dir.sync_all().map_err(ArtifactSealError::io)?;

        Ok(SealedEvaluationArtifacts {
            root: promoted_root.to_path_buf(),
            entries,
            provider_bundle_sha256,
            quiescence_proof_sha256: quiescence.proof_sha256().to_string(),
        })
    }

    fn copy_verified_entries(
        &self,
        root_fd: RawFd,
        temp_root: &Path,
        manifest: &[EvaluationArtifactManifestEntry],
        provider_bundle_path: &str,
    ) -> Result<(Vec<SealedEvaluationArtifact>, String), ArtifactSealError> {
        let mut sealed = Vec::with_capacity(manifest.len());
        let mut provider_bundle_sha256 = None;
        for entry in manifest {
            validate_relative_path(&entry.path)?;
            let mut source = open_relative_regular(root_fd, &entry.path)?;
            let before = source.metadata().map_err(ArtifactSealError::io)?;
            if !before.file_type().is_file() || before.len() != entry.size_bytes {
                return Err(ArtifactSealError::Manifest(format!(
                    "artifact {:?} type/size differed from manifest",
                    entry.path
                )));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if before.nlink() != 1 {
                    return Err(ArtifactSealError::Traversal(format!(
                        "artifact {:?} was a hard link",
                        entry.path
                    )));
                }
            }

            let raw_digest = hash_reader(&mut source)?;
            if raw_digest != entry.artifact_content_sha256 {
                return Err(ArtifactSealError::Manifest(format!(
                    "artifact {:?} digest differed from manifest",
                    entry.path
                )));
            }
            let after = source.metadata().map_err(ArtifactSealError::io)?;
            if metadata_changed(&before, &after) {
                return Err(ArtifactSealError::Race(format!(
                    "artifact {:?} changed while hashing",
                    entry.path
                )));
            }
            source
                .seek(SeekFrom::Start(0))
                .map_err(ArtifactSealError::io)?;

            let destination = temp_root.join(&entry.path);
            let destination_parent = destination.parent().ok_or_else(|| {
                ArtifactSealError::Promotion("artifact destination had no parent".to_string())
            })?;
            std::fs::create_dir_all(destination_parent).map_err(ArtifactSealError::io)?;
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&destination)
                .map_err(ArtifactSealError::io)?;

            let (visibility, schema_sha256, final_size, final_digest) = match entry.visibility {
                ArtifactVisibility::Restricted => {
                    std::io::copy(&mut source, &mut output).map_err(ArtifactSealError::io)?;
                    (
                        ArtifactVisibility::Restricted,
                        None,
                        entry.size_bytes,
                        raw_digest.clone(),
                    )
                }
                ArtifactVisibility::PublicProjection => {
                    let rule = self.policy.rule(&entry.path).ok_or_else(|| {
                        ArtifactSealError::Policy(format!(
                            "provider requested unregistered public projection {:?}",
                            entry.path
                        ))
                    })?;
                    if rule.media_type != entry.media_type
                        || entry.size_bytes as usize > rule.max_bytes
                        || entry.size_bytes as usize > self.limits.max_public_projection_bytes
                    {
                        return Err(ArtifactSealError::Policy(format!(
                            "public projection {:?} exceeded its path/media/size rule",
                            entry.path
                        )));
                    }
                    let mut bytes = Vec::with_capacity(entry.size_bytes as usize);
                    source
                        .read_to_end(&mut bytes)
                        .map_err(ArtifactSealError::io)?;
                    let canonical = CanonicalJson::from_slice(
                        &bytes,
                        CanonicalJsonLimits {
                            max_string_bytes: rule.max_bytes,
                            ..Default::default()
                        },
                    )
                    .map_err(|error| ArtifactSealError::Projection(error.to_string()))?;
                    rule.validator.validate(&canonical)?;
                    let canonical_bytes = canonical.to_bytes();
                    output
                        .write_all(&canonical_bytes)
                        .map_err(ArtifactSealError::io)?;
                    (
                        ArtifactVisibility::PublicProjection,
                        Some(rule.validator.schema_sha256().to_string()),
                        canonical_bytes.len() as u64,
                        sha256_hex(&canonical_bytes),
                    )
                }
            };
            output.sync_all().map_err(ArtifactSealError::io)?;
            if entry.path == provider_bundle_path {
                provider_bundle_sha256 = Some(final_digest.clone());
            }
            sealed.push(SealedEvaluationArtifact {
                artifact_id: entry.artifact_id.clone(),
                path: entry.path.clone(),
                media_type: entry.media_type.clone(),
                visibility,
                size_bytes: final_size,
                artifact_content_sha256: final_digest,
                public_projection_schema_sha256: schema_sha256,
            });
        }
        let provider_bundle_sha256 = provider_bundle_sha256.ok_or_else(|| {
            ArtifactSealError::Manifest("provider bundle path was absent from manifest".to_string())
        })?;
        Ok((sealed, provider_bundle_sha256))
    }
}

/// Recursively strip write and world/group access from a freshly built seal
/// tree: regular files become owner read-only (`0o400`) and directories become
/// owner traverse-only (`0o500`). Children are hardened before their parent so
/// the walk never loses the search permission it needs to descend. Any symlink
/// or non-regular entry is a sealing violation because the tree is built solely
/// with `create_new` regular files under `create_dir`/`create_dir_all`.
fn harden_sealed_tree(path: &Path) -> Result<(), ArtifactSealError> {
    let metadata = std::fs::symlink_metadata(path).map_err(ArtifactSealError::io)?;
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        return Err(ArtifactSealError::Traversal(
            "sealed artifact tree contained a symlink".to_string(),
        ));
    }
    if file_type.is_dir() {
        for entry in std::fs::read_dir(path).map_err(ArtifactSealError::io)? {
            let entry = entry.map_err(ArtifactSealError::io)?;
            harden_sealed_tree(&entry.path())?;
        }
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o500))
            .map_err(ArtifactSealError::io)?;
    } else if file_type.is_file() {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o400))
            .map_err(ArtifactSealError::io)?;
    } else {
        return Err(ArtifactSealError::Traversal(
            "sealed artifact tree contained a non-regular entry".to_string(),
        ));
    }
    Ok(())
}

/// Best-effort restoration of owner write/traverse permissions so a partially
/// hardened tree can be removed on a sealing-failure cleanup path. Directories
/// are made writable before descending, and every error is ignored because this
/// only runs while unwinding an already-failed seal.
fn make_tree_removable(path: &Path) {
    let Ok(metadata) = std::fs::symlink_metadata(path) else {
        return;
    };
    let file_type = metadata.file_type();
    if file_type.is_symlink() {
        return;
    }
    if file_type.is_dir() {
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o700));
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                make_tree_removable(&entry.path());
            }
        }
    } else {
        let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
    }
}

fn open_directory(path: &Path) -> Result<OwnedFd, ArtifactSealError> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| ArtifactSealError::Traversal("directory path contained NUL".to_string()))?;
    // SAFETY: `c_path` is NUL-terminated and flags request a read-only directory
    // descriptor with no symlink following.
    let fd = unsafe {
        libc::open(
            c_path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(ArtifactSealError::io(std::io::Error::last_os_error()));
    }
    // SAFETY: `open` returned a new owned descriptor.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

fn open_relative_regular(root_fd: RawFd, path: &str) -> Result<File, ArtifactSealError> {
    validate_relative_path(path)?;
    let components = Path::new(path)
        .components()
        .map(|component| match component {
            Component::Normal(value) => Ok(value.to_owned()),
            _ => Err(ArtifactSealError::Traversal(format!(
                "artifact path {path:?} contained traversal"
            ))),
        })
        .collect::<Result<Vec<_>, _>>()?;
    // SAFETY: `dup` creates an independent owned descriptor for traversal.
    let duplicate = unsafe { libc::dup(root_fd) };
    if duplicate < 0 {
        return Err(ArtifactSealError::io(std::io::Error::last_os_error()));
    }
    // SAFETY: `dup` returned a new owned descriptor.
    let mut current = unsafe { OwnedFd::from_raw_fd(duplicate) };
    for (index, component) in components.iter().enumerate() {
        let name = CString::new(component.as_bytes()).map_err(|_| {
            ArtifactSealError::Traversal("artifact component contained NUL".to_string())
        })?;
        let final_component = index + 1 == components.len();
        let flags = if final_component {
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC
        } else {
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC
        };
        // SAFETY: `current` and `name` are valid, and no-follow is used at every component.
        let next = unsafe { libc::openat(current.as_raw_fd(), name.as_ptr(), flags) };
        if next < 0 {
            return Err(ArtifactSealError::io(std::io::Error::last_os_error()));
        }
        // SAFETY: `openat` returned a new owned descriptor.
        current = unsafe { OwnedFd::from_raw_fd(next) };
    }
    // SAFETY: ownership moves from `OwnedFd` into `File` exactly once.
    Ok(unsafe { File::from_raw_fd(current.into_raw_fd()) })
}

trait IntoRawFdExt {
    fn into_raw_fd(self) -> RawFd;
}

impl IntoRawFdExt for OwnedFd {
    fn into_raw_fd(self) -> RawFd {
        use std::os::fd::IntoRawFd;
        IntoRawFd::into_raw_fd(self)
    }
}

fn walk_tree(root_fd: RawFd) -> Result<BTreeSet<String>, ArtifactSealError> {
    let mut files = BTreeSet::new();
    walk_directory(root_fd, Path::new(""), &mut files)?;
    Ok(files)
}

fn walk_directory(
    directory_fd: RawFd,
    prefix: &Path,
    files: &mut BTreeSet<String>,
) -> Result<(), ArtifactSealError> {
    // SAFETY: `dup` returns an independent descriptor consumed by `fdopendir`.
    let duplicate = unsafe { libc::dup(directory_fd) };
    if duplicate < 0 {
        return Err(ArtifactSealError::io(std::io::Error::last_os_error()));
    }
    // SAFETY: `duplicate` is a valid directory descriptor; `closedir` below owns it.
    let stream = unsafe { libc::fdopendir(duplicate) };
    if stream.is_null() {
        // SAFETY: `fdopendir` did not consume descriptor on failure.
        unsafe { libc::close(duplicate) };
        return Err(ArtifactSealError::io(std::io::Error::last_os_error()));
    }
    loop {
        // SAFETY: `stream` remains valid until `closedir`.
        let entry = unsafe { libc::readdir(stream) };
        if entry.is_null() {
            break;
        }
        // SAFETY: POSIX `dirent.d_name` is NUL-terminated for this entry.
        let name = unsafe { CStr::from_ptr((*entry).d_name.as_ptr()) };
        if name.to_bytes() == b"." || name.to_bytes() == b".." {
            continue;
        }
        let os_name = OsStr::from_bytes(name.to_bytes());
        let relative = prefix.join(os_name);
        let relative_text = relative.to_str().ok_or_else(|| {
            ArtifactSealError::Traversal("artifact path was not valid UTF-8".to_string())
        })?;
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        // SAFETY: pointers are valid and `stat` is initialized on success.
        let status = unsafe {
            libc::fstatat(
                directory_fd,
                name.as_ptr(),
                stat.as_mut_ptr(),
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if status != 0 {
            // SAFETY: closes the directory stream and its duplicated descriptor.
            unsafe { libc::closedir(stream) };
            return Err(ArtifactSealError::Race(format!(
                "artifact entry {relative_text:?} changed during traversal"
            )));
        }
        // SAFETY: `fstatat` succeeded.
        let stat = unsafe { stat.assume_init() };
        let kind = stat.st_mode & libc::S_IFMT;
        if kind == libc::S_IFLNK {
            // SAFETY: closes the directory stream and its duplicated descriptor.
            unsafe { libc::closedir(stream) };
            return Err(ArtifactSealError::Traversal(format!(
                "artifact entry {relative_text:?} was a symlink"
            )));
        }
        if kind == libc::S_IFDIR {
            // SAFETY: no-follow opens the exact child directory.
            let child = unsafe {
                libc::openat(
                    directory_fd,
                    name.as_ptr(),
                    libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
                )
            };
            if child < 0 {
                // SAFETY: closes the directory stream and its duplicated descriptor.
                unsafe { libc::closedir(stream) };
                return Err(ArtifactSealError::io(std::io::Error::last_os_error()));
            }
            let recurse = walk_directory(child, &relative, files);
            // SAFETY: `child` is a separately opened descriptor.
            unsafe { libc::close(child) };
            recurse?;
        } else if kind == libc::S_IFREG {
            if stat.st_nlink != 1 || !files.insert(relative_text.to_string()) {
                // SAFETY: closes the directory stream and its duplicated descriptor.
                unsafe { libc::closedir(stream) };
                return Err(ArtifactSealError::Traversal(format!(
                    "artifact entry {relative_text:?} was hard-linked or duplicated"
                )));
            }
        } else {
            // SAFETY: closes the directory stream and its duplicated descriptor.
            unsafe { libc::closedir(stream) };
            return Err(ArtifactSealError::Traversal(format!(
                "artifact entry {relative_text:?} was a device, socket, or special file"
            )));
        }
    }
    // SAFETY: closes the directory stream and its duplicated descriptor once.
    unsafe { libc::closedir(stream) };
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<(), ArtifactSealError> {
    let parsed = Path::new(path);
    if path.is_empty()
        || path.len() > 4_096
        || path.contains('\0')
        || parsed.is_absolute()
        || parsed
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ArtifactSealError::Traversal(
            "artifact path was empty, absolute, oversized, or contained traversal".to_string(),
        ));
    }
    Ok(())
}

fn hash_reader(file: &mut File) -> Result<String, ArtifactSealError> {
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).map_err(ArtifactSealError::io)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    Ok(output)
}

fn metadata_changed(before: &std::fs::Metadata, after: &std::fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        before.dev() != after.dev()
            || before.ino() != after.ino()
            || before.len() != after.len()
            || before.mtime() != after.mtime()
            || before.mtime_nsec() != after.mtime_nsec()
            || before.ctime() != after.ctime()
            || before.ctime_nsec() != after.ctime_nsec()
    }
    #[cfg(not(unix))]
    {
        before.len() != after.len() || before.modified().ok() != after.modified().ok()
    }
}

/// Artifact verification, projection, or atomic-promotion failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactSealError {
    /// Candidate manifest was inconsistent.
    Manifest(String),
    /// Path traversal, link, or special-file attempt.
    Traversal(String),
    /// File identity changed during verification.
    Race(String),
    /// Provider requested an unregistered public projection.
    Policy(String),
    /// Public projection did not satisfy its factory schema.
    Projection(String),
    /// Atomic immutable-tree promotion failed.
    Promotion(String),
    /// Local filesystem operation failed.
    Io(String),
}

impl ArtifactSealError {
    fn io(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl Display for ArtifactSealError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Manifest(message) => write!(formatter, "artifact manifest failed: {message}"),
            Self::Traversal(message) => write!(formatter, "artifact traversal failed: {message}"),
            Self::Race(message) => write!(formatter, "artifact race detected: {message}"),
            Self::Policy(message) => {
                write!(formatter, "artifact projection policy failed: {message}")
            }
            Self::Projection(message) => {
                write!(formatter, "artifact public projection failed: {message}")
            }
            Self::Promotion(message) => write!(formatter, "artifact promotion failed: {message}"),
            Self::Io(message) => write!(formatter, "artifact filesystem I/O failed: {message}"),
        }
    }
}

impl std::error::Error for ArtifactSealError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::artifact_content_sha256;
    use crate::provider_protocol::{
        AggregateMetric, ArtifactRef, EvaluationArtifactId, EvaluationCaseTemplateDescriptor,
        EvaluationHostIdentity, EvaluationIdentity, EvaluationIdentityComponent,
        EvaluationProviderId, EvaluationUnitTemplateDescriptor, EvaluationWorkerIdentity,
    };

    struct AnyObjectSchema(String);

    impl PublicArtifactProjectionValidator for AnyObjectSchema {
        fn schema_sha256(&self) -> &str {
            &self.0
        }

        fn validate(&self, value: &CanonicalJson) -> Result<(), ArtifactSealError> {
            if value.value().is_object() {
                Ok(())
            } else {
                Err(ArtifactSealError::Projection("expected object".to_string()))
            }
        }
    }

    fn candidate(bytes: &[u8], path: &str) -> EvaluationFinishCandidate {
        let artifact_id = EvaluationArtifactId::new("bundle").unwrap();
        EvaluationFinishCandidate {
            identity: EvaluationIdentity {
                canonical_json_codec: crate::canonical::CANONICAL_JSON_CODEC.to_string(),
                worker: EvaluationWorkerIdentity {
                    evaluator_protocol: 2,
                    provider_id: EvaluationProviderId::new("fixture").unwrap(),
                    distribution_id: crate::provider_protocol::EvaluationDistributionId::new(
                        "fixture",
                    )
                    .unwrap(),
                    package: "fixture".to_string(),
                    package_version: "1".to_string(),
                    provider_source_sha256: "a".repeat(64),
                    worker_source_sha256: "b".repeat(64),
                    dependency_lock_sha256: "c".repeat(64),
                    python_version: "3.12".to_string(),
                    launch_nonce: "n".repeat(32),
                    oci_digest: None,
                    operations: vec![
                        "plan_session",
                        "bind_assets",
                        "next_units",
                        "instantiate_units",
                        "start_units",
                        "poll_events",
                        "submit_host_events",
                        "cancel_units",
                        "finalize_session",
                        "shutdown",
                    ]
                    .into_iter()
                    .map(str::to_string)
                    .collect(),
                },
                config_schema_sha256: "d".repeat(64),
                resolved_config_sha256: "e".repeat(64),
                dataset: EvaluationIdentityComponent {
                    name: "fixture".to_string(),
                    version: "1".to_string(),
                    source_sha256: "f".repeat(64),
                    source_commit: None,
                    base_source_sha256: None,
                    overlay_policy: None,
                    overlays: Vec::new(),
                },
                components: Vec::new(),
                ordered_manifest_sha256: "1".repeat(64),
                case_templates: vec![EvaluationCaseTemplateDescriptor {
                    template_id: crate::provider_protocol::EvaluationCaseTemplateId::new("case-t")
                        .unwrap(),
                    task: "fixture".to_string(),
                    source: "fixture".to_string(),
                }],
                unit_templates: vec![EvaluationUnitTemplateDescriptor {
                    unit_template_id: crate::provider_protocol::EvaluationUnitTemplateId::new(
                        "unit-t",
                    )
                    .unwrap(),
                    case_template_ids: vec![
                        crate::provider_protocol::EvaluationCaseTemplateId::new("case-t").unwrap(),
                    ],
                    granularity:
                        crate::provider_protocol::EvaluationExecutionGranularity::HostBatch,
                    scheduling_class: "fixture".to_string(),
                }],
                policies: CanonicalJson::new(serde_json::json!({})).unwrap(),
                host: EvaluationHostIdentity {
                    runner_sha256: "2".repeat(64),
                    capability_inventory_sha256: "3".repeat(64),
                    schema_inventory_sha256: "4".repeat(64),
                    isolation_proof_sha256: "5".repeat(64),
                },
                route_map_sha256: "6".repeat(64),
                prepared_endpoints_sha256: "7".repeat(64),
                sandbox_sha256: None,
            },
            outcomes: Vec::new(),
            aggregates: Vec::<AggregateMetric>::new(),
            artifacts: vec![EvaluationArtifactManifestEntry {
                artifact_id: artifact_id.clone(),
                path: path.to_string(),
                media_type: "application/json".to_string(),
                visibility: ArtifactVisibility::Restricted,
                size_bytes: bytes.len() as u64,
                artifact_content_sha256: artifact_content_sha256(bytes),
            }],
            provider_bundle: ArtifactRef {
                artifact_id,
                path: path.to_string(),
                visibility: ArtifactVisibility::Restricted,
            },
            normalized_result_sha256: "8".repeat(64),
        }
    }

    fn temp_roots(label: &str) -> (PathBuf, PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!(
            "aiperf-artifact-{label}-{}-{}",
            std::process::id(),
            SEAL_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let staging = base.join("staging");
        let promoted = base.join("promoted");
        std::fs::create_dir_all(&staging).unwrap();
        (base, staging, promoted)
    }

    #[test]
    fn seals_exact_declared_bytes_only_after_quiescence_proof() {
        let bytes = br#"{"result":1}"#;
        let (base, staging, promoted) = temp_roots("success");
        std::fs::write(staging.join("bundle.json"), bytes).unwrap();
        let mut candidate = candidate(bytes, "bundle.json");
        let proof = IsolationQuiescenceProof::verified(42, "9".repeat(64));
        let sealed = EvaluationArtifactSealer::new(
            ArtifactSealLimits::default(),
            ArtifactProjectionPolicy::restricted_only(),
        )
        .unwrap()
        .seal(&staging, &promoted, &mut candidate, &proof)
        .unwrap();
        assert_eq!(
            sealed.provider_bundle_sha256,
            artifact_content_sha256(bytes)
        );
        assert_eq!(std::fs::read(promoted.join("bundle.json")).unwrap(), bytes);
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn sealed_tree_is_owner_read_only_and_not_world_readable() {
        let bytes = br#"{"result":1}"#;
        let (base, staging, promoted) = temp_roots("hardened");
        std::fs::write(staging.join("bundle.json"), bytes).unwrap();
        let mut candidate = candidate(bytes, "bundle.json");
        let proof = IsolationQuiescenceProof::verified(42, "9".repeat(64));
        EvaluationArtifactSealer::new(
            ArtifactSealLimits::default(),
            ArtifactProjectionPolicy::restricted_only(),
        )
        .unwrap()
        .seal(&staging, &promoted, &mut candidate, &proof)
        .unwrap();

        let dir_mode = std::fs::symlink_metadata(&promoted)
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        let file_mode = std::fs::symlink_metadata(promoted.join("bundle.json"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777;
        // Owner traverse-only directory; owner read-only file. No group/world
        // bits and no owner-write bit: restricted artifacts are immutable and
        // never world-readable once sealed.
        assert_eq!(
            dir_mode, 0o500,
            "sealed directory must be owner traverse-only"
        );
        assert_eq!(file_mode, 0o400, "sealed file must be owner read-only");
        // The owning runner can still read the sealed content.
        assert_eq!(std::fs::read(promoted.join("bundle.json")).unwrap(), bytes);

        make_tree_removable(&promoted);
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn rejects_undeclared_files_and_symlinks() {
        let bytes = b"{}";
        let (base, staging, promoted) = temp_roots("hostile");
        std::fs::write(staging.join("bundle.json"), bytes).unwrap();
        std::fs::write(staging.join("hidden.txt"), b"hidden").unwrap();
        let mut candidate = candidate(bytes, "bundle.json");
        let proof = IsolationQuiescenceProof::verified(42, "9".repeat(64));
        let error = EvaluationArtifactSealer::new(
            ArtifactSealLimits::default(),
            ArtifactProjectionPolicy::restricted_only(),
        )
        .unwrap()
        .seal(&staging, &promoted, &mut candidate, &proof)
        .unwrap_err();
        assert!(matches!(error, ArtifactSealError::Manifest(_)));
        std::fs::remove_file(staging.join("hidden.txt")).unwrap();
        std::os::unix::fs::symlink("bundle.json", staging.join("link.json")).unwrap();
        let error = EvaluationArtifactSealer::new(
            ArtifactSealLimits::default(),
            ArtifactProjectionPolicy::restricted_only(),
        )
        .unwrap()
        .seal(&staging, &promoted, &mut candidate, &proof)
        .unwrap_err();
        assert!(matches!(error, ArtifactSealError::Traversal(_)));
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn public_projection_requires_factory_rule_and_is_canonicalized() {
        let bytes = b"{\n \"z\": 1, \"a\": 2\n}";
        let (base, staging, promoted) = temp_roots("public");
        std::fs::write(staging.join("bundle.json"), bytes).unwrap();
        let mut candidate = candidate(bytes, "bundle.json");
        candidate.artifacts[0].visibility = ArtifactVisibility::PublicProjection;
        candidate.provider_bundle.visibility = ArtifactVisibility::PublicProjection;
        let proof = IsolationQuiescenceProof::verified(42, "9".repeat(64));
        let mut policy = ArtifactProjectionPolicy::restricted_only();
        policy
            .register(
                "bundle.json",
                "application/json",
                1024,
                Arc::new(AnyObjectSchema("a".repeat(64))),
            )
            .unwrap();
        let sealed = EvaluationArtifactSealer::new(ArtifactSealLimits::default(), policy)
            .unwrap()
            .seal(&staging, &promoted, &mut candidate, &proof)
            .unwrap();
        assert_eq!(
            std::fs::read(promoted.join("bundle.json")).unwrap(),
            br#"{"a":2,"z":1}"#
        );
        assert_eq!(
            sealed.entries[0].public_projection_schema_sha256,
            Some("a".repeat(64))
        );
        let _ = std::fs::remove_dir_all(base);
    }
}
