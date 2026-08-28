// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable no-follow acquisition of plugin manifests and artifacts.
//!
//! Every path is opened with `O_NOFOLLOW` (Unix) / reparse-point flags
//! (Windows) so that a symlink substituted at the final component is caught
//! before any bytes are read.  After reading, the bytes are hashed with BLAKE3
//! and compared against the caller-supplied digest to detect content
//! substitution.

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{AcquireError, ManifestError};
use crate::manifest::PluginManifestV2;
use crate::normalize::normalize_manifest;
use crate::platform::fs::{is_symlink, open_no_follow};

/// A manifest that has been read from disk, hashed, parsed, and normalized.
pub struct AcquiredManifest {
    /// Raw bytes exactly as stored on disk.
    pub raw_bytes: Vec<u8>,
    /// Parsed and normalized manifest.
    pub canonical: PluginManifestV2,
    /// Original filesystem path (for diagnostics only; not used for security).
    pub source_path: PathBuf,
    /// BLAKE3 hex digest of `raw_bytes`.
    pub digest: String,
}

impl AcquiredManifest {
    /// Open `path` with `O_NOFOLLOW`, read, hash, parse, and normalize.
    ///
    /// Returns `AcquireError::Symlink` if the final path component is a
    /// symbolic link.
    pub fn acquire(path: &Path) -> Result<Self, AcquireError> {
        if is_symlink(path)? {
            return Err(AcquireError::Symlink(path.to_path_buf()));
        }
        let mut file = open_no_follow(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidInput {
                AcquireError::Symlink(path.to_path_buf())
            } else {
                AcquireError::Io(e)
            }
        })?;
        let mut raw_bytes = Vec::new();
        file.read_to_end(&mut raw_bytes)?;
        let digest = blake3::hash(&raw_bytes).to_hex().to_string();
        let canonical: PluginManifestV2 = serde_yaml::from_slice(&raw_bytes)
            .map_err(|e| ManifestError::ParseError(e.to_string()))?;
        let canonical = normalize_manifest(canonical)?;
        Ok(Self {
            raw_bytes,
            canonical,
            source_path: path.to_path_buf(),
            digest,
        })
    }
}

/// A plugin artifact (.so / .dylib / .dll) that has been read and content-verified.
pub struct AcquiredArtifact {
    /// Raw bytes of the artifact.
    pub raw_bytes: Vec<u8>,
    /// Original filesystem path (for diagnostics only).
    pub source_path: PathBuf,
    /// BLAKE3 hex digest of `raw_bytes` — verified against the manifest record.
    pub digest: String,
    /// Target triple this artifact is built for.
    pub target: String,
}

impl AcquiredArtifact {
    /// Open `path` without following symlinks, read, and verify that the
    /// BLAKE3 digest of the content equals `expected_digest`.
    pub fn acquire(path: &Path, expected_digest: &str, target: &str) -> Result<Self, AcquireError> {
        if is_symlink(path)? {
            return Err(AcquireError::Symlink(path.to_path_buf()));
        }
        let mut file = open_no_follow(path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::InvalidInput {
                AcquireError::Symlink(path.to_path_buf())
            } else {
                AcquireError::Io(e)
            }
        })?;
        let mut raw_bytes = Vec::new();
        file.read_to_end(&mut raw_bytes)?;
        let digest = blake3::hash(&raw_bytes).to_hex().to_string();
        if digest != expected_digest {
            return Err(AcquireError::DigestMismatch {
                expected: expected_digest.to_string(),
                actual: digest,
            });
        }
        Ok(Self {
            raw_bytes,
            source_path: path.to_path_buf(),
            digest,
            target: target.to_string(),
        })
    }
}
