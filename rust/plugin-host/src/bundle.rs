// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic publish and tamper-detected load of a [`PluginLockV1`] on disk.
//!
//! [`LockedCatalogBundle::publish`] writes the lock to a path atomically:
//! it serializes to a temp file in the same directory and renames it into
//! place so a concurrent reader never sees a partial write.
//!
//! [`LockedCatalogBundle::load_and_verify`] reads the file and recomputes the
//! digest, rejecting any mutation made after publish.

use std::{
    fmt,
    path::{Path, PathBuf},
};

use crate::lock::PluginLockV1;

/// Why a bundle publish or load failed.
#[derive(Debug)]
pub enum BundleError {
    /// The lock file could not be read or written.
    Io(std::io::Error),
    /// The file contents could not be parsed as a [`PluginLockV1`].
    Parse(String),
    /// The `schema_version` field is not `"1.0"`.
    UnsupportedSchemaVersion(String),
    /// The on-disk digest did not match the recomputed package-list digest.
    DigestMismatch {
        /// The digest stored in the file.
        stored: String,
        /// The digest recomputed from the package list.
        computed: String,
    },
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "plugin lock I/O error: {e}"),
            Self::Parse(s) => write!(f, "plugin lock parse error: {s}"),
            Self::UnsupportedSchemaVersion(v) => {
                write!(f, "unsupported plugin lock schema version: {v:?}")
            }
            Self::DigestMismatch { stored, computed } => write!(
                f,
                "plugin lock digest mismatch: stored={stored} computed={computed}"
            ),
        }
    }
}

impl std::error::Error for BundleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for BundleError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// An atomically published, tamper-detected plugin lock on disk.
#[derive(Debug)]
pub struct LockedCatalogBundle {
    lock: PluginLockV1,
    path: PathBuf,
}

impl LockedCatalogBundle {
    /// Atomically write `lock` to `path`.
    ///
    /// Serializes to a sibling temp file and renames it into place.  The
    /// rename is atomic on all POSIX systems; a concurrent reader never
    /// observes a partial write.
    pub fn publish(lock: PluginLockV1, path: &Path) -> Result<Self, BundleError> {
        let json = serde_json::to_vec_pretty(&lock)
            .expect("PluginLockV1 serializes infallibly");

        // Write to a temp file in the same directory so rename is atomic.
        let parent = path
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let tmp = parent.join(format!(
            ".tmp-plugin-lock-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.subsec_nanos())
                .unwrap_or(0)
        ));
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, path)?;

        Ok(Self {
            lock,
            path: path.to_owned(),
        })
    }

    /// Read a previously published lock from `path` and verify its digest.
    pub fn load_and_verify(path: &Path) -> Result<Self, BundleError> {
        let raw = std::fs::read(path)?;
        let lock: PluginLockV1 =
            serde_json::from_slice(&raw).map_err(|e| BundleError::Parse(e.to_string()))?;

        if lock.schema_version != "1.0" {
            return Err(BundleError::UnsupportedSchemaVersion(lock.schema_version));
        }

        if !lock.verify() {
            let computed = crate::lock::PluginLockDigest::compute(&lock.packages);
            return Err(BundleError::DigestMismatch {
                stored: lock.digest.hex.clone(),
                computed: computed.hex,
            });
        }

        Ok(Self {
            lock,
            path: path.to_owned(),
        })
    }

    /// The verified lock contents.
    pub fn lock(&self) -> &PluginLockV1 {
        &self.lock
    }

    /// The path where this bundle is stored.
    pub fn path(&self) -> &Path {
        &self.path
    }
}
