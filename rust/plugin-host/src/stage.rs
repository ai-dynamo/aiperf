// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-addressed staging of acquired plugin artifacts.
//!
//! `CanonicalObjectMap` coalesces identical artifacts: two plugins that ship
//! the same bytes for the same target result in exactly one staged file.
//! The loader receives only the private staged path after rehash verification.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use crate::acquire::AcquiredArtifact;
use crate::error::AcquireError;

/// Origin classification of a staged object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectOrigin {
    Executable,
    Baseline,
    CanonicalStage,
}

/// A content-addressed copy of an artifact in the host-owned staging area.
pub struct StagedObject {
    pub digest: String,
    pub staged_path: PathBuf,
    pub origin: ObjectOrigin,
    pub target: String,
}

/// Process-wide content-addressed map from `(loader_identity, digest)` to a
/// single private staged file.
///
/// Identical `(target, digest)` pairs are coalesced: the second call to
/// `stage()` with the same key returns the already-staged path without copying.
pub struct CanonicalObjectMap {
    inner: HashMap<(String, String), StagedObject>,
    staging_dir: PathBuf,
}

impl CanonicalObjectMap {
    /// Create a new map that stages files under `staging_dir`.
    pub fn new(staging_dir: PathBuf) -> Self {
        Self {
            inner: HashMap::new(),
            staging_dir,
        }
    }

    /// Stage `artifact` under `loader_id`.
    ///
    /// If an object with the same `(loader_id, digest)` key already exists,
    /// returns a reference to it without re-copying.  Otherwise copies the
    /// artifact bytes to a private file, rehashes to detect tampering, and
    /// stores the `StagedObject`.
    pub fn stage(
        &mut self,
        artifact: &AcquiredArtifact,
        loader_id: &str,
    ) -> Result<&StagedObject, AcquireError> {
        let key = (loader_id.to_string(), artifact.digest.clone());
        if !self.inner.contains_key(&key) {
            let staged = self.copy_and_verify(artifact, loader_id)?;
            self.inner.insert(key.clone(), staged);
        }
        Ok(&self.inner[&key])
    }

    fn copy_and_verify(
        &self,
        artifact: &AcquiredArtifact,
        loader_id: &str,
    ) -> Result<StagedObject, AcquireError> {
        // Destination: <staging_dir>/<loader_id>/<digest>
        let dest_dir = self
            .staging_dir
            .join(loader_id.replace(['/', '\\', ':'], "_"));
        std::fs::create_dir_all(&dest_dir)?;
        let dest = dest_dir.join(&artifact.digest);
        // Create with 0600 (owner rw only) so no other process can read the
        // binary before the loader verifies and dlopen()s it.
        #[cfg(unix)]
        {
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&dest)?;
            file.write_all(&artifact.raw_bytes)?;
        }
        #[cfg(not(unix))]
        std::fs::write(&dest, &artifact.raw_bytes)?;
        // Rehash after write to detect I/O corruption or tamper.
        let actual = rehash(&dest)?;
        if actual != artifact.digest {
            return Err(AcquireError::StagedTamper(dest));
        }
        Ok(StagedObject {
            digest: artifact.digest.clone(),
            staged_path: dest,
            origin: ObjectOrigin::CanonicalStage,
            target: artifact.target.clone(),
        })
    }
}

fn rehash(path: &Path) -> Result<String, AcquireError> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(blake3::hash(&buf).to_hex().to_string())
}
