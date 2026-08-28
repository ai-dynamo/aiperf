// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical staged-object map (Task 11).
//!
//! Deduplicates acquired artifacts by content digest and writes each unique
//! object exactly once into the staging directory under its digest-named path.
//! Conflicts between two artifacts claiming the same loader identity but
//! different digests are rejected before staging.

use std::{collections::HashMap, path::PathBuf};

use crate::{acquire::AcquiredArtifact, error::AcquireError};

/// Records where an object was sourced from for audit purposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectOrigin {
    /// The binary is the running `aiperf` executable itself (allocator provider).
    Executable,
    /// The binary is a baseline artifact declared in a manifest.
    Baseline,
    /// The binary was already present in the canonical staging directory.
    CanonicalStage,
}

/// One object as it exists in the canonical staging directory.
#[derive(Debug, Clone)]
pub struct StagedObject {
    /// Hex-encoded BLAKE3 digest; also the filename in the staging directory.
    pub digest: String,
    /// Absolute path in the staging directory.
    pub staged_path: PathBuf,
    /// Where this object originated.
    pub origin: ObjectOrigin,
    /// Rustc target triple.
    pub target: String,
}

/// Deduplicating map from `(loader_id, digest)` to staged objects.
///
/// Each unique `(loader_id, digest)` pair is written to the staging directory
/// exactly once.  A second artifact with the same `(loader_id, digest)` is
/// silently accepted (idempotent).  A second artifact with the same `loader_id`
/// but a different `digest` is a `ConflictingLoaderIdentity` error.
pub struct CanonicalObjectMap {
    inner: HashMap<(String, String), StagedObject>,
    staging_dir: PathBuf,
}

impl CanonicalObjectMap {
    /// Create an empty map backed by `staging_dir`.
    pub fn new(staging_dir: PathBuf) -> Self {
        Self {
            inner: HashMap::new(),
            staging_dir,
        }
    }

    /// Stage `artifact` under `loader_id` and return a reference to the
    /// canonical `StagedObject`.
    ///
    /// Idempotent for identical `(loader_id, digest)` pairs.
    pub fn stage(
        &mut self,
        _artifact: &AcquiredArtifact,
        _loader_id: &str,
    ) -> Result<&StagedObject, AcquireError> {
        unimplemented!("Task 11: canonical staging not yet implemented")
    }

    /// Return the staging directory root.
    pub fn staging_dir(&self) -> &PathBuf {
        &self.staging_dir
    }
}
