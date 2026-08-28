// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Process-lifetime plugin residency tracking (Task 14).
//!
//! Once a plugin artifact is loaded via dlopen it resides in the process
//! address space for the lifetime of the process.  The `ResidencyLedger`
//! records every loaded artifact digest and its staged path so that:
//!   1. The same artifact is never opened twice (idempotent).
//!   2. A new path claiming the same digest from a different staged location
//!      is detected as a `LoadError::ResidencyConflict`.

use std::{collections::HashMap, path::PathBuf};

use crate::error::LoadError;

/// Residence record for one loaded artifact.
#[derive(Debug, Clone)]
pub struct ResidencyRecord {
    /// Hex-encoded BLAKE3 digest.
    pub digest: String,
    /// Staged path that was passed to dlopen.
    pub staged_path: PathBuf,
}

/// Process-lifetime ledger of loaded artifact digests.
///
/// Must be kept alive (not dropped) for the process lifetime.  Dropping it
/// does NOT call dlclose; it only loses the bookkeeping records.
#[derive(Debug, Default)]
pub struct ResidencyLedger {
    /// Map from digest → staged path.
    by_digest: HashMap<String, PathBuf>,
}

impl ResidencyLedger {
    /// Create an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempt to register a loaded artifact.
    ///
    /// - If the digest is unknown, the record is inserted and `Ok(())` returned.
    /// - If the digest is already recorded with the **same** staged path, the
    ///   call is idempotent and returns `Ok(())`.
    /// - If the digest is already recorded with a **different** staged path,
    ///   returns `Err(LoadError::ResidencyConflict)`.
    pub fn register(
        &mut self,
        digest: String,
        staged_path: PathBuf,
    ) -> Result<(), LoadError> {
        match self.by_digest.get(&digest) {
            None => {
                self.by_digest.insert(digest, staged_path);
                Ok(())
            }
            Some(existing) if *existing == staged_path => Ok(()),
            Some(existing) => Err(LoadError::ResidencyConflict {
                digest,
                existing: existing.clone(),
                new: staged_path,
            }),
        }
    }

    /// Return the staged path for a known digest, if any.
    pub fn lookup(&self, digest: &str) -> Option<&PathBuf> {
        self.by_digest.get(digest)
    }

    /// Return the number of artifacts currently registered.
    pub fn len(&self) -> usize {
        self.by_digest.len()
    }

    /// Return `true` if no artifacts are registered.
    pub fn is_empty(&self) -> bool {
        self.by_digest.is_empty()
    }

    /// Iterate all registered records.
    pub fn records(&self) -> impl Iterator<Item = ResidencyRecord> + '_ {
        self.by_digest.iter().map(|(digest, path)| ResidencyRecord {
            digest: digest.clone(),
            staged_path: path.clone(),
        })
    }
}
