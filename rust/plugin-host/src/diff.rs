// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Diff between two [`PluginLockV1`] documents.
//!
//! [`diff_locks`] compares by package `id`.  A package present in both with
//! any field changed (version, status, artifact digest, or closure digest) is
//! reported as changed.  Packages only in the old lock are removed; only in
//! the new lock are added.

use crate::lock::{LockedPackageV1, PackageStatus, PluginLockV1};

/// One package entry in a lock diff.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LockDiffEntry {
    /// Normalized package identifier.
    pub id: String,
    /// Semantic version string.
    pub version: String,
    /// Package status at the time this entry was recorded.
    pub status: PackageStatus,
}

impl From<&LockedPackageV1> for LockDiffEntry {
    fn from(p: &LockedPackageV1) -> Self {
        Self {
            id: p.id.clone(),
            version: p.version.clone(),
            status: p.status.clone(),
        }
    }
}

/// The result of comparing two [`PluginLockV1`] documents.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct LockDiff {
    /// Packages present in the new lock but absent from the old.
    pub added: Vec<LockDiffEntry>,
    /// Packages present in the old lock but absent from the new.
    pub removed: Vec<LockDiffEntry>,
    /// Packages present in both locks with at least one field changed.
    /// Each entry is `(old, new)`.
    pub changed: Vec<(LockDiffEntry, LockDiffEntry)>,
}

impl LockDiff {
    /// True when no packages were added, removed, or changed.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// Compute the diff between `old` and `new` plugin locks.
///
/// Comparison key is package `id`.  Order within each result list follows
/// the order in which packages appear in the respective lock.
pub fn diff_locks(old: &PluginLockV1, new: &PluginLockV1) -> LockDiff {
    use std::collections::HashMap;

    let old_map: HashMap<&str, &LockedPackageV1> =
        old.packages.iter().map(|p| (p.id.as_str(), p)).collect();
    let new_map: HashMap<&str, &LockedPackageV1> =
        new.packages.iter().map(|p| (p.id.as_str(), p)).collect();

    let mut diff = LockDiff::default();

    for new_pkg in &new.packages {
        match old_map.get(new_pkg.id.as_str()) {
            None => diff.added.push(LockDiffEntry::from(new_pkg)),
            Some(old_pkg) => {
                if *old_pkg != new_pkg {
                    diff.changed
                        .push((LockDiffEntry::from(*old_pkg), LockDiffEntry::from(new_pkg)));
                }
            }
        }
    }

    for old_pkg in &old.packages {
        if !new_map.contains_key(old_pkg.id.as_str()) {
            diff.removed.push(LockDiffEntry::from(old_pkg));
        }
    }

    diff
}
