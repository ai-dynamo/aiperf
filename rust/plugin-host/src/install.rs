// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic generation installation.
//!
//! An install root holds numbered immutable generations plus two pointer files:
//!
//! ```text
//! <root>/generations/<id>/...        one complete generation per directory
//! <root>/generations/<id>/.ready     written last inside staging
//! <root>/staging/<unique>/...        in-flight material, never observed
//! <root>/current                     id of the generation in service
//! <root>/previous                    id to roll back to
//! ```
//!
//! Every install materializes its files under `staging/`, writes the ready
//! marker there, and only then `rename`s the staging directory into
//! `generations/<id>`.  A generation directory therefore appears fully formed or
//! not at all, and a crash at any point leaves the previous generation current.
//! Resolution ignores any generation without a ready marker, so debris from an
//! interrupted install is invisible rather than dangerous.
//!
//! Pointer updates use the same temporary-file-plus-`rename` discipline, so a
//! reader resolving `current` observes one complete id.

use std::path::{Component, Path, PathBuf};

use crate::error::{InstallError, InventoryError};
use crate::inventory::{AuthenticatedInventory, PluginInventoryV1};

/// Marker file written last inside a staged generation.  Its presence is the
/// only evidence that a generation directory is complete.
pub const READY_MARKER: &str = ".ready";

/// File name of the inventory recorded inside each generation.
pub const GENERATION_INVENTORY: &str = "inventory.json";

const GENERATIONS_DIR: &str = "generations";
const STAGING_DIR: &str = "staging";
const CURRENT_POINTER: &str = "current";
const PREVIOUS_POINTER: &str = "previous";

/// One file to materialize inside a generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallFile {
    /// Path relative to the generation root, using `/` separators.
    relative_path: String,
    /// File contents.
    bytes: Vec<u8>,
}

impl InstallFile {
    /// Build an install file from a generation-relative path and its bytes.
    pub fn new(relative_path: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            relative_path: relative_path.into(),
            bytes,
        }
    }

    /// The generation-relative path.
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    /// The file contents.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// A complete installed generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generation {
    /// Monotonically increasing generation id, starting at 1.
    pub id: u64,
    /// Directory holding the generation's materialized files.
    pub dir: PathBuf,
}

/// An install root that publishes generations atomically.
#[derive(Debug, Clone)]
pub struct InstallRoot {
    root: PathBuf,
}

impl InstallRoot {
    /// Open (creating if absent) the install root at `root`.
    pub fn open(root: &Path) -> Result<Self, InstallError> {
        let this = Self {
            root: root.to_path_buf(),
        };
        create_dir_all(&this.root)?;
        create_dir_all(&this.generations_dir())?;
        create_dir_all(&this.staging_dir())?;
        Ok(this)
    }

    /// The install root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The directory holding installed generations.
    pub fn generations_dir(&self) -> PathBuf {
        self.root.join(GENERATIONS_DIR)
    }

    /// The directory holding in-flight staged material.
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join(STAGING_DIR)
    }

    /// Ids of every complete generation, ascending.
    ///
    /// A generation directory without a [`READY_MARKER`] is skipped: it is
    /// either an install that crashed before completing or material an attacker
    /// dropped in place, and neither is something the host will serve.
    pub fn complete_generations(&self) -> Result<Vec<u64>, InstallError> {
        let dir = self.generations_dir();
        let mut ids = Vec::new();
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(ids),
            Err(source) => return Err(InstallError::Io { path: dir, source }),
        };
        for entry in entries {
            let entry = entry.map_err(|source| InstallError::Io {
                path: dir.clone(),
                source,
            })?;
            let Some(id) = entry.file_name().to_str().and_then(|n| n.parse::<u64>().ok()) else {
                continue;
            };
            if dir.join(id.to_string()).join(READY_MARKER).is_file() {
                ids.push(id);
            }
        }
        ids.sort_unstable();
        Ok(ids)
    }

    /// The generation currently in service, if any.
    pub fn current(&self) -> Result<Option<Generation>, InstallError> {
        self.resolve_pointer(CURRENT_POINTER)
    }

    /// The generation a rollback would restore, if any.
    pub fn previous(&self) -> Result<Option<Generation>, InstallError> {
        self.resolve_pointer(PREVIOUS_POINTER)
    }

    /// Materialize `files` as a new generation described by `inventory`.
    ///
    /// Relative paths are validated before anything is written, so a rejected
    /// request leaves the install root byte-identical.
    pub fn atomic_install(
        &self,
        inventory: &AuthenticatedInventory,
        files: &[InstallFile],
    ) -> Result<Generation, InstallError> {
        for file in files {
            validate_relative_path(&file.relative_path)?;
        }

        let id = self.next_generation_id()?;
        let staged = self.staging_dir().join(format!(
            "{id}.{}.{}",
            std::process::id(),
            monotonic_stamp()
        ));
        // A previous crash may have left this exact name behind; start clean.
        remove_dir_all_if_present(&staged)?;
        create_dir_all(&staged)?;

        match self.materialize(&staged, inventory, files) {
            Ok(()) => {}
            Err(e) => {
                let _ = std::fs::remove_dir_all(&staged);
                return Err(e);
            }
        }

        let target = self.generations_dir().join(id.to_string());
        remove_dir_all_if_present(&target)?;
        std::fs::rename(&staged, &target).map_err(|source| {
            let _ = std::fs::remove_dir_all(&staged);
            InstallError::Io {
                path: target.clone(),
                source,
            }
        })?;

        // Record the outgoing generation before advancing the pointer, so a
        // crash between the two writes leaves `current` unchanged rather than
        // pointing forward with no way back.
        if let Some(outgoing) = self.current()? {
            self.write_pointer(PREVIOUS_POINTER, outgoing.id)?;
        }
        self.write_pointer(CURRENT_POINTER, id)?;

        Ok(Generation { id, dir: target })
    }

    /// Restore the previous generation as current.
    ///
    /// The generation being left is recorded as the new previous, so a rollback
    /// is itself reversible.
    pub fn rollback(&self) -> Result<Generation, InstallError> {
        let restore = self
            .previous()?
            .ok_or(InstallError::NoPreviousGeneration)?;
        if let Some(outgoing) = self.current()? {
            self.write_pointer(PREVIOUS_POINTER, outgoing.id)?;
        }
        self.write_pointer(CURRENT_POINTER, restore.id)?;
        Ok(restore)
    }

    /// Remove all but the newest `keep` complete generations.
    ///
    /// The current and previous generations are never collected regardless of
    /// `keep`: dropping either would strand a running host or make rollback
    /// impossible.  Returns the removed ids, ascending.
    pub fn gc_old_generations(&self, keep: usize) -> Result<Vec<u64>, InstallError> {
        let all = self.complete_generations()?;
        let protected: Vec<u64> = [self.current()?, self.previous()?]
            .into_iter()
            .flatten()
            .map(|g| g.id)
            .collect();

        let collectable = all.len().saturating_sub(keep);
        let mut removed = Vec::new();
        for id in all.into_iter().take(collectable) {
            if protected.contains(&id) {
                continue;
            }
            let dir = self.generations_dir().join(id.to_string());
            std::fs::remove_dir_all(&dir).map_err(|source| InstallError::Io {
                path: dir.clone(),
                source,
            })?;
            removed.push(id);
        }
        Ok(removed)
    }

    /// Verify that generation `id` was installed from `inventory`.
    pub fn verify_generation(
        &self,
        id: u64,
        inventory: &AuthenticatedInventory,
    ) -> Result<(), InstallError> {
        let dir = self.generation_dir_if_complete(id)?;
        let recorded = PluginInventoryV1::load_and_verify(&dir.join(GENERATION_INVENTORY))?;
        // Both digests are already verified against their own payloads, so a
        // plain comparison here decides identity, not authenticity.
        if recorded.inventory_digest != inventory.digest() {
            return Err(InstallError::InventoryDigestMismatch {
                expected: inventory.digest().to_string(),
                actual: recorded.inventory_digest,
            });
        }
        Ok(())
    }

    /// Remove the install root and everything it contains.
    pub fn uninstall(&self) -> Result<(), InstallError> {
        remove_dir_all_if_present(&self.root)
    }

    /// Resolve a pointer file to a complete generation.
    fn resolve_pointer(&self, name: &str) -> Result<Option<Generation>, InstallError> {
        let path = self.root.join(name);
        let raw = match std::fs::read_to_string(&path) {
            Ok(raw) => raw,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(InstallError::Io { path, source }),
        };
        let Ok(id) = raw.trim().parse::<u64>() else {
            return Ok(None);
        };
        match self.generation_dir_if_complete(id) {
            Ok(dir) => Ok(Some(Generation { id, dir })),
            // A pointer naming a generation that has been collected or never
            // completed is treated as absent rather than as a live target.
            Err(InstallError::GenerationNotFound(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Return the directory of `id` when it is a complete generation.
    fn generation_dir_if_complete(&self, id: u64) -> Result<PathBuf, InstallError> {
        let dir = self.generations_dir().join(id.to_string());
        if dir.join(READY_MARKER).is_file() {
            Ok(dir)
        } else {
            Err(InstallError::GenerationNotFound(id))
        }
    }

    /// The next generation id: one past the highest id present on disk,
    /// complete or not, so an interrupted install never has its number reused.
    fn next_generation_id(&self) -> Result<u64, InstallError> {
        let dir = self.generations_dir();
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(1),
            Err(source) => return Err(InstallError::Io { path: dir, source }),
        };
        let mut highest = 0u64;
        for entry in entries {
            let entry = entry.map_err(|source| InstallError::Io {
                path: dir.clone(),
                source,
            })?;
            if let Some(id) = entry.file_name().to_str().and_then(|n| n.parse::<u64>().ok()) {
                highest = highest.max(id);
            }
        }
        Ok(highest + 1)
    }

    /// Write every file, the inventory, and finally the ready marker.
    fn materialize(
        &self,
        staged: &Path,
        inventory: &AuthenticatedInventory,
        files: &[InstallFile],
    ) -> Result<(), InstallError> {
        for file in files {
            let target = staged.join(&file.relative_path);
            if let Some(parent) = target.parent() {
                create_dir_all(parent)?;
            }
            write_file(&target, &file.bytes)?;
        }
        inventory
            .inventory()
            .publish(&staged.join(GENERATION_INVENTORY))
            .map_err(|e| InstallError::Inventory(InventoryError::Io(e)))?;
        write_file(&staged.join(READY_MARKER), inventory.digest().as_bytes())
    }

    /// Atomically replace a pointer file with `id`.
    fn write_pointer(&self, name: &str, id: u64) -> Result<(), InstallError> {
        let path = self.root.join(name);
        let tmp = self
            .root
            .join(format!(".{name}.tmp.{}", std::process::id()));
        write_file(&tmp, id.to_string().as_bytes())?;
        std::fs::rename(&tmp, &path).map_err(|source| {
            let _ = std::fs::remove_file(&tmp);
            InstallError::Io { path, source }
        })
    }
}

/// Reject an install path that is absolute, empty, or escapes the generation
/// directory.
fn validate_relative_path(relative: &str) -> Result<(), InstallError> {
    let path = Path::new(relative);
    if relative.is_empty() {
        return Err(InstallError::InvalidRelativePath(relative.to_string()));
    }
    for component in path.components() {
        match component {
            Component::Normal(_) => {}
            // `..` is rejected lexically rather than after normalization: a
            // normalized path can still traverse a symlink planted in the
            // staging tree by an earlier file in the same request.
            Component::ParentDir
            | Component::RootDir
            | Component::Prefix(_)
            | Component::CurDir => {
                return Err(InstallError::InvalidRelativePath(relative.to_string()));
            }
        }
    }
    Ok(())
}

/// A strictly increasing stamp used to make staging directory names unique.
fn monotonic_stamp() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

fn create_dir_all(path: &Path) -> Result<(), InstallError> {
    std::fs::create_dir_all(path).map_err(|source| InstallError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<(), InstallError> {
    std::fs::write(path, bytes).map_err(|source| InstallError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn remove_dir_all_if_present(path: &Path) -> Result<(), InstallError> {
    match std::fs::remove_dir_all(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(InstallError::Io {
            path: path.to_path_buf(),
            source,
        }),
    }
}
