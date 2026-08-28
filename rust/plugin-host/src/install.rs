// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic installation of immutable plugin generations.
//!
//! An install root holds numbered generations under `generations/`, a
//! `current` pointer, and a `previous` pointer.  A generation is materialized
//! in full under `staging/<id>` and only then renamed into `generations/<id>`,
//! so the directory a reader can reach is always complete.  The last file
//! written into a generation is its [`READY_MARKER`]; a generation without one
//! is debris from an interrupted install and is invisible to every resolver
//! here.
//!
//! Publishing is a pointer swap, not a mutation: the pointer file is written to
//! a sibling temporary and renamed, so a concurrent reader resolves either the
//! previous complete generation or the new one and never a half-built tree.
//! Rollback is the same swap in reverse, which is why the previous generation's
//! bytes are retained until garbage collection removes them.
//!
//! Installed files are made read-only.  Immutability here is a statement about
//! the host's own behavior — it never rewrites a published generation in place
//! — and the authority checks in [`crate::platform`] are what prove nobody else
//! can either.

use std::io::Read as _;
use std::path::{Component, Path, PathBuf};

use crate::error::InstallError;
use crate::inventory::{AuthenticatedInventory, validate_inventory};
use crate::platform::fs::open_no_follow;

/// Name of the file whose presence proves a generation is complete.
pub const READY_MARKER: &str = "generation.marker";

/// Name of the file recording the inventory digest a generation was installed
/// from.
pub const INVENTORY_DIGEST_FILE: &str = "inventory.digest";

/// Directory holding published generations.
const GENERATIONS_DIR: &str = "generations";

/// Directory holding in-flight generation builds.
const STAGING_DIR: &str = "staging";

/// Pointer file naming the generation readers should resolve.
const CURRENT_POINTER: &str = "current";

/// Pointer file naming the generation a rollback would restore.
const PREVIOUS_POINTER: &str = "previous";

/// Read-only mode applied to every installed file.
#[cfg(unix)]
const INSTALLED_FILE_MODE: u32 = 0o444;

/// One file to materialize into a generation.
#[derive(Debug, Clone)]
pub struct InstallFile {
    /// Path relative to the generation directory.
    pub relative_path: String,
    /// Exact bytes to write.
    pub bytes: Vec<u8>,
}

impl InstallFile {
    /// Build an install file from a relative path and its bytes.
    pub fn new(relative_path: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            relative_path: relative_path.into(),
            bytes,
        }
    }
}

/// A published generation resolved from an install root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstallGeneration {
    /// Monotonic generation id.
    pub id: u64,
    /// Directory holding the generation's files.
    pub dir: PathBuf,
}

/// The result of materializing one generation through [`install_generation`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InstalledGeneration {
    /// Generation id that was materialized.
    pub generation: u64,
    /// Directory the generation's files landed in.
    pub root: PathBuf,
}

/// A directory holding immutable plugin generations and their pointers.
#[derive(Debug, Clone)]
pub struct InstallRoot {
    root: PathBuf,
}

impl InstallRoot {
    /// Open (creating if absent) the install root at `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, InstallError> {
        let root = root.as_ref().to_path_buf();
        for dir in [
            root.clone(),
            root.join(GENERATIONS_DIR),
            root.join(STAGING_DIR),
        ] {
            create_dir_all(&dir)?;
        }
        Ok(Self { root })
    }

    /// Path of the install root itself.
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Directory holding published generations.
    pub fn generations_dir(&self) -> PathBuf {
        self.root.join(GENERATIONS_DIR)
    }

    /// Directory holding in-flight generation builds.
    pub fn staging_dir(&self) -> PathBuf {
        self.root.join(STAGING_DIR)
    }

    /// Ascending ids of every generation that carries a ready marker.
    pub fn complete_generations(&self) -> Result<Vec<u64>, InstallError> {
        let mut ids = generation_ids(&self.generations_dir(), true)?;
        ids.sort_unstable();
        Ok(ids)
    }

    /// Resolve the generation readers should use, if one is published.
    pub fn current(&self) -> Result<Option<InstallGeneration>, InstallError> {
        read_pointer(&self.root, CURRENT_POINTER)
    }

    /// Resolve the generation a rollback would restore, if one is recorded.
    pub fn previous(&self) -> Result<Option<InstallGeneration>, InstallError> {
        read_pointer(&self.root, PREVIOUS_POINTER)
    }

    /// Materialize `files` as a new immutable generation and publish it.
    ///
    /// The inventory is validated before anything is written, so an inventory
    /// the host would refuse to trust never produces bytes on disk.  Every
    /// relative path is checked before the staging directory is created, so a
    /// refused install leaves the root exactly as it found it.
    pub fn atomic_install(
        &self,
        inventory: &AuthenticatedInventory,
        files: &[InstallFile],
    ) -> Result<InstallGeneration, InstallError> {
        validate_inventory(inventory)?;
        let checked: Vec<(PathBuf, &[u8])> = files
            .iter()
            .map(|f| checked_relative_path(&f.relative_path).map(|p| (p, f.bytes.as_slice())))
            .collect::<Result<_, _>>()?;

        let id = self.next_generation_id()?;
        let digest = inventory.canonical_digest();
        let dir = materialize(&self.root, id, &checked, Some(&digest))?;

        // Record the outgoing generation before the swap: a crash between the
        // two writes leaves `previous` naming the generation that is still
        // current, which resolves to a redundant rollback rather than a
        // dangling one.
        if let Some(outgoing) = self.current()? {
            write_pointer(&self.root, PREVIOUS_POINTER, &outgoing.dir)?;
        }
        write_pointer(&self.root, CURRENT_POINTER, &dir)?;
        Ok(InstallGeneration { id, dir })
    }

    /// Restore the previous generation, making the outgoing one the new
    /// rollback target.
    pub fn rollback(&self) -> Result<InstallGeneration, InstallError> {
        let previous = self.previous()?.ok_or(InstallError::NoPreviousGeneration)?;
        let outgoing = self.current()?;
        write_pointer(&self.root, CURRENT_POINTER, &previous.dir)?;
        match outgoing {
            Some(outgoing) if outgoing.id != previous.id => {
                write_pointer(&self.root, PREVIOUS_POINTER, &outgoing.dir)?;
            }
            // Rolling back onto the generation already current leaves no
            // distinct target, so the pointer is cleared rather than made
            // self-referential.
            _ => remove_pointer(&self.root, PREVIOUS_POINTER)?,
        }
        Ok(previous)
    }

    /// Prove that generation `id` was installed from `inventory`.
    pub fn verify_generation(
        &self,
        id: u64,
        inventory: &AuthenticatedInventory,
    ) -> Result<(), InstallError> {
        let dir = self.generations_dir().join(id.to_string());
        if !dir.join(READY_MARKER).is_file() {
            return Err(InstallError::GenerationNotFound(id));
        }
        let digest_path = dir.join(INVENTORY_DIGEST_FILE);
        let actual = match std::fs::read_to_string(&digest_path) {
            Ok(text) => text.trim().to_string(),
            Err(_) => return Err(InstallError::IncompleteGeneration(id)),
        };
        let expected = inventory.canonical_digest();
        if actual == expected {
            Ok(())
        } else {
            Err(InstallError::InventoryDigestMismatch { expected, actual })
        }
    }

    /// Remove all but the newest `keep` generations.
    ///
    /// The current and previous generations are retained regardless of `keep`:
    /// collecting either would break a live reader or make rollback impossible.
    /// Returns the ids removed, ascending.
    pub fn gc_old_generations(&self, keep: usize) -> Result<Vec<u64>, InstallError> {
        let ids = self.complete_generations()?;
        let retained_tail: Vec<u64> = ids.iter().rev().take(keep).copied().collect();
        let current = self.current()?.map(|g| g.id);
        let previous = self.previous()?.map(|g| g.id);

        let mut removed = Vec::new();
        for id in ids {
            if retained_tail.contains(&id) || current == Some(id) || previous == Some(id) {
                continue;
            }
            remove_generation(&self.generations_dir(), id)?;
            removed.push(id);
        }
        Ok(removed)
    }

    /// Remove the entire install root, including every generation and pointer.
    pub fn uninstall(&self) -> Result<(), InstallError> {
        remove_dir_all(&self.root)
    }

    /// Next unused generation id.
    ///
    /// Incomplete generations count: reusing the id of an interrupted install
    /// would let its debris masquerade as the new generation's files.
    fn next_generation_id(&self) -> Result<u64, InstallError> {
        let published = generation_ids(&self.generations_dir(), false)?;
        let staged = generation_ids(&self.staging_dir(), false)?;
        Ok(published
            .into_iter()
            .chain(staged)
            .max()
            .unwrap_or(0)
            .saturating_add(1))
    }
}

/// Materialize `artifacts` as generation `generation` under `root`.
///
/// This is the pointer-free half of an install: the generation lands complete
/// and marked ready, but nothing yet resolves to it.  Use
/// [`rollback_to_generation`] to publish it.
///
/// # Warning: caller owns id uniqueness
///
/// If `generation` reuses an id of a previously materialized generation this
/// call silently removes the old directory before writing the new one, so any
/// reader that resolved the old generation may observe an empty or partial tree.
/// Callers that need monotonic ids should query existing generation directories
/// (or use [`InstallRoot::atomic_install`], which handles id allocation) before
/// calling this function.
pub fn install_generation(
    root: &Path,
    generation: u64,
    artifacts: &[(String, &[u8])],
) -> Result<InstalledGeneration, InstallError> {
    create_dir_all(&root.join(GENERATIONS_DIR))?;
    create_dir_all(&root.join(STAGING_DIR))?;
    let checked: Vec<(PathBuf, &[u8])> = artifacts
        .iter()
        .map(|(path, bytes)| checked_relative_path(path).map(|p| (p, *bytes)))
        .collect::<Result<_, _>>()?;
    let dir = materialize(root, generation, &checked, None)?;
    Ok(InstalledGeneration {
        generation,
        root: dir,
    })
}

/// Point `root`'s current pointer at `generation`.
///
/// Fails closed when the target is absent or was never completed, so a rollback
/// can never publish a generation whose bytes are not fully on disk.
pub fn rollback_to_generation(root: &Path, generation: u64) -> Result<(), InstallError> {
    let dir = root.join(GENERATIONS_DIR).join(generation.to_string());
    if !dir.join(READY_MARKER).is_file() {
        return Err(InstallError::GenerationNotFound(generation));
    }
    let outgoing = read_pointer(root, CURRENT_POINTER)?;
    if let Some(outgoing) = outgoing
        && outgoing.id != generation
    {
        write_pointer(root, PREVIOUS_POINTER, &outgoing.dir)?;
    }
    write_pointer(root, CURRENT_POINTER, &dir)
}

/// Remove every complete generation under `root` that is not in `keep`.
///
/// The current generation is retained even when `keep` omits it.
pub fn gc_generations(root: &Path, keep: &[u64]) -> Result<Vec<u64>, InstallError> {
    let generations_dir = root.join(GENERATIONS_DIR);
    let mut ids = generation_ids(&generations_dir, true)?;
    ids.sort_unstable();
    let current = read_pointer(root, CURRENT_POINTER)?.map(|g| g.id);

    let mut removed = Vec::new();
    for id in ids {
        if keep.contains(&id) || current == Some(id) {
            continue;
        }
        remove_generation(&generations_dir, id)?;
        removed.push(id);
    }
    Ok(removed)
}

/// Build generation `id` under `root/staging/<id>` and rename it into place.
fn materialize(
    root: &Path,
    id: u64,
    files: &[(PathBuf, &[u8])],
    inventory_digest: Option<&str>,
) -> Result<PathBuf, InstallError> {
    let staging = root.join(STAGING_DIR).join(id.to_string());
    let target = root.join(GENERATIONS_DIR).join(id.to_string());
    if staging.exists() {
        remove_dir_all(&staging)?;
    }
    if target.exists() {
        remove_dir_all(&target)?;
    }
    create_dir_all(&staging)?;

    for (relative, bytes) in files {
        let path = staging.join(relative);
        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        write_file(&path, bytes)?;
        set_read_only(&path)?;
    }
    if let Some(digest) = inventory_digest {
        let digest_path = staging.join(INVENTORY_DIGEST_FILE);
        write_file(&digest_path, digest.as_bytes())?;
        set_read_only(&digest_path)?;
    }
    // The marker is written last: its presence is the proof that every other
    // file in this generation is already on disk.
    let marker_path = staging.join(READY_MARKER);
    write_file(&marker_path, format!("{id}\n").as_bytes())?;
    set_read_only(&marker_path)?;

    std::fs::rename(&staging, &target).map_err(|source| InstallError::Io {
        path: target.clone(),
        source,
    })?;
    Ok(target)
}

/// Validate that `raw` is a relative path confined to the generation directory.
fn checked_relative_path(raw: &str) -> Result<PathBuf, InstallError> {
    let path = Path::new(raw);
    if raw.is_empty() || path.is_absolute() {
        return Err(InstallError::InvalidRelativePath(raw.to_string()));
    }
    // Only plain components are accepted; `..`, a root, or a Windows prefix
    // would all let a published generation write outside its own directory.
    if !path.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(InstallError::InvalidRelativePath(raw.to_string()));
    }
    Ok(path.to_path_buf())
}

/// Ids of the generation directories under `dir`.
fn generation_ids(dir: &Path, require_marker: bool) -> Result<Vec<u64>, InstallError> {
    let read = match std::fs::read_dir(dir) {
        Ok(read) => read,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(source) => {
            return Err(InstallError::Io {
                path: dir.to_path_buf(),
                source,
            });
        }
    };
    let mut ids = Vec::new();
    for entry in read {
        let entry = entry.map_err(|source| InstallError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        let Some(id) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u64>().ok())
        else {
            continue;
        };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if require_marker && !path.join(READY_MARKER).is_file() {
            tracing::debug!(
                path = %path.display(),
                "ignoring plugin generation with no ready marker"
            );
            continue;
        }
        ids.push(id);
    }
    Ok(ids)
}

/// Resolve a pointer file to the complete generation it names.
fn read_pointer(root: &Path, name: &str) -> Result<Option<InstallGeneration>, InstallError> {
    let pointer = root.join(name);
    // Open with O_NOFOLLOW so a racing symlink at the pointer path cannot
    // redirect the resolver to an arbitrary location.
    let text = match open_no_follow(&pointer) {
        Ok(mut f) => {
            let mut buf = String::new();
            f.read_to_string(&mut buf)
                .map_err(|source| InstallError::Io {
                    path: pointer.clone(),
                    source,
                })?;
            buf
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(source) => {
            return Err(InstallError::Io {
                path: pointer,
                source,
            });
        }
    };
    let dir = PathBuf::from(text.trim());
    let Some(id) = dir
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.parse::<u64>().ok())
    else {
        return Ok(None);
    };
    // A pointer naming a collected or incomplete generation resolves to
    // nothing rather than to a directory a caller would then try to load.
    if !dir.join(READY_MARKER).is_file() {
        return Ok(None);
    }
    Ok(Some(InstallGeneration { id, dir }))
}

/// Atomically point `name` at `dir`.
fn write_pointer(root: &Path, name: &str, dir: &Path) -> Result<(), InstallError> {
    let pointer = root.join(name);
    let temp = root.join(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos()
    ));
    write_file(&temp, format!("{}\n", dir.display()).as_bytes())?;
    std::fs::rename(&temp, &pointer).map_err(|source| {
        let _ = std::fs::remove_file(&temp);
        InstallError::Io {
            path: pointer,
            source,
        }
    })
}

/// Remove a pointer file, tolerating its absence.
fn remove_pointer(root: &Path, name: &str) -> Result<(), InstallError> {
    let pointer = root.join(name);
    match std::fs::remove_file(&pointer) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(InstallError::Io {
            path: pointer,
            source,
        }),
    }
}

/// Remove one generation directory, tolerating its absence.
fn remove_generation(generations_dir: &Path, id: u64) -> Result<(), InstallError> {
    let dir = generations_dir.join(id.to_string());
    if dir.exists() {
        remove_dir_all(&dir)?;
    }
    Ok(())
}

fn create_dir_all(path: &Path) -> Result<(), InstallError> {
    std::fs::create_dir_all(path).map_err(|source| InstallError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn remove_dir_all(path: &Path) -> Result<(), InstallError> {
    std::fs::remove_dir_all(path).map_err(|source| InstallError::Io {
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

/// Drop every write bit from an installed file.
#[cfg(unix)]
fn set_read_only(path: &Path) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(INSTALLED_FILE_MODE)).map_err(
        |source| InstallError::Io {
            path: path.to_path_buf(),
            source,
        },
    )
}

/// Windows marks the file read-only through its attribute rather than a mode.
#[cfg(not(unix))]
fn set_read_only(path: &Path) -> Result<(), InstallError> {
    let mut perms = std::fs::metadata(path)
        .map_err(|source| InstallError::Io {
            path: path.to_path_buf(),
            source,
        })?
        .permissions();
    perms.set_readonly(true);
    std::fs::set_permissions(path, perms).map_err(|source| InstallError::Io {
        path: path.to_path_buf(),
        source,
    })
}
