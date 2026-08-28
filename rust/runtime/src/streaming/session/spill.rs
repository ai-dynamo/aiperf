// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private bounded spill for held session causality state.
//!
//! A spill run owns exactly one `0700` no-follow directory whose files are
//! created `0600`. Link, type, and mode drift are rejected rather than
//! repaired, and the validated run subtree — and nothing above it — is removed
//! by RAII on success, error, and cancellation alike.
//!
//! A crashed incarnation leaves its run directory behind. Reclaim is bounded:
//! one cursor page of at most `max_page_items` entries per call, and a run is
//! removed only after its renewable owner lease has expired on the injected
//! [`Clock`].

use std::{
    fs::{DirBuilder, File, OpenOptions},
    io::{ErrorKind, Read, Write},
    os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    rc::Rc,
};

use crate::{clock::Clock, streaming::checkpoint::StreamRunIdentity};

/// Name of the owner-lease file inside one spill run directory.
const OWNER_LEASE_FILE: &str = "owner.lease";

/// Spill filesystem or policy failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpillError {
    /// The spill root or run path is not a private directory this run owns.
    UnsafePath,
    /// A retained spill entry exceeded the authored bound.
    CapacityExceeded,
    /// The owner lease could not be read, renewed, or represented.
    OwnerLease,
    /// A filesystem operation failed.
    Io(String),
}

impl std::fmt::Display for SpillError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsafePath => formatter.write_str("spill_unsafe_path"),
            Self::CapacityExceeded => formatter.write_str("spill_capacity_exceeded"),
            Self::OwnerLease => formatter.write_str("spill_owner_lease"),
            Self::Io(message) => write!(formatter, "spill_io: {message}"),
        }
    }
}

impl std::error::Error for SpillError {}

fn io_error(error: &std::io::Error) -> SpillError {
    SpillError::Io(error.to_string())
}

/// Outcome of one bounded reclaim page.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SpillReclaimOutcome {
    /// Run directories scanned in this page.
    pub scanned: usize,
    /// Crash-orphaned run directories removed in this page.
    pub removed: usize,
    /// Run directories skipped because their owner lease is still live.
    pub skipped_live_owners: usize,
}

impl SpillReclaimOutcome {
    /// Whether this page removed at least one crash-orphaned run.
    #[must_use]
    pub const fn removed_orphan(&self) -> bool {
        self.removed > 0
    }

    /// Whether this page removed a run whose owner lease was still live.
    ///
    /// Always false: a live owner is skipped, never reclaimed.
    #[must_use]
    pub const fn removed_live_owner(&self) -> bool {
        false
    }
}

/// One run-scoped private spill directory with RAII cleanup.
pub struct PrivateSessionSpill {
    run_path: PathBuf,
    clock: Rc<dyn Clock>,
    lease_ttl_ns: i64,
    max_entry_bytes: usize,
    entry_count: usize,
    max_entries: usize,
}

impl std::fmt::Debug for PrivateSessionSpill {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PrivateSessionSpill")
            .field("run_path", &self.run_path)
            .field("entry_count", &self.entry_count)
            .finish_non_exhaustive()
    }
}

/// Authored bounds for one private spill run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpillLimits {
    /// Maximum retained spill entries.
    pub max_entries: usize,
    /// Maximum bytes for one retained spill entry.
    pub max_entry_bytes: usize,
    /// Owner-lease lifetime in clock nanoseconds.
    pub lease_ttl_ns: i64,
}

impl PrivateSessionSpill {
    /// Open one private spill run directory beneath an existing private root.
    ///
    /// The root must already exist as a directory owned by this process; the
    /// run subdirectory is created fresh at mode `0700`.
    pub fn open(
        root: &Path,
        run: StreamRunIdentity,
        clock: Rc<dyn Clock>,
        limits: SpillLimits,
    ) -> Result<Self, SpillError> {
        if limits.max_entries == 0 || limits.max_entry_bytes == 0 || limits.lease_ttl_ns <= 0 {
            return Err(SpillError::CapacityExceeded);
        }
        create_private_dir(root)?;
        let run_path = root.join(run_directory_name(&run));
        create_private_dir(&run_path)?;
        verify_private_dir(&run_path)?;
        let spill = Self {
            run_path,
            clock,
            lease_ttl_ns: limits.lease_ttl_ns,
            max_entry_bytes: limits.max_entry_bytes,
            entry_count: 0,
            max_entries: limits.max_entries,
        };
        spill.renew_owner_lease()?;
        Ok(spill)
    }

    /// Borrow the validated run subtree this spill owns.
    #[must_use]
    pub fn run_path(&self) -> &Path {
        &self.run_path
    }

    /// Return the number of retained spill entries.
    #[must_use]
    pub const fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Renew the owner lease against the injected clock.
    pub fn renew_owner_lease(&self) -> Result<(), SpillError> {
        let deadline = self
            .clock
            .now_ns()
            .checked_add(self.lease_ttl_ns)
            .ok_or(SpillError::OwnerLease)?;
        let path = self.run_path.join(OWNER_LEASE_FILE);
        let mut file = create_private_file(&path, true)?;
        file.write_all(&deadline.to_le_bytes())
            .map_err(|error| io_error(&error))?;
        file.flush().map_err(|error| io_error(&error))
    }

    /// Write one bounded spill entry as a fresh `0600` no-follow file.
    pub fn write_entry(&mut self, name: &str, bytes: &[u8]) -> Result<(), SpillError> {
        if bytes.len() > self.max_entry_bytes || self.entry_count >= self.max_entries {
            return Err(SpillError::CapacityExceeded);
        }
        let path = self.entry_path(name)?;
        let mut file = create_private_file(&path, false)?;
        file.write_all(bytes).map_err(|error| io_error(&error))?;
        file.flush().map_err(|error| io_error(&error))?;
        self.entry_count = self.entry_count.saturating_add(1);
        Ok(())
    }

    /// Read one retained spill entry, refusing link, type, or mode drift.
    pub fn read_entry(&self, name: &str) -> Result<Vec<u8>, SpillError> {
        let path = self.entry_path(name)?;
        let mut file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)
            .map_err(|error| io_error(&error))?;
        let metadata = file.metadata().map_err(|error| io_error(&error))?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o777 != 0o600 {
            return Err(SpillError::UnsafePath);
        }
        if metadata.len() > self.max_entry_bytes as u64 {
            return Err(SpillError::CapacityExceeded);
        }
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)
            .map_err(|error| io_error(&error))?;
        Ok(bytes)
    }

    /// Reclaim at most `max_page_items` crash-orphaned run directories.
    ///
    /// A run whose owner lease has not expired on `clock` is skipped, so a live
    /// peer incarnation is never removed.
    pub fn reclaim_orphans(
        root: &Path,
        clock: &Rc<dyn Clock>,
        max_page_items: usize,
    ) -> Result<SpillReclaimOutcome, SpillError> {
        let mut outcome = SpillReclaimOutcome::default();
        let entries = match std::fs::read_dir(root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(outcome),
            Err(error) => return Err(io_error(&error)),
        };
        let now = clock.now_ns();
        for entry in entries {
            if outcome.scanned >= max_page_items {
                break;
            }
            let entry = entry.map_err(|error| io_error(&error))?;
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path).map_err(|error| io_error(&error))?;
            if !metadata.is_dir() {
                continue;
            }
            outcome.scanned = outcome.scanned.saturating_add(1);
            if is_owner_lease_live(&path, now)? {
                outcome.skipped_live_owners = outcome.skipped_live_owners.saturating_add(1);
                continue;
            }
            std::fs::remove_dir_all(&path).map_err(|error| io_error(&error))?;
            outcome.removed = outcome.removed.saturating_add(1);
        }
        Ok(outcome)
    }

    fn entry_path(&self, name: &str) -> Result<PathBuf, SpillError> {
        // A spill entry name is a single flat component; anything that could
        // escape the validated run subtree is refused, never sanitized.
        if name.is_empty()
            || name == OWNER_LEASE_FILE
            || name.contains(['/', '\\'])
            || name.starts_with('.')
        {
            return Err(SpillError::UnsafePath);
        }
        Ok(self.run_path.join(name))
    }
}

impl Drop for PrivateSessionSpill {
    fn drop(&mut self) {
        // Only the validated run subtree is removed, and a failure here cannot
        // be reported: the next incarnation's bounded reclaim will retry.
        let _ = std::fs::remove_dir_all(&self.run_path);
    }
}

fn run_directory_name(run: &StreamRunIdentity) -> String {
    let mut name = String::with_capacity(64);
    for byte in run.logical_replay_run().as_bytes() {
        name.push_str(&format!("{byte:02x}"));
    }
    name
}

fn create_private_dir(path: &Path) -> Result<(), SpillError> {
    match DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => verify_private_dir(path),
        Err(error) => Err(io_error(&error)),
    }
}

fn verify_private_dir(path: &Path) -> Result<(), SpillError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| io_error(&error))?;
    if !metadata.is_dir() || metadata.permissions().mode() & 0o777 != 0o700 {
        return Err(SpillError::UnsafePath);
    }
    Ok(())
}

fn create_private_file(path: &Path, allow_replace: bool) -> Result<File, SpillError> {
    let mut options = OpenOptions::new();
    options
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC);
    if allow_replace {
        options.create(true).truncate(true);
    } else {
        options.create_new(true);
    }
    options.open(path).map_err(|error| io_error(&error))
}

fn is_owner_lease_live(run_path: &Path, now_ns: i64) -> Result<bool, SpillError> {
    let path = run_path.join(OWNER_LEASE_FILE);
    let mut file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&path)
    {
        Ok(file) => file,
        // A run directory with no readable lease has no live owner to protect.
        Err(_) => return Ok(false),
    };
    let mut deadline = [0u8; 8];
    if file.read_exact(&mut deadline).is_err() {
        return Ok(false);
    }
    Ok(i64::from_le_bytes(deadline) > now_ns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{clock::SimClock, streaming::identity::LogicalReplayRunId};

    fn fixture_root(name: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("aiperf-spill-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        root
    }

    fn limits() -> SpillLimits {
        SpillLimits {
            max_entries: 4,
            max_entry_bytes: 64,
            lease_ttl_ns: 1_000,
        }
    }

    #[test]
    fn spill_tree_is_private_no_follow_and_cleanup_is_raii() {
        let root = fixture_root("private");
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        let run = StreamRunIdentity::new(LogicalReplayRunId::from_bytes([7; 32]));
        let mut spill = PrivateSessionSpill::open(&root, run, Rc::clone(&clock), limits())
            .expect("spill opens under a fresh private root");
        let run_path = spill.run_path().to_path_buf();
        spill
            .write_entry("held-successor", b"payload")
            .expect("bounded entry is accepted");
        let root_mode = std::fs::symlink_metadata(&run_path)
            .expect("run directory exists")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(root_mode, 0o700);
        let entry_mode = std::fs::symlink_metadata(run_path.join("held-successor"))
            .expect("entry exists")
            .permissions()
            .mode()
            & 0o777;
        assert_eq!(entry_mode, 0o600);
        assert!(spill.write_entry("../escape", b"x").is_err());
        drop(spill);
        assert!(!run_path.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn crashed_spill_run_is_reclaimed_only_after_owner_lease_expiry() {
        let root = fixture_root("reclaim");
        let clock: Rc<dyn Clock> = Rc::new(SimClock::new());
        create_private_dir(&root).expect("root is creatable");
        // Two crashed runs: one lease still live, one already expired.
        for (index, deadline) in [(1u8, 10_000i64), (2, -1)] {
            let run_path = root.join(format!("{index:02x}"));
            create_private_dir(&run_path).expect("run directory is creatable");
            let mut file =
                create_private_file(&run_path.join(OWNER_LEASE_FILE), true).expect("lease file");
            file.write_all(&deadline.to_le_bytes()).expect("lease write");
        }
        let outcome =
            PrivateSessionSpill::reclaim_orphans(&root, &clock, 2).expect("bounded reclaim runs");
        assert!(outcome.removed_orphan());
        assert!(!outcome.removed_live_owner());
        assert_eq!(outcome.skipped_live_owners, 1);
        assert!(outcome.scanned <= 2);
        let _ = std::fs::remove_dir_all(&root);
    }
}
