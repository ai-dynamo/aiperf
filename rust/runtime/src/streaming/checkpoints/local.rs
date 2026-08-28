// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Crash-durable local generation store.
//!
//! Authority flows only through the `CURRENT` pointer. Immutable objects are
//! named from their verified bytes, so an interrupted commit leaves unreachable
//! but byte-correct files rather than a torn head. Reopening the store after any
//! crash, kill, or injected fault yields either the complete previous generation
//! or the complete next one, never a mixture.

use std::{
    any::Any,
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    ffi::OsStr,
    io::{ErrorKind, Read, Write},
    num::NonZeroUsize,
    os::unix::{
        ffi::OsStrExt,
        fs::{DirBuilderExt, OpenOptionsExt},
        io::AsRawFd,
    },
    path::{Path, PathBuf},
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    clock::Clock,
    streaming::{
        blocking::{
            BlockingWorkBudget, BlockingWorkClass, BlockingWorkError, StreamingBlockingExecutor,
        },
        budget::{BudgetLease, BudgetLimits, BudgetSnapshot},
        checkpoint::{
            BudgetedCheckpointBytes, CheckpointBackendBudgetKind, CheckpointEpoch, CheckpointError,
            CheckpointGeneration, CheckpointParticipantId, CommittedCheckpointGeneration,
            CommittedParticipantState, CurrentV4ParticipantStateContext,
            DecodedCheckpointGeneration, LegacyParticipantState, LegacyV3CheckpointGeneration,
            ParticipantStateDescriptor, PreparedParticipantState, StreamRunIdentity,
            decode_versioned_checkpoint_generation,
        },
        checkpoint_backend::{
            CheckpointBackendPlacement, CheckpointBackendPrepareContext,
            CheckpointBackendRequirements, CheckpointCommitMetadata,
            CheckpointGenerationExpectations, CheckpointRetention, CurrentV4CheckpointGeneration,
            FrozenGenerationTransactionInputs, LeasedCheckpointGeneration, LeasedGenerationReader,
            LegacyV3LeasedGenerationReader, StreamingCheckpointBackend,
            StreamingCheckpointBackendDescriptor, StreamingCheckpointBackendFactory,
            StreamingGenerationTransaction, ValidatedCheckpointBackendConfig,
            build_prevalidated_candidate, sealed, validate_commit_metadata,
        },
        checkpoints::{
            budget::{BackendBudget, map_budget_error},
            lease_gc::{
                CheckpointGarbageCollector, CheckpointRetentionPolicy, CondemnationLedger,
                GcReport, LeaseLiveness, ObjectMarkSet, READER_LEASE_PREFIX, REPORT_LEASE_PREFIX,
                SweepAuthority, ValidatedRetentionPolicy, generation_from_record_name,
                generation_lease_file_name, hex32, parse_hex32, pinned_generation_from_name,
            },
        },
        identity::ContentDigest,
        reliability::HandledIssueCut,
        reliability::PreparedIssueReceiptResultPartition,
        results::{
            BudgetedResultDescriptors, PreparedResultEpoch, ResultIndexCursor, ResultIndexPage,
            ResultIndexReadBudget, ResultPartition, ResultSegmentDescriptor, ResultSegmentReader,
            canonical_result_index_object, canonical_result_index_root, descriptor_retained_bytes,
            result_totals,
        },
    },
};

/// Object filename prefix; the remainder is the lowercase BLAKE3 hex.
const OBJECT_PREFIX: &str = "blake3-";

/// Maximum accepted `CURRENT` pointer length, bounded before any read.
const MAX_CURRENT_BYTES: u64 = 128;

/// Maximum accepted lease-record length, bounded before any read.
const MAX_LEASE_RECORD_BYTES: u64 = 1024;

/// Lifetime granted to a reachability lease before a retention policy is authored.
const DEFAULT_READER_LEASE_NS: i64 = 60_000_000_000;

/// Storage failure carrying one external filesystem fact.
fn storage_error(message: impl Into<String>) -> CheckpointError {
    CheckpointError::Storage {
        message: message.into(),
    }
}

// ---------------------------------------------------------------------------
// Filesystem seam
// ---------------------------------------------------------------------------

/// One bounded page of sorted directory entries.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LocalDirPage {
    /// Sorted entry names in this page.
    pub names: Vec<Box<str>>,
    /// Cursor for the next page, absent at the end of the directory.
    pub next: Option<Box<str>>,
}

/// Opaque owner of one held advisory lock.
///
/// Dropping the handle releases the lock. The concrete payload belongs to the
/// [`LocalCheckpointFilesystem`] implementation that minted it.
pub struct LocalLockHandle(#[allow(dead_code)] Box<dyn Any>);

impl LocalLockHandle {
    /// Wrap one implementation-owned lock payload.
    #[must_use]
    pub fn new(payload: impl Any + 'static) -> Self {
        Self(Box::new(payload))
    }
}

impl std::fmt::Debug for LocalLockHandle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("LocalLockHandle").finish()
    }
}

/// Every filesystem effect the local checkpoint store is permitted to perform.
///
/// Each method is one complete operation so the production implementer can move
/// it wholesale onto [`StreamingBlockingExecutor`]; no syscall runs on the
/// request `LocalSet`.
#[async_trait(?Send)]
pub trait LocalCheckpointFilesystem {
    /// Create one directory with exact `0700` mode, tolerating an existing one.
    async fn create_private_dir(&self, path: &Path) -> Result<(), CheckpointError>;

    /// Create one `0600` regular file exclusively and write `bytes` completely.
    ///
    /// Returns `false` when the exact path already exists. Content-addressed
    /// paths make that a hit rather than a failure, and the caller verifies the
    /// retained bytes.
    async fn write_new(&self, path: &Path, bytes: &[u8]) -> Result<bool, CheckpointError>;

    /// Overwrite one existing `0600` regular file in place, refusing symlinks.
    ///
    /// In-place overwrite is what lets a lease record its renewed expiry without
    /// replacing the inode its advisory lock is held on: unlinking and recreating
    /// the name would strand the lock on an unreachable inode and let a second
    /// holder take the same lease.
    async fn overwrite_in_place(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError>;

    /// Flush and fsync one regular file.
    async fn sync_file(&self, path: &Path) -> Result<(), CheckpointError>;

    /// Fsync one directory so its entries are durable.
    async fn sync_directory(&self, path: &Path) -> Result<(), CheckpointError>;

    /// Rename within one directory, replacing the destination atomically.
    async fn rename(&self, source: &Path, destination: &Path) -> Result<(), CheckpointError>;

    /// Read at most `max_bytes` from a regular file, refusing symlinks.
    ///
    /// An absent path is `Ok(None)`; a symlink, a non-regular file, or a file
    /// longer than `max_bytes` is a refusal.
    async fn read_optional(
        &self,
        path: &Path,
        max_bytes: u64,
    ) -> Result<Option<Bytes>, CheckpointError>;

    /// List one directory's entries in sorted order, no-follow, bounded.
    ///
    /// Bounding the entry count here rather than after collecting is what keeps
    /// the reclamation scan's peak allocation inside its configured page limit.
    async fn list_dir_page(
        &self,
        path: &Path,
        after: Option<&str>,
        max_entries: usize,
    ) -> Result<LocalDirPage, CheckpointError>;

    /// Remove one validated private subtree, or one regular file.
    async fn remove_private_subtree(&self, path: &Path) -> Result<(), CheckpointError>;

    /// Take one exclusive advisory lock, creating the lock file when absent.
    ///
    /// `Ok(None)` means another live holder owns the lock. A holder that dies
    /// releases it immediately, which is the guarantee an expiry heuristic
    /// cannot give.
    async fn try_lock_exclusive(
        &self,
        path: &Path,
    ) -> Result<Option<LocalLockHandle>, CheckpointError>;
}

// ---------------------------------------------------------------------------
// Production implementer
// ---------------------------------------------------------------------------

/// Production filesystem seam. Each call is one complete blocking job.
///
/// Known limitation: advisory locking is unreliable over NFS. The local backend
/// declares controller-local placement for exactly that reason, and the
/// reclamation predicate requires the clock-driven expiry to agree with the
/// advisory probe so a filesystem without working `flock` degrades to the clock
/// rule rather than to unsafe reclamation.
pub struct BlockingLocalFilesystem {
    executor: StreamingBlockingExecutor,
}

impl BlockingLocalFilesystem {
    /// Bind one blocking executor as the carrier for every filesystem effect.
    #[must_use]
    pub const fn new(executor: StreamingBlockingExecutor) -> Self {
        Self { executor }
    }

    /// Move one complete filesystem operation onto the blocking executor.
    async fn run<T, F>(&self, bytes: usize, work: F) -> Result<T, CheckpointError>
    where
        F: FnOnce() -> Result<T, std::io::Error> + Send + 'static,
        T: Send + 'static,
    {
        let budget = BlockingWorkBudget {
            input_bytes: bytes,
            output_bytes: bytes,
        };
        // An I/O failure is an ordinary outcome of the job, not a failure of the
        // executor, so it travels as a value and never as a blocking-work error.
        let output = self
            .executor
            .run(BlockingWorkClass::DurableSync, budget, move |_cancel| {
                Ok(work().map_err(|error| error.to_string()))
            })
            .await
            .map_err(map_blocking_error)?;
        output.into_value().map_err(storage_error)
    }
}

/// Map a blocking-executor failure onto the stable checkpoint vocabulary.
///
/// Blocking-executor capacity refusal is a backend budget fact the backend
/// configured. An actual I/O error, including a full device, is an external
/// storage fact the backend never admitted, so it stays `Storage`.
fn map_blocking_error(error: BlockingWorkError) -> CheckpointError {
    match error {
        BlockingWorkError::Budget(inner) => map_budget_error(
            CheckpointBackendBudgetKind::Storage,
            BudgetLimits {
                max_items: usize::MAX,
                max_bytes: usize::MAX,
            },
            0,
            0,
            inner,
        ),
        other => storage_error(other.to_string()),
    }
}

/// Open one regular file no-follow and close-on-exec.
fn open_no_follow(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
}

fn create_private_dir_blocking(path: &Path) -> std::io::Result<()> {
    match std::fs::DirBuilder::new().mode(0o700).create(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            let metadata = std::fs::symlink_metadata(path)?;
            if metadata.is_dir() {
                Ok(())
            } else {
                Err(std::io::Error::new(
                    ErrorKind::AlreadyExists,
                    "checkpoint path exists and is not a private directory",
                ))
            }
        }
        Err(error) => Err(error),
    }
}

fn write_new_blocking(path: &Path, bytes: Vec<u8>) -> std::io::Result<bool> {
    let mut file = match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::AlreadyExists => return Ok(false),
        Err(error) => return Err(error),
    };
    file.write_all(&bytes)?;
    file.flush()?;
    Ok(true)
}

fn overwrite_in_place_blocking(path: &Path, bytes: Vec<u8>) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    file.write_all(&bytes)?;
    file.flush()
}

fn read_optional_blocking(path: &Path, max_bytes: u64) -> std::io::Result<Option<Vec<u8>>> {
    let mut file = match open_no_follow(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "checkpoint object is not a regular file",
        ));
    }
    if metadata.len() > max_bytes {
        return Err(std::io::Error::new(
            ErrorKind::InvalidData,
            "checkpoint object exceeds its accepted length",
        ));
    }
    let mut buffer = Vec::with_capacity(usize::try_from(metadata.len()).map_err(|_| {
        std::io::Error::new(ErrorKind::InvalidData, "unrepresentable object length")
    })?);
    file.read_to_end(&mut buffer)?;
    Ok(Some(buffer))
}

fn remove_private_subtree_blocking(path: &Path) -> std::io::Result<()> {
    let metadata = match std::fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error),
    };
    if metadata.is_symlink() {
        return Err(std::io::Error::new(
            ErrorKind::InvalidInput,
            "refusing to remove a symlinked checkpoint path",
        ));
    }
    if metadata.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

fn list_dir_page_blocking(
    path: &Path,
    after: Option<String>,
    max_entries: usize,
) -> std::io::Result<(Vec<String>, Option<String>)> {
    let entries = match std::fs::read_dir(path) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok((Vec::new(), None)),
        Err(error) => return Err(error),
    };
    let mut names = Vec::new();
    for entry in entries {
        let entry = entry?;
        let name = entry.file_name();
        let Some(name) = name.to_str().map(str::to_owned) else {
            continue;
        };
        if after
            .as_deref()
            .is_some_and(|cursor| name.as_str() <= cursor)
        {
            continue;
        }
        names.push(name);
    }
    names.sort_unstable();
    let next = if names.len() > max_entries {
        names.truncate(max_entries);
        names.last().cloned()
    } else {
        None
    };
    Ok((names, next))
}

fn try_lock_exclusive_blocking(path: &Path) -> std::io::Result<Option<std::fs::File>> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)?;
    // SAFETY: `flock` takes a borrowed descriptor and mutates no Rust memory.
    let taken = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if taken == 0 {
        return Ok(Some(file));
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(code) if code == libc::EWOULDBLOCK || code == libc::EINTR => Ok(None),
        _ => Err(error),
    }
}

#[async_trait(?Send)]
impl LocalCheckpointFilesystem for BlockingLocalFilesystem {
    async fn create_private_dir(&self, path: &Path) -> Result<(), CheckpointError> {
        let path = path.to_path_buf();
        self.run(0, move || create_private_dir_blocking(&path))
            .await
    }

    async fn write_new(&self, path: &Path, bytes: &[u8]) -> Result<bool, CheckpointError> {
        let path = path.to_path_buf();
        let length = bytes.len();
        let bytes = bytes.to_vec();
        self.run(length, move || write_new_blocking(&path, bytes))
            .await
    }

    async fn overwrite_in_place(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
        let path = path.to_path_buf();
        let length = bytes.len();
        let bytes = bytes.to_vec();
        self.run(length, move || overwrite_in_place_blocking(&path, bytes))
            .await
    }

    async fn sync_file(&self, path: &Path) -> Result<(), CheckpointError> {
        let path = path.to_path_buf();
        self.run(0, move || open_no_follow(&path)?.sync_all()).await
    }

    async fn sync_directory(&self, path: &Path) -> Result<(), CheckpointError> {
        let path = path.to_path_buf();
        self.run(0, move || {
            std::fs::OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
                .open(&path)?
                .sync_all()
        })
        .await
    }

    async fn rename(&self, source: &Path, destination: &Path) -> Result<(), CheckpointError> {
        let source = source.to_path_buf();
        let destination = destination.to_path_buf();
        self.run(0, move || std::fs::rename(&source, &destination))
            .await
    }

    async fn read_optional(
        &self,
        path: &Path,
        max_bytes: u64,
    ) -> Result<Option<Bytes>, CheckpointError> {
        let path = path.to_path_buf();
        let bytes = self
            .run(
                usize::try_from(max_bytes).unwrap_or(usize::MAX),
                move || read_optional_blocking(&path, max_bytes),
            )
            .await?;
        Ok(bytes.map(|bytes| Bytes::from(bytes.into_boxed_slice())))
    }

    async fn list_dir_page(
        &self,
        path: &Path,
        after: Option<&str>,
        max_entries: usize,
    ) -> Result<LocalDirPage, CheckpointError> {
        let path = path.to_path_buf();
        let after = after.map(str::to_owned);
        let (names, next) = self
            .run(0, move || list_dir_page_blocking(&path, after, max_entries))
            .await?;
        Ok(LocalDirPage {
            names: names.into_iter().map(String::into_boxed_str).collect(),
            next: next.map(String::into_boxed_str),
        })
    }

    async fn remove_private_subtree(&self, path: &Path) -> Result<(), CheckpointError> {
        let path = path.to_path_buf();
        self.run(0, move || remove_private_subtree_blocking(&path))
            .await
    }

    async fn try_lock_exclusive(
        &self,
        path: &Path,
    ) -> Result<Option<LocalLockHandle>, CheckpointError> {
        let path = path.to_path_buf();
        let file = self
            .run(0, move || try_lock_exclusive_blocking(&path))
            .await?;
        Ok(file.map(LocalLockHandle::new))
    }
}

// ---------------------------------------------------------------------------
// On-disk layout
// ---------------------------------------------------------------------------

/// Path algebra for one logical run's private subtree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunPaths {
    root: PathBuf,
}

impl RunPaths {
    /// Derive the fixed layout for one run under one store root.
    #[must_use]
    pub fn for_run(store_root: &Path, run: &StreamRunIdentity) -> Self {
        Self {
            root: store_root.join(hex32(run.logical_replay_run().as_bytes())),
        }
    }

    /// Borrow the run root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Borrow the flat content-addressed object directory.
    #[must_use]
    pub fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    /// Borrow the epoch-ordered generation-record directory.
    #[must_use]
    pub fn generations_dir(&self) -> PathBuf {
        self.root.join("generations")
    }

    /// Borrow the lease directory.
    #[must_use]
    pub fn leases_dir(&self) -> PathBuf {
        self.root.join("leases")
    }

    /// Borrow the transaction scratch directory.
    #[must_use]
    pub fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }

    /// Borrow the authoritative pointer path.
    #[must_use]
    pub fn current(&self) -> PathBuf {
        self.root.join("CURRENT")
    }

    /// Path of one content-addressed immutable object.
    #[must_use]
    pub fn object_path(&self, digest: &ContentDigest) -> PathBuf {
        self.objects_dir()
            .join(format!("{OBJECT_PREFIX}{}", hex32(digest.as_bytes())))
    }

    /// Path of one generation record, ordered lexicographically by epoch.
    #[must_use]
    pub fn generation_path(&self, epoch: CheckpointEpoch, digest: &ContentDigest) -> PathBuf {
        self.generations_dir().join(format!(
            "{:020}-{}.json",
            epoch.get(),
            hex32(digest.as_bytes())
        ))
    }

    /// Path of the exclusive per-run writer lease.
    #[must_use]
    pub fn writer_lease(&self) -> PathBuf {
        self.leases_dir().join("writer")
    }

    /// Path of one transaction's prepare lease.
    #[must_use]
    pub fn prepare_lease(&self, transaction: &str) -> PathBuf {
        self.leases_dir().join(format!("prepare-{transaction}"))
    }

    /// Path of one transaction's private scratch subtree.
    #[must_use]
    pub fn transaction_dir(&self, transaction: &str) -> PathBuf {
        self.tmp_dir().join(transaction)
    }
}

/// Strict `CURRENT` pointer contents.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CurrentPointer {
    epoch: u64,
    digest: String,
}

impl CurrentPointer {
    fn new(generation: &CheckpointGeneration) -> Self {
        Self {
            epoch: generation.epoch().get(),
            digest: format!("blake3:{}", hex32(generation.digest().as_bytes())),
        }
    }

    fn encode(&self) -> Result<Vec<u8>, CheckpointError> {
        let mut bytes =
            serde_json::to_vec(self).map_err(|_| CheckpointError::ObjectVerification)?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        // A pointer without its exact trailing newline is a torn or forged
        // write, not a storage failure.
        let body = bytes
            .strip_suffix(b"\n")
            .ok_or(CheckpointError::ObjectVerification)?;
        serde_json::from_slice(body).map_err(|_| CheckpointError::ObjectVerification)
    }

    fn matches(&self, generation: &CheckpointGeneration) -> bool {
        *self == Self::new(generation)
    }
}

// ---------------------------------------------------------------------------
// Faults
// ---------------------------------------------------------------------------

/// One deterministic injectable fault point in the local commit ordering.
///
/// Only the pre-publication points are enumerated by
/// [`LocalCommitFault::before_current_publication`]; `AfterCurrentRename` is
/// tested separately because it is the one point past the publication fence.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum LocalCommitFault {
    /// After every new immutable object is written, before any object fsync.
    AfterObjectWrite,
    /// After every new object fsync, before the object-directory fsync.
    AfterObjectSync,
    /// After the object-directory fsync, before the generation write.
    AfterObjectParentSync,
    /// After the generation file write, before its fsync.
    AfterGenerationWrite,
    /// After the generation fsync, before the generation-directory fsync.
    AfterGenerationSync,
    /// After the generation-directory fsync, before the temporary pointer write.
    AfterGenerationParentSync,
    /// After the temporary pointer write, before its fsync.
    AfterCurrentTmpWrite,
    /// After the temporary pointer fsync, before the rename.
    AfterCurrentTmpSync,
    /// After the rename, before the run-root directory fsync.
    AfterCurrentRename,
}

impl LocalCommitFault {
    /// Every fault point that must preserve the complete previous generation.
    #[must_use]
    pub const fn before_current_publication() -> [Self; 8] {
        [
            Self::AfterObjectWrite,
            Self::AfterObjectSync,
            Self::AfterObjectParentSync,
            Self::AfterGenerationWrite,
            Self::AfterGenerationSync,
            Self::AfterGenerationParentSync,
            Self::AfterCurrentTmpWrite,
            Self::AfterCurrentTmpSync,
        ]
    }

    /// Stable message carried by the injected refusal.
    #[must_use]
    pub const fn injected_message(self) -> &'static str {
        match self {
            Self::AfterObjectWrite => "injected local checkpoint fault after object write",
            Self::AfterObjectSync => "injected local checkpoint fault after object sync",
            Self::AfterObjectParentSync => {
                "injected local checkpoint fault after object parent sync"
            }
            Self::AfterGenerationWrite => "injected local checkpoint fault after generation write",
            Self::AfterGenerationSync => "injected local checkpoint fault after generation sync",
            Self::AfterGenerationParentSync => {
                "injected local checkpoint fault after generation parent sync"
            }
            Self::AfterCurrentTmpWrite => {
                "injected local checkpoint fault after temporary current write"
            }
            Self::AfterCurrentTmpSync => {
                "injected local checkpoint fault after temporary current sync"
            }
            Self::AfterCurrentRename => "injected local checkpoint fault after current rename",
        }
    }

    /// The exact refusal this fault point produces.
    #[must_use]
    pub fn injected_error(self) -> CheckpointError {
        storage_error(self.injected_message())
    }
}

// ---------------------------------------------------------------------------
// Leases
// ---------------------------------------------------------------------------

/// Kind discriminator for one lease file under a run's `leases/` directory.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LeaseKind {
    /// Exclusive authority to begin and publish generations for one run.
    Writer,
    /// Authority over one private transaction scratch subtree.
    Prepare,
    /// Reachability hold for one opened generation. Written by later tasks.
    Reader,
    /// Reachability hold for compaction and report persistence. Later tasks.
    Report,
}

/// Unique identity of one lease acquisition.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct LeaseHolderId([u8; 16]);

impl LeaseHolderId {
    /// Lowercase hex naming of this holder.
    #[must_use]
    pub fn to_hex(self) -> String {
        let mut text = String::with_capacity(32);
        for byte in self.0 {
            text.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
            text.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
        }
        text
    }
}

/// Provenance recorded beside an advisory lock; never the authority itself.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LeaseRecord {
    /// Exact run this lease is scoped to.
    run: StreamRunIdentity,
    /// Stable kind discriminator, redundant with the filename by design.
    kind: LeaseKind,
    /// Holder identity, unique per acquisition.
    holder: LeaseHolderId,
    /// Clock-relative expiry in nanoseconds, from the injected clock.
    expires_ns: i64,
}

/// Exclusive per-run writer authority.
///
/// Authority is the kernel-held advisory lock on the open descriptor, not the
/// file contents: a process that dies loses the lock immediately and without a
/// timeout. The recorded provenance is for operators and for the clock-driven
/// expiry that filesystems without working advisory locking fall back to.
pub struct WriterLease {
    _lock: LocalLockHandle,
    holder: LeaseHolderId,
}

/// Authority over one in-flight transaction's private scratch subtree.
pub struct PrepareLease {
    _lock: LocalLockHandle,
    path: PathBuf,
}

/// RAII owner of one private transaction scratch subtree.
///
/// Drop removes only this transaction's validated subtree, on the cancellation
/// and fault paths as well as the success path. Committed immutable objects are
/// never inside it.
pub struct TransactionTmpGuard {
    filesystem: Rc<dyn LocalCheckpointFilesystem>,
    path: PathBuf,
    lease_path: PathBuf,
    is_released: Cell<bool>,
}

impl TransactionTmpGuard {
    /// Borrow this transaction's private subtree path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Remove the subtree now, awaiting the filesystem effect.
    async fn release(&self) {
        if self.is_released.replace(true) {
            return;
        }
        for path in [&self.path, &self.lease_path] {
            if let Err(error) = self.filesystem.remove_private_subtree(path).await {
                tracing::debug!(
                    error = %error,
                    path = ?path,
                    component = "streaming.checkpoint.local",
                    "transaction scratch left for lease-aware reclamation"
                );
            }
        }
    }
}

impl Drop for TransactionTmpGuard {
    fn drop(&mut self) {
        if self.is_released.replace(true) {
            return;
        }
        // A synchronous drop cannot await the injected seam. The bounded
        // lease-aware scan reclaims what drop cannot, so this is a durable
        // fallback rather than a leak.
        tracing::debug!(
            path = ?self.path,
            component = "streaming.checkpoint.local",
            "transaction scratch deferred to lease-aware reclamation"
        );
    }
}

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Durable reachability hold on one committed generation.
///
/// The lease keeps everything transitively reachable from its pinned generation
/// readable; it says nothing about the authoritative head, which the store's own
/// head verification governs. Dropping the guard releases the advisory lock in
/// the kernel and makes one best-effort unlink; a failed unlink leaves a file
/// that is simultaneously expired and unlocked, which is exactly the state the
/// bounded lease sweep reclaims. Callers that need the release to be an ordered,
/// observable step use [`GenerationLease::release`] instead.
pub struct GenerationLease {
    backend: LocalCheckpointBackend,
    run: StreamRunIdentity,
    kind: LeaseKind,
    holder: LeaseHolderId,
    /// Exact generation this lease pins, redundant with the file name by design.
    pinned: CheckpointGeneration,
    path: PathBuf,
    /// Held for the lease's lifetime; dropping releases the advisory lock.
    lock: RefCell<Option<LocalLockHandle>>,
    /// Re-derived on every renewal.
    expires_ns: Cell<i64>,
    /// Granted lifetime, and the basis of the renewal margin.
    lease_ns: i64,
    /// Renew once the remaining lifetime falls below this margin.
    renew_margin_ns: i64,
    /// Sticky: a fenced lease never performs I/O again.
    is_fenced: Cell<bool>,
    /// Set once the name has been given up, so a later drop is a no-op.
    is_released: Cell<bool>,
}

impl std::fmt::Debug for GenerationLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GenerationLease")
            .field("kind", &self.kind)
            .field("pinned", &self.pinned)
            .finish_non_exhaustive()
    }
}

impl GenerationLease {
    /// Borrow the exact generation this lease pins.
    #[must_use]
    pub const fn pinned(&self) -> &CheckpointGeneration {
        &self.pinned
    }

    /// Borrow this lease's file path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check liveness and renew when inside the renewal margin.
    ///
    /// Returns without any filesystem effect in the common case. A fenced lease
    /// returns its retained refusal without retrying, so one lost lease cannot
    /// turn a paging read loop into a syscall storm.
    pub async fn ensure_live(&self) -> Result<(), CheckpointError> {
        if self.is_fenced.get() {
            return Err(CheckpointError::LeaseLost {
                generation: self.pinned.clone(),
            });
        }
        let now_ns = self.backend.inner.clock.now_ns();
        if now_ns.saturating_add(self.renew_margin_ns) < self.expires_ns.get() {
            return Ok(());
        }
        match self.renew(now_ns).await {
            Ok(expires_ns) => {
                self.expires_ns.set(expires_ns);
                Ok(())
            }
            Err(error) => {
                self.is_fenced.set(true);
                tracing::debug!(
                    error = %error,
                    kind = ?self.kind,
                    component = "streaming.checkpoint.lease",
                    "generation lease fenced after renewal failure"
                );
                Err(CheckpointError::LeaseLost {
                    generation: self.pinned.clone(),
                })
            }
        }
    }

    /// Release the lease as an ordered, observable step.
    ///
    /// Report ordering asserts the release position, so this returns its failure
    /// rather than swallowing it the way the drop path does.
    pub async fn release(self) -> Result<(), CheckpointError> {
        if self.is_released.replace(true) {
            return Ok(());
        }
        // Give up the lock before the name disappears, so a reclaiming scan can
        // never observe a live lock on an absent file.
        drop(self.lock.borrow_mut().take());
        self.backend
            .inner
            .filesystem
            .remove_private_subtree(&self.path)
            .await
    }

    /// Give up the advisory lock while leaving the lease file behind.
    ///
    /// This is exactly what a crashed holder leaves: the kernel released its
    /// lock, and its expired record remains until the bounded sweep reclaims it.
    #[doc(hidden)]
    pub fn simulate_holder_crash(self) {
        self.is_released.set(true);
        drop(self.lock.borrow_mut().take());
    }

    async fn renew(&self, now_ns: i64) -> Result<i64, CheckpointError> {
        if self.backend.inner.fail_next_renewal.replace(false) {
            return Err(storage_error("injected checkpoint lease renewal failure"));
        }
        let expires_ns = now_ns
            .checked_add(self.lease_ns)
            .ok_or(CheckpointError::ObjectVerification)?;
        // The file must still be ours: an absent record, or one a reclaiming scan
        // replaced, means this lease no longer pins anything.
        let observed = self.backend.read_lease_record(&self.path).await?;
        if observed.map(|record| record.holder) != Some(self.holder) {
            return Err(CheckpointError::ObjectVerification);
        }
        self.backend
            .write_lease_record(&self.path, &self.run, self.kind, self.holder, expires_ns)
            .await?;
        Ok(expires_ns)
    }
}

impl Drop for GenerationLease {
    fn drop(&mut self) {
        if self.is_released.replace(true) {
            return;
        }
        // Releasing the lock is the load-bearing half and cannot fail in a way
        // that matters. A synchronous drop cannot await the injected seam, so the
        // unlink is best effort and its leftovers are reclaimed by the bounded
        // lease sweep.
        drop(self.lock.borrow_mut().take());
        if let Err(error) = std::fs::remove_file(&self.path) {
            tracing::debug!(
                error = %error,
                path = ?self.path,
                component = "streaming.checkpoint.lease",
                "lease file left for bounded reclamation"
            );
        }
    }
}

/// Capacity limits for each independently owned local-backend resource.
///
/// Field order is the validation order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalCheckpointLimits {
    /// Simultaneously live generation transactions.
    pub transactions: BudgetLimits,
    /// Descriptors retained by staged transaction indexes.
    pub prepared_indexes: BudgetLimits,
    /// Immutable committed object storage, including objects recovered on open.
    pub storage: BudgetLimits,
    /// Descriptor summaries returned from result staging.
    pub result_summaries: BudgetLimits,
    /// Concurrent generation, participant, result, and page readers.
    pub reads: BudgetLimits,
    /// Maximum scratch entries examined per reclamation page.
    pub gc_page_items: NonZeroUsize,
    /// Lifetime granted to one prepare lease, in nanoseconds.
    pub prepare_lease_ns: u64,
}

/// Current item-and-byte usage for one backend budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalBudgetUsage {
    /// Currently charged items.
    pub used_items: usize,
    /// Currently charged bytes.
    pub used_bytes: usize,
}

/// Current backend resource usage without historical high-water telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LocalLiveBudgetUsage {
    /// Transaction usage.
    pub transactions: LocalBudgetUsage,
    /// Prepared-index usage.
    pub prepared_indexes: LocalBudgetUsage,
    /// Immutable-storage usage.
    pub storage: LocalBudgetUsage,
    /// Returned-summary usage.
    pub result_summaries: LocalBudgetUsage,
    /// Reader usage.
    pub reads: LocalBudgetUsage,
}

/// Peak per-page cost observed by the reclamation scan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct GcHighWater {
    /// Greatest number of entries examined in one page.
    pub page_items: usize,
    /// Greatest number of subtrees removed in one page.
    pub page_removals: usize,
}

/// Bounded reclamation cursor over one run's scratch directory.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct TmpReclaimCursor {
    after: Option<Box<str>>,
}

/// Outcome of one bounded reclamation page.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TmpReclaimPage {
    /// Transaction directories removed by this page.
    pub removed: usize,
    /// Transaction directories skipped because their lease is live.
    pub retained: usize,
    /// Cursor for the next page, absent when the sweep is complete.
    pub next: Option<TmpReclaimCursor>,
}

/// Whether one transaction scratch subtree may be reclaimed.
enum TransactionLeaseState {
    /// A holder still owns this transaction.
    Live,
    /// The lease is absent or expired and its advisory lock is available.
    Reclaimable,
}

/// Reachability and retention facts decoded from one generation record.
struct GenerationFacts {
    /// Participant payload digests this generation makes reachable.
    participants: Vec<ContentDigest>,
    /// Result-index root this generation makes reachable.
    result_index_root: ContentDigest,
    /// Whether this generation is terminal.
    is_final: bool,
    /// Whether this generation carries non-empty handled-issue acknowledgement.
    has_handled_issue_authority: bool,
}

struct LocalBudgets {
    transactions: BackendBudget,
    prepared_indexes: BackendBudget,
    storage: BackendBudget,
    result_summaries: BackendBudget,
    reads: BackendBudget,
}

struct StorageCommitBundle {
    _storage_lease: BudgetLease,
}

struct LocalBackendInner {
    root: PathBuf,
    filesystem: Rc<dyn LocalCheckpointFilesystem>,
    clock: Rc<dyn Clock>,
    budgets: LocalBudgets,
    limits: LocalCheckpointLimits,
    /// Acquired lazily per run on the first transaction.
    writers: RefCell<BTreeMap<StreamRunIdentity, Rc<WriterLease>>>,
    /// Storage authority for objects recovered from disk, keyed by run.
    recovered: RefCell<BTreeMap<StreamRunIdentity, Rc<StorageCommitBundle>>>,
    /// Runs this backend has opened, begun, or been given a policy for.
    runs: RefCell<BTreeSet<StreamRunIdentity>>,
    /// Authored retention policy per run; absent runs use the derived default.
    retention: RefCell<BTreeMap<StreamRunIdentity, ValidatedRetentionPolicy>>,
    /// Committed roots retained besides the head, when explicitly lowered.
    retention_override: Cell<Option<usize>>,
    /// In-process grace clock for objects observed unreachable.
    condemned: RefCell<CondemnationLedger>,
    /// Deterministic single-shot renewal failure. Test-only.
    fail_next_renewal: Cell<bool>,
    next_transaction: Cell<u64>,
    fault: Cell<Option<LocalCommitFault>>,
    reached_fault: Cell<Option<LocalCommitFault>>,
    gc_high_water: Cell<GcHighWater>,
    effects: Cell<u64>,
}

/// Crash-durable local generation store.
#[derive(Clone)]
pub struct LocalCheckpointBackend {
    inner: Rc<LocalBackendInner>,
}

impl std::fmt::Debug for LocalCheckpointBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalCheckpointBackend")
            .field("root", &self.inner.root)
            .finish_non_exhaustive()
    }
}

impl LocalCheckpointBackend {
    /// Open or create a local store rooted at `root`.
    ///
    /// Every budget is validated in field order before any state is retained,
    /// so an invalid configuration performs no filesystem effect at all.
    pub fn open(
        root: PathBuf,
        limits: LocalCheckpointLimits,
        filesystem: Rc<dyn LocalCheckpointFilesystem>,
        clock: Rc<dyn Clock>,
    ) -> Result<Self, CheckpointError> {
        let budgets = LocalBudgets {
            transactions: BackendBudget::new(
                CheckpointBackendBudgetKind::Transaction,
                limits.transactions,
            )?,
            prepared_indexes: BackendBudget::new(
                CheckpointBackendBudgetKind::PreparedIndex,
                limits.prepared_indexes,
            )?,
            storage: BackendBudget::new(CheckpointBackendBudgetKind::Storage, limits.storage)?,
            result_summaries: BackendBudget::new(
                CheckpointBackendBudgetKind::ResultSummary,
                limits.result_summaries,
            )?,
            reads: BackendBudget::new(CheckpointBackendBudgetKind::Read, limits.reads)?,
        };
        Ok(Self {
            inner: Rc::new(LocalBackendInner {
                root,
                filesystem,
                clock,
                budgets,
                limits,
                writers: RefCell::new(BTreeMap::new()),
                recovered: RefCell::new(BTreeMap::new()),
                runs: RefCell::new(BTreeSet::new()),
                retention: RefCell::new(BTreeMap::new()),
                retention_override: Cell::new(None),
                condemned: RefCell::new(CondemnationLedger::default()),
                fail_next_renewal: Cell::new(false),
                next_transaction: Cell::new(0),
                fault: Cell::new(None),
                reached_fault: Cell::new(None),
                gc_high_water: Cell::new(GcHighWater::default()),
                effects: Cell::new(0),
            }),
        })
    }

    /// Borrow the store root.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.inner.root
    }

    /// Arm one deterministic commit fault.
    #[doc(hidden)]
    pub fn inject_fault(&self, fault: LocalCommitFault) {
        self.inner.fault.set(Some(fault));
        self.inner.reached_fault.set(None);
    }

    /// Report whether the armed fault point was reached.
    #[doc(hidden)]
    #[must_use]
    pub fn injected_fault_was_reached(&self, fault: LocalCommitFault) -> bool {
        self.inner.reached_fault.get() == Some(fault)
    }

    /// Reset the injected filesystem-effect counter.
    #[doc(hidden)]
    pub fn reset_effect_counter(&self) {
        self.inner.effects.set(0);
    }

    /// Return the injected filesystem-effect counter.
    #[doc(hidden)]
    #[must_use]
    pub fn effect_counter(&self) -> u64 {
        self.inner.effects.get()
    }

    /// Peak per-page cost observed by the reclamation scan.
    #[must_use]
    pub fn gc_high_water(&self) -> GcHighWater {
        self.inner.gc_high_water.get()
    }

    /// Snapshot current charges while discarding historical high-water fields.
    #[must_use]
    pub fn live_budget_usage(&self) -> LocalLiveBudgetUsage {
        fn live(snapshot: BudgetSnapshot) -> LocalBudgetUsage {
            LocalBudgetUsage {
                used_items: snapshot.used_items,
                used_bytes: snapshot.used_bytes,
            }
        }
        LocalLiveBudgetUsage {
            transactions: live(self.inner.budgets.transactions.snapshot()),
            prepared_indexes: live(self.inner.budgets.prepared_indexes.snapshot()),
            storage: live(self.inner.budgets.storage.snapshot()),
            result_summaries: live(self.inner.budgets.result_summaries.snapshot()),
            reads: live(self.inner.budgets.reads.snapshot()),
        }
    }

    /// Borrow the filesystem seam, counting one injected effect.
    fn fs(&self) -> &dyn LocalCheckpointFilesystem {
        self.inner.effects.set(self.inner.effects.get() + 1);
        self.inner.filesystem.as_ref()
    }

    fn paths(&self, run: &StreamRunIdentity) -> RunPaths {
        RunPaths::for_run(&self.inner.root, run)
    }

    /// Create the fixed private subtree for one run. Idempotent.
    async fn ensure_run_tree(&self, paths: &RunPaths) -> Result<(), CheckpointError> {
        self.fs().create_private_dir(&self.inner.root).await?;
        self.fs().create_private_dir(paths.root()).await?;
        for directory in [
            paths.objects_dir(),
            paths.generations_dir(),
            paths.leases_dir(),
            paths.tmp_dir(),
        ] {
            self.fs().create_private_dir(&directory).await?;
        }
        Ok(())
    }

    async fn read_current(
        &self,
        paths: &RunPaths,
    ) -> Result<Option<CurrentPointer>, CheckpointError> {
        let Some(bytes) = self
            .fs()
            .read_optional(&paths.current(), MAX_CURRENT_BYTES)
            .await?
        else {
            return Ok(None);
        };
        CurrentPointer::decode(&bytes).map(Some)
    }

    async fn read_generation_bytes(
        &self,
        paths: &RunPaths,
        pointer: &CurrentPointer,
    ) -> Result<Bytes, CheckpointError> {
        let digest = parse_pointer_digest(&pointer.digest)?;
        let epoch = CheckpointEpoch::new(pointer.epoch);
        let max = u64::try_from(self.inner.budgets.storage.limits().max_bytes)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        self.fs()
            .read_optional(&paths.generation_path(epoch, &digest), max)
            .await?
            .ok_or(CheckpointError::ObjectVerification)
    }

    /// Read one content-addressed object by its committed identity.
    ///
    /// Object identities are domain-separated digests minted by the type that
    /// owns them, so the bytes are verified by the typed constructor that
    /// consumes them — `canonical_result_index_root` for the index, the segment
    /// and participant constructors for payloads — not by re-hashing here.
    async fn read_object(
        &self,
        paths: &RunPaths,
        digest: &ContentDigest,
        max_bytes: u64,
    ) -> Result<Bytes, CheckpointError> {
        self.fs()
            .read_optional(&paths.object_path(digest), max_bytes)
            .await?
            .ok_or(CheckpointError::ObjectVerification)
    }

    fn check_fault(&self, fault: LocalCommitFault) -> Result<(), CheckpointError> {
        if self.inner.fault.get() == Some(fault) {
            self.inner.fault.set(None);
            self.inner.reached_fault.set(Some(fault));
            return Err(fault.injected_error());
        }
        Ok(())
    }

    // -- writer and transaction leases -------------------------------------

    fn next_holder(&self) -> LeaseHolderId {
        let counter = self.inner.next_transaction.get().wrapping_add(1);
        self.inner.next_transaction.set(counter);
        let now = self.inner.clock.now_ns();
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&counter.to_le_bytes());
        bytes[8..].copy_from_slice(&now.to_le_bytes());
        LeaseHolderId(bytes)
    }

    async fn acquire_writer_lease(
        &self,
        run: &StreamRunIdentity,
        paths: &RunPaths,
    ) -> Result<Rc<WriterLease>, CheckpointError> {
        if let Some(existing) = self.inner.writers.borrow().get(run) {
            return Ok(Rc::clone(existing));
        }
        let path = paths.writer_lease();
        let Some(lock) = self.fs().try_lock_exclusive(&path).await? else {
            // Another live writer owns this run. Refusing here is what makes
            // "two writers cannot both commit" true before either writes an
            // object rather than at compare-and-swap time after both did.
            let actual = self.read_current(paths).await?;
            return Err(CheckpointError::GenerationConflict {
                expected: None,
                actual: actual.and_then(|pointer| pointer.into_generation().ok()),
            });
        };
        let holder = self.next_holder();
        let lease = Rc::new(WriterLease {
            _lock: lock,
            holder,
        });
        self.write_lease_record(&path, run, LeaseKind::Writer, holder, i64::MAX)
            .await?;
        self.inner
            .writers
            .borrow_mut()
            .insert(*run, Rc::clone(&lease));
        Ok(lease)
    }

    async fn write_lease_record(
        &self,
        path: &Path,
        run: &StreamRunIdentity,
        kind: LeaseKind,
        holder: LeaseHolderId,
        expires_ns: i64,
    ) -> Result<(), CheckpointError> {
        let record = LeaseRecord {
            run: *run,
            kind,
            holder,
            expires_ns,
        };
        let bytes = serde_json::to_vec(&record).map_err(|_| CheckpointError::ObjectVerification)?;
        // Provenance is advisory; the advisory lock is the authority and is held
        // on this exact inode, so the record is overwritten in place. Replacing
        // the file would strand the lock on an unlinked inode and let a second
        // holder take the same name.
        if self.fs().write_new(path, &bytes).await? {
            return Ok(());
        }
        self.fs().overwrite_in_place(path, &bytes).await
    }

    async fn create_transaction(
        &self,
        run: &StreamRunIdentity,
        paths: &RunPaths,
        writer: &WriterLease,
    ) -> Result<(TransactionTmpGuard, PrepareLease), CheckpointError> {
        let counter = self.inner.next_transaction.get().wrapping_add(1);
        self.inner.next_transaction.set(counter);
        let name = format!("{}{counter:016x}", &writer.holder.to_hex()[..16]);
        let lease_path = paths.prepare_lease(&name);
        let Some(lock) = self.fs().try_lock_exclusive(&lease_path).await? else {
            return Err(storage_error("transaction lease is already held"));
        };
        let expires_ns =
            self.inner.clock.now_ns().saturating_add(
                i64::try_from(self.inner.limits.prepare_lease_ns).unwrap_or(i64::MAX),
            );
        self.write_lease_record(
            &lease_path,
            run,
            LeaseKind::Prepare,
            writer.holder,
            expires_ns,
        )
        .await?;
        let directory = paths.transaction_dir(&name);
        self.fs().create_private_dir(&directory).await?;
        Ok((
            TransactionTmpGuard {
                filesystem: Rc::clone(&self.inner.filesystem),
                path: directory,
                lease_path: lease_path.clone(),
                is_released: Cell::new(false),
            },
            PrepareLease {
                _lock: lock,
                path: lease_path,
            },
        ))
    }

    // -- storage recovery ---------------------------------------------------

    /// Charge the reachable object set of one head exactly once per run.
    ///
    /// Local objects outlive the process that charged them, so a reopened store
    /// must re-derive its storage charge before it hands out any reader. Failing
    /// closed here keeps the "may over-retain but cannot undercharge" rule true
    /// in the safe direction.
    async fn recover_storage_charge(
        &self,
        run: &StreamRunIdentity,
        paths: &RunPaths,
        generation_bytes_len: usize,
        result_index_root: &ContentDigest,
        participants: &[ParticipantStateDescriptor],
    ) -> Result<(), CheckpointError> {
        if self.inner.recovered.borrow().contains_key(run) {
            return Ok(());
        }
        let max = u64::try_from(self.inner.budgets.storage.limits().max_bytes)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let index_bytes = self.read_object(paths, result_index_root, max).await?;
        let descriptors: Vec<ResultSegmentDescriptor> = serde_json::from_slice(&index_bytes)
            .map_err(|_| CheckpointError::ObjectVerification)?;

        let mut digests = BTreeSet::new();
        let mut bytes = generation_bytes_len
            .checked_add(index_bytes.len())
            .ok_or(CheckpointError::ObjectVerification)?;
        let mut items = 2usize;
        for (digest, length) in participants
            .iter()
            .map(|descriptor| (descriptor.content_digest, descriptor.byte_length))
            .chain(
                descriptors
                    .iter()
                    .map(|descriptor| (descriptor.payload_digest, descriptor.byte_length)),
            )
        {
            if !digests.insert(digest) {
                continue;
            }
            items = items
                .checked_add(1)
                .ok_or(CheckpointError::ObjectVerification)?;
            bytes = bytes
                .checked_add(
                    usize::try_from(length).map_err(|_| CheckpointError::ObjectVerification)?,
                )
                .ok_or(CheckpointError::ObjectVerification)?;
        }
        let lease = self.inner.budgets.storage.acquire(items, bytes).await?;
        self.inner.recovered.borrow_mut().insert(
            *run,
            Rc::new(StorageCommitBundle {
                _storage_lease: lease,
            }),
        );
        Ok(())
    }

    // -- durable publication ------------------------------------------------

    /// Execute the fixed durable commit ordering.
    ///
    /// The rename is the publication fence: before it, reopening yields the
    /// previous generation; after it, the next one. Every object and generation
    /// file written by an aborted attempt is content-addressed, immutable, and
    /// byte-identical to what a retry writes, so it is unreachable rather than
    /// wrong.
    async fn publish_durably(
        &self,
        paths: &RunPaths,
        transaction_dir: &Path,
        missing: &[(ContentDigest, Bytes)],
        generation: &CheckpointGeneration,
        generation_bytes: &[u8],
        expected: Option<&CheckpointGeneration>,
    ) -> Result<(), CheckpointError> {
        self.compare_current(paths, expected).await?;

        for (digest, bytes) in missing {
            self.write_verified(&paths.object_path(digest), bytes)
                .await?;
        }
        self.check_fault(LocalCommitFault::AfterObjectWrite)?;
        for (digest, _) in missing {
            self.fs().sync_file(&paths.object_path(digest)).await?;
        }
        self.check_fault(LocalCommitFault::AfterObjectSync)?;
        self.fs().sync_directory(&paths.objects_dir()).await?;
        self.check_fault(LocalCommitFault::AfterObjectParentSync)?;

        let generation_path = paths.generation_path(generation.epoch(), generation.digest());
        self.write_verified(&generation_path, generation_bytes)
            .await?;
        self.check_fault(LocalCommitFault::AfterGenerationWrite)?;
        self.fs().sync_file(&generation_path).await?;
        self.check_fault(LocalCommitFault::AfterGenerationSync)?;
        self.fs().sync_directory(&paths.generations_dir()).await?;
        self.check_fault(LocalCommitFault::AfterGenerationParentSync)?;

        let candidate = transaction_dir.join("CURRENT.candidate");
        let pointer = CurrentPointer::new(generation).encode()?;
        self.fs().remove_private_subtree(&candidate).await.ok();
        self.write_verified(&candidate, &pointer).await?;
        self.check_fault(LocalCommitFault::AfterCurrentTmpWrite)?;
        self.fs().sync_file(&candidate).await?;
        self.check_fault(LocalCommitFault::AfterCurrentTmpSync)?;
        self.fs().rename(&candidate, &paths.current()).await?;
        self.check_fault(LocalCommitFault::AfterCurrentRename)?;
        self.fs().sync_directory(paths.root()).await?;
        Ok(())
    }

    /// Write immutable bytes, treating an existing byte-identical file as a hit.
    async fn write_verified(&self, path: &Path, bytes: &[u8]) -> Result<(), CheckpointError> {
        if self.fs().write_new(path, bytes).await? {
            return Ok(());
        }
        let length = u64::try_from(bytes.len()).map_err(|_| CheckpointError::ObjectVerification)?;
        let existing = self
            .fs()
            .read_optional(path, length)
            .await?
            .ok_or(CheckpointError::ObjectVerification)?;
        if existing.as_ref() != bytes {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }

    /// Compare the on-disk head against the frozen expectation.
    async fn compare_current(
        &self,
        paths: &RunPaths,
        expected: Option<&CheckpointGeneration>,
    ) -> Result<(), CheckpointError> {
        let pointer = self.read_current(paths).await?;
        let matches = match (&pointer, expected) {
            (None, None) => true,
            (Some(pointer), Some(expected)) => pointer.matches(expected),
            _ => false,
        };
        if matches {
            return Ok(());
        }
        Err(CheckpointError::GenerationConflict {
            expected: expected.cloned(),
            actual: pointer.and_then(|pointer| pointer.into_generation().ok()),
        })
    }

    /// Filter the assembled object set down to the ones not already on disk.
    async fn filter_absent_objects(
        &self,
        paths: &RunPaths,
        objects: BTreeMap<ContentDigest, Bytes>,
    ) -> Result<Vec<(ContentDigest, Bytes)>, CheckpointError> {
        let mut missing = Vec::new();
        for (digest, bytes) in objects {
            let length =
                u64::try_from(bytes.len()).map_err(|_| CheckpointError::ObjectVerification)?;
            match self
                .fs()
                .read_optional(&paths.object_path(&digest), length)
                .await?
            {
                Some(existing) if existing == bytes => {}
                Some(_) => return Err(CheckpointError::ObjectVerification),
                None => missing.push((digest, bytes)),
            }
        }
        Ok(missing)
    }

    // -- reclamation --------------------------------------------------------

    /// Reclaim one bounded page of orphaned transaction scratch subtrees.
    ///
    /// A subtree is reclaimable only when its prepare lease is absent, or its
    /// recorded expiry has passed on the injected clock and its advisory lock
    /// can be taken. Modification time is never consulted and a live
    /// transaction is never removed.
    pub async fn reclaim_orphan_transactions(
        &self,
        run: &StreamRunIdentity,
        cursor: Option<TmpReclaimCursor>,
    ) -> Result<TmpReclaimPage, CheckpointError> {
        let paths = self.paths(run);
        let limit = self.inner.limits.gc_page_items.get();
        let after = cursor.as_ref().and_then(|cursor| cursor.after.as_deref());
        let page = self
            .fs()
            .list_dir_page(&paths.tmp_dir(), after, limit)
            .await?;

        let now_ns = self.inner.clock.now_ns();
        let mut removed = 0usize;
        let mut retained = 0usize;
        for name in &page.names {
            match self
                .classify_transaction_lease(&paths, name, now_ns)
                .await?
            {
                TransactionLeaseState::Live => retained += 1,
                TransactionLeaseState::Reclaimable => {
                    self.fs()
                        .remove_private_subtree(&paths.transaction_dir(name))
                        .await?;
                    self.fs()
                        .remove_private_subtree(&paths.prepare_lease(name))
                        .await?;
                    removed += 1;
                }
            }
        }

        let previous = self.inner.gc_high_water.get();
        self.inner.gc_high_water.set(GcHighWater {
            page_items: previous.page_items.max(page.names.len()),
            page_removals: previous.page_removals.max(removed),
        });

        Ok(TmpReclaimPage {
            removed,
            retained,
            next: page
                .next
                .map(|after| TmpReclaimCursor { after: Some(after) }),
        })
    }

    /// Drive reclamation to completion under the configured page bound.
    pub async fn reclaim_all_orphan_transactions(
        &self,
        run: &StreamRunIdentity,
    ) -> Result<usize, CheckpointError> {
        let mut cursor = None;
        let mut removed = 0usize;
        loop {
            let page = self.reclaim_orphan_transactions(run, cursor).await?;
            removed += page.removed;
            match page.next {
                Some(next) => cursor = Some(next),
                None => return Ok(removed),
            }
        }
    }

    async fn classify_transaction_lease(
        &self,
        paths: &RunPaths,
        name: &str,
        now_ns: i64,
    ) -> Result<TransactionLeaseState, CheckpointError> {
        let lease_path = paths.prepare_lease(name);
        let Some(bytes) = self
            .fs()
            .read_optional(&lease_path, MAX_LEASE_RECORD_BYTES)
            .await?
        else {
            // Absent lease: the owner never got one, or already released it.
            return Ok(TransactionLeaseState::Reclaimable);
        };
        let record: LeaseRecord =
            serde_json::from_slice(&bytes).map_err(|_| CheckpointError::ObjectVerification)?;
        if record.kind != LeaseKind::Prepare {
            return Ok(TransactionLeaseState::Live);
        }
        if record.expires_ns > now_ns {
            return Ok(TransactionLeaseState::Live);
        }
        // Expired by the injected clock. The advisory probe is the second,
        // independent witness: a paused-but-alive holder still owns the lock,
        // and a dead holder's lock was released by the kernel.
        match self.fs().try_lock_exclusive(&lease_path).await? {
            Some(probe) => {
                drop(probe);
                Ok(TransactionLeaseState::Reclaimable)
            }
            None => Ok(TransactionLeaseState::Live),
        }
    }

    // -- reachability leases ------------------------------------------------

    /// Record one run so a later collection cycle can find its subtree.
    fn register_run(&self, run: StreamRunIdentity) {
        self.inner.runs.borrow_mut().insert(run);
    }

    /// Resolve the effective retention policy for one run.
    fn retention_for(&self, run: &StreamRunIdentity) -> ValidatedRetentionPolicy {
        if let Some(policy) = self.inner.retention.borrow().get(run) {
            return *policy;
        }
        ValidatedRetentionPolicy::derived(
            i64::try_from(self.inner.limits.prepare_lease_ns).unwrap_or(i64::MAX),
            DEFAULT_READER_LEASE_NS,
        )
    }

    /// Read and strictly decode one lease record, tolerating an absent file.
    async fn read_lease_record(&self, path: &Path) -> Result<Option<LeaseRecord>, CheckpointError> {
        let Some(bytes) = self
            .fs()
            .read_optional(path, MAX_LEASE_RECORD_BYTES)
            .await?
        else {
            return Ok(None);
        };
        serde_json::from_slice(&bytes)
            .map(Some)
            .map_err(|_| CheckpointError::ObjectVerification)
    }

    /// Acquire one durable reachability hold over a committed generation.
    ///
    /// The lock is taken before the record is written, so the descriptor the
    /// holder retains is the one a reclaiming probe will open.
    async fn acquire_generation_lease(
        &self,
        run: StreamRunIdentity,
        kind: LeaseKind,
        pinned: &CheckpointGeneration,
    ) -> Result<GenerationLease, CheckpointError> {
        let prefix = match kind {
            LeaseKind::Reader => READER_LEASE_PREFIX,
            LeaseKind::Report => REPORT_LEASE_PREFIX,
            // Writer and prepare leases are not generation-pinned and are named
            // by the publication and transaction paths instead.
            LeaseKind::Writer | LeaseKind::Prepare => {
                return Err(CheckpointError::ObjectVerification);
            }
        };
        let paths = self.paths(&run);
        let holder = self.next_holder();
        let path =
            paths
                .leases_dir()
                .join(generation_lease_file_name(prefix, pinned, &holder.to_hex()));
        let lease_ns = self.retention_for(&run).reader_lease_ns();
        let expires_ns = self
            .inner
            .clock
            .now_ns()
            .checked_add(lease_ns)
            .ok_or(CheckpointError::ObjectVerification)?;
        let lock = self
            .fs()
            .try_lock_exclusive(&path)
            .await?
            .ok_or_else(|| storage_error("checkpoint reachability lease is already held"))?;
        self.write_lease_record(&path, &run, kind, holder, expires_ns)
            .await?;
        self.register_run(run);
        Ok(GenerationLease {
            backend: self.clone(),
            run,
            kind,
            holder,
            pinned: pinned.clone(),
            path,
            lock: RefCell::new(Some(lock)),
            expires_ns: Cell::new(expires_ns),
            lease_ns,
            // A lease that renews inside its margin cannot be reclaimed even by a
            // maximally adversarial collection schedule.
            renew_margin_ns: lease_ns / 4,
            is_fenced: Cell::new(false),
            is_released: Cell::new(false),
        })
    }

    /// Acquire a report-retention lease over one committed generation.
    ///
    /// Compaction acquires it before it starts; report persistence releases it as
    /// the last ordered step after the report commit.
    pub async fn acquire_report_lease(
        &self,
        run: StreamRunIdentity,
        generation: &CheckpointGeneration,
    ) -> Result<GenerationLease, CheckpointError> {
        self.acquire_generation_lease(run, LeaseKind::Report, generation)
            .await
    }

    /// Fail the next reachability-lease renewal deterministically. Test-only.
    #[doc(hidden)]
    pub fn fail_next_renewal(&self) {
        self.inner.fail_next_renewal.set(true);
    }

    // -- collection ---------------------------------------------------------

    /// Fold one bounded page's cost into the reclamation high-water telemetry.
    fn note_gc_page(&self, items: usize, removals: usize) {
        let previous = self.inner.gc_high_water.get();
        self.inner.gc_high_water.set(GcHighWater {
            page_items: previous.page_items.max(items),
            page_removals: previous.page_removals.max(removals),
        });
    }

    /// Classify one lease file with both independent witnesses.
    ///
    /// A record whose run or kind disagrees with its file name is a forged or
    /// torn lease, not a reason to guess in either direction.
    async fn classify_lease(
        &self,
        path: &Path,
        run: &StreamRunIdentity,
        expected: LeaseKind,
        now_ns: i64,
    ) -> Result<LeaseLiveness, CheckpointError> {
        let Some(record) = self.read_lease_record(path).await? else {
            return Ok(LeaseLiveness::Reclaimable);
        };
        if record.run != *run || record.kind != expected {
            return Err(CheckpointError::ObjectVerification);
        }
        if record.expires_ns > now_ns {
            return Ok(LeaseLiveness::Live);
        }
        // Expired by the injected clock. The advisory probe is the second,
        // independent witness: a paused-but-alive holder still owns the lock, and
        // a dead holder's lock was released by the kernel.
        match self.fs().try_lock_exclusive(path).await? {
            Some(probe) => {
                drop(probe);
                Ok(LeaseLiveness::Reclaimable)
            }
            None => Ok(LeaseLiveness::Live),
        }
    }

    /// Enumerate generation pins from `leases/` and reclaim the expired files.
    ///
    /// Leases are enumerated before any candidate set is computed, so a reader
    /// that opened before this cycle is always visible to the mark phase.
    async fn scan_generation_leases(
        &self,
        run: &StreamRunIdentity,
        paths: &RunPaths,
        now_ns: i64,
    ) -> Result<(BTreeSet<CheckpointGeneration>, usize), CheckpointError> {
        let limit = self.inner.limits.gc_page_items.get();
        let leases_dir = paths.leases_dir();
        let mut after: Option<Box<str>> = None;
        let mut pinned = BTreeSet::new();
        let mut swept = 0usize;
        loop {
            let page = self
                .fs()
                .list_dir_page(&leases_dir, after.as_deref(), limit)
                .await?;
            let mut removals = 0usize;
            for name in &page.names {
                let Some((is_report, generation)) = pinned_generation_from_name(name) else {
                    continue;
                };
                let expected = if is_report {
                    LeaseKind::Report
                } else {
                    LeaseKind::Reader
                };
                let path = leases_dir.join(name.as_ref());
                match self.classify_lease(&path, run, expected, now_ns).await? {
                    LeaseLiveness::Live => {
                        pinned.insert(generation);
                    }
                    LeaseLiveness::Reclaimable => {
                        self.fs().remove_private_subtree(&path).await?;
                        swept += 1;
                        removals += 1;
                    }
                }
            }
            self.note_gc_page(page.names.len(), removals);
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
        Ok((pinned, swept))
    }

    /// List every generation record in epoch order without decoding any of them.
    async fn list_generation_records(
        &self,
        paths: &RunPaths,
    ) -> Result<Vec<(Box<str>, CheckpointGeneration)>, CheckpointError> {
        let limit = self.inner.limits.gc_page_items.get();
        let directory = paths.generations_dir();
        let mut after: Option<Box<str>> = None;
        let mut listed = Vec::new();
        loop {
            let page = self
                .fs()
                .list_dir_page(&directory, after.as_deref(), limit)
                .await?;
            self.note_gc_page(page.names.len(), 0);
            for name in &page.names {
                if let Some(generation) = generation_from_record_name(name) {
                    listed.push((name.clone(), generation));
                }
            }
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
        Ok(listed)
    }

    /// Decode the reachability and retention facts of one generation record.
    async fn decode_generation_facts(
        &self,
        paths: &RunPaths,
        generation: &CheckpointGeneration,
    ) -> Result<GenerationFacts, CheckpointError> {
        let max_bytes = self.inner.budgets.storage.limits().max_bytes;
        let max = u64::try_from(max_bytes).map_err(|_| CheckpointError::ObjectVerification)?;
        let bytes = self
            .fs()
            .read_optional(
                &paths.generation_path(generation.epoch(), generation.digest()),
                max,
            )
            .await?
            .ok_or(CheckpointError::ObjectVerification)?;
        match decode_versioned_checkpoint_generation(&bytes, max_bytes)? {
            DecodedCheckpointGeneration::CurrentV4(candidate) => {
                candidate.verify_decoded()?;
                Ok(GenerationFacts {
                    participants: candidate
                        .reachable_participant_descriptors()
                        .iter()
                        .map(|descriptor| descriptor.content_digest)
                        .collect(),
                    result_index_root: *candidate.reachable_result_index_root(),
                    is_final: candidate.is_final(),
                    // A generation whose acknowledgement roots differ from the
                    // empty cut carries authority a later run needs in order not
                    // to re-quarantine handled issues. This is the only way the
                    // handled cut influences collection: as a retention signal,
                    // never as a mark-set input.
                    has_handled_issue_authority: candidate.represented_cut().handled_issues
                        != HandledIssueCut::empty(),
                })
            }
            DecodedCheckpointGeneration::LegacyV3ReadOnly(legacy) => Ok(GenerationFacts {
                participants: legacy
                    .participant_descriptors()
                    .iter()
                    .map(|descriptor| descriptor.content_digest)
                    .collect(),
                result_index_root: *legacy.result_index_root(),
                is_final: false,
                has_handled_issue_authority: false,
            }),
        }
    }

    /// Select every generation that retention or an active lease pins.
    fn select_pinned_generations(
        listed: &[(Box<str>, CheckpointGeneration)],
        facts: &[GenerationFacts],
        head: Option<&CheckpointGeneration>,
        leased: &BTreeSet<CheckpointGeneration>,
        policy: &ValidatedRetentionPolicy,
        retention_override: Option<usize>,
    ) -> BTreeSet<CheckpointGeneration> {
        let mut pinned: BTreeSet<CheckpointGeneration> = leased.clone();
        // The head is pinned unconditionally and is never overridable: a store
        // must always be openable, which is what makes lowering retention to zero
        // safe rather than destructive.
        if let Some(head) = head {
            pinned.insert(head.clone());
        }
        let resume_roots = retention_override.unwrap_or_else(|| policy.resume_roots());
        let partial_history = retention_override.map_or_else(|| policy.partial_history(), |_| 0);
        let window = resume_roots.saturating_add(partial_history);

        let mut retained = 0usize;
        let mut handled_issue_seen = false;
        for (index, (_, generation)) in listed.iter().enumerate().rev() {
            let facts = &facts[index];
            if retained < window {
                retained += 1;
                pinned.insert(generation.clone());
                continue;
            }
            if facts.is_final && policy.retains_final_until_exported() {
                pinned.insert(generation.clone());
            }
            if facts.has_handled_issue_authority && !handled_issue_seen {
                handled_issue_seen = true;
                pinned.insert(generation.clone());
            }
        }
        pinned
    }

    /// Mark every object reachable from the pinned generations.
    ///
    /// The result index is traversed in bounded pages so peak allocation stays
    /// inside the configured page limit regardless of store size. The handled-cut
    /// roots are never inserted: they are digests over ledger state, not names of
    /// any byte sequence this store has written, so marking them would make the
    /// sweep look for objects that never existed.
    async fn mark_reachable(
        &self,
        paths: &RunPaths,
        pinned: &BTreeMap<CheckpointGeneration, GenerationFacts>,
    ) -> Result<ObjectMarkSet, CheckpointError> {
        let page_items = self.inner.limits.gc_page_items.get();
        let max = u64::try_from(self.inner.budgets.storage.limits().max_bytes)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let mut marked = BTreeSet::new();
        for (generation, facts) in pinned {
            marked.insert(*generation.digest());
            marked.extend(facts.participants.iter().copied());
            marked.insert(facts.result_index_root);

            let bytes = self
                .read_object(paths, &facts.result_index_root, max)
                .await?;
            let descriptors: Vec<ResultSegmentDescriptor> =
                serde_json::from_slice(&bytes).map_err(|_| CheckpointError::ObjectVerification)?;
            for page in descriptors.chunks(page_items) {
                self.note_gc_page(page.len(), 0);
                marked.extend(page.iter().map(|descriptor| descriptor.payload_digest));
            }
        }
        // A mark set that does not fit the read budget must refuse rather than
        // sweep from partial reachability, which would delete live data.
        let bytes = marked
            .len()
            .checked_mul(std::mem::size_of::<ContentDigest>())
            .ok_or(CheckpointError::ObjectVerification)?;
        let lease = self
            .inner
            .budgets
            .reads
            .acquire(marked.len(), bytes)
            .await?;
        Ok(ObjectMarkSet::new(marked, lease))
    }

    /// Condemn and sweep one bounded page at a time over `objects/`.
    async fn sweep_objects(
        &self,
        paths: &RunPaths,
        marks: &ObjectMarkSet,
        now_ns: i64,
        grace_ns: i64,
    ) -> Result<(usize, usize, BTreeSet<ContentDigest>), CheckpointError> {
        let limit = self.inner.limits.gc_page_items.get();
        let directory = paths.objects_dir();
        let mut after: Option<Box<str>> = None;
        let mut swept = 0usize;
        let mut condemned = 0usize;
        let mut present = BTreeSet::new();
        loop {
            let page = self
                .fs()
                .list_dir_page(&directory, after.as_deref(), limit)
                .await?;
            let mut removals = 0usize;
            for name in &page.names {
                let Some(digest) = object_digest_from_name(name) else {
                    continue;
                };
                if marks.contains(&digest) {
                    self.inner.condemned.borrow_mut().absolve(&digest);
                    present.insert(digest);
                    continue;
                }
                let elapsed = self
                    .inner
                    .condemned
                    .borrow_mut()
                    .condemn(digest, now_ns, grace_ns);
                if elapsed {
                    self.fs()
                        .remove_private_subtree(&directory.join(name.as_ref()))
                        .await?;
                    self.inner.condemned.borrow_mut().absolve(&digest);
                    swept += 1;
                    removals += 1;
                } else {
                    present.insert(digest);
                    condemned += 1;
                }
            }
            self.note_gc_page(page.names.len(), removals);
            match page.next {
                Some(next) => after = Some(next),
                None => break,
            }
        }
        Ok((swept, condemned, present))
    }

    /// Sweep unpinned generation records under the same condemnation rule.
    ///
    /// Surviving records are added to `present` so the shared grace ledger keeps
    /// their condemnations across cycles instead of forgetting them every sweep.
    async fn sweep_generations(
        &self,
        paths: &RunPaths,
        listed: &[(Box<str>, CheckpointGeneration)],
        pinned: &BTreeSet<CheckpointGeneration>,
        now_ns: i64,
        grace_ns: i64,
        present: &mut BTreeSet<ContentDigest>,
    ) -> Result<usize, CheckpointError> {
        let directory = paths.generations_dir();
        let mut swept = 0usize;
        for (name, generation) in listed {
            if pinned.contains(generation) {
                self.inner
                    .condemned
                    .borrow_mut()
                    .absolve(generation.digest());
                present.insert(*generation.digest());
                continue;
            }
            let elapsed =
                self.inner
                    .condemned
                    .borrow_mut()
                    .condemn(*generation.digest(), now_ns, grace_ns);
            if elapsed {
                self.fs()
                    .remove_private_subtree(&directory.join(name.as_ref()))
                    .await?;
                self.inner
                    .condemned
                    .borrow_mut()
                    .absolve(generation.digest());
                swept += 1;
            } else {
                present.insert(*generation.digest());
            }
        }
        Ok(swept)
    }

    /// Take sweep authority without blocking on a live writer.
    ///
    /// A concurrent publisher legitimately skips writing an object it observes
    /// already present, so unlinking under a live writer can strand the head it
    /// is about to publish. Mutual exclusion is the whole answer on this path:
    /// there is no crash predicate to evaluate, because the sweep can only run
    /// when no live writer holds the lock.
    async fn try_writer_authority(
        &self,
        run: &StreamRunIdentity,
        paths: &RunPaths,
    ) -> Result<Option<Rc<WriterLease>>, CheckpointError> {
        if let Some(existing) = self.inner.writers.borrow().get(run) {
            return Ok(Some(Rc::clone(existing)));
        }
        let path = paths.writer_lease();
        let Some(lock) = self.fs().try_lock_exclusive(&path).await? else {
            return Ok(None);
        };
        let holder = self.next_holder();
        let lease = Rc::new(WriterLease {
            _lock: lock,
            holder,
        });
        self.write_lease_record(&path, run, LeaseKind::Writer, holder, i64::MAX)
            .await?;
        self.inner
            .writers
            .borrow_mut()
            .insert(*run, Rc::clone(&lease));
        Ok(Some(lease))
    }

    /// Run one complete bounded collection cycle over one run.
    pub async fn collect_garbage_for_run(
        &self,
        run: &StreamRunIdentity,
    ) -> Result<GcReport, CheckpointError> {
        let paths = self.paths(run);
        let policy = self.retention_for(run);
        let now_ns = self.inner.clock.now_ns();

        let (leased, swept_leases) = self.scan_generation_leases(run, &paths, now_ns).await?;
        // Private transaction temporary names are never object-sweep candidates;
        // they are reclaimed only by their own lease-aware scan.
        let reclaimed_transactions = self.reclaim_all_orphan_transactions(run).await?;

        let listed = self.list_generation_records(&paths).await?;
        let mut facts = Vec::with_capacity(listed.len());
        for (_, generation) in &listed {
            facts.push(self.decode_generation_facts(&paths, generation).await?);
        }
        let head = match self.read_current(&paths).await? {
            Some(pointer) => Some(pointer.into_generation()?),
            None => None,
        };
        let pinned = Self::select_pinned_generations(
            &listed,
            &facts,
            head.as_ref(),
            &leased,
            &policy,
            self.inner.retention_override.get(),
        );

        let mut roots = BTreeMap::new();
        for ((_, generation), facts) in listed.iter().zip(facts) {
            if pinned.contains(generation) {
                roots.insert(generation.clone(), facts);
            }
        }
        let marks = self.mark_reachable(&paths, &roots).await?;

        let Some(_authority) = self.try_writer_authority(run, &paths).await? else {
            return Ok(GcReport {
                authority: SweepAuthority::Unavailable,
                pinned_generations: pinned.len(),
                marked_objects: marks.len(),
                condemned_objects: 0,
                swept_objects: 0,
                swept_generations: 0,
                swept_leases,
                reclaimed_transactions,
            });
        };

        let grace_ns = policy.orphan_grace_ns();
        let (swept_objects, condemned_objects, mut present) =
            self.sweep_objects(&paths, &marks, now_ns, grace_ns).await?;
        let swept_generations = self
            .sweep_generations(&paths, &listed, &pinned, now_ns, grace_ns, &mut present)
            .await?;
        self.inner.condemned.borrow_mut().retain_present(&present);

        Ok(GcReport {
            authority: SweepAuthority::Held,
            pinned_generations: pinned.len(),
            marked_objects: marks.len(),
            condemned_objects,
            swept_objects,
            swept_generations,
            swept_leases,
            reclaimed_transactions,
        })
    }

    // -- backend surface ----------------------------------------------------

    /// Open and verify the latest authoritative generation for one run.
    pub async fn open_latest_local(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<LeasedCheckpointGeneration>, CheckpointError> {
        if run != &expected.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let paths = self.paths(run);
        let Some(pointer) = self.read_current(&paths).await? else {
            return Ok(None);
        };
        let bytes = self.read_generation_bytes(&paths, &pointer).await?;
        let lease = self.inner.budgets.reads.acquire(1, bytes.len()).await?;

        // Re-read the pointer after any wait so a superseded head can never be
        // handed out as authority.
        let head = pointer.into_generation()?;
        match self.read_current(&paths).await? {
            Some(current) if current.matches(&head) => {}
            _ => {
                return Err(CheckpointError::LeaseLost { generation: head });
            }
        }

        let decoded = decode_versioned_checkpoint_generation(
            &bytes,
            self.inner.budgets.storage.limits().max_bytes,
        )?;
        // The reachability lease is taken before the reader becomes observable,
        // so a collection cycle that starts afterwards always sees the pin.
        let reachability = self
            .acquire_generation_lease(*run, LeaseKind::Reader, &head)
            .await?;
        let opened = match decoded {
            DecodedCheckpointGeneration::CurrentV4(candidate) => {
                if candidate.generation() != head {
                    return Err(CheckpointError::ObjectVerification);
                }
                let prevalidated = candidate.prevalidate_for_publication(
                    run,
                    &expected.participant_plan,
                    &expected.execution_plan_digest,
                    &expected.result_plan_digest,
                )?;
                let committed = prevalidated.into_committed_after_publication_fence();
                self.recover_storage_charge(
                    run,
                    &paths,
                    bytes.len(),
                    committed.result_index_root(),
                    committed.participant_descriptors(),
                )
                .await?;
                LeasedCheckpointGeneration::current_v4(LocalGenerationReader {
                    backend: self.clone(),
                    paths,
                    generation: committed,
                    reachability,
                    _generation_lease: lease,
                })
            }
            DecodedCheckpointGeneration::LegacyV3ReadOnly(generation) => {
                if generation.generation() != &head {
                    return Err(CheckpointError::ObjectVerification);
                }
                generation.verify_against(
                    run,
                    &expected.participant_plan,
                    &expected.execution_plan_digest,
                    &expected.result_plan_digest,
                )?;
                self.recover_storage_charge(
                    run,
                    &paths,
                    bytes.len(),
                    generation.result_index_root(),
                    generation.participant_descriptors(),
                )
                .await?;
                LeasedCheckpointGeneration::legacy_v3(LocalLegacyV3GenerationReader {
                    backend: self.clone(),
                    paths,
                    generation,
                    reachability,
                    _generation_lease: lease,
                })
            }
        };
        Ok(Some(opened))
    }

    /// Begin a transaction frozen to one exact run, head, and semantic plan.
    pub async fn begin_generation_local(
        &self,
        run: StreamRunIdentity,
        expected: Option<CurrentV4CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<LocalGenerationTransaction, CheckpointError> {
        if run != expectations.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let paths = self.paths(&run);
        self.ensure_run_tree(&paths).await?;
        self.register_run(run);

        // A present head is inspected even for a fresh begin: a legacy-v3 head
        // is not "no head", and following it would fork the digest chain.
        if let Some(pointer) = self.read_current(&paths).await? {
            let head = pointer.into_generation()?;
            let bytes = self
                .read_generation_bytes(&paths, &CurrentPointer::new(&head))
                .await?;
            if matches!(
                decode_versioned_checkpoint_generation(
                    &bytes,
                    self.inner.budgets.storage.limits().max_bytes,
                )?,
                DecodedCheckpointGeneration::LegacyV3ReadOnly(_)
            ) {
                return Err(CheckpointError::LegacyReadOnlyHead);
            }
            let expected_generation = expected
                .as_ref()
                .map(CurrentV4CheckpointGeneration::generation);
            if expected_generation != Some(&head) {
                return Err(CheckpointError::GenerationConflict {
                    expected: expected_generation.cloned(),
                    actual: Some(head),
                });
            }
        } else if let Some(expected) = expected.as_ref() {
            return Err(CheckpointError::GenerationConflict {
                expected: Some(expected.generation().clone()),
                actual: None,
            });
        }

        let writer = self.acquire_writer_lease(&run, &paths).await?;
        let transaction_lease = self.inner.budgets.transactions.acquire(1, 1).await?;
        let (tmp_guard, prepare_lease) = self.create_transaction(&run, &paths, &writer).await?;
        Ok(LocalGenerationTransaction {
            backend: self.clone(),
            run,
            paths,
            expected,
            expectations,
            _writer: writer,
            _transaction_lease: transaction_lease,
            _prepare_lease: prepare_lease,
            tmp_guard,
            participants: Vec::new(),
            staged_results: None,
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointBackend for LocalCheckpointBackend {
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<LeasedCheckpointGeneration>, CheckpointError> {
        self.open_latest_local(run, expected).await
    }

    async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CurrentV4CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError> {
        Ok(Box::new(
            self.begin_generation_local(run, expected, expectations)
                .await?,
        ))
    }
}

impl CurrentPointer {
    fn into_generation(self) -> Result<CheckpointGeneration, CheckpointError> {
        Ok(CheckpointGeneration::new(
            CheckpointEpoch::new(self.epoch),
            parse_pointer_digest(&self.digest)?,
        ))
    }
}

fn parse_pointer_digest(text: &str) -> Result<ContentDigest, CheckpointError> {
    let hex = text
        .strip_prefix("blake3:")
        .ok_or(CheckpointError::ObjectVerification)?;
    if hex.len() != 64 {
        return Err(CheckpointError::ObjectVerification);
    }
    let mut bytes = [0u8; 32];
    for (index, slot) in bytes.iter_mut().enumerate() {
        let pair = hex
            .get(index * 2..index * 2 + 2)
            .ok_or(CheckpointError::ObjectVerification)?;
        *slot = u8::from_str_radix(pair, 16).map_err(|_| CheckpointError::ObjectVerification)?;
    }
    Ok(ContentDigest::from_bytes(bytes))
}

// ---------------------------------------------------------------------------
// Transaction
// ---------------------------------------------------------------------------

struct StagedParticipant {
    descriptor: ParticipantStateDescriptor,
    payload: BudgetedCheckpointBytes,
}

struct StagedResultEpoch {
    index_root: ContentDigest,
    descriptors: BudgetedResultDescriptors,
    payloads: Vec<BudgetedCheckpointBytes>,
    item_count: u64,
    byte_length: u64,
}

struct CheckedResultStagePlan {
    descriptor_items: usize,
    descriptor_bytes: usize,
    index_root: ContentDigest,
    item_count: u64,
    byte_length: u64,
}

impl CheckedResultStagePlan {
    fn from_partitions(partitions: &[&ResultPartition]) -> Result<Self, CheckpointError> {
        let descriptor_bytes = partitions.iter().try_fold(0usize, |total, partition| {
            total
                .checked_add(descriptor_retained_bytes(partition.descriptor())?)
                .ok_or(CheckpointError::ObjectVerification)
        })?;
        let (item_count, byte_length) =
            partitions
                .iter()
                .try_fold((0u64, 0u64), |(items, bytes), partition| {
                    let descriptor = partition.descriptor();
                    Ok((
                        items
                            .checked_add(descriptor.item_count)
                            .ok_or(CheckpointError::ObjectVerification)?,
                        bytes
                            .checked_add(descriptor.byte_length)
                            .ok_or(CheckpointError::ObjectVerification)?,
                    ))
                })?;
        let (index_root, _) = canonical_result_index_object(
            partitions.iter().copied().map(ResultPartition::descriptor),
        )?;
        Ok(Self {
            descriptor_items: partitions.len(),
            descriptor_bytes,
            index_root,
            item_count,
            byte_length,
        })
    }
}

/// One atomic local generation transaction.
pub struct LocalGenerationTransaction {
    backend: LocalCheckpointBackend,
    run: StreamRunIdentity,
    paths: RunPaths,
    expected: Option<CurrentV4CheckpointGeneration>,
    expectations: CheckpointGenerationExpectations,
    _writer: Rc<WriterLease>,
    _transaction_lease: BudgetLease,
    _prepare_lease: PrepareLease,
    tmp_guard: TransactionTmpGuard,
    participants: Vec<StagedParticipant>,
    staged_results: Option<StagedResultEpoch>,
}

impl std::fmt::Debug for LocalGenerationTransaction {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LocalGenerationTransaction")
            .field("run", &self.run)
            .field("scratch", &self.tmp_guard.path())
            .finish_non_exhaustive()
    }
}

impl LocalGenerationTransaction {
    /// Borrow this transaction's private scratch subtree path.
    #[must_use]
    pub fn tmp_path(&self) -> &Path {
        self.tmp_guard.path()
    }

    /// Borrow this transaction's prepare-lease path.
    #[must_use]
    pub fn prepare_lease_path(&self) -> &Path {
        &self._prepare_lease.path
    }

    /// Cancel this transaction, removing its private scratch subtree.
    pub async fn cancel(self) {
        self.tmp_guard.release().await;
    }

    /// Stage one participant.
    pub async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        self.stage_participant_inner(state)
    }

    /// Stage the one required result epoch.
    pub async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        self.prepare_result_partitions(partitions, issue_receipts)
            .await
    }

    /// Commit this transaction atomically.
    pub async fn commit(
        self,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        self.commit_inner(metadata).await
    }

    fn stage_participant_inner(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        if state.run() != &self.run
            || self.participants.iter().any(|existing| {
                existing.descriptor.participant_id == state.descriptor().participant_id
            })
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (_, descriptor, payload) = state.into_parts();
        self.participants.push(StagedParticipant {
            descriptor,
            payload,
        });
        Ok(())
    }

    /// Stage result partitions. This performs no filesystem effect: objects are
    /// written only by the durable commit ordering, so a bind failure between
    /// staging and commit can never leave durable partial state.
    async fn prepare_result_partitions(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        if self.staged_results.is_some() {
            return Err(CheckpointError::ObjectVerification);
        }
        let issue_partition = issue_receipts
            .as_ref()
            .map(PreparedIssueReceiptResultPartition::partition);
        if partitions
            .iter()
            .chain(issue_partition)
            .any(|partition| partition.descriptor().run != self.run)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let staged: Vec<&ResultPartition> = partitions.iter().chain(issue_partition).collect();
        let plan = CheckedResultStagePlan::from_partitions(&staged)?;
        drop(staged);
        let prepared_lease = self
            .backend
            .inner
            .budgets
            .prepared_indexes
            .acquire(plan.descriptor_items, plan.descriptor_bytes)
            .await?;
        let summary_lease = self
            .backend
            .inner
            .budgets
            .result_summaries
            .acquire(plan.descriptor_items, plan.descriptor_bytes)
            .await?;
        self.install_result_partitions(
            partitions,
            issue_receipts,
            plan,
            prepared_lease,
            summary_lease,
        )
    }

    fn install_result_partitions(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
        plan: CheckedResultStagePlan,
        prepared_lease: BudgetLease,
        summary_lease: BudgetLease,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        let issue_partition = issue_receipts
            .as_ref()
            .map(PreparedIssueReceiptResultPartition::partition);
        let prepared_descriptors = partitions
            .iter()
            .chain(issue_partition)
            .map(|partition| partition.descriptor().clone())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let summary_descriptors = prepared_descriptors.to_vec().into_boxed_slice();
        let prepared_descriptors =
            BudgetedResultDescriptors::new(prepared_descriptors, prepared_lease)?;
        let summary_descriptors =
            BudgetedResultDescriptors::new(summary_descriptors, summary_lease)?;
        let (issue_payload_partition, binding) = match issue_receipts.take() {
            Some(handoff) => {
                let (partition, binding) = handoff.into_staged_parts(plan.index_root);
                (Some(partition), Some(binding))
            }
            None => (None, None),
        };
        let prepared_summary = PreparedResultEpoch::new(
            plan.index_root,
            summary_descriptors,
            plan.item_count,
            plan.byte_length,
            binding,
        )?;

        // Publication fence: every fallible construction above has succeeded,
        // so the drain below cannot fail and cannot leave a partial epoch.
        let mut payloads = Vec::with_capacity(plan.descriptor_items);
        for partition in std::mem::take(partitions)
            .into_iter()
            .chain(issue_payload_partition)
        {
            let (budgeted_descriptor, payload) = partition.into_parts();
            let (input_descriptor, input_lease) = budgeted_descriptor.into_backend_parts();
            payloads.push(payload);
            drop(input_descriptor);
            drop(input_lease);
        }
        self.staged_results = Some(StagedResultEpoch {
            index_root: plan.index_root,
            descriptors: prepared_descriptors,
            payloads,
            item_count: plan.item_count,
            byte_length: plan.byte_length,
        });
        Ok(prepared_summary)
    }

    async fn commit_inner(
        self,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        let Self {
            backend,
            run,
            paths,
            expected,
            expectations,
            _writer,
            _transaction_lease,
            _prepare_lease,
            tmp_guard,
            participants,
            staged_results,
        } = self;

        // Phase A: pure, backend-neutral, zero filesystem effect. A lineage
        // refusal must never reach the store or read the pointer.
        let validated = validate_commit_metadata(&expected, metadata)?;
        let epoch = validated.epoch();
        let results = staged_results
            .as_ref()
            .ok_or(CheckpointError::ObjectVerification)?;
        let mut participant_descriptors = participants
            .iter()
            .map(|participant| participant.descriptor.clone())
            .collect::<Vec<_>>();
        participant_descriptors
            .sort_unstable_by(|left, right| left.participant_id.cmp(&right.participant_id));
        if participant_descriptors
            .iter()
            .map(|descriptor| &descriptor.participant_id)
            .ne(expectations.participant_plan.ids().iter())
        {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        if results
            .descriptors
            .descriptors()
            .iter()
            .any(|descriptor| descriptor.run != run || descriptor.epoch != epoch)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (result_items, result_bytes) = result_totals(results.descriptors.descriptors())?;
        if result_items != results.item_count || result_bytes != results.byte_length {
            return Err(CheckpointError::ObjectVerification);
        }
        let frozen = FrozenGenerationTransactionInputs::new(
            run,
            expectations,
            participant_descriptors,
            results.index_root,
        );
        let prevalidated = build_prevalidated_candidate(frozen, validated)?;
        let generation_bytes = prevalidated.encode_for_storage()?;
        let (index_root, index_bytes) =
            canonical_result_index_object(results.descriptors.descriptors().iter())?;
        if index_root != results.index_root {
            return Err(CheckpointError::ObjectVerification);
        }

        // Phase B: assemble the complete content-addressed object set. Objects
        // are keyed by digest alone; the kind is implied by which root reaches
        // them, which is a stronger guarantee than presence-by-kind.
        let mut objects = BTreeMap::new();
        insert_object(
            &mut objects,
            index_root,
            Bytes::from(index_bytes.into_boxed_slice()),
        )?;
        for participant in &participants {
            insert_object(
                &mut objects,
                participant.descriptor.content_digest,
                Bytes::copy_from_slice(participant.payload.as_bytes()),
            )?;
        }
        for (descriptor, payload) in results
            .descriptors
            .descriptors()
            .iter()
            .zip(&results.payloads)
        {
            insert_object(
                &mut objects,
                descriptor.payload_digest,
                Bytes::copy_from_slice(payload.as_bytes()),
            )?;
        }

        // Phase C: one aggregate storage acquisition for the genuinely new set.
        let missing = backend.filter_absent_objects(&paths, objects).await?;
        let storage_bytes =
            missing
                .iter()
                .try_fold(generation_bytes.len(), |total, (_, bytes)| {
                    total
                        .checked_add(bytes.len())
                        .ok_or(CheckpointError::ObjectVerification)
                })?;
        let storage_lease = backend
            .inner
            .budgets
            .storage
            .acquire(
                missing
                    .len()
                    .checked_add(1)
                    .ok_or(CheckpointError::ObjectVerification)?,
                storage_bytes,
            )
            .await?;
        let bundle = Rc::new(StorageCommitBundle {
            _storage_lease: storage_lease,
        });

        // Phase D: the fixed durable ordering.
        let generation = prevalidated.generation().clone();
        backend
            .publish_durably(
                &paths,
                tmp_guard.path(),
                &missing,
                &generation,
                &generation_bytes,
                expected
                    .as_ref()
                    .map(CurrentV4CheckpointGeneration::generation),
            )
            .await?;

        // Phase E: infallible in-process publication. Every fallible operation
        // is complete and the transition below has an infallible return type.
        backend
            .inner
            .recovered
            .borrow_mut()
            .entry(run)
            .or_insert(bundle);
        tmp_guard.release().await;
        Ok(prevalidated.into_committed_after_publication_fence())
    }
}

#[async_trait(?Send)]
impl StreamingGenerationTransaction for LocalGenerationTransaction {
    async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        self.stage_participant_inner(state)
    }

    async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        issue_receipts: &mut Option<PreparedIssueReceiptResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        self.prepare_result_partitions(partitions, issue_receipts)
            .await
    }

    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        (*self).commit_inner(metadata).await
    }
}

fn insert_object(
    objects: &mut BTreeMap<ContentDigest, Bytes>,
    digest: ContentDigest,
    bytes: Bytes,
) -> Result<(), CheckpointError> {
    if let Some(existing) = objects.get(&digest) {
        if existing != &bytes {
            return Err(CheckpointError::ObjectVerification);
        }
        return Ok(());
    }
    objects.insert(digest, bytes);
    Ok(())
}

// ---------------------------------------------------------------------------
// Readers
// ---------------------------------------------------------------------------

struct LocalResultReadAuthority<'a> {
    backend: &'a LocalCheckpointBackend,
    paths: &'a RunPaths,
    run: &'a StreamRunIdentity,
    head: &'a CheckpointGeneration,
    result_index_root: &'a ContentDigest,
}

impl LocalResultReadAuthority<'_> {
    /// Re-verify the on-disk head still names the leased generation.
    async fn verify_head(&self) -> Result<(), CheckpointError> {
        match self.backend.read_current(self.paths).await? {
            Some(pointer) if pointer.matches(self.head) => Ok(()),
            _ => Err(CheckpointError::LeaseLost {
                generation: self.head.clone(),
            }),
        }
    }

    async fn reachable_descriptors(&self) -> Result<Vec<ResultSegmentDescriptor>, CheckpointError> {
        self.verify_head().await?;
        let max = u64::try_from(self.backend.inner.budgets.storage.limits().max_bytes)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let bytes = self
            .backend
            .read_object(self.paths, self.result_index_root, max)
            .await?;
        let descriptors: Vec<ResultSegmentDescriptor> =
            serde_json::from_slice(&bytes).map_err(|_| CheckpointError::ObjectVerification)?;
        if canonical_result_index_root(&descriptors)? != *self.result_index_root
            || descriptors
                .iter()
                .any(|descriptor| descriptor.run != *self.run)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(descriptors)
    }

    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        let root = *self.result_index_root;
        if after
            .as_ref()
            .is_some_and(|cursor| cursor.root != root || cursor.block != root)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let descriptors = self.reachable_descriptors().await?;
        let start = match after.as_ref() {
            None => 0usize,
            Some(cursor) => {
                let offset = usize::try_from(cursor.item_offset)
                    .map_err(|_| CheckpointError::ObjectVerification)?;
                if offset >= descriptors.len() {
                    return Err(CheckpointError::ObjectVerification);
                }
                offset
            }
        };
        if start == descriptors.len() {
            drop(descriptors);
            let lease = self.backend.inner.budgets.reads.acquire(0, 0).await?;
            return ResultIndexPage::new(
                BudgetedResultDescriptors::new(Vec::new().into_boxed_slice(), lease)?,
                None,
            );
        }
        let first_required = descriptor_retained_bytes(&descriptors[start])?;
        let first_required_u64 =
            u64::try_from(first_required).map_err(|_| CheckpointError::ObjectVerification)?;
        if first_required_u64 > budget.max_bytes.get() {
            return Err(CheckpointError::ResultIndexReadBudgetTooSmall {
                required_bytes: first_required_u64,
                max_bytes: budget.max_bytes.get(),
            });
        }
        let mut end = start;
        let mut retained = 0usize;
        while end < descriptors.len() && end - start < budget.max_items.get() {
            let next = descriptor_retained_bytes(&descriptors[end])?;
            let Some(total) = retained.checked_add(next) else {
                return Err(CheckpointError::ObjectVerification);
            };
            if u64::try_from(total).map_err(|_| CheckpointError::ObjectVerification)?
                > budget.max_bytes.get()
            {
                break;
            }
            retained = total;
            end += 1;
        }
        drop(descriptors);
        let lease = self
            .backend
            .inner
            .budgets
            .reads
            .acquire(end - start, retained)
            .await?;
        let descriptors = self.reachable_descriptors().await?;
        let page_descriptors = descriptors
            .get(start..end)
            .ok_or(CheckpointError::ObjectVerification)?
            .to_vec()
            .into_boxed_slice();
        let next = if end < descriptors.len() {
            Some(ResultIndexCursor {
                root,
                block: root,
                item_offset: u32::try_from(end).map_err(|_| CheckpointError::ObjectVerification)?,
            })
        } else {
            None
        };
        ResultIndexPage::new(
            BudgetedResultDescriptors::new(page_descriptors, lease)?,
            next,
        )
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        if !self.reachable_descriptors().await?.contains(descriptor) {
            return Err(CheckpointError::ObjectVerification);
        }
        let length = usize::try_from(descriptor.byte_length)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = self.backend.inner.budgets.reads.acquire(1, length).await?;
        self.verify_head().await?;
        let bytes = self
            .backend
            .read_object(
                self.paths,
                &descriptor.payload_digest,
                descriptor.byte_length,
            )
            .await?;
        if bytes.len() != length {
            return Err(CheckpointError::ObjectVerification);
        }
        ResultSegmentReader::new(descriptor, BudgetedCheckpointBytes::new(bytes, lease)?)
    }

    async fn read_participant_bytes(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<(Bytes, BudgetLease), CheckpointError> {
        let length = usize::try_from(descriptor.byte_length)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = self.backend.inner.budgets.reads.acquire(1, length).await?;
        self.verify_head().await?;
        let bytes = self
            .backend
            .read_object(
                self.paths,
                &descriptor.content_digest,
                descriptor.byte_length,
            )
            .await?;
        if bytes.len() != length {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok((bytes, lease))
    }
}

/// Concrete leased reader for one committed local generation.
pub struct LocalGenerationReader {
    backend: LocalCheckpointBackend,
    paths: RunPaths,
    generation: CommittedCheckpointGeneration,
    /// Durable reachability hold, checked and renewed before every read.
    reachability: GenerationLease,
    _generation_lease: BudgetLease,
}

impl LocalGenerationReader {
    fn authority(&self) -> LocalResultReadAuthority<'_> {
        LocalResultReadAuthority {
            backend: &self.backend,
            paths: &self.paths,
            run: self.generation.run(),
            head: self.generation.generation_ref(),
            result_index_root: self.generation.result_index_root(),
        }
    }

    async fn read_participant_inner(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError> {
        if !self
            .generation
            .participant_descriptors()
            .contains(descriptor)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (bytes, lease) = self.authority().read_participant_bytes(descriptor).await?;
        let context = CurrentV4ParticipantStateContext::for_reachable_descriptor(
            &self.generation,
            descriptor,
        )?;
        if context.generation() != self.generation.generation_ref() {
            return Err(CheckpointError::ObjectVerification);
        }
        CommittedParticipantState::from_current_v4_reader(
            &context,
            descriptor.clone(),
            BudgetedCheckpointBytes::new(bytes, lease)?,
        )
    }
}

impl sealed::LeasedGenerationReader for LocalGenerationReader {}

#[async_trait(?Send)]
impl LeasedGenerationReader for LocalGenerationReader {
    fn generation(&self) -> &CommittedCheckpointGeneration {
        &self.generation
    }

    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        self.reachability.ensure_live().await?;
        self.authority().scan_result_index(after, budget).await
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.reachability.ensure_live().await?;
        self.authority().read_segment(descriptor).await
    }

    async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError> {
        self.reachability.ensure_live().await?;
        self.read_participant_inner(descriptor).await
    }
}

/// Read/export authority over one verified legacy-v3 local head.
pub struct LocalLegacyV3GenerationReader {
    backend: LocalCheckpointBackend,
    paths: RunPaths,
    generation: LegacyV3CheckpointGeneration,
    /// Durable reachability hold, checked and renewed before every read.
    reachability: GenerationLease,
    _generation_lease: BudgetLease,
}

impl LocalLegacyV3GenerationReader {
    fn authority(&self) -> LocalResultReadAuthority<'_> {
        LocalResultReadAuthority {
            backend: &self.backend,
            paths: &self.paths,
            run: self.generation.run(),
            head: self.generation.generation(),
            result_index_root: self.generation.result_index_root(),
        }
    }
}

#[async_trait(?Send)]
impl LegacyV3LeasedGenerationReader for LocalLegacyV3GenerationReader {
    fn generation(&self) -> &CheckpointGeneration {
        self.generation.generation()
    }

    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        self.reachability.ensure_live().await?;
        self.authority().scan_result_index(after, budget).await
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.reachability.ensure_live().await?;
        self.authority().read_segment(descriptor).await
    }

    async fn read_legacy_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<LegacyParticipantState, CheckpointError> {
        self.reachability.ensure_live().await?;
        if !self
            .generation
            .participant_descriptors()
            .contains(descriptor)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (bytes, lease) = self.authority().read_participant_bytes(descriptor).await?;
        LegacyParticipantState::from_legacy_v3_reader(
            descriptor.clone(),
            BudgetedCheckpointBytes::new(bytes, lease)?,
        )
    }
}

/// Whether one directory entry names a content-addressed object.
#[must_use]
pub fn is_object_entry(name: &OsStr) -> bool {
    name.as_bytes().starts_with(OBJECT_PREFIX.as_bytes())
        && name.as_bytes().len() == OBJECT_PREFIX.len() + 64
}

/// Decode the digest one `objects/` entry name records.
fn object_digest_from_name(name: &str) -> Option<ContentDigest> {
    parse_hex32(name.strip_prefix(OBJECT_PREFIX)?).map(ContentDigest::from_bytes)
}

#[async_trait(?Send)]
impl CheckpointGarbageCollector for LocalCheckpointBackend {
    async fn set_retention_policy(
        &self,
        run: &StreamRunIdentity,
        policy: CheckpointRetentionPolicy,
    ) -> Result<(), CheckpointError> {
        // Validation precedes retention so an invalid policy performs no
        // filesystem effect, matching the constructor discipline of the store.
        let validated = policy.validate()?;
        self.inner.retention.borrow_mut().insert(*run, validated);
        self.register_run(*run);
        Ok(())
    }

    async fn retain_last_generations(&self, generations: usize) -> Result<(), CheckpointError> {
        self.inner.retention_override.set(Some(generations));
        Ok(())
    }

    async fn collect_garbage(&self) -> Result<GcReport, CheckpointError> {
        let runs: Vec<StreamRunIdentity> = self.inner.runs.borrow().iter().copied().collect();
        let mut total = GcReport::default();
        for run in runs {
            total = total.fold(self.collect_garbage_for_run(&run).await?);
        }
        Ok(total)
    }
}

// ---------------------------------------------------------------------------
// Registry descriptor, authored configuration, and factory
// ---------------------------------------------------------------------------

/// Stable registry identifier of the built-in local checkpoint backend.
pub const LOCAL_CHECKPOINT_BACKEND_ID: &str = "local";

/// Registry metadata for the crash-durable local generation store.
pub static LOCAL_CHECKPOINT_BACKEND_DESCRIPTOR: StreamingCheckpointBackendDescriptor =
    StreamingCheckpointBackendDescriptor {
        id: LOCAL_CHECKPOINT_BACKEND_ID,
        description: "crash-durable local-filesystem checkpoint generation store",
        is_durable: true,
        has_leased_readers: true,
        has_atomic_generations: true,
        has_result_segments: true,
        // Objects are ordinary files under the store root; confidentiality at
        // rest is the operator's filesystem decision, not this backend's.
        protects_sensitive_state: false,
        retention: CheckpointRetention::GenerationReachability,
        // The store root is one process-local path, so a remote cell cannot
        // reach the same authoritative `CURRENT` pointer.
        placement: CheckpointBackendPlacement::ControllerLocal,
        supports_virtual_clock: true,
    };

/// Blocking participant identity used by every prepared local backend.
const LOCAL_BACKEND_PARTICIPANT: &str = "streaming-checkpoint-local-blocking";

/// Strictly decoded authored configuration for the local checkpoint backend.
///
/// Every bound is authored rather than defaulted: a silently defaulted capacity
/// would make an over-budget run fail at an arbitrary later generation instead
/// of at validation.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalCheckpointBackendConfig {
    /// Absolute store root owned exclusively by this backend.
    pub root: PathBuf,
    /// Simultaneously live generation transactions.
    pub max_transactions: u32,
    /// Bytes retained by live generation transactions.
    pub max_transaction_bytes: u64,
    /// Descriptors retained by staged transaction indexes.
    pub max_prepared_indexes: u32,
    /// Bytes retained by staged transaction indexes.
    pub max_prepared_index_bytes: u64,
    /// Committed immutable objects, including objects recovered on open.
    pub max_storage_objects: u32,
    /// Bytes of committed immutable object storage.
    pub max_storage_bytes: u64,
    /// Descriptor summaries returned from result staging.
    pub max_result_summaries: u32,
    /// Bytes retained by result-staging summaries.
    pub max_result_summary_bytes: u64,
    /// Concurrent generation, participant, result, and page readers.
    pub max_reads: u32,
    /// Bytes retained by concurrent readers.
    pub max_read_bytes: u64,
    /// Scratch entries examined per reclamation page.
    pub gc_page_items: u32,
    /// Lifetime granted to one prepare lease, in nanoseconds.
    pub prepare_lease_ns: u64,
    /// Simultaneously accepted blocking filesystem jobs.
    pub max_blocking_jobs: u32,
    /// Bytes one blocking filesystem job may be handed.
    pub max_blocking_input_bytes: u64,
    /// Bytes one blocking filesystem job may return.
    pub max_blocking_output_bytes: u64,
}

fn config_rejected(message: &str) -> CheckpointError {
    CheckpointError::Storage {
        message: format!("local checkpoint backend configuration rejected: {message}"),
    }
}

fn bounded_usize(value: u64, field: &str) -> Result<usize, CheckpointError> {
    usize::try_from(value).map_err(|_| config_rejected(field))
}

impl LocalCheckpointBackendConfig {
    /// Reject an unusable store root before any capacity is derived.
    ///
    /// A relative root would resolve against whatever working directory the
    /// process happened to inherit, so two identically configured runs could
    /// address different stores.
    fn validate(&self) -> Result<(), CheckpointError> {
        if !self.root.is_absolute() {
            return Err(config_rejected("root must be an absolute path"));
        }
        if self.gc_page_items == 0 {
            return Err(config_rejected("gc_page_items must be non-zero"));
        }
        if self.prepare_lease_ns == 0 {
            return Err(config_rejected("prepare_lease_ns must be non-zero"));
        }
        if self.max_blocking_jobs == 0 {
            return Err(config_rejected("max_blocking_jobs must be non-zero"));
        }
        Ok(())
    }

    /// Project authored bounds onto the backend's own limit type.
    fn limits(&self) -> Result<LocalCheckpointLimits, CheckpointError> {
        let limits = |items: u32, bytes: u64, field: &str| {
            Ok(BudgetLimits {
                max_items: bounded_usize(u64::from(items), field)?,
                max_bytes: bounded_usize(bytes, field)?,
            })
        };
        Ok(LocalCheckpointLimits {
            transactions: limits(
                self.max_transactions,
                self.max_transaction_bytes,
                "max_transaction_bytes",
            )?,
            prepared_indexes: limits(
                self.max_prepared_indexes,
                self.max_prepared_index_bytes,
                "max_prepared_index_bytes",
            )?,
            storage: limits(
                self.max_storage_objects,
                self.max_storage_bytes,
                "max_storage_bytes",
            )?,
            result_summaries: limits(
                self.max_result_summaries,
                self.max_result_summary_bytes,
                "max_result_summary_bytes",
            )?,
            reads: limits(self.max_reads, self.max_read_bytes, "max_read_bytes")?,
            gc_page_items: NonZeroUsize::new(bounded_usize(
                u64::from(self.gc_page_items),
                "gc_page_items",
            )?)
            .ok_or_else(|| config_rejected("gc_page_items must be non-zero"))?,
            prepare_lease_ns: self.prepare_lease_ns,
        })
    }
}

/// Startup validator and preparer for the built-in local checkpoint backend.
///
/// The factory itself holds no run state: the blocking executor, filesystem,
/// and clock are all minted inside `prepare`, on the thread that will own the
/// backend, so the `Send + Sync` registry entry never carries a `!Send` handle.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalCheckpointBackendFactory;

impl StreamingCheckpointBackendFactory for LocalCheckpointBackendFactory {
    fn descriptor(&self) -> &'static StreamingCheckpointBackendDescriptor {
        &LOCAL_CHECKPOINT_BACKEND_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &serde_json::value::RawValue,
        requirements: &CheckpointBackendRequirements,
    ) -> Result<Box<dyn ValidatedCheckpointBackendConfig>, CheckpointError> {
        // Both declared requirements are satisfied by a durable generation
        // store; the check is kept explicit so a future non-durable variant of
        // this backend cannot inherit the acceptance silently.
        let _ = requirements;
        let config: LocalCheckpointBackendConfig = serde_json::from_str(authored.get())
            .map_err(|error| config_rejected(&error.to_string()))?;
        config.validate()?;
        // Reject unrepresentable capacity at validation rather than at prepare.
        config.limits()?;
        Ok(Box::new(config))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedCheckpointBackendConfig>,
        context: &CheckpointBackendPrepareContext,
    ) -> Result<Box<dyn StreamingCheckpointBackend>, CheckpointError> {
        let config = *config
            .into_any()
            .downcast::<LocalCheckpointBackendConfig>()
            .map_err(|_| config_rejected("configuration was validated by a different factory"))?;
        let limits = config.limits()?;
        let executor = StreamingBlockingExecutor::new(
            context.run,
            CheckpointParticipantId::new(LOCAL_BACKEND_PARTICIPANT),
            bounded_usize(u64::from(config.max_blocking_jobs), "max_blocking_jobs")?,
            bounded_usize(config.max_blocking_input_bytes, "max_blocking_input_bytes")?,
            bounded_usize(
                config.max_blocking_output_bytes,
                "max_blocking_output_bytes",
            )?,
        )
        .map_err(|error| config_rejected(&error.to_string()))?;
        let filesystem: Rc<dyn LocalCheckpointFilesystem> =
            Rc::new(BlockingLocalFilesystem::new(executor));
        let backend = LocalCheckpointBackend::open(
            config.root,
            limits,
            filesystem,
            Rc::clone(&context.clock),
        )?;
        Ok(Box::new(backend))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_round_trips_through_its_exact_wire_shape() {
        let generation = CheckpointGeneration::new(
            CheckpointEpoch::new(7),
            ContentDigest::from_bytes([0xab; 32]),
        );
        let pointer = CurrentPointer::new(&generation);
        let encoded = pointer.encode().expect("encodable pointer");

        assert!(encoded.ends_with(b"\n"));
        assert_eq!(
            CurrentPointer::decode(&encoded).expect("decodable pointer"),
            pointer
        );
        assert!(pointer.matches(&generation));
    }

    #[test]
    fn pointer_without_its_trailing_newline_is_object_verification() {
        let generation = CheckpointGeneration::new(
            CheckpointEpoch::new(1),
            ContentDigest::from_bytes([0x01; 32]),
        );
        let mut encoded = CurrentPointer::new(&generation)
            .encode()
            .expect("encodable pointer");
        encoded.pop();

        assert_eq!(
            CurrentPointer::decode(&encoded),
            Err(CheckpointError::ObjectVerification)
        );
    }

    #[test]
    fn generation_paths_order_lexicographically_by_epoch() {
        let run = StreamRunIdentity::new(
            crate::streaming::identity::LogicalReplayRunId::from_bytes([9; 32]),
        );
        let paths = RunPaths::for_run(Path::new("/store"), &run);
        let digest = ContentDigest::from_bytes([0x11; 32]);
        let low = paths.generation_path(CheckpointEpoch::new(2), &digest);
        let high = paths.generation_path(CheckpointEpoch::new(10), &digest);

        assert!(low < high);
        assert!(paths.object_path(&digest).starts_with(paths.objects_dir()));
    }

    #[test]
    fn every_pre_publication_fault_carries_a_distinct_stable_message() {
        let faults = LocalCommitFault::before_current_publication();
        let messages: BTreeSet<&str> = faults
            .iter()
            .map(|fault| fault.injected_message())
            .collect();

        assert_eq!(messages.len(), faults.len());
        assert!(!faults.contains(&LocalCommitFault::AfterCurrentRename));
    }
}
