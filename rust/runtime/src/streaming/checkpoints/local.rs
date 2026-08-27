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
            BudgetedCheckpointBytes, CheckpointBackendBudgetKind, CheckpointEpoch,
            CheckpointError, CheckpointGeneration,
            CommittedCheckpointGeneration, CommittedParticipantState,
            CurrentV4ParticipantStateContext, DecodedCheckpointGeneration, LegacyParticipantState,
            LegacyV3CheckpointGeneration, ParticipantStateDescriptor, PreparedParticipantState,
            StreamRunIdentity, decode_versioned_checkpoint_generation,
        },
        checkpoint_backend::{
            CheckpointCommitMetadata, CheckpointGenerationExpectations, CurrentV4CheckpointGeneration,
            FrozenGenerationTransactionInputs, LeasedCheckpointGeneration, LeasedGenerationReader,
            LegacyV3LeasedGenerationReader, StreamingCheckpointBackend,
            StreamingGenerationTransaction, build_prevalidated_candidate, sealed,
            validate_commit_metadata,
        },
        checkpoints::budget::{BackendBudget, map_budget_error},
        identity::ContentDigest,
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

/// Lowercase hex of one 32-byte digest.
fn hex32(bytes: &[u8; 32]) -> String {
    let mut text = String::with_capacity(64);
    for byte in bytes {
        // Two lowercase nibbles per byte; `write!` on a `String` cannot fail.
        text.push(char::from_digit(u32::from(byte >> 4), 16).unwrap_or('0'));
        text.push(char::from_digit(u32::from(byte & 0x0f), 16).unwrap_or('0'));
    }
    text
}

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
    let mut buffer = Vec::with_capacity(
        usize::try_from(metadata.len()).map_err(|_| {
            std::io::Error::new(ErrorKind::InvalidData, "unrepresentable object length")
        })?,
    );
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
        if after.as_deref().is_some_and(|cursor| name.as_str() <= cursor) {
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
        self.run(0, move || create_private_dir_blocking(&path)).await
    }

    async fn write_new(&self, path: &Path, bytes: &[u8]) -> Result<bool, CheckpointError> {
        let path = path.to_path_buf();
        let length = bytes.len();
        let bytes = bytes.to_vec();
        self.run(length, move || write_new_blocking(&path, bytes))
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
            .run(usize::try_from(max_bytes).unwrap_or(usize::MAX), move || {
                read_optional_blocking(&path, max_bytes)
            })
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
        // Provenance is advisory; the advisory lock is the authority, so an
        // existing record from a released holder is simply overwritten in place
        // by removing it first.
        self.fs().remove_private_subtree(path).await.ok();
        self.fs().write_new(path, &bytes).await?;
        Ok(())
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
        let expires_ns = self
            .inner
            .clock
            .now_ns()
            .saturating_add(i64::try_from(self.inner.limits.prepare_lease_ns).unwrap_or(i64::MAX));
        self.write_lease_record(&lease_path, run, LeaseKind::Prepare, writer.holder, expires_ns)
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
                .checked_add(usize::try_from(length).map_err(|_| {
                    CheckpointError::ObjectVerification
                })?)
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
            match self.classify_transaction_lease(&paths, name, now_ns).await? {
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
            next: page.next.map(|after| TmpReclaimCursor { after: Some(after) }),
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
        let storage_bytes = missing
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
            .read_object(self.paths, &descriptor.payload_digest, descriptor.byte_length)
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
            .read_object(self.paths, &descriptor.content_digest, descriptor.byte_length)
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
        self.authority().scan_result_index(after, budget).await
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.authority().read_segment(descriptor).await
    }

    async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError> {
        self.read_participant_inner(descriptor).await
    }
}

/// Read/export authority over one verified legacy-v3 local head.
pub struct LocalLegacyV3GenerationReader {
    backend: LocalCheckpointBackend,
    paths: RunPaths,
    generation: LegacyV3CheckpointGeneration,
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
        self.authority().scan_result_index(after, budget).await
    }

    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.authority().read_segment(descriptor).await
    }

    async fn read_legacy_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<LegacyParticipantState, CheckpointError> {
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
        let run = StreamRunIdentity::new(crate::streaming::identity::LogicalReplayRunId::from_bytes(
            [9; 32],
        ));
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
