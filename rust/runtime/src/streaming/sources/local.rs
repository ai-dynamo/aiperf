// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable local finite, follow, and reference-manifest streaming source.
//!
//! Discovery acquires the authored root exactly once through a private
//! no-follow descriptor and never re-resolves the root by name afterwards.
//! Follow mode accepts publish-by-rename only and parks on an `inotify`
//! descriptor through `AsyncFd`; it uses no timer and no wall clock. Every
//! `openat`, `fstatat`, `pread`, and whole-file digest runs on
//! [`StreamingBlockingExecutor`].
//!
//! Three discovery policies share one registered source identifier:
//!
//! - [`LocalSourceMode::Finite`] scans once, digests every member at open, and
//!   seals immediately.
//! - [`LocalSourceMode::Follow`] freezes a `(dev, ino, mtime, size)` generation
//!   identity at discovery, proves it again on the acquisition descriptor, and
//!   seals only when an authored marker name is published.
//! - [`LocalSourceMode::Reference`] reads a pre-built immutable JSONL manifest,
//!   binds identity from its declared `(path, size, digest)` triples, and
//!   verifies the declared digest while acquiring.
//!
//! Acquisition, identity verification, budgeting, checkpointing, and the
//! reliability path are mode-independent.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    ffi::{CStr, CString, OsString},
    num::NonZeroUsize,
    os::{
        fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd},
        unix::ffi::{OsStrExt, OsStringExt},
    },
    path::{Component, Path, PathBuf},
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;
use tokio::io::unix::AsyncFd;

use crate::streaming::{
    blocking::{
        BlockingWorkBudget, BlockingWorkClass, BlockingWorkError, StreamingBlockingExecutor,
    },
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::{
        AcquisitionFailureCode, OrdinaryStreamingFailure, OrdinaryStreamingIssue,
        SourceFailureCode, StreamSourceError, StreamingInputDomainIdentity, StreamingIssueClass,
        StreamingIssueReporterHandle,
    },
    identity::{ContentDigest, ImmutableObjectIdentity},
    source::{
        AcquiredPartition, AcquisitionBudget, AcquisitionMemoryLease, BudgetedSourceChunk,
        OpenedStreamingDatasetSource, PartitionAccessKind, PartitionAccessRequest,
        PreparedStreamingDatasetSource, SequentialSourceChunk, SourceEvent, SourceFrontier,
        SourcePartition, SourcePartitionContent, SourceSeal, SourceSnapshotReceipt,
        StreamingDatasetSource, StreamingDatasetSourceFactory, StreamingResumeGranularity,
        StreamingSeekableLocalSnapshot, StreamingSequentialReader, StreamingSourceDescriptor,
        StreamingSourceMode, StreamingSourceOrdering, StreamingSourcePlacement,
        StreamingSourcePrepareContext, StreamingSourceRetention, StreamingStopReceiver,
        ValidatedStreamingSourceConfig,
    },
    unit::SourcePosition,
};

/// Registry identifier of the built-in local source.
pub const LOCAL_SOURCE_ID: &str = "local";
/// Stable schema identity of the local source checkpoint payload.
pub const LOCAL_SOURCE_SCHEMA_ID: &str = "aiperf.streaming.source.local";
/// Current local source checkpoint schema version.
pub const LOCAL_SOURCE_SCHEMA_VERSION: u32 = 1;

/// Fixed prefix of one encoded [`LocalSourceCursor`].
const CURSOR_HEADER_BYTES: usize = 56;
/// Bound on the bytes one directory scan may reserve from the blocking owner.
const SCAN_OUTPUT_BYTES_PER_ENTRY: usize = 256;

static LOCAL_SOURCE_DESCRIPTOR: StreamingSourceDescriptor = StreamingSourceDescriptor {
    id: LOCAL_SOURCE_ID,
    description: "Immutable local finite, follow, and reference-manifest partitions",
    modes: &[StreamingSourceMode::Finite, StreamingSourceMode::Follow],
    access: &[
        PartitionAccessKind::Sequential,
        PartitionAccessKind::SeekableLocal,
    ],
    ordering: StreamingSourceOrdering::Partition,
    resume: &[
        StreamingResumeGranularity::Partition,
        StreamingResumeGranularity::Byte,
    ],
    has_event_time: false,
    has_stable_record_ids: false,
    retention: StreamingSourceRetention::ResumeRootReachability,
    placement: StreamingSourcePlacement::ControllerOnly,
    supports_virtual_clock: true,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Strictly authored local source configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub struct LocalSourceConfig {
    /// Absolute directory acquired once through a no-follow descriptor.
    pub root: PathBuf,
    /// Inventory lifecycle selected for this stream.
    pub mode: LocalSourceMode,
    /// Exact file-name suffix accepted as a partition.
    #[serde(default = "default_suffix")]
    pub suffix: String,
    /// Inclusive upper bound on one partition's immutable length.
    pub max_partition_bytes: u64,
    /// Inclusive bound on names retained by one discovery scan.
    pub max_scan_entries: u32,
    /// Inclusive bound on bytes read per acquisition chunk.
    #[serde(default = "default_chunk_bytes")]
    pub max_chunk_bytes: u32,
    /// Bounded reopen attempts before an ordinary partition fault is reported.
    #[serde(default = "default_open_attempts")]
    pub max_open_attempts: u32,
}

/// Discovery policy selected before any filesystem effect.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields, rename_all = "snake_case")]
pub enum LocalSourceMode {
    /// One byte-sorted scan, then an immediate seal.
    Finite,
    /// Publish-by-rename discovery that parks while quiet.
    Follow {
        /// Root-relative name whose publication seals the inventory.
        #[serde(default)]
        seal_marker: Option<String>,
        /// Whether `IN_CLOSE_WRITE` also wakes discovery.
        #[serde(default)]
        accepts_close_write: bool,
    },
    /// Pre-indexed immutable manifest naming every partition in order.
    Reference {
        /// Root-relative JSONL manifest path.
        manifest: PathBuf,
        /// Inclusive bound on the manifest's own byte length.
        max_manifest_bytes: u64,
    },
}

impl LocalSourceMode {
    const fn tag(&self) -> u8 {
        match self {
            Self::Finite => 0,
            Self::Follow { .. } => 1,
            Self::Reference { .. } => 2,
        }
    }
}

const fn default_chunk_bytes() -> u32 {
    64 * 1024
}

const fn default_open_attempts() -> u32 {
    3
}

fn default_suffix() -> String {
    ".jsonl".to_owned()
}

impl LocalSourceConfig {
    fn validate(&self) -> Result<(), StreamSourceError> {
        if !self.root.is_absolute()
            || self.suffix.is_empty()
            || self.suffix.contains('/')
            || self.max_partition_bytes == 0
            || self.max_scan_entries == 0
            || self.max_chunk_bytes == 0
            || self.max_open_attempts == 0
        {
            return Err(discovery_error());
        }
        match &self.mode {
            LocalSourceMode::Reference {
                manifest,
                max_manifest_bytes,
            } => {
                if *max_manifest_bytes == 0 || !is_valid_relative(manifest) {
                    return Err(discovery_error());
                }
                Ok(())
            }
            LocalSourceMode::Follow {
                seal_marker: Some(marker),
                ..
            } if marker.is_empty() || marker.contains('/') => Err(discovery_error()),
            _ => Ok(()),
        }
    }

    fn seal_marker(&self) -> Option<&str> {
        match &self.mode {
            LocalSourceMode::Follow { seal_marker, .. } => seal_marker.as_deref(),
            _ => None,
        }
    }
}

const fn discovery_error() -> StreamSourceError {
    StreamSourceError::source(SourceFailureCode::Discovery)
}

const fn snapshot_error() -> StreamSourceError {
    StreamSourceError::source(SourceFailureCode::Snapshot)
}

/// Reject absolute, escaping, or non-normal root-relative paths.
fn is_valid_relative(path: &Path) -> bool {
    if path.is_absolute() || path.as_os_str().is_empty() {
        return false;
    }
    path.components().all(|component| match component {
        Component::Normal(name) => !name.as_bytes().contains(&0),
        _ => false,
    })
}

fn u32_to_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

// ---------------------------------------------------------------------------
// Identity derivation
// ---------------------------------------------------------------------------

fn update_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// Immutable `(dev, ino, mtime, size)` generation observed for one name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct FileGeneration {
    dev: u64,
    ino: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    size_bytes: u64,
}

/// Authority the entry's immutable identity is derived from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IdentityBasis {
    /// Whole-object BLAKE3, computed at open or declared by a manifest.
    Content([u8; 32]),
    /// Stat metadata frozen at discovery and proven again at acquisition.
    Generation(FileGeneration),
}

/// One discovered immutable partition candidate.
#[derive(Clone, Debug)]
struct InventoryEntry {
    relative_path: PathBuf,
    size_bytes: u64,
    /// Position the entry would occupy in an order-authoritative inventory.
    ordinal: u64,
    basis: IdentityBasis,
    /// Generation proven again on the acquisition descriptor, when frozen.
    generation: Option<FileGeneration>,
    /// Manifest-declared digest the streamed bytes must reproduce.
    expected_digest: Option<[u8; 32]>,
}

impl InventoryEntry {
    fn identity(&self, stream_identity: &ContentDigest) -> ImmutableObjectIdentity {
        let mut hasher = blake3::Hasher::new();
        match self.basis {
            IdentityBasis::Content(digest) => {
                update_field(&mut hasher, b"aiperf.stream.local.content.v1");
                update_field(&mut hasher, stream_identity.as_bytes());
                update_field(&mut hasher, self.relative_path.as_os_str().as_bytes());
                update_field(&mut hasher, &self.size_bytes.to_le_bytes());
                update_field(&mut hasher, &digest);
            }
            IdentityBasis::Generation(generation) => {
                update_field(&mut hasher, b"aiperf.stream.local.generation.v1");
                update_field(&mut hasher, stream_identity.as_bytes());
                update_field(&mut hasher, self.relative_path.as_os_str().as_bytes());
                update_field(&mut hasher, &generation.dev.to_le_bytes());
                update_field(&mut hasher, &generation.ino.to_le_bytes());
                update_field(&mut hasher, &generation.mtime_sec.to_le_bytes());
                update_field(&mut hasher, &generation.mtime_nsec.to_le_bytes());
                update_field(&mut hasher, &generation.size_bytes.to_le_bytes());
            }
        }
        ImmutableObjectIdentity::from_bytes(*hasher.finalize().as_bytes())
    }
}

// ---------------------------------------------------------------------------
// Bounded syscall helpers
// ---------------------------------------------------------------------------

/// Result of one bounded syscall closure: either a value or a raw `errno`.
type SyscallOutcome<T> = Result<T, i32>;

fn last_errno() -> i32 {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// Duplicate a descriptor so a blocking closure owns what it reads.
///
/// The closure never borrows a caller-owned descriptor, so dropping the caller
/// future can never close a descriptor a detached blocking job still holds.
fn dup_owned(fd: RawFd) -> Result<OwnedFd, i32> {
    // SAFETY: `fd` is a live descriptor owned by the caller for this call, and
    // `F_DUPFD_CLOEXEC` returns a new, independently owned descriptor.
    let raw = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) };
    if raw < 0 {
        return Err(last_errno());
    }
    // SAFETY: a nonnegative `fcntl` result is a fresh descriptor moved once.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Open a fresh directory description for one bounded scan.
///
/// A `dup` would share the caller's file offset, so a second `readdir` over a
/// duplicated descriptor starts at the previous scan's EOF and observes an
/// empty directory. Every scan therefore opens its own description.
fn open_dir_snapshot(dir: RawFd) -> Result<OwnedFd, i32> {
    let dot = c".";
    // SAFETY: `dir` is a live directory descriptor owned by the caller for this
    // call, and a nonnegative result is a fresh descriptor moved once.
    let raw = unsafe {
        libc::openat(
            dir,
            dot.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC,
        )
    };
    if raw < 0 {
        return Err(last_errno());
    }
    // SAFETY: a nonnegative `openat` result is a fresh descriptor moved once.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

// `libc::stat` field widths vary by target and libc; the casts are identities
// on 64-bit glibc and widening conversions elsewhere.
#[allow(clippy::unnecessary_cast)]
fn generation_from_stat(stat: &libc::stat) -> FileGeneration {
    FileGeneration {
        dev: stat.st_dev as u64,
        ino: stat.st_ino as u64,
        mtime_sec: stat.st_mtime as i64,
        mtime_nsec: stat.st_mtime_nsec as i64,
        size_bytes: stat.st_size.max(0) as u64,
    }
}

fn is_regular(stat: &libc::stat) -> bool {
    stat.st_mode & libc::S_IFMT == libc::S_IFREG
}

fn c_path(path: &Path) -> Result<CString, i32> {
    CString::new(path.as_os_str().as_bytes()).map_err(|_| libc::EINVAL)
}

/// Open one root-relative name under an already-owned directory descriptor.
fn openat_nofollow(dir: &OwnedFd, name: &CStr) -> SyscallOutcome<OwnedFd> {
    // SAFETY: `dir` is live for the call, `name` is NUL-terminated, and the
    // returned nonnegative descriptor is immediately moved into `OwnedFd`.
    let raw = unsafe {
        libc::openat(
            dir.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if raw < 0 {
        return Err(last_errno());
    }
    // SAFETY: a nonnegative `openat` result is a fresh descriptor moved once.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

fn fstat_owned(fd: &OwnedFd) -> SyscallOutcome<libc::stat> {
    // SAFETY: `stat` is fully initialized by a successful `fstat`, and `fd` is
    // live for the duration of the call.
    let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
    if unsafe { libc::fstat(fd.as_raw_fd(), &raw mut stat) } < 0 {
        return Err(last_errno());
    }
    Ok(stat)
}

fn pread_bounded(fd: &OwnedFd, buffer: &mut [u8], offset: u64) -> SyscallOutcome<usize> {
    let mut filled = 0_usize;
    while filled < buffer.len() {
        let want = buffer.len() - filled;
        // SAFETY: the slice is valid for `want` bytes at `filled`, and `fd` is
        // live for the call.
        let read = unsafe {
            libc::pread(
                fd.as_raw_fd(),
                buffer[filled..].as_mut_ptr().cast(),
                want,
                (offset + filled as u64) as libc::off_t,
            )
        };
        if read < 0 {
            return Err(last_errno());
        }
        if read == 0 {
            break;
        }
        filled += read as usize;
    }
    Ok(filled)
}

/// Names and frozen generations retained by one bounded directory scan.
#[derive(Debug)]
struct ScanResult {
    entries: Vec<(OsString, FileGeneration)>,
    has_seal_marker: bool,
}

/// Read every accepted name in one directory, bounded by `max_scan_entries`.
fn scan_directory(
    dir: OwnedFd,
    suffix: Vec<u8>,
    seal_marker: Option<Vec<u8>>,
    max_entries: usize,
) -> SyscallOutcome<ScanResult> {
    let dir_fd = dir.as_raw_fd();
    let raw = dir.into_raw_fd();
    // SAFETY: `fdopendir` takes ownership of the descriptor; `closedir` below
    // is the only release path and runs on every return.
    let stream = unsafe { libc::fdopendir(raw) };
    if stream.is_null() {
        let error = last_errno();
        // SAFETY: `fdopendir` failed, so the descriptor is still owned here.
        unsafe { libc::close(raw) };
        return Err(error);
    }
    let mut entries = Vec::new();
    let mut has_seal_marker = false;
    let mut outcome = Ok(());
    loop {
        // SAFETY: `stream` is a live directory stream; a null result ends
        // iteration and the returned pointer is read before the next call.
        let record = unsafe { libc::readdir(stream) };
        if record.is_null() {
            break;
        }
        // SAFETY: `d_name` of a live `readdir` record is a NUL-terminated name.
        let name = unsafe { CStr::from_ptr((*record).d_name.as_ptr()) }.to_bytes();
        if name == b"." || name == b".." {
            continue;
        }
        if seal_marker.as_deref() == Some(name) {
            has_seal_marker = true;
            continue;
        }
        if name.len() <= suffix.len() || !name.ends_with(&suffix) {
            continue;
        }
        if entries.len() >= max_entries {
            continue;
        }
        let owned = OsString::from_vec(name.to_vec());
        let Ok(name_c) = CString::new(name.to_vec()) else {
            continue;
        };
        // SAFETY: `stat` is fully initialized by a successful `fstatat`, and
        // `dir_fd` remains live because `stream` owns it.
        let mut stat = unsafe { std::mem::zeroed::<libc::stat>() };
        let status = unsafe {
            libc::fstatat(
                dir_fd,
                name_c.as_ptr(),
                &raw mut stat,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if status < 0 {
            let error = last_errno();
            if error == libc::ENOENT {
                continue;
            }
            outcome = Err(error);
            break;
        }
        // A symlink or any non-regular entry is never an immutable partition.
        if !is_regular(&stat) {
            continue;
        }
        entries.push((owned, generation_from_stat(&stat)));
    }
    // SAFETY: `stream` is live and is closed exactly once here.
    unsafe { libc::closedir(stream) };
    outcome?;
    entries.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));
    Ok(ScanResult {
        entries,
        has_seal_marker,
    })
}

// ---------------------------------------------------------------------------
// Reference manifest
// ---------------------------------------------------------------------------

/// One strictly decoded immutable manifest record.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
struct ManifestRecordV1 {
    /// Root-relative partition path.
    path: PathBuf,
    /// Exact immutable length the acquired object must have.
    size_bytes: u64,
    /// Lowercase hex BLAKE3 digest of the complete immutable object.
    digest: String,
}

fn decode_hex32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64 {
        return None;
    }
    let mut bytes = [0_u8; 32];
    for (index, chunk) in value.as_bytes().chunks_exact(2).enumerate() {
        let high = (chunk[0] as char).to_digit(16)?;
        let low = (chunk[1] as char).to_digit(16)?;
        bytes[index] = (high * 16 + low) as u8;
    }
    Some(bytes)
}

// ---------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------

/// Discovery-policy-specific inventory of immutable partition candidates.
struct Inventory {
    ready: VecDeque<InventoryEntry>,
    /// Members frozen at open; the snapshot and restore authority.
    members: Vec<(PathBuf, ImmutableObjectIdentity)>,
    observed: BTreeSet<PathBuf>,
    is_complete: bool,
    next_ordinal: u64,
    /// Whether `ordinal` is a stable position a rescan can reproduce.
    is_ordinal_authoritative: bool,
    snapshot_digest: ContentDigest,
}

impl Inventory {
    async fn open(
        config: &LocalSourceConfig,
        stream_identity: &ContentDigest,
        root: &OwnedFd,
        executor: &StreamingBlockingExecutor,
    ) -> Result<Self, StreamSourceError> {
        let mut inventory = match &config.mode {
            LocalSourceMode::Reference {
                manifest,
                max_manifest_bytes,
            } => Self::from_manifest(config, root, executor, manifest, *max_manifest_bytes).await?,
            LocalSourceMode::Finite => Self::from_scan(config, root, executor, true).await?,
            LocalSourceMode::Follow { .. } => {
                Self::from_scan(config, root, executor, false).await?
            }
        };
        inventory.freeze_members(config, stream_identity);
        Ok(inventory)
    }

    async fn from_scan(
        config: &LocalSourceConfig,
        root: &OwnedFd,
        executor: &StreamingBlockingExecutor,
        needs_content_digest: bool,
    ) -> Result<Self, StreamSourceError> {
        let scan = run_scan(config, root, executor).await?;
        let mut ready = VecDeque::with_capacity(scan.entries.len());
        for (ordinal, (name, generation)) in scan.entries.into_iter().enumerate() {
            if generation.size_bytes > config.max_partition_bytes {
                return Err(StreamSourceError::acquisition(
                    AcquisitionFailureCode::ObjectLimitExceeded,
                ));
            }
            let relative_path = PathBuf::from(name);
            let basis = if needs_content_digest {
                IdentityBasis::Content(
                    digest_whole_object(config, root, executor, &relative_path, generation).await?,
                )
            } else {
                IdentityBasis::Generation(generation)
            };
            ready.push_back(InventoryEntry {
                relative_path,
                size_bytes: generation.size_bytes,
                ordinal: ordinal as u64,
                basis,
                generation: Some(generation),
                expected_digest: None,
            });
        }
        let next_ordinal = ready.len() as u64;
        Ok(Self {
            ready,
            members: Vec::new(),
            observed: BTreeSet::new(),
            is_complete: needs_content_digest || scan.has_seal_marker,
            next_ordinal,
            is_ordinal_authoritative: needs_content_digest,
            snapshot_digest: ContentDigest::from_bytes([0; 32]),
        })
    }

    async fn from_manifest(
        config: &LocalSourceConfig,
        root: &OwnedFd,
        executor: &StreamingBlockingExecutor,
        manifest: &Path,
        max_manifest_bytes: u64,
    ) -> Result<Self, StreamSourceError> {
        let bytes = read_whole_file(root, executor, manifest, max_manifest_bytes).await?;
        let text = std::str::from_utf8(&bytes).map_err(|_| discovery_error())?;
        let mut ready = VecDeque::new();
        let mut seen = BTreeSet::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if ready.len() >= u32_to_usize(config.max_scan_entries) {
                return Err(StreamSourceError::acquisition(
                    AcquisitionFailureCode::ObjectLimitExceeded,
                ));
            }
            let record: ManifestRecordV1 =
                serde_json::from_str(line).map_err(|_| discovery_error())?;
            let Some(digest) = decode_hex32(&record.digest) else {
                return Err(discovery_error());
            };
            if !is_valid_relative(&record.path)
                || record.size_bytes > config.max_partition_bytes
                || !seen.insert(record.path.clone())
            {
                return Err(discovery_error());
            }
            let ordinal = ready.len() as u64;
            ready.push_back(InventoryEntry {
                relative_path: record.path,
                size_bytes: record.size_bytes,
                ordinal,
                basis: IdentityBasis::Content(digest),
                generation: None,
                expected_digest: Some(digest),
            });
        }
        let next_ordinal = ready.len() as u64;
        Ok(Self {
            ready,
            members: Vec::new(),
            observed: BTreeSet::new(),
            is_complete: true,
            next_ordinal,
            is_ordinal_authoritative: true,
            snapshot_digest: ContentDigest::from_bytes([0; 32]),
        })
    }

    fn freeze_members(&mut self, config: &LocalSourceConfig, stream_identity: &ContentDigest) {
        self.members = self
            .ready
            .iter()
            .map(|entry| (entry.relative_path.clone(), entry.identity(stream_identity)))
            .collect();
        for entry in &self.ready {
            self.observed.insert(entry.relative_path.clone());
        }
        let mut hasher = blake3::Hasher::new();
        update_field(&mut hasher, b"aiperf.stream.local.snapshot.v1");
        update_field(&mut hasher, stream_identity.as_bytes());
        update_field(&mut hasher, &[config.mode.tag()]);
        update_field(&mut hasher, config.root.as_os_str().as_bytes());
        update_field(&mut hasher, config.suffix.as_bytes());
        update_field(&mut hasher, &(self.members.len() as u64).to_le_bytes());
        for (path, identity) in &self.members {
            update_field(&mut hasher, path.as_os_str().as_bytes());
            update_field(&mut hasher, identity.as_bytes());
        }
        self.snapshot_digest = ContentDigest::from_bytes(*hasher.finalize().as_bytes());
    }

    fn take_ready(&mut self) -> Option<InventoryEntry> {
        self.ready.pop_front()
    }

    const fn is_complete(&self) -> bool {
        self.is_complete
    }

    /// Absorb one rescan, appending only names never observed before.
    fn absorb(
        &mut self,
        config: &LocalSourceConfig,
        scan: ScanResult,
    ) -> Result<(), StreamSourceError> {
        if scan.has_seal_marker {
            self.is_complete = true;
        }
        for (name, generation) in scan.entries {
            let relative_path = PathBuf::from(name);
            if self.observed.contains(&relative_path) {
                continue;
            }
            if generation.size_bytes > config.max_partition_bytes {
                return Err(StreamSourceError::acquisition(
                    AcquisitionFailureCode::ObjectLimitExceeded,
                ));
            }
            let ordinal = self.next_ordinal;
            self.next_ordinal = self
                .next_ordinal
                .checked_add(1)
                .ok_or_else(snapshot_error)?;
            self.observed.insert(relative_path.clone());
            self.ready.push_back(InventoryEntry {
                relative_path,
                size_bytes: generation.size_bytes,
                ordinal,
                basis: IdentityBasis::Generation(generation),
                generation: Some(generation),
                expected_digest: None,
            });
        }
        Ok(())
    }

    /// Return whether the committed cursor is still reachable from this open.
    fn can_resume_at(&self, cursor: &LocalSourceCursor) -> bool {
        if cursor.relative_path.as_os_str().is_empty() {
            return true;
        }
        self.members.iter().any(|(path, identity)| {
            path == &cursor.relative_path && *identity == cursor.object_digest
        })
    }
}

async fn run_scan(
    config: &LocalSourceConfig,
    root: &OwnedFd,
    executor: &StreamingBlockingExecutor,
) -> Result<ScanResult, StreamSourceError> {
    let dir = open_dir_snapshot(root.as_raw_fd()).map_err(|_| discovery_error())?;
    let suffix = config.suffix.as_bytes().to_vec();
    let marker = config.seal_marker().map(|value| value.as_bytes().to_vec());
    let max_entries = u32_to_usize(config.max_scan_entries);
    let reserved = max_entries.saturating_mul(SCAN_OUTPUT_BYTES_PER_ENTRY);
    let output = executor
        .run(
            BlockingWorkClass::Acquisition,
            BlockingWorkBudget {
                input_bytes: 0,
                output_bytes: reserved,
            },
            move |cancellation| {
                if cancellation.is_cancelled() {
                    return Err(BlockingWorkError::Cancelled);
                }
                Ok(scan_directory(dir, suffix, marker, max_entries))
            },
        )
        .await
        .map_err(|_| discovery_error())?;
    output.into_inner().map_err(|_| discovery_error())
}

async fn digest_whole_object(
    config: &LocalSourceConfig,
    root: &OwnedFd,
    executor: &StreamingBlockingExecutor,
    relative_path: &Path,
    generation: FileGeneration,
) -> Result<[u8; 32], StreamSourceError> {
    let dir = dup_owned(root.as_raw_fd()).map_err(|_| discovery_error())?;
    let name = c_path(relative_path).map_err(|_| discovery_error())?;
    let chunk = u32_to_usize(config.max_chunk_bytes);
    let output = executor
        .run(
            BlockingWorkClass::Acquisition,
            BlockingWorkBudget {
                input_bytes: 0,
                output_bytes: 32,
            },
            move |cancellation| {
                if cancellation.is_cancelled() {
                    return Err(BlockingWorkError::Cancelled);
                }
                Ok((|| {
                    let file = openat_nofollow(&dir, &name)?;
                    let stat = fstat_owned(&file)?;
                    if !is_regular(&stat) {
                        return Err(libc::EINVAL);
                    }
                    let mut hasher = blake3::Hasher::new();
                    let mut buffer = vec![0_u8; chunk.max(1)];
                    let mut offset = 0_u64;
                    loop {
                        let read = pread_bounded(&file, &mut buffer, offset)?;
                        if read == 0 {
                            break;
                        }
                        hasher.update(&buffer[..read]);
                        offset += read as u64;
                    }
                    if offset != generation.size_bytes {
                        return Err(libc::EAGAIN);
                    }
                    Ok(*hasher.finalize().as_bytes())
                })())
            },
        )
        .await
        .map_err(|_| discovery_error())?;
    output.into_inner().map_err(|_| discovery_error())
}

async fn read_whole_file(
    root: &OwnedFd,
    executor: &StreamingBlockingExecutor,
    relative_path: &Path,
    max_bytes: u64,
) -> Result<Vec<u8>, StreamSourceError> {
    let dir = dup_owned(root.as_raw_fd()).map_err(|_| discovery_error())?;
    let name = c_path(relative_path).map_err(|_| discovery_error())?;
    let limit = usize::try_from(max_bytes).map_err(|_| discovery_error())?;
    let output = executor
        .run(
            BlockingWorkClass::Acquisition,
            BlockingWorkBudget {
                input_bytes: 0,
                output_bytes: limit,
            },
            move |cancellation| {
                if cancellation.is_cancelled() {
                    return Err(BlockingWorkError::Cancelled);
                }
                Ok((|| {
                    let file = openat_nofollow(&dir, &name)?;
                    let stat = fstat_owned(&file)?;
                    if !is_regular(&stat) {
                        return Err(libc::EINVAL);
                    }
                    let size = stat.st_size.max(0) as u64;
                    if size > max_bytes {
                        return Err(libc::EFBIG);
                    }
                    let mut buffer = vec![0_u8; size as usize];
                    let read = pread_bounded(&file, &mut buffer, 0)?;
                    buffer.truncate(read);
                    Ok(buffer)
                })())
            },
        )
        .await
        .map_err(|_| discovery_error())?;
    output.into_inner().map_err(|_| discovery_error())
}

// ---------------------------------------------------------------------------
// Factory and preparation
// ---------------------------------------------------------------------------

/// Startup validator and preparer for the built-in local source.
#[derive(Debug)]
pub struct LocalSourceFactory;

impl StreamingDatasetSourceFactory for LocalSourceFactory {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor {
        &LOCAL_SOURCE_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError> {
        let config: LocalSourceConfig =
            serde_json::from_str(authored.get()).map_err(|_| discovery_error())?;
        config.validate()?;
        Ok(Box::new(config))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSourceConfig>,
        context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError> {
        let config = *config
            .into_any()
            .downcast::<LocalSourceConfig>()
            .map_err(|_| discovery_error())?;
        // Preparation performs no filesystem effect: the root is opened by
        // `open`, so a prepared-but-never-opened source touches no descriptor.
        let scan_bytes =
            u32_to_usize(config.max_scan_entries).saturating_mul(SCAN_OUTPUT_BYTES_PER_ENTRY);
        let manifest_bytes = match &config.mode {
            LocalSourceMode::Reference {
                max_manifest_bytes, ..
            } => usize::try_from(*max_manifest_bytes).map_err(|_| discovery_error())?,
            _ => 0,
        };
        let max_output_bytes = u32_to_usize(config.max_chunk_bytes)
            .max(scan_bytes)
            .max(manifest_bytes);
        let executor = StreamingBlockingExecutor::new(
            context.run,
            CheckpointParticipantId::new("streaming-source-local-blocking"),
            2,
            u32_to_usize(config.max_chunk_bytes),
            max_output_bytes,
        )
        .map_err(|_| discovery_error())?;
        Ok(Box::new(LocalPreparedSource {
            config: Rc::new(config),
            run: context.run,
            stream_identity: context.stream_semantic_digest,
            budget: context.acquisition_budget.clone(),
            reporter: context.issue_reporter.clone(),
            executor,
        }))
    }
}

struct LocalPreparedSource {
    config: Rc<LocalSourceConfig>,
    run: StreamRunIdentity,
    stream_identity: ContentDigest,
    budget: AcquisitionBudget,
    reporter: StreamingIssueReporterHandle,
    executor: StreamingBlockingExecutor,
}

#[async_trait(?Send)]
impl PreparedStreamingDatasetSource for LocalPreparedSource {
    async fn open(
        self: Box<Self>,
        stop: StreamingStopReceiver,
    ) -> Result<OpenedStreamingDatasetSource, StreamSourceError> {
        let control = stop.control();
        let root = open_root_nofollow(&self.config.root)?;
        let watch = match &self.config.mode {
            LocalSourceMode::Follow {
                accepts_close_write,
                ..
            } => Some(DirectoryWatch::install(
                root.as_raw_fd(),
                *accepts_close_write,
            )?),
            LocalSourceMode::Finite | LocalSourceMode::Reference { .. } => None,
        };
        let inventory =
            Inventory::open(&self.config, &self.stream_identity, &root, &self.executor).await?;
        let snapshot = SourceSnapshotReceipt {
            digest: inventory.snapshot_digest,
        };
        Ok(OpenedStreamingDatasetSource {
            source: Box::new(LocalSource {
                config: self.config,
                root: Rc::new(root),
                watch,
                inventory,
                snapshot,
                run: self.run,
                stream_identity: self.stream_identity,
                budget: self.budget,
                reporter: self.reporter,
                executor: self.executor,
                emitted: BTreeMap::new(),
                next_position: 0,
                generation: 0,
                last_emitted: None,
                committed: None,
                resume_after: None,
                is_sealed: false,
                has_unpublished_frontier: false,
                stop,
                participant_id: CheckpointParticipantId::new("streaming-source-local"),
                initialization: ParticipantInitialization::default(),
            }),
            control,
        })
    }
}

fn open_root_nofollow(root: &Path) -> Result<OwnedFd, StreamSourceError> {
    let path = c_path(root)
        .map_err(|_| StreamSourceError::source(SourceFailureCode::SourceUnavailable))?;
    // SAFETY: `path` is NUL-terminated and is not retained after the call; a
    // nonnegative result is a fresh descriptor moved once into `OwnedFd`.
    let raw = unsafe {
        libc::open(
            path.as_ptr(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    };
    if raw < 0 {
        return Err(StreamSourceError::source(
            SourceFailureCode::SourceUnavailable,
        ));
    }
    // SAFETY: a nonnegative `open` result is a fresh descriptor moved once.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

// ---------------------------------------------------------------------------
// The source
// ---------------------------------------------------------------------------

/// Run-local immutable local partition stream.
pub struct LocalSource {
    config: Rc<LocalSourceConfig>,
    root: Rc<OwnedFd>,
    watch: Option<DirectoryWatch>,
    inventory: Inventory,
    snapshot: SourceSnapshotReceipt,
    run: StreamRunIdentity,
    stream_identity: ContentDigest,
    budget: AcquisitionBudget,
    reporter: StreamingIssueReporterHandle,
    executor: StreamingBlockingExecutor,
    /// Names announced but not yet released by a committed checkpoint.
    emitted: BTreeMap<PathBuf, ImmutableObjectIdentity>,
    next_position: u64,
    generation: u64,
    last_emitted: Option<LocalSourceCursor>,
    committed: Option<LocalSourceCursor>,
    resume_after: Option<LocalSourceCursor>,
    is_sealed: bool,
    has_unpublished_frontier: bool,
    stop: StreamingStopReceiver,
    participant_id: CheckpointParticipantId,
    initialization: ParticipantInitialization,
}

/// Exact resume coordinate retained by the host checkpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LocalSourceCursor {
    /// Position the source will assign to its next announced partition.
    pub next_position: u64,
    /// Monotonic discovery generation that produced the cursor.
    pub generation: u64,
    /// Root-relative path of the last committed partition.
    pub relative_path: PathBuf,
    /// Immutable identity of the last committed partition.
    pub object_digest: ImmutableObjectIdentity,
}

#[async_trait(?Send)]
impl StreamingDatasetSource for LocalSource {
    fn snapshot(&self) -> &SourceSnapshotReceipt {
        &self.snapshot
    }

    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError> {
        self.poll_next_event().await
    }
}

impl LocalSource {
    async fn poll_next_event(&mut self) -> Result<SourceEvent, StreamSourceError> {
        loop {
            // A stop that has already been requested is surfaced by awaiting the
            // host receiver: the stopped outcome is unforgeable by an adapter.
            if self.stop.is_stopped() {
                return self.stop.stopped().await.map(|()| unreachable_event());
            }
            if let Some(entry) = self.inventory.take_ready() {
                if let Some(event) = self.announce(entry)? {
                    return Ok(event);
                }
                continue;
            }
            if self.has_unpublished_frontier {
                self.has_unpublished_frontier = false;
                let through = SourcePosition::new(self.next_position.saturating_sub(1));
                return Ok(SourceEvent::Frontier(SourceFrontier { through }));
            }
            if self.inventory.is_complete() {
                if self.is_sealed {
                    return self.stop.stopped().await.map(|()| unreachable_event());
                }
                self.is_sealed = true;
                return Ok(SourceEvent::Seal(SourceSeal {
                    final_position: self
                        .last_emitted
                        .as_ref()
                        .map(|cursor| SourcePosition::new(cursor.next_position.saturating_sub(1))),
                    digest: self.snapshot.digest,
                }));
            }
            self.await_publication().await?;
        }
    }

    /// Park until the watched directory publishes, or the host stops.
    ///
    /// The wakeup is `inotify` readiness through `AsyncFd`; no timer, wall
    /// clock, or poll interval participates. A readiness edge is followed by a
    /// full bounded rescan, so a dropped-event overflow can never hide a
    /// published name.
    async fn await_publication(&mut self) -> Result<(), StreamSourceError> {
        let Some(watch) = self.watch.as_ref() else {
            // Finite and reference inventories are complete at open, so
            // reaching here would be an internal ordering fault.
            return Err(snapshot_error());
        };
        {
            let stop = &mut self.stop;
            tokio::select! {
                biased;
                stopped = stop.stopped() => return stopped,
                readiness = watch.await_publication() => readiness?,
            }
        }
        self.generation = self.generation.checked_add(1).ok_or_else(snapshot_error)?;
        let scan = run_scan(&self.config, &self.root, &self.executor).await?;
        self.inventory.absorb(&self.config, scan)
    }

    /// Freeze identity, deduplicate, and bind a partition to a stable position.
    fn announce(
        &mut self,
        entry: InventoryEntry,
    ) -> Result<Option<SourceEvent>, StreamSourceError> {
        let identity = entry.identity(&self.stream_identity);
        if let Some(previous) = self.emitted.get(&entry.relative_path) {
            // Rediscovery of an already-announced name never mutates identity
            // and never re-announces a position.
            return if *previous == identity {
                Ok(None)
            } else {
                Err(StreamSourceError::source(SourceFailureCode::MutatedObject))
            };
        }
        if self.is_suppressed_by_resume(&entry) {
            return Ok(None);
        }
        if self.emitted.len() >= u32_to_usize(self.config.max_scan_entries) {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::ObjectLimitExceeded,
            ));
        }
        let position = SourcePosition::new(self.next_position);
        self.next_position = self
            .next_position
            .checked_add(1)
            .ok_or_else(snapshot_error)?;
        self.emitted.insert(entry.relative_path.clone(), identity);
        self.last_emitted = Some(LocalSourceCursor {
            next_position: self.next_position,
            generation: self.generation,
            relative_path: entry.relative_path.clone(),
            object_digest: identity,
        });
        self.has_unpublished_frontier = true;
        Ok(Some(SourceEvent::Partition(SourcePartition::new(
            position,
            Box::new(LocalPartitionContent {
                identity,
                entry,
                config: Rc::clone(&self.config),
                root: Rc::clone(&self.root),
                executor: self.executor.clone(),
                reporter: self.reporter.clone(),
                run: self.run,
                stream_identity: self.stream_identity,
                position,
            }),
        ))))
    }

    fn is_suppressed_by_resume(&self, entry: &InventoryEntry) -> bool {
        // Ordinal suppression applies only where a rescan reproduces the same
        // order; publish-ordered follow relies on the retained `emitted` map.
        self.inventory.is_ordinal_authoritative
            && self
                .resume_after
                .as_ref()
                .is_some_and(|cursor| entry.ordinal < cursor.next_position)
    }

    /// Borrow the acquisition budget the host installed for this source.
    #[must_use]
    pub const fn acquisition_budget(&self) -> &AcquisitionBudget {
        &self.budget
    }
}

/// The stop receiver always resolves to an error, so this is never reached.
fn unreachable_event() -> SourceEvent {
    SourceEvent::Frontier(SourceFrontier {
        through: SourcePosition::new(0),
    })
}

// ---------------------------------------------------------------------------
// Checkpoint participation
// ---------------------------------------------------------------------------

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for LocalSource {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        if barrier.run != self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let cursor = self
            .last_emitted
            .clone()
            .or_else(|| self.committed.clone())
            .unwrap_or_else(|| LocalSourceCursor {
                next_position: self.next_position,
                generation: self.generation,
                relative_path: PathBuf::new(),
                object_digest: ImmutableObjectIdentity::from_bytes([0; 32]),
            });
        let bytes = cursor.encode()?;
        let budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: bytes.len().max(1),
        })
        .map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = budget
            .try_acquire(1, bytes.len())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(bytes), lease)?;
        PreparedParticipantState::new(
            barrier.run,
            self.participant_id.clone(),
            LOCAL_SOURCE_SCHEMA_ID,
            LOCAL_SOURCE_SCHEMA_VERSION,
            barrier.cut.clone(),
            cursor.next_position,
            payload,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()?;
        let Some(state) = state else {
            return Ok(());
        };
        if state.run() != &self.run
            || state.descriptor().schema_id != LOCAL_SOURCE_SCHEMA_ID
            || state.descriptor().schema_version != LOCAL_SOURCE_SCHEMA_VERSION
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let cursor = LocalSourceCursor::decode(state.payload_bytes())?;
        if !self.inventory.can_resume_at(&cursor) {
            // The committed partition is no longer reachable, so no truthful
            // continuation exists.
            return Err(CheckpointError::SourceUnavailableOnResume);
        }
        self.next_position = cursor.next_position;
        self.generation = cursor.generation;
        if !cursor.relative_path.as_os_str().is_empty() {
            self.emitted
                .insert(cursor.relative_path.clone(), cursor.object_digest);
        }
        self.committed = Some(cursor.clone());
        self.resume_after = Some(cursor);
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run || receipt.participant_id() != &self.participant_id {
            return Err(CheckpointError::ObjectVerification);
        }
        let Some(cursor) = self.last_emitted.clone() else {
            return Ok(());
        };
        // Idempotent for one exact receipt: a repeated notification finds the
        // pre-cut state already released and changes nothing.
        self.emitted.retain(|path, _| path >= &cursor.relative_path);
        self.committed = Some(cursor);
        Ok(())
    }
}

impl LocalSourceCursor {
    fn encode(&self) -> Result<Vec<u8>, CheckpointError> {
        let path = self.relative_path.as_os_str().as_bytes();
        let mut bytes = Vec::with_capacity(CURSOR_HEADER_BYTES + path.len());
        bytes.extend_from_slice(&self.next_position.to_le_bytes());
        bytes.extend_from_slice(&self.generation.to_le_bytes());
        bytes.extend_from_slice(self.object_digest.as_bytes());
        let length = u64::try_from(path.len()).map_err(|_| CheckpointError::ObjectVerification)?;
        bytes.extend_from_slice(&length.to_le_bytes());
        bytes.extend_from_slice(path);
        Ok(bytes)
    }

    fn decode(bytes: &[u8]) -> Result<Self, CheckpointError> {
        let invalid = || CheckpointError::ObjectVerification;
        if bytes.len() < CURSOR_HEADER_BYTES {
            return Err(invalid());
        }
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&bytes[16..48]);
        let path_length = usize::try_from(u64::from_le_bytes(
            bytes[48..56].try_into().map_err(|_| invalid())?,
        ))
        .map_err(|_| invalid())?;
        if bytes.len() != CURSOR_HEADER_BYTES + path_length {
            return Err(invalid());
        }
        let relative_path = PathBuf::from(
            std::str::from_utf8(&bytes[CURSOR_HEADER_BYTES..]).map_err(|_| invalid())?,
        );
        if !relative_path.as_os_str().is_empty() && !is_valid_relative(&relative_path) {
            return Err(invalid());
        }
        Ok(Self {
            next_position: u64::from_le_bytes(bytes[0..8].try_into().map_err(|_| invalid())?),
            generation: u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| invalid())?),
            relative_path,
            object_digest: ImmutableObjectIdentity::from_bytes(digest),
        })
    }
}

// ---------------------------------------------------------------------------
// Immutable content authority and bounded acquisition
// ---------------------------------------------------------------------------

/// Immutable content authority for one discovered local partition.
struct LocalPartitionContent {
    identity: ImmutableObjectIdentity,
    entry: InventoryEntry,
    config: Rc<LocalSourceConfig>,
    root: Rc<OwnedFd>,
    executor: StreamingBlockingExecutor,
    reporter: StreamingIssueReporterHandle,
    run: StreamRunIdentity,
    stream_identity: ContentDigest,
    position: SourcePosition,
}

#[async_trait(?Send)]
impl SourcePartitionContent for LocalPartitionContent {
    fn identity(&self) -> &ImmutableObjectIdentity {
        &self.identity
    }

    fn size_bytes(&self) -> Option<u64> {
        Some(self.entry.size_bytes)
    }

    async fn acquire(
        &self,
        request: PartitionAccessRequest,
        budget: &AcquisitionBudget,
    ) -> Result<AcquiredPartition, StreamSourceError> {
        match request {
            PartitionAccessRequest::Sequential { resume_offset } => {
                if resume_offset > self.entry.size_bytes {
                    return Err(StreamSourceError::acquisition(
                        AcquisitionFailureCode::ObjectLimitExceeded,
                    ));
                }
                let file = self.open_bounded().await?;
                let authority = budget.acquire_memory(1, 0).await?;
                AcquiredPartition::sequential(
                    self.position,
                    self.identity,
                    Some(self.entry.size_bytes),
                    resume_offset,
                    Box::new(LocalSequentialReader {
                        file: Some(file),
                        next_offset: resume_offset,
                        size_bytes: self.entry.size_bytes,
                        max_chunk_bytes: self.config.max_chunk_bytes,
                        executor: self.executor.clone(),
                        rolling: blake3::Hasher::new(),
                        expected_digest: if resume_offset == 0 {
                            self.entry.expected_digest
                        } else {
                            None
                        },
                    }),
                    authority,
                )
            }
            PartitionAccessRequest::SeekableLocal => {
                let file = self.open_bounded().await?;
                let charge = usize::try_from(self.entry.size_bytes).map_err(|_| {
                    StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
                })?;
                let disk = budget.acquire_disk(1, charge).await?;
                AcquiredPartition::seekable_local(
                    self.position,
                    self.identity,
                    self.entry.size_bytes,
                    Box::new(LocalSeekableSnapshot {
                        file: Rc::new(file),
                        executor: self.executor.clone(),
                        size_bytes: self.entry.size_bytes,
                        max_chunk_bytes: self.config.max_chunk_bytes,
                    }),
                    disk,
                )
            }
            // The descriptor advertises no range access, so a request for it is
            // a compatibility fault, not an I/O fault.
            PartitionAccessRequest::RangeReadable => {
                Err(StreamSourceError::acquisition(AcquisitionFailureCode::Open))
            }
        }
    }
}

impl LocalPartitionContent {
    /// Reopen the frozen name, proving the frozen generation before any read.
    ///
    /// A vanished object is retried up to `max_open_attempts`; exhaustion
    /// reports one ordinary partition-scoped issue and surfaces `open` so the
    /// host may install a hole. A *changed* object is never retried.
    async fn open_bounded(&self) -> Result<OwnedFd, StreamSourceError> {
        let mut attempt = 0_u32;
        loop {
            match self.open_once().await {
                Ok(file) => return Ok(file),
                Err(error)
                    if error == StreamSourceError::acquisition(AcquisitionFailureCode::Open) =>
                {
                    attempt = attempt.saturating_add(1);
                    if attempt >= self.config.max_open_attempts {
                        self.report_partition_fault(attempt, error).await;
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    async fn open_once(&self) -> Result<OwnedFd, StreamSourceError> {
        let dir = dup_owned(self.root.as_raw_fd())
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Open))?;
        let name = c_path(&self.entry.relative_path)
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Open))?;
        let expected_generation = self.entry.generation;
        let expected_size = self.entry.size_bytes;
        let output = self
            .executor
            .run(
                BlockingWorkClass::Acquisition,
                BlockingWorkBudget {
                    input_bytes: 0,
                    output_bytes: 0,
                },
                move |cancellation| {
                    if cancellation.is_cancelled() {
                        return Err(BlockingWorkError::Cancelled);
                    }
                    Ok((|| {
                        let file = openat_nofollow(&dir, &name)?;
                        let stat = fstat_owned(&file)?;
                        if !is_regular(&stat) {
                            return Err(OpenFault::NotRegular);
                        }
                        let observed = generation_from_stat(&stat);
                        // TOCTOU-free: the proof runs on the descriptor that
                        // will be read, never on a re-resolved name.
                        match expected_generation {
                            Some(frozen) if frozen != observed => Err(OpenFault::Mutated),
                            _ if observed.size_bytes != expected_size => Err(OpenFault::Mutated),
                            _ => Ok(file),
                        }
                    })())
                },
            )
            .await
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        output.into_inner().map_err(|fault| match fault {
            OpenFault::Mutated => StreamSourceError::source(SourceFailureCode::MutatedObject),
            OpenFault::NotRegular | OpenFault::Errno(_) => {
                StreamSourceError::acquisition(AcquisitionFailureCode::Open)
            }
        })
    }

    async fn report_partition_fault(&self, retry_ordinal: u32, error: StreamSourceError) {
        let Ok(issue) = OrdinaryStreamingIssue::partition(
            self.run,
            StreamingInputDomainIdentity::new(self.stream_identity, self.identity),
            self.identity,
            StreamingIssueClass::Retryable,
            self.stream_identity,
            self.position,
            retry_ordinal,
            ContentDigest::from_bytes(*self.identity.as_bytes()),
            OrdinaryStreamingFailure::Source(error),
        ) else {
            // A malformed issue is a programming fault, not a stream fault; the
            // acquisition error is already being returned to the host.
            tracing::debug!(
                component = "streaming.source.local",
                "partition issue facts were internally inconsistent"
            );
            return;
        };
        if let Err(report_error) = self.reporter.report(issue).await {
            tracing::debug!(
                error = %report_error,
                component = "streaming.source.local",
                "host reliability ledger refused a partition issue"
            );
        }
    }
}

/// Closed reasons one bounded reopen can refuse an immutable generation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OpenFault {
    /// The frozen generation or length no longer matches the descriptor.
    Mutated,
    /// The name resolved to a symlink, directory, or other non-regular file.
    NotRegular,
    /// A raw `openat`/`fstat` failure such as `ENOENT`.
    Errno(i32),
}

impl From<i32> for OpenFault {
    fn from(value: i32) -> Self {
        Self::Errno(value)
    }
}

/// Bounded forward reader over one immutable local generation.
struct LocalSequentialReader {
    file: Option<OwnedFd>,
    next_offset: u64,
    size_bytes: u64,
    max_chunk_bytes: u32,
    executor: StreamingBlockingExecutor,
    rolling: blake3::Hasher,
    /// Manifest-declared digest the complete stream must reproduce.
    expected_digest: Option<[u8; 32]>,
}

#[async_trait(?Send)]
impl StreamingSequentialReader for LocalSequentialReader {
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<Option<SequentialSourceChunk>, StreamSourceError> {
        if self.next_offset >= self.size_bytes {
            self.verify_declared_digest()?;
            return Ok(None);
        }
        let remaining = usize::try_from(self.size_bytes - self.next_offset).unwrap_or(usize::MAX);
        let want = max_bytes
            .get()
            .min(u32_to_usize(self.max_chunk_bytes))
            .min(remaining);
        let offset = self.next_offset;
        // The descriptor moves into the blocking closure and back out with the
        // bytes, so no lock guards it and no descriptor is shared.
        let file = self
            .file
            .take()
            .ok_or_else(|| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        let output = self
            .executor
            .run(
                BlockingWorkClass::Acquisition,
                BlockingWorkBudget {
                    input_bytes: 0,
                    output_bytes: want,
                },
                move |cancellation| {
                    if cancellation.is_cancelled() {
                        return Err(BlockingWorkError::Cancelled);
                    }
                    let mut buffer = vec![0_u8; want];
                    match pread_bounded(&file, &mut buffer, offset) {
                        Ok(read) => {
                            buffer.truncate(read);
                            Ok((file, Ok(buffer)))
                        }
                        Err(errno) => Ok((file, Err(errno))),
                    }
                },
            )
            .await
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        let (file, result) = output.into_inner();
        self.file = Some(file);
        let buffer =
            result.map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        if buffer.is_empty() {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::TruncatedObject,
            ));
        }
        self.rolling.update(&buffer);
        self.next_offset += buffer.len() as u64;
        let lease: AcquisitionMemoryLease = budget.acquire_memory(1, buffer.len()).await?;
        let rolling = ContentDigest::from_bytes(*self.rolling.clone().finalize().as_bytes());
        let chunk = BudgetedSourceChunk::new(Bytes::from(buffer), lease)?;
        Ok(Some(SequentialSourceChunk::new(
            chunk,
            self.next_offset,
            rolling,
        )))
    }
}

impl LocalSequentialReader {
    fn verify_declared_digest(&self) -> Result<(), StreamSourceError> {
        let Some(expected) = self.expected_digest else {
            return Ok(());
        };
        if self.next_offset != self.size_bytes
            || self.rolling.clone().finalize().as_bytes() != &expected
        {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::IdentityMismatch,
            ));
        }
        Ok(())
    }
}

/// No-follow seekable authority over one immutable local generation.
struct LocalSeekableSnapshot {
    file: Rc<OwnedFd>,
    executor: StreamingBlockingExecutor,
    size_bytes: u64,
    max_chunk_bytes: u32,
}

#[async_trait(?Send)]
impl StreamingSeekableLocalSnapshot for LocalSeekableSnapshot {
    async fn read_at(
        &self,
        offset: u64,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<BudgetedSourceChunk, StreamSourceError> {
        if offset > self.size_bytes {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::ObjectLimitExceeded,
            ));
        }
        let remaining = usize::try_from(self.size_bytes - offset).unwrap_or(usize::MAX);
        let want = max_bytes
            .get()
            .min(u32_to_usize(self.max_chunk_bytes))
            .min(remaining);
        let file = dup_owned(self.file.as_raw_fd())
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        let output = self
            .executor
            .run(
                BlockingWorkClass::Acquisition,
                BlockingWorkBudget {
                    input_bytes: 0,
                    output_bytes: want,
                },
                move |cancellation| {
                    if cancellation.is_cancelled() {
                        return Err(BlockingWorkError::Cancelled);
                    }
                    let mut buffer = vec![0_u8; want];
                    Ok(pread_bounded(&file, &mut buffer, offset).map(|read| {
                        buffer.truncate(read);
                        buffer
                    }))
                },
            )
            .await
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        let buffer = output
            .into_inner()
            .map_err(|_| StreamSourceError::acquisition(AcquisitionFailureCode::Read))?;
        let lease = budget.acquire_memory(1, buffer.len()).await?;
        BudgetedSourceChunk::new(Bytes::from(buffer), lease)
    }
}

// ---------------------------------------------------------------------------
// Directory watch (follow only)
// ---------------------------------------------------------------------------

/// `inotify` publish-by-rename watch parked through `AsyncFd`.
struct DirectoryWatch {
    descriptor: AsyncFd<OwnedFd>,
}

impl DirectoryWatch {
    /// Install a rename-only (optionally close-write) watch on an open root.
    fn install(root_fd: RawFd, accepts_close_write: bool) -> Result<Self, StreamSourceError> {
        let unavailable = || StreamSourceError::source(SourceFailureCode::SourceUnavailable);
        // SAFETY: `inotify_init1` allocates a new descriptor and borrows nothing.
        let raw = unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) };
        if raw < 0 {
            return Err(unavailable());
        }
        // SAFETY: a nonnegative result is a newly owned descriptor moved once.
        let owned = unsafe { OwnedFd::from_raw_fd(raw) };
        // Publish-by-rename only: `IN_CREATE`/`IN_MODIFY` are deliberately
        // excluded so a partially written object is never a partition.
        let mut mask = libc::IN_MOVED_TO;
        if accepts_close_write {
            mask |= libc::IN_CLOSE_WRITE;
        }
        let path = CString::new(format!("/proc/self/fd/{root_fd}")).map_err(|_| unavailable())?;
        // SAFETY: `owned` is live, `path` is NUL-terminated, and neither is
        // retained by `inotify_add_watch` after it returns.
        if unsafe { libc::inotify_add_watch(owned.as_raw_fd(), path.as_ptr(), mask) } < 0 {
            return Err(unavailable());
        }
        Ok(Self {
            descriptor: AsyncFd::new(owned).map_err(|_| unavailable())?,
        })
    }

    /// Await one readiness edge and drain the queued event buffer.
    async fn await_publication(&self) -> Result<(), StreamSourceError> {
        let unavailable = || StreamSourceError::source(SourceFailureCode::SourceUnavailable);
        loop {
            let mut guard = self
                .descriptor
                .readable()
                .await
                .map_err(|_| unavailable())?;
            match guard.try_io(|inner| drain_inotify(inner.get_ref())) {
                Err(_would_block) => continue,
                Ok(result) => return result.map_err(|_| unavailable()),
            }
        }
    }
}

/// Drain every queued `inotify` record without retaining any name.
///
/// The names are not the discovery authority: a readiness edge only triggers a
/// bounded rescan, so a dropped-event overflow cannot hide a published name.
fn drain_inotify(fd: &OwnedFd) -> std::io::Result<()> {
    // The fixed header plus a generous name bound, times a small batch.
    let mut buffer = [0_u8; 8 * (std::mem::size_of::<libc::inotify_event>() + 256)];
    let mut total = 0_usize;
    loop {
        // SAFETY: `fd` is live and `buffer` is valid for its full length.
        let read = unsafe { libc::read(fd.as_raw_fd(), buffer.as_mut_ptr().cast(), buffer.len()) };
        if read < 0 {
            let error = std::io::Error::last_os_error();
            if error.kind() == std::io::ErrorKind::WouldBlock {
                // Returning `WouldBlock` with nothing drained lets `try_io`
                // clear readiness so the next park is a real wait.
                return if total == 0 { Err(error) } else { Ok(()) };
            }
            return Err(error);
        }
        if read == 0 {
            return Ok(());
        }
        total += read as usize;
    }
}
