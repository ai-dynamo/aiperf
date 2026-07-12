// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Qualified local spool, create-only generations, atomic heads, and file WAL.

use std::ffi::CString;
use std::fmt::{self, Debug, Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::manifest::{generation_key, index_root_key};
use crate::{
    ArchiveId, ArchiveState, Digest, GenerationMutationV1, GenerationObjectV1,
    GenerationTransactionKind, GenerationV1, GenesisV1, HeadDescriptorV1, IndexError,
    IndexMutationSetV1, IndexPageSource, IndexSnapshot, LocalLatestV1, ManifestError, MutationMode,
    RecoveredWal, SEALED_WAL_FOOTER_BYTES, SealedWalSegment, SessionAnchorV1, SessionId, TableId,
    WAL_FOOTER_MAGIC, WalError, WalFrame, WalSegmentBuilder, WalSegmentHeaderV1,
};

const LOCAL_LATEST: &str = "LOCAL-LATEST";
const LOCK_FILE: &str = ".archive.lock";

/// Local filesystem family proved by spool qualification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FilesystemKind {
    /// Linux ext2/ext3/ext4 family.
    Ext,
    /// XFS.
    Xfs,
    /// Btrfs.
    Btrfs,
    /// F2FS.
    F2fs,
    /// ZFS.
    Zfs,
    /// tmpfs, accepted for deterministic tests and explicitly ephemeral deployments.
    Tmpfs,
    /// OverlayFS backed by a qualified local lower/upper filesystem.
    Overlay,
}

/// Filesystem facts captured while the lifetime lock is held.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SpoolQualification {
    /// Canonical absolute spool path.
    pub canonical_path: PathBuf,
    /// Proved local filesystem family.
    pub filesystem_kind: FilesystemKind,
    /// Filesystem device ID.
    pub device_id: u64,
    /// Available filesystem blocks at qualification.
    pub available_blocks: u64,
    /// Available filesystem inodes at qualification.
    pub available_inodes: u64,
}

/// Every post-operation crash edge implemented by this durable-core slice.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DurabilityEdge {
    /// Index temporary file has complete bytes.
    IndexTempWritten,
    /// Index file has been fsynced.
    IndexFileSynced,
    /// Index temporary file was create-only renamed.
    IndexRenamed,
    /// Index parent directory was fsynced.
    IndexDirectorySynced,
    /// Generation temporary file has complete bytes.
    GenerationTempWritten,
    /// Generation file has been fsynced.
    GenerationFileSynced,
    /// Generation temporary file was create-only renamed.
    GenerationRenamed,
    /// Generation parent directory was fsynced.
    GenerationDirectorySynced,
    /// Parquet partition temporary file has complete bytes.
    PartitionTempWritten,
    /// Parquet partition file has been fsynced.
    PartitionFileSynced,
    /// Parquet partition temporary file was create-only renamed.
    PartitionRenamed,
    /// Parquet partition parent directory was fsynced.
    PartitionDirectorySynced,
    /// Pointer temporary file has complete bytes.
    PointerTempWritten,
    /// Pointer temporary file has been fsynced.
    PointerFileSynced,
    /// Pointer was atomically replaced.
    PointerRenamed,
    /// Pointer parent directory was fsynced.
    PointerDirectorySynced,
    /// Receipt batch temporary file has complete bytes.
    ReceiptBatchTempWritten,
    /// Receipt batch file has been fsynced.
    ReceiptBatchFileSynced,
    /// Receipt batch temporary file was create-only renamed.
    ReceiptBatchRenamed,
    /// Receipt batch parent directory was fsynced.
    ReceiptBatchDirectorySynced,
    /// Receipt index temporary file has complete bytes.
    ReceiptIndexTempWritten,
    /// Receipt index file has been fsynced.
    ReceiptIndexFileSynced,
    /// Receipt index temporary file was create-only renamed.
    ReceiptIndexRenamed,
    /// Receipt index parent directory was fsynced.
    ReceiptIndexDirectorySynced,
    /// Receipt head temporary file has complete bytes.
    ReceiptHeadTempWritten,
    /// Receipt head file has been fsynced.
    ReceiptHeadFileSynced,
    /// Receipt head temporary file was create-only renamed.
    ReceiptHeadRenamed,
    /// Receipt head parent directory was fsynced.
    ReceiptHeadDirectorySynced,
    /// Receipt pointer temporary file has complete bytes.
    ReceiptPointerTempWritten,
    /// Receipt pointer temporary file has been fsynced.
    ReceiptPointerFileSynced,
    /// Receipt pointer was atomically replaced.
    ReceiptPointerRenamed,
    /// Receipt pointer parent directory was fsynced.
    ReceiptPointerDirectorySynced,
    /// Open WAL inode was created.
    WalFileCreated,
    /// WAL segment header has complete bytes.
    WalHeaderWritten,
    /// WAL segment header was fsynced.
    WalHeaderSynced,
    /// WAL creation directory entry was fsynced.
    WalDirectorySynced,
    /// One complete WAL frame was written.
    WalFrameWritten,
    /// One complete WAL frame was fsynced.
    WalFrameSynced,
    /// A complete sealed footer was written to the open segment.
    WalFooterWritten,
    /// Open segment plus footer was fsynced.
    WalSealSynced,
    /// Open segment was renamed to its immutable `.wal` name.
    WalRenamed,
    /// Sealed-segment directory entry was fsynced.
    WalSealDirectorySynced,
    /// An incomplete physical open tail was truncated.
    WalTailTruncated,
    /// The tail truncation was fsynced.
    WalTailSynced,
}

/// Injectable crash/fault boundary used by every local transaction.
pub trait DurabilityFaultInjector: Debug + Send + Sync {
    /// Returns an injected failure after the named operation completed.
    fn after(&self, edge: DurabilityEdge) -> Result<(), SpoolError>;
}

/// Production injector that never fails.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoDurabilityFaults;

impl DurabilityFaultInjector for NoDurabilityFaults {
    fn after(&self, _edge: DurabilityEdge) -> Result<(), SpoolError> {
        Ok(())
    }
}

/// Deterministic injector that fails on one selected occurrence of one edge.
#[derive(Debug)]
pub struct FailAtDurabilityEdge {
    edge: DurabilityEdge,
    remaining: AtomicUsize,
}

impl FailAtDurabilityEdge {
    /// Fails on the first occurrence of `edge`.
    #[must_use]
    pub const fn first(edge: DurabilityEdge) -> Self {
        Self {
            edge,
            remaining: AtomicUsize::new(1),
        }
    }

    /// Fails on the selected one-based occurrence of `edge`.
    #[must_use]
    pub const fn occurrence(edge: DurabilityEdge, occurrence: usize) -> Self {
        Self {
            edge,
            remaining: AtomicUsize::new(occurrence),
        }
    }
}

impl DurabilityFaultInjector for FailAtDurabilityEdge {
    fn after(&self, edge: DurabilityEdge) -> Result<(), SpoolError> {
        if edge != self.edge {
            return Ok(());
        }
        let result = self
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                remaining.checked_sub(1)
            });
        if result == Ok(1) {
            return Err(SpoolError::FaultInjected(edge));
        }
        Ok(())
    }
}

/// Qualified spool with one crash-released open-descriptor exclusive lock.
pub struct QualifiedSpool {
    root: PathBuf,
    _lock: File,
    qualification: SpoolQualification,
}

impl Debug for QualifiedSpool {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("QualifiedSpool")
            .field("root", &self.root)
            .field("qualification", &self.qualification)
            .finish_non_exhaustive()
    }
}

impl QualifiedSpool {
    /// Creates/opens, qualifies, and exclusively locks an absolute local spool.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SpoolError> {
        let path = path.as_ref();
        if !path.is_absolute() {
            return Err(SpoolError::RelativeSpoolPath(path.to_path_buf()));
        }
        reject_symlink_components(path)?;
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true).mode(0o700);
        builder
            .create(path)
            .map_err(|error| io_error("create spool directory", path, error))?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .map_err(|error| io_error("set spool permissions", path, error))?;
        reject_symlink_components(path)?;
        let root =
            fs::canonicalize(path).map_err(|error| io_error("canonicalize spool", path, error))?;
        let filesystem_kind = qualified_filesystem(&root)?;

        let lock_path = root.join(LOCK_FILE);
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(&lock_path)
            .map_err(|error| io_error("open archive lock", &lock_path, error))?;
        acquire_exclusive_lock(&lock, &lock_path)?;
        let lock_metadata = lock
            .metadata()
            .map_err(|error| io_error("stat archive lock", &lock_path, error))?;
        if !lock_metadata.is_file() || lock_metadata.nlink() != 1 {
            return Err(SpoolError::UnqualifiedLockFile(lock_path));
        }

        for directory in [
            "manifest-index",
            "manifests",
            "wal",
            "receipts",
            "receipts/batches",
            "receipts/index",
            "receipts/heads",
            "raw",
            "partitions",
        ] {
            let path = root.join(directory);
            let mut builder = fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(&path) {
                Ok(()) => {}
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
                Err(error) => return Err(io_error("create spool layout", &path, error)),
            }
            if !fs::symlink_metadata(&path)
                .map_err(|error| io_error("stat spool layout", &path, error))?
                .file_type()
                .is_dir()
            {
                return Err(SpoolError::SymlinkOrNonDirectory(path));
            }
        }
        sync_directory(&root)?;
        probe_atomic_create_and_rename(&root)?;
        let (available_blocks, available_inodes) = filesystem_capacity(&root)?;
        let metadata =
            fs::metadata(&root).map_err(|error| io_error("stat qualified spool", &root, error))?;
        let qualification = SpoolQualification {
            canonical_path: root.clone(),
            filesystem_kind,
            device_id: metadata.dev(),
            available_blocks,
            available_inodes,
        };
        Ok(Self {
            root,
            _lock: lock,
            qualification,
        })
    }

    /// Returns the canonical absolute spool path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.root
    }

    /// Returns captured filesystem qualification facts.
    #[must_use]
    pub const fn qualification(&self) -> &SpoolQualification {
        &self.qualification
    }

    /// Captures logical spool use and fresh filesystem bytes/inodes.
    ///
    /// Directory traversal and `statvfs` are intentionally blocking and must
    /// run only during preparation or at an owner fence, never on admission.
    pub fn budget_observation(&self) -> Result<crate::budget::ArchiveSpoolObservation, SpoolError> {
        let (logical_bytes, logical_files) = logical_spool_usage(&self.root)?;
        let (filesystem_available_bytes, filesystem_available_files) =
            filesystem_budget_capacity(&self.root)?;
        Ok(crate::budget::ArchiveSpoolObservation {
            logical_bytes,
            logical_files,
            filesystem_available_bytes,
            filesystem_available_files,
        })
    }

    pub(crate) fn read_relative(&self, relative: &Path) -> Result<Vec<u8>, SpoolError> {
        validate_relative(relative)?;
        let path = self.root.join(relative);
        fs::read(&path).map_err(|error| io_error("read durable object", &path, error))
    }

    pub(crate) fn write_immutable(
        &self,
        relative: &Path,
        bytes: &[u8],
        class: ImmutableClass,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<(), SpoolError> {
        validate_relative(relative)?;
        let final_path = self.root.join(relative);
        let parent = final_path
            .parent()
            .ok_or_else(|| SpoolError::UnsafeRelativePath(relative.to_path_buf()))?;
        if final_path.exists() {
            verify_exact_file(&final_path, bytes)?;
            return Ok(());
        }
        let file_name = final_path
            .file_name()
            .ok_or_else(|| SpoolError::UnsafeRelativePath(relative.to_path_buf()))?;
        let mut temporary_name = b".".to_vec();
        temporary_name.extend_from_slice(file_name.as_bytes());
        temporary_name.extend_from_slice(b".tmp");
        let temporary_path = parent.join(std::ffi::OsStr::from_bytes(&temporary_name));

        let mut file = match OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&temporary_path)
        {
            Ok(mut file) => {
                file.write_all(bytes).map_err(|error| {
                    io_error("write immutable temporary", &temporary_path, error)
                })?;
                faults.after(class.written())?;
                file
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                let existing = fs::read(&temporary_path).map_err(|error| {
                    io_error("read immutable temporary", &temporary_path, error)
                })?;
                if existing != bytes {
                    fs::remove_file(&temporary_path).map_err(|error| {
                        io_error("remove stale immutable temporary", &temporary_path, error)
                    })?;
                    sync_directory(parent)?;
                    let mut file = OpenOptions::new()
                        .create_new(true)
                        .write(true)
                        .mode(0o600)
                        .open(&temporary_path)
                        .map_err(|error| {
                            io_error("recreate immutable temporary", &temporary_path, error)
                        })?;
                    file.write_all(bytes).map_err(|error| {
                        io_error("rewrite immutable temporary", &temporary_path, error)
                    })?;
                    faults.after(class.written())?;
                    file
                } else {
                    OpenOptions::new()
                        .write(true)
                        .open(&temporary_path)
                        .map_err(|error| {
                            io_error("reopen immutable temporary", &temporary_path, error)
                        })?
                }
            }
            Err(error) => {
                return Err(io_error(
                    "create immutable temporary",
                    &temporary_path,
                    error,
                ));
            }
        };
        file.flush()
            .map_err(|error| io_error("flush immutable temporary", &temporary_path, error))?;
        file.sync_all()
            .map_err(|error| io_error("fsync immutable temporary", &temporary_path, error))?;
        faults.after(class.synced())?;
        drop(file);
        match rename_noreplace(&temporary_path, &final_path)? {
            true => faults.after(class.renamed())?,
            false => {
                verify_exact_file(&final_path, bytes)?;
                if temporary_path.exists() {
                    fs::remove_file(&temporary_path).map_err(|error| {
                        io_error(
                            "remove redundant immutable temporary",
                            &temporary_path,
                            error,
                        )
                    })?;
                }
            }
        }
        sync_directory(parent)?;
        faults.after(class.directory_synced())?;
        Ok(())
    }

    pub(crate) fn replace_pointer(
        &self,
        name: &str,
        bytes: &[u8],
        class: PointerClass,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<(), SpoolError> {
        if name.contains('/') || name.is_empty() {
            return Err(SpoolError::UnsafeRelativePath(PathBuf::from(name)));
        }
        let final_path = self.root.join(name);
        let temporary_path = self.root.join(format!(".{name}.tmp"));
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(&temporary_path)
            .map_err(|error| io_error("create pointer temporary", &temporary_path, error))?;
        file.write_all(bytes)
            .map_err(|error| io_error("write pointer temporary", &temporary_path, error))?;
        faults.after(class.written())?;
        file.flush()
            .map_err(|error| io_error("flush pointer temporary", &temporary_path, error))?;
        file.sync_all()
            .map_err(|error| io_error("fsync pointer temporary", &temporary_path, error))?;
        faults.after(class.synced())?;
        drop(file);
        fs::rename(&temporary_path, &final_path)
            .map_err(|error| io_error("replace pointer", &final_path, error))?;
        faults.after(class.renamed())?;
        sync_directory(&self.root)?;
        faults.after(class.directory_synced())?;
        Ok(())
    }
}

/// Exact identity checks required before resuming collection or sync.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RecoveryExpectation {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Canonical-spool identity.
    pub canonical_spool_id: Digest,
    /// Persistent collection/writer identity digest.
    pub archive_identity_digest: Digest,
    /// Credential-free normalized archive target digest.
    pub archive_target_digest: Digest,
    /// Archive-key provider digest.
    pub archive_key_digest: Digest,
    /// Writer compatibility ID.
    pub writer_compatibility_id: Digest,
}

impl RecoveryExpectation {
    /// Constructs exact expectations from verified genesis.
    #[must_use]
    pub const fn from_genesis(genesis: &GenesisV1) -> Self {
        Self {
            archive_id: genesis.archive_id,
            canonical_spool_id: genesis.canonical_spool_id,
            archive_identity_digest: genesis.archive_identity_digest,
            archive_target_digest: genesis.archive_target_digest,
            archive_key_digest: genesis.archive_key_digest,
            writer_compatibility_id: genesis.writer_compatibility_id,
        }
    }
}

/// Authoritative local head/index/genesis state under one qualified lock.
#[derive(Debug)]
pub struct LocalArchiveRepository {
    spool: Arc<QualifiedSpool>,
    head: HeadDescriptorV1,
    index: IndexSnapshot,
    genesis: GenesisV1,
    latest_collection_session_id: Option<SessionId>,
    rolled_back_current: bool,
}

struct GenerationCommit<'a> {
    mutation_set: &'a IndexMutationSetV1,
    transaction_kind: GenerationTransactionKind,
    archive_state: ArchiveState,
    session_id: Option<SessionId>,
    session_anchor: Option<SessionAnchorV1>,
    termination_reason: Option<String>,
    next_record_seq: u64,
    active_wal_segment_id: Option<Digest>,
}

impl LocalArchiveRepository {
    /// Creates generation zero and `LOCAL-LATEST` exactly once before activation.
    pub fn create_new(
        spool: QualifiedSpool,
        genesis: GenesisV1,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<Self, SpoolError> {
        if spool.root.join(LOCAL_LATEST).exists() {
            return Err(SpoolError::ArchiveAlreadyExists);
        }
        let spool = Arc::new(spool);
        genesis.validate().map_err(SpoolError::Manifest)?;
        let index = IndexSnapshot::empty().map_err(SpoolError::Index)?;
        persist_index(&spool, &index, faults)?;
        let initial_session_anchor = genesis
            .initial_session_anchor()
            .map_err(SpoolError::Manifest)?;
        let generation = GenerationObjectV1::new(GenerationV1 {
            archive_id: genesis.archive_id,
            local_commit_seq: 0,
            parent_generation_hash: None,
            genesis_hash: None,
            index_root: index.root().clone(),
            archive_state: ArchiveState::Open,
            transaction_kind: GenerationTransactionKind::Genesis,
            session_id: genesis.initial_session_id,
            session_anchor: initial_session_anchor,
            next_record_seq: 0,
            active_wal_segment_id: None,
            mutations: vec![],
            genesis: Some(genesis.clone()),
            termination_reason: None,
        })
        .map_err(SpoolError::Manifest)?;
        spool.write_immutable(
            Path::new(&generation.key),
            &generation.bytes,
            ImmutableClass::Generation,
            faults,
        )?;
        let head = HeadDescriptorV1::from_generation(&generation).map_err(SpoolError::Manifest)?;
        let pointer = LocalLatestV1 {
            current: head.clone(),
            preceding: None,
        };
        spool.replace_pointer(
            LOCAL_LATEST,
            &pointer.canonical_bytes(),
            PointerClass::Primary,
            faults,
        )?;
        Ok(Self {
            spool,
            head,
            index,
            latest_collection_session_id: generation.generation.session_id,
            genesis,
            rolled_back_current: false,
        })
    }

    /// Recovers only from the checksummed pointer and its content-addressed graph.
    pub fn recover(
        spool: QualifiedSpool,
        expectation: RecoveryExpectation,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<Self, SpoolError> {
        let repository = Self::recover_existing(
            spool,
            expectation.archive_id,
            expectation.archive_target_digest,
            faults,
        )?;
        verify_expectation(&repository.genesis, expectation)?;
        Ok(repository)
    }

    /// Recovers a sealed/sync-only archive while discovering genesis-owned identities.
    ///
    /// This path still verifies the complete pointer, generation ancestry,
    /// index graph, and embedded genesis under the qualified spool lock. It
    /// relaxes only caller-supplied identity knowledge: source-free remote
    /// finalization authors the archive ID, while spool/key/writer identities
    /// remain immutable facts discovered from the verified genesis.
    pub fn recover_existing(
        spool: QualifiedSpool,
        expected_archive_id: ArchiveId,
        expected_archive_target_digest: Digest,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<Self, SpoolError> {
        let spool = Arc::new(spool);
        let pointer_path = spool.root.join(LOCAL_LATEST);
        let pointer_bytes = fs::read(&pointer_path)
            .map_err(|error| io_error("read LOCAL-LATEST", &pointer_path, error))?;
        let pointer = LocalLatestV1::decode(&pointer_bytes).map_err(SpoolError::Manifest)?;
        let current = verify_head(&spool, &pointer.current);
        let (verified, rolled_back_current) = match current {
            Ok(verified) => (verified, false),
            Err(current_error) => {
                let Some(preceding) = pointer.preceding.as_ref() else {
                    return Err(current_error);
                };
                match verify_head(&spool, preceding) {
                    Ok(verified) => {
                        let repaired = LocalLatestV1 {
                            current: preceding.clone(),
                            preceding: None,
                        };
                        spool.replace_pointer(
                            LOCAL_LATEST,
                            &repaired.canonical_bytes(),
                            PointerClass::Primary,
                            faults,
                        )?;
                        (verified, true)
                    }
                    Err(preceding_error) => {
                        return Err(SpoolError::NoValidHead {
                            current: current_error.to_string(),
                            preceding: preceding_error.to_string(),
                        });
                    }
                }
            }
        };
        if verified.genesis.archive_id != expected_archive_id {
            return Err(SpoolError::IdentityMismatch("archive ID"));
        }
        if verified.genesis.archive_target_digest != expected_archive_target_digest {
            return Err(SpoolError::IdentityMismatch("archive target digest"));
        }
        Ok(Self {
            spool,
            head: verified.head,
            index: verified.index,
            genesis: verified.genesis,
            latest_collection_session_id: verified.latest_collection_session_id,
            rolled_back_current,
        })
    }

    /// Returns the current authoritative head.
    #[must_use]
    pub const fn head(&self) -> &HeadDescriptorV1 {
        &self.head
    }

    /// Returns the complete current index snapshot.
    #[must_use]
    pub const fn index(&self) -> &IndexSnapshot {
        &self.index
    }

    /// Returns verified generation-zero identity.
    #[must_use]
    pub const fn genesis(&self) -> &GenesisV1 {
        &self.genesis
    }

    /// Returns the latest collection session from the verified current generation.
    #[must_use]
    pub const fn latest_collection_session_id(&self) -> Option<SessionId> {
        self.latest_collection_session_id
    }

    /// Reconstructs the exact manifest-named open WAL header without directory discovery.
    pub fn active_wal_header(
        &self,
        table_schema_fingerprints: Vec<(TableId, Digest)>,
    ) -> Result<Option<WalSegmentHeaderV1>, SpoolError> {
        let Some(active_segment_id) = self.head.active_wal_segment_id else {
            return Ok(None);
        };
        let mut sequence = self.head.local_commit_seq;
        let mut hash = self.head.generation_hash;
        loop {
            let key = generation_key(sequence, hash);
            let bytes = self.spool.read_relative(Path::new(&key))?;
            let object = GenerationObjectV1::decode(&bytes).map_err(SpoolError::Manifest)?;
            if object.hash != hash || object.key != key {
                return Err(SpoolError::IdentityMismatch("active WAL generation"));
            }
            if object.generation.transaction_kind == GenerationTransactionKind::SessionStarted
                && object.generation.active_wal_segment_id == Some(active_segment_id)
            {
                let session_id = object
                    .generation
                    .session_id
                    .ok_or(SpoolError::IdentityMismatch("active WAL session"))?;
                let previous_head_hash = object
                    .generation
                    .parent_generation_hash
                    .ok_or(SpoolError::IdentityMismatch("active WAL parent"))?;
                let genesis_hash = object
                    .generation
                    .genesis_hash
                    .ok_or(SpoolError::IdentityMismatch("active WAL genesis"))?;
                let session_anchor = object
                    .generation
                    .session_anchor
                    .ok_or(SpoolError::IdentityMismatch("active WAL session anchor"))?;
                let header = WalSegmentHeaderV1::new(
                    self.head.archive_id,
                    session_id,
                    previous_head_hash,
                    genesis_hash,
                    self.genesis.writer_compatibility_id,
                    object.generation.next_record_seq,
                    session_anchor,
                    table_schema_fingerprints,
                )
                .map_err(SpoolError::Wal)?;
                if header.segment_id != active_segment_id {
                    return Err(SpoolError::IdentityMismatch(
                        "active WAL schema fingerprints",
                    ));
                }
                return Ok(Some(header));
            }
            if sequence == 0 {
                return Err(SpoolError::IdentityMismatch(
                    "active WAL session generation",
                ));
            }
            sequence -= 1;
            hash = object
                .generation
                .parent_generation_hash
                .ok_or(SpoolError::IdentityMismatch("active WAL ancestry"))?;
        }
    }

    /// Reports whether recovery replaced a bad current head with its preceding head.
    #[must_use]
    pub const fn rolled_back_current(&self) -> bool {
        self.rolled_back_current
    }

    /// Returns the held qualified spool.
    #[must_use]
    pub fn spool(&self) -> &QualifiedSpool {
        &self.spool
    }

    /// Installs one exact session/WAL authority before the WAL file is opened.
    pub fn start_session(
        &mut self,
        header: &WalSegmentHeaderV1,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<&HeadDescriptorV1, SpoolError> {
        if self.head.active_wal_segment_id == Some(header.segment_id) {
            if self.head.archive_state == ArchiveState::Open
                && self.head.parent_generation_hash == Some(header.previous_head_hash)
                && self.head.next_record_seq == header.first_record_seq
                && self.head.session_anchor == Some(header.session_anchor)
            {
                return Ok(&self.head);
            }
            return Err(SpoolError::IdentityMismatch("active WAL session"));
        }
        if self.head.archive_state != ArchiveState::Open
            || self.head.active_wal_segment_id.is_some()
            || header.archive_id != self.head.archive_id
            || header.previous_head_hash != self.head.generation_hash
            || header.genesis_hash != self.head.genesis_hash
            || header.writer_compatibility_id != self.genesis.writer_compatibility_id
            || header.first_record_seq != self.head.next_record_seq
        {
            return Err(SpoolError::IdentityMismatch("session WAL header"));
        }
        let mutation_set =
            IndexMutationSetV1::new(Vec::new(), Vec::new()).map_err(SpoolError::Index)?;
        self.commit_generation(
            GenerationCommit {
                mutation_set: &mutation_set,
                transaction_kind: GenerationTransactionKind::SessionStarted,
                archive_state: ArchiveState::Open,
                session_id: Some(header.session_id),
                session_anchor: Some(header.session_anchor),
                termination_reason: None,
                next_record_seq: self.head.next_record_seq,
                active_wal_segment_id: Some(header.segment_id),
            },
            faults,
        )
    }

    /// Advances indexed coverage and the authoritative global sequence watermark.
    pub fn checkpoint(
        &mut self,
        mutation_set: &IndexMutationSetV1,
        session_id: SessionId,
        next_record_seq: u64,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<&HeadDescriptorV1, SpoolError> {
        let active_wal_segment_id =
            self.head
                .active_wal_segment_id
                .ok_or(SpoolError::InvalidTransaction(
                    "checkpoint requires one active WAL segment",
                ))?;
        if next_record_seq < self.head.next_record_seq {
            return Err(SpoolError::InvalidTransaction(
                "checkpoint cannot regress next record sequence",
            ));
        }
        self.commit_generation(
            GenerationCommit {
                mutation_set,
                transaction_kind: GenerationTransactionKind::Checkpoint,
                archive_state: self.head.archive_state,
                session_id: Some(session_id),
                session_anchor: self.head.session_anchor,
                termination_reason: None,
                next_record_seq,
                active_wal_segment_id: Some(active_wal_segment_id),
            },
            faults,
        )
    }

    /// Clears one sealed session WAL while preserving an open archive for rotation.
    pub fn close_session(
        &mut self,
        session_id: SessionId,
        next_record_seq: u64,
        sealed_wal: &SealedWalSegment,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<&HeadDescriptorV1, SpoolError> {
        let expected_next_record_seq = sealed_wal
            .last_record_seq()
            .checked_add(1)
            .ok_or(SpoolError::SequenceOverflow)?;
        if self.head.archive_state != ArchiveState::Open
            || self.head.active_wal_segment_id != Some(sealed_wal.segment_id())
            || self.latest_collection_session_id != Some(session_id)
            || next_record_seq != expected_next_record_seq
            || next_record_seq != self.head.next_record_seq
        {
            return Err(SpoolError::InvalidTransaction(
                "session close requires its sealed, fully checkpointed active WAL",
            ));
        }
        let mutation_set =
            IndexMutationSetV1::new(Vec::new(), Vec::new()).map_err(SpoolError::Index)?;
        self.commit_generation(
            GenerationCommit {
                mutation_set: &mutation_set,
                transaction_kind: GenerationTransactionKind::SessionClosed,
                archive_state: ArchiveState::Open,
                session_id: Some(session_id),
                session_anchor: self.head.session_anchor,
                termination_reason: None,
                next_record_seq,
                active_wal_segment_id: None,
            },
            faults,
        )
    }

    /// Closes the named WAL authority at an already-checkpointed final watermark.
    pub fn finalize_local(
        &mut self,
        session_id: SessionId,
        next_record_seq: u64,
        termination_reason: String,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<&HeadDescriptorV1, SpoolError> {
        if self.head.active_wal_segment_id.is_none() || next_record_seq != self.head.next_record_seq
        {
            return Err(SpoolError::InvalidTransaction(
                "local finalization requires the checkpointed active WAL watermark",
            ));
        }
        let mutation_set =
            IndexMutationSetV1::new(Vec::new(), Vec::new()).map_err(SpoolError::Index)?;
        self.commit_generation(
            GenerationCommit {
                mutation_set: &mutation_set,
                transaction_kind: GenerationTransactionKind::LocalFinalization,
                archive_state: ArchiveState::LocallyFinalized,
                session_id: Some(session_id),
                session_anchor: self.head.session_anchor,
                termination_reason: Some(termination_reason),
                next_record_seq,
                active_wal_segment_id: None,
            },
            faults,
        )
    }

    fn commit_generation(
        &mut self,
        commit: GenerationCommit<'_>,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<&HeadDescriptorV1, SpoolError> {
        if commit.transaction_kind == GenerationTransactionKind::Genesis {
            return Err(SpoolError::InvalidTransaction(
                "descendant commit cannot use genesis transaction kind",
            ));
        }
        let next_index = self
            .index
            .apply(commit.mutation_set, MutationMode::Normal)
            .map_err(SpoolError::Index)?;
        let commit_seq = self
            .head
            .local_commit_seq
            .checked_add(1)
            .ok_or(SpoolError::SequenceOverflow)?;
        let generation = GenerationObjectV1::new(GenerationV1 {
            archive_id: self.head.archive_id,
            local_commit_seq: commit_seq,
            parent_generation_hash: Some(self.head.generation_hash),
            genesis_hash: Some(self.head.genesis_hash),
            index_root: next_index.root().clone(),
            archive_state: commit.archive_state,
            transaction_kind: commit.transaction_kind,
            session_id: commit.session_id,
            session_anchor: commit.session_anchor,
            next_record_seq: commit.next_record_seq,
            active_wal_segment_id: commit.active_wal_segment_id,
            mutations: GenerationMutationV1::from_set(commit.mutation_set),
            genesis: None,
            termination_reason: commit.termination_reason,
        })
        .map_err(SpoolError::Manifest)?;
        let parent_bytes = self
            .spool
            .read_relative(Path::new(&self.head.generation_key))?;
        let parent = GenerationObjectV1::decode(&parent_bytes).map_err(SpoolError::Manifest)?;
        generation
            .generation
            .validate_descendant_of(&parent.generation)
            .map_err(SpoolError::Manifest)?;
        persist_index(&self.spool, &next_index, faults)?;
        self.spool.write_immutable(
            Path::new(&generation.key),
            &generation.bytes,
            ImmutableClass::Generation,
            faults,
        )?;
        let next_head =
            HeadDescriptorV1::from_generation(&generation).map_err(SpoolError::Manifest)?;
        let pointer = LocalLatestV1 {
            current: next_head.clone(),
            preceding: Some(self.head.clone()),
        };
        self.spool.replace_pointer(
            LOCAL_LATEST,
            &pointer.canonical_bytes(),
            PointerClass::Primary,
            faults,
        )?;
        self.head = next_head;
        self.index = next_index;
        self.latest_collection_session_id = generation.generation.session_id;
        self.rolled_back_current = false;
        Ok(&self.head)
    }

    /// Creates one new open WAL segment bound to the current head/genesis/writer identity.
    pub fn create_wal<'a>(
        &'a self,
        header: WalSegmentHeaderV1,
        faults: &'a dyn DurabilityFaultInjector,
    ) -> Result<LocalWalWriter<'a>, SpoolError> {
        if header.archive_id != self.head.archive_id
            || self.head.archive_state != ArchiveState::Open
            || self.head.active_wal_segment_id != Some(header.segment_id)
            || self.head.parent_generation_hash != Some(header.previous_head_hash)
            || self.head.next_record_seq != header.first_record_seq
            || header.genesis_hash != self.head.genesis_hash
            || header.writer_compatibility_id != self.genesis.writer_compatibility_id
            || self.head.session_anchor != Some(header.session_anchor)
        {
            return Err(SpoolError::IdentityMismatch("WAL header"));
        }
        LocalWalWriter::create(&self.spool, header, faults)
    }

    /// Creates an owned WAL writer that keeps the spool lock and fault policy alive.
    pub fn create_owned_wal(
        &self,
        header: WalSegmentHeaderV1,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<OwnedLocalWalWriter, SpoolError> {
        if header.archive_id != self.head.archive_id
            || self.head.archive_state != ArchiveState::Open
            || self.head.active_wal_segment_id != Some(header.segment_id)
            || self.head.parent_generation_hash != Some(header.previous_head_hash)
            || self.head.next_record_seq != header.first_record_seq
            || header.genesis_hash != self.head.genesis_hash
            || header.writer_compatibility_id != self.genesis.writer_compatibility_id
            || self.head.session_anchor != Some(header.session_anchor)
        {
            return Err(SpoolError::IdentityMismatch("WAL header"));
        }
        OwnedLocalWalWriter::create(Arc::clone(&self.spool), header, faults)
    }

    /// Recovers one known segment, completing a valid interrupted seal or truncating only a tail.
    pub fn recover_wal(
        &self,
        header: &WalSegmentHeaderV1,
        maximum_frame_bytes: u64,
        faults: &dyn DurabilityFaultInjector,
    ) -> Result<RecoveredWal, SpoolError> {
        let (open_relative, sealed_relative) = wal_paths(header.segment_id);
        let open_path = self.spool.root.join(&open_relative);
        let sealed_path = self.spool.root.join(&sealed_relative);
        if sealed_path.exists() {
            let bytes = fs::read(&sealed_path)
                .map_err(|error| io_error("read sealed WAL", &sealed_path, error))?;
            return RecoveredWal::sealed(&bytes, maximum_frame_bytes).map_err(SpoolError::Wal);
        }
        let bytes =
            fs::read(&open_path).map_err(|error| io_error("read open WAL", &open_path, error))?;
        if let Some(footer_start) = bytes
            .windows(WAL_FOOTER_MAGIC.len())
            .rposition(|window| window == WAL_FOOTER_MAGIC)
        {
            let footer_bytes = bytes.len() - footer_start;
            if footer_bytes >= SEALED_WAL_FOOTER_BYTES {
                let recovered =
                    RecoveredWal::sealed(&bytes, maximum_frame_bytes).map_err(SpoolError::Wal)?;
                match rename_noreplace(&open_path, &sealed_path)? {
                    true => {}
                    false => verify_exact_file(&sealed_path, &bytes)?,
                }
                sync_directory(open_path.parent().expect("WAL has parent"))?;
                return Ok(recovered);
            }
        }
        let recovered = RecoveredWal::open(&bytes, maximum_frame_bytes).map_err(SpoolError::Wal)?;
        if recovered.discarded_tail_bytes != 0 {
            let file = OpenOptions::new()
                .write(true)
                .open(&open_path)
                .map_err(|error| io_error("open WAL for tail truncation", &open_path, error))?;
            file.set_len(
                u64::try_from(recovered.valid_len).map_err(|_| SpoolError::SequenceOverflow)?,
            )
            .map_err(|error| io_error("truncate incomplete WAL tail", &open_path, error))?;
            faults.after(DurabilityEdge::WalTailTruncated)?;
            file.sync_all()
                .map_err(|error| io_error("fsync WAL tail truncation", &open_path, error))?;
            faults.after(DurabilityEdge::WalTailSynced)?;
        }
        Ok(recovered)
    }

    /// Recovers and reopens one unsealed WAL under owned spool/fault lifetimes.
    pub fn resume_owned_wal(
        &self,
        header: &WalSegmentHeaderV1,
        maximum_frame_bytes: u64,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<(OwnedLocalWalWriter, RecoveredWal), SpoolError> {
        if header.archive_id != self.head.archive_id
            || !matches!(
                self.head.archive_state,
                ArchiveState::Open | ArchiveState::StopRequested
            )
            || self.head.active_wal_segment_id != Some(header.segment_id)
            || header.first_record_seq > self.head.next_record_seq
            || header.genesis_hash != self.head.genesis_hash
            || header.writer_compatibility_id != self.genesis.writer_compatibility_id
            || self.head.session_anchor != Some(header.session_anchor)
            || !generation_is_ancestor(&self.spool, &self.head, header.previous_head_hash)?
        {
            return Err(SpoolError::IdentityMismatch("recovered WAL header"));
        }
        let recovered = self.recover_wal(header, maximum_frame_bytes, faults.as_ref())?;
        if recovered.header != *header {
            return Err(SpoolError::IdentityMismatch("recovered WAL header bytes"));
        }
        if recovered.segment_digest.is_some() {
            return Err(SpoolError::InvalidTransaction(
                "sealed WAL cannot resume frame admission",
            ));
        }
        let recovered_next_record_seq =
            recovered
                .frames
                .last()
                .map_or(Ok(header.first_record_seq), |frame| {
                    frame
                        .header()
                        .record_seq
                        .checked_add(1)
                        .ok_or(SpoolError::SequenceOverflow)
                })?;
        if recovered_next_record_seq < self.head.next_record_seq {
            return Err(SpoolError::IdentityMismatch(
                "WAL is behind the manifest sequence watermark",
            ));
        }
        let writer =
            OwnedLocalWalWriter::resume(Arc::clone(&self.spool), &recovered, Arc::clone(&faults))?;
        Ok((writer, recovered))
    }

    /// Reopens or creates an empty WAL named by a just-installed session generation.
    ///
    /// This is the idempotent crash-retry path between `session_started` and
    /// source activation. It consults only the exact manifest-derived paths.
    pub fn resume_or_create_empty_owned_wal(
        &self,
        header: &WalSegmentHeaderV1,
        maximum_frame_bytes: u64,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<OwnedLocalWalWriter, SpoolError> {
        let (open_relative, sealed_relative) = wal_paths(header.segment_id);
        let open_exists = self.spool.root.join(open_relative).exists();
        let sealed_exists = self.spool.root.join(sealed_relative).exists();
        if !open_exists && !sealed_exists {
            return self.create_owned_wal(header.clone(), faults);
        }
        let (writer, recovered) = self.resume_owned_wal(header, maximum_frame_bytes, faults)?;
        if !recovered.frames.is_empty() {
            return Err(SpoolError::InvalidTransaction(
                "installed session WAL was already activated",
            ));
        }
        Ok(writer)
    }
}

/// File-backed WAL writer whose lifetime cannot outlive the qualified spool lock.
pub struct LocalWalWriter<'a> {
    _spool: &'a QualifiedSpool,
    faults: &'a dyn DurabilityFaultInjector,
    open_path: PathBuf,
    sealed_path: PathBuf,
    file: File,
    builder: WalSegmentBuilder,
    poisoned: bool,
}

impl Debug for LocalWalWriter<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalWalWriter")
            .field("open_path", &self.open_path)
            .field("sealed_path", &self.sealed_path)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<'a> LocalWalWriter<'a> {
    fn create(
        spool: &'a QualifiedSpool,
        header: WalSegmentHeaderV1,
        faults: &'a dyn DurabilityFaultInjector,
    ) -> Result<Self, SpoolError> {
        let (open_relative, sealed_relative) = wal_paths(header.segment_id);
        let open_path = spool.root.join(open_relative);
        let sealed_path = spool.root.join(sealed_relative);
        if sealed_path.exists() || open_path.exists() {
            return Err(SpoolError::WalAlreadyExists(header.segment_id));
        }
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(&open_path)
            .map_err(|error| io_error("create open WAL", &open_path, error))?;
        faults.after(DurabilityEdge::WalFileCreated)?;
        let builder = WalSegmentBuilder::new(header).map_err(SpoolError::Wal)?;
        let header_bytes = builder.open_bytes();
        file.write_all(&header_bytes)
            .map_err(|error| io_error("write WAL header", &open_path, error))?;
        faults.after(DurabilityEdge::WalHeaderWritten)?;
        file.sync_all()
            .map_err(|error| io_error("fsync WAL header", &open_path, error))?;
        faults.after(DurabilityEdge::WalHeaderSynced)?;
        sync_directory(open_path.parent().expect("WAL has parent"))?;
        faults.after(DurabilityEdge::WalDirectorySynced)?;
        Ok(Self {
            _spool: spool,
            faults,
            open_path,
            sealed_path,
            file,
            builder,
            poisoned: false,
        })
    }

    /// Appends and fsyncs one complete frame before updating writer state.
    pub fn append(&mut self, frame: &WalFrame) -> Result<(), SpoolError> {
        if self.poisoned {
            return Err(SpoolError::WalPoisoned);
        }
        let mut next = self.builder.clone();
        next.append(frame).map_err(SpoolError::Wal)?;
        let encoded = frame.encode().map_err(SpoolError::Wal)?;
        let result = (|| {
            self.file
                .seek(SeekFrom::End(0))
                .map_err(|error| io_error("seek open WAL", &self.open_path, error))?;
            self.file
                .write_all(&encoded)
                .map_err(|error| io_error("append WAL frame", &self.open_path, error))?;
            self.faults.after(DurabilityEdge::WalFrameWritten)?;
            self.file
                .sync_all()
                .map_err(|error| io_error("fsync WAL frame", &self.open_path, error))?;
            self.faults.after(DurabilityEdge::WalFrameSynced)?;
            Ok(())
        })();
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        self.builder = next;
        Ok(())
    }

    /// Returns the immutable segment header bound at file creation.
    #[must_use]
    pub const fn header(&self) -> &WalSegmentHeaderV1 {
        self.builder.header()
    }

    /// Returns the durable cryptographic prefix after the last fsynced frame.
    #[must_use]
    pub const fn durable_prefix(&self) -> Digest {
        self.builder.prefix()
    }

    /// Returns the final durable record sequence, absent before the first frame.
    #[must_use]
    pub const fn last_record_seq(&self) -> Option<u64> {
        self.builder.last_record_seq()
    }

    /// Seals, fsyncs, create-only renames, and directory-fsyncs the whole segment.
    pub fn seal(mut self) -> Result<SealedWalSegment, SpoolError> {
        if self.poisoned {
            return Err(SpoolError::WalPoisoned);
        }
        let sealed = self.builder.clone().seal().map_err(SpoolError::Wal)?;
        let open_len = self.builder.open_bytes().len();
        let footer = &sealed.bytes()[open_len..];
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|error| io_error("seek WAL footer", &self.open_path, error))?;
        self.file
            .write_all(footer)
            .map_err(|error| io_error("write WAL footer", &self.open_path, error))?;
        self.faults.after(DurabilityEdge::WalFooterWritten)?;
        self.file
            .sync_all()
            .map_err(|error| io_error("fsync sealed WAL", &self.open_path, error))?;
        self.faults.after(DurabilityEdge::WalSealSynced)?;
        drop(self.file);
        match rename_noreplace(&self.open_path, &self.sealed_path)? {
            true => self.faults.after(DurabilityEdge::WalRenamed)?,
            false => {
                verify_exact_file(&self.sealed_path, sealed.bytes())?;
                if self.open_path.exists() {
                    fs::remove_file(&self.open_path).map_err(|error| {
                        io_error("remove redundant open WAL", &self.open_path, error)
                    })?;
                }
            }
        }
        sync_directory(self.sealed_path.parent().expect("WAL has parent"))?;
        self.faults.after(DurabilityEdge::WalSealDirectorySynced)?;
        Ok(sealed)
    }
}

/// File-backed WAL writer owning the qualified spool lock and fault policy.
pub struct OwnedLocalWalWriter {
    _spool: Arc<QualifiedSpool>,
    faults: Arc<dyn DurabilityFaultInjector>,
    open_path: PathBuf,
    sealed_path: PathBuf,
    file: File,
    builder: WalSegmentBuilder,
    poisoned: bool,
}

impl Debug for OwnedLocalWalWriter {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedLocalWalWriter")
            .field("open_path", &self.open_path)
            .field("sealed_path", &self.sealed_path)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl OwnedLocalWalWriter {
    fn create(
        spool: Arc<QualifiedSpool>,
        header: WalSegmentHeaderV1,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<Self, SpoolError> {
        let (open_relative, sealed_relative) = wal_paths(header.segment_id);
        let open_path = spool.root.join(open_relative);
        let sealed_path = spool.root.join(sealed_relative);
        if sealed_path.exists() || open_path.exists() {
            return Err(SpoolError::WalAlreadyExists(header.segment_id));
        }
        let mut file = OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .mode(0o600)
            .open(&open_path)
            .map_err(|error| io_error("create open WAL", &open_path, error))?;
        faults.after(DurabilityEdge::WalFileCreated)?;
        let builder = WalSegmentBuilder::new(header).map_err(SpoolError::Wal)?;
        let header_bytes = builder.open_bytes();
        file.write_all(&header_bytes)
            .map_err(|error| io_error("write WAL header", &open_path, error))?;
        faults.after(DurabilityEdge::WalHeaderWritten)?;
        file.sync_all()
            .map_err(|error| io_error("fsync WAL header", &open_path, error))?;
        faults.after(DurabilityEdge::WalHeaderSynced)?;
        sync_directory(open_path.parent().expect("WAL has parent"))?;
        faults.after(DurabilityEdge::WalDirectorySynced)?;
        Ok(Self {
            _spool: spool,
            faults,
            open_path,
            sealed_path,
            file,
            builder,
            poisoned: false,
        })
    }

    fn resume(
        spool: Arc<QualifiedSpool>,
        recovered: &RecoveredWal,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<Self, SpoolError> {
        let (open_relative, sealed_relative) = wal_paths(recovered.header.segment_id);
        let open_path = spool.root.join(open_relative);
        let sealed_path = spool.root.join(sealed_relative);
        if sealed_path.exists() || recovered.segment_digest.is_some() {
            return Err(SpoolError::InvalidTransaction(
                "sealed WAL cannot resume frame admission",
            ));
        }
        let mut builder =
            WalSegmentBuilder::new(recovered.header.clone()).map_err(SpoolError::Wal)?;
        for frame in &recovered.frames {
            builder.append(frame).map_err(SpoolError::Wal)?;
        }
        if builder.open_bytes().len() != recovered.valid_len
            || builder.prefix() != recovered.final_prefix
        {
            return Err(SpoolError::IdentityMismatch("recovered WAL builder"));
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&open_path)
            .map_err(|error| io_error("reopen recovered WAL", &open_path, error))?;
        let physical_len = usize::try_from(
            file.metadata()
                .map_err(|error| io_error("stat recovered WAL", &open_path, error))?
                .len(),
        )
        .map_err(|_| SpoolError::SequenceOverflow)?;
        if physical_len != recovered.valid_len {
            return Err(SpoolError::IdentityMismatch("recovered WAL length"));
        }
        Ok(Self {
            _spool: spool,
            faults,
            open_path,
            sealed_path,
            file,
            builder,
            poisoned: false,
        })
    }

    /// Appends and fsyncs one complete frame before updating writer state.
    pub fn append(&mut self, frame: &WalFrame) -> Result<(), SpoolError> {
        if self.poisoned {
            return Err(SpoolError::WalPoisoned);
        }
        let mut next = self.builder.clone();
        next.append(frame).map_err(SpoolError::Wal)?;
        let encoded = frame.encode().map_err(SpoolError::Wal)?;
        let result = (|| {
            self.file
                .seek(SeekFrom::End(0))
                .map_err(|error| io_error("seek open WAL", &self.open_path, error))?;
            self.file
                .write_all(&encoded)
                .map_err(|error| io_error("append WAL frame", &self.open_path, error))?;
            self.faults.after(DurabilityEdge::WalFrameWritten)?;
            self.file
                .sync_all()
                .map_err(|error| io_error("fsync WAL frame", &self.open_path, error))?;
            self.faults.after(DurabilityEdge::WalFrameSynced)?;
            Ok(())
        })();
        if let Err(error) = result {
            self.poisoned = true;
            return Err(error);
        }
        self.builder = next;
        Ok(())
    }

    /// Returns the immutable segment header bound at file creation.
    #[must_use]
    pub const fn header(&self) -> &WalSegmentHeaderV1 {
        self.builder.header()
    }

    /// Returns the durable cryptographic prefix after the last fsynced frame.
    #[must_use]
    pub const fn durable_prefix(&self) -> Digest {
        self.builder.prefix()
    }

    /// Returns the final durable record sequence, absent before the first frame.
    #[must_use]
    pub const fn last_record_seq(&self) -> Option<u64> {
        self.builder.last_record_seq()
    }

    /// Seals, fsyncs, create-only renames, and directory-fsyncs the segment.
    pub fn seal(mut self) -> Result<SealedWalSegment, SpoolError> {
        if self.poisoned {
            return Err(SpoolError::WalPoisoned);
        }
        let sealed = self.builder.clone().seal().map_err(SpoolError::Wal)?;
        let open_len = self.builder.open_bytes().len();
        let footer = &sealed.bytes()[open_len..];
        self.file
            .seek(SeekFrom::End(0))
            .map_err(|error| io_error("seek WAL footer", &self.open_path, error))?;
        self.file
            .write_all(footer)
            .map_err(|error| io_error("write WAL footer", &self.open_path, error))?;
        self.faults.after(DurabilityEdge::WalFooterWritten)?;
        self.file
            .sync_all()
            .map_err(|error| io_error("fsync sealed WAL", &self.open_path, error))?;
        self.faults.after(DurabilityEdge::WalSealSynced)?;
        drop(self.file);
        match rename_noreplace(&self.open_path, &self.sealed_path)? {
            true => self.faults.after(DurabilityEdge::WalRenamed)?,
            false => {
                verify_exact_file(&self.sealed_path, sealed.bytes())?;
                if self.open_path.exists() {
                    fs::remove_file(&self.open_path).map_err(|error| {
                        io_error("remove redundant open WAL", &self.open_path, error)
                    })?;
                }
            }
        }
        sync_directory(self.sealed_path.parent().expect("WAL has parent"))?;
        self.faults.after(DurabilityEdge::WalSealDirectorySynced)?;
        Ok(sealed)
    }
}

/// Local spool or transaction failure.
#[derive(Debug)]
pub enum SpoolError {
    /// Filesystem operation failed.
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying OS error.
        source: io::Error,
    },
    /// The authored spool path is not absolute.
    RelativeSpoolPath(PathBuf),
    /// A path component is a symlink or expected directory is not a directory.
    SymlinkOrNonDirectory(PathBuf),
    /// Filesystem family is unknown, networked, or FUSE-backed.
    UnsupportedFilesystem(i64),
    /// Another process/handle holds the lifetime lock.
    LockBusy(PathBuf),
    /// Lock file does not have stable regular single-link semantics.
    UnqualifiedLockFile(PathBuf),
    /// Relative durable-object path contains traversal/root components.
    UnsafeRelativePath(PathBuf),
    /// Atomic create-only rename is unavailable.
    AtomicRenameUnavailable,
    /// A content-addressed final path contains unequal bytes.
    ImmutableContentMismatch(PathBuf),
    /// Selected post-operation crash edge failed.
    FaultInjected(DurabilityEdge),
    /// Archive pointer already exists under create-new.
    ArchiveAlreadyExists,
    /// Neither current nor preceding head could be verified.
    NoValidHead {
        /// Current-head verification error.
        current: String,
        /// Preceding-head verification error.
        preceding: String,
    },
    /// Exact collect/sync resume identity differs from genesis.
    IdentityMismatch(&'static str),
    /// Manifest validation failed.
    Manifest(ManifestError),
    /// Persistent-index validation failed.
    Index(IndexError),
    /// WAL validation failed.
    Wal(WalError),
    /// Descendant sequence overflowed.
    SequenceOverflow,
    /// Descendant transaction uses a prohibited shape.
    InvalidTransaction(&'static str),
    /// A known WAL segment path already exists.
    WalAlreadyExists(Digest),
    /// A failed append poisoned the in-process writer handle.
    WalPoisoned,
}

impl Display for SpoolError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                path,
                source,
            } => write!(
                formatter,
                "{operation} at {} failed: {source}",
                path.display()
            ),
            Self::RelativeSpoolPath(path) => {
                write!(formatter, "spool path must be absolute: {}", path.display())
            }
            Self::SymlinkOrNonDirectory(path) => write!(
                formatter,
                "spool path contains a symlink or non-directory: {}",
                path.display()
            ),
            Self::UnsupportedFilesystem(kind) => write!(
                formatter,
                "spool filesystem type 0x{kind:x} is not in the local allowlist"
            ),
            Self::LockBusy(path) => write!(
                formatter,
                "archive spool is already locked: {}",
                path.display()
            ),
            Self::UnqualifiedLockFile(path) => write!(
                formatter,
                "archive lock lacks stable inode/link semantics: {}",
                path.display()
            ),
            Self::UnsafeRelativePath(path) => {
                write!(formatter, "unsafe spool-relative path: {}", path.display())
            }
            Self::AtomicRenameUnavailable => {
                formatter.write_str("create-only atomic rename is unavailable")
            }
            Self::ImmutableContentMismatch(path) => write!(
                formatter,
                "immutable content mismatch at {}",
                path.display()
            ),
            Self::FaultInjected(edge) => write!(formatter, "injected crash after {edge:?}"),
            Self::ArchiveAlreadyExists => {
                formatter.write_str("archive LOCAL-LATEST already exists")
            }
            Self::NoValidHead { current, preceding } => write!(
                formatter,
                "current head invalid ({current}); preceding head invalid ({preceding})"
            ),
            Self::IdentityMismatch(field) => {
                write!(formatter, "archive resume identity mismatch: {field}")
            }
            Self::Manifest(error) => write!(formatter, "archive manifest failure: {error}"),
            Self::Index(error) => write!(formatter, "archive index failure: {error}"),
            Self::Wal(error) => write!(formatter, "archive WAL failure: {error}"),
            Self::SequenceOverflow => formatter.write_str("archive sequence overflow"),
            Self::InvalidTransaction(message) => {
                write!(formatter, "invalid archive transaction: {message}")
            }
            Self::WalAlreadyExists(id) => write!(formatter, "WAL segment {id} already exists"),
            Self::WalPoisoned => {
                formatter.write_str("WAL writer is poisoned after an uncertain append")
            }
        }
    }
}

impl std::error::Error for SpoolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Manifest(error) => Some(error),
            Self::Index(error) => Some(error),
            Self::Wal(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ImmutableClass {
    Index,
    Generation,
    Partition,
    ReceiptBatch,
    ReceiptIndex,
    ReceiptHead,
}

impl ImmutableClass {
    const fn written(self) -> DurabilityEdge {
        match self {
            Self::Index => DurabilityEdge::IndexTempWritten,
            Self::Generation => DurabilityEdge::GenerationTempWritten,
            Self::Partition => DurabilityEdge::PartitionTempWritten,
            Self::ReceiptBatch => DurabilityEdge::ReceiptBatchTempWritten,
            Self::ReceiptIndex => DurabilityEdge::ReceiptIndexTempWritten,
            Self::ReceiptHead => DurabilityEdge::ReceiptHeadTempWritten,
        }
    }

    const fn synced(self) -> DurabilityEdge {
        match self {
            Self::Index => DurabilityEdge::IndexFileSynced,
            Self::Generation => DurabilityEdge::GenerationFileSynced,
            Self::Partition => DurabilityEdge::PartitionFileSynced,
            Self::ReceiptBatch => DurabilityEdge::ReceiptBatchFileSynced,
            Self::ReceiptIndex => DurabilityEdge::ReceiptIndexFileSynced,
            Self::ReceiptHead => DurabilityEdge::ReceiptHeadFileSynced,
        }
    }

    const fn renamed(self) -> DurabilityEdge {
        match self {
            Self::Index => DurabilityEdge::IndexRenamed,
            Self::Generation => DurabilityEdge::GenerationRenamed,
            Self::Partition => DurabilityEdge::PartitionRenamed,
            Self::ReceiptBatch => DurabilityEdge::ReceiptBatchRenamed,
            Self::ReceiptIndex => DurabilityEdge::ReceiptIndexRenamed,
            Self::ReceiptHead => DurabilityEdge::ReceiptHeadRenamed,
        }
    }

    const fn directory_synced(self) -> DurabilityEdge {
        match self {
            Self::Index => DurabilityEdge::IndexDirectorySynced,
            Self::Generation => DurabilityEdge::GenerationDirectorySynced,
            Self::Partition => DurabilityEdge::PartitionDirectorySynced,
            Self::ReceiptBatch => DurabilityEdge::ReceiptBatchDirectorySynced,
            Self::ReceiptIndex => DurabilityEdge::ReceiptIndexDirectorySynced,
            Self::ReceiptHead => DurabilityEdge::ReceiptHeadDirectorySynced,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum PointerClass {
    Primary,
    Receipt,
}

impl PointerClass {
    const fn written(self) -> DurabilityEdge {
        match self {
            Self::Primary => DurabilityEdge::PointerTempWritten,
            Self::Receipt => DurabilityEdge::ReceiptPointerTempWritten,
        }
    }

    const fn synced(self) -> DurabilityEdge {
        match self {
            Self::Primary => DurabilityEdge::PointerFileSynced,
            Self::Receipt => DurabilityEdge::ReceiptPointerFileSynced,
        }
    }

    const fn renamed(self) -> DurabilityEdge {
        match self {
            Self::Primary => DurabilityEdge::PointerRenamed,
            Self::Receipt => DurabilityEdge::ReceiptPointerRenamed,
        }
    }

    const fn directory_synced(self) -> DurabilityEdge {
        match self {
            Self::Primary => DurabilityEdge::PointerDirectorySynced,
            Self::Receipt => DurabilityEdge::ReceiptPointerDirectorySynced,
        }
    }
}

struct VerifiedHead {
    head: HeadDescriptorV1,
    index: IndexSnapshot,
    genesis: GenesisV1,
    latest_collection_session_id: Option<SessionId>,
}

fn verify_head(
    spool: &QualifiedSpool,
    head: &HeadDescriptorV1,
) -> Result<VerifiedHead, SpoolError> {
    let generation_bytes = spool.read_relative(Path::new(&head.generation_key))?;
    let generation = GenerationObjectV1::decode(&generation_bytes).map_err(SpoolError::Manifest)?;
    if generation.hash != head.generation_hash
        || generation.key != head.generation_key
        || generation.generation.archive_id != head.archive_id
        || generation.generation.local_commit_seq != head.local_commit_seq
        || generation.generation.index_root.root_hash != head.index_root_hash
        || generation.generation.parent_generation_hash != head.parent_generation_hash
        || generation.generation.archive_state != head.archive_state
        || generation.generation.session_anchor != head.session_anchor
        || generation.generation.next_record_seq != head.next_record_seq
        || generation.generation.active_wal_segment_id != head.active_wal_segment_id
    {
        return Err(SpoolError::IdentityMismatch("head/generation"));
    }
    verify_ancestry(spool, &generation, head.genesis_hash)?;
    let page_source = FilePageSource { spool };
    let index = IndexSnapshot::load(generation.generation.index_root.clone(), &page_source)
        .map_err(SpoolError::Index)?;
    let genesis_key = generation_key(0, head.genesis_hash);
    let genesis_bytes = spool.read_relative(Path::new(&genesis_key))?;
    let genesis_object =
        GenerationObjectV1::decode(&genesis_bytes).map_err(SpoolError::Manifest)?;
    if genesis_object.hash != head.genesis_hash || genesis_object.generation.local_commit_seq != 0 {
        return Err(SpoolError::IdentityMismatch("genesis hash"));
    }
    let genesis = genesis_object
        .generation
        .genesis
        .ok_or(SpoolError::IdentityMismatch("genesis payload"))?;
    if genesis.archive_id != head.archive_id {
        return Err(SpoolError::IdentityMismatch("genesis archive ID"));
    }
    Ok(VerifiedHead {
        head: head.clone(),
        index,
        genesis,
        latest_collection_session_id: generation.generation.session_id,
    })
}

fn verify_ancestry(
    spool: &QualifiedSpool,
    current: &GenerationObjectV1,
    genesis_hash: Digest,
) -> Result<(), SpoolError> {
    let mut generation = current.clone();
    loop {
        if generation.generation.local_commit_seq == 0 {
            if generation.hash != genesis_hash {
                return Err(SpoolError::IdentityMismatch("generation ancestry genesis"));
            }
            return Ok(());
        }
        if generation.generation.genesis_hash != Some(genesis_hash) {
            return Err(SpoolError::IdentityMismatch("generation genesis link"));
        }
        let parent_hash = generation
            .generation
            .parent_generation_hash
            .ok_or(SpoolError::IdentityMismatch("generation parent"))?;
        let parent_seq = generation.generation.local_commit_seq - 1;
        let parent_key = generation_key(parent_seq, parent_hash);
        let parent_bytes = spool.read_relative(Path::new(&parent_key))?;
        let parent = GenerationObjectV1::decode(&parent_bytes).map_err(SpoolError::Manifest)?;
        if parent.hash != parent_hash
            || parent.generation.archive_id != generation.generation.archive_id
            || parent.generation.local_commit_seq != parent_seq
        {
            return Err(SpoolError::IdentityMismatch("generation parent bytes"));
        }
        generation
            .generation
            .validate_descendant_of(&parent.generation)
            .map_err(SpoolError::Manifest)?;
        generation = parent;
    }
}

fn generation_is_ancestor(
    spool: &QualifiedSpool,
    current: &HeadDescriptorV1,
    candidate: Digest,
) -> Result<bool, SpoolError> {
    let mut sequence = current.local_commit_seq;
    let mut hash = current.generation_hash;
    loop {
        if hash == candidate {
            return Ok(true);
        }
        if sequence == 0 {
            return Ok(false);
        }
        let key = generation_key(sequence, hash);
        let bytes = spool.read_relative(Path::new(&key))?;
        let generation = GenerationObjectV1::decode(&bytes).map_err(SpoolError::Manifest)?;
        if generation.hash != hash
            || generation.key != key
            || generation.generation.local_commit_seq != sequence
        {
            return Err(SpoolError::IdentityMismatch("generation ancestry"));
        }
        hash = generation
            .generation
            .parent_generation_hash
            .ok_or(SpoolError::IdentityMismatch("generation parent"))?;
        sequence -= 1;
    }
}

fn verify_expectation(
    genesis: &GenesisV1,
    expectation: RecoveryExpectation,
) -> Result<(), SpoolError> {
    if genesis.archive_id != expectation.archive_id {
        return Err(SpoolError::IdentityMismatch("archive ID"));
    }
    if genesis.canonical_spool_id != expectation.canonical_spool_id {
        return Err(SpoolError::IdentityMismatch("canonical spool ID"));
    }
    if genesis.archive_identity_digest != expectation.archive_identity_digest {
        return Err(SpoolError::IdentityMismatch("archive identity digest"));
    }
    if genesis.archive_target_digest != expectation.archive_target_digest {
        return Err(SpoolError::IdentityMismatch("archive target digest"));
    }
    if genesis.archive_key_digest != expectation.archive_key_digest {
        return Err(SpoolError::IdentityMismatch("archive key digest"));
    }
    if genesis.writer_compatibility_id != expectation.writer_compatibility_id {
        return Err(SpoolError::IdentityMismatch("writer compatibility ID"));
    }
    Ok(())
}

fn persist_index(
    spool: &QualifiedSpool,
    index: &IndexSnapshot,
    faults: &dyn DurabilityFaultInjector,
) -> Result<(), SpoolError> {
    for (hash, bytes) in index.page_objects() {
        spool.write_immutable(
            Path::new(&index_root_key(hash)),
            bytes,
            ImmutableClass::Index,
            faults,
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct FilePageSource<'a> {
    spool: &'a QualifiedSpool,
}

impl IndexPageSource for FilePageSource<'_> {
    fn get(&self, hash: Digest) -> Result<Vec<u8>, IndexError> {
        self.spool
            .read_relative(Path::new(&index_root_key(hash)))
            .map_err(|error| IndexError::PageSource(error.to_string()))
    }
}

fn wal_paths(segment_id: Digest) -> (PathBuf, PathBuf) {
    (
        PathBuf::from(format!("wal/{}.open", segment_id.to_hex())),
        PathBuf::from(format!("wal/{}.wal", segment_id.to_hex())),
    )
}

fn validate_relative(path: &Path) -> Result<(), SpoolError> {
    if path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(SpoolError::UnsafeRelativePath(path.to_path_buf()));
    }
    Ok(())
}

fn reject_symlink_components(path: &Path) -> Result<(), SpoolError> {
    let mut current = PathBuf::from("/");
    for component in path.components() {
        match component {
            Component::RootDir => continue,
            Component::Normal(component) => current.push(component),
            _ => return Err(SpoolError::SymlinkOrNonDirectory(path.to_path_buf())),
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(SpoolError::SymlinkOrNonDirectory(current));
            }
            Ok(metadata) if current != path && !metadata.is_dir() => {
                return Err(SpoolError::SymlinkOrNonDirectory(current));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => break,
            Err(error) => return Err(io_error("inspect spool component", &current, error)),
        }
    }
    Ok(())
}

fn acquire_exclusive_lock(file: &File, path: &Path) -> Result<(), SpoolError> {
    // SAFETY: `file` owns a valid open descriptor for the duration of this call.
    let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if matches!(
        error.kind(),
        io::ErrorKind::WouldBlock | io::ErrorKind::PermissionDenied
    ) {
        return Err(SpoolError::LockBusy(path.to_path_buf()));
    }
    Err(io_error("lock archive spool", path, error))
}

fn qualified_filesystem(path: &Path) -> Result<FilesystemKind, SpoolError> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| SpoolError::SymlinkOrNonDirectory(path.to_path_buf()))?;
    // SAFETY: `statfs` writes the initialized out-parameter and `c_path` is NUL-terminated.
    let mut stat: libc::statfs = unsafe { std::mem::zeroed() };
    // SAFETY: pointers remain valid for the duration of the call.
    if unsafe { libc::statfs(c_path.as_ptr(), &mut stat) } != 0 {
        return Err(io_error(
            "statfs archive spool",
            path,
            io::Error::last_os_error(),
        ));
    }
    let kind = stat.f_type as i64;
    match kind {
        0x0000_ef53 => Ok(FilesystemKind::Ext),
        0x5846_5342 => Ok(FilesystemKind::Xfs),
        0x9123_683e => Ok(FilesystemKind::Btrfs),
        0xf2f5_2010 => Ok(FilesystemKind::F2fs),
        0x2fc1_2fc1 => Ok(FilesystemKind::Zfs),
        0x0102_1994 => Ok(FilesystemKind::Tmpfs),
        0x794c_7630 => Ok(FilesystemKind::Overlay),
        _ => Err(SpoolError::UnsupportedFilesystem(kind)),
    }
}

fn filesystem_capacity(path: &Path) -> Result<(u64, u64), SpoolError> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| SpoolError::SymlinkOrNonDirectory(path.to_path_buf()))?;
    // SAFETY: `statvfs` writes the initialized out-parameter and `c_path` is NUL-terminated.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: pointers remain valid for the duration of the call.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return Err(io_error(
            "statvfs archive spool",
            path,
            io::Error::last_os_error(),
        ));
    }
    Ok((stat.f_bavail, stat.f_favail))
}

fn filesystem_budget_capacity(path: &Path) -> Result<(u64, u64), SpoolError> {
    let c_path = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| SpoolError::SymlinkOrNonDirectory(path.to_path_buf()))?;
    // SAFETY: `statvfs` writes the initialized out-parameter and `c_path` is NUL-terminated.
    let mut stat: libc::statvfs = unsafe { std::mem::zeroed() };
    // SAFETY: pointers remain valid for the duration of the call.
    if unsafe { libc::statvfs(c_path.as_ptr(), &mut stat) } != 0 {
        return Err(io_error(
            "statvfs archive spool budget",
            path,
            io::Error::last_os_error(),
        ));
    }
    let fragment_bytes = if stat.f_frsize == 0 {
        stat.f_bsize
    } else {
        stat.f_frsize
    };
    let available_bytes = stat
        .f_bavail
        .checked_mul(fragment_bytes)
        .ok_or(SpoolError::SequenceOverflow)?;
    Ok((available_bytes, stat.f_favail))
}

fn logical_spool_usage(root: &Path) -> Result<(u64, u64), SpoolError> {
    let mut pending = vec![root.to_path_buf()];
    let mut bytes = 0_u64;
    let mut files = 0_u64;
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory)
            .map_err(|error| io_error("read spool for budget", &directory, error))?;
        for entry in entries {
            let entry =
                entry.map_err(|error| io_error("read spool budget entry", &directory, error))?;
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path)
                .map_err(|error| io_error("stat spool budget entry", &path, error))?;
            if metadata.file_type().is_symlink() {
                return Err(SpoolError::SymlinkOrNonDirectory(path));
            }
            if metadata.is_dir() {
                pending.push(path);
            } else if metadata.is_file() {
                bytes = bytes
                    .checked_add(metadata.len())
                    .ok_or(SpoolError::SequenceOverflow)?;
                files = files.checked_add(1).ok_or(SpoolError::SequenceOverflow)?;
            } else {
                return Err(SpoolError::SymlinkOrNonDirectory(path));
            }
        }
    }
    Ok((bytes, files))
}

fn probe_atomic_create_and_rename(root: &Path) -> Result<(), SpoolError> {
    let temporary = root.join(".qualification-probe.tmp");
    let final_path = root.join(".qualification-probe.final");
    for path in [&temporary, &final_path] {
        if path.exists() {
            fs::remove_file(path)
                .map_err(|error| io_error("remove stale qualification probe", path, error))?;
        }
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)
        .map_err(|error| io_error("create qualification probe", &temporary, error))?;
    file.write_all(b"aiperf-qualified-spool-v1")
        .map_err(|error| io_error("write qualification probe", &temporary, error))?;
    file.sync_all()
        .map_err(|error| io_error("fsync qualification probe", &temporary, error))?;
    let before = file
        .metadata()
        .map_err(|error| io_error("stat qualification probe", &temporary, error))?;
    drop(file);
    if !rename_noreplace(&temporary, &final_path)? {
        return Err(SpoolError::AtomicRenameUnavailable);
    }
    sync_directory(root)?;
    let after = fs::metadata(&final_path)
        .map_err(|error| io_error("stat renamed qualification probe", &final_path, error))?;
    if before.dev() != after.dev() || before.ino() != after.ino() || after.nlink() != 1 {
        return Err(SpoolError::AtomicRenameUnavailable);
    }
    fs::remove_file(&final_path)
        .map_err(|error| io_error("remove qualification probe", &final_path, error))?;
    sync_directory(root)?;
    Ok(())
}

fn rename_noreplace(source: &Path, destination: &Path) -> Result<bool, SpoolError> {
    let source_c = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| SpoolError::UnsafeRelativePath(source.to_path_buf()))?;
    let destination_c = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| SpoolError::UnsafeRelativePath(destination.to_path_buf()))?;
    // SAFETY: both C strings remain valid; `AT_FDCWD` selects their absolute paths.
    let result = unsafe {
        libc::renameat2(
            libc::AT_FDCWD,
            source_c.as_ptr(),
            libc::AT_FDCWD,
            destination_c.as_ptr(),
            libc::RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.kind() == io::ErrorKind::AlreadyExists {
        return Ok(false);
    }
    if error.raw_os_error() == Some(libc::ENOSYS) || error.raw_os_error() == Some(libc::EINVAL) {
        return Err(SpoolError::AtomicRenameUnavailable);
    }
    Err(io_error("create-only rename", destination, error))
}

fn sync_directory(path: &Path) -> Result<(), SpoolError> {
    let directory =
        File::open(path).map_err(|error| io_error("open directory for fsync", path, error))?;
    directory
        .sync_all()
        .map_err(|error| io_error("fsync directory", path, error))
}

fn verify_exact_file(path: &Path, expected: &[u8]) -> Result<(), SpoolError> {
    let actual = fs::read(path).map_err(|error| io_error("read immutable final", path, error))?;
    if actual != expected {
        return Err(SpoolError::ImmutableContentMismatch(path.to_path_buf()));
    }
    Ok(())
}

fn io_error(operation: &'static str, path: &Path, source: io::Error) -> SpoolError {
    SpoolError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::fs::symlink;

    use tempfile::TempDir;

    use super::*;
    use crate::{
        CanonicalJsonValue, EpochAnchor, FrameIdentityV1, ProjectionEvidence,
        ProjectionReservationId, RequiredProjection, ReservationKind, SessionId, SourceOutcome,
        TableId, TerminalKind, WalFrameHeaderV1,
    };

    fn archive() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    fn genesis() -> GenesisV1 {
        GenesisV1 {
            archive_id: archive(),
            canonical_spool_id: Digest::from_bytes([1; 32]),
            archive_identity_digest: Digest::from_bytes([2; 32]),
            archive_target_digest: crate::manifest::archive_target_digest("file:///archive/"),
            archive_key_digest: Digest::from_bytes([3; 32]),
            writer_compatibility_id: Digest::from_bytes([4; 32]),
            runner_distribution_id: Digest::from_bytes([5; 32]),
            source_descriptors: CanonicalJsonValue::Array(vec![]),
            persistent_writer_identity: CanonicalJsonValue::object([(
                "writer".to_owned(),
                CanonicalJsonValue::String("parquet-v1".to_owned()),
            )])
            .unwrap(),
            initial_session_id: Some(session()),
            time_domain: crate::TimeDomain::Real,
            epoch_anchor: Some(EpochAnchor {
                clock_ns: 10,
                unix_epoch_ns: 1_700_000_000_000_000_000,
                capture_uncertainty_ns: 2,
            }),
        }
    }

    fn create(path: &Path) -> LocalArchiveRepository {
        LocalArchiveRepository::create_new(
            QualifiedSpool::open(path).unwrap(),
            genesis(),
            &NoDurabilityFaults,
        )
        .unwrap()
    }

    fn recover(path: &Path) -> LocalArchiveRepository {
        LocalArchiveRepository::recover(
            QualifiedSpool::open(path).unwrap(),
            RecoveryExpectation::from_genesis(&genesis()),
            &NoDurabilityFaults,
        )
        .unwrap()
    }

    fn index_entry(number: u64) -> crate::IndexEntry {
        crate::IndexEntry::new(
            crate::IndexKey::new(number.to_be_bytes().to_vec()).unwrap(),
            format!("{{\"value\":{number}}}").into_bytes(),
        )
        .unwrap()
    }

    fn wal_header(repository: &LocalArchiveRepository) -> WalSegmentHeaderV1 {
        WalSegmentHeaderV1::new(
            archive(),
            session(),
            repository.head.generation_hash,
            repository.head.genesis_hash,
            repository.genesis.writer_compatibility_id,
            repository.head.next_record_seq,
            repository.head.session_anchor.unwrap(),
            vec![(TableId::Attempts, Digest::from_bytes([9; 32]))],
        )
        .unwrap()
    }

    fn start_session(repository: &mut LocalArchiveRepository) -> WalSegmentHeaderV1 {
        let header = wal_header(repository);
        repository
            .start_session(&header, &NoDurabilityFaults)
            .unwrap();
        header
    }

    fn wal_frame() -> WalFrame {
        let batch = FrameIdentityV1::source_scrape_batch(
            archive(),
            session(),
            "source",
            1,
            SourceOutcome::Success,
            None,
        )
        .unwrap();
        let reservation: ProjectionReservationId = FrameIdentityV1::projection_reservation(
            archive(),
            session(),
            ReservationKind::SourceScrape,
            Some("source"),
            batch,
            0,
        )
        .unwrap();
        WalFrame::new(
            WalFrameHeaderV1::new(
                batch,
                reservation,
                0,
                10,
                TerminalKind::SourceScrape,
                vec![RequiredProjection {
                    table: TableId::Attempts,
                    evidence: ProjectionEvidence {
                        row_count: 1,
                        logical_multiset_digest: Digest::from_bytes([8; 32]),
                    },
                }],
                vec![],
                vec![],
                2,
            )
            .unwrap(),
            b"{}".to_vec(),
        )
        .unwrap()
    }

    #[test]
    fn qualification_rejects_symlinks_and_lock_is_crash_released() {
        let temp = TempDir::new().unwrap();
        let real = temp.path().join("real");
        fs::create_dir(&real).unwrap();
        let linked = temp.path().join("linked");
        symlink(&real, &linked).unwrap();
        assert!(matches!(
            QualifiedSpool::open(&linked),
            Err(SpoolError::SymlinkOrNonDirectory(_))
        ));

        let path = temp.path().join("spool");
        let first = QualifiedSpool::open(&path).unwrap();
        assert!(matches!(
            QualifiedSpool::open(&path),
            Err(SpoolError::LockBusy(_))
        ));
        drop(first);
        QualifiedSpool::open(&path).unwrap();
    }

    #[test]
    fn budget_observation_counts_logical_files_and_fresh_filesystem_capacity() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("spool");
        let spool = QualifiedSpool::open(&path).unwrap();
        let before = spool.budget_observation().unwrap();
        assert!(before.logical_files >= 1);
        assert!(before.filesystem_available_bytes > 0);
        assert!(before.filesystem_available_files > 0);
        fs::write(path.join("raw").join("budget-probe"), b"abc").unwrap();
        let after = spool.budget_observation().unwrap();
        assert_eq!(after.logical_files, before.logical_files + 1);
        assert_eq!(after.logical_bytes, before.logical_bytes + 3);
    }

    #[test]
    fn create_new_survives_every_file_sync_rename_and_pointer_edge() {
        let edges = [
            DurabilityEdge::IndexTempWritten,
            DurabilityEdge::IndexFileSynced,
            DurabilityEdge::IndexRenamed,
            DurabilityEdge::IndexDirectorySynced,
            DurabilityEdge::GenerationTempWritten,
            DurabilityEdge::GenerationFileSynced,
            DurabilityEdge::GenerationRenamed,
            DurabilityEdge::GenerationDirectorySynced,
            DurabilityEdge::PointerTempWritten,
            DurabilityEdge::PointerFileSynced,
            DurabilityEdge::PointerRenamed,
            DurabilityEdge::PointerDirectorySynced,
        ];
        for edge in edges {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("spool");
            let result = LocalArchiveRepository::create_new(
                QualifiedSpool::open(&path).unwrap(),
                genesis(),
                &FailAtDurabilityEdge::first(edge),
            );
            assert!(matches!(result, Err(SpoolError::FaultInjected(actual)) if actual == edge));
            let pointer_exists = path.join(LOCAL_LATEST).exists();
            if pointer_exists {
                let recovered = recover(&path);
                assert_eq!(recovered.head.local_commit_seq, 0, "edge={edge:?}");
            } else {
                let created = create(&path);
                assert_eq!(created.head.local_commit_seq, 0, "edge={edge:?}");
            }
        }
    }

    #[test]
    fn session_start_is_old_or_exact_named_wal_after_every_crash_edge() {
        let edges = [
            DurabilityEdge::GenerationTempWritten,
            DurabilityEdge::GenerationFileSynced,
            DurabilityEdge::GenerationRenamed,
            DurabilityEdge::GenerationDirectorySynced,
            DurabilityEdge::PointerTempWritten,
            DurabilityEdge::PointerFileSynced,
            DurabilityEdge::PointerRenamed,
            DurabilityEdge::PointerDirectorySynced,
        ];
        for edge in edges {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("spool");
            let mut repository = create(&path);
            let header = wal_header(&repository);
            let result = repository.start_session(&header, &FailAtDurabilityEdge::first(edge));
            assert!(matches!(result, Err(SpoolError::FaultInjected(actual)) if actual == edge));
            drop(repository);

            let mut recovered = recover(&path);
            match recovered.head.local_commit_seq {
                0 => {
                    assert_eq!(recovered.head.active_wal_segment_id, None);
                    assert_eq!(recovered.head.next_record_seq, 0);
                }
                1 => {
                    assert_eq!(
                        recovered.head.active_wal_segment_id,
                        Some(header.segment_id)
                    );
                    assert_eq!(recovered.head.next_record_seq, header.first_record_seq);
                }
                sequence => panic!("unexpected recovered session sequence {sequence}"),
            }
            recovered
                .start_session(&header, &NoDurabilityFaults)
                .unwrap();
            assert_eq!(recovered.head.local_commit_seq, 1);
            assert_eq!(
                recovered.head.active_wal_segment_id,
                Some(header.segment_id)
            );
        }
    }

    #[test]
    fn descendant_commit_is_old_or_new_after_every_transaction_edge_never_partial() {
        let edges = [
            DurabilityEdge::IndexTempWritten,
            DurabilityEdge::IndexFileSynced,
            DurabilityEdge::IndexRenamed,
            DurabilityEdge::IndexDirectorySynced,
            DurabilityEdge::GenerationTempWritten,
            DurabilityEdge::GenerationFileSynced,
            DurabilityEdge::GenerationRenamed,
            DurabilityEdge::GenerationDirectorySynced,
            DurabilityEdge::PointerTempWritten,
            DurabilityEdge::PointerFileSynced,
            DurabilityEdge::PointerRenamed,
            DurabilityEdge::PointerDirectorySynced,
        ];
        for edge in edges {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("spool");
            let mut repository = create(&path);
            start_session(&mut repository);
            let mutation = IndexMutationSetV1::new(vec![], vec![index_entry(1)]).unwrap();
            let result =
                repository.checkpoint(&mutation, session(), 0, &FailAtDurabilityEdge::first(edge));
            assert!(matches!(result, Err(SpoolError::FaultInjected(actual)) if actual == edge));
            drop(repository);
            let recovered = recover(&path);
            assert!(
                (1..=2).contains(&recovered.head.local_commit_seq),
                "edge={edge:?}"
            );
            assert_eq!(
                recovered.index.entries().count(),
                usize::try_from(recovered.head.local_commit_seq - 1).unwrap(),
                "edge={edge:?}"
            );
        }
    }

    #[test]
    fn corrupt_current_generation_rolls_back_only_to_verified_preceding() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("spool");
        let mut repository = create(&path);
        start_session(&mut repository);
        repository
            .checkpoint(
                &IndexMutationSetV1::new(vec![], vec![index_entry(1)]).unwrap(),
                session(),
                0,
                &NoDurabilityFaults,
            )
            .unwrap();
        let current_path = path.join(&repository.head.generation_key);
        drop(repository);
        let mut bytes = fs::read(&current_path).unwrap();
        bytes[0] ^= 1;
        fs::write(&current_path, bytes).unwrap();
        let recovered = recover(&path);
        assert!(recovered.rolled_back_current());
        assert_eq!(recovered.head.local_commit_seq, 1);
        assert_eq!(recovered.index.entries().count(), 0);
    }

    #[test]
    fn exact_resume_mismatch_fails_before_returning_authority() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("spool");
        drop(create(&path));
        let mut expectation = RecoveryExpectation::from_genesis(&genesis());
        expectation.archive_key_digest = Digest::from_bytes([0xff; 32]);
        assert!(matches!(
            LocalArchiveRepository::recover(
                QualifiedSpool::open(&path).unwrap(),
                expectation,
                &NoDurabilityFaults,
            ),
            Err(SpoolError::IdentityMismatch("archive key digest"))
        ));
    }

    #[test]
    fn source_free_recovery_pins_archive_and_normalized_target_identity() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("spool");
        drop(create(&path));
        let recovered = LocalArchiveRepository::recover_existing(
            QualifiedSpool::open(&path).unwrap(),
            archive(),
            crate::manifest::archive_target_digest("file:///archive/"),
            &NoDurabilityFaults,
        )
        .unwrap();
        assert_eq!(recovered.genesis(), &genesis());
        drop(recovered);

        let other_archive = ArchiveId::new([0x99; 16]).unwrap();
        assert!(matches!(
            LocalArchiveRepository::recover_existing(
                QualifiedSpool::open(&path).unwrap(),
                other_archive,
                crate::manifest::archive_target_digest("file:///archive/"),
                &NoDurabilityFaults,
            ),
            Err(SpoolError::IdentityMismatch("archive ID"))
        ));
        assert!(matches!(
            LocalArchiveRepository::recover_existing(
                QualifiedSpool::open(&path).unwrap(),
                archive(),
                crate::manifest::archive_target_digest("s3://other/archive/"),
                &NoDurabilityFaults,
            ),
            Err(SpoolError::IdentityMismatch("archive target digest"))
        ));
    }

    #[test]
    fn wal_append_uncertainty_recovers_complete_frame_once() {
        for edge in [
            DurabilityEdge::WalFrameWritten,
            DurabilityEdge::WalFrameSynced,
        ] {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("spool");
            let mut repository = create(&path);
            let header = start_session(&mut repository);
            let injector = FailAtDurabilityEdge::first(edge);
            let mut writer = repository.create_wal(header.clone(), &injector).unwrap();
            assert!(matches!(
                writer.append(&wal_frame()),
                Err(SpoolError::FaultInjected(actual)) if actual == edge
            ));
            drop(writer);
            let recovered = repository
                .recover_wal(
                    &header,
                    crate::wal::DEFAULT_MAX_WAL_FRAME_BYTES,
                    &NoDurabilityFaults,
                )
                .unwrap();
            assert_eq!(recovered.frames.len(), 1, "edge={edge:?}");
            assert_eq!(recovered.frames[0].header().record_seq, 0);
        }
    }

    #[test]
    fn interrupted_wal_seal_is_completed_or_remains_verified_open() {
        for edge in [
            DurabilityEdge::WalFooterWritten,
            DurabilityEdge::WalSealSynced,
            DurabilityEdge::WalRenamed,
            DurabilityEdge::WalSealDirectorySynced,
        ] {
            let temp = TempDir::new().unwrap();
            let path = temp.path().join("spool");
            let mut repository = create(&path);
            let header = start_session(&mut repository);
            let injector = FailAtDurabilityEdge::first(edge);
            let mut writer = repository.create_wal(header.clone(), &injector).unwrap();
            writer.append(&wal_frame()).unwrap();
            assert!(matches!(
                writer.seal(),
                Err(SpoolError::FaultInjected(actual)) if actual == edge
            ));
            let recovered = repository
                .recover_wal(
                    &header,
                    crate::wal::DEFAULT_MAX_WAL_FRAME_BYTES,
                    &NoDurabilityFaults,
                )
                .unwrap();
            assert_eq!(recovered.frames.len(), 1, "edge={edge:?}");
            assert!(recovered.segment_digest.is_some(), "edge={edge:?}");
        }
    }
}
