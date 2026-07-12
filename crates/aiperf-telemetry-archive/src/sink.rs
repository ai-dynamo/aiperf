// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Archive sink seam plus deterministic memory and local WAL/Parquet sinks.
//!
//! The append boundary accepts a fully terminal WAL frame and every required
//! physical table projection together. Validation is complete before the
//! memory sink mutates. The local implementation fail-stops if an fsync result
//! is uncertain, leaving ordinary spool recovery to determine the durable
//! prefix under the same frame identity.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Debug, Display, Formatter};
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::spool::ImmutableClass;
use crate::spool::OwnedLocalWalWriter;
use crate::{
    ArchiveId, ArchiveSchemasV1, ArchiveState, CompositeIndexKeyV1, ControlFrameCodecError,
    ControlFrameCodecV1, Digest, DurabilityFaultInjector, FrameId, FrameTableProjectionV1,
    IndexMutationSetV1, LocalArchiveRepository, LocalWalWriter, ParquetPartitionBuilderV1,
    ParquetProjectionError, ParquetRotationConfigV1, PartitionBuildOutputV1, ProjectionCoverageV1,
    QualifiedSpool, ReceiptError, ReceiptEventId, ReceiptEventV1, ReceiptJournal,
    ReceiptObserverEpochId, ReceiptObserverEpochV1, ReceiptTargetId, ReceiptTargetV1,
    SealedWalSegment, SessionId, SourceFrameCodecError, SourceFrameCodecV1, SpoolError,
    TerminalKind, WalError, WalFrame, WalSegmentBuilder, WalSegmentHeaderV1, domain_digest,
    receipt_range_coverage,
};

/// One finalized WAL frame and all required table projections as one append.
#[derive(Clone, Debug)]
pub struct ArchiveWalFrame {
    /// Exact canonical WAL frame.
    pub wal_frame: WalFrame,
    /// Exactly one projection for every table declared in the WAL header.
    pub table_projections: Vec<FrameTableProjectionV1>,
}

/// Rebuilds complete physical projections from one verified WAL payload.
pub trait ArchiveWalFrameDecoder: Debug + Send + Sync {
    /// Decodes without changing frame identity, ordering, or declared evidence.
    fn decode(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
        frame: WalFrame,
    ) -> Result<ArchiveWalFrame, ArchiveSinkError>;
}

impl ArchiveWalFrameDecoder for SourceFrameCodecV1 {
    fn decode(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
        frame: WalFrame,
    ) -> Result<ArchiveWalFrame, ArchiveSinkError> {
        self.decode_source_frame(archive_id, session_id, frame)
            .map_err(|error| ArchiveSinkError::FrameDecode(Box::new(error)))
    }
}

impl ArchiveWalFrameDecoder for ControlFrameCodecV1 {
    fn decode(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
        frame: WalFrame,
    ) -> Result<ArchiveWalFrame, ArchiveSinkError> {
        self.decode_control_frame(archive_id, session_id, frame)
            .map_err(|error| ArchiveSinkError::ControlFrameDecode(Box::new(error)))
    }
}

/// Closed v1 recovery decoder dispatching every durable terminal frame kind.
#[derive(Clone, Debug)]
pub struct CanonicalArchiveWalFrameDecoderV1 {
    source: SourceFrameCodecV1,
    control: ControlFrameCodecV1,
}

impl CanonicalArchiveWalFrameDecoderV1 {
    /// Shares one loaded schema authority across source and control codecs.
    #[must_use]
    pub fn with_schemas(schemas: ArchiveSchemasV1) -> Self {
        Self {
            source: SourceFrameCodecV1::with_schemas(schemas.clone()),
            control: ControlFrameCodecV1::with_schemas(schemas),
        }
    }
}

impl ArchiveWalFrameDecoder for CanonicalArchiveWalFrameDecoderV1 {
    fn decode(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
        frame: WalFrame,
    ) -> Result<ArchiveWalFrame, ArchiveSinkError> {
        match frame.header().terminal_kind {
            TerminalKind::SourceScrape => self.source.decode(archive_id, session_id, frame),
            TerminalKind::LifecycleMarker
            | TerminalKind::LossExact
            | TerminalKind::LossSaturation
            | TerminalKind::SourceProjectionFailed => {
                self.control.decode(archive_id, session_id, frame)
            }
        }
    }
}

/// Local durability authority returned only after a complete frame is stable.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DurabilityCompletion {
    /// WAL segment containing the frame.
    pub wal_segment_id: Digest,
    /// Exact durable prefix after this frame.
    pub durable_prefix_hash: Digest,
    /// Terminal frame identity.
    pub frame_id: FrameId,
    /// First record in this single-append completion.
    pub first_record_seq: u64,
    /// Last record in this single-append completion.
    pub last_record_seq: u64,
    /// Digest of ascending declared table projection evidence.
    pub projection_coverage_digest: Digest,
}

/// Optional new observer epoch plus one immutable target/event observation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReceiptEventDraft {
    /// Epoch to persist before the event, or `None` when already durable.
    pub observer_epoch: Option<ReceiptObserverEpochV1>,
    /// Earlier immutable WAL/publication target.
    pub target: ReceiptTargetV1,
    /// Later Clock observation bound to the target and epoch.
    pub event: ReceiptEventV1,
}

/// Durable receipt-journal acknowledgment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AppendReceipt {
    /// Immutable target identity.
    pub receipt_target_id: ReceiptTargetId,
    /// Immutable observation event identity.
    pub event_id: ReceiptEventId,
    /// Journal-global receipt sequence.
    pub receipt_seq: u64,
}

/// Verified state returned by recovery before append resumes.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RecoveredArchive {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Current archive lifecycle state.
    pub archive_state: ArchiveState,
    /// Every complete verified frame in owner order.
    pub frames: Vec<WalFrame>,
    /// Calls whose response was uncertain even though recovery found the frame.
    pub uncertain_frame_ids: Vec<FrameId>,
    /// Current verified WAL prefix.
    pub durable_prefix_hash: Digest,
    /// First sequence not present in the recovered WAL.
    pub next_record_seq: u64,
}

/// Locally completed physical checkpoint artifacts.
#[derive(Clone, Debug)]
pub struct CheckpointCompletion {
    /// Monotone sink-local checkpoint sequence.
    pub checkpoint_seq: u64,
    /// Newly completed partitions and explicit zero-row coverage.
    pub physical: PartitionBuildOutputV1,
    /// WAL prefix covered by the checkpoint operation.
    pub durable_prefix_hash: Digest,
    /// Last durable record, absent before the first frame.
    pub last_record_seq: Option<u64>,
    /// Installed local manifest head when this sink owns the repository transaction.
    pub local_head: Option<crate::HeadDescriptorV1>,
}

/// Closed finalization reasons serialized by the owning runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminationReason {
    /// Explicit user request.
    Requested,
    /// Graceful process signal.
    Signal,
    /// Configured duration/stop bound.
    Duration,
    /// Fail-stop after a durable error/loss frame.
    Failure,
    /// Source-free synchronization/finalization invocation.
    SyncOnly,
}

/// Final local checkpoint plus sealed WAL authority.
#[derive(Clone, Debug)]
pub struct FinalizeCompletion {
    /// Final physical checkpoint.
    pub checkpoint: CheckpointCompletion,
    /// Complete sealed WAL; a conforming collection has at least one frame.
    pub sealed_wal: SealedWalSegment,
    /// Terminal reason selected by the owner.
    pub reason: TerminationReason,
    /// Resulting local state.
    pub archive_state: ArchiveState,
    /// Installed terminal local head when this sink owns the repository.
    pub local_head: Option<crate::HeadDescriptorV1>,
}

/// Narrow archive durability/physical-projection extension point.
#[async_trait]
pub trait ArchiveSink: Debug + Send {
    /// Reconciles any uncertain append before accepting more frames.
    async fn recover(&mut self) -> Result<RecoveredArchive, ArchiveSinkError>;

    /// Appends one whole terminal frame through the local-durable boundary.
    async fn append_frame(
        &mut self,
        frame: ArchiveWalFrame,
    ) -> Result<DurabilityCompletion, ArchiveSinkError>;

    /// Persists an observation of an earlier immutable completion target.
    async fn record_receipt(
        &mut self,
        event: ReceiptEventDraft,
    ) -> Result<AppendReceipt, ArchiveSinkError>;

    /// Rotates open physical builders and returns newly completed artifacts.
    async fn checkpoint(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError>;

    /// Closes frame admission, checkpoints, and seals the final WAL segment.
    async fn finalize(
        &mut self,
        reason: TerminationReason,
    ) -> Result<FinalizeCompletion, ArchiveSinkError>;

    /// Borrows the live qualified repository when the sink owns local authority.
    ///
    /// Remote-claim preparation uses this after `session_started` and before
    /// source activation without releasing the lifetime spool lock.
    fn local_repository(&self) -> Option<&LocalArchiveRepository> {
        None
    }

    /// Consumes a finalized sink and transfers its still-held local repository.
    ///
    /// Non-local and borrowed adapters return `None`. The consuming hook keeps
    /// the qualified spool lock continuous across collection, finalization,
    /// and remote publication without cloning a detached authority snapshot.
    fn into_local_repository(
        self: Box<Self>,
    ) -> Result<Option<LocalArchiveRepository>, ArchiveSinkError> {
        Ok(None)
    }
}

/// One deterministic failure injected after a memory mutation is authoritative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryArchiveSinkFault {
    /// No fault.
    None,
    /// The next append applies completely but returns an uncertain result.
    AppendUncertainAfterApply,
}

/// Deterministic IO-free sink used by virtual tests and replay.
#[derive(Debug)]
pub struct MemoryArchiveSink {
    schemas: ArchiveSchemasV1,
    rotation: ParquetRotationConfigV1,
    parquet: Option<ParquetPartitionBuilderV1>,
    pending_physical: PartitionBuildOutputV1,
    wal: Option<WalSegmentBuilder>,
    archive_id: ArchiveId,
    session_id: SessionId,
    state: ArchiveState,
    frames: Vec<WalFrame>,
    completions: Vec<DurabilityCompletion>,
    uncertain_frames: BTreeSet<FrameId>,
    recovery_required: bool,
    checkpoint_seq: u64,
    epochs: BTreeMap<ReceiptObserverEpochId, ReceiptObserverEpochV1>,
    targets: BTreeMap<ReceiptTargetId, ReceiptTargetV1>,
    events: Vec<ReceiptEventV1>,
    fault: MemoryArchiveSinkFault,
}

impl MemoryArchiveSink {
    /// Creates an empty sink bound to one canonical WAL segment.
    pub fn new(
        header: WalSegmentHeaderV1,
        schemas: ArchiveSchemasV1,
        rotation: ParquetRotationConfigV1,
    ) -> Result<Self, ArchiveSinkError> {
        let archive_id = header.archive_id;
        let session_id = header.session_id;
        Ok(Self {
            parquet: Some(ParquetPartitionBuilderV1::new(schemas.clone(), rotation)?),
            wal: Some(WalSegmentBuilder::new(header)?),
            schemas,
            rotation,
            pending_physical: PartitionBuildOutputV1::default(),
            archive_id,
            session_id,
            state: ArchiveState::Open,
            frames: Vec::new(),
            completions: Vec::new(),
            uncertain_frames: BTreeSet::new(),
            recovery_required: false,
            checkpoint_seq: 0,
            epochs: BTreeMap::new(),
            targets: BTreeMap::new(),
            events: Vec::new(),
            fault: MemoryArchiveSinkFault::None,
        })
    }

    /// Injects one deterministic next-operation fault.
    pub fn set_fault(&mut self, fault: MemoryArchiveSinkFault) {
        self.fault = fault;
    }

    /// Returns durable receipt events in sequence order.
    #[must_use]
    pub fn receipt_events(&self) -> &[ReceiptEventV1] {
        &self.events
    }

    fn checkpoint_sync(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError> {
        ensure_operable(self.state, self.recovery_required)?;
        let next_checkpoint_seq = self
            .checkpoint_seq
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)?;
        let output = finish_and_reset_parquet(
            &mut self.parquet,
            &self.schemas,
            self.rotation,
            &mut self.pending_physical,
        )?;
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        let completion = CheckpointCompletion {
            checkpoint_seq: self.checkpoint_seq,
            physical: output,
            durable_prefix_hash: wal.prefix(),
            last_record_seq: wal.last_record_seq(),
            local_head: None,
        };
        self.checkpoint_seq = next_checkpoint_seq;
        Ok(completion)
    }
}

#[async_trait]
impl ArchiveSink for MemoryArchiveSink {
    async fn recover(&mut self) -> Result<RecoveredArchive, ArchiveSinkError> {
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        self.recovery_required = false;
        Ok(recovered_archive(
            self.archive_id,
            self.session_id,
            self.state,
            &self.frames,
            &self.uncertain_frames,
            wal.prefix(),
            wal.header().first_record_seq,
        )?)
    }

    async fn append_frame(
        &mut self,
        frame: ArchiveWalFrame,
    ) -> Result<DurabilityCompletion, ArchiveSinkError> {
        ensure_operable(self.state, self.recovery_required)?;
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        validate_archive_frame(&frame, wal.header(), &self.schemas)?;
        let mut next_wal = wal.clone();
        next_wal.append(&frame.wal_frame)?;
        let output = self
            .parquet
            .as_mut()
            .ok_or(ArchiveSinkError::Finalized)?
            .append_frame(frame.table_projections)?;
        merge_physical(&mut self.pending_physical, output);
        let completion = durability_completion(&next_wal, &frame.wal_frame);
        self.frames.push(frame.wal_frame);
        self.completions.push(completion.clone());
        self.wal = Some(next_wal);
        if self.fault == MemoryArchiveSinkFault::AppendUncertainAfterApply {
            self.fault = MemoryArchiveSinkFault::None;
            self.uncertain_frames.insert(completion.frame_id);
            self.recovery_required = true;
            return Err(ArchiveSinkError::Uncertain(
                "memory append applied before injected response loss".to_owned(),
            ));
        }
        Ok(completion)
    }

    async fn record_receipt(
        &mut self,
        draft: ReceiptEventDraft,
    ) -> Result<AppendReceipt, ArchiveSinkError> {
        ensure_operable(self.state, self.recovery_required)?;
        validate_receipt_draft(
            self.archive_id,
            &draft,
            next_receipt_sequence(self.events.last().map(|event| event.receipt_seq))?,
        )?;
        validate_receipt_target(self.session_id, &draft.target, &self.completions)?;
        if let Some(epoch) = &draft.observer_epoch {
            match self.epochs.get(&epoch.observer_epoch_id) {
                Some(existing) if existing == epoch => {}
                Some(_) => return Err(ArchiveSinkError::ReceiptEpochCollision),
                None => {
                    self.epochs.insert(epoch.observer_epoch_id, epoch.clone());
                }
            }
        }
        if !self.epochs.contains_key(&draft.event.observer_epoch_id) {
            return Err(ArchiveSinkError::MissingObserverEpoch);
        }
        match self.targets.get(&draft.target.receipt_target_id) {
            Some(existing) if existing == &draft.target => {}
            Some(_) => return Err(ArchiveSinkError::ReceiptTargetCollision),
            None => {
                self.targets
                    .insert(draft.target.receipt_target_id, draft.target.clone());
            }
        }
        let receipt = AppendReceipt {
            receipt_target_id: draft.target.receipt_target_id,
            event_id: draft.event.event_id,
            receipt_seq: draft.event.receipt_seq,
        };
        self.events.push(draft.event);
        Ok(receipt)
    }

    async fn checkpoint(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError> {
        self.checkpoint_sync()
    }

    async fn finalize(
        &mut self,
        reason: TerminationReason,
    ) -> Result<FinalizeCompletion, ArchiveSinkError> {
        ensure_operable(self.state, self.recovery_required)?;
        if self.frames.is_empty() {
            return Err(ArchiveSinkError::EmptyFinalization);
        }
        let checkpoint = self.checkpoint_sync()?;
        let sealed_wal = self.wal.take().ok_or(ArchiveSinkError::Finalized)?.seal()?;
        self.state = ArchiveState::LocallyFinalized;
        Ok(FinalizeCompletion {
            checkpoint,
            sealed_wal,
            reason,
            archive_state: self.state,
            local_head: None,
        })
    }
}

/// File-backed WAL plus deterministic Parquet object persistence.
///
/// Manifest/index installation remains an explicit outer transaction after a
/// checkpoint returns its descriptors. Consuming `finalize` releases the WAL
/// writer's spool borrow so the caller can commit those descriptors through
/// [`crate::LocalArchiveRepository`].
pub struct LocalParquetArchiveSink<'a> {
    spool: &'a QualifiedSpool,
    faults: &'a dyn DurabilityFaultInjector,
    schemas: ArchiveSchemasV1,
    rotation: ParquetRotationConfigV1,
    parquet: Option<ParquetPartitionBuilderV1>,
    pending_physical: PartitionBuildOutputV1,
    wal: Option<LocalWalWriter<'a>>,
    receipt_journal: Option<ReceiptJournal<'a>>,
    archive_id: ArchiveId,
    session_id: SessionId,
    state: ArchiveState,
    frames: Vec<WalFrame>,
    completions: Vec<DurabilityCompletion>,
    checkpoint_seq: u64,
    poisoned: bool,
}

impl Debug for LocalParquetArchiveSink<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LocalParquetArchiveSink")
            .field("archive_id", &self.archive_id)
            .field("session_id", &self.session_id)
            .field("state", &self.state)
            .field("frames", &self.frames.len())
            .field("checkpoint_seq", &self.checkpoint_seq)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl<'a> LocalParquetArchiveSink<'a> {
    /// Composes an already-created fsynced WAL and optional durable receipt journal.
    pub fn new(
        spool: &'a QualifiedSpool,
        wal: LocalWalWriter<'a>,
        receipt_journal: Option<ReceiptJournal<'a>>,
        schemas: ArchiveSchemasV1,
        rotation: ParquetRotationConfigV1,
        faults: &'a dyn DurabilityFaultInjector,
    ) -> Result<Self, ArchiveSinkError> {
        let archive_id = wal.header().archive_id;
        let session_id = wal.header().session_id;
        Ok(Self {
            spool,
            faults,
            parquet: Some(ParquetPartitionBuilderV1::new(schemas.clone(), rotation)?),
            schemas,
            rotation,
            pending_physical: PartitionBuildOutputV1::default(),
            wal: Some(wal),
            receipt_journal,
            archive_id,
            session_id,
            state: ArchiveState::Open,
            frames: Vec::new(),
            completions: Vec::new(),
            checkpoint_seq: 0,
            poisoned: false,
        })
    }

    fn ensure_operable(&self) -> Result<(), ArchiveSinkError> {
        if self.poisoned {
            return Err(ArchiveSinkError::Poisoned);
        }
        ensure_operable(self.state, false)
    }

    fn checkpoint_sync(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError> {
        self.ensure_operable()?;
        let next_checkpoint_seq = self
            .checkpoint_seq
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)?;
        let output = finish_and_reset_parquet(
            &mut self.parquet,
            &self.schemas,
            self.rotation,
            &mut self.pending_physical,
        )?;
        if let Err(error) = persist_partitions(self.spool, &output, self.faults) {
            self.pending_physical = output;
            self.poisoned = true;
            return Err(error);
        }
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        let completion = CheckpointCompletion {
            checkpoint_seq: self.checkpoint_seq,
            physical: output,
            durable_prefix_hash: wal.durable_prefix(),
            last_record_seq: wal.last_record_seq(),
            local_head: None,
        };
        self.checkpoint_seq = next_checkpoint_seq;
        Ok(completion)
    }
}

#[async_trait]
impl ArchiveSink for LocalParquetArchiveSink<'_> {
    async fn recover(&mut self) -> Result<RecoveredArchive, ArchiveSinkError> {
        self.ensure_operable()?;
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        Ok(recovered_archive(
            self.archive_id,
            self.session_id,
            self.state,
            &self.frames,
            &BTreeSet::new(),
            wal.durable_prefix(),
            wal.header().first_record_seq,
        )?)
    }

    async fn append_frame(
        &mut self,
        frame: ArchiveWalFrame,
    ) -> Result<DurabilityCompletion, ArchiveSinkError> {
        self.ensure_operable()?;
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        validate_archive_frame(&frame, wal.header(), &self.schemas)?;
        let output = self
            .parquet
            .as_mut()
            .ok_or(ArchiveSinkError::Finalized)?
            .append_frame(frame.table_projections)?;
        merge_physical(&mut self.pending_physical, output);
        let wal = self.wal.as_mut().ok_or(ArchiveSinkError::Finalized)?;
        if let Err(error) = wal.append(&frame.wal_frame) {
            self.poisoned = true;
            return Err(ArchiveSinkError::Spool(error));
        }
        let completion = local_durability_completion(wal, &frame.wal_frame);
        self.frames.push(frame.wal_frame);
        self.completions.push(completion.clone());
        Ok(completion)
    }

    async fn record_receipt(
        &mut self,
        draft: ReceiptEventDraft,
    ) -> Result<AppendReceipt, ArchiveSinkError> {
        self.ensure_operable()?;
        validate_receipt_draft(
            self.archive_id,
            &draft,
            next_receipt_sequence(
                self.receipt_journal
                    .as_ref()
                    .and_then(ReceiptJournal::last_receipt_seq),
            )?,
        )?;
        validate_receipt_target(self.session_id, &draft.target, &self.completions)?;
        let journal = self
            .receipt_journal
            .as_mut()
            .ok_or(ArchiveSinkError::ReceiptJournalUnavailable)?;
        if let Some(epoch) = draft.observer_epoch {
            journal.append_observer_epoch(epoch, self.faults)?;
        }
        let receipt = AppendReceipt {
            receipt_target_id: draft.target.receipt_target_id,
            event_id: draft.event.event_id,
            receipt_seq: draft.event.receipt_seq,
        };
        journal.record_event(draft.target, draft.event, self.faults)?;
        Ok(receipt)
    }

    async fn checkpoint(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError> {
        self.checkpoint_sync()
    }

    async fn finalize(
        &mut self,
        reason: TerminationReason,
    ) -> Result<FinalizeCompletion, ArchiveSinkError> {
        self.ensure_operable()?;
        if self.frames.is_empty() {
            return Err(ArchiveSinkError::EmptyFinalization);
        }
        let checkpoint = self.checkpoint_sync()?;
        let wal = self.wal.take().ok_or(ArchiveSinkError::Finalized)?;
        let sealed_wal = match wal.seal() {
            Ok(sealed) => sealed,
            Err(error) => {
                self.poisoned = true;
                return Err(ArchiveSinkError::Spool(error));
            }
        };
        self.state = ArchiveState::LocallyFinalized;
        Ok(FinalizeCompletion {
            checkpoint,
            sealed_wal,
            reason,
            archive_state: self.state,
            local_head: None,
        })
    }
}

/// Receipt-journal preparation selected by an owned local sink invocation.
#[derive(Clone, Debug)]
pub enum OwnedReceiptJournalMode {
    /// Do not prepare receipt recording for this sink.
    Disabled,
    /// Create the mandatory epoch-only receipt transaction.
    Bootstrap(ReceiptObserverEpochV1),
    /// Recover the journal and optionally register this execution's epoch.
    Recover {
        /// New execution epoch to persist before observing completions.
        observer_epoch: Option<ReceiptObserverEpochV1>,
    },
}

/// Prepared factory for a fully owned repository/WAL/Parquet sink.
#[derive(Clone)]
pub struct OwnedLocalArchiveSinkFactory {
    schemas: ArchiveSchemasV1,
    rotation: ParquetRotationConfigV1,
    faults: Arc<dyn DurabilityFaultInjector>,
}

impl Debug for OwnedLocalArchiveSinkFactory {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedLocalArchiveSinkFactory")
            .field("rotation", &self.rotation)
            .field("faults", &self.faults)
            .finish_non_exhaustive()
    }
}

impl OwnedLocalArchiveSinkFactory {
    /// Validates physical policy and captures the owned durability dependencies.
    pub fn new(
        schemas: ArchiveSchemasV1,
        rotation: ParquetRotationConfigV1,
        faults: Arc<dyn DurabilityFaultInjector>,
    ) -> Result<Self, ArchiveSinkError> {
        // Construction proves the rotation policy against the actual schemas.
        ParquetPartitionBuilderV1::new(schemas.clone(), rotation)?;
        Ok(Self {
            schemas,
            rotation,
            faults,
        })
    }

    /// Consumes a repository and creates one WAL-backed owned sink.
    pub fn prepare(
        &self,
        mut repository: LocalArchiveRepository,
        header: WalSegmentHeaderV1,
        receipt_mode: OwnedReceiptJournalMode,
    ) -> Result<OwnedLocalArchiveSink, ArchiveSinkError> {
        if repository.head().archive_state != ArchiveState::Open {
            return Err(ArchiveSinkError::Finalized);
        }
        let archive_id = header.archive_id;
        let session_id = header.session_id;
        repository.start_session(&header, self.faults.as_ref())?;
        let receipts_enabled = prepare_receipt_journal(
            repository.spool(),
            archive_id,
            receipt_mode,
            self.faults.as_ref(),
        )?;
        let wal = repository.create_owned_wal(header, Arc::clone(&self.faults))?;
        OwnedLocalArchiveSink::new(
            repository,
            wal,
            self.schemas.clone(),
            self.rotation,
            Arc::clone(&self.faults),
            archive_id,
            session_id,
            receipts_enabled,
        )
    }

    /// Recovers and closes the manifest-named prior WAL, then opens a fresh session.
    ///
    /// The prior segment is never selected by directory enumeration. Recovery
    /// replays missing projection coverage, checkpoints the resulting global
    /// sequence, seals that exact segment, and commits `session_closed` before
    /// a fresh `session_started` transaction can authorize source activation.
    pub fn resume(
        &self,
        mut repository: LocalArchiveRepository,
        new_session_id: SessionId,
        maximum_frame_bytes: u64,
        receipt_mode: OwnedReceiptJournalMode,
        decoder: &dyn ArchiveWalFrameDecoder,
    ) -> Result<OwnedLocalArchiveSink, ArchiveSinkError> {
        if repository.head().archive_state != ArchiveState::Open {
            return Err(ArchiveSinkError::Finalized);
        }
        let archive_id = repository.head().archive_id;
        let receipts_enabled = prepare_receipt_journal(
            repository.spool(),
            archive_id,
            receipt_mode,
            self.faults.as_ref(),
        )?;
        let schema_fingerprints = self
            .schemas
            .iter()
            .map(|schema| (schema.table(), schema.fingerprint()))
            .collect::<Vec<_>>();
        if let Some(active_header) = repository.active_wal_header(schema_fingerprints.clone())? {
            let unactivated_retry = active_header.session_id == new_session_id
                && repository.head().parent_generation_hash
                    == Some(active_header.previous_head_hash)
                && repository.head().next_record_seq == active_header.first_record_seq;
            if unactivated_retry {
                let wal = repository.resume_or_create_empty_owned_wal(
                    &active_header,
                    maximum_frame_bytes,
                    Arc::clone(&self.faults),
                )?;
                return OwnedLocalArchiveSink::new(
                    repository,
                    wal,
                    self.schemas.clone(),
                    self.rotation,
                    Arc::clone(&self.faults),
                    archive_id,
                    new_session_id,
                    receipts_enabled,
                );
            }
            repository = self.close_recovered_session(
                repository,
                active_header,
                maximum_frame_bytes,
                decoder,
            )?;
        }
        let header = WalSegmentHeaderV1::new(
            archive_id,
            new_session_id,
            repository.head().generation_hash,
            repository.head().genesis_hash,
            repository.genesis().writer_compatibility_id,
            repository.head().next_record_seq,
            schema_fingerprints,
        )?;
        repository.start_session(&header, self.faults.as_ref())?;
        let wal = repository.create_owned_wal(header, Arc::clone(&self.faults))?;
        OwnedLocalArchiveSink::new(
            repository,
            wal,
            self.schemas.clone(),
            self.rotation,
            Arc::clone(&self.faults),
            archive_id,
            new_session_id,
            receipts_enabled,
        )
    }

    fn close_recovered_session(
        &self,
        mut repository: LocalArchiveRepository,
        header: WalSegmentHeaderV1,
        maximum_frame_bytes: u64,
        decoder: &dyn ArchiveWalFrameDecoder,
    ) -> Result<LocalArchiveRepository, ArchiveSinkError> {
        let recovered =
            repository.recover_wal(&header, maximum_frame_bytes, self.faults.as_ref())?;
        if recovered.header != header {
            return Err(ArchiveSinkError::RecoveredWalMismatch);
        }
        let sealed_wal = if recovered.segment_digest.is_some() {
            checkpoint_recovered_frames(
                &mut repository,
                &recovered.frames,
                &header,
                &self.schemas,
                self.rotation,
                self.faults.as_ref(),
                decoder,
            )?;
            rebuilt_sealed_wal(&recovered)?
        } else {
            let (wal, recovered) = repository.resume_owned_wal(
                &header,
                maximum_frame_bytes,
                Arc::clone(&self.faults),
            )?;
            checkpoint_recovered_frames(
                &mut repository,
                &recovered.frames,
                &header,
                &self.schemas,
                self.rotation,
                self.faults.as_ref(),
                decoder,
            )?;
            wal.seal()?
        };
        let next_record_seq = sealed_wal
            .last_record_seq()
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)?;
        repository.close_session(
            header.session_id,
            next_record_seq,
            &sealed_wal,
            self.faults.as_ref(),
        )?;
        Ok(repository)
    }
}

/// Product-usable local sink owning repository, spool lock, WAL, and builders.
pub struct OwnedLocalArchiveSink {
    repository: LocalArchiveRepository,
    wal: Option<OwnedLocalWalWriter>,
    faults: Arc<dyn DurabilityFaultInjector>,
    schemas: ArchiveSchemasV1,
    rotation: ParquetRotationConfigV1,
    parquet: Option<ParquetPartitionBuilderV1>,
    pending_physical: PartitionBuildOutputV1,
    archive_id: ArchiveId,
    session_id: SessionId,
    state: ArchiveState,
    frames: Vec<WalFrame>,
    completions: Vec<DurabilityCompletion>,
    checkpoint_seq: u64,
    receipts_enabled: bool,
    poisoned: bool,
}

impl Debug for OwnedLocalArchiveSink {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OwnedLocalArchiveSink")
            .field("archive_id", &self.archive_id)
            .field("session_id", &self.session_id)
            .field("state", &self.state)
            .field("frames", &self.frames.len())
            .field("checkpoint_seq", &self.checkpoint_seq)
            .field("receipts_enabled", &self.receipts_enabled)
            .field("poisoned", &self.poisoned)
            .finish_non_exhaustive()
    }
}

impl OwnedLocalArchiveSink {
    #[allow(clippy::too_many_arguments)]
    fn new(
        repository: LocalArchiveRepository,
        wal: OwnedLocalWalWriter,
        schemas: ArchiveSchemasV1,
        rotation: ParquetRotationConfigV1,
        faults: Arc<dyn DurabilityFaultInjector>,
        archive_id: ArchiveId,
        session_id: SessionId,
        receipts_enabled: bool,
    ) -> Result<Self, ArchiveSinkError> {
        Ok(Self {
            repository,
            wal: Some(wal),
            faults,
            parquet: Some(ParquetPartitionBuilderV1::new(schemas.clone(), rotation)?),
            schemas,
            rotation,
            pending_physical: PartitionBuildOutputV1::default(),
            archive_id,
            session_id,
            state: ArchiveState::Open,
            frames: Vec::new(),
            completions: Vec::new(),
            checkpoint_seq: 0,
            receipts_enabled,
            poisoned: false,
        })
    }

    /// Returns the current authoritative local repository/head.
    #[must_use]
    pub const fn repository(&self) -> &LocalArchiveRepository {
        &self.repository
    }

    /// Consumes a finalized sink and returns its repository for publication.
    pub fn into_repository(self) -> Result<LocalArchiveRepository, ArchiveSinkError> {
        if self.state != ArchiveState::LocallyFinalized || self.wal.is_some() || self.poisoned {
            return Err(ArchiveSinkError::NotLocallyFinalized);
        }
        Ok(self.repository)
    }

    fn ensure_operable(&self) -> Result<(), ArchiveSinkError> {
        if self.poisoned {
            return Err(ArchiveSinkError::Poisoned);
        }
        ensure_operable(self.state, false)
    }

    fn checkpoint_sync(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError> {
        self.ensure_operable()?;
        let next_checkpoint_seq = self
            .checkpoint_seq
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)?;
        let output = finish_and_reset_parquet(
            &mut self.parquet,
            &self.schemas,
            self.rotation,
            &mut self.pending_physical,
        )?;
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        let next_record_seq =
            next_record_seq(wal.header().first_record_seq, wal.last_record_seq())?;
        let transaction = (|| {
            persist_partitions(self.repository.spool(), &output, self.faults.as_ref())?;
            let additions = physical_index_entries(&output)?;
            let head = if additions.is_empty()
                && self.repository.head().next_record_seq == next_record_seq
            {
                self.repository.head().clone()
            } else {
                let mutation = IndexMutationSetV1::new(Vec::new(), additions)?;
                self.repository
                    .checkpoint(
                        &mutation,
                        self.session_id,
                        next_record_seq,
                        self.faults.as_ref(),
                    )?
                    .clone()
            };
            Ok::<_, ArchiveSinkError>(head)
        })();
        let local_head = match transaction {
            Ok(head) => head,
            Err(error) => {
                self.pending_physical = output;
                self.poisoned = true;
                return Err(error);
            }
        };
        let completion = CheckpointCompletion {
            checkpoint_seq: self.checkpoint_seq,
            physical: output,
            durable_prefix_hash: wal.durable_prefix(),
            last_record_seq: wal.last_record_seq(),
            local_head: Some(local_head),
        };
        self.checkpoint_seq = next_checkpoint_seq;
        Ok(completion)
    }
}

#[async_trait]
impl ArchiveSink for OwnedLocalArchiveSink {
    async fn recover(&mut self) -> Result<RecoveredArchive, ArchiveSinkError> {
        self.ensure_operable()?;
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        recovered_archive(
            self.archive_id,
            self.session_id,
            self.state,
            &self.frames,
            &BTreeSet::new(),
            wal.durable_prefix(),
            wal.header().first_record_seq,
        )
    }

    async fn append_frame(
        &mut self,
        frame: ArchiveWalFrame,
    ) -> Result<DurabilityCompletion, ArchiveSinkError> {
        self.ensure_operable()?;
        let wal = self.wal.as_ref().ok_or(ArchiveSinkError::Finalized)?;
        validate_archive_frame(&frame, wal.header(), &self.schemas)?;
        let output = self
            .parquet
            .as_mut()
            .ok_or(ArchiveSinkError::Finalized)?
            .append_frame(frame.table_projections)?;
        merge_physical(&mut self.pending_physical, output);
        let wal = self.wal.as_mut().ok_or(ArchiveSinkError::Finalized)?;
        if let Err(error) = wal.append(&frame.wal_frame) {
            self.poisoned = true;
            return Err(ArchiveSinkError::Spool(error));
        }
        let completion = DurabilityCompletion {
            wal_segment_id: wal.header().segment_id,
            durable_prefix_hash: wal.durable_prefix(),
            frame_id: frame.wal_frame.header().frame_id,
            first_record_seq: frame.wal_frame.header().record_seq,
            last_record_seq: frame.wal_frame.header().record_seq,
            projection_coverage_digest: declared_projection_coverage_digest(&frame.wal_frame),
        };
        self.frames.push(frame.wal_frame);
        self.completions.push(completion.clone());
        Ok(completion)
    }

    async fn record_receipt(
        &mut self,
        draft: ReceiptEventDraft,
    ) -> Result<AppendReceipt, ArchiveSinkError> {
        self.ensure_operable()?;
        if !self.receipts_enabled {
            return Err(ArchiveSinkError::ReceiptJournalUnavailable);
        }
        let mut journal = ReceiptJournal::recover(
            self.repository.spool(),
            self.archive_id,
            self.faults.as_ref(),
        )?;
        validate_receipt_draft(
            self.archive_id,
            &draft,
            next_receipt_sequence(journal.last_receipt_seq())?,
        )?;
        validate_receipt_target(self.session_id, &draft.target, &self.completions)?;
        if let Some(epoch) = draft.observer_epoch {
            journal.append_observer_epoch(epoch, self.faults.as_ref())?;
        }
        let receipt = AppendReceipt {
            receipt_target_id: draft.target.receipt_target_id,
            event_id: draft.event.event_id,
            receipt_seq: draft.event.receipt_seq,
        };
        journal.record_event(draft.target, draft.event, self.faults.as_ref())?;
        Ok(receipt)
    }

    async fn checkpoint(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError> {
        self.checkpoint_sync()
    }

    async fn finalize(
        &mut self,
        reason: TerminationReason,
    ) -> Result<FinalizeCompletion, ArchiveSinkError> {
        self.ensure_operable()?;
        if self.frames.is_empty() {
            return Err(ArchiveSinkError::EmptyFinalization);
        }
        let checkpoint = self.checkpoint_sync()?;
        let wal = self.wal.take().ok_or(ArchiveSinkError::Finalized)?;
        let sealed_wal = match wal.seal() {
            Ok(sealed) => sealed,
            Err(error) => {
                self.poisoned = true;
                return Err(ArchiveSinkError::Spool(error));
            }
        };
        let final_next_record_seq = sealed_wal
            .last_record_seq()
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)?;
        let terminal_head = match self.repository.finalize_local(
            self.session_id,
            final_next_record_seq,
            termination_reason_name(reason).to_owned(),
            self.faults.as_ref(),
        ) {
            Ok(head) => head.clone(),
            Err(error) => {
                self.poisoned = true;
                return Err(ArchiveSinkError::Spool(error));
            }
        };
        self.state = ArchiveState::LocallyFinalized;
        Ok(FinalizeCompletion {
            checkpoint,
            sealed_wal,
            reason,
            archive_state: self.state,
            local_head: Some(terminal_head),
        })
    }

    fn local_repository(&self) -> Option<&LocalArchiveRepository> {
        Some(&self.repository)
    }

    fn into_local_repository(
        self: Box<Self>,
    ) -> Result<Option<LocalArchiveRepository>, ArchiveSinkError> {
        Ok(Some(OwnedLocalArchiveSink::into_repository(*self)?))
    }
}

fn prepare_receipt_journal(
    spool: &QualifiedSpool,
    archive_id: ArchiveId,
    mode: OwnedReceiptJournalMode,
    faults: &dyn DurabilityFaultInjector,
) -> Result<bool, ArchiveSinkError> {
    match mode {
        OwnedReceiptJournalMode::Disabled => Ok(false),
        OwnedReceiptJournalMode::Bootstrap(epoch) => {
            ReceiptJournal::bootstrap(spool, archive_id, epoch, faults)?;
            Ok(true)
        }
        OwnedReceiptJournalMode::Recover { observer_epoch } => {
            let mut journal = ReceiptJournal::recover(spool, archive_id, faults)?;
            if let Some(epoch) = observer_epoch {
                journal.append_observer_epoch(epoch, faults)?;
            }
            Ok(true)
        }
    }
}

fn checkpoint_recovered_frames(
    repository: &mut LocalArchiveRepository,
    frames: &[WalFrame],
    header: &WalSegmentHeaderV1,
    schemas: &ArchiveSchemasV1,
    rotation: ParquetRotationConfigV1,
    faults: &dyn DurabilityFaultInjector,
    decoder: &dyn ArchiveWalFrameDecoder,
) -> Result<(), ArchiveSinkError> {
    let mut parquet = ParquetPartitionBuilderV1::new(schemas.clone(), rotation)?;
    let mut physical = PartitionBuildOutputV1::default();
    for frame in frames {
        let decoded = decoder.decode(header.archive_id, header.session_id, frame.clone())?;
        validate_archive_frame(&decoded, header, schemas)?;
        let missing = unindexed_recovered_projections(repository, decoded, schemas)?;
        if !missing.is_empty() {
            merge_physical(&mut physical, parquet.append_frame(missing)?);
        }
    }
    merge_physical(&mut physical, parquet.finish()?);
    persist_partitions(repository.spool(), &physical, faults)?;
    let additions = physical_index_entries(&physical)?;
    let next_record_seq = frames.last().map_or(Ok(header.first_record_seq), |frame| {
        frame
            .header()
            .record_seq
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)
    })?;
    if !additions.is_empty() || repository.head().next_record_seq != next_record_seq {
        repository.checkpoint(
            &IndexMutationSetV1::new(Vec::new(), additions)?,
            header.session_id,
            next_record_seq,
            faults,
        )?;
    }
    Ok(())
}

fn rebuilt_sealed_wal(
    recovered: &crate::RecoveredWal,
) -> Result<SealedWalSegment, ArchiveSinkError> {
    let expected_digest = recovered
        .segment_digest
        .ok_or(ArchiveSinkError::RecoveredWalMismatch)?;
    let mut builder = WalSegmentBuilder::new(recovered.header.clone())?;
    for frame in &recovered.frames {
        builder.append(frame)?;
    }
    let sealed = builder.seal()?;
    if sealed.segment_digest() != expected_digest || sealed.final_prefix() != recovered.final_prefix
    {
        return Err(ArchiveSinkError::RecoveredWalMismatch);
    }
    Ok(sealed)
}

fn unindexed_recovered_projections(
    repository: &LocalArchiveRepository,
    frame: ArchiveWalFrame,
    schemas: &ArchiveSchemasV1,
) -> Result<Vec<FrameTableProjectionV1>, ArchiveSinkError> {
    let mut missing = Vec::new();
    for projection in frame.table_projections {
        let evidence = projection.validate(schemas)?;
        let table = [projection.table as u8];
        let logical_id = domain_digest(
            "aiperf.archive.projection-coverage.v1",
            &[projection.frame_id.digest().as_bytes(), &table],
        );
        let key = CompositeIndexKeyV1::projection_coverage(
            projection.table,
            projection.session_id,
            projection.source_id.as_deref(),
            projection.authoritative_frame_clock_ns,
            logical_id,
        )?;
        let Some(entry) = repository.index().get(&key) else {
            missing.push(projection);
            continue;
        };
        let coverage = ProjectionCoverageV1::from_canonical_bytes(entry.descriptor_bytes())?;
        if coverage.archive_id != projection.archive_id
            || coverage.session_id != projection.session_id
            || coverage.source_id != projection.source_id
            || coverage.frame_id != projection.frame_id
            || coverage.table != projection.table
            || coverage.authoritative_frame_clock_ns != projection.authoritative_frame_clock_ns
            || coverage.row_count != evidence.row_count
            || coverage.logical_multiset_digest != evidence.logical_multiset_digest
        {
            return Err(ArchiveSinkError::RecoveredCoverageMismatch);
        }
    }
    Ok(missing)
}

fn physical_index_entries(
    output: &PartitionBuildOutputV1,
) -> Result<Vec<crate::IndexEntry>, ArchiveSinkError> {
    let mut entries = Vec::new();
    for partition in &output.partitions {
        entries.push(partition.descriptor.index_entry()?);
        for coverage in &partition.coverage {
            entries.push(coverage.index_entry()?);
        }
    }
    for coverage in &output.zero_row_coverage {
        entries.push(coverage.index_entry()?);
    }
    Ok(entries)
}

const fn termination_reason_name(reason: TerminationReason) -> &'static str {
    match reason {
        TerminationReason::Requested => "requested",
        TerminationReason::Signal => "signal",
        TerminationReason::Duration => "duration",
        TerminationReason::Failure => "failure",
        TerminationReason::SyncOnly => "sync_only",
    }
}

fn validate_archive_frame(
    frame: &ArchiveWalFrame,
    segment: &WalSegmentHeaderV1,
    schemas: &ArchiveSchemasV1,
) -> Result<(), ArchiveSinkError> {
    let header = frame.wal_frame.header();
    if frame.table_projections.len() != header.required_projections.len() {
        return Err(ArchiveSinkError::ProjectionSetMismatch);
    }
    let mut evidence = BTreeMap::new();
    for projection in &frame.table_projections {
        if projection.archive_id != segment.archive_id
            || projection.session_id != segment.session_id
            || projection.frame_id != header.frame_id
            || projection.authoritative_frame_clock_ns != header.authoritative_frame_clock_ns
        {
            return Err(ArchiveSinkError::FrameIdentityMismatch);
        }
        let projection_evidence = projection.validate(schemas)?;
        if evidence
            .insert(projection.table, projection_evidence)
            .is_some()
        {
            return Err(ArchiveSinkError::ProjectionSetMismatch);
        }
    }
    if header
        .required_projections
        .iter()
        .any(|required| evidence.get(&required.table).copied() != Some(required.evidence))
    {
        return Err(ArchiveSinkError::ProjectionEvidenceMismatch);
    }
    Ok(())
}

fn durability_completion(wal: &WalSegmentBuilder, frame: &WalFrame) -> DurabilityCompletion {
    DurabilityCompletion {
        wal_segment_id: wal.header().segment_id,
        durable_prefix_hash: wal.prefix(),
        frame_id: frame.header().frame_id,
        first_record_seq: frame.header().record_seq,
        last_record_seq: frame.header().record_seq,
        projection_coverage_digest: declared_projection_coverage_digest(frame),
    }
}

fn local_durability_completion(wal: &LocalWalWriter<'_>, frame: &WalFrame) -> DurabilityCompletion {
    DurabilityCompletion {
        wal_segment_id: wal.header().segment_id,
        durable_prefix_hash: wal.durable_prefix(),
        frame_id: frame.header().frame_id,
        first_record_seq: frame.header().record_seq,
        last_record_seq: frame.header().record_seq,
        projection_coverage_digest: declared_projection_coverage_digest(frame),
    }
}

fn next_receipt_sequence(previous: Option<u64>) -> Result<u64, ArchiveSinkError> {
    previous.map_or(Ok(0), |value| {
        value
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)
    })
}

fn declared_projection_coverage_digest(frame: &WalFrame) -> Digest {
    let mut bytes = Vec::new();
    for projection in &frame.header().required_projections {
        bytes.push(projection.table as u8);
        bytes.extend_from_slice(&projection.evidence.row_count.to_be_bytes());
        bytes.extend_from_slice(projection.evidence.logical_multiset_digest.as_bytes());
    }
    domain_digest("aiperf.archive.frame-projection-coverage.v1", &[&bytes])
}

fn recovered_archive(
    archive_id: ArchiveId,
    session_id: SessionId,
    archive_state: ArchiveState,
    frames: &[WalFrame],
    uncertain: &BTreeSet<FrameId>,
    durable_prefix_hash: Digest,
    first_record_seq: u64,
) -> Result<RecoveredArchive, ArchiveSinkError> {
    let next_record_seq = frames.last().map_or(Ok(first_record_seq), |frame| {
        frame
            .header()
            .record_seq
            .checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)
    })?;
    Ok(RecoveredArchive {
        archive_id,
        session_id,
        archive_state,
        frames: frames.to_vec(),
        uncertain_frame_ids: uncertain.iter().copied().collect(),
        durable_prefix_hash,
        next_record_seq,
    })
}

fn next_record_seq(
    first_record_seq: u64,
    last_record_seq: Option<u64>,
) -> Result<u64, ArchiveSinkError> {
    last_record_seq.map_or(Ok(first_record_seq), |last| {
        last.checked_add(1)
            .ok_or(ArchiveSinkError::SequenceOverflow)
    })
}

fn finish_and_reset_parquet(
    builder: &mut Option<ParquetPartitionBuilderV1>,
    schemas: &ArchiveSchemasV1,
    rotation: ParquetRotationConfigV1,
    pending: &mut PartitionBuildOutputV1,
) -> Result<PartitionBuildOutputV1, ArchiveSinkError> {
    let finished = builder
        .take()
        .ok_or(ArchiveSinkError::Finalized)?
        .finish()?;
    merge_physical(pending, finished);
    let output = std::mem::take(pending);
    *builder = Some(ParquetPartitionBuilderV1::new(schemas.clone(), rotation)?);
    Ok(output)
}

fn merge_physical(target: &mut PartitionBuildOutputV1, mut output: PartitionBuildOutputV1) {
    target.partitions.append(&mut output.partitions);
    target
        .zero_row_coverage
        .append(&mut output.zero_row_coverage);
}

fn persist_partitions(
    spool: &QualifiedSpool,
    output: &PartitionBuildOutputV1,
    faults: &dyn DurabilityFaultInjector,
) -> Result<(), ArchiveSinkError> {
    for partition in &output.partitions {
        let actual = domain_digest("aiperf.archive.partition.v1", &[&partition.parquet_bytes]);
        if actual != partition.descriptor.physical_content_hash {
            return Err(ArchiveSinkError::PartitionContentHash);
        }
        spool.write_immutable(
            Path::new(&partition.descriptor.physical_object_key),
            &partition.parquet_bytes,
            ImmutableClass::Partition,
            faults,
        )?;
    }
    Ok(())
}

fn validate_receipt_draft(
    archive_id: ArchiveId,
    draft: &ReceiptEventDraft,
    expected_sequence: u64,
) -> Result<(), ArchiveSinkError> {
    if draft.target.archive_id() != archive_id
        || draft.event.archive_id != archive_id
        || draft.event.receipt_target_id != draft.target.receipt_target_id
    {
        return Err(ArchiveSinkError::ReceiptIdentityMismatch);
    }
    if draft.event.receipt_seq != expected_sequence {
        return Err(ArchiveSinkError::ReceiptSequence {
            expected: expected_sequence,
            actual: draft.event.receipt_seq,
        });
    }
    if draft
        .observer_epoch
        .as_ref()
        .is_some_and(|epoch| epoch.observer_epoch_id != draft.event.observer_epoch_id)
    {
        return Err(ArchiveSinkError::ReceiptIdentityMismatch);
    }
    Ok(())
}

fn validate_receipt_target(
    session_id: SessionId,
    target: &ReceiptTargetV1,
    completions: &[DurabilityCompletion],
) -> Result<(), ArchiveSinkError> {
    let Some(wal) = target.as_wal_range() else {
        return Ok(());
    };
    if wal.session_id != session_id {
        return Err(ArchiveSinkError::ReceiptTargetNotDurable);
    }
    let selected = completions
        .iter()
        .filter(|completion| {
            (wal.first_record_seq..=wal.last_record_seq).contains(&completion.first_record_seq)
        })
        .collect::<Vec<_>>();
    let expected_count = wal
        .last_record_seq
        .checked_sub(wal.first_record_seq)
        .and_then(|distance| distance.checked_add(1))
        .ok_or(ArchiveSinkError::ReceiptTargetNotDurable)?;
    if u64::try_from(selected.len()).ok() != Some(expected_count)
        || selected.iter().enumerate().any(|(offset, completion)| {
            completion.first_record_seq
                != u64::try_from(offset)
                    .ok()
                    .and_then(|offset| wal.first_record_seq.checked_add(offset))
                    .unwrap_or(u64::MAX)
                || completion.last_record_seq != completion.first_record_seq
                || completion.wal_segment_id != wal.wal_segment_id
        })
    {
        return Err(ArchiveSinkError::ReceiptTargetNotDurable);
    }
    let Some(last) = selected.last() else {
        return Err(ArchiveSinkError::ReceiptTargetNotDurable);
    };
    let coverage = receipt_range_coverage(
        selected
            .iter()
            .map(|completion| {
                (
                    completion.first_record_seq,
                    completion.projection_coverage_digest,
                )
            })
            .collect(),
    )?;
    if last.durable_prefix_hash != wal.durable_prefix_hash
        || coverage != wal.projection_coverage_digest
    {
        return Err(ArchiveSinkError::ReceiptTargetNotDurable);
    }
    Ok(())
}

fn ensure_operable(state: ArchiveState, recovery_required: bool) -> Result<(), ArchiveSinkError> {
    if recovery_required {
        return Err(ArchiveSinkError::RecoveryRequired);
    }
    if state != ArchiveState::Open {
        return Err(ArchiveSinkError::Finalized);
    }
    Ok(())
}

/// Append, projection, receipt, checkpoint, or finalization failure.
#[derive(Debug)]
pub enum ArchiveSinkError {
    /// Persistent index mutation validation failed.
    Index(crate::IndexError),
    /// Canonical WAL validation/assembly failed.
    Wal(WalError),
    /// Physical projection/Parquet encoding failed.
    Parquet(ParquetProjectionError),
    /// Verified WAL payload could not rebuild its source-frame projections.
    FrameDecode(Box<SourceFrameCodecError>),
    /// Verified WAL payload could not rebuild its control-frame projection.
    ControlFrameDecode(Box<ControlFrameCodecError>),
    /// Qualified-spool durability failed.
    Spool(SpoolError),
    /// Receipt journal failed.
    Receipt(ReceiptError),
    /// WAL and physical projections disagree on frame identity.
    FrameIdentityMismatch,
    /// WAL and physical projections name different table sets.
    ProjectionSetMismatch,
    /// WAL-declared logical evidence differs from physical/canonical rows.
    ProjectionEvidenceMismatch,
    /// Recovered coverage descriptor disagrees with WAL-declared evidence.
    RecoveredCoverageMismatch,
    /// Rebuilt recovered prefix differs from the owned WAL writer.
    RecoveredWalMismatch,
    /// A previous response was uncertain and recovery must run first.
    RecoveryRequired,
    /// Sink has already finalized.
    Finalized,
    /// Repository ownership was requested before successful local finalization.
    NotLocallyFinalized,
    /// Local writer failed after potentially mutating durable state.
    Poisoned,
    /// Outcome is uncertain; the caller must recover under the same identity.
    Uncertain(String),
    /// Finalization cannot seal an empty WAL segment.
    EmptyFinalization,
    /// Receipt journal was not configured.
    ReceiptJournalUnavailable,
    /// Receipt observer epoch is absent.
    MissingObserverEpoch,
    /// Same receipt epoch identity resolved to unequal bytes.
    ReceiptEpochCollision,
    /// Same receipt target identity resolved to unequal bytes.
    ReceiptTargetCollision,
    /// Receipt archive/target/event identities disagree.
    ReceiptIdentityMismatch,
    /// WAL target is not an exact earlier durable completion range.
    ReceiptTargetNotDurable,
    /// Receipt sequence differs from the next journal sequence.
    ReceiptSequence {
        /// Required sequence.
        expected: u64,
        /// Supplied sequence.
        actual: u64,
    },
    /// A completed partition's bytes disagree with its descriptor hash.
    PartitionContentHash,
    /// Monotone sequence arithmetic overflowed.
    SequenceOverflow,
}

impl From<WalError> for ArchiveSinkError {
    fn from(value: WalError) -> Self {
        Self::Wal(value)
    }
}

impl From<crate::IndexError> for ArchiveSinkError {
    fn from(value: crate::IndexError) -> Self {
        Self::Index(value)
    }
}

impl From<ParquetProjectionError> for ArchiveSinkError {
    fn from(value: ParquetProjectionError) -> Self {
        Self::Parquet(value)
    }
}

impl From<SpoolError> for ArchiveSinkError {
    fn from(value: SpoolError) -> Self {
        Self::Spool(value)
    }
}

impl From<ReceiptError> for ArchiveSinkError {
    fn from(value: ReceiptError) -> Self {
        Self::Receipt(value)
    }
}

impl Display for ArchiveSinkError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Index(error) => write!(formatter, "archive sink index failed: {error}"),
            Self::Wal(error) => write!(formatter, "archive sink WAL failed: {error}"),
            Self::Parquet(error) => write!(formatter, "archive sink Parquet failed: {error}"),
            Self::FrameDecode(error) => {
                write!(formatter, "archive sink frame decode failed: {error}")
            }
            Self::ControlFrameDecode(error) => {
                write!(
                    formatter,
                    "archive sink control-frame decode failed: {error}"
                )
            }
            Self::Spool(error) => write!(formatter, "archive sink spool failed: {error}"),
            Self::Receipt(error) => write!(formatter, "archive sink receipt failed: {error}"),
            Self::FrameIdentityMismatch => formatter.write_str("archive frame identity mismatch"),
            Self::ProjectionSetMismatch => formatter.write_str("archive projection set mismatch"),
            Self::ProjectionEvidenceMismatch => {
                formatter.write_str("archive projection evidence mismatch")
            }
            Self::RecoveredCoverageMismatch => {
                formatter.write_str("recovered indexed coverage disagrees with WAL evidence")
            }
            Self::RecoveredWalMismatch => {
                formatter.write_str("recovered WAL prefix reconstruction mismatch")
            }
            Self::RecoveryRequired => formatter.write_str("archive sink recovery is required"),
            Self::Finalized => formatter.write_str("archive sink is finalized"),
            Self::NotLocallyFinalized => {
                formatter.write_str("owned local sink is not locally finalized")
            }
            Self::Poisoned => formatter.write_str("archive sink is poisoned"),
            Self::Uncertain(message) => {
                write!(formatter, "archive sink outcome uncertain: {message}")
            }
            Self::EmptyFinalization => formatter.write_str("cannot finalize an empty archive WAL"),
            Self::ReceiptJournalUnavailable => {
                formatter.write_str("archive receipt journal is unavailable")
            }
            Self::MissingObserverEpoch => formatter.write_str("receipt observer epoch is missing"),
            Self::ReceiptEpochCollision => formatter.write_str("receipt observer epoch collision"),
            Self::ReceiptTargetCollision => formatter.write_str("receipt target collision"),
            Self::ReceiptIdentityMismatch => formatter.write_str("receipt identity mismatch"),
            Self::ReceiptTargetNotDurable => {
                formatter.write_str("receipt target is not an earlier durable WAL range")
            }
            Self::ReceiptSequence { expected, actual } => write!(
                formatter,
                "receipt sequence mismatch: expected {expected}, found {actual}"
            ),
            Self::PartitionContentHash => formatter.write_str("partition content hash mismatch"),
            Self::SequenceOverflow => formatter.write_str("archive sink sequence overflow"),
        }
    }
}

impl std::error::Error for ArchiveSinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Index(error) => Some(error),
            Self::Wal(error) => Some(error),
            Self::Parquet(error) => Some(error),
            Self::FrameDecode(error) => Some(error.as_ref()),
            Self::ControlFrameDecode(error) => Some(error.as_ref()),
            Self::Spool(error) => Some(error),
            Self::Receipt(error) => Some(error),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::RecordBatch;

    use super::*;
    use crate::{
        BatchId, CanonicalJsonValue, DurabilityEdge, EpochAnchor, ExecutionId,
        FailAtDurabilityEdge, GenerationObjectV1, GenerationTransactionKind, GenesisV1,
        NoDurabilityFaults, ObservationKind, ProjectionEvidence, ProjectionReservationId,
        ReceiptObserverEpochV1, RecoveryExpectation, RequiredProjection, TableId, TerminalKind,
        TimeDomain, WalFrameHeaderV1, WalRangeTargetV1,
    };

    #[derive(Debug)]
    struct ZeroFrameDecoder {
        schemas: ArchiveSchemasV1,
    }

    impl ArchiveWalFrameDecoder for ZeroFrameDecoder {
        fn decode(
            &self,
            archive_id: ArchiveId,
            session_id: SessionId,
            frame: WalFrame,
        ) -> Result<ArchiveWalFrame, ArchiveSinkError> {
            let required = frame
                .header()
                .required_projections
                .first()
                .ok_or(ArchiveSinkError::ProjectionSetMismatch)?;
            let schema = self
                .schemas
                .table(required.table)
                .map_err(|_| ArchiveSinkError::ProjectionSetMismatch)?;
            Ok(ArchiveWalFrame {
                table_projections: vec![FrameTableProjectionV1 {
                    archive_id,
                    session_id,
                    source_id: Some("source-a".to_owned()),
                    frame_id: frame.header().frame_id,
                    authoritative_frame_clock_ns: frame.header().authoritative_frame_clock_ns,
                    table: required.table,
                    batch: RecordBatch::new_empty(schema.schema().clone()),
                    logical_rows: vec![],
                }],
                wal_frame: frame,
            })
        }
    }

    fn archive() -> ArchiveId {
        ArchiveId::new([0x11; 16]).unwrap()
    }

    fn session() -> SessionId {
        SessionId::new([0x22; 16]).unwrap()
    }

    fn resumed_session() -> SessionId {
        SessionId::new([0x33; 16]).unwrap()
    }

    fn segment(schemas: &ArchiveSchemasV1) -> WalSegmentHeaderV1 {
        WalSegmentHeaderV1::new(
            archive(),
            session(),
            Digest::from_bytes([1; 32]),
            Digest::from_bytes([2; 32]),
            Digest::from_bytes([3; 32]),
            0,
            schemas
                .iter()
                .map(|schema| (schema.table(), schema.fingerprint()))
                .collect(),
        )
        .unwrap()
    }

    fn genesis() -> GenesisV1 {
        GenesisV1 {
            archive_id: archive(),
            canonical_spool_id: Digest::from_bytes([1; 32]),
            archive_identity_digest: Digest::from_bytes([2; 32]),
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
            time_domain: TimeDomain::Real,
            epoch_anchor: Some(EpochAnchor {
                clock_ns: 0,
                unix_epoch_ns: 1_700_000_000_000_000_000,
                capture_uncertainty_ns: 1,
            }),
        }
    }

    fn repository_segment(
        repository: &LocalArchiveRepository,
        schemas: &ArchiveSchemasV1,
    ) -> WalSegmentHeaderV1 {
        WalSegmentHeaderV1::new(
            archive(),
            session(),
            repository.head().generation_hash,
            repository.head().genesis_hash,
            repository.genesis().writer_compatibility_id,
            0,
            schemas
                .iter()
                .map(|schema| (schema.table(), schema.fingerprint()))
                .collect(),
        )
        .unwrap()
    }

    fn zero_frame(
        sequence: u64,
        schemas: &ArchiveSchemasV1,
    ) -> (ArchiveWalFrame, ProjectionEvidence) {
        let table = schemas.table(TableId::Families).unwrap();
        let batch = RecordBatch::new_empty(table.schema().clone());
        let evidence = ProjectionEvidence::empty();
        let reservation = ProjectionReservationId::new(domain_digest(
            "aiperf.archive.sink-test-reservation.v1",
            &[&sequence.to_be_bytes()],
        ));
        let header = WalFrameHeaderV1::new(
            BatchId::new(domain_digest(
                "aiperf.archive.sink-test-batch.v1",
                &[&sequence.to_be_bytes()],
            )),
            reservation,
            sequence,
            i64::try_from(sequence).unwrap(),
            TerminalKind::SourceScrape,
            vec![RequiredProjection {
                table: TableId::Families,
                evidence,
            }],
            vec![],
            vec![],
            0,
        )
        .unwrap();
        let frame_id = header.frame_id;
        (
            ArchiveWalFrame {
                wal_frame: WalFrame::new(header, vec![]).unwrap(),
                table_projections: vec![FrameTableProjectionV1 {
                    archive_id: archive(),
                    session_id: session(),
                    source_id: Some("source-a".to_owned()),
                    frame_id,
                    authoritative_frame_clock_ns: i64::try_from(sequence).unwrap(),
                    table: TableId::Families,
                    batch,
                    logical_rows: vec![],
                }],
            },
            evidence,
        )
    }

    fn two_zero_projection_frame(schemas: &ArchiveSchemasV1) -> ArchiveWalFrame {
        let reservation = ProjectionReservationId::new(domain_digest(
            "aiperf.archive.sink-test-reservation.v1",
            &[&0_u64.to_be_bytes()],
        ));
        let evidence = ProjectionEvidence::empty();
        let header = WalFrameHeaderV1::new(
            BatchId::new(domain_digest(
                "aiperf.archive.sink-test-batch.v1",
                &[&0_u64.to_be_bytes()],
            )),
            reservation,
            0,
            0,
            TerminalKind::SourceScrape,
            vec![
                RequiredProjection {
                    table: TableId::Families,
                    evidence,
                },
                RequiredProjection {
                    table: TableId::Samples,
                    evidence,
                },
            ],
            vec![],
            vec![],
            0,
        )
        .unwrap();
        let frame_id = header.frame_id;
        let table_projections = [TableId::Families, TableId::Samples]
            .into_iter()
            .map(|table| FrameTableProjectionV1 {
                archive_id: archive(),
                session_id: session(),
                source_id: Some("source-a".to_owned()),
                frame_id,
                authoritative_frame_clock_ns: 0,
                table,
                batch: RecordBatch::new_empty(schemas.table(table).unwrap().schema().clone()),
                logical_rows: vec![],
            })
            .collect();
        ArchiveWalFrame {
            wal_frame: WalFrame::new(header, vec![]).unwrap(),
            table_projections,
        }
    }

    #[tokio::test]
    async fn memory_append_uncertainty_recovers_same_frame_once() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let mut sink = MemoryArchiveSink::new(
            segment(&schemas),
            schemas.clone(),
            ParquetRotationConfigV1::default(),
        )
        .unwrap();
        let (first, _) = zero_frame(0, &schemas);
        let first_completion = sink.append_frame(first).await.unwrap();
        let (second, _) = zero_frame(1, &schemas);
        let second_id = second.wal_frame.header().frame_id;
        sink.set_fault(MemoryArchiveSinkFault::AppendUncertainAfterApply);
        assert!(matches!(
            sink.append_frame(second).await,
            Err(ArchiveSinkError::Uncertain(_))
        ));
        let (third, _) = zero_frame(2, &schemas);
        assert!(matches!(
            sink.append_frame(third).await,
            Err(ArchiveSinkError::RecoveryRequired)
        ));
        let recovered = sink.recover().await.unwrap();
        assert_eq!(recovered.frames.len(), 2);
        assert_eq!(recovered.uncertain_frame_ids, vec![second_id]);
        assert_eq!(recovered.next_record_seq, 2);
        assert_ne!(
            first_completion.durable_prefix_hash,
            recovered.durable_prefix_hash
        );
    }

    #[tokio::test]
    async fn receipt_requires_epoch_then_attests_an_earlier_wal_target() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let mut sink = MemoryArchiveSink::new(
            segment(&schemas),
            schemas.clone(),
            ParquetRotationConfigV1::default(),
        )
        .unwrap();
        let (frame, _) = zero_frame(0, &schemas);
        let completion = sink.append_frame(frame).await.unwrap();
        let epoch = ReceiptObserverEpochV1::new(
            ExecutionId::new([9; 16]).unwrap(),
            Some(session()),
            TimeDomain::Real,
            10,
            Some(100),
            1,
            Digest::from_bytes([8; 32]),
        )
        .unwrap();
        let target = ReceiptTargetV1::wal_range(WalRangeTargetV1 {
            archive_id: archive(),
            session_id: session(),
            wal_segment_id: completion.wal_segment_id,
            durable_prefix_hash: completion.durable_prefix_hash,
            first_record_seq: completion.first_record_seq,
            last_record_seq: completion.last_record_seq,
            projection_coverage_digest: receipt_range_coverage(vec![(
                completion.first_record_seq,
                completion.projection_coverage_digest,
            )])
            .unwrap(),
        })
        .unwrap();
        let event = ReceiptEventV1::new(
            archive(),
            0,
            target.receipt_target_id,
            epoch.observer_epoch_id,
            ObservationKind::ResponseObserved,
            11,
        );
        let receipt = sink
            .record_receipt(ReceiptEventDraft {
                observer_epoch: Some(epoch),
                target,
                event,
            })
            .await
            .unwrap();
        assert_eq!(receipt.receipt_seq, 0);
        assert_eq!(sink.receipt_events().len(), 1);
    }

    #[tokio::test]
    async fn checkpoint_preserves_zero_row_coverage_and_finalize_seals_wal() {
        let schemas = ArchiveSchemasV1::load().unwrap();
        let mut sink = MemoryArchiveSink::new(
            segment(&schemas),
            schemas.clone(),
            ParquetRotationConfigV1::default(),
        )
        .unwrap();
        let (frame, _) = zero_frame(0, &schemas);
        sink.append_frame(frame).await.unwrap();
        let checkpoint = sink.checkpoint().await.unwrap();
        assert_eq!(checkpoint.physical.zero_row_coverage.len(), 1);
        assert!(checkpoint.physical.partitions.is_empty());
        let finalized = sink.finalize(TerminationReason::Requested).await.unwrap();
        assert_eq!(finalized.sealed_wal.frame_count(), 1);
        assert_eq!(finalized.archive_state, ArchiveState::LocallyFinalized);
    }

    #[tokio::test]
    async fn owned_sink_installs_coverage_and_returns_finalized_repository() {
        let directory = tempfile::tempdir().unwrap();
        let spool = QualifiedSpool::open(directory.path().join("archive")).unwrap();
        let faults: Arc<dyn DurabilityFaultInjector> = Arc::new(NoDurabilityFaults);
        let repository =
            LocalArchiveRepository::create_new(spool, genesis(), faults.as_ref()).unwrap();
        let schemas = ArchiveSchemasV1::load().unwrap();
        let header = repository_segment(&repository, &schemas);
        let segment_id = header.segment_id;
        let factory = OwnedLocalArchiveSinkFactory::new(
            schemas.clone(),
            ParquetRotationConfigV1::default(),
            Arc::clone(&faults),
        )
        .unwrap();
        let mut sink = factory
            .prepare(repository, header, OwnedReceiptJournalMode::Disabled)
            .unwrap();
        assert_eq!(
            sink.repository().head().active_wal_segment_id,
            Some(segment_id)
        );
        assert_eq!(sink.repository().head().next_record_seq, 0);
        let sink_view: &dyn ArchiveSink = &sink;
        assert_eq!(
            sink_view.local_repository().unwrap().head().generation_hash,
            sink.repository().head().generation_hash
        );
        let (frame, _) = zero_frame(0, &schemas);
        sink.append_frame(frame).await.unwrap();
        let checkpoint = sink.checkpoint().await.unwrap();
        assert_eq!(checkpoint.physical.zero_row_coverage.len(), 1);
        assert_eq!(sink.repository().index().root().logical_entry_count, 1);
        assert_eq!(sink.repository().head().archive_state, ArchiveState::Open);
        assert_eq!(sink.repository().head().next_record_seq, 1);
        assert_eq!(
            sink.repository().head().active_wal_segment_id,
            Some(segment_id)
        );
        let finalized = sink.finalize(TerminationReason::Requested).await.unwrap();
        assert_eq!(finalized.sealed_wal.frame_count(), 1);
        assert_eq!(
            finalized.local_head.as_ref().unwrap().archive_state,
            ArchiveState::LocallyFinalized
        );
        let sink: Box<dyn ArchiveSink> = Box::new(sink);
        let repository = sink.into_local_repository().unwrap().unwrap();
        assert_eq!(
            repository.head().archive_state,
            ArchiveState::LocallyFinalized
        );
        assert_eq!(repository.head().next_record_seq, 1);
        assert_eq!(repository.head().active_wal_segment_id, None);
        assert_eq!(repository.latest_collection_session_id(), Some(session()));
        assert_eq!(
            repository.head().local_commit_seq,
            checkpoint.local_head.unwrap().local_commit_seq + 1
        );
    }

    #[tokio::test]
    async fn owned_factory_rotates_named_open_wal_and_ignores_decoy() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("archive");
        let spool = QualifiedSpool::open(&archive_path).unwrap();
        let faults: Arc<dyn DurabilityFaultInjector> = Arc::new(NoDurabilityFaults);
        let genesis = genesis();
        let expectation = RecoveryExpectation::from_genesis(&genesis);
        let repository =
            LocalArchiveRepository::create_new(spool, genesis, faults.as_ref()).unwrap();
        let schemas = ArchiveSchemasV1::load().unwrap();
        let header = repository_segment(&repository, &schemas);
        let factory = OwnedLocalArchiveSinkFactory::new(
            schemas.clone(),
            ParquetRotationConfigV1::default(),
            Arc::clone(&faults),
        )
        .unwrap();
        let mut sink = factory
            .prepare(
                repository,
                header.clone(),
                OwnedReceiptJournalMode::Disabled,
            )
            .unwrap();
        let (frame, _) = zero_frame(0, &schemas);
        sink.append_frame(frame).await.unwrap();
        drop(sink);

        let decoy_segment = Digest::from_bytes([0xdd; 32]);
        std::fs::write(
            archive_path
                .join("wal")
                .join(format!("{}.open", decoy_segment.to_hex())),
            b"unrelated decoy",
        )
        .unwrap();

        let spool = QualifiedSpool::open(&archive_path).unwrap();
        let repository =
            LocalArchiveRepository::recover(spool, expectation, faults.as_ref()).unwrap();
        assert_eq!(
            repository.head().active_wal_segment_id,
            Some(header.segment_id)
        );
        assert_eq!(repository.head().next_record_seq, 0);
        let decoder = ZeroFrameDecoder {
            schemas: schemas.clone(),
        };
        let mut resumed = factory
            .resume(
                repository,
                resumed_session(),
                1 << 20,
                OwnedReceiptJournalMode::Disabled,
                &decoder,
            )
            .unwrap();
        let recovered = resumed.recover().await.unwrap();
        assert!(recovered.frames.is_empty());
        assert_eq!(recovered.session_id, resumed_session());
        assert_eq!(recovered.next_record_seq, 1);
        assert_eq!(resumed.repository().index().root().logical_entry_count, 1);
        assert_eq!(resumed.repository().head().next_record_seq, 1);
        assert_eq!(resumed.repository().head().local_commit_seq, 4);
        assert_eq!(
            resumed.repository().latest_collection_session_id(),
            Some(resumed_session())
        );
        let active = resumed
            .repository()
            .active_wal_header(
                schemas
                    .iter()
                    .map(|schema| (schema.table(), schema.fingerprint()))
                    .collect(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(active.session_id, resumed_session());
        assert_ne!(active.segment_id, header.segment_id);
        assert!(
            archive_path
                .join("wal")
                .join(format!("{}.wal", header.segment_id.to_hex()))
                .is_file()
        );
        assert!(
            archive_path
                .join("wal")
                .join(format!("{}.open", decoy_segment.to_hex()))
                .is_file()
        );
    }

    #[tokio::test]
    async fn owned_resume_skips_projection_already_committed_after_wal_header() {
        let directory = tempfile::tempdir().unwrap();
        let archive_path = directory.path().join("archive");
        let spool = QualifiedSpool::open(&archive_path).unwrap();
        let faults: Arc<dyn DurabilityFaultInjector> = Arc::new(NoDurabilityFaults);
        let genesis = genesis();
        let expectation = RecoveryExpectation::from_genesis(&genesis);
        let repository =
            LocalArchiveRepository::create_new(spool, genesis, faults.as_ref()).unwrap();
        let schemas = ArchiveSchemasV1::load().unwrap();
        let header = repository_segment(&repository, &schemas);
        let factory = OwnedLocalArchiveSinkFactory::new(
            schemas.clone(),
            ParquetRotationConfigV1::default(),
            Arc::clone(&faults),
        )
        .unwrap();
        let mut sink = factory
            .prepare(
                repository,
                header.clone(),
                OwnedReceiptJournalMode::Disabled,
            )
            .unwrap();
        let (frame, _) = zero_frame(0, &schemas);
        sink.append_frame(frame).await.unwrap();
        sink.checkpoint().await.unwrap();
        drop(sink);

        let spool = QualifiedSpool::open(&archive_path).unwrap();
        let repository =
            LocalArchiveRepository::recover(spool, expectation, faults.as_ref()).unwrap();
        assert_eq!(repository.index().root().logical_entry_count, 1);
        let decoder = ZeroFrameDecoder {
            schemas: schemas.clone(),
        };
        let mut resumed = factory
            .resume(
                repository,
                resumed_session(),
                1 << 20,
                OwnedReceiptJournalMode::Disabled,
                &decoder,
            )
            .unwrap();
        assert_eq!(resumed.repository().head().local_commit_seq, 4);
        assert_eq!(resumed.repository().head().next_record_seq, 1);
        let checkpoint = resumed.checkpoint().await.unwrap();
        assert!(checkpoint.physical.partitions.is_empty());
        assert!(checkpoint.physical.zero_row_coverage.is_empty());
        assert_eq!(resumed.repository().index().root().logical_entry_count, 1);
    }

    #[tokio::test]
    async fn exact_resume_retries_after_seal_close_and_new_session_crashes() {
        let crash_cases = [
            (DurabilityEdge::WalSealDirectorySynced, 1, "prior seal"),
            (DurabilityEdge::PointerDirectorySynced, 1, "session close"),
            (DurabilityEdge::PointerDirectorySynced, 2, "new session"),
        ];
        for (edge, occurrence, stage) in crash_cases {
            let directory = tempfile::tempdir().unwrap();
            let archive_path = directory.path().join("archive");
            let safe_faults: Arc<dyn DurabilityFaultInjector> = Arc::new(NoDurabilityFaults);
            let genesis = genesis();
            let expectation = RecoveryExpectation::from_genesis(&genesis);
            let repository = LocalArchiveRepository::create_new(
                QualifiedSpool::open(&archive_path).unwrap(),
                genesis,
                safe_faults.as_ref(),
            )
            .unwrap();
            let schemas = ArchiveSchemasV1::load().unwrap();
            let prior_header = repository_segment(&repository, &schemas);
            let safe_factory = OwnedLocalArchiveSinkFactory::new(
                schemas.clone(),
                ParquetRotationConfigV1::default(),
                Arc::clone(&safe_faults),
            )
            .unwrap();
            let mut prior = safe_factory
                .prepare(
                    repository,
                    prior_header.clone(),
                    OwnedReceiptJournalMode::Disabled,
                )
                .unwrap();
            prior.append_frame(zero_frame(0, &schemas).0).await.unwrap();
            prior.checkpoint().await.unwrap();
            drop(prior);

            let repository = LocalArchiveRepository::recover(
                QualifiedSpool::open(&archive_path).unwrap(),
                expectation,
                safe_faults.as_ref(),
            )
            .unwrap();
            let crash_faults: Arc<dyn DurabilityFaultInjector> =
                Arc::new(FailAtDurabilityEdge::occurrence(edge, occurrence));
            let crash_factory = OwnedLocalArchiveSinkFactory::new(
                schemas.clone(),
                ParquetRotationConfigV1::default(),
                crash_faults,
            )
            .unwrap();
            let decoder = ZeroFrameDecoder {
                schemas: schemas.clone(),
            };
            let result = crash_factory.resume(
                repository,
                resumed_session(),
                1 << 20,
                OwnedReceiptJournalMode::Disabled,
                &decoder,
            );
            assert!(
                matches!(
                    result,
                    Err(ArchiveSinkError::Spool(SpoolError::FaultInjected(actual)))
                        if actual == edge
                ),
                "stage={stage}"
            );

            let repository = LocalArchiveRepository::recover(
                QualifiedSpool::open(&archive_path).unwrap(),
                expectation,
                safe_faults.as_ref(),
            )
            .unwrap();
            let mut resumed = safe_factory
                .resume(
                    repository,
                    resumed_session(),
                    1 << 20,
                    OwnedReceiptJournalMode::Disabled,
                    &decoder,
                )
                .unwrap();
            let recovered = resumed.recover().await.unwrap();
            assert!(recovered.frames.is_empty(), "stage={stage}");
            assert_eq!(recovered.next_record_seq, 1, "stage={stage}");
            assert_eq!(
                resumed.repository().latest_collection_session_id(),
                Some(resumed_session()),
                "stage={stage}"
            );
            assert_eq!(
                resumed.repository().head().local_commit_seq,
                4,
                "stage={stage}"
            );
            let closed_hash = resumed.repository().head().parent_generation_hash.unwrap();
            let closed_key = crate::manifest::generation_key(3, closed_hash);
            let closed = GenerationObjectV1::decode(
                &resumed
                    .repository()
                    .spool()
                    .read_relative(Path::new(&closed_key))
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                closed.generation.transaction_kind,
                GenerationTransactionKind::SessionClosed,
                "stage={stage}"
            );
            assert_eq!(closed.generation.active_wal_segment_id, None);
            assert!(
                archive_path
                    .join("wal")
                    .join(format!("{}.wal", prior_header.segment_id.to_hex()))
                    .is_file(),
                "stage={stage}"
            );
        }
    }

    #[test]
    fn recovery_replays_only_the_missing_subset_of_frame_coverage() {
        let directory = tempfile::tempdir().unwrap();
        let spool = QualifiedSpool::open(directory.path().join("archive")).unwrap();
        let mut repository =
            LocalArchiveRepository::create_new(spool, genesis(), &NoDurabilityFaults).unwrap();
        let schemas = ArchiveSchemasV1::load().unwrap();
        let frame = two_zero_projection_frame(&schemas);
        let header = repository_segment(&repository, &schemas);
        repository
            .start_session(&header, &NoDurabilityFaults)
            .unwrap();
        let covered = ProjectionCoverageV1::new(
            &frame.table_projections[0],
            ProjectionEvidence::empty(),
            None,
        )
        .unwrap();
        repository
            .checkpoint(
                &IndexMutationSetV1::new(vec![], vec![covered.index_entry().unwrap()]).unwrap(),
                session(),
                1,
                &NoDurabilityFaults,
            )
            .unwrap();

        let missing = unindexed_recovered_projections(&repository, frame, &schemas).unwrap();
        assert_eq!(missing.len(), 1);
        assert_eq!(missing[0].table, TableId::Samples);
    }

    #[test]
    fn local_sink_type_is_send_and_default_faults_are_usable() {
        fn assert_send<T: Send>() {}
        assert_send::<MemoryArchiveSink>();
        assert_send::<OwnedLocalArchiveSink>();
        assert_send::<LocalParquetArchiveSink<'static>>();
        let _: &dyn DurabilityFaultInjector = &NoDurabilityFaults;
    }
}
