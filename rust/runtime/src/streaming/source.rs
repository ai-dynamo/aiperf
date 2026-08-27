// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming source discovery, immutable acquisition, and stop contracts.

use std::{any::Any, num::NonZeroUsize};

use async_trait::async_trait;
use bytes::Bytes;
use serde::Serialize;
use serde_json::value::RawValue;

use super::{
    budget::{BudgetLease, StreamingResourceBudget},
    checkpoint::StreamingCheckpointParticipant,
    failure::StreamingIssueReporterHandle,
    identity::{ContentDigest, ImmutableObjectIdentity},
    unit::SourcePosition,
};

pub use super::failure::{AcquisitionFailureCode, SourceFailureCode, StreamSourceError};

/// Immutable registry metadata for one streaming source implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamingSourceDescriptor {
    /// Stable registry identifier.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Inventory lifecycles the source can support concurrently.
    pub modes: &'static [StreamingSourceMode],
    /// Callable access shapes the source can acquire.
    pub access: &'static [PartitionAccessKind],
    /// Ordering guarantee made by source discovery.
    pub ordering: StreamingSourceOrdering,
    /// Exact resume granularities the source can honor.
    pub resume: &'static [StreamingResumeGranularity],
    /// Whether source records carry event time.
    pub has_event_time: bool,
    /// Whether source records carry producer-stable identities.
    pub has_stable_record_ids: bool,
    /// Retention needed to reacquire immutable content.
    pub retention: StreamingSourceRetention,
    /// Placement behavior supported by discovery and acquisition.
    pub placement: StreamingSourcePlacement,
    /// Whether the source can execute without wall-clock timers.
    pub supports_virtual_clock: bool,
}

/// Source inventory lifecycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingSourceMode {
    /// Inventory is complete after an explicit seal.
    Finite,
    /// Inventory may grow until host control stops it.
    Follow,
}

/// Partition access shape advertised during compatibility validation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PartitionAccessKind {
    /// Bounded forward byte chunks.
    Sequential,
    /// Immutable no-follow seekable local snapshot.
    SeekableLocal,
    /// Bounded reads against one immutable object generation.
    RangeReadable,
}

/// Ordering guaranteed by source discovery.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingSourceOrdering {
    /// No ordering beyond immutable object identity.
    None,
    /// Stable partition positions are monotonic.
    Partition,
    /// Event-time completeness frontiers are monotonic.
    EventTime,
}

/// Resume coordinate accepted by a source/format combination.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingResumeGranularity {
    /// Resume at an immutable partition boundary.
    Partition,
    /// Resume at an exact byte offset.
    Byte,
    /// Resume at a format row-group boundary.
    RowGroup,
    /// Resume at an exact canonical record boundary.
    Record,
}

/// Source-owned retained-state requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingSourceRetention {
    /// Only caller-budgeted acquired chunks are retained.
    BoundedMemory,
    /// Immutable objects remain reachable through the committed resume root.
    ResumeRootReachability,
}

/// Source work placement supported by the implementation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamingSourcePlacement {
    /// Discovery and acquisition remain controller-local.
    ControllerOnly,
    /// Immutable partition assignments may be placed on workers or cells.
    ImmutablePartitionAssignment,
}

/// Type-erased, strictly validated source configuration.
pub trait ValidatedStreamingSourceConfig: std::fmt::Debug + Send + Sync {
    /// Borrow the concrete startup-only value.
    fn as_any(&self) -> &dyn Any;

    /// Consume the concrete startup-only value.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedStreamingSourceConfig for T
where
    T: Any + std::fmt::Debug + Send + Sync,
{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        self
    }
}

/// Host-owned source preparation context.
#[derive(Clone, Debug)]
pub struct StreamingSourcePrepareContext {
    /// Budget used for immutable acquired partition bytes.
    pub acquisition_budget: AcquisitionBudget,
    /// Host-owned reliability issue reporting boundary.
    pub issue_reporter: StreamingIssueReporterHandle,
}

/// Startup source validation and preparation contract.
pub trait StreamingDatasetSourceFactory: std::fmt::Debug + Send + Sync {
    /// Describe the exact compiled source implementation.
    fn descriptor(&self) -> &'static StreamingSourceDescriptor;

    /// Strictly decode and validate source-owned configuration.
    fn validate(
        &self,
        authored: &RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError>;

    /// Prepare one run-local source without beginning discovery.
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingSourceConfig>,
        context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError>;
}

/// Prepared source that has not started discovery.
#[async_trait(?Send)]
pub trait PreparedStreamingDatasetSource {
    /// Open the source with a separately borrowable stop signal.
    async fn open(
        self: Box<Self>,
        stop: StreamingStopReceiver,
    ) -> Result<OpenedStreamingDatasetSource, StreamSourceError>;
}

/// One opened runtime source and its independently cloneable control handle.
pub struct OpenedStreamingDatasetSource {
    /// Sole mutable source event stream.
    pub source: Box<dyn StreamingDatasetSource>,
    /// Control that can wake a pending source event future.
    pub control: StreamingSourceControl,
}

/// Run-local source event stream and checkpoint owner.
#[async_trait(?Send)]
pub trait StreamingDatasetSource: StreamingCheckpointParticipant {
    /// Borrow the immutable source snapshot receipt.
    fn snapshot(&self) -> &SourceSnapshotReceipt;

    /// Wait for the next partition, frontier, or explicit seal.
    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError>;
}

/// Cloneable control that wakes every receiver when stop is requested.
#[derive(Clone, Debug)]
pub struct StreamingSourceControl {
    sender: tokio::sync::watch::Sender<bool>,
}

impl StreamingSourceControl {
    /// Request source shutdown without fabricating a source seal.
    pub fn stop(&self) {
        self.sender.send_replace(true);
    }

    /// Return whether stop has been requested.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        *self.sender.borrow()
    }
}

/// Cloneable receiver held by a runtime source while `next_event` is pending.
#[derive(Clone, Debug)]
pub struct StreamingStopReceiver {
    receiver: tokio::sync::watch::Receiver<bool>,
    control: StreamingSourceControl,
}

impl StreamingStopReceiver {
    /// Borrow a separately cloneable source control.
    #[must_use]
    pub fn control(&self) -> StreamingSourceControl {
        self.control.clone()
    }

    /// Return whether stop has already been requested.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        *self.receiver.borrow()
    }

    /// Wait until stop is requested.
    pub async fn stopped(&mut self) -> Result<(), StreamSourceError> {
        while !*self.receiver.borrow_and_update() {
            if self.receiver.changed().await.is_err() {
                break;
            }
        }
        Err(StreamSourceError::controlled_stop())
    }
}

/// Construct one independently cloneable source stop channel.
#[must_use]
pub fn streaming_stop_channel() -> (StreamingSourceControl, StreamingStopReceiver) {
    let (sender, receiver) = tokio::sync::watch::channel(false);
    let control = StreamingSourceControl { sender };
    (control.clone(), StreamingStopReceiver { receiver, control })
}

/// Digest binding the source's immutable discovery snapshot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceSnapshotReceipt {
    /// Semantic digest of the complete snapshot authority.
    pub digest: ContentDigest,
}

/// Event emitted by one opened source.
pub enum SourceEvent {
    /// One newly discovered immutable partition generation.
    Partition(SourcePartition),
    /// A monotonic source completeness frontier.
    Frontier(SourceFrontier),
    /// Explicit finite or policy-authorized source seal.
    Seal(SourceSeal),
}

/// Discovered partition metadata and opaque content authority.
pub struct SourcePartition {
    position: SourcePosition,
    content: Box<dyn SourcePartitionContent>,
}

impl SourcePartition {
    /// Bind a stable source position to immutable content authority.
    #[must_use]
    pub fn new(position: SourcePosition, content: Box<dyn SourcePartitionContent>) -> Self {
        Self { position, content }
    }

    /// Return the stable position of this partition.
    #[must_use]
    pub const fn position(&self) -> SourcePosition {
        self.position
    }

    /// Borrow the partition's immutable content authority.
    #[must_use]
    pub fn content(&self) -> &dyn SourcePartitionContent {
        self.content.as_ref()
    }
}

/// Monotonic source completeness frontier.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceFrontier {
    /// Greatest source position covered by the frontier.
    pub through: SourcePosition,
}

/// Explicit source exhaustion receipt.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceSeal {
    /// Final source position, absent for an empty source.
    pub final_position: Option<SourcePosition>,
    /// Digest binding the complete sealed inventory.
    pub digest: ContentDigest,
}

/// Access shape requested from immutable partition content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartitionAccessRequest {
    /// Begin or resume bounded sequential reads at an exact byte offset.
    Sequential { resume_offset: u64 },
    /// Acquire an immutable no-follow seekable local snapshot.
    SeekableLocal,
    /// Acquire a bounded reader for immutable byte ranges.
    RangeReadable,
}

/// Budget authority supplied to immutable partition acquisition.
#[derive(Clone, Debug)]
pub struct AcquisitionBudget {
    memory_budget: StreamingResourceBudget,
    disk_budget: StreamingResourceBudget,
}

impl AcquisitionBudget {
    /// Wrap distinct host-owned resident-memory and local-snapshot disk budgets.
    #[must_use]
    pub const fn new(
        memory_budget: StreamingResourceBudget,
        disk_budget: StreamingResourceBudget,
    ) -> Self {
        Self {
            memory_budget,
            disk_budget,
        }
    }

    /// Borrow the exact host-owned resident-memory budget for handles and chunks.
    #[must_use]
    pub const fn memory_budget(&self) -> &StreamingResourceBudget {
        &self.memory_budget
    }

    /// Borrow the exact host-owned disk budget for immutable local snapshots.
    #[must_use]
    pub const fn disk_budget(&self) -> &StreamingResourceBudget {
        &self.disk_budget
    }

    /// Acquire typed resident-memory capacity for a handle or returned chunk.
    pub async fn acquire_memory(
        &self,
        items: usize,
        bytes: usize,
    ) -> Result<AcquisitionMemoryLease, StreamSourceError> {
        self.memory_budget
            .acquire(items, bytes)
            .await
            .map(AcquisitionMemoryLease)
            .map_err(|_| {
                StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
            })
    }

    /// Acquire typed local-snapshot disk capacity.
    pub async fn acquire_disk(
        &self,
        items: usize,
        bytes: usize,
    ) -> Result<AcquisitionDiskLease, StreamSourceError> {
        self.disk_budget
            .acquire(items, bytes)
            .await
            .map(AcquisitionDiskLease)
            .map_err(|_| {
                StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
            })
    }
}

/// Move-only resident-memory capacity acquired from an [`AcquisitionBudget`].
#[derive(Debug)]
pub struct AcquisitionMemoryLease(BudgetLease);

/// Move-only local-snapshot disk capacity acquired from an [`AcquisitionBudget`].
#[derive(Debug)]
pub struct AcquisitionDiskLease(BudgetLease);

/// Opaque immutable partition content supplied by a source implementation.
#[async_trait(?Send)]
pub trait SourcePartitionContent {
    /// Borrow the immutable source-object generation identity.
    fn identity(&self) -> &ImmutableObjectIdentity;

    /// Return the exact byte length when known before acquisition.
    fn size_bytes(&self) -> Option<u64>;

    /// Acquire immutable bytes under the requested access contract and budget.
    async fn acquire(
        &self,
        request: PartitionAccessRequest,
        budget: &AcquisitionBudget,
    ) -> Result<AcquiredPartition, StreamSourceError>;
}

/// Move-only bounded source bytes and their exact resident-memory capacity.
#[derive(Debug)]
pub struct BudgetedSourceChunk {
    bytes: Bytes,
    lease: BudgetLease,
}

impl BudgetedSourceChunk {
    /// Bind compact bytes to an exact one-item resident-memory charge.
    pub fn new(bytes: Bytes, lease: AcquisitionMemoryLease) -> Result<Self, StreamSourceError> {
        let lease = lease.0;
        if lease.charged_items() != 1 || lease.charged_bytes() != bytes.len() {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::BudgetInvariant,
            ));
        }
        let bytes = Bytes::from(bytes.as_ref().to_vec().into_boxed_slice());
        Ok(Self { bytes, lease })
    }

    /// Borrow the bounded immutable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the exact resident byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

/// Bounded sequential chunk and immutable rolling-integrity receipt.
#[derive(Debug)]
pub struct SequentialSourceChunk {
    bytes: BudgetedSourceChunk,
    end_offset: u64,
    rolling_digest: ContentDigest,
}

impl SequentialSourceChunk {
    /// Bind one bounded chunk to the offset and digest immediately after it.
    #[must_use]
    pub const fn new(
        bytes: BudgetedSourceChunk,
        end_offset: u64,
        rolling_digest: ContentDigest,
    ) -> Self {
        Self {
            bytes,
            end_offset,
            rolling_digest,
        }
    }

    /// Borrow the bounded immutable bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_bytes()
    }

    /// Return the exact byte offset immediately after this chunk.
    #[must_use]
    pub const fn end_offset(&self) -> u64 {
        self.end_offset
    }

    /// Borrow the rolling digest through `end_offset`.
    #[must_use]
    pub const fn rolling_digest(&self) -> &ContentDigest {
        &self.rolling_digest
    }
}

/// Bounded forward reader for one immutable source-object generation.
#[async_trait(?Send)]
pub trait StreamingSequentialReader {
    /// Read at most `max_bytes`, retaining returned bytes under the memory budget.
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<Option<SequentialSourceChunk>, StreamSourceError>;
}

/// No-follow seekable authority over one immutable local snapshot.
#[async_trait(?Send)]
pub trait StreamingSeekableLocalSnapshot {
    /// Read at most `max_bytes` from an exact offset under the memory budget.
    async fn read_at(
        &self,
        offset: u64,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<BudgetedSourceChunk, StreamSourceError>;
}

/// Bounded immutable range-read authority.
#[async_trait(?Send)]
pub trait StreamingRangeReader {
    /// Read one exact bounded range under the resident-memory budget.
    async fn read_range(
        &self,
        offset: u64,
        length: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<BudgetedSourceChunk, StreamSourceError>;
}

/// Acquired bounded sequential reader and its exact handle capacity.
pub struct AcquiredSequentialPartition {
    reader: Box<dyn StreamingSequentialReader>,
    authority_lease: BudgetLease,
    next_offset: u64,
    size_bytes: Option<u64>,
    is_eof: bool,
}

impl AcquiredSequentialPartition {
    /// Pull one bounded chunk without retaining the complete logical object.
    pub async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<Option<SequentialSourceChunk>, StreamSourceError> {
        if self.is_eof {
            return Ok(None);
        }
        let Some(chunk) = self.reader.next_chunk(max_bytes, budget).await? else {
            match self.size_bytes {
                Some(size) if self.next_offset < size => {
                    return Err(StreamSourceError::acquisition(
                        AcquisitionFailureCode::TruncatedObject,
                    ));
                }
                Some(size) if self.next_offset > size => {
                    return Err(StreamSourceError::acquisition(
                        AcquisitionFailureCode::InvalidChunk,
                    ));
                }
                Some(_) => {}
                None => self.size_bytes = Some(self.next_offset),
            }
            self.is_eof = true;
            return Ok(None);
        };
        let length = u64::try_from(chunk.as_bytes().len()).map_err(|_| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        let expected_end = self.next_offset.checked_add(length).ok_or_else(|| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        if length == 0
            || chunk.as_bytes().len() > max_bytes.get()
            || chunk.end_offset() != expected_end
            || self.size_bytes.is_some_and(|size| expected_end > size)
        {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::InvalidChunk,
            ));
        }
        self.next_offset = expected_end;
        Ok(Some(chunk))
    }

    /// Return the exact resident byte charge for the reader handle.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.authority_lease.charged_bytes()
    }

    /// Return the advertised or EOF-frozen immutable length.
    #[must_use]
    pub const fn observed_size_bytes(&self) -> Option<u64> {
        self.size_bytes
    }
}

/// Acquired no-follow local snapshot and exact disk capacity ownership.
pub struct AcquiredSeekableLocalPartition {
    snapshot: Box<dyn StreamingSeekableLocalSnapshot>,
    disk_lease: BudgetLease,
    size_bytes: u64,
}

impl AcquiredSeekableLocalPartition {
    /// Read bounded bytes at an exact local-snapshot offset.
    pub async fn read_at(
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
        let chunk = self.snapshot.read_at(offset, max_bytes, budget).await?;
        let length = u64::try_from(chunk.as_bytes().len()).map_err(|_| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        if chunk.as_bytes().len() > max_bytes.get()
            || offset
                .checked_add(length)
                .is_none_or(|end| end > self.size_bytes)
        {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::InvalidChunk,
            ));
        }
        Ok(chunk)
    }

    /// Return the exact disk byte charge for the immutable snapshot.
    #[must_use]
    pub fn charged_disk_bytes(&self) -> usize {
        self.disk_lease.charged_bytes()
    }
}

/// Acquired immutable range reader and exact handle capacity.
pub struct AcquiredRangeReadablePartition {
    reader: Box<dyn StreamingRangeReader>,
    authority_lease: BudgetLease,
    size_bytes: Option<u64>,
}

impl AcquiredRangeReadablePartition {
    /// Read one bounded range, rejecting a known out-of-object request.
    pub async fn read_range(
        &self,
        offset: u64,
        length: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<BudgetedSourceChunk, StreamSourceError> {
        let length_u64 = u64::try_from(length.get()).map_err(|_| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        let end = offset.checked_add(length_u64).ok_or_else(|| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        if self.size_bytes.is_some_and(|size| end > size) {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::ObjectLimitExceeded,
            ));
        }
        let chunk = self.reader.read_range(offset, length, budget).await?;
        if chunk.as_bytes().len() != length.get() {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::InvalidChunk,
            ));
        }
        Ok(chunk)
    }

    /// Return the exact resident byte charge for the range authority.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.authority_lease.charged_bytes()
    }
}

/// Callable access authority selected during descriptor agreement.
pub enum AcquiredPartitionAccess {
    /// Bounded forward chunk reader.
    Sequential(AcquiredSequentialPartition),
    /// Immutable no-follow local snapshot.
    SeekableLocal(AcquiredSeekableLocalPartition),
    /// Bounded immutable range reader.
    RangeReadable(AcquiredRangeReadablePartition),
}

/// Move-only acquired partition identity, position, and bounded content authority.
pub struct AcquiredPartition {
    position: SourcePosition,
    identity: ImmutableObjectIdentity,
    size_bytes: Option<u64>,
    access: AcquiredPartitionAccess,
}

impl AcquiredPartition {
    /// Bind a bounded sequential reader to an exact one-item handle charge.
    pub fn sequential(
        position: SourcePosition,
        identity: ImmutableObjectIdentity,
        size_bytes: Option<u64>,
        resume_offset: u64,
        reader: Box<dyn StreamingSequentialReader>,
        authority_lease: AcquisitionMemoryLease,
    ) -> Result<Self, StreamSourceError> {
        if size_bytes.is_some_and(|size| resume_offset > size) {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::ObjectLimitExceeded,
            ));
        }
        let authority_lease = authority_lease.0;
        validate_memory_authority(&authority_lease)?;
        Ok(Self {
            position,
            identity,
            size_bytes,
            access: AcquiredPartitionAccess::Sequential(AcquiredSequentialPartition {
                reader,
                authority_lease,
                next_offset: resume_offset,
                size_bytes,
                is_eof: false,
            }),
        })
    }

    /// Bind an immutable local snapshot to its exact one-item disk charge.
    pub fn seekable_local(
        position: SourcePosition,
        identity: ImmutableObjectIdentity,
        size_bytes: u64,
        snapshot: Box<dyn StreamingSeekableLocalSnapshot>,
        disk_lease: AcquisitionDiskLease,
    ) -> Result<Self, StreamSourceError> {
        let disk_lease = disk_lease.0;
        let expected_bytes = usize::try_from(size_bytes).map_err(|_| {
            StreamSourceError::acquisition(AcquisitionFailureCode::ObjectLimitExceeded)
        })?;
        if disk_lease.charged_items() != 1 || disk_lease.charged_bytes() != expected_bytes {
            return Err(StreamSourceError::acquisition(
                AcquisitionFailureCode::BudgetInvariant,
            ));
        }
        Ok(Self {
            position,
            identity,
            size_bytes: Some(size_bytes),
            access: AcquiredPartitionAccess::SeekableLocal(AcquiredSeekableLocalPartition {
                snapshot,
                disk_lease,
                size_bytes,
            }),
        })
    }

    /// Bind an immutable range reader to an exact one-item handle charge.
    pub fn range_readable(
        position: SourcePosition,
        identity: ImmutableObjectIdentity,
        size_bytes: Option<u64>,
        reader: Box<dyn StreamingRangeReader>,
        authority_lease: AcquisitionMemoryLease,
    ) -> Result<Self, StreamSourceError> {
        let authority_lease = authority_lease.0;
        validate_memory_authority(&authority_lease)?;
        Ok(Self {
            position,
            identity,
            size_bytes,
            access: AcquiredPartitionAccess::RangeReadable(AcquiredRangeReadablePartition {
                reader,
                authority_lease,
                size_bytes,
            }),
        })
    }

    /// Return the stable position bound to this acquired content.
    #[must_use]
    pub const fn position(&self) -> SourcePosition {
        self.position
    }

    /// Borrow the immutable source-object generation identity.
    #[must_use]
    pub const fn identity(&self) -> &ImmutableObjectIdentity {
        &self.identity
    }

    /// Return the immutable logical object length when known.
    #[must_use]
    pub const fn size_bytes(&self) -> Option<u64> {
        self.size_bytes
    }

    /// Consume the common identity wrapper and retain the selected access authority.
    #[must_use]
    pub fn into_access(self) -> AcquiredPartitionAccess {
        self.access
    }
}

fn validate_memory_authority(lease: &BudgetLease) -> Result<(), StreamSourceError> {
    if lease.charged_items() != 1 || lease.charged_bytes() != 0 {
        return Err(StreamSourceError::acquisition(
            AcquisitionFailureCode::BudgetInvariant,
        ));
    }
    Ok(())
}
