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
    /// Whether the source is finite or follows new objects.
    pub mode: StreamingSourceMode,
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
}

/// Budget authority supplied to immutable partition acquisition.
#[derive(Clone, Debug)]
pub struct AcquisitionBudget {
    budget: StreamingResourceBudget,
}

impl AcquisitionBudget {
    /// Wrap the host-owned acquisition resource budget.
    #[must_use]
    pub const fn new(budget: StreamingResourceBudget) -> Self {
        Self { budget }
    }

    /// Borrow the exact host-owned resource budget for checked acquisition.
    #[must_use]
    pub const fn resource_budget(&self) -> &StreamingResourceBudget {
        &self.budget
    }
}

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

/// Move-only acquired partition bytes and their exact retained capacity.
#[derive(Debug)]
pub struct AcquiredPartition {
    position: SourcePosition,
    identity: ImmutableObjectIdentity,
    bytes: Bytes,
    cursor: usize,
    lease: BudgetLease,
}

impl AcquiredPartition {
    /// Bind immutable bytes to identity and an exact one-item byte charge.
    pub fn new(
        position: SourcePosition,
        identity: ImmutableObjectIdentity,
        resume_offset: u64,
        bytes: Bytes,
        lease: BudgetLease,
    ) -> Result<Self, StreamSourceError> {
        if lease.charged_items() != 1 || lease.charged_bytes() != bytes.len() {
            return Err(StreamSourceError::acquisition(
                super::failure::AcquisitionFailureCode::BudgetInvariant,
            ));
        }
        let cursor = usize::try_from(resume_offset)
            .ok()
            .filter(|cursor| *cursor <= bytes.len())
            .ok_or_else(|| {
                StreamSourceError::acquisition(
                    super::failure::AcquisitionFailureCode::ObjectLimitExceeded,
                )
            })?;
        let bytes = Bytes::from(bytes.as_ref().to_vec().into_boxed_slice());
        Ok(Self {
            position,
            identity,
            bytes,
            cursor,
            lease,
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

    /// Borrow and advance by at most `max_bytes` without minting retained capacity.
    pub fn next_chunk(&mut self, max_bytes: NonZeroUsize) -> Option<&[u8]> {
        if self.cursor == self.bytes.len() {
            return None;
        }
        let end = self
            .cursor
            .saturating_add(max_bytes.get())
            .min(self.bytes.len());
        let chunk = &self.bytes[self.cursor..end];
        self.cursor = end;
        Some(chunk)
    }

    /// Return the exact retained byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}
