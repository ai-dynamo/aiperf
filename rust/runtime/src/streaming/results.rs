// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-neutral checkpoint result descriptors and budgeted read values.

pub mod compactor;
pub mod epoch;
pub mod index;
pub mod sink_status;

pub use compactor::{PreparedStreamingReport, StreamingResultCompactor};
pub use sink_status::SinkFinalizationFailureCode;

use std::{
    fmt,
    mem::size_of,
    num::{NonZeroU64, NonZeroUsize},
};

use serde::{Deserialize, Deserializer, Serialize, Serializer as _, ser::SerializeSeq};

use super::{
    budget::BudgetLease,
    checkpoint::{BudgetedCheckpointBytes, CheckpointEpoch, CheckpointError, StreamRunIdentity},
    identity::{
        ActionAttemptId, ContentDigest, GlobalSequence, SessionOwnershipEpoch, StableActionId,
    },
    reliability::PreparedIssueReceiptEpochBinding,
    unit::DatasetActionKind,
};
use crate::{engine::records::CapturedRecord, metrics_core::RecordIngest};

/// Stable cell coordinate attached to one result segment.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CellId(u32);

impl CellId {
    /// Construct a cell coordinate.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the cell coordinate.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable worker coordinate attached to one result segment.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct WorkerId(u32);

impl WorkerId {
    /// Construct a worker coordinate.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the worker coordinate.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Stable nonempty result projection identity.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub struct ResultProjectionId(Box<str>);

impl ResultProjectionId {
    /// Construct a compact nonempty projection identity.
    pub fn new(value: impl Into<String>) -> Result<Self, CheckpointError> {
        let value = value.into();
        if value.is_empty() {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self(value.into_boxed_str()))
    }

    /// Borrow the projection text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Return the compact nested allocation retained by the projection.
    #[must_use]
    pub fn retained_allocation_bytes(&self) -> usize {
        self.0.len()
    }
}

impl<'de> Deserialize<'de> for ResultProjectionId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(serde::de::Error::custom)
    }
}

/// Version of the immutable result payload schema.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ResultSchemaVersion(u32);

impl ResultSchemaVersion {
    /// Construct a result schema version.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Return the result schema version.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Immutable metadata for one committed result payload.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultSegmentDescriptor {
    /// Logical run owning the segment.
    pub run: StreamRunIdentity,
    /// Result epoch owning the segment.
    pub epoch: CheckpointEpoch,
    /// Producing cell.
    pub cell_id: CellId,
    /// Producing worker.
    pub worker_id: WorkerId,
    /// Result projection identity.
    pub projection: ResultProjectionId,
    /// Result payload schema version.
    pub schema: ResultSchemaVersion,
    /// First global sequence represented by the payload.
    pub first_sequence: GlobalSequence,
    /// Last global sequence represented by the payload.
    pub last_sequence: GlobalSequence,
    /// Logical result item count.
    pub item_count: u64,
    /// Exact payload byte length.
    pub byte_length: u64,
    /// Digest of canonical logical membership.
    pub membership_root: ContentDigest,
    /// Raw BLAKE3 digest of the payload bytes.
    pub payload_digest: ContentDigest,
}

/// Singular result descriptor with inseparable retained-allocation authority.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptor;
/// # fn cannot_separate(value: BudgetedResultDescriptor) {
/// let _descriptor = value.descriptor;
/// let _lease = value.lease;
/// # }
/// ```
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptor;
/// # fn cannot_use_backend_transfer(value: BudgetedResultDescriptor) {
/// let _ = value.into_backend_parts();
/// # }
/// ```
#[derive(Debug)]
pub struct BudgetedResultDescriptor {
    descriptor: ResultSegmentDescriptor,
    lease: BudgetLease,
}

impl BudgetedResultDescriptor {
    /// Bind one descriptor to its exact compact retained allocation charge.
    pub fn new(
        descriptor: ResultSegmentDescriptor,
        lease: BudgetLease,
    ) -> Result<Self, CheckpointError> {
        let bytes = descriptor_retained_bytes(&descriptor)?;
        if lease.charged_items() != 1 || lease.charged_bytes() != bytes {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self { descriptor, lease })
    }

    /// Borrow the checked descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &ResultSegmentDescriptor {
        &self.descriptor
    }

    /// Return the exact descriptor allocation charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }

    pub(crate) fn into_backend_parts(self) -> (ResultSegmentDescriptor, BudgetLease) {
        (self.descriptor, self.lease)
    }
}

/// Verified result payload and its inseparable budgeted descriptor.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::ResultPartition;
/// # fn cannot_separate(value: ResultPartition) {
/// let _descriptor = value.descriptor;
/// let _payload = value.payload;
/// # }
/// ```
#[derive(Debug)]
pub struct ResultPartition {
    descriptor: BudgetedResultDescriptor,
    payload: BudgetedCheckpointBytes,
}

impl ResultPartition {
    /// Verify a descriptor against its separately budgeted payload.
    pub fn new(
        descriptor: BudgetedResultDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError> {
        verify_payload(descriptor.descriptor(), &payload)?;
        Ok(Self {
            descriptor,
            payload,
        })
    }

    /// Borrow the verified descriptor.
    #[must_use]
    pub fn descriptor(&self) -> &ResultSegmentDescriptor {
        self.descriptor.descriptor()
    }

    /// Return the descriptor's compact allocation charge.
    #[must_use]
    pub fn descriptor_charged_bytes(&self) -> usize {
        self.descriptor.charged_bytes()
    }

    /// Borrow the verified payload bytes.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Move both separately budgeted values without dismantling either authority.
    #[must_use]
    pub fn into_parts(self) -> (BudgetedResultDescriptor, BudgetedCheckpointBytes) {
        (self.descriptor, self.payload)
    }

    /// Whether the descriptor's sequence range contains the given value.
    #[must_use]
    pub fn contains_sequence(&self, seq: u64) -> bool {
        let seq = GlobalSequence::new(seq);
        self.descriptor().first_sequence <= seq && seq <= self.descriptor().last_sequence
    }
}

/// Descriptor collection with inseparable aggregate allocation authority.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::BudgetedResultDescriptors;
/// # fn cannot_separate(value: BudgetedResultDescriptors) {
/// let _descriptors = value.descriptors;
/// let _lease = value.lease;
/// # }
/// ```
#[derive(Debug)]
pub struct BudgetedResultDescriptors {
    descriptors: Box<[ResultSegmentDescriptor]>,
    lease: BudgetLease,
}

impl BudgetedResultDescriptors {
    pub(crate) fn new(
        descriptors: Box<[ResultSegmentDescriptor]>,
        lease: BudgetLease,
    ) -> Result<Self, CheckpointError> {
        let bytes = descriptors_retained_bytes(&descriptors)?;
        if lease.charged_items() != descriptors.len() || lease.charged_bytes() != bytes {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self { descriptors, lease })
    }

    /// Borrow all checked descriptors.
    #[must_use]
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor] {
        &self.descriptors
    }

    /// Return the exact aggregate retained allocation charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

/// Prepared result epoch returned by transaction staging.
#[derive(Debug)]
pub struct PreparedResultEpoch {
    index_root: ContentDigest,
    descriptors: BudgetedResultDescriptors,
    item_count: u64,
    byte_length: u64,
    issue_receipts: Option<PreparedIssueReceiptEpochBinding>,
}

impl PreparedResultEpoch {
    pub(crate) fn new(
        index_root: ContentDigest,
        descriptors: BudgetedResultDescriptors,
        item_count: u64,
        byte_length: u64,
        issue_receipts: Option<PreparedIssueReceiptEpochBinding>,
    ) -> Result<Self, CheckpointError> {
        let (computed_items, computed_bytes) = result_totals(descriptors.descriptors())?;
        if computed_items != item_count || computed_bytes != byte_length {
            return Err(CheckpointError::ObjectVerification);
        }
        if issue_receipts
            .as_ref()
            .is_some_and(|binding| binding.result_index_root() != &index_root)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(Self {
            index_root,
            descriptors,
            item_count,
            byte_length,
            issue_receipts,
        })
    }

    /// Borrow the canonical immutable index root.
    #[must_use]
    pub const fn index_root(&self) -> &ContentDigest {
        &self.index_root
    }

    /// Borrow the prepared descriptor inventory.
    #[must_use]
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor] {
        self.descriptors.descriptors()
    }

    /// Return the aggregate logical item count.
    #[must_use]
    pub const fn item_count(&self) -> u64 {
        self.item_count
    }

    /// Return the aggregate payload byte length.
    #[must_use]
    pub const fn byte_length(&self) -> u64 {
        self.byte_length
    }

    /// Borrow the staged detailed-receipt binding, when this epoch stages one.
    #[must_use]
    pub const fn issue_receipt_binding(&self) -> Option<&PreparedIssueReceiptEpochBinding> {
        self.issue_receipts.as_ref()
    }

    /// Move the summary while preserving its descriptor allocation authority.
    ///
    /// The staged detailed-receipt binding travels with the parts, so its view
    /// lease is released exactly once by whichever owner drops it.
    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        ContentDigest,
        BudgetedResultDescriptors,
        u64,
        u64,
        Option<PreparedIssueReceiptEpochBinding>,
    ) {
        (
            self.index_root,
            self.descriptors,
            self.item_count,
            self.byte_length,
            self.issue_receipts,
        )
    }
}

/// Verified budgeted reader for one immutable result payload.
#[derive(Debug)]
pub struct ResultSegmentReader {
    payload: BudgetedCheckpointBytes,
}

impl ResultSegmentReader {
    pub(crate) fn new(
        descriptor: &ResultSegmentDescriptor,
        payload: BudgetedCheckpointBytes,
    ) -> Result<Self, CheckpointError> {
        verify_payload(descriptor, &payload)?;
        Ok(Self { payload })
    }

    /// Borrow the verified result payload.
    #[must_use]
    pub fn payload_bytes(&self) -> &[u8] {
        self.payload.as_bytes()
    }

    /// Move the inseparable payload and read-budget authority.
    #[must_use]
    pub fn into_payload(self) -> BudgetedCheckpointBytes {
        self.payload
    }
}

/// Stable position within one immutable result-index block.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultIndexCursor {
    /// Root whose reachability grants scan authority.
    pub root: ContentDigest,
    /// Reachable immutable block containing the offset.
    pub block: ContentDigest,
    /// Next item offset within the block.
    pub item_offset: u32,
}

/// Caller-owned upper bounds for one result-index page.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResultIndexReadBudget {
    /// Maximum descriptors returned in one page.
    pub max_items: NonZeroUsize,
    /// Maximum retained descriptor bytes returned in one page.
    pub max_bytes: NonZeroU64,
}

/// One budgeted page of reachable result descriptors.
///
/// ```compile_fail
/// # use aiperf_runtime::streaming::results::ResultIndexPage;
/// # fn cannot_separate(value: ResultIndexPage) {
/// let _descriptors = value.descriptors;
/// let _next = value.next;
/// # }
/// ```
#[derive(Debug)]
pub struct ResultIndexPage {
    descriptors: BudgetedResultDescriptors,
    next: Option<ResultIndexCursor>,
}

impl ResultIndexPage {
    pub(crate) fn new(
        descriptors: BudgetedResultDescriptors,
        next: Option<ResultIndexCursor>,
    ) -> Result<Self, CheckpointError> {
        u64::try_from(descriptors.charged_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        Ok(Self { descriptors, next })
    }

    /// Borrow the reachable descriptors in this page.
    #[must_use]
    pub fn descriptors(&self) -> &[ResultSegmentDescriptor] {
        self.descriptors.descriptors()
    }

    /// Borrow the next cursor, when more descriptors are reachable.
    #[must_use]
    pub const fn next(&self) -> Option<&ResultIndexCursor> {
        self.next.as_ref()
    }

    /// Return the page's exact retained allocation charge.
    #[must_use]
    pub fn charged_bytes(&self) -> u64 {
        self.descriptors.charged_bytes() as u64
    }

    /// Move the page while preserving aggregate descriptor authority.
    #[must_use]
    pub fn into_parts(self) -> (BudgetedResultDescriptors, Option<ResultIndexCursor>) {
        (self.descriptors, self.next)
    }
}

/// Result-plane membership of one correlated terminal record.
///
/// Membership decides which projections may retain a record, not whether the
/// record is truthful. A state-only terminal is a real, complete fact; it simply
/// contributes no request metric.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultMembership {
    /// A materialized endpoint request with a terminal transport outcome.
    Request,
    /// A host-owned graph action that reached an endpoint.
    EndpointGraphAction,
    /// A session-state transition with no endpoint request of its own.
    SessionStateOnly,
    /// Per-attempt telemetry retained for provenance, never for request metrics.
    AttemptTelemetry,
}

impl ResultMembership {
    /// Classify a dataset action kind that reached an endpoint.
    #[must_use]
    pub const fn for_endpoint_action(kind: DatasetActionKind) -> Self {
        match kind {
            DatasetActionKind::Request => Self::Request,
            DatasetActionKind::GraphNode => Self::EndpointGraphAction,
            DatasetActionKind::SessionTerminal => Self::SessionStateOnly,
        }
    }

    /// Whether this membership contributes to request-shaped metrics.
    #[must_use]
    pub const fn is_request_shaped(self) -> bool {
        matches!(self, Self::Request | Self::EndpointGraphAction)
    }

    /// Stable lowercase tag used in diagnostics and canonical encodings.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::EndpointGraphAction => "endpoint_graph_action",
            Self::SessionStateOnly => "session_state_only",
            Self::AttemptTelemetry => "attempt_telemetry",
        }
    }
}

/// Stable identity joining one terminal record to its logical action.
///
/// `logical_action_id` is incarnation-free and therefore stable across restart;
/// `attempt_id` is not, and never enters a membership key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingRecordCorrelation {
    /// Semantic identity of the logical action this record completes.
    pub logical_action_id: StableActionId,
    /// Incarnation-local attempt that produced this record.
    pub attempt_id: ActionAttemptId,
    /// Dense host-assigned position in global replay order.
    pub global_sequence: GlobalSequence,
    /// Fencing epoch of the session route that owned the attempt.
    pub ownership_epoch: SessionOwnershipEpoch,
    /// Which projections may retain this record.
    pub membership: ResultMembership,
}

/// One terminal record joined to its correlation and optional captured facts.
///
/// The captured terminal facts are optional because artifact retention is
/// configured: a run with no per-record artifact keeps the correlation and the
/// metric ingest and drops the capture.
pub struct CorrelatedRecordIngest {
    /// Stable logical identity of the completed action.
    pub correlation: StreamingRecordCorrelation,
    /// Native metric ingestion record.
    pub record: RecordIngest,
    /// Retained terminal capture, when artifacts require it.
    pub captured: Option<CapturedRecord>,
}

impl CorrelatedRecordIngest {
    /// Join a captured terminal record to its correlation.
    #[must_use]
    pub fn from_captured(
        correlation: StreamingRecordCorrelation,
        captured: CapturedRecord,
    ) -> Self {
        Self {
            correlation,
            record: captured.record_ingest().clone(),
            captured: Some(captured),
        }
    }
}

/// Initial immutable result payload schema.
pub const RESULT_SCHEMA_V1: ResultSchemaVersion = ResultSchemaVersion::new(1);

/// Metrics retention selected for one checkpointed result plan.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum MetricsCheckpointProjection {
    /// Exact per-record retention as configured artifacts require.
    #[default]
    Exact,
    /// Mergeable t-digest retention; percentiles and deviation are estimates.
    Sketch,
}

impl MetricsCheckpointProjection {
    /// Whether this projection retains a record of the given membership.
    #[must_use]
    pub const fn accepts(self, membership: ResultMembership) -> bool {
        membership.is_request_shaped()
    }
}

/// Exact per-record result projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExactRecordProjection {
    /// Immutable payload schema version.
    pub schema: ResultSchemaVersion,
}

impl Default for ExactRecordProjection {
    fn default() -> Self {
        Self {
            schema: RESULT_SCHEMA_V1,
        }
    }
}

impl ExactRecordProjection {
    /// Whether this projection retains a record of the given membership.
    #[must_use]
    pub const fn accepts(self, membership: ResultMembership) -> bool {
        membership.is_request_shaped()
    }
}

/// Verbatim request/response projection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RawRecordProjection {
    /// Immutable payload schema version.
    pub schema: ResultSchemaVersion,
    /// Semantic digest of the redaction policy applied before retention.
    pub redaction_policy_digest: ContentDigest,
}

impl RawRecordProjection {
    /// Whether this projection retains a record of the given membership.
    ///
    /// Raw wire exists only for a materialized request; a graph action that
    /// reached an endpoint is retained through the exact-record projection.
    #[must_use]
    pub const fn accepts(&self, membership: ResultMembership) -> bool {
        matches!(membership, ResultMembership::Request)
    }
}

/// Session-scoped result projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SessionResultProjection {
    /// Immutable payload schema version.
    pub schema: ResultSchemaVersion,
}

impl Default for SessionResultProjection {
    fn default() -> Self {
        Self {
            schema: RESULT_SCHEMA_V1,
        }
    }
}

impl SessionResultProjection {
    /// Whether this projection retains a record of the given membership.
    #[must_use]
    pub const fn accepts(self, membership: ResultMembership) -> bool {
        !matches!(membership, ResultMembership::AttemptTelemetry)
    }
}

/// Provenance projection retained for every membership.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamingProvenanceProjection {
    /// Immutable payload schema version.
    pub schema: ResultSchemaVersion,
}

impl Default for StreamingProvenanceProjection {
    fn default() -> Self {
        Self {
            schema: RESULT_SCHEMA_V1,
        }
    }
}

impl StreamingProvenanceProjection {
    /// Whether this projection retains a record of the given membership.
    #[must_use]
    pub const fn accepts(self, _membership: ResultMembership) -> bool {
        true
    }
}

/// Barrier cadence selected for one run's result plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointInterval {
    /// Publish one generation per completed phase only.
    PerPhase,
    /// Publish after the stated number of terminal actions.
    EveryActions {
        /// Terminal actions between barriers.
        actions: NonZeroU64,
    },
    /// Publish after the stated elapsed `Clock` duration.
    EveryDuration {
        /// Nanoseconds of clock-driven time between barriers.
        nanos: NonZeroU64,
    },
}

/// Durability required from the checkpoint backend by one result plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CheckpointDurability {
    /// Results survive only while the producing process lives.
    ProcessLocal,
    /// Committed generations survive process replacement.
    Restartable,
}

/// Complete result-plane plan frozen for one logical run.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointResultPlan {
    /// Metrics retention mode.
    pub metrics: MetricsCheckpointProjection,
    /// Exact per-record projection, when configured artifacts require one.
    pub exact_records: Option<ExactRecordProjection>,
    /// Verbatim projection, when configured artifacts require one.
    pub raw_records: Option<RawRecordProjection>,
    /// Session-scoped projection.
    pub session_results: SessionResultProjection,
    /// Provenance projection.
    pub provenance: StreamingProvenanceProjection,
    /// Barrier cadence.
    pub interval: CheckpointInterval,
    /// Required backend durability.
    pub durability: CheckpointDurability,
}

impl CheckpointResultPlan {
    /// Whether any configured projection retains this membership.
    #[must_use]
    pub fn retains(&self, membership: ResultMembership) -> bool {
        self.metrics.accepts(membership)
            || self
                .exact_records
                .is_some_and(|projection| projection.accepts(membership))
            || self
                .raw_records
                .as_ref()
                .is_some_and(|projection| projection.accepts(membership))
            || self.session_results.accepts(membership)
            || self.provenance.accepts(membership)
    }
}

/// Result-plane membership, capacity, coverage, and verification failure.
///
/// Deliberately distinct from [`CheckpointError`]: the result plane and the
/// checkpoint backend do not share a failure vocabulary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ResultPlaneError {
    /// One reachable membership was associated with conflicting content.
    MembershipConflict {
        /// Canonical membership root under contention.
        membership_root: ContentDigest,
    },
    /// Producer-side partition descriptor budget was exhausted.
    ///
    /// Refusal maps only to this variant: never to backend, participant-state,
    /// storage, or provisional-hole capacity.
    PartitionDescriptorCapacityExceeded {
        /// Descriptor items requested at the point of refusal.
        items: u64,
        /// Descriptor retained bytes requested at the point of refusal.
        bytes: u64,
    },
    /// Provisional index capacity was exhausted.
    ProvisionalCapacityExceeded {
        /// Provisional items requested.
        items: u64,
        /// Provisional retained bytes requested.
        bytes: u64,
    },
    /// Declared sequence coverage is empty, inverted, or overlapping.
    InvalidCoverage,
    /// A descriptor's declared membership root did not re-derive.
    SegmentVerification,
    /// Canonical index encoding or block assembly failed.
    Compaction {
        /// Stable, user-readable compaction context.
        message: String,
    },
    /// One derived result sink refused a durable finalization transition.
    SinkFinalization {
        /// Stable machine-readable finalization refusal.
        code: SinkFinalizationFailureCode,
    },
}

impl ResultPlaneError {
    /// Return the stable machine-readable error code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::MembershipConflict { .. } => "membership_conflict",
            Self::PartitionDescriptorCapacityExceeded { .. } => {
                "partition_descriptor_capacity_exceeded"
            }
            Self::ProvisionalCapacityExceeded { .. } => "provisional_capacity_exceeded",
            Self::InvalidCoverage => "invalid_coverage",
            Self::SegmentVerification => "segment_verification",
            Self::Compaction { .. } => "compaction",
            Self::SinkFinalization { .. } => "sink_finalization",
        }
    }
}

impl fmt::Display for ResultPlaneError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MembershipConflict { membership_root } => write!(
                formatter,
                "{}: membership {:?} already holds different content",
                self.code(),
                membership_root
            ),
            Self::PartitionDescriptorCapacityExceeded { items, bytes } => write!(
                formatter,
                "{}: {items} items and {bytes} bytes exceed partition descriptor capacity",
                self.code()
            ),
            Self::ProvisionalCapacityExceeded { items, bytes } => write!(
                formatter,
                "{}: {items} items and {bytes} bytes exceed provisional index capacity",
                self.code()
            ),
            Self::InvalidCoverage | Self::SegmentVerification => {
                write!(formatter, "{}", self.code())
            }
            Self::Compaction { message } => write!(formatter, "{}: {message}", self.code()),
            Self::SinkFinalization { code } => {
                write!(formatter, "{}: {}", self.code(), code.as_str())
            }
        }
    }
}

impl std::error::Error for ResultPlaneError {}

pub(crate) fn descriptor_retained_bytes(
    descriptor: &ResultSegmentDescriptor,
) -> Result<usize, CheckpointError> {
    size_of::<ResultSegmentDescriptor>()
        .checked_add(descriptor.projection.retained_allocation_bytes())
        .ok_or(CheckpointError::ObjectVerification)
}

pub(crate) fn descriptors_retained_bytes(
    descriptors: &[ResultSegmentDescriptor],
) -> Result<usize, CheckpointError> {
    descriptors.iter().try_fold(0usize, |total, descriptor| {
        total
            .checked_add(descriptor_retained_bytes(descriptor)?)
            .ok_or(CheckpointError::ObjectVerification)
    })
}

pub(crate) fn result_totals(
    descriptors: &[ResultSegmentDescriptor],
) -> Result<(u64, u64), CheckpointError> {
    descriptors
        .iter()
        .try_fold((0u64, 0u64), |(items, bytes), descriptor| {
            Ok((
                items
                    .checked_add(descriptor.item_count)
                    .ok_or(CheckpointError::ObjectVerification)?,
                bytes
                    .checked_add(descriptor.byte_length)
                    .ok_or(CheckpointError::ObjectVerification)?,
            ))
        })
}

pub(crate) fn canonical_result_index_root(
    descriptors: &[ResultSegmentDescriptor],
) -> Result<ContentDigest, CheckpointError> {
    canonical_result_index_object(descriptors.iter()).map(|(root, _)| root)
}

pub(crate) fn canonical_result_index_object<'a>(
    descriptors: impl ExactSizeIterator<Item = &'a ResultSegmentDescriptor>,
) -> Result<(ContentDigest, Vec<u8>), CheckpointError> {
    let mut encoded = Vec::new();
    let mut serializer = serde_json::Serializer::new(&mut encoded);
    let mut sequence = serializer
        .serialize_seq(Some(descriptors.len()))
        .map_err(result_index_encoding_error)?;
    for descriptor in descriptors {
        sequence
            .serialize_element(descriptor)
            .map_err(result_index_encoding_error)?;
    }
    sequence.end().map_err(result_index_encoding_error)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"aiperf.streaming.result-index.v1");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(&encoded);
    Ok((
        ContentDigest::from_bytes(*hasher.finalize().as_bytes()),
        encoded,
    ))
}

fn result_index_encoding_error(error: serde_json::Error) -> CheckpointError {
    CheckpointError::Storage {
        message: format!("could not encode result index: {error}"),
    }
}

fn verify_payload(
    descriptor: &ResultSegmentDescriptor,
    payload: &BudgetedCheckpointBytes,
) -> Result<(), CheckpointError> {
    let byte_length =
        u64::try_from(payload.as_bytes().len()).map_err(|_| CheckpointError::ObjectVerification)?;
    let digest = ContentDigest::from_bytes(*blake3::hash(payload.as_bytes()).as_bytes());
    if payload.charged_bytes() != payload.as_bytes().len()
        || descriptor.byte_length != byte_length
        || descriptor.payload_digest != digest
    {
        return Err(CheckpointError::ObjectVerification);
    }
    Ok(())
}
