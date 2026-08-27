// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Streaming format validation, decoding, and canonical event contracts.

use std::any::Any;

use async_trait::async_trait;
use bytes::Bytes;
use serde::Serialize;
use serde_json::value::RawValue;

use super::{
    budget::BudgetLease,
    checkpoint::StreamingCheckpointParticipant,
    failure::StreamingIssueReporterHandle,
    identity::{ContentDigest, ImmutableObjectIdentity},
    source::{
        AcquiredPartition, PartitionAccessKind, SourceFrontier, SourceSeal,
        StreamingSourceDescriptor,
    },
    unit::{EventTimeUtc, StreamingSessionFragment},
};

pub use super::failure::{DecodeFailureCode, OrderingFailureCode, StreamFormatError};

/// Immutable registry metadata for one streaming format implementation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamingFormatDescriptor {
    /// Stable registry identifier.
    pub id: &'static str,
    /// Human-readable implementation description.
    pub description: &'static str,
    /// Semantic digest of canonical decoding behavior.
    pub semantic_digest: ContentDigest,
    /// Accepted immutable input media types.
    pub media_types: &'static [&'static str],
    /// Accepted source record schemas.
    pub input_schemas: &'static [&'static str],
    /// Callable source access required by this format.
    pub required_access: PartitionAccessKind,
    /// Projection behavior supported without whole-dataset retention.
    pub projection: FormatProjection,
    /// Canonical fragment schema emitted by this format.
    pub output_schema: &'static str,
    /// Whether emitted fragments preserve event time.
    pub has_event_time: bool,
    /// Whether emitted fragments preserve stable record identity.
    pub has_stable_record_ids: bool,
    /// Decoder retained-state behavior.
    pub retention: FormatStateRetention,
    /// Whether decoding can run under a virtual clock.
    pub supports_virtual_clock: bool,
}

/// Format projection capability.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatProjection {
    /// Every field of each accepted record is decoded.
    FullRecord,
    /// A validated bounded field projection can be decoded.
    BoundedFields,
}

/// Format-owned retained-state requirement.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FormatStateRetention {
    /// Decoder state is caller-budgeted and independent of total input size.
    BoundedMemory,
    /// Validated spill authority is required for decoder state.
    BoundedSpill,
    /// The implementation requires the complete input resident at once.
    ResidentInput,
}

/// Type-erased, strictly validated format configuration.
pub trait ValidatedStreamingFormatConfig: std::fmt::Debug + Send + Sync {
    /// Borrow the concrete startup-only value.
    fn as_any(&self) -> &dyn Any;

    /// Consume the concrete startup-only value.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

impl<T> ValidatedStreamingFormatConfig for T
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

/// Host-owned format preparation context.
#[derive(Clone, Debug)]
pub struct StreamingFormatPrepareContext {
    /// Semantic namespace of the selected stream.
    pub stream_semantic_digest: ContentDigest,
    /// Host-owned reliability issue reporting boundary.
    pub issue_reporter: StreamingIssueReporterHandle,
}

/// Startup format validation and preparation contract.
pub trait StreamingDatasetFormatFactory: std::fmt::Debug + Send + Sync {
    /// Describe the exact compiled format implementation.
    fn descriptor(&self) -> &'static StreamingFormatDescriptor;

    /// Strictly decode and validate format-owned configuration.
    fn validate(
        &self,
        authored: &RawValue,
        source: &StreamingSourceDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingFormatConfig>, StreamFormatError>;

    /// Prepare one run-scoped format owner.
    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingFormatConfig>,
        context: &StreamingFormatPrepareContext,
    ) -> Result<Box<dyn StreamingDatasetFormat>, StreamFormatError>;
}

/// Run-scoped streaming format and checkpoint participant.
#[async_trait(?Send)]
pub trait StreamingDatasetFormat: StreamingCheckpointParticipant {
    /// Begin or resume decoding one immutable acquired partition.
    async fn begin_partition(
        &mut self,
        partition: AcquiredPartition,
        resume: Option<DecoderCheckpoint>,
    ) -> Result<Box<dyn StreamingPartitionDecoder>, StreamFormatError>;

    /// Translate source completeness into canonical format events.
    async fn advance_source_frontier(
        &mut self,
        frontier: SourceFrontier,
        output: &mut dyn FormatEventSink,
    ) -> Result<(), StreamFormatError>;

    /// Validate source exhaustion and emit final format events.
    async fn seal(
        &mut self,
        seal: SourceSeal,
        output: &mut dyn FormatEventSink,
    ) -> Result<FormatSealReceipt, StreamFormatError>;
}

/// Dynamic decoder for one immutable partition.
#[async_trait(?Send)]
pub trait StreamingPartitionDecoder {
    /// Pull at most one caller-bounded fragment batch.
    async fn next_batch(
        &mut self,
        budget: DecodeBatchBudget,
    ) -> Result<DecodeStep, StreamFormatError>;

    /// Snapshot the exact decoder-private resume cursor.
    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError>;
}

/// Strict upper bounds for one decoder pull.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeBatchBudget {
    /// Maximum fragments returned by one pull.
    pub max_fragments: usize,
    /// Maximum newly retained bytes returned by one pull.
    pub max_bytes: usize,
}

/// Host-bound decoder checkpoint for one immutable partition.
#[derive(Debug)]
pub struct DecoderCheckpoint {
    /// Immutable partition generation being resumed.
    pub partition: ImmutableObjectIdentity,
    /// Selected format semantic digest.
    pub format_semantic_digest: ContentDigest,
    /// Format-private typed resume state.
    pub state: DecoderResumeState,
}

/// Opaque format-private resume cursor retained by the host.
#[derive(Debug)]
pub struct DecoderResumeState {
    bytes: Bytes,
    lease: BudgetLease,
}

impl DecoderResumeState {
    /// Construct a compact opaque decoder cursor with an exact retained charge.
    pub fn new(bytes: Bytes, lease: BudgetLease) -> Result<Self, StreamFormatError> {
        if lease.charged_items() != 1 || lease.charged_bytes() != bytes.len() {
            return Err(StreamFormatError::decode(
                DecodeFailureCode::BudgetInvariant,
            ));
        }
        let bytes = Bytes::from(bytes.as_ref().to_vec().into_boxed_slice());
        Ok(Self { bytes, lease })
    }

    /// Borrow the exact opaque decoder cursor bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the exact retained byte charge.
    #[must_use]
    pub fn charged_bytes(&self) -> usize {
        self.lease.charged_bytes()
    }
}

/// One bounded decoder progress step.
pub enum DecodeStep {
    /// A nonempty canonical fragment batch.
    Batch(DecodedFragmentBatch),
    /// Exact immutable partition exhaustion.
    End(DecodeReceipt),
}

/// Canonical fragments and the cursor immediately after them.
pub struct DecodedFragmentBatch {
    /// Move-only canonical fragments with retained byte ownership.
    pub fragments: Vec<StreamingSessionFragment>,
    /// Decoder state immediately after the final fragment.
    pub resume_after: DecoderResumeState,
}

/// Receipt proving one immutable partition was exhausted.
#[derive(Debug)]
pub struct DecodeReceipt {
    /// Immutable partition generation that was exhausted.
    pub partition: ImmutableObjectIdentity,
    /// Total canonical fragments decoded from the partition.
    pub fragment_count: u64,
    /// Final decoder cursor.
    pub final_state: DecoderResumeState,
}

/// Bounded asynchronous sink for canonical format events.
#[async_trait(?Send)]
pub trait FormatEventSink {
    /// Send one canonical fragment or session frontier downstream.
    async fn send(&mut self, event: FormatEvent) -> Result<(), StreamFormatError>;
}

/// Host-owned canonical output of the format layer.
// Keeping the move-only fragment inline avoids a per-record allocation on the decode path.
#[allow(clippy::large_enum_variant)]
pub enum FormatEvent {
    /// One canonical session-addressed fragment.
    Fragment(StreamingSessionFragment),
    /// Session completeness contributed by the selected format.
    SessionFrontier(SessionWatermark),
}

/// Session-time completeness contributed by a source format.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SessionWatermark {
    /// Greatest event time proven complete for the applicable session scope.
    pub through: EventTimeUtc,
    /// Digest binding the format-specific completeness proof.
    pub digest: ContentDigest,
}

/// Receipt returned after a format accepts an explicit source seal.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FormatSealReceipt {
    /// Digest binding the selected format and final decoder state.
    pub digest: ContentDigest,
    /// Total partitions exhausted before the seal.
    pub partition_count: u64,
}
