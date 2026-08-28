// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict bounded decoder for `dynamo.request.trace.v1` request traces.
//!
//! One immutable JSONL object is one partition, read forward in bounded chunks
//! and resumable at an exact byte offset. Each line is validated by the same
//! `graph::recorded::dynamo::schema` grammar the finite compiler uses, so the
//! two cannot drift on sink envelopes, marker lines, scalar coercions, or the
//! arbitrary-precision cache-block hash domain.
//!
//! This decoder emits *deferred* recorded-request descriptors: validated replay
//! geometry, recorded metric facts, and producer parent/child identity. It never
//! synthesizes content, never resolves a producer root, never infers closure at
//! object EOF, and never allocates the finite compiler's virtual hashes. Content
//! reconstruction happens downstream, after the session coordinator proves the
//! whole producer tree closed.
//!
//! Nothing proportional to the object, the partition inventory, or the session
//! population is ever resident: state is one carry buffer bounded by the
//! authored maximum record size plus a fixed-width cursor.

use std::cell::Cell;
use std::num::NonZeroUsize;
use std::rc::Rc;

use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;
use smallvec::SmallVec;

use crate::graph::recorded::dynamo::schema::{
    AgentContext, EventType, ReplayMetrics, RequestMetrics, TraceRecord, parse_record,
};
use crate::graph::recorded::dynamo::{
    NormalizedReplayGeometry, normalize_replay_geometry, validate_request_counts,
    validate_session_id,
};

use super::super::budget::{BudgetLimits, StreamingResourceBudget};
use super::super::checkpoint::{
    BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
    CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
    PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
};
use super::super::failure::{
    DecodeFailureCode, OrdinaryStreamingFailure, StreamFormatError, StreamSourceError,
};
use super::super::format::{
    DecodeBatchBudget, DecodeReceipt, DecodeStep, DecodedFragmentBatch, DecoderCheckpoint,
    DecoderResumeState, FormatEvent, FormatEventSink, FormatProjection, FormatSealReceipt,
    FormatStateRetention, SessionWatermark, StreamingDatasetFormat, StreamingDatasetFormatFactory,
    StreamingFormatDescriptor, StreamingFormatPrepareContext, StreamingPartitionDecoder,
    ValidatedStreamingFormatConfig,
};
use super::super::identity::{
    ContentDigest, ImmutableObjectIdentity, StableOrderKey, physical_record_id,
    stable_record_id_from_key, stable_session_key,
};
use super::super::reliability::{
    OrdinaryStreamingIssue, StreamingInputDomainIdentity, StreamingIssueClass,
    StreamingIssueReporterHandle,
};
use super::super::source::{
    AcquiredPartition, AcquiredPartitionAccess, AcquiredSequentialPartition, AcquisitionBudget,
    PartitionAccessKind, SourceFrontier, SourceSeal, StreamingSourceDescriptor,
};
use super::super::unit::{
    AgentEventFragment, DeferredRecordedRequestFragment, DeferredReplayGeometry, EventTimeUtc,
    RecordedBlockHash, RecordedRequestFacts, SessionFragmentLease, SessionMutationV1,
    SourcePosition, StreamingSessionFragment, UnitProvenance,
};

/// Stable registry identifier for the strict streaming Dynamo decoder.
pub const STREAMING_DYNAMO_FORMAT_ID: &str = "streaming_dynamo_trace";
/// Exact record schema this decoder accepts.
pub const STREAMING_DYNAMO_INPUT_SCHEMA: &str = "dynamo.request.trace.v1";
/// Canonical event family of a forwarded recorded tool fact.
pub const STREAMING_DYNAMO_TOOL_EVENT_KIND: &str = "aiperf.dynamo.tool-event.v1";
/// Stable schema identity of this decoder's checkpoint payload.
pub const STREAMING_DYNAMO_SCHEMA_ID: &str = "aiperf.streaming.format.dynamo";
/// Current decoder checkpoint schema version.
pub const STREAMING_DYNAMO_SCHEMA_VERSION: u32 = 1;

/// Exact encoded width of the decoder's opaque resume cursor.
const CURSOR_BYTES: usize = 72;
/// Exact encoded width of the format's run-scoped checkpoint payload.
const FORMAT_STATE_BYTES: usize = 56;
/// Cursors that may be alive at once: one batch cursor, one receipt, one probe.
const CURSORS_IN_FLIGHT: usize = 4;

/// Compile-time semantic identity of this decoder implementation.
///
/// Bumped whenever emitted fragment content changes. It is not the per-run bound
/// authority digest, which additionally binds the synthesis profile and the
/// first executable block size.
const STREAMING_DYNAMO_SEMANTIC_DIGEST: ContentDigest = ContentDigest::from_bytes([
    0xd7, 0x9a, 0x11, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
]);

/// Immutable registry metadata for the strict streaming Dynamo decoder.
pub static STREAMING_DYNAMO_DESCRIPTOR: StreamingFormatDescriptor = StreamingFormatDescriptor {
    id: STREAMING_DYNAMO_FORMAT_ID,
    description: "Strict streaming Dynamo/NVCF request-trace decoder",
    semantic_digest: STREAMING_DYNAMO_SEMANTIC_DIGEST,
    media_types: &["application/x-ndjson", "application/jsonl"],
    input_schemas: &[STREAMING_DYNAMO_INPUT_SCHEMA],
    required_access: PartitionAccessKind::Sequential,
    projection: FormatProjection::FullRecord,
    output_schema: "aiperf.stream.session-fragment.v1",
    has_event_time: true,
    has_stable_record_ids: true,
    // No cross-record, cross-session, or cross-partition map: every causal fact
    // travels inside its own fragment.
    retention: FormatStateRetention::BoundedMemory,
    supports_virtual_clock: true,
};

/// Strictly authored Dynamo decoding policy, frozen before any partition.
///
/// Finite-only selection keys (`root_limit`, `max_context_length`, `max_osl`,
/// `idle_gap_cap_seconds`, `prompt_corpus`, `content_root_seed`) are refused by
/// `deny_unknown_fields`: they are whole-capture policies a bounded decoder
/// cannot honor, or profile inputs supplied through the prepared synthesis
/// profile rather than through format configuration.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamingDynamoFormatConfig {
    /// Refuse a line longer than this many bytes before any JSON allocation.
    pub max_record_bytes: usize,
    /// Bytes pulled from the sequential reader in one chunk.
    pub max_chunk_bytes: usize,
    /// Refuse a replay record declaring more complete blocks than this.
    pub max_block_hashes_per_record: usize,
    /// Refuse a bound block size above this many tokens.
    pub max_block_size: u32,
    /// Refuse a recorded `input_length` above this many tokens.
    pub max_input_length: u64,
    /// Forward recorded tool events as non-executable agent-event fragments.
    #[serde(default = "default_true")]
    pub emit_tool_events: bool,
}

const fn default_true() -> bool {
    true
}

/// Configuration proven bounded, paired with the frozen profile receipt.
#[derive(Debug)]
struct ValidatedDynamoConfig {
    config: StreamingDynamoFormatConfig,
    canonical_config_bytes: Vec<u8>,
    profile_digest: ContentDigest,
}

fn validate_bounds(config: &StreamingDynamoFormatConfig) -> Result<(), StreamFormatError> {
    let schema = || StreamFormatError::decode(DecodeFailureCode::Schema);
    if config.max_record_bytes == 0
        || config.max_chunk_bytes == 0
        || config.max_block_hashes_per_record == 0
        || config.max_block_size == 0
        || config.max_input_length == 0
    {
        return Err(schema());
    }
    // A chunk larger than the record bound cannot help and would over-retain.
    if config.max_chunk_bytes > config.max_record_bytes {
        return Err(schema());
    }
    Ok(())
}

fn canonical_config_bytes(config: &StreamingDynamoFormatConfig) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(41);
    bytes.extend_from_slice(&(config.max_record_bytes as u64).to_le_bytes());
    bytes.extend_from_slice(&(config.max_chunk_bytes as u64).to_le_bytes());
    bytes.extend_from_slice(&(config.max_block_hashes_per_record as u64).to_le_bytes());
    bytes.extend_from_slice(&config.max_block_size.to_le_bytes());
    bytes.extend_from_slice(&config.max_input_length.to_le_bytes());
    bytes.push(u8::from(config.emit_tool_events));
    bytes
}

fn digest_fields(domain: &[u8], fields: &[&[u8]]) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&(domain.len() as u64).to_le_bytes());
    hasher.update(domain);
    for field in fields {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

/// Fold the frozen synthesis semantics, the bound block size, and the authored
/// decoder bounds into one digest that names this run's content authority.
fn bound_authority_digest(
    profile_digest: &ContentDigest,
    block_size: u32,
    canonical_config: &[u8],
) -> ContentDigest {
    digest_fields(
        b"aiperf.streaming.dynamo.authority.v1",
        &[
            profile_digest.as_bytes(),
            &block_size.to_le_bytes(),
            canonical_config,
        ],
    )
}

/// Whether this run's content-synthesis authority has been bound yet.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AuthorityTag {
    /// No executable replay record has been decoded.
    Unbound,
    /// One block size and one authority digest are frozen for the run.
    Bound,
}

/// Frozen content-synthesis authority carried by the format and every cursor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AuthorityState {
    tag: AuthorityTag,
    /// Bound block size in tokens; zero while unbound.
    block_size: u32,
    /// Bound authority digest; all-zero while unbound.
    digest: ContentDigest,
}

impl AuthorityState {
    const fn unbound() -> Self {
        Self {
            tag: AuthorityTag::Unbound,
            block_size: 0,
            digest: ContentDigest::from_bytes([0; 32]),
        }
    }
}

/// Registry entry for the strict streaming Dynamo decoder.
///
/// The factory carries the frozen synthesis-profile receipt because this
/// decoder must verify and checkpoint the bound authority digest. It never
/// synthesizes content: expansion belongs to the deferred-reconstruction owner
/// downstream of producer-tree closure.
#[derive(Clone, Copy, Debug)]
pub struct StreamingDynamoFormatFactory {
    profile_digest: ContentDigest,
}

impl StreamingDynamoFormatFactory {
    /// Bind one host-prepared immutable synthesis-profile receipt.
    ///
    /// The digest must cover every semantic that changes reconstructed content:
    /// tokenizer artifact and revision, corpus identity and implementation
    /// version, content root seed, block-sampling algorithm, hash scope, and the
    /// tail/seed rule version.
    #[must_use]
    pub const fn new(profile_digest: ContentDigest) -> Self {
        Self { profile_digest }
    }
}

impl StreamingDatasetFormatFactory for StreamingDynamoFormatFactory {
    fn descriptor(&self) -> &'static StreamingFormatDescriptor {
        &STREAMING_DYNAMO_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        source: &StreamingSourceDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingFormatConfig>, StreamFormatError> {
        if !source
            .access
            .contains(&STREAMING_DYNAMO_DESCRIPTOR.required_access)
        {
            return Err(StreamFormatError::decode(DecodeFailureCode::Schema));
        }
        if self.profile_digest == ContentDigest::from_bytes([0; 32]) {
            return Err(StreamFormatError::decode(
                DecodeFailureCode::SynthesisProfileUnavailable,
            ));
        }
        let config: StreamingDynamoFormatConfig = serde_json::from_str(authored.get())
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Schema))?;
        validate_bounds(&config)?;
        Ok(Box::new(ValidatedDynamoConfig {
            canonical_config_bytes: canonical_config_bytes(&config),
            profile_digest: self.profile_digest,
            config,
        }))
    }

    fn prepare(
        &self,
        config: Box<dyn ValidatedStreamingFormatConfig>,
        context: &StreamingFormatPrepareContext,
    ) -> Result<Box<dyn StreamingDatasetFormat>, StreamFormatError> {
        let config: Box<ValidatedDynamoConfig> = config
            .into_any()
            .downcast()
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Schema))?;
        // Sized from authored bounds, never from any input.
        let cursor_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: CURSORS_IN_FLIGHT,
            max_bytes: CURSOR_BYTES * CURSORS_IN_FLIGHT,
        })
        .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        Ok(Box::new(StreamingDynamoFormat {
            run: context.run,
            stream_identity: context.stream_semantic_digest,
            reporter: context.issue_reporter.clone(),
            fragment_budget: context.fragment_budget.clone(),
            cursor_budget,
            config: Rc::new(*config),
            authority: Rc::new(Cell::new(AuthorityState::unbound())),
            latest_event_time: Rc::new(Cell::new(None)),
            last_partition: None,
            partitions_begun: 0,
            participant_id: CheckpointParticipantId::new("streaming_format_dynamo"),
            initialization: ParticipantInitialization::default(),
        }))
    }
}

/// Exact decoder-private position inside one immutable JSONL object.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DynamoCursor {
    /// Byte offset of the first unconsumed byte, always a line boundary.
    byte_offset: u64,
    /// Lines consumed, including markers and quarantined records.
    line_ordinal: u64,
    /// Fragments emitted from this partition.
    fragments_emitted: u64,
    /// Records quarantined in this partition.
    quarantined: u64,
    /// Content-synthesis authority as of exactly this offset.
    authority: AuthorityState,
}

impl DynamoCursor {
    const fn fresh(authority: AuthorityState) -> Self {
        Self {
            byte_offset: 0,
            line_ordinal: 0,
            fragments_emitted: 0,
            quarantined: 0,
            authority,
        }
    }

    // 8 + 8 + 8 + 8 + 1 tag + 3 reserved + 4 block_size + 32 digest = 72
    fn encode(self) -> [u8; CURSOR_BYTES] {
        let mut buffer = [0_u8; CURSOR_BYTES];
        buffer[0..8].copy_from_slice(&self.byte_offset.to_le_bytes());
        buffer[8..16].copy_from_slice(&self.line_ordinal.to_le_bytes());
        buffer[16..24].copy_from_slice(&self.fragments_emitted.to_le_bytes());
        buffer[24..32].copy_from_slice(&self.quarantined.to_le_bytes());
        buffer[32] = match self.authority.tag {
            AuthorityTag::Unbound => 0,
            AuthorityTag::Bound => 1,
        };
        buffer[36..40].copy_from_slice(&self.authority.block_size.to_le_bytes());
        buffer[40..72].copy_from_slice(self.authority.digest.as_bytes());
        buffer
    }

    fn decode(bytes: &[u8]) -> Result<Self, StreamFormatError> {
        let invalid = || StreamFormatError::decode(DecodeFailureCode::InvalidCursor);
        // Reserved padding is asserted zero so an unset field cannot be
        // smuggled through a checkpoint.
        if bytes.len() != CURSOR_BYTES || bytes[33..36] != [0, 0, 0] {
            return Err(invalid());
        }
        let tag = match bytes[32] {
            0 => AuthorityTag::Unbound,
            1 => AuthorityTag::Bound,
            _ => return Err(invalid()),
        };
        let block_size = read_u32(&bytes[36..40]).ok_or_else(invalid)?;
        // Unbound must be exactly zero-valued; bound must be positive.
        if (tag == AuthorityTag::Unbound) != (block_size == 0) {
            return Err(invalid());
        }
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&bytes[40..72]);
        Ok(Self {
            byte_offset: read_u64(&bytes[0..8]).ok_or_else(invalid)?,
            line_ordinal: read_u64(&bytes[8..16]).ok_or_else(invalid)?,
            fragments_emitted: read_u64(&bytes[16..24]).ok_or_else(invalid)?,
            quarantined: read_u64(&bytes[24..32]).ok_or_else(invalid)?,
            authority: AuthorityState {
                tag,
                block_size,
                digest: ContentDigest::from_bytes(digest),
            },
        })
    }
}

fn read_u64(bytes: &[u8]) -> Option<u64> {
    bytes.try_into().ok().map(u64::from_le_bytes)
}

fn read_u32(bytes: &[u8]) -> Option<u32> {
    bytes.try_into().ok().map(u32::from_le_bytes)
}

/// Run-scoped strict Dynamo format owner and checkpoint participant.
struct StreamingDynamoFormat {
    run: StreamRunIdentity,
    stream_identity: ContentDigest,
    reporter: StreamingIssueReporterHandle,
    fragment_budget: StreamingResourceBudget,
    cursor_budget: StreamingResourceBudget,
    config: Rc<ValidatedDynamoConfig>,
    /// Unbound until the first executable replay record; drift after that is terminal.
    authority: Rc<Cell<AuthorityState>>,
    /// Greatest recorded event time any decoder of this run has emitted.
    latest_event_time: Rc<Cell<Option<EventTimeUtc>>>,
    last_partition: Option<ImmutableObjectIdentity>,
    partitions_begun: u64,
    participant_id: CheckpointParticipantId,
    initialization: ParticipantInitialization,
}

impl StreamingDynamoFormat {
    fn completeness_digest(&self, through: EventTimeUtc) -> ContentDigest {
        digest_fields(
            b"aiperf.streaming.dynamo.frontier.v1",
            &[
                STREAMING_DYNAMO_SEMANTIC_DIGEST.as_bytes(),
                self.authority.get().digest.as_bytes(),
                &through.get().to_le_bytes(),
            ],
        )
    }

    fn seal_digest(&self) -> ContentDigest {
        digest_fields(
            b"aiperf.streaming.dynamo.seal.v1",
            &[
                STREAMING_DYNAMO_SEMANTIC_DIGEST.as_bytes(),
                self.authority.get().digest.as_bytes(),
                &self.partitions_begun.to_le_bytes(),
            ],
        )
    }

    /// Refuse a restored cursor that contradicts the run's frozen authority.
    ///
    /// A resumed bound cursor with a different block size or digest is drift,
    /// not a fresh binding.
    fn reconcile_restored_authority(&self, cursor: &DynamoCursor) -> Result<(), StreamFormatError> {
        let current = self.authority.get();
        match (current.tag, cursor.authority.tag) {
            (AuthorityTag::Unbound, AuthorityTag::Unbound) => Ok(()),
            (AuthorityTag::Unbound, AuthorityTag::Bound) => {
                self.authority.set(cursor.authority);
                Ok(())
            }
            (AuthorityTag::Bound, AuthorityTag::Unbound) => Err(StreamFormatError::decode(
                DecodeFailureCode::SynthesisAuthorityMismatch,
            )),
            (AuthorityTag::Bound, AuthorityTag::Bound) => {
                if current == cursor.authority {
                    Ok(())
                } else {
                    Err(StreamFormatError::decode(
                        DecodeFailureCode::SynthesisAuthorityMismatch,
                    ))
                }
            }
        }
    }

    fn encode_state(&self) -> [u8; FORMAT_STATE_BYTES] {
        let authority = self.authority.get();
        let mut buffer = [0_u8; FORMAT_STATE_BYTES];
        buffer[0..8].copy_from_slice(&self.partitions_begun.to_le_bytes());
        // Zero is a legal absent sentinel: `EventTimeUtc` refuses pre-epoch values
        // and a zero-nanosecond recorded time carries no schedule information.
        buffer[8..16].copy_from_slice(
            &self
                .latest_event_time
                .get()
                .map_or(0_i64, EventTimeUtc::get)
                .to_le_bytes(),
        );
        buffer[16] = match authority.tag {
            AuthorityTag::Unbound => 0,
            AuthorityTag::Bound => 1,
        };
        buffer[20..24].copy_from_slice(&authority.block_size.to_le_bytes());
        buffer[24..56].copy_from_slice(authority.digest.as_bytes());
        buffer
    }

    fn restore_state(&mut self, bytes: &[u8]) -> Result<(), CheckpointError> {
        if bytes.len() != FORMAT_STATE_BYTES || bytes[17..20] != [0, 0, 0] {
            return Err(CheckpointError::ObjectVerification);
        }
        let tag = match bytes[16] {
            0 => AuthorityTag::Unbound,
            1 => AuthorityTag::Bound,
            _ => return Err(CheckpointError::ObjectVerification),
        };
        let block_size = read_u32(&bytes[20..24]).ok_or(CheckpointError::ObjectVerification)?;
        let mut digest = [0_u8; 32];
        digest.copy_from_slice(&bytes[24..56]);
        let digest = ContentDigest::from_bytes(digest);
        let is_zero_digest = digest == ContentDigest::from_bytes([0; 32]);
        if (tag == AuthorityTag::Unbound) != (block_size == 0)
            || (tag == AuthorityTag::Unbound) != is_zero_digest
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let event_time = read_u64(&bytes[8..16]).ok_or(CheckpointError::ObjectVerification)?;
        let event_time =
            i64::try_from(event_time).map_err(|_| CheckpointError::ObjectVerification)?;
        self.partitions_begun =
            read_u64(&bytes[0..8]).ok_or(CheckpointError::ObjectVerification)?;
        self.latest_event_time.set(if event_time == 0 {
            None
        } else {
            Some(EventTimeUtc::new(event_time).map_err(|_| CheckpointError::ObjectVerification)?)
        });
        self.authority.set(AuthorityState {
            tag,
            block_size,
            digest,
        });
        Ok(())
    }
}

#[async_trait(?Send)]
impl StreamingDatasetFormat for StreamingDynamoFormat {
    async fn begin_partition(
        &mut self,
        partition: AcquiredPartition,
        resume: Option<DecoderCheckpoint>,
    ) -> Result<Box<dyn StreamingPartitionDecoder>, StreamFormatError> {
        let identity = *partition.identity();
        let position = partition.position();
        let AcquiredPartitionAccess::Sequential(access) = partition.into_access() else {
            return Err(StreamFormatError::decode(DecodeFailureCode::Schema));
        };

        let cursor = match resume {
            Some(checkpoint) => {
                if checkpoint.partition != identity
                    || checkpoint.format_semantic_digest != STREAMING_DYNAMO_SEMANTIC_DIGEST
                {
                    return Err(StreamFormatError::decode(DecodeFailureCode::InvalidCursor));
                }
                let cursor = DynamoCursor::decode(checkpoint.state.as_bytes())?;
                self.reconcile_restored_authority(&cursor)?;
                cursor
            }
            None => DynamoCursor::fresh(self.authority.get()),
        };

        if self.last_partition != Some(identity) {
            self.last_partition = Some(identity);
            self.partitions_begun = self.partitions_begun.saturating_add(1);
        }

        // Chunk capacity is transient: one chunk is copied into the carry buffer
        // and released before the next pull, so this budget is a bound on a
        // single in-flight read rather than on retained decoder state. The
        // sequential access shape never touches a local snapshot, so the disk
        // budget is a one-byte floor that no acquisition ever draws from.
        let acquisition_budget = AcquisitionBudget::new(
            StreamingResourceBudget::new(BudgetLimits {
                max_items: 1,
                max_bytes: self.config.config.max_chunk_bytes,
            })
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?,
            StreamingResourceBudget::new(BudgetLimits {
                max_items: 1,
                max_bytes: 1,
            })
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?,
        );

        Ok(Box::new(DynamoPartitionDecoder {
            access,
            acquisition_budget,
            identity,
            position,
            carry: Vec::new(),
            cursor,
            config: Rc::clone(&self.config),
            fragment_budget: self.fragment_budget.clone(),
            cursor_budget: self.cursor_budget.clone(),
            input_domain: StreamingInputDomainIdentity::new(self.stream_identity, identity),
            stream_identity: self.stream_identity,
            run: self.run,
            reporter: self.reporter.clone(),
            authority: Rc::clone(&self.authority),
            latest_event_time: Rc::clone(&self.latest_event_time),
        }))
    }

    async fn advance_source_frontier(
        &mut self,
        _frontier: SourceFrontier,
        output: &mut dyn FormatEventSink,
    ) -> Result<(), StreamFormatError> {
        // A source frontier proves discovery completeness over partition
        // positions. Event-time completeness is asserted only through the
        // greatest recorded time this format has actually emitted.
        let Some(through) = self.latest_event_time.get() else {
            return Ok(());
        };
        output
            .send(FormatEvent::SessionFrontier(SessionWatermark {
                through,
                digest: self.completeness_digest(through),
            }))
            .await
    }

    async fn seal(
        &mut self,
        _seal: SourceSeal,
        output: &mut dyn FormatEventSink,
    ) -> Result<FormatSealReceipt, StreamFormatError> {
        if let Some(through) = self.latest_event_time.get() {
            output
                .send(FormatEvent::SessionFrontier(SessionWatermark {
                    through,
                    digest: self.completeness_digest(through),
                }))
                .await?;
        }
        // A seal proves the source inventory exhausted. It is not a producer-tree
        // closure proof: the session coordinator owns that.
        Ok(FormatSealReceipt {
            digest: self.seal_digest(),
            partition_count: self.partitions_begun,
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for StreamingDynamoFormat {
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
        let bytes = Bytes::from(self.encode_state().to_vec());
        let budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: FORMAT_STATE_BYTES,
        })
        .map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = budget
            .acquire(1, bytes.len())
            .await
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let payload = BudgetedCheckpointBytes::new(bytes, lease)?;
        PreparedParticipantState::new(
            barrier.run,
            self.participant_id.clone(),
            STREAMING_DYNAMO_SCHEMA_ID,
            STREAMING_DYNAMO_SCHEMA_VERSION,
            barrier.cut.clone(),
            self.partitions_begun,
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
        if state.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        self.restore_state(state.payload_bytes())
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(())
    }
}

/// One complete newline-delimited record and the bytes it consumed.
struct PartitionLine {
    bytes: Vec<u8>,
    consumed: u64,
}

/// Disposition of one decoded line.
enum LineOutcome {
    /// One canonical fragment was produced.
    Fragment(Box<StreamingSessionFragment>),
    /// One faulty record was excluded and reported.
    Quarantined,
    /// A blank line, a marker line, or a suppressed tool record.
    Skipped,
}

/// Bounded strict line decoder for one immutable JSONL object.
///
/// The acquired partition is assumed to already start at the restored cursor's
/// byte offset: the host acquires with `PartitionAccessRequest::Sequential`
/// carrying that offset. The decoder never seeks and never re-reads consumed
/// bytes.
struct DynamoPartitionDecoder {
    access: AcquiredSequentialPartition,
    acquisition_budget: AcquisitionBudget,
    identity: ImmutableObjectIdentity,
    position: SourcePosition,
    /// Bytes of an incomplete trailing line, never a whole-object buffer.
    carry: Vec<u8>,
    cursor: DynamoCursor,
    config: Rc<ValidatedDynamoConfig>,
    fragment_budget: StreamingResourceBudget,
    cursor_budget: StreamingResourceBudget,
    input_domain: StreamingInputDomainIdentity,
    stream_identity: ContentDigest,
    run: StreamRunIdentity,
    reporter: StreamingIssueReporterHandle,
    authority: Rc<Cell<AuthorityState>>,
    latest_event_time: Rc<Cell<Option<EventTimeUtc>>>,
}

#[async_trait(?Send)]
impl StreamingPartitionDecoder for DynamoPartitionDecoder {
    async fn next_batch(
        &mut self,
        budget: DecodeBatchBudget,
    ) -> Result<DecodeStep, StreamFormatError> {
        let mut fragments = Vec::new();
        let mut retained_bytes = 0_usize;
        loop {
            if fragments.len() >= budget.max_fragments || retained_bytes >= budget.max_bytes {
                break;
            }
            // Capacity for one record is reserved *before* any source byte is
            // consumed, so a decoder parked on a saturated output budget can be
            // cancelled without losing a line. The reservation is the authored
            // record bound and is shrunk to the fragment's real retained size.
            let permit = self
                .fragment_budget
                .acquire(1, self.config.config.max_record_bytes)
                .await
                .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
            let Some(line) = self.next_line().await? else {
                drop(permit);
                break;
            };
            let outcome = self.decode_line(&line.bytes, permit).await?;
            match outcome {
                LineOutcome::Fragment(fragment) => {
                    retained_bytes = retained_bytes.saturating_add(fragment.lease.charged_bytes());
                    fragments.push(*fragment);
                    self.cursor.fragments_emitted += 1;
                }
                LineOutcome::Quarantined => self.cursor.quarantined += 1,
                // A marker line is not a fault and mints no receipt.
                LineOutcome::Skipped => {}
            }
            // Advance only across the exact proven line boundary.
            self.cursor.byte_offset += line.consumed;
            self.cursor.line_ordinal += 1;
        }
        if fragments.is_empty() {
            return Ok(DecodeStep::End(DecodeReceipt {
                partition: self.identity,
                fragment_count: self.cursor.fragments_emitted,
                final_state: self.resume_state()?,
            }));
        }
        Ok(DecodeStep::Batch(DecodedFragmentBatch {
            resume_after: self.resume_state()?,
            fragments,
        }))
    }

    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError> {
        let encoded = self.cursor.encode();
        let lease = self
            .cursor_budget
            .try_acquire(1, encoded.len())
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        DecoderResumeState::new(Bytes::copy_from_slice(&encoded), lease)
    }
}

impl DynamoPartitionDecoder {
    /// Return the next complete line, pulling bounded chunks as needed.
    ///
    /// Raw bytes are preserved until a full line exists: a UTF-8 code point may
    /// span file or network chunks, so `carry` is `Vec<u8>` and is never
    /// interpreted as text before the newline is found. A trailing line without
    /// a terminating newline at EOF is a complete record.
    async fn next_line(&mut self) -> Result<Option<PartitionLine>, StreamFormatError> {
        loop {
            if let Some(index) = self.carry.iter().position(|byte| *byte == b'\n') {
                let mut rest = self.carry.split_off(index + 1);
                std::mem::swap(&mut rest, &mut self.carry);
                let consumed = rest.len() as u64;
                return Ok(Some(PartitionLine {
                    bytes: rest,
                    consumed,
                }));
            }
            if self.carry.len() > self.config.config.max_record_bytes {
                // Refusal precedes any serde_json allocation.
                return Err(StreamFormatError::decode(
                    DecodeFailureCode::OversizedRecord,
                ));
            }
            let max = NonZeroUsize::new(self.config.config.max_chunk_bytes).ok_or(
                StreamFormatError::decode(DecodeFailureCode::BudgetInvariant),
            )?;
            match self.access.next_chunk(max, &self.acquisition_budget).await {
                Ok(Some(chunk)) => self.carry.extend_from_slice(chunk.as_bytes()),
                Ok(None) if self.carry.is_empty() => return Ok(None),
                Ok(None) => {
                    let bytes = std::mem::take(&mut self.carry);
                    let consumed = bytes.len() as u64;
                    return Ok(Some(PartitionLine { bytes, consumed }));
                }
                Err(error) => return Err(map_source_error(error)),
            }
        }
    }

    async fn decode_line(
        &mut self,
        line: &[u8],
        permit: super::super::budget::BudgetLease,
    ) -> Result<LineOutcome, StreamFormatError> {
        let trimmed = trim_ascii_line(line);
        if trimmed.is_empty() {
            return Ok(LineOutcome::Skipped);
        }
        if trimmed.len() > self.config.config.max_record_bytes {
            return Err(StreamFormatError::decode(
                DecodeFailureCode::OversizedRecord,
            ));
        }
        let Ok(text) = std::str::from_utf8(trimmed) else {
            self.quarantine_line(DecodeFailureCode::Syntax).await;
            return Ok(LineOutcome::Quarantined);
        };
        let Ok(raw) = serde_json::from_str::<&RawValue>(text) else {
            self.quarantine_line(DecodeFailureCode::Syntax).await;
            return Ok(LineOutcome::Quarantined);
        };
        // The shared grammar: envelope unwrapping, marker detection, exact
        // schema, closed event/status vocabularies, arbitrary-precision hashes.
        let record = match parse_record(raw, self.cursor.line_ordinal as usize) {
            // A marker line is not a record and not a fault.
            Ok(None) => return Ok(LineOutcome::Skipped),
            Ok(Some(record)) => record,
            Err(_) => {
                self.quarantine_line(DecodeFailureCode::Schema).await;
                return Ok(LineOutcome::Quarantined);
            }
        };
        match record.event_type {
            EventType::RequestEnd => self.decode_request_end(&record, permit).await,
            _ if self.config.config.emit_tool_events => {
                self.decode_tool_event(&record, permit).await
            }
            _ => Ok(LineOutcome::Skipped),
        }
    }

    async fn decode_request_end(
        &mut self,
        record: &TraceRecord,
        permit: super::super::budget::BudgetLease,
    ) -> Result<LineOutcome, StreamFormatError> {
        // Session identity is mandatory: a replay-only record has no producer
        // tree, so no downstream owner could ever reconstruct its content.
        let Some(context) = record.context.as_ref() else {
            self.quarantine_line(DecodeFailureCode::Schema).await;
            return Ok(LineOutcome::Quarantined);
        };
        if validate_session_id(&context.session_id).is_err() {
            self.quarantine_line(DecodeFailureCode::Schema).await;
            return Ok(LineOutcome::Quarantined);
        }
        let Some(request) = record.request.as_ref() else {
            self.quarantine_line(DecodeFailureCode::MissingReplayMetadata)
                .await;
            return Ok(LineOutcome::Quarantined);
        };
        if validate_request_counts(request, &context.session_id).is_err() {
            self.quarantine_line(DecodeFailureCode::Schema).await;
            return Ok(LineOutcome::Quarantined);
        }
        // Generation one refuses the finite virtual-hash fallback outright: its
        // inputs are whole-capture facts a bounded decoder cannot reproduce, and
        // a best-effort version would mint different hashes after a restart.
        let Some(replay) = request.replay.as_ref() else {
            self.quarantine_line(DecodeFailureCode::MissingReplayMetadata)
                .await;
            return Ok(LineOutcome::Quarantined);
        };

        // Authority binding precedes every checked bound and every allocation,
        // so a drifted record can never be masked by an unrelated quarantine.
        let block_size = self.bind_or_verify_block_size(replay).await?;

        if replay.hashes.len() > self.config.config.max_block_hashes_per_record {
            self.quarantine_line(DecodeFailureCode::OversizedRecord)
                .await;
            return Ok(LineOutcome::Quarantined);
        }
        let Ok(input_length) = usize::try_from(replay.input_length) else {
            self.quarantine_line(DecodeFailureCode::InvalidReplayGeometry)
                .await;
            return Ok(LineOutcome::Quarantined);
        };
        if input_length as u64 > self.config.config.max_input_length {
            self.quarantine_line(DecodeFailureCode::OversizedRecord)
                .await;
            return Ok(LineOutcome::Quarantined);
        }
        let Ok(geometry) = normalize_replay_geometry(input_length, block_size, &replay.hashes)
        else {
            self.quarantine_line(DecodeFailureCode::InvalidReplayGeometry)
                .await;
            return Ok(LineOutcome::Quarantined);
        };
        self.emit_deferred_request(record, context, request, geometry, block_size, permit)
            .map(|fragment| LineOutcome::Fragment(Box::new(fragment)))
    }

    async fn decode_tool_event(
        &mut self,
        record: &TraceRecord,
        permit: super::super::budget::BudgetLease,
    ) -> Result<LineOutcome, StreamFormatError> {
        let (Some(context), Some(tool)) = (record.context.as_ref(), record.tool.as_ref()) else {
            self.quarantine_line(DecodeFailureCode::Schema).await;
            return Ok(LineOutcome::Quarantined);
        };
        if validate_session_id(&context.session_id).is_err() {
            self.quarantine_line(DecodeFailureCode::Schema).await;
            return Ok(LineOutcome::Quarantined);
        }
        let kind = event_type_tag(record.event_type);
        let Ok(payload) = serde_json::to_vec(&serde_json::json!({
            "event_type": kind,
            "tool_call_id": tool.tool_call_id,
            "status": tool.status,
        })) else {
            self.quarantine_line(DecodeFailureCode::Schema).await;
            return Ok(LineOutcome::Quarantined);
        };
        let mut producer_key = Vec::new();
        producer_key.extend_from_slice(context.session_id.as_bytes());
        producer_key.push(0);
        producer_key.extend_from_slice(tool.tool_call_id.as_bytes());
        producer_key.push(0);
        producer_key.extend_from_slice(kind.as_bytes());
        producer_key.push(0);
        producer_key.extend_from_slice(&record.event_time_ms.to_le_bytes());

        let mutation = SessionMutationV1::AgentEvent(AgentEventFragment {
            event_kind: STREAMING_DYNAMO_TOOL_EVENT_KIND.to_owned(),
            payload,
            // The line ordinal is the only stable ordinal a bounded decoder can
            // prove; per-session ordering is the coordinator's whole-session fact.
            event_ordinal: self.cursor.line_ordinal,
        });
        let retained = agent_event_retained_bytes(&mutation);
        self.finish_fragment(
            record,
            &producer_key,
            &context.session_id,
            mutation,
            retained,
            permit,
        )
        .map(|fragment| LineOutcome::Fragment(Box::new(fragment)))
    }

    /// Bind the first executable positive block size, or refuse later drift.
    async fn bind_or_verify_block_size(
        &mut self,
        replay: &ReplayMetrics,
    ) -> Result<usize, StreamFormatError> {
        let Ok(block_size) = u32::try_from(replay.block_size) else {
            return Err(self.report_authority_drift().await);
        };
        // Positivity is already guaranteed by the shared grammar; re-checking at
        // the boundary means a future schema relaxation cannot silently unbind
        // the invariant.
        if block_size == 0 || block_size > self.config.config.max_block_size {
            return Err(self.report_authority_drift().await);
        }
        let current = self.cursor.authority;
        match current.tag {
            AuthorityTag::Bound if current.block_size != block_size => {
                Err(self.report_authority_drift().await)
            }
            AuthorityTag::Bound => Ok(replay.block_size),
            AuthorityTag::Unbound => {
                let bound = AuthorityState {
                    tag: AuthorityTag::Bound,
                    block_size,
                    digest: bound_authority_digest(
                        &self.config.profile_digest,
                        block_size,
                        &self.config.canonical_config_bytes,
                    ),
                };
                self.cursor.authority = bound;
                self.authority.set(bound);
                Ok(replay.block_size)
            }
        }
    }

    /// Report a frozen-semantic drift without membership loss, and return the error.
    ///
    /// Never record scope: drift invalidates the run's frozen semantics, and
    /// quarantining it would silently admit a mixed-semantic stream. Partition
    /// scope is unavailable — `reliability.rs` pairs partition scope with a
    /// source-stage failure only — so the fact is reported at run scope, which
    /// loses no membership and leaves the terminal classification to the host.
    async fn report_authority_drift(&mut self) -> StreamFormatError {
        let failure = StreamFormatError::decode(DecodeFailureCode::SynthesisAuthorityMismatch);
        if let Ok(issue) = OrdinaryStreamingIssue::run_diagnostic(
            self.run,
            StreamingIssueClass::Permanent,
            self.cursor.authority.digest,
            0,
            self.cursor.authority.digest,
            OrdinaryStreamingFailure::Format(failure),
        ) {
            let _ = self.reporter.report(issue).await;
        }
        failure
    }

    fn emit_deferred_request(
        &mut self,
        record: &TraceRecord,
        context: &AgentContext,
        request: &RequestMetrics,
        geometry: NormalizedReplayGeometry,
        block_size: usize,
        permit: super::super::budget::BudgetLease,
    ) -> Result<StreamingSessionFragment, StreamFormatError> {
        let hashes = geometry
            .complete_block_hashes
            .iter()
            .map(|hash| RecordedBlockHash::new(*hash))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::InvalidReplayGeometry))?;
        let block_size = u32::try_from(block_size)
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::InvalidReplayGeometry))?;

        let fragment = DeferredRecordedRequestFragment {
            producer_session_id: context.session_id.clone(),
            // The parent *identifier*, not a resolved parent record: resolving it
            // needs the whole tree, which only the coordinator can close.
            parent_producer_session_id: producer_parent_id(context).map(str::to_owned),
            producer_request_id: request.request_id.clone(),
            replay: DeferredReplayGeometry {
                block_size,
                input_length: geometry.input_length as u64,
                complete_block_hashes: hashes,
                tail_tokens: geometry.tail_tokens as u64,
            },
            recorded: RecordedRequestFacts {
                model: request.model.clone(),
                event_time_unix_ms: record.event_time_ms,
                request_received_ms: request.request_received_ms,
                total_time_ms: request.total_time_ms,
                ttft_ms: request.ttft_ms,
                input_tokens: request.input_tokens,
                output_tokens: request.output_tokens,
                cached_tokens: request.cached_tokens,
            },
        };
        let retained = deferred_retained_bytes(&fragment);
        let mut producer_key = Vec::new();
        producer_key.extend_from_slice(context.session_id.as_bytes());
        producer_key.push(0);
        producer_key.extend_from_slice(request.request_id.as_bytes());
        self.finish_fragment(
            record,
            &producer_key,
            &context.session_id,
            SessionMutationV1::DeferredRecordedRequest(fragment),
            retained,
            permit,
        )
    }

    fn finish_fragment(
        &mut self,
        record: &TraceRecord,
        producer_key: &[u8],
        session_id: &str,
        mutation: SessionMutationV1,
        retained_bytes: usize,
        permit: super::super::budget::BudgetLease,
    ) -> Result<StreamingSessionFragment, StreamFormatError> {
        let record_id = stable_record_id_from_key(self.stream_identity.as_bytes(), producer_key);
        let session_key =
            stable_session_key(self.stream_identity.as_bytes(), session_id.as_bytes());
        let event_time = record
            .event_time_ms
            .checked_mul(1_000_000)
            .and_then(|nanoseconds| EventTimeUtc::new(nanoseconds).ok())
            .ok_or(StreamFormatError::decode(DecodeFailureCode::Schema))?;

        let mut lease = SessionFragmentLease::try_from(permit)
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        if retained_bytes < lease.charged_bytes() {
            lease
                .shrink_bytes_to(retained_bytes)
                .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        }

        let latest = self
            .latest_event_time
            .get()
            .map_or(event_time, |seen| seen.max(event_time));
        self.latest_event_time.set(Some(latest));

        Ok(StreamingSessionFragment {
            record_id,
            session_key,
            source_position: self.position,
            source_partition: self.identity,
            event_time: Some(event_time),
            stable_tie_break: StableOrderKey::from_bytes(*record_id.as_bytes()),
            // Turn order inside a producer session is a whole-session fact the
            // coordinator owns; a bounded decoder that named one would produce a
            // different answer after a restart or a partition split.
            predecessors: SmallVec::new(),
            mutation,
            provenance: UnitProvenance {
                source_partition: self.identity,
                source_position: self.position,
                format_semantic_digest: STREAMING_DYNAMO_SEMANTIC_DIGEST,
            },
            lease,
        })
    }

    /// Exclude one invalid line and continue at the next proven boundary.
    ///
    /// The identity is derived from the exact partition coordinate so a line
    /// that failed before its producer key could be parsed still has a stable,
    /// reachable receipt. The adapter never selects the disposition.
    async fn quarantine_line(&mut self, code: DecodeFailureCode) {
        let record_id = physical_record_id(
            self.stream_identity.as_bytes(),
            &self.identity,
            &self.cursor.line_ordinal.to_le_bytes(),
            STREAMING_DYNAMO_SEMANTIC_DIGEST.as_bytes(),
        );
        let Ok(issue) = OrdinaryStreamingIssue::record(
            self.run,
            self.input_domain.clone(),
            record_id,
            StreamingIssueClass::Permanent,
            self.cursor.authority.digest,
            self.position,
            0,
            self.cursor.authority.digest,
            OrdinaryStreamingFailure::Format(StreamFormatError::decode(code)),
        ) else {
            return;
        };
        let _ = self.reporter.report(issue).await;
    }
}

/// Resolve the producer's declared parent session exactly as the finite
/// compiler does.
///
/// `parent_trajectory_id`, when present and non-empty, is *authoritative*: if it
/// names the session itself, the session is a root and `parent_session_id` is
/// deliberately not consulted. Only an absent or empty trajectory id falls back.
/// Collapsing this into `.or_else(..)` would resurrect a stale
/// `parent_session_id` on a self-trajectory root.
fn producer_parent_id(context: &AgentContext) -> Option<&str> {
    match context
        .parent_trajectory_id
        .as_deref()
        .filter(|parent| !parent.is_empty())
    {
        Some(parent) => (parent != context.session_id).then_some(parent),
        None => context
            .parent_session_id
            .as_deref()
            .filter(|parent| !parent.is_empty() && *parent != context.session_id),
    }
}

const fn event_type_tag(event_type: EventType) -> &'static str {
    match event_type {
        EventType::RequestEnd => "request_end",
        EventType::ToolStart => "tool_start",
        EventType::ToolEnd => "tool_end",
        EventType::ToolError => "tool_error",
    }
}

fn deferred_retained_bytes(fragment: &DeferredRecordedRequestFragment) -> usize {
    fragment
        .producer_session_id
        .capacity()
        .saturating_add(
            fragment
                .parent_producer_session_id
                .as_ref()
                .map_or(0, String::capacity),
        )
        .saturating_add(fragment.producer_request_id.capacity())
        .saturating_add(fragment.recorded.model.as_ref().map_or(0, String::capacity))
        .saturating_add(
            fragment
                .replay
                .complete_block_hashes
                .capacity()
                .saturating_mul(std::mem::size_of::<RecordedBlockHash>()),
        )
        .saturating_add(std::mem::size_of::<DeferredRecordedRequestFragment>())
}

fn agent_event_retained_bytes(mutation: &SessionMutationV1) -> usize {
    match mutation {
        SessionMutationV1::AgentEvent(event) => event
            .event_kind
            .capacity()
            .saturating_add(event.payload.capacity())
            .saturating_add(std::mem::size_of::<AgentEventFragment>()),
        _ => std::mem::size_of::<SessionMutationV1>(),
    }
}

fn trim_ascii_line(line: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = line.len();
    while start < end && line[start].is_ascii_whitespace() {
        start += 1;
    }
    while end > start && line[end - 1].is_ascii_whitespace() {
        end -= 1;
    }
    &line[start..end]
}

/// Project an acquisition failure into the format's closed decode vocabulary.
///
/// `StreamFormatError` has no source-stage variant, so a failed read surfaces as
/// a schema-stage refusal of the immutable input rather than as a silent EOF.
const fn map_source_error(_error: StreamSourceError) -> StreamFormatError {
    StreamFormatError::decode(DecodeFailureCode::Schema)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cursor_round_trips_and_refuses_contradictory_authority() {
        let bound = DynamoCursor {
            byte_offset: 41,
            line_ordinal: 3,
            fragments_emitted: 2,
            quarantined: 1,
            authority: AuthorityState {
                tag: AuthorityTag::Bound,
                block_size: 16,
                digest: ContentDigest::from_bytes([0x9a; 32]),
            },
        };
        let encoded = bound.encode();
        assert_eq!(encoded.len(), CURSOR_BYTES);
        assert_eq!(DynamoCursor::decode(&encoded).expect("round trip"), bound);

        // Bound with a zero block size cannot be smuggled through a checkpoint.
        let mut broken = encoded;
        broken[36..40].copy_from_slice(&0_u32.to_le_bytes());
        assert!(DynamoCursor::decode(&broken).is_err());

        // Reserved padding must stay zero.
        let mut padded = encoded;
        padded[34] = 1;
        assert!(DynamoCursor::decode(&padded).is_err());
    }

    #[test]
    fn authoritative_self_trajectory_does_not_use_fallback_parent() {
        let context = AgentContext {
            session_id: "s".to_owned(),
            parent_session_id: Some("stale".to_owned()),
            parent_trajectory_id: Some("s".to_owned()),
        };
        assert_eq!(producer_parent_id(&context), None);

        let context = AgentContext {
            session_id: "s".to_owned(),
            parent_session_id: Some("p".to_owned()),
            parent_trajectory_id: None,
        };
        assert_eq!(producer_parent_id(&context), Some("p"));
    }

    #[test]
    fn bound_authority_digest_binds_profile_block_size_and_config() {
        let profile = ContentDigest::from_bytes([0x11; 32]);
        let config = b"config".as_slice();
        assert_ne!(
            bound_authority_digest(&profile, 16, config),
            bound_authority_digest(&profile, 32, config)
        );
        assert_ne!(
            bound_authority_digest(&profile, 16, config),
            bound_authority_digest(&ContentDigest::from_bytes([0x12; 32]), 16, config)
        );
    }
}
