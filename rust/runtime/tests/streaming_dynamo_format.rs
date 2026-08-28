// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executable contract of the strict streaming Dynamo/NVCF request-trace format.
//!
//! Every fixture is one immutable in-memory JSONL object acquired through the
//! real `AcquiredPartition` sequential seam, so contiguity, chunk bounds, and
//! truncation detection are exercised rather than stubbed.

#[allow(dead_code)]
#[path = "support/streaming_format_conformance.rs"]
mod streaming_format_conformance;

use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::rc::Rc;

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        CheckpointBarrier, CheckpointError, CheckpointParticipantId, CommittedParticipantReceipt,
        CommittedParticipantState, ParticipantInitialization, PreparedParticipantState,
        StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::{StableStreamingFailure, StreamFormatError, StreamSourceError},
    format::{
        DecodeBatchBudget, DecodeStep, DecoderCheckpoint, DecoderResumeState, FormatEvent,
        FormatEventSink, StreamingDatasetFormatFactory, StreamingFormatPrepareContext,
    },
    formats::streaming_dynamo::{
        STREAMING_DYNAMO_DESCRIPTOR, STREAMING_DYNAMO_FORMAT_ID, StreamingDynamoFormatFactory,
    },
    identity::{
        ContentDigest, DuplicateDisposition, ImmutableObjectIdentity, LogicalRecordReceipt,
        LogicalReplayRunId, StableRecordId, classify_logical_duplicate,
    },
    reliability::{
        OrdinaryStreamingIssue, StreamingIssueClass, StreamingIssueReportError,
        StreamingIssueReportStatus, StreamingIssueReporter, StreamingIssueReporterEndpoint,
        StreamingIssueReporterHandle, StreamingIssueScope, StreamingIssueSummary,
        StreamingReliabilityError,
    },
    source::{
        AcquiredPartition, AcquisitionBudget, BudgetedSourceChunk, PartitionAccessKind,
        SequentialSourceChunk, SourceFrontier, SourceSeal, StreamingResumeGranularity,
        StreamingSequentialReader, StreamingSourceDescriptor, StreamingSourceMode,
        StreamingSourceOrdering, StreamingSourcePlacement, StreamingSourceRetention,
    },
    unit::{SessionMutationV1, SourcePosition, StreamingSessionFragment},
};
use async_trait::async_trait;
use bytes::Bytes;
use serde_json::value::RawValue;

use streaming_format_conformance::{FormatConformanceCases, assert_format_conformance};

static LOCAL_LIKE_SOURCE_DESCRIPTOR: StreamingSourceDescriptor = StreamingSourceDescriptor {
    id: "test_in_memory_object",
    description: "In-memory immutable JSONL object",
    modes: &[StreamingSourceMode::Finite],
    access: &[PartitionAccessKind::Sequential],
    ordering: StreamingSourceOrdering::Partition,
    resume: &[StreamingResumeGranularity::Byte],
    has_event_time: true,
    has_stable_record_ids: true,
    retention: StreamingSourceRetention::BoundedMemory,
    placement: StreamingSourcePlacement::ControllerOnly,
    supports_virtual_clock: true,
};

const PARTITION_IDENTITY: ImmutableObjectIdentity = ImmutableObjectIdentity::from_bytes([0x21; 32]);
const STREAM_DIGEST: ContentDigest = ContentDigest::from_bytes([0x71; 32]);
const PROFILE_DIGEST: ContentDigest = ContentDigest::from_bytes([0xa5; 32]);

fn run_identity() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x11; 32]))
}

fn raw(value: serde_json::Value) -> Box<RawValue> {
    RawValue::from_string(value.to_string()).expect("valid raw configuration")
}

fn authored_config() -> serde_json::Value {
    serde_json::json!({
        "max_record_bytes": 4096,
        "max_chunk_bytes": 64,
        "max_block_hashes_per_record": 64,
        "max_block_size": 1024,
        "max_input_length": 1_048_576,
    })
}

// ---------------------------------------------------------------------------
// Immutable in-memory object acquired through the real sequential seam
// ---------------------------------------------------------------------------

struct SliceReader {
    bytes: Rc<Vec<u8>>,
    next: usize,
}

#[async_trait(?Send)]
impl StreamingSequentialReader for SliceReader {
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<Option<SequentialSourceChunk>, StreamSourceError> {
        if self.next >= self.bytes.len() {
            return Ok(None);
        }
        let end = (self.next + max_bytes.get()).min(self.bytes.len());
        let slice = self.bytes[self.next..end].to_vec();
        let length = slice.len();
        let lease = budget.acquire_memory(1, length).await?;
        let chunk = BudgetedSourceChunk::new(Bytes::from(slice), lease)?;
        self.next = end;
        Ok(Some(SequentialSourceChunk::new(
            chunk,
            self.next as u64,
            ContentDigest::from_bytes([0x41; 32]),
        )))
    }
}

async fn acquire(
    bytes: &Rc<Vec<u8>>,
    resume_offset: u64,
    budget: &AcquisitionBudget,
) -> AcquiredPartition {
    let start = usize::try_from(resume_offset).expect("fixture offsets fit usize");
    assert!(
        start <= bytes.len(),
        "fixture resume offset is inside the object"
    );
    let authority = budget
        .acquire_memory(1, 0)
        .await
        .expect("handle capacity is available");
    AcquiredPartition::sequential(
        SourcePosition::new(1),
        PARTITION_IDENTITY,
        Some(bytes.len() as u64),
        resume_offset,
        Box::new(SliceReader {
            bytes: Rc::clone(bytes),
            next: start,
        }),
        authority,
    )
    .expect("in-memory sequential acquisition")
}

fn acquisition_budget() -> AcquisitionBudget {
    AcquisitionBudget::new(
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 16,
            max_bytes: 1 << 20,
        })
        .expect("valid memory limits"),
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 4,
            max_bytes: 1 << 20,
        })
        .expect("valid disk limits"),
    )
}

// ---------------------------------------------------------------------------
// Test-local reliability reporter that retains scope and code facts
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Eq, PartialEq)]
struct IssueFact {
    scope: &'static str,
    code: String,
    class: StreamingIssueClass,
}

#[derive(Debug, Default)]
struct IssueLog {
    facts: RefCell<Vec<IssueFact>>,
    is_closed: Cell<bool>,
}

#[derive(Debug)]
struct RecordingEndpoint {
    log: Rc<IssueLog>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for RecordingEndpoint {
    async fn report(
        &self,
        issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        if self.log.is_closed.get() {
            return Err(StreamingIssueReportError::Closed);
        }
        let scope = match issue.scope() {
            StreamingIssueScope::Run => "run",
            StreamingIssueScope::Partition { .. } => "partition",
            StreamingIssueScope::Record { .. } => "record",
            StreamingIssueScope::Session { .. } => "session",
            StreamingIssueScope::Action { .. } => "action",
            StreamingIssueScope::Export { .. } => "export",
            StreamingIssueScope::CheckpointAttempt { .. } => "checkpoint_attempt",
        };
        self.log.facts.borrow_mut().push(IssueFact {
            scope,
            code: issue.code().as_str().to_owned(),
            class: issue.class(),
        });
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

struct RecordingReporter {
    log: Rc<IssueLog>,
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    initialization: ParticipantInitialization,
}

impl RecordingReporter {
    fn new(run: StreamRunIdentity) -> Self {
        Self {
            log: Rc::new(IssueLog::default()),
            participant_id: CheckpointParticipantId::new("test_issue_reporter"),
            run,
            initialization: ParticipantInitialization::default(),
        }
    }

    fn facts(&self) -> Vec<IssueFact> {
        self.log.facts.borrow().clone()
    }
}

impl Drop for RecordingReporter {
    fn drop(&mut self) {
        self.log.is_closed.set(true);
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for RecordingReporter {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        _barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        Err(CheckpointError::ObjectVerification)
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()
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

#[async_trait(?Send)]
impl StreamingIssueReporter for RecordingReporter {
    fn handle(&self) -> StreamingIssueReporterHandle {
        StreamingIssueReporterHandle::new(RecordingEndpoint {
            log: Rc::clone(&self.log),
        })
    }

    fn summary(&self) -> Result<StreamingIssueSummary, StreamingReliabilityError> {
        Ok(StreamingIssueSummary {
            total: self.log.facts.borrow().len() as u64,
            by_scope: BTreeMap::new(),
            by_class: BTreeMap::new(),
            by_disposition: BTreeMap::new(),
            is_admission_fenced: false,
        })
    }
}

#[derive(Default)]
struct CapturingSink {
    frontiers: usize,
    closes: usize,
}

#[async_trait(?Send)]
impl FormatEventSink for CapturingSink {
    async fn send(&mut self, event: FormatEvent) -> Result<(), StreamFormatError> {
        match event {
            FormatEvent::SessionFrontier(_) => self.frontiers += 1,
            FormatEvent::Fragment(fragment) => {
                if matches!(fragment.mutation, SessionMutationV1::SessionClose(_)) {
                    self.closes += 1;
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Fixture drive
// ---------------------------------------------------------------------------

/// Everything one decode of one immutable object produced.
struct Decoded {
    fragments: Vec<StreamingSessionFragment>,
    fragment_count: u64,
    cursor: Vec<u8>,
    error: Option<StreamFormatError>,
    facts: Vec<IssueFact>,
    /// Kept alive so retained fragment leases stay valid for assertions.
    _fragment_budget: StreamingResourceBudget,
}

fn fragment_budget(max_items: usize) -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items,
        max_bytes: 1 << 20,
    })
    .expect("valid fragment budget")
}

async fn decode_object(object: &str, authored: serde_json::Value) -> Decoded {
    decode_object_from(object, authored, None).await
}

async fn decode_object_from(
    object: &str,
    authored: serde_json::Value,
    resume: Option<Vec<u8>>,
) -> Decoded {
    let reporter = RecordingReporter::new(run_identity());
    let budget = fragment_budget(64);
    let factory = StreamingDynamoFormatFactory::new(PROFILE_DIGEST);
    let validated = factory
        .validate(&raw(authored), &LOCAL_LIKE_SOURCE_DESCRIPTOR)
        .expect("authored configuration validates");
    let acquisition = acquisition_budget();
    let context = StreamingFormatPrepareContext {
        run: run_identity(),
        stream_semantic_digest: STREAM_DIGEST,
        fragment_budget: budget.clone(),
        issue_reporter: reporter.handle(),
        acquisition_budget: acquisition.clone(),
    };
    let mut format = factory.prepare(validated, &context).expect("prepare");
    format.initialize(None).await.expect("fresh initialization");

    let bytes = Rc::new(object.as_bytes().to_vec());
    let resume_offset = resume.as_ref().map_or(0, |state| {
        u64::from_le_bytes(state[0..8].try_into().expect("cursor offset"))
    });
    let partition = acquire(&bytes, resume_offset, &acquisition).await;

    let checkpoint = resume.map(|state| {
        let lease = budget
            .try_acquire(1, state.len())
            .expect("cursor charge fits");
        DecoderCheckpoint {
            partition: PARTITION_IDENTITY,
            format_semantic_digest: STREAMING_DYNAMO_DESCRIPTOR.semantic_digest,
            state: DecoderResumeState::new(Bytes::from(state), lease).expect("exact cursor charge"),
        }
    });

    let mut decoder = match format.begin_partition(partition, checkpoint).await {
        Ok(decoder) => decoder,
        Err(error) => {
            let facts = reporter.facts();
            return Decoded {
                fragments: Vec::new(),
                fragment_count: 0,
                cursor: Vec::new(),
                error: Some(error),
                facts,
                _fragment_budget: budget,
            };
        }
    };

    let decode_budget = DecodeBatchBudget {
        max_fragments: 32,
        max_bytes: 1 << 20,
    };
    let mut fragments = Vec::new();
    let mut cursor = Vec::new();
    let mut fragment_count = 0;
    let mut error = None;
    loop {
        match decoder.next_batch(decode_budget).await {
            Ok(DecodeStep::Batch(batch)) => {
                cursor = batch.resume_after.as_bytes().to_vec();
                fragments.extend(batch.fragments);
            }
            Ok(DecodeStep::End(receipt)) => {
                cursor = receipt.final_state.as_bytes().to_vec();
                fragment_count = receipt.fragment_count;
                break;
            }
            Err(failure) => {
                error = Some(failure);
                break;
            }
        }
    }
    let facts = reporter.facts();
    Decoded {
        fragments,
        fragment_count,
        cursor,
        error,
        facts,
        _fragment_budget: budget,
    }
}

fn request_end_line(
    session: &str,
    request_id: &str,
    time_ms: i64,
    block_size: usize,
    input_length: i64,
    hashes: &[i128],
) -> String {
    let hashes = hashes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        r#"{{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":{time_ms},"agent_context":{{"session_id":"{session}"}},"request":{{"request_id":"{request_id}","model":"m","replay":{{"trace_block_size":{block_size},"input_length":{input_length},"input_sequence_hashes":[{hashes}]}}}}}}"#
    )
}

fn deferred(
    fragment: &StreamingSessionFragment,
) -> &aiperf_runtime::streaming::unit::DeferredRecordedRequestFragment {
    match &fragment.mutation {
        SessionMutationV1::DeferredRecordedRequest(deferred) => deferred,
        other => panic!("expected a deferred recorded request, got {other:?}"),
    }
}

fn mutation_digest(fragment: &StreamingSessionFragment) -> ContentDigest {
    let bytes = serde_json::to_vec(&fragment.mutation).expect("canonical mutation encoding");
    ContentDigest::from_bytes(*blake3::hash(&bytes).as_bytes())
}

fn receipt(fragment: &StreamingSessionFragment) -> LogicalRecordReceipt {
    LogicalRecordReceipt {
        record_id: fragment.record_id,
        content_digest: mutation_digest(fragment),
        provenance: fragment.provenance.clone(),
    }
}

// ---------------------------------------------------------------------------
// Named contract tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn parent_in_later_object_does_not_create_an_early_root() {
    let child = format!(
        "{}\n",
        r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":20,"agent_context":{"session_id":"child","parent_session_id":"parent"},"request":{"request_id":"r-child","replay":{"trace_block_size":16,"input_length":16,"input_sequence_hashes":[7]}}}"#
    );
    let parent = format!(
        "{}\n",
        request_end_line("parent", "r-parent", 10, 16, 16, &[5])
    );

    let first = decode_object(&child, authored_config()).await;
    assert!(first.error.is_none());
    assert_eq!(first.fragments.len(), 1);
    assert_eq!(
        deferred(&first.fragments[0])
            .parent_producer_session_id
            .as_deref(),
        Some("parent"),
        "a child names its producer parent even when the parent record is elsewhere"
    );

    let second = decode_object(&parent, authored_config()).await;
    assert!(second.error.is_none());
    assert_eq!(
        deferred(&second.fragments[0]).parent_producer_session_id,
        None,
        "a root record declares no parent and no root is inferred at object EOF"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn first_executable_request_binds_block_size_and_later_drift_fails() {
    let object = format!(
        "{}\n{}\n",
        request_end_line("s", "r-1", 10, 16, 16, &[1]),
        request_end_line("s", "r-2", 20, 32, 32, &[2]),
    );
    let decoded = decode_object(&object, authored_config()).await;
    let error = decoded.error.expect("a drifted block size is terminal");
    assert_eq!(error.code(), "synthesis_authority_mismatch");
}

#[tokio::test(flavor = "current_thread")]
async fn bound_profile_drift_cannot_be_quarantined() {
    let object = format!(
        "{}\n{}\n",
        request_end_line("s", "r-1", 10, 16, 16, &[1]),
        request_end_line("s", "r-2", 20, 32, 32, &[2]),
    );
    let decoded = decode_object(&object, authored_config()).await;
    assert!(decoded.error.is_some());
    assert_eq!(
        decoded.facts,
        vec![IssueFact {
            scope: "run",
            code: "synthesis_authority_mismatch".to_owned(),
            class: StreamingIssueClass::Permanent,
        }],
        "frozen-semantic drift is never a record quarantine"
    );
    assert!(
        decoded.fragments.is_empty(),
        "the offending batch contributes no fragment"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn sink_envelopes_markers_and_coercions_decode_like_the_finite_parser() {
    // The oversized hash is written as raw text: a `json!`-built value would have
    // already lost digits through `f64`.
    let object = concat!(
        r#"{"verification":"trace-s3-uploader"}"#,
        "\n",
        r#"{"timestamp":1,"event":{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":"1000.0","event_source":"dynamo","agent_context":{"session_id":"s","future":"ok"},"request":{"request_id":"r","output_tokens":true,"replay":{"trace_block_size":"16.0","input_length":"16.0","input_sequence_hashes":[184467440737095516170],"future":"ok"}}}}"#,
        "\n",
    );
    let decoded = decode_object(object, authored_config()).await;
    assert!(decoded.error.is_none());
    assert_eq!(decoded.fragments.len(), 1);
    assert!(
        decoded.facts.is_empty(),
        "a marker line is not a fault and mints no receipt"
    );
    let request = deferred(&decoded.fragments[0]);
    assert_eq!(request.recorded.event_time_unix_ms, 1000);
    assert_eq!(request.recorded.output_tokens, Some(1));
    assert_eq!(request.replay.block_size, 16);
    assert_eq!(
        request.replay.complete_block_hashes[0].get(),
        "184467440737095516170".parse::<i128>().expect("i128 hash"),
        "the arbitrary-precision hash domain survives the decoder boundary"
    );
    // The decimal string form is what downstream content synthesis seeds from.
    let encoded = serde_json::to_string(&decoded.fragments[0].mutation).expect("encode");
    assert!(encoded.contains("\"184467440737095516170\""));
}

#[tokio::test(flavor = "current_thread")]
async fn missing_replay_metadata_quarantines_without_virtual_hashes() {
    let object = concat!(
        r#"{"schema":"dynamo.request.trace.v1","event_type":"request_end","event_time_unix_ms":10,"agent_context":{"session_id":"s"},"request":{"request_id":"r","input_tokens":64}}"#,
        "\n",
    );
    let decoded = decode_object(object, authored_config()).await;
    assert!(decoded.error.is_none());
    assert!(decoded.fragments.is_empty());
    assert_eq!(decoded.fragment_count, 0);
    assert_eq!(
        decoded.facts,
        vec![IssueFact {
            scope: "record",
            code: "missing_replay_metadata".to_owned(),
            class: StreamingIssueClass::Permanent,
        }],
        "generation one never invokes the finite virtual-hash allocator"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn replay_geometry_boundaries_zero_tiny_full_and_full_plus_partial() {
    let object = format!(
        "{}\n{}\n{}\n{}\n",
        request_end_line("s", "zero", 10, 16, 0, &[]),
        request_end_line("s", "tiny", 20, 16, 5, &[1]),
        request_end_line("s", "full", 30, 16, 32, &[1, 2]),
        request_end_line("s", "partial", 40, 16, 40, &[1, 2, 3]),
    );
    let decoded = decode_object(&object, authored_config()).await;
    assert!(decoded.error.is_none());
    let shapes: Vec<(usize, u64, u64)> = decoded
        .fragments
        .iter()
        .map(|fragment| {
            let replay = &deferred(fragment).replay;
            (
                replay.complete_block_hashes.len(),
                replay.tail_tokens,
                replay.input_length,
            )
        })
        .collect();
    assert_eq!(shapes, vec![(0, 0, 0), (0, 5, 5), (2, 0, 32), (2, 8, 40)]);
}

#[tokio::test(flavor = "current_thread")]
async fn nonzero_input_with_empty_hashes_is_refused_as_invalid_geometry() {
    let object = format!("{}\n", request_end_line("s", "r", 10, 16, 8, &[]));
    let decoded = decode_object(&object, authored_config()).await;
    assert!(decoded.fragments.is_empty());
    assert_eq!(
        decoded.facts,
        vec![IssueFact {
            scope: "record",
            code: "invalid_replay_geometry".to_owned(),
            class: StreamingIssueClass::Permanent,
        }]
    );
}

#[tokio::test(flavor = "current_thread")]
async fn repeated_request_id_is_idempotent_and_conflicting_content_conflicts() {
    let line = request_end_line("s", "r", 10, 16, 16, &[1]);
    let same = decode_object(&format!("{line}\n{line}\n"), authored_config()).await;
    assert_eq!(same.fragments.len(), 2);
    assert_eq!(same.fragments[0].record_id, same.fragments[1].record_id);
    assert_eq!(
        classify_logical_duplicate(&receipt(&same.fragments[0]), &receipt(&same.fragments[1]))
            .expect("identical content is idempotent"),
        DuplicateDisposition::Identical
    );

    let divergent = request_end_line("s", "r", 10, 16, 32, &[1, 2]);
    let mixed = decode_object(&format!("{line}\n{divergent}\n"), authored_config()).await;
    assert_eq!(mixed.fragments[0].record_id, mixed.fragments[1].record_id);
    let error =
        classify_logical_duplicate(&receipt(&mixed.fragments[0]), &receipt(&mixed.fragments[1]))
            .expect_err("divergent content under one logical identity conflicts");
    assert_eq!(error.code(), "logical_identity_conflict");
}

#[tokio::test(flavor = "current_thread")]
async fn record_identity_is_stable_across_unbound_to_bound_authority() {
    let first = format!("{}\n", request_end_line("s", "r-1", 10, 16, 16, &[1]));
    let both = format!(
        "{}{}\n",
        first,
        request_end_line("s", "r-2", 20, 16, 16, &[2])
    );
    let alone = decode_object(&first, authored_config()).await;
    let together = decode_object(&both, authored_config()).await;
    assert_eq!(
        alone.fragments[0].record_id, together.fragments[0].record_id,
        "binding the authority does not change a record identity"
    );
    assert_eq!(
        alone.fragments[0].stable_tie_break,
        together.fragments[0].stable_tie_break
    );
}

#[tokio::test(flavor = "current_thread")]
async fn cursor_restore_before_and_after_binding_resumes_without_duplicates() {
    let marker = r#"{"verification":"trace-s3-uploader"}"#;
    let first = request_end_line("s", "r-1", 10, 16, 16, &[1]);
    let second = request_end_line("s", "r-2", 20, 16, 16, &[2]);
    let object = format!("{marker}\n{first}\n{second}\n");

    // A prefix decode leaves an unbound cursor at a proven line boundary.
    let prefix = decode_object(&format!("{marker}\n"), authored_config()).await;
    assert!(prefix.fragments.is_empty());
    assert_eq!(prefix.cursor[32], 0, "no executable record has bound yet");

    let resumed = decode_object_from(&object, authored_config(), Some(prefix.cursor.clone())).await;
    assert!(resumed.error.is_none());
    let ids: Vec<StableRecordId> = resumed
        .fragments
        .iter()
        .map(|fragment| fragment.record_id)
        .collect();
    let whole = decode_object(&object, authored_config()).await;
    let whole_ids: Vec<StableRecordId> = whole
        .fragments
        .iter()
        .map(|fragment| fragment.record_id)
        .collect();
    assert_eq!(
        ids, whole_ids,
        "resuming replays no duplicate and skips no record"
    );
    assert_eq!(
        resumed.cursor[32], 1,
        "the resumed decode bound the authority"
    );

    // Resuming again from the bound cursor keeps the same authority bytes.
    let again = decode_object_from(&object, authored_config(), Some(resumed.cursor.clone())).await;
    assert!(again.error.is_none());
    assert_eq!(&again.cursor[32..72], &resumed.cursor[32..72]);
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_line_and_hash_count_are_refused_before_allocation() {
    let mut tight = authored_config();
    tight["max_record_bytes"] = serde_json::json!(64);
    tight["max_chunk_bytes"] = serde_json::json!(64);
    let object = format!("{}\n", request_end_line("s", "r", 10, 16, 16, &[1]));
    let decoded = decode_object(&object, tight).await;
    assert_eq!(
        decoded
            .error
            .expect("an oversized line is refused before JSON parsing")
            .code(),
        "oversized_record"
    );

    let mut narrow = authored_config();
    narrow["max_block_hashes_per_record"] = serde_json::json!(1);
    let object = format!("{}\n", request_end_line("s", "r", 10, 16, 32, &[1, 2]));
    let decoded = decode_object(&object, narrow).await;
    assert!(decoded.fragments.is_empty());
    assert_eq!(
        decoded.facts,
        vec![IssueFact {
            scope: "record",
            code: "oversized_record".to_owned(),
            class: StreamingIssueClass::Permanent,
        }]
    );
}

#[test]
fn finite_only_selection_keys_are_refused_at_validation() {
    let factory = StreamingDynamoFormatFactory::new(PROFILE_DIGEST);
    for key in [
        "root_limit",
        "max_context_length",
        "max_osl",
        "idle_gap_cap_seconds",
        "prompt_corpus",
        "content_root_seed",
    ] {
        let mut authored = authored_config();
        authored[key] = serde_json::json!(1);
        assert!(
            factory
                .validate(&raw(authored), &LOCAL_LIKE_SOURCE_DESCRIPTOR)
                .is_err(),
            "finite-only selection key {key} is refused before any effect"
        );
    }
    assert_eq!(factory.descriptor().id, STREAMING_DYNAMO_FORMAT_ID);
}

#[tokio::test(flavor = "current_thread")]
async fn partition_eof_emits_end_without_close_root_or_tree_receipt() {
    let object = format!("{}\n", request_end_line("s", "r", 10, 16, 16, &[1]));
    let reporter = RecordingReporter::new(run_identity());
    let budget = fragment_budget(64);
    let factory = StreamingDynamoFormatFactory::new(PROFILE_DIGEST);
    let validated = factory
        .validate(&raw(authored_config()), &LOCAL_LIKE_SOURCE_DESCRIPTOR)
        .expect("validates");
    let acquisition = acquisition_budget();
    let context = StreamingFormatPrepareContext {
        run: run_identity(),
        stream_semantic_digest: STREAM_DIGEST,
        fragment_budget: budget,
        issue_reporter: reporter.handle(),
        acquisition_budget: acquisition.clone(),
    };
    let mut format = factory.prepare(validated, &context).expect("prepare");
    format.initialize(None).await.expect("initialize");

    let bytes = Rc::new(object.into_bytes());
    let partition = acquire(&bytes, 0, &acquisition).await;
    let mut decoder = format
        .begin_partition(partition, None)
        .await
        .expect("decoder begins");
    let decode_budget = DecodeBatchBudget {
        max_fragments: 8,
        max_bytes: 1 << 20,
    };
    let batch = match decoder.next_batch(decode_budget).await.expect("first pull") {
        DecodeStep::Batch(batch) => batch,
        DecodeStep::End(_) => panic!("a nonempty object yields a batch first"),
    };
    drop(batch);
    match decoder
        .next_batch(decode_budget)
        .await
        .expect("second pull")
    {
        DecodeStep::End(receipt) => assert_eq!(receipt.fragment_count, 1),
        DecodeStep::Batch(_) => panic!("the object holds exactly one record"),
    }

    let mut sink = CapturingSink::default();
    format
        .advance_source_frontier(
            SourceFrontier {
                through: SourcePosition::new(1),
            },
            &mut sink,
        )
        .await
        .expect("frontier");
    let seal = format
        .seal(
            SourceSeal {
                final_position: Some(SourcePosition::new(1)),
                digest: ContentDigest::from_bytes([0x31; 32]),
            },
            &mut sink,
        )
        .await
        .expect("seal");
    assert_eq!(seal.partition_count, 1);
    assert_eq!(
        sink.closes, 0,
        "object EOF is never a session close or a producer-tree closure proof"
    );
    assert!(sink.frontiers >= 1);
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_replay_tree_is_quarantined_and_neighbor_tree_continues() {
    let object = format!(
        "{}\n{}\n{}\n",
        request_end_line("tree-a", "a-1", 10, 16, 16, &[1]),
        // Impossible geometry inside tree A: one hash cannot cover 64 tokens.
        request_end_line("tree-a", "a-2", 20, 16, 64, &[9]),
        request_end_line("tree-b", "b-1", 30, 16, 16, &[3]),
    );
    let decoded = decode_object(&object, authored_config()).await;
    assert!(decoded.error.is_none());
    assert_eq!(decoded.fragments.len(), 2);
    let sessions: Vec<&str> = decoded
        .fragments
        .iter()
        .map(|fragment| deferred(fragment).producer_session_id.as_str())
        .collect();
    assert_eq!(sessions, vec!["tree-a", "tree-b"]);
    assert_eq!(
        decoded.facts,
        vec![IssueFact {
            scope: "record",
            code: "invalid_replay_geometry".to_owned(),
            class: StreamingIssueClass::Permanent,
        }],
        "quarantine is record scoped at the exact proven line boundary"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn dynamo_format_satisfies_the_shared_conformance_harness() {
    let object = format!(
        "{}\n{}\n{}\n",
        request_end_line("s", "r-1", 10, 16, 16, &[1]),
        request_end_line("s", "r-2", 20, 16, 16, &[2]),
        request_end_line("s", "r-3", 30, 16, 16, &[3]),
    );
    // Exactly three fragment permits: one batch saturates the output budget, so
    // the decoder's next pull must park before consuming another line.
    let budget = fragment_budget(3);
    let factory = StreamingDynamoFormatFactory::new(PROFILE_DIGEST);
    let acquisition = acquisition_budget();
    let bytes = Rc::new(object.into_bytes());
    let mut partitions = Vec::new();
    for _ in 0..2 {
        partitions.push(acquire(&bytes, 0, &acquisition).await);
    }

    assert_format_conformance(
        &factory,
        Box::new(RecordingReporter::new(run_identity())),
        FormatConformanceCases {
            run: run_identity(),
            authored: raw(authored_config()),
            rejected_authored: raw(serde_json::json!({
                "max_record_bytes": 4096,
                "max_chunk_bytes": 64,
                "max_block_hashes_per_record": 64,
                "max_block_size": 1024,
                "max_input_length": 1_048_576,
                "root_limit": 1,
            })),
            source_descriptor: &LOCAL_LIKE_SOURCE_DESCRIPTOR,
            stream_semantic_digest: STREAM_DIGEST,
            partitions,
            partition_identity: PARTITION_IDENTITY,
            decode_budget: DecodeBatchBudget {
                max_fragments: 3,
                max_bytes: 1 << 20,
            },
            fragment_budget: budget,
            acquisition_budget: acquisition.clone(),
            expected_fragment_count: 3,
            frontier: SourceFrontier {
                through: SourcePosition::new(1),
            },
            seal: SourceSeal {
                final_position: Some(SourcePosition::new(1)),
                digest: ContentDigest::from_bytes([0x31; 32]),
            },
            expected_issue_count: 0,
            advance: Rc::new(|| {}),
        },
    )
    .await;
}

#[test]
fn descriptor_pairs_only_with_a_sequential_source() {
    assert_eq!(
        STREAMING_DYNAMO_DESCRIPTOR.required_access,
        PartitionAccessKind::Sequential
    );
    static RANGE_ONLY: StreamingSourceDescriptor = StreamingSourceDescriptor {
        id: "test_range_only",
        description: "Range-only source",
        modes: &[StreamingSourceMode::Finite],
        access: &[PartitionAccessKind::RangeReadable],
        ordering: StreamingSourceOrdering::Partition,
        resume: &[StreamingResumeGranularity::Byte],
        has_event_time: true,
        has_stable_record_ids: true,
        retention: StreamingSourceRetention::BoundedMemory,
        placement: StreamingSourcePlacement::ControllerOnly,
        supports_virtual_clock: true,
    };
    let factory = StreamingDynamoFormatFactory::new(PROFILE_DIGEST);
    assert_eq!(
        factory
            .validate(&raw(authored_config()), &RANGE_ONLY)
            .expect_err("an access-shape mismatch is refused")
            .code(),
        "schema"
    );
}
