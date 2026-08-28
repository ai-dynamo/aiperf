// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Executable conformance of the streaming source and format adapter contracts.

#[allow(dead_code)]
#[path = "support/streaming_source_conformance.rs"]
mod streaming_source_conformance;

#[allow(dead_code)]
#[path = "support/streaming_format_conformance.rs"]
mod streaming_format_conformance;

use std::cell::Cell;
use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use aiperf_runtime::clock::RealClock;
use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
        CommittedParticipantReceipt, CommittedParticipantState, ParticipantInitialization,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
    },
    failure::{
        AcquisitionFailureCode, DecodeFailureCode, OrderingFailureCode, SourceFailureCode,
        StableStreamingFailure, StreamFormatError, StreamSourceError, StreamingFailureStage,
    },
    format::{
        DecodeBatchBudget, DecodeReceipt, DecodeStep, DecodedFragmentBatch, DecoderCheckpoint,
        DecoderResumeState, FormatEvent, FormatEventSink, FormatProjection, FormatSealReceipt,
        FormatStateRetention, SessionWatermark, StreamingDatasetFormat,
        StreamingDatasetFormatFactory, StreamingFormatDescriptor, StreamingFormatPrepareContext,
        StreamingPartitionDecoder, ValidatedStreamingFormatConfig,
    },
    identity::{
        ContentDigest, ImmutableObjectIdentity, LogicalReplayRunId, StableOrderKey, StableRecordId,
        StableSessionKey,
    },
    reliability::{
        OrdinaryStreamingIssue, StreamingIssueReportError, StreamingIssueReportStatus,
        StreamingIssueReporter, StreamingIssueReporterEndpoint, StreamingIssueReporterHandle,
        StreamingIssueSummary, StreamingReliabilityError,
    },
    source::{
        AcquiredPartition, AcquisitionBudget, BudgetedSourceChunk, OpenedStreamingDatasetSource,
        PartitionAccessKind, PartitionAccessRequest, PreparedStreamingDatasetSource,
        SequentialSourceChunk, SourceEvent, SourceFrontier, SourcePartition,
        SourcePartitionContent, SourceSeal, SourceSnapshotReceipt, StreamingDatasetSource,
        StreamingDatasetSourceFactory, StreamingResumeGranularity, StreamingSequentialReader,
        StreamingSourceDescriptor, StreamingSourceMode, StreamingSourceOrdering,
        StreamingSourcePlacement, StreamingSourcePrepareContext, StreamingSourceRetention,
        StreamingStopReceiver, ValidatedStreamingSourceConfig, streaming_stop_channel,
    },
    unit::{
        ConversationTurnFragment, EventTimeUtc, SessionFragmentLease, SessionMutationV1,
        SourcePosition, StreamingSessionFragment, UnitProvenance,
    },
};
use async_trait::async_trait;
use bytes::Bytes;
use serde::Deserialize;
use serde_json::value::RawValue;
use smallvec::SmallVec;

use streaming_format_conformance::{FormatConformanceCases, assert_format_conformance};
use streaming_source_conformance::{SourceConformanceCases, assert_source_conformance};

const PARTITION_BYTES: &[u8] = b"scripted-partition-bytes";
const CURSOR_BYTES: &[u8] = b"cursor-1";
const FRAGMENT_CONTENT: &[u8] = b"scripted-turn";

fn run_identity() -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([0x11; 32]))
}

fn partition_identity() -> ImmutableObjectIdentity {
    ImmutableObjectIdentity::from_bytes([0x21; 32])
}

fn raw(value: serde_json::Value) -> Box<RawValue> {
    RawValue::from_string(value.to_string()).expect("valid raw configuration")
}

// ---------------------------------------------------------------------------
// Host-owned reliability reporter (test-local, ledger-free)
// ---------------------------------------------------------------------------

/// Counters shared between the sole reporter owner and its erased endpoints.
#[derive(Debug, Default)]
struct CountingState {
    accepted: Cell<u64>,
    is_closed: Cell<bool>,
}

/// Endpoint that counts accepted issues without minting host disposition.
#[derive(Debug)]
struct CountingEndpoint {
    state: Rc<CountingState>,
}

#[async_trait(?Send)]
impl StreamingIssueReporterEndpoint for CountingEndpoint {
    async fn report(
        &self,
        _issue: OrdinaryStreamingIssue,
    ) -> Result<StreamingIssueReportStatus, StreamingIssueReportError> {
        if self.state.is_closed.get() {
            return Err(StreamingIssueReportError::Closed);
        }
        self.state.accepted.set(self.state.accepted.get() + 1);
        Ok(StreamingIssueReportStatus::Accepted)
    }
}

/// Sole mutable owner of the counting endpoint state.
struct CountingReporter {
    state: Rc<CountingState>,
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    initialization: ParticipantInitialization,
}

impl CountingReporter {
    fn new(run: StreamRunIdentity) -> Self {
        Self {
            state: Rc::new(CountingState::default()),
            participant_id: CheckpointParticipantId::new("test_issue_reporter"),
            run,
            initialization: ParticipantInitialization::default(),
        }
    }
}

impl Drop for CountingReporter {
    fn drop(&mut self) {
        // Surviving handles observe `Closed` once the owner is gone.
        self.state.is_closed.set(true);
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for CountingReporter {
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
impl StreamingIssueReporter for CountingReporter {
    fn handle(&self) -> StreamingIssueReporterHandle {
        StreamingIssueReporterHandle::new(CountingEndpoint {
            state: Rc::clone(&self.state),
        })
    }

    fn summary(&self) -> Result<StreamingIssueSummary, StreamingReliabilityError> {
        Ok(StreamingIssueSummary {
            total: self.state.accepted.get(),
            by_scope: BTreeMap::new(),
            by_class: BTreeMap::new(),
            by_disposition: BTreeMap::new(),
            is_admission_fenced: false,
        })
    }
}

// ---------------------------------------------------------------------------
// Scripted source
// ---------------------------------------------------------------------------

/// Shared gate that parks the first scripted step until the harness releases it.
///
/// Adapter factories are `Send + Sync`, so the gate cannot be an `Rc<Cell<_>>`.
#[derive(Debug, Default)]
struct ScriptGate {
    is_open: AtomicBool,
}

impl ScriptGate {
    fn is_open(&self) -> bool {
        self.is_open.load(Ordering::Relaxed)
    }

    fn open(&self) {
        self.is_open.store(true, Ordering::Relaxed);
    }
}

static SOURCE_DESCRIPTOR: StreamingSourceDescriptor = StreamingSourceDescriptor {
    id: "test_scripted",
    description: "Scripted conformance source",
    modes: &[StreamingSourceMode::Finite],
    access: &[PartitionAccessKind::Sequential],
    ordering: StreamingSourceOrdering::Partition,
    resume: &[StreamingResumeGranularity::Byte],
    has_event_time: false,
    has_stable_record_ids: true,
    retention: StreamingSourceRetention::BoundedMemory,
    placement: StreamingSourcePlacement::ControllerOnly,
    supports_virtual_clock: true,
};

static PARTITION_IDENTITY: ImmutableObjectIdentity =
    ImmutableObjectIdentity::from_bytes([0x21; 32]);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScriptedSourceConfig {
    #[allow(dead_code)]
    partitions: u32,
}

#[derive(Debug)]
struct ScriptedSourceFactory {
    gate: Arc<ScriptGate>,
}

impl StreamingDatasetSourceFactory for ScriptedSourceFactory {
    fn descriptor(&self) -> &'static StreamingSourceDescriptor {
        &SOURCE_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
    ) -> Result<Box<dyn ValidatedStreamingSourceConfig>, StreamSourceError> {
        let config: ScriptedSourceConfig = serde_json::from_str(authored.get())
            .map_err(|_| StreamSourceError::source(SourceFailureCode::Discovery))?;
        Ok(Box::new(config))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingSourceConfig>,
        context: &StreamingSourcePrepareContext,
    ) -> Result<Box<dyn PreparedStreamingDatasetSource>, StreamSourceError> {
        Ok(Box::new(ScriptedPreparedSource {
            gate: Arc::clone(&self.gate),
            reporter: context.issue_reporter.clone(),
        }))
    }
}

struct ScriptedPreparedSource {
    gate: Arc<ScriptGate>,
    reporter: StreamingIssueReporterHandle,
}

#[async_trait(?Send)]
impl PreparedStreamingDatasetSource for ScriptedPreparedSource {
    async fn open(
        self: Box<Self>,
        stop: StreamingStopReceiver,
    ) -> Result<OpenedStreamingDatasetSource, StreamSourceError> {
        let control = stop.control();
        Ok(OpenedStreamingDatasetSource {
            source: Box::new(ScriptedSource {
                gate: self.gate,
                stop,
                step: 0,
                snapshot: SourceSnapshotReceipt {
                    digest: ContentDigest::from_bytes([0x31; 32]),
                },
                participant_id: CheckpointParticipantId::new("test_scripted_source"),
                initialization: ParticipantInitialization::default(),
                _reporter: self.reporter,
            }),
            control,
        })
    }
}

struct ScriptedSource {
    gate: Arc<ScriptGate>,
    stop: StreamingStopReceiver,
    step: u8,
    snapshot: SourceSnapshotReceipt,
    participant_id: CheckpointParticipantId,
    initialization: ParticipantInitialization,
    _reporter: StreamingIssueReporterHandle,
}

#[async_trait(?Send)]
impl StreamingDatasetSource for ScriptedSource {
    fn snapshot(&self) -> &SourceSnapshotReceipt {
        &self.snapshot
    }

    async fn next_event(&mut self) -> Result<SourceEvent, StreamSourceError> {
        // A closed gate parks until the harness either releases it or stops the
        // source. Parking is never a seal.
        while !self.gate.is_open() {
            self.stop.stopped().await?;
        }
        self.step += 1;
        match self.step {
            1 | 2 => Ok(SourceEvent::Partition(SourcePartition::new(
                SourcePosition::new(1),
                Box::new(ScriptedPartitionContent),
            ))),
            3 => Ok(SourceEvent::Frontier(SourceFrontier {
                through: SourcePosition::new(1),
            })),
            _ => Ok(SourceEvent::Seal(SourceSeal {
                final_position: Some(SourcePosition::new(1)),
                digest: ContentDigest::from_bytes([0x31; 32]),
            })),
        }
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for ScriptedSource {
    fn participant_id(&self) -> CheckpointParticipantId {
        self.participant_id.clone()
    }

    async fn checkpoint_view(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let bytes = Bytes::from(u64::from(self.step).to_le_bytes().to_vec());
        let budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: bytes.len(),
        })
        .map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = budget
            .acquire(1, bytes.len())
            .await
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let payload = BudgetedCheckpointBytes::new(bytes, lease)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        PreparedParticipantState::new(
            barrier.run,
            self.participant_id.clone(),
            "test.scripted_source",
            1,
            barrier.cut.clone(),
            u64::from(self.step),
            payload,
        )
    }

    async fn initialize(
        &mut self,
        _state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.initialization.initialize_once()
    }

    async fn checkpoint_committed(
        &mut self,
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

struct ScriptedPartitionContent;

#[async_trait(?Send)]
impl SourcePartitionContent for ScriptedPartitionContent {
    fn identity(&self) -> &ImmutableObjectIdentity {
        &PARTITION_IDENTITY
    }

    fn size_bytes(&self) -> Option<u64> {
        Some(PARTITION_BYTES.len() as u64)
    }

    async fn acquire(
        &self,
        request: PartitionAccessRequest,
        budget: &AcquisitionBudget,
    ) -> Result<AcquiredPartition, StreamSourceError> {
        let PartitionAccessRequest::Sequential { resume_offset } = request else {
            return Err(StreamSourceError::acquisition(AcquisitionFailureCode::Open));
        };
        let authority = budget.acquire_memory(1, 0).await?;
        AcquiredPartition::sequential(
            SourcePosition::new(1),
            partition_identity(),
            Some(PARTITION_BYTES.len() as u64),
            resume_offset,
            Box::new(ScriptedSequentialReader { is_done: false }),
            authority,
        )
    }
}

struct ScriptedSequentialReader {
    is_done: bool,
}

#[async_trait(?Send)]
impl StreamingSequentialReader for ScriptedSequentialReader {
    async fn next_chunk(
        &mut self,
        max_bytes: NonZeroUsize,
        budget: &AcquisitionBudget,
    ) -> Result<Option<SequentialSourceChunk>, StreamSourceError> {
        if self.is_done {
            return Ok(None);
        }
        let length = PARTITION_BYTES.len().min(max_bytes.get());
        let bytes = Bytes::from_static(PARTITION_BYTES).slice(0..length);
        let lease = budget.acquire_memory(1, length).await?;
        let chunk = BudgetedSourceChunk::new(bytes, lease)?;
        self.is_done = length == PARTITION_BYTES.len();
        Ok(Some(SequentialSourceChunk::new(
            chunk,
            length as u64,
            ContentDigest::from_bytes([0x41; 32]),
        )))
    }
}

// ---------------------------------------------------------------------------
// Scripted format
// ---------------------------------------------------------------------------

static FORMAT_DESCRIPTOR: StreamingFormatDescriptor = StreamingFormatDescriptor {
    id: "test_scripted_format",
    description: "Scripted conformance format",
    semantic_digest: ContentDigest::from_bytes([0xf0; 32]),
    media_types: &["application/x-ndjson"],
    input_schemas: &["test.scripted.v1"],
    required_access: PartitionAccessKind::Sequential,
    projection: FormatProjection::FullRecord,
    output_schema: "aiperf.streaming.session_fragment.v1",
    has_event_time: false,
    has_stable_record_ids: true,
    retention: FormatStateRetention::BoundedMemory,
    supports_virtual_clock: true,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScriptedFormatConfig {
    #[allow(dead_code)]
    fragments: u32,
}

#[derive(Debug)]
struct ScriptedFormatFactory {
    gate: Arc<ScriptGate>,
    fragment_budget: StreamingResourceBudget,
}

impl StreamingDatasetFormatFactory for ScriptedFormatFactory {
    fn descriptor(&self) -> &'static StreamingFormatDescriptor {
        &FORMAT_DESCRIPTOR
    }

    fn validate(
        &self,
        authored: &RawValue,
        source: &StreamingSourceDescriptor,
    ) -> Result<Box<dyn ValidatedStreamingFormatConfig>, StreamFormatError> {
        if !source.access.contains(&FORMAT_DESCRIPTOR.required_access) {
            return Err(StreamFormatError::decode(DecodeFailureCode::Schema));
        }
        let config: ScriptedFormatConfig = serde_json::from_str(authored.get())
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Schema))?;
        Ok(Box::new(config))
    }

    fn prepare(
        &self,
        _config: Box<dyn ValidatedStreamingFormatConfig>,
        context: &StreamingFormatPrepareContext,
    ) -> Result<Box<dyn StreamingDatasetFormat>, StreamFormatError> {
        Ok(Box::new(ScriptedFormat {
            gate: Arc::clone(&self.gate),
            fragment_budget: self.fragment_budget.clone(),
            partitions_sealed: 0,
            participant_id: CheckpointParticipantId::new("test_scripted_format"),
            initialization: ParticipantInitialization::default(),
            _reporter: context.issue_reporter.clone(),
        }))
    }
}

struct ScriptedFormat {
    gate: Arc<ScriptGate>,
    fragment_budget: StreamingResourceBudget,
    partitions_sealed: u64,
    participant_id: CheckpointParticipantId,
    initialization: ParticipantInitialization,
    _reporter: StreamingIssueReporterHandle,
}

#[async_trait(?Send)]
impl StreamingDatasetFormat for ScriptedFormat {
    async fn begin_partition(
        &mut self,
        partition: AcquiredPartition,
        resume: Option<DecoderCheckpoint>,
    ) -> Result<Box<dyn StreamingPartitionDecoder>, StreamFormatError> {
        if let Some(resume) = &resume
            && resume.partition != *partition.identity()
        {
            return Err(StreamFormatError::decode(DecodeFailureCode::InvalidCursor));
        }
        self.partitions_sealed = 1;
        Ok(Box::new(ScriptedDecoder {
            gate: Arc::clone(&self.gate),
            budget: self.fragment_budget.clone(),
            identity: *partition.identity(),
            cursor: resume.map_or_else(
                || CURSOR_BYTES.to_vec(),
                |resume| resume.state.as_bytes().to_vec(),
            ),
            has_emitted: false,
        }))
    }

    async fn advance_source_frontier(
        &mut self,
        _frontier: SourceFrontier,
        output: &mut dyn FormatEventSink,
    ) -> Result<(), StreamFormatError> {
        output
            .send(FormatEvent::SessionFrontier(SessionWatermark {
                through: EventTimeUtc::new(1)
                    .map_err(|_| StreamFormatError::decode(DecodeFailureCode::Schema))?,
                digest: ContentDigest::from_bytes([0x51; 32]),
            }))
            .await
    }

    async fn seal(
        &mut self,
        _seal: SourceSeal,
        _output: &mut dyn FormatEventSink,
    ) -> Result<FormatSealReceipt, StreamFormatError> {
        Ok(FormatSealReceipt {
            digest: ContentDigest::from_bytes([0x52; 32]),
            partition_count: self.partitions_sealed,
        })
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for ScriptedFormat {
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
        _receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        Ok(())
    }
}

struct ScriptedDecoder {
    gate: Arc<ScriptGate>,
    budget: StreamingResourceBudget,
    identity: ImmutableObjectIdentity,
    cursor: Vec<u8>,
    has_emitted: bool,
}

#[async_trait(?Send)]
impl StreamingPartitionDecoder for ScriptedDecoder {
    async fn next_batch(
        &mut self,
        budget: DecodeBatchBudget,
    ) -> Result<DecodeStep, StreamFormatError> {
        if self.has_emitted {
            // Parks while the previously issued output lease is outstanding; the
            // harness drops the batch and opens the gate to resume.
            while !self.gate.is_open() {
                let probe =
                    self.budget.acquire(1, 0).await.map_err(|_| {
                        StreamFormatError::decode(DecodeFailureCode::BudgetInvariant)
                    })?;
                drop(probe);
            }
            return Ok(DecodeStep::End(DecodeReceipt {
                partition: self.identity,
                fragment_count: 1,
                final_state: self.resume_cursor().await?,
            }));
        }
        assert!(budget.max_fragments >= 1);
        let fragment = self.fragment().await?;
        self.has_emitted = true;
        Ok(DecodeStep::Batch(DecodedFragmentBatch {
            fragments: vec![fragment],
            resume_after: self.resume_cursor().await?,
        }))
    }

    fn resume_state(&self) -> Result<DecoderResumeState, StreamFormatError> {
        let lease = self
            .budget
            .try_acquire(1, self.cursor.len())
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        DecoderResumeState::new(Bytes::from(self.cursor.clone()), lease)
    }
}

impl ScriptedDecoder {
    async fn resume_cursor(&self) -> Result<DecoderResumeState, StreamFormatError> {
        let lease = self
            .budget
            .acquire(1, self.cursor.len())
            .await
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        DecoderResumeState::new(Bytes::from(self.cursor.clone()), lease)
    }

    async fn fragment(&self) -> Result<StreamingSessionFragment, StreamFormatError> {
        let content = FRAGMENT_CONTENT.to_vec();
        let lease = self
            .budget
            .acquire(1, content.len())
            .await
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        let lease = SessionFragmentLease::try_from(lease)
            .map_err(|_| StreamFormatError::decode(DecodeFailureCode::BudgetInvariant))?;
        Ok(StreamingSessionFragment {
            record_id: StableRecordId::from_bytes([0x61; 32]),
            session_key: StableSessionKey::from_bytes([0x62; 32]),
            source_position: SourcePosition::new(1),
            source_partition: self.identity,
            event_time: None,
            stable_tie_break: StableOrderKey::from_bytes([0x63; 32]),
            predecessors: SmallVec::new(),
            mutation: SessionMutationV1::ConversationTurn(ConversationTurnFragment {
                role: "user".to_owned(),
                content,
                turn_ordinal: 0,
            }),
            provenance: UnitProvenance {
                source_partition: self.identity,
                source_position: SourcePosition::new(1),
                format_semantic_digest: FORMAT_DESCRIPTOR.semantic_digest,
            },
            lease,
        })
    }
}

fn harness_memory_limits() -> BudgetLimits {
    BudgetLimits {
        max_items: 16,
        max_bytes: 65_536,
    }
}

fn harness_disk_limits() -> BudgetLimits {
    BudgetLimits {
        max_items: 4,
        max_bytes: 65_536,
    }
}

fn harness_acquisition_budget() -> AcquisitionBudget {
    AcquisitionBudget::new(
        StreamingResourceBudget::new(harness_memory_limits()).expect("valid memory limits"),
        StreamingResourceBudget::new(harness_disk_limits()).expect("valid disk limits"),
    )
}

// ---------------------------------------------------------------------------
// Named conformance tests
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "current_thread")]
async fn conformance_reporter_is_released_before_each_await() {
    let gate = Arc::new(ScriptGate::default());
    let factory = ScriptedSourceFactory {
        gate: Arc::clone(&gate),
    };
    let reporter = CountingReporter::new(run_identity());
    // The harness owns the reporter for the entire run. If any borrow were held
    // across an adapter await, this future would not be constructible: the
    // reporter is `!Send`, and the harness's borrows are scoped between awaits.
    assert_source_conformance(
        &factory,
        Box::new(reporter),
        SourceConformanceCases {
            authored: raw(serde_json::json!({ "partitions": 1 })),
            rejected_authored: raw(serde_json::json!({ "partitions": 1, "extra": true })),
            memory_limits: harness_memory_limits(),
            disk_limits: harness_disk_limits(),
            expected_partition_count: 1,
            expected_duplicate_count: 1,
            expects_frontier: true,
            expected_issue_count: 0,
            run: run_identity(),
            stream_semantic_digest: ContentDigest::from_bytes([0x51; 32]),
            advance: Rc::new(move || gate.open()),
        },
    )
    .await;
}

#[tokio::test(flavor = "current_thread")]
async fn host_stop_wakes_pending_source_without_issue_or_seal() {
    let gate = Arc::new(ScriptGate::default());
    let factory = ScriptedSourceFactory { gate };
    let reporter = CountingReporter::new(run_identity());
    let run = run_identity();
    let context = StreamingSourcePrepareContext {
        run,
        stream_semantic_digest: ContentDigest::from_bytes(*run.logical_replay_run().as_bytes()),
        clock: RealClock::new(),
        acquisition_budget: harness_acquisition_budget(),
        issue_reporter: reporter.handle(),
        clock: RealClock::new(),
    };
    let validated = factory
        .validate(&raw(serde_json::json!({ "partitions": 1 })))
        .expect("authored configuration validates");
    let prepared = factory
        .prepare(validated, &context)
        .expect("source preparation succeeds");
    let (_control, stop) = streaming_stop_channel();
    let mut opened = prepared.open(stop).await.expect("source opens");

    let pending = opened.source.next_event();
    tokio::pin!(pending);
    assert!(
        futures::poll!(&mut pending).is_pending(),
        "a parked source is not a sealed source"
    );

    opened.control.stop();
    // `SourceEvent` carries opaque content authority and is not `Debug`, so the
    // failure path is matched rather than unwrapped.
    let error = match pending.await {
        Ok(_) => panic!("stop wakes the pending source"),
        Err(error) => error,
    };
    assert!(error.is_stopped());
    assert_eq!(error.stage(), StreamingFailureStage::Source);
    assert_eq!(error.code(), "stopped");
    assert_eq!(
        reporter.summary().expect("summary after stop").total,
        0,
        "a controlled stop mints no issue receipt"
    );
}

/// Only the paired host stop channel can produce a stopped outcome.
///
/// The crate-private constructor is unreachable from an adapter. The executing
/// proof of that is the `compile_fail` rustdoc on `StreamSourceError` in
/// `rust/runtime/src/streaming/failure.rs`, which runs under
/// `cargo test -p aiperf-runtime --features streaming --doc`; a `compile_fail`
/// block placed in an integration test would never be collected:
///
/// ```compile_fail
/// use aiperf_runtime::streaming::failure::StreamSourceError;
/// let _ = StreamSourceError::controlled_stop();
/// ```
#[test]
fn external_source_cannot_construct_stopped_error() {
    // Every publicly constructible source error is observably not a stop.
    for error in [
        StreamSourceError::source(SourceFailureCode::Discovery),
        StreamSourceError::source(SourceFailureCode::MutatedObject),
        StreamSourceError::acquisition(AcquisitionFailureCode::IdentityMismatch),
        StreamSourceError::ordering(OrderingFailureCode::LateData),
    ] {
        assert!(
            !error.is_stopped(),
            "no public constructor can forge a controlled stop"
        );
    }

    let (control, receiver) = streaming_stop_channel();
    assert!(!receiver.is_stopped());
    control.stop();
    assert!(receiver.is_stopped());
}

#[tokio::test(flavor = "current_thread")]
async fn scripted_format_satisfies_the_shared_conformance_harness() {
    let gate = Arc::new(ScriptGate::default());
    // Exactly two items: one batch (its fragment plus its resume cursor)
    // saturates the output budget, so the decoder's next pull must park.
    let fragment_budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 2,
        max_bytes: 4096,
    })
    .expect("valid fragment budget");
    let factory = ScriptedFormatFactory {
        gate: Arc::clone(&gate),
        fragment_budget: fragment_budget.clone(),
    };
    let acquisition = harness_acquisition_budget();
    let content = ScriptedPartitionContent;
    let mut partitions = Vec::new();
    for _ in 0..2 {
        partitions.push(
            content
                .acquire(
                    PartitionAccessRequest::Sequential { resume_offset: 0 },
                    &acquisition,
                )
                .await
                .expect("scripted acquisition"),
        );
    }

    assert_format_conformance(
        &factory,
        Box::new(CountingReporter::new(run_identity())),
        FormatConformanceCases {
            run: run_identity(),
            authored: raw(serde_json::json!({ "fragments": 1 })),
            rejected_authored: raw(serde_json::json!({ "fragments": 1, "extra": true })),
            source_descriptor: &SOURCE_DESCRIPTOR,
            stream_semantic_digest: ContentDigest::from_bytes([0x71; 32]),
            partitions,
            partition_identity: partition_identity(),
            decode_budget: DecodeBatchBudget {
                max_fragments: 4,
                max_bytes: 4096,
            },
            fragment_budget,
            acquisition_budget: harness_acquisition_budget(),
            expected_fragment_count: 1,
            frontier: SourceFrontier {
                through: SourcePosition::new(1),
            },
            seal: SourceSeal {
                final_position: Some(SourcePosition::new(1)),
                digest: ContentDigest::from_bytes([0x31; 32]),
            },
            expected_issue_count: 0,
            advance: Rc::new(move || gate.open()),
        },
    )
    .await;
}
