// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, BudgetedCheckpointBytes, CheckpointBarrier,
        CheckpointCut, CheckpointEpoch, CheckpointError, CheckpointGeneration,
        CheckpointParticipantId, CheckpointTerminalReason, CommittedCheckpointGeneration,
        CommittedParticipantReceipt, CommittedParticipantState, DecodeHorizon, DiscoveryHorizon,
        EventTimeWatermark, OrderedActionHorizon, ParticipantInitialization,
        ParticipantStateDescriptor, PreparedParticipantState, StreamRunIdentity,
        StreamingCheckpointParticipant, TerminalActionHorizon,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations, CurrentV4CheckpointGeneration,
        LeasedCheckpointGeneration, LeasedCheckpointGenerationView, LegacyV3FixtureLimits,
        LegacyV3FixturePrecharge, LegacyV3ReadOnlyFixture, StreamingGenerationTransaction,
    },
    checkpoints::memory::{
        ImmutableObjectInventory, MemoryCheckpointBackend, MemoryCheckpointLimits,
        MemoryLiveBudgetUsage,
    },
    identity::{
        ContentDigest, GlobalSequence, ImmutableObjectIdentity, LogicalReplayRunId,
        SessionCausalFrontier, StableRecordId,
    },
    reliability::{
        BudgetOwnedStreamingIssueReporter, HandledIssueCut, IssueSequenceUpdate,
        OrdinaryStreamingIssue, PreparedIssueReceiptResultPartition, PreparedStreamingIssuePolicy,
        StreamingInputDomainIdentity, StreamingIssueClass, StreamingIssueComponentId,
        StreamingIssueDisposition, StreamingIssueReporter, StreamingIssueScopeKind,
        StreamingIssueThresholdRule, submission_queue_charge_bytes,
    },
    results::{
        BudgetedResultDescriptor, CellId, ResultPartition, ResultProjectionId, ResultSchemaVersion,
        ResultSegmentDescriptor, WorkerId,
    },
    unit::{EventTimeUtc, SourcePosition},
};
use async_trait::async_trait;
use bytes::Bytes;
use serde::Serialize;
use std::num::{NonZeroU64, NonZeroUsize};

use aiperf_runtime::streaming::failure::{
    DecodeFailureCode, OrdinaryStreamingFailure, StreamFormatError,
};

pub fn cut_at(value: u64) -> CheckpointCut {
    // Epoch-boundary tests pass `u64::MAX`, which no event time represents;
    // saturating keeps those cuts constructible without changing any value a
    // signed event time can hold.
    let event_time = EventTimeUtc::new(i64::try_from(value).unwrap_or(i64::MAX))
        .expect("non-negative test event time");
    CheckpointCut {
        discovered: DiscoveryHorizon::new(SourcePosition::new(value)),
        acquired: AcquisitionHorizon::new(SourcePosition::new(value)),
        decoded: DecodeHorizon::new(SourcePosition::new(value)),
        ordered: OrderedActionHorizon::new(GlobalSequence::new(value)),
        admitted: AdmissionHorizon::new(GlobalSequence::new(value)),
        terminal: TerminalActionHorizon::new(GlobalSequence::new(value)),
        event_watermark: EventTimeWatermark::Hard {
            through: event_time,
        },
        causal_frontier: SessionCausalFrontier {
            through_sequence: GlobalSequence::new(value),
            event_time: Some(event_time),
            digest: ContentDigest::from_bytes([value as u8; 32]),
        },
        handled_issues: HandledIssueCut::empty(),
    }
}

pub fn run_id(value: u8) -> StreamRunIdentity {
    StreamRunIdentity::new(LogicalReplayRunId::from_bytes([value; 32]))
}

pub fn barrier_for_run(run: u8, value: u64) -> CheckpointBarrier {
    CheckpointBarrier {
        run: run_id(run),
        epoch: CheckpointEpoch::new(value),
        cut: cut_at(value),
        plan_digest: ContentDigest::from_bytes([0x55; 32]),
    }
}

pub fn barrier_at(value: u64) -> CheckpointBarrier {
    barrier_for_run(1, value)
}

pub fn backend_limits() -> MemoryCheckpointLimits {
    let limits = BudgetLimits {
        max_items: 64,
        max_bytes: 1_048_576,
    };
    MemoryCheckpointLimits {
        transactions: limits,
        prepared_indexes: limits,
        storage: limits,
        result_summaries: limits,
        reads: limits,
    }
}

pub fn backend_limits_with_each_capacity(capacity: usize) -> MemoryCheckpointLimits {
    let limits = BudgetLimits {
        max_items: capacity,
        max_bytes: capacity,
    };
    MemoryCheckpointLimits {
        transactions: limits,
        prepared_indexes: limits,
        storage: limits,
        result_summaries: limits,
        reads: limits,
    }
}

pub fn backend_limits_with_storage_bytes(bytes: usize) -> MemoryCheckpointLimits {
    let mut limits = backend_limits();
    limits.storage.max_bytes = bytes;
    limits
}

pub fn backend_limits_with_read_bytes(bytes: usize) -> MemoryCheckpointLimits {
    let mut limits = backend_limits();
    limits.reads.max_bytes = bytes;
    limits
}

pub fn invalid_backend_limits() -> Vec<(
    MemoryCheckpointLimits,
    aiperf_runtime::streaming::checkpoint::CheckpointBackendBudgetKind,
    aiperf_runtime::streaming::checkpoint::CheckpointBackendBudgetFailureCode,
)> {
    use aiperf_runtime::streaming::checkpoint::{
        CheckpointBackendBudgetFailureCode as Code, CheckpointBackendBudgetKind as Kind,
    };

    let first_unrepresentable = usize::try_from(u64::from(u32::MAX) + 1)
        .expect("64-bit test target represents the conversion boundary");
    let mut cases = Vec::with_capacity(20);
    for kind in [
        Kind::Transaction,
        Kind::PreparedIndex,
        Kind::Storage,
        Kind::ResultSummary,
        Kind::Read,
    ] {
        for (items, bytes, code) in [
            (0, 64, Code::ItemCapacity),
            (64, 0, Code::ByteCapacity),
            (first_unrepresentable, 64, Code::Unrepresentable),
            (64, first_unrepresentable, Code::Unrepresentable),
        ] {
            let mut limits = backend_limits();
            let replacement = BudgetLimits {
                max_items: items,
                max_bytes: bytes,
            };
            match kind {
                Kind::Transaction => limits.transactions = replacement,
                Kind::PreparedIndex => limits.prepared_indexes = replacement,
                Kind::Storage => limits.storage = replacement,
                Kind::ResultSummary => limits.result_summaries = replacement,
                Kind::Read => limits.reads = replacement,
            }
            cases.push((limits, kind, code));
        }
    }
    cases
}

pub fn contains_capacity(limits: MemoryCheckpointLimits, capacity: usize) -> bool {
    [
        limits.transactions,
        limits.prepared_indexes,
        limits.storage,
        limits.result_summaries,
        limits.reads,
    ]
    .iter()
    .any(|limits| limits.max_items == capacity || limits.max_bytes == capacity)
}

pub fn expectations(run: StreamRunIdentity) -> CheckpointGenerationExpectations {
    CheckpointGenerationExpectations {
        run,
        participant_plan: aiperf_runtime::streaming::checkpoint::CheckpointParticipantPlan::new([
            CheckpointParticipantId::new("participant"),
        ])
        .expect("valid test participant plan"),
        execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
    }
}

pub fn metadata_with_lineage(
    previous: Option<aiperf_runtime::streaming::checkpoint::CheckpointGeneration>,
    epoch: u64,
) -> CheckpointCommitMetadata {
    CheckpointCommitMetadata {
        previous,
        epoch: CheckpointEpoch::new(epoch),
        cut: cut_at(epoch),
        execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
        is_final: false,
        terminal_reason: None,
    }
}

pub fn metadata_at(epoch: u64) -> CheckpointCommitMetadata {
    metadata_with_lineage(None, epoch)
}

pub fn same_epoch_wrong_digest(
    generation: &aiperf_runtime::streaming::checkpoint::CheckpointGeneration,
) -> aiperf_runtime::streaming::checkpoint::CheckpointGeneration {
    aiperf_runtime::streaming::checkpoint::CheckpointGeneration::new(
        generation.epoch(),
        ContentDigest::from_bytes([0xfd; 32]),
    )
}

pub async fn prepared_participant(run: StreamRunIdentity, epoch: u64) -> PreparedParticipantState {
    prepared_participant_with_bytes(run, epoch, Bytes::from_static(b"participant-state")).await
}

pub async fn prepared_participant_with_bytes(
    run: StreamRunIdentity,
    epoch: u64,
    bytes: Bytes,
) -> PreparedParticipantState {
    PreparedParticipantState::new(
        run,
        CheckpointParticipantId::new("participant"),
        "test.participant",
        1,
        cut_at(epoch),
        1,
        checkpoint_payload(bytes).await,
    )
    .expect("valid prepared participant")
}

pub async fn result_partition(run: StreamRunIdentity, epoch: u64) -> ResultPartition {
    result_partition_with_projection_for(run, epoch, "projection")
        .await
        .1
}

pub async fn result_partition_with_projection(
    projection: &str,
) -> (StreamingResourceBudget, ResultPartition) {
    result_partition_with_projection_for(run_id(1), 1, projection).await
}

pub async fn result_partition_with_projection_for(
    run: StreamRunIdentity,
    epoch: u64,
    projection: &str,
) -> (StreamingResourceBudget, ResultPartition) {
    result_partition_with_projection_and_bytes_for(
        run,
        epoch,
        projection,
        Bytes::from_static(b"result-payload"),
    )
    .await
}

pub async fn result_partition_with_projection_and_bytes_for(
    run: StreamRunIdentity,
    epoch: u64,
    projection: &str,
    payload_bytes: Bytes,
) -> (StreamingResourceBudget, ResultPartition) {
    let payload_budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: payload_bytes.len(),
    })
    .expect("valid result payload budget");
    let payload_lease = payload_budget
        .acquire(1, payload_bytes.len())
        .await
        .expect("result payload lease");
    let payload = BudgetedCheckpointBytes::new(payload_bytes.clone(), payload_lease)
        .expect("exact result payload charge");
    let projection = ResultProjectionId::new(projection).expect("nonempty test projection");
    let descriptor = ResultSegmentDescriptor {
        run,
        epoch: CheckpointEpoch::new(epoch),
        cell_id: CellId::new(0),
        worker_id: WorkerId::new(0),
        projection,
        schema: ResultSchemaVersion::new(1),
        first_sequence: GlobalSequence::new(1),
        last_sequence: GlobalSequence::new(1),
        item_count: 1,
        byte_length: u64::try_from(payload_bytes.len()).expect("small payload"),
        membership_root: ContentDigest::from_bytes([0x41; 32]),
        payload_digest: ContentDigest::from_bytes(*blake3::hash(&payload_bytes).as_bytes()),
    };
    let descriptor_bytes = std::mem::size_of::<ResultSegmentDescriptor>()
        + descriptor.projection.retained_allocation_bytes();
    let descriptor_budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: descriptor_bytes,
    })
    .expect("valid result descriptor budget");
    let descriptor_lease = descriptor_budget
        .acquire(1, descriptor_bytes)
        .await
        .expect("result descriptor lease");
    let descriptor = BudgetedResultDescriptor::new(descriptor, descriptor_lease)
        .expect("exact descriptor charge");
    (
        descriptor_budget,
        ResultPartition::new(descriptor, payload).expect("verified result partition"),
    )
}

pub async fn transaction_with_all_participants(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
) -> aiperf_runtime::streaming::checkpoints::memory::MemoryGenerationTransaction {
    let mut transaction = backend
        .begin_generation(run, None, expectations(run))
        .await
        .expect("begin transaction");
    transaction
        .stage_participant(prepared_participant(run, 1).await)
        .await
        .expect("stage participant");
    transaction
}

pub async fn commit_empty(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
    previous: Option<aiperf_runtime::streaming::checkpoint::CheckpointGeneration>,
    epoch: u64,
) -> Result<aiperf_runtime::streaming::checkpoint::CommittedCheckpointGeneration, CheckpointError> {
    let expected = predecessor_for(backend, run, previous.as_ref()).await?;
    let mut transaction = backend
        .begin_generation(run, expected, expectations(run))
        .await?;
    transaction
        .stage_participant(prepared_participant(run, epoch).await)
        .await?;
    transaction
        .stage_results(&mut Vec::new(), &mut None)
        .await?;
    transaction
        .commit(metadata_with_lineage(previous, epoch))
        .await
}

pub async fn commit_with_segment(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
    previous: Option<aiperf_runtime::streaming::checkpoint::CheckpointGeneration>,
    epoch: u64,
) -> Result<aiperf_runtime::streaming::checkpoint::CommittedCheckpointGeneration, CheckpointError> {
    let expected = predecessor_for(backend, run, previous.as_ref()).await?;
    let mut transaction = backend
        .begin_generation(run, expected, expectations(run))
        .await?;
    transaction
        .stage_participant(prepared_participant(run, epoch).await)
        .await?;
    let mut partitions = vec![result_partition(run, epoch).await];
    transaction
        .stage_results(&mut partitions, &mut None)
        .await?;
    transaction
        .commit(metadata_with_lineage(previous, epoch))
        .await
}

/// Commit one generation with a result segment against any backend.
///
/// The memory-specific helper above returns the concrete transaction type; this
/// one drives the erased trait so a second backend can reuse the same shape.
pub async fn commit_with_segment_on(
    backend: &dyn aiperf_runtime::streaming::checkpoint_backend::StreamingCheckpointBackend,
    run: StreamRunIdentity,
    previous: Option<CheckpointGeneration>,
    epoch: u64,
) -> Result<CommittedCheckpointGeneration, CheckpointError> {
    let expected = match previous.as_ref() {
        None => None,
        Some(previous) => {
            let opened = backend
                .open_latest(&run, &expectations(run))
                .await?
                .ok_or_else(|| CheckpointError::GenerationConflict {
                    expected: Some(previous.clone()),
                    actual: None,
                })?;
            Some(current_v4_predecessor(&opened, previous)?)
        }
    };
    let mut transaction = backend
        .begin_generation(run, expected, expectations(run))
        .await?;
    transaction
        .stage_participant(prepared_participant(run, epoch).await)
        .await?;
    let mut partitions = vec![result_partition(run, epoch).await];
    transaction.stage_results(&mut partitions, &mut None).await?;
    transaction
        .commit(metadata_with_lineage(previous, epoch))
        .await
}

pub fn current_v4_predecessor(
    opened: &LeasedCheckpointGeneration,
    expected: &CheckpointGeneration,
) -> Result<CurrentV4CheckpointGeneration, CheckpointError> {
    match opened.view() {
        LeasedCheckpointGenerationView::CurrentV4(reader) => {
            reader.current_v4_predecessor(expected)
        }
        LeasedCheckpointGenerationView::LegacyV3ReadOnly(_) => {
            Err(CheckpointError::LegacyReadOnlyHead)
        }
    }
}

async fn predecessor_for(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
    expected: Option<&CheckpointGeneration>,
) -> Result<Option<CurrentV4CheckpointGeneration>, CheckpointError> {
    let Some(expected) = expected else {
        return Ok(None);
    };
    let opened = backend
        .open_latest(&run, &expectations(run))
        .await?
        .ok_or_else(|| CheckpointError::GenerationConflict {
            expected: Some(expected.clone()),
            actual: None,
        })?;
    current_v4_predecessor(&opened, expected).map(Some)
}

pub async fn committed_current_v4_participant_state(
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    schema_id: &str,
    schema_version: u32,
    cut: CheckpointCut,
    item_count: u64,
    bytes: Bytes,
) -> CommittedParticipantState {
    let backend = MemoryCheckpointBackend::new(backend_limits()).expect("valid memory backend");
    let participant_plan = aiperf_runtime::streaming::checkpoint::CheckpointParticipantPlan::new([
        participant_id.clone(),
    ])
    .expect("valid one-participant plan");
    let expectations = CheckpointGenerationExpectations {
        run,
        participant_plan,
        execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
    };
    let payload = checkpoint_payload(bytes).await;
    let prepared = PreparedParticipantState::new(
        run,
        participant_id,
        schema_id,
        schema_version,
        cut.clone(),
        item_count,
        payload,
    )
    .expect("valid prepared current-v4 participant");
    let descriptor = prepared.descriptor().clone();
    let mut transaction = backend
        .begin_generation(run, None, expectations.clone())
        .await
        .expect("begin current-v4 fixture generation");
    transaction
        .stage_participant(prepared)
        .await
        .expect("stage current-v4 fixture participant");
    transaction
        .stage_results(&mut Vec::new(), &mut None)
        .await
        .expect("stage canonical empty result epoch");
    transaction
        .commit(CheckpointCommitMetadata {
            previous: None,
            epoch: CheckpointEpoch::new(1),
            cut,
            execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
            result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
            is_final: false,
            terminal_reason: None,
        })
        .await
        .expect("commit current-v4 fixture generation");
    let opened = backend
        .open_latest(&run, &expectations)
        .await
        .expect("open current-v4 fixture generation")
        .expect("current-v4 fixture head");
    match opened.view() {
        LeasedCheckpointGenerationView::CurrentV4(reader) => reader
            .read_participant(&descriptor)
            .await
            .expect("read verified current-v4 participant"),
        LeasedCheckpointGenerationView::LegacyV3ReadOnly(_) => {
            panic!("fresh fixture must be current-v4")
        }
    }
}

pub struct LegacyV3FixtureExpectation {
    pub generation: CheckpointGeneration,
    pub participant: ParticipantStateDescriptor,
    pub result: ResultSegmentDescriptor,
}

pub fn legacy_v3_fixture_limits() -> LegacyV3FixtureLimits {
    LegacyV3FixtureLimits {
        max_objects: NonZeroUsize::new(4).expect("nonzero fixture object limit"),
        max_bytes: NonZeroU64::new(1_048_576).expect("nonzero fixture byte limit"),
    }
}

pub fn legacy_fixture_budget() -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: 16,
        max_bytes: 1_048_576,
    })
    .expect("valid legacy fixture budget")
}

pub async fn legacy_v3_fixture(
    budget: &StreamingResourceBudget,
    run: StreamRunIdentity,
) -> (LegacyV3ReadOnlyFixture, LegacyV3FixtureExpectation) {
    let cut = cut_at(1);
    let participant_bytes = Bytes::from_static(b"legacy-participant-state");
    let participant_digest =
        ContentDigest::from_bytes(*blake3::hash(&participant_bytes).as_bytes());
    let participant = ParticipantStateDescriptor {
        participant_id: CheckpointParticipantId::new("participant"),
        schema_id: "test.participant".into(),
        schema_version: 1,
        represented_cut: cut.clone(),
        content_digest: participant_digest,
        item_count: 1,
        byte_length: u64::try_from(participant_bytes.len()).expect("small fixture payload"),
    };
    let plan = aiperf_runtime::streaming::checkpoint::CheckpointParticipantPlan::new([participant
        .participant_id
        .clone()])
    .expect("valid legacy participant plan");
    let result_bytes = Bytes::from_static(b"legacy-result-payload");
    let result_digest = ContentDigest::from_bytes(*blake3::hash(&result_bytes).as_bytes());
    let result = ResultSegmentDescriptor {
        run,
        epoch: CheckpointEpoch::new(1),
        cell_id: CellId::new(0),
        worker_id: WorkerId::new(0),
        projection: ResultProjectionId::new("legacy").expect("valid legacy projection"),
        schema: ResultSchemaVersion::new(1),
        first_sequence: GlobalSequence::new(1),
        last_sequence: GlobalSequence::new(1),
        item_count: 1,
        byte_length: u64::try_from(result_bytes.len()).expect("small fixture result"),
        membership_root: ContentDigest::from_bytes([0x43; 32]),
        payload_digest: result_digest,
    };
    let result_index_bytes = Bytes::from(
        serde_json::to_vec(std::slice::from_ref(&result))
            .expect("encode canonical legacy result index")
            .into_boxed_slice(),
    );
    let result_index_root = legacy_result_index_root(&result_index_bytes);
    let wire_cut = LegacyV3CheckpointCut::from(&cut);
    let wire_descriptors = vec![LegacyV3ParticipantDescriptor::from(&participant)];
    let generation_digest = legacy_generation_digest(
        &run,
        CheckpointEpoch::new(1),
        None,
        &wire_cut,
        &plan.digest(),
        &ContentDigest::from_bytes([0x31; 32]),
        &ContentDigest::from_bytes([0x32; 32]),
        &wire_descriptors,
        &result_index_root,
        false,
        None,
    );
    let generation = CheckpointGeneration::new(CheckpointEpoch::new(1), generation_digest);
    let generation_bytes = Bytes::from(
        serde_json::to_vec(&LegacyV3GenerationWire {
            run,
            generation: generation.clone(),
            previous: None,
            cut: wire_cut,
            participant_plan_digest: plan.digest(),
            execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
            result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
            participant_descriptors: wire_descriptors,
            result_index_root,
            is_final: false,
            terminal_reason: None,
        })
        .expect("encode valid legacy-v3 generation")
        .into_boxed_slice(),
    );
    let exact_encoded_bytes = generation_bytes.len()
        + participant_bytes.len()
        + result_index_bytes.len()
        + result_bytes.len();
    let mut precharge = LegacyV3FixturePrecharge::acquire(
        budget,
        legacy_v3_fixture_limits(),
        4,
        exact_encoded_bytes,
    )
    .await
    .expect("precharge complete legacy fixture");
    let generation_object = precharge
        .compact_object(*generation.digest(), &generation_bytes)
        .expect("compact generation object");
    let participant_object = precharge
        .compact_object(participant_digest, &participant_bytes)
        .expect("compact participant object");
    let participant_objects = precharge
        .collect_inventory(std::iter::once(participant_object))
        .expect("collect participant inventory");
    let result_index_object = precharge
        .compact_object(result_index_root, &result_index_bytes)
        .expect("compact result-index object");
    let result_object = precharge
        .compact_object(result_digest, &result_bytes)
        .expect("compact result payload object");
    let result_objects = precharge
        .collect_inventory(std::iter::once(result_object))
        .expect("collect result inventory");
    let fixture = precharge
        .finish(
            run,
            generation.clone(),
            generation_object,
            participant_objects,
            result_index_object,
            result_objects,
        )
        .expect("finish valid legacy-v3 fixture");
    (
        fixture,
        LegacyV3FixtureExpectation {
            generation,
            participant,
            result,
        },
    )
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct LegacyV3CheckpointCut {
    discovered: DiscoveryHorizon,
    acquired: AcquisitionHorizon,
    decoded: DecodeHorizon,
    ordered: OrderedActionHorizon,
    admitted: AdmissionHorizon,
    terminal: TerminalActionHorizon,
    event_watermark: EventTimeWatermark,
    causal_frontier: SessionCausalFrontier,
}

impl From<&CheckpointCut> for LegacyV3CheckpointCut {
    fn from(cut: &CheckpointCut) -> Self {
        Self {
            discovered: cut.discovered.clone(),
            acquired: cut.acquired.clone(),
            decoded: cut.decoded.clone(),
            ordered: cut.ordered.clone(),
            admitted: cut.admitted.clone(),
            terminal: cut.terminal.clone(),
            event_watermark: cut.event_watermark.clone(),
            causal_frontier: cut.causal_frontier.clone(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct LegacyV3ParticipantDescriptor {
    participant_id: CheckpointParticipantId,
    schema_id: String,
    schema_version: u32,
    represented_cut: LegacyV3CheckpointCut,
    content_digest: ContentDigest,
    item_count: u64,
    byte_length: u64,
}

impl From<&ParticipantStateDescriptor> for LegacyV3ParticipantDescriptor {
    fn from(descriptor: &ParticipantStateDescriptor) -> Self {
        Self {
            participant_id: descriptor.participant_id.clone(),
            schema_id: descriptor.schema_id.clone(),
            schema_version: descriptor.schema_version,
            represented_cut: LegacyV3CheckpointCut::from(&descriptor.represented_cut),
            content_digest: descriptor.content_digest,
            item_count: descriptor.item_count,
            byte_length: descriptor.byte_length,
        }
    }
}

#[derive(Serialize)]
struct LegacyV3GenerationWire {
    run: StreamRunIdentity,
    generation: CheckpointGeneration,
    previous: Option<ContentDigest>,
    cut: LegacyV3CheckpointCut,
    participant_plan_digest: ContentDigest,
    execution_plan_digest: ContentDigest,
    result_plan_digest: ContentDigest,
    participant_descriptors: Vec<LegacyV3ParticipantDescriptor>,
    result_index_root: ContentDigest,
    is_final: bool,
    terminal_reason: Option<CheckpointTerminalReason>,
}

#[allow(clippy::too_many_arguments)]
fn legacy_generation_digest(
    run: &StreamRunIdentity,
    epoch: CheckpointEpoch,
    previous: Option<&ContentDigest>,
    cut: &LegacyV3CheckpointCut,
    participant_plan_digest: &ContentDigest,
    execution_plan_digest: &ContentDigest,
    result_plan_digest: &ContentDigest,
    descriptors: &[LegacyV3ParticipantDescriptor],
    result_index_root: &ContentDigest,
    is_final: bool,
    terminal_reason: Option<CheckpointTerminalReason>,
) -> ContentDigest {
    let cut = serde_json::to_vec(cut).expect("encode legacy cut");
    let descriptors = serde_json::to_vec(descriptors).expect("encode legacy descriptors");
    let terminal_state = match terminal_reason {
        None => [0, 0],
        Some(CheckpointTerminalReason::Completed) => [1, 1],
        Some(CheckpointTerminalReason::Aborted) => [1, 2],
        Some(CheckpointTerminalReason::Cancelled) => [1, 3],
    };
    let mut hasher = blake3::Hasher::new();
    update_legacy_generation_digest_field(
        &mut hasher,
        b"aiperf.streaming.committed-checkpoint-generation.v3",
    );
    update_legacy_generation_digest_field(&mut hasher, run.logical_replay_run().as_bytes());
    update_legacy_generation_digest_field(&mut hasher, &epoch.get().to_le_bytes());
    match previous {
        None => update_legacy_generation_digest_field(&mut hasher, &[0]),
        Some(previous) => {
            update_legacy_generation_digest_field(&mut hasher, &[1]);
            update_legacy_generation_digest_field(&mut hasher, previous.as_bytes());
        }
    }
    update_legacy_generation_digest_field(&mut hasher, &cut);
    update_legacy_generation_digest_field(&mut hasher, participant_plan_digest.as_bytes());
    update_legacy_generation_digest_field(&mut hasher, execution_plan_digest.as_bytes());
    update_legacy_generation_digest_field(&mut hasher, result_plan_digest.as_bytes());
    update_legacy_generation_digest_field(&mut hasher, &descriptors);
    update_legacy_generation_digest_field(&mut hasher, result_index_root.as_bytes());
    update_legacy_generation_digest_field(&mut hasher, &[u8::from(is_final)]);
    update_legacy_generation_digest_field(&mut hasher, &terminal_state);
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

fn update_legacy_generation_digest_field(hasher: &mut blake3::Hasher, field: &[u8]) {
    hasher.update(&(field.len() as u64).to_le_bytes());
    hasher.update(field);
}

fn legacy_result_index_root(encoded: &[u8]) -> ContentDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"aiperf.streaming.result-index.v1");
    hasher.update(&(encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
    ContentDigest::from_bytes(*hasher.finalize().as_bytes())
}

/// Backend-neutral view of one run's authoritative publication state.
///
/// The conformance helper only needs equality across a refused publication, so
/// backend-specific inventory and usage are rendered into one opaque string
/// rather than forcing every backend to expose the memory backend's types.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicationAuthoritySnapshot {
    generation: Option<CheckpointGeneration>,
    storage: String,
}

impl PublicationAuthoritySnapshot {
    pub fn new(generation: Option<CheckpointGeneration>, storage: String) -> Self {
        Self {
            generation,
            storage,
        }
    }
}

#[async_trait(?Send)]
pub trait PublicationBackendFixture {
    fn run(&self) -> StreamRunIdentity;
    async fn seed_baseline(&self) -> CommittedCheckpointGeneration;
    async fn seed_maximum_epoch(&self) -> (StreamRunIdentity, CommittedCheckpointGeneration);
    async fn staged_after(
        &self,
        run: StreamRunIdentity,
        expected: CheckpointGeneration,
        participant_epoch: u64,
    ) -> Box<dyn StreamingGenerationTransaction>;
    async fn authority_snapshot(&self, run: StreamRunIdentity) -> PublicationAuthoritySnapshot;
    fn reset_effect_counter(&self);
    fn effect_counter(&self) -> u64;
}

pub struct MemoryPublicationBackendFixture {
    backend: MemoryCheckpointBackend,
    run: StreamRunIdentity,
}

pub fn memory_publication_backend_fixture() -> MemoryPublicationBackendFixture {
    MemoryPublicationBackendFixture {
        backend: MemoryCheckpointBackend::new(backend_limits()).expect("valid memory backend"),
        run: run_id(1),
    }
}

#[async_trait(?Send)]
impl PublicationBackendFixture for MemoryPublicationBackendFixture {
    fn run(&self) -> StreamRunIdentity {
        self.run
    }

    async fn seed_baseline(&self) -> CommittedCheckpointGeneration {
        commit_with_segment(&self.backend, self.run, None, 1)
            .await
            .expect("seed baseline generation")
    }

    async fn seed_maximum_epoch(&self) -> (StreamRunIdentity, CommittedCheckpointGeneration) {
        let run = run_id(2);
        let committed = self
            .backend
            .seed_nonempty_committed_generation_at_epoch(
                run,
                CheckpointEpoch::new(u64::MAX),
                &expectations(run),
                vec![prepared_participant(run, u64::MAX).await],
            )
            .await
            .expect("seed maximum current-v4 head in fresh run");
        (run, committed)
    }

    async fn staged_after(
        &self,
        run: StreamRunIdentity,
        expected: CheckpointGeneration,
        participant_epoch: u64,
    ) -> Box<dyn StreamingGenerationTransaction> {
        let opened = self
            .backend
            .open_latest(&run, &expectations(run))
            .await
            .expect("open lineage head")
            .expect("lineage head exists");
        let predecessor =
            current_v4_predecessor(&opened, &expected).expect("verified current-v4 predecessor");
        let mut transaction = self
            .backend
            .begin_generation(run, Some(predecessor), expectations(run))
            .await
            .expect("begin lineage transaction");
        transaction
            .stage_participant(prepared_participant(run, participant_epoch).await)
            .await
            .expect("stage lineage participant");
        transaction
            .stage_results(&mut Vec::new(), &mut None)
            .await
            .expect("stage lineage result epoch");
        Box::new(transaction)
    }

    async fn authority_snapshot(&self, run: StreamRunIdentity) -> PublicationAuthoritySnapshot {
        let generation = self
            .backend
            .open_latest(&run, &expectations(run))
            .await
            .expect("open memory head")
            .map(|reader| reader.generation().clone());
        PublicationAuthoritySnapshot::new(
            generation,
            format!(
                "{:?}|{:?}",
                self.backend.immutable_object_inventory(&run),
                self.backend.live_budget_usage()
            ),
        )
    }

    fn reset_effect_counter(&self) {
        self.backend.reset_test_state_accesses();
    }

    fn effect_counter(&self) -> u64 {
        self.backend.test_state_accesses()
    }
}

pub async fn assert_publication_backend_lineage_conformance(
    fixture: impl PublicationBackendFixture,
) {
    let run = fixture.run();
    let baseline = fixture.seed_baseline().await;
    let baseline_snapshot = fixture.authority_snapshot(run).await;

    let wrong = fixture.staged_after(run, baseline.generation(), 2).await;
    fixture.reset_effect_counter();
    assert_eq!(
        wrong
            .commit(metadata_with_lineage(None, 2))
            .await
            .expect_err("wrong predecessor must fail"),
        CheckpointError::ObjectVerification,
    );
    assert_eq!(fixture.effect_counter(), 0);
    assert_eq!(fixture.authority_snapshot(run).await, baseline_snapshot);

    let skipped = fixture.staged_after(run, baseline.generation(), 2).await;
    fixture.reset_effect_counter();
    assert_eq!(
        skipped
            .commit(metadata_with_lineage(Some(baseline.generation()), 3))
            .await
            .expect_err("skipped epoch must fail"),
        CheckpointError::ObjectVerification,
    );
    assert_eq!(fixture.effect_counter(), 0);
    assert_eq!(fixture.authority_snapshot(run).await, baseline_snapshot);

    let (maximum_run, maximum_generation) = fixture.seed_maximum_epoch().await;
    let maximum = maximum_generation.generation();
    let maximum_snapshot = fixture.authority_snapshot(maximum_run).await;
    let overflow = fixture
        .staged_after(maximum_run, maximum.clone(), u64::MAX)
        .await;
    let mut overflow_metadata = metadata_at(u64::MAX);
    overflow_metadata.previous = Some(maximum.clone());
    fixture.reset_effect_counter();
    assert_eq!(
        overflow
            .commit(overflow_metadata)
            .await
            .expect_err("maximum predecessor must overflow"),
        CheckpointError::GenerationEpochOverflow { previous: maximum },
    );
    assert_eq!(fixture.effect_counter(), 0);
    assert_eq!(
        fixture.authority_snapshot(maximum_run).await,
        maximum_snapshot
    );

    assert_eq!(run, baseline.run().to_owned());
}

async fn checkpoint_payload(bytes: Bytes) -> BudgetedCheckpointBytes {
    let budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: bytes.len().max(1),
    })
    .expect("valid test budget");
    let lease = budget
        .acquire(1, bytes.len())
        .await
        .expect("checkpoint payload budget");
    BudgetedCheckpointBytes::new(bytes, lease).expect("exact payload charge")
}

pub struct CountingParticipant {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    items: u64,
    initialization: ParticipantInitialization,
    released_items: u64,
    commit_notifications: u64,
    prepared_descriptor: Option<ParticipantStateDescriptor>,
    committed_receipt: Option<CommittedParticipantReceipt>,
}

impl CountingParticipant {
    pub fn new(participant_id: &str, items: u64) -> Self {
        Self::for_run(run_id(1), participant_id, items)
    }

    pub fn for_run(run: StreamRunIdentity, participant_id: &str, items: u64) -> Self {
        Self {
            run,
            participant_id: CheckpointParticipantId::new(participant_id),
            items,
            initialization: ParticipantInitialization::default(),
            released_items: 0,
            commit_notifications: 0,
            prepared_descriptor: None,
            committed_receipt: None,
        }
    }

    pub fn released_items(&self) -> u64 {
        self.released_items
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for CountingParticipant {
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
        let bytes = Bytes::from(self.items.to_le_bytes().to_vec());
        let prepared = PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            "test.counting",
            1,
            barrier.cut.clone(),
            self.items,
            checkpoint_payload(bytes).await,
        )?;
        self.prepared_descriptor = Some(prepared.descriptor().clone());
        Ok(prepared)
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        if state.as_ref().is_some_and(|state| state.run() != &self.run) {
            return Err(CheckpointError::ObjectVerification);
        }
        self.initialization.initialize_once()?;
        if let Some(state) = state {
            let bytes: [u8; 8] = state
                .payload_bytes()
                .try_into()
                .map_err(|_| CheckpointError::ObjectVerification)?;
            self.items = u64::from_le_bytes(bytes);
        }
        Ok(())
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        if receipt.participant_id() != &self.participant_id {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        let prepared = self
            .prepared_descriptor
            .as_ref()
            .ok_or(CheckpointError::ObjectVerification)?;
        if receipt.descriptor_digest() != &prepared.digest()?
            || receipt.represented_cut() != &prepared.represented_cut
        {
            return Err(CheckpointError::ObjectVerification);
        }
        if self.committed_receipt.as_ref() == Some(receipt) {
            return Ok(());
        }
        if let Some(committed) = &self.committed_receipt
            && receipt.generation().epoch() <= committed.generation().epoch()
        {
            return Err(CheckpointError::GenerationConflict {
                expected: Some(committed.generation().clone()),
                actual: Some(receipt.generation().clone()),
            });
        }
        self.released_items = self.items;
        self.commit_notifications += 1;
        self.committed_receipt = Some(receipt.clone());
        Ok(())
    }
}

/// Build one real detailed-receipt result partition from a reporter that has
/// classified exactly one record issue.
///
/// The returned payload bytes are the exact canonical encoding the reporter
/// retained, so a caller can assert the committed generation round-trips them.
pub async fn issue_receipt_partition(
    run: StreamRunIdentity,
    epoch: u64,
) -> (
    StreamingResourceBudget,
    PreparedIssueReceiptResultPartition,
    Vec<u8>,
) {
    let reporter_budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 65,
        max_bytes: submission_queue_charge_bytes() + 64 * 1024,
    })
    .expect("valid reporter budget");
    let policy = PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
        StreamingIssueComponentId::new("record_default").expect("valid rule ID"),
        StreamingIssueScopeKind::Record,
        StreamingIssueClass::Permanent,
        None,
        0,
        StreamingIssueDisposition::Quarantine,
        None,
    )
    .expect("valid record rule")])
    .expect("valid record policy");
    let mut reporter = BudgetOwnedStreamingIssueReporter::new(run, policy, reporter_budget.clone())
        .expect("budget-owned reporter");

    let input_domain = StreamingInputDomainIdentity::new(
        ContentDigest::from_bytes([0x21; 32]),
        ImmutableObjectIdentity::from_bytes([0x20; 32]),
    );
    let issue = OrdinaryStreamingIssue::record(
        run,
        input_domain.clone(),
        StableRecordId::from_bytes([0x22; 32]),
        StreamingIssueClass::Permanent,
        ContentDigest::from_bytes([0x33; 32]),
        SourcePosition::new(7),
        0,
        ContentDigest::from_bytes([0x44; 32]),
        OrdinaryStreamingFailure::Format(StreamFormatError::decode(DecodeFailureCode::Syntax)),
    )
    .expect("valid record issue");
    reporter
        .report(IssueSequenceUpdate::Issue(issue))
        .await
        .expect("retain record issue");
    reporter
        .report(IssueSequenceUpdate::NoMoreBefore {
            input_domain,
            through: SourcePosition::new(7),
        })
        .await
        .expect("classify record issue");

    let view = reporter
        .receipt_partition_view(&CheckpointBarrier {
            run,
            epoch: CheckpointEpoch::new(epoch),
            cut: cut_at(epoch),
            plan_digest: ContentDigest::from_bytes([0x55; 32]),
        })
        .await
        .expect("prepare issue receipt partition");
    let payload_bytes = view.payload_bytes().to_vec();
    let descriptor = ResultSegmentDescriptor {
        run,
        epoch: CheckpointEpoch::new(epoch),
        cell_id: CellId::new(0),
        worker_id: WorkerId::new(0),
        projection: ResultProjectionId::new("streaming_issue_receipts")
            .expect("valid issue projection"),
        schema: ResultSchemaVersion::new(2),
        first_sequence: GlobalSequence::new(0),
        last_sequence: GlobalSequence::new(0),
        item_count: 1,
        byte_length: u64::try_from(payload_bytes.len()).expect("small payload"),
        membership_root: *view.receipt_root(),
        payload_digest: ContentDigest::from_bytes(*blake3::hash(&payload_bytes).as_bytes()),
    };
    let descriptor_bytes = std::mem::size_of::<ResultSegmentDescriptor>()
        + descriptor.projection.retained_allocation_bytes();
    let descriptor_budget = StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: descriptor_bytes,
    })
    .expect("valid descriptor budget");
    let descriptor_lease = descriptor_budget
        .acquire(1, descriptor_bytes)
        .await
        .expect("issue descriptor lease");
    let descriptor = BudgetedResultDescriptor::new(descriptor, descriptor_lease)
        .expect("exact issue descriptor charge");
    let handoff = view
        .into_result_partition(descriptor)
        .expect("move issue receipt partition");
    (reporter_budget, handoff, payload_bytes)
}

// ── Task 6C1: deterministic finalization and compaction fixtures ─────────────

use aiperf_runtime::streaming::{
    checkpoint_backend::StreamingCheckpointBackend,
    reliability::{
        PreparedExportAttemptFailure, PreparedExportReceiptPersistence, ResultSinkAttemptOutcome,
        StreamingReliabilityError,
    },
    results::ResultIndexReadBudget,
};

/// Construct the in-memory checkpoint backend used by finalization tests.
pub fn streaming_backend() -> MemoryCheckpointBackend {
    MemoryCheckpointBackend::new(backend_limits()).expect("valid memory checkpoint backend")
}

/// Return the authoritative head generation for one run, when one exists.
pub async fn latest_generation(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
) -> Option<CheckpointGeneration> {
    backend
        .open_latest(&run, &expectations(run))
        .await
        .expect("open latest generation")
        .map(|opened| opened.generation().clone())
}

/// Commit metadata for one final generation with the given terminal reason.
pub fn final_metadata(
    previous: Option<CheckpointGeneration>,
    epoch: u64,
    reason: CheckpointTerminalReason,
) -> CheckpointCommitMetadata {
    CheckpointCommitMetadata {
        previous,
        epoch: CheckpointEpoch::new(epoch),
        cut: cut_at(epoch),
        execution_plan_digest: ContentDigest::from_bytes([0x31; 32]),
        result_plan_digest: ContentDigest::from_bytes([0x32; 32]),
        is_final: true,
        terminal_reason: Some(reason),
    }
}

/// Commit one final generation carrying three distinguishable result segments.
///
/// The segments differ only in projection identity, so the compaction key must
/// order them rather than relying on staging order. A generation with no
/// predecessor is only publishable at the initial epoch, so the fixture owns the
/// epoch rather than accepting one.
pub async fn committed_final_generation(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
) -> CommittedCheckpointGeneration {
    let epoch = 1;
    let mut transaction =
        StreamingCheckpointBackend::begin_generation(backend, run, None, expectations(run))
            .await
            .expect("begin final generation");
    transaction
        .stage_participant(prepared_participant(run, epoch).await)
        .await
        .expect("stage participant");
    let mut partitions = Vec::new();
    // Staged in reverse projection order on purpose.
    for projection in ["zeta_records", "mu_records", "alpha_records"] {
        partitions.push(
            result_partition_with_projection_for(run, epoch, projection)
                .await
                .1,
        );
    }
    transaction
        .stage_results(&mut partitions, &mut None)
        .await
        .expect("stage final results");
    transaction
        .commit(final_metadata(
            None,
            epoch,
            CheckpointTerminalReason::Completed,
        ))
        .await
        .expect("commit final generation")
}

/// Open the leased read authority for one committed generation.
pub async fn open_leased(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
    committed: &CommittedCheckpointGeneration,
) -> LeasedCheckpointGeneration {
    let opened = backend
        .open_latest(&run, &expectations(run))
        .await
        .expect("open latest generation")
        .expect("committed head exists");
    assert_eq!(opened.generation(), committed.generation_ref());
    opened
}

/// Stage a transaction plus metadata for one safe abort at the given epoch.
pub async fn staged_abort_transaction(
    backend: &MemoryCheckpointBackend,
    run: StreamRunIdentity,
    previous: &CommittedCheckpointGeneration,
) -> (
    Box<dyn StreamingGenerationTransaction>,
    CheckpointCommitMetadata,
) {
    let opened = backend
        .open_latest(&run, &expectations(run))
        .await
        .expect("open latest generation")
        .expect("committed head exists");
    let expected = current_v4_predecessor(&opened, previous.generation_ref())
        .expect("current-v4 predecessor authority");
    drop(opened);
    // The backend admits exactly the dense successor epoch.
    let epoch = previous.generation_ref().epoch().get() + 1;
    let mut transaction = StreamingCheckpointBackend::begin_generation(
        backend,
        run,
        Some(expected),
        expectations(run),
    )
    .await
    .expect("begin abort generation");
    transaction
        .stage_participant(prepared_participant(run, epoch).await)
        .await
        .expect("stage participant");
    (
        transaction,
        final_metadata(
            Some(previous.generation()),
            epoch,
            CheckpointTerminalReason::Aborted,
        ),
    )
}

/// Bounded result-index page budget holding at most `items` descriptors.
pub fn page_budget(items: usize) -> ResultIndexReadBudget {
    ResultIndexReadBudget {
        max_items: NonZeroUsize::new(items).expect("nonzero page items"),
        max_bytes: NonZeroU64::new(1_048_576).expect("nonzero page bytes"),
    }
}

/// Budget the prepared report's retained lease is charged against.
pub fn report_budget() -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: 4,
        max_bytes: 256 * 1024,
    })
    .expect("valid report budget")
}

/// Budget for one export receipt's encoded and parsed charges.
pub fn export_budget() -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: 8,
        max_bytes: 64 * 1024,
    })
    .expect("valid export budget")
}

/// Budget admitting several derived-sink attempt tokens.
pub fn sink_attempt_budget() -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: 8,
        max_bytes: 4096,
    })
    .expect("valid attempt budget")
}

/// Budget that cannot admit even one derived-sink attempt token.
pub fn exhausted_attempt_budget() -> StreamingResourceBudget {
    StreamingResourceBudget::new(BudgetLimits {
        max_items: 1,
        max_bytes: 1,
    })
    .expect("valid exhausted attempt budget")
}

/// Construct a checked lowercase component identity.
pub fn component(value: &str) -> StreamingIssueComponentId {
    StreamingIssueComponentId::new(value).expect("valid component identity")
}

/// Frozen export policy retrying a retryable export failure three times.
pub fn export_policy() -> PreparedStreamingIssuePolicy {
    PreparedStreamingIssuePolicy::new([StreamingIssueThresholdRule::new(
        component("export_retryable"),
        StreamingIssueScopeKind::Export,
        StreamingIssueClass::Retryable,
        None,
        3,
        StreamingIssueDisposition::ExportIncomplete,
        None,
    )
    .expect("valid retryable export rule")])
    .expect("valid export policy")
}

/// Prepare one export failure, allowing the issue-side and call-side authority
/// to diverge so foreign run, generation, sink, and ordinal are reachable.
#[allow(clippy::too_many_arguments)]
pub async fn try_prepared_export_failure_for(
    reporter_run: StreamRunIdentity,
    issue_run: StreamRunIdentity,
    issue_generation: &CheckpointGeneration,
    issue_sink: &StreamingIssueComponentId,
    issue_ordinal: u32,
    call_run: StreamRunIdentity,
    call_generation: &CheckpointGeneration,
    call_sink: &StreamingIssueComponentId,
    call_ordinal: u32,
    budget: &StreamingResourceBudget,
) -> Result<PreparedExportAttemptFailure, StreamingReliabilityError> {
    let mut reporter = BudgetOwnedStreamingIssueReporter::new(
        reporter_run,
        export_policy(),
        StreamingResourceBudget::new(BudgetLimits {
            max_items: 64,
            max_bytes: 128 * 1024,
        })
        .expect("valid reporter budget"),
    )
    .expect("budget-owned reporter");
    let issue = OrdinaryStreamingIssue::export(
        issue_run,
        issue_sink.clone(),
        issue_generation.clone(),
        StreamingIssueClass::Retryable,
        ContentDigest::from_bytes([0xc3; 32]),
        issue_ordinal,
        ContentDigest::from_bytes([0xc4; 32]),
        OrdinaryStreamingFailure::Export(ResultExportError::failure(ResultExportFailureCode::Io)),
    )
    .expect("valid export issue");
    reporter
        .prepare_export_attempt_failure(
            &call_run,
            call_generation,
            call_sink,
            call_ordinal,
            ResultSinkAttemptOutcome::Failed(issue),
            budget,
        )
        .await
}

/// Prepare one well-formed export failure at the given dense ordinal.
pub async fn prepared_export_failure(
    run: StreamRunIdentity,
    committed: &CommittedCheckpointGeneration,
    sink: &StreamingIssueComponentId,
    attempt_ordinal: u32,
    budget: &StreamingResourceBudget,
) -> PreparedExportAttemptFailure {
    let generation = committed.generation();
    try_prepared_export_failure_for(
        run,
        run,
        &generation,
        sink,
        attempt_ordinal,
        run,
        &generation,
        sink,
        attempt_ordinal,
        budget,
    )
    .await
    .expect("prepare export failure")
}

/// Prepare one well-formed export failure and consume it into persistence.
pub async fn prepared_export_persistence(
    run: StreamRunIdentity,
    committed: &CommittedCheckpointGeneration,
    sink: &StreamingIssueComponentId,
    attempt_ordinal: u32,
    budget: &StreamingResourceBudget,
) -> PreparedExportReceiptPersistence {
    prepared_export_failure(run, committed, sink, attempt_ordinal, budget)
        .await
        .into_persistence()
}

use aiperf_runtime::streaming::{
    identity::StableActionId,
    results::{
        CheckpointDeliveryMode, DeliveryClaim, DeliveryCrashPoint, DeliveryRestartDecision,
        DeliveryRestartError, DeliveryRestartRequest, DeliveryTopologyBinding, DuplicateWindow,
        OutstandingAction, OutstandingActionState, TargetIdempotencyCapability,
        deliver_restart_decision,
    },
};

/// Committed cut sequence the delivery fixture resumes from.
const DELIVERY_CUT_SEQUENCE: u64 = 16;

fn delivery_binding() -> DeliveryTopologyBinding {
    DeliveryTopologyBinding {
        topology_digest: ContentDigest::from_bytes([0x71; 32]),
        projection: ResultProjectionId::new("aiperf.records.exact").expect("nonempty projection"),
        membership_scheme_digest: ContentDigest::from_bytes([0x72; 32]),
    }
}

fn delivery_action(tag: u8) -> StableActionId {
    StableActionId::from_bytes([tag; 32])
}

/// One restart of the delivery fixture, holding its derived decision.
pub struct RestoredDelivery {
    decision: DeliveryRestartDecision,
}

impl RestoredDelivery {
    /// Whether every re-emitted logical action appears exactly once.
    pub fn logical_membership_is_unique(&self) -> bool {
        self.decision.logical_membership_is_unique()
    }

    /// Return the claim published by the resumed run.
    pub fn claim(&self) -> DeliveryClaim {
        self.decision.claim
    }

    /// Return what this restart leaves possible at the target.
    pub fn duplicate_window(&self) -> DuplicateWindow {
        self.decision.duplicate_window
    }

    /// Borrow the actions the restart re-emits.
    pub fn reissue(&self) -> &[StableActionId] {
        &self.decision.reissue
    }
}

/// A run frozen at one delivery mode and target idempotency capability.
pub struct DeliveryFixture {
    mode: CheckpointDeliveryMode,
    capability: TargetIdempotencyCapability,
    cut: CheckpointCut,
    binding: DeliveryTopologyBinding,
}

impl DeliveryFixture {
    /// Outstanding set a dead incarnation leaves at the given crash point.
    ///
    /// Two never-dispatched actions are always outstanding: a restart re-derives
    /// its undispatched suffix at every crash point, and they give the logical
    /// membership assertion something to be unique about.
    fn outstanding(crash: DeliveryCrashPoint) -> Vec<OutstandingAction> {
        let mut outstanding = vec![
            OutstandingAction {
                action: delivery_action(0xa1),
                sequence: GlobalSequence::new(DELIVERY_CUT_SEQUENCE + 3),
                state: OutstandingActionState::NotDispatched,
            },
            OutstandingAction {
                action: delivery_action(0xa2),
                sequence: GlobalSequence::new(DELIVERY_CUT_SEQUENCE + 4),
                state: OutstandingActionState::NotDispatched,
            },
        ];
        let uncertain = match crash {
            DeliveryCrashPoint::BeforeDispatch | DeliveryCrashPoint::AfterCommit => None,
            DeliveryCrashPoint::AfterDispatchBeforeTerminal => {
                Some(OutstandingActionState::AdmittedNotTerminal)
            }
            DeliveryCrashPoint::AfterTerminalBeforeCommit => {
                Some(OutstandingActionState::TerminalUncommitted)
            }
        };
        if let Some(state) = uncertain {
            outstanding.push(OutstandingAction {
                action: delivery_action(0xb1),
                sequence: GlobalSequence::new(DELIVERY_CUT_SEQUENCE + 1),
                state,
            });
        }
        outstanding
    }

    fn request<'a>(
        &'a self,
        outstanding: &'a [OutstandingAction],
        restarting: &'a DeliveryTopologyBinding,
    ) -> DeliveryRestartRequest<'a> {
        DeliveryRestartRequest {
            mode: self.mode,
            capability: self.capability,
            cut: self.mode.has_authoritative_results().then_some(&self.cut),
            result_index_root: self
                .mode
                .has_authoritative_results()
                .then(|| ContentDigest::from_bytes([0x73; 32])),
            committed_binding: &self.binding,
            restarting_binding: restarting,
            outstanding,
        }
    }

    /// Kill the incarnation at the given point and derive the restart decision.
    pub fn crash_and_restore(&self, crash: DeliveryCrashPoint) -> RestoredDelivery {
        let outstanding = Self::outstanding(crash);
        let restarting = delivery_binding();
        let decision = deliver_restart_decision(&self.request(&outstanding, &restarting))
            .expect("identical binding restart is admissible");
        RestoredDelivery { decision }
    }

    /// Restart under a binding the caller mutates, returning the raw outcome.
    pub fn restart_with_binding(
        &self,
        mutate: impl FnOnce(&mut DeliveryTopologyBinding),
    ) -> Result<DeliveryRestartDecision, DeliveryRestartError> {
        let outstanding = Self::outstanding(DeliveryCrashPoint::AfterDispatchBeforeTerminal);
        let mut restarting = delivery_binding();
        mutate(&mut restarting);
        deliver_restart_decision(&self.request(&outstanding, &restarting))
    }
}

/// Freeze one delivery fixture at the given mode and target capability.
pub fn delivery_fixture(
    mode: CheckpointDeliveryMode,
    capability: TargetIdempotencyCapability,
) -> DeliveryFixture {
    DeliveryFixture {
        mode,
        capability,
        cut: cut_at(DELIVERY_CUT_SEQUENCE),
        binding: delivery_binding(),
    }
}

// ── Task 5F2: conditional object-store checkpoint fixtures ───────────────────

#[cfg(feature = "streaming-s3")]
pub use object_store_support::*;

#[cfg(feature = "streaming-s3")]
mod object_store_support {
    use super::{
        commit_with_segment_on, current_v4_predecessor, expectations, legacy_fixture_budget,
        legacy_v3_fixture, prepared_participant, run_id, PublicationAuthoritySnapshot,
        PublicationBackendFixture,
    };
    use aiperf_runtime::streaming::{
        budget::{BudgetLimits, StreamingResourceBudget},
        checkpoint::{
            CheckpointEpoch, CheckpointError, CheckpointGeneration, CommittedCheckpointGeneration,
            StreamRunIdentity,
        },
        checkpoint_backend::{
            CheckpointGenerationExpectations, StreamingCheckpointBackend,
            StreamingGenerationTransaction,
        },
        checkpoints::object_store::{
            immutable_object_key, stale_writer_error, BudgetOwnedObjectChunk,
            BudgetOwnedObjectPage, BudgetOwnedObjectReader, ConditionalObjectStore,
            ObjectCheckpointBackend, ObjectCheckpointLimits, ObjectKey, ObjectListBudget,
            ObjectListCursor, ObjectMetadata, ObjectReadBudget, ObjectReadRange, ObjectVersion,
            PointerObject,
        },
        identity::ContentDigest,
    };
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::{
        cell::{Cell, RefCell},
        collections::{BTreeMap, BTreeSet},
        num::NonZeroUsize,
        rc::Rc,
    };

    /// Prefix every object-store fixture writes under.
    pub fn object_test_prefix() -> ObjectKey {
        ObjectKey::new("aiperf-test/checkpoints")
    }

    fn nonzero(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("nonzero test bound")
    }

    /// Limits generous enough that only deliberate cases hit a bound.
    pub fn object_backend_limits() -> ObjectCheckpointLimits {
        let limits = BudgetLimits {
            max_items: 256,
            max_bytes: 1_048_576,
        };
        ObjectCheckpointLimits {
            transactions: limits,
            prepared_indexes: limits,
            storage: limits,
            result_summaries: limits,
            reads: limits,
            max_chunk_bytes: nonzero(64 * 1024),
            list: ObjectListBudget {
                max_items: nonzero(256),
                max_metadata_bytes: nonzero(1_048_576),
            },
        }
    }

    pub fn object_io_budget(bytes: usize) -> usize {
        bytes
    }

    pub fn read_budget(max_chunk_bytes: usize) -> ObjectReadBudget {
        ObjectReadBudget { max_chunk_bytes }
    }

    /// Build one backend over the supplied fake store.
    pub fn object_backend(store: FakeConditionalObjectStore) -> ObjectCheckpointBackend {
        ObjectCheckpointBackend::new(Rc::new(store), object_test_prefix(), object_backend_limits())
            .expect("valid object checkpoint backend")
    }

    #[derive(Default)]
    struct FakeState {
        objects: BTreeMap<ObjectKey, (ObjectVersion, Bytes)>,
        read_progress: BTreeMap<(ObjectKey, ObjectVersion), u64>,
        verified: BTreeSet<(ObjectKey, ObjectVersion)>,
        next_version: u64,
    }

    /// Deterministic in-process conditional object store.
    ///
    /// Only the properties the backend contract depends on are modelled: exact
    /// conditional pointer replacement, content-addressed immutable writes, and
    /// bounded ranged reads. Everything else — durability, latency, eventual
    /// consistency — is deliberately absent so a failing test names a contract
    /// violation rather than a simulation artifact.
    #[derive(Clone)]
    pub struct FakeConditionalObjectStore {
        state: Rc<RefCell<FakeState>>,
        retention: StreamingResourceBudget,
        allocated: Rc<Cell<usize>>,
        cas_calls: Rc<Cell<u64>>,
        effects: Rc<Cell<u64>>,
        uploads: Rc<Cell<u64>>,
        fail_upload_at: Rc<Cell<Option<u64>>>,
        declared_length: Option<u64>,
    }

    impl std::fmt::Debug for FakeConditionalObjectStore {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter
                .debug_struct("FakeConditionalObjectStore")
                .field("objects", &self.state.borrow().objects.len())
                .finish()
        }
    }

    impl FakeConditionalObjectStore {
        pub fn new(retained_bytes: usize) -> Self {
            Self {
                state: Rc::new(RefCell::new(FakeState::default())),
                retention: StreamingResourceBudget::new(BudgetLimits {
                    max_items: 1_024,
                    max_bytes: retained_bytes.max(1),
                })
                .expect("valid fake retention budget"),
                allocated: Rc::new(Cell::new(0)),
                cas_calls: Rc::new(Cell::new(0)),
                effects: Rc::new(Cell::new(0)),
                uploads: Rc::new(Cell::new(0)),
                fail_upload_at: Rc::new(Cell::new(None)),
                declared_length: None,
            }
        }

        /// Seed one pointer whose listed length is a hostile declaration.
        pub fn declaring_length(length: usize) -> Self {
            let store = Self::new(64 * 1024);
            let key = ObjectKey::new(format!("{}/pointers/hostile", object_test_prefix().as_str()));
            store.state.borrow_mut().objects.insert(
                key,
                (ObjectVersion::new("v-hostile"), Bytes::from_static(b"{}")),
            );
            Self {
                declared_length: Some(u64::try_from(length).unwrap_or(u64::MAX)),
                ..store
            }
        }

        pub fn allocated_bytes(&self) -> usize {
            self.allocated.get()
        }

        pub fn pointer_cas_calls(&self) -> u64 {
            self.cas_calls.get()
        }

        pub fn effects(&self) -> u64 {
            self.effects.get()
        }

        pub fn reset_counters(&self) {
            self.allocated.set(0);
            self.cas_calls.set(0);
            self.effects.set(0);
        }

        pub fn arm_upload_failure(&self, nth: u64) {
            self.fail_upload_at.set(Some(nth));
        }

        pub fn upload_attempts(&self) -> u64 {
            self.uploads.get()
        }

        /// Render every retained object and pointer for snapshot comparison.
        pub fn state_fingerprint(&self) -> String {
            let state = self.state.borrow();
            state
                .objects
                .iter()
                .map(|(key, (version, bytes))| {
                    format!("{}@{}#{}", key.as_str(), version.as_str(), bytes.len())
                })
                .collect::<Vec<_>>()
                .join(",")
        }

        /// Every object the current pointer names must have been read back whole.
        pub fn current_pointer_references_only_verified_objects(&self) -> bool {
            let state = self.state.borrow();
            let pointer_prefix = format!("{}/pointers/", object_test_prefix().as_str());
            let mut pointers = state
                .objects
                .iter()
                .filter(|(key, _)| key.as_str().starts_with(&pointer_prefix));
            let Some((_, (_, bytes))) = pointers.next() else {
                return false;
            };
            let Ok(document) = serde_json::from_slice::<serde_json::Value>(bytes) else {
                return false;
            };
            let (Some(key), Some(version)) = (
                document.get("generation_object").and_then(|v| v.as_str()),
                document.get("generation_version").and_then(|v| v.as_str()),
            ) else {
                return false;
            };
            let named = (ObjectKey::new(key), ObjectVersion::new(version));
            state.objects.contains_key(&named.0) && state.verified.contains(&named)
        }

        fn note_effect(&self) {
            self.effects.set(self.effects.get().saturating_add(1));
        }

        fn mint_version(&self, state: &mut FakeState) -> ObjectVersion {
            state.next_version += 1;
            ObjectVersion::new(format!("v{}", state.next_version))
        }
    }

    #[async_trait(?Send)]
    impl ConditionalObjectStore for FakeConditionalObjectStore {
        async fn put_immutable(
            &self,
            mut object: Box<dyn BudgetOwnedObjectReader>,
        ) -> Result<ObjectVersion, CheckpointError> {
            self.note_effect();
            let attempt = self.uploads.get() + 1;
            self.uploads.set(attempt);
            if self.fail_upload_at.get() == Some(attempt) {
                return Err(CheckpointError::Storage {
                    message: format!("injected object upload fault on attempt {attempt}"),
                });
            }
            let digest = object.content_digest();
            let declared = object.content_length();
            let mut assembled = Vec::new();
            while let Some(chunk) = object.next_chunk(64 * 1024).await? {
                assembled.extend_from_slice(&chunk.bytes);
            }
            if u64::try_from(assembled.len()).unwrap_or(u64::MAX) != declared
                || ContentDigest::from_bytes(*blake3::hash(&assembled).as_bytes()) != digest
            {
                return Err(CheckpointError::ObjectVerification);
            }
            let key = immutable_object_key(&object_test_prefix(), &digest);
            let mut state = self.state.borrow_mut();
            if let Some((version, existing)) = state.objects.get(&key) {
                // Content-addressed writes are idempotent; a differing body under
                // the same address would be a digest collision, not a rewrite.
                if existing.as_ref() != assembled.as_slice() {
                    return Err(CheckpointError::ObjectVerification);
                }
                return Ok(version.clone());
            }
            let version = self.mint_version(&mut state);
            state.objects.insert(
                key,
                (
                    version.clone(),
                    Bytes::from(assembled.into_boxed_slice()),
                ),
            );
            Ok(version)
        }

        async fn compare_and_swap_pointer(
            &self,
            key: &ObjectKey,
            expected: Option<&ObjectVersion>,
            next: PointerObject,
        ) -> Result<ObjectVersion, CheckpointError> {
            self.note_effect();
            self.cas_calls.set(self.cas_calls.get() + 1);
            let mut state = self.state.borrow_mut();
            let current = state.objects.get(key).map(|(version, _)| version.clone());
            if current.as_ref() != expected {
                return Err(stale_writer_error());
            }
            let version = self.mint_version(&mut state);
            state
                .objects
                .insert(key.clone(), (version.clone(), next.bytes));
            drop(next.lease);
            Ok(version)
        }

        async fn get_version_range(
            &self,
            key: &ObjectKey,
            version: &ObjectVersion,
            range: ObjectReadRange,
            budget: ObjectReadBudget,
        ) -> Result<BudgetOwnedObjectChunk, CheckpointError> {
            self.note_effect();
            let length = usize::try_from(range.length)
                .map_err(|_| CheckpointError::ObjectVerification)?;
            if length == 0 || length > budget.max_chunk_bytes {
                return Err(CheckpointError::ObjectVerification);
            }
            let bytes = {
                let state = self.state.borrow();
                let (stored_version, bytes) = state
                    .objects
                    .get(key)
                    .ok_or(CheckpointError::ObjectVerification)?;
                if stored_version != version {
                    return Err(CheckpointError::ObjectVerification);
                }
                let start = usize::try_from(range.offset)
                    .map_err(|_| CheckpointError::ObjectVerification)?;
                if start >= bytes.len() {
                    return Err(CheckpointError::ObjectVerification);
                }
                bytes.slice(start..bytes.len().min(start + length))
            };
            let lease = self
                .retention
                .acquire(1, bytes.len())
                .await
                .map_err(|_| CheckpointError::ObjectVerification)?;
            self.allocated
                .set(self.allocated.get().saturating_add(bytes.len()));
            let mut state = self.state.borrow_mut();
            let identity = (key.clone(), version.clone());
            let total = state
                .objects
                .get(key)
                .map(|(_, stored)| stored.len() as u64)
                .unwrap_or_default();
            let progress = state.read_progress.entry(identity.clone()).or_default();
            if *progress == range.offset {
                *progress += bytes.len() as u64;
                if *progress >= total {
                    state.verified.insert(identity);
                }
            }
            Ok(BudgetOwnedObjectChunk { bytes, lease })
        }

        async fn list_versions(
            &self,
            prefix: &ObjectKey,
            _cursor: Option<&ObjectListCursor>,
            budget: ObjectListBudget,
        ) -> Result<BudgetOwnedObjectPage, CheckpointError> {
            self.note_effect();
            let entries: Vec<ObjectMetadata> = {
                let state = self.state.borrow();
                state
                    .objects
                    .iter()
                    .filter(|(key, _)| key.as_str().starts_with(prefix.as_str()))
                    .take(budget.max_items.get())
                    .map(|(key, (version, bytes))| ObjectMetadata {
                        key: key.clone(),
                        version: version.clone(),
                        byte_length: self
                            .declared_length
                            .unwrap_or_else(|| bytes.len() as u64),
                    })
                    .collect()
            };
            let lease = self
                .retention
                .acquire(entries.len(), entries.len() * std::mem::size_of::<ObjectMetadata>())
                .await
                .map_err(|_| CheckpointError::ObjectVerification)?;
            Ok(BudgetOwnedObjectPage {
                objects: entries.into_boxed_slice(),
                next: None,
                lease,
            })
        }

        async fn delete_version(
            &self,
            key: &ObjectKey,
            version: &ObjectVersion,
        ) -> Result<(), CheckpointError> {
            self.note_effect();
            let mut state = self.state.borrow_mut();
            match state.objects.get(key) {
                Some((stored, _)) if stored == version => {
                    state.objects.remove(key);
                    Ok(())
                }
                _ => Err(stale_writer_error()),
            }
        }
    }

    /// Stage one transaction ready to commit at `epoch`.
    pub async fn prepared_transaction(
        backend: &ObjectCheckpointBackend,
        previous: Option<CheckpointGeneration>,
        epoch: u64,
    ) -> Box<dyn StreamingGenerationTransaction> {
        let run = run_id(1);
        let expected = match previous.as_ref() {
            None => None,
            Some(previous) => {
                let opened = backend
                    .open_latest(&run, &expectations(run))
                    .await
                    .expect("open object head")
                    .expect("object head exists");
                Some(current_v4_predecessor(&opened, previous).expect("verified predecessor"))
            }
        };
        let mut transaction = backend
            .begin_generation(run, expected, expectations(run))
            .await
            .expect("begin object transaction");
        transaction
            .stage_participant(prepared_participant(run, epoch).await)
            .await
            .expect("stage object participant");
        transaction
            .stage_results(&mut Vec::new(), &mut None)
            .await
            .expect("stage object result epoch");
        transaction
    }

    /// Publication-conformance fixture over the conditional object store.
    pub struct ObjectPublicationBackendFixture {
        backend: ObjectCheckpointBackend,
        store: FakeConditionalObjectStore,
        run: StreamRunIdentity,
    }

    pub fn object_publication_backend_fixture() -> ObjectPublicationBackendFixture {
        let store = FakeConditionalObjectStore::new(1_048_576);
        ObjectPublicationBackendFixture {
            backend: object_backend(store.clone()),
            store,
            run: run_id(1),
        }
    }

    #[async_trait(?Send)]
    impl PublicationBackendFixture for ObjectPublicationBackendFixture {
        fn run(&self) -> StreamRunIdentity {
            self.run
        }

        async fn seed_baseline(&self) -> CommittedCheckpointGeneration {
            commit_with_segment_on(&self.backend, self.run, None, 1)
                .await
                .expect("seed object baseline generation")
        }

        async fn seed_maximum_epoch(&self) -> (StreamRunIdentity, CommittedCheckpointGeneration) {
            let run = run_id(2);
            let committed = self
                .backend
                .seed_nonempty_committed_generation_at_epoch(
                    run,
                    CheckpointEpoch::new(u64::MAX),
                    &expectations(run),
                    vec![prepared_participant(run, u64::MAX).await],
                )
                .await
                .expect("seed maximum object head in fresh run");
            (run, committed)
        }

        async fn staged_after(
            &self,
            run: StreamRunIdentity,
            expected: CheckpointGeneration,
            participant_epoch: u64,
        ) -> Box<dyn StreamingGenerationTransaction> {
            let opened = self
                .backend
                .open_latest(&run, &expectations(run))
                .await
                .expect("open object lineage head")
                .expect("object lineage head exists");
            let predecessor = current_v4_predecessor(&opened, &expected)
                .expect("verified current-v4 predecessor");
            let mut transaction = self
                .backend
                .begin_generation(run, Some(predecessor), expectations(run))
                .await
                .expect("begin object lineage transaction");
            transaction
                .stage_participant(prepared_participant(run, participant_epoch).await)
                .await
                .expect("stage object lineage participant");
            transaction
                .stage_results(&mut Vec::new(), &mut None)
                .await
                .expect("stage object lineage result epoch");
            transaction
        }

        async fn authority_snapshot(&self, run: StreamRunIdentity) -> PublicationAuthoritySnapshot {
            let generation = self
                .backend
                .open_latest(&run, &expectations(run))
                .await
                .expect("open object head")
                .map(|reader| reader.generation().clone());
            PublicationAuthoritySnapshot::new(generation, self.store.state_fingerprint())
        }

        fn reset_effect_counter(&self) {
            self.store.reset_counters();
        }

        fn effect_counter(&self) -> u64 {
            self.store.effects()
        }
    }

    /// One object backend whose only head is a verified legacy-v3 generation.
    pub struct LegacyObjectHeadFixture {
        pub backend: ObjectCheckpointBackend,
        pub store: FakeConditionalObjectStore,
        pub run: StreamRunIdentity,
        pub expectations: CheckpointGenerationExpectations,
        _budget: StreamingResourceBudget,
    }

    pub async fn object_backend_with_legacy_v3_head() -> LegacyObjectHeadFixture {
        let run = run_id(1);
        let store = FakeConditionalObjectStore::new(1_048_576);
        let backend = object_backend(store.clone());
        let budget = legacy_fixture_budget();
        let (fixture, _) = legacy_v3_fixture(&budget, run).await;
        backend
            .import_legacy_v3_read_only_fixture(fixture)
            .await
            .expect("import legacy-v3 head");
        // The import itself replaced the pointer; the assertion under test is
        // about successor writes, so seeding must not be counted.
        store.reset_counters();
        LegacyObjectHeadFixture {
            backend,
            store,
            run,
            expectations: expectations(run),
            _budget: budget,
        }
    }
}
