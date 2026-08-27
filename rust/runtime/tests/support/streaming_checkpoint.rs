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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicationAuthoritySnapshot {
    generation: Option<CheckpointGeneration>,
    inventory: ImmutableObjectInventory,
    usage: MemoryLiveBudgetUsage,
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
        PublicationAuthoritySnapshot {
            generation,
            inventory: self.backend.immutable_object_inventory(&run),
            usage: self.backend.live_budget_usage(),
        }
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
