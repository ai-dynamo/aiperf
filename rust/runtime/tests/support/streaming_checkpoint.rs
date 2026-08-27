// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::streaming::{
    budget::{BudgetLimits, StreamingResourceBudget},
    checkpoint::{
        AcquisitionHorizon, AdmissionHorizon, BudgetedCheckpointBytes, CheckpointBarrier,
        CheckpointCut, CheckpointEpoch, CheckpointError, CheckpointGeneration,
        CheckpointParticipantId, CommittedCheckpointGeneration, CommittedParticipantReceipt,
        CommittedParticipantState, DecodeHorizon, DiscoveryHorizon, EventTimeWatermark,
        OrderedActionHorizon, ParticipantInitialization, ParticipantStateDescriptor,
        PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
        TerminalActionHorizon,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations, StreamingGenerationTransaction,
    },
    checkpoints::memory::{
        ImmutableObjectInventory, MemoryCheckpointBackend, MemoryCheckpointLimits,
        MemoryLiveBudgetUsage,
    },
    identity::{ContentDigest, GlobalSequence, LogicalReplayRunId, SessionCausalFrontier},
    results::{
        BudgetedResultDescriptor, CellId, ResultPartition, ResultProjectionId, ResultSchemaVersion,
        ResultSegmentDescriptor, WorkerId,
    },
    unit::{EventTimeUtc, SourcePosition},
};
use async_trait::async_trait;
use bytes::Bytes;

pub fn cut_at(value: u64) -> CheckpointCut {
    CheckpointCut {
        discovered: DiscoveryHorizon::new(SourcePosition::new(value)),
        acquired: AcquisitionHorizon::new(SourcePosition::new(value)),
        decoded: DecodeHorizon::new(SourcePosition::new(value)),
        ordered: OrderedActionHorizon::new(GlobalSequence::new(value)),
        admitted: AdmissionHorizon::new(GlobalSequence::new(value)),
        terminal: TerminalActionHorizon::new(GlobalSequence::new(value)),
        event_watermark: EventTimeWatermark::Hard {
            through: EventTimeUtc::new(value as i64).expect("non-negative test event time"),
        },
        causal_frontier: SessionCausalFrontier {
            through_sequence: GlobalSequence::new(value),
            event_time: Some(
                EventTimeUtc::new(value as i64).expect("non-negative test event time"),
            ),
            digest: ContentDigest::from_bytes([value as u8; 32]),
        },
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
    let mut transaction = backend
        .begin_generation(run, previous.clone(), expectations(run))
        .await?;
    transaction
        .stage_participant(prepared_participant(run, epoch).await)
        .await?;
    transaction.stage_results(&mut Vec::new()).await?;
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
    let mut transaction = backend
        .begin_generation(run, previous.clone(), expectations(run))
        .await?;
    transaction
        .stage_participant(prepared_participant(run, epoch).await)
        .await?;
    let mut partitions = vec![result_partition(run, epoch).await];
    transaction.stage_results(&mut partitions).await?;
    transaction
        .commit(metadata_with_lineage(previous, epoch))
        .await
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
    async fn staged_after(
        &self,
        expected: CheckpointGeneration,
        participant_epoch: u64,
    ) -> Box<dyn StreamingGenerationTransaction>;
    async fn authority_snapshot(&self) -> PublicationAuthoritySnapshot;
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

    async fn staged_after(
        &self,
        expected: CheckpointGeneration,
        participant_epoch: u64,
    ) -> Box<dyn StreamingGenerationTransaction> {
        let mut transaction = self
            .backend
            .begin_generation(self.run, Some(expected), expectations(self.run))
            .await
            .expect("begin lineage transaction");
        transaction
            .stage_participant(prepared_participant(self.run, participant_epoch).await)
            .await
            .expect("stage lineage participant");
        transaction
            .stage_results(&mut Vec::new())
            .await
            .expect("stage lineage result epoch");
        Box::new(transaction)
    }

    async fn authority_snapshot(&self) -> PublicationAuthoritySnapshot {
        let generation = self
            .backend
            .open_latest(&self.run, &expectations(self.run))
            .await
            .expect("open memory head")
            .map(|reader| reader.generation().generation());
        PublicationAuthoritySnapshot {
            generation,
            inventory: self.backend.immutable_object_inventory(&self.run),
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
    let baseline_snapshot = fixture.authority_snapshot().await;

    let wrong = fixture.staged_after(baseline.generation(), 2).await;
    fixture.reset_effect_counter();
    assert_eq!(
        wrong
            .commit(metadata_with_lineage(None, 2))
            .await
            .expect_err("wrong predecessor must fail"),
        CheckpointError::ObjectVerification,
    );
    assert_eq!(fixture.effect_counter(), 0);
    assert_eq!(fixture.authority_snapshot().await, baseline_snapshot);

    let skipped = fixture.staged_after(baseline.generation(), 2).await;
    fixture.reset_effect_counter();
    assert_eq!(
        skipped
            .commit(metadata_with_lineage(Some(baseline.generation()), 3))
            .await
            .expect_err("skipped epoch must fail"),
        CheckpointError::ObjectVerification,
    );
    assert_eq!(fixture.effect_counter(), 0);
    assert_eq!(fixture.authority_snapshot().await, baseline_snapshot);

    let maximum = CheckpointGeneration::new(
        CheckpointEpoch::new(u64::MAX),
        ContentDigest::from_bytes([0xfe; 32]),
    );
    let overflow = fixture.staged_after(maximum.clone(), 1).await;
    let mut overflow_metadata = metadata_at(1);
    overflow_metadata.previous = Some(maximum.clone());
    overflow_metadata.epoch = CheckpointEpoch::new(u64::MAX);
    fixture.reset_effect_counter();
    assert_eq!(
        overflow
            .commit(overflow_metadata)
            .await
            .expect_err("maximum predecessor must overflow"),
        CheckpointError::GenerationEpochOverflow { previous: maximum },
    );
    assert_eq!(fixture.effect_counter(), 0);
    assert_eq!(fixture.authority_snapshot().await, baseline_snapshot);

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
