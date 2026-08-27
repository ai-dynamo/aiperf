// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-local atomic in-memory checkpoint backend reference.

use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};

use async_trait::async_trait;
use bytes::Bytes;

use crate::streaming::{
    budget::{BudgetError, BudgetLease, BudgetLimits, BudgetSnapshot, StreamingResourceBudget},
    checkpoint::{
        BudgetedCheckpointBytes, CheckpointBackendBudgetFailureCode, CheckpointBackendBudgetKind,
        CheckpointError, CheckpointGeneration, CheckpointGenerationCandidate,
        CommittedCheckpointGeneration, CommittedParticipantState, ParticipantStateDescriptor,
        PreparedParticipantState, PrevalidatedCheckpointGenerationCandidate, StreamRunIdentity,
    },
    checkpoint_backend::{
        CheckpointCommitMetadata, CheckpointGenerationExpectations,
        FrozenGenerationTransactionInputs, LeasedGenerationReader, StreamingCheckpointBackend,
        StreamingGenerationTransaction, build_prevalidated_candidate, validate_commit_metadata,
    },
    identity::ContentDigest,
    results::{
        BudgetedResultDescriptors, PreparedResultEpoch, ResultIndexCursor, ResultIndexPage,
        ResultIndexReadBudget, ResultPartition, ResultSegmentDescriptor, ResultSegmentReader,
        canonical_result_index_root, descriptor_retained_bytes, result_totals,
    },
};

/// Capacity limits for each independently owned memory-backend resource.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryCheckpointLimits {
    /// Simultaneously live generation transactions.
    pub transactions: BudgetLimits,
    /// Descriptors retained by staged transaction indexes.
    pub prepared_indexes: BudgetLimits,
    /// Immutable committed object storage.
    pub storage: BudgetLimits,
    /// Descriptor summaries returned from result staging.
    pub result_summaries: BudgetLimits,
    /// Concurrent generation, participant, result, and page readers.
    pub reads: BudgetLimits,
}

#[derive(Clone, Debug)]
struct BackendBudget {
    kind: CheckpointBackendBudgetKind,
    limits: BudgetLimits,
    resource: StreamingResourceBudget,
}

impl BackendBudget {
    fn new(
        kind: CheckpointBackendBudgetKind,
        limits: BudgetLimits,
    ) -> Result<Self, CheckpointError> {
        if limits.max_items == 0 {
            return Err(backend_error(
                kind,
                CheckpointBackendBudgetFailureCode::ItemCapacity,
            ));
        }
        if limits.max_bytes == 0 {
            return Err(backend_error(
                kind,
                CheckpointBackendBudgetFailureCode::ByteCapacity,
            ));
        }
        let resource = StreamingResourceBudget::new(limits)
            .map_err(|error| map_budget_error(kind, limits, 0, 0, error))?;
        Ok(Self {
            kind,
            limits,
            resource,
        })
    }

    async fn acquire(&self, items: usize, bytes: usize) -> Result<BudgetLease, CheckpointError> {
        if items > self.limits.max_items {
            return Err(backend_error(
                self.kind,
                CheckpointBackendBudgetFailureCode::ItemCapacity,
            ));
        }
        if bytes > self.limits.max_bytes {
            return Err(backend_error(
                self.kind,
                CheckpointBackendBudgetFailureCode::ByteCapacity,
            ));
        }
        self.resource
            .acquire(items, bytes)
            .await
            .map_err(|error| map_budget_error(self.kind, self.limits, items, bytes, error))
    }

    fn snapshot(&self) -> BudgetSnapshot {
        self.resource.snapshot()
    }
}

fn backend_error(
    budget: CheckpointBackendBudgetKind,
    code: CheckpointBackendBudgetFailureCode,
) -> CheckpointError {
    CheckpointError::BackendBudget { budget, code }
}

fn map_budget_error(
    kind: CheckpointBackendBudgetKind,
    limits: BudgetLimits,
    items: usize,
    bytes: usize,
    error: BudgetError,
) -> CheckpointError {
    let code = match error {
        BudgetError::ZeroCapacity if limits.max_items == 0 => {
            CheckpointBackendBudgetFailureCode::ItemCapacity
        }
        BudgetError::ZeroCapacity => CheckpointBackendBudgetFailureCode::ByteCapacity,
        BudgetError::RequestExceedsCapacity if items > limits.max_items => {
            CheckpointBackendBudgetFailureCode::ItemCapacity
        }
        BudgetError::RequestExceedsCapacity if bytes > limits.max_bytes => {
            CheckpointBackendBudgetFailureCode::ByteCapacity
        }
        BudgetError::Closed => CheckpointBackendBudgetFailureCode::Closed,
        BudgetError::PermitCountTooLarge
        | BudgetError::AccountingOverflow
        | BudgetError::CannotGrowLease
        | BudgetError::InvalidFragmentItemCharge { .. }
        | BudgetError::ActionPayloadUndercharged { .. }
        | BudgetError::RequestExceedsCapacity => {
            CheckpointBackendBudgetFailureCode::Unrepresentable
        }
    };
    backend_error(kind, code)
}

#[derive(Clone)]
struct MemoryBudgets {
    transactions: BackendBudget,
    prepared_indexes: BackendBudget,
    storage: BackendBudget,
    result_summaries: BackendBudget,
    reads: BackendBudget,
}

#[derive(Default)]
struct MemoryRunHead {
    generation: Option<CommittedCheckpointGeneration>,
    objects: BTreeMap<ContentDigest, BudgetedStoredObject>,
}

#[derive(Default)]
struct MemoryState {
    heads: BTreeMap<StreamRunIdentity, MemoryRunHead>,
}

struct StorageCommitBundle {
    _storage_lease: BudgetLease,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StoredObjectKind {
    Generation,
    Participant,
    ResultIndex,
    ResultPayload,
}

struct BudgetedStoredObject {
    bytes: Bytes,
    kind: StoredObjectKind,
    _storage_bundle: Rc<StorageCommitBundle>,
}

/// Current item-and-byte usage for one backend budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryBudgetUsage {
    /// Currently charged items.
    pub used_items: usize,
    /// Currently charged bytes.
    pub used_bytes: usize,
}

/// Current backend resource usage without historical high-water telemetry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryLiveBudgetUsage {
    /// Transaction usage.
    pub transactions: MemoryBudgetUsage,
    /// Prepared-index usage.
    pub prepared_indexes: MemoryBudgetUsage,
    /// Immutable-storage usage.
    pub storage: MemoryBudgetUsage,
    /// Returned-summary usage.
    pub result_summaries: MemoryBudgetUsage,
    /// Reader usage.
    pub reads: MemoryBudgetUsage,
}

/// Exact immutable object inventory retained for one logical run.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ImmutableObjectInventory {
    generations: BTreeSet<ContentDigest>,
    participants: BTreeSet<ContentDigest>,
    indexes: BTreeSet<ContentDigest>,
    results: BTreeSet<ContentDigest>,
}

impl ImmutableObjectInventory {
    /// Borrow retained participant payload digests.
    #[must_use]
    pub const fn participant_payloads(&self) -> &BTreeSet<ContentDigest> {
        &self.participants
    }

    /// Borrow retained result payload digests.
    #[must_use]
    pub const fn result_payloads(&self) -> &BTreeSet<ContentDigest> {
        &self.results
    }

    /// Return the total immutable object count.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.generations.len() + self.participants.len() + self.indexes.len() + self.results.len()
    }
}

/// Deterministic injected reference-backend fault point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TestMemoryFault {
    /// Fail after complete candidate prevalidation and before publication.
    AfterPrevalidationBeforePublication,
}

/// Atomic worker-local in-memory checkpoint backend.
#[derive(Clone)]
pub struct MemoryCheckpointBackend {
    state: Rc<RefCell<MemoryState>>,
    budgets: MemoryBudgets,
    fault: Rc<Cell<Option<TestMemoryFault>>>,
    reached_fault: Rc<Cell<Option<TestMemoryFault>>>,
    state_accesses: Rc<Cell<u64>>,
}

/// Opaque test hold over all returned-summary capacity.
#[doc(hidden)]
pub struct MemoryResultSummaryCapacityHold {
    _lease: BudgetLease,
}

impl MemoryCheckpointBackend {
    /// Construct a backend after validating all five budgets in field order.
    pub fn new(limits: MemoryCheckpointLimits) -> Result<Self, CheckpointError> {
        let transactions = BackendBudget::new(
            CheckpointBackendBudgetKind::Transaction,
            limits.transactions,
        )?;
        let prepared_indexes = BackendBudget::new(
            CheckpointBackendBudgetKind::PreparedIndex,
            limits.prepared_indexes,
        )?;
        let storage = BackendBudget::new(CheckpointBackendBudgetKind::Storage, limits.storage)?;
        let result_summaries = BackendBudget::new(
            CheckpointBackendBudgetKind::ResultSummary,
            limits.result_summaries,
        )?;
        let reads = BackendBudget::new(CheckpointBackendBudgetKind::Read, limits.reads)?;
        Ok(Self {
            state: Rc::new(RefCell::new(MemoryState::default())),
            budgets: MemoryBudgets {
                transactions,
                prepared_indexes,
                storage,
                result_summaries,
                reads,
            },
            fault: Rc::new(Cell::new(None)),
            reached_fault: Rc::new(Cell::new(None)),
            state_accesses: Rc::new(Cell::new(0)),
        })
    }

    /// Begin a concrete reference transaction.
    pub async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<MemoryGenerationTransaction, CheckpointError> {
        if run != expectations.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let lease = self.budgets.transactions.acquire(1, 1).await?;
        Ok(MemoryGenerationTransaction {
            backend: self.clone(),
            run,
            expected,
            expectations,
            _transaction_lease: lease,
            participants: Vec::new(),
            staged_results: None,
        })
    }

    /// Open a concrete verified reference reader for the latest run head.
    pub async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<MemoryGenerationReader>, CheckpointError> {
        if run != &expected.run {
            return Err(CheckpointError::ObjectVerification);
        }
        self.note_state_access();
        let (generation, object_bytes) = {
            let state = self.state.borrow();
            let Some(head) = state.heads.get(run) else {
                return Ok(None);
            };
            let Some(generation) = head.generation.as_ref() else {
                return Ok(None);
            };
            let object = head
                .objects
                .get(generation.generation_ref().digest())
                .ok_or(CheckpointError::ObjectVerification)?;
            (generation.generation(), object.bytes.len())
        };
        let lease = self.budgets.reads.acquire(1, 0).await?;
        self.note_state_access();
        let stored = {
            let state = self.state.borrow();
            let object = state
                .heads
                .get(run)
                .and_then(|head| head.objects.get(generation.digest()))
                .ok_or(CheckpointError::LeaseLost {
                    generation: generation.clone(),
                })?;
            if object.bytes.len() != object_bytes {
                return Err(CheckpointError::ObjectVerification);
            }
            object.bytes.clone()
        };
        let candidate: CheckpointGenerationCandidate =
            serde_json::from_slice(&stored).map_err(|_| CheckpointError::ObjectVerification)?;
        let prevalidated = candidate.prevalidate_for_publication(
            run,
            &expected.participant_plan,
            &expected.execution_plan_digest,
            &expected.result_plan_digest,
        )?;
        let committed = prevalidated.into_committed_after_publication_fence();
        Ok(Some(MemoryGenerationReader {
            backend: self.clone(),
            generation: committed,
            _generation_lease: lease,
        }))
    }

    /// Snapshot all backend budgets.
    #[must_use]
    pub fn budget_snapshots(&self) -> MemoryLiveBudgetUsage {
        self.live_budget_usage()
    }

    /// Snapshot current charges while discarding historical high-water fields.
    #[must_use]
    pub fn live_budget_usage(&self) -> MemoryLiveBudgetUsage {
        fn live(snapshot: BudgetSnapshot) -> MemoryBudgetUsage {
            MemoryBudgetUsage {
                used_items: snapshot.used_items,
                used_bytes: snapshot.used_bytes,
            }
        }
        MemoryLiveBudgetUsage {
            transactions: live(self.budgets.transactions.snapshot()),
            prepared_indexes: live(self.budgets.prepared_indexes.snapshot()),
            storage: live(self.budgets.storage.snapshot()),
            result_summaries: live(self.budgets.result_summaries.snapshot()),
            reads: live(self.budgets.reads.snapshot()),
        }
    }

    /// Return the number of currently prepared transactions.
    #[must_use]
    pub fn prepared_transactions(&self) -> usize {
        self.budgets.transactions.snapshot().used_items
    }

    /// Hold all returned-summary capacity for cancellation and contention tests.
    #[doc(hidden)]
    pub async fn hold_all_result_summary_capacity(
        &self,
    ) -> Result<MemoryResultSummaryCapacityHold, CheckpointError> {
        let lease = self
            .budgets
            .result_summaries
            .resource
            .acquire(
                self.budgets.result_summaries.limits.max_items,
                self.budgets.result_summaries.limits.max_bytes,
            )
            .await
            .map_err(|error| {
                map_budget_error(
                    CheckpointBackendBudgetKind::ResultSummary,
                    self.budgets.result_summaries.limits,
                    self.budgets.result_summaries.limits.max_items,
                    self.budgets.result_summaries.limits.max_bytes,
                    error,
                )
            })?;
        Ok(MemoryResultSummaryCapacityHold { _lease: lease })
    }

    /// Inventory immutable objects retained for one exact run.
    #[must_use]
    pub fn immutable_object_inventory(&self, run: &StreamRunIdentity) -> ImmutableObjectInventory {
        self.note_state_access();
        let state = self.state.borrow();
        let Some(head) = state.heads.get(run) else {
            return ImmutableObjectInventory::default();
        };
        let mut inventory = ImmutableObjectInventory::default();
        for (digest, object) in &head.objects {
            match object.kind {
                StoredObjectKind::Generation => {
                    inventory.generations.insert(*digest);
                }
                StoredObjectKind::Participant => {
                    inventory.participants.insert(*digest);
                }
                StoredObjectKind::ResultIndex => {
                    inventory.indexes.insert(*digest);
                }
                StoredObjectKind::ResultPayload => {
                    inventory.results.insert(*digest);
                }
            }
        }
        inventory
    }

    /// Arm one deterministic pre-publication fault.
    pub fn arm_test_fault(&self, fault: TestMemoryFault) {
        self.fault.set(Some(fault));
        self.reached_fault.set(None);
    }

    /// Return whether the named fault point was reached.
    #[must_use]
    pub fn test_fault_was_reached(&self, fault: TestMemoryFault) -> bool {
        self.reached_fault.get() == Some(fault)
    }

    /// Reset the injected state-access counter.
    pub fn reset_test_state_accesses(&self) {
        self.state_accesses.set(0);
    }

    /// Return the injected state-access counter.
    #[must_use]
    pub fn test_state_accesses(&self) -> u64 {
        self.state_accesses.get()
    }

    fn note_state_access(&self) {
        self.state_accesses
            .set(self.state_accesses.get().saturating_add(1));
    }

    fn take_fault(&self, fault: TestMemoryFault) -> bool {
        if self.fault.get() == Some(fault) {
            self.fault.set(None);
            self.reached_fault.set(Some(fault));
            true
        } else {
            false
        }
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointBackend for MemoryCheckpointBackend {
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<Box<dyn LeasedGenerationReader>>, CheckpointError> {
        Ok(MemoryCheckpointBackend::open_latest(self, run, expected)
            .await?
            .map(|reader| Box::new(reader) as Box<dyn LeasedGenerationReader>))
    }

    async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError> {
        Ok(Box::new(
            MemoryCheckpointBackend::begin_generation(self, run, expected, expectations).await?,
        ))
    }
}

struct StagedParticipant {
    descriptor: ParticipantStateDescriptor,
    payload: BudgetedCheckpointBytes,
}

struct StagedResultEpoch {
    index_root: ContentDigest,
    descriptors: BudgetedResultDescriptors,
    payloads: Vec<BudgetedCheckpointBytes>,
    item_count: u64,
    byte_length: u64,
}

/// Concrete in-memory generation transaction used by reference tests.
pub struct MemoryGenerationTransaction {
    backend: MemoryCheckpointBackend,
    run: StreamRunIdentity,
    expected: Option<CheckpointGeneration>,
    expectations: CheckpointGenerationExpectations,
    _transaction_lease: BudgetLease,
    participants: Vec<StagedParticipant>,
    staged_results: Option<StagedResultEpoch>,
}

impl MemoryGenerationTransaction {
    /// Stage one participant.
    pub async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        self.stage_participant_inner(state)
    }

    /// Stage the one required result epoch.
    pub async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        self.prepare_result_partitions(partitions).await
    }

    /// Commit this transaction atomically.
    pub async fn commit(
        self,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        self.commit_inner(metadata).await
    }

    /// Snapshot staged counts for cancellation tests.
    #[must_use]
    pub fn staged_snapshot(&self) -> (usize, bool) {
        (self.participants.len(), self.staged_results.is_some())
    }

    /// Borrow the staged result root, when installed.
    #[must_use]
    pub fn staged_result_root(&self) -> Option<&ContentDigest> {
        self.staged_results
            .as_ref()
            .map(|results| &results.index_root)
    }

    fn stage_participant_inner(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        if state.run() != &self.run
            || self.participants.iter().any(|existing| {
                existing.descriptor.participant_id == state.descriptor().participant_id
            })
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (_, descriptor, payload) = state.into_parts();
        self.participants.push(StagedParticipant {
            descriptor,
            payload,
        });
        Ok(())
    }

    async fn prepare_result_partitions(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        if self.staged_results.is_some() {
            return Err(CheckpointError::ObjectVerification);
        }
        if partitions
            .iter()
            .any(|partition| partition.descriptor().run != self.run)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let plan = CheckedResultStagePlan::from_partitions(partitions)?;
        let prepared_lease = self
            .backend
            .budgets
            .prepared_indexes
            .acquire(plan.descriptor_items, plan.descriptor_bytes)
            .await?;
        let summary_lease = self
            .backend
            .budgets
            .result_summaries
            .acquire(plan.descriptor_items, plan.descriptor_bytes)
            .await?;
        self.install_result_partitions(partitions, plan, prepared_lease, summary_lease)
    }

    fn install_result_partitions(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
        plan: CheckedResultStagePlan,
        prepared_lease: BudgetLease,
        summary_lease: BudgetLease,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        let prepared_descriptors = partitions
            .iter()
            .map(|partition| partition.descriptor().clone())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let summary_descriptors = prepared_descriptors.to_vec().into_boxed_slice();
        let prepared_descriptors =
            BudgetedResultDescriptors::new(prepared_descriptors, prepared_lease)?;
        let summary_descriptors =
            BudgetedResultDescriptors::new(summary_descriptors, summary_lease)?;
        let prepared_summary = PreparedResultEpoch::new(
            plan.index_root,
            summary_descriptors,
            plan.item_count,
            plan.byte_length,
        )?;

        let mut payloads = Vec::with_capacity(plan.descriptor_items);
        for partition in std::mem::take(partitions) {
            let (budgeted_descriptor, payload) = partition.into_parts();
            let (input_descriptor, input_lease) = budgeted_descriptor.into_backend_parts();
            payloads.push(payload);
            drop(input_descriptor);
            drop(input_lease);
        }
        self.staged_results = Some(StagedResultEpoch {
            index_root: plan.index_root,
            descriptors: prepared_descriptors,
            payloads,
            item_count: plan.item_count,
            byte_length: plan.byte_length,
        });
        Ok(prepared_summary)
    }

    async fn commit_inner(
        self,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        validate_commit_metadata(&self.expected, metadata.clone())?;
        let results = self
            .staged_results
            .as_ref()
            .ok_or(CheckpointError::ObjectVerification)?;
        let mut participant_descriptors = self
            .participants
            .iter()
            .map(|participant| participant.descriptor.clone())
            .collect::<Vec<_>>();
        participant_descriptors
            .sort_unstable_by(|left, right| left.participant_id.cmp(&right.participant_id));
        if participant_descriptors
            .iter()
            .map(|descriptor| &descriptor.participant_id)
            .ne(self.expectations.participant_plan.ids().iter())
        {
            return Err(CheckpointError::ParticipantSetMismatch);
        }
        if results
            .descriptors
            .descriptors()
            .iter()
            .any(|descriptor| descriptor.run != self.run || descriptor.epoch != metadata.epoch)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let (result_items, result_bytes) = result_totals(results.descriptors.descriptors())?;
        if result_items != results.item_count || result_bytes != results.byte_length {
            return Err(CheckpointError::ObjectVerification);
        }
        let frozen = FrozenGenerationTransactionInputs::new(
            self.run,
            self.expected.clone(),
            self.expectations.clone(),
            participant_descriptors,
            results.index_root,
        );
        let prevalidated = build_prevalidated_candidate(frozen, metadata)?;
        let generation_bytes = prevalidated.encode_for_storage()?;
        let index_bytes = serde_json::to_vec(results.descriptors.descriptors())
            .map(Bytes::from)
            .map_err(|error| CheckpointError::Storage {
                message: format!("could not encode result index: {error}"),
            })?;

        let mut objects = Vec::with_capacity(self.participants.len() + results.payloads.len() + 2);
        objects.push((
            *prevalidated.generation().digest(),
            StoredObjectKind::Generation,
            generation_bytes,
        ));
        objects.push((
            results.index_root,
            StoredObjectKind::ResultIndex,
            index_bytes,
        ));
        for participant in &self.participants {
            objects.push((
                participant.descriptor.content_digest,
                StoredObjectKind::Participant,
                Bytes::copy_from_slice(participant.payload.as_bytes()),
            ));
        }
        for (descriptor, payload) in results
            .descriptors
            .descriptors()
            .iter()
            .zip(&results.payloads)
        {
            objects.push((
                descriptor.payload_digest,
                StoredObjectKind::ResultPayload,
                Bytes::copy_from_slice(payload.as_bytes()),
            ));
        }

        self.backend.note_state_access();
        let missing = {
            let state = self.backend.state.borrow();
            let existing = state.heads.get(&self.run);
            objects
                .into_iter()
                .filter(|(digest, _, _)| {
                    existing.is_none_or(|head| !head.objects.contains_key(digest))
                })
                .collect::<Vec<_>>()
        };
        let storage_bytes = missing.iter().try_fold(0usize, |total, (_, _, bytes)| {
            total
                .checked_add(bytes.len())
                .ok_or(CheckpointError::ObjectVerification)
        })?;
        let storage_lease = self
            .backend
            .budgets
            .storage
            .acquire(missing.len(), storage_bytes)
            .await?;
        let bundle = Rc::new(StorageCommitBundle {
            _storage_lease: storage_lease,
        });
        let new_objects = missing
            .into_iter()
            .map(|(digest, kind, bytes)| {
                (
                    digest,
                    BudgetedStoredObject {
                        bytes,
                        kind,
                        _storage_bundle: Rc::clone(&bundle),
                    },
                )
            })
            .collect::<Vec<_>>();

        if self
            .backend
            .take_fault(TestMemoryFault::AfterPrevalidationBeforePublication)
        {
            return Err(CheckpointError::Storage {
                message: "injected memory checkpoint fault after prevalidation before publication"
                    .into(),
            });
        }
        self.backend.note_state_access();
        publish_prevalidated(
            &mut self.backend.state.borrow_mut(),
            self.run,
            self.expected,
            prevalidated,
            new_objects,
        )
    }
}

#[async_trait(?Send)]
impl StreamingGenerationTransaction for MemoryGenerationTransaction {
    async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError> {
        self.stage_participant_inner(state)
    }
    async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError> {
        self.prepare_result_partitions(partitions).await
    }
    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError> {
        (*self).commit_inner(metadata).await
    }
}

struct CheckedResultStagePlan {
    descriptor_items: usize,
    descriptor_bytes: usize,
    index_root: ContentDigest,
    item_count: u64,
    byte_length: u64,
}

impl CheckedResultStagePlan {
    fn from_partitions(partitions: &[ResultPartition]) -> Result<Self, CheckpointError> {
        let descriptors = partitions
            .iter()
            .map(ResultPartition::descriptor)
            .collect::<Vec<_>>();
        let descriptor_bytes = descriptors.iter().try_fold(0usize, |total, descriptor| {
            total
                .checked_add(descriptor_retained_bytes(descriptor)?)
                .ok_or(CheckpointError::ObjectVerification)
        })?;
        let (item_count, byte_length) =
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
                })?;
        let encoded =
            serde_json::to_vec(&descriptors).map_err(|error| CheckpointError::Storage {
                message: format!("could not encode result index: {error}"),
            })?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aiperf.streaming.result-index.v1");
        hasher.update(&(encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
        let index_root = ContentDigest::from_bytes(*hasher.finalize().as_bytes());
        Ok(Self {
            descriptor_items: descriptors.len(),
            descriptor_bytes,
            index_root,
            item_count,
            byte_length,
        })
    }
}

fn compare_expected(
    head: Option<CheckpointGeneration>,
    expected: Option<CheckpointGeneration>,
) -> Result<(), CheckpointError> {
    if head != expected {
        return Err(CheckpointError::GenerationConflict {
            expected,
            actual: head,
        });
    }
    Ok(())
}

fn publish_prevalidated(
    state: &mut MemoryState,
    run: StreamRunIdentity,
    expected: Option<CheckpointGeneration>,
    prevalidated: PrevalidatedCheckpointGenerationCandidate,
    new_objects: Vec<(ContentDigest, BudgetedStoredObject)>,
) -> Result<CommittedCheckpointGeneration, CheckpointError> {
    let actual = state
        .heads
        .get(&run)
        .and_then(|head| head.generation.as_ref())
        .map(CommittedCheckpointGeneration::generation);
    compare_expected(actual, expected)?;
    let committed = prevalidated.into_committed_after_publication_fence();
    let run_head = state.heads.entry(run).or_default();
    for (digest, object) in new_objects {
        run_head.objects.insert(digest, object);
    }
    run_head.generation = Some(committed.clone());
    Ok(committed)
}

/// Concrete leased reader for one in-memory committed generation.
pub struct MemoryGenerationReader {
    backend: MemoryCheckpointBackend,
    generation: CommittedCheckpointGeneration,
    _generation_lease: BudgetLease,
}

impl MemoryGenerationReader {
    /// Borrow the authoritative generation.
    #[must_use]
    pub const fn generation(&self) -> &CommittedCheckpointGeneration {
        &self.generation
    }

    /// Scan one reachable descriptor page.
    pub async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        self.scan_result_index_inner(after, budget).await
    }

    /// Read one reachable result payload.
    pub async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.read_segment_inner(descriptor).await
    }

    /// Read one reachable participant payload.
    pub async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError> {
        self.read_participant_inner(descriptor).await
    }

    fn reachable_descriptors(&self) -> Result<Vec<ResultSegmentDescriptor>, CheckpointError> {
        self.backend.note_state_access();
        let descriptors = {
            let state = self.backend.state.borrow();
            let object = state
                .heads
                .get(self.generation.run())
                .and_then(|head| head.objects.get(self.generation.result_index_root()))
                .filter(|object| object.kind == StoredObjectKind::ResultIndex)
                .ok_or(CheckpointError::ObjectVerification)?;
            serde_json::from_slice::<Vec<ResultSegmentDescriptor>>(&object.bytes)
                .map_err(|_| CheckpointError::ObjectVerification)?
        };
        if canonical_result_index_root(&descriptors)? != *self.generation.result_index_root()
            || descriptors
                .iter()
                .any(|descriptor| descriptor.run != *self.generation.run())
        {
            return Err(CheckpointError::ObjectVerification);
        }
        Ok(descriptors)
    }

    async fn scan_result_index_inner(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        let descriptors = self.reachable_descriptors()?;
        let root = *self.generation.result_index_root();
        let start = match after {
            None => 0usize,
            Some(cursor) => {
                if cursor.root != root || cursor.block != root {
                    return Err(CheckpointError::ObjectVerification);
                }
                let offset = usize::try_from(cursor.item_offset)
                    .map_err(|_| CheckpointError::ObjectVerification)?;
                if offset >= descriptors.len() {
                    return Err(CheckpointError::ObjectVerification);
                }
                offset
            }
        };
        if start == descriptors.len() {
            let lease = self.backend.budgets.reads.acquire(0, 0).await?;
            return ResultIndexPage::new(
                BudgetedResultDescriptors::new(Vec::new().into_boxed_slice(), lease)?,
                None,
            );
        }
        let first_required = descriptor_retained_bytes(&descriptors[start])?;
        let first_required_u64 =
            u64::try_from(first_required).map_err(|_| CheckpointError::ObjectVerification)?;
        if first_required_u64 > budget.max_bytes.get() {
            return Err(CheckpointError::ResultIndexReadBudgetTooSmall {
                required_bytes: first_required_u64,
                max_bytes: budget.max_bytes.get(),
            });
        }
        let mut end = start;
        let mut retained = 0usize;
        while end < descriptors.len() && end - start < budget.max_items.get() {
            let next = descriptor_retained_bytes(&descriptors[end])?;
            let Some(total) = retained.checked_add(next) else {
                return Err(CheckpointError::ObjectVerification);
            };
            if u64::try_from(total).map_err(|_| CheckpointError::ObjectVerification)?
                > budget.max_bytes.get()
            {
                break;
            }
            retained = total;
            end += 1;
        }
        let page_descriptors = descriptors[start..end].to_vec().into_boxed_slice();
        let lease = self
            .backend
            .budgets
            .reads
            .acquire(page_descriptors.len(), retained)
            .await?;
        let next = if end < descriptors.len() {
            Some(ResultIndexCursor {
                root,
                block: root,
                item_offset: u32::try_from(end).map_err(|_| CheckpointError::ObjectVerification)?,
            })
        } else {
            None
        };
        ResultIndexPage::new(
            BudgetedResultDescriptors::new(page_descriptors, lease)?,
            next,
        )
    }

    async fn read_segment_inner(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        let reachable = self.reachable_descriptors()?;
        if !reachable.contains(descriptor) {
            return Err(CheckpointError::ObjectVerification);
        }
        self.backend.note_state_access();
        let byte_length = {
            let state = self.backend.state.borrow();
            state
                .heads
                .get(self.generation.run())
                .and_then(|head| head.objects.get(&descriptor.payload_digest))
                .filter(|object| object.kind == StoredObjectKind::ResultPayload)
                .map(|object| object.bytes.len())
                .ok_or(CheckpointError::ObjectVerification)?
        };
        let lease = self.backend.budgets.reads.acquire(1, byte_length).await?;
        self.backend.note_state_access();
        let bytes = {
            let state = self.backend.state.borrow();
            state
                .heads
                .get(self.generation.run())
                .and_then(|head| head.objects.get(&descriptor.payload_digest))
                .filter(|object| object.kind == StoredObjectKind::ResultPayload)
                .map(|object| object.bytes.clone())
                .ok_or(CheckpointError::ObjectVerification)?
        };
        ResultSegmentReader::new(descriptor, BudgetedCheckpointBytes::new(bytes, lease)?)
    }

    async fn read_participant_inner(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError> {
        if !self
            .generation
            .participant_descriptors()
            .contains(descriptor)
        {
            return Err(CheckpointError::ObjectVerification);
        }
        self.backend.note_state_access();
        let byte_length = {
            let state = self.backend.state.borrow();
            state
                .heads
                .get(self.generation.run())
                .and_then(|head| head.objects.get(&descriptor.content_digest))
                .filter(|object| object.kind == StoredObjectKind::Participant)
                .map(|object| object.bytes.len())
                .ok_or(CheckpointError::ObjectVerification)?
        };
        let lease = self.backend.budgets.reads.acquire(1, byte_length).await?;
        self.backend.note_state_access();
        let bytes = {
            let state = self.backend.state.borrow();
            state
                .heads
                .get(self.generation.run())
                .and_then(|head| head.objects.get(&descriptor.content_digest))
                .filter(|object| object.kind == StoredObjectKind::Participant)
                .map(|object| object.bytes.clone())
                .ok_or(CheckpointError::ObjectVerification)?
        };
        CommittedParticipantState::new(
            *self.generation.run(),
            descriptor.clone(),
            BudgetedCheckpointBytes::new(bytes, lease)?,
        )
    }
}

#[async_trait(?Send)]
impl LeasedGenerationReader for MemoryGenerationReader {
    fn generation(&self) -> &CommittedCheckpointGeneration {
        &self.generation
    }
    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError> {
        self.scan_result_index_inner(after, budget).await
    }
    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError> {
        self.read_segment_inner(descriptor).await
    }
    async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError> {
        self.read_participant_inner(descriptor).await
    }
}
