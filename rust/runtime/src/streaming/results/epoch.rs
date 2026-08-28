// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Result-epoch rotation, bounded provisional holes, and partial-result views.
//!
//! One [`EpochResultCoordinator`] owns the result plane of one worker. It joins
//! correlated terminal facts to their logical actions, keeps the contiguous
//! terminal horizon `H`, and rotates everything at or below `H` into one
//! immutable result partition per checkpoint barrier.
//!
//! A completion above `H` is a hole, not a loss: it is retained in a bounded
//! [`ProvisionalResultStore`] and never linked from a committed root until the
//! hole closes. Exhausting that bound fences new admission and refuses with the
//! authored overload decision rather than dropping the fact.
//!
//! The producer holds one descriptor budget and one payload budget. The
//! descriptor authority is charged before a partition is returned and travels
//! inside [`ResultPartition`], so it stays charged until the checkpoint backend
//! has taken the epoch. A descriptor refusal maps only to
//! [`ResultPlaneError::PartitionDescriptorCapacityExceeded`]; backend,
//! participant-state, storage, and provisional-hole capacity are separate.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::{
    metrics_core::{
        AccumulatorSummary, MetricEntry, RecordIngest, accumulator::MetricsAccumulator,
        report::NativeReport,
    },
    streaming::{
        budget::StreamingResourceBudget,
        checkpoint::{
            BudgetedCheckpointBytes, CheckpointBarrier, CheckpointCut, CheckpointEpoch,
            CheckpointError, CheckpointGeneration, CheckpointParticipantId,
            CommittedCheckpointGeneration, CommittedParticipantReceipt, CommittedParticipantState,
            PreparedParticipantState, StreamRunIdentity, StreamingCheckpointParticipant,
            TerminalActionHorizon,
        },
        checkpoint_coordinator::PreparedCheckpointResultInput,
        identity::{ContentDigest, GlobalSequence, StableActionId},
        reliability::{PreparedIssueReceiptPartitionView, StreamingIssueSummary},
        results::{
            BudgetedResultDescriptor, CellId, CorrelatedRecordIngest, RESULT_SCHEMA_V1,
            ResultMembership, ResultPartition, ResultPlaneError, ResultProjectionId,
            ResultSegmentDescriptor, WorkerId, descriptor_retained_bytes,
            index::ResultMembershipKey,
        },
    },
};

/// Participant state schema owned by the result plane.
const RESULT_PARTICIPANT_SCHEMA_ID: &str = "aiperf.streaming.results.epoch";
/// Participant state schema version owned by the result plane.
const RESULT_PARTICIPANT_SCHEMA_VERSION: u32 = 1;
/// Stable projection identity used when the caller authors none.
const DEFAULT_RESULT_PROJECTION: &str = "streaming_results";

/// Fixed placement and bounds frozen for one worker's result plane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResultEpochPlacement {
    /// Producing cell coordinate.
    pub cell_id: CellId,
    /// Producing worker coordinate.
    pub worker_id: WorkerId,
    /// Stable projection identity written into every descriptor.
    pub projection: ResultProjectionId,
    /// Greatest number of completions retained above the terminal horizon.
    pub provisional_limit: usize,
}

impl ResultEpochPlacement {
    /// Build a placement for one worker with the default result projection.
    ///
    /// # Errors
    ///
    /// Returns [`ResultPlaneError::InvalidCoverage`] when the provisional bound
    /// is zero, because a zero-capacity hole store can never close a hole.
    pub fn new(
        cell_id: CellId,
        worker_id: WorkerId,
        provisional_limit: usize,
    ) -> Result<Self, ResultPlaneError> {
        Self::with_projection(cell_id, worker_id, DEFAULT_RESULT_PROJECTION, provisional_limit)
    }

    /// Build a placement with an explicit nonempty projection identity.
    ///
    /// # Errors
    ///
    /// Returns [`ResultPlaneError::InvalidCoverage`] for an empty projection or
    /// a zero provisional bound.
    pub fn with_projection(
        cell_id: CellId,
        worker_id: WorkerId,
        projection: &str,
        provisional_limit: usize,
    ) -> Result<Self, ResultPlaneError> {
        if provisional_limit == 0 {
            return Err(ResultPlaneError::InvalidCoverage);
        }
        let projection =
            ResultProjectionId::new(projection).map_err(|_| ResultPlaneError::InvalidCoverage)?;
        Ok(Self {
            cell_id,
            worker_id,
            projection,
            provisional_limit,
        })
    }
}

/// One retained terminal fact joined to its logical action.
struct RetainedTerminal {
    logical_action_id: StableActionId,
    membership: ResultMembership,
    session_num: u64,
    record: RecordIngest,
}

impl RetainedTerminal {
    fn from_ingest(fact: &CorrelatedRecordIngest) -> Self {
        Self {
            logical_action_id: fact.correlation.logical_action_id,
            membership: fact.correlation.membership,
            session_num: fact.record.session_num,
            record: fact.record.clone(),
        }
    }
}

/// Bounded holder for completions observed above the terminal horizon.
///
/// The store is a strict capacity gate, not a cache: it never evicts, because a
/// dropped completion above a hole would silently shorten the committed
/// membership once the hole closed.
pub struct ProvisionalResultStore {
    limit: usize,
    entries: BTreeMap<GlobalSequence, RetainedTerminal>,
    retained_bytes: usize,
    is_admission_fenced: bool,
}

impl ProvisionalResultStore {
    /// Build an empty store with the authored item bound.
    #[must_use]
    pub const fn new(limit: usize) -> Self {
        Self {
            limit,
            entries: BTreeMap::new(),
            retained_bytes: 0,
            is_admission_fenced: false,
        }
    }

    /// Return the number of retained provisional completions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no completion is currently held above the horizon.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether capacity exhaustion has fenced new admission.
    #[must_use]
    pub const fn is_admission_fenced(&self) -> bool {
        self.is_admission_fenced
    }

    /// Return the exact retained provisional allocation.
    #[must_use]
    pub const fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    /// Summarize the retained hole for a dashboard view.
    ///
    /// Returns `None` when no hole is open, so a partial view never labels an
    /// empty provisional set as data.
    #[must_use]
    pub fn dashboard_summary(&self) -> Option<ProvisionalDashboardSummary> {
        let first = *self.entries.keys().next()?;
        let last = *self.entries.keys().next_back()?;
        Some(ProvisionalDashboardSummary {
            item_count: self.entries.len() as u64,
            retained_bytes: self.retained_bytes as u64,
            first_sequence: first,
            last_sequence: last,
            is_admission_fenced: self.is_admission_fenced,
        })
    }

    fn request_shaped_count(&self) -> u64 {
        self.entries
            .values()
            .filter(|retained| retained.membership.is_request_shaped())
            .count() as u64
    }

    fn session_numbers(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.values().map(|retained| retained.session_num)
    }

    /// Retain one completion above the horizon, refusing beyond the bound.
    fn insert(
        &mut self,
        sequence: GlobalSequence,
        retained: RetainedTerminal,
    ) -> Result<(), ResultPlaneError> {
        if let Some(existing) = self.entries.get(&sequence) {
            return membership_agreement(existing, &retained);
        }
        let entry_bytes = provisional_entry_bytes(&retained);
        if self.entries.len() >= self.limit {
            // Fencing is the authored overload decision: the caller must stop
            // admitting new work rather than lose the completion it just saw.
            self.is_admission_fenced = true;
            return Err(ResultPlaneError::ProvisionalCapacityExceeded {
                items: self.entries.len() as u64 + 1,
                bytes: self.retained_bytes.saturating_add(entry_bytes) as u64,
            });
        }
        self.retained_bytes = self.retained_bytes.saturating_add(entry_bytes);
        self.entries.insert(sequence, retained);
        Ok(())
    }

    /// Remove the entry at `sequence` when the hole below it has closed.
    fn take(&mut self, sequence: GlobalSequence) -> Option<RetainedTerminal> {
        let retained = self.entries.remove(&sequence)?;
        self.retained_bytes = self
            .retained_bytes
            .saturating_sub(provisional_entry_bytes(&retained));
        if self.entries.is_empty() {
            self.is_admission_fenced = false;
        }
        Some(retained)
    }
}

/// Separately labeled provisional data excluded from every committed total.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvisionalDashboardSummary {
    /// Completions retained above the terminal horizon.
    pub item_count: u64,
    /// Exact retained provisional allocation.
    pub retained_bytes: u64,
    /// Lowest retained provisional sequence.
    pub first_sequence: GlobalSequence,
    /// Greatest retained provisional sequence.
    pub last_sequence: GlobalSequence,
    /// Whether capacity exhaustion fenced new admission.
    pub is_admission_fenced: bool,
}

/// One worker's rotated accumulator epoch and the partitions it produced.
///
/// The plan names a committed `CheckpointGeneration` here, which is not
/// knowable before CAS; the rotation is therefore identified by the barrier
/// epoch it was cut at, and the generation is learned from the post-commit
/// receipt.
pub struct WorkerResultEpoch {
    /// Barrier epoch this rotation was cut at.
    pub epoch: CheckpointEpoch,
    /// Producing worker coordinate.
    pub worker_id: WorkerId,
    /// First global sequence represented by the rotation.
    pub first_sequence: GlobalSequence,
    /// Last global sequence represented by the rotation.
    pub last_sequence: GlobalSequence,
    /// Immutable partitions produced by the rotation.
    pub partitions: Vec<ResultPartition>,
}

/// Partial result authority derived from one committed generation.
pub struct CommittedPartialResult {
    /// Generation that made this partial view authoritative.
    pub generation: CheckpointGeneration,
    /// Complete cut represented by the generation.
    pub cut: CheckpointCut,
    /// Contiguous terminal horizon the totals are computed through.
    pub terminal_horizon: TerminalActionHorizon,
    /// Request-shaped actions committed through the horizon.
    pub authoritative_request_count: u64,
    /// Request-shaped completions held above the horizon.
    pub provisional_request_count: u64,
    /// Sessions represented by committed actions.
    pub active_session_count: u64,
    /// Sessions represented only by provisional completions.
    pub incomplete_session_count: u64,
    /// Reliability summary supplied by the ledger, when one was observed.
    pub issue_summary: Option<StreamingIssueSummary>,
    /// Committed actions whose terminal fact reported failure.
    pub failed_action_count: u64,
    /// Metrics computed from committed records only.
    pub metrics: BTreeMap<String, MetricEntry>,
    /// Separately labeled provisional data, excluded from every total above.
    pub provisional: Option<ProvisionalDashboardSummary>,
}

/// Retained checkpointable result-plane state.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResultParticipantState {
    terminal_horizon: GlobalSequence,
    published_horizon: GlobalSequence,
    authoritative_request_count: u64,
    failed_action_count: u64,
    committed_index_root: Option<ContentDigest>,
    provisional: Vec<GlobalSequence>,
    active_sessions: Vec<u64>,
}

/// One worker's result-plane checkpoint participant and epoch producer.
pub struct EpochResultCoordinator {
    run: StreamRunIdentity,
    participant_id: CheckpointParticipantId,
    placement: ResultEpochPlacement,
    descriptor_budget: StreamingResourceBudget,
    payload_budget: StreamingResourceBudget,
    /// Terminal facts at or below `H` that no barrier has rotated yet.
    pending: BTreeMap<GlobalSequence, RetainedTerminal>,
    provisional: ProvisionalResultStore,
    terminal_horizon: GlobalSequence,
    published_horizon: GlobalSequence,
    accumulator: MetricsAccumulator,
    authoritative_request_count: u64,
    failed_action_count: u64,
    active_sessions: BTreeSet<u64>,
    committed_index_root: Option<ContentDigest>,
    committed_generation: Option<CheckpointGeneration>,
    committed_cut: Option<CheckpointCut>,
    issue_summary: Option<StreamingIssueSummary>,
    is_initialized: bool,
}

impl EpochResultCoordinator {
    /// Build one worker's result plane over its two producer-side budgets.
    ///
    /// The descriptor budget is the singular partition-descriptor authority the
    /// plan requires as an explicit dependency; the payload budget charges the
    /// separately budgeted immutable payloads. Neither is created here and
    /// neither is a checkpoint backend budget.
    #[must_use]
    pub fn new(
        run: StreamRunIdentity,
        participant_id: CheckpointParticipantId,
        placement: ResultEpochPlacement,
        descriptor_budget: StreamingResourceBudget,
        payload_budget: StreamingResourceBudget,
    ) -> Self {
        let provisional_limit = placement.provisional_limit;
        Self {
            run,
            participant_id,
            placement,
            descriptor_budget,
            payload_budget,
            pending: BTreeMap::new(),
            provisional: ProvisionalResultStore::new(provisional_limit),
            terminal_horizon: GlobalSequence::new(0),
            published_horizon: GlobalSequence::new(0),
            accumulator: MetricsAccumulator::new(),
            authoritative_request_count: 0,
            failed_action_count: 0,
            active_sessions: BTreeSet::new(),
            committed_index_root: None,
            committed_generation: None,
            committed_cut: None,
            issue_summary: None,
            is_initialized: false,
        }
    }

    /// Borrow the producer-side singular descriptor authority.
    #[must_use]
    pub const fn descriptor_budget(&self) -> &StreamingResourceBudget {
        &self.descriptor_budget
    }

    /// Borrow the bounded provisional hole store.
    #[must_use]
    pub const fn provisional(&self) -> &ProvisionalResultStore {
        &self.provisional
    }

    /// Return the contiguous terminal horizon this plane has proven.
    #[must_use]
    pub const fn terminal_horizon(&self) -> TerminalActionHorizon {
        TerminalActionHorizon::new(self.terminal_horizon)
    }

    /// Adopt the reliability ledger's summary for the next partial view.
    pub fn observe_issue_summary(&mut self, summary: StreamingIssueSummary) {
        self.issue_summary = Some(summary);
    }

    /// Join one correlated terminal fact to the result plane.
    ///
    /// A fact exactly one past the horizon closes the current hole and drains
    /// every provisional completion that becomes contiguous behind it. A fact
    /// further above the horizon is retained provisionally. A fact at or below
    /// the horizon is an idempotent repeat unless it names a different logical
    /// action, which is a membership conflict.
    ///
    /// # Errors
    ///
    /// Returns [`ResultPlaneError::ProvisionalCapacityExceeded`] when the hole
    /// store is full, and [`ResultPlaneError::MembershipConflict`] when a
    /// sequence is reused for a different logical action.
    pub fn observe_terminal(
        &mut self,
        fact: CorrelatedRecordIngest,
    ) -> Result<(), ResultPlaneError> {
        let sequence = fact.correlation.global_sequence;
        let retained = RetainedTerminal::from_ingest(&fact);
        if sequence.get() <= self.terminal_horizon.get() {
            return match self.pending.get(&sequence) {
                Some(existing) => membership_agreement(existing, &retained),
                // Already rotated into a committed epoch: repeating it changes
                // nothing, and the epoch that owns it is immutable.
                None => Ok(()),
            };
        }
        if sequence.get() == self.terminal_horizon.get().saturating_add(1) {
            self.admit(sequence, retained);
            self.drain_contiguous();
            return Ok(());
        }
        self.provisional.insert(sequence, retained)
    }

    /// Rotate every committed action through the barrier horizon.
    ///
    /// # Errors
    ///
    /// Returns [`ResultPlaneError::PartitionDescriptorCapacityExceeded`] when
    /// the singular descriptor authority is unavailable, and
    /// [`ResultPlaneError::InvalidCoverage`] for a foreign barrier.
    pub fn rotate_worker_epoch(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<WorkerResultEpoch, ResultPlaneError> {
        if barrier.run != self.run {
            return Err(ResultPlaneError::InvalidCoverage);
        }
        let horizon = self
            .terminal_horizon
            .get()
            .min(barrier.cut.terminal.get().get());
        let rotated = self.split_pending_through(GlobalSequence::new(horizon));
        let mut partitions = Vec::new();
        let (first, last) = match (rotated.first(), rotated.last()) {
            (Some(first), Some(last)) => (first.0, last.0),
            _ => (GlobalSequence::new(0), GlobalSequence::new(0)),
        };
        if !rotated.is_empty() {
            partitions.push(self.partition_for(barrier.epoch, &rotated)?);
        }
        Ok(WorkerResultEpoch {
            epoch: barrier.epoch,
            worker_id: self.placement.worker_id,
            first_sequence: first,
            last_sequence: last,
            partitions,
        })
    }

    /// Rotate one epoch and carry the ledger's receipt partition with it.
    ///
    /// The receipt view is consumed through `into_result_partition`, so its
    /// payload and view lease move into the returned input rather than being
    /// copied out of it.
    ///
    /// # Errors
    ///
    /// Returns the same refusals as [`Self::rotate_worker_epoch`], plus
    /// [`ResultPlaneError::SegmentVerification`] when the receipt view does not
    /// agree with the descriptor built for it.
    pub fn prepare_epoch(
        &mut self,
        barrier: &CheckpointBarrier,
        issue_receipts: Option<PreparedIssueReceiptPartitionView>,
    ) -> Result<PreparedCheckpointResultInput, ResultPlaneError> {
        let rotated = self.rotate_worker_epoch(barrier)?;
        let receipts = match issue_receipts {
            None => None,
            Some(view) => Some(self.receipt_partition(view)?),
        };
        Ok(PreparedCheckpointResultInput::new(
            rotated.partitions,
            receipts,
        ))
    }

    /// Derive the partial view authorized by one committed generation.
    ///
    /// # Errors
    ///
    /// Returns [`ResultPlaneError::SegmentVerification`] when the generation is
    /// not the one this participant was notified of.
    pub fn committed_partial(
        &self,
        generation: &CommittedCheckpointGeneration,
    ) -> Result<CommittedPartialResult, ResultPlaneError> {
        if generation.run() != &self.run
            || self.committed_generation.as_ref() != Some(generation.generation_ref())
        {
            return Err(ResultPlaneError::SegmentVerification);
        }
        let cut = self
            .committed_cut
            .clone()
            .ok_or(ResultPlaneError::SegmentVerification)?;
        let provisional_sessions: BTreeSet<u64> = self
            .provisional
            .session_numbers()
            .filter(|session| !self.active_sessions.contains(session))
            .collect();
        Ok(CommittedPartialResult {
            generation: generation.generation(),
            cut,
            terminal_horizon: TerminalActionHorizon::new(self.published_horizon),
            authoritative_request_count: self.authoritative_request_count,
            provisional_request_count: self.provisional.request_shaped_count(),
            active_session_count: self.active_sessions.len() as u64,
            incomplete_session_count: provisional_sessions.len() as u64,
            issue_summary: self.issue_summary.clone(),
            failed_action_count: self.failed_action_count,
            metrics: committed_metric_map(&self.accumulator.summarize()),
            provisional: self.provisional.dashboard_summary(),
        })
    }

    fn admit(&mut self, sequence: GlobalSequence, retained: RetainedTerminal) {
        self.terminal_horizon = sequence;
        if retained.membership.is_request_shaped() {
            self.authoritative_request_count += 1;
            self.accumulator.process_record(&retained.record);
        }
        if retained.record.errored {
            self.failed_action_count += 1;
        }
        self.active_sessions.insert(retained.session_num);
        self.pending.insert(sequence, retained);
    }

    fn drain_contiguous(&mut self) {
        while let Some(retained) = self
            .provisional
            .take(GlobalSequence::new(self.terminal_horizon.get() + 1))
        {
            self.admit(
                GlobalSequence::new(self.terminal_horizon.get() + 1),
                retained,
            );
        }
    }

    fn split_pending_through(
        &mut self,
        horizon: GlobalSequence,
    ) -> Vec<(GlobalSequence, RetainedTerminal)> {
        let above = self.pending.split_off(&GlobalSequence::new(
            horizon.get().saturating_add(1),
        ));
        let rotated = std::mem::replace(&mut self.pending, above)
            .into_iter()
            .collect::<Vec<_>>();
        rotated
    }

    fn partition_for(
        &self,
        epoch: CheckpointEpoch,
        rotated: &[(GlobalSequence, RetainedTerminal)],
    ) -> Result<ResultPartition, ResultPlaneError> {
        let projection = rotated
            .iter()
            .map(|(sequence, retained)| EpochRecordProjection {
                global_sequence: *sequence,
                logical_action_id: retained.logical_action_id,
                membership: retained.membership.tag(),
                correlation_id: retained.record.correlation_id.as_str(),
                start_ns: retained.record.start_ns,
                end_ns: retained.record.end_ns,
                errored: retained.record.errored,
            })
            .collect::<Vec<_>>();
        let payload_bytes = serde_json::to_vec(&projection).map_err(|error| {
            ResultPlaneError::Compaction {
                message: format!("could not encode result epoch payload: {error}"),
            }
        })?;
        let membership = ResultMembershipKey::enumerated(
            rotated
                .iter()
                .map(|(_, retained)| retained.logical_action_id)
                .collect(),
        )?;
        // Safe indexing: the caller only reaches here with a nonempty rotation.
        let first_sequence = rotated
            .first()
            .map(|(sequence, _)| *sequence)
            .ok_or(ResultPlaneError::InvalidCoverage)?;
        let last_sequence = rotated
            .last()
            .map(|(sequence, _)| *sequence)
            .ok_or(ResultPlaneError::InvalidCoverage)?;
        let byte_length =
            u64::try_from(payload_bytes.len()).map_err(|_| ResultPlaneError::InvalidCoverage)?;
        let descriptor = ResultSegmentDescriptor {
            run: self.run,
            epoch,
            cell_id: self.placement.cell_id,
            worker_id: self.placement.worker_id,
            projection: self.placement.projection.clone(),
            schema: RESULT_SCHEMA_V1,
            first_sequence,
            last_sequence,
            item_count: membership.item_count(),
            byte_length,
            membership_root: membership.root(),
            payload_digest: ContentDigest::from_bytes(
                *blake3::hash(&payload_bytes).as_bytes(),
            ),
        };
        let payload = self.charge_payload(payload_bytes)?;
        let descriptor = self.charge_descriptor(descriptor)?;
        ResultPartition::new(descriptor, payload).map_err(|_| ResultPlaneError::SegmentVerification)
    }

    fn receipt_partition(
        &self,
        view: PreparedIssueReceiptPartitionView,
    ) -> Result<
        crate::streaming::reliability::PreparedIssueReceiptResultPartition,
        ResultPlaneError,
    > {
        let payload_bytes = view.payload_bytes();
        let byte_length =
            u64::try_from(payload_bytes.len()).map_err(|_| ResultPlaneError::InvalidCoverage)?;
        let projection = ResultProjectionId::new("streaming_issue_receipts")
            .map_err(|_| ResultPlaneError::SegmentVerification)?;
        let descriptor = ResultSegmentDescriptor {
            run: *view.run(),
            epoch: view.barrier().epoch,
            cell_id: self.placement.cell_id,
            worker_id: self.placement.worker_id,
            projection,
            schema: crate::streaming::results::ResultSchemaVersion::new(
                PreparedIssueReceiptPartitionView::required_schema_version(),
            ),
            first_sequence: GlobalSequence::new(0),
            last_sequence: GlobalSequence::new(0),
            item_count: view.receipt_count(),
            byte_length,
            membership_root: *view.receipt_root(),
            payload_digest: ContentDigest::from_bytes(*blake3::hash(payload_bytes).as_bytes()),
        };
        let descriptor = self.charge_descriptor(descriptor)?;
        view.into_result_partition(descriptor)
            .map_err(|_| ResultPlaneError::SegmentVerification)
    }

    /// Charge the exact singular descriptor allocation before it is installed.
    fn charge_descriptor(
        &self,
        descriptor: ResultSegmentDescriptor,
    ) -> Result<BudgetedResultDescriptor, ResultPlaneError> {
        let bytes = descriptor_retained_bytes(&descriptor)
            .map_err(|_| ResultPlaneError::SegmentVerification)?;
        let lease = self.descriptor_budget.try_acquire(1, bytes).map_err(|_| {
            ResultPlaneError::PartitionDescriptorCapacityExceeded {
                items: 1,
                bytes: bytes as u64,
            }
        })?;
        BudgetedResultDescriptor::new(descriptor, lease)
            .map_err(|_| ResultPlaneError::SegmentVerification)
    }

    fn charge_payload(&self, bytes: Vec<u8>) -> Result<BudgetedCheckpointBytes, ResultPlaneError> {
        let lease = self
            .payload_budget
            .try_acquire(1, bytes.len())
            .map_err(|_| ResultPlaneError::Compaction {
                message: "prepared result payload budget exhausted".to_string(),
            })?;
        BudgetedCheckpointBytes::new(Bytes::from(bytes), lease)
            .map_err(|_| ResultPlaneError::SegmentVerification)
    }

    fn retained_state(&self) -> ResultParticipantState {
        ResultParticipantState {
            terminal_horizon: self.terminal_horizon,
            published_horizon: self.published_horizon,
            authoritative_request_count: self.authoritative_request_count,
            failed_action_count: self.failed_action_count,
            committed_index_root: self.committed_index_root,
            provisional: self.provisional.entries.keys().copied().collect(),
            active_sessions: self.active_sessions.iter().copied().collect(),
        }
    }
}

#[async_trait(?Send)]
impl StreamingCheckpointParticipant for EpochResultCoordinator {
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
        let encoded = serde_json::to_vec(&self.retained_state()).map_err(|error| {
            CheckpointError::Storage {
                message: format!("could not encode result participant state: {error}"),
            }
        })?;
        let lease = self
            .payload_budget
            .acquire(1, encoded.len())
            .await
            .map_err(|_| CheckpointError::ObjectVerification)?;
        let payload = BudgetedCheckpointBytes::new(Bytes::from(encoded), lease)?;
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            RESULT_PARTICIPANT_SCHEMA_ID,
            RESULT_PARTICIPANT_SCHEMA_VERSION,
            barrier.cut.clone(),
            self.authoritative_request_count,
            payload,
        )
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        if self.is_initialized {
            return Err(CheckpointError::ObjectVerification);
        }
        self.is_initialized = true;
        let Some(state) = state else {
            return Ok(());
        };
        if state.run() != &self.run {
            return Err(CheckpointError::ObjectVerification);
        }
        let restored: ResultParticipantState = serde_json::from_slice(state.payload_bytes())
            .map_err(|_| CheckpointError::ObjectVerification)?;
        self.terminal_horizon = restored.terminal_horizon;
        self.published_horizon = restored.published_horizon;
        self.authoritative_request_count = restored.authoritative_request_count;
        self.failed_action_count = restored.failed_action_count;
        self.committed_index_root = restored.committed_index_root;
        self.active_sessions = restored.active_sessions.into_iter().collect();
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
        if let Some(committed) = &self.committed_generation
            && receipt.generation() != committed
            && receipt.generation().epoch() <= committed.epoch()
        {
            return Err(CheckpointError::GenerationConflict {
                expected: Some(committed.clone()),
                actual: Some(receipt.generation().clone()),
            });
        }
        self.published_horizon = *receipt.represented_cut().terminal.get();
        self.committed_index_root = Some(*receipt.result_index_root());
        self.committed_generation = Some(receipt.generation().clone());
        self.committed_cut = Some(receipt.represented_cut().clone());
        Ok(())
    }
}

/// Canonical per-record projection written into one result payload.
#[derive(Serialize)]
struct EpochRecordProjection<'a> {
    global_sequence: GlobalSequence,
    logical_action_id: StableActionId,
    membership: &'static str,
    correlation_id: &'a str,
    start_ns: i64,
    end_ns: i64,
    errored: bool,
}

/// Refuse a sequence reused for a different logical action.
fn membership_agreement(
    existing: &RetainedTerminal,
    observed: &RetainedTerminal,
) -> Result<(), ResultPlaneError> {
    if existing.logical_action_id == observed.logical_action_id {
        return Ok(());
    }
    let membership = ResultMembershipKey::enumerated(vec![observed.logical_action_id])?;
    Err(ResultPlaneError::MembershipConflict {
        membership_root: membership.root(),
    })
}

/// Exact retained allocation charged to one provisional entry.
fn provisional_entry_bytes(retained: &RetainedTerminal) -> usize {
    size_of::<RetainedTerminal>()
        .saturating_add(retained.record.correlation_id.len())
        .saturating_add(retained.record.token_arrival_ns.len() * size_of::<i64>())
}

/// Build the committed metric map from records at or below the horizon only.
fn committed_metric_map(summary: &AccumulatorSummary) -> BTreeMap<String, MetricEntry> {
    NativeReport::new(summary, None).metrics
}
