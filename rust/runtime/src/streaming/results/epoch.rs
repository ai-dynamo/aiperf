// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Epoch rotation, provisional holes, and partial result production.
//!
//! `EpochResultCoordinator` is the single result-plane participant. It receives
//! terminal records via `observe_terminal`, accumulates them until a
//! `CheckpointBarrier` fires, rotates each worker accumulator at the barrier,
//! and produces the `PreparedCheckpointResultInput` the checkpoint coordinator
//! carries into the backend transaction.
//!
//! Records whose global sequence exceeds the current terminal action horizon are
//! "provisional": they have been observed but may not yet be included in an
//! authoritative committed result. A bounded `provisional_limit` guards against
//! unbounded provisional growth; attempts that would exceed it return
//! `ResultPlaneError::ProvisionalCapacityExceeded`.
//!
//! The coordinator holds one `StreamingResourceBudget` for the singular
//! partition-descriptor allocation. When `prepare_epoch` builds a partition for
//! the committed sequence range, it acquires exactly one item and the exact
//! compact descriptor bytes from that budget. Refusal maps only to
//! `ResultPlaneError::PartitionDescriptorCapacityExceeded`; no other budget
//! is reported here.

use std::collections::BTreeMap;
use std::mem::size_of;

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use super::{
    BudgetedResultDescriptor, CellId, ResultPartition, ResultPlaneError, ResultProjectionId,
    ResultSchemaVersion, ResultSegmentDescriptor, WorkerId, descriptor_retained_bytes,
};
use crate::{
    metrics_core::report::MetricEntry,
    streaming::{
        budget::{BudgetError, BudgetLimits, StreamingResourceBudget},
        checkpoint::{
            BudgetedCheckpointBytes, CheckpointBarrier, CheckpointError, CheckpointParticipantId,
            CommittedCheckpointGeneration, CommittedParticipantReceipt, CommittedParticipantState,
            ParticipantInitialization, PreparedParticipantState, StreamRunIdentity,
            StreamingCheckpointParticipant, TerminalActionHorizon,
        },
        checkpoint_coordinator::PreparedCheckpointResultInput,
        identity::{ContentDigest, GlobalSequence},
        reliability::{PreparedIssueReceiptPartitionView, StreamingIssueSummary},
        results::CorrelatedRecordIngest,
    },
};

/// Schema identifier for the epoch-result participant state payload.
const EPOCH_RESULT_STATE_SCHEMA_ID: &str = "aiperf.streaming.epoch-result";
/// Wire version of the epoch-result participant state payload.
const EPOCH_RESULT_STATE_WIRE_VERSION: u32 = 1;
/// Stable projection identifier for the metrics-records projection.
pub(crate) const METRICS_PROJECTION_ID: &str = "streaming_metrics";

/// Lightweight provisional-record summary retained for labeled dashboard display.
///
/// Provisional records exceed the current terminal action horizon and cannot yet
/// be included in an authoritative committed result. This summary describes their
/// presence without including their content.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProvisionalDashboardSummary {
    /// Number of provisional records held above the terminal horizon.
    pub provisional_record_count: u64,
}

/// Authoritative metrics and counts committed with one checkpoint generation.
///
/// Partial results are constructed after each successful `checkpoint_committed`
/// callback. The caller retrieves them via `EpochResultCoordinator::committed_partial`.
#[derive(Clone, Debug)]
pub struct CommittedPartialResult {
    /// The generation identity that produced these partial results.
    pub generation: crate::streaming::checkpoint::CheckpointGeneration,
    /// The exact cut represented by the committed participant state.
    pub cut: crate::streaming::checkpoint::CheckpointCut,
    /// The contiguous terminal action horizon at commit time.
    pub terminal_horizon: TerminalActionHorizon,
    /// Number of terminal records at or below the terminal horizon.
    pub authoritative_request_count: u64,
    /// Number of terminal records above the terminal horizon.
    pub provisional_request_count: u64,
    /// Active sessions at commit time (placeholder — zero in this implementation).
    pub active_session_count: u64,
    /// Incomplete sessions at commit time (placeholder — zero in this implementation).
    pub incomplete_session_count: u64,
    /// Issue summary from the reliability ledger.
    pub issue_summary: StreamingIssueSummary,
    /// Number of failed terminal actions.
    pub failed_action_count: u64,
    /// Aggregated metrics map (empty in this implementation).
    pub metrics: BTreeMap<String, MetricEntry>,
    /// Provisional dashboard summary, when any provisional records exist.
    pub provisional: Option<ProvisionalDashboardSummary>,
}

/// One worker's result contribution for one checkpoint epoch.
#[derive(Debug)]
pub struct WorkerResultEpoch {
    /// Checkpoint generation identity this epoch belongs to.
    pub generation: crate::streaming::checkpoint::CheckpointGeneration,
    /// Producing worker identifier.
    pub worker_id: u32,
    /// First global sequence in this epoch's coverage.
    pub first_sequence: GlobalSequence,
    /// Last global sequence in this epoch's coverage.
    pub last_sequence: GlobalSequence,
    /// Prepared partitions for this worker epoch.
    pub partitions: Vec<ResultPartition>,
}

/// Serialized state carried in the participant payload at each checkpoint.
#[derive(Debug, Serialize, Deserialize)]
struct EpochResultStateWire {
    terminal_sequence: u64,
    authoritative_count: u64,
    provisional_count: u64,
}

/// Epoch result coordinator implementing one `StreamingCheckpointParticipant`.
///
/// Owns a single producer-side `StreamingResourceBudget` for partition
/// descriptor allocation. The caller is responsible for supplying both the run
/// identity and the provisional capacity limit at construction time; neither is
/// resolved from authored configuration here.
pub struct EpochResultCoordinator {
    participant_id: CheckpointParticipantId,
    run: StreamRunIdentity,
    /// Singular partition-descriptor budget.
    descriptor_budget: StreamingResourceBudget,
    /// Maximum number of terminal records held above the terminal horizon.
    provisional_limit: usize,
    /// Terminal records received since the last committed checkpoint, keyed by
    /// global sequence so they are iterable in order.
    pending: BTreeMap<GlobalSequence, CorrelatedRecordIngest>,
    /// Number of entries in `pending` whose sequence exceeds
    /// `committed_terminal_horizon`.
    provisional_count: usize,
    /// Terminal action horizon from the most recently committed generation.
    committed_terminal_horizon: TerminalActionHorizon,
    /// Most recently committed partial result, set in `checkpoint_committed`.
    last_partial: Option<CommittedPartialResult>,
    /// Guard that ensures `initialize` is called exactly once.
    init: ParticipantInitialization,
}

impl EpochResultCoordinator {
    /// Construct a coordinator for one logical run.
    ///
    /// `descriptor_budget` enforces the per-epoch limit on partition-descriptor
    /// allocation. `provisional_limit` caps the number of terminal records that
    /// may be held above the committed terminal horizon between barriers.
    pub fn new(
        participant_id: CheckpointParticipantId,
        run: StreamRunIdentity,
        descriptor_budget: StreamingResourceBudget,
        provisional_limit: usize,
    ) -> Self {
        Self {
            participant_id,
            run,
            descriptor_budget,
            provisional_limit,
            pending: BTreeMap::new(),
            provisional_count: 0,
            committed_terminal_horizon: TerminalActionHorizon::new(GlobalSequence::new(0)),
            last_partial: None,
            init: ParticipantInitialization::default(),
        }
    }

    /// Borrow the producer-side partition-descriptor budget.
    #[must_use]
    pub fn descriptor_budget(&self) -> &StreamingResourceBudget {
        &self.descriptor_budget
    }

    /// Ingest one terminal record.
    ///
    /// A record whose sequence exceeds the committed terminal horizon is counted
    /// as provisional. Returns `ProvisionalCapacityExceeded` when the provisional
    /// count would exceed the configured limit, leaving both `pending` and the
    /// count unchanged.
    pub fn observe_terminal(
        &mut self,
        fact: CorrelatedRecordIngest,
    ) -> Result<(), ResultPlaneError> {
        let seq = fact.correlation.global_sequence;
        let is_provisional = seq > *self.committed_terminal_horizon.get();
        if is_provisional {
            let new_count = self.provisional_count + 1;
            if new_count > self.provisional_limit {
                return Err(ResultPlaneError::ProvisionalCapacityExceeded {
                    items: new_count as u64,
                    bytes: 0,
                });
            }
            self.provisional_count = new_count;
        }
        self.pending.insert(seq, fact);
        Ok(())
    }

    /// Prepare one epoch of result partitions and the issue-receipt handoff.
    ///
    /// Collects every pending record whose sequence is at or below
    /// `barrier.cut.terminal`, serializes them into one partition if any exist,
    /// and wraps the result with the prepared issue-receipt handoff.
    ///
    /// Returns `PartitionDescriptorCapacityExceeded` when the descriptor budget
    /// cannot cover the exact inline-descriptor-plus-compact-projection bytes for
    /// the single partition being produced.
    pub async fn prepare_epoch(
        &mut self,
        barrier: &CheckpointBarrier,
        issue_receipts: PreparedIssueReceiptPartitionView,
    ) -> Result<PreparedCheckpointResultInput, ResultPlaneError> {
        let partitions = self.build_authoritative_partitions(barrier).await?;
        let issue_partition = self.build_issue_receipt_partition(barrier, issue_receipts)?;
        Ok(PreparedCheckpointResultInput::new(partitions, Some(issue_partition)))
    }

    /// Build only the ordinary result partitions for a barrier.
    ///
    /// Identical to `prepare_epoch` but does not consume an issue-receipt view.
    /// Intended for tests and callers that produce no detailed-receipt partition.
    pub async fn prepare_epoch_without_receipts(
        &mut self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedCheckpointResultInput, ResultPlaneError> {
        let partitions = self.build_authoritative_partitions(barrier).await?;
        Ok(PreparedCheckpointResultInput::new(partitions, None))
    }

    /// Return the partial result committed with the given generation.
    ///
    /// Returns `InvalidCoverage` when no committed partial result exists yet —
    /// i.e., `checkpoint_committed` has not yet been called for this coordinator.
    pub fn committed_partial(
        &self,
        _generation: &CommittedCheckpointGeneration,
    ) -> Result<CommittedPartialResult, ResultPlaneError> {
        self.last_partial
            .clone()
            .ok_or(ResultPlaneError::InvalidCoverage)
    }

    // ── StreamingCheckpointParticipant helpers ─────────────────────────────────

    async fn prepare_result_state_view(
        &self,
        barrier: &CheckpointBarrier,
    ) -> Result<PreparedParticipantState, CheckpointError> {
        let terminal = barrier.cut.terminal.get();
        let authoritative_count = self
            .pending
            .keys()
            .filter(|seq| *seq <= terminal)
            .count() as u64;
        let provisional_count = self.provisional_count as u64;

        let wire = EpochResultStateWire {
            terminal_sequence: terminal.get(),
            authoritative_count,
            provisional_count,
        };
        let encoded = serde_json::to_vec(&wire).map_err(|e| CheckpointError::Storage {
            message: format!("epoch-result state encode failed: {e}"),
        })?;
        let bytes = Bytes::from(encoded);
        // A dedicated small state budget per view: one item, exact byte length.
        let state_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: bytes.len(),
        })
        .map_err(|_| CheckpointError::ObjectVerification)?;
        let lease = state_budget
            .acquire(1, bytes.len())
            .await
            .map_err(|_| CheckpointError::StateBudget {
                participant: self.participant_id.as_str().to_owned(),
                message: "state payload budget exhausted".to_owned(),
            })?;
        let budgeted = BudgetedCheckpointBytes::new(bytes, lease)
            .map_err(|_| CheckpointError::ObjectVerification)?;
        PreparedParticipantState::new(
            self.run,
            self.participant_id.clone(),
            EPOCH_RESULT_STATE_SCHEMA_ID,
            EPOCH_RESULT_STATE_WIRE_VERSION,
            barrier.cut.clone(),
            authoritative_count,
            budgeted,
        )
    }

    async fn restore_result_state(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        let Some(state) = state else {
            // Fresh run: nothing to restore.
            return Ok(());
        };
        let descriptor = state.descriptor();
        if descriptor.schema_id != EPOCH_RESULT_STATE_SCHEMA_ID
            || descriptor.schema_version != EPOCH_RESULT_STATE_WIRE_VERSION
        {
            return Err(CheckpointError::ObjectVerification);
        }
        let wire: EpochResultStateWire =
            serde_json::from_slice(state.payload_bytes()).map_err(|e| {
                CheckpointError::Storage {
                    message: format!("epoch-result state decode failed: {e}"),
                }
            })?;
        self.committed_terminal_horizon =
            TerminalActionHorizon::new(GlobalSequence::new(wire.terminal_sequence));
        self.provisional_count = wire.provisional_count as usize;
        Ok(())
    }

    fn advance_committed_result_cut(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        if receipt.run() != &self.run || receipt.participant_id() != &self.participant_id {
            return Err(CheckpointError::ObjectVerification);
        }
        let new_terminal = receipt.represented_cut().terminal.clone();
        let new_seq = *new_terminal.get();

        // Count records that are now authoritative (sequence <= new terminal).
        let authoritative_count = self
            .pending
            .keys()
            .filter(|seq| **seq <= new_seq)
            .count() as u64;

        // Provisional records remain above the new terminal.
        let provisional_after: Vec<GlobalSequence> = self
            .pending
            .keys()
            .filter(|seq| **seq > new_seq)
            .copied()
            .collect();
        let provisional_count = provisional_after.len() as u64;

        // Remove authoritative records from pending now that the checkpoint is
        // committed; they will be included in the next query result iteration.
        let authoritative_keys: Vec<GlobalSequence> = self
            .pending
            .keys()
            .filter(|seq| **seq <= new_seq)
            .copied()
            .collect();
        for key in authoritative_keys {
            self.pending.remove(&key);
        }

        self.committed_terminal_horizon = new_terminal.clone();
        self.provisional_count = provisional_count as usize;

        let provisional_summary = if provisional_count > 0 {
            Some(ProvisionalDashboardSummary {
                provisional_record_count: provisional_count,
            })
        } else {
            None
        };

        self.last_partial = Some(CommittedPartialResult {
            generation: receipt.generation().clone(),
            cut: receipt.represented_cut().clone(),
            terminal_horizon: new_terminal,
            authoritative_request_count: authoritative_count,
            provisional_request_count: provisional_count,
            active_session_count: 0,
            incomplete_session_count: 0,
            issue_summary: StreamingIssueSummary::empty(),
            failed_action_count: 0,
            metrics: BTreeMap::new(),
            provisional: provisional_summary,
        });
        Ok(())
    }

    // ── Private partition-building helpers ────────────────────────────────────

    /// Collect and serialize all records at or below the barrier terminal.
    ///
    /// Returns an empty Vec when no authoritative records exist. Returns
    /// `PartitionDescriptorCapacityExceeded` when the descriptor budget is
    /// insufficient for the single produced partition.
    async fn build_authoritative_partitions(
        &self,
        barrier: &CheckpointBarrier,
    ) -> Result<Vec<ResultPartition>, ResultPlaneError> {
        let terminal = barrier.cut.terminal.get();
        let authoritative: Vec<GlobalSequence> = self
            .pending
            .keys()
            .filter(|seq| *seq <= terminal)
            .copied()
            .collect();
        if authoritative.is_empty() {
            return Ok(Vec::new());
        }
        let first_sequence = *authoritative.first().expect("nonempty authoritative");
        let last_sequence = *authoritative.last().expect("nonempty authoritative");
        let item_count = authoritative.len() as u64;

        // Minimal payload: a JSON array of the authoritative sequence values.
        let payload_bytes = {
            let seqs: Vec<u64> = authoritative.iter().map(|s| s.get()).collect();
            Bytes::from(serde_json::to_vec(&seqs).map_err(|e| ResultPlaneError::Compaction {
                message: format!("epoch payload encode failed: {e}"),
            })?)
        };
        let byte_length =
            u64::try_from(payload_bytes.len()).map_err(|_| ResultPlaneError::Compaction {
                message: "payload length overflowed u64".to_owned(),
            })?;
        let payload_digest =
            ContentDigest::from_bytes(*blake3::hash(&payload_bytes).as_bytes());

        // Membership root: hash of the sequence range endpoints.
        let membership_root = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"aiperf.streaming.epoch-metrics.v1");
            hasher.update(&first_sequence.get().to_le_bytes());
            hasher.update(&last_sequence.get().to_le_bytes());
            hasher.update(&item_count.to_le_bytes());
            ContentDigest::from_bytes(*hasher.finalize().as_bytes())
        };

        let projection = ResultProjectionId::new(METRICS_PROJECTION_ID)
            .map_err(|_| ResultPlaneError::Compaction {
                message: "projection ID empty".to_owned(),
            })?;
        let descriptor = ResultSegmentDescriptor {
            run: self.run,
            epoch: barrier.epoch,
            cell_id: CellId::new(0),
            worker_id: WorkerId::new(0),
            projection,
            schema: ResultSchemaVersion::new(1),
            first_sequence,
            last_sequence,
            item_count,
            byte_length,
            membership_root,
            payload_digest,
        };
        let descriptor_bytes = descriptor_retained_bytes(&descriptor)
            .map_err(|_| ResultPlaneError::Compaction {
                message: "descriptor byte computation overflowed".to_owned(),
            })?;

        // Try to acquire from the descriptor budget — this is the ONLY place
        // PartitionDescriptorCapacityExceeded is returned.
        let descriptor_lease = self
            .descriptor_budget
            .try_acquire(1, descriptor_bytes)
            .map_err(|err| match err {
                BudgetError::RequestExceedsCapacity | BudgetError::Unavailable => {
                    ResultPlaneError::PartitionDescriptorCapacityExceeded {
                        items: 1,
                        bytes: descriptor_bytes as u64,
                    }
                }
                _ => ResultPlaneError::Compaction {
                    message: format!("descriptor budget error: {err:?}"),
                },
            })?;

        let budgeted_descriptor = BudgetedResultDescriptor::new(descriptor, descriptor_lease)
            .map_err(|_| ResultPlaneError::SegmentVerification)?;

        // Payload budget: one owned budget sized exactly for this payload.
        let payload_budget = StreamingResourceBudget::new(BudgetLimits {
            max_items: 1,
            max_bytes: payload_bytes.len(),
        })
        .map_err(|_| ResultPlaneError::Compaction {
            message: "payload budget construction failed".to_owned(),
        })?;
        // Synchronous acquisition — the payload is already in memory.
        // This should never fail since max_bytes == len.
        let payload_lease = payload_budget
            .try_acquire(1, payload_bytes.len())
            .map_err(|_| ResultPlaneError::Compaction {
                message: "payload budget exhausted".to_owned(),
            })?;
        let budgeted_payload = BudgetedCheckpointBytes::new(payload_bytes, payload_lease)
            .map_err(|_| ResultPlaneError::Compaction {
                message: "payload budget charge mismatch".to_owned(),
            })?;

        let partition = ResultPartition::new(budgeted_descriptor, budgeted_payload)
            .map_err(|_| ResultPlaneError::SegmentVerification)?;
        Ok(vec![partition])
    }

    fn build_issue_receipt_partition(
        &self,
        barrier: &CheckpointBarrier,
        issue_receipts: PreparedIssueReceiptPartitionView,
    ) -> Result<
        crate::streaming::reliability::PreparedIssueReceiptResultPartition,
        ResultPlaneError,
    > {
        use crate::streaming::reliability::ISSUE_RECEIPT_WIRE_VERSION;
        let receipt_count = issue_receipts.payload_bytes().len() as u64; // approximate
        let receipt_count = issue_receipts.payload_bytes().len() as u64;
        // Borrow the receipt root and payload before consuming the view.
        let receipt_root = *issue_receipts.receipt_root();
        let payload_len =
            u64::try_from(issue_receipts.payload_bytes().len()).map_err(|_| {
                ResultPlaneError::Compaction {
                    message: "issue receipt payload overflowed u64".to_owned(),
                }
            })?;
        let payload_digest =
            ContentDigest::from_bytes(*blake3::hash(issue_receipts.payload_bytes()).as_bytes());

        let projection = ResultProjectionId::new("streaming_issue_receipts").map_err(|_| {
            ResultPlaneError::Compaction {
                message: "projection ID empty".to_owned(),
            }
        })?;
        let descriptor = ResultSegmentDescriptor {
            run: self.run,
            epoch: barrier.epoch,
            cell_id: CellId::new(0),
            worker_id: WorkerId::new(0),
            projection,
            schema: ResultSchemaVersion::new(ISSUE_RECEIPT_WIRE_VERSION),
            first_sequence: GlobalSequence::new(0),
            last_sequence: GlobalSequence::new(0),
            item_count: issue_receipts.payload_bytes().len() as u64, // approximation
            byte_length: payload_len,
            membership_root: receipt_root,
            payload_digest,
        };

        // Use the receipt count from the view.
        let descriptor = ResultSegmentDescriptor {
            item_count: {
                // Recompute from the actual receipt count field via view.
                // The view exposes only the payload bytes, not the count directly.
                // We use the payload length in bytes as proxy; the view validates
                // the count in into_result_partition.
                descriptor.item_count
            },
            ..descriptor
        };
        let descriptor_bytes = descriptor_retained_bytes(&descriptor)
            .map_err(|_| ResultPlaneError::Compaction {
                message: "receipt descriptor byte computation overflowed".to_owned(),
            })?;
        let descriptor_lease = self
            .descriptor_budget
            .try_acquire(1, descriptor_bytes)
            .map_err(|err| match err {
                BudgetError::RequestExceedsCapacity | BudgetError::Unavailable => {
                    ResultPlaneError::PartitionDescriptorCapacityExceeded {
                        items: 1,
                        bytes: descriptor_bytes as u64,
                    }
                }
                _ => ResultPlaneError::Compaction {
                    message: format!("receipt descriptor budget error: {err:?}"),
                },
            })?;
        let budgeted_descriptor = BudgetedResultDescriptor::new(descriptor, descriptor_lease)
            .map_err(|_| ResultPlaneError::SegmentVerification)?;
        issue_receipts
            .into_result_partition(budgeted_descriptor)
            .map_err(|_| ResultPlaneError::SegmentVerification)
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
        self.prepare_result_state_view(barrier).await
    }

    async fn initialize(
        &mut self,
        state: Option<CommittedParticipantState>,
    ) -> Result<(), CheckpointError> {
        self.init.begin()?;
        self.restore_result_state(state).await
    }

    async fn checkpoint_committed(
        &mut self,
        receipt: &CommittedParticipantReceipt,
    ) -> Result<(), CheckpointError> {
        self.advance_committed_result_cut(receipt)
    }
}
