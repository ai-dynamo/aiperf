// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic generation backend contract and shared publication prevalidation.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::{
    checkpoint::{
        CheckpointCut, CheckpointEpoch, CheckpointError, CheckpointGeneration,
        CheckpointGenerationCandidate, CheckpointParticipantPlan, CheckpointTerminalReason,
        CommittedCheckpointGeneration, CommittedParticipantState, ParticipantStateDescriptor,
        PreparedParticipantState, PrevalidatedCheckpointGenerationCandidate, StreamRunIdentity,
    },
    identity::ContentDigest,
    results::{
        PreparedResultEpoch, ResultIndexCursor, ResultIndexPage, ResultIndexReadBudget,
        ResultPartition, ResultSegmentDescriptor, ResultSegmentReader,
    },
};

const INITIAL_CHECKPOINT_EPOCH: CheckpointEpoch = CheckpointEpoch::new(1);

/// Caller-supplied semantic metadata for one atomic generation publication.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CheckpointCommitMetadata {
    /// Complete predecessor expected by the writer.
    pub previous: Option<CheckpointGeneration>,
    /// Exact successor epoch being committed.
    pub epoch: CheckpointEpoch,
    /// Complete checkpoint cut represented by every participant.
    pub cut: CheckpointCut,
    /// Semantic digest of the execution plan.
    pub execution_plan_digest: ContentDigest,
    /// Semantic digest of the result plan.
    pub result_plan_digest: ContentDigest,
    /// Whether this generation terminates the logical run.
    pub is_final: bool,
    /// Terminal reason, present exactly for final generations.
    pub terminal_reason: Option<CheckpointTerminalReason>,
}

/// Exact run and semantic plan expected while opening or writing generations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointGenerationExpectations {
    /// Logical run being opened or written.
    pub run: StreamRunIdentity,
    /// Frozen exact participant inventory.
    pub participant_plan: CheckpointParticipantPlan,
    /// Frozen execution-plan semantic digest.
    pub execution_plan_digest: ContentDigest,
    /// Frozen result-plan semantic digest.
    pub result_plan_digest: ContentDigest,
}

/// Atomic streaming checkpoint storage backend.
#[async_trait(?Send)]
pub trait StreamingCheckpointBackend {
    /// Open and verify the latest authoritative generation for one exact run.
    async fn open_latest(
        &self,
        run: &StreamRunIdentity,
        expected: &CheckpointGenerationExpectations,
    ) -> Result<Option<Box<dyn LeasedGenerationReader>>, CheckpointError>;

    /// Begin a transaction frozen to one exact run, head, and semantic plan.
    async fn begin_generation(
        &self,
        run: StreamRunIdentity,
        expected: Option<CheckpointGeneration>,
        expectations: CheckpointGenerationExpectations,
    ) -> Result<Box<dyn StreamingGenerationTransaction>, CheckpointError>;
}

/// Read authority scoped to the reachable objects of one committed generation.
#[async_trait(?Send)]
pub trait LeasedGenerationReader {
    /// Borrow the authoritative committed generation.
    fn generation(&self) -> &CommittedCheckpointGeneration;

    /// Scan reachable result descriptors under caller and backend budgets.
    async fn scan_result_index(
        &self,
        after: Option<ResultIndexCursor>,
        budget: ResultIndexReadBudget,
    ) -> Result<ResultIndexPage, CheckpointError>;

    /// Read one exact result segment reachable from this generation.
    async fn read_segment(
        &self,
        descriptor: &ResultSegmentDescriptor,
    ) -> Result<ResultSegmentReader, CheckpointError>;

    /// Read one exact participant object reachable from this generation.
    async fn read_participant(
        &self,
        descriptor: &ParticipantStateDescriptor,
    ) -> Result<CommittedParticipantState, CheckpointError>;
}

/// One atomic streaming generation transaction.
#[async_trait(?Send)]
pub trait StreamingGenerationTransaction {
    /// Stage one participant's prepared immutable state.
    async fn stage_participant(
        &mut self,
        state: PreparedParticipantState,
    ) -> Result<(), CheckpointError>;

    /// Stage exactly one complete result epoch.
    async fn stage_results(
        &mut self,
        partitions: &mut Vec<ResultPartition>,
    ) -> Result<PreparedResultEpoch, CheckpointError>;

    /// Atomically publish the fully staged generation.
    async fn commit(
        self: Box<Self>,
        metadata: CheckpointCommitMetadata,
    ) -> Result<CommittedCheckpointGeneration, CheckpointError>;
}

pub(crate) struct ValidatedCommitMetadata {
    previous_digest: Option<ContentDigest>,
    epoch: CheckpointEpoch,
    metadata: CheckpointCommitMetadata,
}

impl ValidatedCommitMetadata {
    pub(crate) const fn epoch(&self) -> CheckpointEpoch {
        self.epoch
    }
}

pub(crate) fn validate_commit_metadata(
    expected: &Option<CheckpointGeneration>,
    metadata: CheckpointCommitMetadata,
) -> Result<ValidatedCommitMetadata, CheckpointError> {
    if metadata.previous.as_ref() != expected.as_ref() {
        return Err(CheckpointError::ObjectVerification);
    }
    let epoch = match expected {
        None => INITIAL_CHECKPOINT_EPOCH,
        Some(previous) => {
            CheckpointEpoch::new(previous.epoch().get().checked_add(1).ok_or_else(|| {
                CheckpointError::GenerationEpochOverflow {
                    previous: previous.clone(),
                }
            })?)
        }
    };
    if metadata.epoch != epoch {
        return Err(CheckpointError::ObjectVerification);
    }
    Ok(ValidatedCommitMetadata {
        previous_digest: expected.as_ref().map(|generation| *generation.digest()),
        epoch,
        metadata,
    })
}

pub(crate) struct FrozenGenerationTransactionInputs {
    run: StreamRunIdentity,
    expectations: CheckpointGenerationExpectations,
    participant_descriptors: Vec<ParticipantStateDescriptor>,
    result_index_root: ContentDigest,
}

impl FrozenGenerationTransactionInputs {
    pub(crate) fn new(
        run: StreamRunIdentity,
        expectations: CheckpointGenerationExpectations,
        participant_descriptors: Vec<ParticipantStateDescriptor>,
        result_index_root: ContentDigest,
    ) -> Self {
        Self {
            run,
            expectations,
            participant_descriptors,
            result_index_root,
        }
    }
}

pub(crate) fn build_prevalidated_candidate(
    transaction: FrozenGenerationTransactionInputs,
    validated: ValidatedCommitMetadata,
) -> Result<PrevalidatedCheckpointGenerationCandidate, CheckpointError> {
    if transaction.run != transaction.expectations.run {
        return Err(CheckpointError::ObjectVerification);
    }
    let ValidatedCommitMetadata {
        previous_digest,
        epoch,
        metadata,
    } = validated;
    CheckpointGenerationCandidate::new(
        transaction.run,
        epoch,
        previous_digest,
        metadata.cut,
        &transaction.expectations.participant_plan,
        metadata.execution_plan_digest,
        metadata.result_plan_digest,
        transaction.participant_descriptors,
        transaction.result_index_root,
        metadata.is_final,
        metadata.terminal_reason,
    )?
    .prevalidate_for_publication(
        &transaction.expectations.run,
        &transaction.expectations.participant_plan,
        &transaction.expectations.execution_plan_digest,
        &transaction.expectations.result_plan_digest,
    )
}
