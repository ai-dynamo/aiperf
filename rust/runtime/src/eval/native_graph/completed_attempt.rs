// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Sealed NativeGraph completion facts over frozen Harbor and rollout evidence.

use std::fmt::{self, Display, Formatter};

use crate::eval::{
    ArtifactDigest, AttemptId, EvidenceEvent, EvidenceKind, FrozenAttemptBundle, FrozenAttemptError,
};

use super::{
    FrozenRolloutEvidence, ResolvedEpisodeTrial, RolloutEvidenceIdentity, RolloutPolicyEvidence,
    RolloutReturnAgreementError, result::EpisodeExecution,
};

/// Immutable imported-attempt authority required before native rollout evidence can freeze.
///
/// The authority is derived only from an already-resolved, importer-owned trial. It binds the
/// rollout provenance to the exact trial digest and attempt identity that may receive its
/// lifecycle evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphAttemptAuthority {
    rollout_identity: RolloutEvidenceIdentity,
    rollout_policy_identity: Option<ArtifactDigest>,
    rollout_selection_digest: Option<ArtifactDigest>,
    trial_digest: ArtifactDigest,
    attempt_id: AttemptId,
}

impl NativeGraphAttemptAuthority {
    /// Derives trusted rollout provenance from one immutable resolved NativeGraph trial.
    pub fn from_resolved_trial(trial: &ResolvedEpisodeTrial) -> Self {
        let rollout = trial
            .imported()
            .package
            .native_graph()
            .and_then(|package| package.rollout());
        let rollout_selection_digest = rollout.map(|rollout| rollout.selection_digest());
        let rollout_policy_identity = rollout.map(|rollout| {
            RolloutPolicyEvidence::from_imported(
                rollout.policy().environment(),
                rollout.policy().horizon(),
                rollout.policy().gamma(),
            )
            .identity()
            .clone()
        });
        let mut rollout_identity = RolloutEvidenceIdentity::new(
            trial.imported().report.source_digest.clone(),
            trial.imported().task.digest.clone(),
            trial.trial().environment.clone(),
        );
        if let Some(selection_digest) = rollout_selection_digest.clone() {
            rollout_identity = rollout_identity.with_rollout_selection_digest(selection_digest);
        }
        Self {
            rollout_identity,
            rollout_policy_identity,
            rollout_selection_digest,
            trial_digest: trial.trial_digest().clone(),
            attempt_id: trial.attempt_id().clone(),
        }
    }

    /// Returns the provenance a trusted environment must embed in its frozen rollout evidence.
    pub fn rollout_identity(&self) -> RolloutEvidenceIdentity {
        self.rollout_identity.clone()
    }

    /// Reports whether the immutable imported trial selected a rollout.
    pub(crate) const fn requires_rollout_evidence(&self) -> bool {
        self.rollout_selection_digest.is_some()
    }

    /// Borrows the one resolved trial that may receive this rollout's lifecycle evidence.
    pub fn trial_digest(&self) -> &ArtifactDigest {
        &self.trial_digest
    }

    /// Borrows the one resolved attempt that may receive this rollout's lifecycle evidence.
    pub fn attempt_id(&self) -> &AttemptId {
        &self.attempt_id
    }
}

/// One fully frozen NativeGraph attempt, optionally bound to one validated RL rollout.
///
/// Its private fields ensure callers cannot add rollout evidence without first preserving
/// Harbor's verifier-input boundary and validating the rollout's independent provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphCompletedAttempt {
    attempt: FrozenAttemptBundle,
    rollout: Option<FrozenRolloutEvidence>,
}

impl NativeGraphCompletedAttempt {
    /// Retains already-frozen Harbor facts when this attempt has no RL rollout.
    pub fn from_frozen(attempt: FrozenAttemptBundle) -> Self {
        Self {
            attempt,
            rollout: None,
        }
    }

    /// Validates and freezes rollout evidence beside its immutable Harbor attempt facts.
    ///
    /// The authority comes from the immutable imported NativeGraph task and selected environment
    /// boundary. A rollout may add its digest only as one lifecycle artifact event; it is never
    /// made verifier input or reward material.
    pub fn freeze(
        authority: &NativeGraphAttemptAuthority,
        attempt: FrozenAttemptBundle,
        rollout: Option<FrozenRolloutEvidence>,
    ) -> Result<Self, NativeGraphCompletedAttemptError> {
        validate_attempt_authority(authority, &attempt)?;
        if authority.requires_rollout_evidence() && rollout.is_none() {
            return Err(NativeGraphCompletedAttemptError::RolloutEvidenceRequired);
        }
        let Some(rollout) = rollout else {
            return Ok(Self::from_frozen(attempt));
        };
        validate_rollout_identity(authority, &rollout)?;
        rollout
            .verifier_input()
            .verify_return_agreement()
            .map_err(NativeGraphCompletedAttemptError::ReturnAgreement)?;
        let attempt = append_rollout_lifecycle(attempt, &rollout)?;
        Ok(Self {
            attempt,
            rollout: Some(rollout),
        })
    }

    /// Borrows the ordinary frozen Harbor attempt facts.
    pub fn frozen_attempt(&self) -> &FrozenAttemptBundle {
        &self.attempt
    }

    /// Borrows the validated rollout evidence, when this attempt has one.
    pub fn rollout(&self) -> Option<&FrozenRolloutEvidence> {
        self.rollout.as_ref()
    }

    pub(crate) const fn has_rollout(&self) -> bool {
        self.rollout.is_some()
    }

    /// Transfers the frozen Harbor facts to a legacy evaluator that has no rollout-aware seam.
    pub(crate) fn into_frozen_attempt(self) -> FrozenAttemptBundle {
        self.attempt
    }

    pub(crate) fn execution(&self) -> EpisodeExecution {
        if self.rollout.as_ref().is_some_and(|rollout| {
            rollout
                .verifier_input()
                .transitions()
                .last()
                .is_some_and(|transition| transition.is_truncated())
        }) {
            EpisodeExecution::Truncated
        } else {
            EpisodeExecution::Completed
        }
    }
}

fn validate_attempt_authority(
    authority: &NativeGraphAttemptAuthority,
    attempt: &FrozenAttemptBundle,
) -> Result<(), NativeGraphCompletedAttemptError> {
    if attempt.trial_digest() != authority.trial_digest() {
        return Err(NativeGraphCompletedAttemptError::TrialIdentityMismatch);
    }
    if attempt.attempt() != authority.attempt_id() {
        return Err(NativeGraphCompletedAttemptError::AttemptIdentityMismatch);
    }
    Ok(())
}

fn validate_rollout_identity(
    authority: &NativeGraphAttemptAuthority,
    rollout: &FrozenRolloutEvidence,
) -> Result<(), NativeGraphCompletedAttemptError> {
    let expected = &authority.rollout_identity;
    let actual = rollout.verifier_input().identity();
    if actual.source() != expected.source() {
        return Err(NativeGraphCompletedAttemptError::SourceIdentityMismatch);
    }
    if actual.task() != expected.task() {
        return Err(NativeGraphCompletedAttemptError::TaskIdentityMismatch);
    }
    if actual.environment_implementation() != expected.environment_implementation() {
        return Err(NativeGraphCompletedAttemptError::EnvironmentIdentityMismatch);
    }
    let expected_selection = authority
        .rollout_selection_digest
        .as_ref()
        .ok_or(NativeGraphCompletedAttemptError::MissingRolloutPolicy)?;
    if actual.rollout_selection_digest() != expected_selection {
        return Err(NativeGraphCompletedAttemptError::RolloutSelectionIdentityMismatch);
    }
    let expected_policy = authority
        .rollout_policy_identity
        .as_ref()
        .ok_or(NativeGraphCompletedAttemptError::MissingRolloutPolicy)?;
    if rollout.verifier_input().policy().identity() != expected_policy {
        return Err(NativeGraphCompletedAttemptError::PolicyIdentityMismatch);
    }
    Ok(())
}

fn append_rollout_lifecycle(
    attempt: FrozenAttemptBundle,
    rollout: &FrozenRolloutEvidence,
) -> Result<FrozenAttemptBundle, NativeGraphCompletedAttemptError> {
    let mut lifecycle = attempt.lifecycle_evidence().to_vec();
    let sequence = u64::try_from(lifecycle.len())
        .map_err(|_| NativeGraphCompletedAttemptError::LifecycleSequenceOverflow)?;
    let parent = lifecycle.last().map(EvidenceEvent::identity_digest);
    lifecycle.push(rollout.lifecycle_evidence(attempt.attempt().clone(), sequence, parent));
    if let Some(policy_calls) = rollout.live_policy_calls() {
        let sequence = u64::try_from(lifecycle.len())
            .map_err(|_| NativeGraphCompletedAttemptError::LifecycleSequenceOverflow)?;
        let parent = lifecycle.last().map(EvidenceEvent::identity_digest);
        lifecycle.push(EvidenceEvent::new(
            attempt.attempt().clone(),
            sequence,
            EvidenceKind::Llm,
            policy_calls.identity_digest(),
            parent,
        ));
    }
    FrozenAttemptBundle::new(
        attempt.trial_digest().clone(),
        attempt.verifier_result().clone(),
        lifecycle,
        attempt.score_lineage().to_vec(),
    )
    .map_err(NativeGraphCompletedAttemptError::Frozen)
}

/// Failure while binding frozen rollout facts to a frozen NativeGraph attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NativeGraphCompletedAttemptError {
    /// The Harbor attempt did not belong to the authority's immutable resolved trial.
    TrialIdentityMismatch,
    /// The Harbor attempt did not belong to the authority's immutable expanded attempt.
    AttemptIdentityMismatch,
    /// Rollout source provenance disagreed with the immutable imported task source.
    SourceIdentityMismatch,
    /// Rollout task provenance disagreed with the immutable selected task.
    TaskIdentityMismatch,
    /// Rollout environment implementation provenance disagreed with the selected environment.
    EnvironmentIdentityMismatch,
    /// The imported task did not select a rollout policy for this attempt.
    MissingRolloutPolicy,
    /// The imported task selected a rollout, but the completion omitted its evidence.
    RolloutEvidenceRequired,
    /// Rollout policy facts disagreed with the imported package-selected policy.
    PolicyIdentityMismatch,
    /// Rollout selection facts disagreed with the immutable imported package selection.
    RolloutSelectionIdentityMismatch,
    /// The frozen rollout failed independent return or terminal-shape validation.
    ReturnAgreement(RolloutReturnAgreementError),
    /// Existing lifecycle evidence could not allocate a following sequence value.
    LifecycleSequenceOverflow,
    /// Re-freezing the append-only Harbor lifecycle facts failed validation.
    Frozen(FrozenAttemptError),
}

impl Display for NativeGraphCompletedAttemptError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::TrialIdentityMismatch => {
                formatter.write_str("completed Harbor attempt does not match the resolved trial")
            }
            Self::AttemptIdentityMismatch => {
                formatter.write_str("completed Harbor attempt does not match the resolved attempt")
            }
            Self::SourceIdentityMismatch => formatter
                .write_str("rollout source provenance disagrees with the completed attempt"),
            Self::TaskIdentityMismatch => {
                formatter.write_str("rollout task provenance disagrees with the completed attempt")
            }
            Self::EnvironmentIdentityMismatch => formatter
                .write_str("rollout environment provenance disagrees with the completed attempt"),
            Self::MissingRolloutPolicy => {
                formatter.write_str("completed attempt has no imported rollout policy")
            }
            Self::RolloutEvidenceRequired => formatter
                .write_str("completed rollout attempt omitted its required rollout evidence"),
            Self::PolicyIdentityMismatch => {
                formatter.write_str("rollout policy disagrees with the completed attempt")
            }
            Self::RolloutSelectionIdentityMismatch => {
                formatter.write_str("rollout selection disagrees with the completed attempt")
            }
            Self::ReturnAgreement(error) => error.fmt(formatter),
            Self::LifecycleSequenceOverflow => {
                formatter.write_str("rollout lifecycle evidence sequence overflowed")
            }
            Self::Frozen(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for NativeGraphCompletedAttemptError {}
