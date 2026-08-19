// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Sealed NativeGraph completion facts over frozen Harbor and rollout evidence.

use std::fmt::{self, Display, Formatter};

use crate::eval::{
    ArtifactDigest, AttemptId, EvidenceEvent, EvidenceKind, FrozenAttemptBundle, FrozenAttemptError,
};

use super::{
    CapturePolicy, CompatibilityCaptureSession, CompatibilityTerminalSupplement, EpisodeFidelity,
    FrozenRolloutEvidence, NativeGraphProfile, ResolvedEpisodeTrial, RolloutEvidenceIdentity,
    RolloutPolicyEvidence, RolloutReturnAgreementError, result::EpisodeExecution,
};

/// Immutable imported-attempt authority required before native rollout evidence can freeze.
///
/// The authority is derived only from an already-resolved, importer-owned trial. It binds the
/// rollout provenance to the exact trial digest and attempt identity that may receive its
/// lifecycle evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct NativeGraphAttemptAuthority {
    profile: NativeGraphProfile,
    compatibility_capture_session: Option<CompatibilityCaptureSession>,
    rollout_identity: RolloutEvidenceIdentity,
    rollout_policy_identity: Option<ArtifactDigest>,
    rollout_selection_digest: Option<ArtifactDigest>,
    trial_digest: ArtifactDigest,
    attempt_id: AttemptId,
}

impl NativeGraphAttemptAuthority {
    /// Derives trusted rollout provenance from one immutable resolved NativeGraph trial.
    pub fn from_resolved_trial(trial: &ResolvedEpisodeTrial) -> Self {
        let package = trial.imported().package.native_graph();
        let rollout = package.and_then(|package| package.rollout());
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
            profile: package.map_or(NativeGraphProfile::NativeGraph, |package| package.profile()),
            compatibility_capture_session: package
                .filter(|package| package.profile() == NativeGraphProfile::ExternallyDriven)
                .and_then(|package| {
                    CapturePolicy::from_package(package).ok().map(|policy| {
                        CompatibilityCaptureSession::new(
                            policy.package_identity().clone(),
                            trial.imported().report.source_digest.clone(),
                            trial.imported().task.digest.clone(),
                            trial.trial().environment.clone(),
                            trial.trial_digest().clone(),
                            trial.attempt_id().clone(),
                        )
                    })
                }),
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

    /// Returns whether the immutable resolved trial selected the external compatibility profile.
    pub(crate) const fn is_externally_driven(&self) -> bool {
        matches!(self.profile, NativeGraphProfile::ExternallyDriven)
    }

    pub(crate) fn compatibility_capture_session(&self) -> Option<&CompatibilityCaptureSession> {
        self.compatibility_capture_session.as_ref()
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
    compatibility: Option<CompatibilityTerminalSupplement>,
    fidelity: EpisodeFidelity,
}

impl NativeGraphCompletedAttempt {
    /// Retains already-frozen Harbor facts when this attempt has no RL rollout.
    pub fn from_frozen(attempt: FrozenAttemptBundle) -> Self {
        Self {
            attempt,
            rollout: None,
            compatibility: None,
            fidelity: EpisodeFidelity::Legacy,
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
        if authority.is_externally_driven() {
            return Err(NativeGraphCompletedAttemptError::NativeRolloutRequiresNativeProfile);
        }
        if authority.requires_rollout_evidence() && rollout.is_none() {
            return Err(NativeGraphCompletedAttemptError::RolloutEvidenceRequired);
        }
        let Some(rollout) = rollout else {
            return Ok(Self {
                attempt,
                rollout: None,
                compatibility: None,
                fidelity: EpisodeFidelity::NativeGraph,
            });
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
            compatibility: None,
            fidelity: EpisodeFidelity::NativeGraph,
        })
    }

    /// Validates and freezes bounded external compatibility facts beside Harbor attempt facts.
    ///
    /// Compatibility facts append exactly one lifecycle event. They never modify verifier
    /// evidence, verifier reward, or score lineage.
    pub fn freeze_compatibility(
        authority: &NativeGraphAttemptAuthority,
        attempt: FrozenAttemptBundle,
        supplement: CompatibilityTerminalSupplement,
    ) -> Result<Self, NativeGraphCompletedAttemptError> {
        validate_attempt_authority(authority, &attempt)?;
        if !authority.is_externally_driven() {
            return Err(NativeGraphCompletedAttemptError::CompatibilityRequiresExternalProfile);
        }
        if authority.requires_rollout_evidence() {
            return Err(NativeGraphCompletedAttemptError::CompatibilityCannotUseRollout);
        }
        let expected_session = authority
            .compatibility_capture_session()
            .ok_or(NativeGraphCompletedAttemptError::CompatibilityRequiresExternalProfile)?;
        if supplement.session() != expected_session {
            return Err(NativeGraphCompletedAttemptError::CompatibilitySessionIdentityMismatch);
        }
        if attempt
            .lifecycle_evidence()
            .iter()
            .any(|event| event.kind == EvidenceKind::Compatibility)
        {
            return Err(NativeGraphCompletedAttemptError::CompatibilityLifecycleAlreadyPresent);
        }
        let attempt = append_compatibility_lifecycle(attempt, &supplement)?;
        Ok(Self {
            attempt,
            rollout: None,
            fidelity: EpisodeFidelity::ExternallyDriven(supplement.fidelity()),
            compatibility: Some(supplement),
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

    /// Borrows the sealed external compatibility facts, when this was externally driven.
    pub fn compatibility(&self) -> Option<&CompatibilityTerminalSupplement> {
        self.compatibility.as_ref()
    }

    pub(crate) const fn has_rollout(&self) -> bool {
        self.rollout.is_some()
    }

    pub(crate) const fn has_compatibility(&self) -> bool {
        self.compatibility.is_some()
    }

    pub(crate) const fn fidelity(&self) -> EpisodeFidelity {
        self.fidelity
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

fn append_compatibility_lifecycle(
    attempt: FrozenAttemptBundle,
    supplement: &CompatibilityTerminalSupplement,
) -> Result<FrozenAttemptBundle, NativeGraphCompletedAttemptError> {
    let mut lifecycle = attempt.lifecycle_evidence().to_vec();
    let sequence = u64::try_from(lifecycle.len())
        .map_err(|_| NativeGraphCompletedAttemptError::LifecycleSequenceOverflow)?;
    let parent = lifecycle.last().map(EvidenceEvent::identity_digest);
    lifecycle.push(supplement.lifecycle_evidence(attempt.attempt().clone(), sequence, parent));
    FrozenAttemptBundle::new(
        attempt.trial_digest().clone(),
        attempt.verifier_result().clone(),
        lifecycle,
        attempt.score_lineage().to_vec(),
    )
    .map_err(NativeGraphCompletedAttemptError::Frozen)
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
    /// A native rollout completion was attempted for an externally driven trial.
    NativeRolloutRequiresNativeProfile,
    /// Compatibility facts require the imported externally driven profile.
    CompatibilityRequiresExternalProfile,
    /// Compatibility facts cannot coexist with an imported native rollout.
    CompatibilityCannotUseRollout,
    /// Compatibility terminal facts do not belong to the resolved capture session.
    CompatibilitySessionIdentityMismatch,
    /// The frozen Harbor attempt already contains compatibility lifecycle evidence.
    CompatibilityLifecycleAlreadyPresent,
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
            Self::NativeRolloutRequiresNativeProfile => {
                formatter.write_str("native rollout completion requires the native_graph profile")
            }
            Self::CompatibilityRequiresExternalProfile => formatter
                .write_str("compatibility completion requires the externally_driven profile"),
            Self::CompatibilityCannotUseRollout => formatter
                .write_str("compatibility completion cannot attach native rollout evidence"),
            Self::CompatibilitySessionIdentityMismatch => formatter.write_str(
                "compatibility terminal facts disagree with the resolved capture session",
            ),
            Self::CompatibilityLifecycleAlreadyPresent => formatter
                .write_str("completed attempt already contains compatibility lifecycle evidence"),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{CaptureError, CompatibilityTerminalReceipt};
    use crate::eval::{
        EpisodeEvaluator, EpisodeScoreState, HarborEpisodeEvaluator, RewardDocument, ScoreVersion,
        VerifierResult,
    };

    fn authority(attempt: &str) -> NativeGraphAttemptAuthority {
        let package = ArtifactDigest::from_bytes(b"external-package");
        let source = ArtifactDigest::from_bytes(b"external-source");
        let task = ArtifactDigest::from_bytes(b"external-task");
        let environment = ArtifactDigest::from_bytes(b"external-environment");
        let trial = ArtifactDigest::from_bytes(b"external-trial");
        let attempt_id = AttemptId::new(attempt).expect("fixture attempt is valid");
        NativeGraphAttemptAuthority {
            profile: NativeGraphProfile::ExternallyDriven,
            compatibility_capture_session: Some(CompatibilityCaptureSession::new(
                package,
                source.clone(),
                task.clone(),
                environment.clone(),
                trial.clone(),
                attempt_id.clone(),
            )),
            rollout_identity: RolloutEvidenceIdentity::new(source, task, environment),
            rollout_policy_identity: None,
            rollout_selection_digest: None,
            trial_digest: trial,
            attempt_id,
        }
    }

    fn frozen_attempt(
        authority: &NativeGraphAttemptAuthority,
        has_compatibility_lifecycle: bool,
    ) -> FrozenAttemptBundle {
        let attempt = authority.attempt_id().clone();
        let verifier = VerifierResult::new(
            attempt.clone(),
            ArtifactDigest::from_bytes(b"verifier"),
            vec![ArtifactDigest::from_bytes(b"declared-verifier-artifact")],
            RewardDocument::parse(Some(br#"{"reward":0.75}"#), None)
                .expect("fixture reward is valid"),
            ArtifactDigest::from_bytes(b"verifier-rationale"),
        )
        .expect("fixture verifier result is valid");
        let score = ScoreVersion::initial(
            attempt.clone(),
            verifier.verifier.clone(),
            verifier.evidence.clone(),
            "reward",
            0.75,
            ArtifactDigest::from_bytes(b"score-rationale"),
        )
        .expect("fixture score is valid");
        let mut lifecycle = vec![EvidenceEvent::new(
            attempt.clone(),
            0,
            EvidenceKind::Evaluator,
            ArtifactDigest::from_bytes(b"existing-lifecycle"),
            None,
        )];
        if has_compatibility_lifecycle {
            let parent = lifecycle.last().map(EvidenceEvent::identity_digest);
            lifecycle.push(EvidenceEvent::new(
                attempt,
                1,
                EvidenceKind::Compatibility,
                ArtifactDigest::from_bytes(b"forged-compatibility"),
                parent,
            ));
        }
        FrozenAttemptBundle::new(
            authority.trial_digest().clone(),
            verifier,
            lifecycle,
            vec![score],
        )
        .expect("fixture Harbor facts freeze")
    }

    fn supplement(authority: &NativeGraphAttemptAuthority) -> CompatibilityTerminalSupplement {
        let session = authority
            .compatibility_capture_session()
            .expect("external authority has a capture session")
            .clone();
        let report = CapturePolicy::from_session(&session)
            .begin_observation()
            .freeze();
        let receipt = CompatibilityTerminalReceipt::from_canonical_terminal_bytes(
            session,
            br#"{"terminal":"accepted"}"#,
        )
        .expect("bounded fixture terminal receipt seals");
        report
            .into_terminal_supplement(receipt)
            .expect("matching report and receipt seal one supplement")
    }

    #[tokio::test(flavor = "current_thread")]
    async fn compatibility_completion_binds_one_session_and_preserves_harbor_scoring() {
        let authority = authority("external-attempt-a");
        let attempt = frozen_attempt(&authority, false);
        let verifier_evidence = attempt.verifier_input_evidence().to_vec();
        let reward = attempt.verifier_result().reward.clone();
        let scores = attempt.score_lineage().to_vec();
        let completed = NativeGraphCompletedAttempt::freeze_compatibility(
            &authority,
            attempt,
            supplement(&authority),
        )
        .expect("matching external session freezes exactly one compatibility event");

        assert_eq!(
            completed.frozen_attempt().verifier_input_evidence(),
            verifier_evidence
        );
        assert_eq!(completed.frozen_attempt().verifier_result().reward, reward);
        assert_eq!(completed.frozen_attempt().score_lineage(), scores);
        assert_eq!(completed.frozen_attempt().lifecycle_evidence().len(), 2);
        assert_eq!(
            completed.frozen_attempt().lifecycle_evidence()[1].kind,
            EvidenceKind::Compatibility
        );
        let result = HarborEpisodeEvaluator::new()
            .evaluate_native_graph(completed)
            .await
            .expect("sealed compatibility completion remains Harbor-scorable");
        assert!(result.fidelity().is_externally_driven());
        assert_eq!(result.score(), EpisodeScoreState::Verified { reward: 0.75 });
    }

    #[test]
    fn compatibility_completion_refuses_same_package_receipt_from_another_attempt() {
        let primary = authority("external-attempt-a");
        let foreign = authority("external-attempt-b");
        let error = NativeGraphCompletedAttempt::freeze_compatibility(
            &foreign,
            frozen_attempt(&foreign, false),
            supplement(&primary),
        )
        .expect_err("same-package receipt from another attempt cannot be replayed");

        assert_eq!(
            error,
            NativeGraphCompletedAttemptError::CompatibilitySessionIdentityMismatch
        );
    }

    #[test]
    fn compatibility_supplement_refuses_a_report_from_another_capture_session() {
        let report_authority = authority("external-attempt-a");
        let receipt_authority = authority("external-attempt-b");
        let report_session = report_authority
            .compatibility_capture_session()
            .expect("external authority has a capture session");
        let receipt_session = receipt_authority
            .compatibility_capture_session()
            .expect("external authority has a capture session")
            .clone();
        let report = CapturePolicy::from_session(report_session)
            .begin_observation()
            .freeze();
        let receipt = CompatibilityTerminalReceipt::from_canonical_terminal_bytes(
            receipt_session,
            br#"{"terminal":"accepted"}"#,
        )
        .expect("bounded fixture terminal receipt seals");

        assert_eq!(
            report
                .into_terminal_supplement(receipt)
                .expect_err("a report and receipt from different sessions cannot pair"),
            CaptureError::CaptureSessionIdentityMismatch
        );
    }

    #[test]
    fn compatibility_completion_refuses_a_native_no_rollout_authority() {
        let external = authority("external-attempt-a");
        let mut native = authority("external-attempt-a");
        native.profile = NativeGraphProfile::NativeGraph;
        native.compatibility_capture_session = None;

        let error = NativeGraphCompletedAttempt::freeze_compatibility(
            &native,
            frozen_attempt(&native, false),
            supplement(&external),
        )
        .expect_err("native no-rollout completion cannot accept compatibility evidence");

        assert_eq!(
            error,
            NativeGraphCompletedAttemptError::CompatibilityRequiresExternalProfile
        );
    }

    #[test]
    fn compatibility_completion_refuses_an_existing_compatibility_lifecycle_event() {
        let authority = authority("external-attempt-a");
        let error = NativeGraphCompletedAttempt::freeze_compatibility(
            &authority,
            frozen_attempt(&authority, true),
            supplement(&authority),
        )
        .expect_err("only one compatibility lifecycle event may be appended");

        assert_eq!(
            error,
            NativeGraphCompletedAttemptError::CompatibilityLifecycleAlreadyPresent
        );
    }
}
