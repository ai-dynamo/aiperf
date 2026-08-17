// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, AttemptId, EpisodeComparability, EpisodeEvaluator, EpisodeExecution,
    EpisodeIntegrity, EvidenceEvent, EvidenceKind, FrozenAttemptBundle, FrozenAttemptError,
    HarborEpisodeEvaluator, RegradeRequest, RewardDocument, ScoreVersion, VerifierResult, regrade,
};

fn frozen_attempt(reward: f64) -> FrozenAttemptBundle {
    let attempt = AttemptId::new("zero-score-attempt").unwrap();
    let verifier = VerifierResult::new(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"verifier"),
        vec![ArtifactDigest::from_bytes(b"patch")],
        RewardDocument::parse(Some(format!("{{\"reward\":{reward}}}").as_bytes()), None).unwrap(),
        ArtifactDigest::from_bytes(b"rationale"),
    )
    .unwrap();
    let initial = ScoreVersion::initial(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"verifier"),
        verifier.evidence.clone(),
        "reward",
        reward,
        ArtifactDigest::from_bytes(b"initial-rationale"),
    )
    .unwrap();
    let regrade_score =
        regrade(RegradeRequest::new(initial.clone(), verifier.clone(), "reward").unwrap()).unwrap();

    FrozenAttemptBundle::new(
        ArtifactDigest::from_bytes(b"trial"),
        verifier,
        vec![EvidenceEvent::new(
            attempt,
            0,
            EvidenceKind::Evaluator,
            ArtifactDigest::from_bytes(b"lifecycle"),
            None,
        )],
        vec![initial, regrade_score],
    )
    .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn evaluator_keeps_a_valid_zero_score_in_the_scored_episode_axes() {
    let attempt = frozen_attempt(0.0);
    let evidence = attempt.identity_digest();
    let result = HarborEpisodeEvaluator::new()
        .evaluate(attempt)
        .await
        .unwrap();

    assert_eq!(result.integrity(), EpisodeIntegrity::Valid);
    assert_eq!(result.execution(), EpisodeExecution::Completed);
    assert_eq!(result.verified_reward(), Some(0.0));
    assert_eq!(result.comparability(), EpisodeComparability::Scored);
    assert_eq!(result.evidence(), [evidence]);
}

#[test]
fn frozen_attempt_rejects_a_score_from_another_evaluator() {
    let attempt = AttemptId::new("foreign-evaluator-attempt").unwrap();
    let verifier = VerifierResult::new(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"verifier"),
        vec![ArtifactDigest::from_bytes(b"patch")],
        RewardDocument::parse(Some(br#"{"reward":0.0}"#), None).unwrap(),
        ArtifactDigest::from_bytes(b"rationale"),
    )
    .unwrap();
    let foreign_score = ScoreVersion::initial(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"foreign-evaluator"),
        verifier.evidence.clone(),
        "reward",
        0.0,
        ArtifactDigest::from_bytes(b"score-rationale"),
    )
    .unwrap();

    let error = FrozenAttemptBundle::new(
        ArtifactDigest::from_bytes(b"trial"),
        verifier,
        vec![EvidenceEvent::new(
            attempt,
            0,
            EvidenceKind::Evaluator,
            ArtifactDigest::from_bytes(b"lifecycle"),
            None,
        )],
        vec![foreign_score],
    )
    .unwrap_err();

    assert_eq!(
        error,
        FrozenAttemptError::ScoreEvaluatorMismatch { index: 0 }
    );
}

#[test]
fn frozen_attempt_rejects_a_score_outside_the_verifier_reward() {
    let attempt = AttemptId::new("wrong-reward-attempt").unwrap();
    let verifier = VerifierResult::new(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"verifier"),
        vec![ArtifactDigest::from_bytes(b"patch")],
        RewardDocument::parse(Some(br#"{"reward":0.0}"#), None).unwrap(),
        ArtifactDigest::from_bytes(b"rationale"),
    )
    .unwrap();
    let wrong_score = ScoreVersion::initial(
        attempt.clone(),
        verifier.verifier.clone(),
        verifier.evidence.clone(),
        "reward",
        1.0,
        ArtifactDigest::from_bytes(b"score-rationale"),
    )
    .unwrap();

    let error = FrozenAttemptBundle::new(
        ArtifactDigest::from_bytes(b"trial"),
        verifier,
        vec![EvidenceEvent::new(
            attempt,
            0,
            EvidenceKind::Evaluator,
            ArtifactDigest::from_bytes(b"lifecycle"),
            None,
        )],
        vec![wrong_score],
    )
    .unwrap_err();

    assert_eq!(error, FrozenAttemptError::ScoreRewardMismatch { index: 0 });
}
