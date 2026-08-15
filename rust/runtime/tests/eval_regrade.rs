// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    regrade, ArtifactDigest, AttemptId, RegradeError, RegradeRequest, RewardDocument, ScoreVersion,
    VerifierResult,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

fn reward() -> RewardDocument {
    RewardDocument::parse(Some(br#"{"accuracy":1.0,"cost":2.5}"#), None).unwrap()
}

fn previous(attempt: AttemptId) -> ScoreVersion {
    ScoreVersion::initial(
        attempt,
        digest('a'),
        vec![digest('b')],
        "accuracy",
        0.0,
        digest('c'),
    )
    .unwrap()
}

#[test]
fn regrade_appends_a_rationale_bearing_score_without_changing_original() {
    let attempt = AttemptId::new("attempt-1").unwrap();
    let original = previous(attempt.clone());
    let original_snapshot = original.clone();
    let result = VerifierResult::new(
        attempt.clone(),
        digest('d'),
        vec![digest('e')],
        reward(),
        digest('f'),
    )
    .unwrap();

    let regraded = regrade(RegradeRequest::new(original.clone(), result, "cost").unwrap()).unwrap();

    assert_eq!(original, original_snapshot);
    assert_eq!(regraded.attempt, attempt);
    assert_eq!(regraded.version, 1);
    assert_eq!(regraded.metric, "cost");
    assert_eq!(regraded.value, 2.5);
    assert_eq!(regraded.evaluator, digest('d'));
    assert_eq!(regraded.evidence, vec![digest('e')]);
    assert_eq!(regraded.rationale, digest('f'));
    assert_eq!(regraded.predecessor, Some(original.identity_digest()));
    assert_ne!(original.identity_digest(), regraded.identity_digest());
}

#[test]
fn regrade_refuses_an_unknown_metric_without_falling_back() {
    let attempt = AttemptId::new("attempt-1").unwrap();
    let result = VerifierResult::new(
        attempt.clone(),
        digest('d'),
        vec![digest('e')],
        reward(),
        digest('f'),
    )
    .unwrap();

    let error = regrade(RegradeRequest::new(previous(attempt), result, "reward").unwrap()).unwrap_err();

    assert_eq!(error, RegradeError::MetricNotFound("reward".to_owned()));
}

#[test]
fn regrade_refuses_mismatched_attempt_lineage() {
    let result = VerifierResult::new(
        AttemptId::new("attempt-2").unwrap(),
        digest('d'),
        vec![digest('e')],
        reward(),
        digest('f'),
    )
    .unwrap();

    let error = regrade(
        RegradeRequest::new(previous(AttemptId::new("attempt-1").unwrap()), result, "accuracy").unwrap(),
    )
    .unwrap_err();

    assert_eq!(error, RegradeError::AttemptMismatch);
}

#[test]
fn regrade_refuses_a_version_that_cannot_append() {
    let attempt = AttemptId::new("attempt-1").unwrap();
    let original = ScoreVersion::new(
        attempt.clone(),
        u32::MAX,
        digest('a'),
        vec![digest('b')],
        "accuracy",
        0.0,
        digest('c'),
        None,
    )
    .unwrap();
    let result = VerifierResult::new(
        attempt,
        digest('d'),
        vec![digest('e')],
        reward(),
        digest('f'),
    )
    .unwrap();

    let error = regrade(RegradeRequest::new(original, result, "accuracy").unwrap()).unwrap_err();

    assert_eq!(error, RegradeError::VersionOverflow);
}
