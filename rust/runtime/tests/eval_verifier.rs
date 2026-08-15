// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, AttemptId, DeclaredArtifactTransfer, RewardDocument, ScoreVersion,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

#[test]
fn reward_json_precedes_reward_txt_and_preserves_multiple_metrics() {
    let reward = RewardDocument::parse(
        Some(br#"{"accuracy":1.0,"cost":2.5}"#),
        Some("0.25".as_bytes()),
    )
    .unwrap();

    assert_eq!(reward.metrics.get("accuracy"), Some(&1.0));
    assert_eq!(reward.metrics.get("cost"), Some(&2.5));
}

#[test]
fn declared_artifact_transfer_excludes_undeclared_workspace() {
    let transfer = DeclaredArtifactTransfer::new(vec![("/results/patch.diff", digest('a'))]).unwrap();

    assert_eq!(transfer.artifacts().len(), 1);
    assert!(DeclaredArtifactTransfer::new(vec![("relative/path", digest('b'))]).is_err());
}

#[test]
fn regrade_appends_score_without_changing_original_attempt() {
    let attempt = AttemptId::new("attempt-1").unwrap();
    let original = ScoreVersion::new(attempt.clone(), 0, digest('a'), vec![digest('b')], 0.0).unwrap();
    let regrade = ScoreVersion::new(attempt.clone(), 1, digest('c'), vec![digest('b')], 1.0).unwrap();

    assert_eq!(original.attempt, attempt);
    assert_ne!(original.identity_digest(), regrade.identity_digest());
}
