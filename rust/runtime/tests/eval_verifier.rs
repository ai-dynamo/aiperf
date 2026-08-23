// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "engine")]

use aiperf_runtime::eval::{
    ArtifactDigest, AttemptId, DeclaredArtifactTransfer, EvidenceKind, RewardDocument,
    ScoreVersion, VerifierMode, VerifierSandboxFactory, invalid_reward_evidence,
    parse_reward_with_evidence, prepare_verifier,
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
fn malformed_reward_becomes_evaluator_evidence() {
    let error = RewardDocument::parse(Some(b"not-json"), None).unwrap_err();

    let event = invalid_reward_evidence(AttemptId::new("attempt-1").unwrap(), 3, &error);

    assert_eq!(event.kind, EvidenceKind::Evaluator);
    assert_eq!(event.sequence, 3);
    assert_eq!(event.attempt.as_str(), "attempt-1");
}

#[test]
fn reward_parser_returns_evaluator_evidence_for_a_malformed_document() {
    let outcome = parse_reward_with_evidence(
        AttemptId::new("attempt-1").unwrap(),
        3,
        Some(b"not-json"),
        None,
    );

    assert!(outcome.reward.is_err());
    assert_eq!(outcome.evidence.unwrap().kind, EvidenceKind::Evaluator);
}

#[test]
fn declared_artifact_transfer_refuses_workspace_paths_and_duplicates() {
    assert!(DeclaredArtifactTransfer::new(vec![("relative/path", digest('a'))]).is_err());
    assert!(DeclaredArtifactTransfer::new(vec![("/../../agent-secret", digest('a'))]).is_err());
    assert!(
        DeclaredArtifactTransfer::new(vec![
            ("/results/patch.diff", digest('a')),
            ("/results/patch.diff", digest('b')),
        ])
        .is_err()
    );
}

#[test]
fn separate_verifier_materializes_only_declared_artifacts_in_fresh_sandbox() {
    let transfer =
        DeclaredArtifactTransfer::new(vec![("/results/patch.diff", digest('a'))]).unwrap();
    let sandbox = RecordingSandbox::default();

    prepare_verifier(&sandbox, VerifierMode::Separate, &transfer).unwrap();

    assert_eq!(sandbox.mode(), Some(VerifierMode::Separate));
    assert_eq!(
        sandbox.artifacts(),
        vec![("/results/patch.diff".to_owned(), digest('a'))]
    );
}

#[test]
fn initial_score_pins_metric_rationale_and_evidence() {
    let attempt = AttemptId::new("attempt-1").unwrap();
    let score = ScoreVersion::initial(
        attempt.clone(),
        digest('a'),
        vec![digest('b')],
        "reward",
        0.0,
        digest('c'),
    )
    .unwrap();

    assert_eq!(score.attempt, attempt);
    assert_eq!(score.metric, "reward");
    assert_eq!(score.rationale, digest('c'));
}

#[derive(Default)]
struct RecordingSandbox {
    prepared: std::cell::RefCell<Option<(VerifierMode, Vec<(String, ArtifactDigest)>)>>,
}

impl RecordingSandbox {
    fn mode(&self) -> Option<VerifierMode> {
        self.prepared.borrow().as_ref().map(|(mode, _)| *mode)
    }

    fn artifacts(&self) -> Vec<(String, ArtifactDigest)> {
        self.prepared
            .borrow()
            .as_ref()
            .map(|(_, artifacts)| artifacts.clone())
            .unwrap_or_default()
    }
}

impl VerifierSandboxFactory for RecordingSandbox {
    fn prepare(
        &self,
        mode: VerifierMode,
        artifacts: &[(String, ArtifactDigest)],
    ) -> Result<(), aiperf_runtime::eval::VerifierExecutionError> {
        self.prepared.replace(Some((mode, artifacts.to_vec())));
        Ok(())
    }
}
