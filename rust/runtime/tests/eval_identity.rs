// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    AgentVariantRef, ArtifactDigest, AttemptId, EvalDatasetManifest, EvalTaskRef, EvidenceEvent,
    EvidenceKind, ImportDisposition, ImportReport, ModelIdentity, PolicyIdentity, RuntimeIdentity,
    ScoreVersion, TrialBudget, TrialSpec,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

fn trial(seed: u64) -> TrialSpec {
    TrialSpec::new(
        EvalTaskRef::new("task-1", digest('a')).unwrap(),
        AgentVariantRef::new("external-agent").unwrap(),
        ModelIdentity::new("provider", "model").unwrap(),
        seed,
        PolicyIdentity::new(digest('b')),
        TrialBudget::new(1.0, 2.0).unwrap(),
        digest('c'),
        digest('d'),
        RuntimeIdentity::new("runtime-v1").unwrap(),
    )
    .unwrap()
}

#[test]
fn resolved_trial_digest_changes_with_seed() {
    assert_eq!(trial(7).identity_digest(), trial(7).identity_digest());
    assert_ne!(trial(7).identity_digest(), trial(8).identity_digest());
}

#[test]
fn import_report_rejects_unknown_disposition() {
    let report = r#"{
        "source_digest":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "normalized_digest":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "disposition":"bridge"
    }"#;
    assert!(serde_json::from_str::<ImportReport>(report).is_err());
    assert_eq!(ImportDisposition::Unsupported.as_str(), "unsupported");
}

#[test]
fn evaluation_evidence_and_scores_are_append_only_identity_records() {
    let task = EvalTaskRef::new("task-1", digest('a')).unwrap();
    let manifest = EvalDatasetManifest::new("suite-v1", "2026.08", vec![task.clone()]).unwrap();
    let attempt = AttemptId::new("attempt-1").unwrap();
    let event = EvidenceEvent::new(attempt.clone(), 0, EvidenceKind::Agent, digest('b'), None);
    let score = ScoreVersion::initial(
        attempt,
        digest('c'),
        vec![event.identity_digest()],
        "reward",
        1.0,
        digest('d'),
    )
    .unwrap();

    assert_eq!(manifest.tasks, vec![task]);
    assert_eq!(event.identity_digest(), event.identity_digest());
    assert!(score.identity_digest().as_str().starts_with("blake3:"));
    assert!(ScoreVersion::initial(
        AttemptId::new("attempt-1").unwrap(),
        digest('c'),
        vec![],
        "reward",
        f64::NAN,
        digest('d'),
    )
    .is_err());
}

#[test]
fn serde_rejects_noncanonical_identity_values() {
    assert!(serde_json::from_str::<ArtifactDigest>(r#""blake3:gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg""#).is_err());
    assert!(serde_json::from_str::<AttemptId>(r#""   ""#).is_err());
    assert!(serde_json::from_str::<EvalTaskRef>(
        r#"{"id":"","digest":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#,
    )
    .is_err());
}
