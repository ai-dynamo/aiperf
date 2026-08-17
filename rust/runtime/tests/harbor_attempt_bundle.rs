// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    AgentVariantRef, ArtifactDigest, AttemptId, DeclaredArtifactManifest, EvidenceKind,
    HarborEvaluationCoordinator, HarborImporter, HarborLifecycleAgentContract,
    HarborLifecycleRequest, HarborLifecycleScoreRequest, HarborSource, LocalExecutionResult,
    MaterializedArtifactManifest, ModelIdentity, PolicyIdentity, RewardDocument, RuntimeIdentity,
    SourceAcquirer, TrialBudget,
};

struct StaticAcquirer {
    bytes: Vec<u8>,
}

impl SourceAcquirer for StaticAcquirer {
    fn acquire(
        &self,
        _: &HarborSource,
    ) -> Result<Vec<u8>, aiperf_runtime::eval::HarborImportError> {
        Ok(self.bytes.clone())
    }
}

fn completed_harbor_attempt(reward: f64) -> aiperf_runtime::eval::HarborCompletedEvaluation {
    let acquirer = StaticAcquirer {
        bytes: br#"{"id":"frozen-harbor-attempt","instruction":"Repair","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["true"],"verifier_command":["true"],"declared_artifacts":["/results/patch.diff"]}"#.to_vec(),
    };
    let imported = HarborImporter::new(&acquirer)
        .import(&HarborSource::local("task.json").unwrap())
        .unwrap();
    let request = HarborLifecycleRequest {
        version: 1,
        agent_variant: AgentVariantRef::new("native-agent").unwrap(),
        model: ModelIdentity::new("native", "unit-test").unwrap(),
        seed: 7,
        policy: PolicyIdentity::new(ArtifactDigest::from_bytes(b"frozen-policy")),
        runtime: RuntimeIdentity::new("native-unit").unwrap(),
        attempt: AttemptId::new("frozen-attempt").unwrap(),
        budget: TrialBudget::new(30.0, 30.0).unwrap(),
        agent_contract: HarborLifecycleAgentContract::External,
        command: vec!["true".to_owned()],
        initial_score: HarborLifecycleScoreRequest {
            metric: "reward".to_owned(),
            rationale: ArtifactDigest::from_bytes(b"initial-rationale"),
        },
        regrade: HarborLifecycleScoreRequest {
            metric: "reward".to_owned(),
            rationale: ArtifactDigest::from_bytes(b"regrade-rationale"),
        },
    };
    let trial = HarborEvaluationCoordinator::resolve_trial(&imported, &request).unwrap();
    let execution = LocalExecutionResult {
        artifacts: vec![(
            "/results/patch.diff".to_owned(),
            ArtifactDigest::from_bytes(b"patch"),
        )],
        reward: RewardDocument::parse(Some(format!("{{\"reward\":{reward}}}").as_bytes()), None)
            .unwrap(),
        verifier: ArtifactDigest::from_bytes(b"verifier"),
    };
    HarborEvaluationCoordinator::complete_attempt(
        imported,
        trial,
        &request.command,
        execution,
        &request,
    )
    .unwrap()
}

#[test]
fn manifests_sort_paths_and_change_only_for_identity_bearing_content() {
    let declared =
        DeclaredArtifactManifest::new(["/results/z.txt".to_owned(), "/results/a.txt".to_owned()])
            .unwrap();
    let reordered =
        DeclaredArtifactManifest::new(["/results/a.txt".to_owned(), "/results/z.txt".to_owned()])
            .unwrap();
    assert_eq!(declared, reordered);

    let materialized = MaterializedArtifactManifest::new([
        (
            "/results/a.txt".to_owned(),
            ArtifactDigest::from_bytes(b"a"),
        ),
        (
            "/results/z.txt".to_owned(),
            ArtifactDigest::from_bytes(b"z"),
        ),
    ])
    .unwrap();
    assert_ne!(declared.digest, materialized.digest);
}

#[test]
fn manifests_canonicalize_import_aliases_and_reject_invalid_or_tampered_values() {
    let aliases =
        DeclaredArtifactManifest::new(["//results//a/".to_owned(), "/results/z.txt".to_owned()])
            .unwrap();
    let canonical =
        DeclaredArtifactManifest::new(["/results/a".to_owned(), "/results/z.txt".to_owned()])
            .unwrap();
    assert_eq!(aliases, canonical);
    assert!(
        DeclaredArtifactManifest::new(["/results/a".to_owned(), "//results/a/".to_owned(),])
            .is_err()
    );
    assert!(DeclaredArtifactManifest::new(["relative".to_owned()]).is_err());
    assert!(serde_json::from_str::<DeclaredArtifactManifest>(
        r#"{"paths":["/results/a"],"digest":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}"#,
    )
    .is_err());
}

#[test]
fn declared_manifest_identity_uses_versioned_length_delimited_bytes() {
    let manifest = DeclaredArtifactManifest::new(["/results/a".to_owned()]).unwrap();
    let expected = ArtifactDigest::from_bytes(
        b"harbor-declared-artifacts-v1\x1f\0\0\0\0\0\0\0\x01\x1e\0\0\0\0\0\0\0\x0a/results/a",
    );

    assert_eq!(manifest.digest, expected);
}

#[test]
fn materialized_manifest_sorts_paths_and_binds_content_digests() {
    let first = MaterializedArtifactManifest::new([
        (
            "/results/z.txt".to_owned(),
            ArtifactDigest::from_bytes(b"z"),
        ),
        (
            "/results/a.txt".to_owned(),
            ArtifactDigest::from_bytes(b"a"),
        ),
    ])
    .unwrap();
    let reordered = MaterializedArtifactManifest::new([
        (
            "/results/a.txt".to_owned(),
            ArtifactDigest::from_bytes(b"a"),
        ),
        (
            "/results/z.txt".to_owned(),
            ArtifactDigest::from_bytes(b"z"),
        ),
    ])
    .unwrap();
    let changed_content = MaterializedArtifactManifest::new([
        (
            "/results/a.txt".to_owned(),
            ArtifactDigest::from_bytes(b"changed"),
        ),
        (
            "/results/z.txt".to_owned(),
            ArtifactDigest::from_bytes(b"z"),
        ),
    ])
    .unwrap();

    assert_eq!(first, reordered);
    assert_ne!(first.digest, changed_content.digest);
}

#[test]
fn manifests_reject_serde_tampering_and_unknown_fields() {
    let declared = DeclaredArtifactManifest::new(["/results/a".to_owned()]).unwrap();
    let materialized = MaterializedArtifactManifest::new([(
        "/results/a".to_owned(),
        ArtifactDigest::from_bytes(b"a"),
    )])
    .unwrap();

    let declared_json = serde_json::to_value(&declared).unwrap();
    let materialized_json = serde_json::to_value(&materialized).unwrap();
    assert_eq!(
        serde_json::from_value::<DeclaredArtifactManifest>(declared_json.clone()).unwrap(),
        declared
    );
    assert_eq!(
        serde_json::from_value::<MaterializedArtifactManifest>(materialized_json.clone()).unwrap(),
        materialized
    );

    let mut declared_unknown = declared_json.clone();
    declared_unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(true));
    assert!(serde_json::from_value::<DeclaredArtifactManifest>(declared_unknown).is_err());
    let mut materialized_unknown = materialized_json.clone();
    materialized_unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".to_owned(), serde_json::json!(true));
    assert!(serde_json::from_value::<MaterializedArtifactManifest>(materialized_unknown).is_err());

    let mut declared_tampered = declared_json;
    declared_tampered.as_object_mut().unwrap().insert(
        "digest".to_owned(),
        serde_json::json!(ArtifactDigest::from_bytes(b"tampered")),
    );
    assert!(serde_json::from_value::<DeclaredArtifactManifest>(declared_tampered).is_err());
    let mut materialized_tampered = materialized_json;
    materialized_tampered.as_object_mut().unwrap().insert(
        "digest".to_owned(),
        serde_json::json!(ArtifactDigest::from_bytes(b"tampered")),
    );
    assert!(serde_json::from_value::<MaterializedArtifactManifest>(materialized_tampered).is_err());
}

#[test]
fn empty_manifest_kinds_have_distinct_identities() {
    let declared = DeclaredArtifactManifest::new([]).unwrap();
    let materialized = MaterializedArtifactManifest::new([]).unwrap();

    assert_ne!(declared.digest, materialized.digest);
}

#[test]
fn frozen_existing_harbor_attempt_preserves_verifier_input_evidence() {
    let completed = completed_harbor_attempt(0.75);
    let bundle = completed.freeze().unwrap();

    assert_eq!(
        bundle.verifier_input_evidence(),
        completed.verifier_result.evidence
    );
    assert_eq!(bundle.lifecycle_evidence(), completed.evidence);
    assert_eq!(bundle.lifecycle_evidence()[0].kind, EvidenceKind::Sandbox);
    assert_ne!(
        bundle.lifecycle_evidence_digest(),
        bundle.verifier_input_evidence()[0],
    );
}

#[test]
fn frozen_existing_harbor_attempt_retains_append_only_score_lineage() {
    let completed = completed_harbor_attempt(0.75);
    let initial_score = completed.initial_score.clone();
    let verifier_input_evidence = completed.verifier_result.evidence.clone();
    let bundle = completed.freeze().unwrap();

    assert_eq!(bundle.score_lineage()[0], initial_score);
    assert_eq!(bundle.score_lineage()[1].version, 1);
    assert_eq!(
        bundle.score_lineage()[1].predecessor,
        Some(bundle.score_lineage()[0].identity_digest()),
    );
    assert_eq!(bundle.score_lineage()[1].evidence, verifier_input_evidence);
}
