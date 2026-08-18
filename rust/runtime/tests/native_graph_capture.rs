// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path};

use aiperf_runtime::eval::{
    ArtifactDigest, AttemptId, CaptureFidelity, CapturePolicy, CompatibilityFidelity, EvidenceKind,
    FrozenAttemptBundle, HarborImporter, HarborSource, NativeGraphProfile, NativeSourceAcquirer,
    RewardDocument, ScoreVersion, VerifierResult,
};

#[test]
fn externally_driven_policy_is_derived_from_the_immutable_package_and_never_claims_native_control()
{
    let (_task, imported) = externally_driven_import();
    let package = imported.package.native_graph().unwrap();

    let policy = CapturePolicy::from_package(package).unwrap();

    assert_eq!(policy.profile(), NativeGraphProfile::ExternallyDriven);
    assert_eq!(policy.fidelity_ceiling(), CaptureFidelity::ObservedProxy);
    assert_ne!(policy.fidelity_ceiling(), CaptureFidelity::NativeControlled);
    assert_ne!(
        policy.package_identity(),
        &ArtifactDigest::from_bytes(b"unbound-policy")
    );
}

#[test]
fn compatibility_observation_is_digest_only_bounded_and_degrades_for_partial_or_bypassed_calls() {
    let (_task, imported) = externally_driven_import();
    let policy = CapturePolicy::from_package(imported.package.native_graph().unwrap()).unwrap();

    let mut observed = policy.begin_observation();
    for index in 0..CapturePolicy::MAX_OBSERVATIONS {
        observed
            .record_observed_https(ArtifactDigest::from_bytes(index.to_le_bytes().as_slice()))
            .unwrap();
    }
    assert!(
        observed
            .record_observed_https(ArtifactDigest::from_bytes(b"over-capacity"))
            .is_err()
    );
    let observed = observed.freeze();
    assert_eq!(
        observed.observed_https_calls(),
        CapturePolicy::MAX_OBSERVATIONS
    );
    assert_eq!(observed.fidelity(), CaptureFidelity::ObservedProxy);

    let mut incomplete = policy.begin_observation();
    incomplete
        .record_observed_https(ArtifactDigest::from_bytes(b"redacted-http"))
        .unwrap();
    incomplete.record_partial_call().unwrap();
    assert_eq!(incomplete.freeze().fidelity(), CaptureFidelity::Partial);

    let mut bypassed = policy.begin_observation();
    bypassed.record_unobservable_or_bypassed_call().unwrap();
    assert_eq!(bypassed.freeze().fidelity(), CaptureFidelity::Missing);
}

#[test]
fn compatibility_observation_without_any_capture_is_missing() {
    let (_task, imported) = externally_driven_import();
    let policy = CapturePolicy::from_package(imported.package.native_graph().unwrap()).unwrap();

    let report = policy.begin_observation().freeze();

    assert_eq!(report.fidelity(), CaptureFidelity::Missing);
    assert_eq!(report.observed_https_calls(), 0);
    assert_eq!(report.partial_calls(), 0);
    assert_eq!(report.unobservable_or_bypassed_calls(), 0);
}

#[test]
fn compatibility_terminal_supplement_is_sealed_to_external_fidelity_and_lifecycle_evidence() {
    let (_task, imported) = externally_driven_import();
    let policy = CapturePolicy::from_package(imported.package.native_graph().unwrap()).unwrap();
    let mut observation = policy.begin_observation();
    observation
        .record_observed_https(ArtifactDigest::from_bytes(b"bounded-redacted-exchange"))
        .unwrap();

    let supplement = observation.freeze().into_terminal_supplement();

    assert_eq!(
        supplement.fidelity(),
        CompatibilityFidelity::ObservedProxy,
        "the externally driven result contract has no native/exact fidelity variant"
    );
    assert_eq!(
        supplement
            .lifecycle_evidence(AttemptId::new("attempt-1").unwrap(), 0, None)
            .kind,
        EvidenceKind::Compatibility,
        "compatibility facts remain lifecycle-only rather than verifier input"
    );
}

#[test]
fn compatibility_report_projects_only_lifecycle_evidence_and_preserves_verifier_facts() {
    let (_task, imported) = externally_driven_import();
    let policy = CapturePolicy::from_package(imported.package.native_graph().unwrap()).unwrap();
    let mut observations = policy.begin_observation();
    observations
        .record_observed_https(ArtifactDigest::from_bytes(b"redacted-http"))
        .unwrap();
    let report = observations.freeze();

    let attempt = AttemptId::new("external-attempt").unwrap();
    let lifecycle = report.lifecycle_evidence(attempt.clone(), 0, None);
    assert_eq!(lifecycle.kind, EvidenceKind::Compatibility);
    assert_eq!(lifecycle.payload, report.digest().clone());

    let verifier_input = ArtifactDigest::from_bytes(b"declared-verifier-artifact");
    let verifier = VerifierResult::new(
        attempt.clone(),
        ArtifactDigest::from_bytes(b"verifier"),
        vec![verifier_input.clone()],
        RewardDocument::parse(Some(br#"{"reward":0.75}"#), None).unwrap(),
        ArtifactDigest::from_bytes(b"verifier-rationale"),
    )
    .unwrap();
    let score = ScoreVersion::initial(
        attempt,
        verifier.verifier.clone(),
        verifier.evidence.clone(),
        "reward",
        0.75,
        ArtifactDigest::from_bytes(b"score-rationale"),
    )
    .unwrap();

    let frozen = FrozenAttemptBundle::new(
        ArtifactDigest::from_bytes(b"trial"),
        verifier,
        vec![lifecycle.clone()],
        vec![score],
    )
    .unwrap();

    assert_eq!(frozen.lifecycle_evidence(), [lifecycle]);
    assert_eq!(frozen.verifier_input_evidence(), [verifier_input]);
    assert_eq!(frozen.selected_score().unwrap().value, 0.75);
}

fn externally_driven_import() -> (tempfile::TempDir, aiperf_runtime::eval::ImportedTask) {
    let task = tempfile::tempdir().unwrap();
    write_standard_task(task.path());
    fs::write(
        task.path().join("task.toml"),
        r#"schema_version = "1.1"

[task]
name = "example/external-driver"

[native_graph]
profile = "externally_driven"
adapter_manifest = "adapters.toml"
driver = "driver-adapter"
external_driver_factory_id = "refuse"
"#,
    )
    .unwrap();
    fs::create_dir_all(task.path().join("tools")).unwrap();
    fs::write(
        task.path().join("adapters.toml"),
        r#"[[adapters]]
id = "driver-adapter"
role = "driver"
argv = ["tools/driver.sh"]
executable = "tools/driver.sh"
"#,
    )
    .unwrap();
    fs::write(task.path().join("tools/driver.sh"), b"#!/bin/sh\nexit 0\n").unwrap();

    let source = HarborSource::local(task.path().to_string_lossy()).unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&source)
        .unwrap();
    (task, imported)
}

fn write_standard_task(task_root: &Path) {
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(task_root.join("environment/Dockerfile"), b"FROM scratch\n").unwrap();
    fs::write(task_root.join("instruction.md"), b"Do work.\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), b"exit 0\n").unwrap();
}
