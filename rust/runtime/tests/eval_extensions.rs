// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, AttemptId, EvalTaskRef, ProviderCapability, ProviderProfile, RegistryReference,
    TrajectoryExportManifest,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

#[test]
fn provider_without_required_capability_is_refused_before_trial_start() {
    let provider =
        ProviderProfile::new("local", vec![ProviderCapability::NetworkIsolation]).unwrap();

    assert!(
        provider
            .require(&[ProviderCapability::OverlayWorkspace])
            .is_err()
    );
}

#[test]
fn local_manifest_remains_valid_when_registry_is_offline() {
    let task = EvalTaskRef::new("task-1", digest('a')).unwrap();
    let registry = RegistryReference::local("suite-v1", vec![task]).unwrap();

    assert!(registry.is_offline_valid());
}

#[test]
fn trajectory_export_references_immutable_attempt_evidence() {
    let manifest =
        TrajectoryExportManifest::new(AttemptId::new("attempt-1").unwrap(), vec![digest('b')])
            .unwrap();

    assert_eq!(manifest.attempt.as_str(), "attempt-1");
    assert_eq!(manifest.evidence, vec![digest('b')]);
}
