// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use aiperf_runtime::eval::{
    ArtifactDigest, EvalTaskRef, ProviderCapability, ProviderError, ProviderProfile, RegistryError,
    RegistryReference,
};

fn digest(seed: char) -> ArtifactDigest {
    ArtifactDigest::parse(format!("blake3:{}", seed.to_string().repeat(64))).unwrap()
}

#[test]
fn offline_manifest_refuses_duplicate_task_ids_before_execution() {
    let tasks = vec![
        EvalTaskRef::new("repair-1", digest('a')).unwrap(),
        EvalTaskRef::new("repair-1", digest('b')).unwrap(),
    ];

    assert_eq!(
        RegistryReference::local("private-suite", tasks),
        Err(RegistryError::DuplicateTaskId("repair-1".to_owned()))
    );
}

#[test]
fn offline_manifest_revalidates_public_task_selection() {
    let task = EvalTaskRef::new("repair-1", digest('a')).unwrap();
    let mut manifest = RegistryReference::local("private-suite", vec![task]).unwrap();
    manifest.tasks.clear();

    assert_eq!(
        manifest.validate_offline(),
        Err(RegistryError::EmptyTaskSelection)
    );
    assert!(!manifest.is_offline_valid());
}

#[test]
fn provider_preflight_reports_every_distinct_missing_capability() {
    let provider =
        ProviderProfile::new("local", vec![ProviderCapability::NetworkIsolation]).unwrap();

    assert_eq!(
        provider.require_all(&[
            ProviderCapability::OverlayWorkspace,
            ProviderCapability::NetworkIsolation,
            ProviderCapability::SecretIsolation,
            ProviderCapability::OverlayWorkspace,
        ]),
        Err(ProviderError::MissingCapabilities(vec![
            ProviderCapability::OverlayWorkspace,
            ProviderCapability::SecretIsolation,
        ]))
    );
}

#[test]
fn provider_profile_refuses_ambiguous_duplicate_capabilities() {
    assert_eq!(
        ProviderProfile::new(
            "local",
            vec![
                ProviderCapability::NetworkIsolation,
                ProviderCapability::NetworkIsolation,
            ],
        ),
        Err(ProviderError::DuplicateCapability(
            ProviderCapability::NetworkIsolation
        ))
    );
}
