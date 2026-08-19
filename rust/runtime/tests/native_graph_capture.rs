// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{fs, path::Path};

use aiperf_runtime::eval::{
    ArtifactDigest, CaptureFidelity, CapturePolicy, HarborImporter, HarborSource,
    NativeGraphProfile, NativeSourceAcquirer,
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
