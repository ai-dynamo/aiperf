// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for PluginLockV1 construction and serialization (Task 16).

use aiperf_plugin_host::lock::{LockedPackageV1, PackageStatus, PluginLockDigest, PluginLockV1};

fn minimal_lock() -> PluginLockV1 {
    PluginLockV1::new(vec![LockedPackageV1 {
        id: "aiperf_export_otlp".to_string(),
        version: "0.3.0".to_string(),
        status: PackageStatus::Active,
        artifact_digest: "blake3:".to_string() + &"a".repeat(64),
        closure_digest: "blake3:".to_string() + &"b".repeat(64),
    }])
}

#[test]
fn plugin_lock_v1_roundtrips_through_json() {
    let lock = minimal_lock();
    let json = serde_json::to_string(&lock).expect("lock serializes");
    let decoded: PluginLockV1 = serde_json::from_str(&json).expect("lock deserializes");
    assert_eq!(decoded.schema_version, "1.0");
    assert_eq!(decoded.packages.len(), 1);
    assert_eq!(decoded.packages[0].id, "aiperf_export_otlp");
    assert_eq!(decoded.packages[0].status, PackageStatus::Active);
}

#[test]
fn plugin_lock_digest_is_stable_for_same_content() {
    let lock1 = minimal_lock();
    let lock2 = minimal_lock();
    assert_eq!(lock1.digest.hex, lock2.digest.hex);
    assert_eq!(lock1.digest.algorithm, "blake3");
}

#[test]
fn plugin_lock_digest_differs_when_packages_differ() {
    let lock1 = minimal_lock();
    let lock2 = PluginLockV1::new(vec![LockedPackageV1 {
        id: "aiperf_transport_h2c".to_string(),
        version: "1.0.0".to_string(),
        status: PackageStatus::Active,
        artifact_digest: "blake3:".to_string() + &"c".repeat(64),
        closure_digest: "blake3:".to_string() + &"d".repeat(64),
    }]);
    assert_ne!(lock1.digest.hex, lock2.digest.hex);
}

#[test]
fn disabled_package_roundtrips_through_json() {
    let lock = PluginLockV1::new(vec![LockedPackageV1 {
        id: "aiperf_export_otlp".to_string(),
        version: "0.3.0".to_string(),
        status: PackageStatus::Disabled,
        artifact_digest: "blake3:".to_string() + &"e".repeat(64),
        closure_digest: "blake3:".to_string() + &"f".repeat(64),
    }]);
    let json = serde_json::to_string(&lock).expect("serializes");
    let decoded: PluginLockV1 = serde_json::from_str(&json).expect("deserializes");
    assert_eq!(decoded.packages[0].status, PackageStatus::Disabled);
}

#[test]
fn lock_rejects_unknown_top_level_field() {
    let lock = minimal_lock();
    let mut value: serde_json::Value = serde_json::to_value(&lock).expect("serializes");
    value["surprise"] = serde_json::Value::Bool(true);
    let err = serde_json::from_value::<PluginLockV1>(value)
        .expect_err("unknown top-level field must be rejected");
    assert!(
        err.to_string().contains("surprise"),
        "error must name the unknown field: {err}"
    );
}

#[test]
fn locked_package_rejects_unknown_field() {
    let lock = minimal_lock();
    let mut value: serde_json::Value = serde_json::to_value(&lock).expect("serializes");
    value["packages"][0]["surprise"] = serde_json::Value::Bool(true);
    let err = serde_json::from_value::<PluginLockV1>(value)
        .expect_err("unknown package field must be rejected");
    assert!(
        err.to_string().contains("surprise"),
        "error must name the unknown field: {err}"
    );
}

#[test]
fn lock_digest_field_matches_recomputed() {
    let lock = minimal_lock();
    let recomputed = PluginLockDigest::compute(&lock.packages);
    assert_eq!(lock.digest.hex, recomputed.hex);
    assert_eq!(lock.digest.algorithm, recomputed.algorithm);
}
