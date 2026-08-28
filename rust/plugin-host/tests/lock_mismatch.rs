// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for bundle tamper detection — digest mismatch and schema rejection (Task 16).

use aiperf_plugin_host::{
    bundle::{BundleError, LockedCatalogBundle},
    lock::{LockedPackageV1, PackageStatus, PluginLockV1},
};

fn make_lock(id: &str) -> PluginLockV1 {
    PluginLockV1::new(vec![LockedPackageV1 {
        id: id.to_string(),
        version: "1.0.0".to_string(),
        status: PackageStatus::Active,
        artifact_digest: "blake3:".to_string() + &"a".repeat(64),
        closure_digest: "blake3:".to_string() + &"b".repeat(64),
    }])
}

fn tempdir() -> TempDir {
    let path = std::env::temp_dir().join(format!(
        "aiperf-lock-mismatch-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&path).expect("create temp dir");
    TempDir(path)
}

struct TempDir(std::path::PathBuf);
impl TempDir {
    fn path(&self) -> &std::path::Path { &self.0 }
}
impl Drop for TempDir {
    fn drop(&mut self) { let _ = std::fs::remove_dir_all(&self.0); }
}

#[test]
fn package_list_mutation_is_detected_by_digest_check() {
    let dir = tempdir();
    let path = dir.path().join("plugin.lock");
    let lock = make_lock("aiperf_export_otlp");
    LockedCatalogBundle::publish(lock, &path).expect("publish");

    // Mutate: change a package id on disk without updating the digest.
    let raw = std::fs::read(&path).expect("read");
    let mut on_disk: serde_json::Value = serde_json::from_slice(&raw).expect("parse");
    on_disk["packages"][0]["id"] = serde_json::Value::String("injected_package".into());
    std::fs::write(&path, serde_json::to_vec(&on_disk).expect("serialize")).expect("write");

    let err = LockedCatalogBundle::load_and_verify(&path)
        .expect_err("mutated package list must be rejected");
    assert!(
        matches!(err, BundleError::DigestMismatch { .. }),
        "expected DigestMismatch, got: {err:?}"
    );
}

#[test]
fn invalid_json_is_rejected_with_parse_error() {
    let dir = tempdir();
    let path = dir.path().join("bad.lock");
    std::fs::write(&path, b"not json at all").expect("write");

    let err = LockedCatalogBundle::load_and_verify(&path).expect_err("invalid json rejected");
    assert!(
        matches!(err, BundleError::Parse(_)),
        "expected Parse error, got: {err:?}"
    );
}

#[test]
fn wrong_schema_version_is_rejected() {
    let dir = tempdir();
    let path = dir.path().join("plugin.lock");
    let json = r#"{"schema_version":"0.0","packages":[],"digest":{"algorithm":"blake3","hex":"0000000000000000000000000000000000000000000000000000000000000000"}}"#;
    std::fs::write(&path, json).expect("write");

    let err = LockedCatalogBundle::load_and_verify(&path)
        .expect_err("wrong schema version rejected");
    assert!(
        matches!(err, BundleError::UnsupportedSchemaVersion(_)),
        "expected UnsupportedSchemaVersion, got: {err:?}"
    );
}
