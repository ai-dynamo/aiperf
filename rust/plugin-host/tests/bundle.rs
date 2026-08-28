// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for LockedCatalogBundle publish and load (Task 16).

use aiperf_plugin_host::{
    bundle::LockedCatalogBundle,
    lock::{LockedPackageV1, PackageStatus, PluginLockV1},
};

fn sample_lock() -> PluginLockV1 {
    PluginLockV1::new(vec![LockedPackageV1 {
        id: "aiperf_export_otlp".to_string(),
        version: "0.3.0".to_string(),
        status: PackageStatus::Active,
        artifact_digest: "blake3:".to_string() + &"a".repeat(64),
        closure_digest: "blake3:".to_string() + &"b".repeat(64),
    }])
}

#[test]
fn publish_writes_and_load_and_verify_roundtrips() {
    let dir = tempdir();
    let path = dir.path().join("plugin.lock");
    let lock = sample_lock();

    LockedCatalogBundle::publish(lock.clone(), &path).expect("publish succeeds");
    assert!(path.exists(), "lock file must be written");

    let bundle = LockedCatalogBundle::load_and_verify(&path).expect("load succeeds");
    assert_eq!(bundle.lock().packages.len(), 1);
    assert_eq!(bundle.lock().packages[0].id, "aiperf_export_otlp");
    assert_eq!(bundle.lock().digest.hex, lock.digest.hex);
}

#[test]
fn load_and_verify_rejects_corrupted_digest() {
    let dir = tempdir();
    let path = dir.path().join("plugin.lock");
    let mut lock = sample_lock();
    LockedCatalogBundle::publish(lock.clone(), &path).expect("publish succeeds");

    // Tamper: flip the digest hex after writing.
    lock.digest.hex = "0".repeat(64);
    let json = serde_json::to_vec(&lock).expect("serializes");
    std::fs::write(&path, &json).expect("write tampered file");

    let err =
        LockedCatalogBundle::load_and_verify(&path).expect_err("tampered file must be rejected");
    assert!(
        matches!(
            err,
            aiperf_plugin_host::bundle::BundleError::DigestMismatch { .. }
        ),
        "wrong error variant: {err:?}"
    );
}

#[test]
fn load_and_verify_rejects_missing_file() {
    let dir = tempdir();
    let path = dir.path().join("missing.lock");
    let err = LockedCatalogBundle::load_and_verify(&path).expect_err("missing file must error");
    assert!(
        matches!(err, aiperf_plugin_host::bundle::BundleError::Io(_)),
        "wrong error variant: {err:?}"
    );
}

// Minimal temp-dir helper that cleans up on drop.
fn tempdir() -> TempDir {
    let path = std::env::temp_dir().join(format!(
        "aiperf-plugin-host-test-{}",
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
    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}
