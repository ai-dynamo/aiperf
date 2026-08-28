// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for diff_locks — added, removed, and changed packages (Task 16).

use aiperf_plugin_host::{
    diff::{LockDiff, diff_locks},
    lock::{LockedPackageV1, PackageStatus, PluginLockV1},
};

fn pkg(id: &str, version: &str, status: PackageStatus) -> LockedPackageV1 {
    LockedPackageV1 {
        id: id.to_string(),
        version: version.to_string(),
        status,
        artifact_digest: "blake3:".to_string() + &"a".repeat(64),
        closure_digest: "blake3:".to_string() + &"b".repeat(64),
    }
}

fn lock(packages: Vec<LockedPackageV1>) -> PluginLockV1 {
    PluginLockV1::new(packages)
}

#[test]
fn diff_reports_added_package() {
    let old = lock(vec![]);
    let new = lock(vec![pkg("aiperf_export_otlp", "0.3.0", PackageStatus::Active)]);
    let diff = diff_locks(&old, &new);
    assert_eq!(diff.added.len(), 1);
    assert_eq!(diff.added[0].id, "aiperf_export_otlp");
    assert!(diff.removed.is_empty());
    assert!(diff.changed.is_empty());
}

#[test]
fn diff_reports_removed_package() {
    let old = lock(vec![pkg("aiperf_export_otlp", "0.3.0", PackageStatus::Active)]);
    let new = lock(vec![]);
    let diff = diff_locks(&old, &new);
    assert_eq!(diff.removed.len(), 1);
    assert_eq!(diff.removed[0].id, "aiperf_export_otlp");
    assert!(diff.added.is_empty());
    assert!(diff.changed.is_empty());
}

#[test]
fn diff_reports_version_change() {
    let old = lock(vec![pkg("aiperf_export_otlp", "0.3.0", PackageStatus::Active)]);
    let new = lock(vec![pkg("aiperf_export_otlp", "0.4.0", PackageStatus::Active)]);
    let diff = diff_locks(&old, &new);
    assert_eq!(diff.changed.len(), 1);
    assert_eq!(diff.changed[0].0.version, "0.3.0");
    assert_eq!(diff.changed[0].1.version, "0.4.0");
    assert!(diff.added.is_empty());
    assert!(diff.removed.is_empty());
}

#[test]
fn diff_on_identical_locks_is_empty() {
    let pkgs = vec![
        pkg("aiperf_export_otlp", "0.3.0", PackageStatus::Active),
        pkg("aiperf_transport_h2c", "1.0.0", PackageStatus::Active),
    ];
    let old = lock(pkgs.clone());
    let new = lock(pkgs);
    let diff = diff_locks(&old, &new);
    assert!(diff.is_empty(), "identical locks must produce empty diff");
}

#[test]
fn diff_reports_status_change_as_changed() {
    let old = lock(vec![pkg("aiperf_export_otlp", "0.3.0", PackageStatus::Active)]);
    let new = lock(vec![pkg("aiperf_export_otlp", "0.3.0", PackageStatus::Disabled)]);
    let diff = diff_locks(&old, &new);
    assert_eq!(diff.changed.len(), 1);
    assert_eq!(diff.changed[0].0.status, PackageStatus::Active);
    assert_eq!(diff.changed[0].1.status, PackageStatus::Disabled);
}
