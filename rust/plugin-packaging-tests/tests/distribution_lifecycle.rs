// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Distribution lifecycle tests: an authenticated first-party inventory is
//! accepted only when it is internally complete and key-bound, and the install
//! lifecycle it drives survives tamper, rollback, and generation mixing.

use std::collections::BTreeMap;

use aiperf_plugin_host::error::{InstallError, InventoryError};
use aiperf_plugin_host::install::{InstallFile, InstallRoot};
use aiperf_plugin_host::inventory::{
    AuthenticatedInventory, DistributionEntry, validate_inventory,
};

const KEY: &str = "nvidia-aiperf-first-party-2026";
const DIGEST_A: &str = "aa00000000000000000000000000000000000000000000000000000000000000";
const DIGEST_B: &str = "bb00000000000000000000000000000000000000000000000000000000000000";

fn entry(id: &str, digest: &str) -> DistributionEntry {
    let mut artifact_digests = BTreeMap::new();
    artifact_digests.insert("x86_64-unknown-linux-gnu".to_string(), digest.to_string());
    DistributionEntry {
        package_id: id.to_string(),
        version: "1.0.0".to_string(),
        manifest_digest: digest.to_string(),
        artifact_digests,
        depends_on: Vec::new(),
        signing_key_id: KEY.to_string(),
    }
}

fn valid_inventory() -> AuthenticatedInventory {
    AuthenticatedInventory {
        universe_id: "aiperf-first-party".to_string(),
        build_id: "2026.08.27+1".to_string(),
        authentication_root: DIGEST_A.to_string(),
        required_packages: vec!["aiperf.exporter.parquet".to_string()],
        required_keys: vec![KEY.to_string()],
        entries: vec![
            entry("aiperf.exporter.parquet", DIGEST_A),
            entry("aiperf.transport.grpc", DIGEST_B),
        ],
    }
}

#[test]
fn a_complete_first_party_inventory_validates() {
    validate_inventory(&valid_inventory()).expect("a complete inventory validates");
}

#[test]
fn an_inventory_never_carries_an_absolute_path() {
    let inv = valid_inventory();
    assert!(!inv.contains_absolute_path());

    let mut tampered = valid_inventory();
    tampered.entries[0].package_id = "/opt/aiperf/plugins/parquet".to_string();
    assert!(tampered.contains_absolute_path());
    let err = validate_inventory(&tampered).expect_err("absolute paths must be refused");
    assert!(
        matches!(err, InventoryError::AbsolutePath(_)),
        "expected AbsolutePath, got {err:?}"
    );
}

#[test]
fn a_tampered_digest_is_refused() {
    let mut inv = valid_inventory();
    inv.entries[1].manifest_digest = "not-a-digest".to_string();
    let err = validate_inventory(&inv).expect_err("a malformed digest must be refused");
    assert!(
        matches!(err, InventoryError::MalformedDigest { .. }),
        "expected MalformedDigest, got {err:?}"
    );
}

#[test]
fn an_entry_signed_by_an_unlisted_key_is_refused() {
    let mut inv = valid_inventory();
    inv.entries[1].signing_key_id = "attacker-key".to_string();
    let err = validate_inventory(&inv).expect_err("an unlisted signing key must be refused");
    assert!(
        matches!(err, InventoryError::UntrustedSigningKey { .. }),
        "expected UntrustedSigningKey, got {err:?}"
    );
}

#[test]
fn a_missing_required_package_is_refused() {
    let mut inv = valid_inventory();
    inv.required_packages
        .push("aiperf.dataset.absent".to_string());
    let err = validate_inventory(&inv).expect_err("a missing required package must be refused");
    assert!(
        matches!(err, InventoryError::MissingRequiredPackage(_)),
        "expected MissingRequiredPackage, got {err:?}"
    );
}

#[test]
fn an_incomplete_dependency_closure_is_refused() {
    let mut inv = valid_inventory();
    inv.entries[0].depends_on.push("aiperf.core.absent".to_string());
    let err = validate_inventory(&inv).expect_err("a dangling dependency must be refused");
    assert!(
        matches!(err, InventoryError::IncompleteClosure { .. }),
        "expected IncompleteClosure, got {err:?}"
    );
}

#[test]
fn a_duplicate_package_entry_is_refused() {
    let mut inv = valid_inventory();
    inv.entries.push(entry("aiperf.exporter.parquet", DIGEST_B));
    let err = validate_inventory(&inv).expect_err("a duplicate package entry must be refused");
    assert!(
        matches!(err, InventoryError::DuplicatePackage(_)),
        "expected DuplicatePackage, got {err:?}"
    );
}

#[test]
fn the_canonical_digest_changes_with_any_field() {
    let base = valid_inventory().canonical_digest();
    let mut changed = valid_inventory();
    changed.build_id = "2026.08.27+2".to_string();
    assert_ne!(base, changed.canonical_digest());
    assert_eq!(base, valid_inventory().canonical_digest(), "digest is stable");
}

#[test]
fn a_rolled_back_generation_reports_the_inventory_it_was_installed_from() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");

    let first = valid_inventory();
    let mut second = valid_inventory();
    second.build_id = "2026.08.28+1".to_string();

    let g1 = root
        .atomic_install(&first, &[InstallFile::new("lib.so", b"one".to_vec())])
        .expect("install one");
    let g2 = root
        .atomic_install(&second, &[InstallFile::new("lib.so", b"two".to_vec())])
        .expect("install two");

    root.verify_generation(g2.id, &second).expect("g2 verifies");
    let restored = root.rollback().expect("rollback");
    assert_eq!(restored.id, g1.id);
    root.verify_generation(restored.id, &first)
        .expect("the restored generation still carries its own inventory digest");

    // Generations may not be mixed: the restored generation must not verify
    // against the inventory of the generation it replaced.
    let err = root
        .verify_generation(restored.id, &second)
        .expect_err("mixing generations must be refused");
    assert!(
        matches!(err, InstallError::InventoryDigestMismatch { .. }),
        "expected InventoryDigestMismatch, got {err:?}"
    );
}

#[test]
fn an_unvalidated_inventory_cannot_be_installed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = InstallRoot::open(dir.path()).expect("open install root");
    let mut inv = valid_inventory();
    inv.entries[0].signing_key_id = "attacker-key".to_string();

    let err = root
        .atomic_install(&inv, &[InstallFile::new("lib.so", b"one".to_vec())])
        .expect_err("installation must validate the inventory first");
    assert!(
        matches!(err, InstallError::Inventory(_)),
        "expected Inventory, got {err:?}"
    );
    assert!(root.current().expect("current").is_none());
}
