// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Residency ledger tests (Task 14).

use std::path::PathBuf;

use aiperf_plugin_host::residency::ResidencyLedger;

#[test]
fn empty_ledger_is_empty() {
    let ledger = ResidencyLedger::new();
    assert!(ledger.is_empty());
    assert_eq!(ledger.len(), 0);
}

#[test]
fn register_new_entry_succeeds() {
    let mut ledger = ResidencyLedger::new();
    ledger
        .register("abc123".to_owned(), PathBuf::from("/staged/abc123.so"))
        .expect("new entry");
    assert_eq!(ledger.len(), 1);
}

#[test]
fn register_same_digest_same_path_is_idempotent() {
    let mut ledger = ResidencyLedger::new();
    let digest = "abc123".to_owned();
    let path = PathBuf::from("/staged/abc123.so");
    ledger.register(digest.clone(), path.clone()).expect("first");
    ledger.register(digest.clone(), path.clone()).expect("second — idempotent");
    assert_eq!(ledger.len(), 1);
}

#[test]
fn register_same_digest_different_path_is_conflict() {
    let mut ledger = ResidencyLedger::new();
    ledger
        .register("abc123".to_owned(), PathBuf::from("/staged/abc123.so"))
        .expect("first");
    let err = ledger
        .register("abc123".to_owned(), PathBuf::from("/other/abc123.so"))
        .expect_err("conflict");
    // The error should mention the digest.
    let msg = err.to_string();
    assert!(msg.contains("abc123"), "error should reference digest: {msg}");
}

#[test]
fn lookup_known_digest_returns_path() {
    let mut ledger = ResidencyLedger::new();
    let path = PathBuf::from("/staged/abc123.so");
    ledger.register("abc123".to_owned(), path.clone()).expect("register");
    let found = ledger.lookup("abc123").expect("should find");
    assert_eq!(*found, path);
}

#[test]
fn lookup_unknown_digest_returns_none() {
    let ledger = ResidencyLedger::new();
    assert!(ledger.lookup("nonexistent").is_none());
}

#[test]
fn records_iterator_yields_all() {
    let mut ledger = ResidencyLedger::new();
    ledger.register("d1".to_owned(), PathBuf::from("/a.so")).expect("ok");
    ledger.register("d2".to_owned(), PathBuf::from("/b.so")).expect("ok");
    let mut records: Vec<_> = ledger.records().collect();
    records.sort_by_key(|r| r.digest.clone());
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].digest, "d1");
    assert_eq!(records[1].digest, "d2");
}
