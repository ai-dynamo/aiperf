// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static binary inspection tests (Task 12).
//!
//! Tests operate on the `inspect_bytes` path so they don't need real binaries
//! on disk.  The ELF construction helpers produce minimal valid ELF64 LE
//! structures that exercise the inspection logic.

use aiperf_plugin_host::inspect::{ArtifactKind, PLUGIN_ENTRY_SYMBOL, SearchPolicy, inspect_bytes};

/// A minimal blob that is not a valid ELF/MachO/PE produces an Unknown receipt
/// with a quarantine reason, not an error.
#[test]
fn unknown_format_quarantined_not_error() {
    let bytes = b"not a valid binary at all \xFF\xFF";
    let receipt = inspect_bytes(bytes, "fake-digest".to_owned())
        .expect("inspect_bytes should not return Err for unknown format");
    assert_eq!(receipt.artifact_kind, ArtifactKind::Unknown);
    assert!(!receipt.has_entry_symbol);
    assert!(
        !receipt.quarantine_reasons.is_empty(),
        "unknown format must have quarantine reasons"
    );
}

/// An empty file produces an Unknown receipt (not a panic or error).
#[test]
fn empty_bytes_quarantined() {
    let receipt =
        inspect_bytes(&[], "empty-digest".to_owned()).expect("empty bytes should not error");
    assert_eq!(receipt.artifact_kind, ArtifactKind::Unknown);
}

/// The digest passed in is preserved verbatim in the receipt.
#[test]
fn digest_preserved_verbatim() {
    let bytes = b"garbage";
    let digest = "abc123def456".to_owned();
    let receipt = inspect_bytes(bytes, digest.clone()).expect("should not error");
    assert_eq!(receipt.digest, digest);
}

/// For an unknown format the dependency search policy is Rejected.
#[test]
fn unknown_format_search_policy_rejected() {
    let receipt = inspect_bytes(b"bad", "d".to_owned()).expect("no error");
    assert_eq!(receipt.dependency_search_policy, SearchPolicy::Rejected);
}
