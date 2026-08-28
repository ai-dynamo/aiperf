// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authority verification tests (Task 13).

use aiperf_plugin_host::authority::{AuthorityVerdict, verify_digest_authority_bytes};

#[test]
fn matching_digests_yield_trusted() {
    let verdict = verify_digest_authority_bytes("abc123", "abc123");
    assert_eq!(verdict, AuthorityVerdict::Trusted);
}

#[test]
fn mismatched_digests_yield_digest_mismatch() {
    let verdict = verify_digest_authority_bytes("actual_digest", "expected_digest");
    assert!(
        matches!(verdict, AuthorityVerdict::DigestMismatch { .. }),
        "mismatched digests must return DigestMismatch"
    );
}

#[test]
fn digest_mismatch_preserves_both_values() {
    let verdict = verify_digest_authority_bytes("AAA", "BBB");
    match verdict {
        AuthorityVerdict::DigestMismatch { expected, actual } => {
            assert_eq!(expected, "BBB");
            assert_eq!(actual, "AAA");
        }
        other => panic!("expected DigestMismatch, got {other:?}"),
    }
}
