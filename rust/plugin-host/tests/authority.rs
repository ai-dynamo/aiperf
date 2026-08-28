// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Authority verification tests (Task 13).

use aiperf_plugin_host::authority::{AuthorityVerdict, verify_digest_authority_bytes};

/// BLAKE3 of the empty input, as canonical lowercase hex.
const DIGEST_A: &str = "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262";
/// A second, distinct well-formed 64-hex digest.
const DIGEST_B: &str = "0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8";

#[test]
fn matching_digests_yield_trusted() {
    let verdict = verify_digest_authority_bytes(DIGEST_A, DIGEST_A);
    assert_eq!(verdict, AuthorityVerdict::Trusted);
}

#[test]
fn mismatched_digests_yield_digest_mismatch() {
    let verdict = verify_digest_authority_bytes(DIGEST_A, DIGEST_B);
    assert!(
        matches!(verdict, AuthorityVerdict::DigestMismatch { .. }),
        "mismatched digests must return DigestMismatch"
    );
}

#[test]
fn identical_unparseable_digests_are_not_trusted() {
    // Both sides must parse as a BLAKE3 hash; equal-but-invalid strings such as
    // an empty digest or a placeholder must never be admitted as authority.
    for bogus in ["", "none", "abc123", "sha256:deadbeef"] {
        assert!(
            matches!(
                verify_digest_authority_bytes(bogus, bogus),
                AuthorityVerdict::DigestMismatch { .. }
            ),
            "identical invalid digest {bogus:?} must not be Trusted"
        );
    }
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
