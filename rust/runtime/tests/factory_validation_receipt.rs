// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for `ValidatedRunPlan`: deterministic digest, opaque-state non-transfer,
//! and canonical ordering semantics.

use aiperf_runtime::engine::validated_run_plan::ValidatedRunPlan;

/// Minimal stub run bytes — content is arbitrary; only determinism matters.
const RUN_BYTES_A: &[u8] = b"run-dto-canonical-bytes-a";
const RUN_BYTES_B: &[u8] = b"run-dto-canonical-bytes-b";

#[test]
fn deterministic_digest_same_inputs() {
    let p1 = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &[]);
    let p2 = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &[]);
    assert_eq!(
        p1.canonical_digest(),
        p2.canonical_digest(),
        "same inputs must produce same digest"
    );
}

#[test]
fn different_run_bytes_different_digest() {
    let p1 = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &[]);
    let p2 = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_B, &[]);
    assert_ne!(
        p1.canonical_digest(),
        p2.canonical_digest(),
        "different run bytes must produce different digest"
    );
}

#[test]
fn receipt_bytes_change_digest() {
    let no_receipts = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &[]);
    let with_receipts =
        ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &[b"receipt-1".as_ref()]);
    assert_ne!(
        no_receipts.canonical_digest(),
        with_receipts.canonical_digest(),
        "adding receipts must change the digest"
    );
}

#[test]
fn receipt_order_is_sorted_for_determinism() {
    // Two orderings of the same receipts must produce the same digest.
    let receipts_ab: Vec<&[u8]> = vec![b"receipt-a", b"receipt-b"];
    let receipts_ba: Vec<&[u8]> = vec![b"receipt-b", b"receipt-a"];
    let p_ab = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &receipts_ab);
    let p_ba = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &receipts_ba);
    assert_eq!(
        p_ab.canonical_digest(),
        p_ba.canonical_digest(),
        "receipts must be sorted before hashing so order does not matter"
    );
}

#[test]
fn framing_prevents_concatenation_collision() {
    // Without length framing, hashing "ab" with no receipts and "a" with the
    // receipt "b" would stream the identical byte sequence.
    let joined = ValidatedRunPlan::from_canonical_bytes(b"ab", &[]);
    let split = ValidatedRunPlan::from_canonical_bytes(b"a", &[b"b".as_ref()]);
    assert_ne!(
        joined.canonical_digest(),
        split.canonical_digest(),
        "boundary-ambiguous inputs must not collide"
    );
}

#[test]
fn canonical_digest_is_hex_string() {
    let plan = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &[]);
    // BLAKE3 hex is 64 lowercase hex chars
    assert_eq!(plan.canonical_digest().len(), 64);
    assert!(
        plan.canonical_digest()
            .chars()
            .all(|c| c.is_ascii_hexdigit()),
        "digest must be lowercase hex"
    );
}

#[test]
fn capture_plan_accessible() {
    let plan = ValidatedRunPlan::from_canonical_bytes(RUN_BYTES_A, &[]);
    // By default (no receipts with requirements) capture plan is empty.
    assert!(!plan.capture_plan().requires_exact_records);
}
