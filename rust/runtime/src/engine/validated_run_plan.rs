// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable validated run plan produced before runtime construction.
//!
//! A [`ValidatedRunPlan`] pairs a BLAKE3 digest of the canonical run DTO with
//! the combined capture requirements from all configured exporters.  It is
//! constructed once, before sockets or files are opened, so capture
//! incompatibilities (e.g. exact-record requirements under sketch mode) surface
//! before any effect.

use crate::export::capture::ExportCapturePlan;

/// Immutable plan produced by validating the canonical run DTO against the
/// configured exporter set before any runtime effect.
///
/// Fields are private: a plan is only ever produced by hashing its own inputs,
/// so a caller must not be able to substitute a digest that does not describe
/// the accompanying capture plan.
#[derive(Debug, Clone)]
pub struct ValidatedRunPlan {
    /// BLAKE3 hex digest of the canonical run DTO bytes concatenated with the
    /// sorted factory-receipt bytes.  64 lowercase hex characters.
    canonical_digest: String,
    /// Combined capture requirements across all configured exporters.
    capture_plan: ExportCapturePlan,
}

impl ValidatedRunPlan {
    /// Construct from raw canonical run bytes and an optional set of
    /// factory-receipt byte slices (sorted before hashing for determinism).
    ///
    /// This is the only constructor: it performs no I/O, opens no file, and
    /// binds no opaque factory state.  The resulting plan is immutable.
    pub fn from_canonical_bytes(run_bytes: &[u8], receipt_bytes: &[&[u8]]) -> Self {
        Self::from_canonical_bytes_with_capture(
            run_bytes,
            receipt_bytes,
            ExportCapturePlan::default(),
        )
    }

    /// Construct from raw canonical run bytes, factory-receipt bytes, and the
    /// combined capture requirements from all configured exporters.
    ///
    /// Use this constructor when the caller has already assembled the
    /// [`ExportCapturePlan`] from the registered exporter set.
    /// [`from_canonical_bytes`] is a convenience shorthand that supplies an
    /// empty plan, suitable for callers that have not yet wired exporter
    /// capture requirements.
    pub fn from_canonical_bytes_with_capture(
        run_bytes: &[u8],
        receipt_bytes: &[&[u8]],
        capture_plan: ExportCapturePlan,
    ) -> Self {
        let digest = compute_digest(run_bytes, receipt_bytes);
        Self {
            canonical_digest: digest,
            capture_plan,
        }
    }

    /// The BLAKE3 hex digest binding the canonical run DTO to its sorted
    /// factory receipts. 64 lowercase hex characters.
    pub fn canonical_digest(&self) -> &str {
        &self.canonical_digest
    }

    /// The combined capture requirements across all configured exporters.
    pub fn capture_plan(&self) -> &ExportCapturePlan {
        &self.capture_plan
    }
}

/// Domain separator prefixed to every canonical run-plan digest preimage so a
/// digest from this construction can never collide with one from another.
const DIGEST_DOMAIN: &[u8] = b"aiperf.run.plan.v1\x00";

/// Compute the BLAKE3 hex digest of `run_bytes` followed by sorted
/// `receipt_bytes`.  Sorting the receipts makes the digest independent of the
/// order in which factory registrations were encountered.
///
/// Every variable-length field is length-framed with its little-endian `u64`
/// byte length before its bytes, and the receipt count is framed before the
/// receipts.  Without framing the streaming append is ambiguous: `(b"ab", [])`
/// and `(b"a", [b"b"])` would produce the same preimage.
fn compute_digest(run_bytes: &[u8], receipt_bytes: &[&[u8]]) -> String {
    let mut sorted: Vec<&[u8]> = receipt_bytes.to_vec();
    sorted.sort();

    let mut hasher = blake3::Hasher::new();
    hasher.update(DIGEST_DOMAIN);
    hasher.update(&(run_bytes.len() as u64).to_le_bytes());
    hasher.update(run_bytes);
    hasher.update(&(sorted.len() as u64).to_le_bytes());
    for receipt in &sorted {
        hasher.update(&(receipt.len() as u64).to_le_bytes());
        hasher.update(receipt);
    }
    hasher.finalize().to_hex().to_string()
}
