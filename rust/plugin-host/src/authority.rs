// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Artifact authority checking (Task 13).
//!
//! Verifies that an acquired artifact's digest matches the manifest
//! declaration and, optionally, a separate authority record (e.g., a
//! detached BLAKE3 digest file or a signature from a trusted key).
//!
//! For the initial implementation the authority check is limited to
//! digest-match verification.  Cryptographic signature verification is
//! reserved for a future task.

use crate::{acquire::AcquiredArtifact, error::AcquireError};

/// The result of an authority check against one acquired artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityVerdict {
    /// The artifact passes all configured authority checks.
    Trusted,
    /// The artifact's digest did not match the manifest declaration.
    DigestMismatch { expected: String, actual: String },
    /// The authority record (e.g., detached signature file) is absent.
    AuthorityRecordMissing,
}

impl AuthorityVerdict {
    /// Return `true` if the verdict is `Trusted`.
    pub fn is_trusted(&self) -> bool {
        matches!(self, AuthorityVerdict::Trusted)
    }
}

/// Verify that `artifact.digest` matches the manifest-declared `expected_digest`.
///
/// This is the minimal authority check: the host trusts the digest comparison
/// alone when no external authority record is present.
pub fn verify_digest_authority(
    artifact: &AcquiredArtifact,
    expected_digest: &str,
) -> Result<AuthorityVerdict, AcquireError> {
    // Parse both sides as `blake3::Hash` so the comparison is constant-time,
    // matching the pattern in `PluginInventoryV1::verify_digest`.
    match (
        artifact.digest.parse::<blake3::Hash>(),
        expected_digest.parse::<blake3::Hash>(),
    ) {
        (Ok(actual), Ok(expected)) if actual == expected => Ok(AuthorityVerdict::Trusted),
        (Ok(_), Ok(_)) => Ok(AuthorityVerdict::DigestMismatch {
            expected: expected_digest.to_owned(),
            actual: artifact.digest.clone(),
        }),
        _ => Ok(AuthorityVerdict::DigestMismatch {
            expected: expected_digest.to_owned(),
            actual: artifact.digest.clone(),
        }),
    }
}

/// Verify authority for a raw byte slice + known digest.
///
/// Both sides are parsed as `blake3::Hash` before comparison, so two identical
/// but unparseable strings (e.g. `("", "")`) are a `DigestMismatch` rather than
/// an accidental `Trusted`.
pub fn verify_digest_authority_bytes(
    actual_digest: &str,
    expected_digest: &str,
) -> AuthorityVerdict {
    let mismatch = || AuthorityVerdict::DigestMismatch {
        expected: expected_digest.to_owned(),
        actual: actual_digest.to_owned(),
    };
    match (
        actual_digest.parse::<blake3::Hash>(),
        expected_digest.parse::<blake3::Hash>(),
    ) {
        (Ok(actual), Ok(expected)) if actual == expected => AuthorityVerdict::Trusted,
        _ => mismatch(),
    }
}
