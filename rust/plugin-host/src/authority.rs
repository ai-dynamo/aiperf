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
    if artifact.digest == expected_digest {
        Ok(AuthorityVerdict::Trusted)
    } else {
        Ok(AuthorityVerdict::DigestMismatch {
            expected: expected_digest.to_owned(),
            actual: artifact.digest.clone(),
        })
    }
}

/// Verify authority for a raw byte slice + known digest.
pub fn verify_digest_authority_bytes(
    actual_digest: &str,
    expected_digest: &str,
) -> AuthorityVerdict {
    if actual_digest == expected_digest {
        AuthorityVerdict::Trusted
    } else {
        AuthorityVerdict::DigestMismatch {
            expected: expected_digest.to_owned(),
            actual: actual_digest.to_owned(),
        }
    }
}
