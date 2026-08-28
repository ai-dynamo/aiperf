// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Propagate plugin lock identity to re-exec subprocesses.
//!
//! The parent sets [`ENV_LOCK_PATH`] and [`ENV_LOCK_DIGEST`] on the child
//! [`Command`] before spawning so the subprocess can reconstruct the identical
//! frozen plugin universe without re-discovering the lock file.
//!
//! The child calls [`read_lock_env`] early in bootstrap, before any socket or
//! resource open, and verifies the digest matches the loaded bundle before
//! proceeding. A mismatch means the lock file was modified between parent spawn
//! and child read and the child refuses with [`PropagateError::DigestMismatch`].

use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment variable carrying the absolute path to the lock bundle.
pub const ENV_LOCK_PATH: &str = "AIPERF_PLUGIN_LOCK_PATH";

/// Environment variable carrying the BLAKE3 hex digest of the lock bundle.
pub const ENV_LOCK_DIGEST: &str = "AIPERF_PLUGIN_LOCK_DIGEST";

/// Error returned when lock propagation verification fails in the child.
#[derive(Debug)]
pub enum PropagateError {
    /// The lock file's actual digest does not match what the parent recorded.
    ///
    /// The lock was modified between parent spawn and child bootstrap, or the
    /// environment was corrupted in transit.
    DigestMismatch { expected: String, actual: String },
}

impl fmt::Display for PropagateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DigestMismatch { expected, actual } => write!(
                f,
                "plugin lock digest mismatch: expected {expected}, got {actual}; \
                 the lock file was modified between parent spawn and child bootstrap"
            ),
        }
    }
}

impl std::error::Error for PropagateError {}

/// Set the plugin lock environment variables on a subprocess [`Command`].
///
/// Call this in the parent after [`compose_plugin_universe`] succeeds, before
/// [`Command::spawn`].
pub fn set_lock_env(cmd: &mut Command, lock_path: &Path, lock_digest: &str) {
    cmd.env(ENV_LOCK_PATH, lock_path);
    cmd.env(ENV_LOCK_DIGEST, lock_digest);
}

/// Read the plugin lock environment variables in a child process.
///
/// Returns `None` when neither variable is set, meaning the parent was launched
/// without plugins and the child should proceed with an empty universe.
///
/// Returns `Some((path, digest))` when both variables are set and both are
/// structurally valid. The caller must load the bundle at `path` and verify
/// its digest matches before using it.
///
/// A non-absolute path or a digest that is not exactly 64 lower-case hex
/// characters causes a warning log and returns `None` so that a corrupted or
/// injected environment is treated the same as an absent lock rather than
/// silently accepted with bad data.
pub fn read_lock_env() -> Option<(PathBuf, String)> {
    let path_str = std::env::var(ENV_LOCK_PATH).ok()?;
    let digest = std::env::var(ENV_LOCK_DIGEST).ok()?;

    let path = PathBuf::from(&path_str);
    if !path.is_absolute() {
        tracing::warn!(
            path = %path.display(),
            "ignoring non-absolute {ENV_LOCK_PATH}; must be an absolute path"
        );
        return None;
    }

    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        tracing::warn!(
            "ignoring malformed {ENV_LOCK_DIGEST}: expected 64 hex characters, got {:?}",
            digest
        );
        return None;
    }

    Some((path, digest))
}

/// Verify that a loaded bundle's digest matches the propagated digest.
///
/// Returns `Ok(())` when the digests agree, or [`PropagateError::DigestMismatch`]
/// when they differ.
pub fn verify_propagated_digest(expected: &str, actual: &str) -> Result<(), PropagateError> {
    if expected == actual {
        Ok(())
    } else {
        Err(PropagateError::DigestMismatch {
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        })
    }
}
