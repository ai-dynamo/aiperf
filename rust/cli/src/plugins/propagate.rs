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
//!
//! The read fails closed: only a completely absent pair means "no plugins".
//! A half-set or structurally invalid environment is an error, never silently
//! downgraded to an unlocked run.

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
    /// Exactly one of the two propagation variables is present.
    ///
    /// The parent always sets both or neither, so a half-set environment means
    /// the child's view of the plugin universe was tampered with or truncated.
    PartialEnvironment {
        present: &'static str,
        missing: &'static str,
    },
    /// [`ENV_LOCK_DIGEST`] is not exactly 64 hexadecimal characters.
    MalformedDigest { value: String },
    /// [`ENV_LOCK_PATH`] is not an absolute path.
    NonAbsolutePath { value: String },
    /// An environment variable is set but its bytes are not valid UTF-8.
    ///
    /// A non-UTF-8 value cannot be a valid lock path or BLAKE3 hex digest, so
    /// the child refuses rather than silently treating it as absent.
    NonUnicode { variable: &'static str },
}

impl fmt::Display for PropagateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DigestMismatch { expected, actual } => write!(
                f,
                "plugin lock digest mismatch: expected {expected}, got {actual}; \
                 the lock file was modified between parent spawn and child bootstrap"
            ),
            Self::PartialEnvironment { present, missing } => write!(
                f,
                "incomplete plugin lock propagation: {present} is set but {missing} is not; \
                 the parent always sets both or neither"
            ),
            Self::MalformedDigest { value } => write!(
                f,
                "malformed {ENV_LOCK_DIGEST}: expected 64 hexadecimal characters, got {value:?}"
            ),
            Self::NonAbsolutePath { value } => write!(
                f,
                "malformed {ENV_LOCK_PATH}: expected an absolute path, got {value:?}"
            ),
            Self::NonUnicode { variable } => write!(
                f,
                "plugin lock environment variable {variable} contains non-UTF-8 bytes; \
                 the child refuses rather than treating it as absent"
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
/// Returns `Ok(None)` only when **neither** variable is set, meaning the parent
/// was launched without plugins and the child should proceed with an empty
/// universe.
///
/// Returns `Ok(Some((path, digest)))` when both variables are set and both are
/// structurally valid. The caller must load the bundle at `path` and verify its
/// digest matches before using it.
///
/// Every other shape is an error, not an absent lock: a half-set environment,
/// a non-absolute path, or a digest that is not exactly 64 hexadecimal
/// characters. Treating those as "no plugins" would let a truncated or injected
/// environment silently downgrade a locked run to an unlocked one, so the
/// caller must refuse instead.
pub fn read_lock_env() -> Result<Option<(PathBuf, String)>, PropagateError> {
    let path_var = env_opt(ENV_LOCK_PATH)?;
    let digest_var = env_opt(ENV_LOCK_DIGEST)?;

    let (path_str, digest) = match (path_var, digest_var) {
        (None, None) => return Ok(None),
        (Some(path), Some(digest)) => (path, digest),
        (Some(_), None) => {
            return Err(PropagateError::PartialEnvironment {
                present: ENV_LOCK_PATH,
                missing: ENV_LOCK_DIGEST,
            });
        }
        (None, Some(_)) => {
            return Err(PropagateError::PartialEnvironment {
                present: ENV_LOCK_DIGEST,
                missing: ENV_LOCK_PATH,
            });
        }
    };

    let path = PathBuf::from(&path_str);
    if !path.is_absolute() {
        return Err(PropagateError::NonAbsolutePath { value: path_str });
    }

    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(PropagateError::MalformedDigest { value: digest });
    }

    Ok(Some((path, digest)))
}

/// Return `Ok(None)` when the variable is absent, `Ok(Some(v))` when it is set
/// and valid UTF-8, and `Err(NonUnicode)` when it is set but not valid UTF-8.
///
/// `std::env::var(..).ok()` collapses both cases into `None`, which would let a
/// non-UTF-8 lock path silently downgrade a locked run to an unlocked one.
fn env_opt(variable: &'static str) -> Result<Option<String>, PropagateError> {
    match std::env::var(variable) {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(PropagateError::NonUnicode { variable })
        }
    }
}

/// Verify that a loaded bundle's digest matches the propagated digest.
///
/// Returns `Ok(())` when the digests agree, or [`PropagateError::DigestMismatch`]
/// when they differ.
pub fn verify_propagated_digest(expected: &str, actual: &str) -> Result<(), PropagateError> {
    let expected_hash =
        blake3::Hash::from_hex(expected).map_err(|_| PropagateError::MalformedDigest {
            value: expected.to_owned(),
        })?;
    let actual_hash =
        blake3::Hash::from_hex(actual).map_err(|_| PropagateError::MalformedDigest {
            value: actual.to_owned(),
        })?;
    if expected_hash == actual_hash {
        Ok(())
    } else {
        Err(PropagateError::DigestMismatch {
            expected: expected.to_owned(),
            actual: actual.to_owned(),
        })
    }
}
