// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for plugin lock propagation to re-exec subprocess.
//!
//! `read_lock_env` reads process-global state, so every test here mutates and
//! restores the same two variables. Cargo runs the tests in one binary on
//! multiple threads, so they are serialized behind one mutex.

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard, OnceLock};

use aiperf_cli::plugins::propagate::{
    ENV_LOCK_DIGEST, ENV_LOCK_PATH, PropagateError, read_lock_env,
};

/// A structurally valid BLAKE3 digest: exactly 64 lower-case hex characters.
const VALID_DIGEST: &str = "a3b4c5d6e7f8a1b2c3d4e5f6a7b8c9d0a3b4c5d6e7f8a1b2c3d4e5f6a7b8c9d0";

fn env_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Set (or clear, on `None`) both variables and read them back.
fn read_with_env(
    path: Option<&str>,
    digest: Option<&str>,
) -> Result<Option<(PathBuf, String)>, PropagateError> {
    let _guard = env_guard();
    // Safety: all env mutation in this binary is serialized behind `env_guard`.
    unsafe {
        match path {
            Some(value) => std::env::set_var(ENV_LOCK_PATH, value),
            None => std::env::remove_var(ENV_LOCK_PATH),
        }
        match digest {
            Some(value) => std::env::set_var(ENV_LOCK_DIGEST, value),
            None => std::env::remove_var(ENV_LOCK_DIGEST),
        }
    }
    let result = read_lock_env();
    unsafe {
        std::env::remove_var(ENV_LOCK_PATH);
        std::env::remove_var(ENV_LOCK_DIGEST);
    }
    result
}

#[test]
fn env_vars_roundtrip() {
    let (path, digest) = read_with_env(Some("/tmp/test.lock"), Some(VALID_DIGEST))
        .expect("well-formed environment is accepted")
        .expect("both env vars are set");
    assert_eq!(path, PathBuf::from("/tmp/test.lock"));
    assert_eq!(digest, VALID_DIGEST);
}

#[test]
fn no_env_vars_returns_none() {
    assert!(
        read_with_env(None, None)
            .expect("an absent lock is not an error")
            .is_none(),
        "neither variable set means an empty plugin universe"
    );
}

#[test]
fn partial_env_path_only_is_rejected() {
    assert!(
        matches!(
            read_with_env(Some("/tmp/test.lock"), None),
            Err(PropagateError::PartialEnvironment { .. })
        ),
        "a half-set environment must fail closed, not read as absent"
    );
}

#[test]
fn partial_env_digest_only_is_rejected() {
    assert!(
        matches!(
            read_with_env(None, Some(VALID_DIGEST)),
            Err(PropagateError::PartialEnvironment { .. })
        ),
        "a half-set environment must fail closed, not read as absent"
    );
}

#[test]
fn short_digest_is_rejected() {
    assert!(
        matches!(
            read_with_env(Some("/tmp/test.lock"), Some("deadbeef")),
            Err(PropagateError::MalformedDigest { .. })
        ),
        "a digest shorter than 64 characters must be refused"
    );
}

#[test]
fn non_hex_digest_is_rejected() {
    let non_hex = "z".repeat(64);
    assert!(
        matches!(
            read_with_env(Some("/tmp/test.lock"), Some(&non_hex)),
            Err(PropagateError::MalformedDigest { .. })
        ),
        "a 64-character non-hex digest must be refused"
    );
}

#[test]
fn non_absolute_path_is_rejected() {
    assert!(
        matches!(
            read_with_env(Some("relative/test.lock"), Some(VALID_DIGEST)),
            Err(PropagateError::NonAbsolutePath { .. })
        ),
        "a non-absolute lock path must be refused"
    );
}

#[test]
fn set_lock_env_propagates_to_command() {
    use aiperf_cli::plugins::propagate::set_lock_env;
    use std::process::Command;

    // `Command`'s env map is not readable, so run a child that prints both
    // variables and assert on what it inherited.
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(format!("printf '%s\\n%s' \"${ENV_LOCK_PATH}\" \"${ENV_LOCK_DIGEST}\""));
    set_lock_env(
        &mut cmd,
        std::path::Path::new("/tmp/lock.bundle"),
        VALID_DIGEST,
    );
    let output = cmd.output().expect("spawn child shell");
    let stdout = String::from_utf8(output.stdout).expect("child printed utf-8");
    assert_eq!(stdout, format!("/tmp/lock.bundle\n{VALID_DIGEST}"));
}
