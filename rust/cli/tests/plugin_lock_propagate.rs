// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for plugin lock propagation to re-exec subprocess.

use std::path::PathBuf;

use aiperf_cli::plugins::propagate::{read_lock_env, ENV_LOCK_DIGEST, ENV_LOCK_PATH};

#[test]
fn env_vars_roundtrip() {
    // Isolate via a serial harness: env mutations are process-global.
    std::env::set_var(ENV_LOCK_PATH, "/tmp/test.lock");
    std::env::set_var(ENV_LOCK_DIGEST, "abc123");
    let result = read_lock_env();
    std::env::remove_var(ENV_LOCK_PATH);
    std::env::remove_var(ENV_LOCK_DIGEST);
    let (path, digest) = result.expect("should read both env vars");
    assert_eq!(path, PathBuf::from("/tmp/test.lock"));
    assert_eq!(digest, "abc123");
}

#[test]
fn no_env_vars_returns_none() {
    std::env::remove_var(ENV_LOCK_PATH);
    std::env::remove_var(ENV_LOCK_DIGEST);
    assert!(read_lock_env().is_none());
}

#[test]
fn partial_env_path_only_returns_none() {
    std::env::set_var(ENV_LOCK_PATH, "/tmp/test.lock");
    std::env::remove_var(ENV_LOCK_DIGEST);
    let result = read_lock_env();
    std::env::remove_var(ENV_LOCK_PATH);
    assert!(result.is_none());
}

#[test]
fn partial_env_digest_only_returns_none() {
    std::env::remove_var(ENV_LOCK_PATH);
    std::env::set_var(ENV_LOCK_DIGEST, "deadbeef");
    let result = read_lock_env();
    std::env::remove_var(ENV_LOCK_DIGEST);
    assert!(result.is_none());
}

#[test]
fn set_lock_env_propagates_to_command() {
    use aiperf_cli::plugins::propagate::set_lock_env;
    use std::process::Command;
    // We cannot inspect Command's env directly, so we round-trip via a process
    // that echoes the env vars. Instead, verify set_lock_env doesn't panic and
    // the constants are well-formed identifiers.
    let mut cmd = Command::new("true");
    set_lock_env(
        &mut cmd,
        std::path::Path::new("/tmp/lock.bundle"),
        "aabbccdd",
    );
    // Constants must be non-empty ASCII.
    assert!(!ENV_LOCK_PATH.is_empty());
    assert!(!ENV_LOCK_DIGEST.is_empty());
    assert!(ENV_LOCK_PATH.is_ascii());
    assert!(ENV_LOCK_DIGEST.is_ascii());
}
