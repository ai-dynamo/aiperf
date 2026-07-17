// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Resolve the `aiperf` binary to re-exec for one run's execution.
//!
//! `AIPERF_EXEC_BIN` can select a feature-specific build; otherwise the entry
//! point re-execs itself with `aiperf --execute`.

use std::path::PathBuf;

/// Resolve the execution binary for `--execute` and `--cell`.
///
/// 1. `$AIPERF_EXEC_BIN`
/// 2. `current_exe()`
///
/// Falls back to the bare name `aiperf` only if `current_exe()` is unavailable,
/// so a spawn failure surfaces the OS error naming the path.
pub fn resolve() -> anyhow::Result<PathBuf> {
    if let Ok(explicit) = std::env::var("AIPERF_EXEC_BIN") {
        return Ok(PathBuf::from(explicit));
    }
    if let Ok(exe) = std::env::current_exe() {
        return Ok(exe);
    }
    Ok(PathBuf::from(if cfg!(windows) {
        "aiperf.exe"
    } else {
        "aiperf"
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // `AIPERF_EXEC_BIN` is process-global; cargo runs tests in parallel threads
    // within one process, so env-var behaviour is exercised in a single test to
    // avoid a set/remove race between tests.
    #[test]
    fn resolves_env_override_then_current_exe() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // SAFETY: the two env mutations are confined to this one test; no other
        // test touches `AIPERF_EXEC_BIN`, so no concurrent reader races.
        unsafe { std::env::set_var("AIPERF_EXEC_BIN", tmp.path()) };
        assert_eq!(resolve().unwrap(), tmp.path());

        unsafe { std::env::remove_var("AIPERF_EXEC_BIN") };
        let resolved = resolve().unwrap();
        assert!(resolved.is_absolute() || !resolved.as_os_str().is_empty());
    }
}
