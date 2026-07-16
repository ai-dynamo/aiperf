// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Resolve the `aiperf` binary to re-exec for one run's execution.
//!
//! There is no separate execution binary anymore: the front door re-execs
//! **itself** (`aiperf --execute`) for each run/probe/cell. The happy path is
//! therefore `current_exe()`. A single non-"runner" override, `AIPERF_EXEC_BIN`,
//! lets dev/test point the execution child at a different-features build (e.g. a
//! `--features dynosim` binary) without rebuilding the front door.

use std::path::PathBuf;

/// Resolve the execution binary to spawn in `--execute`/`--cell` mode:
///
/// 1. `$AIPERF_EXEC_BIN` (explicit override — dev/test point at a specific build),
/// 2. `current_exe()` (this same `aiperf` binary — the normal path).
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
        // Without an override the resolver yields this test binary's own path
        // (current_exe) and never errors.
        let resolved = resolve().unwrap();
        assert!(resolved.is_absolute() || !resolved.as_os_str().is_empty());
    }
}
