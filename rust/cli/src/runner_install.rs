// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Locate the `aiperf-runner` executable this CLI drives.

use std::path::PathBuf;

/// Resolve the runner binary path, in precedence order:
///
/// 1. `$AIPERF_RUNNER_BIN` (explicit override — used by tests and by the wheel's
///    `importlib.resources` discovery once packaging lands),
/// 2. a sibling `aiperf-runner` next to the current executable (the common
///    cargo `target/<profile>/` layout, and the interned-wheel layout),
/// 3. `aiperf-runner` resolved via `$PATH`.
///
/// Resolution is intentionally lenient: it returns a path without proving the
/// file is executable, so the failure surfaces at spawn time with the OS error
/// (which names the path), matching the Python `RunnerInstallation` behaviour.
pub fn resolve() -> anyhow::Result<PathBuf> {
    if let Ok(explicit) = std::env::var("AIPERF_RUNNER_BIN") {
        return Ok(PathBuf::from(explicit));
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let sibling = dir.join(runner_file_name());
        if sibling.exists() {
            return Ok(sibling);
        }
    }
    Ok(PathBuf::from(runner_file_name()))
}

/// Platform-correct runner file name.
fn runner_file_name() -> &'static str {
    if cfg!(windows) {
        "aiperf-runner.exe"
    } else {
        "aiperf-runner"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // `AIPERF_RUNNER_BIN` is process-global; cargo runs tests in parallel
    // threads within one process, so env-var behaviour is exercised in a single
    // test to avoid a set/remove race between tests.
    #[test]
    fn resolves_env_override_then_falls_back() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        // SAFETY: the two env mutations are confined to this one test; no other
        // test touches `AIPERF_RUNNER_BIN`, so no concurrent reader races.
        unsafe { std::env::set_var("AIPERF_RUNNER_BIN", tmp.path()) };
        assert_eq!(resolve().unwrap(), tmp.path());

        unsafe { std::env::remove_var("AIPERF_RUNNER_BIN") };
        // Without an override the resolver still yields a runner path (sibling
        // or bare name) and never errors.
        let resolved = resolve().unwrap();
        assert!(
            resolved
                .as_os_str()
                .to_string_lossy()
                .contains("aiperf-runner")
        );
    }
}
