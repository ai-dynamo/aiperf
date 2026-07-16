// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Transparent passthrough to the Python frontend for not-yet-ported commands.

use std::process::Command;

/// Re-exec `python -m aiperf <argv>` inheriting this process's stdio, returning
/// the child's exit code.
///
/// Used for every subcommand the native front door does not yet own. The Python
/// interpreter is `$AIPERF_PYTHON` when set, else `python`. This is a plain
/// spawn-and-wait rather than an `exec(2)` replacement so the caller keeps
/// control of the process (and so the same code path works on Windows).
pub fn exec_python(argv: &[String]) -> anyhow::Result<i32> {
    let python = std::env::var("AIPERF_PYTHON").unwrap_or_else(|_| "python".to_string());
    let status = Command::new(&python)
        .arg("-m")
        .arg("aiperf")
        .args(argv)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to delegate to `{python} -m aiperf`: {e}"))?;
    // A signal-terminated child has no code; surface a nonzero status.
    Ok(status.code().unwrap_or(1))
}

/// Delegate `aiperf profile <args>` to the pure-Python app: `python -m aiperf
/// profile <args>`. Used as the parity fallback when the native front door
/// encounters a `profile` flag it does not model (accuracy benchmarks,
/// adaptive/BO search recipes, and other Python-only surfaces) — the native
/// binary stays at least as capable as Python for every flag.
pub fn exec_python_profile(args: &[String]) -> anyhow::Result<i32> {
    let mut argv = vec!["profile".to_string()];
    argv.extend_from_slice(args);
    exec_python(&argv)
}
