// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Process-only delegation for allowlisted Python-owned utilities.
//!
//! Native benchmark commands never enter the Python product. The small utility
//! allowlist and named Rust shims execute only through external Python processes.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Spawn the configured Python interpreter for an allowlisted utility command.
///
/// `$AIPERF_PYTHON` selects the interpreter and defaults to `python`.
pub fn exec_python_utility(argv: &[String]) -> anyhow::Result<i32> {
    let python = python_executable()?;
    exec_python_module(&python, "aiperf", argv)
}

/// Spawn a named Rust-support shim with inherited stdio.
///
/// The caller selects only a fixed shim name from the native command router.
pub fn exec_rust_shim(python: &Path, shim: &str, argv: &[String]) -> anyhow::Result<i32> {
    let mut arguments = Vec::with_capacity(argv.len() + 1);
    arguments.push(shim.to_owned());
    arguments.extend(argv.iter().cloned());
    exec_python_module(python, "aiperf.rust_shims", &arguments)
}

/// Return the interpreter selected for Python utility subprocesses.
pub fn python_executable() -> anyhow::Result<PathBuf> {
    match std::env::var_os("AIPERF_PYTHON") {
        Some(python) if python.is_empty() => anyhow::bail!("AIPERF_PYTHON must not be empty"),
        Some(python) => Ok(PathBuf::from(python)),
        None => Ok(PathBuf::from("python")),
    }
}

fn exec_python_module(python: &Path, module: &str, argv: &[String]) -> anyhow::Result<i32> {
    let status = Command::new(python)
        .arg("-m")
        .arg(module)
        .args(argv)
        .status()
        .map_err(|error| anyhow::anyhow!("failed to delegate to `{}`: {error}", python.display()))?;
    Ok(status.code().unwrap_or(1))
}
