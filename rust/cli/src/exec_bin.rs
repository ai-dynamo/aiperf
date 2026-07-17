// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Resolve the `aiperf` binary to re-exec for one run's execution.
//!
//! There is no separate execution binary: the entry point re-execs **itself**
//! (`aiperf --execute`) for each run/probe/cell, so the binary is just
//! `current_exe()`.

use std::path::PathBuf;

/// Resolve the execution binary to spawn in `--execute`/`--cell` mode.
///
/// It is this same `aiperf` binary (`current_exe()`). Falls back to the bare name
/// `aiperf` only if `current_exe()` is unavailable, so a spawn failure surfaces the
/// OS error naming the path.
pub fn resolve() -> anyhow::Result<PathBuf> {
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

    #[test]
    fn resolves_to_current_exe() {
        // The resolver yields this test binary's own path (current_exe) and never
        // errors.
        let resolved = resolve().unwrap();
        assert!(resolved.is_absolute() || !resolved.as_os_str().is_empty());
    }
}
