// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical default lock file path derivation.
//!
//! The plugin lock file lives as a sibling of the config file with the same
//! stem and a `.plugin-lock` extension.  This derivation is the single
//! authoritative rule so the CLI, the lock tool, and future integrations all
//! agree on the path without coordination.

use std::path::{Path, PathBuf};

/// Return the canonical default plugin lock path for a given config path.
///
/// The result is a sibling of `config_path` with the same stem and a
/// `.plugin-lock` extension:
/// - `/path/to/benchmark.yaml` → `/path/to/benchmark.plugin-lock`
/// - `/path/to/benchmark`      → `/path/to/benchmark.plugin-lock`
///
/// The caller supplies the config path and may override the result with a
/// `--plugin-lock` CLI flag.
pub fn default_lock_path(config_path: &Path) -> PathBuf {
    let stem = config_path.file_stem().unwrap_or(config_path.as_os_str());
    let mut name = stem.to_os_string();
    name.push(".plugin-lock");
    config_path
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(name)
}
