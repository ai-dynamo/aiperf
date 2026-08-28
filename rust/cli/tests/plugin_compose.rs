// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for CLI plugin composition bootstrap (Task 17).

use std::path::Path;

use aiperf_cli::plugins::compose::{compose_plugin_universe, ComposeError};
use aiperf_cli::plugins::lock_path::default_lock_path;

/// An absent lock file is a valid no-plugin configuration: compose returns an
/// empty frozen universe without error.
#[test]
fn no_lock_file_is_noop() {
    let tmp = tempfile::tempdir().unwrap();
    let absent = tmp.path().join("nonexistent.plugin-lock");
    let result = compose_plugin_universe(&absent);
    assert!(
        result.is_ok(),
        "compose_plugin_universe on absent lock should succeed: {result:?}"
    );
    let universe = result.unwrap();
    assert!(universe.is_empty(), "absent lock should yield empty universe");
}

/// The default lock path is a sibling of the config file, with the same stem
/// and a `.plugin-lock` extension.
#[test]
fn default_lock_path_is_sibling_with_stem() {
    let config = Path::new("/some/path/benchmark.yaml");
    let lock = default_lock_path(config);
    assert_eq!(lock, Path::new("/some/path/benchmark.plugin-lock"));
}

/// A config with no extension still gets a `.plugin-lock` sibling.
#[test]
fn default_lock_path_no_extension() {
    let config = Path::new("/run/benchmark");
    let lock = default_lock_path(config);
    assert_eq!(lock, Path::new("/run/benchmark.plugin-lock"));
}

/// The error type is displayable.
#[test]
fn compose_error_is_display() {
    let e = ComposeError::PluginLoad {
        package_id: "my-plugin".to_owned(),
        reason: "dlopen failed".to_owned(),
    };
    let s = e.to_string();
    assert!(s.contains("my-plugin"), "error should mention package id");
    assert!(s.contains("dlopen failed"), "error should mention reason");
}
