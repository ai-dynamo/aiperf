// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the hermetic plugin build machinery.

use aiperf_plugin_sdk::build::{BuildConfig, BuildError, build_plugin};
use std::path::PathBuf;

fn temp_dir() -> tempfile::TempDir {
    tempfile::TempDir::new().expect("temp dir")
}

fn write_file(dir: &std::path::Path, name: &str, content: &str) {
    std::fs::write(dir.join(name), content).expect("write file");
}

/// Minimal valid plugin directory fixture.
fn make_valid_plugin_dir(dir: &std::path::Path) {
    write_file(
        dir,
        "Cargo.toml",
        r#"[package]
name = "my-plugin"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[profile.release]
panic = "abort"
"#,
    );
    write_file(
        dir,
        "plugin.toml",
        r#"schema_version = 1

[plugin]
name = "my-plugin"
version = "0.1.0"

[requires]
aiperf_sdk = ">=0.13.0"
target = "x86_64-unknown-linux-gnu"
"#,
    );
    std::fs::create_dir_all(dir.join("src")).expect("create src dir");
    write_file(dir.join("src").as_path(), "lib.rs", "// empty plugin\n");
}

#[test]
fn rejects_missing_cdylib_crate_type() {
    let td = temp_dir();
    let dir = td.path();
    write_file(
        dir,
        "Cargo.toml",
        r#"[package]
name = "bad-plugin"
version = "0.1.0"
edition = "2024"
"#,
    );
    write_file(
        dir,
        "plugin.toml",
        "schema_version = 1\n[plugin]\nname = \"bad\"\nversion = \"0.1.0\"\n[requires]\naiperf_sdk = \">=0.13.0\"\ntarget = \"x86_64-unknown-linux-gnu\"\n",
    );
    let cfg = BuildConfig {
        plugin_dir: dir.to_path_buf(),
        sdk_dir: None,
        release: false,
        target: None,
    };
    let err = build_plugin(&cfg).unwrap_err();
    assert!(
        matches!(err, BuildError::MissingCdylibCrateType),
        "expected MissingCdylibCrateType, got {err:?}"
    );
}

#[test]
fn rejects_missing_panic_abort() {
    let td = temp_dir();
    let dir = td.path();
    write_file(
        dir,
        "Cargo.toml",
        r#"[package]
name = "bad-plugin"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]
"#,
    );
    write_file(
        dir,
        "plugin.toml",
        "schema_version = 1\n[plugin]\nname = \"bad\"\nversion = \"0.1.0\"\n[requires]\naiperf_sdk = \">=0.13.0\"\ntarget = \"x86_64-unknown-linux-gnu\"\n",
    );
    let cfg = BuildConfig {
        plugin_dir: dir.to_path_buf(),
        sdk_dir: None,
        release: true,
        target: None,
    };
    let err = build_plugin(&cfg).unwrap_err();
    assert!(
        matches!(err, BuildError::MissingPanicAbort),
        "expected MissingPanicAbort, got {err:?}"
    );
}

#[test]
fn rejects_missing_manifest() {
    let td = temp_dir();
    let dir = td.path();
    write_file(
        dir,
        "Cargo.toml",
        r#"[package]
name = "bad-plugin"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib"]

[profile.release]
panic = "abort"
"#,
    );
    // No plugin.toml
    let cfg = BuildConfig {
        plugin_dir: dir.to_path_buf(),
        sdk_dir: None,
        release: false,
        target: None,
    };
    let err = build_plugin(&cfg).unwrap_err();
    assert!(
        matches!(err, BuildError::MissingManifest),
        "expected MissingManifest, got {err:?}"
    );
}

#[test]
fn accepts_valid_plugin_structure() {
    // Verify that a well-formed directory passes preflight validation
    // (we don't actually run cargo build in tests — just preflight).
    let td = temp_dir();
    let dir = td.path();
    make_valid_plugin_dir(dir);
    let cfg = BuildConfig {
        plugin_dir: dir.to_path_buf(),
        sdk_dir: None,
        release: false,
        target: None,
    };
    // preflight_only returns Ok(()) when structure is valid
    aiperf_plugin_sdk::build::preflight_plugin(&cfg).expect("preflight should pass");
}

#[test]
fn build_config_paths() {
    let cfg = BuildConfig {
        plugin_dir: PathBuf::from("/some/plugin"),
        sdk_dir: Some(PathBuf::from("/some/sdk")),
        release: true,
        target: Some("aarch64-unknown-linux-gnu".to_string()),
    };
    assert_eq!(cfg.plugin_dir, PathBuf::from("/some/plugin"));
    assert_eq!(cfg.sdk_dir, Some(PathBuf::from("/some/sdk")));
    assert!(cfg.release);
    assert_eq!(cfg.target.as_deref(), Some("aarch64-unknown-linux-gnu"));
}
