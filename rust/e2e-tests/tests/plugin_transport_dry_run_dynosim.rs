// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 34: Dry-run and Dynosim transport candidate package staging tests.
//!
//! RED — fail until Task 34 GREEN builds both cdylib candidates.
//! Static production dry-run and Dynosim transports remain unchanged until Task 39a.

use std::path::PathBuf;

fn dry_run_lib_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIPERF_TRANSPORT_DRY_RUN_PLUGIN_LIB") {
        return PathBuf::from(p);
    }
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    let libname = if cfg!(target_os = "macos") {
        "libaiperf_plugin_transport_dry_run.dylib"
    } else if cfg!(windows) {
        "aiperf_plugin_transport_dry_run.dll"
    } else {
        "libaiperf_plugin_transport_dry_run.so"
    };
    target.join(libname)
}

fn dynosim_lib_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIPERF_TRANSPORT_DYNOSIM_PLUGIN_LIB") {
        return PathBuf::from(p);
    }
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    let libname = if cfg!(target_os = "macos") {
        "libaiperf_plugin_transport_dynosim.dylib"
    } else if cfg!(windows) {
        "aiperf_plugin_transport_dynosim.dll"
    } else {
        "libaiperf_plugin_transport_dynosim.so"
    };
    target.join(libname)
}

/// RED: fails until Task 34 GREEN builds the dry-run transport cdylib.
#[test]
fn dry_run_transport_candidate_library_exists() {
    let lib = dry_run_lib_path();
    assert!(
        lib.exists(),
        "Dry-run transport candidate library not found at {}: \
         build `cargo build -p aiperf-plugin-transport-dry-run --lib` (Task 34 GREEN)",
        lib.display()
    );
}

/// RED: fails until Task 34 GREEN builds the Dynosim transport cdylib.
///
/// NOTE: Requires `--features dynosim`.
#[test]
fn dynosim_transport_candidate_library_exists() {
    let lib = dynosim_lib_path();
    assert!(
        lib.exists(),
        "Dynosim transport candidate library not found at {}: \
         build `cargo build -p aiperf-plugin-transport-dynosim --lib --features dynosim` (Task 34 GREEN)",
        lib.display()
    );
}

/// RED: fails until plugins.yaml.in declares `canonical_id: dry_run`.
#[test]
fn dry_run_manifest_declares_dry_run_id() {
    let plugins_yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("plugins")
        .join("transport-dry-run")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in missing for transport-dry-run (Task 34)"
    );
    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("dry_run"),
        "plugins.yaml.in does not declare canonical_id: dry_run (Task 34 pending)"
    );
}

/// RED: fails until plugins.yaml.in declares `canonical_id: dynosim_offline`.
#[test]
fn dynosim_manifest_declares_dynosim_ids() {
    let plugins_yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("plugins")
        .join("transport-dynosim")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in missing for transport-dynosim (Task 34)"
    );
    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("dynosim"),
        "plugins.yaml.in does not declare canonical_id: dynosim_offline (Task 34 pending)"
    );
}
