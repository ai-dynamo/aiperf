// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 29: Endpoint factory candidate package staging tests.
//!
//! These tests are **RED** — they fail because no SDK-built dynamic endpoint
//! candidate cdylib exists yet. When Task 29 (GREEN) lands:
//! - `aiperf-plugin-endpoints` builds as a `cdylib`.
//! - The candidate registers atomic endpoint capabilities with canonical IDs,
//!   `EndpointFactory`, optional companion bindings, and effective aliases.
//! - The static production endpoint authority remains unchanged until Task 39.

use std::path::PathBuf;

/// Path where the endpoint plugin candidate library is expected.
fn candidate_lib_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIPERF_ENDPOINTS_PLUGIN_LIB") {
        return PathBuf::from(p);
    }
    let manifest = env!("CARGO_MANIFEST_DIR");
    let target_dir = PathBuf::from(manifest)
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    let libname = if cfg!(target_os = "macos") {
        "libaiperf_plugin_endpoints.dylib"
    } else if cfg!(windows) {
        "aiperf_plugin_endpoints.dll"
    } else {
        "libaiperf_plugin_endpoints.so"
    };
    target_dir.join(libname)
}

/// RED: The endpoint candidate cdylib must exist before the plugin can be loaded.
///
/// Fails until Task 29 GREEN builds the candidate library.
#[test]
fn endpoint_plugin_candidate_library_exists() {
    let lib = candidate_lib_path();
    assert!(
        lib.exists(),
        "Endpoint plugin candidate library not found at {}: build with \
         `cargo build -p aiperf-plugin-endpoints --lib` (Task 29 GREEN)",
        lib.display()
    );
}

/// RED: The candidate library must export `aiperf_plugin_entry_v1`.
#[test]
#[cfg(unix)]
fn endpoint_plugin_entry_point_exported() {
    let lib = candidate_lib_path();
    if !lib.exists() {
        return;
    }
    let output = std::process::Command::new("nm")
        .arg("-D")
        .arg("--defined-only")
        .arg(&lib)
        .output()
        .expect("nm must be available");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let symbol = "aiperf_plugin_entry_v1";
    let found = stdout
        .lines()
        .any(|line| line.split_ascii_whitespace().last().is_some_and(|s| s == symbol));
    assert!(
        found,
        "symbol `{symbol}` not exported from {}",
        lib.display()
    );
}

/// RED: The candidate `plugins.yaml.in` must declare endpoint capabilities.
///
/// Fails until Task 29 authors the manifest with canonical IDs and aliases.
#[test]
fn endpoint_plugin_manifest_declares_capabilities() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let plugins_yaml = manifest_dir
        .parent()
        .unwrap()
        .join("plugins")
        .join("endpoints")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in not found at {}: Task 29 must author it",
        plugins_yaml.display()
    );

    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    // Must declare at least one endpoint canonical_id and category: endpoint.
    assert!(
        content.contains("canonical_id") && content.contains("endpoint"),
        "plugins.yaml.in at {} does not declare endpoint canonical_id",
        plugins_yaml.display()
    );
}

/// Static invariant: production endpoint registration must remain present and
/// unchanged. The candidate is additive; it must not remove any static ID.
///
/// This test verifies that the built-in endpoint IDs are still referenced in
/// the static registry after Task 29 stages the candidate copy.
#[test]
fn static_endpoint_registration_unchanged() {
    // Scan for static endpoint type registrations in the runtime source.
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("runtime")
        .join("src")
        .join("endpoints");

    assert!(
        src_dir.exists(),
        "runtime/src/endpoints must exist; static production path is gone"
    );

    // The static registry must still have at least the chat endpoint entry.
    let found_chat = walkdir_contains(&src_dir, "Chat")
        || walkdir_contains(
            &PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("runtime")
                .join("src"),
            "EndpointType::Chat",
        );
    assert!(
        found_chat,
        "Static `EndpointType::Chat` registration is gone — Task 29 must not remove static IDs"
    );
}

fn walkdir_contains(dir: &PathBuf, pattern: &str) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rs") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if content.contains(pattern) {
                    return true;
                }
            }
        } else if path.is_dir() {
            if walkdir_contains(&path, pattern) {
                return true;
            }
        }
    }
    false
}
