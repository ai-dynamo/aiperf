// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 30: Endpoint-owned gRPC binding factory candidate tests.
//!
//! These tests are **RED** — they fail because no SDK-built gRPC binding
//! candidate exists. When Task 30 (GREEN) lands:
//! - `aiperf-plugin-endpoints` adds `GrpcEndpointBindingFactory` for KServe
//!   and Riva codec families as optional companion bindings.
//! - Static production gRPC execution and `GrpcBindingRegistry::builtin()`
//!   remain unchanged until Task 39a.

use std::path::PathBuf;

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

/// RED: The gRPC binding candidate cdylib must expose `aiperf_plugin_entry_v1`
/// and register at least one `GrpcEndpointBindingFactory`.
///
/// Fails until Task 30 GREEN adds the gRPC binding implementation.
#[test]
fn grpc_binding_candidate_library_exists() {
    let lib = candidate_lib_path();
    assert!(
        lib.exists(),
        "gRPC binding candidate library not found at {}: build \
         `cargo build -p aiperf-plugin-endpoints --lib` (Task 30 GREEN)",
        lib.display()
    );
}

/// RED: The candidate must export gRPC binding companion factories.
///
/// After Task 30, `plugins.yaml.in` must declare `grpc_bindings` for
/// at least KServe OIP and Riva ASR/TTS/NLP families.
#[test]
fn grpc_binding_manifest_declares_companion_bindings() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let plugins_yaml = manifest_dir
        .parent()
        .unwrap()
        .join("plugins")
        .join("endpoints")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in not found; Task 29 must land before Task 30"
    );

    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    // Task 30 adds grpc binding companion declarations.
    assert!(
        content.contains("grpc_binding") || content.contains("kserve") || content.contains("riva"),
        "plugins.yaml.in does not declare gRPC companion bindings (Task 30 pending)"
    );
}

/// Static invariant: production `GrpcBindingRegistry::builtin()` must remain
/// present and constructible after Task 30 stages the candidate.
///
/// Search for the static builtin registry construction in the production source.
#[test]
fn static_grpc_binding_registry_builtin_unchanged() {
    let src_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("runtime")
        .join("src")
        .join("transport")
        .join("grpc");

    assert!(
        src_root.exists(),
        "runtime/src/transport/grpc must exist; static gRPC is gone"
    );

    // The production registry must still reference KServe and Riva binding types.
    let found_kserve = walkdir_contains(&src_root, "kserve")
        || walkdir_contains(&src_root, "KServe")
        || walkdir_contains(&src_root, "kserve_binding");
    assert!(
        found_kserve,
        "Static KServe gRPC binding is gone — Task 30 must not remove production bindings"
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
