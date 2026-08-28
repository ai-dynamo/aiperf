// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 32: gRPC transport candidate package staging tests.
//!
//! RED — fail until Task 32 GREEN builds the gRPC transport cdylib candidate.
//! Static production gRPC transport remains unchanged until Task 39a.

use std::path::PathBuf;

fn candidate_lib_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIPERF_TRANSPORT_GRPC_PLUGIN_LIB") {
        return PathBuf::from(p);
    }
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    let libname = if cfg!(target_os = "macos") {
        "libaiperf_plugin_transport_grpc.dylib"
    } else if cfg!(windows) {
        "aiperf_plugin_transport_grpc.dll"
    } else {
        "libaiperf_plugin_transport_grpc.so"
    };
    target.join(libname)
}

/// RED: fails until Task 32 GREEN builds the gRPC transport cdylib.
#[test]
fn grpc_transport_candidate_library_exists() {
    let lib = candidate_lib_path();
    assert!(
        lib.exists(),
        "gRPC transport candidate library not found at {}: \
         build `cargo build -p aiperf-plugin-transport-grpc --lib` (Task 32 GREEN)",
        lib.display()
    );
}

/// RED: fails until plugins.yaml.in declares `canonical_id: grpc`.
#[test]
fn grpc_transport_manifest_declares_grpc_id() {
    let plugins_yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("plugins")
        .join("transport-grpc")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in missing for transport-grpc (Task 32)"
    );
    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("grpc"),
        "plugins.yaml.in does not declare canonical_id: grpc (Task 32 pending)"
    );
}

/// Static invariant: production gRPC transport must remain registered.
#[test]
fn static_grpc_transport_registration_unchanged() {
    let transport_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("runtime")
        .join("src")
        .join("transport")
        .join("grpc");
    assert!(
        transport_dir.exists(),
        "runtime/src/transport/grpc is gone — static gRPC transport removed"
    );
}
