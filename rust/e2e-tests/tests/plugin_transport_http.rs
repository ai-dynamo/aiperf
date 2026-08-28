// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 31: HTTP transport candidate package staging tests.
//!
//! RED — fail until Task 31 GREEN builds the HTTP transport cdylib candidate.
//! Static production HTTP transport remains unchanged until Task 39a.

use std::path::PathBuf;

fn candidate_lib_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIPERF_TRANSPORT_HTTP_PLUGIN_LIB") {
        return PathBuf::from(p);
    }
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    let libname = if cfg!(target_os = "macos") {
        "libaiperf_plugin_transport_http.dylib"
    } else if cfg!(windows) {
        "aiperf_plugin_transport_http.dll"
    } else {
        "libaiperf_plugin_transport_http.so"
    };
    target.join(libname)
}

/// RED: fails until Task 31 GREEN builds the HTTP transport cdylib.
#[test]
fn http_transport_candidate_library_exists() {
    let lib = candidate_lib_path();
    assert!(
        lib.exists(),
        "HTTP transport candidate library not found at {}: \
         build `cargo build -p aiperf-plugin-transport-http --lib` (Task 31 GREEN)",
        lib.display()
    );
}

/// RED: fails until plugins.yaml.in declares `canonical_id: http`.
#[test]
fn http_transport_manifest_declares_http_id() {
    let plugins_yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("plugins")
        .join("transport-http")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in missing for transport-http (Task 31)"
    );
    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("http"),
        "plugins.yaml.in does not declare canonical_id: http (Task 31 pending)"
    );
}

/// Static invariant: production HTTP transport must remain registered.
#[test]
fn static_http_transport_registration_unchanged() {
    let transport_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("runtime")
        .join("src")
        .join("transport")
        .join("http");
    assert!(
        transport_dir.exists(),
        "runtime/src/transport/http is gone — static HTTP transport removed"
    );
}
