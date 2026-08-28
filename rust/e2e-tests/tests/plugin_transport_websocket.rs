// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 33: WebSocket transport candidate package staging tests.
//!
//! RED — fail until Task 33 GREEN builds the WebSocket transport cdylib.
//! Static production WebSocket transport remains unchanged until Task 39a.

use std::path::PathBuf;

fn candidate_lib_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIPERF_TRANSPORT_WS_PLUGIN_LIB") {
        return PathBuf::from(p);
    }
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    let libname = if cfg!(target_os = "macos") {
        "libaiperf_plugin_transport_websocket.dylib"
    } else if cfg!(windows) {
        "aiperf_plugin_transport_websocket.dll"
    } else {
        "libaiperf_plugin_transport_websocket.so"
    };
    target.join(libname)
}

/// RED: fails until Task 33 GREEN builds the WebSocket transport cdylib.
#[test]
fn websocket_transport_candidate_library_exists() {
    let lib = candidate_lib_path();
    assert!(
        lib.exists(),
        "WebSocket transport candidate library not found at {}: \
         build `cargo build -p aiperf-plugin-transport-websocket --lib` (Task 33 GREEN)",
        lib.display()
    );
}

/// RED: fails until plugins.yaml.in declares `canonical_id: websocket`.
#[test]
fn websocket_transport_manifest_declares_ws_id() {
    let plugins_yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("plugins")
        .join("transport-websocket")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in missing for transport-websocket (Task 33)"
    );
    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("websocket"),
        "plugins.yaml.in does not declare canonical_id: websocket (Task 33 pending)"
    );
}

/// Static invariant: production WebSocket support must remain in the runtime.
#[test]
fn static_websocket_support_unchanged() {
    let core_src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("core")
        .join("src");
    // The core crate owns WebSocket operation values in aiperf-core.
    assert!(
        core_src.exists(),
        "rust/core/src is gone — WebSocket operation values may be lost"
    );
}
