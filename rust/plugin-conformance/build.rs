// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for aiperf-plugin-conformance.
//!
//! Propagates the provider cdylib name, target-directory path, and the
//! mimalloc native library link directives to integration tests.
//!
//! The mimalloc search path and link flag are emitted here because Cargo
//! does not propagate `cargo:rustc-link-lib` from dev-dependency build
//! scripts into integration test binary link commands; only the search paths
//! (`cargo:rustc-link-search`) propagate through that channel.

use std::env;
use std::path::PathBuf;

fn main() {
    // DEP_* metadata is only propagated for [dependencies], not [dev-dependencies].
    // The provider cdylib filename is deterministic from the target OS, so we
    // derive it here rather than relying on the DEP_ mechanism.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cdylib_name = match target_os.as_str() {
        "macos" | "ios" => "libaiperf_alloc_v1.dylib",
        "windows" => "aiperf_alloc_v1.dll",
        _ => "libaiperf_alloc_v1.so",
    };
    println!("cargo:rustc-env=AIPERF_ALLOC_V1_CDYLIB_NAME={cdylib_name}");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        println!(
            "cargo:rustc-env=AIPERF_CARGO_PROFILE_TARGET_DIR={}",
            profile_dir.display()
        );

        // Emit the mimalloc search path so the #[link(name = "mimalloc")]
        // attribute in allocator.rs can find libmimalloc.a at link time.
        // Cargo propagates cargo:rustc-link-search to all targets (including
        // integration tests) but suppresses cargo:rustc-link-lib through the
        // links = "mimalloc" deduplication.  The #[link] attribute in the test
        // source bypasses that propagation path entirely.
        //
        // libmimalloc-sys is in [dependencies] so its build script runs before
        // this one, guaranteeing libmimalloc.a exists in the profile build dir
        // when we scan.
        if let Ok(entries) = std::fs::read_dir(profile_dir.join("build")) {
            for entry in entries.flatten() {
                if entry.file_name().to_string_lossy().starts_with("libmimalloc-sys-") {
                    let lib_a = entry.path().join("out").join("libmimalloc.a");
                    if lib_a.exists() {
                        println!(
                            "cargo:rustc-link-search=native={}",
                            entry.path().join("out").display()
                        );
                        break;
                    }
                }
            }
        }
    }

    // Rerun if the provider Cargo.toml changes (new exports, version bump).
    println!(
        "cargo:rerun-if-changed={}",
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .parent()
            .unwrap()
            .join("allocator-provider/Cargo.toml")
            .display()
    );
}
