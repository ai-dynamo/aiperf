// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for aiperf-plugin-conformance.
//!
//! Injects the provider cdylib name and target-directory path into the test
//! binary as compile-time constants.  The mimalloc link search path is
//! propagated automatically by libmimalloc-sys's own dev-dep build script;
//! the `#[link(name = "mimalloc", kind = "static")]` attribute in allocator.rs
//! emits the -lmimalloc flag directly.

use std::env;
use std::path::PathBuf;

fn main() {
    // DEP_* metadata is only propagated for [dependencies], not [dev-dependencies].
    // The provider cdylib filename is deterministic from the target OS.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cdylib_name = match target_os.as_str() {
        "macos" | "ios" => "libaiperf_alloc_v1.dylib",
        "windows" => "aiperf_alloc_v1.dll",
        _ => "libaiperf_alloc_v1.so",
    };
    println!("cargo:rustc-env=AIPERF_ALLOC_V1_CDYLIB_NAME={cdylib_name}");

    // Always point to target/debug: the nested cargo builds for the provider
    // and fixtures always build in debug mode regardless of the outer profile,
    // so the cdylib and fixture binaries always land in target/debug.
    // OUT_DIR is <workspace>/target/<profile>/build/<crate-hash>/out;
    // ancestor #4 is <workspace>/target.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Some(target_dir) = out_dir.ancestors().nth(4) {
        let profile_dir = target_dir.join("debug");
        println!(
            "cargo:rustc-env=AIPERF_CARGO_PROFILE_TARGET_DIR={}",
            profile_dir.display()
        );
    }

    // Inject the target triple for use in integration tests.
    let target = env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=AIPERF_BUILD_TARGET={target}");

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
