// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for aiperf-plugin-conformance.
//!
//! Propagates the provider cdylib name and target-directory path to
//! integration tests as compile-time environment variables.

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

    // Infer the Cargo target directory from OUT_DIR.
    // OUT_DIR is:  <target_dir>/<profile>/build/<pkg>-<hash>/out
    // We need:     <target_dir>/<profile>/
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    if let Some(profile_dir) = out_dir.ancestors().nth(3) {
        println!(
            "cargo:rustc-env=AIPERF_CARGO_PROFILE_TARGET_DIR={}",
            profile_dir.display()
        );
    }

    // The test binary links aiperf-allocator-shim, which references mi_* symbols
    // as extern "C". Cargo propagates cargo:rustc-link-search from dev-dependency
    // build scripts but NOT cargo:rustc-link-lib. The search path comes from
    // libmimalloc-sys's build script (which IS propagated), so libmimalloc.a is
    // findable; we just need to tell the linker to include it.
    println!("cargo:rustc-link-lib=static=mimalloc");

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
