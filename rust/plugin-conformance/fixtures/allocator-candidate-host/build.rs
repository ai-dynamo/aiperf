// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for the allocator candidate host fixture.
//!
//! Links against the provider cdylib so that `mi_*` symbols imported by the
//! `aiperf-allocator-shim` are resolved dynamically from the provider, not
//! from a statically linked mimalloc.  The provider directory is passed via
//! `AIPERF_ALLOC_V1_DIR` when the conformance test builds this fixture.

use std::env;

fn main() {
    let provider_dir = env::var("AIPERF_ALLOC_V1_DIR")
        .expect("AIPERF_ALLOC_V1_DIR must be set to the directory containing the provider cdylib");

    // Tell the linker where to find the provider cdylib.
    println!("cargo:rustc-link-search=native={provider_dir}");

    // Link against the provider cdylib dynamically.
    println!("cargo:rustc-link-lib=dylib=aiperf_alloc_v1");

    // Embed the provider directory in the RPATH so the binary can find the
    // cdylib at runtime without relying solely on LD_LIBRARY_PATH.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{provider_dir}");
    } else if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{provider_dir}");
    }
    // Windows: runtime search is via PATH or the DLL co-location rule.

    println!("cargo:rerun-if-env-changed=AIPERF_ALLOC_V1_DIR");
}
