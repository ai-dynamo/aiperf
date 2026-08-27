// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for the allocator candidate plugin fixture.
//!
//! Same pattern as the host fixture: links `mi_*` dynamically from the
//! provider cdylib so the plugin carries no embedded allocator.

use std::env;

fn main() {
    let provider_dir = env::var("AIPERF_ALLOC_V1_DIR")
        .expect("AIPERF_ALLOC_V1_DIR must be set to the directory containing the provider cdylib");

    println!("cargo:rustc-link-search=native={provider_dir}");
    println!("cargo:rustc-link-lib=dylib=aiperf_alloc_v1");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" || target_os == "android" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{provider_dir}");
    } else if target_os == "macos" || target_os == "ios" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{provider_dir}");
    }

    println!("cargo:rerun-if-env-changed=AIPERF_ALLOC_V1_DIR");
}
