// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CLI build script: links the shared allocator provider at build time.
//!
//! The `aiperf` binary uses `aiperf_allocator_shim::MiMallocShim` as its
//! global allocator.  The shim's `mi_*` `extern "C"` references are satisfied
//! by `libaiperf_alloc_v1.so`, which must be present in the target directory
//! before the CLI binary is linked.
//!
//! Building the provider first (via `cargo build -p aiperf-allocator-provider`)
//! produces the cdylib in the workspace's target directory; this build script
//! then locates it through `OUT_DIR` ancestry.

use std::env;
use std::path::PathBuf;

fn main() {
    // The provider cdylib lives in target/<profile>/.  OUT_DIR is
    // target/<profile>/build/<crate-hash>/out; ancestor #4 is target/.
    // Always point to the debug profile's directory: the provider is built in
    // debug mode for local development, and the CLI links it from there.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let profile_dir = if let Some(target_root) = out_dir.ancestors().nth(4) {
        target_root.join("debug")
    } else {
        panic!("OUT_DIR ancestry did not yield a workspace target directory");
    };

    // Link the provider cdylib and add an rpath so the dynamic linker finds
    // it at runtime without requiring LD_LIBRARY_PATH.
    println!("cargo:rustc-link-lib=dylib=aiperf_alloc_v1");
    println!("cargo:rustc-link-search=native={}", profile_dir.display());

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" | "android" => {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                profile_dir.display()
            );
        }
        "macos" | "ios" => {
            println!(
                "cargo:rustc-link-arg=-Wl,-rpath,{}",
                profile_dir.display()
            );
        }
        _ => {
            // Windows: the provider DLL must be in the same directory as the
            // executable or on the PATH; rpath is not supported.
        }
    }

    // Rerun only if the provider itself changes.
    let provider_toml = out_dir
        .ancestors()
        .nth(4)
        .map(|t| t.join("debug").join("libaiperf_alloc_v1.so"))
        .unwrap_or_default();
    if provider_toml.exists() {
        println!("cargo:rerun-if-changed={}", provider_toml.display());
    }
    // Also rerun if the provider source changes.
    println!(
        "cargo:rerun-if-changed={}",
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
            .parent()
            .unwrap()
            .join("allocator-provider/Cargo.toml")
            .display()
    );
}
