// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for the allocator provider.
//!
//! When building the cdylib, exports the required `mi_*` symbols from the
//! statically linked mimalloc so that host and plugin processes share one
//! allocator instance through the provider shared library.

use std::env;
use std::io::Write;
use std::path::PathBuf;

/// Symbols that must be exported from the provider cdylib.
///
/// These are the symbols that `aiperf-allocator-shim` imports directly.
/// The list is the canonical boundary: add here only, never remove.
const EXPORTED_SYMBOLS: &[&str] = &[
    "mi_malloc",
    "mi_zalloc",
    "mi_malloc_aligned",
    "mi_zalloc_aligned",
    "mi_realloc",
    "mi_realloc_aligned",
    "mi_free",
    "mi_free_size",
    "mi_free_aligned",
    "mi_free_size_aligned",
    "mi_subproc_main",
    "mi_version",
    "mi_malloc_size",
    "mi_malloc_good_size",
    "mi_malloc_usable_size",
];

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Cargo does NOT propagate `cargo:rustc-link-lib` from a dependency's
    // build script into a cdylib link command; it only propagates the search
    // paths (`rustc-link-search`).  Emit the mimalloc link directive ourselves
    // so that libmimalloc.a is actually linked into the provider cdylib.
    //
    // We link it with `+whole-archive` so that every object file in the archive
    // is unconditionally included.  The version script below then restricts the
    // dynamic export table to exactly the boundary surface; the exported symbols
    // (listed in `global:`) become live roots for `--gc-sections`, keeping the
    // mimalloc code that implements them while eliminating internal helpers that
    // nothing outside the cdylib will ever call.
    println!("cargo:rustc-link-lib=static:+whole-archive=mimalloc");

    match target_os.as_str() {
        "linux" | "android" => export_linux(&out_dir),
        "macos" | "ios" => export_macos(&out_dir),
        "windows" => export_windows(&out_dir),
        other => {
            println!("cargo:warning=aiperf-allocator-provider: unknown target_os={other}; skipping symbol export");
        }
    }
}

fn export_linux(out_dir: &PathBuf) {
    // Write a version script that exports exactly the required symbols and
    // hides everything else.  The `global:` entries also act as linker GC
    // roots when `--gc-sections` is active, keeping the mimalloc code that
    // implements them while stripping unreachable internal helpers.
    let script_path = out_dir.join("aiperf_alloc_v1.map");
    let mut script = std::fs::File::create(&script_path).unwrap();

    writeln!(script, "AIPERF_ALLOC_V1 {{").unwrap();
    writeln!(script, "    global:").unwrap();
    for sym in EXPORTED_SYMBOLS {
        writeln!(script, "        {sym};").unwrap();
    }
    writeln!(script, "    local: *;").unwrap();
    writeln!(script, "}};").unwrap();

    println!(
        "cargo:rustc-link-arg=-Wl,--version-script={}",
        script_path.display()
    );
}

fn export_macos(out_dir: &PathBuf) {
    // On macOS, write an exported symbols list and pass it to the linker.
    let list_path = out_dir.join("aiperf_alloc_v1.exp");
    let mut list = std::fs::File::create(&list_path).unwrap();

    for sym in EXPORTED_SYMBOLS {
        // macOS exported symbols lists use a leading underscore.
        writeln!(list, "_{sym}").unwrap();
    }

    println!(
        "cargo:rustc-link-arg=-Wl,-exported_symbols_list,{}",
        list_path.display()
    );
}

fn export_windows(out_dir: &PathBuf) {
    // On Windows, write a module-definition file listing all exports.
    let def_path = out_dir.join("aiperf_alloc_v1.def");
    let mut def = std::fs::File::create(&def_path).unwrap();

    writeln!(def, "EXPORTS").unwrap();
    for sym in EXPORTED_SYMBOLS {
        writeln!(def, "    {sym}").unwrap();
    }

    println!("cargo:rustc-link-arg=/DEF:{}", def_path.display());
}
