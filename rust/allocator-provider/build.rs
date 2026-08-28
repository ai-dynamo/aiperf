// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Build script for the allocator provider.
//!
//! When building the cdylib, exports the required `mi_*` symbols from the
//! statically linked mimalloc so that host and plugin processes share one
//! allocator instance through the provider shared library.
//!
//! Also compiles `src/options.c`, which contains the
//! `mi_aiperf_option_purge_delay()` helper that returns the
//! `mi_option_purge_delay` enum index from the exact linked header.

use std::env;
use std::io::Write;
use std::path::PathBuf;

/// Symbols that must be exported from the provider cdylib.
///
/// These are the symbols that `aiperf-allocator-shim` and the CLI import
/// directly.  The list is the canonical boundary: add here only, never remove.
///
/// `mi_option_set` and `mi_aiperf_option_purge_delay` are added to support
/// the CLI's execute-mode purge-delay adjustment without re-linking mimalloc.
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
    // Runtime option mutation (used by the CLI execute-mode to disable purge).
    "mi_option_set",
    // Returns the `mi_option_purge_delay` index from the provider's header;
    // defined in src/options.c so consumers never duplicate the numeric value.
    "mi_aiperf_option_purge_delay",
];

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Compile src/options.c: the mi_aiperf_option_purge_delay() helper.
    // The mimalloc include directory is propagated by libmimalloc-sys through
    // the DEP_MIMALLOC_INCLUDE_DIR metadata variable (set via its
    // `links = "mimalloc"` key).
    //
    // cargo_metadata(false) suppresses cc's automatic `cargo:rustc-link-lib`
    // emission so we can link with `+whole-archive` below.  Without
    // whole-archive, the linker GC's mi_aiperf_option_purge_delay because no
    // Rust code references it, even though it appears in the version-script
    // global: section.
    let include_dir = env::var_os("DEP_MIMALLOC_INCLUDE_DIR")
        .map(PathBuf::from)
        .expect("libmimalloc-sys did not expose its compiled header directory");
    println!("cargo:rerun-if-changed=src/options.c");
    cc::Build::new()
        .include(&include_dir)
        .file("src/options.c")
        .warnings_into_errors(true)
        .cargo_metadata(false)
        .compile("aiperf_alloc_options");
    // Link with whole-archive so the GC root from the version script can reach it.
    println!("cargo:rustc-link-lib=static:+whole-archive=aiperf_alloc_options");
    println!("cargo:rustc-link-search=native={}", out_dir.display());

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
    // Use an anonymous version-script block (no version name) to control the
    // dynamic symbol table.  The key design constraints:
    //
    // 1. Rust 1.76+ passes --exclude-libs=ALL when linking cdylibs, which
    //    hides all symbols from static archives (including our whole-archive
    //    mimalloc) from the dynamic symbol table.  A version-script's global:
    //    clause overrides --exclude-libs=ALL; --dynamic-list does not.
    //
    // 2. Rust 1.76+ also injects an anonymous version tag into cdylib outputs
    //    for its own ABI stability machinery.  A NAMED version-script block
    //    combined with that anonymous tag triggers:
    //      ld: anonymous version tag cannot be combined with other version tags
    //
    // An anonymous version block { global: ...; local: *; } avoids both
    // problems: it overrides --exclude-libs=ALL through the global: clause,
    // and it does not introduce a named tag that conflicts with Rust's own
    // anonymous tag.  The trade-off is that exported symbols carry no version
    // suffix (@@AIPERF_ALLOC_V1) in nm output; this is acceptable because the
    // provider is a process-local singleton and not an installed multi-version
    // shared library.  The global: entries also act as GC roots under
    // --gc-sections, retaining the mimalloc code that implements them while
    // stripping unreachable internal helpers.
    let script_path = out_dir.join("aiperf_alloc_v1.map");
    let mut script = std::fs::File::create(&script_path).unwrap();

    writeln!(script, "{{").unwrap();
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
    // GNU ld (≥2.35) rejects two anonymous version-script blocks with
    // "anonymous version tag cannot be combined with other version tags".
    // Rust 1.76+ injects its own anonymous block; ours is a second.
    // LLD (available on Linux ≥ Debian 11) merges anonymous blocks without
    // error.  Force LLD so the version-script global: clause can override
    // --exclude-libs=ALL on both old and new GNU toolchains.
    println!("cargo:rustc-link-arg=-fuse-ld=lld");
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
