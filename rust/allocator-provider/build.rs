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

    // Compile a C translation unit with a constructor function that references
    // every exported mi_* symbol.  `__attribute__((constructor))` places the
    // function in `.init_array`, which is a linker GC root — so the constructor
    // survives --gc-sections, and its references to mi_* force those symbols
    // from libmimalloc.a into the link.
    //
    // We compile it with cargo_metadata(false) so cc does NOT emit
    // cargo:rustc-link-lib for this archive, letting us emit
    // static:+whole-archive=<name> ourselves.  +whole-archive ensures every
    // object in the archive is force-included rather than being dead-stripped.
    let glue_src = write_glue_c(&out_dir, &target_os);
    let archive_name = "aiperf_alloc_pull";

    cc::Build::new()
        .cargo_metadata(false)
        .file(&glue_src)
        .compile(archive_name);

    // Link the compiled glue archive with +whole-archive so the constructor
    // object is included unconditionally.
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive={archive_name}");

    match target_os.as_str() {
        "linux" | "android" => export_linux(&out_dir),
        "macos" | "ios" => export_macos(&out_dir),
        "windows" => export_windows(&out_dir),
        other => {
            println!("cargo:warning=aiperf-allocator-provider: unknown target_os={other}; skipping symbol export");
        }
    }

    // Emit the cdylib filename pattern so that conformance tests can locate
    // the built artifact without hard-coding platform-specific names.
    let cdylib_name = cdylib_filename(&target_os);
    println!("cargo:CDYLIB_NAME={cdylib_name}");

    // Propagate the mimalloc include directory for any C bridge consumers.
    if let Ok(include) = env::var("DEP_LIBMIMALLOC_SYS_INCLUDE_DIR") {
        println!("cargo:MIMALLOC_INCLUDE_DIR={include}");
    }
}

/// Writes the C glue file and returns its path.
///
/// The file defines a `__attribute__((constructor))` function that takes the
/// address of every boundary symbol.  Constructor functions reside in
/// `.init_array`, which the linker treats as a GC root, so they survive
/// `--gc-sections`.  Their references to `mi_*` force those symbols out of
/// `libmimalloc.a` before GC runs, keeping them in the final output.
fn write_glue_c(out_dir: &PathBuf, target_os: &str) -> PathBuf {
    let glue_path = out_dir.join("aiperf_alloc_pull.c");
    let mut src = String::new();

    src.push_str("/* Auto-generated: forces mi_* inclusion as GC-root via constructor. */\n");
    src.push_str("#include <stddef.h>\n\n");

    // Forward-declare every boundary symbol without including the full header.
    for sym in EXPORTED_SYMBOLS {
        src.push_str(&format!("extern void* {sym}(void);\n"));
    }

    src.push('\n');

    // On Windows, __attribute__((constructor)) is not portable.
    // Use a pragma init_seg or DllMain equivalent if needed.
    // For now, use a pragma-based approach on MSVC and the GCC attribute elsewhere.
    if target_os == "windows" {
        // On Windows with MSVC, use #pragma init_seg to run before user code.
        // This places the references in a CRT init section.
        src.push_str("#pragma section(\".CRT$XCU\", read)\n");
        src.push_str("static void _aiperf_alloc_pull_fn(void);\n");
        src.push_str("__declspec(allocate(\".CRT$XCU\"))\n");
        src.push_str("static void (*_aiperf_init_ptr)(void) = _aiperf_alloc_pull_fn;\n");
        src.push_str("static void _aiperf_alloc_pull_fn(void) {\n");
    } else {
        src.push_str("__attribute__((constructor, visibility(\"hidden\")))\n");
        src.push_str("static void _aiperf_alloc_pull_fn(void) {\n");
    }

    src.push_str("    /* volatile prevents the optimizer from removing these references. */\n");
    src.push_str("    volatile const void* _refs[] = {\n");
    for sym in EXPORTED_SYMBOLS {
        src.push_str(&format!("        (const void*){sym},\n"));
    }
    src.push_str("    };\n");
    src.push_str("    (void)_refs;\n");
    src.push_str("}\n");

    std::fs::write(&glue_path, &src).expect("failed to write aiperf_alloc_pull.c");
    glue_path
}

fn export_linux(out_dir: &PathBuf) {
    // Write a version script that exports exactly the required symbols and
    // hides everything else.  This gives the cdylib a minimal, stable ABI.
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

fn cdylib_filename(target_os: &str) -> String {
    match target_os {
        "windows" => "aiperf_alloc_v1.dll".to_string(),
        "macos" | "ios" => "libaiperf_alloc_v1.dylib".to_string(),
        _ => "libaiperf_alloc_v1.so".to_string(),
    }
}
