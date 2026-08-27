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

    // Compile a C glue object that takes the address of every exported mi_*
    // symbol.  This prevents --gc-sections from removing them from the cdylib
    // even though no Rust code references them directly.
    //
    // libmimalloc-sys exposes its include directory as DEP_LIBMIMALLOC_SYS_INCLUDE_DIR.
    // If that env var is unavailable, fall back to a forward-declaration approach
    // that doesn't need the header.
    let glue_src = write_glue_c(&out_dir);
    let mut build = cc::Build::new();
    build.file(&glue_src);
    if let Ok(inc) = env::var("DEP_LIBMIMALLOC_SYS_INCLUDE_DIR") {
        build.include(inc);
    }
    build.compile("aiperf_alloc_pull");

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

/// Writes the C glue file that prevents GC of mimalloc symbols and returns its path.
///
/// The file takes the address of each exported symbol into a hidden static array.
/// `__attribute__((used))` prevents the compiler from eliminating the array,
/// which in turn keeps the referenced symbols alive through the linker's GC pass.
fn write_glue_c(out_dir: &PathBuf) -> PathBuf {
    let glue_path = out_dir.join("aiperf_alloc_pull.c");
    let mut src = String::new();

    src.push_str("/* Auto-generated: forces mi_* symbol inclusion from libmimalloc.a */\n");
    src.push_str("#include <stddef.h>\n\n");

    // Forward-declare all exported symbols so we can take their address
    // without requiring the full mimalloc header.
    for sym in EXPORTED_SYMBOLS {
        src.push_str(&format!("extern void* {sym}(void);\n"));
    }

    src.push_str("\n");
    src.push_str("#if defined(__GNUC__) || defined(__clang__)\n");
    src.push_str("__attribute__((used, visibility(\"hidden\")))\n");
    src.push_str("#endif\n");
    src.push_str("static const void* _aiperf_alloc_pull[] = {\n");
    for sym in EXPORTED_SYMBOLS {
        src.push_str(&format!("    (const void*){sym},\n"));
    }
    src.push_str("};\n");

    std::fs::write(&glue_path, src).expect("failed to write aiperf_alloc_pull.c");
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
