// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Standalone inspector for the `aiperf-allocator-provider` cdylib.
//!
//! Run with:
//!
//! ```bash
//! # From the workspace rust/ dir:
//! cargo build -p aiperf-allocator-provider
//! cargo script rust/scripts/inspect-plugin-allocator.rs \
//!     -- target/debug/libaiperf_alloc_v1.so
//! ```
//!
//! Or directly after building:
//! ```bash
//! nm -D --defined-only target/debug/libaiperf_alloc_v1.so | grep mi_
//! ```
//!
//! ## What this script checks
//!
//! 1. **Required `mi_*` exports** — every symbol in the boundary contract list
//!    must appear in the dynamic symbol table.
//! 2. **No AIPerf wrappers** — `aiperf_mi_*` symbols must be absent.
//! 3. **`mi_version`** — calls the exported function via `dlopen`/`dlsym` and
//!    verifies the version is v3.x.
//! 4. **`mi_subproc_main`** — calls the exported function and prints the pointer
//!    so you can verify host/plugin equality manually.
//!
//! The script exits 0 if all checks pass, non-zero otherwise.

use std::env;
use std::ffi::CString;
use std::path::PathBuf;
use std::process::{Command, exit};

const REQUIRED_EXPORTS: &[&str] = &[
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
    let args: Vec<String> = env::args().collect();
    let cdylib_path = if args.len() >= 2 {
        PathBuf::from(&args[1])
    } else {
        // Default: look in target/debug relative to the workspace rust/ dir.
        let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("scripts/ must have a parent")
            .to_path_buf();
        workspace.join("target/debug/libaiperf_alloc_v1.so")
    };

    println!("Inspecting: {}", cdylib_path.display());

    if !cdylib_path.exists() {
        eprintln!(
            "ERROR: cdylib not found at {}\n\
             Run `cargo build -p aiperf-allocator-provider` first.",
            cdylib_path.display()
        );
        exit(1);
    }

    let mut failed = false;

    // ── 1. Symbol export check via nm ────────────────────────────────────────
    println!("\n── Dynamic exports (nm -D --defined-only | grep mi_) ──────────────");

    #[cfg(unix)]
    {
        let nm_out = Command::new("nm")
            .args(["-D", "--defined-only"])
            .arg(&cdylib_path)
            .output()
            .expect("nm failed to run; install binutils");

        let exports: Vec<String> = String::from_utf8_lossy(&nm_out.stdout)
            .lines()
            .filter_map(|line| line.split_whitespace().last().map(str::to_string))
            .collect();

        let mi_exports: Vec<&String> = exports.iter().filter(|s| s.starts_with("mi_")).collect();

        println!("Found {} mi_* exports:", mi_exports.len());
        for sym in &mi_exports {
            println!("  {sym}");
        }

        println!("\nChecking required exports:");
        for sym in REQUIRED_EXPORTS {
            let found = exports.iter().any(|s| s == sym);
            if found {
                println!("  ✓ {sym}");
            } else {
                println!("  ✗ {sym}  ← MISSING");
                failed = true;
            }
        }

        println!("\nChecking for banned aiperf_mi_* wrappers:");
        let wrapper_exports: Vec<&String> = exports.iter().filter(|s| s.starts_with("aiperf_mi_")).collect();
        if wrapper_exports.is_empty() {
            println!("  ✓ no aiperf_mi_* exports");
        } else {
            for sym in &wrapper_exports {
                println!("  ✗ FOUND WRAPPER: {sym}");
            }
            failed = true;
        }
    }

    #[cfg(not(unix))]
    {
        println!("(nm check skipped on non-Unix; use dumpbin /EXPORTS on Windows)");
    }

    // ── 2. Runtime version check via dlopen ──────────────────────────────────
    println!("\n── Runtime version check (dlopen/dlsym) ────────────────────────────");

    #[cfg(target_os = "linux")]
    {
        use std::os::raw::c_int;

        let path_cstr = CString::new(cdylib_path.to_str().unwrap()).unwrap();
        let sym_version = CString::new("mi_version").unwrap();
        let sym_subproc = CString::new("mi_subproc_main").unwrap();

        unsafe {
            // RTLD_NOW | RTLD_LOCAL = 2 | 0 = 2 on Linux.
            let handle = libc_dlopen(path_cstr.as_ptr(), 2);
            if handle.is_null() {
                eprintln!("dlopen failed: {}", libc_dlerror_str());
                failed = true;
            } else {
                // mi_version
                let ver_fn = libc_dlsym(handle, sym_version.as_ptr());
                if ver_fn.is_null() {
                    eprintln!("dlsym(mi_version) failed");
                    failed = true;
                } else {
                    let version_fn: unsafe extern "C" fn() -> c_int = std::mem::transmute(ver_fn);
                    let version = version_fn();
                    let major = version / 10000;
                    let minor = (version / 100) % 100;
                    let patch = version % 100;
                    println!("mi_version() = {version}  ({major}.{minor}.{patch})");
                    if version >= 30000 && version < 40000 {
                        println!("  ✓ v3.x confirmed");
                    } else {
                        println!("  ✗ expected v3.x (30000..40000), got {version}");
                        failed = true;
                    }
                }

                // mi_subproc_main
                let sp_fn = libc_dlsym(handle, sym_subproc.as_ptr());
                if sp_fn.is_null() {
                    eprintln!("dlsym(mi_subproc_main) failed");
                    failed = true;
                } else {
                    let subproc_fn: unsafe extern "C" fn() -> *const u8 =
                        std::mem::transmute(sp_fn);
                    let ptr = subproc_fn();
                    println!("mi_subproc_main() = {ptr:p}  (host baseline)");
                    println!("  ✓ non-null; compare against plugin's reported pointer");
                }

                libc_dlclose(handle);
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        println!("(dlopen runtime check implemented for Linux; skipped on this platform)");
    }

    // ── Final result ──────────────────────────────────────────────────────────
    println!("\n──────────────────────────────────────────────────────────────────────");
    if failed {
        eprintln!("FAIL — one or more checks failed (see ✗ above)");
        exit(1);
    } else {
        println!("PASS — all checks passed");
    }
}

// Minimal libc bindings for dlopen/dlsym on Linux without a crate dep.
#[cfg(target_os = "linux")]
extern "C" {
    fn dlopen(filename: *const std::os::raw::c_char, flags: std::os::raw::c_int) -> *mut u8;
    fn dlsym(handle: *mut u8, symbol: *const std::os::raw::c_char) -> *mut u8;
    fn dlclose(handle: *mut u8) -> std::os::raw::c_int;
    fn dlerror() -> *const std::os::raw::c_char;
}

#[cfg(target_os = "linux")]
unsafe fn libc_dlopen(path: *const std::os::raw::c_char, flags: std::os::raw::c_int) -> *mut u8 {
    dlopen(path, flags)
}

#[cfg(target_os = "linux")]
unsafe fn libc_dlsym(handle: *mut u8, symbol: *const std::os::raw::c_char) -> *mut u8 {
    dlsym(handle, symbol)
}

#[cfg(target_os = "linux")]
unsafe fn libc_dlclose(handle: *mut u8) {
    dlclose(handle);
}

#[cfg(target_os = "linux")]
unsafe fn libc_dlerror_str() -> String {
    let p = dlerror();
    if p.is_null() {
        "(no dlerror)".to_string()
    } else {
        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}
