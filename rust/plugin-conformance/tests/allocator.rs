// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Allocator conformance suite for the shared provider topology.
//!
//! Tests verified by this suite:
//!
//! 1. **Provider exports** — the provider cdylib exports every required `mi_*`
//!    symbol.  No provider-level AIPerf wrappers are present.
//! 2. **Shim imports** — the fixture host binary has dynamic import entries for
//!    every required `mi_*` symbol; none resolve to a static wrapper.
//! 3. **Version** — `mi_version()` returns a v3.x value.
//! 4. **Shared instance** — host and plugin report an identical
//!    `mi_subproc_main()` pointer, confirming one allocator instance.
//! 5. **Allocation operations** — `String`, `Vec`, `Box`, `Arc`, `Rc`, trait
//!    objects, and cross-boundary free/alloc sequences succeed and exit 0.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// Cargo's build-script `cargo:rustc-link-lib` is suppressed by the
// `links = "mimalloc"` deduplication system when forwarding to integration
// test binaries; only `cargo:rustc-link-search` (the -L paths) reaches them.
// The `#[link]` attribute below emits -lmimalloc directly through rustc,
// bypassing Cargo's link-lib propagation path entirely.  The matching
// search path is provided by the conformance build.rs.
#[cfg(unix)]
#[link(name = "mimalloc", kind = "static")]
unsafe extern "C" {}

// ── compile-time constants injected by build.rs ───────────────────────────

/// Platform-specific cdylib filename (e.g. `libaiperf_alloc_v1.so`).
const CDYLIB_NAME: &str = env!("AIPERF_ALLOC_V1_CDYLIB_NAME");

/// Cargo profile target directory (e.g. `.../target/debug`).
const PROFILE_TARGET_DIR: &str = env!("AIPERF_CARGO_PROFILE_TARGET_DIR");

// ── symbols that must be present in the provider and shim ─────────────────

/// Symbols that the provider cdylib must export (visible in dynamic table).
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

/// Symbols that the fixture host must import dynamically (from the provider).
const REQUIRED_IMPORTS: &[&str] = &[
    "mi_malloc",
    "mi_zalloc",
    "mi_malloc_aligned",
    "mi_zalloc_aligned",
    "mi_realloc",
    "mi_realloc_aligned",
    "mi_free",
    "mi_subproc_main",
    "mi_version",
];

// ── path helpers ──────────────────────────────────────────────────────────

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("fixtures")
}

fn workspace_rust_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("plugin-conformance dir must have a parent")
        .to_path_buf()
}

fn provider_cdylib_path() -> PathBuf {
    Path::new(PROFILE_TARGET_DIR).join(CDYLIB_NAME)
}

fn fixture_host_path() -> PathBuf {
    Path::new(PROFILE_TARGET_DIR).join("allocator-candidate-host")
}

fn fixture_plugin_path() -> PathBuf {
    let prefix = if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    };
    let ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    Path::new(PROFILE_TARGET_DIR).join(format!("{prefix}allocator_candidate_plugin.{ext}"))
}

// ── build helpers ──────────────────────────────────────────────────────────

/// Builds the workspace-member package `name` if not already built.
///
/// Uses the same profile as the current test run (debug by default).
static PROVIDER_BUILT: OnceLock<bool> = OnceLock::new();
static FIXTURES_BUILT: OnceLock<bool> = OnceLock::new();

fn ensure_provider_built() {
    PROVIDER_BUILT.get_or_init(|| {
        let status = Command::new(env!("CARGO"))
            .args(["build", "-p", "aiperf-allocator-provider"])
            .current_dir(workspace_rust_dir())
            .status()
            .expect("cargo build aiperf-allocator-provider failed to launch");
        assert!(
            status.success(),
            "cargo build aiperf-allocator-provider failed"
        );
        true
    });
}

fn ensure_fixtures_built() {
    FIXTURES_BUILT.get_or_init(|| {
        ensure_provider_built();
        let provider_dir = Path::new(PROFILE_TARGET_DIR);
        let provider_dir_str = provider_dir.to_str().unwrap();

        // Build the fixture host into the main workspace target dir so that
        // fixture_host_path() finds it under PROFILE_TARGET_DIR.
        let manifest = fixtures_dir().join("allocator-candidate-host/Cargo.toml");
        let status = Command::new(env!("CARGO"))
            .args([
                "build",
                "--manifest-path",
                manifest.to_str().unwrap(),
                "--target-dir",
                provider_dir
                    .parent()
                    .unwrap_or(provider_dir)
                    .to_str()
                    .unwrap(),
            ])
            .env("AIPERF_ALLOC_V1_DIR", provider_dir_str)
            .status()
            .expect("cargo build allocator-candidate-host failed to launch");
        assert!(
            status.success(),
            "cargo build allocator-candidate-host failed"
        );

        // Build the fixture plugin into the main workspace target dir.
        let manifest = fixtures_dir().join("allocator-candidate-plugin/Cargo.toml");
        let status = Command::new(env!("CARGO"))
            .args([
                "build",
                "--manifest-path",
                manifest.to_str().unwrap(),
                "--target-dir",
                provider_dir
                    .parent()
                    .unwrap_or(provider_dir)
                    .to_str()
                    .unwrap(),
            ])
            .env("AIPERF_ALLOC_V1_DIR", provider_dir_str)
            .status()
            .expect("cargo build allocator-candidate-plugin failed to launch");
        assert!(
            status.success(),
            "cargo build allocator-candidate-plugin failed"
        );

        true
    });
}

// ── symbol-table inspection ────────────────────────────────────────────────

/// Returns all dynamic symbol names from the given binary using `nm -D`.
///
/// On Linux, `nm -D` shows the dynamic (DYNSYM) table.  On macOS, `nm -m`
/// approximates this.  On Windows, `dumpbin /EXPORTS` would be used instead;
/// this helper is a no-op there (tests skip with `cfg(unix)`).
#[cfg(unix)]
fn nm_dynamic_symbols(path: &Path) -> Vec<String> {
    let output = Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(path)
        .output()
        .unwrap_or_else(|_| panic!("nm failed on {}", path.display()));
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            // nm output format: "<addr> <type> <name>" or "         U <name>"
            let parts: Vec<&str> = line.split_whitespace().collect();
            // Strip GNU version suffix (e.g. "mi_malloc@@AIPERF_ALLOC_V1" → "mi_malloc").
            parts
                .last()
                .map(|s| s.split('@').next().unwrap_or(s).to_string())
        })
        .collect()
}

/// Returns all dynamic import names from the given binary using `nm -D`.
///
/// Looks for undefined symbols (type `U`), which represent dynamic imports.
#[cfg(unix)]
fn nm_dynamic_imports(path: &Path) -> Vec<String> {
    let output = Command::new("nm")
        .args(["-D", "--undefined-only"])
        .arg(path)
        .output()
        .unwrap_or_else(|_| panic!("nm --undefined-only failed on {}", path.display()));
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            // Strip GNU version suffix (e.g. "mi_malloc@AIPERF_ALLOC_V1" → "mi_malloc").
            parts
                .last()
                .map(|s| s.split('@').next().unwrap_or(s).to_string())
        })
        .collect()
}

// ── tests ──────────────────────────────────────────────────────────────────

/// The provider cdylib must exist and export every required `mi_*` symbol.
#[test]
#[cfg(unix)]
fn provider_exports_required_mi_symbols() {
    ensure_provider_built();

    let cdylib = provider_cdylib_path();
    assert!(
        cdylib.exists(),
        "provider cdylib not found at {}: ensure `cargo build -p aiperf-allocator-provider` succeeds",
        cdylib.display()
    );

    let exports = nm_dynamic_symbols(&cdylib);

    for sym in REQUIRED_EXPORTS {
        assert!(
            exports.iter().any(|s| s == sym),
            "provider cdylib {CDYLIB_NAME} must export `{sym}`; found exports: {:?}",
            exports
                .iter()
                .filter(|s| s.starts_with("mi_"))
                .collect::<Vec<_>>()
        );
    }
}

/// The provider cdylib must not export any AIPerf-named wrapper around `mi_*`.
#[test]
#[cfg(unix)]
fn provider_has_no_aiperf_allocator_wrappers() {
    ensure_provider_built();

    let cdylib = provider_cdylib_path();
    let exports = nm_dynamic_symbols(&cdylib);

    for sym in &exports {
        // "aiperf_mi_" prefix is the banned wrapper namespace.
        assert!(
            !sym.starts_with("aiperf_mi_"),
            "provider cdylib must not export AIPerf wrapper `{sym}`"
        );
    }
}

/// The `MiMallocShim` type compiles and satisfies the `GlobalAlloc` contract.
#[test]
fn shim_compiles_and_satisfies_global_alloc() {
    use aiperf_allocator_shim::MiMallocShim;
    use std::alloc::{GlobalAlloc, Layout};

    let shim = MiMallocShim;

    // Allocate, write, read, reallocate, and free through the shim.
    let layout = Layout::array::<u8>(64).unwrap();
    let ptr = unsafe { shim.alloc(layout) };
    assert!(
        !ptr.is_null(),
        "MiMallocShim::alloc returned null for 64 bytes"
    );

    unsafe { std::ptr::write_bytes(ptr, 0xAB, 64) };

    let new_layout = Layout::array::<u8>(128).unwrap();
    let rptr = unsafe { shim.realloc(ptr, layout, new_layout.size()) };
    assert!(
        !rptr.is_null(),
        "MiMallocShim::realloc returned null for 128 bytes"
    );

    unsafe { shim.dealloc(rptr, new_layout) };

    // Zeroed allocation.
    let zptr = unsafe { shim.alloc_zeroed(layout) };
    assert!(!zptr.is_null(), "MiMallocShim::alloc_zeroed returned null");
    let zeroed = unsafe { std::slice::from_raw_parts(zptr, 64) };
    assert!(
        zeroed.iter().all(|&b| b == 0),
        "alloc_zeroed must return zeroed memory"
    );
    unsafe { shim.dealloc(zptr, layout) };
}

/// `mi_version()` must return a v3.x mimalloc version.
#[test]
fn shim_mi_version_is_v3() {
    let version = unsafe { aiperf_allocator_shim::mi_version() };
    assert!(
        version >= 30000 && version < 40000,
        "mi_version must be v3.x (30000..40000), got {version}"
    );
}

/// The fixture host binary must dynamically import `mi_*` from the provider.
///
/// This confirms eager binding (symbols in the DYNSYM import table) rather
/// than lazy dlopen-based loading.
#[test]
#[cfg(unix)]
fn fixture_host_dynamically_imports_mi_symbols() {
    ensure_fixtures_built();

    let host = fixture_host_path();
    assert!(
        host.exists(),
        "fixture host binary not found at {}",
        host.display()
    );

    let imports = nm_dynamic_imports(&host);

    for sym in REQUIRED_IMPORTS {
        assert!(
            imports.iter().any(|s| s == sym),
            "fixture host must dynamically import `{sym}` from provider; \
             found mi_* imports: {:?}",
            imports
                .iter()
                .filter(|s| s.starts_with("mi_"))
                .collect::<Vec<_>>()
        );
    }
}

/// The fixture host must not import any AIPerf-named `mi_*` wrapper.
#[test]
#[cfg(unix)]
fn fixture_host_has_no_aiperf_allocator_wrapper_imports() {
    ensure_fixtures_built();

    let host = fixture_host_path();
    let imports = nm_dynamic_imports(&host);

    for sym in &imports {
        assert!(
            !sym.starts_with("aiperf_mi_"),
            "fixture host must not import AIPerf wrapper `{sym}`"
        );
    }
}

/// The fixture host subprocess must exit 0, confirming all allocation
/// operations and the shared-instance check pass.
#[test]
#[cfg(unix)]
fn fixture_host_subprocess_exits_zero() {
    ensure_fixtures_built();

    let host = fixture_host_path();
    let plugin = fixture_plugin_path();

    assert!(
        host.exists(),
        "fixture host not found at {}",
        host.display()
    );
    assert!(
        plugin.exists(),
        "fixture plugin not found at {}",
        plugin.display()
    );

    let provider_dir = Path::new(PROFILE_TARGET_DIR);

    // Set LD_LIBRARY_PATH so both the host and the plugin can locate the
    // provider cdylib at runtime.
    let lib_path = {
        let existing = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();
        if existing.is_empty() {
            provider_dir.to_str().unwrap().to_string()
        } else {
            format!("{}:{existing}", provider_dir.display())
        }
    };

    let output = Command::new(&host)
        .arg(plugin.to_str().unwrap())
        .env("LD_LIBRARY_PATH", lib_path)
        .output()
        .expect("fixture host failed to launch");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "fixture host exited with status {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status.code()
    );
}

/// The fixture host and plugin must report the same `mi_subproc_main()` pointer,
/// confirming they share one allocator instance.
///
/// This check is embedded inside the fixture host subprocess (verified via
/// exit status in `fixture_host_subprocess_exits_zero`).  This test documents
/// the requirement explicitly.
#[test]
fn shared_allocator_instance_requirement_is_documented() {
    // The actual runtime check happens in fixture_host_subprocess_exits_zero.
    // The fixture host calls mi_subproc_main() both in the host and via an
    // exported plugin function, and asserts they are equal before exiting 0.
    //
    // If this test is the only failure, fix fixture_host_subprocess_exits_zero.

    // Verify the documented contract at the shim level: two calls to
    // mi_subproc_main() within the same process return the same pointer.
    let a = unsafe { aiperf_allocator_shim::mi_subproc_main() };
    let b = unsafe { aiperf_allocator_shim::mi_subproc_main() };
    assert_eq!(
        a, b,
        "mi_subproc_main() must return the same pointer on repeated calls within one process"
    );
}
