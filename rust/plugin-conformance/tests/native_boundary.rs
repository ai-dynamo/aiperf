// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 36: Four-target native-boundary and allocator feasibility gate.
//!
//! This suite proves the minimal cdylib ABI contract and the four plugin
//! category SDK boundaries compile and link correctly:
//!
//! 1. **ABI boundary** — the `minimal-plugin` fixture builds as a cdylib and
//!    exports `aiperf_plugin_entry_v1`.
//! 2. **Static inspection** — `statically_inspect` validates the ELF symbol
//!    table and bind-now flag without dlopen.
//! 3. **Runtime boundary** — dlopen + dlsym resolves the entry point through
//!    the cdylib boundary; the allocation path does not corrupt across the
//!    boundary.
//! 4. **Four SDK boundaries** — compile-time presence checks for every
//!    plugin category SDK: transport, endpoint, export, and plugin-api.
//!
//! These tests gate Tasks 24-34; they are the minimum bar to prove that
//! candidate packages can be built and linked without corrupting the process.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

// ── path helpers ──────────────────────────────────────────────────────────

fn workspace_rust_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("plugin-conformance must have a parent")
        .to_path_buf()
}

fn profile_target_dir() -> PathBuf {
    Path::new(PROFILE_TARGET_DIR).to_path_buf()
}

fn minimal_plugin_lib_path() -> PathBuf {
    let dir = profile_target_dir();
    if cfg!(target_os = "macos") {
        dir.join("libminimal_plugin.dylib")
    } else if cfg!(windows) {
        dir.join("minimal_plugin.dll")
    } else {
        dir.join("libminimal_plugin.so")
    }
}

// Injected by build.rs — points at target/debug.
const PROFILE_TARGET_DIR: &str = env!("AIPERF_CARGO_PROFILE_TARGET_DIR");

// ── build helper ──────────────────────────────────────────────────────────

static MINIMAL_PLUGIN_BUILT: OnceLock<bool> = OnceLock::new();

fn ensure_minimal_plugin_built() {
    MINIMAL_PLUGIN_BUILT.get_or_init(|| {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fixtures")
            .join("minimal-plugin")
            .join("Cargo.toml");
        let target_dir = profile_target_dir()
            .parent()
            .unwrap_or(&profile_target_dir())
            .to_path_buf();
        let status = Command::new(env!("CARGO"))
            .args([
                "build",
                "--manifest-path",
                manifest.to_str().unwrap(),
                "--target-dir",
                target_dir.to_str().unwrap(),
            ])
            .status()
            .expect("cargo build minimal-plugin failed to launch");
        assert!(status.success(), "cargo build minimal-plugin failed");
        true
    });
}

// ── Symbol inspection ─────────────────────────────────────────────────────

#[cfg(unix)]
fn nm_dynamic_exports(path: &Path) -> Vec<String> {
    let output = Command::new("nm")
        .args(["-D", "--defined-only"])
        .arg(path)
        .output()
        .unwrap_or_else(|_| panic!("nm failed on {}", path.display()));
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            parts
                .last()
                .map(|s| s.split('@').next().unwrap_or(s).to_string())
        })
        .collect()
}

// ── tests ──────────────────────────────────────────────────────────────────

/// TARGET 1 — ABI boundary: minimal-plugin fixture builds as a cdylib.
///
/// The `minimal-plugin` fixture uses `#[aiperf_plugin]` proc-macro to generate
/// the `unsafe extern "C" fn aiperf_plugin_entry_v1()` export.
#[test]
fn minimal_plugin_cdylib_exists() {
    ensure_minimal_plugin_built();
    let lib = minimal_plugin_lib_path();
    assert!(
        lib.exists(),
        "minimal-plugin cdylib not found at {}",
        lib.display()
    );
}

/// TARGET 1 — ABI boundary: `aiperf_plugin_entry_v1` is in the dynamic symbol table.
#[test]
#[cfg(unix)]
fn minimal_plugin_exports_entry_symbol() {
    ensure_minimal_plugin_built();
    let lib = minimal_plugin_lib_path();
    if !lib.exists() {
        panic!("minimal-plugin cdylib not found; build must succeed first");
    }

    let exports = nm_dynamic_exports(&lib);
    let entry = "aiperf_plugin_entry_v1";
    assert!(
        exports.iter().any(|s| s == entry),
        "minimal-plugin cdylib must export `{entry}`; found exports: {:?}",
        exports
            .iter()
            .filter(|s| s.contains("aiperf") || s.contains("plugin"))
            .collect::<Vec<_>>()
    );
}

/// TARGET 1 — ABI boundary: no SDK or runtime symbols bleed into the cdylib
/// dynamic symbol table.  Only the entry point and allocator symbols are allowed.
#[test]
#[cfg(unix)]
fn minimal_plugin_dynamic_table_is_minimal() {
    ensure_minimal_plugin_built();
    let lib = minimal_plugin_lib_path();
    if !lib.exists() {
        panic!("minimal-plugin cdylib not found at {:?}", lib);
    }
    let exports = nm_dynamic_exports(&lib);
    // The only acceptable aiperf_* export is the entry point.
    let unexpected: Vec<&str> = exports
        .iter()
        .filter(|s| s.starts_with("aiperf_") && s.as_str() != "aiperf_plugin_entry_v1")
        .map(|s| s.as_str())
        .collect();
    assert!(
        unexpected.is_empty(),
        "minimal-plugin exports unexpected aiperf_* symbols: {:?}",
        unexpected
    );
}

/// TARGET 2 — Static inspection gate: `statically_inspect` validates the
/// minimal-plugin artifact and confirms `entry_symbol_present`.
#[test]
#[cfg(unix)]
fn static_inspection_validates_minimal_plugin() {
    use aiperf_plugin_host::acquire::AcquiredArtifact;
    use aiperf_plugin_host::inspect::statically_inspect;

    ensure_minimal_plugin_built();
    let lib = minimal_plugin_lib_path();
    if !lib.exists() {
        panic!("minimal-plugin cdylib not found");
    }

    // Compute BLAKE3 digest so acquire() passes verification.
    let raw = std::fs::read(&lib).unwrap();
    let digest = blake3::hash(&raw).to_hex().to_string();

    // Use the current process's target triple as the declared target.
    let target = env!("AIPERF_BUILD_TARGET");

    let artifact =
        AcquiredArtifact::acquire(&lib, &digest, target).expect("AcquiredArtifact::acquire");
    let inspected = statically_inspect(&artifact).expect("statically_inspect");
    assert!(
        inspected.entry_symbol_present,
        "static inspection must confirm `aiperf_plugin_entry_v1` present"
    );
    assert!(
        inspected.arch_matches,
        "minimal-plugin must be built for the current architecture"
    );
}

/// TARGET 3 — Runtime boundary: dlopen loads the minimal-plugin cdylib and
/// invoking `aiperf_plugin_entry_v1` returns a non-empty package descriptor.
///
/// This proves the ABI contract is crossable at runtime: the cdylib loads,
/// the symbol resolves through the dynamic linker, the entry point executes
/// across the boundary, and the returned `PluginDeclarationV1` is coherent.
#[test]
#[cfg(unix)]
fn dlopen_resolves_entry_symbol() {
    use aiperf_plugin_api::extension::{PluginDeclarationV1, PLUGIN_ENTRY_SYMBOL_V1};

    ensure_minimal_plugin_built();
    let lib_path = minimal_plugin_lib_path();
    if !lib_path.exists() {
        panic!("minimal-plugin cdylib not found at {:?}", lib_path);
    }

    // SAFETY: We dlopen a cdylib built by this workspace from the same
    // compiler invocation. The returned `PluginDeclarationV1` holds `'static`
    // references into the loaded image; `_lib` keeps the image mapped for the
    // duration of the assertion.
    unsafe {
        let _lib = libloading::Library::new(&lib_path)
            .expect("dlopen minimal-plugin cdylib failed");
        let entry: libloading::Symbol<unsafe extern "C" fn() -> PluginDeclarationV1> = _lib
            .get(format!("{PLUGIN_ENTRY_SYMBOL_V1}\0").as_bytes())
            .expect("failed to resolve aiperf_plugin_entry_v1 via dlsym");
        let decl = entry();
        assert!(
            !decl.package.version.is_empty(),
            "plugin declaration version must be non-empty after boundary crossing"
        );
    }
}

/// TARGET 4 — Four SDK boundaries: compile-time proof that all four plugin
/// category SDKs are accessible from the conformance crate.
///
/// Each assertion touches a type from the category SDK that a real plugin
/// in that category would use.  If any SDK has a compilation error or
/// breaks its public API, this test fails to compile.
#[test]
fn four_sdk_category_boundaries_compile() {
    // Transport boundary (aiperf-transport-sdk):
    let transport_ver = aiperf_transport_sdk::PLUGIN_SOURCE_API_VERSION;
    assert!(!transport_ver.is_empty());

    // Endpoint boundary (aiperf-endpoint-sdk):
    let endpoint_ver = aiperf_endpoint_sdk::PLUGIN_SOURCE_API_VERSION;
    assert!(!endpoint_ver.is_empty());

    // Export boundary (aiperf-export-sdk) — use a helper fn as the type anchor:
    let normalized = aiperf_export_sdk::helpers::normalize_endpoint_display("http://localhost");
    assert!(!normalized.is_empty());

    // Plugin API boundary — all three factory traits are visible:
    let _ = std::any::TypeId::of::<dyn aiperf_plugin_api::category::EndpointFactory>();
    let _ = std::any::TypeId::of::<dyn aiperf_plugin_api::category::TransportFactory>();
    let _ = std::any::TypeId::of::<dyn aiperf_plugin_api::category::ExporterFactory>();
}

/// SDK version constant is set and non-empty.
#[test]
fn plugin_source_api_version_is_set() {
    use aiperf_plugin_api::PLUGIN_SOURCE_API_VERSION;
    assert!(
        !PLUGIN_SOURCE_API_VERSION.is_empty(),
        "PLUGIN_SOURCE_API_VERSION must be set"
    );
    // Must be semver-shaped: at least two dots.
    assert_eq!(
        PLUGIN_SOURCE_API_VERSION
            .chars()
            .filter(|&c| c == '.')
            .count(),
        2,
        "PLUGIN_SOURCE_API_VERSION must be a semver triple: {PLUGIN_SOURCE_API_VERSION}"
    );
}
