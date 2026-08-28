// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Task 28: OTel exporter candidate package staging and telemetry capture tests.
//!
//! These tests are **RED** — they fail because no SDK-built dynamic OTel
//! candidate exists yet. The plugin crate (`aiperf-plugin-export-otel`) must
//! expose `aiperf_plugin_entry_v1` as a `cdylib` entry point and the candidate
//! library must be present before these pass.
//!
//! When Task 28 (GREEN) lands:
//! - `aiperf-plugin-export-otel` builds as a `cdylib`.
//! - The candidate `plugins.yaml.in` declares `canonical_id = "otel"` and
//!   `requires = "FoldedProjectionV1(GenAiClientHistogramsV1)"`.
//! - The static production `otel` exporter remains unchanged; the candidate
//!   is identified only via its lock entry.

use std::path::PathBuf;

/// Path where the OTel plugin candidate library is expected after `cargo build
/// -p aiperf-plugin-export-otel --lib`.
fn candidate_lib_path() -> PathBuf {
    // Prefer the env override used in CI; fall back to a conventional location.
    if let Ok(p) = std::env::var("AIPERF_OTEL_PLUGIN_LIB") {
        return PathBuf::from(p);
    }
    // Try debug build location relative to cargo target dir.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let target_dir = PathBuf::from(manifest)
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    let libname = if cfg!(target_os = "macos") {
        "libaiperf_plugin_export_otel.dylib"
    } else if cfg!(windows) {
        "aiperf_plugin_export_otel.dll"
    } else {
        "libaiperf_plugin_export_otel.so"
    };
    target_dir.join(libname)
}

/// RED: The candidate cdylib must exist before the OTel plugin can be loaded.
///
/// This test fails until Task 28 (GREEN) builds the candidate library.
#[test]
fn otel_plugin_candidate_library_exists() {
    let lib = candidate_lib_path();
    assert!(
        lib.exists(),
        "OTel plugin candidate library not found at {}: build with \
         `cargo build -p aiperf-plugin-export-otel --lib` (Task 28 GREEN)",
        lib.display()
    );
}

/// RED: The candidate library must export the `aiperf_plugin_entry_v1` symbol.
///
/// Fails until the plugin crate is declared as `crate-type = ["cdylib"]` and
/// the `#[aiperf_plugin]` macro generates the entry point.
#[test]
#[cfg(unix)]
fn otel_plugin_entry_point_exported() {
    let lib = candidate_lib_path();
    if !lib.exists() {
        // Let the prior test report the missing library; skip here.
        return;
    }

    // Use `nm` / `objdump` to verify the symbol is present without loading.
    let output = std::process::Command::new("nm")
        .arg("-D")
        .arg("--defined-only")
        .arg(&lib)
        .output()
        .expect("nm must be available");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let symbol = "aiperf_plugin_entry_v1";
    let found = stdout.lines().any(|line| {
        line.split_ascii_whitespace()
            .last()
            .is_some_and(|s| s == symbol)
    });
    assert!(
        found,
        "symbol `{symbol}` not exported from {}: \
         add `#[aiperf_plugin]` to the plugin registration block (Task 28 GREEN)",
        lib.display()
    );
}

/// RED: The candidate `plugins.yaml.in` must declare canonical_id = "otel".
///
/// Fails until `plugins.yaml.in` is authored for Task 28.
#[test]
fn otel_plugin_manifest_declares_canonical_id() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let plugins_yaml = manifest_dir
        .parent()
        .unwrap()
        .join("plugins")
        .join("export-otel")
        .join("plugins.yaml.in");

    assert!(
        plugins_yaml.exists(),
        "plugins.yaml.in not found at {}: Task 28 must author it",
        plugins_yaml.display()
    );

    let content = std::fs::read_to_string(&plugins_yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("otel"),
        "plugins.yaml.in at {} does not declare `canonical_id: otel`",
        plugins_yaml.display()
    );
    assert!(
        content.contains("FoldedProjectionV1") || content.contains("folded_projection"),
        "plugins.yaml.in at {} does not declare FoldedProjectionV1 requirement",
        plugins_yaml.display()
    );
}

/// Static source scan: the production OTel implementation must not contain a
/// per-record accumulator or an OTel-specific report side channel after Task 19.
///
/// This test verifies a Task-19 invariant: the static production exporter no
/// longer drives a dedicated per-record accumulator after Task 19 removed it.
#[test]
fn production_otel_source_has_no_per_record_accumulator() {
    // Scan the production otel exporter source for known banned accumulator patterns.
    let src_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("runtime")
        .join("src")
        .join("export");

    let otel_source_files: Vec<_> = std::fs::read_dir(&src_dir)
        .expect("runtime/src/export must exist")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|e| e == "rs")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("otel"))
        })
        .collect();

    for path in &otel_source_files {
        let source = std::fs::read_to_string(path).unwrap();
        // Banned: a per-record callback that drives OTel-specific accumulation.
        // These patterns were removed by Task 19; their presence is a regression.
        assert!(
            !source.contains("OtelAccumulator"),
            "{} contains banned OtelAccumulator (Task 19 regression)",
            path.display()
        );
        assert!(
            !source.contains("otel_record_callback"),
            "{} contains banned otel_record_callback (Task 19 regression)",
            path.display()
        );
    }
}
