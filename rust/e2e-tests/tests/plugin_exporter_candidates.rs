// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tasks 24-27: Exporter candidate package staging tests (basic, mlflow, parquet, wandb).
//!
//! RED — fail until Tasks 24-27 GREEN build the respective cdylibs.
//! Static production exporters remain unchanged until Task 39b.

use std::path::PathBuf;

fn exporter_lib_path(name: &str, env_var: &str) -> PathBuf {
    if let Ok(p) = std::env::var(env_var) {
        return PathBuf::from(p);
    }
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("target")
        .join("debug");
    target.join(if cfg!(target_os = "macos") {
        format!("lib{name}.dylib")
    } else if cfg!(windows) {
        format!("{name}.dll")
    } else {
        format!("lib{name}.so")
    })
}

/// RED: fails until Task 24 GREEN builds the basic exporter cdylib.
#[test]
fn export_basic_candidate_library_exists() {
    let lib = exporter_lib_path(
        "aiperf_plugin_export_basic",
        "AIPERF_EXPORT_BASIC_PLUGIN_LIB",
    );
    assert!(
        lib.exists(),
        "basic exporter candidate not found at {}: build Task 24 GREEN",
        lib.display()
    );
}

/// RED: fails until Task 25 GREEN builds the MLflow exporter cdylib.
#[test]
fn export_mlflow_candidate_library_exists() {
    let lib = exporter_lib_path(
        "aiperf_plugin_export_mlflow",
        "AIPERF_EXPORT_MLFLOW_PLUGIN_LIB",
    );
    assert!(
        lib.exists(),
        "MLflow exporter candidate not found at {}: build Task 25 GREEN",
        lib.display()
    );
}

/// RED: fails until Task 26 GREEN builds the Parquet exporter cdylib.
#[test]
fn export_parquet_candidate_library_exists() {
    let lib = exporter_lib_path(
        "aiperf_plugin_export_parquet",
        "AIPERF_EXPORT_PARQUET_PLUGIN_LIB",
    );
    assert!(
        lib.exists(),
        "Parquet exporter candidate not found at {}: build Task 26 GREEN (--features parquet)",
        lib.display()
    );
}

/// RED: fails until Task 27 GREEN builds the W&B exporter cdylib.
#[test]
fn export_wandb_candidate_library_exists() {
    let lib = exporter_lib_path(
        "aiperf_plugin_export_wandb",
        "AIPERF_EXPORT_WANDB_PLUGIN_LIB",
    );
    assert!(
        lib.exists(),
        "W&B exporter candidate not found at {}: build Task 27 GREEN",
        lib.display()
    );
}

/// RED: plugins.yaml.in must declare basic exporter canonical IDs.
#[test]
fn export_basic_manifest_declares_capabilities() {
    let yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("plugins")
        .join("export-basic")
        .join("plugins.yaml.in");
    assert!(yaml.exists(), "export-basic/plugins.yaml.in missing");
    let content = std::fs::read_to_string(&yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("export"),
        "export-basic manifest does not declare export capabilities (Task 24 pending)"
    );
}

/// RED: plugins.yaml.in must declare MLflow exporter canonical ID.
#[test]
fn export_mlflow_manifest_declares_capability() {
    let yaml = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("plugins")
        .join("export-mlflow")
        .join("plugins.yaml.in");
    assert!(yaml.exists(), "export-mlflow/plugins.yaml.in missing");
    let content = std::fs::read_to_string(&yaml).unwrap();
    assert!(
        content.contains("canonical_id") && content.contains("mlflow"),
        "export-mlflow manifest does not declare mlflow capability (Task 25 pending)"
    );
}

/// Static invariant: production exporter implementations must remain in runtime.
#[test]
fn static_exporter_runtime_source_unchanged() {
    let export_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("runtime")
        .join("src")
        .join("export");
    assert!(
        export_dir.exists(),
        "runtime/src/export is gone — static production exporters removed"
    );
}
