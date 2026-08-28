// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for author tools: manifest validation, conformance runner.

use aiperf_plugin_sdk::conformance::{ConformanceReport, run_conformance};
use aiperf_plugin_sdk::manifest::{PluginManifest, PluginEntry, Requirements, validate_manifest};

fn minimal_manifest() -> PluginManifest {
    PluginManifest {
        schema_version: 1,
        plugin: PluginEntry {
            name: "test".to_string(),
            version: "0.1.0".to_string(),
            description: None,
        },
        requires: Requirements {
            aiperf_sdk: ">=0.13.0".to_string(),
            target: "x86_64-unknown-linux-gnu".to_string(),
        },
    }
}

#[test]
fn manifest_valid_passes_validation() {
    let m = minimal_manifest();
    assert!(validate_manifest(&m).is_ok());
}

#[test]
fn manifest_rejects_schema_version_zero() {
    let mut m = minimal_manifest();
    m.schema_version = 0;
    assert!(validate_manifest(&m).is_err(), "schema_version=0 must be rejected");
}

#[test]
fn manifest_rejects_empty_plugin_name() {
    let mut m = minimal_manifest();
    m.plugin.name = "".to_string();
    assert!(validate_manifest(&m).is_err());
}

#[test]
fn manifest_with_description() {
    let m = PluginManifest {
        schema_version: 1,
        plugin: PluginEntry {
            name: "my-plugin".to_string(),
            version: "1.0.0".to_string(),
            description: Some("A useful plugin".to_string()),
        },
        requires: Requirements {
            aiperf_sdk: ">=0.13.0".to_string(),
            target: "x86_64-unknown-linux-gnu".to_string(),
        },
    };
    assert_eq!(m.plugin.description.as_deref(), Some("A useful plugin"));
    assert!(validate_manifest(&m).is_ok());
}

#[test]
fn manifest_parse_roundtrip() {
    let m = minimal_manifest();
    let serialized = aiperf_plugin_sdk::manifest::serialize_manifest(&m)
        .expect("serialize manifest");
    let m2 = aiperf_plugin_sdk::manifest::parse_manifest(&serialized)
        .expect("parse manifest");
    assert_eq!(m2.schema_version, m.schema_version);
    assert_eq!(m2.plugin.name, m.plugin.name);
    assert_eq!(m2.plugin.version, m.plugin.version);
    assert_eq!(m2.requires.aiperf_sdk, m.requires.aiperf_sdk);
    assert_eq!(m2.requires.target, m.requires.target);
}

#[test]
fn conformance_report_structure() {
    let report = ConformanceReport {
        passed: vec!["entry_symbol".to_string(), "manifest_present".to_string()],
        failed: vec![],
    };
    assert!(report.failed.is_empty());
    assert_eq!(report.passed.len(), 2);
}

#[test]
fn run_conformance_on_nonexistent_path_errors() {
    let path = std::path::Path::new("/this/path/does/not/exist/plugin.so");
    assert!(run_conformance(path).is_err());
}
