// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest fixture matrix for Task 10.
//!
//! Tests strict schema-2.0 decoding, normalization, and stable error codes.

use aiperf_plugin_host::{
    error::ManifestError, manifest::PluginManifestV2, normalize::normalize_manifest,
};

fn load_fixture(name: &str) -> String {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/manifests")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read fixture {name}: {e}"))
}

fn parse_yaml(yaml: &str) -> Result<PluginManifestV2, serde_yaml::Error> {
    serde_yaml::from_str(yaml)
}

fn parse_and_normalize(yaml: &str) -> Result<PluginManifestV2, ManifestError> {
    let raw: PluginManifestV2 =
        serde_yaml::from_str(yaml).map_err(|e| ManifestError::ParseError(e.to_string()))?;
    normalize_manifest(raw)
}

// --- valid ---

#[test]
fn valid_complete_parses() {
    let yaml = load_fixture("valid_complete.yaml");
    let manifest = parse_and_normalize(&yaml).expect("valid_complete should parse");
    assert_eq!(manifest.schema_version, "2.0");
    assert_eq!(manifest.packages.len(), 1);
    let pkg = &manifest.packages[0];
    assert_eq!(pkg.id, "my-exporter");
    assert_eq!(pkg.version, "1.0.0");
    assert!(!pkg.categories.is_empty());
}

// --- schema_version ---

#[test]
fn python_schema_1_0_returns_stable_code() {
    let yaml = load_fixture("python_schema_1_0.yaml");
    let err = parse_and_normalize(&yaml).unwrap_err();
    assert!(
        matches!(err, ManifestError::PythonManifest(_)),
        "expected PythonManifest, got: {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("python-plugin-manifest-not-native"),
        "stable code missing: {msg}"
    );
}

#[test]
fn unsupported_schema_version_rejected() {
    let yaml = "schema_version: \"3.0\"\npackages: []";
    let err = parse_and_normalize(yaml).unwrap_err();
    assert!(matches!(err, ManifestError::UnsupportedSchemaVersion(_)));
}

// --- unknown fields ---

#[test]
fn unknown_field_at_root_rejected() {
    let yaml = load_fixture("unknown_field_root.yaml");
    // serde deny_unknown_fields fires at parse time
    let result = parse_yaml(&yaml);
    assert!(result.is_err(), "unknown root field should be rejected");
}

#[test]
fn unknown_field_in_package_rejected() {
    let yaml = load_fixture("unknown_field_package.yaml");
    let result = parse_yaml(&yaml);
    assert!(result.is_err(), "unknown package field should be rejected");
}

// --- path validation ---

#[test]
fn absolute_artifact_path_rejected() {
    let yaml = load_fixture("absolute_path.yaml");
    let err = parse_and_normalize(&yaml).unwrap_err();
    assert!(
        matches!(err, ManifestError::InvalidPath(_)),
        "expected InvalidPath, got: {err:?}"
    );
}

#[test]
fn parent_traversal_path_rejected() {
    let yaml = load_fixture("parent_traversal.yaml");
    let err = parse_and_normalize(&yaml).unwrap_err();
    assert!(
        matches!(err, ManifestError::InvalidPath(_)),
        "expected InvalidPath, got: {err:?}"
    );
}

#[test]
fn windows_ads_path_rejected() {
    let yaml = r#"
schema_version: "2.0"
packages:
  - id: my-plugin
    version: "1.0.0"
    artifacts:
      - target: x86_64-unknown-linux-gnu
        path: "plugin.dll:stream"
        digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    categories:
      - category: exporter
        id: my-exporter
"#;
    let err = parse_and_normalize(yaml).unwrap_err();
    assert!(
        matches!(err, ManifestError::InvalidPath(_)),
        "expected InvalidPath for ADS, got: {err:?}"
    );
}

// --- duplicate artifacts ---

#[test]
fn duplicate_artifact_for_same_target_rejected() {
    let yaml = load_fixture("duplicate_artifact.yaml");
    let err = parse_and_normalize(&yaml).unwrap_err();
    assert!(
        matches!(err, ManifestError::DuplicateBaselineArtifact(_)),
        "expected DuplicateBaselineArtifact, got: {err:?}"
    );
}

// --- semver ---

#[test]
fn noncanonical_semver_rejected() {
    let yaml = load_fixture("noncanonical_semver.yaml");
    let err = parse_and_normalize(&yaml).unwrap_err();
    assert!(
        matches!(err, ManifestError::InvalidSemVer(_)),
        "expected InvalidSemVer, got: {err:?}"
    );
}

// --- categories ---

#[test]
fn no_categories_rejected() {
    let yaml = r#"
schema_version: "2.0"
packages:
  - id: my-plugin
    version: "1.0.0"
    artifacts: []
    categories: []
"#;
    let err = parse_and_normalize(yaml).unwrap_err();
    assert!(
        matches!(err, ManifestError::NoCategories),
        "expected NoCategories, got: {err:?}"
    );
}

// --- defaults and normalization ---

#[test]
fn priority_defaults_to_zero() {
    let yaml = r#"
schema_version: "2.0"
packages:
  - id: my-plugin
    version: "1.0.0"
    artifacts:
      - target: x86_64-unknown-linux-gnu
        path: "plugin.so"
        digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    categories:
      - category: exporter
        id: my-exporter
"#;
    let manifest = parse_and_normalize(yaml).expect("should parse");
    assert_eq!(manifest.packages[0].priority, 0);
}

#[test]
fn aliases_are_sorted_and_deduplicated() {
    let yaml = r#"
schema_version: "2.0"
packages:
  - id: my-plugin
    version: "1.0.0"
    aliases: ["zebra", "apple", "apple", "mango"]
    artifacts:
      - target: x86_64-unknown-linux-gnu
        path: "plugin.so"
        digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    categories:
      - category: exporter
        id: my-exporter
"#;
    let manifest = parse_and_normalize(yaml).expect("should parse");
    let aliases = &manifest.packages[0].aliases;
    assert_eq!(aliases, &["apple", "mango", "zebra"]);
}

#[test]
fn depends_on_is_sorted() {
    let yaml = r#"
schema_version: "2.0"
packages:
  - id: my-plugin
    version: "1.0.0"
    depends_on:
      - id: "z-dep"
        version: ">=1.0.0"
      - id: "a-dep"
        version: ">=2.0.0"
    artifacts:
      - target: x86_64-unknown-linux-gnu
        path: "plugin.so"
        digest: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    categories:
      - category: exporter
        id: my-exporter
"#;
    let manifest = parse_and_normalize(yaml).expect("should parse");
    let deps = &manifest.packages[0].depends_on;
    assert_eq!(deps[0].id, "a-dep");
    assert_eq!(deps[1].id, "z-dep");
}

// --- schema golden ---

#[test]
fn schema_golden_file_exists_and_is_valid_json() {
    let schema_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("schema/plugins-2.0.schema.json");
    let content =
        std::fs::read_to_string(&schema_path).expect("schema/plugins-2.0.schema.json must exist");
    let parsed: serde_json::Value =
        serde_json::from_str(&content).expect("schema file must be valid JSON");
    assert_eq!(
        parsed["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    // Verify schema_version const is present
    let schema_ver = &parsed["properties"]["schema_version"]["const"];
    assert_eq!(
        schema_ver, "2.0",
        "schema must constrain schema_version to 2.0"
    );
}

// --- canonical serialization ---

#[test]
fn normalized_serialization_is_deterministic() {
    let yaml = load_fixture("valid_complete.yaml");
    let m1 = parse_and_normalize(&yaml).expect("parse 1");
    let m2 = parse_and_normalize(&yaml).expect("parse 2");
    let j1 = serde_json::to_string(&m1).expect("serialize 1");
    let j2 = serde_json::to_string(&m2).expect("serialize 2");
    assert_eq!(j1, j2, "canonical serialization must be deterministic");
}
