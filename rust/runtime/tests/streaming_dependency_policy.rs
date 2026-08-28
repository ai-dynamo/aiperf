// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::{Path, PathBuf};

fn parse_manifest(path: &Path) -> toml::Value {
    let manifest = std::fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
    manifest
        .parse()
        .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn runtime_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml")
}

fn workspace_manifest_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("runtime manifest has workspace parent")
        .join("Cargo.toml")
}

/// Collect the transitive feature closure of `feature` over the manifest's own
/// feature table, including `dep:` activations, so an indirect enablement of a
/// crypto crate through some intermediate feature cannot slip past the check.
fn feature_closure(features: &toml::Value, feature: &str) -> Vec<String> {
    let mut pending = vec![feature.to_string()];
    let mut seen: Vec<String> = Vec::new();
    while let Some(current) = pending.pop() {
        if seen.contains(&current) {
            continue;
        }
        seen.push(current.clone());
        let Some(entries) = features.get(&current).and_then(toml::Value::as_array) else {
            continue;
        };
        for entry in entries {
            if let Some(text) = entry.as_str() {
                pending.push(text.to_string());
            }
        }
    }
    seen
}

#[test]
fn base_streaming_owns_no_encryption_or_zeroization_dependency() {
    let runtime = parse_manifest(&runtime_manifest_path());
    let workspace = parse_manifest(&workspace_manifest_path());

    let features = &runtime["features"];
    let streaming = features["streaming"]
        .as_array()
        .expect("runtime streaming feature is an array");
    assert_eq!(streaming, &[toml::Value::String("engine".into())]);

    // The crypto crates exist, but only behind the opt-in `streaming-crypto`
    // feature: they must be optional in the member manifest, and the base
    // `streaming` closure must not reach them by any path.
    let crypto = features["streaming-crypto"]
        .as_array()
        .expect("runtime streaming-crypto feature is an array");
    assert_eq!(
        crypto,
        &[
            toml::Value::String("streaming".into()),
            toml::Value::String("dep:chacha20poly1305".into()),
            toml::Value::String("dep:zeroize".into()),
        ]
    );

    let streaming_closure = feature_closure(features, "streaming");
    for dependency in ["chacha20poly1305", "zeroize"] {
        // A workspace-level version pin carries no activation; what matters is
        // that the member entry stays optional.
        assert!(
            workspace["workspace"]["dependencies"]
                .get(dependency)
                .is_some(),
            "workspace does not pin {dependency}"
        );
        let entry = runtime["dependencies"]
            .get(dependency)
            .unwrap_or_else(|| panic!("runtime does not declare {dependency}"));
        assert_eq!(
            entry.get("optional").and_then(toml::Value::as_bool),
            Some(true),
            "runtime declares non-optional {dependency} dependency"
        );
        assert!(
            !streaming_closure.iter().any(|feature| {
                feature == dependency || feature == &format!("dep:{dependency}")
            }),
            "base streaming feature closure enables {dependency}"
        );
    }
}
