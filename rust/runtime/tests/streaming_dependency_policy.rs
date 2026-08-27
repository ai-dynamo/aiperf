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

#[test]
fn base_streaming_owns_no_encryption_or_zeroization_dependency() {
    let runtime = parse_manifest(&runtime_manifest_path());
    let workspace = parse_manifest(&workspace_manifest_path());

    let streaming = runtime["features"]["streaming"]
        .as_array()
        .expect("runtime streaming feature is an array");
    assert_eq!(streaming, &[toml::Value::String("engine".into())]);

    for dependency in ["chacha20poly1305", "zeroize"] {
        assert!(
            workspace["workspace"]["dependencies"]
                .get(dependency)
                .is_none(),
            "workspace declares direct {dependency} dependency"
        );
        assert!(
            runtime["dependencies"].get(dependency).is_none(),
            "runtime declares direct {dependency} dependency"
        );
    }
}
