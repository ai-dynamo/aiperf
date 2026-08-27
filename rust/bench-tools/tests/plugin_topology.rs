// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract for the measured native-plugin package and feature topology.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    process::Command,
};

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct BaselineTopology {
    host_commit: String,
    workspace_packages: Vec<BaselinePackage>,
}

#[derive(Debug, Deserialize)]
struct BaselinePackage {
    name: String,
    features: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct OwnershipMatrix {
    schema_version: u64,
    baseline_topology: BaselineBinding,
    package_ownership: Vec<PackageOwnership>,
    feature_ownership: Vec<FeatureOwnership>,
}

#[derive(Debug, Deserialize)]
struct BaselineBinding {
    path: String,
    host_commit: String,
    blake3: String,
    state: String,
}

#[derive(Debug, Deserialize)]
struct PackageOwnership {
    package: String,
    owner: String,
    review_state: String,
    coupling_evidence: String,
    dependencies: Vec<OwnedDependency>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
struct OwnedDependency {
    package: String,
    kind: String,
    justification: String,
}

#[derive(Debug, Deserialize)]
struct FeatureOwnership {
    baseline_package: String,
    feature: String,
    owners: Vec<String>,
    justification: String,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<MetadataPackage>,
    workspace_members: Vec<String>,
    metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct MetadataPackage {
    id: String,
    name: String,
    dependencies: Vec<MetadataDependency>,
}

#[derive(Debug, Deserialize)]
struct MetadataDependency {
    name: String,
    kind: Option<String>,
    path: Option<String>,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench-tools is directly below the workspace root")
        .to_path_buf()
}

#[test]
fn every_plugin_dependency_and_baseline_feature_has_one_reviewed_owner() {
    let root = workspace_root();
    let matrix_path = root.join("plugin-api/feature-ownership.toml");
    let matrix: OwnershipMatrix = toml::from_str(
        &std::fs::read_to_string(&matrix_path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", matrix_path.display())),
    )
    .expect("ownership matrix is valid TOML");
    assert_eq!(matrix.schema_version, 1);

    let topology_path = root.join("../").join(&matrix.baseline_topology.path);
    let topology_bytes = std::fs::read(&topology_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", topology_path.display()));
    assert_eq!(
        matrix.baseline_topology.blake3,
        format!("blake3:{}", blake3::hash(&topology_bytes).to_hex())
    );
    assert_eq!(
        matrix.baseline_topology.state,
        "provisional_task1_rebase_required"
    );
    let baseline: BaselineTopology =
        serde_json::from_slice(&topology_bytes).expect("baseline JSON");
    assert_eq!(baseline.host_commit, matrix.baseline_topology.host_commit);

    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
        .current_dir(&root)
        .output()
        .expect("cargo metadata executes");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata: Metadata = serde_json::from_slice(&output.stdout).expect("metadata JSON");
    let policy = &metadata.metadata["plugin-parity"];
    for (field, expected) in [
        ("schema-version", 1_u64),
        ("retained-pairs", 30),
        ("bootstrap-resamples", 100_000),
        ("max-replacement-pairs", 5),
        ("max-experiment-attempts", 3),
        ("exporter-corpus-records", 100_000),
        ("exporter-sample-repetitions", 16),
        ("exporter-processed-records", 1_600_000),
        ("exporter-retained-artifact-records", 100_000),
    ] {
        assert_eq!(policy[field].as_u64(), Some(expected), "{field}");
    }
    assert_eq!(policy["confidence"].as_f64(), Some(0.95));
    assert_eq!(policy["max-relative-regression"].as_f64(), Some(0.01));
    assert_eq!(policy["max-coefficient-of-variation"].as_f64(), Some(0.02));
    let members = metadata
        .workspace_members
        .into_iter()
        .collect::<BTreeSet<_>>();
    let packages = metadata
        .packages
        .into_iter()
        .filter(|package| members.contains(&package.id))
        .map(|package| (package.name.clone(), package))
        .collect::<BTreeMap<_, _>>();

    let rows = matrix
        .package_ownership
        .iter()
        .map(|row| {
            assert!(!row.owner.is_empty());
            assert!(matches!(
                row.review_state.as_str(),
                "measured" | "reviewed_neutral"
            ));
            assert!(!row.coupling_evidence.is_empty());
            (row.package.as_str(), row)
        })
        .collect::<BTreeMap<_, _>>();
    let reviewed_roots = BTreeSet::from([
        "aiperf-core",
        "aiperf-plugin-sdk-macros",
        "aiperf-plugin-static-comparator",
        "aiperf-allocator-provider",
        "aiperf-allocator-shim",
    ]);
    for row in matrix
        .package_ownership
        .iter()
        .filter(|row| row.dependencies.is_empty())
    {
        assert!(
            reviewed_roots.contains(row.package.as_str()),
            "dependency-neutral shell {} was left unresolved",
            row.package
        );
        assert_eq!(row.review_state, "reviewed_neutral");
    }
    let plugin_packages = packages
        .keys()
        .filter(|name| {
            name.starts_with("aiperf-plugin-")
                || matches!(
                    name.as_str(),
                    "aiperf-core"
                        | "aiperf-endpoint-sdk"
                        | "aiperf-transport-sdk"
                        | "aiperf-export-sdk"
                        | "aiperf-allocator-provider"
                        | "aiperf-allocator-shim"
                )
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        rows.keys().copied().collect::<BTreeSet<_>>(),
        plugin_packages.iter().map(String::as_str).collect()
    );

    for package_name in plugin_packages {
        let package = &packages[&package_name];
        let actual = package
            .dependencies
            .iter()
            .filter(|dependency| dependency.path.is_some())
            .map(|dependency| OwnedDependency {
                package: dependency.name.clone(),
                kind: dependency
                    .kind
                    .clone()
                    .unwrap_or_else(|| "normal".to_owned()),
                justification: rows[package_name.as_str()]
                    .dependencies
                    .iter()
                    .find(|owned| {
                        owned.package == dependency.name
                            && owned.kind
                                == dependency
                                    .kind
                                    .clone()
                                    .unwrap_or_else(|| "normal".to_owned())
                    })
                    .map(|owned| owned.justification.clone())
                    .unwrap_or_default(),
            })
            .collect::<BTreeSet<_>>();
        let reviewed = rows[package_name.as_str()]
            .dependencies
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        assert_eq!(
            actual, reviewed,
            "unmeasured dependency edge for {package_name}"
        );
        assert!(reviewed.iter().all(|edge| !edge.justification.is_empty()));
    }

    let baseline_features = baseline
        .workspace_packages
        .iter()
        .flat_map(|package| {
            package
                .features
                .iter()
                .map(move |feature| (package.name.as_str(), feature.as_str()))
        })
        .collect::<BTreeSet<_>>();
    let mut owned_features = BTreeSet::new();
    for row in matrix.feature_ownership {
        assert!(baseline_features.contains(&(row.baseline_package.as_str(), row.feature.as_str())));
        assert!(!row.owners.is_empty());
        assert!(!row.justification.is_empty());
        assert!(
            row.owners
                .iter()
                .all(|owner| rows.contains_key(owner.as_str()))
        );
        assert!(owned_features.insert((row.baseline_package, row.feature)));
    }
    let measured_runtime_features = baseline
        .workspace_packages
        .iter()
        .find(|package| package.name == "aiperf-runtime")
        .expect("Task-1 topology contains aiperf-runtime")
        .features
        .iter()
        .map(|feature| ("aiperf-runtime".to_owned(), feature.clone()))
        .collect::<BTreeSet<_>>();
    assert_eq!(owned_features, measured_runtime_features);
}
