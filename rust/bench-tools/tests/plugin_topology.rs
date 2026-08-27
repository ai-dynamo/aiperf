// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract for the measured native-plugin package and feature topology.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    process::Command,
};

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
struct BaselineTopology {
    host_commit: String,
    cargo_projection: Vec<CargoPackageProjection>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CargoPackageProjection {
    name: String,
    version: String,
    edition: String,
    dependencies: Vec<CargoDependencyIdentity>,
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct CargoDependencyIdentity {
    package: String,
    local_name: String,
    kind: String,
    source: Option<String>,
    requirement: String,
    registry: Option<String>,
    path: Option<String>,
    target: Option<String>,
    is_optional: bool,
    uses_default_features: bool,
    features: Vec<String>,
    is_workspace: bool,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnershipMatrix {
    schema_version: u64,
    symbol_ownership: Vec<toml::Value>,
    topology_amendment: TopologyAmendment,
    baseline_topology: BaselineBinding,
    package_ownership: Vec<PackageOwnership>,
    feature_ownership: Vec<FeatureOwnership>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct TopologyAmendment {
    schema_version: u64,
    producer_task: u64,
    from_state: String,
    to_state: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BaselineBinding {
    path: String,
    host_commit: String,
    blake3: String,
    state: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PackageOwnership {
    package: String,
    owner: String,
    review_state: String,
    coupling_evidence: String,
    source_package: Option<String>,
    dependencies: Vec<OwnedDependency>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OwnedDependency {
    package: String,
    kind: String,
    justification: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureOwnership {
    baseline_package: String,
    feature: String,
    baseline_forwarding: Vec<String>,
    splits: Vec<FeatureSplit>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FeatureSplit {
    entry: String,
    owner: String,
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

fn insert_unique<T: Ord>(set: &mut BTreeSet<T>, value: T, description: &str) -> Result<(), String> {
    if set.insert(value) {
        Ok(())
    } else {
        Err(format!("duplicate {description}"))
    }
}

fn validate_projection(
    matrix: &OwnershipMatrix,
    baseline: &BaselineTopology,
) -> Result<(), String> {
    if matrix.schema_version != 2 {
        return Err("ownership schema must be version 2".to_owned());
    }
    if matrix.topology_amendment
        != (TopologyAmendment {
            schema_version: 1,
            producer_task: 3,
            from_state: "task2_neutral".to_owned(),
            to_state: "task3_reviewed".to_owned(),
        })
        || matrix.baseline_topology.state != "task3_reviewed"
    {
        return Err("ownership matrix is not the final Task-3 reviewed state".to_owned());
    }
    if !matrix.symbol_ownership.iter().all(toml::Value::is_table) {
        return Err("symbol ownership rows must be tables".to_owned());
    }

    let mut cargo_packages = BTreeMap::new();
    for package in &baseline.cargo_projection {
        if package.version.is_empty() || package.edition.is_empty() {
            return Err(format!("incomplete Cargo identity for {}", package.name));
        }
        let mut dependencies = BTreeSet::new();
        for dependency in &package.dependencies {
            insert_unique(
                &mut dependencies,
                dependency.clone(),
                &format!("Task-1 dependency row in {}", package.name),
            )?;
        }
        if cargo_packages
            .insert(package.name.as_str(), package)
            .is_some()
        {
            return Err(format!("duplicate Task-1 package row {}", package.name));
        }
    }

    let mut package_rows = BTreeMap::new();
    for row in &matrix.package_ownership {
        if row.owner.is_empty() || row.coupling_evidence.trim().is_empty() {
            return Err(format!(
                "empty owner or coupling evidence for {}",
                row.package
            ));
        }
        match row.review_state.as_str() {
            "measured" => {
                let source = row.source_package.as_deref().ok_or_else(|| {
                    format!(
                        "measured package {} lacks a Task-1 source witness",
                        row.package
                    )
                })?;
                if !cargo_packages.contains_key(source) {
                    return Err(format!(
                        "package {} cites unknown Task-1 source package {source}",
                        row.package
                    ));
                }
            }
            "reviewed_neutral" => {
                if row.source_package.is_some() || !row.dependencies.is_empty() {
                    return Err(format!(
                        "reviewed-neutral package {} has projected coupling",
                        row.package
                    ));
                }
            }
            state => return Err(format!("unsupported review state {state}")),
        }
        let mut dependencies = BTreeSet::new();
        for dependency in &row.dependencies {
            if dependency.justification.trim().is_empty() {
                return Err(format!("empty dependency justification in {}", row.package));
            }
            insert_unique(
                &mut dependencies,
                (dependency.package.as_str(), dependency.kind.as_str()),
                &format!("dependency row in {}", row.package),
            )?;
        }
        if package_rows.insert(row.package.as_str(), row).is_some() {
            return Err(format!("duplicate package row {}", row.package));
        }
    }

    let mut feature_rows = BTreeSet::new();
    for row in &matrix.feature_ownership {
        let key = (row.baseline_package.as_str(), row.feature.as_str());
        insert_unique(&mut feature_rows, key, "feature row")?;
        let source = cargo_packages
            .get(row.baseline_package.as_str())
            .ok_or_else(|| format!("unknown Task-1 feature package {}", row.baseline_package))?;
        let forwarding = source.features.get(&row.feature).ok_or_else(|| {
            format!(
                "unknown Task-1 feature {}:{}",
                row.baseline_package, row.feature
            )
        })?;
        if forwarding != &row.baseline_forwarding {
            return Err(format!(
                "Task-1 feature forwarding drift for {}:{}",
                row.baseline_package, row.feature
            ));
        }
        if row.splits.is_empty() {
            return Err(format!(
                "feature {}:{} has no owned split entry",
                row.baseline_package, row.feature
            ));
        }
        let mut splits = BTreeSet::new();
        for split in &row.splits {
            if split.entry.is_empty() || split.owner.is_empty() {
                return Err("feature split entry and owner must be non-empty".to_owned());
            }
            insert_unique(&mut splits, split.entry.as_str(), "feature split entry")?;
            if !package_rows.contains_key(split.owner.as_str()) {
                return Err(format!("unknown feature owner {}", split.owner));
            }
        }
        let expected_splits = if row.baseline_package == "aiperf-runtime" && row.feature == "grpc" {
            BTreeSet::from(["endpoint_bindings", "transport"])
        } else {
            BTreeSet::from(["complete"])
        };
        if splits != expected_splits {
            return Err(format!(
                "feature {}:{} has an inexact split projection",
                row.baseline_package, row.feature
            ));
        }
    }

    let runtime = cargo_packages
        .get("aiperf-runtime")
        .ok_or_else(|| "Task-1 Cargo projection lacks aiperf-runtime".to_owned())?;
    let expected_features = runtime
        .features
        .keys()
        .map(|feature| ("aiperf-runtime", feature.as_str()))
        .collect::<BTreeSet<_>>();
    if feature_rows != expected_features {
        return Err("runtime feature ownership projection is incomplete".to_owned());
    }
    Ok(())
}

fn fixture() -> (OwnershipMatrix, BaselineTopology) {
    let baseline = BaselineTopology {
        host_commit: "caa3ff6fcf20ffe36a7704abe16274bedadbb9fb".to_owned(),
        cargo_projection: vec![CargoPackageProjection {
            name: "aiperf-runtime".to_owned(),
            version: "0.12.0".to_owned(),
            edition: "2024".to_owned(),
            dependencies: vec![CargoDependencyIdentity {
                package: "tonic".to_owned(),
                local_name: "tonic".to_owned(),
                kind: "normal".to_owned(),
                source: Some("registry+https://github.com/rust-lang/crates.io-index".to_owned()),
                requirement: "^0.14".to_owned(),
                registry: None,
                path: None,
                target: None,
                is_optional: true,
                uses_default_features: false,
                features: vec!["channel".to_owned()],
                is_workspace: false,
            }],
            features: BTreeMap::from([("grpc".to_owned(), vec!["dep:tonic".to_owned()])]),
        }],
    };
    let packages = [
        ("aiperf-plugin-endpoints", "endpoints"),
        ("aiperf-plugin-transport-grpc", "transport-grpc"),
    ]
    .into_iter()
    .map(|(package, owner)| PackageOwnership {
        package: package.to_owned(),
        owner: owner.to_owned(),
        review_state: "measured".to_owned(),
        source_package: Some("aiperf-runtime".to_owned()),
        dependencies: vec![OwnedDependency {
            package: "aiperf-plugin-api".to_owned(),
            kind: "normal".to_owned(),
            justification: "fixture witness".to_owned(),
        }],
        coupling_evidence: "Task-1 fixture source coupling".to_owned(),
    })
    .collect();
    let matrix = OwnershipMatrix {
        schema_version: 2,
        symbol_ownership: vec![],
        topology_amendment: TopologyAmendment {
            schema_version: 1,
            producer_task: 3,
            from_state: "task2_neutral".to_owned(),
            to_state: "task3_reviewed".to_owned(),
        },
        baseline_topology: BaselineBinding {
            path: String::new(),
            host_commit: baseline.host_commit.clone(),
            blake3: String::new(),
            state: "task3_reviewed".to_owned(),
        },
        package_ownership: packages,
        feature_ownership: vec![FeatureOwnership {
            baseline_package: "aiperf-runtime".to_owned(),
            feature: "grpc".to_owned(),
            baseline_forwarding: vec!["dep:tonic".to_owned()],
            splits: vec![
                FeatureSplit {
                    entry: "endpoint_bindings".to_owned(),
                    owner: "aiperf-plugin-endpoints".to_owned(),
                },
                FeatureSplit {
                    entry: "transport".to_owned(),
                    owner: "aiperf-plugin-transport-grpc".to_owned(),
                },
            ],
        }],
    };
    (matrix, baseline)
}

#[test]
fn projection_rejects_duplicate_package_dependency_feature_and_split_rows() {
    let (matrix, baseline) = fixture();
    assert!(validate_projection(&matrix, &baseline).is_ok());

    let mut duplicate = matrix.clone();
    duplicate
        .package_ownership
        .push(duplicate.package_ownership[0].clone());
    assert!(
        validate_projection(&duplicate, &baseline)
            .unwrap_err()
            .contains("duplicate package row")
    );

    let mut duplicate = matrix.clone();
    let edge = duplicate.package_ownership[0].dependencies[0].clone();
    duplicate.package_ownership[0].dependencies.push(edge);
    assert!(
        validate_projection(&duplicate, &baseline)
            .unwrap_err()
            .contains("duplicate dependency row")
    );

    let mut duplicate = matrix.clone();
    duplicate
        .feature_ownership
        .push(duplicate.feature_ownership[0].clone());
    assert!(
        validate_projection(&duplicate, &baseline)
            .unwrap_err()
            .contains("duplicate feature row")
    );

    let mut duplicate = matrix;
    let split = duplicate.feature_ownership[0].splits[0].clone();
    duplicate.feature_ownership[0].splits.push(split);
    assert!(
        validate_projection(&duplicate, &baseline)
            .unwrap_err()
            .contains("duplicate feature split entry")
    );
}

#[test]
fn projection_requires_concrete_task1_cargo_witnesses_and_exact_grpc_splits() {
    let (matrix, baseline) = fixture();

    let mut unwitnessed = matrix.clone();
    unwitnessed.package_ownership[0].source_package = None;
    assert!(
        validate_projection(&unwitnessed, &baseline)
            .unwrap_err()
            .contains("lacks a Task-1 source witness")
    );

    let mut drifted = matrix.clone();
    drifted.feature_ownership[0].baseline_forwarding = vec![];
    assert!(
        validate_projection(&drifted, &baseline)
            .unwrap_err()
            .contains("Task-1 feature forwarding drift")
    );

    let mut unsplit = matrix;
    unsplit.feature_ownership[0].splits = vec![FeatureSplit {
        entry: "complete".to_owned(),
        owner: "aiperf-plugin-transport-grpc".to_owned(),
    }];
    assert!(
        validate_projection(&unsplit, &baseline)
            .unwrap_err()
            .contains("inexact split projection")
    );

    let (mut ownerless, baseline) = fixture();
    ownerless.feature_ownership[0].splits[0].owner.clear();
    assert!(
        validate_projection(&ownerless, &baseline)
            .unwrap_err()
            .contains("owner must be non-empty")
    );

    let (mut unevidenced, baseline) = fixture();
    unevidenced.package_ownership[0].coupling_evidence.clear();
    assert!(
        validate_projection(&unevidenced, &baseline)
            .unwrap_err()
            .contains("empty owner or coupling evidence")
    );

    let (mut unjustified, baseline) = fixture();
    unjustified.package_ownership[0].dependencies[0]
        .justification
        .clear();
    assert!(
        validate_projection(&unjustified, &baseline)
            .unwrap_err()
            .contains("empty dependency justification")
    );
}

#[test]
fn toml_is_a_test_only_dependency() {
    let root = workspace_root();
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
        .current_dir(&root)
        .output()
        .expect("cargo metadata executes");
    assert!(output.status.success());
    let metadata: Metadata = serde_json::from_slice(&output.stdout).expect("metadata JSON");
    let package = metadata
        .packages
        .iter()
        .find(|package| package.name == "aiperf-bench-tools")
        .expect("bench-tools package");
    let toml_dependencies = package
        .dependencies
        .iter()
        .filter(|dependency| dependency.name == "toml")
        .collect::<Vec<_>>();
    assert_eq!(toml_dependencies.len(), 1);
    assert_eq!(toml_dependencies[0].kind.as_deref(), Some("dev"));
}

#[test]
fn every_plugin_dependency_and_baseline_feature_has_one_reviewed_owner() {
    let root = workspace_root();
    let matrix_path = root.join("plugin-api/feature-ownership.toml");
    let matrix: OwnershipMatrix = toml::from_str(
        &std::fs::read_to_string(&matrix_path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", matrix_path.display())),
    )
    .expect("ownership matrix has the strict version-2 shape");

    assert_eq!(
        matrix.topology_amendment,
        TopologyAmendment {
            schema_version: 1,
            producer_task: 3,
            from_state: "task2_neutral".to_owned(),
            to_state: "task3_reviewed".to_owned(),
        }
    );
    assert_eq!(matrix.baseline_topology.state, "task3_reviewed");
    let topology_path = root.join("../").join(&matrix.baseline_topology.path);
    let topology_bytes = std::fs::read(&topology_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", topology_path.display()));
    let baseline: BaselineTopology = serde_json::from_slice(&topology_bytes)
        .expect("Task-1 topology must contain the exact Cargo projection");
    assert_eq!(baseline.host_commit, matrix.baseline_topology.host_commit);
    assert_eq!(
        matrix.baseline_topology.blake3,
        format!("blake3:{}", blake3::hash(&topology_bytes).to_hex())
    );
    validate_projection(&matrix, &baseline).expect("ownership projection must be exact");

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
    let mut packages = BTreeMap::new();
    for package in metadata
        .packages
        .into_iter()
        .filter(|package| members.contains(&package.id))
    {
        let name = package.name.clone();
        assert!(
            packages.insert(name.clone(), package).is_none(),
            "duplicate Cargo package row {name}"
        );
    }
    let rows = matrix
        .package_ownership
        .iter()
        .map(|row| (row.package.as_str(), row))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(rows.len(), matrix.package_ownership.len());
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
        let actual = packages[&package_name]
            .dependencies
            .iter()
            .filter(|dependency| dependency.path.is_some())
            .map(|dependency| {
                (
                    dependency.name.as_str(),
                    dependency.kind.as_deref().unwrap_or("normal"),
                )
            })
            .collect::<Vec<_>>();
        let unique = actual.iter().copied().collect::<BTreeSet<_>>();
        assert_eq!(unique.len(), actual.len(), "duplicate Cargo dependency row");
        assert_eq!(
            rows[package_name.as_str()]
                .dependencies
                .iter()
                .map(|dependency| (dependency.package.as_str(), dependency.kind.as_str()))
                .collect::<BTreeSet<_>>(),
            unique,
            "unmeasured dependency edge for {package_name}"
        );
    }
}
