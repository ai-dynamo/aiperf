// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract for the reviewed native-plugin package and feature projection.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    process::Command,
};

use serde::Deserialize;

const IMPLEMENTATION_TASK_PACKAGES: &[(u64, &[&str])] = &[
    (4, &["aiperf-core"]),
    (5, &["aiperf-plugin-api"]),
    (
        6,
        &[
            "aiperf-endpoint-sdk",
            "aiperf-export-sdk",
            "aiperf-plugin-test-support",
            "aiperf-transport-sdk",
        ],
    ),
    (7, &["aiperf-allocator-provider", "aiperf-allocator-shim"]),
    (9, &["aiperf-plugin-sdk", "aiperf-plugin-sdk-macros"]),
    (16, &["aiperf-plugin-host"]),
    (24, &["aiperf-plugin-export-basic"]),
    (25, &["aiperf-plugin-export-parquet"]),
    (26, &["aiperf-plugin-export-mlflow"]),
    (27, &["aiperf-plugin-export-wandb"]),
    (28, &["aiperf-plugin-export-otel"]),
    (30, &["aiperf-plugin-endpoints"]),
    (31, &["aiperf-plugin-transport-http"]),
    (32, &["aiperf-plugin-transport-grpc"]),
    (
        33,
        &[
            "aiperf-plugin-transport-dry-run",
            "aiperf-plugin-transport-websocket",
        ],
    ),
    (34, &["aiperf-plugin-transport-dynosim"]),
    (35, &["aiperf-plugin-packaging-tests"]),
    (36, &["aiperf-plugin-conformance"]),
    (37, &["aiperf-plugin-static-comparator"]),
    (38, &["aiperf-plugin-perf"]),
];

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
    projection_basis: String,
    source_package: Option<String>,
    dependencies: Vec<OwnedDependency>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
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
    features: BTreeMap<String, Vec<String>>,
    manifest_path: String,
}

#[derive(Debug, Deserialize)]
struct MetadataDependency {
    name: String,
    kind: Option<String>,
    path: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImplementationWitness {
    schema_version: u64,
    task: u64,
    packages: Vec<ImplementedPackageWitness>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ImplementedPackageWitness {
    package: String,
    source_files: Vec<String>,
    dependencies: Vec<OwnedDependency>,
    features: BTreeMap<String, Vec<String>>,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("bench-tools is directly below the workspace root")
        .to_path_buf()
}

fn fixture_from_checked_files() -> (OwnershipMatrix, BaselineTopology) {
    let root = workspace_root();
    let matrix_path = root.join("plugin-api/feature-ownership.toml");
    let matrix: OwnershipMatrix = toml::from_str(
        &std::fs::read_to_string(&matrix_path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", matrix_path.display())),
    )
    .expect("ownership matrix has the strict version-2 shape");
    let topology_path = root.join("../").join(&matrix.baseline_topology.path);
    let baseline: BaselineTopology = serde_json::from_slice(
        &std::fs::read(&topology_path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", topology_path.display())),
    )
    .expect("Task-1 topology must contain the exact Cargo projection");
    (matrix, baseline)
}

fn checked_matrix_and_metadata(root: &std::path::Path) -> (OwnershipMatrix, Metadata) {
    let (matrix, baseline) = fixture_from_checked_files();
    validate_projection(&matrix, &baseline).expect("ownership projection must be exact");
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .expect("cargo metadata executes");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let metadata = serde_json::from_slice(&output.stdout).expect("metadata JSON");
    (matrix, metadata)
}

fn collect_rust_sources(
    workspace: &std::path::Path,
    directory: &std::path::Path,
    files: &mut Vec<String>,
) -> Result<(), String> {
    for entry in std::fs::read_dir(directory)
        .map_err(|error| format!("cannot read {}: {error}", directory.display()))?
    {
        let entry = entry.map_err(|error| format!("cannot read directory entry: {error}"))?;
        let file_type = entry
            .file_type()
            .map_err(|error| format!("cannot inspect {}: {error}", entry.path().display()))?;
        if file_type.is_dir() {
            collect_rust_sources(workspace, &entry.path(), files)?;
        } else if file_type.is_file() && entry.path().extension().is_some_and(|ext| ext == "rs") {
            let relative = entry
                .path()
                .strip_prefix(workspace)
                .map_err(|_| format!("source outside workspace: {}", entry.path().display()))?
                .to_str()
                .ok_or_else(|| format!("non-UTF-8 source path: {}", entry.path().display()))?
                .replace(std::path::MAIN_SEPARATOR, "/");
            files.push(relative);
        }
    }
    Ok(())
}

fn validate_implementation_witness(
    workspace: &std::path::Path,
    matrix: &OwnershipMatrix,
    metadata: &Metadata,
    task: u64,
    witness: &ImplementationWitness,
) -> Result<(), String> {
    if witness.schema_version != 1 || witness.task != task {
        return Err(format!(
            "implemented topology witness identity mismatch for Task {task}"
        ));
    }
    let required = IMPLEMENTATION_TASK_PACKAGES
        .iter()
        .find_map(|(owner, packages)| (*owner == task).then_some(*packages))
        .unwrap_or_default();
    let witnessed = witness
        .packages
        .iter()
        .map(|package| package.package.as_str())
        .collect::<BTreeSet<_>>();
    if witnessed.len() != witness.packages.len()
        || witnessed != required.iter().copied().collect::<BTreeSet<_>>()
    {
        return Err(format!("inexact implemented package set for Task {task}"));
    }
    let matrix_packages = matrix
        .package_ownership
        .iter()
        .map(|row| row.package.as_str())
        .collect::<BTreeSet<_>>();
    let packages = metadata
        .packages
        .iter()
        .map(|package| (package.name.as_str(), package))
        .collect::<BTreeMap<_, _>>();
    for package in &witness.packages {
        if !matrix_packages.contains(package.package.as_str()) {
            return Err(format!("unknown projected package {}", package.package));
        }
        let actual = packages
            .get(package.package.as_str())
            .ok_or_else(|| format!("Cargo metadata lacks {}", package.package))?;
        let package_root = std::path::Path::new(&actual.manifest_path)
            .parent()
            .ok_or_else(|| format!("manifest has no parent for {}", package.package))?;
        let mut source_files = Vec::new();
        collect_rust_sources(workspace, package_root, &mut source_files)?;
        source_files.sort();
        let mut declared_sources = package.source_files.clone();
        declared_sources.sort();
        declared_sources.dedup();
        if package.source_files.is_empty()
            || declared_sources.len() != package.source_files.len()
            || declared_sources != source_files
        {
            return Err(format!(
                "source-file census mismatch for {}",
                package.package
            ));
        }
        let actual_dependencies = actual
            .dependencies
            .iter()
            .filter(|dependency| dependency.path.is_some())
            .map(|dependency| {
                (
                    dependency.name.as_str(),
                    dependency.kind.as_deref().unwrap_or("normal"),
                )
            })
            .collect::<BTreeSet<_>>();
        let declared_dependencies = package
            .dependencies
            .iter()
            .map(|dependency| (dependency.package.as_str(), dependency.kind.as_str()))
            .collect::<BTreeSet<_>>();
        if declared_dependencies.len() != package.dependencies.len()
            || declared_dependencies != actual_dependencies
        {
            return Err(format!(
                "dependency census mismatch for {}",
                package.package
            ));
        }
        if package.features != actual.features {
            return Err(format!("feature census mismatch for {}", package.package));
        }
    }
    Ok(())
}

fn validate_configured_implementation_task(
    workspace: &std::path::Path,
    task: u64,
) -> Result<(), String> {
    let (matrix, metadata) = checked_matrix_and_metadata(workspace);
    let tasks = if task == 40 {
        IMPLEMENTATION_TASK_PACKAGES
            .iter()
            .map(|(task, _)| *task)
            .collect::<Vec<_>>()
    } else if IMPLEMENTATION_TASK_PACKAGES
        .iter()
        .any(|(owner, _)| *owner == task)
    {
        vec![task]
    } else {
        Vec::new()
    };
    for task in tasks {
        let path = workspace.join(format!("plugin-api/implemented-topology/task-{task}.toml"));
        let witness: ImplementationWitness = toml::from_str(
            &std::fs::read_to_string(&path)
                .map_err(|error| format!("cannot read {}: {error}", path.display()))?,
        )
        .map_err(|error| format!("invalid {}: {error}", path.display()))?;
        validate_implementation_witness(workspace, &matrix, &metadata, task, &witness)?;
    }
    Ok(())
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
        if row.owner.is_empty() || row.projection_basis.trim().is_empty() {
            return Err(format!(
                "empty owner or projection basis for {}",
                row.package
            ));
        }
        match row.review_state.as_str() {
            "reviewed_projection" => {
                let source = row.source_package.as_deref().ok_or_else(|| {
                    format!(
                        "projected package {} lacks a Task-1 package witness",
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
        review_state: "reviewed_projection".to_owned(),
        source_package: Some("aiperf-runtime".to_owned()),
        dependencies: vec![OwnedDependency {
            package: "aiperf-plugin-api".to_owned(),
            kind: "normal".to_owned(),
            justification: "fixture witness".to_owned(),
        }],
        projection_basis: "Task-1 fixture package projection".to_owned(),
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
fn projection_requires_task1_package_witnesses_and_exact_grpc_splits() {
    let (matrix, baseline) = fixture();

    let mut unwitnessed = matrix.clone();
    unwitnessed.package_ownership[0].source_package = None;
    assert!(
        validate_projection(&unwitnessed, &baseline)
            .unwrap_err()
            .contains("lacks a Task-1 package witness")
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
    unevidenced.package_ownership[0].projection_basis.clear();
    assert!(
        validate_projection(&unevidenced, &baseline)
            .unwrap_err()
            .contains("empty owner or projection basis")
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
fn every_plugin_dependency_and_baseline_feature_has_one_reviewed_projection() {
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
            "unprojected dependency edge for {package_name}"
        );
    }
}

#[test]
fn implementation_task_map_assigns_every_projected_package_once() {
    let (matrix, _) = fixture_from_checked_files();
    let mut assigned = BTreeSet::new();
    for (_, packages) in IMPLEMENTATION_TASK_PACKAGES {
        for package in *packages {
            assert!(assigned.insert(*package), "duplicate implementation owner");
        }
    }
    assert_eq!(
        assigned,
        matrix
            .package_ownership
            .iter()
            .map(|row| row.package.as_str())
            .collect()
    );
}

#[test]
fn implemented_witness_matches_real_package_sources_dependencies_and_features() {
    let root = workspace_root();
    let (matrix, metadata) = checked_matrix_and_metadata(&root);
    let witness = ImplementationWitness {
        schema_version: 1,
        task: 4,
        packages: vec![ImplementedPackageWitness {
            package: "aiperf-core".to_owned(),
            source_files: vec![
                    "core/src/artifact.rs".to_owned(),
                    "core/src/capture.rs".to_owned(),
                    "core/src/clock.rs".to_owned(),
                    "core/src/dispatch.rs".to_owned(),
                    "core/src/endpoint.rs".to_owned(),
                    "core/src/histogram.rs".to_owned(),
                    "core/src/lib.rs".to_owned(),
                    "core/src/measure/error.rs".to_owned(),
                    "core/src/measure/eventstream.rs".to_owned(),
                    "core/src/measure/mod.rs".to_owned(),
                    "core/src/measure/record.rs".to_owned(),
                    "core/src/measure/response.rs".to_owned(),
                    "core/src/measure/reuse.rs".to_owned(),
                    "core/src/measure/sse.rs".to_owned(),
                    "core/src/measure/trace.rs".to_owned(),
                    "core/src/report.rs".to_owned(),
                    "core/src/services.rs".to_owned(),
                    "core/tests/histogram_contract.rs".to_owned(),
                    "core/tests/public_contract.rs".to_owned(),
                ],
            dependencies: vec![],
            features: BTreeMap::new(),
        }],
    };
    validate_implementation_witness(&root, &matrix, &metadata, 4, &witness)
        .expect("exact core witness");

    let mut missing_source = witness.clone();
    missing_source.packages[0].source_files.clear();
    assert!(
        validate_implementation_witness(&root, &matrix, &metadata, 4, &missing_source)
            .unwrap_err()
            .contains("source-file census")
    );

    let mut wrong_dependency = witness;
    wrong_dependency.packages[0]
        .dependencies
        .push(OwnedDependency {
            package: "aiperf-plugin-api".to_owned(),
            kind: "normal".to_owned(),
            justification: "not present in Cargo metadata".to_owned(),
        });
    assert!(
        validate_implementation_witness(&root, &matrix, &metadata, 4, &wrong_dependency)
            .unwrap_err()
            .contains("dependency census")
    );
}

#[test]
fn configured_task_requires_its_implemented_topology_witness() {
    let Some(task) = std::env::var("AIPERF_PLUGIN_TOPOLOGY_TASK")
        .ok()
        .and_then(|task| task.parse::<u64>().ok())
    else {
        return;
    };
    validate_configured_implementation_task(&workspace_root(), task)
        .expect("configured implementation topology witness");
}
