// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract for the reviewed native-plugin package and feature projection, and
//! for the per-task implemented-topology witnesses that make it real.
//!
//! The projection half binds Task 3's reviewed ownership matrix to the exact
//! Task-1 Cargo census. The witness half is not projection: it derives each
//! finalizing task's package set, Rust source census, local dependency edges,
//! and feature map from the live workspace and rejects any drift. A witness may
//! mirror its projection row's justification prose, but only the exact
//! `(package, kind)` edge is ever validated against Cargo.

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
    dependencies: Vec<WitnessDependency>,
    features: BTreeMap<String, Vec<String>>,
}

/// A witness dependency edge. Only the exact `(package, kind)` pair is
/// validated against live Cargo metadata; the projection's `OwnedDependency`
/// owns the reviewed justification prose. The authored witnesses mirror their
/// projection row verbatim, so the prose is accepted here and carried inert,
/// with a blank one refused so a mirrored justification cannot rot into an
/// empty string.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct WitnessDependency {
    package: String,
    kind: String,
    #[serde(default)]
    justification: Option<String>,
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

/// Renders the symmetric difference of a hand-authored census and the census
/// derived from the live workspace. Twenty later tasks author these by hand, so
/// a failure has to say which entries are absent and which are surplus.
fn census_difference<T: Ord + std::fmt::Debug>(
    declared: &BTreeSet<T>,
    actual: &BTreeSet<T>,
) -> String {
    format!(
        "missing {:?}, unexpected {:?}",
        actual.difference(declared).collect::<Vec<_>>(),
        declared.difference(actual).collect::<Vec<_>>()
    )
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
    if witnessed.len() != witness.packages.len() {
        return Err(format!(
            "duplicate implemented package rows for Task {task}"
        ));
    }
    let required = required.iter().copied().collect::<BTreeSet<_>>();
    if witnessed != required {
        return Err(format!(
            "inexact implemented package set for Task {task}: {}",
            census_difference(&witnessed, &required)
        ));
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
        let actual_sources = source_files.into_iter().collect::<BTreeSet<_>>();
        let declared_sources = package
            .source_files
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        if package.source_files.is_empty() {
            return Err(format!("empty source-file census for {}", package.package));
        }
        if declared_sources.len() != package.source_files.len() {
            return Err(format!(
                "duplicate source-file census rows for {}",
                package.package
            ));
        }
        if declared_sources != actual_sources {
            return Err(format!(
                "source-file census mismatch for {}: {}",
                package.package,
                census_difference(&declared_sources, &actual_sources)
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
        for dependency in &package.dependencies {
            if dependency
                .justification
                .as_ref()
                .is_some_and(|prose| prose.trim().is_empty())
            {
                return Err(format!(
                    "empty dependency justification in {}",
                    package.package
                ));
            }
        }
        let declared_dependencies = package
            .dependencies
            .iter()
            .map(|dependency| (dependency.package.as_str(), dependency.kind.as_str()))
            .collect::<BTreeSet<_>>();
        if declared_dependencies.len() != package.dependencies.len() {
            return Err(format!(
                "duplicate dependency census rows for {}",
                package.package
            ));
        }
        if declared_dependencies != actual_dependencies {
            return Err(format!(
                "dependency census mismatch for {}: {}",
                package.package,
                census_difference(&declared_dependencies, &actual_dependencies)
            ));
        }
        if package.features != actual.features {
            return Err(format!(
                "feature census mismatch for {}: {}",
                package.package,
                census_difference(
                    &package.features.iter().collect::<BTreeSet<_>>(),
                    &actual.features.iter().collect::<BTreeSet<_>>()
                )
            ));
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

/// Parses the plan's witness table. The table is what a future task author
/// reads before writing a witness, so it is bound to the constant the gate
/// enforces rather than left to drift as prose.
fn planned_witness_tasks(plan: &str) -> Vec<(u64, Vec<String>)> {
    plan.split_once("| Witness task | Exact packages |")
        .expect("plan must document the witness table")
        .1
        .lines()
        .skip(1)
        .map(str::trim)
        .take_while(|line| line.starts_with('|'))
        .filter_map(|line| {
            let cells = line.split('|').map(str::trim).collect::<Vec<_>>();
            let task = cells.get(1)?.parse::<u64>().ok()?;
            let packages = cells
                .get(2)?
                .split(',')
                .map(|package| package.trim().trim_matches('`').to_owned())
                .collect();
            Some((task, packages))
        })
        .collect()
}

#[test]
fn planned_witness_table_matches_the_implementation_task_map() {
    let plan_path = workspace_root()
        .join("../docs/superpowers/plans/2026-08-26-native-rust-runtime-plugins-implementation.md");
    let plan = std::fs::read_to_string(&plan_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", plan_path.display()));
    let enforced = IMPLEMENTATION_TASK_PACKAGES
        .iter()
        .map(|(task, packages)| {
            (
                *task,
                packages
                    .iter()
                    .map(|package| (*package).to_owned())
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(
        planned_witness_tasks(&plan),
        enforced,
        "the plan's witness table and IMPLEMENTATION_TASK_PACKAGES must agree exactly"
    );
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

/// Every authored witness under `plugin-api/implemented-topology/` has to load
/// under this file's witness DTOs. The authored corpus mirrors the projection
/// matrix's dependency rows verbatim, justification prose included, so both the
/// mirrored and the bare edge shape must load. The inline fixtures keep that
/// binding non-vacuous on a checkout that carries no authored witness yet.
#[test]
fn authored_implemented_topology_witnesses_load_under_the_witness_schema() {
    const MIRRORED: &str = r#"
schema_version = 1
task = 5
packages = [
  { package = "aiperf-plugin-api", source_files = ["plugin-api/src/lib.rs"], dependencies = [{ package = "aiperf-core", kind = "normal", justification = "API boundary values are owned by aiperf-core" }], features = {} }
]
"#;
    const BARE: &str = r#"
schema_version = 1
task = 5
packages = [
  { package = "aiperf-plugin-api", source_files = ["plugin-api/src/lib.rs"], dependencies = [{ package = "aiperf-core", kind = "normal" }], features = {} }
]
"#;
    for text in [MIRRORED, BARE] {
        let witness: ImplementationWitness =
            toml::from_str(text).expect("both authored dependency-row shapes load");
        assert_eq!(witness.packages[0].dependencies.len(), 1);
    }

    let directory = workspace_root().join("plugin-api/implemented-topology");
    for (task, packages) in IMPLEMENTATION_TASK_PACKAGES {
        let path = directory.join(format!("task-{task}.toml"));
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let witness: ImplementationWitness = toml::from_str(&text)
            .unwrap_or_else(|error| panic!("cannot load {}: {error}", path.display()));
        assert_eq!(
            (witness.schema_version, witness.task),
            (1, *task),
            "{} must declare schema 1 and its own task",
            path.display()
        );
        assert_eq!(
            witness
                .packages
                .iter()
                .map(|row| row.package.as_str())
                .collect::<BTreeSet<_>>(),
            packages.iter().copied().collect::<BTreeSet<_>>(),
            "{} must witness exactly its task's packages",
            path.display()
        );
    }
}

/// A self-contained `aiperf-core` package tree, Cargo projection, and exact
/// witness. Witness validation is behavior, so its positive and negative arms
/// are asserted here rather than against the live workspace, whose contents 36
/// later tasks rewrite.
struct SyntheticPackage {
    _directory: tempfile::TempDir,
    workspace: PathBuf,
    matrix: OwnershipMatrix,
    metadata: Metadata,
    witness: ImplementationWitness,
}

fn synthetic_core_package() -> SyntheticPackage {
    let directory = tempfile::tempdir().expect("temporary workspace");
    let workspace = directory.path().to_path_buf();
    let sources = [
        "core/src/lib.rs",
        "core/src/clock.rs",
        "core/tests/public_contract.rs",
    ];
    for source in sources {
        let path = workspace.join(source);
        std::fs::create_dir_all(path.parent().expect("source has a parent"))
            .expect("source directory");
        std::fs::write(&path, "// synthetic\n").expect("source file");
    }
    std::fs::write(workspace.join("core/Cargo.toml"), "# synthetic\n").expect("manifest");
    // A non-Rust sibling proves the census walks by extension, not by entry.
    std::fs::write(workspace.join("core/README.md"), "synthetic\n").expect("sibling");

    let features = BTreeMap::from([("default".to_owned(), vec!["clock".to_owned()])]);
    let metadata = Metadata {
        packages: vec![MetadataPackage {
            id: "aiperf-core 0.0.0".to_owned(),
            name: "aiperf-core".to_owned(),
            dependencies: vec![MetadataDependency {
                name: "aiperf-plugin-api".to_owned(),
                kind: None,
                path: Some(workspace.join("plugin-api").display().to_string()),
            }],
            features: features.clone(),
            manifest_path: workspace.join("core/Cargo.toml").display().to_string(),
        }],
        workspace_members: vec!["aiperf-core 0.0.0".to_owned()],
        metadata: serde_json::Value::Null,
    };
    let (mut matrix, _) = fixture();
    matrix.package_ownership[0].package = "aiperf-core".to_owned();
    let mut source_files = sources.map(str::to_owned).to_vec();
    source_files.sort();
    let witness = ImplementationWitness {
        schema_version: 1,
        task: 4,
        packages: vec![ImplementedPackageWitness {
            package: "aiperf-core".to_owned(),
            source_files,
            dependencies: vec![WitnessDependency {
                package: "aiperf-plugin-api".to_owned(),
                kind: "normal".to_owned(),
                justification: None,
            }],
            features,
        }],
    };
    SyntheticPackage {
        _directory: directory,
        workspace,
        matrix,
        metadata,
        witness,
    }
}

impl SyntheticPackage {
    fn validate(&self, witness: &ImplementationWitness) -> Result<(), String> {
        validate_implementation_witness(
            &self.workspace,
            &self.matrix,
            &self.metadata,
            witness.task,
            witness,
        )
    }
}

#[test]
fn implemented_witness_census_failures_name_the_exact_difference() {
    let synthetic = synthetic_core_package();
    synthetic
        .validate(&synthetic.witness)
        .expect("exact synthetic witness");

    let mut stale_source = synthetic.witness.clone();
    stale_source.packages[0].source_files =
        vec!["core/src/gone.rs".to_owned(), "core/src/lib.rs".to_owned()];
    let error = synthetic.validate(&stale_source).unwrap_err();
    assert!(
        error.contains("core/src/gone.rs")
            && error.contains("core/src/clock.rs")
            && error.contains("core/tests/public_contract.rs"),
        "source census failure must name both sides: {error}"
    );

    let mut empty_source = synthetic.witness.clone();
    empty_source.packages[0].source_files.clear();
    assert!(
        synthetic
            .validate(&empty_source)
            .unwrap_err()
            .contains("source-file census"),
        "an empty census must never satisfy a package that has sources"
    );

    let mut duplicate_source = synthetic.witness.clone();
    duplicate_source.packages[0]
        .source_files
        .push("core/src/lib.rs".to_owned());
    assert!(
        synthetic
            .validate(&duplicate_source)
            .unwrap_err()
            .contains("duplicate source-file census"),
        "a duplicated census row must be rejected as a duplicate"
    );

    let mut wrong_dependency = synthetic.witness.clone();
    wrong_dependency.packages[0].dependencies[0].package = "aiperf-core-utils".to_owned();
    let error = synthetic.validate(&wrong_dependency).unwrap_err();
    assert!(
        error.contains("aiperf-core-utils") && error.contains("aiperf-plugin-api"),
        "dependency census failure must name both sides: {error}"
    );

    let mut wrong_feature = synthetic.witness.clone();
    wrong_feature.packages[0]
        .features
        .insert("extra".to_owned(), vec![]);
    let error = synthetic.validate(&wrong_feature).unwrap_err();
    assert!(
        error.contains("extra"),
        "feature census failure must name the differing entry: {error}"
    );

    let mut wrong_packages = synthetic.witness.clone();
    wrong_packages.packages[0].package = "aiperf-plugin-api".to_owned();
    let error = synthetic.validate(&wrong_packages).unwrap_err();
    assert!(
        error.contains("aiperf-plugin-api") && error.contains("aiperf-core"),
        "package-set failure must name both sides: {error}"
    );
}

/// Resolves the implementation task the gate was configured for. An absent
/// variable is the only legitimate skip: a present value that is not a task
/// number is a gate failure, so a typo cannot report a green topology gate
/// having validated no witness at all.
fn configured_topology_task(value: Option<std::ffi::OsString>) -> Result<Option<u64>, String> {
    let Some(raw) = value else {
        return Ok(None);
    };
    let text = raw.to_str().ok_or_else(|| {
        format!(
            "AIPERF_PLUGIN_TOPOLOGY_TASK={} is not a task number",
            raw.to_string_lossy()
        )
    })?;
    text.parse::<u64>().map(Some).map_err(|error| {
        format!("AIPERF_PLUGIN_TOPOLOGY_TASK={text} is not a task number: {error}")
    })
}

#[test]
fn configured_task_requires_its_implemented_topology_witness() {
    let configured = configured_topology_task(std::env::var_os("AIPERF_PLUGIN_TOPOLOGY_TASK"))
        .expect("AIPERF_PLUGIN_TOPOLOGY_TASK must name a task number when it is set");
    let Some(task) = configured else {
        return;
    };
    validate_configured_implementation_task(&workspace_root(), task)
        .expect("configured implementation topology witness");
}

#[test]
fn configured_topology_task_rejects_a_present_but_invalid_value() {
    assert_eq!(configured_topology_task(None), Ok(None));
    assert_eq!(
        configured_topology_task(Some(std::ffi::OsString::from("4"))),
        Ok(Some(4))
    );
    for invalid in ["task-4", "12-elf", ""] {
        assert!(
            configured_topology_task(Some(std::ffi::OsString::from(invalid)))
                .unwrap_err()
                .contains("is not a task number"),
            "{invalid} must fail the gate rather than skip it"
        );
    }
}

/// Builds a witness for one package entirely from the live workspace: the
/// source census from the package's own tree, the dependency census from its
/// path dependencies, and the feature table verbatim from Cargo metadata.
///
/// Deriving the expectation is what keeps this usable past Task 4. A witness
/// with a hand-written source list would have to be re-authored every time a
/// task adds a file to the package it owns.
fn witness_from_live_metadata(
    workspace: &std::path::Path,
    metadata: &Metadata,
    task: u64,
    package: &str,
) -> ImplementationWitness {
    let actual = metadata
        .packages
        .iter()
        .find(|candidate| candidate.name == package)
        .unwrap_or_else(|| panic!("Cargo metadata must describe {package}"));
    let package_root = std::path::Path::new(&actual.manifest_path)
        .parent()
        .unwrap_or_else(|| panic!("manifest for {package} must have a parent"));
    let mut source_files = Vec::new();
    collect_rust_sources(workspace, package_root, &mut source_files)
        .unwrap_or_else(|error| panic!("cannot census {package}: {error}"));
    let dependencies = actual
        .dependencies
        .iter()
        .filter(|dependency| dependency.path.is_some())
        .map(|dependency| WitnessDependency {
            package: dependency.name.clone(),
            kind: dependency
                .kind
                .clone()
                .unwrap_or_else(|| "normal".to_owned()),
            justification: None,
        })
        .collect();
    ImplementationWitness {
        schema_version: 1,
        task,
        packages: vec![ImplementedPackageWitness {
            package: package.to_owned(),
            source_files,
            dependencies,
            features: actual.features.clone(),
        }],
    }
}

/// Runs the witness validator against the real workspace on every `cargo test`,
/// with no environment selector in front of it.
///
/// The task-selected gate reaches `checked_matrix_and_metadata` only when
/// `AIPERF_PLUGIN_TOPOLOGY_TASK` names an owning task, so without this test a
/// bare `cargo test` never executes `cargo metadata --locked`, never walks a
/// real package tree, and never proves the projection is exact against checked
/// files. The negative arms below are what make the positive arm meaningful:
/// they prove the census actually compared something.
#[test]
fn implemented_witness_validates_against_the_live_workspace() {
    let workspace = workspace_root();
    let (matrix, metadata) = checked_matrix_and_metadata(&workspace);
    let (task, packages) = IMPLEMENTATION_TASK_PACKAGES
        .first()
        .expect("the implementation task map is never empty");
    let package = packages
        .first()
        .expect("every implementation task owns at least one package");
    let witness = witness_from_live_metadata(&workspace, &metadata, *task, package);
    assert!(
        !witness.packages[0].source_files.is_empty(),
        "{package} must own at least one Rust source file"
    );
    validate_implementation_witness(&workspace, &matrix, &metadata, *task, &witness)
        .expect("a witness derived from the live workspace must validate");

    let mut empty_sources = witness.clone();
    empty_sources.packages[0].source_files.clear();
    assert!(
        validate_implementation_witness(&workspace, &matrix, &metadata, *task, &empty_sources)
            .expect_err("an empty source census must be refused")
            .contains("empty source-file census"),
    );

    let mut surplus_source = witness.clone();
    surplus_source.packages[0]
        .source_files
        .push("core/src/this-file-does-not-exist.rs".to_owned());
    assert!(
        validate_implementation_witness(&workspace, &matrix, &metadata, *task, &surplus_source)
            .expect_err("a surplus source row must be refused")
            .contains("source-file census mismatch"),
    );

    let mut surplus_dependency = witness;
    surplus_dependency.packages[0]
        .dependencies
        .push(WitnessDependency {
            package: "aiperf-plugin-api".to_owned(),
            kind: "normal".to_owned(),
            justification: None,
        });
    assert!(
        validate_implementation_witness(&workspace, &matrix, &metadata, *task, &surplus_dependency)
            .expect_err("a dependency absent from Cargo metadata must be refused")
            .contains("dependency census"),
    );
}
