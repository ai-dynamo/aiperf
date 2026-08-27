// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Foundation policy checks for the provisional native plugin workspace.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::Command,
};

use serde::Deserialize;

const FORBIDDEN: &[&str] = &[
    "aiperf-runtime",
    "tokio",
    "hyper",
    "tonic",
    "clap",
    "parquet",
    "opentelemetry",
    "reqwest",
    "wandb",
];

const DISTRIBUTABLE_PACKAGES: &[(&str, &str)] = &[
    ("aiperf-plugin-export-basic", "plugins/export-basic"),
    ("aiperf-plugin-export-parquet", "plugins/export-parquet"),
    ("aiperf-plugin-export-mlflow", "plugins/export-mlflow"),
    ("aiperf-plugin-export-wandb", "plugins/export-wandb"),
    ("aiperf-plugin-export-otel", "plugins/export-otel"),
    ("aiperf-plugin-endpoints", "plugins/endpoints"),
    ("aiperf-plugin-transport-http", "plugins/transport-http"),
    ("aiperf-plugin-transport-grpc", "plugins/transport-grpc"),
    (
        "aiperf-plugin-transport-websocket",
        "plugins/transport-websocket",
    ),
    (
        "aiperf-plugin-transport-dry-run",
        "plugins/transport-dry-run",
    ),
    (
        "aiperf-plugin-transport-dynosim",
        "plugins/transport-dynosim",
    ),
];

const FOUNDATION_PACKAGES: &[(&str, &str)] = &[
    ("aiperf-core", "core"),
    ("aiperf-plugin-api", "plugin-api"),
    ("aiperf-plugin-sdk", "plugin-sdk"),
    ("aiperf-endpoint-sdk", "endpoint-sdk"),
    ("aiperf-transport-sdk", "transport-sdk"),
    ("aiperf-export-sdk", "export-sdk"),
    ("aiperf-plugin-sdk-macros", "plugin-sdk-macros"),
    ("aiperf-plugin-host", "plugin-host"),
    ("aiperf-plugin-conformance", "plugin-conformance"),
    ("aiperf-plugin-test-support", "plugin-test-support"),
    ("aiperf-plugin-packaging-tests", "plugin-packaging-tests"),
    ("aiperf-plugin-perf", "plugin-perf"),
    (
        "aiperf-plugin-static-comparator",
        "plugin-static-comparator",
    ),
    ("aiperf-allocator-provider", "allocator-provider"),
    ("aiperf-allocator-shim", "allocator-shim"),
];

#[derive(Debug, Deserialize)]
struct Metadata {
    packages: Vec<Package>,
    workspace_members: Vec<String>,
    workspace_root: String,
}

#[derive(Debug, Deserialize)]
struct Package {
    id: String,
    name: String,
    manifest_path: String,
    publish: Option<Vec<String>>,
    dependencies: Vec<Dependency>,
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct Dependency {
    name: String,
    kind: Option<String>,
    source: Option<String>,
}

#[derive(Deserialize)]
struct BaselineTopology {
    workspace_packages: Vec<BaselinePackage>,
}
#[derive(Deserialize)]
struct BaselinePackage {
    name: String,
    direct_dependencies: Vec<BaselineDependency>,
    features: Vec<String>,
}
#[derive(Deserialize)]
struct BaselineDependency {
    name: String,
    kind: String,
    is_workspace: bool,
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("plugin-api is directly below the Rust workspace root")
        .to_path_buf()
}

fn metadata(root: &Path) -> Metadata {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .expect("cargo metadata must execute");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("cargo metadata must emit JSON")
}

fn standalone_metadata(root: &Path) -> Metadata {
    let manifest = root.join("tests/plugin-third-party/Cargo.toml");
    let output = Command::new("cargo")
        .args([
            "metadata",
            "--locked",
            "--format-version",
            "1",
            "--no-deps",
            "--manifest-path",
        ])
        .arg(&manifest)
        .current_dir(root)
        .output()
        .expect("standalone cargo metadata must execute");
    assert!(
        output.status.success(),
        "standalone metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("standalone metadata must emit JSON")
}

fn package_map(metadata: Metadata) -> BTreeMap<String, Package> {
    let members = metadata
        .workspace_members
        .into_iter()
        .collect::<BTreeSet<_>>();
    metadata
        .packages
        .into_iter()
        .filter(|package| members.contains(&package.id))
        .map(|package| (package.name.clone(), package))
        .collect()
}

fn normal_and_build_dependencies(package: &Package) -> BTreeSet<&str> {
    package
        .dependencies
        .iter()
        .filter(|dependency| dependency.kind.as_deref() != Some("dev"))
        .map(|dependency| dependency.name.as_str())
        .collect()
}

fn dev_dependencies(package: &Package) -> BTreeSet<&str> {
    package
        .dependencies
        .iter()
        .filter(|dependency| dependency.kind.as_deref() == Some("dev"))
        .map(|dependency| dependency.name.as_str())
        .collect()
}

#[test]
fn workspace_and_template_policy() {
    // Removing a shell, moving it out of the parent workspace, or publishing the
    // test helper would make the public package boundary incomplete.
    let root = workspace_root();
    let parent_metadata = metadata(&root);
    assert_eq!(Path::new(&parent_metadata.workspace_root), root);
    let packages = package_map(parent_metadata);

    let baseline: BaselineTopology = serde_json::from_slice(
        &std::fs::read(
            root.parent()
                .expect("repo root")
                .join("artifacts/native-plugin-baseline/package-topology.json"),
        )
        .expect("Task 1 package topology must exist"),
    )
    .expect("Task 1 package topology must be JSON");
    let baseline_names = baseline
        .workspace_packages
        .iter()
        .map(|package| package.name.clone())
        .collect::<BTreeSet<_>>();
    let expected_names = baseline_names
        .into_iter()
        .chain(
            FOUNDATION_PACKAGES
                .iter()
                .chain(DISTRIBUTABLE_PACKAGES)
                .map(|(name, _)| (*name).to_owned()),
        )
        .collect::<BTreeSet<_>>();
    assert_eq!(
        packages.keys().cloned().collect::<BTreeSet<_>>(),
        expected_names
    );
    let lock: toml::Value = std::fs::read_to_string(root.join("Cargo.lock"))
        .expect("parent lock")
        .parse()
        .expect("parent lock TOML");
    let lock_names = lock["package"]
        .as_array()
        .expect("parent lock packages")
        .iter()
        .filter(|package| package.get("source").is_none())
        .map(|package| {
            package["name"]
                .as_str()
                .expect("lock package name")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(lock_names, packages.keys().cloned().collect());
    for captured in &baseline.workspace_packages {
        let current = &packages[&captured.name];
        let expected = captured
            .direct_dependencies
            .iter()
            .map(|edge| (edge.name.as_str(), edge.kind.as_str(), edge.is_workspace))
            .collect::<BTreeSet<_>>();
        let actual = current
            .dependencies
            .iter()
            .map(|edge| {
                (
                    edge.name.as_str(),
                    edge.kind.as_deref().unwrap_or("normal"),
                    edge.source.is_none(),
                )
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected, "baseline DAG drift for {}", captured.name);
        let mut expected_features = captured.features.iter().cloned().collect::<BTreeSet<_>>();
        if captured.name == "aiperf-e2e-tests" {
            expected_features.extend(["grpc", "websocket", "dynosim"].map(str::to_owned));
        }
        assert_eq!(
            current.features.keys().cloned().collect::<BTreeSet<_>>(),
            expected_features,
            "baseline feature drift for {}",
            captured.name
        );
    }

    for (name, relative_manifest_dir) in FOUNDATION_PACKAGES.iter().chain(DISTRIBUTABLE_PACKAGES) {
        let package = packages
            .get(*name)
            .unwrap_or_else(|| panic!("missing workspace shell {name}"));
        assert_eq!(
            Path::new(&package.manifest_path),
            root.join(relative_manifest_dir).join("Cargo.toml"),
            "{name} must retain its assigned package directory"
        );
    }

    for (name, relative_manifest_dir) in DISTRIBUTABLE_PACKAGES {
        let template = root.join(relative_manifest_dir).join("plugins.yaml.in");
        assert!(
            template.is_file(),
            "{name} must retain its distributable template"
        );
        assert!(
            std::fs::read_to_string(template)
                .expect("template text")
                .starts_with("# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n# SPDX-License-Identifier: Apache-2.0\n"),
            "{name} template must retain its SPDX header"
        );
    }
    for (_, relative_manifest_dir) in FOUNDATION_PACKAGES {
        assert!(
            !root
                .join(relative_manifest_dir)
                .join("plugins.yaml.in")
                .exists(),
            "non-distributable shell {relative_manifest_dir} must not look installable"
        );
    }

    let standalone = root.join("tests/plugin-third-party/Cargo.toml");
    assert!(standalone.is_file(), "third-party exemplar must exist");
    assert!(
        !packages
            .values()
            .any(|package| Path::new(&package.manifest_path) == standalone),
        "third-party exemplar must remain outside the parent workspace"
    );
    let standalone_metadata = standalone_metadata(&root);
    assert_eq!(
        Path::new(&standalone_metadata.workspace_root),
        root.join("tests/plugin-third-party")
    );
    let standalone_packages = package_map(standalone_metadata);
    assert_eq!(standalone_packages.len(), 1);
    assert!(standalone_packages.contains_key("aiperf-plugin-third-party-example"));
    let standalone_lock: toml::Value =
        std::fs::read_to_string(root.join("tests/plugin-third-party/Cargo.lock"))
            .expect("standalone lock")
            .parse()
            .expect("standalone lock TOML");
    assert_eq!(standalone_lock["version"].as_integer(), Some(4));
    assert_eq!(
        standalone_lock["package"]
            .as_array()
            .expect("standalone packages")
            .len(),
        1
    );
    assert_eq!(
        standalone_lock["package"][0]["name"].as_str(),
        Some("aiperf-plugin-third-party-example")
    );

    let api = &packages["aiperf-plugin-api"];
    let allowlist_path = root.join("plugin-api/api-allowlist.toml");
    let allowlist: toml::Value = std::fs::read_to_string(&allowlist_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", allowlist_path.display()))
        .parse()
        .expect("API allowlist must be valid TOML");
    assert_eq!(allowlist["schema_version"].as_integer(), Some(1));
    assert_eq!(
        allowlist
            .as_table()
            .expect("allowlist table")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "schema_version",
            "allowed_dependencies",
            "allowed_std_modules"
        ])
    );
    let allowed = allowlist["allowed_dependencies"]
        .as_array()
        .expect("API allowlist dependencies must be an array")
        .iter()
        .map(|value| {
            value
                .as_str()
                .expect("allowlist dependency must be a string")
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        allowlist["allowed_dependencies"]
            .as_array()
            .expect("dependencies")
            .len(),
        allowed.len()
    );
    let api_dependencies = normal_and_build_dependencies(api);
    assert!(api_dependencies.is_subset(&allowed));
    for forbidden in FORBIDDEN {
        assert!(!api_dependencies.contains(forbidden));
    }
    assert_eq!(api_dependencies, BTreeSet::from(["aiperf-core"]));
    assert_eq!(
        allowed,
        BTreeSet::from(["aiperf-core", "blake3", "serde", "serde_json", "thiserror"])
    );
    assert_eq!(
        allowlist["allowed_std_modules"]
            .as_array()
            .expect("API standard-library allowlist")
            .iter()
            .map(|value| value.as_str().expect("standard-library module"))
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["alloc", "core", "std"])
    );
    assert_eq!(
        allowlist["allowed_std_modules"]
            .as_array()
            .expect("standard modules")
            .len(),
        3
    );

    let host_dependencies = normal_and_build_dependencies(&packages["aiperf-plugin-host"]);
    assert!(host_dependencies.is_subset(&BTreeSet::from([
        "aiperf-plugin-api",
        "aiperf-core",
        "aiperf-plugin-sdk",
    ])));
    assert!(!host_dependencies.contains("aiperf-runtime"));
    assert!(
        !normal_and_build_dependencies(&packages["aiperf-runtime"]).contains("aiperf-plugin-host")
    );

    let test_support = &packages["aiperf-plugin-test-support"];
    assert_eq!(test_support.publish, Some(Vec::new()));
    assert_eq!(
        normal_and_build_dependencies(test_support),
        BTreeSet::from(["aiperf-core", "tempfile"])
    );
    assert!(!normal_and_build_dependencies(api).contains("aiperf-plugin-test-support"));
    assert!(
        dev_dependencies(&packages["aiperf-export-sdk"]).contains("aiperf-plugin-test-support"),
        "export SDK must reserve its independent-leaf test seam"
    );
    for package in packages.values() {
        assert!(
            !normal_and_build_dependencies(package).contains("aiperf-plugin-test-support"),
            "{} must not have a production dependency on test support",
            package.name
        );
    }
    let expected_new_edges = BTreeSet::from([
        ("aiperf-plugin-api", "aiperf-core", "normal"),
        ("aiperf-plugin-test-support", "aiperf-core", "normal"),
        ("aiperf-plugin-test-support", "tempfile", "normal"),
        ("aiperf-export-sdk", "aiperf-plugin-test-support", "dev"),
    ]);
    let new_names = FOUNDATION_PACKAGES
        .iter()
        .chain(DISTRIBUTABLE_PACKAGES)
        .map(|(name, _)| *name)
        .collect::<BTreeSet<_>>();
    let new_edges = packages
        .values()
        .filter(|package| new_names.contains(package.name.as_str()))
        .flat_map(|package| {
            package
                .dependencies
                .iter()
                .filter(|dependency| {
                    dependency.kind.as_deref() != Some("dev")
                        || dependency.name == "aiperf-plugin-test-support"
                })
                .map(|dependency| {
                    (
                        package.name.as_str(),
                        dependency.name.as_str(),
                        dependency.kind.as_deref().unwrap_or("normal"),
                    )
                })
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(new_edges, expected_new_edges);
    assert_eq!(
        packages["aiperf-e2e-tests"].features.get("grpc"),
        Some(&vec![
            "aiperf-runtime/grpc".to_owned(),
            "aiperf-cli/grpc".to_owned()
        ])
    );
    assert_eq!(
        packages["aiperf-e2e-tests"].features.get("websocket"),
        Some(&vec![
            "aiperf-runtime/websocket".to_owned(),
            "aiperf-cli/websocket".to_owned()
        ])
    );
    assert_eq!(
        packages["aiperf-e2e-tests"].features.get("dynosim"),
        Some(&vec![
            "aiperf-runtime/dynosim".to_owned(),
            "aiperf-cli/dynosim".to_owned()
        ])
    );
    for package in packages.values() {
        let dependencies = normal_and_build_dependencies(package);
        assert!(
            package.name == "aiperf-cli"
                || !(dependencies.contains("aiperf-plugin-host")
                    && dependencies.contains("aiperf-runtime"))
        );
        assert!(
            package.name == "aiperf-export-sdk"
                || !dev_dependencies(package).contains("aiperf-plugin-test-support")
        );
    }
}

#[test]
fn symbolic_ownership_policy() {
    // Changing either symbolic row would let a later task steal the frozen
    // universe construction or CLI composition responsibility.
    let ownership_path = workspace_root().join("plugin-api/feature-ownership.toml");
    let value: toml::Value = std::fs::read_to_string(&ownership_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", ownership_path.display()))
        .parse()
        .expect("feature ownership policy must be valid TOML");
    assert_eq!(value["schema_version"].as_integer(), Some(1));
    let rows = value["symbol_ownership"]
        .as_array()
        .expect("symbol ownership rows must be an array");
    assert_eq!(rows.len(), 2);

    for (row, symbol) in rows
        .iter()
        .zip(["FrozenAIPerfRegistry", "FrozenPluginUniverse"])
    {
        assert_eq!(
            row.as_table()
                .expect("symbol ownership row")
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "symbol",
                "owner_crate",
                "source_path",
                "producer_task",
                "construction_crate",
                "consumer_crates",
                "composition_crate",
                "state",
                "composition_state",
            ])
        );
        assert_eq!(row["symbol"].as_str(), Some(symbol));
        assert_eq!(row["owner_crate"].as_str(), Some("aiperf-plugin-api"));
        assert_eq!(
            row["source_path"].as_str(),
            Some("plugin-api/src/frozen.rs")
        );
        assert_eq!(row["producer_task"].as_integer(), Some(15));
        assert_eq!(
            row["construction_crate"].as_str(),
            Some("aiperf-plugin-host")
        );
        assert_eq!(
            row["consumer_crates"]
                .as_array()
                .expect("consumer crates")
                .iter()
                .map(|value| value.as_str().expect("consumer crate"))
                .collect::<Vec<_>>(),
            vec!["aiperf-runtime"]
        );
        assert_eq!(row["composition_crate"].as_str(), Some("aiperf-cli"));
        assert_eq!(row["state"].as_str(), Some("planned"));
        assert_eq!(row["composition_state"].as_str(), Some("planned"));
    }
}

#[test]
fn candidate_inventory_policy() {
    // Dropping a leaf, treating a planned Task-6 split as present, or losing a
    // byte digest would make later candidate staging non-reproducible.
    let inventory_path =
        workspace_root().join("plugin-conformance/candidate-source-inventory.toml");
    let source_root = workspace_root();
    let repository_root = source_root.parent().expect("repository root");
    let generator = source_root.join("scripts/generate-plugin-candidate-inventory.py");
    let output = Command::new("python")
        .args([
            generator.to_str().expect("generator path"),
            repository_root.to_str().expect("repository path"),
            "--check",
        ])
        .output()
        .expect("candidate generator must execute");
    assert!(
        output.status.success(),
        "candidate generator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let value: toml::Value = std::fs::read_to_string(&inventory_path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", inventory_path.display()))
        .parse()
        .expect("candidate inventory must be valid TOML");
    let rows = value["source"]
        .as_array()
        .expect("candidate inventory rows must be an array");
    assert_eq!(rows.len(), 126);
    assert_eq!(
        value["base_commit"].as_str(),
        Some("057d116850cd059bcfa8e259c1e929e913e6ef07")
    );
    assert_eq!(
        value["provisional_against"].as_str(),
        Some("Task 1 pending integration")
    );

    let planned = BTreeSet::from([
        "runtime/src/transport/grpc/kserve_binding.rs",
        "runtime/src/transport/ws/sink.rs",
        "runtime/src/transport/dry_run.rs",
        "runtime/src/dynosim/direct.rs",
    ]);
    let facades = BTreeSet::from([
        "runtime/src/endpoints/mod.rs",
        "runtime/src/transport/grpc/mod.rs",
    ]);
    let expected_package = BTreeMap::from([
        (24, "export-basic"),
        (25, "export-parquet"),
        (26, "export-mlflow"),
        (27, "export-wandb"),
        (28, "export-otel"),
        (29, "endpoints"),
        (30, "endpoints"),
        (31, "transport-http"),
        (32, "transport-grpc"),
        (33, "transport-websocket"),
        (34, "transport-dynosim"),
    ]);
    let source_root = workspace_root();
    let mut source_paths = BTreeSet::new();
    let mut candidate_paths = BTreeSet::new();
    let mut present = 0;
    let mut assets = 0;
    let mut implementation_leaves = 0;
    let mut facade_rows = 0;
    for row in rows {
        let table = row.as_table().expect("inventory row must be a table");
        let source_path = table["source_path"].as_str().expect("source path");
        assert!(
            source_paths.insert(source_path),
            "duplicate source path {source_path}"
        );
        let candidate_path = table["candidate_path"].as_str().expect("candidate path");
        assert!(
            candidate_paths.insert(candidate_path),
            "duplicate candidate path {candidate_path}"
        );
        let owner_task = table["owner_task"].as_integer().expect("owner task");
        let package = if owner_task == 33 && source_path.starts_with("dry-run-tests/")
            || owner_task == 33 && source_path == "runtime/src/transport/dry_run.rs"
        {
            "transport-dry-run"
        } else {
            expected_package
                .get(&(owner_task as i32))
                .unwrap_or_else(|| panic!("unknown inventory owner task {owner_task}"))
        };
        assert!(candidate_path.starts_with(&format!("plugins/{package}/")));
        let classification = table["classification"].as_str().expect("classification");
        let state = table["state"].as_str().expect("state");
        match (classification, state) {
            ("implementation_leaf", "present") => {
                present += 1;
                implementation_leaves += 1;
                let digest = table["blake3"].as_str().expect("present digest");
                assert_eq!(digest.len(), 64);
                assert!(
                    digest
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
                );
                let bytes = std::fs::read(source_root.join(source_path))
                    .expect("present source must exist");
                assert_eq!(blake3::hash(&bytes).to_hex().as_str(), digest);
            }
            ("implementation_leaf", "planned") => {
                implementation_leaves += 1;
                assert!(planned.contains(source_path));
                assert_eq!(table["producer_task"].as_integer(), Some(6));
                assert!(!table.contains_key("blake3"));
            }
            ("asset", "present") => {
                present += 1;
                assets += 1;
                let digest = table["blake3"].as_str().expect("asset digest");
                let bytes =
                    std::fs::read(source_root.join(source_path)).expect("asset source must exist");
                assert_eq!(blake3::hash(&bytes).to_hex().as_str(), digest);
            }
            ("facade", "present") => {
                present += 1;
                facade_rows += 1;
                assert!(facades.contains(source_path));
                let digest = table["blake3"].as_str().expect("facade digest");
                let bytes =
                    std::fs::read(source_root.join(source_path)).expect("facade source must exist");
                assert_eq!(blake3::hash(&bytes).to_hex().as_str(), digest);
            }
            _ => panic!("invalid inventory state for {source_path}: {classification}/{state}"),
        }
    }
    assert_eq!(source_paths.intersection(&planned).count(), 4);
    assert_eq!(present, 122);
    assert_eq!(implementation_leaves, 115);
    assert_eq!(assets, 9);
    assert_eq!(facade_rows, 2);
}
