// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Foundation policy checks for the provisional native plugin workspace.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    process::Command,
};

use serde::{Deserialize, Serialize};

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
    version: String,
    edition: String,
    manifest_path: String,
    publish: Option<Vec<String>>,
    dependencies: Vec<Dependency>,
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct Dependency {
    name: String,
    kind: Option<String>,
    rename: Option<String>,
    source: Option<String>,
    req: String,
    optional: bool,
    uses_default_features: bool,
    features: Vec<String>,
    target: Option<String>,
    registry: Option<String>,
    path: Option<String>,
}

#[derive(Deserialize)]
struct BaselineTopology {
    host_commit: String,
    workspace_packages: Vec<BaselinePackage>,
    cargo_projection: Vec<CargoPackageProjection>,
}
#[derive(Deserialize)]
struct BaselinePackage {
    name: String,
    version: String,
    direct_dependencies: Vec<BaselineDependency>,
    features: Vec<String>,
}
#[derive(Deserialize)]
struct BaselineDependency {
    name: String,
    kind: String,
    is_workspace: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TopologyAmendmentPolicy {
    schema_version: u32,
    matrix: Vec<TopologyMatrix>,
    task3_transition: Task3Transition,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TopologyMatrix {
    state: String,
    dependency_projection_blake3: String,
    test_support_dev_consumers: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Task3Transition {
    schema_version: u32,
    producer_task: u32,
    from_state: String,
    to_state: String,
    ownership_record_path: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Task3ActivationRecord {
    schema_version: u32,
    producer_task: u32,
    from_state: String,
    to_state: String,
}

#[derive(Debug, Deserialize)]
struct PackageManifest {
    package: PackageManifestBoundary,
}

#[derive(Debug, Deserialize)]
struct PackageManifestBoundary {
    name: String,
    version: WorkspaceInheritance,
    edition: WorkspaceInheritance,
    license: WorkspaceInheritance,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkspaceInheritance {
    workspace: bool,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, PartialOrd, Ord, Serialize)]
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

#[derive(Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
struct CargoPackageProjection {
    name: String,
    version: String,
    edition: String,
    dependencies: Vec<CargoDependencyIdentity>,
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Serialize)]
struct ShellPackageProjection {
    name: String,
    dependencies: Vec<CargoDependencyIdentity>,
    features: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ApiAllowlist {
    schema_version: u32,
    allowed_dependencies: Vec<String>,
    allowed_std_modules: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CargoLock {
    version: u32,
    package: Vec<CargoLockPackage>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct CargoLockPackage {
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
    #[serde(rename = "dependencies")]
    _dependencies: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StandaloneCargoLock {
    version: u32,
    package: Vec<StandaloneLockPackage>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct StandaloneLockPackage {
    name: String,
    version: String,
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
    let mut packages = BTreeMap::new();
    for package in metadata
        .packages
        .into_iter()
        .filter(|package| members.contains(&package.id))
    {
        let name = package.name.clone();
        assert!(
            packages.insert(name.clone(), package).is_none(),
            "duplicate workspace package name {name}"
        );
    }
    packages
}

fn normalized_path(root: &Path, path: Option<&str>) -> Option<String> {
    path.map(|path| {
        let path = Path::new(path);
        path.strip_prefix(root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/")
    })
}

fn dependency_identity(
    root: &Path,
    workspace_names: &BTreeSet<&str>,
    dependency: &Dependency,
) -> CargoDependencyIdentity {
    let mut features = dependency.features.clone();
    features.sort();
    CargoDependencyIdentity {
        package: dependency.name.clone(),
        local_name: dependency
            .rename
            .clone()
            .unwrap_or_else(|| dependency.name.clone()),
        kind: dependency
            .kind
            .clone()
            .unwrap_or_else(|| "normal".to_owned()),
        source: dependency.source.clone(),
        requirement: dependency.req.clone(),
        registry: dependency.registry.clone(),
        path: normalized_path(root, dependency.path.as_deref()),
        target: dependency.target.clone(),
        is_optional: dependency.optional,
        uses_default_features: dependency.uses_default_features,
        features,
        is_workspace: workspace_names.contains(dependency.name.as_str()),
    }
}

fn package_dependencies(
    root: &Path,
    workspace_names: &BTreeSet<&str>,
    package: &Package,
) -> Vec<CargoDependencyIdentity> {
    let mut dependencies = package
        .dependencies
        .iter()
        .map(|dependency| dependency_identity(root, workspace_names, dependency))
        .collect::<Vec<_>>();
    dependencies.sort();
    dependencies
}

fn canonical_blake3<T: Serialize>(value: &T) -> String {
    let bytes = serde_json::to_vec(value).expect("policy projection must serialize");
    format!("blake3:{}", blake3::hash(&bytes).to_hex())
}

fn parse_toml<T: for<'de> Deserialize<'de>>(path: &Path, description: &str) -> T {
    toml::from_str(
        &std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("{description} must have the exact supported shape: {error}"))
}

fn is_git_object_present(repository_root: &Path, revision_path: &str) -> bool {
    let output = Command::new("git")
        .args(["cat-file", "-e", revision_path])
        .current_dir(repository_root)
        .output()
        .expect("git cat-file must execute");
    match output.status.code() {
        Some(0) => true,
        Some(128) => false,
        code => panic!("git cat-file returned unexpected status {code:?} for {revision_path}"),
    }
}

fn task1_projection(
    root: &Path,
    packages: &BTreeMap<String, Package>,
    baseline: &BaselineTopology,
) -> Vec<CargoPackageProjection> {
    let workspace_names = packages.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let mut projection = baseline
        .workspace_packages
        .iter()
        .map(|captured| {
            let package = &packages[&captured.name];
            let mut features = package.features.clone();
            for values in features.values_mut() {
                values.sort();
            }
            if package.name == "aiperf-e2e-tests" {
                for task2_feature in ["grpc", "websocket", "dynosim"] {
                    features.remove(task2_feature);
                }
            }
            CargoPackageProjection {
                name: package.name.clone(),
                version: package.version.clone(),
                edition: package.edition.clone(),
                dependencies: package_dependencies(root, &workspace_names, package),
                features,
            }
        })
        .collect::<Vec<_>>();
    projection.sort_by(|left, right| left.name.cmp(&right.name));
    projection
}

fn shell_projection(
    root: &Path,
    packages: &BTreeMap<String, Package>,
) -> Vec<ShellPackageProjection> {
    let workspace_names = packages.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let mut projection = FOUNDATION_PACKAGES
        .iter()
        .chain(DISTRIBUTABLE_PACKAGES)
        .map(|(name, _)| {
            let package = &packages[*name];
            let mut features = package.features.clone();
            for values in features.values_mut() {
                values.sort();
            }
            ShellPackageProjection {
                name: package.name.clone(),
                dependencies: package_dependencies(root, &workspace_names, package),
                features,
            }
        })
        .collect::<Vec<_>>();
    projection.sort_by(|left, right| left.name.cmp(&right.name));
    projection
}

fn active_topology_state(root: &Path, transition: &Task3Transition) -> String {
    assert_eq!(
        transition.ownership_record_path,
        "plugin-api/feature-ownership.toml"
    );
    let ownership: toml::Value =
        std::fs::read_to_string(root.join(&transition.ownership_record_path))
            .expect("feature ownership policy")
            .parse()
            .expect("feature ownership policy TOML");
    let Some(record) = ownership.get("topology_amendment") else {
        return transition.from_state.clone();
    };
    let activation: Task3ActivationRecord = record
        .clone()
        .try_into()
        .expect("Task 3 topology amendment record must be typed and exact");
    let expected = Task3ActivationRecord {
        schema_version: transition.schema_version,
        producer_task: transition.producer_task,
        from_state: transition.from_state.clone(),
        to_state: transition.to_state.clone(),
    };
    assert_eq!(
        activation, expected,
        "unauthorized topology amendment record"
    );
    transition.to_state.clone()
}

fn production_dependency_closure(
    packages: &BTreeMap<String, Package>,
    roots: &[&str],
) -> BTreeSet<String> {
    let mut pending = roots
        .iter()
        .map(|root| (*root).to_owned())
        .collect::<Vec<_>>();
    let mut closure = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !closure.insert(name.clone()) {
            continue;
        }
        let Some(package) = packages.get(&name) else {
            continue;
        };
        pending.extend(
            package
                .dependencies
                .iter()
                .filter(|dependency| dependency.kind.as_deref() != Some("dev"))
                .filter(|dependency| packages.contains_key(&dependency.name))
                .map(|dependency| dependency.name.clone()),
        );
    }
    closure
}

fn normal_and_build_dependencies(package: &Package) -> BTreeSet<&str> {
    package
        .dependencies
        .iter()
        .filter(|dependency| dependency.kind.as_deref() != Some("dev"))
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
    assert_eq!(
        baseline.host_commit,
        "caa3ff6fcf20ffe36a7704abe16274bedadbb9fb"
    );
    assert_eq!(
        task1_projection(&root, &packages, &baseline),
        baseline.cargo_projection,
        "complete Task 1 Cargo projection drift"
    );
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
    let expected_workspace_identities = baseline
        .workspace_packages
        .iter()
        .map(|package| (package.name.clone(), package.version.clone()))
        .chain(
            FOUNDATION_PACKAGES
                .iter()
                .chain(DISTRIBUTABLE_PACKAGES)
                .map(|(name, _)| ((*name).to_owned(), "0.12.0".to_owned())),
        )
        .collect::<BTreeSet<_>>();
    assert_eq!(expected_workspace_identities.len(), expected_names.len());
    let lock: CargoLock =
        toml::from_str(&std::fs::read_to_string(root.join("Cargo.lock")).expect("parent lock"))
            .expect("parent lock TOML");
    assert_eq!(lock.version, 4);
    let source_less_identities = lock
        .package
        .iter()
        .filter(|package| package.source.is_none())
        .map(|package| {
            assert!(
                package.checksum.is_none(),
                "source-less package {} must not have a checksum",
                package.name
            );
            (package.name.clone(), package.version.clone())
        })
        .collect::<Vec<_>>();
    let unique_source_less_identities = source_less_identities
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    assert_eq!(
        source_less_identities.len(),
        unique_source_less_identities.len(),
        "parent lock contains duplicate source-less package identities"
    );
    assert_eq!(
        unique_source_less_identities, expected_workspace_identities,
        "parent lock source-less package identities must exactly match the workspace"
    );
    // Task 1 classified metadata dependency package names by workspace
    // membership; source-less external path dependencies are not workspace edges.
    let workspace_names = packages.keys().map(String::as_str).collect::<BTreeSet<_>>();
    for captured in &baseline.workspace_packages {
        let current = &packages[&captured.name];
        assert_eq!(
            current.version, captured.version,
            "baseline package version drift for {}",
            captured.name
        );
        let mut expected = captured
            .direct_dependencies
            .iter()
            .map(|edge| (edge.name.as_str(), edge.kind.as_str(), edge.is_workspace))
            .collect::<Vec<_>>();
        expected.sort_unstable();
        let mut actual = current
            .dependencies
            .iter()
            .map(|edge| {
                let package_name = edge.name.as_str();
                (
                    package_name,
                    edge.kind.as_deref().unwrap_or("normal"),
                    workspace_names.contains(package_name),
                )
            })
            .collect::<Vec<_>>();
        actual.sort_unstable();
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

    let root_manifest: toml::Value = std::fs::read_to_string(root.join("Cargo.toml"))
        .expect("workspace manifest")
        .parse()
        .expect("workspace manifest TOML");
    assert_eq!(
        root_manifest["workspace"]["package"]["version"].as_str(),
        Some("0.12.0")
    );
    assert_eq!(
        root_manifest["workspace"]["package"]["edition"].as_str(),
        Some("2024")
    );
    assert_eq!(
        root_manifest["workspace"]["package"]["license"].as_str(),
        Some("Apache-2.0")
    );
    for (name, relative_manifest_dir) in FOUNDATION_PACKAGES.iter().chain(DISTRIBUTABLE_PACKAGES) {
        let package = packages
            .get(*name)
            .unwrap_or_else(|| panic!("missing workspace shell {name}"));
        assert_eq!(
            Path::new(&package.manifest_path),
            root.join(relative_manifest_dir).join("Cargo.toml"),
            "{name} must retain its assigned package directory"
        );
        assert_eq!(package.version, "0.12.0", "{name} package version drift");
        assert_eq!(package.edition, "2024", "{name} package edition drift");
        let manifest: PackageManifest = parse_toml(
            &root.join(relative_manifest_dir).join("Cargo.toml"),
            "plugin shell manifest",
        );
        assert_eq!(manifest.package.name, *name);
        assert!(
            manifest.package.version.workspace,
            "{name} must inherit version"
        );
        assert!(
            manifest.package.edition.workspace,
            "{name} must inherit edition"
        );
        assert!(
            manifest.package.license.workspace,
            "{name} must inherit license"
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
    assert_eq!(
        standalone_packages["aiperf-plugin-third-party-example"].version,
        "0.0.0"
    );
    let standalone_lock: StandaloneCargoLock = toml::from_str(
        &std::fs::read_to_string(root.join("tests/plugin-third-party/Cargo.lock"))
            .expect("standalone lock"),
    )
    .expect("standalone lock must have the exact supported shape");
    assert_eq!(
        standalone_lock,
        StandaloneCargoLock {
            version: 4,
            package: vec![StandaloneLockPackage {
                name: "aiperf-plugin-third-party-example".to_owned(),
                version: "0.0.0".to_owned(),
            }],
        }
    );

    let api = &packages["aiperf-plugin-api"];
    let allowlist_path = root.join("plugin-api/api-allowlist.toml");
    let allowlist: ApiAllowlist = toml::from_str(
        &std::fs::read_to_string(&allowlist_path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", allowlist_path.display())),
    )
    .expect("API allowlist must have the exact supported shape");
    assert_eq!(allowlist.schema_version, 1);
    let allowed = allowlist
        .allowed_dependencies
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    assert_eq!(allowlist.allowed_dependencies.len(), allowed.len());
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
        allowlist
            .allowed_std_modules
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["alloc", "core", "std"])
    );
    assert_eq!(allowlist.allowed_std_modules.len(), 3);

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
    for package in packages.values() {
        assert!(
            !normal_and_build_dependencies(package).contains("aiperf-plugin-test-support"),
            "{} must not have a production dependency on test support",
            package.name
        );
    }
    let topology_policy: TopologyAmendmentPolicy = parse_toml(
        &root.join("plugin-api/topology-amendment.toml"),
        "topology amendment policy",
    );
    assert_eq!(topology_policy.schema_version, 1);
    assert_eq!(topology_policy.matrix.len(), 2);
    assert_eq!(topology_policy.task3_transition.schema_version, 1);
    assert_eq!(topology_policy.task3_transition.producer_task, 3);
    assert_eq!(topology_policy.task3_transition.from_state, "task2_neutral");
    assert_eq!(topology_policy.task3_transition.to_state, "task3_reviewed");
    let matrices = topology_policy
        .matrix
        .iter()
        .map(|matrix| (matrix.state.as_str(), matrix))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(matrices.len(), topology_policy.matrix.len());
    assert_eq!(
        matrices.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from(["task2_neutral", "task3_reviewed"])
    );
    let active_state = active_topology_state(&root, &topology_policy.task3_transition);
    let active_matrix = matrices[active_state.as_str()];
    assert_eq!(
        canonical_blake3(&shell_projection(&root, &packages)),
        active_matrix.dependency_projection_blake3,
        "complete {active_state} shell dependency matrix drift"
    );
    let test_support_dev_consumers = packages
        .values()
        .filter(|package| {
            package.dependencies.iter().any(|dependency| {
                dependency.name == "aiperf-plugin-test-support"
                    && dependency.kind.as_deref() == Some("dev")
            })
        })
        .map(|package| package.name.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        test_support_dev_consumers,
        active_matrix
            .test_support_dev_consumers
            .iter()
            .map(String::as_str)
            .collect()
    );
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
    }
}

#[test]
fn distribution_exclusion_policy() {
    let root = workspace_root();
    let packages = package_map(metadata(&root));
    let host_universe = production_dependency_closure(
        &packages,
        &["aiperf-cli", "aiperf-runtime", "aiperf-plugin-host"],
    );
    assert!(
        !host_universe.contains("aiperf-plugin-test-support"),
        "host-universe dependency closure includes test support"
    );
    let repository_root = root.parent().expect("repository root");
    let verifier = root.join("scripts/verify-plugin-test-support-boundaries.py");
    let current_executable = std::env::current_exe().expect("current policy-test executable");
    let output = Command::new("python")
        .arg(&verifier)
        .arg(repository_root)
        .arg(current_executable)
        .output()
        .expect("distribution boundary verifier must execute");
    assert!(
        output.status.success(),
        "distribution boundary verification failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
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
        value
            .as_table()
            .expect("candidate inventory top level")
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>(),
        BTreeSet::from(["base_commit", "source"])
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
                assert!(
                    !source_root.join(source_path).exists(),
                    "planned source exists in worktree: {source_path}"
                );
                assert!(
                    !is_git_object_present(
                        repository_root,
                        &format!("057d116850cd059bcfa8e259c1e929e913e6ef07:rust/{source_path}")
                    ),
                    "planned source exists in pinned base: {source_path}"
                );
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
}
