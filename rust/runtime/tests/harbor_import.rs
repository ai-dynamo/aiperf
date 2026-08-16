// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::process::Command;
use std::time::Duration;

use aiperf_runtime::eval::{
    ArtifactDigest, EnvBinding, HarborImporter, HarborSource, ImageSourceKind, ImportDisposition,
    MultiStepRewardStrategy, NativeSourceAcquirer, NetworkPolicy, ProviderCapabilities,
    SourceAcquirer, VerifierMode,
};

#[derive(Default)]
struct MemoryAcquirer {
    packages: BTreeMap<String, Vec<u8>>,
}

impl SourceAcquirer for MemoryAcquirer {
    fn acquire(
        &self,
        source: &HarborSource,
    ) -> Result<Vec<u8>, aiperf_runtime::eval::HarborImportError> {
        self.packages
            .get(source.location())
            .cloned()
            .ok_or_else(|| {
                aiperf_runtime::eval::HarborImportError::Unavailable(source.location().to_owned())
            })
    }
}

#[test]
fn retains_executable_task_material() {
    let bytes = supported_package();
    let source = HarborSource::local("fixtures/repair-1").unwrap();
    let mut acquirer = MemoryAcquirer::default();
    acquirer
        .packages
        .insert(source.location().to_owned(), bytes.clone());

    let imported = HarborImporter::new(&acquirer).import(&source).unwrap();

    assert_eq!(
        imported.report.source_digest,
        ArtifactDigest::from_bytes(&bytes)
    );
    assert_eq!(
        imported.report.disposition,
        ImportDisposition::LosslessNormalized
    );
    assert_eq!(imported.task.id.as_str(), "repair-1");
    assert_eq!(imported.package.id(), "repair-1");
    assert_eq!(imported.package.instruction(), "Fix the failing test");
    assert_eq!(
        imported.package.environment(),
        "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    );
    assert_eq!(
        imported.package.verifier(),
        "blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
    );
    assert_eq!(
        imported.package.agent_command(),
        ["sh", "-c", "printf patch > out/patch.diff"]
    );
    assert_eq!(
        imported.package.verifier_command(),
        ["sh", "-c", "printf '{\"reward\":1.0}' > reward.json"]
    );
    assert_eq!(
        imported.package.declared_artifacts(),
        ["/results/patch.diff"]
    );
    assert_eq!(
        imported
            .package
            .execution_plan()
            .environment()
            .image_source()
            .kind(),
        ImageSourceKind::LegacyArtifact
    );
    assert!(
        imported
            .package
            .execution_plan()
            .environment()
            .image_source()
            .dockerfile_digest()
            .is_none()
    );
    assert_eq!(
        imported.package.source_digest(),
        imported.report.source_digest
    );
}

#[test]
fn rejects_malformed_executable_task_material() {
    let malformed_packages = [
        package_with_mutation(|package| package["agent_command"] = serde_json::json!([])),
        package_with_mutation(|package| {
            package["verifier_command"] = serde_json::json!(["sh", " "])
        }),
        package_with_mutation(|package| {
            package["declared_artifacts"] = serde_json::json!(["results/patch.diff"])
        }),
        package_with_mutation(|package| package["declared_artifacts"] = serde_json::json!(["/"])),
        package_with_mutation(|package| {
            package["declared_artifacts"] = serde_json::json!(["/results/../secret"])
        }),
        package_with_mutation(|package| {
            package["declared_artifacts"] =
                serde_json::json!(["/results/patch.diff", "//results/patch.diff"])
        }),
        package_with_mutation(|package| package["unknown_package_field"] = serde_json::json!(true)),
        package_with_mutation(|package| {
            package.as_object_mut().unwrap().remove("agent_command");
        }),
    ];

    for bytes in malformed_packages {
        let source = HarborSource::local("fixtures/malformed").unwrap();
        let mut acquirer = MemoryAcquirer::default();
        acquirer
            .packages
            .insert(source.location().to_owned(), bytes);

        assert!(matches!(
            HarborImporter::new(&acquirer).import(&source),
            Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
        ));
    }
}

fn supported_package() -> Vec<u8> {
    br#"{
        "id":"repair-1",
        "instruction":"Fix the failing test",
        "environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        "agent_command":["sh","-c","printf patch > out/patch.diff"],
        "verifier_command":["sh","-c","printf '{\"reward\":1.0}' > reward.json"],
        "declared_artifacts":["/results/patch.diff"]
    }"#
    .to_vec()
}

fn package_with_mutation(mutate: impl FnOnce(&mut serde_json::Value)) -> Vec<u8> {
    let mut package = serde_json::from_slice(&supported_package()).unwrap();
    mutate(&mut package);
    serde_json::to_vec(&package).unwrap()
}

#[test]
fn unsupported_semantics_return_report_before_provisioning() {
    let source = HarborSource::local("fixtures/unsupported").unwrap();
    let mut acquirer = MemoryAcquirer::default();
    acquirer.packages.insert(
        source.location().to_owned(),
        br#"{"id":"repair-1","unsupported_semantics":"sidecar"}"#.to_vec(),
    );

    let refusal = HarborImporter::new(&acquirer).import(&source).unwrap_err();

    assert_eq!(refusal.disposition(), Some(ImportDisposition::Unsupported));
}

#[test]
fn native_acquirer_reads_local_and_pinned_git_package_bytes() {
    let temporary = tempfile::tempdir().unwrap();
    let package = br#"{"id":"repair-1","instruction":"Fix","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}"#;
    let local_path = temporary.path().join("local-task.json");
    fs::write(&local_path, package).unwrap();
    let acquirer = NativeSourceAcquirer;

    assert_eq!(
        acquirer
            .acquire(&HarborSource::local(local_path.to_string_lossy()).unwrap())
            .unwrap(),
        package
    );

    let repository = temporary.path().join("task-repository");
    fs::create_dir(&repository).unwrap();
    run_git(&repository, ["init"]);
    run_git(
        &repository,
        ["config", "user.email", "eval@example.invalid"],
    );
    run_git(&repository, ["config", "user.name", "Native Eval"]);
    let package_path = repository.join("task.json");
    fs::write(&package_path, package).unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "task A"]);
    let revision = git_output(&repository, ["rev-parse", "HEAD"]);
    fs::write(&package_path, br#"{"different":"head"}"#).unwrap();
    run_git(&repository, ["add", "task.json"]);
    run_git(&repository, ["commit", "-m", "task B"]);

    assert_eq!(
        acquirer
            .acquire(
                &HarborSource::pinned_git(repository.to_string_lossy(), revision, "task.json",)
                    .unwrap(),
            )
            .unwrap(),
        package
    );
}

#[test]
fn native_acquirer_owns_directory_tree_identity_but_preserves_file_byte_identity() {
    let temporary = tempfile::tempdir().unwrap();
    let package = supported_package();
    let directory = temporary.path().join("directory-task");
    fs::create_dir(&directory).unwrap();
    fs::write(directory.join("task.json"), &package).unwrap();
    fs::write(directory.join("fixture.txt"), b"original fixture\n").unwrap();
    let acquirer = NativeSourceAcquirer;

    let acquired = acquirer
        .acquire_artifact(&HarborSource::local(directory.to_string_lossy()).unwrap())
        .unwrap();
    let directory_digest = acquired.source_digest();
    assert_eq!(acquired.primary_bytes(), package);
    assert_ne!(directory_digest, ArtifactDigest::from_bytes(&package));

    fs::write(directory.join("task.json"), b"mutated manifest").unwrap();
    fs::remove_file(directory.join("fixture.txt")).unwrap();
    assert_eq!(acquired.primary_bytes(), package);
    assert_eq!(acquired.source_digest(), directory_digest);

    let file = temporary.path().join("single-task.json");
    fs::write(&file, &package).unwrap();
    let acquired = acquirer
        .acquire_artifact(&HarborSource::local(file.to_string_lossy()).unwrap())
        .unwrap();
    assert_eq!(acquired.primary_bytes(), package);
    assert_eq!(
        acquired.source_digest(),
        ArtifactDigest::from_bytes(&package)
    );
}

#[test]
fn imports_standard_directory_manifest_with_instruction_and_verifier() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("standard-task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"

[task]
name = "example/repair-1"

[agent]
timeout_sec = 7.5

[verifier]
timeout_sec = 3.25

[environment]
cpus = 1
memory_mb = 512
"#,
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Fix the failing test.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(
        task_root.join("tests/test.sh"),
        "#!/bin/sh\nprintf 1 > reward.txt\n",
    )
    .unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();

    assert_eq!(imported.task.id.as_str(), "example/repair-1");
    assert_eq!(imported.package.instruction(), "Fix the failing test.\n");
    assert_eq!(
        imported.package.verifier_command(),
        ["/bin/sh", "tests/test.sh"]
    );
    assert_eq!(imported.package.verifier_mode(), VerifierMode::Shared);
    let plan = imported.package.execution_plan();
    assert_eq!(plan.environment().network(), &NetworkPolicy::public());
    assert!(matches!(
        plan.validate_for(ProviderCapabilities::none()),
        Err(aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement("docker"))
    ));
    assert!(
        plan.validate_for(
            ProviderCapabilities::none()
                .with_docker()
                .with_image_source()
                .with_resource_limits()
                .with_phase_timeouts()
                .with_public_network(),
        )
        .is_ok()
    );
    assert_eq!(imported.package.container_resources(), Some((1, 512)));
    assert_eq!(
        imported.package.timeouts(),
        Some((Duration::from_millis(7500), Duration::from_millis(3250)))
    );

    fs::write(
        task_root.join("instruction.md"),
        "Fix a different failing test.\n",
    )
    .unwrap();
    let changed = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_ne!(imported.report.source_digest, changed.report.source_digest);
}

#[test]
fn rejects_standard_zero_or_subnanosecond_timeout() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("zero-timeout");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    for timeout in ["0.0000000001", "0"] {
        fs::write(
            task_root.join("task.toml"),
            format!(
                r#"schema_version = "1.0"

[task]
name = "example/zero-timeout"

[agent]
timeout_sec = {timeout}

[verifier]
timeout_sec = 3
"#
            ),
        )
        .unwrap();

        assert!(matches!(
            HarborImporter::new(&NativeSourceAcquirer)
                .import(&HarborSource::local(task_root.to_string_lossy()).unwrap()),
            Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
        ));
    }
}

#[test]
fn imports_standard_separate_verifier_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("separate-task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = ["/work/result.txt"]
[task]
name = "example/separate-1"
[verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Write a result.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();

    assert_eq!(imported.package.verifier_mode(), VerifierMode::Separate);
    assert_eq!(imported.package.declared_artifacts(), ["/work/result.txt"]);
}

#[test]
fn normalizes_standard_execution_fields_without_reading_host_secrets() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "full-plan");
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = ["/work/result.txt", { source = "/work/output", destination = "results", exclude = ["*.tmp"] }]

[task]
name = "example/full-plan"

[environment]
cpus = 2
memory_mb = 1024
workdir = "/workspace"
user = "1000:1001"
network = "allowlist"
allowed_hosts = ["EXAMPLE.com", "*.Example.ORG", "10.0.0.1", "10.0.0.0/24", "2001:DB8::1"]

[environment.env]
BASE = "base"
API_TOKEN = "${HOST_API_TOKEN}"

[environment.healthcheck]
command = ["sh", "-c", "true"]
start_period_sec = 1.0
start_interval_sec = 0.5
interval_sec = 2.0
timeout_sec = 3.0
retries = 4

[agent]
timeout_sec = 7.5
user = "agent"
network = "no-network"

[agent.env]
BASE = "agent"
AGENT_TOKEN = "${AGENT_TOKEN}"

[verifier]
environment_mode = "separate"
timeout_sec = 3.25
user = "verifier"
network = "public"

[verifier.env]
CHECK = "1"

[verifier.environment]
workdir = "/verify"
user = "2000"
network = "public"

[verifier.environment.env]
VERIFY_BASE = "present"
"#,
    )
    .unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let plan = imported.package.execution_plan();

    assert_eq!(
        plan.environment()
            .image_source()
            .dockerfile_digest()
            .unwrap()
            .as_str(),
        imported.package.environment()
    );
    assert_eq!(plan.environment().resources().unwrap().cpus(), 2);
    assert_eq!(plan.environment().resources().unwrap().memory_mb(), 1024);
    assert_eq!(plan.environment().workdir(), Some("/workspace"));
    assert_eq!(plan.environment().user(), Some("1000:1001"));
    assert_eq!(
        plan.environment()
            .network()
            .allowed_hosts()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        [
            "*.example.org",
            "10.0.0.0/24",
            "10.0.0.1",
            "2001:db8::1",
            "example.com",
        ]
    );
    assert_eq!(
        plan.environment()
            .env()
            .get("API_TOKEN")
            .and_then(EnvBinding::secret_reference),
        Some("HOST_API_TOKEN")
    );
    assert_eq!(plan.agent().user(), Some("agent"));
    assert_eq!(plan.agent().network(), &NetworkPolicy::no_network());
    assert_eq!(
        plan.agent()
            .env()
            .get("AGENT_TOKEN")
            .and_then(EnvBinding::secret_reference),
        Some("AGENT_TOKEN")
    );
    assert_eq!(plan.verifier().mode(), VerifierMode::Separate);
    assert_eq!(plan.verifier().phase().user(), Some("verifier"));
    assert_eq!(plan.verifier().environment().workdir(), Some("/verify"));
    assert_eq!(plan.verifier().environment().user(), Some("2000"));
    assert_eq!(plan.artifacts().len(), 2);
    assert_eq!(plan.artifacts()[0].source(), "/work/result.txt");
    assert!(plan.artifacts()[0].is_exact_file());
    assert_eq!(plan.artifacts()[1].source(), "/work/output");
    assert_eq!(plan.artifacts()[1].destination(), Some("results"));
    assert_eq!(plan.artifacts()[1].exclude(), ["*.tmp"]);
    let existing_capabilities = ProviderCapabilities::none()
        .with_docker()
        .with_resource_limits()
        .with_users()
        .with_phase_env()
        .with_healthchecks()
        .with_no_network()
        .with_public_network()
        .with_allowlist_egress();
    assert!(matches!(
        plan.validate_for(existing_capabilities),
        Err(aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement("image_source"))
    ));
    let with_image_source = existing_capabilities.with_image_source();
    assert!(matches!(
        plan.validate_for(with_image_source),
        Err(aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement("workdir"))
    ));
    let with_workdir = with_image_source.with_workdir();
    assert!(matches!(
        plan.validate_for(with_workdir),
        Err(aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement("phase_timeouts"))
    ));
    let with_phase_timeouts = with_workdir.with_phase_timeouts();
    assert!(matches!(
        plan.validate_for(with_phase_timeouts),
        Err(aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement("separate_verifier"))
    ));
    assert!(
        plan.validate_for(with_phase_timeouts.with_separate_verifier())
            .is_ok()
    );
}

#[test]
fn separate_verifier_without_environment_receives_fresh_environment_copy() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "fresh-verifier-environment");
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
[task]
name = "example/fresh-verifier-environment"
[environment]
workdir = "/workspace"
user = "1000"
network = "no-network"
[environment.env]
BASE = "value"
[verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let plan = imported.package.execution_plan();

    assert_eq!(plan.verifier().mode(), VerifierMode::Separate);
    assert_eq!(plan.verifier().environment(), plan.environment());
}

#[test]
fn rejects_explicit_environment_for_shared_verifier() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "shared-verifier-environment");
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
[task]
name = "example/shared-verifier-environment"
[verifier]
environment_mode = "shared"
[verifier.environment]
workdir = "/must-not-apply"
"#,
    )
    .unwrap();

    assert!(matches!(
        HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap()),
        Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
    ));
}

#[test]
fn rejects_invalid_standard_execution_fields_during_import() {
    let cases = [
        ("unknown", "[environment]\nunknown = true\n"),
        ("incomplete-resources", "[environment]\ncpus = 1\n"),
        (
            "relative-workdir",
            "[environment]\nworkdir = \"workspace\"\n",
        ),
        (
            "dot-workdir",
            "[environment]\nworkdir = \"/work/../secret\"\n",
        ),
        ("invalid-user", "[agent]\nuser = \"root:wheel:extra\"\n"),
        (
            "invalid-template",
            "[environment.env]\nTOKEN = \"${NOT CLOSED\"\n",
        ),
        (
            "non-positive-health-timing",
            "[environment.healthcheck]\ncommand = [\"true\"]\ninterval_sec = 0\n",
        ),
        (
            "non-finite-health-timing",
            "[environment.healthcheck]\ncommand = [\"true\"]\ntimeout_sec = inf\n",
        ),
        (
            "allowlist-with-public",
            "[environment]\nnetwork = \"public\"\nallowed_hosts = [\"example.com\"]\n",
        ),
        (
            "allowlist-url",
            "[environment]\nnetwork = \"allowlist\"\nallowed_hosts = [\"https://example.com\"]\n",
        ),
        (
            "allowlist-missing-hosts",
            "[environment]\nnetwork = \"allowlist\"\n",
        ),
        (
            "duplicate-normalized-allowlist",
            "[environment]\nnetwork = \"allowlist\"\nallowed_hosts = [\"EXAMPLE.com\", \"example.com\"]\n",
        ),
        (
            "malformed-allowlist",
            "[environment]\nnetwork = \"allowlist\"\nallowed_hosts = [\"*bad.example\"]\n",
        ),
    ];
    for (name, authored) in cases {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!("schema_version = \"1.0\"\n[task]\nname = \"example/{name}\"\n{authored}"),
        )
        .unwrap();

        assert!(matches!(
            HarborImporter::new(&NativeSourceAcquirer)
                .import(&HarborSource::local(task_root.to_string_lossy()).unwrap()),
            Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
        ));
    }
}

fn standard_task_root(temporary: &tempfile::TempDir, name: &str) -> std::path::PathBuf {
    let task_root = temporary.path().join(name);
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    task_root
}

#[test]
fn compose_identity_binds_services_evidence_hooks_timeouts_and_owned_environment_tree() {
    let manifest = r#"schema_version = "1.0"
artifacts = [{ source = "/var/lib/api/dump", destination = "api.dump", service = "api" }]

[task]
name = "example/compose-identity"

[environment]
build_timeout_sec = 601
startup_timeout_sec = 121

[verifier]
environment_mode = "separate"
collection_timeout_sec = 122

[[verifier.collect]]
service = "main"
command = ["prepare-main", "--flush"]
timeout_sec = 61
user = "1000"

[[verifier.collect]]
service = "api"
command = ["dbctl", "dump"]
timeout_sec = 62
"#;
    let compose = "services:\n  main:\n    depends_on: [api]\n  api:\n    image: api:fixture\n";
    let baseline = import_compose_task_digest(manifest, compose, b"nested input\n");
    let cases = [
        (
            "service target",
            manifest.replacen("service = \"api\"", "service = \"main\"", 1),
            compose.to_owned(),
            b"nested input\n".as_slice(),
        ),
        (
            "hook argv",
            manifest.replacen("[\"dbctl\", \"dump\"]", "[\"dbctl\", \"export\"]", 1),
            compose.to_owned(),
            b"nested input\n".as_slice(),
        ),
        (
            "hook user",
            manifest.replacen("user = \"1000\"", "user = \"1001\"", 1),
            compose.to_owned(),
            b"nested input\n".as_slice(),
        ),
        (
            "hook timeout",
            manifest.replacen("timeout_sec = 62", "timeout_sec = 63", 1),
            compose.to_owned(),
            b"nested input\n".as_slice(),
        ),
        (
            "build timeout",
            manifest.replacen("build_timeout_sec = 601", "build_timeout_sec = 602", 1),
            compose.to_owned(),
            b"nested input\n".as_slice(),
        ),
        (
            "startup timeout",
            manifest.replacen("startup_timeout_sec = 121", "startup_timeout_sec = 123", 1),
            compose.to_owned(),
            b"nested input\n".as_slice(),
        ),
        (
            "collection timeout",
            manifest.replacen(
                "collection_timeout_sec = 122",
                "collection_timeout_sec = 124",
                1,
            ),
            compose.to_owned(),
            b"nested input\n".as_slice(),
        ),
        (
            "sidecar service",
            manifest.replace("api", "db"),
            compose.replace("api", "db"),
            b"nested input\n".as_slice(),
        ),
        (
            "nested build input",
            manifest.to_owned(),
            compose.to_owned(),
            b"changed nested input\n".as_slice(),
        ),
    ];

    for (name, changed_manifest, changed_compose, nested_input) in cases {
        assert_ne!(
            baseline,
            import_compose_task_digest(&changed_manifest, &changed_compose, nested_input),
            "compose identity did not change for {name}"
        );
    }
}

#[test]
fn compose_source_selection_and_service_shape_fail_closed() {
    let valid_manifest = "schema_version = \"1.0\"\n[task]\nname = \"example/compose-invalid\"\n";
    let invalid_compose_documents = [
        ("non-utf8", vec![0xff]),
        ("empty", Vec::new()),
        ("non-mapping", b"[]\n".to_vec()),
        ("missing-services", b"version: '3'\n".to_vec()),
        ("non-mapping-services", b"services: []\n".to_vec()),
        ("empty-services", b"services: {}\n".to_vec()),
        ("uppercase-service", b"services:\n  API: {}\n".to_vec()),
        ("unsafe-service", b"services:\n  '../api': {}\n".to_vec()),
        (
            "long-service",
            format!("services:\n  {}: {{}}\n", "a".repeat(64)).into_bytes(),
        ),
        ("non-mapping-main", b"services:\n  main: image\n".to_vec()),
        (
            "main-image",
            b"services:\n  main:\n    image: task-controlled\n".to_vec(),
        ),
        (
            "main-command",
            b"services:\n  main:\n    command: [true]\n".to_vec(),
        ),
        (
            "duplicate-service",
            b"services:\n  api: {}\n  api: {}\n".to_vec(),
        ),
    ];
    for (name, compose) in invalid_compose_documents {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(task_root.join("task.toml"), valid_manifest).unwrap();
        fs::write(task_root.join("environment/docker-compose.yaml"), compose).unwrap();
        assert_invalid_standard_task(&task_root, name);
    }

    for alternate in ["docker-compose.yml", "compose.yaml", "compose.yml"] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, alternate);
        fs::write(task_root.join("task.toml"), valid_manifest).unwrap();
        fs::write(
            task_root.join("environment").join(alternate),
            "services:\n  api: {}\n",
        )
        .unwrap();
        assert_invalid_standard_task(&task_root, alternate);
    }

    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "missing-dockerfile");
    fs::write(task_root.join("task.toml"), valid_manifest).unwrap();
    fs::remove_file(task_root.join("environment/Dockerfile")).unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api: {}\n",
    )
    .unwrap();
    assert_invalid_standard_task(&task_root, "missing Dockerfile");
}

#[test]
fn compose_evidence_hooks_and_topology_fail_closed() {
    let compose = "services:\n  api: {}\n  db: {}\n";
    let invalid_manifests = [
        (
            "unknown-artifact-service",
            r#"artifacts = [{ source = "/data/dump", service = "missing" }]
[task]
name = "example/invalid"
[verifier]
environment_mode = "separate"
"#,
        ),
        (
            "empty-hook-argv",
            r#"[task]
name = "example/invalid"
[verifier]
[[verifier.collect]]
command = []
"#,
        ),
        (
            "blank-hook-argument",
            r#"[task]
name = "example/invalid"
[verifier]
[[verifier.collect]]
command = ["dbctl", " "]
"#,
        ),
        (
            "bad-hook-user",
            r#"[task]
name = "example/invalid"
[verifier]
[[verifier.collect]]
command = ["true"]
user = "bad user"
"#,
        ),
        (
            "unknown-hook-field",
            r#"[task]
name = "example/invalid"
[verifier]
[[verifier.collect]]
command = ["true"]
environment = { TOKEN = "value" }
"#,
        ),
        (
            "unknown-artifact-field",
            r#"artifacts = [{ source = "/data/dump", command = ["true"] }]
[task]
name = "example/invalid"
"#,
        ),
        (
            "duplicate-service-source",
            r#"artifacts = [
  { source = "/data/dump", destination = "one", service = "api" },
  { source = "/data/dump", destination = "two", service = "api" },
]
[task]
name = "example/invalid"
[verifier]
environment_mode = "separate"
"#,
        ),
        (
            "cross-service-output-collision",
            r#"artifacts = [
  { source = "/main/result", service = "main" },
  { source = "/api/result", service = "api" },
]
[task]
name = "example/invalid"
[verifier]
environment_mode = "separate"
"#,
        ),
        (
            "main-hook-after-sidecar",
            r#"[task]
name = "example/invalid"
[verifier]
environment_mode = "separate"
[[verifier.collect]]
service = "api"
command = ["api-dump"]
[[verifier.collect]]
service = "main"
command = ["main-flush"]
"#,
        ),
        (
            "shared-sidecar-evidence",
            r#"artifacts = [{ source = "/data/dump", service = "api" }]
[task]
name = "example/invalid"
"#,
        ),
        (
            "restricted-environment-network",
            r#"[task]
name = "example/invalid"
[environment]
network = "no-network"
"#,
        ),
        (
            "restricted-agent-network",
            r#"[task]
name = "example/invalid"
[agent]
network = "no-network"
"#,
        ),
        (
            "restricted-verifier-network",
            r#"[task]
name = "example/invalid"
[verifier]
network = "no-network"
"#,
        ),
    ];
    for (name, suffix) in invalid_manifests {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!("schema_version = \"1.0\"\n{suffix}"),
        )
        .unwrap();
        fs::write(task_root.join("environment/docker-compose.yaml"), compose).unwrap();
        assert_invalid_standard_task(&task_root, name);
    }

    for (name, timeout) in [
        ("nonfinite", "inf"),
        ("zero", "0"),
        ("negative", "-1"),
        ("subnanosecond", "0.0000000001"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!(
                "schema_version = \"1.0\"\n[task]\nname = \"example/{name}\"\n[verifier]\n[[verifier.collect]]\ncommand = [\"true\"]\ntimeout_sec = {timeout}\n"
            ),
        )
        .unwrap();
        fs::write(task_root.join("environment/docker-compose.yaml"), compose).unwrap();
        assert_invalid_standard_task(&task_root, name);
    }

    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "without-compose");
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = [{ source = "/data/dump", service = "api" }]
[task]
name = "example/without-compose"
[verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();
    assert_invalid_standard_task(&task_root, "sidecar reference without Compose");

    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "non-final-sidecar");
    for step in ["one", "two"] {
        fs::create_dir_all(task_root.join(format!("steps/{step}"))).unwrap();
        fs::write(
            task_root.join(format!("steps/{step}/instruction.md")),
            "Do step work.\n",
        )
        .unwrap();
    }
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
[task]
name = "example/non-final-sidecar"
[verifier]
environment_mode = "separate"
[[steps]]
name = "one"
artifacts = [{ source = "/data/dump", service = "api" }]
[[steps]]
name = "two"
"#,
    )
    .unwrap();
    fs::write(task_root.join("environment/docker-compose.yaml"), compose).unwrap();
    assert_invalid_standard_task(&task_root, "non-final sidecar evidence");
}

fn import_compose_task_digest(
    manifest: &str,
    compose: &str,
    nested_input: &[u8],
) -> ArtifactDigest {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "compose-identity");
    fs::create_dir_all(task_root.join("environment/context")).unwrap();
    fs::write(task_root.join("task.toml"), manifest).unwrap();
    fs::write(task_root.join("environment/docker-compose.yaml"), compose).unwrap();
    fs::write(
        task_root.join("environment/context/build-input.txt"),
        nested_input,
    )
    .unwrap();
    import_task_digest(&task_root)
}

fn assert_invalid_standard_task(task_root: &std::path::Path, case: &str) {
    assert!(
        matches!(
            HarborImporter::new(&NativeSourceAcquirer)
                .import(&HarborSource::local(task_root.to_string_lossy()).unwrap()),
            Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
        ),
        "{case} must be refused during import"
    );
}

#[test]
fn imports_standard_multi_step_manifest_in_authored_order() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("multi-step");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::create_dir_all(task_root.join("steps/prepare")).unwrap();
    fs::create_dir_all(task_root.join("steps/verify/tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "final"
artifacts = ["/work/root.txt"]

[task]
name = "example/multi"

[agent]
timeout_sec = 5
user = "root-agent"

[agent.env]
ROOT = "present"

[verifier]
timeout_sec = 3

[[steps]]
name = "prepare"
artifacts = ["/work/prepare.txt"]

[steps.agent]
user = "prepare-agent"

[steps.agent.env]
ROOT = "overridden"

[[steps]]
name = "verify"
artifacts = ["/work/verify.txt"]

[steps.verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Legacy instruction.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        task_root.join("steps/prepare/instruction.md"),
        "Prepare the workspace.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("steps/verify/instruction.md"),
        "Verify the workspace.\n",
    )
    .unwrap();
    fs::write(task_root.join("steps/verify/tests/test.sh"), "exit 2\n").unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .expect("valid explicit steps must normalize");
    let plan = imported.package.execution_plan();

    assert!(plan.is_multi_step());
    assert_eq!(
        plan.multi_step_reward_strategy(),
        Some(MultiStepRewardStrategy::Final)
    );
    assert_eq!(
        plan.steps()
            .iter()
            .map(|step| step.name())
            .collect::<Vec<_>>(),
        ["prepare", "verify"]
    );
    assert_eq!(plan.steps()[0].instruction(), "Prepare the workspace.\n");
    assert_eq!(plan.steps()[1].instruction(), "Verify the workspace.\n");
    assert_eq!(plan.steps()[0].agent().user(), Some("prepare-agent"));
    assert_eq!(
        plan.steps()[0]
            .agent()
            .env()
            .get("ROOT")
            .and_then(EnvBinding::literal),
        Some("overridden")
    );
    assert_eq!(
        plan.steps()[0].agent().timeout(),
        Some(Duration::from_secs(5))
    );
    assert_eq!(
        plan.steps()[0]
            .artifacts()
            .iter()
            .map(|artifact| artifact.source())
            .collect::<Vec<_>>(),
        ["/work/root.txt", "/work/prepare.txt"]
    );
    assert_eq!(plan.steps()[0].verifier_test_root(), "tests");
    assert_eq!(plan.steps()[1].verifier_test_root(), "steps/verify/tests");
    assert_eq!(plan.steps()[1].verifier().mode(), VerifierMode::Separate);

    fs::write(
        task_root.join("steps/verify/instruction.md"),
        "A different verifier instruction.\n",
    )
    .unwrap();
    let changed_instruction = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_ne!(imported.task.digest, changed_instruction.task.digest);

    fs::write(task_root.join("steps/verify/tests/test.sh"), "exit 3\n").unwrap();
    let changed_test = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_ne!(changed_instruction.task.digest, changed_test.task.digest);
}

#[test]
fn standard_task_identity_tracks_normalized_execution_policy_and_artifacts() {
    let cases = [
        ("environment resources", "cpus = 2", "cpus = 3"),
        ("environment memory", "memory_mb = 1024", "memory_mb = 2048"),
        (
            "environment workdir",
            "workdir = \"/workspace\"",
            "workdir = \"/other-workspace\"",
        ),
        (
            "environment literal",
            "BASE = \"baseline\"",
            "BASE = \"changed\"",
        ),
        (
            "environment user",
            "user = \"1000:1001\"",
            "user = \"1000:1002\"",
        ),
        (
            "environment secret reference",
            "TOKEN = \"${HOST_ONE}\"",
            "TOKEN = \"${HOST_TWO}\"",
        ),
        (
            "environment allowlist",
            "allowed_hosts = [\"EXAMPLE.com\", \"10.0.0.1\"]",
            "allowed_hosts = [\"other.example.com\", \"10.0.0.1\"]",
        ),
        (
            "healthcheck command",
            "command = [\"sh\", \"-c\", \"true\"]",
            "command = [\"sh\", \"-c\", \"test -f /ready\"]",
        ),
        (
            "healthcheck start period",
            "start_period_sec = 1",
            "start_period_sec = 2",
        ),
        (
            "healthcheck start interval",
            "start_interval_sec = 0.5",
            "start_interval_sec = 0.75",
        ),
        (
            "healthcheck interval",
            "interval_sec = 2",
            "interval_sec = 3",
        ),
        (
            "healthcheck timeout",
            "timeout_sec = 3\nretries = 4",
            "timeout_sec = 4\nretries = 4",
        ),
        ("healthcheck retries", "retries = 4", "retries = 5"),
        (
            "agent user",
            "user = \"agent\"\nnetwork = \"no-network\"",
            "user = \"other-agent\"\nnetwork = \"no-network\"",
        ),
        (
            "agent phase environment",
            "AGENT = \"root\"",
            "AGENT = \"changed\"",
        ),
        (
            "agent network",
            "user = \"agent\"\nnetwork = \"no-network\"",
            "user = \"agent\"\nnetwork = \"public\"",
        ),
        (
            "agent timeout",
            "[agent]\ntimeout_sec = 5",
            "[agent]\ntimeout_sec = 6",
        ),
        (
            "verifier phase environment",
            "VERIFY_ROOT = \"present\"",
            "VERIFY_ROOT = \"changed\"",
        ),
        (
            "step agent user",
            "user = \"step-agent\"",
            "user = \"other-step-agent\"",
        ),
        (
            "step agent environment",
            "STEP_TOKEN = \"${STEP_ONE}\"",
            "STEP_TOKEN = \"${STEP_TWO}\"",
        ),
        (
            "step agent network",
            "allowed_hosts = [\"STEP.example.com\"]",
            "allowed_hosts = [\"other-step.example.com\"]",
        ),
        (
            "step agent timeout",
            "[steps.agent]\nuser = \"step-agent\"\ntimeout_sec = 7",
            "[steps.agent]\nuser = \"step-agent\"\ntimeout_sec = 9",
        ),
        (
            "step verifier mode",
            "environment_mode = \"separate\"",
            "environment_mode = \"shared\"",
        ),
        (
            "step verifier user",
            "user = \"step-verifier\"",
            "user = \"other-step-verifier\"",
        ),
        (
            "step verifier environment",
            "VERIFY_STEP = \"${VERIFY_ONE}\"",
            "VERIFY_STEP = \"${VERIFY_TWO}\"",
        ),
        (
            "step verifier network",
            "[steps.verifier]\nenvironment_mode = \"separate\"\nuser = \"step-verifier\"\ntimeout_sec = 8\nnetwork = \"no-network\"",
            "[steps.verifier]\nenvironment_mode = \"separate\"\nuser = \"step-verifier\"\ntimeout_sec = 8\nnetwork = \"public\"",
        ),
        (
            "step verifier timeout",
            "user = \"step-verifier\"\ntimeout_sec = 8",
            "user = \"step-verifier\"\ntimeout_sec = 10",
        ),
        (
            "artifact source",
            "source = \"/work/root\"",
            "source = \"/work/other-root\"",
        ),
        (
            "artifact kind",
            "artifacts = [{ source = \"/work/root\", destination = \"root-output\", exclude = [\"*.tmp\"] }]",
            "artifacts = [\"/work/root\"]",
        ),
        (
            "artifact destination",
            "destination = \"root-output\"",
            "destination = \"other-root-output\"",
        ),
        (
            "artifact exclusion",
            "exclude = [\"*.tmp\"]",
            "exclude = [\"*.cache\"]",
        ),
        (
            "reward strategy",
            "multi_step_reward_strategy = \"mean\"",
            "multi_step_reward_strategy = \"final\"",
        ),
    ];
    let mut collisions = Vec::new();
    for (name, before, after) in cases {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = identity_task_root(&temporary);
        let baseline = import_task_digest(&task_root);
        let manifest = fs::read_to_string(task_root.join("task.toml")).unwrap();
        assert!(manifest.contains(before), "fixture is missing {before:?}");
        fs::write(
            task_root.join("task.toml"),
            manifest.replacen(before, after, 1),
        )
        .unwrap();
        let changed = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap_or_else(|error| panic!("{name} mutation must remain valid: {error}"));
        if changed.task.digest == baseline {
            collisions.push(name);
        }
    }
    assert!(
        collisions.is_empty(),
        "task identity did not change for {collisions:?}"
    );
}

#[test]
fn standard_task_identity_tracks_every_selected_verifier_tree_file() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = identity_task_root(&temporary);
    let baseline = import_task_digest(&task_root);

    fs::write(task_root.join("tests/fixtures/root.txt"), "changed root\n").unwrap();
    let changed_root_helper = import_task_digest(&task_root);
    assert_ne!(baseline, changed_root_helper);

    fs::write(
        task_root.join("steps/two/tests/fixtures/step.txt"),
        "changed step\n",
    )
    .unwrap();
    let changed_step_helper = import_task_digest(&task_root);
    assert_ne!(changed_root_helper, changed_step_helper);
}

#[test]
fn standard_task_identity_uses_normalized_policy_not_toml_spelling() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = identity_task_root(&temporary);
    let baseline = import_task_digest(&task_root);
    let manifest = fs::read_to_string(task_root.join("task.toml")).unwrap();
    let equivalent = manifest
        .replace(
            "allowed_hosts = [\"EXAMPLE.com\", \"10.0.0.1\"]",
            "allowed_hosts = [\"10.0.0.1\", \"example.COM\"]",
        )
        .replace(
            "BASE = \"baseline\"\nTOKEN = \"${HOST_ONE}\"",
            "TOKEN = \"${HOST_ONE}\"\nBASE = \"baseline\"",
        );
    fs::write(task_root.join("task.toml"), equivalent).unwrap();

    assert_eq!(baseline, import_task_digest(&task_root));
}

#[test]
fn artifact_exclusions_use_one_sorted_unique_normal_form_for_execution_and_identity() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "canonical-exclusions");
    let manifest = |exclude: &str| {
        format!(
            r#"schema_version = "1.0"
artifacts = [{{ source = "/work/results", destination = "results", exclude = {exclude} }}]
[task]
name = "example/canonical-exclusions"
"#,
        )
    };
    fs::write(
        task_root.join("task.toml"),
        manifest("[\"tmp/**\", \"*.cache\"]"),
    )
    .unwrap();
    let baseline = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_eq!(
        baseline.package.execution_plan().artifacts()[0].exclude(),
        ["*.cache", "tmp/**"]
    );

    fs::write(
        task_root.join("task.toml"),
        manifest("[\"*.cache\", \"tmp/**\", \"*.cache\"]"),
    )
    .unwrap();
    let equivalent = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_eq!(
        equivalent.package.execution_plan().artifacts()[0].exclude(),
        ["*.cache", "tmp/**"]
    );
    assert_eq!(baseline.task.digest, equivalent.task.digest);

    fs::write(
        task_root.join("task.toml"),
        manifest("[\"*.cache\", \"kept/**\"]"),
    )
    .unwrap();
    assert_ne!(baseline.task.digest, import_task_digest(&task_root));
}

#[test]
fn artifact_exclusions_reject_non_relative_or_non_normal_patterns() {
    for (name, pattern) in [
        ("empty", ""),
        ("absolute", "/tmp/**"),
        ("empty-component", "tmp//**"),
        ("current-component", "tmp/./**"),
        ("parent-component", "tmp/../**"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!(
                r#"schema_version = "1.0"
artifacts = [{{ source = "/work/results", exclude = ["{pattern}"] }}]
[task]
name = "example/{name}"
"#,
            ),
        )
        .unwrap();

        assert!(matches!(
            HarborImporter::new(&NativeSourceAcquirer)
                .import(&HarborSource::local(task_root.to_string_lossy()).unwrap()),
            Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
        ));
    }
}

#[test]
fn standard_package_identity_binds_context_selected_tree_entries_and_modes() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = identity_task_root(&temporary);
    let baseline = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_eq!(baseline.package.identity_digest(), baseline.task.digest);
    assert_eq!(baseline.report.normalized_digest, baseline.task.digest);
    assert_eq!(
        baseline.package.source_digest(),
        baseline.report.source_digest
    );

    fs::write(task_root.join("environment/context.txt"), b"context\n").unwrap();
    let with_context = import_task_digest(&task_root);
    assert_ne!(baseline.task.digest, with_context);

    fs::create_dir(task_root.join("environment/empty")).unwrap();
    let with_environment_empty_directory = import_task_digest(&task_root);
    assert_ne!(with_context, with_environment_empty_directory);

    fs::set_permissions(
        task_root.join("environment/context.txt"),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    let with_executable_context = import_task_digest(&task_root);
    assert_ne!(with_environment_empty_directory, with_executable_context);

    fs::create_dir(task_root.join("tests/empty")).unwrap();
    let with_selected_empty_directory = import_task_digest(&task_root);
    assert_ne!(with_executable_context, with_selected_empty_directory);

    fs::set_permissions(
        task_root.join("tests/fixtures/root.txt"),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    assert_ne!(
        with_selected_empty_directory,
        import_task_digest(&task_root)
    );
}

#[test]
fn standard_source_provenance_includes_unselected_entries_without_changing_package_identity() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = identity_task_root(&temporary);
    let baseline = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();

    fs::write(task_root.join("notes.txt"), b"not executable\n").unwrap();
    let with_unselected_file = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_ne!(
        baseline.report.source_digest,
        with_unselected_file.report.source_digest
    );
    assert_eq!(baseline.task.digest, with_unselected_file.task.digest);

    let manifest = fs::read_to_string(task_root.join("task.toml")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!("# equivalent formatting\n{manifest}"),
    )
    .unwrap();
    let reformatted = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_ne!(
        with_unselected_file.report.source_digest,
        reformatted.report.source_digest
    );
    assert_eq!(baseline.task.digest, reformatted.task.digest);
}

#[test]
fn directory_json_package_identity_binds_the_whole_tree_and_canonical_modes() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("json-tree");
    fs::create_dir_all(task_root.join("fixtures")).unwrap();
    fs::write(task_root.join("task.json"), supported_package()).unwrap();
    fs::write(task_root.join("fixtures/input.txt"), b"fixture\n").unwrap();
    let baseline = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    assert_eq!(baseline.package.identity_digest(), baseline.task.digest);
    assert_eq!(baseline.report.normalized_digest, baseline.task.digest);

    fs::create_dir(task_root.join("fixtures/empty")).unwrap();
    let with_empty = import_task_digest(&task_root);
    assert_ne!(baseline.task.digest, with_empty);

    fs::set_permissions(
        task_root.join("fixtures/input.txt"),
        fs::Permissions::from_mode(0o755),
    )
    .unwrap();
    assert_ne!(with_empty, import_task_digest(&task_root));
}

#[test]
fn standard_task_identity_distinguishes_implicit_and_explicit_layouts() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "layout-identity");
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/layout-identity\"\n",
    )
    .unwrap();
    let implicit = import_task_digest(&task_root);
    fs::create_dir_all(task_root.join("steps/default")).unwrap();
    fs::write(task_root.join("steps/default/instruction.md"), "Do work.\n").unwrap();
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
[task]
name = "example/layout-identity"
[[steps]]
name = "default"
"#,
    )
    .unwrap();

    assert_ne!(implicit, import_task_digest(&task_root));
}

#[test]
fn standard_task_import_rejects_artifacts_overlapping_verifier_owned_paths() {
    for (name, source) in [
        ("tests-file", "/tests/result.txt"),
        ("tests-directory", "/tests"),
        ("tests-normalized-alias", "//tests//result.txt"),
        ("reward-file", "/logs/verifier/reward.json"),
        ("reward-directory", "/logs/verifier"),
        ("reward-parent", "/logs"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!(
                "schema_version = \"1.0\"\nartifacts = [\"{source}\"]\n[task]\nname = \"example/{name}\"\n"
            ),
        )
        .unwrap();

        let error = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .expect_err("reserved verifier artifact path must fail import");
        assert!(matches!(
            error,
            aiperf_runtime::eval::HarborImportError::InvalidPackage(reason)
                if reason.contains("reserved verifier path")
        ));
    }

    for source in [
        "/tests-output/result.txt",
        "/logs/verifier-output/reward.json",
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, "allowed-neighbor");
        fs::write(
            task_root.join("task.toml"),
            format!(
                "schema_version = \"1.0\"\nartifacts = [\"{source}\"]\n[task]\nname = \"example/allowed-neighbor\"\n"
            ),
        )
        .unwrap();

        HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .expect("component-neighbor paths are not verifier-owned");
    }
}

#[test]
fn standard_task_import_rejects_known_verifier_staging_collisions() {
    for (name, workdir, destination) in [
        ("tests-workdir", "/tests", "result.txt"),
        ("tests-alias", "//tests//", "result.txt"),
        ("reward-workdir", "/logs/verifier", "result.txt"),
        ("reward-target", "/", "logs/verifier/reward.json"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!(
                r#"schema_version = "1.0"
artifacts = [{{ source = "/work/output", destination = "{destination}" }}]
[task]
name = "example/{name}"
[verifier]
environment_mode = "separate"
[verifier.environment]
workdir = "{workdir}"
"#,
            ),
        )
        .unwrap();

        let error = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .expect_err("known reserved verifier staging target must fail import");
        assert!(matches!(
            error,
            aiperf_runtime::eval::HarborImportError::InvalidPackage(reason)
                if reason.contains("reserved verifier path")
        ));
    }
}

#[test]
fn shared_verifier_workdirs_reserve_only_their_directional_namespaces() {
    for (name, workdir) in [
        ("tests", "/tests"),
        ("nested-reward", "/logs/verifier/nested"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!(
                "schema_version = \"1.0\"\n[task]\nname = \"example/{name}\"\n[environment]\nworkdir = \"{workdir}\"\n"
            ),
        )
        .unwrap();

        let error = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .expect_err("a shared verifier workdir must not occupy evaluator-owned paths");
        assert!(matches!(
            error,
            aiperf_runtime::eval::HarborImportError::InvalidPackage(reason)
                if reason.contains("shared verifier workdir")
        ));
    }

    for (name, workdir) in [
        ("root", "/"),
        ("logs-parent", "/logs"),
        ("tests-neighbor", "/tests-output"),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = standard_task_root(&temporary, name);
        fs::write(
            task_root.join("task.toml"),
            format!(
                "schema_version = \"1.0\"\n[task]\nname = \"example/{name}\"\n[environment]\nworkdir = \"{workdir}\"\n"
            ),
        )
        .unwrap();

        HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .expect("an ancestor or component-neighbor workdir remains valid");
    }
}

#[test]
fn separate_only_verifier_allows_reserved_environment_workdir_without_artifacts() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "separate-reserved-workdir");
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
[task]
name = "example/separate-reserved-workdir"
[environment]
workdir = "/tests"
[verifier]
environment_mode = "separate"
"#,
    )
    .unwrap();

    HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .expect("a separate verifier does not reserve the agent workdir itself");
}

fn import_task_digest(task_root: &std::path::Path) -> ArtifactDigest {
    HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap()
        .task
        .digest
}

fn identity_task_root(temporary: &tempfile::TempDir) -> std::path::PathBuf {
    let task_root = temporary.path().join("identity-task");
    for directory in [
        "environment",
        "tests/fixtures",
        "steps/one",
        "steps/two/tests/fixtures",
    ] {
        fs::create_dir_all(task_root.join(directory)).unwrap();
    }
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
multi_step_reward_strategy = "mean"
artifacts = [{ source = "/work/root", destination = "root-output", exclude = ["*.tmp"] }]

[task]
name = "example/identity"

[environment]
cpus = 2
memory_mb = 1024
workdir = "/workspace"
user = "1000:1001"
network = "allowlist"
allowed_hosts = ["EXAMPLE.com", "10.0.0.1"]

[environment.env]
BASE = "baseline"
TOKEN = "${HOST_ONE}"

[environment.healthcheck]
command = ["sh", "-c", "true"]
start_period_sec = 1
start_interval_sec = 0.5
interval_sec = 2
timeout_sec = 3
retries = 4

[agent]
timeout_sec = 5
user = "agent"
network = "no-network"

[agent.env]
AGENT = "root"

[verifier]
timeout_sec = 3
user = "verifier"
network = "public"

[verifier.env]
VERIFY_ROOT = "present"

[[steps]]
name = "one"
artifacts = ["/work/one.txt"]

[steps.agent]
user = "step-agent"
timeout_sec = 7
network = "allowlist"
allowed_hosts = ["STEP.example.com"]

[steps.agent.env]
STEP_TOKEN = "${STEP_ONE}"

[[steps]]
name = "two"
artifacts = [{ source = "/work/two", destination = "step-output", exclude = ["*.bak"] }]

[steps.verifier]
environment_mode = "separate"
user = "step-verifier"
timeout_sec = 8
network = "no-network"

[steps.verifier.env]
VERIFY_STEP = "${VERIFY_ONE}"
"#,
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Legacy instruction.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    fs::write(task_root.join("tests/fixtures/root.txt"), "root\n").unwrap();
    fs::write(task_root.join("steps/one/instruction.md"), "Step one.\n").unwrap();
    fs::write(task_root.join("steps/two/instruction.md"), "Step two.\n").unwrap();
    fs::write(task_root.join("steps/two/tests/test.sh"), "exit 0\n").unwrap();
    fs::write(
        task_root.join("steps/two/tests/fixtures/step.txt"),
        "step\n",
    )
    .unwrap();
    task_root
}

#[test]
fn rejects_invalid_explicit_step_contracts_before_provisioning() {
    let cases = [
        ("unsafe-name", "[[steps]]\nname = \"../unsafe\"\n", &[][..]),
        (
            "duplicate-name",
            "[[steps]]\nname = \"one\"\n\n[[steps]]\nname = \"one\"\n",
            &["one"],
        ),
        ("blank-instruction", "[[steps]]\nname = \"one\"\n", &["one"]),
        ("missing-test", "[[steps]]\nname = \"one\"\n", &["one"]),
        (
            "fractional-timeout",
            "[agent]\ntimeout_sec = 1\n\n[verifier]\ntimeout_sec = 1\n\n[[steps]]\nname = \"one\"\n\n[steps.agent]\ntimeout_sec = 0.5\n",
            &["one"],
        ),
        (
            "zero-timeout",
            "[agent]\ntimeout_sec = 1\n\n[verifier]\ntimeout_sec = 1\n\n[[steps]]\nname = \"one\"\n\n[steps.agent]\ntimeout_sec = 0\n",
            &["one"],
        ),
        (
            "shared-verifier-environment",
            "[[steps]]\nname = \"one\"\n\n[steps.verifier]\nenvironment_mode = \"shared\"\n\n[steps.verifier.environment]\nworkdir = \"/verify\"\n",
            &["one"],
        ),
        (
            "duplicate-output",
            "artifacts = [\"/work/root.txt\"]\n\n[[steps]]\nname = \"one\"\nartifacts = [\"/other/root.txt\"]\n",
            &["one"],
        ),
        (
            "invalid-strategy",
            "multi_step_reward_strategy = \"maximum\"\n\n[[steps]]\nname = \"one\"\n",
            &["one"],
        ),
    ];

    for (name, manifest, instructions) in cases {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = temporary.path().join(name);
        fs::create_dir_all(task_root.join("environment")).unwrap();
        fs::create_dir_all(task_root.join("tests")).unwrap();
        fs::write(
            task_root.join("task.toml"),
            format!("schema_version = \"1.0\"\n[task]\nname = \"example/{name}\"\n\n{manifest}"),
        )
        .unwrap();
        fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
        fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
        for instruction in instructions {
            fs::create_dir_all(task_root.join(format!("steps/{instruction}"))).unwrap();
            fs::write(
                task_root.join(format!("steps/{instruction}/instruction.md")),
                if name == "blank-instruction" {
                    " \n"
                } else {
                    "Step work.\n"
                },
            )
            .unwrap();
        }
        if name == "missing-test" {
            fs::remove_file(task_root.join("tests/test.sh")).unwrap();
        }

        assert!(matches!(
            HarborImporter::new(&NativeSourceAcquirer)
                .import(&HarborSource::local(task_root.to_string_lossy()).unwrap()),
            Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
        ));
    }

    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "strategy-without-steps");
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\nmulti_step_reward_strategy = \"final\"\n[task]\nname = \"example/strategy-without-steps\"\n",
    )
    .unwrap();
    assert!(matches!(
        HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap()),
        Err(aiperf_runtime::eval::HarborImportError::InvalidPackage(_))
    ));
}

fn run_git<const N: usize>(repository: &std::path::Path, arguments: [&str; N]) {
    let status = Command::new("git")
        .arg("-c")
        .arg("commit.gpgsign=false")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .status()
        .unwrap();
    assert!(status.success());
}

fn git_output<const N: usize>(repository: &std::path::Path, arguments: [&str; N]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repository)
        .args(arguments)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}
