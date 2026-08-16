// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fs;
use std::process::Command;
use std::time::Duration;

use aiperf_runtime::eval::{
    ArtifactDigest, EnvBinding, HarborImporter, HarborSource, ImportDisposition,
    NativeSourceAcquirer, NetworkPolicy, ProviderCapabilities, SourceAcquirer, VerifierMode,
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
                .with_resource_limits()
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
    assert!(matches!(
        plan.validate_for(ProviderCapabilities::none().with_docker()),
        Err(aiperf_runtime::eval::EvalExecutionError::UnsupportedEnforcement("allowlist_egress"))
    ));
    assert!(
        plan.validate_for(
            ProviderCapabilities::none()
                .with_docker()
                .with_resource_limits()
                .with_users()
                .with_phase_env()
                .with_healthchecks()
                .with_no_network()
                .with_public_network()
                .with_allowlist_egress(),
        )
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
fn rejects_standard_multi_step_manifest_before_execution() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = temporary.path().join("multi-step");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        "schema_version = \"1.0\"\n[task]\nname = \"example/multi\"\n[[steps]]\nname = \"one\"\n",
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();

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
