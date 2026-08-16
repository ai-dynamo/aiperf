// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Pure contracts for normalized Harbor benchmark execution plans.

use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    fs,
};

use aiperf_runtime::eval::{
    DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerExecRequest,
    DockerRemoveRequest, DockerRuntime, DockerStartRequest, EnvBinding, EnvName,
    EvalExecutionError, HarborImporter, HarborSandboxRecipe, HarborSource, LocalProcessSandbox,
    NativeSourceAcquirer, NetworkPolicy, ProviderCapabilities, SecretProvider, SecretValue,
    VerifierMode, collect_artifacts, resolve_phase_environment,
};

/// A Docker boundary that supplies precise archive bytes for collection tests.
struct ArchiveRuntime {
    archives: RefCell<BTreeMap<String, VecDeque<Vec<u8>>>>,
}

impl ArchiveRuntime {
    fn new(archives: impl IntoIterator<Item = (impl Into<String>, Vec<u8>)>) -> Self {
        Self {
            archives: RefCell::new(
                archives
                    .into_iter()
                    .map(|(source, archive)| (source.into(), VecDeque::from([archive])))
                    .collect(),
            ),
        }
    }
}

impl DockerRuntime for ArchiveRuntime {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::none()
    }

    fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
        unreachable!("artifact collection does not build containers")
    }

    fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
        unreachable!("artifact collection does not create containers")
    }

    fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
        unreachable!("artifact collection does not start containers")
    }

    fn exec(&self, _: &DockerExecRequest) -> Result<(), EvalExecutionError> {
        unreachable!("artifact collection does not execute commands")
    }

    fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
        unreachable!("artifact collection does not copy arbitrary paths")
    }

    fn copy_archive(&self, _: &str, source: &str) -> Result<Vec<u8>, EvalExecutionError> {
        self.archives
            .borrow_mut()
            .get_mut(source)
            .and_then(VecDeque::pop_front)
            .ok_or_else(|| EvalExecutionError::ArtifactCollection(format!("missing {source}")))
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        unreachable!("artifact collection does not remove containers")
    }
}

#[test]
fn declared_files_and_directories_are_collected_at_deterministic_destinations() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = artifact_task_root(
        &temporary,
        "[\"/work/result.txt\", { source = \"/work/output\", destination = \"reports\", exclude = [\"*.tmp\"] }]",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ArchiveRuntime::new([
        (
            "/work/result.txt",
            tar_archive(&[("result.txt", b"answer")]),
        ),
        (
            "/work/output",
            tar_archive(&[
                ("output/keep.txt", b"keep"),
                ("output/nested/also.txt", b"nested"),
                ("output/drop.tmp", b"drop"),
            ]),
        ),
    ]);
    let destination = temporary.path().join("collected");

    let collected = collect_artifacts(
        &runtime,
        "agent-container",
        imported.package.execution_plan().artifacts(),
        &destination,
    )
    .unwrap();

    assert_eq!(
        collected,
        vec![
            (
                "reports/keep.txt".to_owned(),
                aiperf_runtime::eval::ArtifactDigest::from_bytes(b"keep"),
            ),
            (
                "reports/nested/also.txt".to_owned(),
                aiperf_runtime::eval::ArtifactDigest::from_bytes(b"nested"),
            ),
            (
                "result.txt".to_owned(),
                aiperf_runtime::eval::ArtifactDigest::from_bytes(b"answer"),
            ),
        ]
    );
    assert_eq!(
        fs::read(destination.join("reports/keep.txt")).unwrap(),
        b"keep"
    );
    assert!(!destination.join("reports/drop.tmp").exists());
}

#[test]
fn malicious_archive_path_never_escapes_the_declared_artifact_root() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = artifact_task_root(&temporary, "[\"/work/result.txt\"]");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ArchiveRuntime::new([(
        "/work/result.txt",
        tar_archive(&[("../agent-secret", b"must not escape")]),
    )]);
    let destination = temporary.path().join("collected");

    let error = collect_artifacts(
        &runtime,
        "agent-container",
        imported.package.execution_plan().artifacts(),
        &destination,
    )
    .expect_err("archive traversal must be rejected");

    assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
    assert!(!temporary.path().join("agent-secret").exists());
}

#[test]
fn local_legacy_execution_rejects_a_declared_artifact_symlink() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("task.json");
    fs::write(
        &package_path,
        br#"{"id":"artifact-link","instruction":"test","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","ln -s /etc/hosts \"$AIPERF_EVAL_ROOT/results/patch.diff\""],"verifier_command":["true"],"declared_artifacts":["/results/patch.diff"]}"#,
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(package_path.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    let error = LocalProcessSandbox::new()
        .execute(&recipe, &imported.package, VerifierMode::Separate)
        .expect_err("artifact symlinks must not be copied into a verifier sandbox");

    assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
}

/// Constructs a POSIX tar archive with regular files only.
fn tar_archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut archive = Vec::new();
    for (path, contents) in entries {
        let mut header = [0_u8; 512];
        header[..path.len()].copy_from_slice(path.as_bytes());
        header[100..108].copy_from_slice(b"0000644\0");
        header[108..116].copy_from_slice(b"0000000\0");
        header[116..124].copy_from_slice(b"0000000\0");
        let size = format!("{:011o}\0", contents.len());
        header[124..136].copy_from_slice(size.as_bytes());
        header[136..148].copy_from_slice(b"00000000000\0");
        header[148..156].fill(b' ');
        header[156] = b'0';
        header[257..263].copy_from_slice(b"ustar\0");
        header[263..265].copy_from_slice(b"00");
        let checksum = header.iter().map(|byte| u32::from(*byte)).sum::<u32>();
        let checksum = format!("{:06o}\0 ", checksum);
        header[148..156].copy_from_slice(checksum.as_bytes());
        archive.extend_from_slice(&header);
        archive.extend_from_slice(contents);
        archive.resize(archive.len().next_multiple_of(512), 0);
    }
    archive.resize(archive.len() + 1024, 0);
    archive
}

fn artifact_task_root(temporary: &tempfile::TempDir, artifacts: &str) -> std::path::PathBuf {
    let task_root = standard_task_root(temporary, "");
    fs::write(
        task_root.join("task.toml"),
        format!(
            "schema_version = \"1.0\"\nartifacts = {artifacts}\n\n[task]\nname = \"example/artifacts\"\n"
        ),
    )
    .unwrap();
    task_root
}

#[test]
fn normalizes_equivalent_allowlist_spelling_deterministically() {
    let policy =
        NetworkPolicy::allowlist(["EXAMPLE.com", "*.Example.ORG", "2001:DB8::1", "10.0.0.0/24"])
            .unwrap();

    assert_eq!(
        policy
            .allowed_hosts()
            .unwrap()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["*.example.org", "10.0.0.0/24", "2001:db8::1", "example.com"]
    );
}

#[test]
fn environment_binding_never_captures_the_host_secret_value() {
    let binding = EnvBinding::parse("${HOST_API_TOKEN}").unwrap();

    assert_eq!(binding.secret_reference(), Some("HOST_API_TOKEN"));
    assert!(!format!("{binding:?}").contains("super-secret"));
}

#[derive(Default)]
struct RecordingSecrets {
    requested: RefCell<Vec<String>>,
    values: BTreeMap<String, String>,
}

impl SecretProvider for RecordingSecrets {
    fn resolve(&self, name: &EnvName) -> Result<SecretValue, EvalExecutionError> {
        self.requested.borrow_mut().push(name.to_string());
        self.values
            .get(name.as_str())
            .cloned()
            .map(SecretValue::new)
            .ok_or_else(|| EvalExecutionError::MissingSecret(name.to_string()))
    }
}

#[test]
fn resolves_only_active_phase_secrets_and_keeps_later_phase_bindings_independent() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(
        &temporary,
        r#"[environment.env]
BASE = "base"
TOKEN = "${BASE_TOKEN}"

[agent.env]
BASE = "agent"
AGENT_TOKEN = "${AGENT_TOKEN}"

[verifier.env]
BASE = "verifier"
VERIFY_TOKEN = "${VERIFY_TOKEN}"
"#,
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let plan = imported.package.execution_plan();
    let secrets = RecordingSecrets {
        values: BTreeMap::from([
            ("BASE_TOKEN".to_owned(), "base-secret".to_owned()),
            ("AGENT_TOKEN".to_owned(), "agent-secret".to_owned()),
            ("VERIFY_TOKEN".to_owned(), "verify-secret".to_owned()),
        ]),
        ..Default::default()
    };

    let agent = resolve_phase_environment(plan.environment(), plan.agent(), &secrets).unwrap();
    assert_eq!(
        secrets.requested.borrow().as_slice(),
        ["AGENT_TOKEN", "BASE_TOKEN"]
    );
    assert_eq!(
        agent.public().get("BASE").map(String::as_str),
        Some("agent")
    );
    assert_eq!(agent.secret_names(), ["AGENT_TOKEN", "TOKEN"]);

    let verifier = resolve_phase_environment(
        plan.verifier().environment(),
        plan.verifier().phase(),
        &secrets,
    )
    .unwrap();
    assert_eq!(
        secrets.requested.borrow().as_slice(),
        ["AGENT_TOKEN", "BASE_TOKEN", "BASE_TOKEN", "VERIFY_TOKEN"]
    );
    assert_eq!(
        verifier.public().get("BASE").map(String::as_str),
        Some("verifier")
    );
    assert_eq!(verifier.secret_names(), ["TOKEN", "VERIFY_TOKEN"]);
}

#[test]
fn missing_secret_names_the_binding_without_rendering_secret_values() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[agent.env]\nTOKEN = \"${MISSING_TOKEN}\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let secrets = RecordingSecrets::default();

    let error = resolve_phase_environment(
        imported.package.execution_plan().environment(),
        imported.package.execution_plan().agent(),
        &secrets,
    )
    .unwrap_err();

    assert_eq!(
        error,
        EvalExecutionError::MissingSecret("MISSING_TOKEN".to_owned())
    );
    assert!(error.to_string().contains("MISSING_TOKEN"));
    assert!(!error.to_string().contains("secret-value"));
}

struct AdversarialSecretProvider;

impl SecretProvider for AdversarialSecretProvider {
    fn resolve(&self, _: &EnvName) -> Result<SecretValue, EvalExecutionError> {
        Err(EvalExecutionError::MissingSecret(
            "untrusted-secret-payload".to_owned(),
        ))
    }
}

#[test]
fn secret_provider_failures_are_replaced_with_the_declared_reference_name() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[agent.env]\nTOKEN = \"${DECLARED_TOKEN}\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();

    let error = resolve_phase_environment(
        imported.package.execution_plan().environment(),
        imported.package.execution_plan().agent(),
        &AdversarialSecretProvider,
    )
    .unwrap_err();

    assert_eq!(
        error,
        EvalExecutionError::MissingSecret("DECLARED_TOKEN".to_owned())
    );
    assert!(!error.to_string().contains("untrusted-secret-payload"));
}

#[test]
fn local_sandbox_refuses_standard_tasks_before_running_commands() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "[agent]\nnetwork = \"no-network\"\n");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let recipe = HarborSandboxRecipe::new(
        "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "/work",
    )
    .unwrap();

    assert_eq!(
        LocalProcessSandbox::new().execute(&recipe, &imported.package, VerifierMode::Shared),
        Err(EvalExecutionError::UnsupportedEnforcement("docker"))
    );
}

fn standard_task_root(temporary: &tempfile::TempDir, manifest_suffix: &str) -> std::path::PathBuf {
    let task_root = temporary.path().join("task");
    fs::create_dir_all(task_root.join("environment")).unwrap();
    fs::create_dir_all(task_root.join("tests")).unwrap();
    fs::write(
        task_root.join("task.toml"),
        format!("schema_version = \"1.0\"\n\n[task]\nname = \"example/task\"\n\n{manifest_suffix}"),
    )
    .unwrap();
    fs::write(task_root.join("instruction.md"), "Do work.\n").unwrap();
    fs::write(task_root.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
    fs::write(task_root.join("tests/test.sh"), "exit 0\n").unwrap();
    task_root
}
