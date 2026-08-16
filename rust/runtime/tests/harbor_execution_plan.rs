// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Pure contracts for normalized Harbor benchmark execution plans.

use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    fs,
    io::{self, Read},
    os::unix::fs::PermissionsExt,
    sync::Mutex,
    time::Duration,
};

use aiperf_runtime::eval::{
    DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerExecRequest,
    DockerProcessSandbox, DockerRemoveRequest, DockerRuntime, DockerStartRequest, EnvBinding,
    EnvName, EvalExecutionError, HarborImportError, HarborImporter, HarborSandboxRecipe,
    HarborSource, LocalProcessSandbox, NativeSourceAcquirer, NetworkPolicy, ProviderCapabilities,
    SecretProvider, SecretValue, VerifierMode, collect_artifacts, resolve_phase_environment,
    transfer_artifacts,
};

static UMASK_TEST_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn compose_standard_task_exposes_generated_main_services_evidence_and_defaults() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "");
    fs::write(
        task_root.join("task.toml"),
        r#"schema_version = "1.0"
artifacts = [{ source = "/var/lib/api/dump", destination = "api.dump", service = "api" }]

[task]
name = "example/compose"

[verifier]
environment_mode = "separate"

[[verifier.collect]]
service = "main"
command = ["prepare-main", "--flush"]

[[steps]]
name = "final"

[[steps.verifier.collect]]
service = "api"
command = ["dbctl", "dump"]
"#,
    )
    .unwrap();
    fs::create_dir_all(task_root.join("steps/final")).unwrap();
    fs::write(
        task_root.join("steps/final/instruction.md"),
        "Collect evidence.\n",
    )
    .unwrap();
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  main:\n    depends_on: [api]\n  api:\n    image: api:fixture\n",
    )
    .unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let plan = imported.package.execution_plan();
    let compose = plan.compose().unwrap();
    let step = &plan.steps()[0];

    assert_eq!(compose.definition_path(), "environment/docker-compose.yaml");
    assert_eq!(
        compose
            .services()
            .iter()
            .map(|service| service.as_str())
            .collect::<Vec<_>>(),
        ["api", "main"]
    );
    assert_eq!(compose.build_timeout(), Duration::from_secs(600));
    assert_eq!(compose.startup_timeout(), Duration::from_secs(120));
    assert_eq!(step.collection_timeout(), Duration::from_secs(120));
    assert_eq!(step.artifacts()[0].service(), "api");
    assert_eq!(step.collect_hooks().len(), 2);
    assert_eq!(step.collect_hooks()[0].service().as_str(), "main");
    assert_eq!(
        step.collect_hooks()[0].command(),
        ["prepare-main", "--flush"]
    );
    assert_eq!(step.collect_hooks()[0].timeout(), Duration::from_secs(60));
    assert_eq!(step.collect_hooks()[0].user(), None);
    assert_eq!(step.collect_hooks()[1].service().as_str(), "api");
    assert_eq!(step.collect_hooks()[1].command(), ["dbctl", "dump"]);
}

#[test]
fn compose_plan_requires_the_complete_provider_contract_before_execution() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = standard_task_root(&temporary, "");
    fs::write(
        task_root.join("environment/docker-compose.yaml"),
        "services:\n  api:\n    image: api:fixture\n",
    )
    .unwrap();
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let plan = imported.package.execution_plan();

    let baseline = ProviderCapabilities::none()
        .with_docker()
        .with_image_source()
        .with_public_network();
    assert!(matches!(
        plan.validate_for(baseline),
        Err(EvalExecutionError::UnsupportedEnforcement(
            "compose_project"
        ))
    ));
    assert!(matches!(
        plan.validate_for(baseline.with_compose_project()),
        Err(EvalExecutionError::UnsupportedEnforcement("compose_config"))
    ));
    assert!(matches!(
        plan.validate_for(
            baseline
                .with_compose_project()
                .with_compose_config()
                .with_service_exec()
                .with_service_archive()
        ),
        Err(EvalExecutionError::UnsupportedEnforcement("service_stop"))
    ));
    assert!(
        plan.validate_for(
            baseline
                .with_compose_project()
                .with_compose_config()
                .with_service_exec()
                .with_service_archive()
                .with_service_stop(),
        )
        .is_ok()
    );
}

/// A Docker boundary that supplies precise archive bytes for collection tests.
struct ArchiveRuntime {
    archives: RefCell<BTreeMap<String, VecDeque<Vec<u8>>>>,
}

/// A bounded reader that models a Docker archive pipe without holding file data.
struct ChunkedReader {
    remaining: usize,
    max_read: std::rc::Rc<std::cell::Cell<usize>>,
}

impl Read for ChunkedReader {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let count = self.remaining.min(buffer.len()).min(4096);
        buffer[..count].fill(b'x');
        self.remaining -= count;
        self.max_read.set(self.max_read.get().max(count));
        Ok(count)
    }
}

/// A Docker boundary that exposes one archive as a chunked reader.
struct ChunkedArchiveRuntime {
    source: String,
    archive: RefCell<Option<Box<dyn Read>>>,
    max_read: std::rc::Rc<std::cell::Cell<usize>>,
}

impl ChunkedArchiveRuntime {
    fn new(source: impl Into<String>, file_len: usize) -> Self {
        let max_read = std::rc::Rc::new(std::cell::Cell::new(0));
        Self {
            source: source.into(),
            archive: RefCell::new(Some(file_tar_stream(
                "result.txt",
                file_len,
                max_read.clone(),
            ))),
            max_read,
        }
    }
}

impl DockerRuntime for ChunkedArchiveRuntime {
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

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        if source != self.source {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "missing {source}"
            )));
        }
        self.archive.borrow_mut().take().ok_or_else(|| {
            EvalExecutionError::ArtifactCollection("archive consumed twice".to_owned())
        })
    }

    fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
        unreachable!("artifact collection does not remove containers")
    }
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

    fn copy_archive(&self, _: &str, source: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.archives
            .borrow_mut()
            .get_mut(source)
            .and_then(VecDeque::pop_front)
            .map(|archive| Box::new(io::Cursor::new(archive)) as Box<dyn Read>)
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

    assert_eq!(
        imported.package.execution_plan().artifacts()[0].service(),
        "main"
    );

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
fn directory_collection_rejects_archive_members_outside_the_declared_root() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = artifact_task_root(
        &temporary,
        "[{ source = \"/work/output\", destination = \"published\" }]",
    );
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let runtime = ArchiveRuntime::new([(
        "/work/output",
        tar_archive(&[("unrelated.txt", b"must not collect")]),
    )]);

    let error = collect_artifacts(
        &runtime,
        "agent-container",
        imported.package.execution_plan().artifacts(),
        &temporary.path().join("collected"),
    )
    .expect_err("a directory archive must remain rooted under its source name");

    assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
}

#[test]
fn collection_rejects_link_and_special_archive_entries() {
    for entry_type in [b'2', b'1', b'3', b'4', b'6'] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = artifact_task_root(&temporary, "[\"/work/result.txt\"]");
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap();
        let runtime = ArchiveRuntime::new([(
            "/work/result.txt",
            tar_archive_entry("result.txt", b"", entry_type),
        )]);

        let error = collect_artifacts(
            &runtime,
            "agent-container",
            imported.package.execution_plan().artifacts(),
            &temporary.path().join("collected"),
        )
        .expect_err("links and special files must not cross the artifact boundary");

        assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
    }
}

#[test]
fn collection_rejects_missing_sources_and_duplicate_destinations() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = artifact_task_root(&temporary, "[\"/work/missing.txt\"]");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let missing = ArchiveRuntime::new([] as [(String, Vec<u8>); 0]);
    assert!(matches!(
        collect_artifacts(
            &missing,
            "agent-container",
            imported.package.execution_plan().artifacts(),
            &temporary.path().join("missing"),
        ),
        Err(EvalExecutionError::ArtifactCollection(_))
    ));

    let task_root = artifact_task_root(
        &temporary,
        "[{ source = \"/work/one\", destination = \"same\" }, { source = \"/work/two\", destination = \"same\" }]",
    );
    let error = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .expect_err("duplicate destinations must be rejected before execution");
    assert!(matches!(error, HarborImportError::InvalidPackage(_)));
}

#[test]
fn transfer_revalidates_the_collected_file_digest() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("result.txt"), b"modified").unwrap();

    let error = transfer_artifacts(
        &source,
        &temporary.path().join("destination"),
        &[(
            "result.txt".to_owned(),
            aiperf_runtime::eval::ArtifactDigest::from_bytes(b"original"),
        )],
    )
    .expect_err("transfer must verify the collection digest before copying");

    assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
}

#[test]
fn transfer_makes_a_verified_artifact_readable_by_the_verifier() {
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    fs::create_dir_all(&source).unwrap();
    fs::write(source.join("result.txt"), b"result").unwrap();
    fs::set_permissions(source.join("result.txt"), fs::Permissions::from_mode(0o600)).unwrap();
    let destination = temporary.path().join("destination");

    transfer_artifacts(
        &source,
        &destination,
        &[(
            "result.txt".to_owned(),
            aiperf_runtime::eval::ArtifactDigest::from_bytes(b"result"),
        )],
    )
    .unwrap();

    assert_eq!(
        fs::metadata(destination.join("result.txt"))
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o644
    );
}

#[test]
fn transfer_makes_nested_collection_directories_traversable_under_a_restrictive_umask() {
    let _umask_lock = UMASK_TEST_LOCK.lock().unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let source = temporary.path().join("source");
    fs::create_dir_all(source.join("published/nested")).unwrap();
    fs::write(source.join("published/nested/result.txt"), b"result").unwrap();
    let destination = temporary.path().join("destination");
    let previous_umask = unsafe { libc::umask(0o077) };
    let result = transfer_artifacts(
        &source,
        &destination,
        &[(
            "published/nested/result.txt".to_owned(),
            aiperf_runtime::eval::ArtifactDigest::from_bytes(b"result"),
        )],
    );
    unsafe { libc::umask(previous_umask) };
    result.unwrap();

    for directory in [
        destination.join("published"),
        destination.join("published/nested"),
    ] {
        assert_eq!(
            fs::metadata(directory).unwrap().permissions().mode() & 0o777,
            0o755
        );
    }
}

#[test]
fn collection_consumes_a_chunked_archive_without_allocating_the_file_contents() {
    let temporary = tempfile::tempdir().unwrap();
    let task_root = artifact_task_root(&temporary, "[\"/work/result.txt\"]");
    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
        .unwrap();
    let file_len = 8 * 1024 * 1024;
    let runtime = ChunkedArchiveRuntime::new("/work/result.txt", file_len);

    let collected = collect_artifacts(
        &runtime,
        "agent-container",
        imported.package.execution_plan().artifacts(),
        &temporary.path().join("collected"),
    )
    .unwrap();

    assert_eq!(collected.len(), 1);
    assert_eq!(
        fs::metadata(temporary.path().join("collected/result.txt"))
            .unwrap()
            .len(),
        file_len as u64
    );
    assert!(
        runtime.max_read.get() <= 16 * 1024,
        "collector requested an unbounded read"
    );
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
        .execute(&recipe, &imported.package, VerifierMode::Shared)
        .expect_err("artifact symlinks must not be accepted from the agent workspace");

    assert!(matches!(error, EvalExecutionError::ArtifactCollection(_)));
}

/// Constructs a POSIX tar archive with regular files only.
fn tar_archive(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut archive = Vec::new();
    for (path, contents) in entries {
        archive.extend_from_slice(&tar_header(path, contents.len(), b'0'));
        archive.extend_from_slice(contents);
        archive.resize(archive.len().next_multiple_of(512), 0);
    }
    archive.resize(archive.len() + 1024, 0);
    archive
}

fn tar_archive_entry(path: &str, contents: &[u8], entry_type: u8) -> Vec<u8> {
    let mut archive = tar_header(path, contents.len(), entry_type).to_vec();
    archive.extend_from_slice(contents);
    archive.resize(archive.len().next_multiple_of(512) + 1024, 0);
    archive
}

fn file_tar_stream(
    path: &str,
    file_len: usize,
    max_read: std::rc::Rc<std::cell::Cell<usize>>,
) -> Box<dyn Read> {
    Box::new(
        io::Cursor::new(tar_header(path, file_len, b'0').to_vec())
            .chain(ChunkedReader {
                remaining: file_len,
                max_read,
            })
            .chain(io::Cursor::new(vec![0; 1024])),
    )
}

fn tar_header(path: &str, size: usize, entry_type: u8) -> [u8; 512] {
    let mut header = [0_u8; 512];
    header[..path.len()].copy_from_slice(path.as_bytes());
    header[100..108].copy_from_slice(b"0000644\0");
    header[108..116].copy_from_slice(b"0000000\0");
    header[116..124].copy_from_slice(b"0000000\0");
    let size = format!("{:011o}\0", size);
    header[124..136].copy_from_slice(size.as_bytes());
    header[136..148].copy_from_slice(b"00000000000\0");
    header[148..156].fill(b' ');
    header[156] = entry_type;
    header[257..263].copy_from_slice(b"ustar\0");
    header[263..265].copy_from_slice(b"00");
    let checksum = header.iter().map(|byte| u32::from(*byte)).sum::<u32>();
    let checksum = format!("{:06o}\0 ", checksum);
    header[148..156].copy_from_slice(checksum.as_bytes());
    header
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

#[test]
fn legacy_tasks_synthesize_one_logical_step() {
    let temporary = tempfile::tempdir().unwrap();
    let package_path = temporary.path().join("legacy-task.json");
    fs::write(
        &package_path,
        br#"{"id":"legacy","instruction":"Fix it","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["true"],"verifier_command":["true"],"declared_artifacts":[]}"#,
    )
    .unwrap();

    let imported = HarborImporter::new(&NativeSourceAcquirer)
        .import(&HarborSource::local(package_path.to_string_lossy()).unwrap())
        .unwrap();
    let plan = imported.package.execution_plan();

    assert!(!plan.is_multi_step());
    assert!(plan.compose().is_none());
    assert!(plan.multi_step_reward_strategy().is_none());
    assert_eq!(plan.steps().len(), 1);
    assert_eq!(plan.steps()[0].name(), "default");
    assert_eq!(plan.steps()[0].instruction(), "Fix it");
}

#[test]
fn existing_sandbox_entry_points_refuse_every_explicit_step_layout() {
    for steps in [
        [("one", "Only step.\n")].as_slice(),
        [("one", "First step.\n"), ("two", "Second step.\n")].as_slice(),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let task_root = explicit_step_task_root(&temporary, steps);
        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&HarborSource::local(task_root.to_string_lossy()).unwrap())
            .unwrap();
        assert!(imported.package.execution_plan().is_multi_step());
        assert!(
            imported
                .package
                .execution_plan()
                .multi_step_reward_strategy()
                .is_some()
        );
        let recipe = HarborSandboxRecipe::new(
            "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "/work",
        )
        .unwrap();

        assert_eq!(
            LocalProcessSandbox::new().execute(&recipe, &imported.package, VerifierMode::Shared),
            Err(EvalExecutionError::UnsupportedMultiStep)
        );
        assert_eq!(
            DockerProcessSandbox::new().execute(
                &recipe,
                &imported.package,
                &["true".to_owned()],
                VerifierMode::Shared,
            ),
            Err(EvalExecutionError::UnsupportedMultiStep)
        );
    }
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

fn explicit_step_task_root(
    temporary: &tempfile::TempDir,
    steps: &[(&str, &str)],
) -> std::path::PathBuf {
    let manifest = steps
        .iter()
        .map(|(name, _)| format!("[[steps]]\nname = \"{name}\"\n"))
        .collect::<Vec<_>>()
        .join("\n");
    let task_root = standard_task_root(temporary, &manifest);
    for (name, instruction) in steps {
        fs::create_dir_all(task_root.join(format!("steps/{name}"))).unwrap();
        fs::write(
            task_root.join(format!("steps/{name}/instruction.md")),
            instruction,
        )
        .unwrap();
    }
    task_root
}
