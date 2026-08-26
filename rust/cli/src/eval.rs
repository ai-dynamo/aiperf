// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0
//! Native execution of one Harbor-compatible evaluation package.

mod native_graph;

use std::{
    collections::BTreeMap,
    fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::{fs::OpenOptions, os::unix::fs::OpenOptionsExt};

#[cfg(not(unix))]
use std::fs::File;

use aiperf_runtime::eval::{
    ArtifactDigest, DockerProcessSandbox, EvalExecutionError, HarborEvaluationCoordinator,
    HarborImporter, HarborLifecycleAgentContract, HarborLifecycleRequest, HarborSandboxRecipe,
    HarborSource, LocalExecutionResult, LocalProcessSandbox, MultiStepExecutionResult,
    NativeSourceAcquirer, TrialSpec, VerifierMode,
};
use clap::{Parser, ValueEnum};
use serde::Serialize;

/// Run a single native Harbor-compatible package without a Harbor runtime.
#[derive(Debug, Parser)]
#[command(name = "eval", disable_help_subcommand = true)]
struct EvalFlags {
    /// Local task package directory or manifest file (`task.toml` / `task.json`).
    #[arg(long, conflicts_with_all = ["git_repository", "git_revision", "git_path"])]
    task: Option<PathBuf>,
    /// Local or remote Git repository containing a package pinned by `--git-revision`.
    #[arg(long, requires_all = ["git_revision", "git_path"])]
    git_repository: Option<String>,
    /// Exact 40-hex commit for a Git package source.
    #[arg(long, requires_all = ["git_repository", "git_path"])]
    git_revision: Option<String>,
    /// Repository-relative path to a Git package manifest (`task.toml` / `task.json`).
    #[arg(long, requires_all = ["git_repository", "git_revision"])]
    git_path: Option<String>,
    /// Immutable image identity used by the sandbox recipe.
    #[arg(long)]
    image: Option<String>,
    /// Absolute runtime working-directory override for a standard task.
    #[arg(long)]
    workdir: Option<String>,
    /// Whether the verifier shares the agent sandbox or receives a fresh root.
    #[arg(long, value_enum)]
    verifier_mode: Option<VerifierModeFlag>,
    /// Shell command for an external agent; required by standard task directories.
    #[arg(long)]
    agent_command: Option<String>,
    /// Sandbox backend; `auto` uses Docker for standard directories and separate verification.
    #[arg(long, value_enum, default_value_t = SandboxFlag::Auto)]
    sandbox: SandboxFlag,
    /// Strict host-owned NativeGraph model-secret mapping.
    #[arg(long)]
    model_runtime: Option<PathBuf>,
    /// Strict NativeGraph suite document.
    #[arg(long, conflicts_with_all = ["task", "git_repository", "git_revision", "git_path"])]
    suite: Option<PathBuf>,
    /// Strict versioned JSON request that persists the full immutable evaluation lifecycle.
    #[arg(long)]
    lifecycle_request: Option<PathBuf>,
    /// Destination for canonical schema-1.1 NativeGraph node records in JSONL format.
    #[arg(long)]
    records_output: Option<PathBuf>,
    /// Destination for the canonical lifecycle record; defaults to `aiperf-eval-lifecycle.json`.
    #[arg(long, requires = "lifecycle_request")]
    lifecycle_output: Option<PathBuf>,
}

const MAX_LIFECYCLE_REQUEST_BYTES: u64 = 1024 * 1024;
const DEFAULT_SANDBOX_IMAGE: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

/// User-facing verifier sandbox topology.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum VerifierModeFlag {
    /// Run the verifier in the task sandbox.
    Shared,
    /// Copy only declared artifacts into a fresh verifier sandbox.
    Separate,
}

/// User-facing execution backend selection.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SandboxFlag {
    /// Use Docker for conventional task directories or separate verification, and
    /// local execution otherwise.
    Auto,
    /// Use the temporary-root local process backend.
    Local,
    /// Build and execute a conventional task directory in Docker.
    Docker,
}

impl From<VerifierModeFlag> for VerifierMode {
    fn from(value: VerifierModeFlag) -> Self {
        match value {
            VerifierModeFlag::Shared => Self::Shared,
            VerifierModeFlag::Separate => Self::Separate,
        }
    }
}

#[derive(Serialize)]
struct EvalOutput<'a> {
    task: &'a str,
    artifacts: &'a [(String, ArtifactDigest)],
    reward: &'a BTreeMap<String, f64>,
}

/// Completed evaluation output selected by the resolved task layout.
pub enum EvalExecutionResult {
    /// The unchanged output from an implicit single-step task.
    Single(LocalExecutionResult),
    /// The additive output from an explicit multi-step task.
    MultiStep(MultiStepExecutionResult),
}

#[derive(Serialize)]
struct MultiStepEvalOutput {
    task: String,
    artifacts: Vec<(String, ArtifactDigest)>,
    reward: BTreeMap<String, f64>,
    steps: Vec<MultiStepEvalStepOutput>,
}

#[derive(Serialize)]
struct MultiStepEvalStepOutput {
    name: String,
    artifacts: Vec<(String, ArtifactDigest)>,
    reward: BTreeMap<String, f64>,
}

#[derive(Serialize)]
struct LifecycleRecord<'a> {
    version: u32,
    source: LifecycleSourceProvenance<'a>,
    trial: &'a TrialSpec,
    verifier_result: &'a aiperf_runtime::eval::VerifierResult,
    initial_score: &'a aiperf_runtime::eval::ScoreVersion,
    regraded_score: &'a aiperf_runtime::eval::ScoreVersion,
    evidence: &'a [aiperf_runtime::eval::EvidenceEvent],
}

/// Immutable caller-selected package source retained with a lifecycle record.
#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LifecycleSourceProvenance<'a> {
    /// A caller-selected local source location.
    Local { location: &'a str },
    /// A Git package identified by its repository, exact revision, and package path.
    PinnedGit {
        repository: &'a str,
        revision: &'a str,
        package_path: &'a str,
    },
    /// A caller-selected registry package reference.
    Registry { reference: &'a str },
}

impl<'a> From<&'a HarborSource> for LifecycleSourceProvenance<'a> {
    fn from(source: &'a HarborSource) -> Self {
        match source {
            HarborSource::Local(location) => Self::Local { location },
            HarborSource::PinnedGit {
                repository,
                revision,
                package_path,
            } => Self::PinnedGit {
                repository,
                revision,
                package_path,
            },
            HarborSource::Registry(reference) => Self::Registry { reference },
        }
    }
}

/// Serializes a completed native evaluation result into its public JSON shape.
pub fn serialize_eval_result(
    task: &str,
    result: EvalExecutionResult,
) -> anyhow::Result<serde_json::Value> {
    match result {
        EvalExecutionResult::Single(result) => Ok(serde_json::to_value(EvalOutput {
            task,
            artifacts: &result.artifacts,
            reward: &result.reward.metrics,
        })?),
        EvalExecutionResult::MultiStep(result) => {
            let MultiStepExecutionResult { steps, reward, .. } = result;
            let artifacts = steps
                .last()
                .map(|step| step.artifacts.clone())
                .ok_or_else(|| {
                    anyhow::anyhow!("multi-step execution returned no verified steps")
                })?;
            Ok(serde_json::to_value(MultiStepEvalOutput {
                task: task.to_owned(),
                artifacts,
                reward: reward.metrics,
                steps: steps
                    .into_iter()
                    .map(|step| MultiStepEvalStepOutput {
                        name: step.name,
                        artifacts: step.artifacts,
                        reward: step.reward.metrics,
                    })
                    .collect(),
            })?)
        }
    }
}

/// Runs the native Harbor package lifecycle and emits one JSON summary.
pub fn run(args: &[String]) -> anyhow::Result<i32> {
    let flags =
        EvalFlags::try_parse_from(std::iter::once("eval".to_owned()).chain(args.iter().cloned()))?;
    let lifecycle = flags
        .lifecycle_request
        .as_deref()
        .map(read_lifecycle_request)
        .transpose()?;
    let lifecycle_output_explicit = flags.lifecycle_output.is_some();
    let lifecycle_output = flags
        .lifecycle_output
        .unwrap_or_else(|| PathBuf::from("aiperf-eval-lifecycle.json"));
    let has_external_agent_command = flags.agent_command.is_some();
    let native_options = native_graph::NativeGraphCliOptions {
        image: flags.image.clone(),
        workdir: flags.workdir.clone(),
        sandbox: flags.sandbox,
        requested_verifier_mode: flags.verifier_mode.map(VerifierMode::from),
        has_external_agent_command,
        lifecycle_output_explicit,
        records_output: flags.records_output.clone(),
    };
    if let Some(suite) = flags.suite.as_deref() {
        let model_runtime = flags.model_runtime.as_deref().ok_or_else(|| {
            anyhow::anyhow!("--model-runtime is required for schema-1.1 NativeGraph evaluation")
        })?;
        let lifecycle = lifecycle.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--lifecycle-request is required for scored NativeGraph evaluation")
        })?;
        return native_graph::run_suite(suite, model_runtime, lifecycle, native_options);
    }
    let requested_source = source_from_flags(
        flags.task,
        flags.git_repository,
        flags.git_revision,
        flags.git_path,
    )?;
    let (_pinned_tree, source) = materialize_pinned_directory(&requested_source)?;
    let imported = HarborImporter::new(&NativeSourceAcquirer).import(&source)?;
    if imported.package.native_graph().is_some() {
        let profile = imported
            .package
            .native_graph()
            .map(|native| native.profile())
            .ok_or_else(|| {
                anyhow::anyhow!("NativeGraph task snapshot disappeared before execution")
            })?;
        let model_runtime = match profile {
            aiperf_runtime::eval::NativeGraphProfile::NativeGraph => {
                Some(flags.model_runtime.as_deref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "--model-runtime is required for schema-1.1 NativeGraph evaluation"
                    )
                })?)
            }
            aiperf_runtime::eval::NativeGraphProfile::ExternallyDriven => {
                flags.model_runtime.as_deref()
            }
        };
        let lifecycle = lifecycle.as_ref().ok_or_else(|| {
            anyhow::anyhow!("--lifecycle-request is required for scored NativeGraph evaluation")
        })?;
        return native_graph::run_task(imported, model_runtime, lifecycle, native_options);
    }
    if flags.records_output.is_some() {
        anyhow::bail!("--records-output is available only for schema-1.1 NativeGraph evaluation");
    }
    let requested_verifier_mode = flags.verifier_mode.map(VerifierMode::from);
    let verifier_mode = requested_verifier_mode.unwrap_or_else(|| imported.package.verifier_mode());
    let use_docker = uses_docker(
        flags.sandbox,
        imported.package.is_standard_directory(),
        verifier_mode,
    );
    let image = match flags.image {
        Some(image) => image,
        None if use_docker && !imported.package.is_standard_directory() => {
            anyhow::bail!("--image is required for a legacy package with separate verification")
        }
        None if use_docker => DEFAULT_SANDBOX_IMAGE.to_owned(),
        None => anyhow::bail!("--image is required for the local sandbox backend"),
    };
    let recipe = if imported.package.is_standard_directory() {
        HarborSandboxRecipe::for_standard_task(image, flags.workdir)?
    } else {
        HarborSandboxRecipe::new(image, flags.workdir.unwrap_or_else(|| "/work".to_owned()))?
    };
    let agent_command = flags
        .agent_command
        .as_deref()
        .map(agent_command_argv)
        .unwrap_or_else(|| imported.package.agent_command().to_vec());
    let lifecycle_trial = if let Some(lifecycle) = &lifecycle {
        validate_lifecycle_execution(
            lifecycle,
            &imported,
            use_docker,
            verifier_mode,
            has_external_agent_command,
            &agent_command,
        )?;
        Some(HarborEvaluationCoordinator::resolve_trial(
            &imported, lifecycle,
        )?)
    } else {
        None
    };
    if use_docker && imported.package.is_standard_directory() {
        let plan = imported.package.execution_plan();
        let has_verifier_mode_conflict = requested_verifier_mode.is_some_and(|requested| {
            if plan.is_multi_step() {
                plan.steps()
                    .iter()
                    .any(|step| requested != step.verifier().mode())
            } else {
                requested != plan.verifier().mode()
            }
        });
        if has_verifier_mode_conflict {
            anyhow::bail!(
                "--verifier-mode conflicts with the standard task's normalized verifier environment"
            );
        }
    }
    if imported.package.execution_plan().is_multi_step() {
        if lifecycle.is_some() {
            anyhow::bail!(
                "lifecycle records require a single resolved verifier attempt; explicit multi-step packages are not yet lifecycle-addressable"
            );
        }
        if !use_docker {
            return Err(EvalExecutionError::UnsupportedMultiStep.into());
        }
        let result = DockerProcessSandbox::new().execute_multi_step(
            &recipe,
            &imported.package,
            &agent_command,
        )?;
        println!(
            "{}",
            serde_json::to_string(&serialize_eval_result(
                imported.task.id.as_str(),
                EvalExecutionResult::MultiStep(result),
            )?)?
        );
    } else {
        let result = if use_docker {
            DockerProcessSandbox::new().execute(
                &recipe,
                &imported.package,
                &agent_command,
                verifier_mode,
            )?
        } else {
            LocalProcessSandbox::new().execute_with_agent_command(
                &recipe,
                &imported.package,
                &agent_command,
                verifier_mode,
            )?
        };
        let mut output = serde_json::to_value(EvalOutput {
            task: imported.task.id.as_str(),
            artifacts: &result.artifacts,
            reward: &result.reward.metrics,
        })?;
        if let (Some(lifecycle), Some(trial)) = (lifecycle, lifecycle_trial) {
            let completed = HarborEvaluationCoordinator::complete_attempt(
                imported.clone(),
                trial,
                &agent_command,
                result,
                &lifecycle,
            )?;
            let record = LifecycleRecord {
                version: lifecycle.version,
                source: LifecycleSourceProvenance::from(&requested_source),
                trial: &completed.trial,
                verifier_result: &completed.verifier_result,
                initial_score: &completed.initial_score,
                regraded_score: &completed.regraded_score,
                evidence: &completed.evidence,
            };
            let record_json = serde_json::to_value(&record)?;
            output
                .as_object_mut()
                .ok_or_else(|| anyhow::anyhow!("evaluation output was not an object"))?
                .insert("lifecycle".to_owned(), record_json.clone());
            persist_lifecycle_record(&lifecycle_output, &record_json)?;
        }
        println!("{}", serde_json::to_string(&output)?);
    }
    Ok(0)
}

fn uses_docker(
    sandbox: SandboxFlag,
    is_standard_directory: bool,
    verifier_mode: VerifierMode,
) -> bool {
    match sandbox {
        SandboxFlag::Auto => is_standard_directory || verifier_mode == VerifierMode::Separate,
        SandboxFlag::Local => false,
        SandboxFlag::Docker => true,
    }
}

fn read_lifecycle_request(path: &Path) -> anyhow::Result<HarborLifecycleRequest> {
    let bytes = read_lifecycle_request_bytes(path)?;
    let request = serde_json::from_slice::<HarborLifecycleRequest>(&bytes).map_err(|error| {
        anyhow::anyhow!("invalid lifecycle request {}: {error}", path.display())
    })?;
    request.validate()?;
    Ok(request)
}

fn read_lifecycle_request_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    #[cfg(unix)]
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| {
            anyhow::anyhow!(
                "unable to read lifecycle request {}: {error}",
                path.display()
            )
        })?;
    #[cfg(not(unix))]
    let file = File::open(path).map_err(|error| {
        anyhow::anyhow!(
            "unable to read lifecycle request {}: {error}",
            path.display()
        )
    })?;
    if !file.metadata()?.is_file() {
        anyhow::bail!("lifecycle request {} is not a regular file", path.display());
    }
    let mut bytes = Vec::new();
    file.take(MAX_LIFECYCLE_REQUEST_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| {
            anyhow::anyhow!(
                "unable to read lifecycle request {}: {error}",
                path.display()
            )
        })?;
    if bytes.len() as u64 > MAX_LIFECYCLE_REQUEST_BYTES {
        anyhow::bail!(
            "lifecycle request {} exceeds {MAX_LIFECYCLE_REQUEST_BYTES} bytes",
            path.display()
        );
    }
    Ok(bytes)
}

fn validate_lifecycle_execution(
    lifecycle: &HarborLifecycleRequest,
    imported: &aiperf_runtime::eval::ImportedTask,
    use_docker: bool,
    verifier_mode: VerifierMode,
    has_external_agent_command: bool,
    command: &[String],
) -> anyhow::Result<()> {
    lifecycle.validate()?;
    if lifecycle.command != command {
        anyhow::bail!("lifecycle command provenance disagrees with the selected agent command");
    }
    match lifecycle.agent_contract {
        HarborLifecycleAgentContract::External if !has_external_agent_command => {
            anyhow::bail!("an external lifecycle contract requires --agent-command")
        }
        HarborLifecycleAgentContract::Installed if has_external_agent_command => {
            anyhow::bail!("an installed lifecycle contract must use the package agent command")
        }
        HarborLifecycleAgentContract::NativeGraph => {
            anyhow::bail!(
                "native_graph lifecycle contracts are not executable through --agent-command"
            )
        }
        HarborLifecycleAgentContract::ExternallyDriven => {
            anyhow::bail!(
                "externally_driven lifecycle contracts are not executable through --agent-command"
            )
        }
        HarborLifecycleAgentContract::External | HarborLifecycleAgentContract::Installed => {}
    }
    if !use_docker && verifier_mode == VerifierMode::Separate {
        return Err(EvalExecutionError::UnsupportedEnforcement(
            "separate verifier isolation for local lifecycle execution",
        )
        .into());
    }
    let (execution, verifier) = imported.package.timeouts().ok_or_else(|| {
        anyhow::anyhow!(
            "lifecycle execution requires paired agent.timeout_sec and verifier.timeout_sec for enforceable budgets"
        )
    })?;
    if lifecycle.budget.execution_seconds != execution.as_secs_f64()
        || lifecycle.budget.verifier_seconds != verifier.as_secs_f64()
    {
        anyhow::bail!("lifecycle budget disagrees with the package's effective phase deadlines");
    }
    Ok(())
}

fn persist_lifecycle_record(path: &Path, record: &serde_json::Value) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|error| {
        anyhow::anyhow!(
            "unable to create lifecycle record beside {}: {error}",
            path.display()
        )
    })?;
    let bytes = serde_json::to_vec(record)?;
    temporary.write_all(&bytes)?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).map_err(|error| {
        anyhow::anyhow!(
            "unable to atomically persist lifecycle record {}: {}",
            path.display(),
            error.error
        )
    })?;
    Ok(())
}

fn materialize_pinned_directory(
    source: &HarborSource,
) -> anyhow::Result<(Option<tempfile::TempDir>, HarborSource)> {
    let HarborSource::PinnedGit {
        repository,
        revision,
        package_path,
    } = source
    else {
        return Ok((None, source.clone()));
    };
    let root = tempfile::tempdir()?;
    let checkout = root.path().join("repository");
    let initialize =
        git_output_bounded(Command::new("git").args(["init", "--quiet"]).arg(&checkout))?;
    if !initialize.status.success() {
        anyhow::bail!(
            "unable to initialize pinned Git repository: {}",
            String::from_utf8_lossy(&initialize.stderr).trim()
        );
    }
    let fetch = git_output_bounded(Command::new("git").args([
        "-C",
        checkout.to_string_lossy().as_ref(),
        "-c",
        "protocol.version=2",
        "fetch",
        "--depth=1",
        "--filter=blob:none",
        "--no-tags",
        repository,
        revision,
    ]))?;
    if !fetch.status.success() {
        anyhow::bail!(
            "unable to fetch pinned Git revision: {}",
            String::from_utf8_lossy(&fetch.stderr).trim()
        );
    }
    let object_type = git_output_bounded(Command::new("git").args([
        "-C",
        checkout.to_string_lossy().as_ref(),
        "cat-file",
        "-t",
        revision,
    ]))?;
    if !object_type.status.success() || object_type.stdout.trim_ascii() != b"commit" {
        anyhow::bail!("pinned Git revision must identify a commit");
    }
    let checkout_result = git_output_bounded(Command::new("git").args([
        "-C",
        checkout.to_string_lossy().as_ref(),
        "checkout",
        "--detach",
        revision,
    ]))?;
    if !checkout_result.status.success() {
        anyhow::bail!(
            "unable to check out pinned Git revision: {}",
            String::from_utf8_lossy(&checkout_result.stderr).trim()
        );
    }
    let head = git_output_bounded(Command::new("git").args([
        "-C",
        checkout.to_string_lossy().as_ref(),
        "rev-parse",
        "--verify",
        "HEAD",
    ]))?;
    if !head.status.success() || head.stdout.trim_ascii() != revision.as_bytes() {
        anyhow::bail!("checked out Git revision does not match the requested commit");
    }
    fs::remove_dir_all(checkout.join(".git"))?;
    let package = checkout.join(package_path);
    let is_package_manifest = matches!(
        Path::new(package_path).file_name(),
        Some(name) if name == "task.toml" || name == "task.json"
    );
    let location = if is_package_manifest {
        package
            .parent()
            .map_or_else(|| checkout.clone(), std::path::Path::to_path_buf)
    } else {
        package
    };
    Ok((Some(root), HarborSource::local(location.to_string_lossy())?))
}

const MAX_GIT_COMMAND_OUTPUT_BYTES: usize = 1024 * 1024;
const GIT_MATERIALIZATION_TIMEOUT: Duration = Duration::from_secs(60);

fn git_output_bounded(command: &mut Command) -> anyhow::Result<Output> {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = command.spawn()?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| anyhow::anyhow!("could not capture Git command stdout"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| anyhow::anyhow!("could not capture Git command stderr"))?;
    let stdout = thread::spawn(move || read_git_output(stdout));
    let stderr = thread::spawn(move || read_git_output(stderr));
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if started.elapsed() >= GIT_MATERIALIZATION_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout.join();
            let _ = stderr.join();
            anyhow::bail!("pinned Git materialization exceeded its time limit");
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = stdout
        .join()
        .map_err(|_| anyhow::anyhow!("Git stdout reader panicked"))??;
    let stderr = stderr
        .join()
        .map_err(|_| anyhow::anyhow!("Git stderr reader panicked"))??;
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

fn read_git_output(mut source: impl Read) -> io::Result<Vec<u8>> {
    let mut output = Vec::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            return Ok(output);
        }
        if output.len().saturating_add(count) > MAX_GIT_COMMAND_OUTPUT_BYTES {
            return Err(io::Error::other("Git command output exceeded its limit"));
        }
        output.extend_from_slice(&buffer[..count]);
    }
}

fn agent_command_argv(command: &str) -> Vec<String> {
    vec!["/bin/sh".to_owned(), "-c".to_owned(), command.to_owned()]
}

fn source_from_flags(
    task: Option<PathBuf>,
    git_repository: Option<String>,
    git_revision: Option<String>,
    git_path: Option<String>,
) -> anyhow::Result<HarborSource> {
    match (task, git_repository, git_revision, git_path) {
        (Some(path), None, None, None) => Ok(HarborSource::local(path.to_string_lossy())?),
        (None, Some(repository), Some(revision), Some(path)) => {
            Ok(HarborSource::pinned_git(repository, revision, path)?)
        }
        _ => anyhow::bail!(
            "provide either --task or --git-repository with --git-revision and --git-path"
        ),
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, process::Command};

    #[cfg(unix)]
    use std::os::unix::{ffi::OsStrExt, fs::FileTypeExt};

    use super::{
        LifecycleSourceProvenance, SandboxFlag, agent_command_argv, git_output_bounded,
        materialize_pinned_directory, persist_lifecycle_record, read_lifecycle_request,
        source_from_flags, uses_docker,
    };
    use aiperf_runtime::eval::{
        HarborImporter, HarborSandboxRecipe, HarborSource, LocalProcessSandbox,
        NativeSourceAcquirer, SandboxRole, VerifierMode,
    };

    #[test]
    fn source_flags_require_one_complete_source_form() {
        assert!(source_from_flags(Some("task.json".into()), None, None, None).is_ok());
        assert!(
            source_from_flags(
                None,
                Some("tasks".to_owned()),
                Some("a".repeat(40)),
                Some("task.json".to_owned()),
            )
            .is_ok()
        );
        assert!(source_from_flags(None, Some("tasks".to_owned()), None, None).is_err());
    }

    #[test]
    fn external_agent_command_uses_a_shell_argv() {
        assert_eq!(
            agent_command_argv("printf result > result.txt"),
            ["/bin/sh", "-c", "printf result > result.txt"]
        );
    }

    #[test]
    fn lifecycle_request_is_versioned_strict_and_persists_canonical_record() {
        let temporary = tempfile::tempdir().unwrap();
        let request_path = temporary.path().join("request.json");
        let digest = "blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        fs::write(
            &request_path,
            format!(
                r#"{{"version":1,"agent_variant":"agent:v1","model":{{"provider":"provider","model":"model"}},"seed":7,"policy":"{digest}","runtime":"native:v1","attempt":"attempt-1","budget":{{"execution_seconds":2.0,"verifier_seconds":3.0}},"agent_contract":"external","command":["/bin/sh","-c","true"],"initial_score":{{"metric":"reward","rationale":"{digest}"}},"regrade":{{"metric":"reward","rationale":"{digest}"}}}}"#
            ),
        )
        .unwrap();
        let request = read_lifecycle_request(&request_path).unwrap();
        assert_eq!(request.version, 1);
        let output_path = temporary.path().join("record.json");
        let record = serde_json::json!({"version": request.version, "trial": {"seed": 7}});
        persist_lifecycle_record(&output_path, &record).unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&fs::read(output_path).unwrap()).unwrap(),
            record
        );

        fs::write(
            &request_path,
            format!(
                r#"{{"version":1,"agent_variant":"agent:v1","model":{{"provider":"provider","model":"model"}},"seed":7,"policy":"{digest}","runtime":"native:v1","attempt":"attempt-1","budget":{{"execution_seconds":2.0,"verifier_seconds":3.0}},"agent_contract":"external","command":["/bin/sh"],"initial_score":{{"metric":"reward","rationale":"{digest}"}},"regrade":{{"metric":"reward","rationale":"{digest}"}},"forged_environment":"no"}}"#
            ),
        )
        .unwrap();
        assert!(read_lifecycle_request(&request_path).is_err());
    }

    #[test]
    fn lifecycle_request_rejects_oversize_input_before_json_parsing() {
        let temporary = tempfile::tempdir().unwrap();
        let request_path = temporary.path().join("request.json");
        fs::write(&request_path, vec![b'x'; 1024 * 1024 + 1]).unwrap();

        let error = read_lifecycle_request(&request_path).unwrap_err();

        assert!(error.to_string().contains("exceeds 1048576 bytes"));
    }

    #[cfg(unix)]
    #[test]
    fn lifecycle_request_rejects_fifo_without_waiting_for_a_writer() {
        let temporary = tempfile::tempdir().unwrap();
        let request_path = temporary.path().join("request.json");
        let path = std::ffi::CString::new(request_path.as_os_str().as_bytes()).unwrap();
        // SAFETY: `path` is NUL-terminated and this fresh temporary path does not exist.
        assert_eq!(unsafe { libc::mkfifo(path.as_ptr(), 0o600) }, 0);
        assert!(fs::metadata(&request_path).unwrap().file_type().is_fifo());

        let error = read_lifecycle_request(&request_path).unwrap_err();

        assert!(error.to_string().contains("not a regular file"));
    }

    #[test]
    fn lifecycle_provenance_keeps_the_original_pinned_git_identity() {
        let first = HarborSource::pinned_git(
            "https://example.invalid/tasks.git",
            "a".repeat(40),
            "tasks/example/task.toml",
        )
        .unwrap();
        let second = HarborSource::pinned_git(
            "https://example.invalid/tasks.git",
            "b".repeat(40),
            "tasks/example/task.toml",
        )
        .unwrap();

        let first = serde_json::to_value(LifecycleSourceProvenance::from(&first)).unwrap();
        let second = serde_json::to_value(LifecycleSourceProvenance::from(&second)).unwrap();

        assert_ne!(first, second);
        assert_eq!(first["kind"], "pinned_git");
        assert_eq!(first["repository"], "https://example.invalid/tasks.git");
        assert_eq!(first["revision"], "a".repeat(40));
        assert_eq!(first["package_path"], "tasks/example/task.toml");
    }

    #[test]
    fn materializes_a_pinned_standard_task_tree() {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("tasks");
        fs::create_dir(&repository).unwrap();
        for arguments in [
            ["init"].as_slice(),
            ["config", "user.email", "eval@example.invalid"].as_slice(),
            ["config", "user.name", "Eval"].as_slice(),
        ] {
            assert!(
                Command::new("git")
                    .arg("-C")
                    .arg(&repository)
                    .args(arguments)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        fs::create_dir_all(repository.join("task/environment")).unwrap();
        fs::create_dir_all(repository.join("task/tests")).unwrap();
        fs::write(
            repository.join("task/task.toml"),
            "schema_version = \"1.0\"\n[task]\nname = \"example/pinned\"\n",
        )
        .unwrap();
        fs::write(repository.join("task/instruction.md"), "Do work.\n").unwrap();
        fs::write(
            repository.join("task/environment/Dockerfile"),
            "FROM scratch\n",
        )
        .unwrap();
        fs::write(repository.join("task/tests/test.sh"), "exit 0\n").unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["add", "."])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .arg("-c")
                .arg("commit.gpgsign=false")
                .arg("-C")
                .arg(&repository)
                .args(["commit", "-m", "task"])
                .status()
                .unwrap()
                .success()
        );
        let revision = String::from_utf8(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_owned();
        let (tree, source) = materialize_pinned_directory(
            &HarborSource::pinned_git(repository.to_string_lossy(), revision, "task/task.toml")
                .unwrap(),
        )
        .unwrap();
        assert!(
            tree.unwrap()
                .path()
                .join("repository/task/task.toml")
                .is_file()
        );
        assert!(matches!(source, HarborSource::Local(_)));
    }

    #[test]
    fn pinned_git_materialization_rejects_an_annotated_tag_object_id() {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("tasks");
        fs::create_dir(&repository).unwrap();
        for arguments in [
            ["init"].as_slice(),
            ["config", "user.email", "eval@example.invalid"].as_slice(),
            ["config", "user.name", "Eval"].as_slice(),
        ] {
            assert!(
                Command::new("git")
                    .arg("-C")
                    .arg(&repository)
                    .args(arguments)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        fs::write(repository.join("task.json"), "{\"id\":\"pinned\"}").unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["add", "task.json"])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .arg("-c")
                .arg("commit.gpgsign=false")
                .arg("-C")
                .arg(&repository)
                .args(["commit", "-m", "task"])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["tag", "-a", "release", "-m", "release"])
                .status()
                .unwrap()
                .success()
        );
        let tag = String::from_utf8(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["rev-parse", "release^{tag}"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_owned();

        let error = materialize_pinned_directory(
            &HarborSource::pinned_git(repository.to_string_lossy(), tag, "task.json").unwrap(),
        )
        .expect_err("annotated tags are not immutable commit pins");

        assert!(error.to_string().contains("commit"));
    }

    #[test]
    fn git_materialization_bounds_command_output() {
        let error =
            git_output_bounded(Command::new("sh").args(["-c", "head -c 1048577 /dev/zero"]))
                .expect_err("Git command output must be bounded");

        assert!(error.to_string().contains("output exceeded"));
    }

    #[test]
    fn pinned_git_json_package_retains_its_directory_snapshot_after_mutation() {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("tasks");
        fs::create_dir(&repository).unwrap();
        for arguments in [
            ["init"].as_slice(),
            ["config", "user.email", "eval@example.invalid"].as_slice(),
            ["config", "user.name", "Eval"].as_slice(),
        ] {
            assert!(
                Command::new("git")
                    .arg("-C")
                    .arg(&repository)
                    .args(arguments)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        fs::create_dir(repository.join("task")).unwrap();
        fs::write(
            repository.join("task/task.json"),
            r#"{"id":"pinned-json","instruction":"Inspect the sibling.","environment":"blake3:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","verifier":"blake3:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","agent_command":["sh","-c","true"],"verifier_command":["sh","-c","true"],"declared_artifacts":[]}"#,
        )
        .unwrap();
        fs::write(repository.join("task/sibling.txt"), "pinned sibling\n").unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["add", "task"])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .arg("-c")
                .arg("commit.gpgsign=false")
                .arg("-C")
                .arg(&repository)
                .args(["commit", "-m", "JSON package"])
                .status()
                .unwrap()
                .success()
        );
        let revision = String::from_utf8(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_owned();
        let (tree, source) = materialize_pinned_directory(
            &HarborSource::pinned_git(repository.to_string_lossy(), revision, "task/task.json")
                .unwrap(),
        )
        .unwrap();
        let pinned_tree = tree.unwrap();
        let source_path = match &source {
            HarborSource::Local(location) => std::path::PathBuf::from(location),
            _ => panic!("pinned source must materialize as a local directory"),
        };
        assert_eq!(source_path, pinned_tree.path().join("repository/task"));

        let imported = HarborImporter::new(&NativeSourceAcquirer)
            .import(&source)
            .unwrap();
        fs::write(source_path.join("sibling.txt"), "mutated sibling\n").unwrap();

        let sandbox = LocalProcessSandbox::new()
            .materialize(
                &HarborSandboxRecipe::new(
                    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "/work",
                )
                .unwrap(),
                &imported.package,
                SandboxRole::Agent,
            )
            .unwrap();
        assert_eq!(
            fs::read(sandbox.root().join("sibling.txt")).unwrap(),
            b"pinned sibling\n"
        );
    }

    #[test]
    fn root_pinned_manifest_materialization_excludes_git_metadata() {
        let temporary = tempfile::tempdir().unwrap();
        let repository = temporary.path().join("tasks");
        fs::create_dir(&repository).unwrap();
        for arguments in [
            ["init"].as_slice(),
            ["config", "user.email", "eval@example.invalid"].as_slice(),
            ["config", "user.name", "Eval"].as_slice(),
        ] {
            assert!(
                Command::new("git")
                    .arg("-C")
                    .arg(&repository)
                    .args(arguments)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        fs::write(
            repository.join("task.toml"),
            "schema_version = \"1.0\"\n[task]\nname = \"example/root\"\n",
        )
        .unwrap();
        fs::write(repository.join("instruction.md"), "Do work.\n").unwrap();
        fs::create_dir_all(repository.join("environment")).unwrap();
        fs::write(repository.join("environment/Dockerfile"), "FROM scratch\n").unwrap();
        fs::create_dir_all(repository.join("tests")).unwrap();
        fs::write(repository.join("tests/test.sh"), "exit 0\n").unwrap();
        assert!(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["add", "."])
                .status()
                .unwrap()
                .success()
        );
        assert!(
            Command::new("git")
                .arg("-c")
                .arg("commit.gpgsign=false")
                .arg("-C")
                .arg(&repository)
                .args(["commit", "-m", "root task"])
                .status()
                .unwrap()
                .success()
        );
        let revision = String::from_utf8(
            Command::new("git")
                .arg("-C")
                .arg(&repository)
                .args(["rev-parse", "HEAD"])
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_owned();

        let (tree, source) = materialize_pinned_directory(
            &HarborSource::pinned_git(repository.to_string_lossy(), revision, "task.toml").unwrap(),
        )
        .unwrap();
        let tree = tree.unwrap();
        let source_path = match source {
            HarborSource::Local(location) => std::path::PathBuf::from(location),
            _ => panic!("pinned source must materialize as a local directory"),
        };
        assert_eq!(source_path, tree.path().join("repository"));
        assert!(!source_path.join(".git").exists());
    }

    #[test]
    fn auto_sandbox_uses_docker_for_separate_legacy_verification() {
        assert!(uses_docker(
            SandboxFlag::Auto,
            false,
            VerifierMode::Separate
        ));
        assert!(!uses_docker(SandboxFlag::Auto, false, VerifierMode::Shared));
    }
}
