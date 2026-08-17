// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Temporary-root local process execution for shared-verifier compatibility.

use std::{
    fs::{self, OpenOptions},
    io::Read,
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use tempfile::TempDir;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    process::{Child as TokioChild, ChildStdin, ChildStdout, Command as TokioCommand},
};

use crate::eval::{
    AdapterExit, AdapterProcess, AdapterSpawnRequest, AdapterSpawnTransaction, AdapterSpawner,
    AdapterSupervisionError, ArtifactDigest, AttemptId, CancelReason, HarborTaskPackage,
    RegradeError, RewardDocument, ScoreVersion, VerifierMode,
};

use super::{EvalExecutionError, HarborSandboxRecipe, ProviderCapabilities};

const MAX_LOCAL_FILE_BYTES: u64 = 1024 * 1024;

/// Tokio-process implementation of the streaming local adapter process seam.
#[derive(Clone, Debug)]
pub struct LocalAdapterSpawner {
    workdir: PathBuf,
}

impl LocalAdapterSpawner {
    /// Launches adapter argv within this already-isolated local task root.
    pub fn new(workdir: impl Into<PathBuf>) -> Self {
        Self {
            workdir: workdir.into(),
        }
    }
}

impl AdapterSpawner for LocalAdapterSpawner {
    fn begin_spawn(
        &self,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        let (program, arguments) = request
            .argv()
            .split_first()
            .ok_or(AdapterSupervisionError::InvalidSpawnRequest("argv"))?;
        let mut command = TokioCommand::new(program);
        command
            .args(arguments)
            .current_dir(&self.workdir)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .envs(request.environment())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        configure_adapter_process_group(&mut command);
        let process = spawn_adapter_child(command, request.max_stderr_bytes())?;
        Ok(Box::new(LocalAdapterSpawnTransaction {
            process: Some(process),
        }))
    }
}

/// Launch transaction which keeps a local child fenced until ownership transfers.
struct LocalAdapterSpawnTransaction {
    process: Option<Box<dyn AdapterProcess>>,
}

#[async_trait(?Send)]
impl AdapterSpawnTransaction for LocalAdapterSpawnTransaction {
    async fn await_process(&mut self) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
        self.process
            .take()
            .ok_or(AdapterSupervisionError::AlreadyReaped)
    }

    async fn abort(&mut self, deadline: Duration) -> Result<(), AdapterSupervisionError> {
        let Some(mut process) = self.process.take() else {
            return Ok(());
        };
        let cancel = process.cancel(CancelReason::HostShutdown, deadline).await;
        let reap = process.reap(deadline).await;
        match (cancel, reap) {
            (Ok(()), Ok(_)) => Ok(()),
            (Err(primary), Ok(_)) => Err(primary),
            (Ok(()), Err(recovery)) => {
                self.process = Some(process);
                Err(recovery)
            }
            (Err(primary), Err(recovery)) => {
                self.process = Some(process);
                Err(AdapterSupervisionError::Recovery {
                    primary: Box::new(primary),
                    recovery: Box::new(recovery),
                })
            }
        }
    }

    fn fence(&mut self) {
        if let Some(process) = self.process.as_deref_mut() {
            process.fence();
        }
    }
}

/// Starts a process whose stdin/stdout/stderr are retained for adapter supervision.
pub(crate) fn spawn_adapter_child(
    mut command: TokioCommand,
    max_stderr_bytes: usize,
) -> Result<Box<dyn AdapterProcess>, AdapterSupervisionError> {
    let mut child = command.spawn().map_err(|error| {
        AdapterSupervisionError::Process(format!("cannot start adapter: {error}"))
    })?;
    let process_group_id = child.id().ok_or_else(|| {
        AdapterSupervisionError::Process("adapter has no process identifier".to_owned())
    })? as i32;
    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| AdapterSupervisionError::Process("adapter has no stdin".to_owned()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| AdapterSupervisionError::Process("adapter has no stdout".to_owned()))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| AdapterSupervisionError::Process("adapter has no stderr".to_owned()))?;
    let diagnostics = Arc::new(Mutex::new(AdapterDiagnostics::default()));
    let stderr_diagnostics = diagnostics.clone();
    let stderr_task = tokio::spawn(async move {
        let mut stderr = stderr;
        let mut chunk = [0_u8; 4096];
        loop {
            let read = match stderr.read(&mut chunk).await {
                Ok(read) => read,
                Err(_) => return,
            };
            if read == 0 {
                return;
            }
            let Ok(mut diagnostics) = stderr_diagnostics.lock() else {
                return;
            };
            let remaining = max_stderr_bytes
                .saturating_add(1)
                .saturating_sub(diagnostics.bytes.len());
            let retained = remaining.min(read);
            diagnostics.bytes.extend_from_slice(&chunk[..retained]);
            diagnostics.overflowed |= retained < read;
        }
    });
    Ok(Box::new(TokioAdapterProcess {
        child,
        process_group_id,
        stdin,
        stdout,
        diagnostics,
        stderr_task,
    }))
}

#[derive(Default)]
struct AdapterDiagnostics {
    bytes: Vec<u8>,
    overflowed: bool,
}

struct TokioAdapterProcess {
    child: TokioChild,
    process_group_id: i32,
    stdin: ChildStdin,
    stdout: ChildStdout,
    diagnostics: Arc<Mutex<AdapterDiagnostics>>,
    stderr_task: tokio::task::JoinHandle<()>,
}

#[async_trait(?Send)]
impl AdapterProcess for TokioAdapterProcess {
    async fn write_frame(
        &mut self,
        frame: &[u8],
        deadline: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        tokio::time::timeout(deadline, async {
            self.stdin.write_all(frame).await?;
            self.stdin.flush().await
        })
        .await
        .map_err(|_| AdapterSupervisionError::Process("adapter stdin deadline elapsed".to_owned()))?
        .map_err(|error| {
            AdapterSupervisionError::Process(format!("cannot write adapter stdin: {error}"))
        })
    }

    async fn read_stdout_frame(
        &mut self,
        max_bytes: usize,
        deadline: std::time::Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        let deadline = tokio::time::Instant::now()
            .checked_add(deadline)
            .ok_or_else(|| {
                AdapterSupervisionError::Process("adapter stdout deadline is invalid".to_owned())
            })?;
        let mut frame = Vec::with_capacity(max_bytes.min(4096));
        loop {
            let mut byte = [0_u8; 1];
            let remaining = deadline
                .checked_duration_since(tokio::time::Instant::now())
                .filter(|remaining| !remaining.is_zero())
                .ok_or_else(|| {
                    AdapterSupervisionError::Process("adapter stdout deadline elapsed".to_owned())
                })?;
            let read = tokio::time::timeout(remaining, self.stdout.read(&mut byte))
                .await
                .map_err(|_| {
                    AdapterSupervisionError::Process("adapter stdout deadline elapsed".to_owned())
                })?
                .map_err(|error| {
                    AdapterSupervisionError::Process(format!("cannot read adapter stdout: {error}"))
                })?;
            if read == 0 {
                return if frame.is_empty() {
                    Err(AdapterSupervisionError::EndOfStream)
                } else {
                    Err(AdapterSupervisionError::Process(
                        "adapter stdout ended before a JSONL frame terminator".to_owned(),
                    ))
                };
            }
            if frame.len() == max_bytes {
                return Err(AdapterSupervisionError::bounded_stdout_frame(
                    max_bytes.saturating_add(1),
                    max_bytes,
                ));
            }
            frame.push(byte[0]);
            if byte[0] == b'\n' {
                return Ok(frame);
            }
        }
    }

    async fn drain_stderr(&mut self, max_bytes: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        let mut diagnostics = self.diagnostics.lock().map_err(|_| {
            AdapterSupervisionError::Process("adapter diagnostics lock poisoned".to_owned())
        })?;
        if diagnostics.overflowed || diagnostics.bytes.len() > max_bytes {
            return Err(AdapterSupervisionError::bounded_diagnostic_output(
                diagnostics.bytes.len(),
                max_bytes,
            ));
        }
        Ok(std::mem::take(&mut diagnostics.bytes))
    }

    async fn cancel(
        &mut self,
        _: CancelReason,
        _: std::time::Duration,
    ) -> Result<(), AdapterSupervisionError> {
        signal_adapter_process_group(self.process_group_id, libc::SIGTERM)
    }

    async fn reap(
        &mut self,
        deadline: std::time::Duration,
    ) -> Result<AdapterExit, AdapterSupervisionError> {
        let deadline = tokio::time::Instant::now()
            .checked_add(deadline)
            .ok_or_else(|| {
                AdapterSupervisionError::Process("adapter reap deadline is invalid".to_owned())
            })?;
        signal_adapter_process_group(self.process_group_id, libc::SIGKILL)?;
        let remaining = remaining_adapter_deadline(deadline, "adapter reap deadline elapsed")?;
        tokio::time::timeout(remaining, self.child.wait())
            .await
            .map_err(|_| {
                AdapterSupervisionError::Process("adapter reap deadline elapsed".to_owned())
            })?
            .map_err(|error| {
                AdapterSupervisionError::Process(format!("cannot reap adapter: {error}"))
            })?;
        wait_for_adapter_process_group_exit(self.process_group_id, deadline).await?;
        self.stderr_task.abort();
        Ok(AdapterExit::Reaped)
    }

    fn fence(&mut self) {
        let _ = signal_adapter_process_group(self.process_group_id, libc::SIGKILL);
        self.stderr_task.abort();
    }
}

#[cfg(unix)]
pub(crate) fn configure_adapter_process_group(command: &mut TokioCommand) {
    // SAFETY: `setsid` is async-signal-safe in the forked process, giving this
    // session ownership over its direct child and every adapter descendant.
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() < 0 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(())
        });
    }
}

#[cfg(not(unix))]
pub(crate) fn configure_adapter_process_group(_: &mut TokioCommand) {}

#[cfg(unix)]
fn signal_adapter_process_group(
    process_group_id: i32,
    signal: libc::c_int,
) -> Result<(), AdapterSupervisionError> {
    let result = unsafe { libc::kill(-process_group_id, signal) };
    if result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(AdapterSupervisionError::Process(format!(
            "cannot signal adapter process group: {}",
            std::io::Error::last_os_error()
        )))
    }
}

#[cfg(unix)]
async fn wait_for_adapter_process_group_exit(
    process_group_id: i32,
    deadline: tokio::time::Instant,
) -> Result<(), AdapterSupervisionError> {
    while adapter_process_group_exists(process_group_id)? {
        let remaining = remaining_adapter_deadline(deadline, "adapter reap deadline elapsed")?;
        tokio::time::sleep(remaining.min(Duration::from_millis(10))).await;
    }
    Ok(())
}

#[cfg(unix)]
fn adapter_process_group_exists(process_group_id: i32) -> Result<bool, AdapterSupervisionError> {
    let result = unsafe { libc::kill(-process_group_id, 0) };
    if result == 0 {
        return Ok(true);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(AdapterSupervisionError::Process(format!(
            "cannot inspect adapter process group: {}",
            std::io::Error::last_os_error()
        ))),
    }
}

#[cfg(unix)]
fn remaining_adapter_deadline(
    deadline: tokio::time::Instant,
    message: &str,
) -> Result<Duration, AdapterSupervisionError> {
    deadline
        .checked_duration_since(tokio::time::Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| AdapterSupervisionError::Process(message.to_owned()))
}

#[cfg(not(unix))]
fn signal_adapter_process_group(_: i32, _: libc::c_int) -> Result<(), AdapterSupervisionError> {
    Err(AdapterSupervisionError::Process(
        "adapter process-group cleanup requires a Unix host".to_owned(),
    ))
}

#[cfg(not(unix))]
async fn wait_for_adapter_process_group_exit(
    _: i32,
    _: tokio::time::Instant,
) -> Result<(), AdapterSupervisionError> {
    Err(AdapterSupervisionError::Process(
        "adapter process-group cleanup requires a Unix host".to_owned(),
    ))
}

/// Selects the isolated root materialized for one evaluation participant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SandboxRole {
    /// Root in which the selected agent executes.
    Agent,
    /// Root for a verifier explicitly authorized to share the task sandbox.
    SharedVerifier,
    /// Fresh root for a separately provisioned verifier.
    SeparateVerifier,
}

/// Concrete local process provider for deterministic P0 package execution.
#[derive(Debug, Default)]
pub struct LocalProcessSandbox;

impl LocalProcessSandbox {
    /// Creates an empty local-process provider.
    pub const fn new() -> Self {
        Self
    }

    /// Writes exactly the acquired package bytes into an isolated temporary root.
    pub fn materialize(
        &self,
        _: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        _: SandboxRole,
    ) -> Result<MaterializedSandbox, EvalExecutionError> {
        let lease = tempfile::tempdir()
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        package.materialize_source_into(lease.path())?;
        fs::create_dir_all(lease.path().join("results"))
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        Ok(MaterializedSandbox { lease })
    }

    /// Runs the package agent and an explicitly shared verifier in one temporary root.
    pub fn execute(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        verifier_mode: VerifierMode,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        self.execute_with_agent_command(recipe, package, package.agent_command(), verifier_mode)
    }

    /// Runs a package with a caller-supplied external agent command.
    pub fn execute_with_agent_command(
        &self,
        recipe: &HarborSandboxRecipe,
        package: &HarborTaskPackage,
        agent_command: &[String],
        verifier_mode: VerifierMode,
    ) -> Result<LocalExecutionResult, EvalExecutionError> {
        if verifier_mode == VerifierMode::Separate {
            return Err(EvalExecutionError::UnsupportedEnforcement(
                "separate verifier isolation",
            ));
        }
        if package.execution_plan().is_multi_step() {
            return Err(EvalExecutionError::UnsupportedMultiStep);
        }
        if package.is_standard_directory() {
            package
                .execution_plan()
                .validate_for(ProviderCapabilities::none())?;
        }
        let agent = self.materialize(recipe, package, SandboxRole::Agent)?;
        let environment = vec![(
            "AIPERF_EVAL_INSTRUCTION".to_owned(),
            package.instruction().to_owned(),
        )];
        agent.run(agent_command, &environment)?;
        let artifacts = collect_declared_artifacts(&agent, package)?;
        agent.run(package.verifier_command(), &environment)?;
        let reward = parse_reward(&agent)?;
        let verifier = ArtifactDigest::parse(package.verifier())
            .map_err(|error| EvalExecutionError::Materialization(error.to_string()))?;
        Ok(LocalExecutionResult {
            artifacts,
            reward,
            verifier,
        })
    }
}

/// A live local sandbox root whose lease removes it when evaluation finishes.
#[derive(Debug)]
pub struct MaterializedSandbox {
    lease: TempDir,
}

impl MaterializedSandbox {
    /// Returns the private process root.
    pub fn root(&self) -> &Path {
        self.lease.path()
    }

    /// Maps a declared absolute artifact path into this private root.
    pub fn artifact_path(&self, declared_path: &str) -> Result<PathBuf, EvalExecutionError> {
        let relative = declared_path
            .strip_prefix('/')
            .filter(|path| {
                !path.is_empty() && !path.split('/').any(|part| part == "." || part == "..")
            })
            .ok_or_else(|| {
                EvalExecutionError::Materialization("invalid declared artifact path".to_owned())
            })?;
        Ok(self.root().join(relative))
    }

    /// Runs an argv with no inherited environment in this sandbox root.
    pub fn run(
        &self,
        argv: &[String],
        environment: &[(String, String)],
    ) -> Result<ProcessOutput, EvalExecutionError> {
        let (program, arguments) = argv
            .split_first()
            .ok_or(EvalExecutionError::InvalidCommand)?;
        if program.trim().is_empty() || arguments.iter().any(|argument| argument.trim().is_empty())
        {
            return Err(EvalExecutionError::InvalidCommand);
        }
        let status = Command::new(program)
            .args(arguments)
            .current_dir(self.root())
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("AIPERF_EVAL_ROOT", self.root())
            .envs(environment.iter().map(|(key, value)| (key, value)))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map_err(|_| EvalExecutionError::ProcessSpawn(program.clone()))?;
        if !status.success() {
            return Err(EvalExecutionError::ProcessFailure(program.clone()));
        }
        Ok(ProcessOutput {
            status,
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
}

/// Captured output from a successful local sandbox process.
#[derive(Debug)]
pub struct ProcessOutput {
    /// Terminal process status.
    pub status: ExitStatus,
    /// Captured standard output bytes.
    pub stdout: Vec<u8>,
    /// Captured standard error bytes.
    pub stderr: Vec<u8>,
}

/// Immutable artifacts and reward emitted by one completed native local evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct LocalExecutionResult {
    /// Declared artifact paths paired with content digests.
    pub artifacts: Vec<(String, ArtifactDigest)>,
    /// Finite verifier reward metrics.
    pub reward: RewardDocument,
    /// Immutable verifier implementation identity that produced the reward.
    pub verifier: ArtifactDigest,
}

impl LocalExecutionResult {
    /// Produces the initial immutable score revision from this verifier result.
    pub fn initial_score(
        &self,
        attempt: AttemptId,
        metric: impl Into<String>,
        rationale: ArtifactDigest,
    ) -> Result<ScoreVersion, RegradeError> {
        let metric = metric.into();
        let value = self
            .reward
            .metrics
            .get(&metric)
            .copied()
            .ok_or_else(|| RegradeError::MetricNotFound(metric.clone()))?;
        ScoreVersion::initial(
            attempt,
            self.verifier.clone(),
            self.artifacts
                .iter()
                .map(|(_, digest)| digest.clone())
                .collect(),
            metric,
            value,
            rationale,
        )
        .map_err(RegradeError::InvalidScore)
    }
}

fn collect_declared_artifacts(
    sandbox: &MaterializedSandbox,
    package: &HarborTaskPackage,
) -> Result<Vec<(String, ArtifactDigest)>, EvalExecutionError> {
    package
        .declared_artifacts()
        .iter()
        .map(|path| {
            let artifact_path = sandbox.artifact_path(path)?;
            let bytes = read_file_bounded(&artifact_path, "declared artifact")?;
            Ok((path.clone(), ArtifactDigest::from_bytes(&bytes)))
        })
        .collect()
}

fn parse_reward(sandbox: &MaterializedSandbox) -> Result<RewardDocument, EvalExecutionError> {
    let reward_json = read_optional_file_bounded(&sandbox.root().join("reward.json"))?;
    let reward_txt = read_optional_file_bounded(&sandbox.root().join("reward.txt"))?;
    RewardDocument::parse(reward_json.as_deref(), reward_txt.as_deref())
        .map_err(|error| EvalExecutionError::ProcessFailure(format!("verifier reward: {error}")))
}

fn read_optional_file_bounded(path: &Path) -> Result<Option<Vec<u8>>, EvalExecutionError> {
    let file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) if error.raw_os_error() == Some(libc::ELOOP) => {
            return Err(EvalExecutionError::ArtifactCollection(format!(
                "verifier reward is not a regular file: {}",
                path.display()
            )));
        }
        Err(error) => return Err(EvalExecutionError::ArtifactCollection(error.to_string())),
    };
    ensure_regular_file(&file, path, "verifier reward")?;
    read_open_file_bounded(file).map(Some)
}

fn read_file_bounded(path: &Path, kind: &str) -> Result<Vec<u8>, EvalExecutionError> {
    let file = open_regular_file(path, kind)?;
    read_open_file_bounded(file)
}

fn open_regular_file(path: &Path, kind: &str) -> Result<fs::File, EvalExecutionError> {
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
    ensure_regular_file(&file, path, kind)?;
    Ok(file)
}

fn ensure_regular_file(file: &fs::File, path: &Path, kind: &str) -> Result<(), EvalExecutionError> {
    let metadata = file
        .metadata()
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
    if !metadata.file_type().is_file() {
        return Err(EvalExecutionError::ArtifactCollection(format!(
            "{kind} is not a regular file: {}",
            path.display()
        )));
    }
    Ok(())
}

fn read_open_file_bounded(file: fs::File) -> Result<Vec<u8>, EvalExecutionError> {
    let metadata = file
        .metadata()
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
    if metadata.len() > MAX_LOCAL_FILE_BYTES {
        return Err(EvalExecutionError::ArtifactCollection(
            "local artifact exceeds the maximum size".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_LOCAL_FILE_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| EvalExecutionError::ArtifactCollection(error.to_string()))?;
    if bytes.len() as u64 > MAX_LOCAL_FILE_BYTES {
        return Err(EvalExecutionError::ArtifactCollection(
            "local artifact exceeds the maximum size".to_owned(),
        ));
    }
    Ok(bytes)
}
