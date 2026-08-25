// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Supervised stdio client for the canonical Python/Lighteval evaluator.
//!
//! This control-plane client never contacts an inference server. Rust sends LLM
//! requests and reads streaming responses through the normal AIPerf transport;
//! the child receives completed response text only for grading.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fmt::{self, Display};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::sync::{Mutex, oneshot};
use tokio::task::JoinHandle;

use crate::accuracy_core::protocol::{
    EVALUATOR_PROTOCOL_VERSION, EvaluatorGradeBatch, EvaluatorGradeItem, EvaluatorIdentity,
    EvaluatorLoadConfig, EvaluatorLoadResult, EvaluatorProblemPage, ShutdownResult, WorkerRequest,
    WorkerResponse,
};

const MAX_PROTOCOL_LINE_BYTES: usize = 64 * 1024 * 1024;
const REQUIRED_CAPABILITIES: &[&str] = &["load", "next_problems", "grade_batch", "shutdown"];
const PROCESS_GROUP_REAP_TIMEOUT: Duration = Duration::from_secs(2);

/// Exact parent-environment keys forwarded to the evaluator subprocess.
///
/// [`PythonEvaluator::spawn`] calls `env_clear()` before installing a curated
/// allowlist: passing the full parent environment leaks unrelated host secrets
/// (cloud credentials, database URLs, foreign API tokens) into a third-party
/// Python evaluator. Only the interpreter/runtime essentials below survive
/// `env_clear()`; callers add anything else explicitly through
/// [`WorkerProcessConfig::env`].
const FORWARDED_ENVIRONMENT_KEYS: &[&str] = &[
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "PYTHONPATH",
    "PYTHONHOME",
    "VIRTUAL_ENV",
];

/// Parent-environment key prefixes forwarded to the evaluator subprocess.
///
/// Namespaced runtime configuration the canonical evaluator stack genuinely
/// reads — Hugging Face hub/dataset roots and tokens, transformers caches,
/// freedesktop dirs, and AIPerf worker settings. These are the vars a curated
/// deployment would supply; unrelated host secrets outside these namespaces stay
/// cleared, matching the supervised path's intent.
const FORWARDED_ENVIRONMENT_PREFIXES: &[&str] =
    &["AIPERF_", "HF_", "HUGGINGFACE_", "TRANSFORMERS_", "XDG_"];
const GRADER_OVERRIDE_CAPABILITY: &str = "grader_override";

/// Sink for worker stderr lines.
pub trait EvaluatorLogSink: Send + Sync {
    /// Consume one complete stderr line.
    fn log_line(&self, line: &str);
}

/// Default log sink forwarding evaluator diagnostics to process stderr.
#[derive(Debug, Clone, Copy, Default)]
pub struct StderrEvaluatorLogSink;

impl EvaluatorLogSink for StderrEvaluatorLogSink {
    fn log_line(&self, line: &str) {
        eprintln!("[accuracy-evaluator] {line}");
    }
}

/// Process launch configuration for a local evaluator.
#[derive(Clone)]
pub struct WorkerProcessConfig {
    program: OsString,
    args: Vec<OsString>,
    environment: BTreeMap<OsString, OsString>,
    current_dir: Option<PathBuf>,
    log_sink: Arc<dyn EvaluatorLogSink>,
}

impl WorkerProcessConfig {
    /// Build a launch configuration without invoking a shell.
    pub fn new(program: impl Into<OsString>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            environment: BTreeMap::new(),
            current_dir: None,
            log_sink: Arc::new(StderrEvaluatorLogSink),
        }
    }

    /// Build the standard `python -u -m aiperf.accuracy.worker` command.
    pub fn python_module() -> Self {
        let program =
            std::env::var_os("AIPERF_ACCURACY_PYTHON").unwrap_or_else(|| OsString::from("python"));
        Self::new(program)
            .arg("-u")
            .arg("-m")
            .arg("aiperf.accuracy.worker")
    }

    /// Append one literal argv element.
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Set one child environment entry.
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.environment.insert(key.into(), value.into());
        self
    }

    /// Set the child working directory.
    pub fn current_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.current_dir = Some(path.into());
        self
    }

    /// Inject worker-log handling.
    pub fn log_sink(mut self, sink: Arc<dyn EvaluatorLogSink>) -> Self {
        self.log_sink = sink;
        self
    }

    /// Executable path/name.
    pub fn program(&self) -> &OsStr {
        &self.program
    }

    /// Literal argument vector.
    pub fn args(&self) -> &[OsString] {
        &self.args
    }
}

impl fmt::Debug for WorkerProcessConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerProcessConfig")
            .field("program", &self.program)
            .field("args", &self.args)
            .field(
                "environment_keys",
                &self.environment.keys().collect::<Vec<_>>(),
            )
            .field("current_dir", &self.current_dir)
            .finish_non_exhaustive()
    }
}

/// Evaluator control-plane seam used by the application layer.
#[async_trait(?Send)]
pub trait AccuracyEvaluator {
    /// Exact environment negotiated at startup.
    fn identity(&self) -> &EvaluatorIdentity;

    /// Load and freeze one benchmark.
    async fn load(
        &mut self,
        benchmark: &str,
        config: &EvaluatorLoadConfig,
    ) -> Result<EvaluatorLoadResult, EvaluatorWorkerError>;

    /// Load a benchmark with an optional evaluator-owned grader override.
    ///
    /// Implementations without this protocol capability retain a fail-closed
    /// default. The override never moves grading into Rust.
    async fn load_with_grader(
        &mut self,
        benchmark: &str,
        config: &EvaluatorLoadConfig,
        grader: Option<&str>,
    ) -> Result<EvaluatorLoadResult, EvaluatorWorkerError> {
        if let Some(grader) = grader {
            return Err(EvaluatorWorkerError::Protocol(format!(
                "evaluator does not support grader override {grader:?}"
            )));
        }
        self.load(benchmark, config).await
    }

    /// Retrieve one ordered page of opaque problems.
    async fn next_problems(
        &mut self,
        offset: usize,
        limit: usize,
    ) -> Result<EvaluatorProblemPage, EvaluatorWorkerError>;

    /// Grade terminal Rust inference responses in one canonical batch.
    async fn grade_batch(
        &mut self,
        items: &[EvaluatorGradeItem],
    ) -> Result<EvaluatorGradeBatch, EvaluatorWorkerError>;

    /// Ask the worker to shut down and wait for the child.
    async fn shutdown(&mut self) -> Result<(), EvaluatorWorkerError>;
}

/// Long-lived supervised Python evaluator process.
pub struct PythonEvaluator {
    session: Option<EvaluatorSession>,
    next_id: u64,
    identity: EvaluatorIdentity,
}

type PendingResponse = Result<WorkerResponse, EvaluatorWorkerError>;

struct EvaluatorSession {
    stdin: Arc<Mutex<BufWriter<ChildStdin>>>,
    child: Arc<Mutex<Child>>,
    pending: Arc<Mutex<BTreeMap<u64, oneshot::Sender<PendingResponse>>>>,
    stderr_task: Option<JoinHandle<()>>,
    reader_task: Option<JoinHandle<()>>,
    #[cfg(unix)]
    process_group_id: i32,
}

/// Whether a parent-environment key is on the evaluator forwarding allowlist.
fn is_forwarded_environment_key(key: &OsStr) -> bool {
    let Some(name) = key.to_str() else {
        return false;
    };
    FORWARDED_ENVIRONMENT_KEYS.contains(&name)
        || FORWARDED_ENVIRONMENT_PREFIXES
            .iter()
            .any(|prefix| name.starts_with(prefix))
}

/// Build the child environment as the curated parent allowlist plus explicit
/// caller entries.
///
/// The launching-process environment is filtered to interpreter/runtime
/// essentials and evaluator-owned namespaces; caller-provided entries are
/// layered last so explicit values win. Applied under `env_clear()`, this stops
/// the [`PythonEvaluator`] subprocess from inheriting the full parent
/// environment.
fn allowlisted_child_environment(
    explicit: &BTreeMap<OsString, OsString>,
) -> BTreeMap<OsString, OsString> {
    let mut environment: BTreeMap<OsString, OsString> = std::env::vars_os()
        .filter(|(key, _)| is_forwarded_environment_key(key))
        .collect();
    for (key, value) in explicit {
        environment.insert(key.clone(), value.clone());
    }
    environment
}

#[cfg(unix)]
fn configure_evaluator_process_group(command: &mut Command) {
    // SAFETY: `setsid` is async-signal-safe in the forked child before exec.
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
fn configure_evaluator_process_group(_: &mut Command) {}

#[cfg(unix)]
fn signal_evaluator_process_group(
    process_group_id: i32,
    signal: libc::c_int,
) -> Result<(), EvaluatorWorkerError> {
    let result = unsafe { libc::kill(-process_group_id, signal) };
    if result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(EvaluatorWorkerError::Process(format!(
            "cannot signal evaluator process group: {}",
            std::io::Error::last_os_error()
        )))
    }
}

#[cfg(unix)]
fn evaluator_process_group_exists(process_group_id: i32) -> Result<bool, EvaluatorWorkerError> {
    let result = unsafe { libc::kill(-process_group_id, 0) };
    if result == 0 {
        return Ok(true);
    }
    match std::io::Error::last_os_error().raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(EvaluatorWorkerError::Process(format!(
            "cannot inspect evaluator process group: {}",
            std::io::Error::last_os_error()
        ))),
    }
}

#[cfg(unix)]
fn remaining_evaluator_deadline(
    deadline: tokio::time::Instant,
    message: &str,
) -> Result<Duration, EvaluatorWorkerError> {
    deadline
        .checked_duration_since(tokio::time::Instant::now())
        .filter(|remaining| !remaining.is_zero())
        .ok_or_else(|| EvaluatorWorkerError::Process(message.to_owned()))
}

#[cfg(unix)]
async fn wait_for_evaluator_process_group_exit(
    process_group_id: i32,
    deadline: tokio::time::Instant,
) -> Result<(), EvaluatorWorkerError> {
    while evaluator_process_group_exists(process_group_id)? {
        let remaining = remaining_evaluator_deadline(
            deadline,
            "evaluator process-group reap deadline elapsed",
        )?;
        tokio::time::sleep(remaining.min(Duration::from_millis(10))).await;
    }
    Ok(())
}

async fn fault_pending_requests(
    pending: &Arc<Mutex<BTreeMap<u64, oneshot::Sender<PendingResponse>>>>,
    error: EvaluatorWorkerError,
) {
    let drained = {
        let mut pending = pending.lock().await;
        std::mem::take(&mut *pending)
            .into_values()
            .collect::<Vec<_>>()
    };
    for sender in drained {
        let _ = sender.send(Err(error.clone()));
    }
}

async fn run_response_reader(
    mut stdout: BufReader<ChildStdout>,
    pending: Arc<Mutex<BTreeMap<u64, oneshot::Sender<PendingResponse>>>>,
) {
    loop {
        let mut line = String::new();
        let bytes = match stdout.read_line(&mut line).await {
            Ok(bytes) => bytes,
            Err(error) => {
                fault_pending_requests(&pending, EvaluatorWorkerError::Io(error.to_string())).await;
                break;
            }
        };
        if bytes == 0 {
            fault_pending_requests(
                &pending,
                EvaluatorWorkerError::Crashed {
                    status: "worker exited before responding".to_string(),
                },
            )
            .await;
            break;
        }
        if bytes > MAX_PROTOCOL_LINE_BYTES {
            fault_pending_requests(
                &pending,
                EvaluatorWorkerError::Protocol(format!(
                    "evaluator response exceeded {MAX_PROTOCOL_LINE_BYTES} bytes"
                )),
            )
            .await;
            break;
        }
        let response: WorkerResponse = match serde_json::from_str(&line) {
            Ok(response) => response,
            Err(error) => {
                fault_pending_requests(&pending, EvaluatorWorkerError::Json(error.to_string()))
                    .await;
                break;
            }
        };
        let Some(id) = response.id else {
            fault_pending_requests(
                &pending,
                EvaluatorWorkerError::Protocol("evaluator response omitted request id".to_string()),
            )
            .await;
            break;
        };
        let sender = {
            let mut pending = pending.lock().await;
            pending.remove(&id)
        };
        if let Some(sender) = sender {
            let _ = sender.send(Ok(response));
        }
    }
}

impl PythonEvaluator {
    async fn dispatch_request<T: DeserializeOwned>(
        stdin: Arc<Mutex<BufWriter<ChildStdin>>>,
        pending: Arc<Mutex<BTreeMap<u64, oneshot::Sender<PendingResponse>>>>,
        request: WorkerRequest<'_>,
    ) -> Result<T, EvaluatorWorkerError> {
        let expected_id = request.id();
        let encoded = serde_json::to_vec(&request)
            .map_err(|error| EvaluatorWorkerError::Json(error.to_string()))?;
        let (sender, receiver) = oneshot::channel();
        {
            let mut pending = pending.lock().await;
            if pending.insert(expected_id, sender).is_some() {
                return Err(EvaluatorWorkerError::Protocol(format!(
                    "duplicate pending evaluator request id {expected_id}"
                )));
            }
        }

        let write_result = async {
            let mut stdin = stdin.lock().await;
            stdin.write_all(&encoded).await?;
            stdin.write_all(b"\n").await?;
            stdin.flush().await
        }
        .await;
        if let Err(error) = write_result {
            let mut pending = pending.lock().await;
            pending.remove(&expected_id);
            return Err(EvaluatorWorkerError::io(error));
        }

        let response = match receiver.await {
            Ok(response) => response?,
            Err(_) => {
                return Err(EvaluatorWorkerError::Crashed {
                    status: "worker response channel closed".to_string(),
                });
            }
        };
        if !response.ok {
            let error = response.error.ok_or_else(|| {
                EvaluatorWorkerError::Protocol(
                    "failed evaluator response omitted error details".to_string(),
                )
            })?;
            return Err(EvaluatorWorkerError::Remote {
                kind: error.kind,
                message: error.message,
                retryable: error.retryable,
            });
        }
        let result = response.result.ok_or_else(|| {
            EvaluatorWorkerError::Protocol(
                "successful evaluator response omitted result".to_string(),
            )
        })?;
        serde_json::from_value(result)
            .map_err(|error| EvaluatorWorkerError::Json(error.to_string()))
    }

    async fn reap_session(
        mut session: EvaluatorSession,
        require_success: bool,
    ) -> Result<(), EvaluatorWorkerError> {
        fault_pending_requests(
            &session.pending,
            EvaluatorWorkerError::Process("evaluator session terminated".to_string()),
        )
        .await;
        {
            let mut stdin = session.stdin.lock().await;
            let _ = stdin.shutdown().await;
        }

        let status = {
            let mut child = session.child.lock().await;
            #[cfg(unix)]
            {
                let deadline = tokio::time::Instant::now()
                    .checked_add(PROCESS_GROUP_REAP_TIMEOUT)
                    .ok_or_else(|| {
                        EvaluatorWorkerError::Process(
                            "evaluator process-group reap deadline is invalid".to_string(),
                        )
                    })?;
                let remaining =
                    remaining_evaluator_deadline(deadline, "evaluator reap deadline elapsed")?;
                let status = match tokio::time::timeout(remaining, child.wait()).await {
                    Ok(result) => result.map_err(EvaluatorWorkerError::io)?,
                    Err(_) => {
                        signal_evaluator_process_group(session.process_group_id, libc::SIGKILL)?;
                        let remaining = remaining_evaluator_deadline(
                            deadline,
                            "evaluator reap deadline elapsed",
                        )?;
                        tokio::time::timeout(remaining, child.wait())
                            .await
                            .map_err(|_| {
                                EvaluatorWorkerError::Process(
                                    "evaluator reap deadline elapsed".to_string(),
                                )
                            })?
                            .map_err(EvaluatorWorkerError::io)?
                    }
                };
                signal_evaluator_process_group(session.process_group_id, libc::SIGKILL)?;
                wait_for_evaluator_process_group_exit(session.process_group_id, deadline).await?;
                status
            }
            #[cfg(not(unix))]
            {
                child.kill().await.map_err(EvaluatorWorkerError::io)?;
                child.wait().await.map_err(EvaluatorWorkerError::io)?
            }
        };

        if let Some(task) = session.reader_task.take() {
            task.abort();
            let _ = task.await;
        }
        if let Some(task) = session.stderr_task.take() {
            task.abort();
            let _ = task.await;
        }

        if require_success && !status.success() {
            return Err(EvaluatorWorkerError::Crashed {
                status: status.to_string(),
            });
        }
        Ok(())
    }

    async fn force_reap_session(&mut self) -> Result<(), EvaluatorWorkerError> {
        if let Some(session) = self.session.take() {
            Self::reap_session(session, false).await?;
        }
        Ok(())
    }

    fn should_reap_after_request(error: &EvaluatorWorkerError) -> bool {
        !matches!(error, EvaluatorWorkerError::Remote { .. })
    }

    /// Spawn the worker, drain stderr, and negotiate protocol version 1.
    pub async fn spawn(config: WorkerProcessConfig) -> Result<Self, EvaluatorWorkerError> {
        let child_environment = allowlisted_child_environment(&config.environment);
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .env_clear()
            .envs(&child_environment)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        configure_evaluator_process_group(&mut command);
        if let Some(current_dir) = &config.current_dir {
            command.current_dir(current_dir);
        }
        let mut child = command
            .spawn()
            .map_err(|error| EvaluatorWorkerError::Spawn {
                program: config.program.to_string_lossy().into_owned(),
                message: error.to_string(),
            })?;
        let stdin = child
            .stdin
            .take()
            .ok_or(EvaluatorWorkerError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(EvaluatorWorkerError::MissingPipe("stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or(EvaluatorWorkerError::MissingPipe("stderr"))?;
        #[cfg(unix)]
        let process_group_id = i32::try_from(child.id().ok_or_else(|| {
            EvaluatorWorkerError::Process("evaluator child process id was unavailable".to_string())
        })?)
        .map_err(|_| {
            EvaluatorWorkerError::Process("evaluator child process id overflowed i32".to_string())
        })?;
        let sink = config.log_sink;
        let stderr_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => sink.log_line(&line),
                    Ok(None) => break,
                    Err(error) => {
                        sink.log_line(&format!("failed to read evaluator stderr: {error}"));
                        break;
                    }
                }
            }
        });
        let pending = Arc::new(Mutex::new(BTreeMap::new()));
        let reader_task = tokio::spawn(run_response_reader(
            BufReader::new(stdout),
            Arc::clone(&pending),
        ));
        let placeholder = EvaluatorIdentity {
            protocol: 0,
            worker_version: String::new(),
            python_version: String::new(),
            python_executable: String::new(),
            packages: BTreeMap::new(),
            worker_source_sha256: String::new(),
            dependency_lock_sha256: None,
            container_digest: None,
            capabilities: Vec::new(),
        };
        let mut worker = Self {
            session: Some(EvaluatorSession {
                stdin: Arc::new(Mutex::new(BufWriter::new(stdin))),
                child: Arc::new(Mutex::new(child)),
                pending,
                stderr_task: Some(stderr_task),
                reader_task: Some(reader_task),
                #[cfg(unix)]
                process_group_id,
            }),
            next_id: 1,
            identity: placeholder,
        };
        let id = worker.take_id()?;
        let identity: EvaluatorIdentity = match worker
            .request(WorkerRequest::Hello {
                id,
                protocol: EVALUATOR_PROTOCOL_VERSION,
            })
            .await
        {
            Ok(identity) => identity,
            Err(error) => {
                let _ = worker.force_reap_session().await;
                return Err(error);
            }
        };
        if identity.protocol != EVALUATOR_PROTOCOL_VERSION {
            let error = EvaluatorWorkerError::Protocol(format!(
                "worker negotiated protocol {}, expected {}",
                identity.protocol, EVALUATOR_PROTOCOL_VERSION
            ));
            let _ = worker.force_reap_session().await;
            return Err(error);
        }
        if let Err(error) = validate_identity(&identity) {
            let _ = worker.force_reap_session().await;
            return Err(error);
        }
        worker.identity = identity;
        Ok(worker)
    }

    fn take_id(&mut self) -> Result<u64, EvaluatorWorkerError> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| EvaluatorWorkerError::Protocol("request id overflow".to_string()))?;
        Ok(id)
    }

    async fn request<T: DeserializeOwned>(
        &mut self,
        request: WorkerRequest<'_>,
    ) -> Result<T, EvaluatorWorkerError> {
        let session = self.session.as_ref().ok_or_else(|| {
            EvaluatorWorkerError::Protocol("evaluator request attempted after shutdown".to_string())
        })?;
        let result = Self::dispatch_request(
            Arc::clone(&session.stdin),
            Arc::clone(&session.pending),
            request,
        )
        .await;
        if let Err(error) = &result
            && Self::should_reap_after_request(error)
        {
            let _ = self.force_reap_session().await;
        }
        result
    }

    /// Test-only external hook for issuing one grade request with an explicit id.
    ///
    /// The public evaluator trait is sequential (`&mut self`), so integration
    /// coverage for the internal request demultiplexer needs a narrow direct
    /// entry that can submit multiple in-flight grade requests against the same
    /// session. Production code should continue using [`AccuracyEvaluator`].
    #[doc(hidden)]
    pub async fn grade_batch_with_request_id_for_testing(
        &self,
        id: u64,
        items: &[EvaluatorGradeItem],
    ) -> Result<EvaluatorGradeBatch, EvaluatorWorkerError> {
        let session = self.session.as_ref().ok_or_else(|| {
            EvaluatorWorkerError::Protocol("evaluator request attempted after shutdown".to_string())
        })?;
        Self::dispatch_request(
            Arc::clone(&session.stdin),
            Arc::clone(&session.pending),
            WorkerRequest::GradeBatch { id, items },
        )
        .await
    }
}

fn validate_identity(identity: &EvaluatorIdentity) -> Result<(), EvaluatorWorkerError> {
    for (field, value) in [
        ("worker_version", identity.worker_version.as_str()),
        ("python_version", identity.python_version.as_str()),
        ("python_executable", identity.python_executable.as_str()),
        (
            "worker_source_sha256",
            identity.worker_source_sha256.as_str(),
        ),
    ] {
        if value.trim().is_empty() {
            return Err(EvaluatorWorkerError::Protocol(format!(
                "evaluator identity field {field} was empty"
            )));
        }
    }
    if identity.packages.is_empty() {
        return Err(EvaluatorWorkerError::Protocol(
            "evaluator identity reported no package versions".to_string(),
        ));
    }
    if !is_sha256(&identity.worker_source_sha256) {
        return Err(EvaluatorWorkerError::Protocol(
            "evaluator worker_source_sha256 was not a 64-digit lowercase hex digest".to_string(),
        ));
    }
    if let Some(lock) = &identity.dependency_lock_sha256
        && !is_sha256(lock)
    {
        return Err(EvaluatorWorkerError::Protocol(
            "evaluator dependency_lock_sha256 was not a 64-digit lowercase hex digest".to_string(),
        ));
    }
    if let Some(container) = &identity.container_digest
        && !container.strip_prefix("sha256:").is_some_and(is_sha256)
    {
        return Err(EvaluatorWorkerError::Protocol(
            "evaluator container_digest was not a sha256 OCI digest".to_string(),
        ));
    }
    if identity.dependency_lock_sha256.is_none() && identity.container_digest.is_none() {
        return Err(EvaluatorWorkerError::Protocol(
            "evaluator identity must report a dependency lock or container digest".to_string(),
        ));
    }
    let capabilities = identity
        .capabilities
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if capabilities.len() != identity.capabilities.len() {
        return Err(EvaluatorWorkerError::Protocol(
            "evaluator identity contained duplicate capabilities".to_string(),
        ));
    }
    let missing = REQUIRED_CAPABILITIES
        .iter()
        .copied()
        .filter(|required| !capabilities.contains(required))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(EvaluatorWorkerError::Protocol(format!(
            "evaluator identity omitted required capabilities: {}",
            missing.join(", ")
        )));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[async_trait(?Send)]
impl AccuracyEvaluator for PythonEvaluator {
    fn identity(&self) -> &EvaluatorIdentity {
        &self.identity
    }

    async fn load(
        &mut self,
        benchmark: &str,
        config: &EvaluatorLoadConfig,
    ) -> Result<EvaluatorLoadResult, EvaluatorWorkerError> {
        self.load_with_grader(benchmark, config, None).await
    }

    async fn load_with_grader(
        &mut self,
        benchmark: &str,
        config: &EvaluatorLoadConfig,
        grader: Option<&str>,
    ) -> Result<EvaluatorLoadResult, EvaluatorWorkerError> {
        if grader.is_some()
            && !self
                .identity
                .capabilities
                .iter()
                .any(|capability| capability == GRADER_OVERRIDE_CAPABILITY)
        {
            return Err(EvaluatorWorkerError::Protocol(
                "evaluator worker does not report grader_override".to_string(),
            ));
        }
        let id = self.take_id()?;
        self.request(WorkerRequest::Load {
            id,
            benchmark,
            config,
            grader,
        })
        .await
    }

    async fn next_problems(
        &mut self,
        offset: usize,
        limit: usize,
    ) -> Result<EvaluatorProblemPage, EvaluatorWorkerError> {
        if limit == 0 {
            return Err(EvaluatorWorkerError::Protocol(
                "next_problems limit must be positive".to_string(),
            ));
        }
        let id = self.take_id()?;
        self.request(WorkerRequest::NextProblems { id, offset, limit })
            .await
    }

    async fn grade_batch(
        &mut self,
        items: &[EvaluatorGradeItem],
    ) -> Result<EvaluatorGradeBatch, EvaluatorWorkerError> {
        if items.is_empty() {
            return Err(EvaluatorWorkerError::Protocol(
                "grade_batch requires at least one item".to_string(),
            ));
        }
        let id = self.take_id()?;
        self.request(WorkerRequest::GradeBatch { id, items }).await
    }

    async fn shutdown(&mut self) -> Result<(), EvaluatorWorkerError> {
        if self.session.is_none() {
            return Ok(());
        }
        let id = self.take_id()?;
        let result: ShutdownResult = match self.request(WorkerRequest::Shutdown { id }).await {
            Ok(result) => result,
            Err(error) => {
                let _ = self.force_reap_session().await;
                return Err(error);
            }
        };
        if !result.shutdown {
            return Err(EvaluatorWorkerError::Protocol(
                "worker did not acknowledge shutdown".to_string(),
            ));
        }
        if let Some(session) = self.session.take() {
            Self::reap_session(session, true).await?;
        }
        Ok(())
    }
}

/// Supervision, protocol, or remote evaluator failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EvaluatorWorkerError {
    /// Child process could not start.
    Spawn {
        /// Executable that failed.
        program: String,
        /// OS diagnostic.
        message: String,
    },
    /// Required child stdio pipe was unavailable.
    MissingPipe(&'static str),
    /// Stdio or process-wait failure.
    Io(String),
    /// JSON serialization/deserialization failure.
    Json(String),
    /// JSONL protocol invariant failure.
    Protocol(String),
    /// Session setup or teardown failure outside the JSONL protocol.
    Process(String),
    /// Structured worker operation error. This is infrastructure failure, not an incorrect answer.
    Remote {
        /// Python exception type.
        kind: String,
        /// Python diagnostic.
        message: String,
        /// Whether the worker declared the operation retryable.
        retryable: bool,
    },
    /// Worker exited before or during an operation.
    Crashed {
        /// Child exit status.
        status: String,
    },
}

impl EvaluatorWorkerError {
    fn io(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

impl Display for EvaluatorWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Spawn { program, message } => {
                write!(
                    formatter,
                    "failed to spawn evaluator {program:?}: {message}"
                )
            }
            Self::MissingPipe(pipe) => write!(formatter, "evaluator child had no {pipe} pipe"),
            Self::Io(message) => write!(formatter, "evaluator I/O failed: {message}"),
            Self::Json(message) => write!(formatter, "evaluator JSON failed: {message}"),
            Self::Protocol(message) => write!(formatter, "evaluator protocol failed: {message}"),
            Self::Process(message) => write!(formatter, "evaluator process failed: {message}"),
            Self::Remote {
                kind,
                message,
                retryable,
            } => write!(
                formatter,
                "evaluator operation failed ({kind}, retryable={retryable}): {message}"
            ),
            Self::Crashed { status } => {
                write!(formatter, "evaluator worker exited unexpectedly: {status}")
            }
        }
    }
}

impl std::error::Error for EvaluatorWorkerError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accuracy_core::protocol::{EvaluatorGradeItem, ProblemId};
    use tempfile::tempdir;

    const FAKE_WORKER: &str = r#"
import json, sys
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'hello':
        result = {'protocol': 1, 'worker_version': 'fixture', 'python_version': '3', 'python_executable': sys.executable, 'packages': {'lighteval': 'fixture'}, 'worker_source_sha256': 'a' * 64, 'dependency_lock_sha256': 'b' * 64, 'container_digest': None, 'capabilities': ['load', 'next_problems', 'grade_batch', 'shutdown']}
    elif op == 'load':
        result = {'benchmark': request['benchmark'], 'problem_count': 1, 'dataset': {'provider': 'fixture', 'revision': 'rev', 'evaluation_splits': ['test']}, 'grader': 'fixture'}
    elif op == 'next_problems':
        result = {'items': [{'problem_id': 'opaque-1', 'task': 'fixture', 'prompt': 'Question?', 'messages': [{'role': 'user', 'content': 'Question?'}], 'generation': {'max_tokens': 8, 'temperature': 0.0, 'top_p': 1.0, 'stop': []}}], 'next_offset': 1, 'done': True}
    elif op == 'grade_batch':
        result = {'items': [{'problem_id': item['problem_id'], 'task': 'fixture', 'correct': item['response'] == 'A', 'unparsed': False, 'confidence': 1.0 if item['response'] == 'A' else 0.0, 'reasoning': 'fixture', 'extracted_answer': item['response']} for item in request['items']]}
    elif op == 'shutdown':
        result = {'shutdown': True}
    else:
        raise RuntimeError(op)
    print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
    if op == 'shutdown':
        break
"#;

    fn fixture_config(script: &str) -> WorkerProcessConfig {
        WorkerProcessConfig::new(
            std::env::var_os("PYTHON").unwrap_or_else(|| OsString::from("python")),
        )
        .arg("-u")
        .arg("-c")
        .arg(script)
    }

    fn fixture_config_with_arg(script: &str, arg: &OsStr) -> WorkerProcessConfig {
        fixture_config(script).arg(arg.to_os_string())
    }

    #[tokio::test]
    async fn supervises_versioned_problem_and_grade_protocol() {
        let mut evaluator = PythonEvaluator::spawn(fixture_config(FAKE_WORKER))
            .await
            .unwrap();
        assert_eq!(evaluator.identity().worker_version, "fixture");
        let loaded = evaluator
            .load("fixture", &EvaluatorLoadConfig::default())
            .await
            .unwrap();
        assert_eq!(loaded.problem_count, 1);
        let page = evaluator.next_problems(0, 100).await.unwrap();
        assert!(page.done);
        assert_eq!(page.items[0].problem_id.as_str(), "opaque-1");
        let grades = evaluator
            .grade_batch(&[EvaluatorGradeItem {
                problem_id: ProblemId::new("opaque-1").unwrap(),
                response: "A".to_string(),
            }])
            .await
            .unwrap();
        assert!(grades.items[0].correct);
        evaluator.shutdown().await.unwrap();
    }

    #[tokio::test]
    async fn worker_exit_is_infrastructure_error() {
        let script = FAKE_WORKER.replace(
            "elif op == 'grade_batch':",
            "elif op == 'grade_batch':\n        raise SystemExit(7)\n    elif False:",
        );
        let mut evaluator = PythonEvaluator::spawn(fixture_config(&script))
            .await
            .unwrap();
        let error = evaluator
            .grade_batch(&[EvaluatorGradeItem {
                problem_id: ProblemId::new("opaque-1").unwrap(),
                response: "A".to_string(),
            }])
            .await
            .unwrap_err();
        assert!(matches!(error, EvaluatorWorkerError::Crashed { .. }));
    }

    #[tokio::test]
    async fn grade_batch_demuxes_out_of_order_responses() {
        let script = r#"
import json, sys
pending = []
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'hello':
        result = {'protocol': 1, 'worker_version': 'fixture', 'python_version': '3', 'python_executable': sys.executable, 'packages': {'lighteval': 'fixture'}, 'worker_source_sha256': 'a' * 64, 'dependency_lock_sha256': 'b' * 64, 'container_digest': None, 'capabilities': ['load', 'next_problems', 'grade_batch', 'shutdown']}
        print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
    elif op == 'grade_batch':
        pending.append(request)
        if len(pending) == 2:
            for queued in reversed(pending):
                item = queued['items'][0]
                result = {'items': [{'problem_id': item['problem_id'], 'task': 'fixture', 'correct': True, 'unparsed': False, 'confidence': float(queued['id']), 'reasoning': 'fixture', 'extracted_answer': item['response']}]}
                print(json.dumps({'id': queued['id'], 'ok': True, 'result': result}), flush=True)
            pending = []
    elif op == 'shutdown':
        print(json.dumps({'id': request['id'], 'ok': True, 'result': {'shutdown': True}}), flush=True)
        break
"#;
        let mut evaluator = PythonEvaluator::spawn(fixture_config(script))
            .await
            .unwrap();
        let first_id = evaluator.take_id().unwrap();
        let second_id = evaluator.take_id().unwrap();
        let session = evaluator.session.as_ref().unwrap();
        let stdin = Arc::clone(&session.stdin);
        let pending = Arc::clone(&session.pending);
        let first_items = vec![EvaluatorGradeItem {
            problem_id: ProblemId::new("opaque-1").unwrap(),
            response: "first".to_string(),
        }];
        let second_items = vec![EvaluatorGradeItem {
            problem_id: ProblemId::new("opaque-2").unwrap(),
            response: "second".to_string(),
        }];

        let (first, second) = tokio::join!(
            PythonEvaluator::dispatch_request::<EvaluatorGradeBatch>(
                Arc::clone(&stdin),
                Arc::clone(&pending),
                WorkerRequest::GradeBatch {
                    id: first_id,
                    items: &first_items,
                },
            ),
            PythonEvaluator::dispatch_request::<EvaluatorGradeBatch>(
                stdin,
                pending,
                WorkerRequest::GradeBatch {
                    id: second_id,
                    items: &second_items,
                },
            ),
        );
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(first.items[0].problem_id.as_str(), "opaque-1");
        assert_eq!(second.items[0].problem_id.as_str(), "opaque-2");
        assert_eq!(first.items[0].confidence, first_id as f64);
        assert_eq!(second.items[0].confidence, second_id as f64);
        evaluator.shutdown().await.unwrap();
    }

    #[cfg(unix)]
    fn process_exists(pid: i32) -> Result<bool, std::io::Error> {
        let result = unsafe { libc::kill(pid, 0) };
        if result == 0 {
            return Ok(true);
        }
        match std::io::Error::last_os_error().raw_os_error() {
            Some(libc::ESRCH) => Ok(false),
            Some(libc::EPERM) => Ok(true),
            _ => Err(std::io::Error::last_os_error()),
        }
    }

    #[cfg(unix)]
    async fn wait_for_text_file(path: &std::path::Path) -> String {
        for _ in 0..100 {
            if let Ok(text) = std::fs::read_to_string(path) {
                return text;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for {}", path.display());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shutdown_reaps_worker_process_group_descendants() {
        let tempdir = tempdir().unwrap();
        let pid_file = tempdir.path().join("descendant.pid");
        let script = r#"
import json, pathlib, subprocess, sys
pid_file = pathlib.Path(sys.argv[1])
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'hello':
        sleeper = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'])
        pid_file.write_text(str(sleeper.pid))
        result = {'protocol': 1, 'worker_version': 'fixture', 'python_version': '3', 'python_executable': sys.executable, 'packages': {'lighteval': 'fixture'}, 'worker_source_sha256': 'a' * 64, 'dependency_lock_sha256': 'b' * 64, 'container_digest': None, 'capabilities': ['load', 'next_problems', 'grade_batch', 'shutdown']}
        print(json.dumps({'id': request['id'], 'ok': True, 'result': result}), flush=True)
    elif op == 'shutdown':
        print(json.dumps({'id': request['id'], 'ok': True, 'result': {'shutdown': True}}), flush=True)
        break
"#;
        let mut evaluator =
            PythonEvaluator::spawn(fixture_config_with_arg(script, pid_file.as_os_str()))
                .await
                .unwrap();
        let descendant_pid = wait_for_text_file(&pid_file)
            .await
            .trim()
            .parse::<i32>()
            .unwrap();
        assert!(process_exists(descendant_pid).unwrap());
        evaluator.shutdown().await.unwrap();
        assert!(!process_exists(descendant_pid).unwrap());
    }

    #[tokio::test]
    async fn handshake_rejects_missing_required_capability() {
        let script = FAKE_WORKER.replace(
            "'load', 'next_problems', 'grade_batch', 'shutdown'",
            "'load', 'next_problems', 'shutdown'",
        );
        let error = match PythonEvaluator::spawn(fixture_config(&script)).await {
            Ok(_) => panic!("worker without grade_batch capability was accepted"),
            Err(error) => error,
        };
        assert!(matches!(error, EvaluatorWorkerError::Protocol(_)));
        assert!(error.to_string().contains("grade_batch"));
    }

    #[tokio::test]
    async fn handshake_requires_locked_dependencies_or_container() {
        let script = FAKE_WORKER.replace("'dependency_lock_sha256': 'b' * 64, ", "");
        let error = match PythonEvaluator::spawn(fixture_config(&script)).await {
            Ok(_) => panic!("worker without dependency identity was accepted"),
            Err(error) => error,
        };
        assert!(matches!(error, EvaluatorWorkerError::Protocol(_)));
        assert!(error.to_string().contains("dependency lock or container"));
    }

    #[tokio::test]
    async fn grade_rejects_ground_truth_outside_protocol() {
        let script = FAKE_WORKER.replace(
            "'extracted_answer': item['response']",
            "'extracted_answer': item['response'], 'ground_truth': 'private'",
        );
        let mut evaluator = PythonEvaluator::spawn(fixture_config(&script))
            .await
            .unwrap();
        let error = evaluator
            .grade_batch(&[EvaluatorGradeItem {
                problem_id: ProblemId::new("opaque-1").unwrap(),
                response: "A".to_string(),
            }])
            .await
            .unwrap_err();
        assert!(matches!(error, EvaluatorWorkerError::Json(_)));
        assert!(error.to_string().contains("unknown field `ground_truth`"));
    }

    #[tokio::test]
    async fn problem_rejects_private_fields_inside_prompt_messages() {
        let script = FAKE_WORKER.replace(
            "{'role': 'user', 'content': 'Question?'}",
            "{'role': 'user', 'content': 'Question?', 'private_tests': ['secret']}",
        );
        let mut evaluator = PythonEvaluator::spawn(fixture_config(&script))
            .await
            .unwrap();
        evaluator
            .load("fixture", &EvaluatorLoadConfig::default())
            .await
            .unwrap();
        let error = evaluator.next_problems(0, 1).await.unwrap_err();
        assert!(matches!(error, EvaluatorWorkerError::Json(_)));
        assert!(error.to_string().contains("unknown field `private_tests`"));
    }

    #[tokio::test]
    async fn problem_rejects_empty_wire_id() {
        let script = FAKE_WORKER.replace("'problem_id': 'opaque-1'", "'problem_id': ' '");
        let mut evaluator = PythonEvaluator::spawn(fixture_config(&script))
            .await
            .unwrap();
        evaluator
            .load("fixture", &EvaluatorLoadConfig::default())
            .await
            .unwrap();
        let error = evaluator.next_problems(0, 1).await.unwrap_err();
        assert!(matches!(error, EvaluatorWorkerError::Json(_)));
        assert!(error.to_string().contains("problem_id must not be empty"));
    }
}
