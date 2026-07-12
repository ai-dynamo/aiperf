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

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tokio::task::JoinHandle;

use crate::protocol::{
    AcceptedModelResults, AgenticEpisodeOutcome, AgenticEpisodePage, AgenticEvaluatorEvent,
    AgenticEvaluatorIdentity, AgenticEvaluatorLoadConfig, AgenticEventBatch,
    AgenticInferenceStatus, AgenticModelResult, AgenticResultBatch, CancelledEpisodesResult,
    EVALUATOR_PROTOCOL_VERSION, EpisodeId, EvaluatorGradeBatch, EvaluatorGradeItem,
    EvaluatorIdentity, EvaluatorLoadConfig, EvaluatorLoadResult, EvaluatorProblemPage,
    ShutdownResult, StartedEpisodesResult, WorkerRequest, WorkerResponse,
};

const MAX_PROTOCOL_LINE_BYTES: usize = 64 * 1024 * 1024;
const REQUIRED_CAPABILITIES: &[&str] = &["load", "next_problems", "grade_batch", "shutdown"];
const AGENTIC_CAPABILITY: &str = "agentic_harbor";
const AGENTIC_INFERENCE_GATEWAY_CAPABILITY: &str = "agentic_inference_gateway";
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

/// Stateful agent-harness seam whose model calls remain owned by Rust.
///
/// Implementations prepare task environments and canonical verifiers, then
/// publish model-call events. The application must dispatch those calls through
/// the ordinary AIPerf transport and submit terminal results back here.
#[async_trait(?Send)]
pub trait AgenticEvaluator: AccuracyEvaluator {
    /// Whether the worker reported the pinned Harbor capability.
    fn supports_agentic(&self) -> bool;

    /// Whether evaluator environments accept Rust callback-ingress injection.
    fn supports_agentic_inference_gateway(&self) -> bool {
        false
    }

    /// Resolve and freeze one versioned agentic dataset.
    async fn load_agentic(
        &mut self,
        dataset: &str,
        model: &str,
        config: &AgenticEvaluatorLoadConfig,
    ) -> Result<AgenticEvaluatorIdentity, EvaluatorWorkerError>;

    /// Retrieve one ordered page of opaque task episodes.
    async fn next_episodes(
        &mut self,
        offset: usize,
        limit: usize,
    ) -> Result<AgenticEpisodePage, EvaluatorWorkerError>;

    /// Start evaluator-owned environment setup for selected episodes.
    async fn start_episodes(
        &mut self,
        episode_ids: &[EpisodeId],
    ) -> Result<(), EvaluatorWorkerError>;

    /// Bounded long-poll for ready model calls and terminal episode results.
    async fn poll_agentic(
        &mut self,
        limit: usize,
        wait_ms: u64,
    ) -> Result<AgenticEventBatch, EvaluatorWorkerError>;

    /// Resume evaluator-owned agents with terminal responses produced by Rust.
    async fn submit_model_results(
        &mut self,
        items: &[AgenticModelResult],
    ) -> Result<(), EvaluatorWorkerError>;

    /// Cancel active evaluator environments selected by Rust policy.
    async fn cancel_episodes(
        &mut self,
        episode_ids: &[EpisodeId],
    ) -> Result<(), EvaluatorWorkerError>;

    /// Validate terminal state and return canonical results in dataset order.
    async fn finish_agentic(&mut self) -> Result<AgenticResultBatch, EvaluatorWorkerError>;
}

/// Long-lived supervised Python evaluator process.
pub struct PythonEvaluator {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    stderr_task: Option<JoinHandle<()>>,
    next_id: u64,
    identity: EvaluatorIdentity,
    shutdown: bool,
}

impl PythonEvaluator {
    /// Spawn the worker, drain stderr, and negotiate protocol version 1.
    pub async fn spawn(config: WorkerProcessConfig) -> Result<Self, EvaluatorWorkerError> {
        let mut command = Command::new(&config.program);
        command
            .args(&config.args)
            .envs(&config.environment)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
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
            child,
            stdin: BufWriter::new(stdin),
            stdout: BufReader::new(stdout),
            stderr_task: Some(stderr_task),
            next_id: 1,
            identity: placeholder,
            shutdown: false,
        };
        let id = worker.take_id()?;
        let identity: EvaluatorIdentity = worker
            .request(WorkerRequest::Hello {
                id,
                protocol: EVALUATOR_PROTOCOL_VERSION,
            })
            .await?;
        if identity.protocol != EVALUATOR_PROTOCOL_VERSION {
            return Err(EvaluatorWorkerError::Protocol(format!(
                "worker negotiated protocol {}, expected {}",
                identity.protocol, EVALUATOR_PROTOCOL_VERSION
            )));
        }
        validate_identity(&identity)?;
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
        if self.shutdown {
            return Err(EvaluatorWorkerError::Protocol(
                "evaluator request attempted after shutdown".to_string(),
            ));
        }
        let expected_id = request.id();
        let encoded = serde_json::to_vec(&request)
            .map_err(|error| EvaluatorWorkerError::Json(error.to_string()))?;
        self.stdin
            .write_all(&encoded)
            .await
            .map_err(EvaluatorWorkerError::io)?;
        self.stdin
            .write_all(b"\n")
            .await
            .map_err(EvaluatorWorkerError::io)?;
        self.stdin.flush().await.map_err(EvaluatorWorkerError::io)?;

        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .await
            .map_err(EvaluatorWorkerError::io)?;
        if bytes == 0 {
            return Err(self.crashed_error().await);
        }
        if bytes > MAX_PROTOCOL_LINE_BYTES {
            return Err(EvaluatorWorkerError::Protocol(format!(
                "evaluator response exceeded {MAX_PROTOCOL_LINE_BYTES} bytes"
            )));
        }
        let response: WorkerResponse = serde_json::from_str(&line)
            .map_err(|error| EvaluatorWorkerError::Json(error.to_string()))?;
        if response.id != Some(expected_id) {
            return Err(EvaluatorWorkerError::Protocol(format!(
                "evaluator response id {:?} did not match request {expected_id}",
                response.id
            )));
        }
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

    async fn crashed_error(&mut self) -> EvaluatorWorkerError {
        match self.child.wait().await {
            Ok(status) => EvaluatorWorkerError::Crashed {
                status: status.to_string(),
            },
            Err(error) => EvaluatorWorkerError::Io(error.to_string()),
        }
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

fn validate_agentic_identity(
    identity: &AgenticEvaluatorIdentity,
) -> Result<(), EvaluatorWorkerError> {
    for (field, value) in [
        ("harness", identity.harness.as_str()),
        ("harness_version", identity.harness_version.as_str()),
        ("agent", identity.agent.as_str()),
        ("agent_version", identity.agent_version.as_str()),
        ("environment", identity.environment.as_str()),
        ("verifier", identity.verifier.as_str()),
        ("dataset.provider", identity.dataset.provider.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(EvaluatorWorkerError::Protocol(format!(
                "agentic evaluator identity field {field} was empty"
            )));
        }
    }
    if !is_sha256(&identity.harness_source_sha256) {
        return Err(EvaluatorWorkerError::Protocol(
            "agentic harness_source_sha256 was not a 64-digit lowercase hex digest".to_string(),
        ));
    }
    if identity.episode_count == 0 {
        return Err(EvaluatorWorkerError::Protocol(
            "agentic evaluator selected zero episodes".to_string(),
        ));
    }
    if identity
        .dataset
        .revision
        .as_deref()
        .is_none_or(|revision| revision.trim().is_empty())
    {
        return Err(EvaluatorWorkerError::Protocol(
            "agentic evaluator omitted immutable dataset revision".to_string(),
        ));
    }
    if identity.dataset.evaluation_splits.is_empty()
        || identity
            .dataset
            .evaluation_splits
            .iter()
            .any(|split| split.trim().is_empty())
    {
        return Err(EvaluatorWorkerError::Protocol(
            "agentic evaluator reported no valid evaluation split".to_string(),
        ));
    }
    if identity
        .primary_reward
        .as_ref()
        .is_some_and(|reward| reward.trim().is_empty())
    {
        return Err(EvaluatorWorkerError::Protocol(
            "agentic evaluator reported an empty primary_reward".to_string(),
        ));
    }
    Ok(())
}

fn validate_agentic_result(
    result: &crate::protocol::AgenticEpisodeResult,
) -> Result<(), EvaluatorWorkerError> {
    if result.task.trim().is_empty() {
        return Err(EvaluatorWorkerError::Protocol(format!(
            "agentic episode {:?} returned an empty task",
            result.episode_id.as_str()
        )));
    }
    if !result.duration_seconds.is_finite() || result.duration_seconds < 0.0 {
        return Err(EvaluatorWorkerError::Protocol(format!(
            "agentic episode {:?} returned invalid duration {}",
            result.episode_id.as_str(),
            result.duration_seconds
        )));
    }
    if result.rewards.values().any(|reward| !reward.is_finite()) {
        return Err(EvaluatorWorkerError::Protocol(format!(
            "agentic episode {:?} returned a non-finite reward",
            result.episode_id.as_str()
        )));
    }
    match result.outcome {
        AgenticEpisodeOutcome::Completed => {
            if result.rewards.is_empty() {
                return Err(EvaluatorWorkerError::Protocol(format!(
                    "completed agentic episode {:?} returned no rewards",
                    result.episode_id.as_str()
                )));
            }
            if result.error_kind.is_some() || result.error_message.is_some() {
                return Err(EvaluatorWorkerError::Protocol(format!(
                    "completed agentic episode {:?} also returned an infrastructure error",
                    result.episode_id.as_str()
                )));
            }
        }
        AgenticEpisodeOutcome::InfrastructureError | AgenticEpisodeOutcome::Cancelled => {
            if !result.rewards.is_empty() || result.error_kind.is_none() {
                return Err(EvaluatorWorkerError::Protocol(format!(
                    "non-completed agentic episode {:?} must have no rewards and an error_kind",
                    result.episode_id.as_str()
                )));
            }
        }
    }
    if let Some(primary) = &result.primary_reward
        && !result.rewards.contains_key(primary)
    {
        return Err(EvaluatorWorkerError::Protocol(format!(
            "agentic episode {:?} selected absent primary reward {primary:?}",
            result.episode_id.as_str()
        )));
    }
    Ok(())
}

fn validate_agentic_events(batch: &AgenticEventBatch) -> Result<(), EvaluatorWorkerError> {
    for event in &batch.events {
        match event {
            AgenticEvaluatorEvent::ModelCall { call } => {
                if call.prompt.trim().is_empty() || call.messages.is_empty() {
                    return Err(EvaluatorWorkerError::Protocol(format!(
                        "agentic call {:?} omitted its prompt or messages",
                        call.call_id.as_str()
                    )));
                }
                if call.generation.max_tokens == 0
                    || !call.generation.temperature.is_finite()
                    || call.generation.temperature < 0.0
                    || !call.generation.top_p.is_finite()
                    || !(0.0..=1.0).contains(&call.generation.top_p)
                {
                    return Err(EvaluatorWorkerError::Protocol(format!(
                        "agentic call {:?} returned invalid generation controls",
                        call.call_id.as_str()
                    )));
                }
            }
            AgenticEvaluatorEvent::EpisodeCompleted { result } => {
                validate_agentic_result(result)?;
            }
        }
    }
    Ok(())
}

fn validate_submitted_model_result(item: &AgenticModelResult) -> Result<(), EvaluatorWorkerError> {
    match item.status {
        AgenticInferenceStatus::Completed => {
            if item.error_kind.is_some() || item.error_message.is_some() {
                return Err(EvaluatorWorkerError::Protocol(format!(
                    "completed model call {:?} contained an infrastructure error",
                    item.call_id.as_str()
                )));
            }
        }
        AgenticInferenceStatus::Failed | AgenticInferenceStatus::Cancelled => {
            if item
                .error_kind
                .as_ref()
                .is_none_or(|kind| kind.trim().is_empty())
            {
                return Err(EvaluatorWorkerError::Protocol(format!(
                    "non-completed model call {:?} omitted error_kind",
                    item.call_id.as_str()
                )));
            }
        }
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
        if self.shutdown {
            return Ok(());
        }
        let id = self.take_id()?;
        let result: ShutdownResult = self.request(WorkerRequest::Shutdown { id }).await?;
        if !result.shutdown {
            return Err(EvaluatorWorkerError::Protocol(
                "worker did not acknowledge shutdown".to_string(),
            ));
        }
        self.shutdown = true;
        self.stdin
            .shutdown()
            .await
            .map_err(EvaluatorWorkerError::io)?;
        let status = self.child.wait().await.map_err(EvaluatorWorkerError::io)?;
        if !status.success() {
            return Err(EvaluatorWorkerError::Crashed {
                status: status.to_string(),
            });
        }
        if let Some(task) = self.stderr_task.take() {
            let _ = task.await;
        }
        Ok(())
    }
}

#[async_trait(?Send)]
impl AgenticEvaluator for PythonEvaluator {
    fn supports_agentic(&self) -> bool {
        self.identity
            .capabilities
            .iter()
            .any(|capability| capability == AGENTIC_CAPABILITY)
    }

    fn supports_agentic_inference_gateway(&self) -> bool {
        self.identity
            .capabilities
            .iter()
            .any(|capability| capability == AGENTIC_INFERENCE_GATEWAY_CAPABILITY)
    }

    async fn load_agentic(
        &mut self,
        dataset: &str,
        model: &str,
        config: &AgenticEvaluatorLoadConfig,
    ) -> Result<AgenticEvaluatorIdentity, EvaluatorWorkerError> {
        if !self.supports_agentic() {
            return Err(EvaluatorWorkerError::Protocol(
                "evaluator worker does not report agentic_harbor; launch it from the pinned agentic worker environment"
                    .to_string(),
            ));
        }
        if dataset.trim().is_empty() || model.trim().is_empty() {
            return Err(EvaluatorWorkerError::Protocol(
                "agentic dataset and model must not be empty".to_string(),
            ));
        }
        let id = self.take_id()?;
        let identity = self
            .request(WorkerRequest::LoadAgentic {
                id,
                dataset,
                model,
                config,
            })
            .await?;
        validate_agentic_identity(&identity)?;
        Ok(identity)
    }

    async fn next_episodes(
        &mut self,
        offset: usize,
        limit: usize,
    ) -> Result<AgenticEpisodePage, EvaluatorWorkerError> {
        if limit == 0 {
            return Err(EvaluatorWorkerError::Protocol(
                "next_episodes limit must be positive".to_string(),
            ));
        }
        let id = self.take_id()?;
        let page: AgenticEpisodePage = self
            .request(WorkerRequest::NextEpisodes { id, offset, limit })
            .await?;
        if page
            .items
            .iter()
            .any(|episode| episode.task.trim().is_empty() || episode.source.trim().is_empty())
        {
            return Err(EvaluatorWorkerError::Protocol(
                "agentic episode page contained an empty task or source".to_string(),
            ));
        }
        Ok(page)
    }

    async fn start_episodes(
        &mut self,
        episode_ids: &[EpisodeId],
    ) -> Result<(), EvaluatorWorkerError> {
        if episode_ids.is_empty() {
            return Err(EvaluatorWorkerError::Protocol(
                "start_episodes requires at least one episode".to_string(),
            ));
        }
        let id = self.take_id()?;
        let result: StartedEpisodesResult = self
            .request(WorkerRequest::StartEpisodes { id, episode_ids })
            .await?;
        if result.started != episode_ids {
            return Err(EvaluatorWorkerError::Protocol(
                "worker start_episodes acknowledgement did not match submitted IDs".to_string(),
            ));
        }
        Ok(())
    }

    async fn poll_agentic(
        &mut self,
        limit: usize,
        wait_ms: u64,
    ) -> Result<AgenticEventBatch, EvaluatorWorkerError> {
        if limit == 0 {
            return Err(EvaluatorWorkerError::Protocol(
                "poll_agentic limit must be positive".to_string(),
            ));
        }
        let id = self.take_id()?;
        let batch = self
            .request(WorkerRequest::PollAgentic { id, limit, wait_ms })
            .await?;
        validate_agentic_events(&batch)?;
        Ok(batch)
    }

    async fn submit_model_results(
        &mut self,
        items: &[AgenticModelResult],
    ) -> Result<(), EvaluatorWorkerError> {
        if items.is_empty() {
            return Err(EvaluatorWorkerError::Protocol(
                "submit_model_results requires at least one item".to_string(),
            ));
        }
        for item in items {
            validate_submitted_model_result(item)?;
        }
        let id = self.take_id()?;
        let result: AcceptedModelResults = self
            .request(WorkerRequest::SubmitModelResults { id, items })
            .await?;
        let submitted = items
            .iter()
            .map(|item| item.call_id.clone())
            .collect::<Vec<_>>();
        if result.accepted != submitted {
            return Err(EvaluatorWorkerError::Protocol(
                "worker submit_model_results acknowledgement did not match submitted calls"
                    .to_string(),
            ));
        }
        Ok(())
    }

    async fn cancel_episodes(
        &mut self,
        episode_ids: &[EpisodeId],
    ) -> Result<(), EvaluatorWorkerError> {
        if episode_ids.is_empty() {
            return Err(EvaluatorWorkerError::Protocol(
                "cancel_episodes requires at least one episode".to_string(),
            ));
        }
        let id = self.take_id()?;
        let result: CancelledEpisodesResult = self
            .request(WorkerRequest::CancelEpisodes { id, episode_ids })
            .await?;
        if result.cancelled != episode_ids {
            return Err(EvaluatorWorkerError::Protocol(
                "worker cancel_episodes acknowledgement did not match submitted IDs".to_string(),
            ));
        }
        Ok(())
    }

    async fn finish_agentic(&mut self) -> Result<AgenticResultBatch, EvaluatorWorkerError> {
        let id = self.take_id()?;
        let batch: AgenticResultBatch = self.request(WorkerRequest::FinishAgentic { id }).await?;
        for result in &batch.items {
            validate_agentic_result(result)?;
        }
        Ok(batch)
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
    use crate::protocol::{
        AgenticEpisodeOutcome, AgenticEvaluatorEvent, AgenticInferenceStatus, EvaluatorGradeItem,
        ModelCallId, ProblemId,
    };

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

    const AGENTIC_FAKE_WORKER: &str = r#"
import json, sys
submitted = False
for line in sys.stdin:
    request = json.loads(line)
    op = request['op']
    if op == 'hello':
        result = {'protocol': 1, 'worker_version': 'fixture', 'python_version': '3', 'python_executable': sys.executable, 'packages': {'harbor': '0.18.0'}, 'worker_source_sha256': 'a' * 64, 'dependency_lock_sha256': 'b' * 64, 'container_digest': None, 'capabilities': ['load', 'next_problems', 'grade_batch', 'shutdown', 'agentic_harbor']}
    elif op == 'load_agentic':
        result = {'harness': 'harbor', 'harness_version': '0.18.0', 'harness_source_sha256': 'c' * 64, 'dataset': {'provider': 'fixture', 'benchmark': request['dataset'], 'repository': request['dataset'], 'revision': 'd' * 64, 'evaluation_splits': ['tasks']}, 'agent': 'aiperf-terminus-2', 'agent_version': 'fixture', 'environment': 'fixture', 'verifier': 'fixture verifier', 'episode_count': 1, 'primary_reward': 'reward'}
    elif op == 'next_episodes':
        result = {'items': [{'episode_id': 'episode-1', 'task': 'swebench.task', 'source': 'fixture/swebench'}], 'next_offset': 1, 'done': True}
    elif op == 'start_episodes':
        result = {'started': request['episode_ids']}
    elif op == 'poll_agentic' and not submitted:
        result = {'events': [{'kind': 'model_call', 'call': {'episode_id': 'episode-1', 'call_id': 'call-1', 'turn_index': 0, 'prompt': 'fix it', 'messages': [{'role': 'user', 'content': 'fix it'}], 'generation': {'max_tokens': 64, 'temperature': 0.0, 'top_p': 1.0, 'stop': []}, 'tools': []}}]}
    elif op == 'submit_model_results':
        submitted = True
        result = {'accepted': [item['call_id'] for item in request['items']]}
    elif op == 'poll_agentic':
        result = {'events': [{'kind': 'episode_completed', 'result': {'episode_id': 'episode-1', 'task': 'swebench.task', 'outcome': 'completed', 'rewards': {'reward': 1.0}, 'primary_reward': 'reward', 'duration_seconds': 1.0, 'model_calls': 1, 'prompt_tokens': 8, 'completion_tokens': 4}}]}
    elif op == 'finish_agentic':
        result = {'items': [{'episode_id': 'episode-1', 'task': 'swebench.task', 'outcome': 'completed', 'rewards': {'reward': 1.0}, 'primary_reward': 'reward', 'duration_seconds': 1.0, 'model_calls': 1, 'prompt_tokens': 8, 'completion_tokens': 4}]}
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
    async fn supervises_agentic_episode_callbacks_without_python_http() {
        let mut evaluator = PythonEvaluator::spawn(fixture_config(AGENTIC_FAKE_WORKER))
            .await
            .unwrap();
        assert!(AgenticEvaluator::supports_agentic(&evaluator));
        let identity = evaluator
            .load_agentic(
                "fixture/swebench@locked",
                "fixture-model",
                &AgenticEvaluatorLoadConfig::default(),
            )
            .await
            .unwrap();
        assert_eq!(identity.harness, "harbor");
        assert_eq!(identity.episode_count, 1);
        let page = evaluator.next_episodes(0, 10).await.unwrap();
        assert!(page.done);
        let episode_id = page.items[0].episode_id.clone();
        evaluator
            .start_episodes(std::slice::from_ref(&episode_id))
            .await
            .unwrap();
        let events = evaluator.poll_agentic(10, 0).await.unwrap();
        let AgenticEvaluatorEvent::ModelCall { call } = &events.events[0] else {
            panic!("expected model call")
        };
        assert_eq!(call.prompt, "fix it");
        evaluator
            .submit_model_results(&[AgenticModelResult {
                episode_id: episode_id.clone(),
                call_id: ModelCallId::new("call-1").unwrap(),
                status: AgenticInferenceStatus::Completed,
                response: "fixed".to_string(),
                reasoning: None,
                prompt_tokens: Some(8),
                completion_tokens: Some(4),
                cached_tokens: None,
                response_id: None,
                finish_reason: Some("stop".to_string()),
                assistant_message: None,
                error_kind: None,
                error_message: None,
            }])
            .await
            .unwrap();
        let events = evaluator.poll_agentic(10, 0).await.unwrap();
        let AgenticEvaluatorEvent::EpisodeCompleted { result } = &events.events[0] else {
            panic!("expected terminal episode")
        };
        assert_eq!(result.outcome, AgenticEpisodeOutcome::Completed);
        assert_eq!(result.rewards["reward"], 1.0);
        let final_results = evaluator.finish_agentic().await.unwrap();
        assert_eq!(final_results.items.len(), 1);
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
