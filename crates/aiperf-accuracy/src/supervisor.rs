// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dedicated-control-FD evaluator process supervision and protocol-v2 client.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs::File;
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::process::CommandExt;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use tokio::io::{
    AsyncBufRead, AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader,
    BufWriter,
};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;

use crate::canonical::{CanonicalJson, CanonicalJsonLimits};
use crate::isolation::{
    AttestedWorkerLaunch, EvaluatorIsolation, EvaluatorLaunchAttestor, IsolationQuiescenceProof,
};
use crate::lifecycle::{EvaluationLifecycle, EvaluationLifecycleState};
use crate::provider::{
    EvaluationDistributionDescriptor, EvaluationProvider, EvaluationProviderDescriptor,
    EvaluationProviderError, EvaluationProviderLauncher, EvaluatorProtocolLimits,
    PreparedEvaluationProviderLaunch, ProviderLaunchContext, ValidatedProviderConfig,
};
use crate::provider_protocol::{
    BindAssetsResult, CancelledUnitsResult, EVALUATOR_WORKER_PROTOCOL_V2, EvaluationEventBatch,
    EvaluationFinishCandidate, EvaluationIdentity, EvaluationPlan, EvaluationPlanRequest,
    EvaluationSchedulingMode, EvaluationUnitId, EvaluationUnitOccurrence,
    EvaluationUnitOccurrenceRequest, EvaluationUnitPage, EvaluationWorkerIdentity,
    EvaluatorWorkerRequest, EvaluatorWorkerResponse, HostOperationEvent, ProviderShutdownResult,
    ResolvedEvaluationAsset, StartedUnitsResult, SubmittedHostEventsResult,
};

const CONTROL_READ_FD: i32 = 3;
const CONTROL_WRITE_FD: i32 = 4;
const DEFAULT_CONTROL_TIMEOUT: Duration = Duration::from_secs(120);
const MAX_CONTROL_TIMEOUT: Duration = Duration::from_secs(300);
const STDERR_DRAIN_CHUNK_BYTES: usize = 8_192;
const STDERR_RESTRICTED_OUTPUT: &str = "provider_stderr_restricted_output";
const STDERR_DRAIN_FAILED: &str = "provider_stderr_drain_failed";

trait DynAsyncRead: AsyncBufRead + Unpin {}
impl<T: AsyncBufRead + Unpin> DynAsyncRead for T {}

trait DynAsyncWrite: AsyncWrite + Unpin {}
impl<T: AsyncWrite + Unpin> DynAsyncWrite for T {}

/// Sink for one fixed restricted evaluator-provider diagnostic classification.
pub trait EvaluationProviderLogSink: Send + Sync {
    /// Consume a fixed restricted diagnostic classification, never worker bytes.
    fn log_line(&self, classification: &str);
}

/// Default provider stderr sink.
#[derive(Debug, Clone, Copy, Default)]
pub struct StderrEvaluationProviderLogSink;

impl EvaluationProviderLogSink for StderrEvaluationProviderLogSink {
    fn log_line(&self, classification: &str) {
        eprintln!("[evaluation-provider] {classification}");
    }
}

#[derive(Clone)]
struct RestrictedProviderDiagnostics {
    sink: Arc<dyn EvaluationProviderLogSink>,
    output_reported: Arc<AtomicBool>,
    failure_reported: Arc<AtomicBool>,
}

impl RestrictedProviderDiagnostics {
    fn new(sink: Arc<dyn EvaluationProviderLogSink>) -> Self {
        Self {
            sink,
            output_reported: Arc::new(AtomicBool::new(false)),
            failure_reported: Arc::new(AtomicBool::new(false)),
        }
    }

    fn report_output(&self) {
        if !self.output_reported.swap(true, Ordering::Relaxed) {
            self.sink.log_line(STDERR_RESTRICTED_OUTPUT);
        }
    }

    fn report_failure(&self) {
        if !self.failure_reported.swap(true, Ordering::Relaxed) {
            self.sink.log_line(STDERR_DRAIN_FAILED);
        }
    }
}

async fn drain_provider_stderr<R: AsyncRead + Unpin>(
    mut stderr: R,
    diagnostics: RestrictedProviderDiagnostics,
) {
    let mut chunk = [0_u8; STDERR_DRAIN_CHUNK_BYTES];
    loop {
        match stderr.read(&mut chunk).await {
            Ok(0) => break,
            Ok(_) => diagnostics.report_output(),
            Err(_) => {
                diagnostics.report_failure();
                break;
            }
        }
    }
}

/// Concrete factory launcher using attestation, isolation, inherited pipes, and strict framing.
pub struct SupervisedEvaluationProviderLauncher {
    launches: BTreeMap<crate::provider_protocol::EvaluationDistributionId, AttestedWorkerLaunch>,
    attestor: Arc<dyn EvaluatorLaunchAttestor>,
    isolation: Arc<dyn EvaluatorIsolation>,
    log_sink: Arc<dyn EvaluationProviderLogSink>,
    shutdown_timeout: Duration,
    control_timeout: Duration,
    token_authority: Arc<()>,
}

impl fmt::Debug for SupervisedEvaluationProviderLauncher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SupervisedEvaluationProviderLauncher")
            .field("distributions", &self.launches.keys().collect::<Vec<_>>())
            .field("shutdown_timeout", &self.shutdown_timeout)
            .field("control_timeout", &self.control_timeout)
            .finish_non_exhaustive()
    }
}

impl SupervisedEvaluationProviderLauncher {
    /// Construct a launcher from factory-owned recipes and platform policy.
    pub fn new(
        launches: Vec<AttestedWorkerLaunch>,
        attestor: Arc<dyn EvaluatorLaunchAttestor>,
        isolation: Arc<dyn EvaluatorIsolation>,
    ) -> Result<Self, EvaluationProviderError> {
        let mut by_id = BTreeMap::new();
        for launch in launches {
            launch.validate()?;
            let id = launch.distribution_id.clone();
            if by_id.insert(id.clone(), launch).is_some() {
                return Err(EvaluationProviderError::Launch(format!(
                    "duplicate worker launch recipe for distribution {id}"
                )));
            }
        }
        if by_id.is_empty() {
            return Err(EvaluationProviderError::Launch(
                "supervised provider launcher had no immutable distributions".to_string(),
            ));
        }
        Ok(Self {
            launches: by_id,
            attestor,
            isolation,
            log_sink: Arc::new(StderrEvaluationProviderLogSink),
            shutdown_timeout: Duration::from_secs(30),
            control_timeout: DEFAULT_CONTROL_TIMEOUT,
            token_authority: Arc::new(()),
        })
    }

    /// Inject diagnostic handling.
    pub fn with_log_sink(mut self, sink: Arc<dyn EvaluationProviderLogSink>) -> Self {
        self.log_sink = sink;
        self
    }

    /// Set bounded graceful shutdown time.
    pub fn with_shutdown_timeout(
        mut self,
        timeout: Duration,
    ) -> Result<Self, EvaluationProviderError> {
        if timeout.is_zero() || timeout > Duration::from_secs(300) {
            return Err(EvaluationProviderError::Launch(
                "provider shutdown timeout was zero or exceeded 300 seconds".to_string(),
            ));
        }
        self.shutdown_timeout = timeout;
        Ok(self)
    }

    /// Set the base deadline for every evaluator control operation.
    ///
    /// A worker-requested long-poll duration is added to this base so a valid
    /// poll retains its authored wait while still having a finite deadline.
    pub fn with_control_timeout(
        mut self,
        timeout: Duration,
    ) -> Result<Self, EvaluationProviderError> {
        if timeout.is_zero() || timeout > MAX_CONTROL_TIMEOUT {
            return Err(EvaluationProviderError::Launch(
                "provider control timeout was zero or exceeded 300 seconds".to_string(),
            ));
        }
        self.control_timeout = timeout;
        Ok(self)
    }
}

#[derive(Debug)]
struct SupervisedPreparedEvaluationProviderLaunch {
    provider_id: crate::provider_protocol::EvaluationProviderId,
    distribution_id: crate::provider_protocol::EvaluationDistributionId,
    prelaunch_context_sha256: String,
    context: ProviderLaunchContext,
    launch: AttestedWorkerLaunch,
    attestation: crate::isolation::LaunchAttestation,
    prepared: crate::isolation::PreparedEvaluatorLaunch,
    token_authority: Arc<()>,
}

impl PreparedEvaluationProviderLaunch for SupervisedPreparedEvaluationProviderLaunch {
    fn distribution_id(&self) -> &crate::provider_protocol::EvaluationDistributionId {
        &self.distribution_id
    }

    fn prelaunch_context_sha256(&self) -> &str {
        &self.prelaunch_context_sha256
    }

    fn isolation_evidence(&self) -> &crate::isolation::EvaluatorIsolationEvidence {
        &self.prepared.evidence
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[async_trait(?Send)]
impl EvaluationProviderLauncher for SupervisedEvaluationProviderLauncher {
    fn check_distribution_available(
        &self,
        distribution: &EvaluationDistributionDescriptor,
    ) -> Result<(), EvaluationProviderError> {
        let launch = self
            .launches
            .get(&distribution.distribution_id)
            .ok_or_else(|| {
                EvaluationProviderError::FactoryMismatch(format!(
                    "no factory-owned launch recipe for distribution {}",
                    distribution.distribution_id
                ))
            })?;
        self.attestor.attest(launch, distribution)?;
        self.isolation.check_available()
    }

    fn prepare_launch(
        &self,
        descriptor: &EvaluationProviderDescriptor,
        distribution: &EvaluationDistributionDescriptor,
        mut context: ProviderLaunchContext,
    ) -> Result<Box<dyn PreparedEvaluationProviderLaunch>, EvaluationProviderError> {
        let launch = self
            .launches
            .get(&distribution.distribution_id)
            .ok_or_else(|| {
                EvaluationProviderError::FactoryMismatch(format!(
                    "no factory-owned launch recipe for distribution {}",
                    distribution.distribution_id
                ))
            })?;
        let staging_dir = std::fs::canonicalize(&context.staging_dir).map_err(|error| {
            EvaluationProviderError::Launch(format!(
                "failed to resolve evaluator staging root: {error}"
            ))
        })?;
        if !staging_dir.is_dir() {
            return Err(EvaluationProviderError::Launch(
                "evaluator staging root was not a directory".to_string(),
            ));
        }
        context.staging_dir = staging_dir;
        context
            .validate()
            .map_err(EvaluationProviderError::registry)?;
        let prelaunch_context_sha256 = context
            .binding_sha256()
            .map_err(EvaluationProviderError::registry)?;
        let attestation = self.attestor.attest(launch, distribution)?;
        let prepared = self.isolation.prepare(launch, &attestation, &context)?;
        prepared.evidence.validate_strict()?;
        if prepared.evidence.enforced != descriptor.isolation {
            return Err(EvaluationProviderError::FactoryMismatch(
                "prepared isolation outcomes differed from the provider descriptor".to_string(),
            ));
        }
        Ok(Box::new(SupervisedPreparedEvaluationProviderLaunch {
            provider_id: descriptor.provider_id.clone(),
            distribution_id: distribution.distribution_id.clone(),
            prelaunch_context_sha256,
            context,
            launch: launch.clone(),
            attestation,
            prepared,
            token_authority: Arc::clone(&self.token_authority),
        }))
    }

    async fn launch(
        &self,
        descriptor: &EvaluationProviderDescriptor,
        config: &ValidatedProviderConfig,
        host_binding: crate::provider_protocol::EvaluationHostBinding,
        prepared: Box<dyn PreparedEvaluationProviderLaunch>,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
        if config.provider_id() != &descriptor.provider_id
            || config.schema_sha256() != descriptor.config_schema_sha256
            || config.schema_version() != descriptor.config_schema_version
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "launcher received config from a different provider/schema".to_string(),
            ));
        }
        let prepared = prepared
            .into_any()
            .downcast::<SupervisedPreparedEvaluationProviderLaunch>()
            .map_err(|_| {
                EvaluationProviderError::FactoryMismatch(
                    "provider launcher received a foreign prepared token".to_string(),
                )
            })?;
        if !Arc::ptr_eq(&prepared.token_authority, &self.token_authority)
            || prepared.provider_id != descriptor.provider_id
            || prepared.prelaunch_context_sha256
                != prepared
                    .context
                    .binding_sha256()
                    .map_err(EvaluationProviderError::registry)?
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "prepared provider launch authority/context drifted".to_string(),
            ));
        }
        let distribution = descriptor
            .distribution(&prepared.distribution_id)
            .ok_or_else(|| {
                EvaluationProviderError::FactoryMismatch(format!(
                    "prepared distribution {} is not registered for provider {}",
                    prepared.distribution_id, descriptor.provider_id
                ))
            })?;
        let measured = self.attestor.attest(&prepared.launch, distribution)?;
        if measured != prepared.attestation {
            return Err(EvaluationProviderError::FactoryMismatch(
                "worker launch closure drifted after preparation".to_string(),
            ));
        }
        self.isolation.check_available()?;
        host_binding.validate()?;
        if host_binding.host.isolation_proof_sha256 != prepared.prepared.evidence.proof_sha256 {
            return Err(EvaluationProviderError::FactoryMismatch(
                "host binding did not preserve prepared isolation evidence".to_string(),
            ));
        }
        let SupervisedPreparedEvaluationProviderLaunch {
            context, prepared, ..
        } = *prepared;
        let provider = SupervisedEvaluationProvider::spawn(
            descriptor,
            distribution,
            SupervisedSpawnInput {
                prepared,
                context,
                host_binding,
            },
            Arc::clone(&self.isolation),
            Arc::clone(&self.log_sink),
            self.shutdown_timeout,
            self.control_timeout,
        )
        .await?;
        Ok(Box::new(provider))
    }
}

struct SupervisedProcess {
    child: Child,
    root_pid: u32,
    stderr_task: Option<JoinHandle<()>>,
    diagnostics: RestrictedProviderDiagnostics,
    isolation: Arc<dyn EvaluatorIsolation>,
    shutdown_timeout: Duration,
    quiescence_result: Option<Result<IsolationQuiescenceProof, EvaluationProviderError>>,
}

impl SupervisedProcess {
    async fn join_stderr(&mut self) {
        let Some(mut task) = self.stderr_task.take() else {
            return;
        };
        match tokio::time::timeout(self.shutdown_timeout, &mut task).await {
            Ok(Ok(())) => {}
            Ok(Err(_)) => self.diagnostics.report_failure(),
            Err(_) => {
                task.abort();
                let _ = task.await;
                self.diagnostics.report_failure();
            }
        }
    }

    async fn force_kill_and_wait(&mut self) {
        if self.child.start_kill().is_err() {
            // The process may already have exited between the failed wait and
            // this cleanup path. A bounded wait plus the isolation verifier is
            // authoritative, so the kill syscall outcome is not exposed.
        }
        let _ = tokio::time::timeout(self.shutdown_timeout, self.child.wait()).await;
    }

    fn verify_quiescent_once(
        &mut self,
    ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
        if let Some(result) = &self.quiescence_result {
            return result.clone();
        }
        let result = self.isolation.verify_quiescent(self.root_pid);
        self.quiescence_result = Some(result.clone());
        result
    }

    async fn terminate_with_primary(
        &mut self,
        primary: EvaluationProviderError,
    ) -> EvaluationProviderError {
        match self.abort_quiescent().await {
            Ok(_) => primary,
            Err(quiescence) => quiescence,
        }
    }

    async fn abort_quiescent(
        &mut self,
    ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
        if self.quiescence_result.is_none() {
            self.force_kill_and_wait().await;
            self.join_stderr().await;
        }
        self.verify_quiescent_once()
    }

    async fn wait_quiescent(
        &mut self,
    ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
        let waited = tokio::time::timeout(self.shutdown_timeout, self.child.wait()).await;
        let primary = match waited {
            Ok(Ok(status)) if status.success() => None,
            Ok(Ok(status)) => Some(EvaluationProviderError::Crashed(status.to_string())),
            Ok(Err(error)) => {
                self.force_kill_and_wait().await;
                Some(EvaluationProviderError::Io(error.to_string()))
            }
            Err(_) => {
                self.force_kill_and_wait().await;
                Some(EvaluationProviderError::Crashed(
                    "evaluator worker exceeded graceful shutdown timeout and was force-killed"
                        .to_string(),
                ))
            }
        };
        self.join_stderr().await;
        let proof = self.verify_quiescent_once()?;
        match primary {
            Some(error) => Err(error),
            None => Ok(proof),
        }
    }
}

/// Long-lived process-backed implementation of [`EvaluationProvider`].
pub struct SupervisedEvaluationProvider {
    process: SupervisedProcess,
    reader: Box<dyn DynAsyncRead>,
    writer: BufWriter<Box<dyn DynAsyncWrite>>,
    identity: EvaluationWorkerIdentity,
    lifecycle: EvaluationLifecycle,
    limits: EvaluatorProtocolLimits,
    control_timeout: Duration,
    next_request_id: u64,
    proxy: Option<crate::provider_protocol::ScopedProxyBinding>,
    host_binding: crate::provider_protocol::EvaluationHostBinding,
    plan_request: Option<EvaluationPlanRequest>,
    plan: Option<EvaluationPlan>,
    frozen_identity: Option<EvaluationIdentity>,
    expected_page_offset: usize,
    finite_pages_done: bool,
    paged_unit_count: usize,
    paged_case_count: usize,
    finish_candidate: Option<EvaluationFinishCandidate>,
    quiescence_proof: Option<IsolationQuiescenceProof>,
}

struct SupervisedSpawnInput {
    prepared: crate::isolation::PreparedEvaluatorLaunch,
    context: ProviderLaunchContext,
    host_binding: crate::provider_protocol::EvaluationHostBinding,
}

impl fmt::Debug for SupervisedEvaluationProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SupervisedEvaluationProvider")
            .field("identity", &self.identity)
            .field("lifecycle", &self.lifecycle.state())
            .field(
                "outstanding_host_operations",
                &self.lifecycle.outstanding_host_operations(),
            )
            .finish_non_exhaustive()
    }
}

impl SupervisedEvaluationProvider {
    async fn spawn(
        descriptor: &EvaluationProviderDescriptor,
        distribution: &EvaluationDistributionDescriptor,
        input: SupervisedSpawnInput,
        isolation: Arc<dyn EvaluatorIsolation>,
        log_sink: Arc<dyn EvaluationProviderLogSink>,
        shutdown_timeout: Duration,
        control_timeout: Duration,
    ) -> Result<Self, EvaluationProviderError> {
        let SupervisedSpawnInput {
            prepared,
            context,
            host_binding,
        } = input;
        let lifecycle = EvaluationLifecycle::new(context.protocol_limits)?;
        let (request_child, request_parent) = create_control_pipe()?;
        let (response_parent, response_child) = create_control_pipe()?;

        let request_parent_fd = request_parent.as_raw_fd();
        let request_child_fd = request_child.as_raw_fd();
        let response_parent_fd = response_parent.as_raw_fd();
        let response_child_fd = response_child.as_raw_fd();
        let limits = prepared.resource_limits;

        let mut command = Command::new(&prepared.program);
        command
            .args(&prepared.args)
            .env_clear()
            .envs(&prepared.environment)
            .current_dir(&prepared.current_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        // SAFETY: the closure uses only async-signal-safe libc calls between
        // fork and exec. All captured values are plain integers.
        unsafe {
            command.as_std_mut().pre_exec(move || {
                duplicate_control_fd(request_child_fd, CONTROL_READ_FD)?;
                duplicate_control_fd(response_child_fd, CONTROL_WRITE_FD)?;
                for fd in [
                    request_parent_fd,
                    request_child_fd,
                    response_parent_fd,
                    response_child_fd,
                ] {
                    if fd > CONTROL_WRITE_FD {
                        libc::close(fd);
                    }
                }
                if libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                set_resource_limit(libc::RLIMIT_AS, limits.address_space_bytes)?;
                set_resource_limit(libc::RLIMIT_FSIZE, limits.file_size_bytes)?;
                set_resource_limit(libc::RLIMIT_NOFILE, limits.open_files)?;
                set_resource_limit(libc::RLIMIT_NPROC, limits.processes)?;
                set_resource_limit(libc::RLIMIT_CPU, limits.cpu_seconds)?;
                Ok(())
            });
        }
        let mut child = command.spawn().map_err(|error| {
            EvaluationProviderError::Launch(format!("isolated worker spawn failed: {error}"))
        })?;
        drop(request_child);
        drop(response_child);
        let Some(root_pid) = child.id() else {
            let _ = child.start_kill();
            let _ = tokio::time::timeout(shutdown_timeout, child.wait()).await;
            return Err(EvaluationProviderError::Launch(
                "spawned evaluator worker had no PID".to_string(),
            ));
        };
        let diagnostics = RestrictedProviderDiagnostics::new(log_sink);
        let stderr_task = child.stderr.take().map(|stderr| {
            let diagnostics = diagnostics.clone();
            tokio::spawn(drain_provider_stderr(stderr, diagnostics))
        });
        let mut process = SupervisedProcess {
            child,
            root_pid,
            stderr_task,
            diagnostics,
            isolation,
            shutdown_timeout,
            quiescence_result: None,
        };
        if process.stderr_task.is_none() {
            let error = process
                .terminate_with_primary(EvaluationProviderError::Launch(
                    "evaluator worker had no stderr pipe".to_string(),
                ))
                .await;
            return Err(error);
        }
        if let Some(binder) = &context.process_root_binder
            && let Err(error) = binder.bind_attested_root(root_pid)
        {
            let error = process
                .terminate_with_primary(EvaluationProviderError::registry(error))
                .await;
            return Err(error);
        }
        let request_stream = tokio::fs::File::from_std(request_parent);
        let response_stream = tokio::fs::File::from_std(response_parent);
        let reader: Box<dyn DynAsyncRead> = Box::new(BufReader::new(response_stream));
        let writer: BufWriter<Box<dyn DynAsyncWrite>> = BufWriter::new(Box::new(request_stream));
        let mut provider = Self {
            process,
            reader,
            writer,
            identity: placeholder_identity(descriptor, distribution, &context.launch_nonce),
            lifecycle,
            limits: context.protocol_limits,
            control_timeout,
            next_request_id: 1,
            proxy: context.proxy.clone(),
            host_binding,
            plan_request: None,
            plan: None,
            frozen_identity: None,
            expected_page_offset: 0,
            finite_pages_done: false,
            paged_unit_count: 0,
            paged_case_count: 0,
            finish_candidate: None,
            quiescence_proof: None,
        };
        let handshake = async {
            let id = provider.take_request_id()?;
            let identity: EvaluationWorkerIdentity = provider
                .request(EvaluatorWorkerRequest::Hello {
                    id,
                    protocol: EVALUATOR_WORKER_PROTOCOL_V2,
                    max_message_bytes: provider.limits.max_message_bytes,
                    max_collection_items: provider.limits.max_collection_items,
                    launch_nonce: context.launch_nonce.clone(),
                })
                .await?;
            identity.validate()?;
            validate_negotiated_identity(
                descriptor,
                distribution,
                &context.launch_nonce,
                &identity,
            )?;
            provider.identity = identity;
            provider.lifecycle.negotiated()?;
            Ok::<(), EvaluationProviderError>(())
        }
        .await;
        match handshake {
            Ok(()) => Ok(provider),
            Err(error) => {
                let error = provider.process.terminate_with_primary(error).await;
                Err(error)
            }
        }
    }

    fn take_request_id(&mut self) -> Result<u64, EvaluationProviderError> {
        let id = self.next_request_id;
        self.next_request_id = self.next_request_id.checked_add(1).ok_or_else(|| {
            EvaluationProviderError::Protocol("request correlation ID overflow".to_string())
        })?;
        Ok(id)
    }

    async fn request<T: DeserializeOwned>(
        &mut self,
        request: EvaluatorWorkerRequest,
    ) -> Result<T, EvaluationProviderError> {
        let operation = request_operation(&request);
        let deadline = request_deadline(self.control_timeout, &request);
        match tokio::time::timeout(deadline, self.request_once(request)).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(error)) => Err(self.process.terminate_with_primary(error).await),
            Err(_) => {
                let primary = EvaluationProviderError::Crashed(format!(
                    "evaluator control operation {operation} exceeded its bounded deadline"
                ));
                Err(self.process.terminate_with_primary(primary).await)
            }
        }
    }

    async fn request_once<T: DeserializeOwned>(
        &mut self,
        request: EvaluatorWorkerRequest,
    ) -> Result<T, EvaluationProviderError> {
        let expected_id = request.id();
        let bytes = serde_json::to_vec(&request)
            .map_err(|error| EvaluationProviderError::Protocol(error.to_string()))?;
        if bytes.len() > self.limits.max_message_bytes {
            return Err(EvaluationProviderError::Protocol(format!(
                "evaluator-worker protocol v2 request exceeded {} bytes",
                self.limits.max_message_bytes
            )));
        }
        self.writer.write_all(&bytes).await.map_err(|error| {
            EvaluationProviderError::Io(format!("writing request control descriptor: {error}"))
        })?;
        self.writer.write_all(b"\n").await.map_err(|error| {
            EvaluationProviderError::Io(format!("writing request newline: {error}"))
        })?;
        self.writer.flush().await.map_err(|error| {
            EvaluationProviderError::Io(format!("flushing request control descriptor: {error}"))
        })?;

        let line = read_bounded_line(&mut *self.reader, self.limits.max_message_bytes).await?;
        let strict = CanonicalJson::from_slice(
            &line,
            CanonicalJsonLimits {
                max_collection_items: self.limits.max_collection_items,
                max_string_bytes: self.limits.max_message_bytes,
                ..Default::default()
            },
        )
        .map_err(|error| EvaluationProviderError::Protocol(error.to_string()))?;
        let mut response: EvaluatorWorkerResponse = serde_json::from_value(strict.into_value())
            .map_err(|error| EvaluationProviderError::Protocol(error.to_string()))?;
        if response.id != expected_id {
            return Err(EvaluationProviderError::Protocol(format!(
                "response correlation ID {} did not match request {expected_id}",
                response.id
            )));
        }
        match (response.ok, response.result.take(), response.error.take()) {
            (true, Some(result), None) => serde_json::from_value(result)
                .map_err(|error| EvaluationProviderError::Protocol(error.to_string())),
            (false, None, Some(mut error)) => {
                error.validate_and_redact()?;
                Err(EvaluationProviderError::Remote(error))
            }
            _ => Err(EvaluationProviderError::Protocol(
                "worker reply did not contain exactly one success result or failure error"
                    .to_string(),
            )),
        }
    }

    async fn finish_worker_response<T>(
        &mut self,
        validation: Result<T, EvaluationProviderError>,
    ) -> Result<T, EvaluationProviderError> {
        match validation {
            Ok(value) => Ok(value),
            Err(error) => Err(self.process.terminate_with_primary(error).await),
        }
    }

    fn ensure_collection(
        &self,
        len: usize,
        operation: &str,
    ) -> Result<(), EvaluationProviderError> {
        if len == 0 || len > self.limits.max_collection_items {
            Err(EvaluationProviderError::Protocol(format!(
                "{operation} collection was empty or exceeded {} items",
                self.limits.max_collection_items
            )))
        } else {
            Ok(())
        }
    }
}

#[async_trait(?Send)]
impl EvaluationProvider for SupervisedEvaluationProvider {
    fn identity(&self) -> &EvaluationWorkerIdentity {
        &self.identity
    }

    fn lifecycle_state(&self) -> EvaluationLifecycleState {
        self.lifecycle.state()
    }

    fn quiescence_proof(&self) -> Option<&IsolationQuiescenceProof> {
        self.quiescence_proof.as_ref()
    }

    async fn plan(
        &mut self,
        request: &EvaluationPlanRequest,
    ) -> Result<EvaluationPlan, EvaluationProviderError> {
        request.validate()?;
        if request.provider_id != self.identity.provider_id
            || request.distribution_id != self.identity.distribution_id
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "plan request provider/distribution did not match the launched worker".to_string(),
            ));
        }
        let id = self.take_request_id()?;
        let plan: EvaluationPlan = self
            .request(EvaluatorWorkerRequest::PlanSession {
                id,
                request: request.clone(),
            })
            .await?;
        let validation = (|| {
            self.lifecycle.planned(&plan)?;
            self.plan_request = Some(request.clone());
            self.plan = Some(plan.clone());
            Ok(plan)
        })();
        self.finish_worker_response(validation).await
    }

    async fn bind_assets(
        &mut self,
        assets: &[ResolvedEvaluationAsset],
    ) -> Result<EvaluationIdentity, EvaluationProviderError> {
        if assets.len() > self.limits.max_collection_items {
            return Err(EvaluationProviderError::Protocol(
                "bind_assets exceeded collection limit".to_string(),
            ));
        }
        let plan = self.plan.as_ref().ok_or_else(|| {
            EvaluationProviderError::Lifecycle("bind_assets preceded plan_session".to_string())
        })?;
        validate_asset_binding(plan, assets)?;
        let id = self.take_request_id()?;
        let result: BindAssetsResult = self
            .request(EvaluatorWorkerRequest::BindAssets {
                id,
                assets: assets.to_vec(),
                proxy: self.proxy.clone(),
                host_binding: Box::new(self.host_binding.clone()),
            })
            .await?;
        let validation = (|| {
            let identity = result.identity;
            identity.validate()?;
            let plan_request = self
                .plan_request
                .as_ref()
                .expect("plan and request set together");
            if identity.worker != self.identity
                || identity.config_schema_sha256 != plan_request.config_schema_sha256
                || identity.resolved_config_sha256
                    != plan_request.provider_config.normalized_result_sha256()
                || !self.host_binding.matches(&identity)
            {
                return Err(EvaluationProviderError::FactoryMismatch(
                    "bound evaluation identity drifted from worker/schema/config evidence"
                        .to_string(),
                ));
            }
            self.lifecycle.assets_bound_and_ready()?;
            self.frozen_identity = Some(identity.clone());
            Ok(identity)
        })();
        self.finish_worker_response(validation).await
    }

    async fn next_units(
        &mut self,
        offset: usize,
        limit: usize,
    ) -> Result<EvaluationUnitPage, EvaluationProviderError> {
        self.ensure_collection(limit, "next_units limit")?;
        let plan = self.plan.as_ref().ok_or_else(|| {
            EvaluationProviderError::Lifecycle("next_units preceded plan_session".to_string())
        })?;
        let scheduling_mode = plan.scheduling_mode;
        let finite_unit_count = plan.finite_unit_count;
        let finite_case_count = plan.finite_case_count;
        if scheduling_mode != EvaluationSchedulingMode::Finite
            || offset != self.expected_page_offset
            || self.finite_pages_done
        {
            return Err(EvaluationProviderError::Protocol(
                "next_units used a non-finite plan, noncanonical offset, or exhausted page stream"
                    .to_string(),
            ));
        }
        let id = self.take_request_id()?;
        let page: EvaluationUnitPage = self
            .request(EvaluatorWorkerRequest::NextUnits { id, offset, limit })
            .await?;
        let validation = (|| {
            if page.items.len() > limit
                || page.next_offset != offset + page.items.len()
                || (!page.done && page.items.is_empty())
            {
                return Err(EvaluationProviderError::Protocol(
                    "next_units page violated offset/limit/progress invariants".to_string(),
                ));
            }
            self.lifecycle.register_units(&page.items)?;
            self.expected_page_offset = page.next_offset;
            self.paged_unit_count += page.items.len();
            self.paged_case_count += page
                .items
                .iter()
                .map(|unit| unit.cases.len())
                .sum::<usize>();
            if page.done {
                if self.paged_unit_count != finite_unit_count.expect("finite plan validated")
                    || self.paged_case_count != finite_case_count.expect("finite plan validated")
                {
                    return Err(EvaluationProviderError::Protocol(
                        "finite unit pages did not exhaust declared unit/case counts".to_string(),
                    ));
                }
                self.finite_pages_done = true;
            }
            Ok(page)
        })();
        self.finish_worker_response(validation).await
    }

    async fn instantiate_units(
        &mut self,
        requests: &[EvaluationUnitOccurrenceRequest],
    ) -> Result<Vec<EvaluationUnitOccurrence>, EvaluationProviderError> {
        self.ensure_collection(requests.len(), "instantiate_units")?;
        self.lifecycle.validate_instantiate()?;
        let identities = requests
            .iter()
            .map(|request| {
                (
                    &request.unit_template_id,
                    &request.phase_id,
                    request.issue_ordinal,
                    request.cycle_index,
                )
            })
            .collect::<BTreeSet<_>>();
        if identities.len() != requests.len() {
            return Err(EvaluationProviderError::Protocol(
                "instantiate_units contained duplicate occurrence identity".to_string(),
            ));
        }
        let identity = self.frozen_identity.as_ref().ok_or_else(|| {
            EvaluationProviderError::Lifecycle("instantiate_units preceded bind_assets".to_string())
        })?;
        let templates = identity
            .unit_templates
            .iter()
            .map(|template| &template.unit_template_id)
            .collect::<BTreeSet<_>>();
        if requests
            .iter()
            .any(|request| !templates.contains(&request.unit_template_id))
        {
            return Err(EvaluationProviderError::Protocol(
                "instantiate_units referenced an unknown frozen template".to_string(),
            ));
        }
        let id = self.take_request_id()?;
        let units: Vec<EvaluationUnitOccurrence> = self
            .request(EvaluatorWorkerRequest::InstantiateUnits {
                id,
                requests: requests.to_vec(),
            })
            .await?;
        let validation = (|| {
            if units.len() != requests.len()
                || units
                    .iter()
                    .zip(requests)
                    .any(|(unit, request)| unit.unit_template_id != request.unit_template_id)
            {
                return Err(EvaluationProviderError::Protocol(
                    "instantiate_units response did not match requested template order".to_string(),
                ));
            }
            self.lifecycle.register_units(&units)?;
            Ok(units)
        })();
        self.finish_worker_response(validation).await
    }

    async fn start_units(
        &mut self,
        ids: &[EvaluationUnitId],
    ) -> Result<(), EvaluationProviderError> {
        self.ensure_collection(ids.len(), "start_units")?;
        let unique = ids.iter().collect::<BTreeSet<_>>();
        if unique.len() != ids.len() {
            return Err(EvaluationProviderError::Protocol(
                "start_units contained duplicate IDs".to_string(),
            ));
        }
        let id = self.take_request_id()?;
        let result: StartedUnitsResult = self
            .request(EvaluatorWorkerRequest::StartUnits {
                id,
                unit_ids: ids.to_vec(),
            })
            .await?;
        let validation = (|| {
            if result.started != ids {
                return Err(EvaluationProviderError::Protocol(
                    "start_units acknowledgement did not exactly match submitted IDs".to_string(),
                ));
            }
            self.lifecycle.start_units(ids)
        })();
        self.finish_worker_response(validation).await
    }

    async fn poll_events(
        &mut self,
        limit: usize,
        wait_ms: u64,
    ) -> Result<EvaluationEventBatch, EvaluationProviderError> {
        self.ensure_collection(limit, "poll_events limit")?;
        if wait_ms > self.limits.max_poll_wait_ms {
            return Err(EvaluationProviderError::Protocol(
                "poll_events wait exceeded negotiated bound".to_string(),
            ));
        }
        let id = self.take_request_id()?;
        let mut batch: EvaluationEventBatch = self
            .request(EvaluatorWorkerRequest::PollEvents { id, limit, wait_ms })
            .await?;
        let validation = (|| {
            if batch.events.len() > limit {
                return Err(EvaluationProviderError::Protocol(
                    "poll_events worker returned more events than requested".to_string(),
                ));
            }
            self.lifecycle.record_event_batch(&mut batch)?;
            Ok(batch)
        })();
        self.finish_worker_response(validation).await
    }

    async fn submit_host_events(
        &mut self,
        events: &[HostOperationEvent],
    ) -> Result<(), EvaluationProviderError> {
        self.ensure_collection(events.len(), "submit_host_events")?;
        let mut checked = events.to_vec();
        let expected = self.lifecycle.record_host_events(&mut checked)?;
        let id = self.take_request_id()?;
        let result: SubmittedHostEventsResult = self
            .request(EvaluatorWorkerRequest::SubmitHostEvents {
                id,
                events: checked,
            })
            .await?;
        let validation = if result.accepted == expected {
            Ok(())
        } else {
            Err(EvaluationProviderError::Protocol(
                "submit_host_events acknowledgement did not exactly match event order".to_string(),
            ))
        };
        self.finish_worker_response(validation).await
    }

    async fn cancel_units(
        &mut self,
        ids: &[EvaluationUnitId],
    ) -> Result<(), EvaluationProviderError> {
        self.ensure_collection(ids.len(), "cancel_units")?;
        self.lifecycle.begin_cancellation(ids)?;
        let id = self.take_request_id()?;
        let result: CancelledUnitsResult = self
            .request(EvaluatorWorkerRequest::CancelUnits {
                id,
                unit_ids: ids.to_vec(),
            })
            .await?;
        let validation = if result.cancelled == ids {
            Ok(())
        } else {
            Err(EvaluationProviderError::Protocol(
                "cancel_units acknowledgement did not exactly match submitted IDs".to_string(),
            ))
        };
        self.finish_worker_response(validation).await
    }

    async fn finalize_candidate(
        &mut self,
    ) -> Result<EvaluationFinishCandidate, EvaluationProviderError> {
        if self
            .plan
            .as_ref()
            .is_some_and(|plan| plan.scheduling_mode == EvaluationSchedulingMode::Finite)
            && !self.finite_pages_done
        {
            return Err(EvaluationProviderError::Protocol(
                "finite provider cannot finalize before every unit page is consumed".to_string(),
            ));
        }
        if self.finish_candidate.is_some() {
            return Err(EvaluationProviderError::Protocol(
                "finalize_session returned more than one manifest candidate".to_string(),
            ));
        }
        let id = self.take_request_id()?;
        let mut candidate: EvaluationFinishCandidate = self
            .request(EvaluatorWorkerRequest::FinalizeSession { id })
            .await?;
        let validation = (|| {
            candidate.validate()?;
            if candidate.identity
                != *self.frozen_identity.as_ref().ok_or_else(|| {
                    EvaluationProviderError::Lifecycle(
                        "finalize_session preceded bind_assets".to_string(),
                    )
                })?
            {
                return Err(EvaluationProviderError::FactoryMismatch(
                    "final evaluation identity drifted after asset binding".to_string(),
                ));
            }
            self.lifecycle.validate_finish_candidate(&candidate)?;
            self.lifecycle.finalized_candidate()?;
            self.finish_candidate = Some(candidate.clone());
            Ok(candidate)
        })();
        self.finish_worker_response(validation).await
    }

    async fn shutdown(&mut self) -> Result<(), EvaluationProviderError> {
        if matches!(
            self.lifecycle.state(),
            EvaluationLifecycleState::WorkerExited
                | EvaluationLifecycleState::ArtifactsSealed
                | EvaluationLifecycleState::ReportCommitted
        ) {
            return Ok(());
        }
        self.lifecycle.begin_shutdown()?;
        let id = self.take_request_id()?;
        let result: ProviderShutdownResult =
            match self.request(EvaluatorWorkerRequest::Shutdown { id }).await {
                Ok(result) => result,
                Err(error) => {
                    return Err(self.process.terminate_with_primary(error).await);
                }
            };
        if !result.shutdown {
            let error = EvaluationProviderError::Protocol(
                "worker did not acknowledge shutdown".to_string(),
            );
            return Err(self.process.terminate_with_primary(error).await);
        }
        match tokio::time::timeout(self.control_timeout, self.writer.shutdown()).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let error = EvaluationProviderError::Io(error.to_string());
                return Err(self.process.terminate_with_primary(error).await);
            }
            Err(_) => {
                let error = EvaluationProviderError::Crashed(
                    "evaluator control descriptor shutdown exceeded its bounded deadline"
                        .to_string(),
                );
                return Err(self.process.terminate_with_primary(error).await);
            }
        }
        let proof = self.process.wait_quiescent().await?;
        self.lifecycle.worker_exited()?;
        self.quiescence_proof = Some(proof);
        Ok(())
    }

    async fn abort(&mut self) -> Result<(), EvaluationProviderError> {
        if self.quiescence_proof.is_some() {
            return Ok(());
        }
        self.lifecycle.abort_to_quiescing();
        let proof = self.process.abort_quiescent().await?;
        self.lifecycle.worker_exited()?;
        self.quiescence_proof = Some(proof);
        Ok(())
    }

    fn mark_artifacts_sealed(&mut self) -> Result<(), EvaluationProviderError> {
        if self.quiescence_proof.is_none() {
            return Err(EvaluationProviderError::Lifecycle(
                "artifact sealing cannot precede process-tree quiescence".to_string(),
            ));
        }
        self.lifecycle.artifacts_sealed()
    }

    fn mark_report_committed(&mut self) -> Result<(), EvaluationProviderError> {
        self.lifecycle.report_committed()
    }
}

fn request_operation(request: &EvaluatorWorkerRequest) -> &'static str {
    match request {
        EvaluatorWorkerRequest::Hello { .. } => "hello",
        EvaluatorWorkerRequest::PlanSession { .. } => "plan_session",
        EvaluatorWorkerRequest::BindAssets { .. } => "bind_assets",
        EvaluatorWorkerRequest::NextUnits { .. } => "next_units",
        EvaluatorWorkerRequest::InstantiateUnits { .. } => "instantiate_units",
        EvaluatorWorkerRequest::StartUnits { .. } => "start_units",
        EvaluatorWorkerRequest::PollEvents { .. } => "poll_events",
        EvaluatorWorkerRequest::SubmitHostEvents { .. } => "submit_host_events",
        EvaluatorWorkerRequest::CancelUnits { .. } => "cancel_units",
        EvaluatorWorkerRequest::FinalizeSession { .. } => "finalize_session",
        EvaluatorWorkerRequest::Shutdown { .. } => "shutdown",
    }
}

fn request_deadline(base: Duration, request: &EvaluatorWorkerRequest) -> Duration {
    let worker_wait = match request {
        EvaluatorWorkerRequest::PollEvents { wait_ms, .. } => Duration::from_millis(*wait_ms),
        _ => Duration::ZERO,
    };
    base.saturating_add(worker_wait)
}

async fn read_bounded_line<R: AsyncBufRead + Unpin + ?Sized>(
    reader: &mut R,
    max_bytes: usize,
) -> Result<Vec<u8>, EvaluationProviderError> {
    let mut output = Vec::new();
    loop {
        let buffer = reader.fill_buf().await.map_err(|error| {
            EvaluationProviderError::Io(format!("reading response control descriptor: {error}"))
        })?;
        if buffer.is_empty() {
            return Err(EvaluationProviderError::Crashed(
                "worker closed the response descriptor before a complete reply".to_string(),
            ));
        }
        let newline = buffer.iter().position(|byte| *byte == b'\n');
        let take = newline.unwrap_or(buffer.len());
        if output.len().saturating_add(take) > max_bytes {
            return Err(EvaluationProviderError::Protocol(format!(
                "evaluator-worker protocol v2 response exceeded {max_bytes} bytes"
            )));
        }
        output.extend_from_slice(&buffer[..take]);
        reader.consume(take + usize::from(newline.is_some()));
        if newline.is_some() {
            if output.is_empty() {
                return Err(EvaluationProviderError::Protocol(
                    "worker emitted an empty protocol line".to_string(),
                ));
            }
            return Ok(output);
        }
    }
}

fn validate_negotiated_identity(
    descriptor: &EvaluationProviderDescriptor,
    distribution: &EvaluationDistributionDescriptor,
    launch_nonce: &str,
    identity: &EvaluationWorkerIdentity,
) -> Result<(), EvaluationProviderError> {
    if identity.provider_id != descriptor.provider_id
        || identity.distribution_id != distribution.distribution_id
        || identity.package != distribution.package
        || identity.package_version != distribution.package_version
        || identity.provider_source_sha256 != distribution.provider_source_sha256
        || identity.worker_source_sha256 != distribution.worker_source_sha256
        || identity.dependency_lock_sha256 != distribution.dependency_lock_sha256
        || identity.oci_digest != distribution.oci_digest
        || identity.launch_nonce != launch_nonce
    {
        return Err(EvaluationProviderError::FactoryMismatch(
            "worker hello identity did not match factory launch evidence".to_string(),
        ));
    }
    Ok(())
}

fn validate_asset_binding(
    plan: &EvaluationPlan,
    assets: &[ResolvedEvaluationAsset],
) -> Result<(), EvaluationProviderError> {
    let expected = plan
        .assets
        .iter()
        .map(|asset| (asset.asset_id.as_str(), asset))
        .collect::<BTreeMap<_, _>>();
    let actual = assets
        .iter()
        .map(|asset| (asset.asset_id.as_str(), asset))
        .collect::<BTreeMap<_, _>>();
    if expected.len() != plan.assets.len()
        || actual.len() != assets.len()
        || expected.len() != actual.len()
    {
        return Err(EvaluationProviderError::Protocol(
            "asset binding contained duplicate, missing, or extra identities".to_string(),
        ));
    }
    for (id, requirement) in expected {
        let resolved = actual.get(id).ok_or_else(|| {
            EvaluationProviderError::Protocol(format!("missing resolved asset {id}"))
        })?;
        if resolved.content_sha256 != requirement.content_sha256
            || resolved.immutable_revision != requirement.immutable_revision
            || resolved.media_type != requirement.media_type
            || !resolved.contained_path.starts_with('/')
            || resolved.contained_path.contains("..")
        {
            return Err(EvaluationProviderError::Protocol(format!(
                "resolved asset {id} did not match immutable requirement/containment"
            )));
        }
    }
    Ok(())
}

fn placeholder_identity(
    descriptor: &EvaluationProviderDescriptor,
    distribution: &EvaluationDistributionDescriptor,
    launch_nonce: &str,
) -> EvaluationWorkerIdentity {
    EvaluationWorkerIdentity {
        evaluator_protocol: 0,
        provider_id: descriptor.provider_id.clone(),
        distribution_id: distribution.distribution_id.clone(),
        package: String::new(),
        package_version: String::new(),
        provider_source_sha256: String::new(),
        worker_source_sha256: String::new(),
        dependency_lock_sha256: String::new(),
        python_version: String::new(),
        launch_nonce: launch_nonce.to_string(),
        oci_digest: None,
        operations: Vec::new(),
    }
}

unsafe fn duplicate_control_fd(source: i32, target: i32) -> std::io::Result<()> {
    // SAFETY: caller runs between fork/exec with valid inherited descriptors.
    if source != target && unsafe { libc::dup2(source, target) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `target` is the newly duplicated fixed control descriptor.
    if unsafe { libc::fcntl(target, libc::F_SETFD, 0) } < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

fn create_control_pipe() -> Result<(File, File), EvaluationProviderError> {
    let mut descriptors = [0_i32; 2];
    // SAFETY: `descriptors` has room for exactly the two descriptors returned
    // by `pipe2`; CLOEXEC prevents accidental inheritance before fixed-FD dup.
    if unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) } != 0 {
        return Err(EvaluationProviderError::Io(format!(
            "creating dedicated control pipe: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: successful `pipe2` returned two independently owned descriptors.
    let read = unsafe { File::from_raw_fd(descriptors[0]) };
    // SAFETY: successful `pipe2` returned two independently owned descriptors.
    let write = unsafe { File::from_raw_fd(descriptors[1]) };
    Ok((read, write))
}

unsafe fn set_resource_limit(
    resource: libc::__rlimit_resource_t,
    value: u64,
) -> std::io::Result<()> {
    let mut current = std::mem::MaybeUninit::<libc::rlimit>::uninit();
    // SAFETY: `current` is initialized by a successful `getrlimit` call.
    if unsafe { libc::getrlimit(resource, current.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    // SAFETY: `getrlimit` succeeded.
    let current = unsafe { current.assume_init() };
    let limit = libc::rlimit {
        rlim_cur: (value as libc::rlim_t).min(current.rlim_max),
        // Never try to raise the inherited hard ceiling; an unprivileged
        // worker launch must remain valid inside a stricter parent sandbox.
        rlim_max: current.rlim_max,
    };
    // SAFETY: `resource` is a fixed RLIMIT constant and `limit` lives through the syscall.
    if unsafe { libc::setrlimit(resource, &limit) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    use super::*;
    use crate::canonical::sha256_hex;
    use crate::isolation::{
        EvaluatorIsolationEvidence, EvaluatorResourceLimits, IsolationQuiescenceProof,
        LaunchClosureFile, PreparedEvaluatorLaunch, Sha256LaunchAttestor,
    };
    use crate::provider::{
        EvaluationOperationDescriptor, EvaluationProviderFactory, EvaluatorIsolationRequirements,
        EvaluatorProcessRootBinder, NemoEvaluatorProviderFactory, ProviderRegistryError,
    };
    use crate::provider_protocol::{
        EvaluationDistributionId, EvaluationExecutionGranularity, EvaluationProviderId,
        EvaluationSchedulingMode, EvaluationSessionId, LogicalServiceId, OperationPurpose,
        ScopedProxyBinding, ScopedProxyGrant, ScopedProxySecret, SemanticOperationId,
    };

    #[derive(Default)]
    struct CapturingRestrictedLogSink {
        classifications: Mutex<Vec<String>>,
    }

    impl EvaluationProviderLogSink for CapturingRestrictedLogSink {
        fn log_line(&self, classification: &str) {
            self.classifications
                .lock()
                .unwrap()
                .push(classification.to_string());
        }
    }

    #[derive(Debug, Default)]
    struct CapturingProcessRootBinder {
        root_pid: Mutex<Option<u32>>,
    }

    impl EvaluatorProcessRootBinder for CapturingProcessRootBinder {
        fn bind_attested_root(&self, root_pid: u32) -> Result<(), ProviderRegistryError> {
            let mut captured = self.root_pid.lock().unwrap();
            if captured.replace(root_pid).is_some() {
                return Err(ProviderRegistryError::InvalidLaunch(
                    "fixture process root was bound twice".to_string(),
                ));
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn bounded_reader_rejects_oversized_line_before_newline() {
        let (mut writer, reader) = tokio::io::duplex(64);
        tokio::spawn(async move {
            writer.write_all(b"123456789\n").await.unwrap();
        });
        let mut reader = BufReader::new(reader);
        let error = read_bounded_line(&mut reader, 8).await.unwrap_err();
        assert!(matches!(error, EvaluationProviderError::Protocol(_)));
    }

    #[tokio::test]
    async fn bounded_reader_rejects_eof_and_empty_line() {
        let (mut writer, reader) = tokio::io::duplex(64);
        writer.write_all(b"\n").await.unwrap();
        let mut reader = BufReader::new(reader);
        assert!(matches!(
            read_bounded_line(&mut reader, 8).await.unwrap_err(),
            EvaluationProviderError::Protocol(_)
        ));
    }

    async fn assert_stderr_payload_is_classified_once(payload: Vec<u8>) {
        let (mut writer, reader) = tokio::io::duplex(16 * 1024);
        let sink = Arc::new(CapturingRestrictedLogSink::default());
        let diagnostics = RestrictedProviderDiagnostics::new(sink.clone());
        let drain = tokio::spawn(drain_provider_stderr(reader, diagnostics));
        let write = tokio::spawn(async move {
            writer.write_all(&payload).await.unwrap();
            writer.shutdown().await.unwrap();
        });
        write.await.unwrap();
        drain.await.unwrap();
        let classifications = sink.classifications.lock().unwrap();
        assert_eq!(classifications.as_slice(), [STDERR_RESTRICTED_OUTPUT]);
    }

    #[tokio::test]
    async fn stderr_drain_discards_multi_mib_newline_free_invalid_utf8() {
        assert_stderr_payload_is_classified_once(vec![0xff; 3 * 1024 * 1024]).await;
    }

    #[tokio::test]
    async fn stderr_drain_classifies_many_tiny_lines_only_once() {
        assert_stderr_payload_is_classified_once(b"worker-secret\n".repeat(200_000)).await;
    }

    struct FixtureIsolation;

    impl EvaluatorIsolation for FixtureIsolation {
        fn check_available(&self) -> Result<(), EvaluationProviderError> {
            Ok(())
        }

        fn prepare(
            &self,
            launch: &AttestedWorkerLaunch,
            _attestation: &crate::isolation::LaunchAttestation,
            context: &ProviderLaunchContext,
        ) -> Result<PreparedEvaluatorLaunch, EvaluationProviderError> {
            Ok(PreparedEvaluatorLaunch {
                program: launch.program.clone(),
                args: launch.args.clone(),
                environment: launch.environment.clone(),
                current_dir: launch.current_dir.clone(),
                resource_limits: EvaluatorResourceLimits::default(),
                evidence: EvaluatorIsolationEvidence {
                    profile_id: "fixture-isolation-v1".to_string(),
                    proof_sha256: context
                        .binding_sha256()
                        .map_err(EvaluationProviderError::registry)?,
                    enforced: EvaluatorIsolationRequirements::strict_process_tree(),
                },
            })
        }

        fn verify_quiescent(
            &self,
            root_pid: u32,
        ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
            if Path::new(&format!("/proc/{root_pid}")).exists() {
                return Err(EvaluationProviderError::Quiescence(
                    "fixture worker remained live".to_string(),
                ));
            }
            Ok(IsolationQuiescenceProof::verified(
                root_pid,
                sha256_hex(format!("fixture-quiescent:{root_pid}").as_bytes()),
            ))
        }
    }

    struct CountingIsolation {
        verification_count: Arc<AtomicUsize>,
        fail_verification: bool,
    }

    impl CountingIsolation {
        fn new(fail_verification: bool) -> (Arc<Self>, Arc<AtomicUsize>) {
            let verification_count = Arc::new(AtomicUsize::new(0));
            (
                Arc::new(Self {
                    verification_count: Arc::clone(&verification_count),
                    fail_verification,
                }),
                verification_count,
            )
        }
    }

    impl EvaluatorIsolation for CountingIsolation {
        fn check_available(&self) -> Result<(), EvaluationProviderError> {
            Ok(())
        }

        fn prepare(
            &self,
            _launch: &AttestedWorkerLaunch,
            _attestation: &crate::isolation::LaunchAttestation,
            _context: &ProviderLaunchContext,
        ) -> Result<PreparedEvaluatorLaunch, EvaluationProviderError> {
            Err(EvaluationProviderError::Launch(
                "counting isolation cannot prepare launches".to_string(),
            ))
        }

        fn verify_quiescent(
            &self,
            root_pid: u32,
        ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
            self.verification_count.fetch_add(1, Ordering::SeqCst);
            if self.fail_verification {
                return Err(EvaluationProviderError::Quiescence(
                    "fixture subtree was not quiescent".to_string(),
                ));
            }
            if Path::new(&format!("/proc/{root_pid}")).exists() {
                return Err(EvaluationProviderError::Quiescence(
                    "fixture root remained live".to_string(),
                ));
            }
            Ok(IsolationQuiescenceProof::verified(
                root_pid,
                sha256_hex(format!("counting-quiescent:{root_pid}").as_bytes()),
            ))
        }
    }

    async fn spawn_process_for_wait_test(
        script: &str,
        isolation: Arc<dyn EvaluatorIsolation>,
        sink: Arc<dyn EvaluationProviderLogSink>,
        shutdown_timeout: Duration,
    ) -> SupervisedProcess {
        let mut child = Command::new(find_python())
            .args(["-u", "-c", script])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let root_pid = child.id().unwrap();
        let diagnostics = RestrictedProviderDiagnostics::new(sink);
        let stderr = child.stderr.take().unwrap();
        let stderr_task = Some(tokio::spawn(drain_provider_stderr(
            stderr,
            diagnostics.clone(),
        )));
        SupervisedProcess {
            child,
            root_pid,
            stderr_task,
            diagnostics,
            isolation,
            shutdown_timeout,
            quiescence_result: None,
        }
    }

    #[tokio::test]
    async fn graceful_wait_timeout_kills_reaps_and_invokes_quiescence_verifier() {
        let (isolation, count) = CountingIsolation::new(false);
        let sink = Arc::new(CapturingRestrictedLogSink::default());
        let mut process = spawn_process_for_wait_test(
            "import time; time.sleep(60)",
            isolation,
            sink,
            Duration::from_millis(50),
        )
        .await;
        let error = process.wait_quiescent().await.unwrap_err();
        assert!(matches!(error, EvaluationProviderError::Crashed(_)));
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert!(!Path::new(&format!("/proc/{}", process.root_pid)).exists());
    }

    #[tokio::test]
    async fn nonzero_exit_invokes_verifier_and_prioritizes_quiescence_failure() {
        let (isolation, count) = CountingIsolation::new(true);
        let sink = Arc::new(CapturingRestrictedLogSink::default());
        let mut process = spawn_process_for_wait_test(
            "raise SystemExit(17)",
            isolation,
            sink,
            Duration::from_secs(1),
        )
        .await;
        let error = process.wait_quiescent().await.unwrap_err();
        assert_eq!(
            error,
            EvaluationProviderError::Quiescence("fixture subtree was not quiescent".to_string())
        );
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    fn find_python() -> PathBuf {
        let candidate = std::env::var_os("PYTHON").unwrap_or_else(|| "python".into());
        let candidate = PathBuf::from(candidate);
        if candidate.is_absolute() {
            return std::fs::canonicalize(candidate).unwrap();
        }
        std::env::var_os("PATH")
            .and_then(|path| {
                std::env::split_paths(&path)
                    .map(|directory| directory.join(&candidate))
                    .find(|path| path.is_file())
            })
            .and_then(|path| std::fs::canonicalize(path).ok())
            .expect("Python fixture executable must be on PATH")
    }

    #[tokio::test]
    async fn prepared_token_binds_context_and_must_match_host_isolation_proof() {
        let source_python = find_python();
        let base =
            std::env::temp_dir().join(format!("aiperf-provider-token-test-{}", std::process::id()));
        let worker_root = base.join("rootfs");
        let python = worker_root.join("bin/python");
        let staging = base.join("staging");
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::copy(source_python, &python).unwrap();
        let executable_sha256 = sha256_hex(&std::fs::read(&python).unwrap());
        let logical_program = python.strip_prefix(&worker_root).unwrap();
        let mut closure_bytes = Vec::new();
        closure_bytes.extend_from_slice(logical_program.to_string_lossy().as_bytes());
        closure_bytes.push(0);
        closure_bytes.extend_from_slice(executable_sha256.as_bytes());
        closure_bytes.push(b'\n');
        let distribution = EvaluationDistributionDescriptor {
            distribution_id: EvaluationDistributionId::new("fixture-prepared").unwrap(),
            package: "fixture-provider".to_string(),
            package_version: "1.0".to_string(),
            provider_source_sha256: "a".repeat(64),
            worker_source_sha256: "b".repeat(64),
            dependency_lock_sha256: "c".repeat(64),
            oci_digest: None,
            launch_closure_sha256: sha256_hex(&closure_bytes),
        };
        let launch = AttestedWorkerLaunch {
            distribution_id: distribution.distribution_id.clone(),
            program: python.clone(),
            args: Vec::new(),
            environment: BTreeMap::new(),
            current_dir: worker_root.clone(),
            worker_root,
            closure: vec![LaunchClosureFile {
                path: python,
                artifact_content_sha256: executable_sha256,
            }],
        };
        let launcher = Arc::new(
            SupervisedEvaluationProviderLauncher::new(
                vec![launch],
                Arc::new(Sha256LaunchAttestor),
                Arc::new(FixtureIsolation),
            )
            .unwrap(),
        );
        let factory =
            NemoEvaluatorProviderFactory::new(vec![distribution.clone()], launcher).unwrap();
        let context = ProviderLaunchContext {
            session_id: EvaluationSessionId::new("fixture-token-session").unwrap(),
            staging_dir: staging.clone(),
            proxy: None,
            process_root_binder: None,
            protocol_limits: EvaluatorProtocolLimits::default(),
            launch_nonce: "prepared-launch-nonce-0123456789abcdef".to_string(),
        };
        let expected_context_sha256 = context.binding_sha256().unwrap();
        let prepared = factory
            .prepare_launch(&distribution.distribution_id, context)
            .unwrap();
        assert_eq!(prepared.prelaunch_context_sha256(), expected_context_sha256);
        assert_eq!(
            prepared.isolation_evidence().proof_sha256,
            expected_context_sha256
        );
        let config = factory
            .validate_authored_config(
                &CanonicalJson::new(serde_json::json!({
                    "environment": "gsm8k",
                    "solver": "chat",
                    "solver_config": {"max_tokens": 64},
                    "selection": {"limit": 1, "seed": 0}
                }))
                .unwrap(),
            )
            .unwrap();
        let host_binding = crate::provider_protocol::EvaluationHostBinding {
            host: crate::provider_protocol::EvaluationHostIdentity {
                runner_sha256: "6".repeat(64),
                capability_inventory_sha256: "7".repeat(64),
                schema_inventory_sha256: "8".repeat(64),
                isolation_proof_sha256: "9".repeat(64),
            },
            route_map_sha256: "a".repeat(64),
            prepared_endpoints_sha256: "b".repeat(64),
            sandbox_sha256: None,
        };
        let error = match factory.launch(prepared, &config, host_binding).await {
            Ok(_) => panic!("mismatched host isolation proof launched a worker"),
            Err(error) => error,
        };
        assert!(matches!(error, EvaluationProviderError::FactoryMismatch(_)));
        let _ = std::fs::remove_dir_all(base);
    }

    fn fixture_descriptor() -> (
        EvaluationProviderDescriptor,
        EvaluationDistributionDescriptor,
    ) {
        let distribution = EvaluationDistributionDescriptor {
            distribution_id: EvaluationDistributionId::new("fixture-dist").unwrap(),
            package: "fixture-provider".to_string(),
            package_version: "1.0".to_string(),
            provider_source_sha256: "a".repeat(64),
            worker_source_sha256: "b".repeat(64),
            dependency_lock_sha256: "c".repeat(64),
            oci_digest: None,
            launch_closure_sha256: "d".repeat(64),
        };
        let descriptor = EvaluationProviderDescriptor {
            provider_id: EvaluationProviderId::new("fixture").unwrap(),
            display_name: "Fixture".to_string(),
            worker_protocol_versions: vec![2],
            execution_granularities: vec![EvaluationExecutionGranularity::Case],
            scheduling_modes: vec![EvaluationSchedulingMode::Finite],
            operations: vec![EvaluationOperationDescriptor {
                operation_id: SemanticOperationId::new("model.generate").unwrap(),
                input_schema_sha256: "1".repeat(64),
                output_schema_sha256: "2".repeat(64),
                stream_schema_sha256: Some("3".repeat(64)),
                reports_usage: true,
                modalities: vec!["text".to_string()],
                endpoint_capabilities: vec!["chat".to_string()],
            }],
            isolation: EvaluatorIsolationRequirements::strict_process_tree(),
            config_schema_version: 1,
            config_schema_sha256: "4".repeat(64),
            public_projection_schemas: BTreeMap::new(),
            distributions: vec![distribution.clone()],
        };
        (descriptor, distribution)
    }

    async fn launch_fixture_script(
        label: &str,
        script: &str,
        control_timeout: Duration,
    ) -> (
        Result<SupervisedEvaluationProvider, EvaluationProviderError>,
        Arc<AtomicUsize>,
        PathBuf,
    ) {
        let python = find_python();
        let (descriptor, distribution) = fixture_descriptor();
        let base =
            std::env::temp_dir().join(format!("aiperf-provider-{label}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(&base).unwrap();
        let context = ProviderLaunchContext {
            session_id: EvaluationSessionId::new(format!("fixture-{label}-session")).unwrap(),
            staging_dir: base.clone(),
            proxy: None,
            process_root_binder: None,
            protocol_limits: EvaluatorProtocolLimits::default(),
            launch_nonce: format!("fixture-{label}-nonce-0123456789abcdef"),
        };
        let host_binding = crate::provider_protocol::EvaluationHostBinding {
            host: crate::provider_protocol::EvaluationHostIdentity {
                runner_sha256: "6".repeat(64),
                capability_inventory_sha256: "7".repeat(64),
                schema_inventory_sha256: "8".repeat(64),
                isolation_proof_sha256: "9".repeat(64),
            },
            route_map_sha256: "a".repeat(64),
            prepared_endpoints_sha256: "b".repeat(64),
            sandbox_sha256: None,
        };
        let prepared = PreparedEvaluatorLaunch {
            program: python,
            args: vec!["-u".into(), "-c".into(), script.into()],
            environment: BTreeMap::new(),
            current_dir: base.clone(),
            resource_limits: EvaluatorResourceLimits::default(),
            evidence: EvaluatorIsolationEvidence {
                profile_id: "test-process-tree".to_string(),
                proof_sha256: "5".repeat(64),
                enforced: EvaluatorIsolationRequirements::strict_process_tree(),
            },
        };
        let (isolation, verification_count) = CountingIsolation::new(false);
        let result = SupervisedEvaluationProvider::spawn(
            &descriptor,
            &distribution,
            SupervisedSpawnInput {
                prepared,
                context,
                host_binding,
            },
            isolation,
            Arc::new(CapturingRestrictedLogSink::default()),
            Duration::from_secs(1),
            control_timeout,
        )
        .await;
        (result, verification_count, base)
    }

    #[tokio::test]
    async fn hanging_hello_is_bounded_and_fully_quiesced() {
        let launch = launch_fixture_script(
            "hanging-hello",
            "import time; time.sleep(60)",
            Duration::from_millis(50),
        );
        let error = match tokio::time::timeout(Duration::from_secs(5), launch).await {
            Ok((Err(error), verification_count, base)) => {
                assert_eq!(verification_count.load(Ordering::SeqCst), 1);
                let _ = std::fs::remove_dir_all(base);
                error
            }
            Ok((Ok(_), _, _)) => panic!("hanging worker unexpectedly negotiated"),
            Err(_) => panic!("supervisor failed to bound the hanging hello"),
        };
        assert!(matches!(error, EvaluationProviderError::Crashed(_)));
        assert!(error.to_string().contains("hello"));
    }

    #[tokio::test]
    async fn malformed_immediate_reply_is_terminated_and_verified_once() {
        const SCRIPT: &str = r#"
import os, time
reader = os.fdopen(3, 'r', encoding='utf-8', closefd=False)
writer = os.fdopen(4, 'w', encoding='utf-8', closefd=False)
reader.readline()
writer.write('{not-json}\n')
writer.flush()
time.sleep(60)
"#;
        let (result, verification_count, base) =
            launch_fixture_script("malformed-reply", SCRIPT, Duration::from_secs(1)).await;
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("malformed worker unexpectedly negotiated"),
        };
        assert!(matches!(error, EvaluationProviderError::Protocol(_)));
        assert_eq!(verification_count.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn semantic_plan_error_is_terminated_and_verified_once() {
        const SCRIPT: &str = r#"
import json, os, sys, time
reader = os.fdopen(3, 'r', encoding='utf-8', closefd=False)
writer = os.fdopen(4, 'w', encoding='utf-8', closefd=False)
ops = ['plan_session','bind_assets','next_units','instantiate_units','start_units','poll_events','submit_host_events','cancel_units','finalize_session','shutdown']
for line in reader:
    request = json.loads(line)
    if request['op'] == 'hello':
        result = {
            'evaluator_protocol': 2,
            'provider_id': 'fixture',
            'distribution_id': 'fixture-dist',
            'package': 'fixture-provider',
            'package_version': '1.0',
            'provider_source_sha256': 'a' * 64,
            'worker_source_sha256': 'b' * 64,
            'dependency_lock_sha256': 'c' * 64,
            'python_version': sys.version.split()[0],
            'launch_nonce': request['launch_nonce'],
            'operations': ops,
        }
    elif request['op'] == 'plan_session':
        result = {
            'assets': [],
            'host_requirements': [],
            'logical_services': [],
            'aggregation_policy': {
                'policy_id': 'fixture',
                'exclude_infrastructure': True,
                'exclude_cancelled': True,
                'definition': {},
            },
            'execution_granularity': 'case',
            'scheduling_mode': 'finite',
            'finite_unit_count': 1,
            'finite_case_count': 1,
            'queue_credits': {
                'units': 0,
                'host_operations': 0,
                'host_operations_per_unit': 0,
                'stream_events': 0,
                'sandboxes': 0,
                'processes': 0,
                'artifacts': 0,
                'artifact_bytes': 0,
            },
        }
    else:
        raise RuntimeError(request['op'])
    writer.write(json.dumps({'id': request['id'], 'ok': True, 'result': result}) + '\n')
    writer.flush()
    if request['op'] == 'plan_session':
        time.sleep(60)
"#;
        let (result, verification_count, base) =
            launch_fixture_script("semantic-plan", SCRIPT, Duration::from_secs(1)).await;
        let mut provider = result.unwrap();
        let request = EvaluationPlanRequest {
            session_id: EvaluationSessionId::new("semantic-plan-session").unwrap(),
            provider_id: EvaluationProviderId::new("fixture").unwrap(),
            distribution_id: EvaluationDistributionId::new("fixture-dist").unwrap(),
            config_schema_version: 1,
            config_schema_sha256: "4".repeat(64),
            provider_config: CanonicalJson::new(serde_json::json!({})).unwrap(),
            reproducible: true,
        };
        let error = provider.plan(&request).await.unwrap_err();
        assert!(matches!(error, EvaluationProviderError::Protocol(_)));
        assert_eq!(verification_count.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn abort_from_negotiated_state_is_idempotent_and_verified_once() {
        const SCRIPT: &str = r#"
import json, os, sys, time
reader = os.fdopen(3, 'r', encoding='utf-8', closefd=False)
writer = os.fdopen(4, 'w', encoding='utf-8', closefd=False)
request = json.loads(reader.readline())
result = {
    'evaluator_protocol': 2,
    'provider_id': 'fixture',
    'distribution_id': 'fixture-dist',
    'package': 'fixture-provider',
    'package_version': '1.0',
    'provider_source_sha256': 'a' * 64,
    'worker_source_sha256': 'b' * 64,
    'dependency_lock_sha256': 'c' * 64,
    'python_version': sys.version.split()[0],
    'launch_nonce': request['launch_nonce'],
    'operations': ['plan_session','bind_assets','next_units','instantiate_units','start_units','poll_events','submit_host_events','cancel_units','finalize_session','shutdown'],
}
writer.write(json.dumps({'id': request['id'], 'ok': True, 'result': result}) + '\n')
writer.flush()
time.sleep(60)
"#;
        let (result, verification_count, base) =
            launch_fixture_script("abort-negotiated", SCRIPT, Duration::from_secs(1)).await;
        let mut provider = result.unwrap();
        provider.abort().await.unwrap();
        provider.abort().await.unwrap();
        assert_eq!(
            provider.lifecycle_state(),
            EvaluationLifecycleState::WorkerExited
        );
        assert!(provider.quiescence_proof().is_some());
        assert_eq!(verification_count.load(Ordering::SeqCst), 1);
        let _ = std::fs::remove_dir_all(base);
    }

    #[tokio::test]
    async fn real_child_uses_dedicated_fds_not_stdin_or_stdout() {
        const SCRIPT: &str = r#"
import json, os, sys
print('EXPECTED_ANSWER_SENTINEL Bearer provider-secret', file=sys.stderr, flush=True)
reader = os.fdopen(3, 'r', encoding='utf-8', closefd=False)
writer = os.fdopen(4, 'w', encoding='utf-8', closefd=False)
print('ordinary stdout is not protocol', flush=True)
ops = ['plan_session','bind_assets','next_units','instantiate_units','start_units','poll_events','submit_host_events','cancel_units','finalize_session','shutdown']
for line in reader:
    request = json.loads(line)
    if request['op'] == 'hello':
        result = {
            'evaluator_protocol': 2,
            'provider_id': 'fixture',
            'distribution_id': 'fixture-dist',
            'package': 'fixture-provider',
            'package_version': '1.0',
            'provider_source_sha256': 'a' * 64,
            'worker_source_sha256': 'b' * 64,
            'dependency_lock_sha256': 'c' * 64,
            'python_version': sys.version.split()[0],
            'launch_nonce': request['launch_nonce'],
            'operations': ops,
        }
    elif request['op'] == 'shutdown':
        result = {'shutdown': True}
    else:
        raise RuntimeError(request['op'])
    writer.write(json.dumps({'id': request['id'], 'ok': True, 'result': result}) + '\n')
    writer.flush()
    if request['op'] == 'shutdown':
        break
"#;
        let python = find_python();
        let (descriptor, distribution) = fixture_descriptor();
        let base =
            std::env::temp_dir().join(format!("aiperf-provider-fd-test-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let process_root_binder = Arc::new(CapturingProcessRootBinder::default());
        let session_id = EvaluationSessionId::new("fixture-session").unwrap();
        let context = ProviderLaunchContext {
            session_id: session_id.clone(),
            staging_dir: base.clone(),
            proxy: Some(ScopedProxyBinding {
                local_locator: "unix:///run/aiperf/evaluator-proxy.sock".to_string(),
                host_socket_path: base.join("evaluator-proxy.sock"),
                grant: ScopedProxyGrant {
                    grant_id: "fixture-grant".to_string(),
                    session_id,
                    secret: ScopedProxySecret::new("s".repeat(48)).unwrap(),
                    service_ids: vec![LogicalServiceId::new("candidate").unwrap()],
                    semantic_operation_ids: vec![
                        SemanticOperationId::new("model.generate").unwrap(),
                    ],
                    purposes: vec![OperationPurpose::new("primary").unwrap()],
                    process_scope_sha256: "f".repeat(64),
                    max_operations: 1,
                    max_concurrent_operations: 1,
                    max_request_bytes: 1024,
                    max_response_bytes: 1024,
                    max_stream_events: 4,
                    expires_after_ms: 1000,
                },
            }),
            process_root_binder: Some(process_root_binder.clone()),
            protocol_limits: EvaluatorProtocolLimits::default(),
            launch_nonce: "launch-nonce-fixture-0123456789abcdef".to_string(),
        };
        let host_binding = crate::provider_protocol::EvaluationHostBinding {
            host: crate::provider_protocol::EvaluationHostIdentity {
                runner_sha256: "6".repeat(64),
                capability_inventory_sha256: "7".repeat(64),
                schema_inventory_sha256: "8".repeat(64),
                isolation_proof_sha256: "9".repeat(64),
            },
            route_map_sha256: "a".repeat(64),
            prepared_endpoints_sha256: "b".repeat(64),
            sandbox_sha256: None,
        };
        let prepared = PreparedEvaluatorLaunch {
            program: python,
            args: vec!["-u".into(), "-c".into(), SCRIPT.into()],
            environment: BTreeMap::new(),
            current_dir: base.clone(),
            resource_limits: EvaluatorResourceLimits::default(),
            evidence: EvaluatorIsolationEvidence {
                profile_id: "test-process-tree".to_string(),
                proof_sha256: "5".repeat(64),
                enforced: EvaluatorIsolationRequirements::strict_process_tree(),
            },
        };
        let log_sink = Arc::new(CapturingRestrictedLogSink::default());
        let mut provider = SupervisedEvaluationProvider::spawn(
            &descriptor,
            &distribution,
            SupervisedSpawnInput {
                prepared,
                context,
                host_binding,
            },
            Arc::new(FixtureIsolation),
            log_sink.clone(),
            Duration::from_secs(5),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
        assert_eq!(
            *process_root_binder.root_pid.lock().unwrap(),
            Some(provider.process.root_pid)
        );
        assert_eq!(provider.identity().provider_id.as_str(), "fixture");
        provider.lifecycle.abort_to_quiescing();
        let id = provider.take_request_id().unwrap();
        let result: ProviderShutdownResult = provider
            .request(EvaluatorWorkerRequest::Shutdown { id })
            .await
            .unwrap();
        assert!(result.shutdown);
        provider.writer.shutdown().await.unwrap();
        let proof = provider.process.wait_quiescent().await.unwrap();
        provider.lifecycle.worker_exited().unwrap();
        assert!(proof.proof_sha256().len() == 64);
        let classifications = log_sink.classifications.lock().unwrap();
        assert_eq!(classifications.as_slice(), [STDERR_RESTRICTED_OUTPUT]);
        assert!(
            !classifications
                .iter()
                .any(|line| line.contains("EXPECTED_ANSWER_SENTINEL")
                    || line.contains("provider-secret"))
        );
        let _ = std::fs::remove_dir_all(base);
    }
}
