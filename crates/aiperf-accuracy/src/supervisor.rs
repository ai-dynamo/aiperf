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
use std::time::Duration;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter};
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
    ProviderLaunchContext, ValidatedProviderConfig,
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

trait DynAsyncRead: AsyncBufRead + Unpin {}
impl<T: AsyncBufRead + Unpin> DynAsyncRead for T {}

trait DynAsyncWrite: AsyncWrite + Unpin {}
impl<T: AsyncWrite + Unpin> DynAsyncWrite for T {}

/// Sink for one complete evaluator-provider stderr diagnostic line.
pub trait EvaluationProviderLogSink: Send + Sync {
    /// Consume one already redacted worker diagnostic line.
    fn log_line(&self, line: &str);
}

/// Default provider stderr sink.
#[derive(Debug, Clone, Copy, Default)]
pub struct StderrEvaluationProviderLogSink;

impl EvaluationProviderLogSink for StderrEvaluationProviderLogSink {
    fn log_line(&self, line: &str) {
        eprintln!(
            "[evaluation-provider] {}",
            crate::canonical::redact_diagnostic(line)
        );
    }
}

/// Concrete factory launcher using attestation, isolation, inherited pipes, and strict framing.
pub struct SupervisedEvaluationProviderLauncher {
    launches: BTreeMap<crate::provider_protocol::EvaluationDistributionId, AttestedWorkerLaunch>,
    attestor: Arc<dyn EvaluatorLaunchAttestor>,
    isolation: Arc<dyn EvaluatorIsolation>,
    log_sink: Arc<dyn EvaluationProviderLogSink>,
    shutdown_timeout: Duration,
}

impl fmt::Debug for SupervisedEvaluationProviderLauncher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SupervisedEvaluationProviderLauncher")
            .field("distributions", &self.launches.keys().collect::<Vec<_>>())
            .field("shutdown_timeout", &self.shutdown_timeout)
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
}

#[async_trait(?Send)]
impl EvaluationProviderLauncher for SupervisedEvaluationProviderLauncher {
    async fn launch(
        &self,
        descriptor: &EvaluationProviderDescriptor,
        distribution: &EvaluationDistributionDescriptor,
        config: &ValidatedProviderConfig,
        context: &ProviderLaunchContext,
    ) -> Result<Box<dyn EvaluationProvider>, EvaluationProviderError> {
        if config.provider_id() != &descriptor.provider_id
            || config.schema_sha256() != descriptor.config_schema_sha256
            || config.schema_version() != descriptor.config_schema_version
        {
            return Err(EvaluationProviderError::FactoryMismatch(
                "launcher received config from a different provider/schema".to_string(),
            ));
        }
        let launch = self
            .launches
            .get(&distribution.distribution_id)
            .ok_or_else(|| {
                EvaluationProviderError::FactoryMismatch(format!(
                    "no factory-owned launch recipe for distribution {}",
                    distribution.distribution_id
                ))
            })?;
        let attestation = self.attestor.attest(launch, distribution)?;
        let prepared = self.isolation.prepare(launch, &attestation, context)?;
        prepared.evidence.validate_strict()?;
        let provider = SupervisedEvaluationProvider::spawn(
            descriptor,
            distribution,
            prepared,
            context,
            Arc::clone(&self.isolation),
            Arc::clone(&self.log_sink),
            self.shutdown_timeout,
        )
        .await?;
        Ok(Box::new(provider))
    }
}

struct SupervisedProcess {
    child: Child,
    root_pid: u32,
    stderr_task: Option<JoinHandle<()>>,
    isolation: Arc<dyn EvaluatorIsolation>,
    shutdown_timeout: Duration,
}

impl SupervisedProcess {
    async fn wait_quiescent(
        &mut self,
    ) -> Result<IsolationQuiescenceProof, EvaluationProviderError> {
        let waited = tokio::time::timeout(self.shutdown_timeout, self.child.wait()).await;
        let status = match waited {
            Ok(result) => result.map_err(|error| EvaluationProviderError::Io(error.to_string()))?,
            Err(_) => {
                self.child
                    .start_kill()
                    .map_err(|error| EvaluationProviderError::Io(error.to_string()))?;
                let _ = self.child.wait().await;
                return Err(EvaluationProviderError::Quiescence(
                    "evaluator worker exceeded graceful shutdown timeout and was force-killed"
                        .to_string(),
                ));
            }
        };
        if !status.success() {
            return Err(EvaluationProviderError::Crashed(status.to_string()));
        }
        if let Some(task) = self.stderr_task.take() {
            let _ = task.await;
        }
        self.isolation.verify_quiescent(self.root_pid)
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
        prepared: crate::isolation::PreparedEvaluatorLaunch,
        context: &ProviderLaunchContext,
        isolation: Arc<dyn EvaluatorIsolation>,
        log_sink: Arc<dyn EvaluationProviderLogSink>,
        shutdown_timeout: Duration,
    ) -> Result<Self, EvaluationProviderError> {
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
        let root_pid = child.id().ok_or_else(|| {
            EvaluationProviderError::Launch("spawned evaluator worker had no PID".to_string())
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            EvaluationProviderError::Launch("evaluator worker had no stderr pipe".to_string())
        })?;
        let stderr_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => log_sink.log_line(&line),
                    Ok(None) => break,
                    Err(error) => {
                        log_sink.log_line(&format!("failed to drain provider stderr: {error}"));
                        break;
                    }
                }
            }
        });
        let request_stream = tokio::fs::File::from_std(request_parent);
        let response_stream = tokio::fs::File::from_std(response_parent);
        let reader: Box<dyn DynAsyncRead> = Box::new(BufReader::new(response_stream));
        let writer: BufWriter<Box<dyn DynAsyncWrite>> = BufWriter::new(Box::new(request_stream));
        let mut provider = Self {
            process: SupervisedProcess {
                child,
                root_pid,
                stderr_task: Some(stderr_task),
                isolation,
                shutdown_timeout,
            },
            reader,
            writer,
            identity: placeholder_identity(descriptor, distribution, &context.launch_nonce),
            lifecycle: EvaluationLifecycle::new(context.protocol_limits)?,
            limits: context.protocol_limits,
            next_request_id: 1,
            proxy: context.proxy.clone(),
            host_binding: context.host_binding.clone(),
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
        validate_negotiated_identity(descriptor, distribution, &context.launch_nonce, &identity)?;
        provider.identity = identity;
        provider.lifecycle.negotiated()?;
        Ok(provider)
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
        self.lifecycle.planned(&plan)?;
        self.plan_request = Some(request.clone());
        self.plan = Some(plan.clone());
        Ok(plan)
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
                "bound evaluation identity drifted from worker/schema/config evidence".to_string(),
            ));
        }
        self.lifecycle.assets_bound_and_ready()?;
        self.frozen_identity = Some(identity.clone());
        Ok(identity)
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
        if result.started != ids {
            return Err(EvaluationProviderError::Protocol(
                "start_units acknowledgement did not exactly match submitted IDs".to_string(),
            ));
        }
        self.lifecycle.start_units(ids)
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
        if batch.events.len() > limit {
            return Err(EvaluationProviderError::Protocol(
                "poll_events worker returned more events than requested".to_string(),
            ));
        }
        self.lifecycle.record_event_batch(&mut batch)?;
        Ok(batch)
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
        if result.accepted != expected {
            return Err(EvaluationProviderError::Protocol(
                "submit_host_events acknowledgement did not exactly match event order".to_string(),
            ));
        }
        Ok(())
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
        if result.cancelled != ids {
            return Err(EvaluationProviderError::Protocol(
                "cancel_units acknowledgement did not exactly match submitted IDs".to_string(),
            ));
        }
        Ok(())
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
    }

    async fn shutdown(&mut self) -> Result<(), EvaluationProviderError> {
        if self.lifecycle.state() == EvaluationLifecycleState::WorkerExited {
            return Ok(());
        }
        self.lifecycle.begin_shutdown()?;
        let id = self.take_request_id()?;
        let result: ProviderShutdownResult = self
            .request(EvaluatorWorkerRequest::Shutdown { id })
            .await?;
        if !result.shutdown {
            return Err(EvaluationProviderError::Protocol(
                "worker did not acknowledge shutdown".to_string(),
            ));
        }
        self.writer
            .shutdown()
            .await
            .map_err(|error| EvaluationProviderError::Io(error.to_string()))?;
        let proof = self.process.wait_quiescent().await?;
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

    use super::*;
    use crate::canonical::sha256_hex;
    use crate::isolation::{
        EvaluatorIsolationEvidence, EvaluatorResourceLimits, IsolationQuiescenceProof,
        PreparedEvaluatorLaunch,
    };
    use crate::provider::{EvaluationOperationDescriptor, EvaluatorIsolationRequirements};
    use crate::provider_protocol::{
        EvaluationDistributionId, EvaluationExecutionGranularity, EvaluationProviderId,
        EvaluationSchedulingMode, EvaluationSessionId, SemanticOperationId,
    };

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

    struct FixtureIsolation;

    impl EvaluatorIsolation for FixtureIsolation {
        fn prepare(
            &self,
            _launch: &AttestedWorkerLaunch,
            _attestation: &crate::isolation::LaunchAttestation,
            _context: &ProviderLaunchContext,
        ) -> Result<PreparedEvaluatorLaunch, EvaluationProviderError> {
            unreachable!("dedicated-FD test supplies an already prepared launch")
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

    #[tokio::test]
    async fn real_child_uses_dedicated_fds_not_stdin_or_stdout() {
        const SCRIPT: &str = r#"
import json, os, sys
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
        let context = ProviderLaunchContext {
            session_id: EvaluationSessionId::new("fixture-session").unwrap(),
            staging_dir: base.clone(),
            proxy: None,
            host_binding: crate::provider_protocol::EvaluationHostBinding {
                host: crate::provider_protocol::EvaluationHostIdentity {
                    runner_sha256: "6".repeat(64),
                    capability_inventory_sha256: "7".repeat(64),
                    schema_inventory_sha256: "8".repeat(64),
                    isolation_proof_sha256: "9".repeat(64),
                },
                route_map_sha256: "a".repeat(64),
                prepared_endpoints_sha256: "b".repeat(64),
                sandbox_sha256: None,
            },
            protocol_limits: EvaluatorProtocolLimits::default(),
            launch_nonce: "launch-nonce-fixture-0123456789abcdef".to_string(),
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
        let mut provider = SupervisedEvaluationProvider::spawn(
            &descriptor,
            &distribution,
            prepared,
            &context,
            Arc::new(FixtureIsolation),
            Arc::new(StderrEvaluationProviderLogSink),
            Duration::from_secs(5),
        )
        .await
        .unwrap();
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
        let _ = std::fs::remove_dir_all(base);
    }
}
