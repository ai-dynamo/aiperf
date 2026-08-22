// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Redacted, injectable Docker command contracts for benchmark execution.

use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
    time::Duration,
};

use std::rc::Rc;

use async_trait::async_trait;

#[cfg(feature = "engine")]
use crate::eval::native_graph::{CompatibilityCaptureSession, ProtocolExternalDriverSession};
use crate::eval::{
    AdapterExit, AdapterProcess, AdapterSpawnRequest, AdapterSpawnTransaction, AdapterSpawner,
    AdapterSupervisionError, CancelReason, ExternallyDrivenAdapterAuthorization, HarborTaskPackage,
    ModelSecretId, NativeGraphAdapterAuthorization, NativeGraphPackagePlan, NativeGraphProfile,
    PreparedExternalDriver, PreparedExternalDriverCapability, ProviderProfile,
    ResolvedEpisodeTrial,
};
#[cfg(feature = "engine")]
use crate::eval::{
    AdapterProtocolConfig, CompatibilityTerminalReceipt, StrictSupervisedAdapter, SupervisedAdapter,
};

use super::{
    BenchmarkExecutionPlan, ComposeServiceName, EnvName, EnvironmentPlan, EvalExecutionError,
    EvalExecutionPhase, PhasePlan, ProviderCapabilities, SecretProvider, SecretValue,
};

/// Preflights Docker enforcement before the executor can build an image.
pub fn preflight_docker(
    runtime: &dyn DockerRuntime,
    plan: &BenchmarkExecutionPlan,
) -> Result<(), EvalExecutionError> {
    plan.validate_for(runtime.capabilities())?;
    if plan.compose().is_some() && runtime.compose_runtime().is_none() {
        return Err(EvalExecutionError::UnsupportedEnforcement(
            "Docker Compose runtime",
        ));
    }
    Ok(())
}

/// Resolves the non-forgeable NativeGraph exact-profile authorization before provisioning.
///
/// Only a NativeGraph exact profile needs this boundary. The selected Docker
/// runtime supplies its actual isolation proof and the concrete host environment
/// names backing each logical model secret; this function validates both against
/// the imported package before any image build or container creation.
pub(crate) fn resolve_native_graph_adapter_authorization(
    runtime: &dyn DockerRuntime,
    package: &HarborTaskPackage,
    plan: &BenchmarkExecutionPlan,
) -> Result<Option<NativeGraphAdapterAuthorization>, EvalExecutionError> {
    let Some(native_graph) = package.native_graph() else {
        return Ok(None);
    };
    if native_graph.profile() != NativeGraphProfile::NativeGraph {
        return Ok(None);
    }
    let profile = runtime.native_graph_provider_profile_for_plan(native_graph, plan)?;
    let secret_environment = runtime.native_graph_model_secret_environment(native_graph)?;
    NativeGraphAdapterAuthorization::resolve(
        native_graph,
        runtime.capabilities(),
        profile,
        secret_environment,
    )
    .map(Some)
}

/// Request admitted by the sealed external Driver authorization.
pub struct AuthorizedExternalDriverSpawn {
    container: String,
    request: AdapterSpawnRequest,
}

impl AuthorizedExternalDriverSpawn {
    /// Borrows the exact task container selected by authorization.
    pub fn container(&self) -> &str {
        &self.container
    }

    /// Borrows the exact declared non-shell argv.
    pub fn argv(&self) -> &[String] {
        self.request.argv()
    }

    /// Borrows the validated empty environment.
    pub fn environment(&self) -> &BTreeMap<String, String> {
        self.request.environment()
    }

    /// Returns the exact lifecycle deadlines bound at authorization.
    pub const fn deadlines(&self) -> crate::eval::AdapterLifecycleDeadlines {
        self.request.deadlines()
    }

    pub(crate) fn into_request(self) -> AdapterSpawnRequest {
        self.request
    }
}

/// Provider-specific process start invoked only after external authorization succeeds.
pub trait ExternalDriverSpawnExecutor {
    /// Starts the one already-authorized external Driver request.
    fn begin_spawn(
        &self,
        request: AuthorizedExternalDriverSpawn,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError>;
}

/// Typed Docker spawn seam that always enforces the sealed external authorization.
pub struct ExternalDriverDockerSpawner {
    container: String,
    executor: Rc<dyn ExternalDriverSpawnExecutor>,
}

impl ExternalDriverDockerSpawner {
    /// Binds a provider executor to the exact labelled task container.
    pub fn new(
        request: &DockerAdapterSpawnerRequest,
        executor: Rc<dyn ExternalDriverSpawnExecutor>,
    ) -> Self {
        Self {
            container: request.container().to_owned(),
            executor,
        }
    }

    fn begin_spawn(
        self,
        authorization: &ExternallyDrivenAdapterAuthorization,
        request: AdapterSpawnRequest,
    ) -> Result<Box<dyn AdapterSpawnTransaction>, AdapterSupervisionError> {
        let request = authorization.authorize_spawn_request(&self.container, request)?;
        self.executor.begin_spawn(AuthorizedExternalDriverSpawn {
            container: self.container,
            request,
        })
    }
}

/// One pre-provisioned external Driver operation with no model or result authority.
pub struct ExternalDriverDockerSpawnOperation {
    prepared_driver: Box<dyn PreparedExternalDriver>,
    authorization: ExternallyDrivenAdapterAuthorization,
    spawner: ExternalDriverDockerSpawner,
}

impl std::fmt::Debug for ExternalDriverDockerSpawnOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExternalDriverDockerSpawnOperation")
            .field("has_prepared_driver", &true)
            .field("authorization", &self.authorization)
            .finish_non_exhaustive()
    }
}

impl ExternalDriverDockerSpawnOperation {
    /// Starts the authorized Driver while retaining its exact prepared capability.
    pub fn start(
        self,
        deadlines: crate::eval::AdapterLifecycleDeadlines,
    ) -> Result<StartedExternalDriverDockerSpawn, AdapterSupervisionError> {
        let request = self.authorization.spawn_request_with_deadlines(deadlines)?;
        #[cfg(feature = "engine")]
        let max_stdout_frame_bytes = request.max_stdout_frame_bytes();
        #[cfg(feature = "engine")]
        let max_stderr_bytes = request.max_stderr_bytes();
        let transaction = self.spawner.begin_spawn(&self.authorization, request)?;
        Ok(StartedExternalDriverDockerSpawn {
            prepared_driver: Some(self.prepared_driver),
            authorization: self.authorization,
            transaction: Some(transaction),
            #[cfg(feature = "engine")]
            adapter: None,
            #[cfg(feature = "engine")]
            deadlines,
            #[cfg(feature = "engine")]
            max_stdout_frame_bytes,
            #[cfg(feature = "engine")]
            max_stderr_bytes,
        })
    }
}

/// Started external Driver state retaining preparation and launch ownership together.
pub struct StartedExternalDriverDockerSpawn {
    prepared_driver: Option<Box<dyn PreparedExternalDriver>>,
    authorization: ExternallyDrivenAdapterAuthorization,
    transaction: Option<Box<dyn AdapterSpawnTransaction>>,
    #[cfg(feature = "engine")]
    adapter: Option<StrictSupervisedAdapter>,
    #[cfg(feature = "engine")]
    deadlines: crate::eval::AdapterLifecycleDeadlines,
    #[cfg(feature = "engine")]
    max_stdout_frame_bytes: usize,
    #[cfg(feature = "engine")]
    max_stderr_bytes: usize,
}

impl std::fmt::Debug for StartedExternalDriverDockerSpawn {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = formatter.debug_struct("StartedExternalDriverDockerSpawn");
        debug
            .field(
                "prepared_driver",
                &self
                    .prepared_driver
                    .as_deref()
                    .map(std::any::type_name_of_val),
            )
            .field("authorization", &self.authorization)
            .field("has_transaction", &self.transaction.is_some());
        #[cfg(feature = "engine")]
        debug.field("has_adapter", &self.adapter.is_some());
        debug.finish_non_exhaustive()
    }
}

#[cfg(feature = "engine")]
pub(crate) struct ExternalDriverDockerSession {
    prepared_driver: Box<dyn PreparedExternalDriver>,
    adapter: Option<Box<dyn SupervisedAdapter>>,
    protocol: AdapterProtocolConfig,
    capture_session: CompatibilityCaptureSession,
}

#[cfg(feature = "engine")]
impl StartedExternalDriverDockerSpawn {
    pub(crate) async fn establish_session(
        &mut self,
        protocol: &AdapterProtocolConfig,
    ) -> Result<(), AdapterSupervisionError> {
        let process = self
            .transaction
            .as_deref_mut()
            .ok_or(AdapterSupervisionError::AlreadyReaped)?
            .await_process()
            .await?;
        self.transaction.take();
        self.adapter = Some(StrictSupervisedAdapter::from_prestarted_process(
            protocol.clone(),
            process,
            self.deadlines,
            self.max_stdout_frame_bytes,
            self.max_stderr_bytes,
        ));
        self.adapter
            .as_mut()
            .ok_or(AdapterSupervisionError::AlreadyReaped)?
            .negotiate_startup(protocol)
            .await
    }

    pub(crate) fn into_session(
        mut self,
        protocol: AdapterProtocolConfig,
        capture_session: CompatibilityCaptureSession,
    ) -> Result<ExternalDriverDockerSession, EvalExecutionError> {
        if self.transaction.is_some() {
            return Err(EvalExecutionError::InvalidRecipe(
                "established external Driver session",
            ));
        }
        let adapter = self
            .adapter
            .take()
            .ok_or(EvalExecutionError::InvalidRecipe(
                "established external Driver adapter",
            ))?;
        let prepared_driver =
            self.prepared_driver
                .take()
                .ok_or(EvalExecutionError::InvalidRecipe(
                    "prepared external Driver session",
                ))?;
        Ok(ExternalDriverDockerSession {
            prepared_driver,
            adapter: Some(Box::new(adapter)),
            protocol,
            capture_session,
        })
    }

    pub(crate) async fn cancel_and_reap(&mut self) -> Result<(), EvalExecutionError> {
        let cleanup = self
            .take_cleanup()
            .ok_or(EvalExecutionError::InvalidRecipe(
                "external Driver startup cleanup",
            ))?;
        cleanup
            .complete()
            .await
            .map_err(external_driver_supervision_error)
    }

    fn take_cleanup(&mut self) -> Option<ExternalDriverCleanup> {
        ExternalDriverCleanup::new(
            self.adapter
                .take()
                .map(|adapter| Box::new(adapter) as Box<dyn SupervisedAdapter>),
            self.transaction.take(),
            self.deadlines.reap(),
        )
    }
}

#[cfg(feature = "engine")]
#[async_trait(?Send)]
impl super::native_graph_episode::ExternallyDrivenEpisodeSession for ExternalDriverDockerSession {
    async fn request_terminal(
        &mut self,
    ) -> Result<CompatibilityTerminalReceipt, EvalExecutionError> {
        let adapter = self
            .adapter
            .as_deref_mut()
            .ok_or(EvalExecutionError::InvalidRecipe(
                "live external Driver session",
            ))?;
        let mut session = ProtocolExternalDriverSession::new(
            adapter,
            self.protocol.clone(),
            self.capture_session.clone(),
        )
        .map_err(|error| {
            EvalExecutionError::ProcessFailure(format!(
                "external Driver protocol admission failed: {error}"
            ))
        })?;
        self.prepared_driver
            .run(&mut session)
            .await
            .map_err(|error| {
                EvalExecutionError::ProcessFailure(format!("external Driver failed: {error}"))
            })
    }

    async fn cancel_and_reap(&mut self) -> Result<(), EvalExecutionError> {
        let adapter = self
            .adapter
            .take()
            .ok_or(EvalExecutionError::InvalidRecipe(
                "live external Driver cleanup",
            ))?;
        ExternalDriverCleanup::new(Some(adapter), None, Duration::ZERO)
            .ok_or(EvalExecutionError::InvalidRecipe(
                "live external Driver cleanup",
            ))?
            .complete()
            .await
            .map_err(external_driver_supervision_error)
    }
}

#[cfg(feature = "engine")]
struct ExternalDriverCleanup {
    adapter: Option<Box<dyn SupervisedAdapter>>,
    transaction: Option<Box<dyn AdapterSpawnTransaction>>,
    startup_abort_deadline: Duration,
}

#[cfg(feature = "engine")]
impl ExternalDriverCleanup {
    fn new(
        adapter: Option<Box<dyn SupervisedAdapter>>,
        transaction: Option<Box<dyn AdapterSpawnTransaction>>,
        startup_abort_deadline: Duration,
    ) -> Option<Self> {
        if adapter.is_none() && transaction.is_none() {
            return None;
        }
        Some(Self {
            adapter,
            transaction,
            startup_abort_deadline,
        })
    }

    async fn run(mut self) -> Result<(), AdapterSupervisionError> {
        if let Some(mut adapter) = self.adapter.take() {
            return adapter
                .cancel_and_reap(CancelReason::HostShutdown)
                .await
                .map(|_| ());
        }
        let mut transaction = self
            .transaction
            .take()
            .ok_or(AdapterSupervisionError::AlreadyReaped)?;
        let result = transaction.abort(self.startup_abort_deadline).await;
        if result.is_err() {
            transaction.fence();
        }
        result
    }

    async fn complete(self) -> Result<(), AdapterSupervisionError> {
        self.spawn().await.map_err(|_| {
            AdapterSupervisionError::Process(
                "external Driver cleanup task ended unexpectedly".to_owned(),
            )
        })?
    }

    fn spawn(self) -> tokio::sync::oneshot::Receiver<Result<(), AdapterSupervisionError>> {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        tokio::task::spawn_local(async move {
            let _ = sender.send(self.run().await);
        });
        receiver
    }

    fn schedule(self) {
        drop(self.spawn());
    }
}

#[cfg(feature = "engine")]
impl Drop for ExternalDriverCleanup {
    fn drop(&mut self) {
        if let Some(transaction) = self.transaction.as_deref_mut() {
            transaction.fence();
        }
    }
}

#[cfg(feature = "engine")]
impl Drop for ExternalDriverDockerSession {
    fn drop(&mut self) {
        if let Some(cleanup) = ExternalDriverCleanup::new(self.adapter.take(), None, Duration::ZERO)
        {
            cleanup.schedule();
        }
    }
}

#[cfg(feature = "engine")]
fn external_driver_supervision_error(error: AdapterSupervisionError) -> EvalExecutionError {
    EvalExecutionError::ProcessFailure(format!("external Driver supervision failed: {error}"))
}

impl Drop for StartedExternalDriverDockerSpawn {
    fn drop(&mut self) {
        #[cfg(feature = "engine")]
        if let Some(cleanup) = self.take_cleanup() {
            cleanup.schedule();
        }
        #[cfg(not(feature = "engine"))]
        if let Some(transaction) = self.transaction.as_deref_mut() {
            transaction.fence();
        }
    }
}

pub(crate) fn prepare_external_driver_spawn(
    runtime: &dyn DockerRuntime,
    package: &HarborTaskPackage,
    trial: &ResolvedEpisodeTrial,
    plan: &BenchmarkExecutionPlan,
    prepared_driver: Option<PreparedExternalDriverCapability>,
    container: &str,
    project: ComposeProjectId,
    deadlines: crate::eval::AdapterLifecycleDeadlines,
) -> Result<ExternalDriverDockerSpawnOperation, EvalExecutionError> {
    if plan.is_multi_step() || package.execution_plan().is_multi_step() {
        return Err(EvalExecutionError::UnsupportedMultiStep);
    }
    if plan.compose().is_some() || package.execution_plan().compose().is_some() {
        return Err(EvalExecutionError::UnsupportedEnforcement(
            "external Driver Docker Compose",
        ));
    }
    if plan.verifier().mode() != crate::eval::VerifierMode::Separate
        || package.execution_plan().verifier().mode() != crate::eval::VerifierMode::Separate
    {
        return Err(EvalExecutionError::UnsupportedEnforcement(
            "external Driver shared verifier isolation",
        ));
    }
    let prepared_driver = prepared_driver.ok_or(EvalExecutionError::InvalidRecipe(
        "prepared external Driver",
    ))?;
    if !package.is_standard_directory()
        || package.execution_plan() != plan
        || trial.package().execution_plan() != plan
    {
        return Err(EvalExecutionError::InvalidRecipe(
            "external Driver execution plan",
        ));
    }
    let prepared_driver = prepared_driver
        .into_driver_for(package, trial)
        .map_err(|_| EvalExecutionError::InvalidRecipe("prepared external Driver capability"))?;
    preflight_docker(runtime, plan)?;
    let authorization =
        ExternallyDrivenAdapterAuthorization::resolve(package, trial, container, deadlines)?;
    let spawner_request = DockerAdapterSpawnerRequest::new(container, project)?;
    let spawner = runtime.external_driver_spawner(&spawner_request)?;
    Ok(ExternalDriverDockerSpawnOperation {
        prepared_driver,
        authorization,
        spawner,
    })
}

pub(crate) struct ComposePreflightRequest<'a> {
    pub(crate) runtime: &'a dyn DockerRuntime,
    pub(crate) plan: &'a BenchmarkExecutionPlan,
    pub(crate) environment: &'a EnvironmentPlan,
    pub(crate) environment_root: &'a Path,
    pub(crate) project: ComposeProjectId,
    pub(crate) project_directory: &'a Path,
    pub(crate) image_tag: &'a str,
    pub(crate) project_labels: &'a BTreeMap<String, String>,
    pub(crate) workspace: &'a Path,
    pub(crate) authored_overlay: &'a [u8],
    pub(crate) deadline: Duration,
}

/// Runs the read-only provider configuration check for a Compose-backed plan.
///
/// Callers must pass paths below an acquired package materialization. This
/// function never builds, creates, or starts project resources.
pub(crate) fn preflight_compose_configuration(
    request: ComposePreflightRequest<'_>,
) -> Result<super::compose_policy::RenderedGeneratedMainCompose, EvalExecutionError> {
    preflight_docker(request.runtime, request.plan)?;
    let compose_plan = request
        .plan
        .compose()
        .ok_or(EvalExecutionError::InvalidRecipe("Compose project plan"))?;
    let authored = super::compose_policy::validate_authored_compose(
        request.authored_overlay,
        compose_plan,
        request.environment_root,
    )?;
    let rendered = super::compose_policy::render_generated_project_compose(
        request.image_tag,
        request.project_labels,
        request.environment,
        request.workspace,
        &authored,
    )?;
    let config_request = DockerComposeConfigRequest::new(
        request.project,
        request.project_directory,
        rendered.bytes().to_vec(),
        request
            .project_directory
            .join(compose_plan.definition_path()),
    )
    .with_deadline(request.deadline);
    let compose_runtime =
        request
            .runtime
            .compose_runtime()
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "Docker Compose runtime",
            ))?;
    let canonical = compose_runtime.compose_config(&config_request)?;
    super::compose_policy::validate_provider_compose_config(
        &canonical,
        compose_plan,
        request.environment_root,
        &rendered,
        &authored,
    )?;
    Ok(rendered)
}

/// An opaque task-owned Docker Compose project identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComposeProjectId {
    project: String,
    run: String,
}

impl ComposeProjectId {
    /// Creates an opaque project identifier selected by the evaluation runtime.
    pub fn new(value: impl Into<String>) -> Self {
        Self {
            project: value.into(),
            run: uuid::Uuid::new_v4().simple().to_string(),
        }
    }

    /// Returns the project identifier for provider argument construction.
    pub fn as_str(&self) -> &str {
        &self.project
    }

    /// Returns the exact labels used to select resources owned by this run.
    pub fn ownership_labels(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("aiperf.project".to_owned(), self.project.clone()),
            ("aiperf.run".to_owned(), self.run.clone()),
        ])
    }
}

/// A read-only request for a fully resolved Compose configuration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerComposeConfigRequest {
    project: ComposeProjectId,
    project_directory: PathBuf,
    generated_definition: Vec<u8>,
    overlay_definition: PathBuf,
    deadline: Option<Duration>,
}

impl DockerComposeConfigRequest {
    /// Creates a config request with interpolation and env-file loading disabled.
    pub fn new(
        project: ComposeProjectId,
        project_directory: impl Into<PathBuf>,
        generated_definition: Vec<u8>,
        overlay_definition: impl Into<PathBuf>,
    ) -> Self {
        Self {
            project,
            project_directory: project_directory.into(),
            generated_definition,
            overlay_definition: overlay_definition.into(),
            deadline: None,
        }
    }

    /// Returns the task-owned project identifier.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }

    /// Returns the owned, materialized project directory.
    pub fn project_directory(&self) -> &Path {
        &self.project_directory
    }

    /// Returns the generated runtime-owned base definition bytes.
    pub fn generated_definition(&self) -> &[u8] {
        &self.generated_definition
    }

    /// Returns the materialized authored overlay path.
    pub fn overlay_definition(&self) -> &Path {
        &self.overlay_definition
    }

    /// Bounds the host-side provider configuration operation.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns the host-side provider configuration deadline, if configured.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }

    /// Reports that Compose interpolation is prohibited for this request.
    pub const fn interpolation_disabled(&self) -> bool {
        true
    }

    /// Reports that Compose env-file resolution is prohibited for this request.
    pub const fn env_file_disabled(&self) -> bool {
        true
    }
}

macro_rules! compose_project_request {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Debug, PartialEq, Eq)]
        pub struct $name {
            project: ComposeProjectId,
            project_directory: PathBuf,
            labels: BTreeMap<String, String>,
            deadline: Option<Duration>,
            container_grace: Duration,
        }

        impl $name {
            /// Creates a request scoped to one task-owned Compose project.
            pub fn new(project: ComposeProjectId, project_directory: impl Into<PathBuf>) -> Self {
                let labels = project.ownership_labels();
                Self {
                    project,
                    project_directory: project_directory.into(),
                    labels,
                    deadline: None,
                    container_grace: Duration::from_secs(10),
                }
            }

            /// Returns the task-owned project identifier.
            pub fn project(&self) -> &ComposeProjectId {
                &self.project
            }

            /// Returns the owned, materialized project directory.
            pub fn project_directory(&self) -> &Path {
                &self.project_directory
            }

            /// Returns the exact AIPerf ownership labels for this operation.
            pub fn labels(&self) -> &BTreeMap<String, String> {
                &self.labels
            }

            /// Bounds the host-side provider operation.
            pub fn with_deadline(mut self, deadline: Duration) -> Self {
                self.deadline = Some(deadline);
                self
            }

            /// Returns the host-side provider operation deadline, if configured.
            pub const fn deadline(&self) -> Option<Duration> {
                self.deadline
            }
        }
    };
}

compose_project_request!(
    DockerComposeBuildRequest,
    "A Compose project build request."
);

impl DockerComposeUpRequest {
    /// Compose startup is detached so a task lease owns the project lifecycle.
    pub const fn detached(&self) -> bool {
        true
    }
    /// Startup waits for service readiness before exposing the lease.
    pub const fn wait_for_readiness(&self) -> bool {
        true
    }
}

impl DockerComposeDownRequest {
    /// Gives containers a bounded graceful-stop interval before forced cleanup.
    pub const fn container_grace(&self) -> Duration {
        self.container_grace
    }
    /// Forces task-owned containers down after a terminal benchmark failure.
    pub fn with_terminal_failure(mut self) -> Self {
        self.container_grace = Duration::ZERO;
        self
    }
    /// Requests removal of task-owned anonymous volumes.
    pub const fn removes_volumes(&self) -> bool {
        true
    }
    /// Requests removal only of orphans in this exact project.
    pub const fn removes_orphans(&self) -> bool {
        true
    }
}
compose_project_request!(
    DockerComposeUpRequest,
    "A detached Compose project startup request."
);
compose_project_request!(
    DockerComposeDownRequest,
    "A Compose project teardown request."
);

/// A redacted Compose service command request.
#[derive(Debug)]
pub struct DockerComposeExecRequest {
    project: ComposeProjectId,
    service: ComposeServiceName,
    public_arguments: Vec<String>,
    public_environment: BTreeMap<EnvName, String>,
    secret_environment: BTreeMap<EnvName, SecretValue>,
    phase: EvalExecutionPhase,
    user: Option<String>,
    workdir: Option<String>,
    deadline: Option<Duration>,
    labels: BTreeMap<String, String>,
}

/// Identifies one task-owned Compose service for a streaming adapter client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerComposeAdapterSpawnerRequest {
    project: ComposeProjectId,
    service: ComposeServiceName,
    user: Option<String>,
    workdir: Option<String>,
    deadline: Duration,
}

impl DockerComposeAdapterSpawnerRequest {
    /// Binds a streaming adapter to one service in an already-owned project.
    pub fn new(project: ComposeProjectId, service: ComposeServiceName) -> Self {
        Self {
            project,
            service,
            user: None,
            workdir: None,
            deadline: Duration::from_secs(10),
        }
    }

    /// Pins the effective user before the Docker client process is spawned.
    pub fn with_user(mut self, user: Option<&str>) -> Self {
        self.user = user.map(ToOwned::to_owned);
        self
    }

    /// Pins the effective workdir before the Docker client process is spawned.
    pub fn with_workdir(mut self, workdir: Option<&str>) -> Self {
        self.workdir = workdir.map(ToOwned::to_owned);
        self
    }

    /// Bounds the ownership lookup before any streaming client is started.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = deadline;
        self
    }

    /// Borrows the exact Compose project identity.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }

    /// Borrows the task-owned service identity.
    pub fn service(&self) -> &ComposeServiceName {
        &self.service
    }

    /// Borrows the requested user, if any.
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
    }

    /// Borrows the requested workdir, if any.
    pub fn workdir(&self) -> Option<&str> {
        self.workdir.as_deref()
    }

    /// Returns the bounded task-owned container lookup deadline.
    pub const fn deadline(&self) -> Duration {
        self.deadline
    }
}

impl DockerComposeExecRequest {
    /// Creates a service command with renderable and secret environments separated.
    pub fn new(
        project: ComposeProjectId,
        service: ComposeServiceName,
        arguments: impl IntoIterator<Item = impl Into<String>>,
        public_environment: BTreeMap<EnvName, String>,
        secret_environment: BTreeMap<EnvName, SecretValue>,
    ) -> Self {
        let labels = project.ownership_labels();
        Self {
            project,
            service,
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            public_environment,
            secret_environment,
            phase: EvalExecutionPhase::Agent,
            user: None,
            workdir: None,
            deadline: None,
            labels,
        }
    }

    /// Returns the task-owned project identifier.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }
    /// Returns the target service.
    pub fn service(&self) -> &ComposeServiceName {
        &self.service
    }
    /// Returns command arguments safe for diagnostics.
    pub fn public_arguments(&self) -> &[String] {
        &self.public_arguments
    }
    /// Returns literal environment bindings.
    pub fn public_environment(&self) -> &BTreeMap<EnvName, String> {
        &self.public_environment
    }
    /// Returns secret variable names without their values.
    pub fn secret_names(&self) -> Vec<&str> {
        self.secret_environment.keys().map(String::as_str).collect()
    }
    /// Returns the command phase.
    pub const fn phase(&self) -> EvalExecutionPhase {
        self.phase
    }
    /// Returns the effective user, if any.
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
    }
    /// Returns the effective workdir, if any.
    pub fn workdir(&self) -> Option<&str> {
        self.workdir.as_deref()
    }
    /// Returns the command deadline, if any.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }
    /// Returns the exact AIPerf ownership labels for this operation.
    pub fn labels(&self) -> &BTreeMap<String, String> {
        &self.labels
    }

    /// Adds phase execution policy to this service command.
    pub fn with_phase(
        mut self,
        phase: EvalExecutionPhase,
        user: Option<&str>,
        workdir: Option<&str>,
        deadline: Option<Duration>,
    ) -> Self {
        self.phase = phase;
        self.user = user.map(ToOwned::to_owned);
        self.workdir = workdir.map(ToOwned::to_owned);
        self.deadline = deadline;
        self
    }

    #[allow(dead_code)]
    pub(crate) fn secret_environment(&self) -> &BTreeMap<EnvName, SecretValue> {
        &self.secret_environment
    }
}

/// A Compose service archive request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerComposeArchiveRequest {
    project: ComposeProjectId,
    service: ComposeServiceName,
    source: String,
    labels: BTreeMap<String, String>,
}

/// A controlled host-to-service transfer from an immutable task snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerComposeCopyRequest {
    project: ComposeProjectId,
    service: ComposeServiceName,
    source: String,
    destination: String,
    labels: BTreeMap<String, String>,
    deadline: Option<Duration>,
}

impl DockerComposeCopyRequest {
    /// Creates one exact service-local copy request.
    pub fn new(
        project: ComposeProjectId,
        service: ComposeServiceName,
        source: impl Into<String>,
        destination: impl Into<String>,
    ) -> Self {
        let labels = project.ownership_labels();
        Self {
            project,
            service,
            source: source.into(),
            destination: destination.into(),
            labels,
            deadline: None,
        }
    }
    /// Returns the project selection.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }
    /// Returns the target service.
    pub fn service(&self) -> &ComposeServiceName {
        &self.service
    }
    /// Returns the snapshot source path.
    pub fn source(&self) -> &str {
        &self.source
    }
    /// Returns the service-local destination path.
    pub fn destination(&self) -> &str {
        &self.destination
    }
    /// Returns exact ownership labels.
    pub fn labels(&self) -> &BTreeMap<String, String> {
        &self.labels
    }
    /// Bounds lookup and transfer as part of a phase deadline.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }
    /// Returns the optional transfer deadline.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }
}

impl DockerComposeArchiveRequest {
    /// Creates an archive request for one service path.
    pub fn new(
        project: ComposeProjectId,
        service: ComposeServiceName,
        source: impl Into<String>,
    ) -> Self {
        let labels = project.ownership_labels();
        Self {
            project,
            service,
            source: source.into(),
            labels,
        }
    }
    /// Returns the task-owned project identifier.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }
    /// Returns the target service.
    pub fn service(&self) -> &ComposeServiceName {
        &self.service
    }
    /// Returns the source path inside the target service.
    pub fn source(&self) -> &str {
        &self.source
    }
    /// Returns the exact AIPerf ownership labels for this operation.
    pub fn labels(&self) -> &BTreeMap<String, String> {
        &self.labels
    }
}

/// A request to stop one service in a task-owned Compose project.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerComposeStopRequest {
    project: ComposeProjectId,
    service: ComposeServiceName,
    labels: BTreeMap<String, String>,
    deadline: Option<Duration>,
}

impl DockerComposeStopRequest {
    /// Creates a service stop request.
    pub fn new(project: ComposeProjectId, service: ComposeServiceName) -> Self {
        let labels = project.ownership_labels();
        Self {
            project,
            service,
            labels,
            deadline: None,
        }
    }
    /// Returns the task-owned project identifier.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }
    /// Returns the target service.
    pub fn service(&self) -> &ComposeServiceName {
        &self.service
    }
    /// Returns the exact AIPerf ownership labels for this operation.
    pub fn labels(&self) -> &BTreeMap<String, String> {
        &self.labels
    }

    /// Applies a bounded stop deadline.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns the stop deadline, if one was configured.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }
}

/// Exact resources discovered under one task-owned Compose project label set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OwnedComposeResources {
    containers: Vec<String>,
    networks: Vec<String>,
    volumes: Vec<String>,
}

impl OwnedComposeResources {
    /// Creates resources returned by an exact project-label discovery operation.
    pub fn new(containers: Vec<String>, networks: Vec<String>, volumes: Vec<String>) -> Self {
        Self {
            containers,
            networks,
            volumes,
        }
    }
    /// Returns exact owned container identifiers.
    pub fn containers(&self) -> &[String] {
        &self.containers
    }
    /// Returns exact owned network identifiers.
    pub fn networks(&self) -> &[String] {
        &self.networks
    }
    /// Returns exact owned volume identifiers.
    pub fn volumes(&self) -> &[String] {
        &self.volumes
    }
}

/// Resolved environment data split between renderable and secret values.
#[derive(Debug)]
pub struct DockerEnvironment {
    public: BTreeMap<EnvName, String>,
    secrets: BTreeMap<EnvName, SecretValue>,
}

impl DockerEnvironment {
    /// Returns literal values that may appear in command diagnostics.
    pub fn public(&self) -> &BTreeMap<EnvName, String> {
        &self.public
    }

    /// Returns secret variable names in deterministic order.
    pub fn secret_names(&self) -> Vec<&str> {
        self.secrets.keys().map(String::as_str).collect()
    }

    pub(crate) fn secrets(&self) -> &BTreeMap<EnvName, SecretValue> {
        &self.secrets
    }
}

/// Resolves exactly the bindings needed by one execution phase.
pub fn resolve_phase_environment(
    environment: &EnvironmentPlan,
    phase: &PhasePlan,
    secrets: &dyn SecretProvider,
) -> Result<DockerEnvironment, EvalExecutionError> {
    let mut bindings = environment.env().clone();
    bindings.extend(phase.env().clone());
    resolve_bindings(bindings, secrets)
}

/// Resolves the environment baseline without activating any phase bindings.
pub fn resolve_environment(
    environment: &EnvironmentPlan,
    secrets: &dyn SecretProvider,
) -> Result<DockerEnvironment, EvalExecutionError> {
    resolve_bindings(environment.env().clone(), secrets)
}

fn resolve_bindings(
    bindings: BTreeMap<EnvName, super::EnvBinding>,
    secrets: &dyn SecretProvider,
) -> Result<DockerEnvironment, EvalExecutionError> {
    let mut public = BTreeMap::new();
    let mut resolved_secrets = BTreeMap::new();
    for (name, binding) in bindings {
        if let Some(value) = binding.literal() {
            public.insert(name, value.to_owned());
            continue;
        }
        let reference = binding
            .secret_reference()
            .ok_or(EvalExecutionError::InvalidRecipe("environment binding"))?;
        let value = secrets
            .resolve(&reference.to_owned())
            .map_err(|_| EvalExecutionError::MissingSecret(reference.to_owned()))?;
        resolved_secrets.insert(name, value);
    }
    Ok(DockerEnvironment {
        public,
        secrets: resolved_secrets,
    })
}

/// A redacted Docker image-build request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerBuildRequest {
    public_arguments: Vec<String>,
    network_lease: Option<String>,
    deadline: Option<Duration>,
}

impl DockerBuildRequest {
    /// Creates a request from non-secret Docker arguments.
    pub fn new(arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            network_lease: None,
            deadline: None,
        }
    }

    /// Returns Docker arguments that may appear in diagnostics.
    pub fn public_arguments(&self) -> &[String] {
        &self.public_arguments
    }

    /// Associates the build with its provider-managed network lease.
    pub fn with_network_lease(mut self, network_lease: impl Into<String>) -> Self {
        self.network_lease = Some(network_lease.into());
        self
    }

    /// Returns the provider-managed network lease, when the build requires one.
    pub fn network_lease(&self) -> Option<&str> {
        self.network_lease.as_deref()
    }

    /// Bounds the host Docker build operation.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns the host build deadline when the plan configures one.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }
}

/// A redacted Docker container-create request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerCreateRequest {
    public_arguments: Vec<String>,
    network_lease: Option<String>,
    deadline: Option<Duration>,
    creation_target: Option<String>,
    creation_phase: Option<EvalExecutionPhase>,
}

impl DockerCreateRequest {
    /// Creates a request from non-secret Docker arguments.
    pub fn new(arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            network_lease: None,
            deadline: None,
            creation_target: None,
            creation_phase: None,
        }
    }

    /// Returns Docker arguments that may appear in diagnostics.
    pub fn public_arguments(&self) -> &[String] {
        &self.public_arguments
    }

    /// Associates the container with its provider-managed network lease.
    pub fn with_network_lease(mut self, network_lease: impl Into<String>) -> Self {
        self.network_lease = Some(network_lease.into());
        self
    }

    /// Returns the provider-managed network lease, when the container requires one.
    pub fn network_lease(&self) -> Option<&str> {
        self.network_lease.as_deref()
    }

    /// Bounds container creation by the enclosing phase deadline.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns the host creation deadline when one is configured.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }

    /// Identifies a bounded create so its exact resource can be compensated if
    /// the Docker client times out after the daemon has accepted the request.
    pub fn with_creation_identity(
        mut self,
        target: impl Into<String>,
        phase: EvalExecutionPhase,
    ) -> Self {
        self.creation_target = Some(target.into());
        self.creation_phase = Some(phase);
        self
    }

    /// Returns the exact container name that must be compensated on timeout.
    pub fn creation_target(&self) -> Option<&str> {
        self.creation_target.as_deref()
    }

    /// Returns the phase that owns this bounded create.
    pub const fn creation_phase(&self) -> Option<EvalExecutionPhase> {
        self.creation_phase
    }
}

/// A Docker container-start request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerStartRequest {
    container: String,
    deadline: Option<Duration>,
}

impl DockerStartRequest {
    /// Creates a container-start request.
    pub fn new(container: impl Into<String>) -> Self {
        Self {
            container: container.into(),
            deadline: None,
        }
    }

    /// Returns the container identifier.
    pub fn container(&self) -> &str {
        &self.container
    }

    /// Bounds container startup by the enclosing phase deadline.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns the host startup deadline when one is configured.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }
}

/// A Docker command execution request whose secrets cannot be rendered.
#[derive(Debug)]
pub struct DockerExecRequest {
    container: String,
    public_arguments: Vec<String>,
    public_environment: BTreeMap<EnvName, String>,
    secret_environment: BTreeMap<EnvName, SecretValue>,
    phase: EvalExecutionPhase,
    user: Option<String>,
    workdir: Option<String>,
    network_lease: String,
    deadline: Option<Duration>,
}

/// Identifies one labelled task container for a streaming `docker exec -i` client.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerAdapterSpawnerRequest {
    container: String,
    project: ComposeProjectId,
    user: Option<String>,
    workdir: Option<String>,
}

impl DockerAdapterSpawnerRequest {
    /// Binds a streaming adapter to one container resolved from its owning project.
    pub(crate) fn new(
        container: impl Into<String>,
        project: ComposeProjectId,
    ) -> Result<Self, EvalExecutionError> {
        let container = container.into();
        if container.trim().is_empty() {
            return Err(EvalExecutionError::InvalidRecipe("adapter container"));
        }
        Ok(Self {
            container,
            project,
            user: None,
            workdir: None,
        })
    }

    /// Pins the container user for every adapter child spawned through this lease.
    pub fn with_user(mut self, user: Option<&str>) -> Self {
        self.user = user.map(ToOwned::to_owned);
        self
    }

    /// Pins the container workdir for every adapter child spawned through this lease.
    pub fn with_workdir(mut self, workdir: Option<&str>) -> Self {
        self.workdir = workdir.map(ToOwned::to_owned);
        self
    }

    /// Borrows the previously labelled task container identity.
    pub fn container(&self) -> &str {
        &self.container
    }

    /// Borrows the project whose ownership labels selected this container.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }

    /// Borrows the requested user, if any.
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
    }

    /// Borrows the requested workdir, if any.
    pub fn workdir(&self) -> Option<&str> {
        self.workdir.as_deref()
    }
}

/// A task-owned remote container lease for one supervised adapter client.
///
/// The lease is deliberately separate from the local `docker exec` client:
/// closing that client alone does not stop code already running in the task
/// container. Implementations must terminate only the previously validated
/// task-owned container represented by this lease.
#[async_trait(?Send)]
pub trait DockerAdapterLease {
    /// Terminates the exact task-owned remote container within the supplied budget.
    async fn terminate(&self, deadline: Duration) -> Result<(), AdapterSupervisionError>;
    /// Starts best-effort termination when `Drop` cannot await the provider.
    fn fence(&self);
}

/// Couples a local streaming client to its exact task-owned remote container lease.
///
/// A remote termination is required before this wrapper allows the local client
/// to report `Reaped`, so fencing only the host-side `docker exec` process can
/// never leave its remote adapter runnable.
pub struct DockerAdapterProcess {
    client: Box<dyn AdapterProcess>,
    lease: Rc<dyn DockerAdapterLease>,
    has_terminated_remote: bool,
}

impl DockerAdapterProcess {
    /// Binds one local streaming client to its validated task-owned container lease.
    pub fn new(client: Box<dyn AdapterProcess>, lease: Rc<dyn DockerAdapterLease>) -> Self {
        Self {
            client,
            lease,
            has_terminated_remote: false,
        }
    }

    async fn terminate_remote(
        &mut self,
        deadline: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        if self.has_terminated_remote {
            return Ok(());
        }
        self.lease.terminate(deadline).await?;
        self.has_terminated_remote = true;
        Ok(())
    }
}

#[async_trait(?Send)]
impl AdapterProcess for DockerAdapterProcess {
    async fn write_frame(
        &mut self,
        frame: &[u8],
        deadline: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        self.client.write_frame(frame, deadline).await
    }

    async fn read_stdout_frame(
        &mut self,
        max_bytes: usize,
        deadline: Duration,
    ) -> Result<Vec<u8>, AdapterSupervisionError> {
        self.client.read_stdout_frame(max_bytes, deadline).await
    }

    async fn drain_stderr(&mut self, max_bytes: usize) -> Result<Vec<u8>, AdapterSupervisionError> {
        self.client.drain_stderr(max_bytes).await
    }

    async fn cancel(
        &mut self,
        reason: CancelReason,
        deadline: Duration,
    ) -> Result<(), AdapterSupervisionError> {
        let client = self.client.cancel(reason, deadline).await;
        let remote = self.terminate_remote(deadline).await;
        match (client, remote) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(primary), Err(recovery)) => Err(AdapterSupervisionError::Recovery {
                primary: Box::new(primary),
                recovery: Box::new(recovery),
            }),
        }
    }

    async fn reap(&mut self, deadline: Duration) -> Result<AdapterExit, AdapterSupervisionError> {
        self.terminate_remote(deadline).await?;
        self.client.reap(deadline).await
    }

    fn fence(&mut self) {
        self.client.fence();
        self.lease.fence();
    }
}

impl DockerExecRequest {
    /// Creates a command request with literal and secret environments kept separate.
    pub fn new(
        container: impl Into<String>,
        arguments: impl IntoIterator<Item = impl Into<String>>,
        public_environment: BTreeMap<EnvName, String>,
        secret_environment: BTreeMap<EnvName, SecretValue>,
    ) -> Self {
        Self {
            container: container.into(),
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            public_environment,
            secret_environment,
            phase: EvalExecutionPhase::Agent,
            user: None,
            workdir: None,
            network_lease: String::new(),
            deadline: None,
        }
    }

    /// Returns the container identifier.
    pub fn container(&self) -> &str {
        &self.container
    }

    /// Returns non-secret command arguments.
    pub fn public_arguments(&self) -> &[String] {
        &self.public_arguments
    }

    /// Returns literal environment bindings.
    pub fn public_environment(&self) -> &BTreeMap<EnvName, String> {
        &self.public_environment
    }

    /// Returns secret environment variable names without their values.
    pub fn secret_names(&self) -> Vec<&str> {
        self.secret_environment.keys().map(String::as_str).collect()
    }

    /// Adds the resolved phase policy to this execution request.
    pub fn with_phase(
        mut self,
        phase: EvalExecutionPhase,
        user: Option<&str>,
        workdir: Option<&str>,
        network_lease: impl Into<String>,
        deadline: Option<Duration>,
    ) -> Self {
        self.phase = phase;
        self.user = user.map(ToOwned::to_owned);
        self.workdir = workdir.map(ToOwned::to_owned);
        self.network_lease = network_lease.into();
        self.deadline = deadline;
        self
    }

    /// Returns the command phase this request belongs to.
    pub const fn phase(&self) -> EvalExecutionPhase {
        self.phase
    }

    /// Returns the authored effective user, if one was supplied.
    pub fn user(&self) -> Option<&str> {
        self.user.as_deref()
    }

    /// Returns the resolved working directory, if one was supplied.
    pub fn workdir(&self) -> Option<&str> {
        self.workdir.as_deref()
    }

    /// Returns the provider-managed network lease for this phase.
    pub fn network_lease(&self) -> &str {
        &self.network_lease
    }

    /// Returns the optional phase command deadline.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }

    pub(crate) fn secret_environment(&self) -> &BTreeMap<EnvName, SecretValue> {
        &self.secret_environment
    }
}

/// A Docker copy request containing only safe public arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerCopyRequest {
    public_arguments: Vec<String>,
    deadline: Option<Duration>,
}

impl DockerCopyRequest {
    /// Creates a copy request from non-secret Docker arguments.
    pub fn new(arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            deadline: None,
        }
    }

    /// Returns Docker arguments that may appear in diagnostics.
    pub fn public_arguments(&self) -> &[String] {
        &self.public_arguments
    }

    /// Bounds host file transfer by the enclosing phase deadline.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns the host transfer deadline when one is configured.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }
}

/// A Docker container or image removal request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerRemoveRequest {
    public_arguments: Vec<String>,
    deadline: Option<Duration>,
}

impl DockerRemoveRequest {
    /// Creates a removal request from non-secret Docker arguments.
    pub fn new(arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            deadline: None,
        }
    }

    /// Returns Docker arguments that may appear in diagnostics.
    pub fn public_arguments(&self) -> &[String] {
        &self.public_arguments
    }

    /// Bounds this exact-resource removal operation.
    pub fn with_deadline(mut self, deadline: Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Returns the removal deadline, if one was configured.
    pub const fn deadline(&self) -> Option<Duration> {
        self.deadline
    }
}

/// Injectable Docker boundary used by benchmark execution.
pub trait DockerRuntime {
    /// Returns provider guarantees available to this Docker implementation.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Reports whether the provider can transition one running environment between
    /// distinct phase network leases without widening connectivity.
    fn supports_phase_network_transitions(&self) -> bool {
        false
    }

    /// Returns the complete Compose provider contract when this runtime supports
    /// a task-owned Compose project.
    ///
    /// A runtime that advertises Compose capabilities but returns `None` is
    /// refused during preflight, before any build or container creation.
    fn compose_runtime(&self) -> Option<&dyn DockerComposeRuntime> {
        None
    }

    /// Resolves the provider profile that actually governs NativeGraph adapters.
    ///
    /// Returning a profile supplied by a caller instead of the selected runtime
    /// would let a task claim endpoint mediation it does not receive, so exact
    /// NativeGraph profiles fail closed unless the runtime provides this value.
    fn native_graph_provider_profile(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<ProviderProfile, EvalExecutionError> {
        Err(EvalExecutionError::UnsupportedEnforcement(
            "model endpoint isolation",
        ))
    }

    /// Resolves plan-bound adapter isolation proof before environment provisioning.
    ///
    /// The default preserves providers whose profile does not depend on a task
    /// network plan. Providers with plan-sensitive isolation, such as Docker's
    /// no-egress proof, must override this method rather than overstating a
    /// package-only guarantee.
    fn native_graph_provider_profile_for_plan(
        &self,
        package: &NativeGraphPackagePlan,
        _: &BenchmarkExecutionPlan,
    ) -> Result<ProviderProfile, EvalExecutionError> {
        self.native_graph_provider_profile(package)
    }

    /// Resolves every logical model secret to its host-only environment name.
    ///
    /// Values never cross this boundary. The exact-profile authorization checks
    /// this map against the imported bindings and strips every returned name from
    /// each adapter launch environment.
    fn native_graph_model_secret_environment(
        &self,
        _: &NativeGraphPackagePlan,
    ) -> Result<BTreeMap<ModelSecretId, EnvName>, EvalExecutionError> {
        Err(EvalExecutionError::UnsupportedEnforcement(
            "native graph model secret environment",
        ))
    }

    /// Returns a streaming adapter spawner bound to one labelled task container.
    fn adapter_spawner(
        &self,
        _: &DockerAdapterSpawnerRequest,
        _: &NativeGraphAdapterAuthorization,
    ) -> Result<Rc<dyn AdapterSpawner>, EvalExecutionError> {
        Err(EvalExecutionError::UnsupportedEnforcement(
            "streaming Docker adapter spawn",
        ))
    }

    /// Returns a spawner bound only to one externally driven task container.
    fn external_driver_spawner(
        &self,
        _: &DockerAdapterSpawnerRequest,
    ) -> Result<ExternalDriverDockerSpawner, EvalExecutionError> {
        Err(EvalExecutionError::UnsupportedEnforcement(
            "external Driver Docker adapter spawn",
        ))
    }

    /// Builds the requested immutable task environment.
    fn build(&self, request: &DockerBuildRequest) -> Result<(), EvalExecutionError>;

    /// Creates the requested container.
    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError>;

    /// Compensates a bounded create whose client completion is uncertain.
    ///
    /// A provider must override this only when it can retry the exact creation
    /// target until it is confirmed absent within `cleanup_deadline`.
    fn compensate_create_timeout(
        &self,
        request: &DockerCreateRequest,
        _: Duration,
    ) -> Result<(), EvalExecutionError> {
        let target = request
            .creation_target()
            .unwrap_or("unknown Docker create target");
        Err(EvalExecutionError::ContainerTeardown {
            container: target.to_owned(),
            reason: "Docker provider cannot compensate an uncertain create".to_owned(),
        })
    }

    /// Starts a created container.
    fn start(&self, request: &DockerStartRequest) -> Result<(), EvalExecutionError>;

    /// Executes one redacted phase command.
    fn exec(&self, request: &DockerExecRequest) -> Result<(), EvalExecutionError>;

    /// Transfers files through the Docker provider boundary.
    fn copy(&self, request: &DockerCopyRequest) -> Result<(), EvalExecutionError>;

    /// Returns the created container's absolute configured working directory.
    ///
    /// Providers must return `/` when the image and container leave the working
    /// directory unset.
    fn container_workdir(&self, _: &str) -> Result<String, EvalExecutionError> {
        Err(EvalExecutionError::UnsupportedEnforcement(
            "container workdir inspection",
        ))
    }

    /// Inspects the configured workdir within an enclosing phase deadline.
    fn container_workdir_bounded(
        &self,
        container: &str,
        _: Duration,
    ) -> Result<String, EvalExecutionError> {
        self.container_workdir(container)
    }

    /// Opens one declared container path as a streaming Docker archive.
    fn copy_archive(&self, _: &str, _: &str) -> Result<Box<dyn Read>, EvalExecutionError> {
        Err(EvalExecutionError::ArtifactCollection(
            "Docker provider cannot collect artifact archives".to_owned(),
        ))
    }

    /// Opens one declared container path while enforcing its collection deadline.
    fn copy_archive_bounded(
        &self,
        container: &str,
        source: &str,
        _: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError> {
        self.copy_archive(container, source)
    }

    /// Removes a container, image, or related lease.
    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError>;
}

/// Required Docker Compose operations for a task-owned benchmark project.
///
/// The trait intentionally has no operation defaults: a runtime must provide
/// every lifecycle operation before it can advertise Compose support.
pub trait DockerComposeRuntime: DockerRuntime {
    /// Returns the normalized JSON for the supplied read-only Compose request.
    fn compose_config(
        &self,
        request: &DockerComposeConfigRequest,
    ) -> Result<Vec<u8>, EvalExecutionError>;

    /// Builds the project images.
    fn compose_build(&self, request: &DockerComposeBuildRequest) -> Result<(), EvalExecutionError>;

    /// Starts the project services.
    fn compose_up(&self, request: &DockerComposeUpRequest) -> Result<(), EvalExecutionError>;

    /// Executes a redacted command in one project service.
    fn compose_exec(&self, request: &DockerComposeExecRequest) -> Result<(), EvalExecutionError>;

    /// Returns a streaming adapter spawner for one verified Compose service.
    fn compose_adapter_spawner(
        &self,
        _: &DockerComposeAdapterSpawnerRequest,
        _: &NativeGraphAdapterAuthorization,
    ) -> Result<Rc<dyn AdapterSpawner>, EvalExecutionError> {
        Err(EvalExecutionError::UnsupportedEnforcement(
            "streaming Docker Compose adapter spawn",
        ))
    }

    /// Opens a service path as a streaming archive.
    fn compose_copy_archive(
        &self,
        request: &DockerComposeArchiveRequest,
    ) -> Result<Box<dyn Read>, EvalExecutionError>;

    /// Opens one service path while enforcing its collection deadline.
    fn compose_copy_archive_bounded(
        &self,
        request: &DockerComposeArchiveRequest,
        phase: EvalExecutionPhase,
        deadline: Duration,
    ) -> Result<Box<dyn Read>, EvalExecutionError>;

    /// Copies an explicitly selected immutable snapshot path into one service.
    fn compose_copy_into(
        &self,
        request: &DockerComposeCopyRequest,
    ) -> Result<(), EvalExecutionError>;

    /// Stops one service.
    fn compose_stop_service(
        &self,
        request: &DockerComposeStopRequest,
    ) -> Result<(), EvalExecutionError>;

    /// Stops one service while enforcing a collection deadline.
    fn compose_stop_service_bounded(
        &self,
        request: &DockerComposeStopRequest,
    ) -> Result<(), EvalExecutionError>;

    /// Tears down the project and owned resources.
    fn compose_down(&self, request: &DockerComposeDownRequest) -> Result<(), EvalExecutionError>;

    /// Discovers resources matching the exact task-owned project labels.
    fn compose_owned_resources(
        &self,
        project: &ComposeProjectId,
        deadline: Duration,
    ) -> Result<OwnedComposeResources, EvalExecutionError>;
}

#[cfg(test)]
mod compose_lease_tests {
    use std::{
        cell::{Cell, RefCell},
        collections::BTreeSet,
        rc::Rc,
        time::Duration,
    };

    use super::*;
    use crate::clock::{RealClock, SimClock};
    use crate::eval::execution::{
        compose_project::ComposeProjectLease,
        plan::ComposeProjectPlan,
        task_environment::{ServiceArchiveRequest, TaskEnvironmentLease},
    };

    struct Runtime {
        events: Rc<RefCell<Vec<String>>>,
        down_graces: Rc<RefCell<Vec<Duration>>>,
        advance_after_up: Option<(Rc<SimClock>, i64)>,
    }

    fn reserve<'a>(
        runtime: &'a dyn DockerComposeRuntime,
        plan: &ComposeProjectPlan,
        source_digest: &str,
        project_directory: &str,
        main_image: &str,
    ) -> ComposeProjectLease<'a> {
        ComposeProjectLease::reserve_with_clock(
            runtime,
            RealClock::new(),
            plan,
            source_digest,
            project_directory,
            main_image,
        )
        .unwrap()
    }

    #[test]
    fn terminal_failure_teardown_forces_compose_containers_without_grace() {
        let request = DockerComposeDownRequest::new(
            ComposeProjectId::new("aiperf-fixture"),
            "/tmp/aiperf-fixture",
        )
        .with_terminal_failure()
        .with_deadline(Duration::from_secs(60));

        assert_eq!(request.container_grace(), Duration::ZERO);
        assert_eq!(request.deadline(), Some(Duration::from_secs(60)));
    }

    impl DockerRuntime for Runtime {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::none()
        }
        fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn exec(&self, _: &DockerExecRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn remove(&self, _: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
    }

    impl DockerComposeRuntime for Runtime {
        fn compose_config(
            &self,
            _: &DockerComposeConfigRequest,
        ) -> Result<Vec<u8>, EvalExecutionError> {
            Ok(Vec::new())
        }
        fn compose_build(
            &self,
            request: &DockerComposeBuildRequest,
        ) -> Result<(), EvalExecutionError> {
            assert_eq!(
                request.labels().get("aiperf.project"),
                Some(&request.project().as_str().to_owned())
            );
            assert_ne!(
                request.labels().get("aiperf.project"),
                request.labels().get("aiperf.run")
            );
            assert_eq!(request.deadline(), Some(std::time::Duration::from_secs(1)));
            self.events
                .borrow_mut()
                .push(format!("build:{}", request.project().as_str()));
            Ok(())
        }
        fn compose_up(&self, request: &DockerComposeUpRequest) -> Result<(), EvalExecutionError> {
            assert!(request.detached());
            assert!(request.wait_for_readiness());
            assert_ne!(
                request.labels().get("aiperf.project"),
                request.labels().get("aiperf.run")
            );
            self.events
                .borrow_mut()
                .push(format!("up:{}", request.project().as_str()));
            if let Some((clock, time_ns)) = &self.advance_after_up {
                clock.advance_to(*time_ns);
            }
            Ok(())
        }
        fn compose_exec(&self, _: &DockerComposeExecRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn compose_copy_archive(
            &self,
            _: &DockerComposeArchiveRequest,
        ) -> Result<Box<dyn Read>, EvalExecutionError> {
            unreachable!()
        }
        fn compose_copy_archive_bounded(
            &self,
            request: &DockerComposeArchiveRequest,
            _: EvalExecutionPhase,
            deadline: Duration,
        ) -> Result<Box<dyn Read>, EvalExecutionError> {
            assert!(!deadline.is_zero());
            self.events.borrow_mut().push(format!(
                "archive:{}:{}",
                request.service().as_str(),
                deadline.as_nanos()
            ));
            Ok(Box::new(std::io::Cursor::new(Vec::new())))
        }
        fn compose_copy_into(
            &self,
            request: &DockerComposeCopyRequest,
        ) -> Result<(), EvalExecutionError> {
            self.events.borrow_mut().push(format!(
                "copy:{}:{}",
                request.service().as_str(),
                request.destination()
            ));
            Ok(())
        }
        fn compose_stop_service(
            &self,
            _: &DockerComposeStopRequest,
        ) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn compose_stop_service_bounded(
            &self,
            request: &DockerComposeStopRequest,
        ) -> Result<(), EvalExecutionError> {
            assert!(
                request
                    .deadline()
                    .is_some_and(|deadline| !deadline.is_zero())
            );
            self.events
                .borrow_mut()
                .push(format!("stop:{}", request.service().as_str()));
            Ok(())
        }
        fn compose_down(
            &self,
            request: &DockerComposeDownRequest,
        ) -> Result<(), EvalExecutionError> {
            self.down_graces
                .borrow_mut()
                .push(request.container_grace());
            assert!(request.removes_volumes());
            assert!(request.removes_orphans());
            assert!(request.deadline().is_some_and(|deadline| {
                !deadline.is_zero() && deadline <= std::time::Duration::from_secs(60)
            }));
            self.events
                .borrow_mut()
                .push(format!("down:{}", request.project().as_str()));
            Ok(())
        }
        fn compose_owned_resources(
            &self,
            _: &ComposeProjectId,
            _: Duration,
        ) -> Result<OwnedComposeResources, EvalExecutionError> {
            if self
                .events
                .borrow()
                .iter()
                .any(|event| event.starts_with("down:"))
            {
                Ok(OwnedComposeResources::default())
            } else {
                Ok(OwnedComposeResources::new(
                    vec!["main-id".to_owned(), "api-id".to_owned()],
                    Vec::new(),
                    Vec::new(),
                ))
            }
        }
    }

    #[test]
    fn compose_project_ownership_separates_stable_project_and_unique_run_labels() {
        let project = ComposeProjectId::new("aiperf-fixture");
        let labels = project.ownership_labels();

        assert_eq!(
            labels.get("aiperf.project"),
            Some(&"aiperf-fixture".to_owned())
        );
        assert_ne!(labels.get("aiperf.project"), labels.get("aiperf.run"));
    }

    #[test]
    fn compose_lease_builds_and_starts_exact_reserved_project_once() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime: Rc<dyn DockerComposeRuntime> = Rc::new(Runtime {
            events: Rc::clone(&events),
            down_graces: Rc::new(RefCell::new(Vec::new())),
            advance_after_up: None,
        });
        let plan = ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([
                ComposeServiceName::main(),
                ComposeServiceName::parse("api").unwrap(),
            ]),
            build_timeout: std::time::Duration::from_secs(1),
            startup_timeout: std::time::Duration::from_secs(1),
        };
        let mut lease = reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        );
        assert!(lease.project().as_str().starts_with("aiperf-abcdef012345"));
        lease.start().unwrap();
        assert_eq!(
            &*events.borrow(),
            &[
                format!("build:{}", lease.project().as_str()),
                format!("up:{}", lease.project().as_str())
            ]
        );
        lease.teardown().unwrap();
    }

    #[test]
    fn compose_lease_up_that_returns_after_startup_deadline_uses_terminal_cleanup() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let down_graces = Rc::new(RefCell::new(Vec::new()));
        let clock = Rc::new(SimClock::new());
        let runtime = Runtime {
            events: Rc::clone(&events),
            down_graces: Rc::clone(&down_graces),
            advance_after_up: Some((clock.clone(), Duration::from_secs(1).as_nanos() as i64)),
        };
        let plan = ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([ComposeServiceName::main()]),
            build_timeout: Duration::from_secs(1),
            startup_timeout: Duration::from_secs(1),
        };
        let mut lease = ComposeProjectLease::reserve_with_clock(
            &runtime,
            clock,
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        )
        .unwrap();

        assert!(matches!(
            lease.start(),
            Err(EvalExecutionError::ContainerTeardown { .. })
        ));
        assert!(
            events
                .borrow()
                .iter()
                .any(|event| event.starts_with("down:"))
        );
        assert_eq!(&*down_graces.borrow(), &[Duration::ZERO]);
    }

    #[test]
    fn compose_leases_with_the_same_step_input_keep_project_cleanup_disjoint() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime: Rc<dyn DockerComposeRuntime> = Rc::new(Runtime {
            events: Rc::clone(&events),
            down_graces: Rc::new(RefCell::new(Vec::new())),
            advance_after_up: None,
        });
        let plan = ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([ComposeServiceName::main()]),
            build_timeout: Duration::from_secs(1),
            startup_timeout: Duration::from_secs(1),
        };
        let mut first = reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        );
        let mut second = reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        );
        assert_ne!(first.project(), second.project());
        first.start().unwrap();
        second.start().unwrap();
        let first_project = first.project().as_str().to_owned();
        let second_project = second.project().as_str().to_owned();
        first.teardown().unwrap();
        second.teardown().unwrap();
        assert_eq!(
            events
                .borrow()
                .iter()
                .filter(|event| event.as_str() == format!("down:{first_project}"))
                .count(),
            1
        );
        assert_eq!(
            events
                .borrow()
                .iter()
                .filter(|event| event.as_str() == format!("down:{second_project}"))
                .count(),
            1
        );
    }

    #[test]
    fn compose_lease_uses_explicit_bounded_service_operations() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime: Rc<dyn DockerComposeRuntime> = Rc::new(Runtime {
            events: Rc::clone(&events),
            down_graces: Rc::new(RefCell::new(Vec::new())),
            advance_after_up: None,
        });
        let plan = ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([ComposeServiceName::main()]),
            build_timeout: Duration::from_secs(1),
            startup_timeout: Duration::from_secs(1),
        };
        let mut lease = reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        );
        lease.start().unwrap();
        let main = ComposeServiceName::main();
        let archive = lease.archive(ServiceArchiveRequest {
            service: &main,
            source: "/evidence/result.json",
            deadline: Duration::from_secs(2),
            phase: EvalExecutionPhase::CollectionHook,
        });
        assert!(archive.is_ok());
        lease.stop_main(Duration::from_secs(3)).unwrap();

        assert!(
            events
                .borrow()
                .iter()
                .any(|event| event.starts_with("archive:main:"))
        );
        assert!(events.borrow().iter().any(|event| event == "stop:main"));
    }

    struct CleanupRuntime {
        fail_up: bool,
        down_failures: Cell<usize>,
        down_clears_resources: bool,
        remove_failures: Cell<usize>,
        resources: RefCell<OwnedComposeResources>,
        removals: RefCell<Vec<String>>,
        removal_deadlines: RefCell<Vec<Duration>>,
        discovery_deadlines: RefCell<Vec<Duration>>,
        down_requests: RefCell<Vec<(Duration, Duration)>>,
    }

    impl DockerRuntime for CleanupRuntime {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::none()
        }
        fn build(&self, _: &DockerBuildRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn create(&self, _: &DockerCreateRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn start(&self, _: &DockerStartRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn exec(&self, _: &DockerExecRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn copy(&self, _: &DockerCopyRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError> {
            self.removal_deadlines
                .borrow_mut()
                .push(request.deadline().unwrap_or_default());
            self.removals.borrow_mut().push(
                request
                    .public_arguments()
                    .last()
                    .cloned()
                    .unwrap_or_default(),
            );
            assert!(
                request
                    .deadline()
                    .is_some_and(|deadline| !deadline.is_zero())
            );
            if self.remove_failures.get() > 0 {
                self.remove_failures.set(self.remove_failures.get() - 1);
                Err(EvalExecutionError::ProcessFailure("remove".to_owned()))
            } else {
                Ok(())
            }
        }
    }

    impl DockerComposeRuntime for CleanupRuntime {
        fn compose_config(
            &self,
            _: &DockerComposeConfigRequest,
        ) -> Result<Vec<u8>, EvalExecutionError> {
            Ok(Vec::new())
        }
        fn compose_build(&self, _: &DockerComposeBuildRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn compose_up(&self, _: &DockerComposeUpRequest) -> Result<(), EvalExecutionError> {
            if self.fail_up {
                Err(EvalExecutionError::ProcessFailure("up".to_owned()))
            } else {
                Ok(())
            }
        }
        fn compose_exec(&self, _: &DockerComposeExecRequest) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn compose_copy_archive(
            &self,
            _: &DockerComposeArchiveRequest,
        ) -> Result<Box<dyn Read>, EvalExecutionError> {
            unreachable!()
        }
        fn compose_copy_archive_bounded(
            &self,
            _: &DockerComposeArchiveRequest,
            _: EvalExecutionPhase,
            _: Duration,
        ) -> Result<Box<dyn Read>, EvalExecutionError> {
            Ok(Box::new(std::io::Cursor::new(Vec::new())))
        }
        fn compose_copy_into(
            &self,
            _: &DockerComposeCopyRequest,
        ) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn compose_stop_service(
            &self,
            _: &DockerComposeStopRequest,
        ) -> Result<(), EvalExecutionError> {
            Ok(())
        }
        fn compose_stop_service_bounded(
            &self,
            request: &DockerComposeStopRequest,
        ) -> Result<(), EvalExecutionError> {
            assert!(
                request
                    .deadline()
                    .is_some_and(|deadline| !deadline.is_zero())
            );
            Ok(())
        }
        fn compose_down(
            &self,
            request: &DockerComposeDownRequest,
        ) -> Result<(), EvalExecutionError> {
            self.down_requests.borrow_mut().push((
                request.container_grace(),
                request.deadline().unwrap_or_default(),
            ));
            if self.down_failures.get() > 0 {
                self.down_failures.set(self.down_failures.get() - 1);
                Err(EvalExecutionError::ProcessFailure("down".to_owned()))
            } else {
                if self.down_clears_resources {
                    *self.resources.borrow_mut() = OwnedComposeResources::default();
                }
                Ok(())
            }
        }
        fn compose_owned_resources(
            &self,
            _: &ComposeProjectId,
            deadline: Duration,
        ) -> Result<OwnedComposeResources, EvalExecutionError> {
            self.discovery_deadlines.borrow_mut().push(deadline);
            std::thread::sleep(Duration::from_millis(1));
            Ok(self.resources.borrow().clone())
        }
    }

    fn cleanup_plan() -> ComposeProjectPlan {
        ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([ComposeServiceName::main()]),
            build_timeout: std::time::Duration::from_secs(1),
            startup_timeout: std::time::Duration::from_secs(30),
        }
    }

    #[test]
    fn compose_lease_start_failure_cleans_owned_resources_without_caller_teardown() {
        let runtime = Rc::new(CleanupRuntime {
            fail_up: true,
            down_failures: Cell::new(0),
            down_clears_resources: true,
            remove_failures: Cell::new(0),
            resources: RefCell::new(OwnedComposeResources::new(
                vec!["partial".to_owned()],
                Vec::new(),
                Vec::new(),
            )),
            removals: RefCell::new(Vec::new()),
            removal_deadlines: RefCell::new(Vec::new()),
            discovery_deadlines: RefCell::new(Vec::new()),
            down_requests: RefCell::new(Vec::new()),
        });
        let mut lease = reserve(runtime.as_ref(), &cleanup_plan(), "abcdef", "/tmp", "main");
        assert!(lease.start().is_err());
        assert_eq!(
            *runtime.resources.borrow(),
            OwnedComposeResources::default()
        );
        let requests = runtime.down_requests.borrow();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].0, Duration::ZERO);
        assert!(
            requests[0].1
                <= crate::eval::execution::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE
        );
        assert!(runtime.discovery_deadlines.borrow().iter().all(|deadline| {
            *deadline <= crate::eval::execution::compose_project::TERMINAL_COMPOSE_CLEANUP_DEADLINE
        }));
    }

    #[test]
    fn compose_lease_drop_retries_failed_start_cleanup_before_releasing_ownership() {
        let runtime = Rc::new(CleanupRuntime {
            fail_up: true,
            down_failures: Cell::new(1),
            down_clears_resources: true,
            remove_failures: Cell::new(0),
            resources: RefCell::new(OwnedComposeResources::new(
                vec!["partial".to_owned()],
                Vec::new(),
                Vec::new(),
            )),
            removals: RefCell::new(Vec::new()),
            removal_deadlines: RefCell::new(Vec::new()),
            discovery_deadlines: RefCell::new(Vec::new()),
            down_requests: RefCell::new(Vec::new()),
        });
        {
            let mut lease = reserve(runtime.as_ref(), &cleanup_plan(), "abcdef", "/tmp", "main");
            assert!(lease.start().is_err());
        }
        assert_eq!(
            *runtime.resources.borrow(),
            OwnedComposeResources::default()
        );
    }

    #[test]
    fn compose_lease_failed_down_is_retryable_and_does_not_mark_down() {
        let runtime = Rc::new(CleanupRuntime {
            fail_up: false,
            down_failures: Cell::new(1),
            down_clears_resources: true,
            remove_failures: Cell::new(0),
            resources: RefCell::new(OwnedComposeResources::default()),
            removals: RefCell::new(Vec::new()),
            removal_deadlines: RefCell::new(Vec::new()),
            discovery_deadlines: RefCell::new(Vec::new()),
            down_requests: RefCell::new(Vec::new()),
        });
        let mut lease = reserve(runtime.as_ref(), &cleanup_plan(), "abcdef", "/tmp", "main");
        lease.start().unwrap();
        assert!(lease.teardown().is_err());
        lease.teardown().unwrap();
    }

    #[test]
    fn compose_lease_drop_retries_terminal_down_without_grace() {
        let runtime = Rc::new(CleanupRuntime {
            fail_up: false,
            down_failures: Cell::new(1),
            down_clears_resources: true,
            remove_failures: Cell::new(0),
            resources: RefCell::new(OwnedComposeResources::default()),
            removals: RefCell::new(Vec::new()),
            removal_deadlines: RefCell::new(Vec::new()),
            discovery_deadlines: RefCell::new(Vec::new()),
            down_requests: RefCell::new(Vec::new()),
        });
        {
            let mut lease = reserve(runtime.as_ref(), &cleanup_plan(), "abcdef", "/tmp", "main");
            lease.start().unwrap();
            assert!(
                lease
                    .teardown_after_terminal_failure(Duration::from_secs(3))
                    .is_err()
            );
        }
        let requests = runtime.down_requests.borrow();
        assert_eq!(requests.len(), 2);
        assert!(requests.iter().all(|(grace, deadline)| {
            *grace == Duration::ZERO && !deadline.is_zero() && *deadline <= Duration::from_secs(3)
        }));
    }

    #[test]
    fn compose_lease_attempts_every_exact_resource_when_one_forced_remove_fails() {
        let runtime = Rc::new(CleanupRuntime {
            fail_up: false,
            down_failures: Cell::new(0),
            down_clears_resources: false,
            remove_failures: Cell::new(1),
            resources: RefCell::new(OwnedComposeResources::new(
                vec!["one".to_owned(), "two".to_owned()],
                vec!["network".to_owned()],
                vec!["volume".to_owned()],
            )),
            removals: RefCell::new(Vec::new()),
            removal_deadlines: RefCell::new(Vec::new()),
            discovery_deadlines: RefCell::new(Vec::new()),
            down_requests: RefCell::new(Vec::new()),
        });
        let mut lease = reserve(runtime.as_ref(), &cleanup_plan(), "abcdef", "/tmp", "main");
        lease.start().unwrap();
        assert!(lease.teardown().is_err());
        assert_eq!(
            &*runtime.removals.borrow(),
            &["one", "two", "network", "volume"]
        );
    }

    #[test]
    fn compose_lease_terminal_cleanup_consumes_one_deadline_across_removals() {
        let runtime = Rc::new(CleanupRuntime {
            fail_up: false,
            down_failures: Cell::new(0),
            down_clears_resources: false,
            remove_failures: Cell::new(0),
            resources: RefCell::new(OwnedComposeResources::new(
                vec!["one".to_owned(), "two".to_owned()],
                Vec::new(),
                Vec::new(),
            )),
            removals: RefCell::new(Vec::new()),
            removal_deadlines: RefCell::new(Vec::new()),
            discovery_deadlines: RefCell::new(Vec::new()),
            down_requests: RefCell::new(Vec::new()),
        });
        let mut lease = reserve(runtime.as_ref(), &cleanup_plan(), "abcdef", "/tmp", "main");
        lease.start().unwrap();
        assert!(
            lease
                .teardown_after_terminal_failure(Duration::from_secs(3))
                .is_err()
        );
        let deadlines = runtime.removal_deadlines.borrow();
        assert_eq!(deadlines.len(), 2);
        assert!(deadlines[0] <= Duration::from_secs(3));
        assert!(deadlines[1] < deadlines[0]);
        let discoveries = runtime.discovery_deadlines.borrow();
        assert!(discoveries.len() >= 3);
        let first_terminal_discovery = discoveries.len() - 2;
        assert!(discoveries[first_terminal_discovery] <= Duration::from_secs(3));
        assert!(discoveries[first_terminal_discovery + 1] < discoveries[first_terminal_discovery]);
    }
}
