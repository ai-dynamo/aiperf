// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Redacted, injectable Docker command contracts for benchmark execution.

use std::{
    collections::BTreeMap,
    io::Read,
    path::{Path, PathBuf},
    time::Duration,
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

/// Runs the read-only provider configuration check for a Compose-backed plan.
///
/// Callers must pass paths below an acquired package materialization. This
/// function never builds, creates, or starts project resources.
pub(crate) fn preflight_compose_configuration(
    runtime: &dyn DockerRuntime,
    plan: &BenchmarkExecutionPlan,
    environment: &EnvironmentPlan,
    environment_root: &Path,
    project: ComposeProjectId,
    project_directory: &Path,
    image_tag: &str,
    project_labels: &BTreeMap<String, String>,
    workspace: &Path,
    authored_overlay: &[u8],
) -> Result<(), EvalExecutionError> {
    preflight_docker(runtime, plan)?;
    let compose_plan = plan
        .compose()
        .ok_or(EvalExecutionError::InvalidRecipe("Compose project plan"))?;
    let authored = super::compose_policy::validate_authored_compose(
        authored_overlay,
        compose_plan,
        environment_root,
    )?;
    let rendered = super::compose_policy::render_generated_main_compose(
        image_tag,
        project_labels,
        environment,
        workspace,
    )?;
    let request = DockerComposeConfigRequest::new(
        project,
        project_directory,
        rendered.bytes().to_vec(),
        project_directory.join(compose_plan.definition_path()),
    );
    let compose_runtime =
        runtime
            .compose_runtime()
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "Docker Compose runtime",
            ))?;
    let canonical = compose_runtime.compose_config(&request)?;
    super::compose_policy::validate_provider_compose_config(
        &canonical,
        compose_plan,
        environment_root,
        &rendered,
        &authored,
    )
    .map(|_| ())
}

/// An opaque task-owned Docker Compose project identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ComposeProjectId(String);

impl ComposeProjectId {
    /// Creates an opaque project identifier selected by the evaluation runtime.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// Returns the project identifier for provider argument construction.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns the exact labels used to select resources owned by this run.
    pub fn ownership_labels(&self) -> BTreeMap<String, String> {
        BTreeMap::from([
            ("aiperf.project".to_owned(), self.0.clone()),
            ("aiperf.run".to_owned(), self.0.clone()),
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
        Duration::from_secs(10)
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
}

impl DockerBuildRequest {
    /// Creates a request from non-secret Docker arguments.
    pub fn new(arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            network_lease: None,
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
}

/// A redacted Docker container-create request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerCreateRequest {
    public_arguments: Vec<String>,
    network_lease: Option<String>,
}

impl DockerCreateRequest {
    /// Creates a request from non-secret Docker arguments.
    pub fn new(arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            public_arguments: arguments.into_iter().map(Into::into).collect(),
            network_lease: None,
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
}

/// A Docker container-start request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerStartRequest {
    container: String,
}

impl DockerStartRequest {
    /// Creates a container-start request.
    pub fn new(container: impl Into<String>) -> Self {
        Self {
            container: container.into(),
        }
    }

    /// Returns the container identifier.
    pub fn container(&self) -> &str {
        &self.container
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
}

impl DockerCopyRequest {
    /// Creates a copy request from non-secret Docker arguments.
    pub fn new(arguments: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            public_arguments: arguments.into_iter().map(Into::into).collect(),
        }
    }

    /// Returns Docker arguments that may appear in diagnostics.
    pub fn public_arguments(&self) -> &[String] {
        &self.public_arguments
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

    /// Builds the requested immutable task environment.
    fn build(&self, request: &DockerBuildRequest) -> Result<(), EvalExecutionError>;

    /// Creates the requested container.
    fn create(&self, request: &DockerCreateRequest) -> Result<(), EvalExecutionError>;

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

    /// Opens a service path as a streaming archive.
    fn compose_copy_archive(
        &self,
        request: &DockerComposeArchiveRequest,
    ) -> Result<Box<dyn Read>, EvalExecutionError>;

    /// Opens one service path while enforcing its collection deadline.
    fn compose_copy_archive_bounded(
        &self,
        request: &DockerComposeArchiveRequest,
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
    use crate::eval::execution::{
        compose_project::ComposeProjectLease,
        plan::ComposeProjectPlan,
        task_environment::{ServiceArchiveRequest, TaskEnvironmentLease},
    };

    struct Runtime {
        events: Rc<RefCell<Vec<String>>>,
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
            assert_eq!(request.deadline(), Some(std::time::Duration::from_secs(1)));
            self.events
                .borrow_mut()
                .push(format!("build:{}", request.project().as_str()));
            Ok(())
        }
        fn compose_up(&self, request: &DockerComposeUpRequest) -> Result<(), EvalExecutionError> {
            assert!(request.detached());
            assert!(request.wait_for_readiness());
            assert_eq!(
                request.labels().get("aiperf.run"),
                Some(&request.project().as_str().to_owned())
            );
            self.events
                .borrow_mut()
                .push(format!("up:{}", request.project().as_str()));
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
            assert_eq!(
                request.container_grace(),
                std::time::Duration::from_secs(10)
            );
            assert!(request.removes_volumes());
            assert!(request.removes_orphans());
            assert_eq!(request.deadline(), Some(std::time::Duration::from_secs(60)));
            self.events
                .borrow_mut()
                .push(format!("down:{}", request.project().as_str()));
            Ok(())
        }
        fn compose_owned_resources(
            &self,
            _: &ComposeProjectId,
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
    fn compose_lease_builds_and_starts_exact_reserved_project_once() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime: Rc<dyn DockerComposeRuntime> = Rc::new(Runtime {
            events: Rc::clone(&events),
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
        let mut lease = ComposeProjectLease::reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        )
        .unwrap();
        assert!(lease.project().as_str().starts_with("aiperf-abcdef012345"));
        lease.start().unwrap();
        assert_eq!(
            lease.state(),
            super::super::compose_project::ComposeLeaseState::Started
        );
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
    fn compose_leases_with_the_same_step_input_keep_project_cleanup_disjoint() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let runtime: Rc<dyn DockerComposeRuntime> = Rc::new(Runtime {
            events: Rc::clone(&events),
        });
        let plan = ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([ComposeServiceName::main()]),
            build_timeout: Duration::from_secs(1),
            startup_timeout: Duration::from_secs(1),
        };
        let mut first = ComposeProjectLease::reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        )
        .unwrap();
        let mut second = ComposeProjectLease::reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        )
        .unwrap();
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
        });
        let plan = ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([ComposeServiceName::main()]),
            build_timeout: Duration::from_secs(1),
            startup_timeout: Duration::from_secs(1),
        };
        let mut lease = ComposeProjectLease::reserve(
            runtime.as_ref(),
            &plan,
            "abcdef0123456789",
            "/tmp",
            "main:image",
        )
        .unwrap();
        lease.start().unwrap();
        let main = ComposeServiceName::main();
        let archive = lease.archive(ServiceArchiveRequest {
            service: &main,
            source: "/evidence/result.json",
            deadline: Duration::from_secs(2),
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
            self.removals.borrow_mut().push(
                request
                    .public_arguments()
                    .last()
                    .cloned()
                    .unwrap_or_default(),
            );
            assert_eq!(request.deadline(), Some(std::time::Duration::from_secs(10)));
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
        fn compose_down(&self, _: &DockerComposeDownRequest) -> Result<(), EvalExecutionError> {
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
        ) -> Result<OwnedComposeResources, EvalExecutionError> {
            Ok(self.resources.borrow().clone())
        }
    }

    fn cleanup_plan() -> ComposeProjectPlan {
        ComposeProjectPlan {
            definition_path: "environment/docker-compose.yaml".to_owned(),
            services: BTreeSet::from([ComposeServiceName::main()]),
            build_timeout: std::time::Duration::from_secs(1),
            startup_timeout: std::time::Duration::from_secs(1),
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
        });
        let mut lease = ComposeProjectLease::reserve(
            runtime.as_ref(),
            &cleanup_plan(),
            "abcdef",
            "/tmp",
            "main",
        )
        .unwrap();
        assert!(lease.start().is_err());
        assert_eq!(
            *runtime.resources.borrow(),
            OwnedComposeResources::default()
        );
        assert_eq!(
            lease.state(),
            super::super::compose_project::ComposeLeaseState::Down
        );
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
        });
        {
            let mut lease = ComposeProjectLease::reserve(
                runtime.as_ref(),
                &cleanup_plan(),
                "abcdef",
                "/tmp",
                "main",
            )
            .unwrap();
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
        });
        let mut lease = ComposeProjectLease::reserve(
            runtime.as_ref(),
            &cleanup_plan(),
            "abcdef",
            "/tmp",
            "main",
        )
        .unwrap();
        lease.start().unwrap();
        assert!(lease.teardown().is_err());
        assert_eq!(
            lease.state(),
            super::super::compose_project::ComposeLeaseState::Started
        );
        lease.teardown().unwrap();
        assert_eq!(
            lease.state(),
            super::super::compose_project::ComposeLeaseState::Down
        );
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
        });
        let mut lease = ComposeProjectLease::reserve(
            runtime.as_ref(),
            &cleanup_plan(),
            "abcdef",
            "/tmp",
            "main",
        )
        .unwrap();
        lease.start().unwrap();
        assert!(lease.teardown().is_err());
        assert_eq!(
            &*runtime.removals.borrow(),
            &["one", "two", "network", "volume"]
        );
        assert_eq!(
            lease.state(),
            super::super::compose_project::ComposeLeaseState::Started
        );
    }
}
