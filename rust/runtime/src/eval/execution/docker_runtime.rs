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
pub fn preflight_compose_configuration(
    runtime: &dyn DockerRuntime,
    plan: &BenchmarkExecutionPlan,
    environment_root: &Path,
    request: &DockerComposeConfigRequest,
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
        plan.environment(),
        workspace,
    )?;
    if request.generated_definition() != rendered.bytes() {
        return Err(EvalExecutionError::InvalidRecipe(
            "generated Compose definition",
        ));
    }
    let compose_runtime =
        runtime
            .compose_runtime()
            .ok_or(EvalExecutionError::UnsupportedEnforcement(
                "Docker Compose runtime",
            ))?;
    let canonical = compose_runtime.compose_config(request)?;
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
        }

        impl $name {
            /// Creates a request scoped to one task-owned Compose project.
            pub fn new(project: ComposeProjectId, project_directory: impl Into<PathBuf>) -> Self {
                Self {
                    project,
                    project_directory: project_directory.into(),
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
        }
    };
}

compose_project_request!(
    DockerComposeBuildRequest,
    "A Compose project build request."
);
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
}

impl DockerComposeArchiveRequest {
    /// Creates an archive request for one service path.
    pub fn new(
        project: ComposeProjectId,
        service: ComposeServiceName,
        source: impl Into<String>,
    ) -> Self {
        Self {
            project,
            service,
            source: source.into(),
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
}

/// A request to stop one service in a task-owned Compose project.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DockerComposeStopRequest {
    project: ComposeProjectId,
    service: ComposeServiceName,
}

impl DockerComposeStopRequest {
    /// Creates a service stop request.
    pub fn new(project: ComposeProjectId, service: ComposeServiceName) -> Self {
        Self { project, service }
    }
    /// Returns the task-owned project identifier.
    pub fn project(&self) -> &ComposeProjectId {
        &self.project
    }
    /// Returns the target service.
    pub fn service(&self) -> &ComposeServiceName {
        &self.service
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
}

impl DockerRemoveRequest {
    /// Creates a removal request from non-secret Docker arguments.
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

    /// Stops one service.
    fn compose_stop_service(
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
