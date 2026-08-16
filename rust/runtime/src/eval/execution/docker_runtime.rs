// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Redacted, injectable Docker command contracts for benchmark execution.

use std::{collections::BTreeMap, time::Duration};

use super::{
    BenchmarkExecutionPlan, EnvName, EnvironmentPlan, EvalExecutionError, EvalExecutionPhase,
    PhasePlan, ProviderCapabilities, SecretProvider, SecretValue,
};

/// Preflights Docker enforcement before the executor can build an image.
pub fn preflight_docker(
    runtime: &dyn DockerRuntime,
    plan: &BenchmarkExecutionPlan,
) -> Result<(), EvalExecutionError> {
    plan.validate_for(runtime.capabilities())
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

    /// Removes a container, image, or related lease.
    fn remove(&self, request: &DockerRemoveRequest) -> Result<(), EvalExecutionError>;
}
