// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Agent capability contracts and fail-closed sandbox preflight.

use std::{
    fmt::{self, Display, Formatter},
    time::Duration,
};

use super::HarborSandboxRecipe;

/// A validated environment-variable name.
pub type EnvName = String;

/// A host secret resolved immediately before a command is executed.
#[derive(Clone, PartialEq, Eq)]
pub struct SecretValue(String);

impl SecretValue {
    /// Creates a secret value that redacts itself in diagnostics.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub(crate) fn exposed(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for SecretValue {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("SecretValue([REDACTED])")
    }
}

impl Display for SecretValue {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str("[REDACTED]")
    }
}

/// Resolves named host secrets for the command phase that requires them.
pub trait SecretProvider {
    /// Resolves one declared secret reference without exposing it in diagnostics.
    fn resolve(&self, name: &EnvName) -> Result<SecretValue, EvalExecutionError>;
}

/// A capability that an agent contract requires from its sandbox provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentCapability {
    /// The base workspace cannot be modified by an agent branch.
    ReadOnlyBase,
    /// Branches receive an isolated copy-on-write workspace.
    OverlayWorkspace,
}

/// A native evaluation agent selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HarborAgentContract {
    /// A command supplied by the caller.
    External { required: Vec<AgentCapability> },
    /// An agent installed into the resolved sandbox image.
    Installed { required: Vec<AgentCapability> },
    /// A native graph agent executed by the runtime.
    NativeGraph { required: Vec<AgentCapability> },
}

/// An evaluation phase that owns a sandbox command timeout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvalExecutionPhase {
    /// Readiness validation before the task agent is allowed to run.
    Healthcheck,
    /// The externally supplied agent command.
    Agent,
    /// The task-supplied verifier command.
    Verifier,
}

impl Display for EvalExecutionPhase {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthcheck => formatter.write_str("healthcheck"),
            Self::Agent => formatter.write_str("agent"),
            Self::Verifier => formatter.write_str("verifier"),
        }
    }
}

impl HarborAgentContract {
    /// Creates an installed-agent contract.
    pub fn installed(required: Vec<AgentCapability>) -> Self {
        Self::Installed { required }
    }

    /// Returns every capability required by this contract.
    pub fn required_capabilities(&self) -> &[AgentCapability] {
        match self {
            Self::External { required }
            | Self::Installed { required }
            | Self::NativeGraph { required } => required,
        }
    }
}

/// Provider boundary for capability preflight and environment opening.
pub trait EvalSandboxFactory {
    /// Returns capabilities guaranteed by this factory.
    fn capabilities(&self) -> &[AgentCapability];

    /// Opens an already preflighted evaluation environment.
    fn open(&self, recipe: &HarborSandboxRecipe) -> Result<(), EvalExecutionError>;

    /// Rejects missing capabilities before environment opening.
    fn preflight(
        &self,
        _: &HarborSandboxRecipe,
        contract: &HarborAgentContract,
    ) -> Result<(), EvalExecutionError> {
        for required in contract.required_capabilities() {
            if !self.capabilities().contains(required) {
                return Err(EvalExecutionError::MissingCapability(*required));
            }
        }
        Ok(())
    }
}

/// Failed native sandbox validation or capability preflight.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EvalExecutionError {
    /// A sandbox recipe lacked a required immutable field.
    InvalidRecipe(&'static str),
    /// A synchronous sandbox operation was invoked from an incompatible async runtime.
    RuntimeContext(&'static str),
    /// The selected provider cannot honor a required contract capability.
    MissingCapability(AgentCapability),
    /// The selected provider cannot enforce an authored benchmark requirement.
    UnsupportedEnforcement(&'static str),
    /// A declared host-secret reference could not be resolved for the active phase.
    MissingSecret(String),
    /// The task environment did not become ready before its agent phase.
    Unhealthy(String),
    /// Declared task artifacts could not be collected safely.
    ArtifactCollection(String),
    /// An immutable workspace or artifact identity was invalid.
    InvalidWorkspace(String),
    /// The requested local process command was not a nonempty argv.
    InvalidCommand,
    /// Materializing an immutable package into a local sandbox failed.
    Materialization(String),
    /// Starting a local sandbox process failed.
    ProcessSpawn(String),
    /// A local sandbox process returned a non-success status.
    ProcessFailure(String),
    /// A phase exceeded its configured execution limit after its container was removed.
    Timeout {
        /// The command phase that exceeded its limit.
        phase: EvalExecutionPhase,
        /// The configured execution limit.
        timeout: Duration,
    },
    /// Explicit removal of a timed-out container could not be verified.
    ContainerTeardown {
        /// The container that may still be running.
        container: String,
        /// The failed Docker operation.
        reason: String,
    },
    /// The Docker exec client could not be conclusively terminated after its container ended.
    TerminalUncertainty {
        /// The timed-out command phase.
        phase: EvalExecutionPhase,
        /// The removed container that owned the phase.
        container: String,
        /// The unresolved host-client operation.
        reason: String,
    },
}

impl Display for EvalExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRecipe(field) => write!(formatter, "invalid sandbox recipe {field}"),
            Self::RuntimeContext(operation) => {
                write!(formatter, "invalid runtime context for {operation}")
            }
            Self::MissingCapability(capability) => {
                write!(formatter, "missing sandbox capability {capability:?}")
            }
            Self::UnsupportedEnforcement(requirement) => {
                write!(
                    formatter,
                    "provider cannot enforce benchmark requirement {requirement}"
                )
            }
            Self::MissingSecret(name) => write!(formatter, "missing required secret {name}"),
            Self::Unhealthy(reason) => write!(formatter, "task environment is unhealthy: {reason}"),
            Self::ArtifactCollection(reason) => {
                write!(formatter, "artifact collection failed: {reason}")
            }
            Self::InvalidWorkspace(reason) => write!(formatter, "invalid workspace {reason}"),
            Self::InvalidCommand => formatter.write_str("sandbox command must be a nonempty argv"),
            Self::Materialization(reason) => {
                write!(formatter, "sandbox materialization failed: {reason}")
            }
            Self::ProcessSpawn(command) => {
                write!(formatter, "failed to start sandbox command {command:?}")
            }
            Self::ProcessFailure(command) => {
                write!(formatter, "sandbox command failed: {command:?}")
            }
            Self::Timeout { phase, timeout } => {
                write!(formatter, "{phase} phase timed out after {timeout:?}")
            }
            Self::ContainerTeardown { container, reason } => {
                write!(
                    formatter,
                    "failed to remove timed-out container {container:?}: {reason}"
                )
            }
            Self::TerminalUncertainty {
                phase,
                container,
                reason,
            } => {
                write!(
                    formatter,
                    "{phase} phase terminal state is uncertain after removing container {container:?}: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for EvalExecutionError {}
