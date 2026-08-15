// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Agent capability contracts and fail-closed sandbox preflight.

use std::fmt::{self, Display, Formatter};

use super::HarborSandboxRecipe;

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
    /// The selected provider cannot honor a required contract capability.
    MissingCapability(AgentCapability),
    /// An immutable workspace or artifact identity was invalid.
    InvalidWorkspace(String),
}

impl Display for EvalExecutionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRecipe(field) => write!(formatter, "invalid sandbox recipe {field}"),
            Self::MissingCapability(capability) => write!(formatter, "missing sandbox capability {capability:?}"),
            Self::InvalidWorkspace(reason) => write!(formatter, "invalid workspace {reason}"),
        }
    }
}

impl std::error::Error for EvalExecutionError {}
