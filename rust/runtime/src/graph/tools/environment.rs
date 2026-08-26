// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure environment-recipe resolution and capability descriptions.

use std::error::Error;
use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};

use crate::graph::driver::{ReplayTaskIdentity, TraceEnvironmentSpec};

use super::workspace::WorkspaceSpec;

/// Failure while resolving or preparing one replay environment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceEnvironmentError(String);

impl TraceEnvironmentError {
    /// Construct a contextual environment boundary failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for TraceEnvironmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TraceEnvironmentError {}

/// Pure resolver for transportable trace environment specifications.
pub trait TraceEnvironmentResolver: Send + Sync {
    /// Resolve a task identity before placement provisions any worker resources.
    fn resolve(
        &self,
        task: &ReplayTaskIdentity,
    ) -> Result<TraceEnvironmentSpec, TraceEnvironmentError>;
}

/// Stock recipe family selected from a recorded task identity.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub enum EnvironmentRecipe {
    /// A staged PinchBench workspace mounted at `/workspace`.
    #[serde(rename = "pinchbench")]
    PinchBench,
    /// A SWE-Bench task rooted at `/testbed`, image-native unless a local replay stages it.
    #[serde(rename = "swebench")]
    SweBench,
}

/// Concrete execution backend selected while resolving a trace recipe.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolExecutionBackend {
    /// Execute in a host-local persistent shell.
    Local,
    /// Execute in an isolated Docker container.
    Docker,
}

/// Fully resolved recipe used only while composing worker-local resources.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResolvedTraceEnvironment {
    /// Recipe family.
    pub kind: EnvironmentRecipe,
    /// Fully resolved execution backend; workers never infer this from image text.
    pub backend: ToolExecutionBackend,
    /// Fully selected container image.
    pub image: String,
    /// Staged or image-native workspace policy.
    pub workspace: WorkspaceSpec,
}

/// Capability requirements checked before a sandbox is created.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ToolSandboxCapabilities {
    /// The backend retains filesystem state across commands.
    pub has_persistent_workspace: bool,
    /// The backend can materialize uploaded fixture files.
    pub has_workspace_materialization: bool,
    /// The backend can enforce a disabled network.
    pub has_network_disabled: bool,
    /// The backend can kill timed-out commands and their descendants.
    pub has_timeout_recycle: bool,
}

impl ToolSandboxCapabilities {
    /// Refuse a backend that cannot enforce this recipe's required isolation.
    pub fn validate(self, recipe: &ResolvedTraceEnvironment) -> Result<(), TraceEnvironmentError> {
        if !self.has_persistent_workspace || !self.has_timeout_recycle {
            return Err(TraceEnvironmentError::new(
                "tool sandbox cannot provide persistent workspace and timeout recycle",
            ));
        }
        if recipe.backend == ToolExecutionBackend::Docker && !self.has_network_disabled {
            return Err(TraceEnvironmentError::new(
                "Docker tool sandbox cannot enforce disabled network",
            ));
        }
        if recipe.workspace.mount_workspace && !self.has_workspace_materialization {
            return Err(TraceEnvironmentError::new(
                "tool sandbox cannot materialize the staged Pinch workspace",
            ));
        }
        Ok(())
    }
}
