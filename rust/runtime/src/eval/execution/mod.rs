// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Native sandbox recipes, agent contracts, and immutable workspace overlays.

mod agent;
mod coordinator;
mod docker_process;
mod local_process;
mod recipe;
mod workspace;

pub use agent::{
    AgentCapability, EvalExecutionError, EvalExecutionPhase, EvalSandboxFactory,
    HarborAgentContract,
};
pub use coordinator::{HarborEvaluationCoordinator, HarborEvaluationError};
pub use docker_process::DockerProcessSandbox;
pub use local_process::{
    LocalExecutionResult, LocalProcessSandbox, MaterializedSandbox, ProcessOutput, SandboxRole,
};
pub use recipe::HarborSandboxRecipe;
pub use workspace::{ImmutablePatch, WorkspaceOverlay};
