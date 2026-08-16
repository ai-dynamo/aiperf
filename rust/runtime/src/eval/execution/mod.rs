// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Native sandbox recipes, agent contracts, and immutable workspace overlays.

mod agent;
mod artifacts;
mod coordinator;
mod docker_process;
mod docker_runtime;
mod local_process;
mod multi_step;
mod plan;
mod recipe;
mod workspace;

pub use agent::{
    AgentCapability, EnvName, EvalExecutionError, EvalExecutionPhase, EvalSandboxFactory,
    HarborAgentContract, SecretProvider, SecretValue,
};
pub use artifacts::{collect_artifacts, transfer_artifacts};
pub use coordinator::{HarborEvaluationCoordinator, HarborEvaluationError};
pub use docker_process::DockerProcessSandbox;
pub use docker_runtime::{
    DockerBuildRequest, DockerCopyRequest, DockerCreateRequest, DockerEnvironment,
    DockerExecRequest, DockerRemoveRequest, DockerRuntime, DockerStartRequest, preflight_docker,
    resolve_environment, resolve_phase_environment,
};
pub use local_process::{
    LocalExecutionResult, LocalProcessSandbox, MaterializedSandbox, ProcessOutput, SandboxRole,
};
pub use multi_step::{MultiStepExecutionResult, StepExecutionResult};
pub use plan::{
    ArtifactSpec, BenchmarkExecutionPlan, BenchmarkStepPlan, ContainerResources, EnvBinding,
    EnvironmentPlan, HealthcheckPlan, ImageSource, ImageSourceKind, MultiStepRewardStrategy,
    NetworkPolicy, PhasePlan, ProviderCapabilities, VerifierPlan,
};
pub(crate) use plan::{validate_env_name, validate_user};
pub use recipe::HarborSandboxRecipe;
pub use workspace::{ImmutablePatch, WorkspaceOverlay};
