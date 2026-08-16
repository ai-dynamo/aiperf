// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Native sandbox recipes, agent contracts, and immutable workspace overlays.

mod agent;
mod artifacts;
mod compose_policy;
mod compose_project;
mod coordinator;
mod docker_process;
mod docker_runtime;
mod local_process;
mod multi_step;
mod plan;
mod recipe;
mod task_environment;
mod workspace;

pub use agent::{
    AgentCapability, EnvName, EvalExecutionError, EvalExecutionPhase, EvalSandboxFactory,
    HarborAgentContract, SecretProvider, SecretValue,
};
pub use artifacts::{collect_artifacts, transfer_artifacts};
pub use coordinator::{
    HarborCompletedEvaluation, HarborEvaluationCoordinator, HarborEvaluationError,
    HarborLocalEvaluationRequest,
};
pub use docker_process::DockerProcessSandbox;
#[allow(unused_imports)]
pub use docker_runtime::{
    ComposeProjectId, DockerBuildRequest, DockerComposeArchiveRequest, DockerComposeBuildRequest,
    DockerComposeConfigRequest, DockerComposeCopyRequest, DockerComposeDownRequest,
    DockerComposeExecRequest, DockerComposeRuntime, DockerComposeStopRequest,
    DockerComposeUpRequest, DockerCopyRequest, DockerCreateRequest, DockerEnvironment,
    DockerExecRequest, DockerRemoveRequest, DockerRuntime, DockerStartRequest,
    OwnedComposeResources, preflight_docker, resolve_environment, resolve_phase_environment,
};
pub use local_process::{
    LocalExecutionResult, LocalProcessSandbox, MaterializedSandbox, ProcessOutput, SandboxRole,
};
pub use multi_step::{MultiStepExecutionResult, StepExecutionResult};
pub use plan::{
    ArtifactSpec, BenchmarkExecutionPlan, BenchmarkStepPlan, ComposeProjectPlan,
    ComposeServiceName, ContainerResources, EnvBinding, EnvironmentPlan, HealthcheckPlan,
    ImageSource, ImageSourceKind, MultiStepRewardStrategy, NetworkPolicy, PhasePlan,
    ProviderCapabilities, VerifierCollectHook, VerifierPlan,
};
pub(crate) use plan::{
    CanonicalPackagePlan, append_identity_field, artifact_source_overlaps_reserved_verifier_path,
    shared_workdir_conflicts_reserved_verifier_path, validate_env_name, validate_user,
    verifier_artifact_target_collision,
};
pub use recipe::HarborSandboxRecipe;
pub use workspace::{ImmutablePatch, WorkspaceOverlay};
