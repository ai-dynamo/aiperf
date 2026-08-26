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
mod native_graph_episode;
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
    HarborLifecycleAgentContract, HarborLifecycleRequest, HarborLifecycleScoreRequest,
    HarborLocalEvaluationRequest,
};
pub use docker_process::DockerProcessSandbox;
pub(crate) use docker_runtime::prepare_external_driver_spawn;
pub(crate) use docker_runtime::resolve_native_graph_adapter_authorization;
#[allow(unused_imports)]
pub use docker_runtime::{
    AuthorizedExternalDriverSpawn, ComposeProjectId, DockerAdapterLease, DockerAdapterProcess,
    DockerAdapterSpawnerRequest, DockerBuildRequest, DockerComposeAdapterSpawnerRequest,
    DockerComposeArchiveRequest, DockerComposeBuildRequest, DockerComposeConfigRequest,
    DockerComposeCopyRequest, DockerComposeDownRequest, DockerComposeExecRequest,
    DockerComposeRuntime, DockerComposeStopRequest, DockerComposeUpRequest, DockerCopyRequest,
    DockerCreateRequest, DockerEnvironment, DockerExecRequest, DockerRemoveRequest, DockerRuntime,
    DockerStartRequest, ExternalDriverDockerSpawnOperation, ExternalDriverDockerSpawner,
    ExternalDriverSpawnExecutor, OwnedComposeResources, StartedExternalDriverDockerSpawn,
    preflight_docker, resolve_environment, resolve_phase_environment,
};
pub use local_process::{
    LocalAdapterSpawner, LocalExecutionResult, LocalProcessSandbox, MaterializedSandbox,
    ProcessOutput, SandboxRole,
};
pub use multi_step::{MultiStepExecutionResult, StepExecutionResult};
pub use native_graph_episode::{
    NativeGraphEnvironmentAdapterStart, NativeGraphEpisodeBackendLease, NativeGraphEpisodeCallback,
    NativeGraphEpisodeLease, run_native_graph_episode_callback,
};
#[cfg(feature = "engine")]
pub use native_graph_episode::{
    NativeGraphEnvironmentRolloutSession, NativeGraphLeaseRolloutStart,
};
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
