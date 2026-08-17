// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable evaluation identities and evidence contracts.

mod artifact_manifest;
mod evidence;
mod execution;
mod health;
mod identity;
mod import;
mod import_report;
mod native_graph;
mod provider;
mod registry;
mod score;
mod semantic;
mod source;
mod training;
mod trial;
mod verifier;

pub(crate) use execution::{
    CanonicalPackagePlan, append_identity_field, artifact_source_overlaps_reserved_verifier_path,
    shared_workdir_conflicts_reserved_verifier_path, validate_env_name, validate_user,
    verifier_artifact_target_collision,
};
pub(crate) use native_graph::GENERATION_METADATA_KEY;

pub use artifact_manifest::{
    ArtifactManifestError, DeclaredArtifactManifest, MaterializedArtifactManifest,
};
pub use evidence::{
    AttemptId, EvidenceEvent, EvidenceKind, FrozenAttemptBundle, FrozenAttemptError,
};
pub use execution::{
    AgentCapability, ArtifactSpec, BenchmarkExecutionPlan, BenchmarkStepPlan, ComposeProjectId,
    ComposeProjectPlan, ComposeServiceName, ContainerResources, DockerAdapterLease,
    DockerAdapterProcess, DockerAdapterSpawnerRequest, DockerBuildRequest,
    DockerComposeAdapterSpawnerRequest, DockerComposeArchiveRequest, DockerComposeBuildRequest,
    DockerComposeConfigRequest, DockerComposeCopyRequest, DockerComposeDownRequest,
    DockerComposeExecRequest, DockerComposeRuntime, DockerComposeStopRequest,
    DockerComposeUpRequest, DockerCopyRequest, DockerCreateRequest, DockerEnvironment,
    DockerExecRequest, DockerProcessSandbox, DockerRemoveRequest, DockerRuntime,
    DockerStartRequest, EnvBinding, EnvName, EnvironmentPlan, EvalExecutionError,
    EvalExecutionPhase, EvalSandboxFactory, HarborAgentContract, HarborCompletedEvaluation,
    HarborEvaluationCoordinator, HarborEvaluationError, HarborLifecycleAgentContract,
    HarborLifecycleRequest, HarborLifecycleScoreRequest, HarborLocalEvaluationRequest,
    HarborSandboxRecipe, HealthcheckPlan, ImageSource, ImageSourceKind, ImmutablePatch,
    LocalAdapterSpawner, LocalExecutionResult, LocalProcessSandbox, MaterializedSandbox,
    MultiStepExecutionResult, MultiStepRewardStrategy, NativeGraphEpisodeCallback,
    NativeGraphEpisodeLease, NetworkPolicy, OwnedComposeResources, PhasePlan, ProcessOutput,
    ProviderCapabilities, SandboxRole, SecretProvider, SecretValue, StepExecutionResult,
    VerifierCollectHook, VerifierPlan, WorkspaceOverlay, collect_artifacts, preflight_docker,
    resolve_phase_environment, run_native_graph_episode_callback, transfer_artifacts,
};
pub use health::{TaskHealthError, TaskHealthRecord, TaskVerdict};
pub use identity::{
    AgentVariantRef, ArtifactDigest, EvalIdentityError, EvalTaskId, EvalTaskRef, ModelIdentity,
    PolicyIdentity, RuntimeIdentity,
};
pub use import::{
    AcquiredSource, HarborImportError, HarborImporter, HarborSource, HarborTaskPackage,
    ImportedTask, NativeSourceAcquirer, SourceAcquirer,
};
pub use import_report::{ImportDisposition, ImportReport};
pub use native_graph::{
    AdapterCheckout, AdapterCheckoutOrigin, AdapterEnvelope, AdapterExit, AdapterId,
    AdapterLifecycleDeadlines, AdapterMessage, AdapterPool, AdapterPoolKey, AdapterProcess,
    AdapterProtocol, AdapterProtocolConfig, AdapterProtocolFactory, AdapterRole,
    AdapterRuntimeFactory, AdapterSpawnRequest, AdapterSpawnTransaction, AdapterSpawner,
    AdapterSpec, AdapterSupervisionError, ArtifactDownloadHandle, ArtifactError, ArtifactQuota,
    ArtifactUploadHandle, AuthoredNativeGraphSuite, BoundedControlFlowContract, CancelReason,
    ConfirmedNativeGraphProviderRecoveryFactory, CurrentNativeGraphModelBindingResolver,
    DockerNativeGraphEpisodeExecutor, EngineNativeGraphEpisodeCallback, EpisodeAggregate,
    EpisodeArtifactStore, EpisodeAssignment, EpisodeAssignmentId, EpisodeComparability,
    EpisodeEvaluationError, EpisodeEvaluator, EpisodeEvaluatorFactory, EpisodeExecution,
    EpisodeExecutionError, EpisodeIntegrity, EpisodeResult, EpisodeResultError, EpisodeRunner,
    EpisodeScoreState, ExactNativeGraphFidelityObserverFactory, FrozenArtifact,
    FrozenArtifactManifest, GenerationDefaults, HarborEpisodeEvaluator,
    HarborEpisodeEvaluatorFactory, HeaderSecretRef, HostEnvelope, HostMessage,
    LocalNativeGraphSuiteScheduler, LocalNativeGraphSuiteSchedulerFactory, MatrixError,
    ModelBindingId, ModelBindingSpec, ModelCapacityKey, ModelCapturePolicy, ModelRuntimeConfig,
    ModelRuntimeError, ModelSecretId, NativeGraphAdapterAuthorization,
    NativeGraphAdapterRuntimeProvider, NativeGraphControlContract, NativeGraphEnvironmentStepper,
    NativeGraphEnvironmentStepperFactory, NativeGraphEpisodeExecutor, NativeGraphEpisodeRunner,
    NativeGraphExternalDriver, NativeGraphExternalDriverFactory, NativeGraphFactoryError,
    NativeGraphFidelityObserver, NativeGraphFidelityObserverFactory,
    NativeGraphLiveTraceProgramDriverFactory, NativeGraphLowererFactory,
    NativeGraphLowererProvider, NativeGraphLoweringError, NativeGraphLoweringReport,
    NativeGraphModelBindingResolver, NativeGraphModelStageError, NativeGraphNodeFidelity,
    NativeGraphNodeLowering, NativeGraphPackagePlan, NativeGraphProfile, NativeGraphProgramSource,
    NativeGraphProviderRecoveryFactory, NativeGraphSuiteDefinition, NativeGraphSuiteManifest,
    NativeGraphSuiteScheduler, ObservedNativeGraphTransportEvidence, PROTOCOL_VERSION,
    PackageNativeGraphLowererProvider, ProtocolAdapterRuntimeFactory, ProtocolCapability,
    ProtocolError, ProtocolLimits, ProtocolOperationState, ProtocolSessionState,
    RefusingEnvironmentStepperFactory, RefusingExternalDriverFactory, ReservedNativeGraphBranch,
    ReservedNativeGraphJoin, ReservedNativeGraphLoop, ResolvedEpisodeTrial, ResolvedModelBinding,
    ResolvedModelBindingSet, ResolvedNativeGraphSuite, ResolvedTokenizerBinding,
    ResourceLeaseRequest, ResourceLimits, SelectedModelBinding, StrictAdapterProtocolFactory,
    StrictAdapterRuntimeProvider, StrictSupervisedAdapter, SuiteError, SuiteRunId,
    SuiteSchedulerFactory, SuiteTrialSpec, SupervisedAdapter, TokenizerBindingSpec,
    ValidatedAdapterMessage, ValidatedHostMessage, aggregate_episode_results, lower_native_graph,
    parse_native_graph_suite_toml, run_resolved_suite, select_native_graph_scheduler,
    validate_native_graph_trace_plan,
};
pub use provider::{
    ModelEndpointAuthority, ModelEndpointIsolationProof, ProviderCapability, ProviderError,
    ProviderProfile, ProviderRecovery,
};
pub use registry::{RegistryError, RegistryReference};
pub use score::{ScoreError, ScoreVersion};
pub use semantic::{
    ExecutableSemanticNode, FidelityError, FidelityOutcome, GraphLowererCapabilities,
    GraphLowererFactory, GraphLoweringError, GraphLoweringRequest, LoweredSemanticGraph,
    PairedComparisonError, PairedComparisonReport, PairedComparisonSpec, PairedMeasurements,
    SemanticGraph, SemanticNode, lower_semantic_graph,
};
pub use source::{EvalDatasetId, EvalDatasetManifest};
pub use training::{TrainingError, TrajectoryExportManifest};
pub use trial::{TrialBudget, TrialIdentityError, TrialSpec};
pub use verifier::{
    ArtifactTransferError, DeclaredArtifactTransfer, RegradeError, RegradeRequest, RewardDocument,
    RewardError, RewardParseOutcome, VerifierExecutionError, VerifierMode, VerifierResult,
    VerifierSandboxFactory, invalid_reward_evidence, parse_reward_with_evidence, prepare_verifier,
    regrade,
};
