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
#[cfg(feature = "engine")]
pub(crate) use native_graph::GENERATION_METADATA_KEY;
#[cfg(feature = "engine")]
pub(crate) use native_graph::NATIVE_GRAPH_LIVE_DRIVER_KIND;
pub(crate) use native_graph::{ActionAdmissionAuthority, ActionSessionAuthority};

pub use artifact_manifest::{
    ArtifactManifestError, DeclaredArtifactManifest, MaterializedArtifactManifest,
};
pub use evidence::{
    AttemptId, EvidenceEvent, EvidenceKind, FrozenAttemptBundle, FrozenAttemptError,
};
pub use execution::{
    AgentCapability, ArtifactSpec, AuthorizedExternalDriverSpawn, BenchmarkExecutionPlan,
    BenchmarkStepPlan, ComposeProjectId, ComposeProjectPlan, ComposeServiceName,
    ContainerResources, DockerAdapterLease, DockerAdapterProcess, DockerAdapterSpawnerRequest,
    DockerBuildRequest, DockerComposeAdapterSpawnerRequest, DockerComposeArchiveRequest,
    DockerComposeBuildRequest, DockerComposeConfigRequest, DockerComposeCopyRequest,
    DockerComposeDownRequest, DockerComposeExecRequest, DockerComposeRuntime,
    DockerComposeStopRequest, DockerComposeUpRequest, DockerCopyRequest, DockerCreateRequest,
    DockerEnvironment, DockerExecRequest, DockerProcessSandbox, DockerRemoveRequest, DockerRuntime,
    DockerStartRequest, EnvBinding, EnvName, EnvironmentPlan, EvalExecutionError,
    EvalExecutionPhase, EvalSandboxFactory, ExternalDriverDockerSpawnOperation,
    ExternalDriverDockerSpawner, ExternalDriverSpawnExecutor, HarborAgentContract,
    HarborCompletedEvaluation, HarborEvaluationCoordinator, HarborEvaluationError,
    HarborLifecycleAgentContract, HarborLifecycleRequest, HarborLifecycleScoreRequest,
    HarborLocalEvaluationRequest, HarborSandboxRecipe, HealthcheckPlan, ImageSource,
    ImageSourceKind, ImmutablePatch, LocalAdapterSpawner, LocalExecutionResult,
    LocalProcessSandbox, MaterializedSandbox, MultiStepExecutionResult, MultiStepRewardStrategy,
    NativeGraphEnvironmentAdapterStart, NativeGraphEpisodeBackendLease, NativeGraphEpisodeCallback,
    NativeGraphEpisodeLease, NetworkPolicy, OwnedComposeResources, PhasePlan, ProcessOutput,
    ProviderCapabilities, SandboxRole, SecretProvider, SecretValue,
    StartedExternalDriverDockerSpawn, StepExecutionResult, VerifierCollectHook, VerifierPlan,
    WorkspaceOverlay, collect_artifacts, preflight_docker, resolve_phase_environment,
    run_native_graph_episode_callback, transfer_artifacts,
};
#[cfg(feature = "engine")]
pub use execution::{NativeGraphEnvironmentRolloutSession, NativeGraphLeaseRolloutStart};
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
#[cfg(feature = "engine")]
pub use native_graph::select_native_graph_scheduler;
pub use native_graph::{
    ActionEncoderFactoryId, ActionEncodingLimits, AdapterCheckout, AdapterCheckoutOrigin,
    AdapterEnvelope, AdapterExit, AdapterId, AdapterLifecycleDeadlines, AdapterMessage,
    AdapterPool, AdapterPoolKey, AdapterProcess, AdapterProtocol, AdapterProtocolConfig,
    AdapterProtocolFactory, AdapterRole, AdapterRuntimeFactory, AdapterSpawnRequest,
    AdapterSpawnTransaction, AdapterSpawner, AdapterSpec, AdapterSupervisionError,
    AdmittedEnvironmentAction, ArtifactDownloadHandle, ArtifactError, ArtifactQuota,
    ArtifactUploadHandle, AuthoredNativeGraphSuite, BoundNativeGraphActionEncoder,
    BoundedControlFlowContract, CancelReason, CaptureError, CaptureFidelity, CapturePolicy,
    CompatibilityFidelity, CompatibilityObservation, CompatibilityObservationReport,
    CompatibilityTerminalReceipt, CompatibilityTerminalSupplement,
    ConfirmedNativeGraphProviderRecoveryFactory, DeclaredPolicyDecision,
    EnvironmentTransitionRecord, EpisodeActionEncodingError, EpisodeAggregate,
    EpisodeArtifactStore, EpisodeAssignment, EpisodeAssignmentId, EpisodeComparability,
    EpisodeEvaluationError, EpisodeEvaluator, EpisodeEvaluatorFactory, EpisodeExecution,
    EpisodeFidelity, EpisodeIntegrity, EpisodeResult, EpisodeResultError, EpisodeRunner,
    EpisodeScoreState, ExactNativeGraphFidelityObserverFactory, ExternalDriverError,
    ExternalDriverFactoryId, ExternalDriverSession, ExternallyDrivenAdapterAuthorization,
    FrozenArtifact, FrozenArtifactManifest, FrozenArtifactReference, FrozenRolloutEvidence,
    FrozenRolloutTrajectory, GenerationDefaults, HarborEpisodeEvaluator,
    HarborEpisodeEvaluatorFactory, HeaderSecretRef, HostEnvelope, HostMessage,
    LocalNativeGraphSuiteScheduler, LocalNativeGraphSuiteSchedulerFactory, MatrixError,
    ModelBindingId, ModelBindingSpec, ModelCapacityKey, ModelCapturePolicy, ModelSecretId,
    MoveV1ActionEncoderFactory, NativeGraphActionEncoder, NativeGraphActionEncoderFactory,
    NativeGraphAdapterAuthorization, NativeGraphAdapterRuntimeProvider,
    NativeGraphAdapterRuntimeResolution, NativeGraphAttemptAuthority, NativeGraphCompletedAttempt,
    NativeGraphCompletedAttemptError, NativeGraphControlContract, NativeGraphEnvironmentStepper,
    NativeGraphEnvironmentStepperFactory, NativeGraphExternalDriverFactory,
    NativeGraphFactoryError, NativeGraphFidelityObserver, NativeGraphFidelityObserverFactory,
    NativeGraphLiveAgentLoopFactories, NativeGraphLivePolicyCallEvidence,
    NativeGraphLiveTraceProgramDriverFactory, NativeGraphLowererFactory,
    NativeGraphLowererProvider, NativeGraphLoweringError, NativeGraphLoweringReport,
    NativeGraphNodeFidelity, NativeGraphNodeLowering, NativeGraphPackagePlan, NativeGraphProfile,
    NativeGraphProgramSource, NativeGraphProviderRecoveryFactory,
    NativeGraphRolloutPolicyPromptSource, NativeGraphRolloutReceipt,
    NativeGraphRolloutReceiptError, NativeGraphRolloutTransitionReceipt,
    NativeGraphSuiteDefinition, NativeGraphSuiteManifest, NativeGraphSuiteScheduler,
    NativeGraphWorkspacePatchContract, PROTOCOL_VERSION, PackageNativeGraphLowererProvider,
    PreparedExternalDriver, PreparedExternalDriverCapability, ProtocolAdapterRuntimeFactory,
    ProtocolCapability, ProtocolError, ProtocolLimits, ProtocolOperationState,
    ProtocolSessionState, RefusingEnvironmentStepperFactory, RefusingExternalDriverFactory,
    ReservedNativeGraphBranch, ReservedNativeGraphJoin, ReservedNativeGraphLoop,
    ResolvedEpisodeTrial, ResolvedNativeGraphSuite, ResourceLeaseRequest, ResourceLimits,
    RlEvaluationLimits, RlEvaluationPolicy, RlRolloutError, RolloutAdmissionError,
    RolloutEvidenceError, RolloutEvidenceIdentity, RolloutEvidenceLimits,
    RolloutEvidenceLimitsError, RolloutPolicyEvidence, RolloutReturnAgreementError, RolloutReturns,
    RolloutTransitionEvidence, RolloutVerifierDecodeError, RolloutVerifierInput,
    SelectedModelBinding, StrictAdapterProtocolFactory, StrictAdapterRuntimeProvider,
    StrictSupervisedAdapter, SuiteError, SuiteRunId, SuiteSchedulerFactory, SuiteTrialSpec,
    SupervisedAdapter, SupervisedEnvironmentStepperBinder, TokenizerBindingSpec,
    ValidatedAdapterMessage, ValidatedHostMessage, aggregate_episode_results, lower_native_graph,
    parse_native_graph_suite_toml, run_resolved_suite, validate_native_graph_trace_plan,
};
#[cfg(feature = "engine")]
pub use native_graph::{
    BoundNativeGraphEnvironmentStepper, CurrentNativeGraphModelBindingResolver,
    DockerNativeGraphEpisodeExecutor, EngineNativeGraphEpisodeCallback, EpisodeExecutionError,
    EvalNodeRecordArtifact, IssuedNativeGraphPolicyDecision, ModelRuntimeConfig, ModelRuntimeError,
    NativeGraphEpisodeExecutor, NativeGraphEpisodeRunner, NativeGraphLiveRolloutCoordinator,
    NativeGraphLiveRolloutError, NativeGraphModelBindingResolver, NativeGraphModelDecisionError,
    NativeGraphModelStageError, NativeGraphPolicyModelRuntime,
    ObservedNativeGraphTransportEvidence, PreparedNativeGraphLiveRolloutCoordinator,
    ResolvedModelBinding, ResolvedModelBindingSet, ResolvedTokenizerBinding,
    StartedNativeGraphEnvironmentStepper, bind_native_graph_environment_stepper,
    select_native_graph_external_driver,
};
pub use native_graph::{
    CellularFoldError, NativeGraphCellAssignment, NativeGraphCellLease,
    NativeGraphCellLeaseAuthority, NativeGraphCellLeaseError, NativeGraphCellLeaseId,
    NativeGraphCellPlacement, NativeGraphCellResultAuthority, NativeGraphCellResultReceipt,
    NativeGraphCellularFold, NativeGraphCellularPlan, NativeGraphCellularReceiptError,
    NativeGraphCellularReceiptLimits,
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
