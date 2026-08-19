// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph package, protocol, artifact, result, and suite contracts.

mod action_encoder;
mod artifacts;
mod capture;
mod cellular;
mod completed_attempt;
#[cfg(feature = "engine")]
mod episode_runner;
mod evaluator;
mod factories;
mod live_driver;
mod lowering;
mod matrix;
#[cfg(feature = "engine")]
mod model_runtime;
mod package;
mod protocol;
mod result;
mod rl;
mod rollout_evidence;
mod suite;
mod supervision;
pub(crate) mod workspace_patch;

#[cfg(feature = "engine")]
pub use crate::engine::record_lane::EvalNodeRecordArtifact;
pub(crate) use action_encoder::{ActionAdmissionAuthority, ActionSessionAuthority};
pub use action_encoder::{
    ActionEncodingLimits, AdmittedEnvironmentAction, BoundNativeGraphActionEncoder,
    DeclaredPolicyDecision, EpisodeActionEncodingError, NativeGraphActionEncoder,
};
pub use artifacts::{
    ArtifactDownloadHandle, ArtifactError, ArtifactQuota, ArtifactUploadHandle,
    EpisodeArtifactStore, FrozenArtifact, FrozenArtifactManifest, FrozenArtifactReference,
};
pub(crate) use capture::CompatibilityCaptureSession;
pub use capture::{
    CaptureError, CaptureFidelity, CapturePolicy, CompatibilityFidelity, CompatibilityObservation,
    CompatibilityObservationReport, CompatibilityTerminalReceipt, CompatibilityTerminalSupplement,
};
pub use cellular::{
    CellularFoldError, NativeGraphCellAssignment, NativeGraphCellLease,
    NativeGraphCellLeaseAuthority, NativeGraphCellLeaseError, NativeGraphCellLeaseId,
    NativeGraphCellPlacement, NativeGraphCellResultAuthority, NativeGraphCellResultReceipt,
    NativeGraphCellularFold, NativeGraphCellularPlan, NativeGraphCellularReceiptError,
    NativeGraphCellularReceiptLimits,
};
pub use completed_attempt::{
    NativeGraphAttemptAuthority, NativeGraphCompletedAttempt, NativeGraphCompletedAttemptError,
};
#[cfg(feature = "engine")]
pub use episode_runner::{
    DockerNativeGraphEpisodeExecutor, EpisodeExecutionError, NativeGraphEpisodeExecutor,
    NativeGraphEpisodeRunner,
};
pub use evaluator::{
    EpisodeEvaluationError, EpisodeEvaluator, EpisodeEvaluatorFactory, HarborEpisodeEvaluator,
    HarborEpisodeEvaluatorFactory,
};
#[cfg(feature = "engine")]
pub use factories::select_native_graph_external_driver;
#[cfg(feature = "engine")]
pub use factories::{
    BoundNativeGraphEnvironmentStepper, StartedNativeGraphEnvironmentStepper,
    bind_native_graph_environment_stepper,
};
pub use factories::{
    ConfirmedNativeGraphProviderRecoveryFactory, ExactNativeGraphFidelityObserverFactory,
    ExternalDriverError, ExternalDriverSession, MoveV1ActionEncoderFactory,
    NativeGraphActionEncoderFactory, NativeGraphAdapterRuntimeProvider,
    NativeGraphAdapterRuntimeResolution, NativeGraphEnvironmentStepper,
    NativeGraphEnvironmentStepperFactory, NativeGraphExternalDriverFactory,
    NativeGraphFactoryError, NativeGraphFidelityObserver, NativeGraphFidelityObserverFactory,
    NativeGraphLowererProvider, NativeGraphProviderRecoveryFactory,
    PackageNativeGraphLowererProvider, PreparedExternalDriver, RefusingEnvironmentStepperFactory,
    RefusingExternalDriverFactory, StrictAdapterRuntimeProvider,
    SupervisedEnvironmentStepperBinder,
};
pub(crate) use live_driver::NATIVE_GRAPH_LIVE_DRIVER_KIND;
pub use live_driver::{
    NativeGraphLiveAgentLoopFactories, NativeGraphLiveTraceProgramDriverFactory,
};
pub(crate) use lowering::GENERATION_METADATA_KEY;
pub use lowering::{
    BoundedControlFlowContract, NativeGraphControlContract, NativeGraphLowererFactory,
    NativeGraphLoweringError, NativeGraphLoweringReport, NativeGraphNodeFidelity,
    NativeGraphNodeLowering, ReservedNativeGraphBranch, ReservedNativeGraphJoin,
    ReservedNativeGraphLoop, lower_native_graph, validate_native_graph_trace_plan,
};
#[cfg(feature = "engine")]
pub use matrix::select_native_graph_scheduler;
pub use matrix::{
    EpisodeAssignment, EpisodeRunner, LocalNativeGraphSuiteScheduler,
    LocalNativeGraphSuiteSchedulerFactory, MatrixError, NativeGraphSuiteScheduler, ResourceLimits,
    SuiteSchedulerFactory, run_resolved_suite,
};
#[cfg(feature = "engine")]
pub use model_runtime::{
    CurrentNativeGraphModelBindingResolver, EngineNativeGraphEpisodeCallback,
    IssuedNativeGraphPolicyDecision, ModelRuntimeConfig, ModelRuntimeError,
    NativeGraphLiveRolloutCoordinator, NativeGraphLiveRolloutError,
    NativeGraphModelBindingResolver, NativeGraphModelDecisionError, NativeGraphModelStageError,
    NativeGraphPolicyModelRuntime, ObservedNativeGraphTransportEvidence,
    PreparedNativeGraphLiveRolloutCoordinator, ResolvedModelBinding, ResolvedModelBindingSet,
    ResolvedTokenizerBinding,
};
pub use result::{
    EpisodeAggregate, EpisodeComparability, EpisodeExecution, EpisodeFidelity, EpisodeIntegrity,
    EpisodeResult, EpisodeResultError, EpisodeScoreState, aggregate_episode_results,
};
pub use rl::{
    EnvironmentTransitionRecord, FrozenRolloutTrajectory, RlEvaluationLimits, RlEvaluationPolicy,
    RlRolloutError,
};
pub use rollout_evidence::{
    FrozenRolloutEvidence, NativeGraphLivePolicyCallEvidence, NativeGraphRolloutReceipt,
    NativeGraphRolloutReceiptError, NativeGraphRolloutTransitionReceipt, RolloutAdmissionError,
    RolloutEvidenceError, RolloutEvidenceIdentity, RolloutEvidenceLimits,
    RolloutEvidenceLimitsError, RolloutPolicyEvidence, RolloutReturnAgreementError, RolloutReturns,
    RolloutTransitionEvidence, RolloutVerifierDecodeError, RolloutVerifierInput,
};
pub use suite::{
    AuthoredNativeGraphSuite, EpisodeAssignmentId, ModelCapacityKey, NativeGraphSuiteDefinition,
    NativeGraphSuiteManifest, ResolvedEpisodeTrial, ResolvedNativeGraphSuite, ResourceLeaseRequest,
    SelectedModelBinding, SuiteError, SuiteRunId, SuiteTrialSpec, parse_native_graph_suite_toml,
};

pub use package::{
    ActionEncoderFactoryId, AdapterId, AdapterRole, AdapterSpec, ExternalDriverFactoryId,
    GenerationDefaults, HeaderSecretRef, ModelBindingId, ModelBindingSpec, ModelCapturePolicy,
    ModelSecretId, NativeGraphPackagePlan, NativeGraphProfile, NativeGraphProgramSource,
    NativeGraphRolloutPolicyPromptSource, NativeGraphWorkspacePatchContract, TokenizerBindingSpec,
};
pub(crate) use package::{
    NativeGraphPackageDraft, NativeGraphSectionDto, resolve_native_graph_package,
};
pub use protocol::{
    AdapterEnvelope, AdapterMessage, AdapterProtocol, AdapterProtocolConfig,
    AdapterProtocolFactory, HostEnvelope, HostMessage, PROTOCOL_VERSION, ProtocolCapability,
    ProtocolError, ProtocolLimits, ProtocolOperationState, ProtocolSessionState,
    StrictAdapterProtocolFactory, ValidatedAdapterMessage, ValidatedHostMessage,
};
pub use supervision::{
    AdapterCheckout, AdapterCheckoutOrigin, AdapterExit, AdapterLifecycleDeadlines, AdapterPool,
    AdapterPoolKey, AdapterProcess, AdapterRuntimeFactory, AdapterSpawnRequest,
    AdapterSpawnTransaction, AdapterSpawner, AdapterSupervisionError, CancelReason,
    ExternallyDrivenAdapterAuthorization, NativeGraphAdapterAuthorization,
    ProtocolAdapterRuntimeFactory, StrictSupervisedAdapter, SupervisedAdapter,
};
