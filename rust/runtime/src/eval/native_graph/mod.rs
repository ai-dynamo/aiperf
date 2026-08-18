// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! NativeGraph package, protocol, artifact, result, and suite contracts.

mod artifacts;
mod episode_runner;
mod evaluator;
mod factories;
mod live_driver;
mod lowering;
mod matrix;
mod model_runtime;
mod package;
mod protocol;
mod result;
mod suite;
mod supervision;

pub use artifacts::{
    ArtifactDownloadHandle, ArtifactError, ArtifactQuota, ArtifactUploadHandle,
    EpisodeArtifactStore, FrozenArtifact, FrozenArtifactManifest,
};

pub use episode_runner::{
    DockerNativeGraphEpisodeExecutor, EpisodeExecutionError, NativeGraphEpisodeExecutor,
    NativeGraphEpisodeRunner,
};
pub use evaluator::{
    EpisodeEvaluationError, EpisodeEvaluator, EpisodeEvaluatorFactory, HarborEpisodeEvaluator,
    HarborEpisodeEvaluatorFactory,
};
pub use factories::{
    ConfirmedNativeGraphProviderRecoveryFactory, ExactNativeGraphFidelityObserverFactory,
    NativeGraphAdapterRuntimeProvider, NativeGraphEnvironmentStepper,
    NativeGraphEnvironmentStepperFactory, NativeGraphExternalDriver,
    NativeGraphExternalDriverFactory, NativeGraphFactoryError, NativeGraphFidelityObserver,
    NativeGraphFidelityObserverFactory, NativeGraphLowererProvider,
    NativeGraphProviderRecoveryFactory, PackageNativeGraphLowererProvider,
    RefusingEnvironmentStepperFactory, RefusingExternalDriverFactory, StrictAdapterRuntimeProvider,
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
pub use matrix::{
    EpisodeAssignment, EpisodeRunner, LocalNativeGraphSuiteScheduler,
    LocalNativeGraphSuiteSchedulerFactory, MatrixError, NativeGraphSuiteScheduler, ResourceLimits,
    SuiteSchedulerFactory, run_resolved_suite, select_native_graph_scheduler,
};
pub use model_runtime::{
    CurrentNativeGraphModelBindingResolver, EngineNativeGraphEpisodeCallback, ModelRuntimeConfig,
    ModelRuntimeError, NativeGraphModelBindingResolver, NativeGraphModelStageError,
    ObservedNativeGraphTransportEvidence, ResolvedModelBinding, ResolvedModelBindingSet,
    ResolvedTokenizerBinding,
};
pub use result::{
    EpisodeAggregate, EpisodeComparability, EpisodeExecution, EpisodeIntegrity, EpisodeResult,
    EpisodeResultError, EpisodeScoreState, aggregate_episode_results,
};
pub use suite::{
    AuthoredNativeGraphSuite, EpisodeAssignmentId, ModelCapacityKey, NativeGraphSuiteDefinition,
    NativeGraphSuiteManifest, ResolvedEpisodeTrial, ResolvedNativeGraphSuite, ResourceLeaseRequest,
    SelectedModelBinding, SuiteError, SuiteRunId, SuiteTrialSpec, parse_native_graph_suite_toml,
};

pub use package::{
    AdapterId, AdapterRole, AdapterSpec, GenerationDefaults, HeaderSecretRef, ModelBindingId,
    ModelBindingSpec, ModelCapturePolicy, ModelSecretId, NativeGraphPackagePlan,
    NativeGraphProfile, NativeGraphProgramSource, TokenizerBindingSpec,
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
    NativeGraphAdapterAuthorization, ProtocolAdapterRuntimeFactory, StrictSupervisedAdapter,
    SupervisedAdapter,
};
