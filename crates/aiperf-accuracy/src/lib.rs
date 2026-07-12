// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust evaluator control plane and supervised provider boundary.
//!
//! Rust deliberately contains no benchmark prompt builders, answer extractors,
//! hidden-test decoders, code runners, or graders. The provider-neutral v2 seam
//! keeps evaluator semantics in an isolated worker while Rust owns every host
//! effect. The legacy [`AccuracyEvaluator`] and [`AgenticEvaluator`] protocols
//! remain exported only for staged migration.

pub mod artifacts;
pub mod canonical;
pub mod isolation;
pub mod lifecycle;
pub mod provider;
pub mod provider_protocol;
pub mod score_projection;

pub mod protocol;
pub mod supervisor;
pub mod worker;

pub use artifacts::{
    ArtifactProjectionPolicy, ArtifactSealError, ArtifactSealLimits, EvaluationArtifactSealer,
    PublicArtifactProjectionRule, PublicArtifactProjectionValidator, SealedEvaluationArtifact,
    SealedEvaluationArtifacts,
};
pub use canonical::{
    CANONICAL_JSON_CODEC, CanonicalJson, CanonicalJsonError, CanonicalJsonLimits,
    artifact_content_sha256, is_sha256, normalized_result_sha256, redact_diagnostic, sha256_hex,
    validate_no_secret_control_value, validate_no_secret_host_payload,
};
pub use isolation::{
    AttestedWorkerLaunch, BubblewrapEvaluatorIsolation, EvaluatorIsolation,
    EvaluatorIsolationEvidence, EvaluatorLaunchAttestor, EvaluatorResourceLimits,
    IsolationQuiescenceProof, LaunchAttestation, LaunchClosureFile, PreparedEvaluatorLaunch,
    Sha256LaunchAttestor,
};
pub use lifecycle::{EvaluationLifecycle, EvaluationLifecycleState};
pub use provider::{
    EvaluationDistributionDescriptor, EvaluationOperationDescriptor, EvaluationProvider,
    EvaluationProviderDescriptor, EvaluationProviderError, EvaluationProviderFactory,
    EvaluationProviderLauncher, EvaluationProviderRegistry, EvaluationProviderRegistryBuilder,
    EvaluatorIsolationRequirements, EvaluatorProcessRootBinder, EvaluatorProtocolLimits,
    NemoEvaluatorProviderFactory, OpenBenchProviderFactory, PreparedEvaluationProviderLaunch,
    ProviderConfigValidator, ProviderLaunchContext, ProviderRegistryError,
    STOCK_EVALUATION_OPERATION_SCHEMAS, StockEvaluationOperationSchema, ValidatedProviderConfig,
};
pub use provider_protocol::{
    AggregateMetric, AggregationPolicy, ArtifactRef, ArtifactVisibility, CaseOutcome,
    CaseOutcomeKind, CompletedCaseOutcome, EVALUATOR_WORKER_PROTOCOL_V2, EvaluationArtifactId,
    EvaluationArtifactManifestEntry, EvaluationAssetRequirement, EvaluationCaseId,
    EvaluationCaseOccurrenceDescriptor, EvaluationCaseTemplateDescriptor, EvaluationCaseTemplateId,
    EvaluationDistributionId, EvaluationError, EvaluationEvent, EvaluationEventBatch,
    EvaluationExecutionGranularity, EvaluationFinishCandidate, EvaluationHostBinding,
    EvaluationHostIdentity, EvaluationIdentity, EvaluationIdentityComponent, EvaluationPhaseId,
    EvaluationPlan, EvaluationPlanRequest, EvaluationProgress, EvaluationProtocolError,
    EvaluationProviderId, EvaluationQueueCredits, EvaluationSchedulingMode, EvaluationSessionId,
    EvaluationStage, EvaluationUnitId, EvaluationUnitOccurrence, EvaluationUnitOccurrenceRequest,
    EvaluationUnitPage, EvaluationUnitTemplateDescriptor, EvaluationUnitTemplateId,
    EvaluationWorkerIdentity, FiniteF64, HostCallContext, HostCapabilityId,
    HostCapabilityRequirement, HostOperationCancelRequest, HostOperationDisposition,
    HostOperationEvent, HostOperationId, HostOperationRequest, HostOperationTerminal,
    HostOperationUsage, HostResponseMode, LogicalCallId, LogicalServiceId,
    LogicalServiceRequirement, OperationPurpose, ProviderScore, ResolvedEvaluationAsset,
    RestrictedDisclosure, RestrictedInferencePayload, ScopedProxyBinding, ScopedProxyGrant,
    ScopedProxySecret, SemanticAttemptId, SemanticOperationId, SequencedEvaluationEvent,
};
pub use score_projection::{
    PublicScoreProjectionError, PublicScoreProjectionPolicy, PublicScoreProjectionValidator,
};
pub use supervisor::{
    EvaluationProviderLogSink, StderrEvaluationProviderLogSink, SupervisedEvaluationProvider,
    SupervisedEvaluationProviderLauncher,
};

pub use protocol::{
    AgenticEpisode, AgenticEpisodeOutcome, AgenticEpisodePage, AgenticEpisodeResult,
    AgenticEvaluatorEvent, AgenticEvaluatorIdentity, AgenticEvaluatorLoadConfig, AgenticEventBatch,
    AgenticInferenceGatewayConfig, AgenticInferenceStatus, AgenticMessage, AgenticModelCall,
    AgenticModelResult, AgenticResultBatch, EVALUATOR_PROTOCOL_VERSION, EpisodeId,
    EvaluatorDatasetIdentity, EvaluatorGenerationConfig, EvaluatorGrade, EvaluatorGradeBatch,
    EvaluatorGradeItem, EvaluatorIdentity, EvaluatorLoadConfig, EvaluatorLoadResult,
    EvaluatorMessage, EvaluatorProblem, EvaluatorProblemPage, ModelCallId, ProblemId,
};
pub use worker::{
    AccuracyEvaluator, AgenticEvaluator, EvaluatorLogSink, EvaluatorWorkerError, PythonEvaluator,
    StderrEvaluatorLogSink, WorkerProcessConfig,
};
