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
mod provider;
mod registry;
mod score;
mod semantic;
mod source;
mod training;
mod trial;
mod verifier;

pub(crate) use execution::{validate_env_name, validate_user};

pub use artifact_manifest::{
    ArtifactManifestError, DeclaredArtifactManifest, MaterializedArtifactManifest,
};
pub use evidence::{AttemptId, EvidenceEvent, EvidenceKind};
pub use execution::{
    AgentCapability, ArtifactSpec, BenchmarkExecutionPlan, ContainerResources,
    DockerProcessSandbox, EnvBinding, EnvironmentPlan, EvalExecutionError, EvalExecutionPhase,
    EvalSandboxFactory, HarborAgentContract, HarborEvaluationCoordinator, HarborEvaluationError,
    HarborSandboxRecipe, HealthcheckPlan, ImmutablePatch, LocalExecutionResult,
    LocalProcessSandbox, MaterializedSandbox, NetworkPolicy, PhasePlan, ProcessOutput, SandboxRole,
    VerifierPlan, WorkspaceOverlay,
};
pub use health::{TaskHealthError, TaskHealthRecord, TaskVerdict};
pub use identity::{
    AgentVariantRef, ArtifactDigest, EvalIdentityError, EvalTaskId, EvalTaskRef, ModelIdentity,
    PolicyIdentity, RuntimeIdentity,
};
pub use import::{
    HarborImportError, HarborImporter, HarborSource, HarborTaskPackage, ImportedTask,
    NativeSourceAcquirer, SourceAcquirer,
};
pub use import_report::{ImportDisposition, ImportReport};
pub use provider::{ProviderCapability, ProviderError, ProviderProfile};
pub use registry::{RegistryError, RegistryReference};
pub use score::{ScoreError, ScoreVersion};
pub use semantic::{
    ExecutableSemanticNode, FidelityError, FidelityOutcome, LoweredSemanticGraph,
    PairedComparisonError, PairedComparisonReport, PairedComparisonSpec, PairedMeasurements,
    SemanticGraph, SemanticNode, lower_semantic_graph,
};
pub use source::{EvalDatasetId, EvalDatasetManifest};
pub use training::{TrainingError, TrajectoryExportManifest};
pub use trial::{TrialBudget, TrialSpec};
pub use verifier::{
    ArtifactTransferError, DeclaredArtifactTransfer, RegradeError, RegradeRequest, RewardDocument,
    RewardError, RewardParseOutcome, VerifierExecutionError, VerifierMode, VerifierResult,
    VerifierSandboxFactory, invalid_reward_evidence, parse_reward_with_evidence, prepare_verifier,
    regrade,
};
