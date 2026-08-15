// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable evaluation identities and evidence contracts.

mod evidence;
mod execution;
mod identity;
mod import;
mod import_report;
mod score;
mod semantic;
mod source;
mod trial;
mod verifier;

pub use evidence::{AttemptId, EvidenceEvent, EvidenceKind};
pub use execution::{
    AgentCapability, EvalExecutionError, EvalSandboxFactory, HarborAgentContract, HarborSandboxRecipe,
    ImmutablePatch, WorkspaceOverlay,
};
pub use identity::{
    AgentVariantRef, ArtifactDigest, EvalIdentityError, EvalTaskId, EvalTaskRef, ModelIdentity,
    PolicyIdentity, RuntimeIdentity,
};
pub use import::{HarborImportError, HarborImporter, HarborSource, ImportedTask, SourceAcquirer};
pub use import_report::{ImportDisposition, ImportReport};
pub use score::{ScoreError, ScoreVersion};
pub use semantic::{
    lower_semantic_graph, FidelityError, FidelityOutcome, PairedComparisonError, PairedComparisonSpec,
    SemanticGraph, SemanticNode,
};
pub use source::{EvalDatasetId, EvalDatasetManifest};
pub use trial::{TrialBudget, TrialSpec};
pub use verifier::{ArtifactTransferError, DeclaredArtifactTransfer, RewardDocument, RewardError};
