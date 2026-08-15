// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable evaluation identities and evidence contracts.

mod evidence;
mod identity;
mod import_report;
mod score;
mod source;
mod trial;

pub use evidence::{AttemptId, EvidenceEvent, EvidenceKind};
pub use identity::{
    AgentVariantRef, ArtifactDigest, EvalIdentityError, EvalTaskId, EvalTaskRef, ModelIdentity,
    PolicyIdentity, RuntimeIdentity,
};
pub use import_report::{ImportDisposition, ImportReport};
pub use score::{ScoreError, ScoreVersion};
pub use source::{EvalDatasetId, EvalDatasetManifest};
pub use trial::{TrialBudget, TrialSpec};
