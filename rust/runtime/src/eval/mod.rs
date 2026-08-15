// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable evaluation identities and evidence contracts.

mod identity;
mod import_report;
mod trial;

pub use identity::{
    AgentVariantRef, ArtifactDigest, EvalTaskId, EvalTaskRef, ModelIdentity, PolicyIdentity,
    RuntimeIdentity,
};
pub use import_report::{ImportDisposition, ImportReport};
pub use trial::{TrialBudget, TrialSpec};
