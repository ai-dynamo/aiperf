// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable trajectory export references.

use crate::eval::{ArtifactDigest, AttemptId};

/// Immutable evidence references exported for downstream training preparation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryExportManifest {
    /// Source attempt whose history is exported.
    pub attempt: AttemptId,
    /// Preserved append-only evidence identities.
    pub evidence: Vec<ArtifactDigest>,
}

impl TrajectoryExportManifest {
    /// Creates an export manifest that never embeds mutable attempt state.
    pub fn new(attempt: AttemptId, evidence: Vec<ArtifactDigest>) -> Result<Self, TrainingError> {
        if evidence.is_empty() {
            return Err(TrainingError::EmptyEvidence);
        }
        Ok(Self { attempt, evidence })
    }
}

/// Invalid immutable trajectory export request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingError {
    /// Training exports must pin preserved evidence.
    EmptyEvidence,
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("trajectory export requires immutable evidence")
    }
}

impl std::error::Error for TrainingError {}
