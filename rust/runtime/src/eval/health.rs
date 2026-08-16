// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Task health and quarantine verdict contracts.

use std::collections::BTreeSet;

use crate::eval::ArtifactDigest;

/// Independent validity verdict for an evaluation task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskVerdict {
    /// Task and verifier are valid.
    Valid,
    /// Task is usable only under declared conditions.
    ConditionallyValid,
    /// Task must be excluded from aggregate capability results.
    Broken,
}

/// Immutable health record used to quarantine invalid tasks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskHealthRecord {
    /// Evaluated validity result.
    pub verdict: TaskVerdict,
    /// Immutable supporting evidence.
    pub evidence: Vec<ArtifactDigest>,
}

impl TaskHealthRecord {
    /// Creates a health record with evidence required for quarantine decisions.
    pub fn new(
        verdict: TaskVerdict,
        evidence: Vec<ArtifactDigest>,
    ) -> Result<Self, TaskHealthError> {
        validate_evidence(&evidence)?;
        Ok(Self { verdict, evidence })
    }

    /// Returns whether aggregate evaluation results must exclude this task.
    pub const fn is_quarantined(&self) -> bool {
        matches!(self.verdict, TaskVerdict::Broken)
    }

    /// Revalidates the supporting evidence after record deserialization or transfer.
    pub fn validate(&self) -> Result<(), TaskHealthError> {
        validate_evidence(&self.evidence)
    }
}

fn validate_evidence(evidence: &[ArtifactDigest]) -> Result<(), TaskHealthError> {
    if evidence.is_empty() {
        return Err(TaskHealthError::EmptyEvidence);
    }
    if evidence.iter().collect::<BTreeSet<_>>().len() != evidence.len() {
        return Err(TaskHealthError::DuplicateEvidence);
    }
    Ok(())
}

/// Invalid task-health record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskHealthError {
    /// Quarantine decisions require supporting evidence.
    EmptyEvidence,
    /// Each evidence identity may support a decision only once.
    DuplicateEvidence,
}

impl std::fmt::Display for TaskHealthError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyEvidence => formatter.write_str("task health requires immutable evidence"),
            Self::DuplicateEvidence => formatter.write_str("task health evidence must be unique"),
        }
    }
}
impl std::error::Error for TaskHealthError {}
