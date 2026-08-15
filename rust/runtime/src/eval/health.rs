// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Task health and quarantine verdict contracts.

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
        if evidence.is_empty() {
            return Err(TaskHealthError::EmptyEvidence);
        }
        Ok(Self { verdict, evidence })
    }
}

/// Invalid task-health record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TaskHealthError {
    EmptyEvidence,
}

impl std::fmt::Display for TaskHealthError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("task health requires immutable evidence")
    }
}
impl std::error::Error for TaskHealthError {}
