// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable versioned scores over preserved attempt evidence.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

use super::{ArtifactDigest, AttemptId};

/// An immutable score produced by one evaluator over preserved evidence.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ScoreVersion {
    /// Attempt evaluated by this score.
    pub attempt: AttemptId,
    /// Monotonic score revision for this attempt.
    pub version: u32,
    /// Evaluator or verifier identity.
    pub evaluator: ArtifactDigest,
    /// Immutable evidence identities considered by the evaluator.
    pub evidence: Vec<ArtifactDigest>,
    /// Finite numeric outcome supplied by the evaluator.
    pub value: f64,
}

impl ScoreVersion {
    /// Creates one versioned immutable score without changing prior score records.
    pub fn new(
        attempt: AttemptId,
        version: u32,
        evaluator: ArtifactDigest,
        evidence: Vec<ArtifactDigest>,
        value: f64,
    ) -> Result<Self, ScoreError> {
        if !value.is_finite() {
            return Err(ScoreError::NonFiniteValue);
        }
        Ok(Self {
            attempt,
            version,
            evaluator,
            evidence,
            value,
        })
    }

    /// Computes the immutable identity of this complete score revision.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut bytes = format!(
            "attempt={}\u{1f}version={}\u{1f}evaluator={}\u{1f}value={}",
            self.attempt.as_str(),
            self.version,
            self.evaluator.as_str(),
            self.value.to_bits(),
        )
        .into_bytes();
        for evidence in &self.evidence {
            bytes.extend_from_slice(b"\x1eevidence=");
            bytes.extend_from_slice(evidence.as_str().as_bytes());
        }
        ArtifactDigest::from_bytes(&bytes)
    }
}

/// Failed immutable-score validation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreError {
    /// Scores must remain serializable finite numbers.
    NonFiniteValue,
}

impl Display for ScoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteValue => formatter.write_str("score value must be finite"),
        }
    }
}

impl std::error::Error for ScoreError {}
