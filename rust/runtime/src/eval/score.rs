// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable versioned scores over preserved attempt evidence.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

use super::{ArtifactDigest, AttemptId};

/// An immutable score produced by one evaluator over preserved evidence.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct ScoreVersion {
    /// Attempt evaluated by this score.
    pub attempt: AttemptId,
    /// Monotonic score revision for this attempt.
    pub version: u32,
    /// Evaluator or verifier identity.
    pub evaluator: ArtifactDigest,
    /// Immutable evidence identities considered by the evaluator.
    pub evidence: Vec<ArtifactDigest>,
    /// Named verifier metric represented by this score.
    pub metric: String,
    /// Finite numeric outcome supplied by the evaluator.
    pub value: f64,
    /// Immutable rationale emitted by the evaluator.
    pub rationale: ArtifactDigest,
    /// Identity of the preceding score revision, when this score is a regrade.
    pub predecessor: Option<ArtifactDigest>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawScoreVersion {
    attempt: AttemptId,
    version: u32,
    evaluator: ArtifactDigest,
    evidence: Vec<ArtifactDigest>,
    metric: String,
    value: f64,
    rationale: ArtifactDigest,
    predecessor: Option<ArtifactDigest>,
}

impl<'de> Deserialize<'de> for ScoreVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawScoreVersion::deserialize(deserializer)?;
        Self::new(
            raw.attempt,
            raw.version,
            raw.evaluator,
            raw.evidence,
            raw.metric,
            raw.value,
            raw.rationale,
            raw.predecessor,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl ScoreVersion {
    /// Creates one immutable score revision without changing prior score records.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        attempt: AttemptId,
        version: u32,
        evaluator: ArtifactDigest,
        evidence: Vec<ArtifactDigest>,
        metric: impl Into<String>,
        value: f64,
        rationale: ArtifactDigest,
        predecessor: Option<ArtifactDigest>,
    ) -> Result<Self, ScoreError> {
        let metric = metric.into();
        if metric.trim().is_empty() {
            return Err(ScoreError::EmptyMetric);
        }
        if evidence.is_empty() {
            return Err(ScoreError::EmptyEvidence);
        }
        if !value.is_finite() {
            return Err(ScoreError::NonFiniteValue);
        }
        Ok(Self {
            attempt,
            version,
            evaluator,
            evidence,
            metric,
            value,
            rationale,
            predecessor,
        })
    }

    /// Creates the initial score revision for an attempt.
    pub fn initial(
        attempt: AttemptId,
        evaluator: ArtifactDigest,
        evidence: Vec<ArtifactDigest>,
        metric: impl Into<String>,
        value: f64,
        rationale: ArtifactDigest,
    ) -> Result<Self, ScoreError> {
        Self::new(
            attempt, 0, evaluator, evidence, metric, value, rationale, None,
        )
    }

    /// Computes the immutable identity of this complete score revision.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut bytes = format!(
            "attempt={}\u{1f}version={}\u{1f}evaluator={}\u{1f}metric={}\u{1f}value={}\u{1f}rationale={}",
            self.attempt.as_str(),
            self.version,
            self.evaluator.as_str(),
            self.metric,
            self.value.to_bits(),
            self.rationale.as_str(),
        )
        .into_bytes();
        if let Some(predecessor) = &self.predecessor {
            bytes.extend_from_slice(b"\x1epredecessor=");
            bytes.extend_from_slice(predecessor.as_str().as_bytes());
        }
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
    /// Scores must retain a named metric.
    EmptyMetric,
    /// Scores must pin at least one immutable evidence identity.
    EmptyEvidence,
    /// Scores must remain serializable finite numbers.
    NonFiniteValue,
}

impl Display for ScoreError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyMetric => formatter.write_str("score metric must not be empty"),
            Self::EmptyEvidence => formatter.write_str("score evidence must not be empty"),
            Self::NonFiniteValue => formatter.write_str("score value must be finite"),
        }
    }
}

impl std::error::Error for ScoreError {}
