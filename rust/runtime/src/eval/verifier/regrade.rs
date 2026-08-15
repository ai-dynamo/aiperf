// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable verifier results and append-only score regrades.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

use crate::eval::{ArtifactDigest, AttemptId, ScoreError, ScoreVersion};

use super::RewardDocument;

/// Immutable output of a verifier over one preserved evaluation attempt.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct VerifierResult {
    /// Attempt evaluated by this verifier.
    pub attempt: AttemptId,
    /// Immutable verifier implementation identity.
    pub verifier: ArtifactDigest,
    /// Evidence provided to the verifier.
    pub evidence: Vec<ArtifactDigest>,
    /// Finite named reward metrics emitted by the verifier.
    pub reward: RewardDocument,
    /// Immutable verifier rationale artifact.
    pub rationale: ArtifactDigest,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawVerifierResult {
    attempt: AttemptId,
    verifier: ArtifactDigest,
    evidence: Vec<ArtifactDigest>,
    reward: RewardDocument,
    rationale: ArtifactDigest,
}

impl<'de> Deserialize<'de> for VerifierResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawVerifierResult::deserialize(deserializer)?;
        Self::new(
            raw.attempt,
            raw.verifier,
            raw.evidence,
            raw.reward,
            raw.rationale,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl VerifierResult {
    /// Creates a scoreable verifier result over preserved evidence.
    pub fn new(
        attempt: AttemptId,
        verifier: ArtifactDigest,
        evidence: Vec<ArtifactDigest>,
        reward: RewardDocument,
        rationale: ArtifactDigest,
    ) -> Result<Self, RegradeError> {
        if evidence.is_empty() {
            return Err(RegradeError::EmptyEvidence);
        }
        Ok(Self {
            attempt,
            verifier,
            evidence,
            reward,
            rationale,
        })
    }
}

/// A request to append one score revision from a pinned verifier result.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct RegradeRequest {
    /// The existing immutable score revision that this regrade follows.
    pub previous: ScoreVersion,
    /// Result from the newly selected verifier.
    pub result: VerifierResult,
    /// Exact verifier metric to preserve as the score value.
    pub metric: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRegradeRequest {
    previous: ScoreVersion,
    result: VerifierResult,
    metric: String,
}

impl<'de> Deserialize<'de> for RegradeRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = RawRegradeRequest::deserialize(deserializer)?;
        Self::new(raw.previous, raw.result, raw.metric).map_err(serde::de::Error::custom)
    }
}

impl RegradeRequest {
    /// Creates a request that selects one explicit verifier metric.
    pub fn new(
        previous: ScoreVersion,
        result: VerifierResult,
        metric: impl Into<String>,
    ) -> Result<Self, RegradeError> {
        let metric = metric.into();
        if metric.trim().is_empty() {
            return Err(RegradeError::EmptyMetric);
        }
        Ok(Self {
            previous,
            result,
            metric,
        })
    }
}

/// Appends an immutable score revision without changing the prior score or evidence.
pub fn regrade(request: RegradeRequest) -> Result<ScoreVersion, RegradeError> {
    if request.previous.attempt != request.result.attempt {
        return Err(RegradeError::AttemptMismatch);
    }
    let value = request
        .result
        .reward
        .metrics
        .get(&request.metric)
        .copied()
        .ok_or_else(|| RegradeError::MetricNotFound(request.metric.clone()))?;
    let version = request
        .previous
        .version
        .checked_add(1)
        .ok_or(RegradeError::VersionOverflow)?;
    let predecessor = request.previous.identity_digest();
    ScoreVersion::new(
        request.previous.attempt,
        version,
        request.result.verifier,
        request.result.evidence,
        request.metric,
        value,
        request.result.rationale,
        Some(predecessor),
    )
    .map_err(RegradeError::InvalidScore)
}

/// Invalid verifier-result lineage or score-regrade request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegradeError {
    /// Regrades cannot select a score from a different attempt.
    AttemptMismatch,
    /// Regrades must select a named metric.
    EmptyMetric,
    /// Verifier results must retain immutable evidence.
    EmptyEvidence,
    /// The selected verifier metric was absent.
    MetricNotFound(String),
    /// The prior score already uses the last possible revision number.
    VersionOverflow,
    /// The appended score violates its immutable contract.
    InvalidScore(ScoreError),
}

impl Display for RegradeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::AttemptMismatch => {
                formatter.write_str("regrade attempt does not match prior score")
            }
            Self::EmptyMetric => formatter.write_str("regrade metric must not be empty"),
            Self::EmptyEvidence => formatter.write_str("verifier evidence must not be empty"),
            Self::MetricNotFound(metric) => {
                write!(formatter, "verifier metric is absent: {metric}")
            }
            Self::VersionOverflow => formatter.write_str("score revision cannot be appended"),
            Self::InvalidScore(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for RegradeError {}
