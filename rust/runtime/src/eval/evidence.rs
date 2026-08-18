// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Append-only immutable evidence identities for evaluation attempts.

use std::fmt::{self, Display, Formatter};

use serde::{Deserialize, Serialize};

use super::{
    ArtifactDigest, EvalIdentityError, ScoreVersion, VerifierResult, append_identity_field,
};

/// Stable identifier for one execution attempt of a resolved trial.
#[derive(Clone, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AttemptId(String);

impl AttemptId {
    /// Creates a nonempty attempt identifier.
    pub fn new(value: impl Into<String>) -> Result<Self, EvalIdentityError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(EvalIdentityError::Empty("attempt id"));
        }
        Ok(Self(value))
    }

    /// Borrows the attempt identifier.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl<'de> Deserialize<'de> for AttemptId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(serde::de::Error::custom)
    }
}

/// Category of one append-only evaluation evidence event.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKind {
    /// Agent lifecycle or decision evidence.
    Agent,
    /// Model request or response evidence.
    Llm,
    /// Tool request or result evidence.
    Tool,
    /// Sandbox lifecycle or policy evidence.
    Sandbox,
    /// Materialized artifact evidence.
    Artifact,
    /// Evaluator or verifier evidence.
    Evaluator,
    /// Security-policy evidence.
    Security,
    /// Bounded observation evidence from an externally driven compatibility episode.
    Compatibility,
}

impl EvidenceKind {
    /// Returns the stable evidence-kind spelling.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Llm => "llm",
            Self::Tool => "tool",
            Self::Sandbox => "sandbox",
            Self::Artifact => "artifact",
            Self::Evaluator => "evaluator",
            Self::Security => "security",
            Self::Compatibility => "compatibility",
        }
    }
}

/// One ordered immutable fact emitted by an evaluation attempt.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceEvent {
    /// Attempt that emitted this event.
    pub attempt: AttemptId,
    /// Monotonic sequence within the attempt.
    pub sequence: u64,
    /// Typed event category.
    pub kind: EvidenceKind,
    /// Digest of the immutable event payload.
    pub payload: ArtifactDigest,
    /// Optional parent event identity.
    pub parent: Option<ArtifactDigest>,
}

impl EvidenceEvent {
    /// Creates one immutable evidence event.
    pub fn new(
        attempt: AttemptId,
        sequence: u64,
        kind: EvidenceKind,
        payload: ArtifactDigest,
        parent: Option<ArtifactDigest>,
    ) -> Self {
        Self {
            attempt,
            sequence,
            kind,
            payload,
            parent,
        }
    }

    /// Computes the immutable identity of this event's complete contents.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let parent = self.parent.as_ref().map_or("", ArtifactDigest::as_str);
        ArtifactDigest::from_bytes(
            format!(
                "attempt={}\u{1f}sequence={}\u{1f}kind={}\u{1f}payload={}\u{1f}parent={parent}",
                self.attempt.as_str(),
                self.sequence,
                self.kind.as_str(),
                self.payload.as_str(),
            )
            .as_bytes(),
        )
    }
}

/// Immutable evidence and score lineage frozen from one completed Harbor attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct FrozenAttemptBundle {
    trial_digest: ArtifactDigest,
    attempt: AttemptId,
    lifecycle_evidence: Vec<EvidenceEvent>,
    verifier_result: VerifierResult,
    score_lineage: Vec<ScoreVersion>,
}

impl FrozenAttemptBundle {
    /// Freezes a completed attempt without treating lifecycle facts as verifier inputs.
    pub fn new(
        trial_digest: ArtifactDigest,
        verifier_result: VerifierResult,
        lifecycle_evidence: Vec<EvidenceEvent>,
        score_lineage: Vec<ScoreVersion>,
    ) -> Result<Self, FrozenAttemptError> {
        let attempt = verifier_result.attempt.clone();
        if lifecycle_evidence.is_empty() {
            return Err(FrozenAttemptError::EmptyLifecycleEvidence);
        }
        if verifier_result.evidence.is_empty() {
            return Err(FrozenAttemptError::EmptyVerifierInputEvidence);
        }
        for (index, evidence) in lifecycle_evidence.iter().enumerate() {
            if evidence.attempt != attempt {
                return Err(FrozenAttemptError::LifecycleAttemptMismatch { index });
            }
            let expected = u64::try_from(index)
                .map_err(|_| FrozenAttemptError::LifecycleSequenceOverflow { index })?;
            if evidence.sequence != expected {
                return Err(FrozenAttemptError::NonContiguousLifecycleSequence {
                    index,
                    expected,
                    actual: evidence.sequence,
                });
            }
        }
        if score_lineage.is_empty() {
            return Err(FrozenAttemptError::EmptyScoreLineage);
        }
        let mut predecessor = None;
        for (index, score) in score_lineage.iter().enumerate() {
            if score.attempt != attempt {
                return Err(FrozenAttemptError::ScoreAttemptMismatch { index });
            }
            let expected = u32::try_from(index)
                .map_err(|_| FrozenAttemptError::ScoreVersionOverflow { index })?;
            if score.version != expected {
                return Err(FrozenAttemptError::NonContiguousScoreVersion {
                    index,
                    expected,
                    actual: score.version,
                });
            }
            if score.evidence != verifier_result.evidence {
                return Err(FrozenAttemptError::ScoreVerifierInputMismatch { index });
            }
            if score.evaluator != verifier_result.verifier {
                return Err(FrozenAttemptError::ScoreEvaluatorMismatch { index });
            }
            if verifier_result
                .reward
                .metrics
                .get(&score.metric)
                .is_none_or(|value| value.to_bits() != score.value.to_bits())
            {
                return Err(FrozenAttemptError::ScoreRewardMismatch { index });
            }
            if score.predecessor != predecessor {
                return Err(FrozenAttemptError::ScorePredecessorMismatch { index });
            }
            predecessor = Some(score.identity_digest());
        }
        Ok(Self {
            trial_digest,
            attempt,
            lifecycle_evidence,
            verifier_result,
            score_lineage,
        })
    }

    /// Borrows the immutable trial identity.
    pub fn trial_digest(&self) -> &ArtifactDigest {
        &self.trial_digest
    }

    /// Borrows the append-only attempt identity.
    pub fn attempt(&self) -> &AttemptId {
        &self.attempt
    }

    /// Borrows the ordered lifecycle facts without relabeling them as verifier inputs.
    pub fn lifecycle_evidence(&self) -> &[EvidenceEvent] {
        &self.lifecycle_evidence
    }

    /// Borrows exactly the declared-artifact evidence supplied to the verifier.
    pub fn verifier_input_evidence(&self) -> &[ArtifactDigest] {
        &self.verifier_result.evidence
    }

    /// Borrows the immutable verifier result over the declared artifact inputs.
    pub fn verifier_result(&self) -> &VerifierResult {
        &self.verifier_result
    }

    /// Borrows every append-only score revision for the frozen attempt.
    pub fn score_lineage(&self) -> &[ScoreVersion] {
        &self.score_lineage
    }

    /// Borrows the selected latest immutable score revision.
    pub fn selected_score(&self) -> Option<&ScoreVersion> {
        self.score_lineage.last()
    }

    /// Computes a stable identity for the ordered lifecycle facts only.
    pub fn lifecycle_evidence_digest(&self) -> ArtifactDigest {
        let mut bytes = Vec::new();
        append_identity_field(&mut bytes, "domain", b"aiperf-eval-lifecycle-evidence-v1");
        for evidence in &self.lifecycle_evidence {
            append_identity_field(
                &mut bytes,
                "lifecycle-evidence",
                evidence.identity_digest().as_str().as_bytes(),
            );
        }
        ArtifactDigest::from_bytes(&bytes)
    }

    /// Computes an identity that covers the frozen lifecycle, verifier inputs, and score lineage.
    pub fn identity_digest(&self) -> ArtifactDigest {
        let mut bytes = Vec::new();
        append_identity_field(&mut bytes, "domain", b"aiperf-eval-frozen-attempt-v1");
        append_identity_field(&mut bytes, "trial", self.trial_digest.as_str().as_bytes());
        append_identity_field(&mut bytes, "attempt", self.attempt.as_str().as_bytes());
        append_identity_field(
            &mut bytes,
            "lifecycle",
            self.lifecycle_evidence_digest().as_str().as_bytes(),
        );
        for evidence in self.verifier_input_evidence() {
            append_identity_field(&mut bytes, "verifier-input", evidence.as_str().as_bytes());
        }
        for score in &self.score_lineage {
            append_identity_field(
                &mut bytes,
                "score",
                score.identity_digest().as_str().as_bytes(),
            );
        }
        ArtifactDigest::from_bytes(&bytes)
    }
}

/// Invalid evidence or score lineage supplied for a frozen attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FrozenAttemptError {
    /// A completed attempt must retain at least one lifecycle fact.
    EmptyLifecycleEvidence,
    /// A verifier must receive at least one declared artifact identity.
    EmptyVerifierInputEvidence,
    /// A lifecycle fact belonged to another attempt.
    LifecycleAttemptMismatch {
        /// Index of the mismatched lifecycle fact.
        index: usize,
    },
    /// The lifecycle index did not fit in its wire sequence field.
    LifecycleSequenceOverflow {
        /// Index that could not be represented.
        index: usize,
    },
    /// Lifecycle sequences must begin at zero and append without gaps.
    NonContiguousLifecycleSequence {
        /// Index of the invalid lifecycle fact.
        index: usize,
        /// Required sequence value.
        expected: u64,
        /// Observed sequence value.
        actual: u64,
    },
    /// A frozen attempt requires an initial score and its regrades.
    EmptyScoreLineage,
    /// A score belonged to another attempt.
    ScoreAttemptMismatch {
        /// Index of the mismatched score.
        index: usize,
    },
    /// The score index did not fit in its immutable version field.
    ScoreVersionOverflow {
        /// Index that could not be represented.
        index: usize,
    },
    /// Score versions must begin at zero and append without gaps.
    NonContiguousScoreVersion {
        /// Index of the invalid score.
        index: usize,
        /// Required version value.
        expected: u32,
        /// Observed version value.
        actual: u32,
    },
    /// A score changed the declared artifact evidence supplied to the verifier.
    ScoreVerifierInputMismatch {
        /// Index of the score with different verifier inputs.
        index: usize,
    },
    /// A score was not produced by the frozen verifier identity.
    ScoreEvaluatorMismatch {
        /// Index of the score with a foreign evaluator identity.
        index: usize,
    },
    /// A score did not preserve a named value emitted by the frozen verifier.
    ScoreRewardMismatch {
        /// Index of the score outside the frozen verifier reward document.
        index: usize,
    },
    /// A score did not name the immediately preceding immutable revision.
    ScorePredecessorMismatch {
        /// Index of the score with an invalid predecessor.
        index: usize,
    },
}

impl Display for FrozenAttemptError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLifecycleEvidence => {
                formatter.write_str("frozen attempt requires lifecycle evidence")
            }
            Self::EmptyVerifierInputEvidence => {
                formatter.write_str("frozen attempt requires verifier input evidence")
            }
            Self::LifecycleAttemptMismatch { index } => {
                write!(
                    formatter,
                    "lifecycle evidence {index} belongs to another attempt"
                )
            }
            Self::LifecycleSequenceOverflow { index } => {
                write!(
                    formatter,
                    "lifecycle evidence {index} exceeds sequence capacity"
                )
            }
            Self::NonContiguousLifecycleSequence {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "lifecycle evidence {index} has sequence {actual}, expected {expected}"
            ),
            Self::EmptyScoreLineage => formatter.write_str("frozen attempt requires score lineage"),
            Self::ScoreAttemptMismatch { index } => {
                write!(formatter, "score {index} belongs to another attempt")
            }
            Self::ScoreVersionOverflow { index } => {
                write!(formatter, "score {index} exceeds version capacity")
            }
            Self::NonContiguousScoreVersion {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "score {index} has version {actual}, expected {expected}"
            ),
            Self::ScoreVerifierInputMismatch { index } => {
                write!(formatter, "score {index} changed verifier input evidence")
            }
            Self::ScoreEvaluatorMismatch { index } => {
                write!(formatter, "score {index} has a foreign evaluator identity")
            }
            Self::ScoreRewardMismatch { index } => {
                write!(formatter, "score {index} differs from the verifier reward")
            }
            Self::ScorePredecessorMismatch { index } => {
                write!(formatter, "score {index} has an invalid predecessor")
            }
        }
    }
}

impl std::error::Error for FrozenAttemptError {}
