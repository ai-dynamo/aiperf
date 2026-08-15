// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Append-only immutable evidence identities for evaluation attempts.

use serde::{Deserialize, Serialize};

use super::{ArtifactDigest, EvalIdentityError};

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
