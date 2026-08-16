// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Immutable trajectory export references.

use std::collections::BTreeSet;

use crate::eval::{ArtifactDigest, AttemptId, EvidenceEvent};

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
        validate_evidence(&evidence)?;
        Ok(Self { attempt, evidence })
    }

    /// Builds an export manifest from ordered immutable events for one attempt.
    pub fn from_events(
        attempt: AttemptId,
        events: &[EvidenceEvent],
    ) -> Result<Self, TrainingError> {
        Self::new(attempt.clone(), event_identities(&attempt, events)?)
    }

    /// Verifies that an exported trajectory still names the supplied event history exactly.
    pub fn validate_against(&self, events: &[EvidenceEvent]) -> Result<(), TrainingError> {
        validate_evidence(&self.evidence)?;
        if self.evidence == event_identities(&self.attempt, events)? {
            Ok(())
        } else {
            Err(TrainingError::EvidenceMismatch)
        }
    }
}

fn event_identities(
    attempt: &AttemptId,
    events: &[EvidenceEvent],
) -> Result<Vec<ArtifactDigest>, TrainingError> {
    if events.is_empty() {
        return Err(TrainingError::EmptyEvidence);
    }

    let mut previous_sequence = None;
    let mut identities = Vec::with_capacity(events.len());
    for event in events {
        if &event.attempt != attempt {
            return Err(TrainingError::AttemptMismatch);
        }
        if previous_sequence.is_some_and(|sequence| event.sequence <= sequence) {
            return Err(TrainingError::OutOfOrderEvidence);
        }
        previous_sequence = Some(event.sequence);
        identities.push(event.identity_digest());
    }
    Ok(identities)
}

fn validate_evidence(evidence: &[ArtifactDigest]) -> Result<(), TrainingError> {
    if evidence.is_empty() {
        return Err(TrainingError::EmptyEvidence);
    }
    if evidence.iter().collect::<BTreeSet<_>>().len() != evidence.len() {
        return Err(TrainingError::DuplicateEvidence);
    }
    Ok(())
}

/// Invalid immutable trajectory export request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingError {
    /// Training exports must pin preserved evidence.
    EmptyEvidence,
    /// An evidence identity cannot appear twice in one immutable export.
    DuplicateEvidence,
    /// An event belongs to a different attempt than the export manifest.
    AttemptMismatch,
    /// Events were not supplied in their append-only sequence order.
    OutOfOrderEvidence,
    /// The supplied event history no longer matches the exported identities.
    EvidenceMismatch,
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyEvidence => {
                formatter.write_str("trajectory export requires immutable evidence")
            }
            Self::DuplicateEvidence => {
                formatter.write_str("trajectory export evidence must be unique")
            }
            Self::AttemptMismatch => {
                formatter.write_str("trajectory export evidence belongs to another attempt")
            }
            Self::OutOfOrderEvidence => {
                formatter.write_str("trajectory export evidence is not append-only")
            }
            Self::EvidenceMismatch => {
                formatter.write_str("trajectory export evidence no longer matches its manifest")
            }
        }
    }
}

impl std::error::Error for TrainingError {}
