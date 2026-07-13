// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed lifecycle-marker DTOs and descriptor-level validity rules.
//!
//! These values deliberately stop before terminal frame identity. The archive
//! owner assigns `record_seq`, then the control-frame codec validates this DTO,
//! derives the lifecycle batch/reservation/frame identities, and inserts the
//! resulting frame ID into the one Arrow marker row.

use std::collections::BTreeMap;
use std::fmt::{self, Display, Formatter};

use crate::{ArchiveId, BoundaryReference, BoundaryRole, Digest, SessionId};

/// Frozen marker-kind vocabulary from `markers-arrow-schema-v1.json`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LifecycleMarkerKindV1 {
    /// A collection session became active.
    SessionStarted = 1,
    /// A collection session stopped.
    SessionStopped = 2,
    /// A benchmark run started.
    RunStarted = 3,
    /// A benchmark run stopped.
    RunStopped = 4,
    /// A phase entered `STARTED`.
    PhaseStarted = 5,
    /// A phase entered `SENDING_COMPLETE`.
    PhaseSendingComplete = 6,
    /// A phase entered `COMPLETE`.
    PhaseComplete = 7,
    /// A physical telemetry source changed state.
    SourceState = 8,
    /// A source attribute epoch changed.
    TopologyChange = 9,
    /// Archive health degraded.
    ArchiveDegraded = 10,
    /// Archive health recovered.
    ArchiveRecovered = 11,
}

impl LifecycleMarkerKindV1 {
    /// Returns the exact frozen Enum8 value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SessionStarted => "session_started",
            Self::SessionStopped => "session_stopped",
            Self::RunStarted => "run_started",
            Self::RunStopped => "run_stopped",
            Self::PhaseStarted => "phase_started",
            Self::PhaseSendingComplete => "phase_sending_complete",
            Self::PhaseComplete => "phase_complete",
            Self::SourceState => "source_state",
            Self::TopologyChange => "topology_change",
            Self::ArchiveDegraded => "archive_degraded",
            Self::ArchiveRecovered => "archive_recovered",
        }
    }
}

/// Frozen phase-state vocabulary copied from one phase observer snapshot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LifecyclePhaseStateV1 {
    /// Phase accepts scheduled work.
    Started = 1,
    /// No new work is sent but accepted work may remain.
    SendingComplete = 2,
    /// Phase terminal accounting is frozen.
    Complete = 3,
}

impl LifecyclePhaseStateV1 {
    /// Returns the exact frozen Enum8 value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Started => "started",
            Self::SendingComplete => "sending_complete",
            Self::Complete => "complete",
        }
    }
}

/// Frozen terminal reason vocabulary for stopped/complete markers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LifecycleCompletionReasonV1 {
    /// Work drained normally.
    Completed = 1,
    /// The authored duration bound fired.
    Duration = 2,
    /// The authored request-count bound fired.
    RequestCount = 3,
    /// The authored session-count bound fired.
    SessionCount = 4,
    /// Cancellation policy stopped the work.
    Cancelled = 5,
    /// A runtime component failed.
    Failed = 6,
    /// Process shutdown stopped the work.
    Shutdown = 7,
}

impl LifecycleCompletionReasonV1 {
    /// Returns the exact frozen Enum8 value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Duration => "duration",
            Self::RequestCount => "request_count",
            Self::SessionCount => "session_count",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
            Self::Shutdown => "shutdown",
        }
    }
}

/// One owner-sequenced lifecycle marker before terminal frame derivation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LifecycleMarkerV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Owner-assigned global archive sequence.
    pub record_seq: u64,
    /// Marker sequence; v1 requires exact equality with `record_seq`.
    pub marker_seq: u64,
    /// Closed marker class.
    pub kind: LifecycleMarkerKindV1,
    /// Authoritative injected-Clock observation.
    pub clock_ns: i64,
    /// Optional Unix placement derived from the session epoch anchor.
    pub unix_epoch_ns: Option<i128>,
    /// Run identity when the marker is run-scoped.
    pub run_id: Option<String>,
    /// Phase identity for phase lifecycle markers.
    pub phase_id: Option<String>,
    /// Physical source for source-scoped markers or forced boundaries.
    pub source_id: Option<String>,
    /// Exact phase state for phase lifecycle markers.
    pub phase_state: Option<LifecyclePhaseStateV1>,
    /// Terminal reason for stopped/complete markers.
    pub completion_reason: Option<LifecycleCompletionReasonV1>,
    /// Complete forced-boundary join key, never a partial field set.
    pub boundary: Option<BoundaryReference>,
    /// Phase start Clock snapshot.
    pub phase_start_ns: Option<i64>,
    /// Phase sending-complete Clock snapshot.
    pub sent_end_ns: Option<i64>,
    /// Phase complete Clock snapshot.
    pub requests_end_ns: Option<i64>,
    /// New source attribute epoch for topology changes.
    pub attribute_epoch_id: Option<Digest>,
    /// Sanitized, UTF-8-byte-keyed marker attributes.
    pub attributes: BTreeMap<String, String>,
}

impl LifecycleMarkerV1 {
    /// Validates the closed marker-kind/field matrix before identity derivation.
    pub fn validate(&self) -> Result<(), LifecycleMarkerError> {
        if self.marker_seq != self.record_seq {
            return Err(LifecycleMarkerError::MarkerSequenceMismatch {
                record_seq: self.record_seq,
                marker_seq: self.marker_seq,
            });
        }
        for (field, value) in [
            ("run_id", self.run_id.as_deref()),
            ("phase_id", self.phase_id.as_deref()),
            ("source_id", self.source_id.as_deref()),
        ] {
            if let Some(value) = value {
                validate_identifier(field, value)?;
            }
        }
        validate_attributes(&self.attributes)?;

        match self.kind {
            LifecycleMarkerKindV1::SessionStarted => {
                self.require_scope(false, false, false)?;
                self.require_completion(false)?;
                self.require_non_phase_fields()?;
            }
            LifecycleMarkerKindV1::SessionStopped => {
                self.require_scope(false, false, false)?;
                self.require_completion(true)?;
                self.require_non_phase_fields()?;
            }
            LifecycleMarkerKindV1::RunStarted => {
                self.require_scope(true, false, false)?;
                self.require_completion(false)?;
                self.require_non_phase_fields()?;
            }
            LifecycleMarkerKindV1::RunStopped => {
                self.require_scope(true, false, false)?;
                self.require_completion(true)?;
                self.require_non_phase_fields()?;
            }
            LifecycleMarkerKindV1::PhaseStarted => {
                self.validate_phase(LifecyclePhaseStateV1::Started)?;
            }
            LifecycleMarkerKindV1::PhaseSendingComplete => {
                self.validate_phase(LifecyclePhaseStateV1::SendingComplete)?;
            }
            LifecycleMarkerKindV1::PhaseComplete => {
                self.validate_phase(LifecyclePhaseStateV1::Complete)?;
            }
            LifecycleMarkerKindV1::SourceState => {
                self.require_source_scope(false)?;
            }
            LifecycleMarkerKindV1::TopologyChange => {
                self.require_source_scope(true)?;
            }
            LifecycleMarkerKindV1::ArchiveDegraded | LifecycleMarkerKindV1::ArchiveRecovered => {
                if self.phase_id.is_some() || self.phase_state.is_some() || self.boundary.is_some()
                {
                    return Err(LifecycleMarkerError::UnexpectedPhaseFields(self.kind));
                }
                self.require_completion(false)?;
                self.require_phase_times()?;
                if self.attribute_epoch_id.is_some() {
                    return Err(LifecycleMarkerError::UnexpectedAttributeEpoch(self.kind));
                }
            }
        }
        Ok(())
    }

    fn require_scope(
        &self,
        run_required: bool,
        phase_required: bool,
        source_required: bool,
    ) -> Result<(), LifecycleMarkerError> {
        require_presence("run_id", self.run_id.is_some(), run_required, self.kind)?;
        require_presence(
            "phase_id",
            self.phase_id.is_some(),
            phase_required,
            self.kind,
        )?;
        require_presence(
            "source_id",
            self.source_id.is_some(),
            source_required,
            self.kind,
        )
    }

    fn require_completion(&self, required: bool) -> Result<(), LifecycleMarkerError> {
        require_presence(
            "completion_reason",
            self.completion_reason.is_some(),
            required,
            self.kind,
        )
    }

    fn require_non_phase_fields(&self) -> Result<(), LifecycleMarkerError> {
        if self.phase_state.is_some() || self.boundary.is_some() {
            return Err(LifecycleMarkerError::UnexpectedPhaseFields(self.kind));
        }
        self.require_phase_times()?;
        if self.attribute_epoch_id.is_some() {
            return Err(LifecycleMarkerError::UnexpectedAttributeEpoch(self.kind));
        }
        Ok(())
    }

    fn require_source_scope(&self, topology_change: bool) -> Result<(), LifecycleMarkerError> {
        if self.source_id.is_none() {
            return Err(LifecycleMarkerError::MissingField {
                kind: self.kind,
                field: "source_id",
            });
        }
        if self.phase_id.is_some() || self.phase_state.is_some() || self.boundary.is_some() {
            return Err(LifecycleMarkerError::UnexpectedPhaseFields(self.kind));
        }
        self.require_completion(false)?;
        self.require_phase_times()?;
        require_presence(
            "attribute_epoch_id",
            self.attribute_epoch_id.is_some(),
            topology_change,
            self.kind,
        )
    }

    fn validate_phase(
        &self,
        expected_state: LifecyclePhaseStateV1,
    ) -> Result<(), LifecycleMarkerError> {
        self.require_scope(true, true, self.boundary.is_some())?;
        if self.phase_state != Some(expected_state) {
            return Err(LifecycleMarkerError::PhaseStateMismatch {
                kind: self.kind,
                expected: expected_state,
                actual: self.phase_state,
            });
        }
        self.require_completion(expected_state == LifecyclePhaseStateV1::Complete)?;
        self.require_phase_times()?;
        if self.attribute_epoch_id.is_some() {
            return Err(LifecycleMarkerError::UnexpectedAttributeEpoch(self.kind));
        }
        if let Some(boundary) = &self.boundary {
            validate_boundary(boundary)?;
            if self.phase_id.as_deref() != Some(boundary.phase_id.as_str()) {
                return Err(LifecycleMarkerError::BoundaryPhaseMismatch);
            }
            if self.source_id.as_deref() != Some(boundary.source_id.as_str()) {
                return Err(LifecycleMarkerError::BoundarySourceMismatch);
            }
            let expected_role = if expected_state == LifecyclePhaseStateV1::Started {
                BoundaryRole::PhaseStart
            } else {
                BoundaryRole::PhaseEnd
            };
            if boundary.role != expected_role {
                return Err(LifecycleMarkerError::BoundaryRoleMismatch {
                    expected: expected_role,
                    actual: boundary.role,
                });
            }
        } else if self.source_id.is_some() {
            return Err(LifecycleMarkerError::OrdinaryPhaseHasSource);
        }
        Ok(())
    }

    fn require_phase_times(&self) -> Result<(), LifecycleMarkerError> {
        let expected = match self.kind {
            LifecycleMarkerKindV1::PhaseStarted => (true, false, false),
            LifecycleMarkerKindV1::PhaseSendingComplete => (true, true, false),
            LifecycleMarkerKindV1::PhaseComplete => (true, true, true),
            _ => (false, false, false),
        };
        let actual = (
            self.phase_start_ns.is_some(),
            self.sent_end_ns.is_some(),
            self.requests_end_ns.is_some(),
        );
        if actual != expected {
            return Err(LifecycleMarkerError::PhaseTimeShape {
                kind: self.kind,
                expected,
                actual,
            });
        }
        if let (Some(start), Some(sent)) = (self.phase_start_ns, self.sent_end_ns)
            && sent < start
        {
            return Err(LifecycleMarkerError::PhaseTimeOrder);
        }
        if let (Some(sent), Some(requests)) = (self.sent_end_ns, self.requests_end_ns)
            && requests < sent
        {
            return Err(LifecycleMarkerError::PhaseTimeOrder);
        }
        Ok(())
    }
}

fn require_presence(
    field: &'static str,
    present: bool,
    required: bool,
    kind: LifecycleMarkerKindV1,
) -> Result<(), LifecycleMarkerError> {
    match (present, required) {
        (false, true) => Err(LifecycleMarkerError::MissingField { kind, field }),
        (true, false) => Err(LifecycleMarkerError::UnexpectedField { kind, field }),
        _ => Ok(()),
    }
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), LifecycleMarkerError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(LifecycleMarkerError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_attributes(attributes: &BTreeMap<String, String>) -> Result<(), LifecycleMarkerError> {
    for (key, value) in attributes {
        validate_identifier("attribute key", key)?;
        if value.chars().any(char::is_control) {
            return Err(LifecycleMarkerError::InvalidAttributeValue(key.clone()));
        }
    }
    Ok(())
}

fn validate_boundary(boundary: &BoundaryReference) -> Result<(), LifecycleMarkerError> {
    for (field, value) in [
        ("boundary.transition_id", boundary.transition_id.as_str()),
        ("boundary.boundary_id", boundary.boundary_id.as_str()),
        ("boundary.phase_id", boundary.phase_id.as_str()),
        ("boundary.source_id", boundary.source_id.as_str()),
    ] {
        validate_identifier(field, value)?;
    }
    if let Some(group) = &boundary.coalescing_group_id {
        validate_identifier("boundary.coalescing_group_id", group)?;
    }
    Ok(())
}

/// Invalid lifecycle-marker DTO.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LifecycleMarkerError {
    /// Marker and global record sequences diverged.
    MarkerSequenceMismatch {
        /// Owner-assigned global sequence.
        record_seq: u64,
        /// Supplied marker sequence.
        marker_seq: u64,
    },
    /// A required field was absent.
    MissingField {
        /// Marker kind selecting the validity row.
        kind: LifecycleMarkerKindV1,
        /// Missing field.
        field: &'static str,
    },
    /// A forbidden field was present.
    UnexpectedField {
        /// Marker kind selecting the validity row.
        kind: LifecycleMarkerKindV1,
        /// Present field.
        field: &'static str,
    },
    /// Phase-only fields appeared on a non-phase marker.
    UnexpectedPhaseFields(LifecycleMarkerKindV1),
    /// Attribute epoch appeared outside topology change.
    UnexpectedAttributeEpoch(LifecycleMarkerKindV1),
    /// Identifier was empty, padded, or contained controls.
    InvalidIdentifier {
        /// Invalid field.
        field: &'static str,
        /// Redaction-safe value.
        value: String,
    },
    /// Attribute value contained a control character.
    InvalidAttributeValue(String),
    /// Phase kind and state disagreed.
    PhaseStateMismatch {
        /// Marker kind.
        kind: LifecycleMarkerKindV1,
        /// Required state.
        expected: LifecyclePhaseStateV1,
        /// Supplied state.
        actual: Option<LifecyclePhaseStateV1>,
    },
    /// Phase time fields did not match the selected transition.
    PhaseTimeShape {
        /// Marker kind.
        kind: LifecycleMarkerKindV1,
        /// Required start/sent/requests presence.
        expected: (bool, bool, bool),
        /// Actual start/sent/requests presence.
        actual: (bool, bool, bool),
    },
    /// Phase Clock snapshots were not monotonic.
    PhaseTimeOrder,
    /// Boundary phase did not equal the marker phase.
    BoundaryPhaseMismatch,
    /// Boundary source did not equal the marker source.
    BoundarySourceMismatch,
    /// Boundary role did not match the phase transition.
    BoundaryRoleMismatch {
        /// Role required by the transition.
        expected: BoundaryRole,
        /// Supplied role.
        actual: BoundaryRole,
    },
    /// A non-forced phase marker carried a source.
    OrdinaryPhaseHasSource,
}

impl Display for LifecycleMarkerError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MarkerSequenceMismatch {
                record_seq,
                marker_seq,
            } => write!(
                formatter,
                "lifecycle marker_seq {marker_seq} does not equal record_seq {record_seq}"
            ),
            Self::MissingField { kind, field } => {
                write!(formatter, "lifecycle marker {kind:?} requires {field}")
            }
            Self::UnexpectedField { kind, field } => {
                write!(formatter, "lifecycle marker {kind:?} forbids {field}")
            }
            Self::UnexpectedPhaseFields(kind) => {
                write!(
                    formatter,
                    "lifecycle marker {kind:?} carried phase-only fields"
                )
            }
            Self::UnexpectedAttributeEpoch(kind) => write!(
                formatter,
                "lifecycle marker {kind:?} carried a topology-only attribute epoch"
            ),
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "{field} has invalid identifier {value:?}")
            }
            Self::InvalidAttributeValue(key) => {
                write!(
                    formatter,
                    "lifecycle attribute {key:?} contains a control character"
                )
            }
            Self::PhaseStateMismatch {
                kind,
                expected,
                actual,
            } => write!(
                formatter,
                "lifecycle marker {kind:?} has phase state {actual:?}; expected {expected:?}"
            ),
            Self::PhaseTimeShape {
                kind,
                expected,
                actual,
            } => write!(
                formatter,
                "lifecycle marker {kind:?} has phase-time presence {actual:?}; expected {expected:?}"
            ),
            Self::PhaseTimeOrder => {
                formatter.write_str("lifecycle phase Clock snapshots are not monotonic")
            }
            Self::BoundaryPhaseMismatch => {
                formatter.write_str("lifecycle boundary phase does not match marker phase")
            }
            Self::BoundarySourceMismatch => {
                formatter.write_str("lifecycle boundary source does not match marker source")
            }
            Self::BoundaryRoleMismatch { expected, actual } => write!(
                formatter,
                "lifecycle boundary role {actual:?} does not match expected {expected:?}"
            ),
            Self::OrdinaryPhaseHasSource => formatter.write_str(
                "ordinary lifecycle phase marker cannot carry a source without a boundary",
            ),
        }
    }
}

impl std::error::Error for LifecycleMarkerError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(seed: u8) -> [u8; 16] {
        let mut value = [seed; 16];
        value[15] = seed.wrapping_add(1);
        value
    }

    fn phase_started() -> LifecycleMarkerV1 {
        LifecycleMarkerV1 {
            archive_id: ArchiveId::new(id(1)).unwrap(),
            session_id: SessionId::new(id(2)).unwrap(),
            record_seq: 7,
            marker_seq: 7,
            kind: LifecycleMarkerKindV1::PhaseStarted,
            clock_ns: 100,
            unix_epoch_ns: Some(1_700_000_000_000_000_100),
            run_id: Some("run-a".to_owned()),
            phase_id: Some("profiling".to_owned()),
            source_id: None,
            phase_state: Some(LifecyclePhaseStateV1::Started),
            completion_reason: None,
            boundary: None,
            phase_start_ns: Some(100),
            sent_end_ns: None,
            requests_end_ns: None,
            attribute_epoch_id: None,
            attributes: BTreeMap::new(),
        }
    }

    #[test]
    fn phase_matrix_rejects_sequence_state_and_clock_ambiguity() {
        let mut marker = phase_started();
        assert_eq!(marker.validate(), Ok(()));

        marker.marker_seq += 1;
        assert!(matches!(
            marker.validate(),
            Err(LifecycleMarkerError::MarkerSequenceMismatch { .. })
        ));
        marker.marker_seq = marker.record_seq;
        marker.phase_state = Some(LifecyclePhaseStateV1::Complete);
        assert!(matches!(
            marker.validate(),
            Err(LifecycleMarkerError::PhaseStateMismatch { .. })
        ));
        marker.phase_state = Some(LifecyclePhaseStateV1::Started);
        marker.sent_end_ns = Some(101);
        assert!(matches!(
            marker.validate(),
            Err(LifecycleMarkerError::PhaseTimeShape { .. })
        ));
    }

    #[test]
    fn forced_phase_boundary_requires_complete_matching_reference() {
        let mut marker = phase_started();
        marker.source_id = Some("server-a".to_owned());
        marker.boundary = Some(BoundaryReference {
            transition_id: "warmup-to-profiling".to_owned(),
            boundary_id: "server-a-profiling-start".to_owned(),
            phase_id: "profiling".to_owned(),
            source_id: "server-a".to_owned(),
            role: BoundaryRole::PhaseStart,
            coalescing_group_id: None,
        });
        assert_eq!(marker.validate(), Ok(()));

        marker.boundary.as_mut().unwrap().role = BoundaryRole::PhaseEnd;
        assert!(matches!(
            marker.validate(),
            Err(LifecycleMarkerError::BoundaryRoleMismatch { .. })
        ));
    }
}
