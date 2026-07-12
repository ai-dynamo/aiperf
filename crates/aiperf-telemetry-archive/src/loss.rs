// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed exact-loss and cumulative saturation DTOs.
//!
//! Missed cadence is represented only by tick/deadline ranges. Loss after an
//! issued source event is represented only by source/request sequences. The
//! split is validated before frame identity, Arrow evidence, or WAL bytes can
//! become durable, so absence never masquerades as a scrape attempt.

use std::collections::BTreeSet;
use std::fmt::{self, Display, Formatter};

use crate::{ArchiveId, BoundaryReference, Digest, SessionId, domain_digest};

/// Frozen loss-kind vocabulary from `losses-arrow-schema-v1.json`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LossKindV1 {
    /// No source work was issued for one or more fixed cadence ticks.
    MissedCadence = 1,
    /// Native work was delivered but bounded archive admission rejected it.
    ArchiveRejected = 2,
    /// Accepted source projection or owner terminalization failed.
    ProjectionFailed = 3,
    /// The archive writer failed after work existed.
    WriterFailed = 4,
    /// Shutdown abandoned accepted work at its deadline.
    ShutdownAbandoned = 5,
}

impl LossKindV1 {
    /// Returns the exact frozen Enum8 value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MissedCadence => "missed_cadence",
            Self::ArchiveRejected => "archive_rejected",
            Self::ProjectionFailed => "projection_failed",
            Self::WriterFailed => "writer_failed",
            Self::ShutdownAbandoned => "shutdown_abandoned",
        }
    }

    /// Returns the only legal v1 reason for this loss class.
    #[must_use]
    pub const fn reason(self) -> LossReasonV1 {
        match self {
            Self::MissedCadence => LossReasonV1::CadenceOverrun,
            Self::ArchiveRejected => LossReasonV1::ArchiveAdmissionRejected,
            Self::ProjectionFailed => LossReasonV1::ProjectionError,
            Self::WriterFailed => LossReasonV1::WriterError,
            Self::ShutdownAbandoned => LossReasonV1::ShutdownDeadline,
        }
    }
}

/// Frozen loss-reason vocabulary from `losses-arrow-schema-v1.json`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum LossReasonV1 {
    /// A source was still active at its next fixed cadence deadline.
    CadenceOverrun = 1,
    /// Bounded archive admission was unavailable.
    ArchiveAdmissionRejected = 2,
    /// Projection of an accepted event failed.
    ProjectionError = 3,
    /// The archive writer stopped making valid progress.
    WriterError = 4,
    /// The shutdown deadline expired.
    ShutdownDeadline = 5,
}

impl LossReasonV1 {
    /// Returns the exact frozen Enum8 value.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CadenceOverrun => "cadence_overrun",
            Self::ArchiveAdmissionRejected => "archive_admission_rejected",
            Self::ProjectionError => "projection_error",
            Self::WriterError => "writer_error",
            Self::ShutdownDeadline => "shutdown_deadline",
        }
    }
}

/// One exact, coalesced loss row before terminal frame derivation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExactLossRangeV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Physical source, or the explicit global sentinel.
    pub source_id: Option<String>,
    /// Owner-assigned global archive sequence.
    pub record_seq: u64,
    /// Session-global loss sequence.
    pub loss_seq: u64,
    /// Number of omitted entries represented by the inclusive range.
    pub count: u64,
    /// Semantic loss class.
    pub loss_kind: LossKindV1,
    /// Closed reason paired with the loss class.
    pub reason: LossReasonV1,
    /// First omitted source-record sequence.
    pub first_source_record_seq: Option<u64>,
    /// Last omitted source-record sequence.
    pub last_source_record_seq: Option<u64>,
    /// First omitted physical request-attempt sequence.
    pub first_request_attempt_seq: Option<u64>,
    /// Last omitted physical request-attempt sequence.
    pub last_request_attempt_seq: Option<u64>,
    /// First missed cadence tick.
    pub first_tick: Option<u64>,
    /// Last missed cadence tick.
    pub last_tick: Option<u64>,
    /// First missed absolute Clock deadline.
    pub first_deadline_ns: Option<i64>,
    /// Last missed absolute Clock deadline.
    pub last_deadline_ns: Option<i64>,
    /// Injected-Clock observation at which the range was sealed.
    pub loss_observed_ns: i64,
    /// Exact retained boundary references.
    pub boundary_refs: Vec<BoundaryReference>,
    /// Boundary references folded into the overflow digest.
    pub boundary_overflow_count: u64,
    /// Digest over canonical overflowed boundary references.
    pub boundary_overflow_digest: Option<Digest>,
}

impl ExactLossRangeV1 {
    /// Validates the exact-loss role matrix and every inclusive count equation.
    pub fn validate(&self) -> Result<(), LossValidationError> {
        validate_kind_reason(self.loss_kind, self.reason)?;
        validate_source(self.source_id.as_deref())?;
        if self.count == 0 {
            return Err(LossValidationError::ZeroCount);
        }
        validate_optional_u64_range(
            "source_record_seq",
            self.first_source_record_seq,
            self.last_source_record_seq,
        )?;
        validate_optional_u64_range(
            "request_attempt_seq",
            self.first_request_attempt_seq,
            self.last_request_attempt_seq,
        )?;
        validate_optional_u64_range("tick", self.first_tick, self.last_tick)?;
        validate_optional_i64_range("deadline_ns", self.first_deadline_ns, self.last_deadline_ns)?;
        validate_boundary_evidence(
            self.source_id.as_deref(),
            &self.boundary_refs,
            self.boundary_overflow_count,
            self.boundary_overflow_digest,
        )?;

        if self.loss_kind == LossKindV1::MissedCadence {
            if self.source_id.is_none() {
                return Err(LossValidationError::MissedCadenceRequiresSource);
            }
            require_absent("source_record_seq", self.first_source_record_seq)?;
            require_absent("request_attempt_seq", self.first_request_attempt_seq)?;
            let ticks = required_width("tick", self.first_tick, self.last_tick)?;
            if ticks != self.count {
                return Err(LossValidationError::CountEquation {
                    field: "tick",
                    expected: ticks,
                    actual: self.count,
                });
            }
            require_present("deadline_ns", self.first_deadline_ns)?;
        } else {
            require_absent("tick", self.first_tick)?;
            require_absent("deadline_ns", self.first_deadline_ns)?;
            match self.source_id.as_deref() {
                Some(_) => {
                    let source_width = required_width(
                        "source_record_seq",
                        self.first_source_record_seq,
                        self.last_source_record_seq,
                    )?;
                    if source_width != self.count {
                        return Err(LossValidationError::CountEquation {
                            field: "source_record_seq",
                            expected: source_width,
                            actual: self.count,
                        });
                    }
                    if self.first_request_attempt_seq.is_some() {
                        let request_width = required_width(
                            "request_attempt_seq",
                            self.first_request_attempt_seq,
                            self.last_request_attempt_seq,
                        )?;
                        if request_width != self.count {
                            return Err(LossValidationError::CountEquation {
                                field: "request_attempt_seq",
                                expected: request_width,
                                actual: self.count,
                            });
                        }
                    }
                }
                None => {
                    if matches!(
                        self.loss_kind,
                        LossKindV1::ArchiveRejected | LossKindV1::ProjectionFailed
                    ) {
                        return Err(LossValidationError::IssuedLossRequiresSource(
                            self.loss_kind,
                        ));
                    }
                    require_absent("source_record_seq", self.first_source_record_seq)?;
                    require_absent("request_attempt_seq", self.first_request_attempt_seq)?;
                }
            }
        }
        Ok(())
    }
}

/// Latest cumulative snapshot for one fixed-memory loss-saturation slot.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LossSaturationSnapshotV1 {
    /// Archive identity.
    pub archive_id: ArchiveId,
    /// Collection session identity.
    pub session_id: SessionId,
    /// Physical source, or the explicit global sentinel.
    pub source_id: Option<String>,
    /// Owner-assigned global archive sequence.
    pub record_seq: u64,
    /// Session-global loss sequence for this snapshot row.
    pub loss_seq: u64,
    /// Semantic loss class shared by the saturation slot.
    pub loss_kind: LossKindV1,
    /// Closed reason paired with the loss class.
    pub reason: LossReasonV1,
    /// First cumulative source-record sequence, when applicable.
    pub first_source_record_seq: Option<u64>,
    /// Last cumulative source-record sequence, when applicable.
    pub last_source_record_seq: Option<u64>,
    /// First cumulative request-attempt sequence, when applicable.
    pub first_request_attempt_seq: Option<u64>,
    /// Last cumulative request-attempt sequence, when applicable.
    pub last_request_attempt_seq: Option<u64>,
    /// First cumulative missed tick, when applicable.
    pub first_tick: Option<u64>,
    /// Last cumulative missed tick, when applicable.
    pub last_tick: Option<u64>,
    /// First cumulative missed deadline, when applicable.
    pub first_deadline_ns: Option<i64>,
    /// Last cumulative missed deadline, when applicable.
    pub last_deadline_ns: Option<i64>,
    /// Injected-Clock observation at which this snapshot was sealed.
    pub loss_observed_ns: i64,
    /// Stable slot identity derived from archive/session/source/kind/reason.
    pub saturation_slot_id: Digest,
    /// Monotonic slot-local snapshot sequence.
    pub saturation_snapshot_seq: u64,
    /// Exact cumulative number of non-enumerable ranges.
    pub cumulative_omitted_range_count: u64,
    /// Exact cumulative number of entries represented by those ranges.
    pub cumulative_omitted_entry_count: u64,
    /// Order-sensitive digest over canonical omitted entries.
    pub omitted_rolling_digest: Digest,
}

impl LossSaturationSnapshotV1 {
    /// Returns the row's `count`, fixed to cumulative omitted entries in v1.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.cumulative_omitted_entry_count
    }

    /// Validates cumulative slot identity and the overflow-summary field matrix.
    pub fn validate(&self) -> Result<(), LossValidationError> {
        validate_kind_reason(self.loss_kind, self.reason)?;
        validate_source(self.source_id.as_deref())?;
        if self.cumulative_omitted_range_count == 0
            || self.cumulative_omitted_entry_count == 0
            || self.cumulative_omitted_range_count > self.cumulative_omitted_entry_count
        {
            return Err(LossValidationError::InvalidCumulativeCounts {
                ranges: self.cumulative_omitted_range_count,
                entries: self.cumulative_omitted_entry_count,
            });
        }
        for (field, first, last) in [
            (
                "source_record_seq",
                self.first_source_record_seq,
                self.last_source_record_seq,
            ),
            (
                "request_attempt_seq",
                self.first_request_attempt_seq,
                self.last_request_attempt_seq,
            ),
            ("tick", self.first_tick, self.last_tick),
        ] {
            validate_optional_u64_range(field, first, last)?;
        }
        validate_optional_i64_range("deadline_ns", self.first_deadline_ns, self.last_deadline_ns)?;

        if self.loss_kind == LossKindV1::MissedCadence {
            if self.source_id.is_none() {
                return Err(LossValidationError::MissedCadenceRequiresSource);
            }
            require_absent("source_record_seq", self.first_source_record_seq)?;
            require_absent("request_attempt_seq", self.first_request_attempt_seq)?;
            require_present("tick", self.first_tick)?;
            require_present("deadline_ns", self.first_deadline_ns)?;
        } else {
            require_absent("tick", self.first_tick)?;
            require_absent("deadline_ns", self.first_deadline_ns)?;
            match self.source_id.as_deref() {
                Some(_) => require_present("source_record_seq", self.first_source_record_seq)?,
                None => {
                    if matches!(
                        self.loss_kind,
                        LossKindV1::ArchiveRejected | LossKindV1::ProjectionFailed
                    ) {
                        return Err(LossValidationError::IssuedLossRequiresSource(
                            self.loss_kind,
                        ));
                    }
                    require_absent("source_record_seq", self.first_source_record_seq)?;
                    require_absent("request_attempt_seq", self.first_request_attempt_seq)?;
                }
            }
        }

        let expected = loss_saturation_slot_id_v1(
            self.archive_id,
            self.session_id,
            self.source_id.as_deref(),
            self.loss_kind,
            self.reason,
        );
        if self.saturation_slot_id != expected {
            return Err(LossValidationError::SaturationSlotIdentityMismatch);
        }
        Ok(())
    }
}

/// Derives the stable v1 saturation-slot identity for one bounded tuple.
#[must_use]
pub fn loss_saturation_slot_id_v1(
    archive_id: ArchiveId,
    session_id: SessionId,
    source_id: Option<&str>,
    loss_kind: LossKindV1,
    reason: LossReasonV1,
) -> Digest {
    let source = optional_source_bytes(source_id);
    domain_digest(
        "aiperf.archive.loss-saturation-slot.v1",
        &[
            archive_id.as_bytes(),
            session_id.as_bytes(),
            &source,
            &[loss_kind as u8],
            &[reason as u8],
        ],
    )
}

fn validate_kind_reason(kind: LossKindV1, reason: LossReasonV1) -> Result<(), LossValidationError> {
    if reason != kind.reason() {
        return Err(LossValidationError::ReasonMismatch {
            kind,
            expected: kind.reason(),
            actual: reason,
        });
    }
    Ok(())
}

fn validate_source(source_id: Option<&str>) -> Result<(), LossValidationError> {
    if let Some(source_id) = source_id {
        validate_identifier("source_id", source_id)?;
    }
    Ok(())
}

fn validate_boundary_evidence(
    source_id: Option<&str>,
    references: &[BoundaryReference],
    overflow_count: u64,
    overflow_digest: Option<Digest>,
) -> Result<(), LossValidationError> {
    match (overflow_count, overflow_digest) {
        (0, None) | (1.., Some(_)) => {}
        _ => return Err(LossValidationError::BoundaryOverflowShape),
    }
    if source_id.is_none() && !references.is_empty() {
        return Err(LossValidationError::GlobalBoundaryReference);
    }
    let mut keys = BTreeSet::new();
    for reference in references {
        for (field, value) in [
            ("boundary.transition_id", reference.transition_id.as_str()),
            ("boundary.boundary_id", reference.boundary_id.as_str()),
            ("boundary.phase_id", reference.phase_id.as_str()),
            ("boundary.source_id", reference.source_id.as_str()),
        ] {
            validate_identifier(field, value)?;
        }
        if let Some(group) = &reference.coalescing_group_id {
            validate_identifier("boundary.coalescing_group_id", group)?;
        }
        if source_id != Some(reference.source_id.as_str()) {
            return Err(LossValidationError::BoundarySourceMismatch);
        }
        if !keys.insert((
            reference.transition_id.as_str(),
            reference.source_id.as_str(),
            reference.boundary_id.as_str(),
        )) {
            return Err(LossValidationError::DuplicateBoundaryReference);
        }
    }
    Ok(())
}

fn validate_identifier(field: &'static str, value: &str) -> Result<(), LossValidationError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(LossValidationError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_optional_u64_range(
    field: &'static str,
    first: Option<u64>,
    last: Option<u64>,
) -> Result<(), LossValidationError> {
    match (first, last) {
        (None, None) => Ok(()),
        (Some(first), Some(last)) if first <= last => Ok(()),
        (Some(first), Some(last)) => Err(LossValidationError::ReversedRange {
            field,
            first: i128::from(first),
            last: i128::from(last),
        }),
        _ => Err(LossValidationError::IncompleteRange(field)),
    }
}

fn validate_optional_i64_range(
    field: &'static str,
    first: Option<i64>,
    last: Option<i64>,
) -> Result<(), LossValidationError> {
    match (first, last) {
        (None, None) => Ok(()),
        (Some(first), Some(last)) if first <= last => Ok(()),
        (Some(first), Some(last)) => Err(LossValidationError::ReversedRange {
            field,
            first: i128::from(first),
            last: i128::from(last),
        }),
        _ => Err(LossValidationError::IncompleteRange(field)),
    }
}

fn required_width(
    field: &'static str,
    first: Option<u64>,
    last: Option<u64>,
) -> Result<u64, LossValidationError> {
    let first = first.ok_or(LossValidationError::MissingRange(field))?;
    let last = last.ok_or(LossValidationError::MissingRange(field))?;
    last.checked_sub(first)
        .and_then(|width| width.checked_add(1))
        .ok_or(LossValidationError::RangeWidthOverflow(field))
}

fn require_absent<T>(field: &'static str, value: Option<T>) -> Result<(), LossValidationError> {
    if value.is_some() {
        Err(LossValidationError::ForbiddenRange(field))
    } else {
        Ok(())
    }
}

fn require_present<T>(field: &'static str, value: Option<T>) -> Result<(), LossValidationError> {
    if value.is_some() {
        Ok(())
    } else {
        Err(LossValidationError::MissingRange(field))
    }
}

fn optional_source_bytes(source_id: Option<&str>) -> Vec<u8> {
    match source_id {
        None => vec![0],
        Some(source_id) => {
            let mut bytes = Vec::with_capacity(source_id.len() + 1);
            bytes.push(1);
            bytes.extend_from_slice(source_id.as_bytes());
            bytes
        }
    }
}

/// Rejected exact-loss or saturation DTO.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LossValidationError {
    /// Loss count must be positive.
    ZeroCount,
    /// Frozen kind/reason pairing disagreed.
    ReasonMismatch {
        /// Loss kind.
        kind: LossKindV1,
        /// Required reason.
        expected: LossReasonV1,
        /// Supplied reason.
        actual: LossReasonV1,
    },
    /// Identifier was empty, padded, or contained controls.
    InvalidIdentifier {
        /// Invalid field.
        field: &'static str,
        /// Redaction-safe value.
        value: String,
    },
    /// Only one endpoint of a nullable range was present.
    IncompleteRange(&'static str),
    /// Inclusive range endpoints were reversed.
    ReversedRange {
        /// Range field.
        field: &'static str,
        /// First endpoint.
        first: i128,
        /// Last endpoint.
        last: i128,
    },
    /// A role-required range was absent.
    MissingRange(&'static str),
    /// A role-forbidden range was present.
    ForbiddenRange(&'static str),
    /// Inclusive width overflowed `u64`.
    RangeWidthOverflow(&'static str),
    /// Row count disagreed with its authoritative inclusive range.
    CountEquation {
        /// Authoritative range.
        field: &'static str,
        /// Inclusive width.
        expected: u64,
        /// Supplied row count.
        actual: u64,
    },
    /// Missed cadence lacked its physical source.
    MissedCadenceRequiresSource,
    /// Source-backed issued loss used the global sentinel.
    IssuedLossRequiresSource(LossKindV1),
    /// Boundary overflow count/digest presence disagreed.
    BoundaryOverflowShape,
    /// Global loss attempted to carry a source-scoped boundary.
    GlobalBoundaryReference,
    /// Boundary source did not match the loss source.
    BoundarySourceMismatch,
    /// Exact boundary join key occurred more than once.
    DuplicateBoundaryReference,
    /// Cumulative saturation counts were empty or inconsistent.
    InvalidCumulativeCounts {
        /// Omitted range count.
        ranges: u64,
        /// Omitted entry count.
        entries: u64,
    },
    /// Supplied saturation slot did not match its bounded tuple.
    SaturationSlotIdentityMismatch,
}

impl Display for LossValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroCount => formatter.write_str("exact telemetry loss count must be positive"),
            Self::ReasonMismatch {
                kind,
                expected,
                actual,
            } => write!(
                formatter,
                "telemetry loss {kind:?} has reason {actual:?}; expected {expected:?}"
            ),
            Self::InvalidIdentifier { field, value } => {
                write!(formatter, "{field} has invalid identifier {value:?}")
            }
            Self::IncompleteRange(field) => {
                write!(
                    formatter,
                    "telemetry loss range {field} has only one endpoint"
                )
            }
            Self::ReversedRange { field, first, last } => write!(
                formatter,
                "telemetry loss range {field} is reversed ({first}..={last})"
            ),
            Self::MissingRange(field) => {
                write!(formatter, "telemetry loss requires {field} range")
            }
            Self::ForbiddenRange(field) => {
                write!(formatter, "telemetry loss forbids {field} range")
            }
            Self::RangeWidthOverflow(field) => {
                write!(
                    formatter,
                    "telemetry loss {field} inclusive width overflowed"
                )
            }
            Self::CountEquation {
                field,
                expected,
                actual,
            } => write!(
                formatter,
                "telemetry loss count {actual} disagrees with {field} width {expected}"
            ),
            Self::MissedCadenceRequiresSource => {
                formatter.write_str("missed cadence loss requires a physical source")
            }
            Self::IssuedLossRequiresSource(kind) => {
                write!(
                    formatter,
                    "issued telemetry loss {kind:?} requires a source"
                )
            }
            Self::BoundaryOverflowShape => formatter.write_str(
                "boundary overflow count and digest must be absent together or present together",
            ),
            Self::GlobalBoundaryReference => {
                formatter.write_str("global telemetry loss cannot carry boundary references")
            }
            Self::BoundarySourceMismatch => {
                formatter.write_str("telemetry loss boundary source does not match loss source")
            }
            Self::DuplicateBoundaryReference => {
                formatter.write_str("telemetry loss repeats a boundary reference")
            }
            Self::InvalidCumulativeCounts { ranges, entries } => write!(
                formatter,
                "loss saturation counts are invalid: {ranges} ranges, {entries} entries"
            ),
            Self::SaturationSlotIdentityMismatch => {
                formatter.write_str("loss saturation slot identity does not match its tuple")
            }
        }
    }
}

impl std::error::Error for LossValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(seed: u8) -> [u8; 16] {
        let mut value = [seed; 16];
        value[15] = seed.wrapping_add(1);
        value
    }

    fn missed() -> ExactLossRangeV1 {
        ExactLossRangeV1 {
            archive_id: ArchiveId::new(id(1)).unwrap(),
            session_id: SessionId::new(id(2)).unwrap(),
            source_id: Some("server-a".to_owned()),
            record_seq: 7,
            loss_seq: 3,
            count: 2,
            loss_kind: LossKindV1::MissedCadence,
            reason: LossReasonV1::CadenceOverrun,
            first_source_record_seq: None,
            last_source_record_seq: None,
            first_request_attempt_seq: None,
            last_request_attempt_seq: None,
            first_tick: Some(10),
            last_tick: Some(11),
            first_deadline_ns: Some(1_000),
            last_deadline_ns: Some(2_000),
            loss_observed_ns: 2_100,
            boundary_refs: Vec::new(),
            boundary_overflow_count: 0,
            boundary_overflow_digest: None,
        }
    }

    #[test]
    fn missed_cadence_requires_exact_tick_equation_and_forbids_attempt_ranges() {
        let mut loss = missed();
        assert_eq!(loss.validate(), Ok(()));

        loss.count = 3;
        assert!(matches!(
            loss.validate(),
            Err(LossValidationError::CountEquation { field: "tick", .. })
        ));
        loss.count = 2;
        loss.first_source_record_seq = Some(4);
        loss.last_source_record_seq = Some(5);
        assert_eq!(
            loss.validate(),
            Err(LossValidationError::ForbiddenRange("source_record_seq"))
        );
    }

    #[test]
    fn overflow_evidence_and_saturation_slot_cannot_be_forged() {
        let mut loss = missed();
        loss.boundary_overflow_count = 1;
        assert_eq!(
            loss.validate(),
            Err(LossValidationError::BoundaryOverflowShape)
        );

        let archive_id = loss.archive_id;
        let session_id = loss.session_id;
        let expected = loss_saturation_slot_id_v1(
            archive_id,
            session_id,
            Some("server-a"),
            LossKindV1::MissedCadence,
            LossReasonV1::CadenceOverrun,
        );
        let mut saturation = LossSaturationSnapshotV1 {
            archive_id,
            session_id,
            source_id: Some("server-a".to_owned()),
            record_seq: 8,
            loss_seq: 4,
            loss_kind: LossKindV1::MissedCadence,
            reason: LossReasonV1::CadenceOverrun,
            first_source_record_seq: None,
            last_source_record_seq: None,
            first_request_attempt_seq: None,
            last_request_attempt_seq: None,
            first_tick: Some(20),
            last_tick: Some(30),
            first_deadline_ns: Some(3_000),
            last_deadline_ns: Some(4_000),
            loss_observed_ns: 4_100,
            saturation_slot_id: expected,
            saturation_snapshot_seq: 0,
            cumulative_omitted_range_count: 2,
            cumulative_omitted_entry_count: 11,
            omitted_rolling_digest: Digest::from_bytes([9; 32]),
        };
        assert_eq!(saturation.validate(), Ok(()));
        saturation.saturation_slot_id = Digest::from_bytes([8; 32]);
        assert_eq!(
            saturation.validate(),
            Err(LossValidationError::SaturationSlotIdentityMismatch)
        );
    }
}
