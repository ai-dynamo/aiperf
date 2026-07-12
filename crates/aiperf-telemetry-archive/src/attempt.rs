// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! All-outcome scrape attempt rows and whole-frame projection validation.

use std::fmt::{self, Display, Formatter};

use aiperf_prometheus::ExpositionFormat;

use crate::{
    ArchiveId, BatchId, BoundaryReference, Digest, ExpositionRowsV1, FrameId, SessionId,
    SourceOutcome,
};

/// Why one physical telemetry source event was issued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScrapeReasonV1 {
    /// Anchor-relative cadence deadline.
    Continuous,
    /// Forced transition capture carrying structured boundary references.
    Boundary,
}

/// One queryable row for every issued source event, including failures.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchiveScrapeRecordV1 {
    /// Archive UUID.
    pub archive_id: ArchiveId,
    /// Collection session UUID.
    pub session_id: SessionId,
    /// Physical source identity.
    pub source_id: String,
    /// Owner-assigned global frame sequence.
    pub record_seq: u64,
    /// Per-source sequence assigned for every issued source event.
    pub source_record_seq: u64,
    /// Per-source network-attempt sequence, absent when no IO began.
    pub request_attempt_seq: Option<u64>,
    /// Terminal frame identity.
    pub frame_id: FrameId,
    /// Stable source-event batch identity.
    pub batch_id: BatchId,
    /// Continuous or forced-boundary issuance reason.
    pub reason: ScrapeReasonV1,
    /// Terminal source outcome.
    pub outcome: SourceOutcome,
    /// Structured forced-boundary joins satisfied by this attempt.
    pub boundary_refs: Vec<BoundaryReference>,
    /// Normalized declared response media type.
    pub declared_media_type: Option<String>,
    /// Strict archive grammar selected from `Content-Type`.
    pub strict_parser_format: Option<ExpositionFormat>,
    /// Separately named grammar used only for native compatibility.
    pub native_compatibility_format: Option<ExpositionFormat>,
    /// Whether native compatibility intentionally used a second grammar.
    pub native_compatibility_fallback: bool,
    /// Anchor-relative cadence target, absent for a forced-only command.
    pub scheduled_ns: Option<i64>,
    /// Clock instant immediately before control-plane IO.
    pub request_start_ns: Option<i64>,
    /// Clock instant of first response body byte.
    pub first_byte_ns: Option<i64>,
    /// Authoritative source snapshot/capture instant.
    pub capture_ns: Option<i64>,
    /// Clock instant when bounded decode became terminal.
    pub parse_done_ns: Option<i64>,
    /// Clock instant of archive-admission observation.
    pub archive_enqueue_ns: Option<i64>,
    /// Clock instant when the all-outcome classification became immutable.
    pub outcome_observed_ns: i64,
    /// Approximate Unix placement derived from the session anchor.
    pub unix_epoch_ns: Option<i128>,
    /// HTTP status when a response was received.
    pub http_status: Option<u16>,
    /// Non-negative request latency when measurable.
    pub latency_ns: Option<i64>,
    /// Keyed digest of exact decoded exposition bytes.
    pub decoded_body_digest: Option<Digest>,
    /// Optional keyed digest of exact encoded entity bytes.
    pub encoded_body_digest: Option<Digest>,
    /// Optional exact raw-envelope object identity.
    pub raw_object_id: Option<Digest>,
    /// Whether decoded bytes equal the preceding successful/empty body.
    pub body_unchanged: bool,
    /// Prior source event owning those identical decoded bytes.
    pub same_body_as_source_record_seq: Option<u64>,
    /// Number of projected family metadata rows.
    pub family_count: u64,
    /// Number of projected structured MetricPoint rows.
    pub metric_point_count: u64,
    /// Number of exact retained wire samples across all points.
    pub wire_sample_count: u64,
    /// Stable typed error category for a non-success outcome.
    pub error_kind: Option<String>,
    /// Bounded redaction-safe diagnostic for a non-success outcome.
    pub error_message: Option<String>,
}

impl ArchiveScrapeRecordV1 {
    /// Validates the closed outcome/field matrix before WAL encoding.
    pub fn validate(&self) -> Result<(), AttemptValidationError> {
        validate_source_id(&self.source_id)?;
        validate_boundary_membership(self)?;
        validate_timestamp_order(self)?;

        if self.body_unchanged {
            if !matches!(self.outcome, SourceOutcome::Success | SourceOutcome::Empty) {
                return Err(AttemptValidationError::UnchangedOnFailedOutcome);
            }
            let previous = self
                .same_body_as_source_record_seq
                .ok_or(AttemptValidationError::MissingUnchangedPredecessor)?;
            if previous >= self.source_record_seq {
                return Err(AttemptValidationError::InvalidUnchangedPredecessor {
                    previous,
                    current: self.source_record_seq,
                });
            }
            if self.decoded_body_digest.is_none() {
                return Err(AttemptValidationError::MissingDecodedDigestForUnchanged);
            }
        } else if self.same_body_as_source_record_seq.is_some() {
            return Err(AttemptValidationError::UnexpectedUnchangedPredecessor);
        }

        let successful = matches!(self.outcome, SourceOutcome::Success | SourceOutcome::Empty);
        if successful {
            if self.strict_parser_format.is_none() {
                return Err(AttemptValidationError::MissingStrictParserFormat);
            }
            if self.capture_ns.is_none() || self.parse_done_ns.is_none() {
                return Err(AttemptValidationError::MissingSuccessfulTimestamp);
            }
            if self.error_kind.is_some() || self.error_message.is_some() {
                return Err(AttemptValidationError::ErrorOnSuccessfulOutcome);
            }
            if self
                .http_status
                .is_some_and(|status| !(200..300).contains(&status))
            {
                return Err(AttemptValidationError::NonSuccessStatusOnSuccessfulOutcome);
            }
            if self.outcome == SourceOutcome::Empty
                && (self.family_count != 0
                    || self.metric_point_count != 0
                    || self.wire_sample_count != 0)
            {
                return Err(AttemptValidationError::RowsOnEmptyOutcome);
            }
        } else {
            if self.family_count != 0 || self.metric_point_count != 0 || self.wire_sample_count != 0
            {
                return Err(AttemptValidationError::RowsOnFailedOutcome);
            }
            if self.error_kind.is_none() {
                return Err(AttemptValidationError::MissingFailureKind);
            }
        }

        match self.outcome {
            SourceOutcome::Http => {
                let status = self
                    .http_status
                    .ok_or(AttemptValidationError::MissingHttpFailureStatus)?;
                if (200..300).contains(&status) {
                    return Err(AttemptValidationError::SuccessStatusOnHttpFailure(status));
                }
                if self.strict_parser_format.is_some() || self.parse_done_ns.is_some() {
                    return Err(AttemptValidationError::ParsedHttpFailureBody);
                }
            }
            SourceOutcome::Transport | SourceOutcome::Disabled | SourceOutcome::Shutdown => {
                if self.http_status.is_some() {
                    return Err(AttemptValidationError::UnexpectedHttpStatus);
                }
            }
            SourceOutcome::Timeout => {}
            SourceOutcome::Parse => {
                if self.strict_parser_format.is_none() || self.parse_done_ns.is_none() {
                    return Err(AttemptValidationError::IncompleteParseFailure);
                }
            }
            SourceOutcome::UnsupportedFormat => {
                if self.strict_parser_format.is_some() {
                    return Err(AttemptValidationError::ParserOnUnsupportedFormat);
                }
            }
            SourceOutcome::UnsupportedFeature | SourceOutcome::Success | SourceOutcome::Empty => {}
        }

        if self.native_compatibility_fallback && self.native_compatibility_format.is_none() {
            return Err(AttemptValidationError::MissingCompatibilityFormat);
        }
        if !self.native_compatibility_fallback && self.native_compatibility_format.is_some() {
            return Err(AttemptValidationError::UnexpectedCompatibilityFormat);
        }
        if self.latency_ns.is_some_and(|latency| latency < 0) {
            return Err(AttemptValidationError::NegativeLatency);
        }
        Ok(())
    }
}

/// One terminal source frame whose projections are validated as a whole.
#[derive(Clone, Debug, PartialEq)]
pub struct ArchiveScrapeFrameV1 {
    /// Required all-outcome attempt row.
    pub attempt: ArchiveScrapeRecordV1,
    /// Family/sample rows; empty for every non-success/empty attempt.
    pub exposition: ExpositionRowsV1,
}

impl ArchiveScrapeFrameV1 {
    /// Creates a frame only when every projection shares identity/cardinality.
    pub fn new(
        attempt: ArchiveScrapeRecordV1,
        exposition: ExpositionRowsV1,
    ) -> Result<Self, AttemptValidationError> {
        attempt.validate()?;
        let family_count = usize_to_u64(exposition.families.len())?;
        let metric_point_count = usize_to_u64(exposition.samples.len())?;
        let wire_sample_count = exposition.samples.iter().try_fold(0_u64, |total, point| {
            total
                .checked_add(usize_to_u64(point.wire_samples.len())?)
                .ok_or(AttemptValidationError::CountOverflow)
        })?;
        if (
            attempt.family_count,
            attempt.metric_point_count,
            attempt.wire_sample_count,
        ) != (family_count, metric_point_count, wire_sample_count)
        {
            return Err(AttemptValidationError::ProjectionCountMismatch {
                attempt: (
                    attempt.family_count,
                    attempt.metric_point_count,
                    attempt.wire_sample_count,
                ),
                actual: (family_count, metric_point_count, wire_sample_count),
            });
        }
        if !matches!(attempt.outcome, SourceOutcome::Success)
            && (!exposition.families.is_empty() || !exposition.samples.is_empty())
        {
            return Err(AttemptValidationError::RowsOnFailedOutcome);
        }
        let capture_ns = attempt.capture_ns;
        for family in &exposition.families {
            if family.archive_id != attempt.archive_id
                || family.session_id != attempt.session_id
                || family.source_id != attempt.source_id
                || family.frame_id != attempt.frame_id
                || family.batch_id != attempt.batch_id
                || family.record_seq != attempt.record_seq
            {
                return Err(AttemptValidationError::ProjectionIdentityMismatch {
                    table: "families",
                });
            }
        }
        for sample in &exposition.samples {
            if sample.archive_id != attempt.archive_id
                || sample.session_id != attempt.session_id
                || sample.source_id != attempt.source_id
                || sample.frame_id != attempt.frame_id
                || sample.batch_id != attempt.batch_id
                || sample.record_seq != attempt.record_seq
            {
                return Err(AttemptValidationError::ProjectionIdentityMismatch {
                    table: "samples",
                });
            }
            if Some(sample.clock_ns) != capture_ns {
                return Err(AttemptValidationError::SampleCaptureMismatch {
                    expected: capture_ns,
                    actual: sample.clock_ns,
                });
            }
        }
        Ok(Self {
            attempt,
            exposition,
        })
    }
}

fn validate_boundary_membership(
    attempt: &ArchiveScrapeRecordV1,
) -> Result<(), AttemptValidationError> {
    match attempt.reason {
        ScrapeReasonV1::Continuous if !attempt.boundary_refs.is_empty() => {
            return Err(AttemptValidationError::BoundaryReferencesOnContinuous);
        }
        ScrapeReasonV1::Boundary if attempt.boundary_refs.is_empty() => {
            return Err(AttemptValidationError::MissingBoundaryReferences);
        }
        ScrapeReasonV1::Continuous | ScrapeReasonV1::Boundary => {}
    }
    let mut seen = std::collections::BTreeSet::new();
    for reference in &attempt.boundary_refs {
        if reference.source_id != attempt.source_id {
            return Err(AttemptValidationError::BoundarySourceMismatch {
                expected: attempt.source_id.clone(),
                actual: reference.source_id.clone(),
            });
        }
        if !seen.insert((
            reference.transition_id.as_str(),
            reference.source_id.as_str(),
            reference.boundary_id.as_str(),
        )) {
            return Err(AttemptValidationError::DuplicateBoundaryReference);
        }
    }
    Ok(())
}

fn validate_timestamp_order(attempt: &ArchiveScrapeRecordV1) -> Result<(), AttemptValidationError> {
    if let (Some(start), Some(first_byte)) = (attempt.request_start_ns, attempt.first_byte_ns)
        && first_byte < start
    {
        return Err(AttemptValidationError::TimestampOrder {
            earlier: "request_start_ns",
            later: "first_byte_ns",
        });
    }
    if let (Some(first_byte), Some(capture)) = (attempt.first_byte_ns, attempt.capture_ns)
        && capture < first_byte
    {
        return Err(AttemptValidationError::TimestampOrder {
            earlier: "first_byte_ns",
            later: "capture_ns",
        });
    }
    if let (Some(capture), Some(parsed)) = (attempt.capture_ns, attempt.parse_done_ns)
        && parsed < capture
    {
        return Err(AttemptValidationError::TimestampOrder {
            earlier: "capture_ns",
            later: "parse_done_ns",
        });
    }
    for (field, timestamp) in [
        ("scheduled_ns", attempt.scheduled_ns),
        ("request_start_ns", attempt.request_start_ns),
        ("first_byte_ns", attempt.first_byte_ns),
        ("capture_ns", attempt.capture_ns),
        ("parse_done_ns", attempt.parse_done_ns),
        ("archive_enqueue_ns", attempt.archive_enqueue_ns),
    ] {
        if timestamp.is_some_and(|timestamp| timestamp > attempt.outcome_observed_ns) {
            return Err(AttemptValidationError::AfterOutcomeObservation(field));
        }
    }
    Ok(())
}

fn validate_source_id(value: &str) -> Result<(), AttemptValidationError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(AttemptValidationError::InvalidSourceId);
    }
    Ok(())
}

fn usize_to_u64(value: usize) -> Result<u64, AttemptValidationError> {
    u64::try_from(value).map_err(|_| AttemptValidationError::CountOverflow)
}

/// Closed attempt-row or whole-frame invariant violation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AttemptValidationError {
    /// Source identity is empty, padded, or contains controls.
    InvalidSourceId,
    /// Continuous attempts cannot satisfy forced-boundary references.
    BoundaryReferencesOnContinuous,
    /// Boundary attempts require at least one structured reference.
    MissingBoundaryReferences,
    /// A boundary reference named a different physical source.
    BoundarySourceMismatch {
        /// Attempt source.
        expected: String,
        /// Reference source.
        actual: String,
    },
    /// The same exact join key appeared twice on one attempt.
    DuplicateBoundaryReference,
    /// Unchanged is valid only for successful/empty parsed bodies.
    UnchangedOnFailedOutcome,
    /// An unchanged body omitted its predecessor sequence.
    MissingUnchangedPredecessor,
    /// The predecessor was not strictly earlier than this source event.
    InvalidUnchangedPredecessor {
        /// Authored predecessor.
        previous: u64,
        /// Current source event.
        current: u64,
    },
    /// Unchanged detection requires the keyed decoded-body digest.
    MissingDecodedDigestForUnchanged,
    /// A predecessor was present while `body_unchanged=false`.
    UnexpectedUnchangedPredecessor,
    /// A successful/empty archive parse omitted its selected grammar.
    MissingStrictParserFormat,
    /// Successful parsing omitted capture or parse-completion Clock time.
    MissingSuccessfulTimestamp,
    /// A successful outcome carried failure diagnostics.
    ErrorOnSuccessfulOutcome,
    /// A successful parse carried a non-2xx response status.
    NonSuccessStatusOnSuccessfulOutcome,
    /// An explicitly empty exposition carried family/sample rows.
    RowsOnEmptyOutcome,
    /// A non-success outcome carried successful archive rows.
    RowsOnFailedOutcome,
    /// A failed outcome omitted its stable error kind.
    MissingFailureKind,
    /// HTTP failure omitted its non-2xx status.
    MissingHttpFailureStatus,
    /// HTTP failure carried a successful status.
    SuccessStatusOnHttpFailure(u16),
    /// A non-2xx body was passed to the metrics parser.
    ParsedHttpFailureBody,
    /// A transport/pre-IO outcome unexpectedly carried an HTTP status.
    UnexpectedHttpStatus,
    /// Parse failure omitted selected grammar or parse-completion time.
    IncompleteParseFailure,
    /// Unsupported media was incorrectly assigned a parser grammar.
    ParserOnUnsupportedFormat,
    /// Named native fallback omitted its actual grammar.
    MissingCompatibilityFormat,
    /// Compatibility grammar was present when fallback was false.
    UnexpectedCompatibilityFormat,
    /// Request latency cannot be negative.
    NegativeLatency,
    /// One Clock timestamp preceded its causal predecessor.
    TimestampOrder {
        /// Earlier field.
        earlier: &'static str,
        /// Later field.
        later: &'static str,
    },
    /// A timestamp occurred after the immutable outcome observation.
    AfterOutcomeObservation(&'static str),
    /// Projection count exceeded UInt64.
    CountOverflow,
    /// Attempt counts and materialized rows disagree.
    ProjectionCountMismatch {
        /// Counts retained by the attempt.
        attempt: (u64, u64, u64),
        /// Counts computed from the frame projections.
        actual: (u64, u64, u64),
    },
    /// A family/sample projection belongs to another frame.
    ProjectionIdentityMismatch {
        /// Affected table.
        table: &'static str,
    },
    /// MetricPoint capture time differs from the attempt snapshot.
    SampleCaptureMismatch {
        /// Attempt capture time.
        expected: Option<i64>,
        /// Sample Clock time.
        actual: i64,
    },
}

impl Display for AttemptValidationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceId => formatter.write_str("invalid telemetry source ID"),
            Self::BoundaryReferencesOnContinuous => {
                formatter.write_str("continuous scrape carried boundary references")
            }
            Self::MissingBoundaryReferences => {
                formatter.write_str("boundary scrape has no boundary references")
            }
            Self::BoundarySourceMismatch { expected, actual } => write!(
                formatter,
                "boundary source {actual:?} does not match attempt source {expected:?}"
            ),
            Self::DuplicateBoundaryReference => {
                formatter.write_str("duplicate exact boundary reference")
            }
            Self::UnchangedOnFailedOutcome => {
                formatter.write_str("failed attempt cannot be body-unchanged")
            }
            Self::MissingUnchangedPredecessor => {
                formatter.write_str("unchanged body omitted predecessor sequence")
            }
            Self::InvalidUnchangedPredecessor { previous, current } => write!(
                formatter,
                "unchanged predecessor {previous} is not earlier than source sequence {current}"
            ),
            Self::MissingDecodedDigestForUnchanged => {
                formatter.write_str("unchanged body omitted decoded digest")
            }
            Self::UnexpectedUnchangedPredecessor => {
                formatter.write_str("unchanged predecessor present while unchanged=false")
            }
            Self::MissingStrictParserFormat => {
                formatter.write_str("successful archive parse omitted strict format")
            }
            Self::MissingSuccessfulTimestamp => {
                formatter.write_str("successful archive parse omitted capture/parse timestamp")
            }
            Self::ErrorOnSuccessfulOutcome => {
                formatter.write_str("successful scrape carried failure diagnostics")
            }
            Self::NonSuccessStatusOnSuccessfulOutcome => {
                formatter.write_str("successful scrape carried non-2xx HTTP status")
            }
            Self::RowsOnEmptyOutcome => {
                formatter.write_str("empty exposition carried projected rows")
            }
            Self::RowsOnFailedOutcome => {
                formatter.write_str("failed scrape carried projected rows")
            }
            Self::MissingFailureKind => formatter.write_str("failed scrape omitted error kind"),
            Self::MissingHttpFailureStatus => {
                formatter.write_str("HTTP failure omitted response status")
            }
            Self::SuccessStatusOnHttpFailure(status) => {
                write!(formatter, "HTTP failure carried successful status {status}")
            }
            Self::ParsedHttpFailureBody => {
                formatter.write_str("non-2xx HTTP body was passed to metrics parsing")
            }
            Self::UnexpectedHttpStatus => {
                formatter.write_str("non-HTTP outcome carried response status")
            }
            Self::IncompleteParseFailure => {
                formatter.write_str("parse failure omitted grammar or completion time")
            }
            Self::ParserOnUnsupportedFormat => {
                formatter.write_str("unsupported format selected a parser grammar")
            }
            Self::MissingCompatibilityFormat => {
                formatter.write_str("native compatibility fallback omitted grammar")
            }
            Self::UnexpectedCompatibilityFormat => {
                formatter.write_str("native compatibility grammar present without fallback")
            }
            Self::NegativeLatency => formatter.write_str("negative scrape latency"),
            Self::TimestampOrder { earlier, later } => {
                write!(formatter, "{later} precedes {earlier}")
            }
            Self::AfterOutcomeObservation(field) => {
                write!(formatter, "{field} occurs after outcome_observed_ns")
            }
            Self::CountOverflow => formatter.write_str("attempt projection count overflowed"),
            Self::ProjectionCountMismatch { attempt, actual } => write!(
                formatter,
                "attempt projection counts {attempt:?} do not match rows {actual:?}"
            ),
            Self::ProjectionIdentityMismatch { table } => {
                write!(
                    formatter,
                    "{table} projection identity does not match attempt"
                )
            }
            Self::SampleCaptureMismatch { expected, actual } => write!(
                formatter,
                "sample Clock {actual} does not match attempt capture {expected:?}"
            ),
        }
    }
}

impl std::error::Error for AttemptValidationError {}

#[cfg(test)]
mod tests {
    use aiperf_prometheus::{
        ExpositionFormat, ExpositionParser, ParseLimits, StrictExpositionParser,
    };

    use super::*;
    use crate::{
        Blake3ArchiveKeyProvider, ExpositionProjectionContextV1, FrameIdentityV1, NoopSanitizer,
        ReservationKind, TerminalKind, project_exposition_v1,
    };

    fn identities() -> (ArchiveId, SessionId, BatchId, FrameId) {
        let archive = ArchiveId::new([1; 16]).unwrap();
        let session = SessionId::new([2; 16]).unwrap();
        let batch = FrameIdentityV1::source_scrape_batch(
            archive,
            session,
            "source-a",
            1,
            SourceOutcome::Success,
            None,
        )
        .unwrap();
        let reservation = FrameIdentityV1::projection_reservation(
            archive,
            session,
            ReservationKind::SourceScrape,
            Some("source-a"),
            batch,
            7,
        )
        .unwrap();
        let frame = FrameIdentityV1::terminal_frame(TerminalKind::SourceScrape, reservation, 7);
        (archive, session, batch, frame)
    }

    fn successful_attempt(counts: (u64, u64, u64)) -> ArchiveScrapeRecordV1 {
        let (archive_id, session_id, batch_id, frame_id) = identities();
        ArchiveScrapeRecordV1 {
            archive_id,
            session_id,
            source_id: "source-a".to_owned(),
            record_seq: 7,
            source_record_seq: 1,
            request_attempt_seq: Some(1),
            frame_id,
            batch_id,
            reason: ScrapeReasonV1::Continuous,
            outcome: SourceOutcome::Success,
            boundary_refs: Vec::new(),
            declared_media_type: Some("text/plain;version=0.0.4".to_owned()),
            strict_parser_format: Some(ExpositionFormat::PrometheusText004),
            native_compatibility_format: None,
            native_compatibility_fallback: false,
            scheduled_ns: Some(90),
            request_start_ns: Some(100),
            first_byte_ns: Some(105),
            capture_ns: Some(110),
            parse_done_ns: Some(115),
            archive_enqueue_ns: Some(118),
            outcome_observed_ns: 120,
            unix_epoch_ns: Some(1_000),
            http_status: Some(200),
            latency_ns: Some(10),
            decoded_body_digest: Some(crate::domain_digest(
                "aiperf.archive.body-decoded.v1",
                &[b"body"],
            )),
            encoded_body_digest: None,
            raw_object_id: None,
            body_unchanged: false,
            same_body_as_source_record_seq: None,
            family_count: counts.0,
            metric_point_count: counts.1,
            wire_sample_count: counts.2,
            error_kind: None,
            error_message: None,
        }
    }

    #[test]
    fn non_2xx_metric_looking_body_cannot_become_archive_rows() {
        let mut attempt = successful_attempt((0, 0, 0));
        attempt.outcome = SourceOutcome::Http;
        attempt.http_status = Some(500);
        attempt.strict_parser_format = Some(ExpositionFormat::PrometheusText004);
        attempt.parse_done_ns = Some(115);
        attempt.error_kind = Some("http_status".to_owned());
        assert_eq!(
            attempt.validate(),
            Err(AttemptValidationError::ParsedHttpFailureBody)
        );

        attempt.strict_parser_format = None;
        attempt.parse_done_ns = None;
        attempt.capture_ns = None;
        assert_eq!(attempt.validate(), Ok(()));
    }

    #[test]
    fn unchanged_is_orthogonal_success_and_requires_an_earlier_sequence() {
        let mut attempt = successful_attempt((1, 1, 1));
        attempt.body_unchanged = true;
        attempt.same_body_as_source_record_seq = Some(0);
        assert_eq!(attempt.validate(), Ok(()));
        attempt.same_body_as_source_record_seq = Some(1);
        assert!(matches!(
            attempt.validate(),
            Err(AttemptValidationError::InvalidUnchangedPredecessor { .. })
        ));
    }

    #[test]
    fn whole_frame_binds_counts_identity_and_capture_clock() {
        let exposition = StrictExpositionParser
            .parse(
                ExpositionFormat::PrometheusText004,
                b"# TYPE value gauge\nvalue{label=\"a\"} 1\n",
                &ParseLimits::default(),
            )
            .unwrap();
        let key = Blake3ArchiveKeyProvider::new("fixture_key", [7; 32]).unwrap();
        let (archive_id, session_id, batch_id, frame_id) = identities();
        let rows = project_exposition_v1(
            &exposition,
            &ExpositionProjectionContextV1 {
                archive_id,
                session_id,
                source_id: "source-a",
                frame_id,
                batch_id,
                record_seq: 7,
                clock_ns: 110,
                unix_epoch_ns: Some(1_000),
                attribute_epoch_id: crate::domain_digest(
                    "aiperf.archive.test-epoch.v1",
                    &[b"epoch"],
                ),
                archive_key: &key,
                enrichers: &[],
                sanitizer: &NoopSanitizer,
            },
        )
        .unwrap();
        let counts = (
            u64::try_from(rows.families.len()).unwrap(),
            u64::try_from(rows.samples.len()).unwrap(),
            rows.samples
                .iter()
                .map(|row| u64::try_from(row.wire_samples.len()).unwrap())
                .sum(),
        );
        let attempt = successful_attempt(counts);
        ArchiveScrapeFrameV1::new(attempt.clone(), rows.clone()).unwrap();

        let mut wrong = rows;
        wrong.samples[0].clock_ns += 1;
        assert!(matches!(
            ArchiveScrapeFrameV1::new(attempt, wrong),
            Err(AttemptValidationError::SampleCaptureMismatch { .. })
        ));
    }

    #[test]
    fn failed_attempts_never_accept_success_projections() {
        let mut attempt = successful_attempt((1, 1, 1));
        attempt.outcome = SourceOutcome::Parse;
        attempt.error_kind = Some("parse".to_owned());
        assert_eq!(
            attempt.validate(),
            Err(AttemptValidationError::RowsOnFailedOutcome)
        );
    }
}
