// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sole-owner source-frame sequencing and terminal projection.
//!
//! This pure state machine assigns the one global record sequence, derives the
//! outcome-neutral reservation and terminal frame identities, tracks exact
//! per-source body continuity, and constructs a complete validated scrape
//! frame. Durable sinks consume only these terminal frames; no worker-produced
//! preliminary identity or hash is accepted.

use std::collections::BTreeMap;
use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;

use aiperf_prometheus::{Exposition, ExpositionFormat};

use crate::{
    ArchiveId, ArchiveKeyProvider, ArchiveSanitizer, ArchiveScrapeFrameV1, ArchiveScrapeRecordV1,
    AttributeMap, DecodedAttempt, Digest, EpochAnchor, ExpositionProjectionContextV1,
    ExpositionRowsV1, FrameIdentityV1, NoopSanitizer, ParseOutcome, ProjectionReservationId,
    ReservationKind, ScrapeReasonV1, SessionId, SourceOutcome, StaticLabelEnricher,
    TelemetryEnricher, TerminalKind, domain_digest, project_exposition_v1,
};

/// LocalSet timestamps observed around one completed decode/archive handoff.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArchiveFrameTimingV1 {
    /// Clock instant after bounded strict/native decoding became terminal.
    pub parse_done_ns: i64,
    /// Clock instant immediately before owner handoff.
    pub archive_enqueue_ns: i64,
}

/// One source's genesis-persisted projection policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceProjectionPolicyV1 {
    /// Static additive source attributes.
    pub attributes: AttributeMap,
}

/// Complete terminal source frame plus its WAL reservation authority.
#[derive(Clone, Debug, PartialEq)]
pub struct SequencedArchiveFrameV1 {
    /// Outcome-neutral reservation used by the final WAL header.
    pub projection_reservation_id: ProjectionReservationId,
    /// Fully validated all-outcome attempt and exposition projections.
    pub frame: ArchiveScrapeFrameV1,
}

struct SourceProjectionState {
    expected_source_record_seq: u64,
    attribute_epoch_id: Digest,
    enricher: StaticLabelEnricher,
    previous_success: Option<(u64, Digest)>,
}

impl Debug for SourceProjectionState {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SourceProjectionState")
            .field(
                "expected_source_record_seq",
                &self.expected_source_record_seq,
            )
            .field("attribute_epoch_id", &self.attribute_epoch_id)
            .field("enricher", &self.enricher)
            .field("previous_success", &self.previous_success)
            .finish()
    }
}

/// Single mutable sequencing authority shared by local and memory sinks.
pub struct ArchiveFrameSequencerV1 {
    archive_id: ArchiveId,
    session_id: SessionId,
    epoch_anchor: Option<EpochAnchor>,
    archive_key: Arc<dyn ArchiveKeyProvider>,
    sanitizer: Arc<dyn ArchiveSanitizer>,
    sources: BTreeMap<String, SourceProjectionState>,
    next_record_seq: u64,
}

impl ArchiveFrameSequencerV1 {
    /// Freeze source projection policy before any attempt is admitted.
    pub fn new(
        archive_id: ArchiveId,
        session_id: SessionId,
        epoch_anchor: Option<EpochAnchor>,
        archive_key: Arc<dyn ArchiveKeyProvider>,
        source_policies: BTreeMap<String, SourceProjectionPolicyV1>,
    ) -> Result<Self, FrameSequencingError> {
        Self::with_sanitizer_at_record_seq(
            archive_id,
            session_id,
            epoch_anchor,
            archive_key,
            source_policies,
            Arc::new(NoopSanitizer),
            0,
        )
    }

    /// Freeze a new session whose global sequence continues an existing archive.
    ///
    /// Source-local sequences and unchanged-body continuity deliberately start
    /// fresh for the new session; only the archive-global terminal record
    /// sequence crosses session boundaries.
    pub fn at_record_seq(
        archive_id: ArchiveId,
        session_id: SessionId,
        epoch_anchor: Option<EpochAnchor>,
        archive_key: Arc<dyn ArchiveKeyProvider>,
        source_policies: BTreeMap<String, SourceProjectionPolicyV1>,
        first_record_seq: u64,
    ) -> Result<Self, FrameSequencingError> {
        Self::with_sanitizer_at_record_seq(
            archive_id,
            session_id,
            epoch_anchor,
            archive_key,
            source_policies,
            Arc::new(NoopSanitizer),
            first_record_seq,
        )
    }

    /// Freeze source policy with an explicitly prepared structured sanitizer.
    pub fn with_sanitizer(
        archive_id: ArchiveId,
        session_id: SessionId,
        epoch_anchor: Option<EpochAnchor>,
        archive_key: Arc<dyn ArchiveKeyProvider>,
        source_policies: BTreeMap<String, SourceProjectionPolicyV1>,
        sanitizer: Arc<dyn ArchiveSanitizer>,
    ) -> Result<Self, FrameSequencingError> {
        Self::with_sanitizer_at_record_seq(
            archive_id,
            session_id,
            epoch_anchor,
            archive_key,
            source_policies,
            sanitizer,
            0,
        )
    }

    /// Freeze explicit sanitizer policy and an archive-global starting sequence.
    pub fn with_sanitizer_at_record_seq(
        archive_id: ArchiveId,
        session_id: SessionId,
        epoch_anchor: Option<EpochAnchor>,
        archive_key: Arc<dyn ArchiveKeyProvider>,
        source_policies: BTreeMap<String, SourceProjectionPolicyV1>,
        sanitizer: Arc<dyn ArchiveSanitizer>,
        first_record_seq: u64,
    ) -> Result<Self, FrameSequencingError> {
        if source_policies.is_empty() {
            return Err(FrameSequencingError::NoSources);
        }
        let mut sources = BTreeMap::new();
        for (source_id, policy) in source_policies {
            validate_source_id(&source_id)?;
            let attribute_epoch_id = source_attribute_epoch(&source_id, &policy.attributes)?;
            let enricher = StaticLabelEnricher::new(policy.attributes)
                .map_err(|error| FrameSequencingError::Projection(error.to_string()))?;
            sources.insert(
                source_id,
                SourceProjectionState {
                    expected_source_record_seq: 0,
                    attribute_epoch_id,
                    enricher,
                    previous_success: None,
                },
            );
        }
        Ok(Self {
            archive_id,
            session_id,
            epoch_anchor,
            archive_key,
            sanitizer,
            sources,
            next_record_seq: first_record_seq,
        })
    }

    /// Next unassigned global terminal record sequence.
    #[must_use]
    pub const fn next_record_seq(&self) -> u64 {
        self.next_record_seq
    }

    /// Atomically project one decoded source event or leave all state unchanged.
    pub fn project_attempt(
        &mut self,
        decoded: DecodedAttempt<Exposition, ()>,
        timing: ArchiveFrameTimingV1,
    ) -> Result<SequencedArchiveFrameV1, FrameSequencingError> {
        let source_id = decoded.facts.source_id.clone();
        let source = self
            .sources
            .get(&source_id)
            .ok_or_else(|| FrameSequencingError::UnknownSource(source_id.clone()))?;
        if decoded.facts.source_record_seq != source.expected_source_record_seq {
            return Err(FrameSequencingError::SourceSequence {
                source_id,
                expected: source.expected_source_record_seq,
                actual: decoded.facts.source_record_seq,
            });
        }
        if timing.parse_done_ns > timing.archive_enqueue_ns {
            return Err(FrameSequencingError::TimestampOrder);
        }
        if decoded.facts.scheduled_ns.is_none() {
            return Err(FrameSequencingError::BoundaryContextRequired);
        }

        let record_seq = self.next_record_seq;
        let encoded_body_digest = decoded
            .exact_entity
            .as_ref()
            .map(|lease| lease.encoded_digest(self.archive_key.as_ref()))
            .transpose()
            .map_err(|error| FrameSequencingError::Projection(error.to_string()))?;
        let decoded_body_digest = decoded
            .exact_entity
            .as_ref()
            .map(|lease| lease.decoded_digest(self.archive_key.as_ref()))
            .transpose()
            .map_err(|error| FrameSequencingError::Projection(error.to_string()))?;
        let successful = matches!(
            decoded.facts.outcome,
            SourceOutcome::Success | SourceOutcome::Empty
        );
        let previous_success = source.previous_success;
        let same_body_as_source_record_seq = if successful {
            match (decoded_body_digest, previous_success) {
                (Some(current), Some((sequence, previous))) if current == previous => {
                    Some(sequence)
                }
                _ => None,
            }
        } else {
            None
        };
        let body_unchanged = same_body_as_source_record_seq.is_some();

        let batch_id = FrameIdentityV1::source_scrape_batch(
            self.archive_id,
            self.session_id,
            &decoded.facts.source_id,
            decoded.facts.source_record_seq,
            decoded.facts.outcome,
            decoded_body_digest,
        )
        .map_err(|error| FrameSequencingError::Identity(error.to_string()))?;
        let projection_reservation_id = FrameIdentityV1::projection_reservation(
            self.archive_id,
            self.session_id,
            ReservationKind::SourceScrape,
            Some(&decoded.facts.source_id),
            batch_id,
            record_seq,
        )
        .map_err(|error| FrameSequencingError::Identity(error.to_string()))?;
        let frame_id = FrameIdentityV1::terminal_frame(
            TerminalKind::SourceScrape,
            projection_reservation_id,
            record_seq,
        );
        let authoritative_clock_ns = if successful {
            decoded
                .facts
                .capture_ns
                .ok_or(FrameSequencingError::MissingCapture)?
        } else {
            timing.archive_enqueue_ns
        };
        let unix_epoch_ns = self
            .epoch_anchor
            .map(|anchor| anchor.unix_ns_at(authoritative_clock_ns))
            .transpose()
            .map_err(|error| FrameSequencingError::Projection(error.to_string()))?;
        let strict_parser_format = strict_format(&decoded.strict_parse_outcome);
        let parse_done_ns = (!matches!(decoded.strict_parse_outcome, ParseOutcome::NotAttempted))
            .then_some(timing.parse_done_ns);
        let native_compatibility_format = decoded
            .native_compatibility
            .as_ref()
            .map(|compatibility| compatibility.format);
        let native_compatibility_fallback = decoded.native_compatibility.is_some();

        let exposition = match (&decoded.strict_archive_entity, decoded.facts.outcome) {
            (Some(exposition), SourceOutcome::Success) => {
                let enrichers: [&dyn TelemetryEnricher; 1] = [&source.enricher];
                project_exposition_v1(
                    exposition,
                    &ExpositionProjectionContextV1 {
                        archive_id: self.archive_id,
                        session_id: self.session_id,
                        source_id: &decoded.facts.source_id,
                        frame_id,
                        batch_id,
                        record_seq,
                        clock_ns: authoritative_clock_ns,
                        unix_epoch_ns,
                        attribute_epoch_id: source.attribute_epoch_id,
                        archive_key: self.archive_key.as_ref(),
                        enrichers: &enrichers,
                        sanitizer: self.sanitizer.as_ref(),
                    },
                )
                .map_err(|error| FrameSequencingError::Projection(error.to_string()))?
            }
            (Some(exposition), SourceOutcome::Empty) if exposition.families.is_empty() => {
                ExpositionRowsV1 {
                    families: Vec::new(),
                    samples: Vec::new(),
                }
            }
            (None, outcome)
                if !matches!(outcome, SourceOutcome::Success | SourceOutcome::Empty) =>
            {
                ExpositionRowsV1 {
                    families: Vec::new(),
                    samples: Vec::new(),
                }
            }
            _ => return Err(FrameSequencingError::DecodeOutcomeMismatch),
        };
        let family_count = u64::try_from(exposition.families.len())
            .map_err(|_| FrameSequencingError::CountOverflow)?;
        let metric_point_count = u64::try_from(exposition.samples.len())
            .map_err(|_| FrameSequencingError::CountOverflow)?;
        let wire_sample_count = exposition.samples.iter().try_fold(0_u64, |total, point| {
            total
                .checked_add(
                    u64::try_from(point.wire_samples.len())
                        .map_err(|_| FrameSequencingError::CountOverflow)?,
                )
                .ok_or(FrameSequencingError::CountOverflow)
        })?;

        let attempt = ArchiveScrapeRecordV1 {
            archive_id: self.archive_id,
            session_id: self.session_id,
            source_id: decoded.facts.source_id.clone(),
            record_seq,
            source_record_seq: decoded.facts.source_record_seq,
            request_attempt_seq: decoded.facts.request_attempt_seq,
            frame_id,
            batch_id,
            reason: ScrapeReasonV1::Continuous,
            outcome: decoded.facts.outcome,
            boundary_refs: Vec::new(),
            declared_media_type: decoded.facts.declared_media_type,
            strict_parser_format,
            native_compatibility_format,
            native_compatibility_fallback,
            scheduled_ns: decoded.facts.scheduled_ns,
            request_start_ns: decoded.facts.request_start_ns,
            first_byte_ns: decoded.facts.first_byte_ns,
            capture_ns: decoded.facts.capture_ns,
            parse_done_ns,
            archive_enqueue_ns: Some(timing.archive_enqueue_ns),
            outcome_observed_ns: timing.archive_enqueue_ns,
            unix_epoch_ns,
            http_status: decoded.facts.http_status,
            latency_ns: decoded.facts.latency_ns,
            decoded_body_digest,
            encoded_body_digest,
            raw_object_id: None,
            body_unchanged,
            same_body_as_source_record_seq,
            family_count,
            metric_point_count,
            wire_sample_count,
            error_kind: decoded.facts.error_kind,
            error_message: decoded.facts.error_message,
        };
        let frame = ArchiveScrapeFrameV1::new(attempt, exposition)
            .map_err(|error| FrameSequencingError::Projection(error.to_string()))?;

        let source = self
            .sources
            .get_mut(&frame.attempt.source_id)
            .expect("source was resolved before projection");
        source.expected_source_record_seq = source
            .expected_source_record_seq
            .checked_add(1)
            .ok_or(FrameSequencingError::CountOverflow)?;
        if successful {
            let digest = decoded_body_digest.ok_or(FrameSequencingError::MissingDecodedDigest)?;
            source.previous_success = Some((frame.attempt.source_record_seq, digest));
        }
        self.next_record_seq = self
            .next_record_seq
            .checked_add(1)
            .ok_or(FrameSequencingError::CountOverflow)?;
        Ok(SequencedArchiveFrameV1 {
            projection_reservation_id,
            frame,
        })
    }
}

impl Debug for ArchiveFrameSequencerV1 {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ArchiveFrameSequencerV1")
            .field("archive_id", &self.archive_id)
            .field("session_id", &self.session_id)
            .field("epoch_anchor", &self.epoch_anchor)
            .field("archive_key", &self.archive_key.provider_id())
            .field("sanitizer", &self.sanitizer)
            .field("sources", &self.sources)
            .field("next_record_seq", &self.next_record_seq)
            .finish()
    }
}

fn strict_format(outcome: &ParseOutcome) -> Option<ExpositionFormat> {
    match outcome {
        ParseOutcome::Success { format } | ParseOutcome::Failed { format, .. } => Some(*format),
        ParseOutcome::NotAttempted => None,
    }
}

fn source_attribute_epoch(
    source_id: &str,
    attributes: &AttributeMap,
) -> Result<Digest, FrameSequencingError> {
    let mut fields = Vec::with_capacity(attributes.len() * 2 + 1);
    fields.push(source_id.as_bytes());
    for (key, value) in attributes {
        fields.push(key.as_bytes());
        fields.push(value.as_bytes());
    }
    Ok(domain_digest("aiperf.archive.attribute-epoch.v1", &fields))
}

fn validate_source_id(value: &str) -> Result<(), FrameSequencingError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(FrameSequencingError::InvalidSourceId(value.to_owned()));
    }
    Ok(())
}

/// Terminal source-frame sequencing failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrameSequencingError {
    /// No source policy was frozen.
    NoSources,
    /// Source ID is empty, padded, or contains controls.
    InvalidSourceId(String),
    /// Attempt named a source outside genesis.
    UnknownSource(String),
    /// Per-source all-outcome order was not contiguous.
    SourceSequence {
        /// Physical source.
        source_id: String,
        /// Next expected sequence.
        expected: u64,
        /// Returned sequence.
        actual: u64,
    },
    /// Parse completion occurred after enqueue observation.
    TimestampOrder,
    /// Boundary attempts require an explicit boundary-plan projection path.
    BoundaryContextRequired,
    /// Successful decode omitted its authoritative capture instant.
    MissingCapture,
    /// Successful body continuity omitted exact decoded digest evidence.
    MissingDecodedDigest,
    /// Strict entity presence disagreed with terminal decode outcome.
    DecodeOutcomeMismatch,
    /// Count/sequence overflowed UInt64.
    CountOverflow,
    /// Closed identity preimage validation failed.
    Identity(String),
    /// Enrichment/sanitization/frame validation failed atomically.
    Projection(String),
}

impl Display for FrameSequencingError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoSources => {
                formatter.write_str("archive frame sequencer requires at least one source")
            }
            Self::InvalidSourceId(source) => {
                write!(formatter, "invalid archive source ID {source:?}")
            }
            Self::UnknownSource(source) => {
                write!(formatter, "attempt named unknown archive source {source:?}")
            }
            Self::SourceSequence {
                source_id,
                expected,
                actual,
            } => write!(
                formatter,
                "source {source_id:?} returned sequence {actual}, expected {expected}"
            ),
            Self::TimestampOrder => {
                formatter.write_str("archive enqueue preceded parse completion")
            }
            Self::BoundaryContextRequired => {
                formatter.write_str("boundary attempt omitted its sealed boundary context")
            }
            Self::MissingCapture => {
                formatter.write_str("successful archive attempt omitted capture time")
            }
            Self::MissingDecodedDigest => {
                formatter.write_str("successful archive attempt omitted decoded-body digest")
            }
            Self::DecodeOutcomeMismatch => {
                formatter.write_str("strict entity presence disagreed with source outcome")
            }
            Self::CountOverflow => {
                formatter.write_str("archive frame sequence or row count overflowed")
            }
            Self::Identity(message) => {
                write!(formatter, "archive frame identity failed: {message}")
            }
            Self::Projection(message) => {
                write!(formatter, "archive frame projection failed: {message}")
            }
        }
    }
}

impl std::error::Error for FrameSequencingError {}

#[cfg(test)]
mod tests {
    use aiperf_prometheus::{ExpositionParser, ParseLimits, StrictExpositionParser};
    use bytes::Bytes;

    use super::*;
    use crate::{
        AttemptDecoder, Blake3ArchiveKeyProvider, DecodeLimits, FetchDisposition, FetchedAttempt,
        NoopNativeEntityDecoder, PrometheusAttemptDecoder,
    };

    fn ids() -> (ArchiveId, SessionId) {
        (
            ArchiveId::new([1; 16]).unwrap(),
            SessionId::new([2; 16]).unwrap(),
        )
    }

    fn sequencer() -> ArchiveFrameSequencerV1 {
        let (archive, session) = ids();
        ArchiveFrameSequencerV1::new(
            archive,
            session,
            Some(EpochAnchor {
                clock_ns: 0,
                unix_epoch_ns: 1_000,
                capture_uncertainty_ns: 0,
            }),
            Arc::new(Blake3ArchiveKeyProvider::new("fixture_key", [7; 32]).unwrap()),
            BTreeMap::from([(
                "source-a".to_owned(),
                SourceProjectionPolicyV1 {
                    attributes: BTreeMap::from([("cluster".to_owned(), "a".to_owned())]),
                },
            )]),
        )
        .unwrap()
    }

    fn decode(sequence: u64, body: &'static [u8]) -> DecodedAttempt<Exposition, ()> {
        let fetched = FetchedAttempt {
            source_id: "source-a".to_owned(),
            source_record_seq: sequence,
            request_attempt_seq: Some(sequence),
            scheduled_ns: Some(i64::try_from(sequence).unwrap() * 10),
            request_start_ns: Some(i64::try_from(sequence).unwrap() * 10),
            first_byte_ns: Some(i64::try_from(sequence).unwrap() * 10 + 1),
            capture_ns: Some(i64::try_from(sequence).unwrap() * 10 + 1),
            latency_ns: Some(1),
            disposition: FetchDisposition::Response {
                status: 200,
                content_type: Some("text/plain; version=0.0.4".to_owned()),
                content_encoding: None,
                encoded_body: Bytes::from_static(body),
                decoded_body: Bytes::from_static(body),
            },
        };
        PrometheusAttemptDecoder::new(
            Arc::new(StrictExpositionParser),
            Arc::new(NoopNativeEntityDecoder),
        )
        .decode(fetched, &DecodeLimits::default())
    }

    #[test]
    fn terminal_identity_rows_and_body_continuity_commit_atomically() {
        let mut sequencer = sequencer();
        let body = b"# TYPE precise gauge\nprecise 16777217\n";
        let first = sequencer
            .project_attempt(
                decode(0, body),
                ArchiveFrameTimingV1 {
                    parse_done_ns: 2,
                    archive_enqueue_ns: 3,
                },
            )
            .unwrap();
        assert_eq!(first.frame.attempt.record_seq, 0);
        assert!(!first.frame.attempt.body_unchanged);
        assert_eq!(first.frame.exposition.samples.len(), 1);
        assert_eq!(first.frame.exposition.samples[0].attributes["cluster"], "a");
        assert_eq!(first.frame.exposition.samples[0].unix_epoch_ns, Some(1_001));

        let second = sequencer
            .project_attempt(
                decode(1, body),
                ArchiveFrameTimingV1 {
                    parse_done_ns: 12,
                    archive_enqueue_ns: 13,
                },
            )
            .unwrap();
        assert_eq!(second.frame.attempt.record_seq, 1);
        assert!(second.frame.attempt.body_unchanged);
        assert_eq!(second.frame.attempt.same_body_as_source_record_seq, Some(0));
        assert_ne!(first.frame.attempt.frame_id, second.frame.attempt.frame_id);
        assert_eq!(sequencer.next_record_seq(), 2);
    }

    #[test]
    fn resumed_session_continues_only_the_archive_global_sequence() {
        let (archive, session) = ids();
        let mut sequencer = ArchiveFrameSequencerV1::at_record_seq(
            archive,
            session,
            Some(EpochAnchor {
                clock_ns: 0,
                unix_epoch_ns: 1_000,
                capture_uncertainty_ns: 0,
            }),
            Arc::new(Blake3ArchiveKeyProvider::new("fixture_key", [7; 32]).unwrap()),
            BTreeMap::from([(
                "source-a".to_owned(),
                SourceProjectionPolicyV1 {
                    attributes: BTreeMap::new(),
                },
            )]),
            41,
        )
        .unwrap();

        let resumed = sequencer
            .project_attempt(
                decode(0, b"metric 1\n"),
                ArchiveFrameTimingV1 {
                    parse_done_ns: 2,
                    archive_enqueue_ns: 3,
                },
            )
            .unwrap();
        assert_eq!(resumed.frame.attempt.record_seq, 41);
        assert_eq!(resumed.frame.attempt.source_record_seq, 0);
        assert!(!resumed.frame.attempt.body_unchanged);
        assert_eq!(sequencer.next_record_seq(), 42);
    }

    #[test]
    fn failed_projection_does_not_consume_sequence_or_source_epoch() {
        let mut sequencer = sequencer();
        let error = sequencer
            .project_attempt(
                decode(1, b"metric 1\n"),
                ArchiveFrameTimingV1 {
                    parse_done_ns: 2,
                    archive_enqueue_ns: 3,
                },
            )
            .unwrap_err();
        assert!(matches!(error, FrameSequencingError::SourceSequence { .. }));
        assert_eq!(sequencer.next_record_seq(), 0);
        sequencer
            .project_attempt(
                decode(0, b"metric 1\n"),
                ArchiveFrameTimingV1 {
                    parse_done_ns: 2,
                    archive_enqueue_ns: 3,
                },
            )
            .unwrap();
    }

    #[test]
    fn parser_fixture_is_lossless_before_owner_projection() {
        let exposition = StrictExpositionParser
            .parse(
                ExpositionFormat::PrometheusText004,
                b"quoted{label=\"a,b\\\"c\"} 1\n",
                &ParseLimits::default(),
            )
            .unwrap();
        assert_eq!(exposition.families[0].metrics[0].labels["label"], "a,b\"c");
    }
}
