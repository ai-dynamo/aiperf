// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless exposition-to-archive family and metric-point projection.

use std::collections::BTreeMap;
use std::fmt::{self, Debug, Display, Formatter};

use aiperf_prometheus::{
    Exposition, InfoLabelPartitionStatus, LabelSet, MetricValue, PointTimeStatus, SemanticType,
    SourceTimestamp, WireSample,
};

use crate::key::{ArchiveKeyError, ArchiveKeyProvider, ArchiveSubkey, keyed_domain_digest};
use crate::{ArchiveId, BatchId, Digest, FrameId, SessionId, domain_digest};

/// Additive stored attributes keyed in canonical UTF-8 order.
pub type AttributeMap = BTreeMap<String, String>;

/// Read-only sample identity presented to enrichment and sanitization policies.
#[derive(Clone, Copy, Debug)]
pub struct ArchiveSampleView<'a> {
    /// Stable physical source identity.
    pub source_id: &'a str,
    /// Exact source metric-family name.
    pub metric_family: &'a str,
    /// Format-resolved semantic type.
    pub semantic_type: SemanticType,
    /// Complete pre-redaction source identity labels.
    pub labels: &'a LabelSet,
    /// Additive attributes accumulated before sanitization.
    pub attributes: &'a AttributeMap,
}

/// Additive topology/static metadata policy.
pub trait TelemetryEnricher: Debug + Send + Sync {
    /// Produces attributes without changing source labels or numeric values.
    fn attributes(&self, sample: ArchiveSampleView<'_>) -> Result<AttributeMap, EnrichmentError>;
}

/// Structured content policy applied only after protected source identity.
pub trait ArchiveSanitizer: Debug + Send + Sync {
    /// Produces the stored label/attribute surfaces.
    fn sanitize_sample(
        &self,
        sample: ArchiveSampleView<'_>,
    ) -> Result<SanitizedSample, SanitizationError>;
}

/// Stored identity and attributes returned by one sanitizer chain.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SanitizedSample {
    /// Stored post-redaction source labels.
    pub labels: LabelSet,
    /// Stored post-redaction additive attributes.
    pub attributes: AttributeMap,
}

/// Enricher that deliberately adds no attributes.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopEnricher;

impl TelemetryEnricher for NoopEnricher {
    fn attributes(&self, _sample: ArchiveSampleView<'_>) -> Result<AttributeMap, EnrichmentError> {
        Ok(AttributeMap::new())
    }
}

/// Static additive attributes applied to every point in a prepared source.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaticLabelEnricher {
    attributes: AttributeMap,
}

impl StaticLabelEnricher {
    /// Validates and freezes one static attribute map.
    pub fn new(attributes: AttributeMap) -> Result<Self, EnrichmentError> {
        validate_attribute_patch(&attributes)?;
        Ok(Self { attributes })
    }
}

impl TelemetryEnricher for StaticLabelEnricher {
    fn attributes(&self, _sample: ArchiveSampleView<'_>) -> Result<AttributeMap, EnrichmentError> {
        Ok(self.attributes.clone())
    }
}

/// No additional content policy after the mandatory credential baseline.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopSanitizer;

impl ArchiveSanitizer for NoopSanitizer {
    fn sanitize_sample(
        &self,
        sample: ArchiveSampleView<'_>,
    ) -> Result<SanitizedSample, SanitizationError> {
        Ok(SanitizedSample {
            labels: sample.labels.clone(),
            attributes: sample.attributes.clone(),
        })
    }
}

/// Archive-level availability of an analytical Info label partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArchiveInfoLabelPartitionStatus {
    /// Non-Info point.
    NotApplicable,
    /// Text bytes expose only one merged label map.
    UnavailableFromText,
    /// A named genesis-persisted policy supplied a partition.
    PolicyApplied,
}

/// Exact metadata row for one parsed family, including metadata-only families.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricFamilyRowV1 {
    /// Archive UUID bytes.
    pub archive_id: ArchiveId,
    /// Collection-session UUID bytes.
    pub session_id: SessionId,
    /// Terminal frame identity.
    pub frame_id: FrameId,
    /// Stable scrape batch identity.
    pub batch_id: BatchId,
    /// Owner-assigned global frame sequence.
    pub record_seq: u64,
    /// Source-order family sequence.
    pub family_seq: u64,
    /// Exact source family name.
    pub metric_family: String,
    /// Exact source TYPE token, including unknown/untyped distinction.
    pub source_type_token: String,
    /// Resolved semantic family type.
    pub semantic_type: SemanticType,
    /// Whether HELP was present, even when its value was empty.
    pub help_present: bool,
    /// Whether TYPE was explicitly present.
    pub type_present: bool,
    /// Whether UNIT was present, even when its value was empty.
    pub unit_present: bool,
    /// Decoded HELP value when emitted.
    pub help: Option<String>,
    /// Decoded UNIT value when emitted.
    pub unit: Option<String>,
    /// One-based HELP source line.
    pub help_line_seq: Option<u64>,
    /// One-based TYPE source line.
    pub type_line_seq: Option<u64>,
    /// One-based UNIT source line.
    pub unit_line_seq: Option<u64>,
    /// Number of source metric identities in this family.
    pub metric_count: u64,
    /// Number of structured points in this family.
    pub metric_point_count: u64,
    /// Number of exact wire samples owned by those points.
    pub wire_sample_count: u64,
}

/// Exact structured metric-point row and all contributing wire samples.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricPointRowV1 {
    /// Archive UUID bytes.
    pub archive_id: ArchiveId,
    /// Collection-session UUID bytes.
    pub session_id: SessionId,
    /// Physical source identity.
    pub source_id: String,
    /// Terminal frame identity.
    pub frame_id: FrameId,
    /// Stable scrape batch identity.
    pub batch_id: BatchId,
    /// Owner-assigned global frame sequence.
    pub record_seq: u64,
    /// Source-order family sequence.
    pub family_seq: u64,
    /// Source-order point sequence.
    pub metric_point_seq: u64,
    /// Authoritative capture Clock instant.
    pub clock_ns: i64,
    /// Approximate Unix placement derived from the session anchor.
    pub unix_epoch_ns: Option<i128>,
    /// Exact source family name.
    pub metric_family: String,
    /// Exact source TYPE token.
    pub source_type_token: String,
    /// Resolved semantic type.
    pub semantic_type: SemanticType,
    /// Protected keyed pre-redaction source identity.
    pub source_series_key: Digest,
    /// Ordinary stored post-redaction identity bound to the protected key.
    pub series_key: Digest,
    /// Stored post-redaction labels.
    pub labels: LabelSet,
    /// Stored additive attributes.
    pub attributes: AttributeMap,
    /// Complete text-native merged Info labels when this is an Info point.
    pub wire_merged_info_labels: Option<LabelSet>,
    /// Whether an abstract Info partition exists.
    pub info_label_partition_status: ArchiveInfoLabelPartitionStatus,
    /// Named partition policy when one was applied.
    pub info_label_partition_policy_id: Option<String>,
    /// Source-local attribute-epoch identity.
    pub attribute_epoch_id: Digest,
    /// Relationship among exact component timestamps.
    pub point_time_status: PointTimeStatus,
    /// Common explicit point timestamp only when uniformly present/equal.
    pub source_timestamp: SourceTimestamp,
    /// Structured semantic payload.
    pub payload: MetricValue,
    /// Exact emitted source-order evidence owned by this point.
    pub wire_samples: Vec<WireSample>,
}

/// Atomic family/sample projection of one successful exposition.
#[derive(Clone, Debug, PartialEq)]
pub struct ExpositionRowsV1 {
    /// One row per family, including metadata-only families.
    pub families: Vec<MetricFamilyRowV1>,
    /// One row per structured MetricPoint.
    pub samples: Vec<MetricPointRowV1>,
}

/// Immutable context supplied after terminal frame identity is known.
pub struct ExpositionProjectionContextV1<'a> {
    /// Archive UUID.
    pub archive_id: ArchiveId,
    /// Collection session UUID.
    pub session_id: SessionId,
    /// Physical source identity.
    pub source_id: &'a str,
    /// Terminal frame ID inserted before logical-row hashing.
    pub frame_id: FrameId,
    /// Stable scrape batch ID.
    pub batch_id: BatchId,
    /// Owner-assigned global sequence.
    pub record_seq: u64,
    /// Authoritative capture Clock instant shared by every point in this frame.
    pub clock_ns: i64,
    /// Approximate Unix placement derived from the session anchor.
    pub unix_epoch_ns: Option<i128>,
    /// Installed source-local attribute epoch.
    pub attribute_epoch_id: Digest,
    /// Prepared secret key provider.
    pub archive_key: &'a dyn ArchiveKeyProvider,
    /// Ordered additive enrichers.
    pub enrichers: &'a [&'a dyn TelemetryEnricher],
    /// Prepared structured sanitizer after the mandatory baseline.
    pub sanitizer: &'a dyn ArchiveSanitizer,
}

impl Debug for ExpositionProjectionContextV1<'_> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ExpositionProjectionContextV1")
            .field("archive_id", &self.archive_id)
            .field("session_id", &self.session_id)
            .field("source_id", &self.source_id)
            .field("frame_id", &self.frame_id)
            .field("batch_id", &self.batch_id)
            .field("record_seq", &self.record_seq)
            .field("clock_ns", &self.clock_ns)
            .field("unix_epoch_ns", &self.unix_epoch_ns)
            .field("attribute_epoch_id", &self.attribute_epoch_id)
            .field("archive_key", &self.archive_key.provider_id())
            .field("enricher_count", &self.enrichers.len())
            .field("sanitizer", &self.sanitizer)
            .finish()
    }
}

/// Projects every family/point or returns no partial row batch.
pub fn project_exposition_v1(
    exposition: &Exposition,
    context: &ExpositionProjectionContextV1<'_>,
) -> Result<ExpositionRowsV1, ExpositionProjectionError> {
    validate_source_id(context.source_id)?;
    let source_series_key = context
        .archive_key
        .derive_subkey(ArchiveSubkey::SourceSeries)
        .map_err(ExpositionProjectionError::ArchiveKey)?;
    let mut rows = ExpositionRowsV1 {
        families: Vec::with_capacity(exposition.families.len()),
        samples: Vec::with_capacity(exposition.metric_point_count()),
    };
    let mut stored_identities = BTreeMap::<Digest, Digest>::new();

    for family in &exposition.families {
        let metric_point_count = family.metrics.iter().try_fold(
            0_u64,
            |total, metric| -> Result<u64, ExpositionProjectionError> {
                total
                    .checked_add(usize_to_u64(metric.points.len())?)
                    .ok_or(ExpositionProjectionError::CountOverflow)
            },
        )?;
        let wire_sample_count = family.metrics.iter().try_fold(0_u64, |total, metric| {
            metric.points.iter().try_fold(total, |total, point| {
                total
                    .checked_add(usize_to_u64(point.wire_samples.len())?)
                    .ok_or(ExpositionProjectionError::CountOverflow)
            })
        })?;
        rows.families.push(MetricFamilyRowV1 {
            archive_id: context.archive_id,
            session_id: context.session_id,
            frame_id: context.frame_id,
            batch_id: context.batch_id,
            record_seq: context.record_seq,
            family_seq: family.family_seq,
            metric_family: family.name.clone(),
            source_type_token: family.source_type_token.clone(),
            semantic_type: family.semantic_type,
            help_present: family.help.is_some(),
            type_present: family.type_line.is_some(),
            unit_present: family.unit.is_some(),
            help: family.help.as_ref().map(|line| line.value.clone()),
            unit: family.unit.as_ref().map(|line| line.value.clone()),
            help_line_seq: family
                .help
                .as_ref()
                .map(|line| usize_to_u64(line.line))
                .transpose()?,
            type_line_seq: family.type_line.map(usize_to_u64).transpose()?,
            unit_line_seq: family
                .unit
                .as_ref()
                .map(|line| usize_to_u64(line.line))
                .transpose()?,
            metric_count: usize_to_u64(family.metrics.len())?,
            metric_point_count,
            wire_sample_count,
        });

        for metric in &family.metrics {
            for point in &metric.points {
                if point.labels != metric.labels {
                    return Err(ExpositionProjectionError::PointLabelMismatch {
                        family: family.name.clone(),
                        metric_point_seq: point.metric_point_seq,
                    });
                }
                let mut attributes = AttributeMap::new();
                for enricher in context.enrichers {
                    let patch = enricher
                        .attributes(ArchiveSampleView {
                            source_id: context.source_id,
                            metric_family: &family.name,
                            semantic_type: family.semantic_type,
                            labels: &point.labels,
                            attributes: &attributes,
                        })
                        .map_err(ExpositionProjectionError::Enrichment)?;
                    validate_attribute_patch(&patch)
                        .map_err(ExpositionProjectionError::Enrichment)?;
                    for (key, value) in patch {
                        if attributes.insert(key.clone(), value).is_some() {
                            return Err(ExpositionProjectionError::Enrichment(
                                EnrichmentError::DuplicateAttribute(key),
                            ));
                        }
                    }
                }

                let source_labels = canonical_map_bytes(&point.labels)?;
                let semantic = semantic_type_id(family.semantic_type);
                let protected = keyed_domain_digest(
                    &source_series_key,
                    "aiperf.archive.series-source.v1",
                    &[
                        context.source_id.as_bytes(),
                        family.name.as_bytes(),
                        semantic.as_bytes(),
                        &source_labels,
                    ],
                );
                let sanitized = context
                    .sanitizer
                    .sanitize_sample(ArchiveSampleView {
                        source_id: context.source_id,
                        metric_family: &family.name,
                        semantic_type: family.semantic_type,
                        labels: &point.labels,
                        attributes: &attributes,
                    })
                    .map_err(ExpositionProjectionError::Sanitization)?;
                validate_stored_map(&sanitized.labels, "stored label")?;
                validate_stored_map(&sanitized.attributes, "stored attribute")?;
                let stored_labels = canonical_map_bytes(&sanitized.labels)?;
                let stored_identity = domain_digest(
                    "aiperf.archive.stored-series-identity.v1",
                    &[
                        context.source_id.as_bytes(),
                        family.name.as_bytes(),
                        semantic.as_bytes(),
                        &stored_labels,
                    ],
                );
                if let Some(previous) = stored_identities.insert(stored_identity, protected)
                    && previous != protected
                {
                    return Err(ExpositionProjectionError::SanitizerMergedSeries {
                        stored_identity,
                        first_source_series_key: previous,
                        second_source_series_key: protected,
                    });
                }
                let displayed = domain_digest(
                    "aiperf.archive.series.v1",
                    &[stored_identity.as_bytes(), protected.as_bytes()],
                );

                let (wire_merged_info_labels, partition_status, partition_policy_id) =
                    match &point.value {
                        MetricValue::Info(info) => (
                            Some(info.wire_merged_labels.clone()),
                            match info.partition_status {
                                InfoLabelPartitionStatus::UnavailableFromText => {
                                    ArchiveInfoLabelPartitionStatus::UnavailableFromText
                                }
                                InfoLabelPartitionStatus::PolicyApplied => {
                                    ArchiveInfoLabelPartitionStatus::PolicyApplied
                                }
                            },
                            info.partition_policy_id.clone(),
                        ),
                        _ => (None, ArchiveInfoLabelPartitionStatus::NotApplicable, None),
                    };
                rows.samples.push(MetricPointRowV1 {
                    archive_id: context.archive_id,
                    session_id: context.session_id,
                    source_id: context.source_id.to_owned(),
                    frame_id: context.frame_id,
                    batch_id: context.batch_id,
                    record_seq: context.record_seq,
                    family_seq: family.family_seq,
                    metric_point_seq: point.metric_point_seq,
                    clock_ns: context.clock_ns,
                    unix_epoch_ns: context.unix_epoch_ns,
                    metric_family: family.name.clone(),
                    source_type_token: family.source_type_token.clone(),
                    semantic_type: family.semantic_type,
                    source_series_key: protected,
                    series_key: displayed,
                    labels: sanitized.labels,
                    attributes: sanitized.attributes,
                    wire_merged_info_labels,
                    info_label_partition_status: partition_status,
                    info_label_partition_policy_id: partition_policy_id,
                    attribute_epoch_id: context.attribute_epoch_id,
                    point_time_status: point.point_time_status,
                    source_timestamp: point.source_timestamp.clone(),
                    payload: point.value.clone(),
                    wire_samples: point.wire_samples.clone(),
                });
            }
        }
    }

    if rows.samples.len() != exposition.metric_point_count()
        || rows
            .samples
            .iter()
            .map(|row| row.wire_samples.len())
            .sum::<usize>()
            != exposition.wire_sample_count
    {
        return Err(ExpositionProjectionError::ProjectionCardinalityMismatch);
    }
    Ok(rows)
}

fn semantic_type_id(value: SemanticType) -> &'static str {
    match value {
        SemanticType::Unknown => "unknown",
        SemanticType::Gauge => "gauge",
        SemanticType::Counter => "counter",
        SemanticType::StateSet => "stateset",
        SemanticType::Info => "info",
        SemanticType::Histogram => "histogram",
        SemanticType::GaugeHistogram => "gauge_histogram",
        SemanticType::Summary => "summary",
    }
}

fn canonical_map_bytes(
    map: &BTreeMap<String, String>,
) -> Result<Vec<u8>, ExpositionProjectionError> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&usize_to_u64(map.len())?.to_be_bytes());
    for (key, value) in map {
        append_bytes(&mut bytes, key.as_bytes())?;
        append_bytes(&mut bytes, value.as_bytes())?;
    }
    Ok(bytes)
}

fn append_bytes(output: &mut Vec<u8>, value: &[u8]) -> Result<(), ExpositionProjectionError> {
    output.extend_from_slice(&usize_to_u64(value.len())?.to_be_bytes());
    output.extend_from_slice(value);
    Ok(())
}

fn usize_to_u64(value: usize) -> Result<u64, ExpositionProjectionError> {
    u64::try_from(value).map_err(|_| ExpositionProjectionError::CountOverflow)
}

fn validate_source_id(value: &str) -> Result<(), ExpositionProjectionError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ExpositionProjectionError::InvalidSourceId);
    }
    Ok(())
}

fn validate_attribute_patch(value: &AttributeMap) -> Result<(), EnrichmentError> {
    for (key, item) in value {
        if key.is_empty() || key.trim() != key || key.starts_with("aiperf.") {
            return Err(EnrichmentError::InvalidAttributeKey(key.clone()));
        }
        if item.contains('\0') {
            return Err(EnrichmentError::InvalidAttributeValue(key.clone()));
        }
    }
    Ok(())
}

fn validate_stored_map(
    value: &BTreeMap<String, String>,
    kind: &'static str,
) -> Result<(), ExpositionProjectionError> {
    for (key, item) in value {
        if key.is_empty() || key.trim() != key || key.contains('\0') || item.contains('\0') {
            return Err(ExpositionProjectionError::InvalidStoredMapEntry {
                kind,
                key: key.clone(),
            });
        }
    }
    Ok(())
}

/// Additive enrichment rejected before sanitization.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EnrichmentError {
    /// Empty/padded/reserved attribute key.
    InvalidAttributeKey(String),
    /// Attribute value contains a NUL byte.
    InvalidAttributeValue(String),
    /// Two enrichers attempted to own the same key.
    DuplicateAttribute(String),
    /// Implementation-specific bounded failure.
    Failed(String),
}

impl Display for EnrichmentError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAttributeKey(key) => write!(
                formatter,
                "telemetry enricher produced invalid or reserved attribute key {key:?}"
            ),
            Self::InvalidAttributeValue(key) => {
                write!(formatter, "telemetry attribute {key:?} contains a NUL byte")
            }
            Self::DuplicateAttribute(key) => {
                write!(formatter, "duplicate telemetry attribute {key:?}")
            }
            Self::Failed(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for EnrichmentError {}

/// Structured sanitizer rejected one sample atomically.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SanitizationError {
    /// Bounded redaction-safe diagnostic.
    pub message: String,
}

impl Display for SanitizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for SanitizationError {}

/// Atomic exposition projection failure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpositionProjectionError {
    /// Stable source identity was empty, padded, or contained a control byte.
    InvalidSourceId,
    /// Key-provider resolution/derivation failed.
    ArchiveKey(ArchiveKeyError),
    /// A parser invariant was violated before durable projection.
    PointLabelMismatch {
        /// Source family.
        family: String,
        /// Source-order point sequence.
        metric_point_seq: u64,
    },
    /// Additive enrichment failed.
    Enrichment(EnrichmentError),
    /// Structured sanitization failed.
    Sanitization(SanitizationError),
    /// A stored map contained an invalid key/value surface.
    InvalidStoredMapEntry {
        /// Label or attribute map.
        kind: &'static str,
        /// Offending key.
        key: String,
    },
    /// Post-redaction identity merged two protected source series.
    SanitizerMergedSeries {
        /// Colliding post-redaction identity before protected-key binding.
        stored_identity: Digest,
        /// First protected source identity.
        first_source_series_key: Digest,
        /// Second protected source identity.
        second_source_series_key: Digest,
    },
    /// A platform count/line length did not fit the UInt64 wire type.
    CountOverflow,
    /// Family/point/wire cardinality changed during projection.
    ProjectionCardinalityMismatch,
}

impl Display for ExpositionProjectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSourceId => formatter.write_str("invalid telemetry source ID"),
            Self::ArchiveKey(error) => write!(formatter, "archive key provider failed: {error}"),
            Self::PointLabelMismatch {
                family,
                metric_point_seq,
            } => write!(
                formatter,
                "metric point {metric_point_seq} in family {family:?} disagrees with its metric labels"
            ),
            Self::Enrichment(error) => write!(formatter, "telemetry enrichment failed: {error}"),
            Self::Sanitization(error) => {
                write!(formatter, "telemetry sanitization failed: {error}")
            }
            Self::InvalidStoredMapEntry { kind, key } => {
                write!(formatter, "{kind} has invalid key/value at {key:?}")
            }
            Self::SanitizerMergedSeries {
                stored_identity,
                first_source_series_key,
                second_source_series_key,
            } => write!(
                formatter,
                "sanitizer merged protected source series {first_source_series_key} and {second_source_series_key} into stored identity {stored_identity}"
            ),
            Self::CountOverflow => formatter.write_str("archive row count overflowed UInt64"),
            Self::ProjectionCardinalityMismatch => {
                formatter.write_str("archive exposition projection changed parser cardinality")
            }
        }
    }
}

impl std::error::Error for ExpositionProjectionError {}

#[cfg(test)]
mod tests {
    use aiperf_prometheus::{
        ExpositionFormat, ExpositionParser, NumberKind, ParseLimits, StrictExpositionParser,
    };

    use super::*;
    use crate::{Blake3ArchiveKeyProvider, FrameIdentityV1, SourceOutcome};

    fn parse(body: &str) -> Exposition {
        parse_as(ExpositionFormat::PrometheusText004, body)
    }

    fn parse_as(format: ExpositionFormat, body: &str) -> Exposition {
        StrictExpositionParser
            .parse(format, body.as_bytes(), &ParseLimits::default())
            .unwrap()
    }

    fn fixture_ids() -> (ArchiveId, SessionId, BatchId, FrameId, Digest) {
        let archive = ArchiveId::new([1; 16]).unwrap();
        let session = SessionId::new([2; 16]).unwrap();
        let batch = FrameIdentityV1::source_scrape_batch(
            archive,
            session,
            "source-a",
            0,
            SourceOutcome::Success,
            None,
        )
        .unwrap();
        let reservation = FrameIdentityV1::projection_reservation(
            archive,
            session,
            crate::ReservationKind::SourceScrape,
            Some("source-a"),
            batch,
            0,
        )
        .unwrap();
        let frame =
            FrameIdentityV1::terminal_frame(crate::TerminalKind::SourceScrape, reservation, 0);
        let epoch = domain_digest("aiperf.archive.test-epoch.v1", &[b"epoch"]);
        (archive, session, batch, frame, epoch)
    }

    fn project_with<'a>(
        exposition: &Exposition,
        key: &'a dyn ArchiveKeyProvider,
        enrichers: &'a [&'a dyn TelemetryEnricher],
        sanitizer: &'a dyn ArchiveSanitizer,
    ) -> Result<ExpositionRowsV1, ExpositionProjectionError> {
        let (archive_id, session_id, batch_id, frame_id, attribute_epoch_id) = fixture_ids();
        project_exposition_v1(
            exposition,
            &ExpositionProjectionContextV1 {
                archive_id,
                session_id,
                source_id: "source-a",
                frame_id,
                batch_id,
                record_seq: 0,
                clock_ns: 123,
                unix_epoch_ns: Some(456),
                attribute_epoch_id,
                archive_key: key,
                enrichers,
                sanitizer,
            },
        )
    }

    #[test]
    fn metadata_only_families_and_exact_wire_samples_survive_projection() {
        let exposition = parse(
            "# HELP empty documented only\n# TYPE empty gauge\n\
             # TYPE latency histogram\n\
             latency_bucket{instance=\"a\",le=\"1\"} 16777217\n\
             latency_bucket{instance=\"a\",le=\"+Inf\"} 16777218\n\
             latency_sum{instance=\"a\"} 3.5\n\
             latency_count{instance=\"a\"} 16777218\n",
        );
        let key = Blake3ArchiveKeyProvider::new("fixture_key", [9; 32]).unwrap();
        let rows = project_with(&exposition, &key, &[], &NoopSanitizer).unwrap();

        assert_eq!(rows.families.len(), 2);
        assert_eq!(rows.families[0].metric_family, "empty");
        assert_eq!(rows.families[0].metric_point_count, 0);
        assert_eq!(rows.samples.len(), 1);
        assert_eq!(rows.samples[0].wire_samples.len(), 4);
        assert_eq!(
            rows.samples[0].wire_samples[0]
                .value
                .source_lexeme
                .as_deref(),
            Some("16777217")
        );
        assert_eq!(
            rows.samples[0].wire_samples[0].value.kind,
            NumberKind::Finite
        );
    }

    #[test]
    fn additive_attributes_never_change_series_identity() {
        let exposition = parse("# TYPE temperature gauge\ntemperature{sensor=\"a\"} 1\n");
        let key = Blake3ArchiveKeyProvider::new("fixture_key", [9; 32]).unwrap();
        let left =
            StaticLabelEnricher::new(BTreeMap::from([("cluster".to_owned(), "left".to_owned())]))
                .unwrap();
        let right =
            StaticLabelEnricher::new(BTreeMap::from([("cluster".to_owned(), "right".to_owned())]))
                .unwrap();
        let left_rows = project_with(&exposition, &key, &[&left], &NoopSanitizer).unwrap();
        let right_rows = project_with(&exposition, &key, &[&right], &NoopSanitizer).unwrap();

        assert_eq!(
            left_rows.samples[0].source_series_key,
            right_rows.samples[0].source_series_key
        );
        assert_eq!(
            left_rows.samples[0].series_key,
            right_rows.samples[0].series_key
        );
        assert_ne!(
            left_rows.samples[0].attributes,
            right_rows.samples[0].attributes
        );
    }

    #[derive(Debug)]
    struct RemoveInstance;

    impl ArchiveSanitizer for RemoveInstance {
        fn sanitize_sample(
            &self,
            sample: ArchiveSampleView<'_>,
        ) -> Result<SanitizedSample, SanitizationError> {
            let mut labels = sample.labels.clone();
            labels.remove("instance");
            Ok(SanitizedSample {
                labels,
                attributes: sample.attributes.clone(),
            })
        }
    }

    #[test]
    fn sanitizer_cannot_silently_merge_wire_distinct_series() {
        let exposition = parse(
            "# TYPE temperature gauge\n\
             temperature{instance=\"a\"} 1\n\
             temperature{instance=\"b\"} 2\n",
        );
        let key = Blake3ArchiveKeyProvider::new("fixture_key", [9; 32]).unwrap();
        assert!(matches!(
            project_with(&exposition, &key, &[], &RemoveInstance),
            Err(ExpositionProjectionError::SanitizerMergedSeries { .. })
        ));
    }

    #[test]
    fn text_info_preserves_the_complete_merged_identity() {
        let exposition = parse_as(
            ExpositionFormat::OpenMetricsText100,
            "# TYPE build info\n\
             build_info{instance=\"a\",version=\"1,2\\\"3\"} 1\n\
             # EOF\n",
        );
        let key = Blake3ArchiveKeyProvider::new("fixture_key", [9; 32]).unwrap();
        let rows = project_with(&exposition, &key, &[], &NoopSanitizer).unwrap();
        let point = &rows.samples[0];
        assert_eq!(
            point.info_label_partition_status,
            ArchiveInfoLabelPartitionStatus::UnavailableFromText
        );
        assert_eq!(point.wire_merged_info_labels.as_ref(), Some(&point.labels));
        assert_eq!(point.info_label_partition_policy_id, None);
    }

    #[test]
    fn secret_key_changes_protected_and_display_identities() {
        let exposition = parse("# TYPE temperature gauge\ntemperature{sensor=\"a\"} 1\n");
        let left = Blake3ArchiveKeyProvider::new("fixture_key", [1; 32]).unwrap();
        let right = Blake3ArchiveKeyProvider::new("fixture_key", [2; 32]).unwrap();
        let left = project_with(&exposition, &left, &[], &NoopSanitizer).unwrap();
        let right = project_with(&exposition, &right, &[], &NoopSanitizer).unwrap();
        assert_ne!(
            left.samples[0].source_series_key,
            right.samples[0].source_series_key
        );
        assert_ne!(left.samples[0].series_key, right.samples[0].series_key);
    }
}
