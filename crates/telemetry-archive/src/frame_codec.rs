// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed source-scrape frame projection and recoverable WAL payload encoding.
//!
//! The archive owner hands this module a terminally sequenced source frame. It
//! builds all three mandatory table projections, derives their canonical
//! logical evidence from the same Arrow values, and stores exact Arrow IPC
//! streams in a versioned WAL payload. Recovery can therefore rebuild Parquet
//! after a crash between WAL fsync and partition publication without reparsing
//! source bytes or inventing another identity.

use std::fmt::{self, Display, Formatter};
use std::io::Cursor;
use std::sync::Arc;

use aiperf_prometheus::{
    CountOrigin, CreatedTimestamp, ExactNumber, Exemplar, ExpositionFormat, F64Status, MetricValue,
    NumberKind, PointTimeStatus, SemanticType, SourceTimestamp, TimestampStatus, WireSample,
    WireSampleRole,
};
use arrow_array::builder::{
    ArrayBuilder, BooleanBuilder, Decimal128Builder, FixedSizeBinaryBuilder, Float64Builder,
    Int8Builder, Int16Builder, Int32Builder, Int64Builder, ListBuilder, MapBuilder, StringBuilder,
    StringDictionaryBuilder, StructBuilder, UInt8Builder, UInt16Builder, UInt32Builder,
    UInt64Builder, make_builder,
};
use arrow_array::types::Int8Type;
use arrow_array::{ArrayRef, RecordBatch, make_array};
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{ArrowError, DataType, SchemaRef};

use crate::{
    ArchiveId, ArchiveInfoLabelPartitionStatus, ArchiveSchemasV1, ArchiveScrapeRecordV1,
    ArchiveWalFrame, BoundaryRole, CanonicalLogicalRow, Digest, FrameTableProjectionV1,
    LogicalValue, MetricFamilyRowV1, MetricPointRowV1, ParquetProjectionError, RequiredProjection,
    ScrapeReasonV1, SequencedArchiveFrameV1, SessionId, SourceOutcome, TableId, TerminalKind,
    WalError, WalFrame, WalFrameHeaderV1,
};

const PAYLOAD_MAGIC: &[u8; 8] = b"AIPFWP01";
const PAYLOAD_VERSION: u16 = 1;

/// Closed v1 converter between terminal source frames and durable WAL frames.
#[derive(Clone, Debug)]
pub struct SourceFrameCodecV1 {
    schemas: ArchiveSchemasV1,
}

impl SourceFrameCodecV1 {
    /// Loads and retains the checked-in six-table schema authority.
    pub fn new() -> Result<Self, SourceFrameCodecError> {
        Ok(Self {
            schemas: ArchiveSchemasV1::load().map_err(SourceFrameCodecError::Schema)?,
        })
    }

    /// Uses an already loaded schema authority.
    #[must_use]
    pub fn with_schemas(schemas: ArchiveSchemasV1) -> Self {
        Self { schemas }
    }

    /// Projects one terminal source event into its complete WAL authority.
    pub fn encode_source_frame(
        &self,
        sequenced: SequencedArchiveFrameV1,
    ) -> Result<ArchiveWalFrame, SourceFrameCodecError> {
        let attempt = &sequenced.frame.attempt;
        let authoritative_frame_clock_ns = match attempt.outcome {
            SourceOutcome::Success | SourceOutcome::Empty => attempt
                .capture_ns
                .ok_or(SourceFrameCodecError::MissingAuthoritativeClock)?,
            _ => attempt.outcome_observed_ns,
        };

        let mut projections = vec![
            self.attempt_projection(attempt, authoritative_frame_clock_ns)?,
            self.family_projection(
                attempt,
                &sequenced.frame.exposition.families,
                authoritative_frame_clock_ns,
            )?,
            self.sample_projection(
                attempt,
                &sequenced.frame.exposition.samples,
                authoritative_frame_clock_ns,
            )?,
        ];
        projections.sort_unstable_by_key(|projection| projection.table);

        let required_projections = projections
            .iter()
            .map(|projection| {
                Ok(RequiredProjection {
                    table: projection.table,
                    evidence: projection.validate(&self.schemas)?,
                })
            })
            .collect::<Result<Vec<_>, ParquetProjectionError>>()
            .map_err(SourceFrameCodecError::Projection)?;
        let payload = encode_payload(&projections)?;
        let payload_len =
            u64::try_from(payload.len()).map_err(|_| SourceFrameCodecError::LengthOverflow)?;
        let header = WalFrameHeaderV1::new(
            attempt.batch_id,
            sequenced.projection_reservation_id,
            attempt.record_seq,
            authoritative_frame_clock_ns,
            TerminalKind::SourceScrape,
            required_projections,
            Vec::new(),
            Vec::new(),
            payload_len,
        )
        .map_err(SourceFrameCodecError::Wal)?;
        if header.frame_id != attempt.frame_id {
            return Err(SourceFrameCodecError::TerminalIdentityMismatch);
        }
        let wal_frame = WalFrame::new(header, payload).map_err(SourceFrameCodecError::Wal)?;
        Ok(ArchiveWalFrame {
            wal_frame,
            table_projections: projections,
        })
    }

    /// Rebuilds table projections from a verified source-scrape WAL frame.
    ///
    /// The caller supplies the segment-bound archive/session identities because
    /// those are authoritative in the WAL segment header rather than repeated
    /// in the final frame header.
    pub fn decode_source_frame(
        &self,
        archive_id: ArchiveId,
        session_id: SessionId,
        wal_frame: WalFrame,
    ) -> Result<ArchiveWalFrame, SourceFrameCodecError> {
        if wal_frame.header().terminal_kind != TerminalKind::SourceScrape {
            return Err(SourceFrameCodecError::UnsupportedTerminalKind);
        }
        let projections = decode_payload(
            &self.schemas,
            archive_id,
            session_id,
            wal_frame.header().frame_id,
            wal_frame.header().authoritative_frame_clock_ns,
            wal_frame.payload(),
        )?;
        let declared = &wal_frame.header().required_projections;
        if projections.len() != declared.len() {
            return Err(SourceFrameCodecError::ProjectionSetMismatch);
        }
        for (projection, required) in projections.iter().zip(declared) {
            if projection.table != required.table
                || projection
                    .validate(&self.schemas)
                    .map_err(SourceFrameCodecError::Projection)?
                    != required.evidence
            {
                return Err(SourceFrameCodecError::ProjectionSetMismatch);
            }
        }
        Ok(ArchiveWalFrame {
            wal_frame,
            table_projections: projections,
        })
    }

    fn attempt_projection(
        &self,
        attempt: &ArchiveScrapeRecordV1,
        authoritative_frame_clock_ns: i64,
    ) -> Result<FrameTableProjectionV1, SourceFrameCodecError> {
        self.projection(
            attempt,
            authoritative_frame_clock_ns,
            TableId::Attempts,
            vec![attempt_values(attempt)],
        )
    }

    fn family_projection(
        &self,
        attempt: &ArchiveScrapeRecordV1,
        rows: &[MetricFamilyRowV1],
        authoritative_frame_clock_ns: i64,
    ) -> Result<FrameTableProjectionV1, SourceFrameCodecError> {
        self.projection(
            attempt,
            authoritative_frame_clock_ns,
            TableId::Families,
            rows.iter().map(family_values).collect(),
        )
    }

    fn sample_projection(
        &self,
        attempt: &ArchiveScrapeRecordV1,
        rows: &[MetricPointRowV1],
        authoritative_frame_clock_ns: i64,
    ) -> Result<FrameTableProjectionV1, SourceFrameCodecError> {
        self.projection(
            attempt,
            authoritative_frame_clock_ns,
            TableId::Samples,
            rows.iter().map(sample_values).collect(),
        )
    }

    fn projection(
        &self,
        attempt: &ArchiveScrapeRecordV1,
        authoritative_frame_clock_ns: i64,
        table: TableId,
        rows: Vec<Vec<LogicalValue>>,
    ) -> Result<FrameTableProjectionV1, SourceFrameCodecError> {
        let schema = self
            .schemas
            .table(table)
            .map_err(SourceFrameCodecError::Schema)?;
        let batch = logical_record_batch(schema.schema().clone(), &rows)?;
        let logical_rows = rows
            .iter()
            .map(|row| {
                CanonicalLogicalRow::encode(schema.logical_schema(), row)
                    .map_err(SourceFrameCodecError::LogicalRow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(FrameTableProjectionV1 {
            archive_id: attempt.archive_id,
            session_id: attempt.session_id,
            source_id: Some(attempt.source_id.clone()),
            frame_id: attempt.frame_id,
            authoritative_frame_clock_ns,
            table,
            batch,
            logical_rows,
        })
    }
}

impl Default for SourceFrameCodecV1 {
    fn default() -> Self {
        Self::new().expect("checked-in archive schemas are valid")
    }
}

fn attempt_values(row: &ArchiveScrapeRecordV1) -> Vec<LogicalValue> {
    vec![
        uuid(row.archive_id),
        uuid_session(row.session_id),
        text(&row.source_id),
        u64_value(row.record_seq),
        u64_value(row.source_record_seq),
        optional_u64(row.request_attempt_seq),
        digest(row.frame_id.digest()),
        digest(row.batch_id.digest()),
        enum_value(match row.reason {
            ScrapeReasonV1::Continuous => "continuous",
            ScrapeReasonV1::Boundary => "boundary",
        }),
        enum_value(source_outcome(row.outcome)),
        LogicalValue::List(
            row.boundary_refs
                .iter()
                .map(|reference| {
                    LogicalValue::Struct(vec![
                        text(&reference.transition_id),
                        text(&reference.boundary_id),
                        text(&reference.phase_id),
                        text(&reference.source_id),
                        enum_value(match reference.role {
                            BoundaryRole::PhaseStart => "phase_start",
                            BoundaryRole::PhaseEnd => "phase_end",
                        }),
                        optional_text(reference.coalescing_group_id.as_deref()),
                    ])
                })
                .collect(),
        ),
        optional_text(row.declared_media_type.as_deref()),
        optional_format(row.strict_parser_format),
        optional_format(row.native_compatibility_format),
        LogicalValue::Bool(row.native_compatibility_fallback),
        optional_i64(row.scheduled_ns),
        optional_i64(row.request_start_ns),
        optional_i64(row.first_byte_ns),
        optional_i64(row.capture_ns),
        optional_i64(row.parse_done_ns),
        optional_i64(row.archive_enqueue_ns),
        i64_value(row.outcome_observed_ns),
        optional_decimal(row.unix_epoch_ns),
        optional_u16(row.http_status),
        optional_i64(row.latency_ns),
        optional_digest(row.decoded_body_digest),
        optional_digest(row.encoded_body_digest),
        optional_digest(row.raw_object_id),
        LogicalValue::Bool(row.body_unchanged),
        optional_u64(row.same_body_as_source_record_seq),
        u64_value(row.family_count),
        u64_value(row.metric_point_count),
        u64_value(row.wire_sample_count),
        optional_text(row.error_kind.as_deref()),
        optional_text(row.error_message.as_deref()),
    ]
}

fn family_values(row: &MetricFamilyRowV1) -> Vec<LogicalValue> {
    vec![
        uuid(row.archive_id),
        uuid_session(row.session_id),
        text(&row.source_id),
        digest(row.frame_id.digest()),
        digest(row.batch_id.digest()),
        u64_value(row.record_seq),
        u64_value(row.family_seq),
        text(&row.metric_family),
        text(&row.source_type_token),
        enum_value(semantic_type(row.semantic_type)),
        LogicalValue::Bool(row.help_present),
        LogicalValue::Bool(row.type_present),
        LogicalValue::Bool(row.unit_present),
        optional_text(row.help.as_deref()),
        optional_text(row.unit.as_deref()),
        optional_u64(row.help_line_seq),
        optional_u64(row.type_line_seq),
        optional_u64(row.unit_line_seq),
        u64_value(row.metric_count),
        u64_value(row.metric_point_count),
        u64_value(row.wire_sample_count),
    ]
}

fn sample_values(row: &MetricPointRowV1) -> Vec<LogicalValue> {
    vec![
        uuid(row.archive_id),
        uuid_session(row.session_id),
        text(&row.source_id),
        digest(row.frame_id.digest()),
        digest(row.batch_id.digest()),
        u64_value(row.record_seq),
        u64_value(row.family_seq),
        u64_value(row.metric_point_seq),
        i64_value(row.clock_ns),
        optional_decimal(row.unix_epoch_ns),
        text(&row.metric_family),
        text(&row.source_type_token),
        enum_value(semantic_type(row.semantic_type)),
        digest(row.source_series_key),
        digest(row.series_key),
        string_map(&row.labels),
        string_map(&row.attributes),
        row.wire_merged_info_labels
            .as_ref()
            .map_or(LogicalValue::Null, string_map),
        enum_value(match row.info_label_partition_status {
            ArchiveInfoLabelPartitionStatus::NotApplicable => "not_applicable",
            ArchiveInfoLabelPartitionStatus::UnavailableFromText => "unavailable_from_text",
            ArchiveInfoLabelPartitionStatus::PolicyApplied => "policy_applied",
        }),
        optional_text(row.info_label_partition_policy_id.as_deref()),
        digest(row.attribute_epoch_id),
        enum_value(match row.point_time_status {
            PointTimeStatus::AllAbsent => "all_absent",
            PointTimeStatus::UniformExplicit => "uniform_explicit",
            PointTimeStatus::MixedComponents => "mixed_components",
            PointTimeStatus::PartialComponents => "partial_components",
        }),
        timestamp(&row.source_timestamp),
        metric_payload(&row.payload),
        LogicalValue::List(row.wire_samples.iter().map(wire_sample).collect()),
    ]
}

fn metric_payload(value: &MetricValue) -> LogicalValue {
    let mut branches = vec![LogicalValue::Null; 6];
    match value {
        MetricValue::Scalar { value, exemplar } => {
            branches[0] = LogicalValue::Struct(vec![
                archive_number(value),
                optional_exemplar(exemplar.as_ref()),
            ]);
        }
        MetricValue::Counter(counter) => {
            branches[1] = LogicalValue::Struct(vec![
                archive_number(&counter.total),
                created_timestamp(&counter.created),
                optional_exemplar(counter.exemplar.as_ref()),
            ]);
        }
        MetricValue::StateSet(states) => {
            branches[2] = LogicalValue::List(
                states
                    .iter()
                    .map(|state| {
                        LogicalValue::Struct(vec![
                            text(&state.state),
                            archive_number(&state.enabled),
                        ])
                    })
                    .collect(),
            );
        }
        MetricValue::Info(info) => {
            branches[3] = LogicalValue::Struct(vec![
                string_map(&info.wire_merged_labels),
                info.partitioned_metric_labels
                    .as_ref()
                    .map_or(LogicalValue::Null, string_map),
                info.partitioned_value_labels
                    .as_ref()
                    .map_or(LogicalValue::Null, string_map),
                optional_text(info.partition_policy_id.as_deref()),
            ]);
        }
        MetricValue::Histogram(histogram) => {
            branches[4] = LogicalValue::Struct(vec![
                archive_number(&histogram.sum),
                archive_number(&histogram.count),
                enum_value(match histogram.count_origin {
                    CountOrigin::EmittedAndValidated => "emitted_and_validated",
                    CountOrigin::DerivedFromPositiveInfinity => "derived_from_pos_inf",
                }),
                created_timestamp(&histogram.created),
                LogicalValue::List(
                    histogram
                        .buckets
                        .iter()
                        .map(|bucket| {
                            LogicalValue::Struct(vec![
                                text(&bucket.upper_bound_lexeme),
                                archive_number(&bucket.upper_bound),
                                archive_number(&bucket.cumulative_count),
                                optional_exemplar(bucket.exemplar.as_ref()),
                            ])
                        })
                        .collect(),
                ),
            ]);
        }
        MetricValue::Summary(summary) => {
            branches[5] = LogicalValue::Struct(vec![
                archive_number(&summary.sum),
                archive_number(&summary.count),
                created_timestamp(&summary.created),
                LogicalValue::List(
                    summary
                        .quantiles
                        .iter()
                        .map(|quantile| {
                            LogicalValue::Struct(vec![
                                text(&quantile.quantile_lexeme),
                                archive_number(&quantile.quantile),
                                archive_number(&quantile.value),
                            ])
                        })
                        .collect(),
                ),
            ]);
        }
    }
    LogicalValue::Struct(branches)
}

fn wire_sample(sample: &WireSample) -> LogicalValue {
    LogicalValue::Struct(vec![
        text(&sample.emitted_name),
        enum_value(wire_role(sample.role)),
        string_map(&sample.labels),
        archive_number(&sample.value),
        timestamp(&sample.source_timestamp),
        optional_exemplar(sample.exemplar.as_ref()),
    ])
}

fn optional_exemplar(value: Option<&Exemplar>) -> LogicalValue {
    value.map_or(LogicalValue::Null, |value| {
        LogicalValue::Struct(vec![
            string_map(&value.labels),
            archive_number(&value.value),
            timestamp(&value.timestamp),
        ])
    })
}

fn archive_number(value: &ExactNumber) -> LogicalValue {
    LogicalValue::Struct(vec![
        enum_value(match value.kind {
            NumberKind::Finite => "finite",
            NumberKind::PositiveInfinity => "pos_inf",
            NumberKind::NegativeInfinity => "neg_inf",
            NumberKind::NaN => "nan",
            NumberKind::Absent => "absent",
        }),
        optional_text(value.source_lexeme.as_deref()),
        value
            .finite_value
            .map_or(LogicalValue::Null, LogicalValue::Float64),
        optional_u64(value.exact_u64),
        enum_value(match value.f64_status {
            F64Status::Exact => "exact",
            F64Status::Rounded => "rounded",
            F64Status::Unavailable => "unavailable",
            F64Status::NotApplicable => "not_applicable",
        }),
    ])
}

fn created_timestamp(value: &CreatedTimestamp) -> LogicalValue {
    timestamp(&value.value)
}

fn timestamp(value: &SourceTimestamp) -> LogicalValue {
    LogicalValue::Struct(vec![
        optional_text(value.lexeme.as_deref()),
        optional_decimal(value.normalized_unix_ns),
        enum_value(match value.status {
            TimestampStatus::Absent => "absent",
            TimestampStatus::ExactNanoseconds => "exact_ns",
            TimestampStatus::SubNanosecondPrecision => "sub_ns_precision",
            TimestampStatus::OutOfRange => "out_of_range",
            TimestampStatus::SubNanosecondOutOfRange => "sub_ns_out_of_range",
        }),
    ])
}

fn source_outcome(value: SourceOutcome) -> &'static str {
    match value {
        SourceOutcome::Success => "success",
        SourceOutcome::Empty => "empty",
        SourceOutcome::Http => "http",
        SourceOutcome::Transport => "transport",
        SourceOutcome::Timeout => "timeout",
        SourceOutcome::Parse => "parse",
        SourceOutcome::UnsupportedFormat => "unsupported_format",
        SourceOutcome::UnsupportedFeature => "unsupported_feature",
        SourceOutcome::Disabled => "disabled",
        SourceOutcome::Shutdown => "shutdown",
    }
}

fn semantic_type(value: SemanticType) -> &'static str {
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

fn wire_role(value: WireSampleRole) -> &'static str {
    match value {
        WireSampleRole::Scalar => "scalar",
        WireSampleRole::CounterTotal => "counter_total",
        WireSampleRole::CounterCreated => "counter_created",
        WireSampleRole::State => "state",
        WireSampleRole::Info => "info",
        WireSampleRole::HistogramBucket => "histogram_bucket",
        WireSampleRole::HistogramSum => "histogram_sum",
        WireSampleRole::HistogramCount => "histogram_count",
        WireSampleRole::HistogramCreated => "histogram_created",
        WireSampleRole::GaugeHistogramBucket => "gauge_histogram_bucket",
        WireSampleRole::GaugeHistogramSum => "gauge_histogram_sum",
        WireSampleRole::GaugeHistogramCount => "gauge_histogram_count",
        WireSampleRole::SummaryCount => "summary_count",
        WireSampleRole::SummarySum => "summary_sum",
        WireSampleRole::SummaryCreated => "summary_created",
        WireSampleRole::SummaryQuantile => "summary_quantile",
    }
}

fn optional_format(value: Option<ExpositionFormat>) -> LogicalValue {
    value.map_or(LogicalValue::Null, |format| {
        LogicalValue::String(
            match format {
                ExpositionFormat::PrometheusText004 => "prometheus_text_0_0_4",
                ExpositionFormat::OpenMetricsText100 => "openmetrics_text_1_0_0",
            }
            .to_owned(),
        )
    })
}

fn text(value: &str) -> LogicalValue {
    LogicalValue::String(value.to_owned())
}

fn enum_value(value: &str) -> LogicalValue {
    LogicalValue::String(value.to_owned())
}

fn u64_value(value: u64) -> LogicalValue {
    LogicalValue::Unsigned(u128::from(value))
}

fn i64_value(value: i64) -> LogicalValue {
    LogicalValue::Signed(i128::from(value))
}

fn optional_u64(value: Option<u64>) -> LogicalValue {
    value.map_or(LogicalValue::Null, u64_value)
}

fn optional_u16(value: Option<u16>) -> LogicalValue {
    value.map_or(LogicalValue::Null, |value| {
        LogicalValue::Unsigned(u128::from(value))
    })
}

fn optional_i64(value: Option<i64>) -> LogicalValue {
    value.map_or(LogicalValue::Null, i64_value)
}

fn optional_decimal(value: Option<i128>) -> LogicalValue {
    value.map_or(LogicalValue::Null, LogicalValue::Decimal128)
}

fn optional_text(value: Option<&str>) -> LogicalValue {
    value.map_or(LogicalValue::Null, text)
}

fn string_map<K, V>(value: &std::collections::BTreeMap<K, V>) -> LogicalValue
where
    K: AsRef<str> + Ord,
    V: AsRef<str>,
{
    LogicalValue::StringMap(
        value
            .iter()
            .map(|(key, value)| (key.as_ref().to_owned(), value.as_ref().to_owned()))
            .collect(),
    )
}

fn uuid(value: ArchiveId) -> LogicalValue {
    LogicalValue::Binary(value.as_bytes().to_vec())
}

fn uuid_session(value: SessionId) -> LogicalValue {
    LogicalValue::Binary(value.as_bytes().to_vec())
}

fn digest(value: Digest) -> LogicalValue {
    LogicalValue::Binary(value.as_bytes().to_vec())
}

fn optional_digest(value: Option<Digest>) -> LogicalValue {
    value.map_or(LogicalValue::Null, digest)
}

fn logical_record_batch(
    schema: SchemaRef,
    rows: &[Vec<LogicalValue>],
) -> Result<RecordBatch, SourceFrameCodecError> {
    if rows.iter().any(|row| row.len() != schema.fields().len()) {
        return Err(SourceFrameCodecError::LogicalFieldCount);
    }
    let builder_types = schema
        .fields()
        .iter()
        .map(|field| builder_data_type(field.data_type()))
        .collect::<Vec<_>>();
    let mut builders = builder_types
        .iter()
        .map(|data_type| make_builder(data_type, rows.len()))
        .collect::<Vec<_>>();
    for row in rows {
        for ((builder, data_type), value) in builders.iter_mut().zip(&builder_types).zip(row) {
            append_value(builder.as_mut(), data_type, value)?;
        }
    }
    let arrays = builders
        .iter_mut()
        .zip(schema.fields())
        .map(|(builder, field)| retype_array(builder.finish(), field.data_type()))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(schema, arrays).map_err(SourceFrameCodecError::Arrow)
}

fn builder_data_type(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Struct(fields) => DataType::Struct(
            fields
                .iter()
                .map(|field| {
                    Arc::new(
                        field
                            .as_ref()
                            .clone()
                            .with_data_type(builder_data_type(field.data_type())),
                    )
                })
                .collect(),
        ),
        DataType::List(field) => DataType::List(Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(builder_data_type(field.data_type())),
        )),
        DataType::Map(field, _) => DataType::Map(
            Arc::new(
                field
                    .as_ref()
                    .clone()
                    .with_data_type(builder_data_type(field.data_type())),
            ),
            false,
        ),
        _ => data_type.clone(),
    }
}

fn retype_array(
    array: ArrayRef,
    desired_type: &DataType,
) -> Result<ArrayRef, SourceFrameCodecError> {
    let data = array.to_data();
    let desired_children = match desired_type {
        DataType::Struct(fields) => fields
            .iter()
            .map(|field| field.data_type())
            .collect::<Vec<_>>(),
        DataType::List(field) | DataType::Map(field, _) => vec![field.data_type()],
        DataType::Dictionary(_, value) => vec![value.as_ref()],
        _ => Vec::new(),
    };
    if data.child_data().len() != desired_children.len() {
        return Err(SourceFrameCodecError::LogicalFieldCount);
    }
    let child_data = data
        .child_data()
        .iter()
        .zip(desired_children)
        .map(|(child, desired)| {
            retype_array(make_array(child.clone()), desired).map(|value| value.to_data())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let rebuilt = data
        .into_builder()
        .data_type(desired_type.clone())
        .child_data(child_data)
        .build()
        .map_err(SourceFrameCodecError::Arrow)?;
    Ok(make_array(rebuilt))
}

fn append_value(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
    value: &LogicalValue,
) -> Result<(), SourceFrameCodecError> {
    if matches!(value, LogicalValue::Null) {
        return append_null(builder, data_type);
    }
    macro_rules! primitive {
        ($builder:ty, $pattern:pat => $value:expr) => {{
            let builder = downcast_builder::<$builder>(builder, data_type)?;
            let $pattern = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            builder.append_value($value);
            Ok(())
        }};
    }
    match data_type {
        DataType::Boolean => primitive!(BooleanBuilder, LogicalValue::Bool(value) => *value),
        DataType::Int8 => {
            primitive!(Int8Builder, LogicalValue::Signed(value) => narrow_signed::<i8>(*value)?)
        }
        DataType::Int16 => {
            primitive!(Int16Builder, LogicalValue::Signed(value) => narrow_signed::<i16>(*value)?)
        }
        DataType::Int32 => {
            primitive!(Int32Builder, LogicalValue::Signed(value) => narrow_signed::<i32>(*value)?)
        }
        DataType::Int64 => {
            primitive!(Int64Builder, LogicalValue::Signed(value) => narrow_signed::<i64>(*value)?)
        }
        DataType::UInt8 => {
            primitive!(UInt8Builder, LogicalValue::Unsigned(value) => narrow_unsigned::<u8>(*value)?)
        }
        DataType::UInt16 => {
            primitive!(UInt16Builder, LogicalValue::Unsigned(value) => narrow_unsigned::<u16>(*value)?)
        }
        DataType::UInt32 => {
            primitive!(UInt32Builder, LogicalValue::Unsigned(value) => narrow_unsigned::<u32>(*value)?)
        }
        DataType::UInt64 => {
            primitive!(UInt64Builder, LogicalValue::Unsigned(value) => narrow_unsigned::<u64>(*value)?)
        }
        DataType::Float64 => {
            let builder = downcast_builder::<Float64Builder>(builder, data_type)?;
            let LogicalValue::Float64(value) = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            if !value.is_finite() {
                return Err(SourceFrameCodecError::NonFiniteFloat);
            }
            builder.append_value(*value);
            Ok(())
        }
        DataType::Decimal128(38, 0) => {
            let builder = downcast_builder::<Decimal128Builder>(builder, data_type)?;
            let LogicalValue::Decimal128(value) = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            builder.append_value(*value);
            Ok(())
        }
        DataType::Utf8 => primitive!(StringBuilder, LogicalValue::String(value) => value),
        DataType::FixedSizeBinary(width) => {
            let builder = downcast_builder::<FixedSizeBinaryBuilder>(builder, data_type)?;
            let LogicalValue::Binary(value) = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            if value.len() != usize::try_from(*width).unwrap_or(usize::MAX) {
                return Err(SourceFrameCodecError::FixedBinaryLength);
            }
            builder
                .append_value(value)
                .map_err(SourceFrameCodecError::Arrow)
        }
        DataType::Dictionary(index, values)
            if index.as_ref() == &DataType::Int8 && values.as_ref() == &DataType::Utf8 =>
        {
            let builder =
                downcast_builder::<StringDictionaryBuilder<Int8Type>>(builder, data_type)?;
            let LogicalValue::String(value) = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            builder
                .append(value)
                .map(|_| ())
                .map_err(SourceFrameCodecError::Arrow)
        }
        DataType::Struct(fields) => {
            let builder = downcast_builder::<StructBuilder>(builder, data_type)?;
            let LogicalValue::Struct(values) = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            if values.len() != fields.len() {
                return Err(SourceFrameCodecError::LogicalFieldCount);
            }
            for ((child, field), value) in builder
                .field_builders_mut()
                .iter_mut()
                .zip(fields)
                .zip(values)
            {
                append_value(child.as_mut(), field.data_type(), value)?;
            }
            builder.append(true);
            Ok(())
        }
        DataType::List(field) => {
            let builder =
                downcast_builder::<ListBuilder<Box<dyn ArrayBuilder>>>(builder, data_type)?;
            let LogicalValue::List(values) = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            for value in values {
                append_value(builder.values().as_mut(), field.data_type(), value)?;
            }
            builder.append(true);
            Ok(())
        }
        DataType::Map(field, _) => {
            let DataType::Struct(fields) = field.data_type() else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            if fields.len() != 2 {
                return Err(SourceFrameCodecError::LogicalFieldCount);
            }
            let builder = downcast_builder::<
                MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>,
            >(builder, data_type)?;
            let LogicalValue::StringMap(entries) = value else {
                return Err(SourceFrameCodecError::LogicalTypeMismatch);
            };
            let (keys, values) = builder.entries();
            for (key, value) in entries {
                append_value(keys.as_mut(), fields[0].data_type(), &text(key))?;
                append_value(values.as_mut(), fields[1].data_type(), &text(value))?;
            }
            builder.append(true).map_err(SourceFrameCodecError::Arrow)
        }
        _ => Err(SourceFrameCodecError::UnsupportedArrowType(
            data_type.clone(),
        )),
    }
}

fn append_null(
    builder: &mut dyn ArrayBuilder,
    data_type: &DataType,
) -> Result<(), SourceFrameCodecError> {
    macro_rules! null {
        ($builder:ty) => {{
            downcast_builder::<$builder>(builder, data_type)?.append_null();
            Ok(())
        }};
    }
    match data_type {
        DataType::Boolean => null!(BooleanBuilder),
        DataType::Int8 => null!(Int8Builder),
        DataType::Int16 => null!(Int16Builder),
        DataType::Int32 => null!(Int32Builder),
        DataType::Int64 => null!(Int64Builder),
        DataType::UInt8 => null!(UInt8Builder),
        DataType::UInt16 => null!(UInt16Builder),
        DataType::UInt32 => null!(UInt32Builder),
        DataType::UInt64 => null!(UInt64Builder),
        DataType::Float64 => null!(Float64Builder),
        DataType::Decimal128(38, 0) => null!(Decimal128Builder),
        DataType::Utf8 => null!(StringBuilder),
        DataType::FixedSizeBinary(_) => null!(FixedSizeBinaryBuilder),
        DataType::Dictionary(index, values)
            if index.as_ref() == &DataType::Int8 && values.as_ref() == &DataType::Utf8 =>
        {
            null!(StringDictionaryBuilder<Int8Type>)
        }
        DataType::Struct(fields) => {
            let builder = downcast_builder::<StructBuilder>(builder, data_type)?;
            for (child, field) in builder.field_builders_mut().iter_mut().zip(fields) {
                append_null(child.as_mut(), field.data_type())?;
            }
            builder.append(false);
            Ok(())
        }
        DataType::List(_) => {
            downcast_builder::<ListBuilder<Box<dyn ArrayBuilder>>>(builder, data_type)?
                .append(false);
            Ok(())
        }
        DataType::Map(_, _) => downcast_builder::<
            MapBuilder<Box<dyn ArrayBuilder>, Box<dyn ArrayBuilder>>,
        >(builder, data_type)?
        .append(false)
        .map_err(SourceFrameCodecError::Arrow),
        _ => Err(SourceFrameCodecError::UnsupportedArrowType(
            data_type.clone(),
        )),
    }
}

fn downcast_builder<'a, T: 'static>(
    builder: &'a mut dyn ArrayBuilder,
    data_type: &DataType,
) -> Result<&'a mut T, SourceFrameCodecError> {
    builder
        .as_any_mut()
        .downcast_mut::<T>()
        .ok_or_else(|| SourceFrameCodecError::BuilderType(data_type.clone()))
}

fn narrow_signed<T>(value: i128) -> Result<T, SourceFrameCodecError>
where
    T: TryFrom<i128>,
{
    T::try_from(value).map_err(|_| SourceFrameCodecError::IntegerOutOfRange)
}

fn narrow_unsigned<T>(value: u128) -> Result<T, SourceFrameCodecError>
where
    T: TryFrom<u128>,
{
    T::try_from(value).map_err(|_| SourceFrameCodecError::IntegerOutOfRange)
}

fn encode_payload(
    projections: &[FrameTableProjectionV1],
) -> Result<Vec<u8>, SourceFrameCodecError> {
    let mut payload = Vec::new();
    payload.extend_from_slice(PAYLOAD_MAGIC);
    payload.extend_from_slice(&PAYLOAD_VERSION.to_be_bytes());
    payload.extend_from_slice(
        &u16::try_from(projections.len())
            .map_err(|_| SourceFrameCodecError::LengthOverflow)?
            .to_be_bytes(),
    );
    for projection in projections {
        payload.push(projection.table as u8);
        match projection.source_id.as_deref() {
            None => payload.push(0),
            Some(source_id) => {
                payload.push(1);
                encode_bytes(&mut payload, source_id.as_bytes())?;
            }
        }
        let mut ipc = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc, projection.batch.schema().as_ref())
                .map_err(SourceFrameCodecError::Arrow)?;
            writer
                .write(&projection.batch)
                .map_err(SourceFrameCodecError::Arrow)?;
            writer.finish().map_err(SourceFrameCodecError::Arrow)?;
        }
        encode_bytes(&mut payload, &ipc)?;
    }
    Ok(payload)
}

fn decode_payload(
    schemas: &ArchiveSchemasV1,
    archive_id: ArchiveId,
    session_id: SessionId,
    frame_id: crate::FrameId,
    authoritative_frame_clock_ns: i64,
    payload: &[u8],
) -> Result<Vec<FrameTableProjectionV1>, SourceFrameCodecError> {
    let mut cursor = PayloadCursor::new(payload);
    if cursor.take(PAYLOAD_MAGIC.len())? != PAYLOAD_MAGIC {
        return Err(SourceFrameCodecError::PayloadMagic);
    }
    let version = cursor.u16()?;
    if version != PAYLOAD_VERSION {
        return Err(SourceFrameCodecError::PayloadVersion(version));
    }
    let count = usize::from(cursor.u16()?);
    let mut projections = Vec::with_capacity(count);
    let mut previous = None;
    for _ in 0..count {
        let table = TableId::from_u8(cursor.u8()?).map_err(SourceFrameCodecError::LogicalRow)?;
        if previous >= Some(table) {
            return Err(SourceFrameCodecError::ProjectionSetMismatch);
        }
        previous = Some(table);
        let source_id = match cursor.u8()? {
            0 => None,
            1 => Some(
                std::str::from_utf8(cursor.bytes()?)
                    .map_err(|_| SourceFrameCodecError::PayloadUtf8)?
                    .to_owned(),
            ),
            _ => return Err(SourceFrameCodecError::PayloadTag),
        };
        let ipc = cursor.bytes()?;
        let mut reader =
            StreamReader::try_new(Cursor::new(ipc), None).map_err(SourceFrameCodecError::Arrow)?;
        let batch = reader
            .next()
            .ok_or(SourceFrameCodecError::MissingIpcBatch)?
            .map_err(SourceFrameCodecError::Arrow)?;
        if reader.next().is_some() {
            return Err(SourceFrameCodecError::ExtraIpcBatch);
        }
        let schema = schemas
            .table(table)
            .map_err(SourceFrameCodecError::Schema)?;
        if batch.schema().as_ref() != schema.schema().as_ref() {
            return Err(SourceFrameCodecError::IpcSchemaMismatch(table));
        }
        let logical_rows = schema
            .canonical_rows(&batch)
            .map_err(SourceFrameCodecError::Schema)?;
        projections.push(FrameTableProjectionV1 {
            archive_id,
            session_id,
            source_id,
            frame_id,
            authoritative_frame_clock_ns,
            table,
            batch,
            logical_rows,
        });
    }
    if !cursor.is_empty() {
        return Err(SourceFrameCodecError::PayloadTrailingBytes);
    }
    Ok(projections)
}

fn encode_bytes(output: &mut Vec<u8>, value: &[u8]) -> Result<(), SourceFrameCodecError> {
    output.extend_from_slice(
        &u64::try_from(value.len())
            .map_err(|_| SourceFrameCodecError::LengthOverflow)?
            .to_be_bytes(),
    );
    output.extend_from_slice(value);
    Ok(())
}

struct PayloadCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> PayloadCursor<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], SourceFrameCodecError> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or(SourceFrameCodecError::LengthOverflow)?;
        let value = self
            .bytes
            .get(self.offset..end)
            .ok_or(SourceFrameCodecError::PayloadTruncated)?;
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, SourceFrameCodecError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SourceFrameCodecError> {
        Ok(u16::from_be_bytes(
            self.take(2)?.try_into().expect("checked two bytes"),
        ))
    }

    fn bytes(&mut self) -> Result<&'a [u8], SourceFrameCodecError> {
        let length = u64::from_be_bytes(self.take(8)?.try_into().expect("checked eight bytes"));
        let length = usize::try_from(length).map_err(|_| SourceFrameCodecError::LengthOverflow)?;
        self.take(length)
    }

    fn is_empty(&self) -> bool {
        self.offset == self.bytes.len()
    }
}

/// Terminal source-frame projection or WAL payload failure.
#[derive(Debug)]
pub enum SourceFrameCodecError {
    /// Checked-in schema load or value validation failed.
    Schema(crate::SchemaError),
    /// Arrow array, record-batch, or IPC operation failed.
    Arrow(ArrowError),
    /// Canonical logical-row encoding failed.
    LogicalRow(crate::LogicalRowError),
    /// Whole-frame projection invariants failed.
    Projection(ParquetProjectionError),
    /// WAL header/frame construction failed.
    Wal(WalError),
    /// A source outcome omitted its required authoritative Clock value.
    MissingAuthoritativeClock,
    /// The sequencer and final WAL identity derivations disagreed.
    TerminalIdentityMismatch,
    /// This codec accepts only source-scrape terminal frames.
    UnsupportedTerminalKind,
    /// Required projection declarations and payload tables disagreed.
    ProjectionSetMismatch,
    /// A logical row did not have the schema's exact field count.
    LogicalFieldCount,
    /// A logical value variant disagreed with its Arrow field.
    LogicalTypeMismatch,
    /// An integer exceeded the frozen physical width.
    IntegerOutOfRange,
    /// A non-finite value attempted to enter an analytical Float64 field.
    NonFiniteFloat,
    /// Fixed-size binary bytes had the wrong length.
    FixedBinaryLength,
    /// The dynamic builder type disagreed with its Arrow field.
    BuilderType(DataType),
    /// The generic row builder encountered an unsupported Arrow type.
    UnsupportedArrowType(DataType),
    /// A count or byte offset overflowed its frozen width.
    LengthOverflow,
    /// WAL payload magic was not v1 source-frame payload magic.
    PayloadMagic,
    /// WAL payload version is unsupported.
    PayloadVersion(u16),
    /// WAL payload ended before a declared field.
    PayloadTruncated,
    /// WAL payload carried an unknown option tag.
    PayloadTag,
    /// WAL payload source identity was not UTF-8.
    PayloadUtf8,
    /// WAL payload carried bytes after its declared table streams.
    PayloadTrailingBytes,
    /// An IPC table stream did not contain its one required batch.
    MissingIpcBatch,
    /// An IPC table stream contained more than one batch.
    ExtraIpcBatch,
    /// An IPC table stream did not use the checked-in table schema.
    IpcSchemaMismatch(TableId),
}

impl Display for SourceFrameCodecError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Schema(error) => write!(formatter, "archive schema: {error}"),
            Self::Arrow(error) => write!(formatter, "Arrow/IPC: {error}"),
            Self::LogicalRow(error) => write!(formatter, "logical row: {error}"),
            Self::Projection(error) => write!(formatter, "frame projection: {error}"),
            Self::Wal(error) => write!(formatter, "WAL frame: {error}"),
            Self::MissingAuthoritativeClock => {
                formatter.write_str("source frame omitted its authoritative Clock value")
            }
            Self::TerminalIdentityMismatch => {
                formatter.write_str("sequenced and WAL terminal frame identities differ")
            }
            Self::UnsupportedTerminalKind => {
                formatter.write_str("source-frame codec received a non-source terminal kind")
            }
            Self::ProjectionSetMismatch => {
                formatter.write_str("WAL payload projection set disagrees with its header")
            }
            Self::LogicalFieldCount => formatter.write_str("logical row field count mismatch"),
            Self::LogicalTypeMismatch => formatter.write_str("logical value type mismatch"),
            Self::IntegerOutOfRange => formatter.write_str("logical integer is out of range"),
            Self::NonFiniteFloat => formatter.write_str("non-finite analytical Float64"),
            Self::FixedBinaryLength => formatter.write_str("fixed binary length mismatch"),
            Self::BuilderType(data_type) => {
                write!(formatter, "Arrow builder type mismatch for {data_type}")
            }
            Self::UnsupportedArrowType(data_type) => {
                write!(formatter, "unsupported Arrow row type {data_type}")
            }
            Self::LengthOverflow => formatter.write_str("frame payload length overflow"),
            Self::PayloadMagic => formatter.write_str("invalid source-frame WAL payload magic"),
            Self::PayloadVersion(version) => {
                write!(
                    formatter,
                    "unsupported source-frame WAL payload version {version}"
                )
            }
            Self::PayloadTruncated => formatter.write_str("truncated source-frame WAL payload"),
            Self::PayloadTag => formatter.write_str("invalid source-frame WAL payload tag"),
            Self::PayloadUtf8 => formatter.write_str("source-frame payload identity is not UTF-8"),
            Self::PayloadTrailingBytes => {
                formatter.write_str("trailing source-frame WAL payload bytes")
            }
            Self::MissingIpcBatch => formatter.write_str("source-frame IPC stream has no batch"),
            Self::ExtraIpcBatch => {
                formatter.write_str("source-frame IPC stream has multiple batches")
            }
            Self::IpcSchemaMismatch(table) => {
                write!(formatter, "source-frame IPC schema mismatch for {table:?}")
            }
        }
    }
}

impl std::error::Error for SourceFrameCodecError {}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use aiperf_prometheus::StrictExpositionParser;
    use bytes::Bytes;

    use super::*;
    use crate::{
        ArchiveFrameSequencerV1, ArchiveFrameTimingV1, AttemptDecoder, Blake3ArchiveKeyProvider,
        DecodeLimits, EpochAnchor, FetchDisposition, FetchedAttempt, NativeEntityDecoder,
        NoopNativeEntityDecoder, ProjectionEvidence, PrometheusAttemptDecoder,
        SourceProjectionPolicyV1,
    };

    fn id(seed: u8) -> [u8; 16] {
        let mut bytes = [seed; 16];
        bytes[15] = seed.wrapping_add(1);
        bytes
    }

    fn sequenced(body: &[u8]) -> SequencedArchiveFrameV1 {
        let decoder = PrometheusAttemptDecoder::new(
            Arc::new(StrictExpositionParser),
            Arc::new(NoopNativeEntityDecoder) as Arc<dyn NativeEntityDecoder<()>>,
        );
        let decoded = decoder.decode(
            FetchedAttempt {
                source_id: "node-a".to_owned(),
                source_record_seq: 0,
                request_attempt_seq: Some(0),
                scheduled_ns: Some(10),
                request_start_ns: Some(11),
                first_byte_ns: Some(12),
                capture_ns: Some(13),
                latency_ns: Some(2),
                disposition: FetchDisposition::Response {
                    status: 200,
                    content_type: Some("text/plain; version=0.0.4; charset=utf-8".to_owned()),
                    content_encoding: None,
                    encoded_body: Bytes::copy_from_slice(body),
                    decoded_body: Bytes::copy_from_slice(body),
                },
            },
            &DecodeLimits::default(),
        );
        let archive_id = ArchiveId::new(id(1)).unwrap();
        let session_id = SessionId::new(id(2)).unwrap();
        let key = Arc::new(Blake3ArchiveKeyProvider::new("test", [7; 32]).unwrap());
        let mut sequencer = ArchiveFrameSequencerV1::new(
            archive_id,
            session_id,
            Some(EpochAnchor {
                clock_ns: 0,
                unix_epoch_ns: 1_700_000_000_000_000_000,
                capture_uncertainty_ns: 0,
            }),
            key,
            BTreeMap::from([(
                "node-a".to_owned(),
                SourceProjectionPolicyV1 {
                    attributes: BTreeMap::from([("cluster".to_owned(), "lab-a".to_owned())]),
                },
            )]),
        )
        .unwrap();
        sequencer
            .project_attempt(
                decoded,
                ArchiveFrameTimingV1 {
                    parse_done_ns: 14,
                    archive_enqueue_ns: 15,
                },
            )
            .unwrap()
    }

    #[test]
    fn source_frame_round_trips_all_required_projection_evidence() {
        let sequenced = sequenced(
            b"# HELP requests_total served\n# TYPE requests_total counter\nrequests_total{route=\"/v1/chat\"} 9007199254740993\n",
        );
        let archive_id = sequenced.frame.attempt.archive_id;
        let session_id = sequenced.frame.attempt.session_id;
        let codec = SourceFrameCodecV1::new().unwrap();
        let encoded = codec.encode_source_frame(sequenced).unwrap();
        assert_eq!(
            encoded
                .table_projections
                .iter()
                .map(|projection| projection.table)
                .collect::<Vec<_>>(),
            vec![TableId::Attempts, TableId::Families, TableId::Samples]
        );
        let recovered = codec
            .decode_source_frame(archive_id, session_id, encoded.wal_frame.clone())
            .unwrap();
        assert_eq!(
            recovered
                .table_projections
                .iter()
                .map(|projection| {
                    projection
                        .validate(&codec.schemas)
                        .unwrap_or_else(|error| panic!("{error}"))
                })
                .collect::<Vec<ProjectionEvidence>>(),
            encoded
                .wal_frame
                .header()
                .required_projections
                .iter()
                .map(|projection| projection.evidence)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn zero_row_family_and_sample_projections_remain_explicit() {
        let sequenced = sequenced(b"# EOF\n");
        let codec = SourceFrameCodecV1::new().unwrap();
        let encoded = codec.encode_source_frame(sequenced).unwrap();
        assert_eq!(encoded.wal_frame.header().required_projections.len(), 3);
        assert_eq!(
            encoded
                .wal_frame
                .header()
                .required_projections
                .iter()
                .map(|projection| projection.evidence.row_count)
                .collect::<Vec<_>>(),
            vec![1, 0, 0]
        );
    }

    #[test]
    fn payload_tampering_fails_before_projection_acceptance() {
        let sequenced = sequenced(b"# TYPE temperature gauge\ntemperature 42\n");
        let codec = SourceFrameCodecV1::new().unwrap();
        let encoded = codec.encode_source_frame(sequenced).unwrap();
        let mut bytes = encoded.wal_frame.encode().unwrap();
        let index = bytes.len() / 2;
        bytes[index] ^= 0x80;
        assert!(WalFrame::decode(&bytes, crate::wal::DEFAULT_MAX_WAL_FRAME_BYTES).is_err());
    }
}
