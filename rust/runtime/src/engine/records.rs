// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy-compatible per-request JSONL generated from native metric records.
//!
//! Python's convergence and detailed-aggregation consumers read the
//! `MetricRecordInfo` shape. Rust retains that wire shape, but every value is
//! produced by the same [`crate::metrics_core::MetricsAccumulator`] used for
//! native-v2 aggregation.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::export::otel::{OtelRecordAccumulator, classify_spec_error_type};
use crate::metrics_core::{
    CATALOG, MetricFlags, MetricType, MetricsAccumulator, MetricsConfig, Phase, RecordIngest,
    ReportError,
};
use crate::transport_http::models::{
    ErrorKind, RequestRecord, Response, SseFieldName, SseMessage, TextResponse,
};
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::value::RawValue;
use serde_json::{Value, json};
use uuid::Uuid;

/// Identity retained beside a native metric ingestion record.
pub struct CapturedRecord {
    pub uuid: Uuid,
    pub x_correlation_id: String,
    pub output: CapturedModelOutput,
    pub raw: Option<CapturedHttpExchange>,
    pub ingest: RecordIngest,
}

/// Endpoint-normalized text retained for processed-output artifacts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CapturedModelOutput {
    /// User-visible assistant content, excluding provider reasoning.
    pub response_text: Option<String>,
    /// Provider-emitted reasoning content, when exposed separately.
    pub reasoning_text: Option<String>,
}

impl CapturedModelOutput {
    /// Preserve the structured endpoint split, falling back to the legacy flat
    /// text only for backends that do not expose normalized visible content.
    pub fn from_parts(
        flattened_text: &str,
        visible_text: Option<&str>,
        reasoning_text: Option<&str>,
    ) -> Self {
        let response_text = match visible_text {
            Some(text) => non_empty_text(text),
            None => non_empty_text(flattened_text),
        };
        Self {
            response_text,
            reasoning_text: reasoning_text.and_then(non_empty_text),
        }
    }
}

fn non_empty_text(text: &str) -> Option<String> {
    (!text.is_empty()).then(|| text.to_string())
}

/// Exact HTTP facts retained only when Config v2 requests raw artifacts.
pub struct CapturedHttpExchange {
    /// Canonical JSON payload before multipart/media transport preparation.
    pub request_payload: Vec<u8>,
    /// Exact terminal transport record.
    pub record: RequestRecord,
}

#[derive(Serialize)]
struct RecordRow {
    metadata: RecordMetadata,
    metrics: BTreeMap<String, RecordMetric>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_data: Option<Value>,
    error: Option<RecordError>,
}

#[derive(Serialize)]
struct RecordMetadata {
    session_num: u64,
    x_request_id: String,
    x_correlation_id: String,
    conversation_id: Option<String>,
    turn_index: u32,
    credit_issued_ns: Option<i64>,
    request_start_ns: i64,
    request_ack_ns: Option<i64>,
    request_end_ns: i64,
    worker_id: &'static str,
    record_processor_id: &'static str,
    benchmark_phase: Phase,
    was_cancelled: bool,
    cancellation_time_ns: Option<i64>,
}

#[derive(Serialize)]
struct RecordMetric {
    value: f64,
    unit: String,
}

#[derive(Serialize)]
struct RecordError {
    /// HTTP or pseudo-status code, e.g. 499 for a post-send cancellation.
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<u16>,
    #[serde(rename = "type")]
    error_type: &'static str,
    message: String,
}

/// Terminal error classification shared by the per-request record row and the
/// run-level error summary so both agree on the code, the stable type, and the
/// message for one failed or cancelled request.
struct ClassifiedRecordError {
    /// HTTP or pseudo-status code; `Some(499)` for post-send cancellation.
    code: Option<u16>,
    /// Stable error type mirrored into the Python `ErrorDetails.type`.
    error_type: &'static str,
    /// Human-readable message.
    message: String,
}

/// Classify a captured record's terminal error.
///
/// The exact transport [`ErrorDetails`](crate::transport_http::models::ErrorDetails)
/// is preferred when raw artifacts retained it, so a real HTTP status and kind
/// survive. Otherwise the code and stable type are derived from the record's
/// terminal disposition: post-send cancellation is HTTP 499
/// `RequestCancellationError`. Records that reached a normal terminal return
/// `None`.
fn classify_record_error(captured: &CapturedRecord) -> Option<ClassifiedRecordError> {
    if let Some(error) = captured
        .raw
        .as_ref()
        .and_then(|raw| raw.record.error.as_ref())
    {
        return Some(ClassifiedRecordError {
            code: error.code,
            error_type: error_kind_type_name(error.kind),
            message: error.message.clone(),
        });
    }
    let record = &captured.ingest;
    if record.canceled {
        return Some(ClassifiedRecordError {
            code: Some(499),
            error_type: "RequestCancellationError",
            message: "request was cancelled by benchmark policy".to_string(),
        });
    }
    if record.errored {
        return Some(ClassifiedRecordError {
            code: None,
            error_type: "NativeRequestError",
            message: "request failed in the native transport".to_string(),
        });
    }
    None
}

/// Stable `ErrorDetails.type` name for a transport [`ErrorKind`].
fn error_kind_type_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Http => "HttpError",
        ErrorKind::Sse => "SSEResponseError",
        ErrorKind::Cancelled => "RequestCancellationError",
        ErrorKind::Connect => "ConnectError",
        ErrorKind::Timeout => "TimeoutError",
        ErrorKind::Other => "TransportError",
    }
}

/// Group profiling-phase terminal errors into the run-level report summary,
/// keyed by `(code, stable type, message)`.
///
/// This preserves the HTTP 499 post-send cancellation code and its
/// `RequestCancellationError` type in the aggregated `error_summary`, which the
/// Python native-report projection reads from the report `errors` array. Only
/// profiling-phase records are grouped so the summary matches the profiling
/// `error_request_count`.
pub fn group_record_errors(records: &[CapturedRecord]) -> Vec<ReportError> {
    let mut grouped: BTreeMap<(Option<u16>, &'static str, String), usize> = BTreeMap::new();
    for captured in records
        .iter()
        .filter(|captured| captured.ingest.phase == Phase::Profiling)
    {
        if let Some(classified) = classify_record_error(captured) {
            *grouped
                .entry((classified.code, classified.error_type, classified.message))
                .or_insert(0) += 1;
        }
    }
    grouped
        .into_iter()
        .map(|((code, error_type, message), count)| ReportError {
            code,
            error_type: error_type.to_string(),
            message,
            count,
        })
        .collect()
}

#[derive(Serialize)]
struct RawRecordRow<'a> {
    metadata: RawRecordMetadata,
    start_perf_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<&'a RawValue>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_headers: Option<BTreeMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    status: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_headers: Option<&'a BTreeMap<String, String>>,
    responses: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Value>,
}

#[derive(Serialize)]
struct RawRecordMetadata {
    session_num: u64,
    x_request_id: String,
    x_correlation_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    conversation_id: Option<String>,
    turn_index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    credit_issued_ns: Option<i64>,
    request_start_ns: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_ack_ns: Option<i64>,
    request_end_ns: i64,
    worker_id: &'static str,
    record_processor_id: &'static str,
    benchmark_phase: Phase,
    was_cancelled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    cancellation_time_ns: Option<i64>,
    agent_depth: u32,
}

#[derive(Serialize)]
struct OutputsDocument<'a> {
    schema_version: &'static str,
    data: Vec<OutputRow<'a>>,
}

#[derive(Serialize)]
struct OutputRow<'a> {
    session_num: u64,
    conversation_id: Option<&'a str>,
    turn_index: u32,
    x_request_id: String,
    request_start_ns: i64,
    request_end_ns: i64,
    metrics: BTreeMap<&'static str, f64>,
    response_text: Option<&'a str>,
    reasoning_text: Option<&'a str>,
}

const OUTPUT_METRICS: &[&str] = &[
    "output_token_count",
    "output_sequence_length",
    "request_latency",
];

/// Serialize one record row (compact JSON + trailing newline) into `writer`,
/// exactly as [`write_records_jsonl`] emits each line. Shared by the batch writer
/// and the streaming [`crate::engine::record_lane::RecordArtifactLane`] so both are
/// byte-identical for the same record.
pub(crate) fn write_record_jsonl_row(
    writer: &mut dyn Write,
    captured: &CapturedRecord,
    config: &MetricsConfig,
    include_trace: bool,
) -> Result<()> {
    let row = record_row(captured, config, include_trace);
    serde_json::to_writer(&mut *writer, &row).context("serializing record export row")?;
    writer
        .write_all(b"\n")
        .context("writing record export row newline")
}

/// Write finalized request metrics in deterministic arrival order.
pub fn write_records_jsonl(
    path: &Path,
    records: &[CapturedRecord],
    config: &MetricsConfig,
    include_trace: bool,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating record export directory {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("creating native record export {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for captured in records {
        write_record_jsonl_row(&mut writer, captured, config, include_trace)
            .with_context(|| format!("writing record export {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing record export {}", path.display()))
}

/// Write finalized request metrics as a wide, columnar Parquet sidecar beside
/// the per-request JSONL.
///
/// This is a faithful columnar mirror of [`write_records_jsonl`]: it reuses the
/// exact same [`record_metrics`] projection and [`classify_record_error`]
/// classification (no logic is duplicated), so the Parquet metric columns and
/// error triple agree with the JSONL row for the same record. The columnar
/// assembly lives in [`crate::export::per_record_parquet`] because the runner has
/// no direct `arrow`/`parquet` dependency. `include_trace` mirrors the JSONL's
/// conditional `trace_data`, appending flat `trace_*` columns.
#[cfg(feature = "parquet")]
pub(crate) fn write_records_parquet(
    path: &Path,
    records: &[CapturedRecord],
    config: &MetricsConfig,
    include_trace: bool,
) -> Result<()> {
    use crate::export::per_record_parquet::{record_metric_columns, write_per_record_parquet};

    let columns = record_metric_columns();
    let rows: Vec<_> = records
        .iter()
        .map(|captured| per_record_parquet_row(captured, config, include_trace))
        .collect();
    write_per_record_parquet(path, &rows, &columns, include_trace)
        .with_context(|| format!("writing per-record parquet export {}", path.display()))
}

/// Serialized [`Phase`] discriminant matching the JSONL `benchmark_phase` field
/// (serde `snake_case`). Shared by the Parquet and CSV per-record writers.
fn phase_str(phase: Phase) -> &'static str {
    match phase {
        Phase::Warmup => "warmup",
        Phase::Profiling => "profiling",
    }
}

/// Map one [`CapturedRecord`] into the crate-neutral wide Parquet row.
///
/// Shared by the batch [`write_records_parquet`] and the streaming
/// [`crate::engine::record_lane::RecordArtifactLane`] so both produce the identical wide
/// row for the same record.
#[cfg(feature = "parquet")]
pub(crate) fn per_record_parquet_row(
    captured: &CapturedRecord,
    config: &MetricsConfig,
    include_trace: bool,
) -> crate::export::per_record_parquet::PerRecordRow {
    use crate::export::per_record_parquet::{PerRecordRow, PerRecordTrace};

    let record = &captured.ingest;
    let metrics = record_metrics(captured, config)
        .into_iter()
        .map(|(name, metric)| (name, metric.value))
        .collect();
    let error = classify_record_error(captured);
    let trace = include_trace.then(|| {
        let http = &record.http;
        PerRecordTrace {
            stream_setup_ns: http.stream_setup_ns,
            blocked_ns: http.blocked_ns,
            dns_lookup_ns: http.dns_lookup_ns,
            connecting_ns: http.connecting_ns,
            sending_ns: http.sending_ns,
            waiting_ns: http.waiting_ns,
            receiving_ns: http.receiving_ns,
            duration_ns: http.duration_ns,
            connection_reused: http.connection_reused,
            data_sent_bytes: http.data_sent_bytes.map(|value| value as i64),
            data_received_bytes: http.data_received_bytes.map(|value| value as i64),
            chunks_sent: http.chunks_sent.map(|value| value as i64),
            chunks_received: http.chunks_received.map(|value| value as i64),
        }
    });
    PerRecordRow {
        session_num: record.session_num,
        x_request_id: captured.uuid.to_string(),
        x_correlation_id: captured.x_correlation_id.clone(),
        conversation_id: record.conversation_id.clone(),
        turn_index: record.turn_index,
        credit_issued_ns: record.admit_ns,
        request_start_ns: record.start_ns,
        request_ack_ns: record.first_token_ns,
        request_end_ns: record.end_ns,
        benchmark_phase: phase_str(record.phase),
        was_cancelled: record.canceled,
        cancellation_time_ns: record.canceled.then_some(record.end_ns),
        error_code: error.as_ref().and_then(|classified| classified.code),
        error_type: error.as_ref().map(|classified| classified.error_type),
        error_message: error.map(|classified| classified.message),
        metrics,
        trace,
    }
}

/// Fixed per-record metadata columns, in JSONL/Parquet order. Shared header for
/// the records CSV.
const CSV_METADATA_COLUMNS: &[&str] = &[
    "session_num",
    "x_request_id",
    "x_correlation_id",
    "conversation_id",
    "turn_index",
    "credit_issued_ns",
    "request_start_ns",
    "request_ack_ns",
    "request_end_ns",
    "worker_id",
    "record_processor_id",
    "benchmark_phase",
    "was_cancelled",
    "cancellation_time_ns",
];

/// Flat `trace_*` CSV columns, appended when trace capture is enabled. Names and
/// order match the Parquet sink's trace columns.
const CSV_TRACE_COLUMNS: &[&str] = &[
    "trace_stream_setup_ns",
    "trace_blocked_ns",
    "trace_dns_lookup_ns",
    "trace_connecting_ns",
    "trace_sending_ns",
    "trace_waiting_ns",
    "trace_receiving_ns",
    "trace_duration_ns",
    "trace_data_sent_bytes",
    "trace_data_received_bytes",
    "trace_chunks_sent",
    "trace_chunks_received",
    "trace_connection_reused",
];

/// Escape one CSV field, mirroring the legacy Python
/// `BufferedCsvWriterMixin._escape_csv_value`: quote when the value contains a
/// comma, double-quote, or newline, doubling any embedded quote.
fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

/// Render an optional integer cell: the number, or empty when absent.
fn csv_opt_i64(value: Option<i64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

/// Render an optional unsigned integer cell.
fn csv_opt_u64(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

/// Write finalized request metrics as a per-record CSV.
///
/// Ported from the legacy Python `RecordExportCsvResultsProcessor`
/// (`src/aiperf/post_processors/record_export_csv_results_processor.py`) plus its
/// `BufferedCsvWriterMixin` (`src/aiperf/common/mixins/buffered_csv_writer_mixin.py`)
/// onto the native post-run path. The runner already holds every finalized
/// record, so the Python streaming/column-evolution/temp-file machinery collapses
/// into one deterministic write: metadata columns, then one column per catalog
/// record-metric (in catalog order, empty when a request lacks the metric), then
/// the error columns. Reuses the exact [`record_metrics`] projection and
/// [`classify_record_error`] classification the JSONL/Parquet sinks use, so all
/// three per-record artifacts agree.
///
/// Metric columns follow the summary CSV's convention rather than the legacy
/// per-record CSV's `{tag}_value`/`{tag}_unit` pairs: one column per metric named
/// `{Header} ({unit})` (`RecordMetricColumn::csv_display_name`), so the unit lives
/// in the header exactly like `profile_export_aiperf.csv`.
///
/// Two fields the legacy Python CSV lacked are included for parity with the newer
/// JSONL/Parquet per-record artifacts: the error status `error_code` (e.g. 499 for
/// a post-send cancellation) and, when `include_trace` is set, the flat `trace_*`
/// HTTP-timing columns (the Python CSV dropped trace entirely). A record with no
/// projected metrics and no error is skipped (mirroring the Python
/// `if not display_metrics and not error: return`); an all-skipped run writes no
/// file.
/// Build the records CSV header line (no trailing newline): fixed metadata
/// columns, one column per catalog record-metric (`{Header} ({unit})`), the error
/// triple, then the optional flat `trace_*` columns. Shared by [`write_records_csv`]
/// and the streaming [`crate::engine::record_lane::RecordArtifactLane`] so both emit the
/// exact same header bytes.
pub(crate) fn record_csv_header(include_trace: bool) -> String {
    use crate::metrics_core::record_metric_columns;

    let columns = record_metric_columns();
    let mut header: Vec<String> = CSV_METADATA_COLUMNS
        .iter()
        .map(|name| name.to_string())
        .collect();
    for column in &columns {
        header.push(csv_escape(&column.csv_display_name()));
    }
    header.push("error_code".to_string());
    header.push("error_type".to_string());
    header.push("error_message".to_string());
    if include_trace {
        header.extend(CSV_TRACE_COLUMNS.iter().map(|name| name.to_string()));
    }
    header.join(",")
}

/// Build one records CSV data row (no trailing newline) for `captured`, or `None`
/// when the record has neither a projected metric nor an error and must be skipped
/// (mirroring the Python `if not display_metrics and not error: return`). Shared by
/// [`write_records_csv`] and the streaming lane so both emit identical row bytes and
/// apply the identical skip-empty rule.
pub(crate) fn record_csv_row(
    captured: &CapturedRecord,
    config: &MetricsConfig,
    include_trace: bool,
) -> Option<String> {
    use crate::metrics_core::record_metric_columns;

    let columns = record_metric_columns();
    let metrics = record_metrics(captured, config);
    let error = classify_record_error(captured);
    // Skip records with nothing to report unless they carry an error.
    if metrics.is_empty() && error.is_none() {
        return None;
    }
    let record = &captured.ingest;
    let mut cells: Vec<String> = Vec::with_capacity(
        CSV_METADATA_COLUMNS.len() + columns.len() + 3 + CSV_TRACE_COLUMNS.len(),
    );

    // Metadata (order matches CSV_METADATA_COLUMNS).
    cells.push(record.session_num.to_string());
    cells.push(csv_escape(&captured.uuid.to_string()));
    cells.push(csv_escape(&captured.x_correlation_id));
    cells.push(
        record
            .conversation_id
            .as_deref()
            .map(csv_escape)
            .unwrap_or_default(),
    );
    cells.push(record.turn_index.to_string());
    cells.push(csv_opt_i64(record.admit_ns));
    cells.push(record.start_ns.to_string());
    cells.push(csv_opt_i64(record.first_token_ns));
    cells.push(record.end_ns.to_string());
    cells.push("rust-0".to_string());
    cells.push("aiperf-runner".to_string());
    cells.push(phase_str(record.phase).to_string());
    cells.push(record.canceled.to_string());
    cells.push(csv_opt_i64(record.canceled.then_some(record.end_ns)));

    // Metrics: one value cell per catalog column (unit is carried in the
    // header), empty when the record lacks the metric.
    for column in &columns {
        match metrics.get(&column.tag) {
            Some(metric) => cells.push(metric.value.to_string()),
            None => cells.push(String::new()),
        }
    }

    // Error: code / type / message (empty when the record reached a normal
    // terminal).
    match &error {
        Some(classified) => {
            cells.push(csv_opt_i64(classified.code.map(i64::from)));
            cells.push(classified.error_type.to_string());
            cells.push(csv_escape(&classified.message));
        }
        None => {
            cells.push(String::new());
            cells.push(String::new());
            cells.push(String::new());
        }
    }

    // Trace columns, only when requested (mirrors the JSONL's conditional
    // trace_data and the Parquet trace_* tail).
    if include_trace {
        let http = &record.http;
        cells.push(csv_opt_i64(http.stream_setup_ns));
        cells.push(csv_opt_i64(http.blocked_ns));
        cells.push(csv_opt_i64(http.dns_lookup_ns));
        cells.push(csv_opt_i64(http.connecting_ns));
        cells.push(csv_opt_i64(http.sending_ns));
        cells.push(csv_opt_i64(http.waiting_ns));
        cells.push(csv_opt_i64(http.receiving_ns));
        cells.push(csv_opt_i64(http.duration_ns));
        cells.push(csv_opt_u64(http.data_sent_bytes));
        cells.push(csv_opt_u64(http.data_received_bytes));
        cells.push(csv_opt_u64(http.chunks_sent));
        cells.push(csv_opt_u64(http.chunks_received));
        cells.push(
            http.connection_reused
                .map(|v| v.to_string())
                .unwrap_or_default(),
        );
    }

    Some(cells.join(","))
}

pub(crate) fn write_records_csv(
    path: &Path,
    records: &[CapturedRecord],
    config: &MetricsConfig,
    include_trace: bool,
) -> Result<()> {
    let rows: Vec<String> = records
        .iter()
        .filter_map(|captured| record_csv_row(captured, config, include_trace))
        .collect();

    // No displayable records -> no file (mirrors the Python zero-row cleanup and
    // the Parquet empty-skip).
    if rows.is_empty() {
        return Ok(());
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating records CSV directory {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("creating records CSV export {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    writer
        .write_all(record_csv_header(include_trace).as_bytes())
        .and_then(|()| writer.write_all(b"\n"))
        .with_context(|| format!("writing records CSV header {}", path.display()))?;
    for row in &rows {
        writer
            .write_all(row.as_bytes())
            .and_then(|()| writer.write_all(b"\n"))
            .with_context(|| format!("writing records CSV row {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing records CSV export {}", path.display()))
}

/// Serialize one terminal record through the canonical compatibility shape.
///
/// Live extension workers consume this exact object while the post-run JSONL
/// writer above consumes the same private row builder. This keeps streaming
/// and persisted records on one Rust-owned metric projection.
pub fn record_json_value(
    captured: &CapturedRecord,
    config: &MetricsConfig,
    include_trace: bool,
) -> Result<Value> {
    serde_json::to_value(record_row(captured, config, include_trace))
        .context("serializing live native metric record")
}

/// Write Python-compatible raw request/response records in dispatch order.
///
/// The request payload is serialized through [`RawValue`], which validates the
/// one-time captured JSON while preserving its original bytes verbatim in the
/// enclosing JSONL object. The response side comes from the terminal
/// `aiperf-transport-http` record, so no SSE frame, status, response header, or
/// structured transport error is reconstructed from aggregate metrics.
/// Serialize one raw request/response row (compact JSON + trailing newline) into
/// `writer`, exactly as [`write_raw_records_jsonl`] emits each line. Shared by the
/// batch writer and the streaming
/// [`crate::engine::record_lane::RecordArtifactLane`] so both are
/// byte-identical for the same record. The captured request payload is validated
/// through [`RawValue`] here (preserving its original bytes verbatim), matching the
/// batch writer.
pub(crate) fn write_raw_record_jsonl_row(
    writer: &mut dyn Write,
    captured: &CapturedRecord,
) -> Result<()> {
    let payload = captured
        .raw
        .as_ref()
        .map(|raw| serde_json::from_slice::<Box<RawValue>>(&raw.request_payload))
        .transpose()
        .with_context(|| {
            format!(
                "validating captured request payload for raw record {}",
                captured.uuid
            )
        })?;
    let row = raw_record_row(captured, payload.as_deref());
    serde_json::to_writer(&mut *writer, &row).context("serializing raw record export row")?;
    writer
        .write_all(b"\n")
        .context("writing raw record export row newline")
}

pub fn write_raw_records_jsonl(path: &Path, records: &[CapturedRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating raw record directory {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("creating native raw record export {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for captured in records {
        write_raw_record_jsonl_row(&mut writer, captured)
            .with_context(|| format!("writing raw record export {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing raw record export {}", path.display()))
}

/// One dataset session and its per-turn formatted request payloads.
///
/// This is the native-path source for `inputs.json`. The legacy multiprocess
/// `DatasetManager._generate_inputs_json_file` produced the same
/// `{session_id, payloads[]}` shape by formatting every dataset turn through
/// the endpoint; the native runner already builds the exact canonical request
/// body per dispatched turn, so we retain those bytes (deduplicated per
/// `(conversation_id, turn_index)`) and serialize them here without a
/// decode-then-encode round-trip.
pub struct InputSession {
    /// Conversation/session identity — mirrors legacy `SessionPayloads.session_id`.
    pub session_id: String,
    /// One canonical request body per turn, ordered by turn index.
    pub payloads: Vec<Box<RawValue>>,
}

#[derive(Serialize)]
struct InputsDocument<'a> {
    data: Vec<InputsSessionRow<'a>>,
}

#[derive(Serialize)]
struct InputsSessionRow<'a> {
    session_id: &'a str,
    payloads: Vec<&'a RawValue>,
}

/// Write the per-session formatted request payloads as `inputs.json`.
///
/// The shape matches the Pydantic `InputsFile` model
/// (`aiperf.common.models.dataset_models`): a top-level `data` array of
/// `{session_id, payloads}` objects, where each payload is the exact JSON body
/// the runner sent for that turn. Consumers (integration harness, GenAI-Perf
/// compatibility) read multimodal presence directly from
/// `payloads[].messages[].content[]`.
pub fn write_inputs_json(path: &Path, sessions: &[InputSession]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating inputs export directory {}", parent.display()))?;
    }
    let document = InputsDocument {
        data: sessions
            .iter()
            .map(|session| InputsSessionRow {
                session_id: &session.session_id,
                payloads: session.payloads.iter().map(Box::as_ref).collect(),
            })
            .collect(),
    };
    let file = File::create(path)
        .with_context(|| format!("creating native inputs export {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, &document)
        .with_context(|| format!("serializing inputs export {}", path.display()))?;
    writer
        .write_all(b"\n")
        .with_context(|| format!("writing inputs export {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing inputs export {}", path.display()))
}

/// Outputs schema version, shared by the batch writer and the streaming lane.
pub(crate) const OUTPUTS_SCHEMA_VERSION: &str = "1.1";

/// The `outputs.json` document prefix up to (and including) the opening `[` of the
/// `data` array, byte-identical to what `serde_json::to_writer_pretty` emits for the
/// enclosing [`OutputsDocument`]. The streaming lane writes this once, then appends
/// pretty entries (see [`outputs_entry_indented`]) and the matching suffix.
pub(crate) const OUTPUTS_PREFIX: &str = "{\n  \"schema_version\": \"1.1\",\n  \"data\": [";

/// Project one captured record into the profiling `outputs.json` row, or `None` for a
/// non-profiling (warmup) record. Shared by the batch [`write_outputs_json`] and the
/// streaming lane so both select the same `OUTPUT_METRICS` and text fields.
fn output_row<'a>(captured: &'a CapturedRecord, config: &MetricsConfig) -> Option<OutputRow<'a>> {
    if captured.ingest.phase != Phase::Profiling {
        return None;
    }
    let metrics = record_metrics(captured, config)
        .into_iter()
        .filter_map(|(name, metric)| {
            OUTPUT_METRICS
                .contains(&name.as_str())
                .then_some((name, metric.value))
        })
        .map(|(name, value)| {
            let name = OUTPUT_METRICS
                .iter()
                .copied()
                .find(|candidate| *candidate == name)
                .expect("output metric names come from the static allowlist");
            (name, value)
        })
        .collect();
    Some(OutputRow {
        session_num: captured.ingest.session_num,
        conversation_id: captured.ingest.conversation_id.as_deref(),
        turn_index: captured.ingest.turn_index,
        x_request_id: captured.uuid.to_string(),
        request_start_ns: captured.ingest.start_ns,
        request_end_ns: captured.ingest.end_ns,
        metrics,
        response_text: captured.output.response_text.as_deref(),
        reasoning_text: captured.output.reasoning_text.as_deref(),
    })
}

/// Serialize one profiling record's `outputs.json` entry indented to sit inside the
/// `data` array, byte-for-byte as `serde_json::to_writer_pretty` would render that
/// element within the enclosing [`OutputsDocument`] (every line shifted right by four
/// spaces: two for the object nesting, two for the array nesting). Returns `None` for
/// a non-profiling record, which the outputs stream skips exactly as [`write_outputs_json`]
/// filters warmup rows. Sharing [`output_row`] with the batch writer keeps a single
/// pretty entry byte-identical, so a set-comparison of the two documents (sorted by
/// `(session_num, turn_index)`) is exact.
pub(crate) fn outputs_entry_indented(
    captured: &CapturedRecord,
    config: &MetricsConfig,
) -> Result<Option<String>> {
    let Some(row) = output_row(captured, config) else {
        return Ok(None);
    };
    let pretty = serde_json::to_string_pretty(&row).context("serializing outputs export entry")?;
    let mut indented = String::with_capacity(pretty.len() + pretty.lines().count() * 4);
    for (index, line) in pretty.lines().enumerate() {
        if index > 0 {
            indented.push('\n');
        }
        indented.push_str("    ");
        indented.push_str(line);
    }
    Ok(Some(indented))
}

/// Write profiling response and reasoning text with selected metric values.
///
/// This collapses Python's per-processor fragment/aggregation implementation
/// into one post-run write because the native runner already owns every
/// finalized record. Schema 1.1 keeps provider reasoning separate from
/// user-visible response text.
pub fn write_outputs_json(
    path: &Path,
    records: &[CapturedRecord],
    config: &MetricsConfig,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating outputs export directory {}", parent.display()))?;
    }
    let mut rows = records
        .iter()
        .filter_map(|captured| output_row(captured, config))
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| (row.session_num, row.turn_index));

    let file = File::create(path)
        .with_context(|| format!("creating native outputs export {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(
        &mut writer,
        &OutputsDocument {
            schema_version: OUTPUTS_SCHEMA_VERSION,
            data: rows,
        },
    )
    .with_context(|| format!("serializing outputs export {}", path.display()))?;
    writer
        .flush()
        .with_context(|| format!("flushing outputs export {}", path.display()))
}

fn record_row(captured: &CapturedRecord, config: &MetricsConfig, include_trace: bool) -> RecordRow {
    let record = &captured.ingest;
    let metrics = record_metrics(captured, config);
    let error = classify_record_error(captured).map(|classified| RecordError {
        code: classified.code,
        error_type: classified.error_type,
        message: classified.message,
    });
    RecordRow {
        metadata: RecordMetadata {
            session_num: record.session_num,
            x_request_id: captured.uuid.to_string(),
            x_correlation_id: captured.x_correlation_id.clone(),
            conversation_id: record.conversation_id.clone(),
            turn_index: record.turn_index,
            credit_issued_ns: record.admit_ns,
            request_start_ns: record.start_ns,
            request_ack_ns: record.first_token_ns,
            request_end_ns: record.end_ns,
            worker_id: "rust-0",
            record_processor_id: "aiperf-runner",
            benchmark_phase: record.phase,
            was_cancelled: record.canceled,
            cancellation_time_ns: record.canceled.then_some(record.end_ns),
        },
        metrics,
        trace_data: include_trace.then(|| trace_value(record)),
        error,
    }
}

fn raw_record_row<'a>(
    captured: &'a CapturedRecord,
    payload: Option<&'a RawValue>,
) -> RawRecordRow<'a> {
    let ingest = &captured.ingest;
    let raw = captured.raw.as_ref();
    let request_headers = raw.map(|raw| redact_headers(&raw.record.request_headers));
    let status = raw.and_then(|raw| raw.record.status);
    let response_headers = raw
        .map(|raw| &raw.record.response_headers)
        .filter(|headers| !headers.is_empty());
    let responses = raw
        .map(|raw| raw.record.responses.iter().map(response_value).collect())
        .unwrap_or_default();
    let error = raw
        .and_then(|raw| raw.record.error.as_ref().map(error_value))
        .or_else(|| native_error_value(ingest));
    RawRecordRow {
        metadata: RawRecordMetadata {
            session_num: ingest.session_num,
            x_request_id: captured.uuid.to_string(),
            x_correlation_id: captured.x_correlation_id.clone(),
            conversation_id: ingest.conversation_id.clone(),
            turn_index: ingest.turn_index,
            credit_issued_ns: ingest.admit_ns,
            request_start_ns: ingest.start_ns,
            request_ack_ns: ingest.first_token_ns,
            request_end_ns: ingest.end_ns,
            worker_id: "rust-0",
            record_processor_id: "aiperf-runner",
            benchmark_phase: ingest.phase,
            was_cancelled: ingest.canceled,
            cancellation_time_ns: ingest.canceled.then_some(ingest.end_ns),
            agent_depth: 0,
        },
        start_perf_ns: raw.map_or(ingest.start_ns, |raw| raw.record.start_ns),
        payload,
        request_headers,
        status,
        response_headers,
        responses,
        error,
    }
}

fn redact_headers(headers: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    const SENSITIVE: &[&str] = &[
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "api-key",
        "ocp-apim-subscription-key",
        "x-goog-api-key",
        "x-functions-key",
        "aeg-sas-key",
        "x-amz-security-token",
    ];
    headers
        .iter()
        .map(|(name, value)| {
            let value = if SENSITIVE
                .iter()
                .any(|sensitive| name.eq_ignore_ascii_case(sensitive))
            {
                "<redacted>".to_string()
            } else {
                value.clone()
            };
            (name.clone(), value)
        })
        .collect()
}

fn response_value(response: &Response) -> Value {
    match response {
        Response::Sse(message) => sse_response_value(message),
        Response::Text(response) => text_response_value(response),
    }
}

fn sse_response_value(message: &SseMessage) -> Value {
    let packets = message
        .packets
        .iter()
        .map(|packet| {
            json!({
                "name": sse_field_name(&packet.name),
                "value": packet.value,
            })
        })
        .collect::<Vec<_>>();
    json!({"perf_ns": message.perf_ns, "packets": packets})
}

fn sse_field_name(name: &SseFieldName) -> &str {
    match name {
        SseFieldName::Data => "data",
        SseFieldName::Event => "event",
        SseFieldName::Id => "id",
        SseFieldName::Retry => "retry",
        SseFieldName::Comment => "comment",
        SseFieldName::Other(name) => name,
    }
}

fn text_response_value(response: &TextResponse) -> Value {
    json!({
        "perf_ns": response.perf_ns,
        "text": response.text,
        "content_type": response.content_type,
    })
}

fn error_value(error: &crate::transport_http::models::ErrorDetails) -> Value {
    json!({
        "code": error.code,
        "type": error_kind_type_name(error.kind),
        "message": error.message,
    })
}

fn native_error_value(record: &RecordIngest) -> Option<Value> {
    (record.errored || record.canceled).then(|| {
        json!({
            "code": if record.canceled { Some(499) } else { None },
            "type": if record.canceled {
                "NativeRequestCancelled"
            } else {
                "NativeRequestError"
            },
            "message": if record.canceled {
                "request was cancelled by benchmark policy"
            } else {
                "request failed before the native transport produced a record"
            },
        })
    })
}

/// Feed one record's projected per-request metrics into the per-record OTLP
/// histogram accumulator (the native analogue of Python's
/// `MetricResultsStrategy.process`).
///
/// The projection is the exact same [`record_metrics`] shape the live-streaming
/// sink forwards to the Python OTel processor, so the bucketed distribution
/// matches what a collector aggregating Python's per-record stream would
/// compute. The record's terminal error (if any) is classified into the spec
/// `error.type` attribute; successful records contribute no `error.type` and
/// only successful records carry the semconv-mapped metrics, so errored requests
/// never reach a mapped histogram.
pub fn observe_otel_record(
    accumulator: &mut OtelRecordAccumulator,
    captured: &CapturedRecord,
    config: &MetricsConfig,
) {
    let projected = record_metrics(captured, config);
    let lookup: BTreeMap<&str, (f64, &str)> = projected
        .iter()
        .map(|(name, metric)| (name.as_str(), (metric.value, metric.unit.as_str())))
        .collect();
    let error_type = classify_record_error(captured).map(|classified| {
        classify_spec_error_type(classified.code, classified.error_type, &classified.message)
    });
    accumulator.observe_record(&lookup, error_type.as_deref());
}

fn record_metrics(
    captured: &CapturedRecord,
    config: &MetricsConfig,
) -> BTreeMap<String, RecordMetric> {
    // Process into a throwaway single-row store, not at the record's absolute
    // `request_index`. Inserting at `request_index` made this store `request_index`
    // rows wide, so the per-record `summarize()` scanned O(request_index) rows for
    // every metric — O(N^2) across the run and the dominant export cost. Row 0
    // yields byte-identical results (only the one occupied row is ever selected).
    let mut ingest = captured.ingest.clone();
    ingest.request_index = None;
    let mut accumulator = MetricsAccumulator::with_config(config.clone());
    accumulator.process_record(&ingest);
    let summary = accumulator.summarize();
    let hidden =
        MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::INTERNAL | MetricFlags::EXPERIMENTAL;
    CATALOG
        .iter()
        .filter(|spec| spec.kind == MetricType::Record && !spec.flags.intersects(hidden))
        .filter_map(|spec| {
            let result = summary.result(spec.tag)?;
            Some((
                spec.tag.as_str().to_string(),
                RecordMetric {
                    value: result.finite_value()?,
                    unit: result.unit.clone(),
                },
            ))
        })
        .collect()
}

fn trace_value(record: &RecordIngest) -> Value {
    json!({
        "trace_type": "aiperf-transport-http",
        "stream_setup_ns": record.http.stream_setup_ns,
        "blocked_ns": record.http.blocked_ns,
        "dns_lookup_ns": record.http.dns_lookup_ns,
        "connecting_ns": record.http.connecting_ns,
        "sending_ns": record.http.sending_ns,
        "waiting_ns": record.http.waiting_ns,
        "receiving_ns": record.http.receiving_ns,
        "duration_ns": record.http.duration_ns,
        "connection_reused": record.http.connection_reused,
        "data_sent_bytes": record.http.data_sent_bytes,
        "data_received_bytes": record.http.data_received_bytes,
        "chunks_sent": record.http.chunks_sent,
        "chunks_received": record.http.chunks_received,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics_core::{Phase, TokenCounts};

    #[test]
    fn captured_output_prefers_structured_visible_and_reasoning_text() {
        let split = CapturedModelOutput::from_parts("whyanswer", Some("answer"), Some("why"));
        assert_eq!(split.response_text.as_deref(), Some("answer"));
        assert_eq!(split.reasoning_text.as_deref(), Some("why"));

        let reasoning_only = CapturedModelOutput::from_parts("why", Some(""), Some("why"));
        assert_eq!(reasoning_only.response_text, None);
        assert_eq!(reasoning_only.reasoning_text.as_deref(), Some("why"));

        let legacy = CapturedModelOutput::from_parts("answer", None, None);
        assert_eq!(legacy.response_text.as_deref(), Some("answer"));
        assert_eq!(legacy.reasoning_text, None);
    }

    #[test]
    fn jsonl_uses_native_record_metrics_and_legacy_metadata_shape() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile_export.jsonl");
        let mut ingest = RecordIngest::minimal(1_000_000, 11_000_000, Phase::Profiling);
        ingest.first_token_ns = Some(6_000_000);
        ingest.token_arrival_ns = vec![6_000_000, 8_000_000, 11_000_000];
        ingest.tokens = TokenCounts {
            input: Some(8),
            output: Some(3),
            requested_output: Some(3),
            ..TokenCounts::default()
        };
        let captured = CapturedRecord {
            uuid: Uuid::from_u128(7),
            x_correlation_id: "session-7".into(),
            output: CapturedModelOutput::from_parts("hello", None, None),
            raw: None,
            ingest,
        };

        write_records_jsonl(&path, &[captured], &MetricsConfig::default(), false).unwrap();

        let row: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(row["metadata"]["benchmark_phase"], "profiling");
        assert_eq!(row["metadata"]["x_correlation_id"], "session-7");
        assert_eq!(row["metrics"]["request_latency"]["value"], 10.0);
        assert_eq!(row["metrics"]["time_to_first_token"]["value"], 5.0);
        assert_eq!(row["metrics"]["inter_token_latency"]["value"], 2.5);
        assert!(row.get("trace_data").is_none());
        assert!(row["error"].is_null());
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn parquet_sidecar_mirrors_jsonl_records() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile_export.parquet");

        let mut ok = RecordIngest::minimal(1_000_000, 11_000_000, Phase::Profiling);
        ok.first_token_ns = Some(6_000_000);
        ok.token_arrival_ns = vec![6_000_000, 8_000_000, 11_000_000];
        ok.tokens = TokenCounts {
            input: Some(8),
            output: Some(3),
            requested_output: Some(3),
            ..TokenCounts::default()
        };
        let success = CapturedRecord {
            uuid: Uuid::from_u128(7),
            x_correlation_id: "session-7".into(),
            output: CapturedModelOutput::from_parts("hello", None, None),
            raw: None,
            ingest: ok,
        };

        let mut cancel = RecordIngest::minimal(1_000_000, 5_000_000, Phase::Profiling);
        cancel.canceled = true;
        cancel.tokens.input = Some(128);
        let cancelled = CapturedRecord {
            uuid: Uuid::from_u128(3),
            x_correlation_id: "session-3".into(),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest: cancel,
        };

        // The mapped rows agree with the JSONL projection for the same records.
        let success_row = per_record_parquet_row(&success, &MetricsConfig::default(), false);
        assert_eq!(success_row.benchmark_phase, "profiling");
        assert_eq!(
            success_row.metrics.get("request_latency").copied(),
            Some(10.0)
        );
        assert!(success_row.error_code.is_none());

        let cancelled_row = per_record_parquet_row(&cancelled, &MetricsConfig::default(), false);
        assert!(cancelled_row.was_cancelled);
        assert_eq!(cancelled_row.error_code, Some(499));
        assert_eq!(cancelled_row.error_type, Some("RequestCancellationError"));
        assert!(!cancelled_row.metrics.contains_key("request_latency"));

        write_records_parquet(
            &path,
            &[success, cancelled],
            &MetricsConfig::default(),
            false,
        )
        .unwrap();
        assert!(path.exists());
        assert!(std::fs::metadata(&path).unwrap().len() > 0);

        // Empty records write no file (mirrors the aiperf writer contract).
        let empty = directory.path().join("empty.parquet");
        write_records_parquet(&empty, &[], &MetricsConfig::default(), false).unwrap();
        assert!(!empty.exists());
    }

    #[test]
    fn records_csv_has_metadata_metric_and_error_columns() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile_export_records.csv");

        let mut ok = RecordIngest::minimal(1_000_000, 11_000_000, Phase::Profiling);
        ok.session_num = 7;
        ok.first_token_ns = Some(6_000_000);
        ok.token_arrival_ns = vec![6_000_000, 8_000_000, 11_000_000];
        ok.tokens = TokenCounts {
            input: Some(8),
            output: Some(3),
            requested_output: Some(3),
            ..TokenCounts::default()
        };
        let success = CapturedRecord {
            uuid: Uuid::from_u128(7),
            x_correlation_id: "session-7".into(),
            output: CapturedModelOutput::from_parts("hello", None, None),
            raw: None,
            ingest: ok,
        };

        let mut cancel = RecordIngest::minimal(1_000_000, 5_000_000, Phase::Profiling);
        cancel.session_num = 3;
        cancel.canceled = true;
        cancel.tokens.input = Some(128);
        let cancelled = CapturedRecord {
            uuid: Uuid::from_u128(3),
            x_correlation_id: "session-3".into(),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest: cancel,
        };

        write_records_csv(
            &path,
            &[success, cancelled],
            &MetricsConfig::default(),
            false,
        )
        .unwrap();

        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let header: Vec<&str> = lines.next().unwrap().split(',').collect();
        // Metadata + one column per metric (unit in the header, summary-CSV style)
        // + error columns are all present.
        for column in [
            "session_num",
            "x_request_id",
            "benchmark_phase",
            "was_cancelled",
            "Request Latency (ms)",
            "error_code",
            "error_type",
            "error_message",
        ] {
            assert!(
                header.contains(&column),
                "header missing {column}: {header:?}"
            );
        }
        // Units live in the header, not in a separate per-metric unit column.
        assert!(!header.iter().any(|c| c.ends_with("_unit")));
        // No trace columns when trace is disabled.
        assert!(!header.iter().any(|c| c.starts_with("trace_")));

        let idx = |name: &str| header.iter().position(|c| *c == name).unwrap();
        let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
        assert_eq!(rows.len(), 2);

        let ok_row = &rows[0];
        assert_eq!(ok_row[idx("session_num")], "7");
        assert_eq!(ok_row[idx("benchmark_phase")], "profiling");
        assert_eq!(ok_row[idx("was_cancelled")], "false");
        assert_eq!(ok_row[idx("Request Latency (ms)")], "10");
        assert_eq!(ok_row[idx("error_code")], "");
        assert_eq!(ok_row[idx("error_type")], "");

        let cancel_row = &rows[1];
        assert_eq!(cancel_row[idx("session_num")], "3");
        assert_eq!(cancel_row[idx("was_cancelled")], "true");
        assert_eq!(cancel_row[idx("error_code")], "499");
        assert_eq!(cancel_row[idx("error_type")], "RequestCancellationError");
        // The cancelled request produced no latency -> empty metric cell.
        assert_eq!(cancel_row[idx("Request Latency (ms)")], "");

        // Empty input writes no file.
        let empty = directory.path().join("empty_records.csv");
        write_records_csv(&empty, &[], &MetricsConfig::default(), false).unwrap();
        assert!(!empty.exists());
    }

    #[test]
    fn cancelled_record_projects_http_499_and_cancellation_type() {
        let mut ingest = RecordIngest::minimal(1_000_000, 5_000_000, Phase::Profiling);
        ingest.canceled = true;
        ingest.tokens.input = Some(128);
        let captured = CapturedRecord {
            uuid: Uuid::from_u128(3),
            x_correlation_id: "session-3".into(),
            output: CapturedModelOutput::default(),
            raw: None,
            ingest,
        };

        let row = record_row(&captured, &MetricsConfig::default(), false);
        let value = serde_json::to_value(&row).unwrap();
        assert_eq!(value["metadata"]["was_cancelled"], true);
        assert_eq!(value["error"]["code"], 499);
        assert_eq!(value["error"]["type"], "RequestCancellationError");
        // error_isl is still computed for the cancelled request's input tokens.
        assert!(value["metrics"]["error_isl"]["value"].as_f64().unwrap() > 0.0);

        let errors = group_record_errors(std::slice::from_ref(&captured));
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code, Some(499));
        assert_eq!(errors[0].error_type, "RequestCancellationError");
        assert_eq!(errors[0].count, 1);
    }

    #[test]
    fn outputs_json_is_profiling_only_sorted_and_uses_selected_native_metrics() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("outputs.json");
        let mut profiling = RecordIngest::minimal(2_000_000, 12_000_000, Phase::Profiling);
        profiling.session_num = 2;
        profiling.turn_index = 1;
        profiling.conversation_id = Some("conversation-2".into());
        profiling.tokens.output = Some(3);
        profiling.tokens.requested_output = Some(3);
        let mut warmup = RecordIngest::minimal(1_000_000, 3_000_000, Phase::Warmup);
        warmup.session_num = 1;

        write_outputs_json(
            &path,
            &[
                CapturedRecord {
                    uuid: Uuid::from_u128(2),
                    x_correlation_id: "session-2".into(),
                    output: CapturedModelOutput::from_parts(
                        "whyanswer",
                        Some("answer"),
                        Some("why"),
                    ),
                    raw: None,
                    ingest: profiling,
                },
                CapturedRecord {
                    uuid: Uuid::from_u128(1),
                    x_correlation_id: "session-1".into(),
                    output: CapturedModelOutput::from_parts("warmup", None, None),
                    raw: None,
                    ingest: warmup,
                },
            ],
            &MetricsConfig::default(),
        )
        .unwrap();

        let document: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(document["schema_version"], "1.1");
        assert_eq!(document["data"].as_array().unwrap().len(), 1);
        assert_eq!(document["data"][0]["session_num"], 2);
        assert_eq!(document["data"][0]["turn_index"], 1);
        assert_eq!(document["data"][0]["conversation_id"], "conversation-2");
        assert_eq!(document["data"][0]["response_text"], "answer");
        assert_eq!(document["data"][0]["reasoning_text"], "why");
        assert_eq!(document["data"][0]["metrics"]["request_latency"], 10.0);
        assert_eq!(document["data"][0]["metrics"]["output_token_count"], 3.0);
        assert!(
            document["data"][0]["metrics"]
                .get("time_to_first_token")
                .is_none()
        );
    }

    #[test]
    fn raw_jsonl_preserves_payload_frames_headers_and_redaction() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("profile_export_raw.jsonl");
        let payload = br#"{"model":"m",  "messages": []}"#.to_vec();
        let transport_record = RequestRecord {
            start_ns: 2_000_000,
            end_ns: Some(12_000_000),
            request_body: payload.clone().into(),
            request_headers: BTreeMap::from([
                ("Authorization".into(), "Bearer super-secret".into()),
                ("X-Custom-Tracking".into(), "trace-123".into()),
            ]),
            status: Some(200),
            response_headers: BTreeMap::from([("content-type".into(), "text/event-stream".into())]),
            responses: vec![
                Response::Sse(SseMessage::parse(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}",
                    6_000_000,
                )),
                Response::Sse(SseMessage::parse("data: [DONE]", 12_000_000)),
            ],
            ..RequestRecord::default()
        };
        let captured = CapturedRecord {
            uuid: Uuid::from_u128(9),
            x_correlation_id: "session-9".into(),
            output: CapturedModelOutput::from_parts("hi", None, None),
            raw: Some(CapturedHttpExchange {
                request_payload: payload.clone(),
                record: transport_record,
            }),
            ingest: RecordIngest::minimal(2_000_000, 12_000_000, Phase::Profiling),
        };

        write_raw_records_jsonl(&path, &[captured]).unwrap();

        let bytes = std::fs::read(path).unwrap();
        assert!(bytes.windows(payload.len()).any(|window| window == payload));
        assert!(
            !bytes
                .windows(b"super-secret".len())
                .any(|window| window == b"super-secret")
        );
        let row: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(row["metadata"]["benchmark_phase"], "profiling");
        assert_eq!(row["payload"]["model"], "m");
        assert_eq!(row["request_headers"]["Authorization"], "<redacted>");
        assert_eq!(row["request_headers"]["X-Custom-Tracking"], "trace-123");
        assert_eq!(row["response_headers"]["content-type"], "text/event-stream");
        assert_eq!(row["responses"].as_array().unwrap().len(), 2);
        assert_eq!(row["responses"][0]["packets"][0]["name"], "data");
        assert!(row.get("error").is_none());
    }
}
