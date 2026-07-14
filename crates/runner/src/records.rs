// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy-compatible per-request JSONL generated from native metric records.
//!
//! Python's convergence and detailed-aggregation consumers read the
//! `MetricRecordInfo` shape. Rust retains that wire shape, but every value is
//! produced by the same [`aiperf::metrics_core::MetricsAccumulator`] used for
//! native-v2 aggregation.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use aiperf::metrics_core::{
    CATALOG, MetricFlags, MetricType, MetricsAccumulator, MetricsConfig, Phase, RecordIngest,
    ReportError,
};
use aiperf::transport_http::models::{
    ErrorKind, RequestRecord, Response, SseFieldName, SseMessage, TextResponse,
};
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::value::RawValue;
use serde_json::{Value, json};
use uuid::Uuid;

/// Identity retained beside a native metric ingestion record.
pub(crate) struct CapturedRecord {
    pub(crate) uuid: Uuid,
    pub(crate) x_correlation_id: String,
    pub(crate) output: CapturedModelOutput,
    pub(crate) raw: Option<CapturedHttpExchange>,
    pub(crate) ingest: RecordIngest,
}

/// Endpoint-normalized text retained for processed-output artifacts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CapturedModelOutput {
    /// User-visible assistant content, excluding provider reasoning.
    pub(crate) response_text: Option<String>,
    /// Provider-emitted reasoning content, when exposed separately.
    pub(crate) reasoning_text: Option<String>,
}

impl CapturedModelOutput {
    /// Preserve the structured endpoint split, falling back to the legacy flat
    /// text only for backends that do not expose normalized visible content.
    pub(crate) fn from_parts(
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
pub(crate) struct CapturedHttpExchange {
    /// Canonical JSON payload before multipart/media transport preparation.
    pub(crate) request_payload: Vec<u8>,
    /// Exact terminal transport record.
    pub(crate) record: RequestRecord,
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
/// The exact transport [`ErrorDetails`](aiperf::transport_http::models::ErrorDetails)
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
pub(crate) fn group_record_errors(records: &[CapturedRecord]) -> Vec<ReportError> {
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

/// Write finalized request metrics in deterministic arrival order.
pub(crate) fn write_records_jsonl(
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
        let row = record_row(captured, config, include_trace);
        serde_json::to_writer(&mut writer, &row)
            .with_context(|| format!("serializing record export {}", path.display()))?;
        writer
            .write_all(b"\n")
            .with_context(|| format!("writing record export {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing record export {}", path.display()))
}

/// Serialize one terminal record through the canonical compatibility shape.
///
/// Live extension workers consume this exact object while the post-run JSONL
/// writer above consumes the same private row builder. This keeps streaming
/// and persisted records on one Rust-owned metric projection.
pub(crate) fn record_json_value(
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
pub(crate) fn write_raw_records_jsonl(path: &Path, records: &[CapturedRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating raw record directory {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("creating native raw record export {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for captured in records {
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
        serde_json::to_writer(&mut writer, &row)
            .with_context(|| format!("serializing raw record export {}", path.display()))?;
        writer
            .write_all(b"\n")
            .with_context(|| format!("writing raw record export {}", path.display()))?;
    }
    writer
        .flush()
        .with_context(|| format!("flushing raw record export {}", path.display()))
}

/// Write profiling response and reasoning text with selected metric values.
///
/// This collapses Python's per-processor fragment/aggregation implementation
/// into one post-run write because the native runner already owns every
/// finalized record. Schema 1.1 keeps provider reasoning separate from
/// user-visible response text.
pub(crate) fn write_outputs_json(
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
        .filter(|captured| captured.ingest.phase == Phase::Profiling)
        .map(|captured| {
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
            OutputRow {
                session_num: captured.ingest.session_num,
                conversation_id: captured.ingest.conversation_id.as_deref(),
                turn_index: captured.ingest.turn_index,
                x_request_id: captured.uuid.to_string(),
                request_start_ns: captured.ingest.start_ns,
                request_end_ns: captured.ingest.end_ns,
                metrics,
                response_text: captured.output.response_text.as_deref(),
                reasoning_text: captured.output.reasoning_text.as_deref(),
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| (row.session_num, row.turn_index));

    let file = File::create(path)
        .with_context(|| format!("creating native outputs export {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(
        &mut writer,
        &OutputsDocument {
            schema_version: "1.1",
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

fn error_value(error: &aiperf::transport_http::models::ErrorDetails) -> Value {
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

fn record_metrics(
    captured: &CapturedRecord,
    config: &MetricsConfig,
) -> BTreeMap<String, RecordMetric> {
    let record = &captured.ingest;
    let mut accumulator = MetricsAccumulator::with_config(config.clone());
    accumulator.process_record(record);
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
    use aiperf::metrics_core::{Phase, TokenCounts};

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
