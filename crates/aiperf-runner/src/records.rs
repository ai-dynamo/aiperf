// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Legacy-compatible per-request JSONL generated from native metric records.
//!
//! Python's convergence and detailed-aggregation consumers read the
//! `MetricRecordInfo` shape written by
//! `src/aiperf/post_processors/record_export_jsonl_writer.py:17-123`. Rust
//! retains that wire shape, but every value is produced by the same
//! [`aiperf_metrics::MetricsAccumulator`] used for native-v2 aggregation.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use aiperf_metrics::{
    CATALOG, MetricFlags, MetricType, MetricsAccumulator, MetricsConfig, Phase, RecordIngest,
};
use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::{Value, json};
use uuid::Uuid;

/// Identity retained beside a native metric ingestion record.
pub(crate) struct CapturedRecord {
    pub(crate) uuid: Uuid,
    pub(crate) x_correlation_id: String,
    pub(crate) response_text: Option<String>,
    pub(crate) ingest: RecordIngest,
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
    #[serde(rename = "type")]
    error_type: &'static str,
    message: &'static str,
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

/// Write profiling response text and the legacy selected metric values.
///
/// This collapses Python's per-processor fragment/aggregation implementation
/// (`src/aiperf/post_processors/outputs_json_record_processor.py:83-108` and
/// `src/aiperf/exporters/outputs_json_exporter.py:58-100`) into one post-run
/// write because the native runner already owns every finalized record.
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
                response_text: captured.response_text.as_deref(),
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
            schema_version: "1.0",
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
    let error = (record.errored || record.canceled).then_some(RecordError {
        error_type: if record.canceled {
            "NativeRequestCancelled"
        } else {
            "NativeRequestError"
        },
        message: if record.canceled {
            "request was cancelled by benchmark policy"
        } else {
            "request failed in the native transport"
        },
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
        "trace_type": "aiperf-transport",
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
    use aiperf_metrics::{Phase, TokenCounts};

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
            response_text: Some("hello".into()),
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
                    response_text: Some("answer".into()),
                    ingest: profiling,
                },
                CapturedRecord {
                    uuid: Uuid::from_u128(1),
                    x_correlation_id: "session-1".into(),
                    response_text: Some("warmup".into()),
                    ingest: warmup,
                },
            ],
            &MetricsConfig::default(),
        )
        .unwrap();

        let document: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        assert_eq!(document["schema_version"], "1.0");
        assert_eq!(document["data"].as_array().unwrap().len(), 1);
        assert_eq!(document["data"][0]["session_num"], 2);
        assert_eq!(document["data"][0]["turn_index"], 1);
        assert_eq!(document["data"][0]["conversation_id"], "conversation-2");
        assert_eq!(document["data"][0]["response_text"], "answer");
        assert_eq!(document["data"][0]["metrics"]["request_latency"], 10.0);
        assert_eq!(document["data"][0]["metrics"]["output_token_count"], 3.0);
        assert!(
            document["data"][0]["metrics"]
                .get("time_to_first_token")
                .is_none()
        );
    }
}
