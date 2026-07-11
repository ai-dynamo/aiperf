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

fn record_row(captured: &CapturedRecord, config: &MetricsConfig, include_trace: bool) -> RecordRow {
    let record = &captured.ingest;
    let mut accumulator = MetricsAccumulator::with_config(config.clone());
    accumulator.process_record(record);
    let summary = accumulator.summarize();
    let hidden =
        MetricFlags::NO_INDIVIDUAL_RECORDS | MetricFlags::INTERNAL | MetricFlags::EXPERIMENTAL;
    let metrics = CATALOG
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
        .collect();
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
}
