// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Wide per-request Parquet artifact.
//!
//! The runner writes row-oriented per-request `profile_export.jsonl`, one object
//! per request with the shape `{metadata, metrics{tag:{value,unit}},
//! trace_data?, error}`. That shape is fine for streaming/replay but awkward for
//! analytical queries over millions of records. This module writes the same data
//! columnarly: one Parquet row per request, one nullable
//! `Float64` column per catalog record-metric, so a run can be analyzed
//! column-wise without reshaping the JSONL.
//!
//! # Crate boundary
//! The runner has no direct `arrow`/`parquet` dependency (it only forwards the
//! `parquet` feature to this crate), and the metric
//! [`CATALOG`](crate::metrics_core::CATALOG) lives here, so the
//! columnar assembly lives here. The runner maps each of its `CapturedRecord`s
//! into the crate-neutral
//! [`PerRecordRow`](crate::export::per_record_parquet::PerRecordRow) and calls
//! [`write_per_record_parquet`](crate::export::per_record_parquet::write_per_record_parquet);
//! no metric or error logic is duplicated across the boundary.
//!
//! # Schema
//! Byte identity is not a target. The target is a correct, stable, self-describing
//! schema: fixed metadata head, one metric column per
//! [`record_metric_columns`](crate::metrics_core::record_metric_columns)
//! entry (catalog order, null when the request produced no finite value), fixed
//! error tail, and — only when `include_trace` — flat `trace_*` HTTP-timing
//! columns. Per-metric units are
//! constant, so they live in the `aiperf.units` file metadata rather than a
//! redundant per-row column.
//!
//! Metric columns follow
//! [`record_metric_columns`](crate::metrics_core::record_metric_columns) in
//! catalog order; metadata and trace fields use fixed columns.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use arrow::array::{ArrayRef, BooleanArray, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;

use crate::export::parquet_util::{float_column, string_column, writer_properties};

// The per-record metric column set is shared with the records CSV writer and is
// derived from the catalog, so it lives (ungated) in `metrics_core`. Re-exported
// here as `PerRecordMetricColumn`.
pub use crate::metrics_core::RecordMetricColumn as PerRecordMetricColumn;
pub use crate::metrics_core::record_metric_columns;

/// Schema version stamped into the file's key-value metadata.
const SCHEMA_VERSION: &str = "1.0";

/// Flat HTTP-timing fields shared with `records.rs::trace_value`. Present on a
/// [`PerRecordRow`] only when
/// trace capture is enabled; every field is nullable in the emitted columns.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PerRecordTrace {
    pub stream_setup_ns: Option<i64>,
    pub blocked_ns: Option<i64>,
    pub dns_lookup_ns: Option<i64>,
    pub connecting_ns: Option<i64>,
    pub sending_ns: Option<i64>,
    pub waiting_ns: Option<i64>,
    pub receiving_ns: Option<i64>,
    pub duration_ns: Option<i64>,
    pub connection_reused: Option<bool>,
    pub data_sent_bytes: Option<i64>,
    pub data_received_bytes: Option<i64>,
    pub chunks_sent: Option<i64>,
    pub chunks_received: Option<i64>,
}

/// One request's wide row prior to columnarization.
///
/// Crate-neutral: the runner fills this from its `CapturedRecord` (metadata +
/// projected metrics + classified error), and this module owns the arrow/parquet
/// assembly. `metrics` maps a catalog metric tag to its finite value; a tag absent
/// from the map becomes a null cell in that metric's column, matching the JSONL's
/// metric-absence semantics.
#[derive(Debug, Clone, Default)]
pub struct PerRecordRow {
    pub session_num: u64,
    pub x_request_id: String,
    pub x_correlation_id: String,
    pub conversation_id: Option<String>,
    pub turn_index: u32,
    pub credit_issued_ns: Option<i64>,
    pub request_start_ns: i64,
    pub request_ack_ns: Option<i64>,
    pub request_end_ns: i64,
    /// Identity of the worker that executed the request, matching the JSONL
    /// `metadata.worker_id`.
    pub worker_id: String,
    /// Dense global dispatch ordinal, matching the JSONL
    /// `metadata.global_dispatch_index`. `None` for a record the workload assigned
    /// no ordinal, which becomes a null cell rather than a fabricated position.
    pub global_dispatch_index: Option<i64>,
    /// Serialized [`crate::metrics_core::Phase`] (`"warmup"` / `"profiling"`).
    pub benchmark_phase: &'static str,
    pub was_cancelled: bool,
    pub cancellation_time_ns: Option<i64>,
    /// HTTP or pseudo-status code; `Some(499)` for post-send cancellation.
    pub error_code: Option<u16>,
    /// Stable error type shared with the JSONL row.
    pub error_type: Option<&'static str>,
    pub error_message: Option<String>,
    /// Metric tag → finite value. Missing tag ⇒ null cell in that column.
    pub metrics: BTreeMap<String, f64>,
    /// Flat HTTP timing; `Some` only when trace capture is enabled.
    pub trace: Option<PerRecordTrace>,
}

/// Constant `record_processor_id` column value, matching the JSONL metadata.
const RECORD_PROCESSOR_ID: &str = "aiperf runner";

/// Write the wide per-record Parquet file.
///
/// `columns` is the ordered metric column set (typically
/// [`record_metric_columns`]);
/// `include_trace` toggles the trailing `trace_*` columns (schema-affecting, so it
/// must match how the rows were built). An empty `rows` writes no file. Snappy compression matches
/// the sibling server-metrics sink.
pub fn write_per_record_parquet(
    path: &Path,
    rows: &[PerRecordRow],
    columns: &[PerRecordMetricColumn],
    include_trace: bool,
) -> Result<()> {
    if rows.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!("creating per-record parquet directory {}", parent.display())
        })?;
    }
    let schema = build_schema(columns, include_trace);
    let batch = build_record_batch(&schema, rows, columns, include_trace)?;
    write_parquet(path, schema, &batch)
        .with_context(|| format!("writing per-record parquet {}", path.display()))
}

/// Build the arrow schema: fixed metadata head, metric columns, error tail, and
/// (when requested) the trace tail. All metric/error/trace/optional columns are
/// nullable. The `aiperf.units` and `aiperf.schema_version` metadata is attached
/// here.
fn build_schema(columns: &[PerRecordMetricColumn], include_trace: bool) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("session_num", DataType::Int64, false),
        Field::new("x_request_id", DataType::Utf8, false),
        Field::new("x_correlation_id", DataType::Utf8, false),
        Field::new("conversation_id", DataType::Utf8, true),
        Field::new("turn_index", DataType::Int64, false),
        Field::new("credit_issued_ns", DataType::Int64, true),
        Field::new("request_start_ns", DataType::Int64, false),
        Field::new("request_ack_ns", DataType::Int64, true),
        Field::new("request_end_ns", DataType::Int64, false),
        Field::new("worker_id", DataType::Utf8, false),
        Field::new("global_dispatch_index", DataType::Int64, true),
        Field::new("record_processor_id", DataType::Utf8, false),
        Field::new("benchmark_phase", DataType::Utf8, false),
        Field::new("was_cancelled", DataType::Boolean, false),
        Field::new("cancellation_time_ns", DataType::Int64, true),
    ];
    for column in columns {
        fields.push(Field::new(&column.tag, DataType::Float64, true));
    }
    fields.extend([
        Field::new("error_code", DataType::Int64, true),
        Field::new("error_type", DataType::Utf8, true),
        Field::new("error_message", DataType::Utf8, true),
    ]);
    if include_trace {
        for name in TRACE_INT_COLUMNS {
            fields.push(Field::new(*name, DataType::Int64, true));
        }
        fields.push(Field::new(
            "trace_connection_reused",
            DataType::Boolean,
            true,
        ));
    }

    Arc::new(Schema::new_with_metadata(fields, build_metadata(columns)))
}

/// Int64 trace columns in emission order. `trace_connection_reused` (Boolean) is
/// appended after these by the schema/batch builders.
const TRACE_INT_COLUMNS: &[&str] = &[
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
];

/// File key-value metadata: schema version, crate version, and the constant
/// per-metric units map (units are per-metric constant, so they belong here rather
/// than in a redundant per-row column).
fn build_metadata(columns: &[PerRecordMetricColumn]) -> HashMap<String, String> {
    let units: BTreeMap<&str, &str> = columns
        .iter()
        .map(|column| (column.tag.as_str(), column.unit.as_str()))
        .collect();
    let mut metadata = HashMap::new();
    metadata.insert(
        "aiperf.schema_version".to_string(),
        SCHEMA_VERSION.to_string(),
    );
    metadata.insert(
        "aiperf.version".to_string(),
        env!("CARGO_PKG_VERSION").to_string(),
    );
    metadata.insert(
        "aiperf.units".to_string(),
        serde_json::to_string(&units).unwrap_or_else(|_| "{}".to_string()),
    );
    metadata
}

/// Columnarize the rows against the schema, in the schema's column order.
fn build_record_batch(
    schema: &Arc<Schema>,
    rows: &[PerRecordRow],
    columns: &[PerRecordMetricColumn],
    include_trace: bool,
) -> Result<RecordBatch> {
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

    arrays.push(int_column(rows.iter().map(|r| Some(r.session_num as i64))));
    arrays.push(string_column(
        rows.iter().map(|r| Some(r.x_request_id.clone())),
    ));
    arrays.push(string_column(
        rows.iter().map(|r| Some(r.x_correlation_id.clone())),
    ));
    arrays.push(string_column(
        rows.iter().map(|r| r.conversation_id.clone()),
    ));
    arrays.push(int_column(rows.iter().map(|r| Some(r.turn_index as i64))));
    arrays.push(int_column(rows.iter().map(|r| r.credit_issued_ns)));
    arrays.push(int_column(rows.iter().map(|r| Some(r.request_start_ns))));
    arrays.push(int_column(rows.iter().map(|r| r.request_ack_ns)));
    arrays.push(int_column(rows.iter().map(|r| Some(r.request_end_ns))));
    arrays.push(string_column(
        rows.iter().map(|r| Some(r.worker_id.clone())),
    ));
    arrays.push(int_column(rows.iter().map(|r| r.global_dispatch_index)));
    arrays.push(string_column(
        rows.iter().map(|_| Some(RECORD_PROCESSOR_ID.to_string())),
    ));
    arrays.push(string_column(
        rows.iter().map(|r| Some(r.benchmark_phase.to_string())),
    ));
    arrays.push(bool_column(rows.iter().map(|r| Some(r.was_cancelled))));
    arrays.push(int_column(rows.iter().map(|r| r.cancellation_time_ns)));

    for column in columns {
        arrays.push(float_column(
            rows.iter().map(|r| r.metrics.get(&column.tag).copied()),
        ));
    }

    arrays.push(int_column(rows.iter().map(|r| r.error_code.map(i64::from))));
    arrays.push(string_column(
        rows.iter().map(|r| r.error_type.map(str::to_string)),
    ));
    arrays.push(string_column(rows.iter().map(|r| r.error_message.clone())));

    if include_trace {
        // Absent trace ⇒ null across every trace column for that row.
        let trace_int = |select: fn(&PerRecordTrace) -> Option<i64>| -> ArrayRef {
            int_column(rows.iter().map(|r| r.trace.as_ref().and_then(select)))
        };
        arrays.push(trace_int(|t| t.stream_setup_ns));
        arrays.push(trace_int(|t| t.blocked_ns));
        arrays.push(trace_int(|t| t.dns_lookup_ns));
        arrays.push(trace_int(|t| t.connecting_ns));
        arrays.push(trace_int(|t| t.sending_ns));
        arrays.push(trace_int(|t| t.waiting_ns));
        arrays.push(trace_int(|t| t.receiving_ns));
        arrays.push(trace_int(|t| t.duration_ns));
        arrays.push(trace_int(|t| t.data_sent_bytes));
        arrays.push(trace_int(|t| t.data_received_bytes));
        arrays.push(trace_int(|t| t.chunks_sent));
        arrays.push(trace_int(|t| t.chunks_received));
        arrays.push(bool_column(
            rows.iter()
                .map(|r| r.trace.as_ref().and_then(|t| t.connection_reused)),
        ));
    }

    RecordBatch::try_new(schema.clone(), arrays)
        .context("assembling per-record parquet record batch")
}

/// Build a nullable Int64 column.
fn int_column<I: Iterator<Item = Option<i64>>>(values: I) -> ArrayRef {
    Arc::new(Int64Array::from_iter(values)) as ArrayRef
}

/// Build a nullable Boolean column.
fn bool_column<I: Iterator<Item = Option<bool>>>(values: I) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(values)) as ArrayRef
}

/// Write the record batch to Parquet with Snappy compression and file-level
/// key-value metadata copied from the schema.
fn write_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) -> Result<()> {
    super::parquet_util::write_parquet_table(path, schema, batch, "per-record parquet")
}

/// Default row-group row bound for the incremental streaming writer. Each buffer
/// of up to this many rows is flushed as one Parquet row group and dropped, so
/// peak writer memory is O(bound) rather than O(records).
pub const DEFAULT_ROW_GROUP_ROWS: usize = 4096;

/// Incremental, bounded-memory sibling of
/// [`write_per_record_parquet`].
///
/// The one-shot
/// [`write_per_record_parquet`]
/// columnarizes ALL rows into one
/// in-memory `RecordBatch`, which requires the caller to retain every record
/// until run end. This writer instead buffers a bounded window of
/// [`PerRecordRow`]s, flushes
/// each full window as one Parquet **row group** via a
/// held-open [`ArrowWriter`], and clears the buffer — so a fold-and-drop caller
/// can push each record at completion and immediately drop it, bounding peak
/// memory at the buffer size.
///
/// The emitted schema, per-metric column set, Snappy codec, and file metadata are
/// byte-for-byte the same builders
/// [`write_per_record_parquet`]
/// uses; only the
/// physical row-group chunking differs (an internal detail with no bearing on the
/// logical row set — see the parity test). Rows are appended in the order they are
/// pushed (completion order for the fold path), which is the accepted decision for
/// streamed per-record artifacts.
///
/// File creation is lazy: an instance that is finished without a single pushed row
/// leaves no file, matching
/// [`write_per_record_parquet`]'s
/// empty-rows contract.
pub struct StreamingPerRecordParquetWriter {
    path: PathBuf,
    schema: Arc<Schema>,
    columns: Vec<PerRecordMetricColumn>,
    include_trace: bool,
    buffer: Vec<PerRecordRow>,
    row_group_rows: usize,
    writer: Option<ArrowWriter<File>>,
}

impl StreamingPerRecordParquetWriter {
    /// Build a streaming writer for `path` over the ordered metric `columns`
    /// (typically
    /// [`record_metric_columns`]);
    /// `include_trace` toggles the trailing
    /// `trace_*` columns (schema-affecting, so it must match how rows are built).
    /// `row_group_rows` bounds each row group; it is clamped to at least 1. No file
    /// is created until the first row is flushed.
    pub fn new(
        path: PathBuf,
        columns: Vec<PerRecordMetricColumn>,
        include_trace: bool,
        row_group_rows: usize,
    ) -> Self {
        let schema = build_schema(&columns, include_trace);
        Self {
            path,
            schema,
            columns,
            include_trace,
            buffer: Vec::new(),
            row_group_rows: row_group_rows.max(1),
            writer: None,
        }
    }

    /// Append one row, flushing a row group once the buffer reaches the bound.
    pub fn push(&mut self, row: PerRecordRow) -> Result<()> {
        self.buffer.push(row);
        if self.buffer.len() >= self.row_group_rows {
            self.flush_row_group()?;
        }
        Ok(())
    }

    /// Flush the buffered rows as one row group and clear the buffer. Creates the
    /// file + [`ArrowWriter`] on the first non-empty flush. A no-op when the buffer
    /// is empty, so it is safe to call on `finish` with nothing pending.
    fn flush_row_group(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        if self.writer.is_none() {
            if let Some(parent) = self.path.parent() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("creating per-record parquet directory {}", parent.display())
                })?;
            }
            let file = File::create(&self.path)
                .with_context(|| format!("creating per-record parquet {}", self.path.display()))?;
            let props = writer_properties(&self.schema);
            self.writer = Some(
                ArrowWriter::try_new(file, self.schema.clone(), Some(props)).with_context(
                    || {
                        format!(
                            "constructing streaming per-record parquet writer {}",
                            self.path.display()
                        )
                    },
                )?,
            );
        }
        let batch = build_record_batch(
            &self.schema,
            &self.buffer,
            &self.columns,
            self.include_trace,
        )?;
        let writer = self
            .writer
            .as_mut()
            .expect("writer created above on first flush");
        writer.write(&batch).with_context(|| {
            format!(
                "writing streaming per-record parquet row group {}",
                self.path.display()
            )
        })?;
        // `ArrowWriter::write` coalesces successive batches into a row group sized by
        // `max_row_group_size` (default ~1M rows), so without an explicit flush every
        // buffered window would land in ONE row group and defeat the bounded-memory
        // goal. Flushing here closes the current row group, so each buffered window is
        // its own group and the encoded column pages for it can be released.
        writer.flush().with_context(|| {
            format!(
                "flushing streaming per-record parquet row group {}",
                self.path.display()
            )
        })?;
        self.buffer.clear();
        Ok(())
    }

    /// Flush the final partial buffer and close the file. A writer that never saw a
    /// row leaves no file (matching the batch writer's empty-rows contract).
    pub fn finish(mut self) -> Result<()> {
        self.flush_row_group()?;
        if let Some(writer) = self.writer.take() {
            writer.close().with_context(|| {
                format!(
                    "finalizing streaming per-record parquet file {}",
                    self.path.display()
                )
            })?;
        }
        Ok(())
    }
}

/// Concatenate several per-shard per-record Parquet files into one combined file.
///
/// Each thread-per-core shard streams its own
/// [`StreamingPerRecordParquetWriter`] to a per-shard temp file, so the coordinator
/// must fuse those into the single final `profile_export.parquet`. Because every
/// shard file was produced by the same `build_schema`/`writer_properties`
/// builders (identical `columns` and `include_trace`), they share one schema and
/// one `aiperf.units`/`aiperf.schema_version`/`aiperf.version` metadata set; this
/// reader-to-writer copy reads each shard's row groups back as [`RecordBatch`]es and
/// re-emits them into a fresh combined file carrying that same schema and file KV
/// metadata. Row order across shards is completion order (the accepted streamed
/// decision) — the logical row set is the union of the shard rows, which is exactly
/// what the batch writer over the union produces.
///
/// Only shard paths that exist are read (a shard that saw no displayable row left no
/// file, matching the empty-rows contract); if NONE exist, no combined file is
/// written (the whole run had zero displayable rows). The combined file's schema and
/// KV metadata are taken from the first existing shard file, so the empty-tail
/// (`ARROW:schema` + `aiperf.*`) round-trips unchanged.
pub fn concat_per_record_parquet(shard_paths: &[PathBuf], final_path: &Path) -> Result<()> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let existing: Vec<&PathBuf> = shard_paths.iter().filter(|path| path.exists()).collect();
    if existing.is_empty() {
        return Ok(());
    }
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "creating combined per-record parquet directory {}",
                parent.display()
            )
        })?;
    }
    let mut writer: Option<ArrowWriter<File>> = None;
    for shard in existing {
        let file = File::open(shard)
            .with_context(|| format!("opening shard per-record parquet {}", shard.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).with_context(|| {
            format!(
                "reading shard per-record parquet metadata {}",
                shard.display()
            )
        })?;
        // Every shard shares one schema + KV metadata (same columns/include_trace);
        // seed the combined writer from the first shard so its `ARROW:schema` and
        // `aiperf.*` file metadata round-trip verbatim.
        if writer.is_none() {
            let schema = builder.schema().clone();
            let out = File::create(final_path).with_context(|| {
                format!(
                    "creating combined per-record parquet {}",
                    final_path.display()
                )
            })?;
            let props = writer_properties(&schema);
            writer = Some(
                ArrowWriter::try_new(out, schema, Some(props)).with_context(|| {
                    format!(
                        "constructing combined per-record parquet writer {}",
                        final_path.display()
                    )
                })?,
            );
        }
        let reader = builder.build().with_context(|| {
            format!(
                "opening shard per-record parquet reader {}",
                shard.display()
            )
        })?;
        let sink = writer.as_mut().expect("writer seeded on first shard");
        for batch in reader {
            let batch = batch.with_context(|| {
                format!("reading shard per-record parquet batch {}", shard.display())
            })?;
            sink.write(&batch).with_context(|| {
                format!(
                    "writing combined per-record parquet batch {}",
                    final_path.display()
                )
            })?;
        }
    }
    writer
        .expect("at least one existing shard seeded the writer")
        .close()
        .with_context(|| {
            format!(
                "finalizing combined per-record parquet {}",
                final_path.display()
            )
        })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    fn read_back(path: &Path) -> (RecordBatch, HashMap<String, String>) {
        let file = File::open(path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let metadata: HashMap<String, String> = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .map(|kv| {
                kv.iter()
                    .filter_map(|entry| entry.value.clone().map(|value| (entry.key.clone(), value)))
                    .collect()
            })
            .unwrap_or_default();
        let mut reader = builder.build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        (batch, metadata)
    }

    fn column<'a, T: 'static>(batch: &'a RecordBatch, name: &str) -> &'a T {
        let idx = batch.schema().index_of(name).unwrap();
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<T>()
            .expect("column type mismatch")
    }

    fn success_row() -> PerRecordRow {
        PerRecordRow {
            session_num: 7,
            x_request_id: "req-7".into(),
            x_correlation_id: "session-7".into(),
            conversation_id: Some("conv-7".into()),
            turn_index: 1,
            request_start_ns: 1_000_000,
            request_end_ns: 11_000_000,
            worker_id: "rust-0".into(),
            global_dispatch_index: Some(7),
            benchmark_phase: "profiling",
            metrics: BTreeMap::from([
                ("request_latency".to_string(), 10.0),
                ("time_to_first_token".to_string(), 5.0),
            ]),
            ..PerRecordRow::default()
        }
    }

    fn cancelled_row() -> PerRecordRow {
        PerRecordRow {
            session_num: 3,
            x_request_id: "req-3".into(),
            x_correlation_id: "session-3".into(),
            turn_index: 0,
            request_start_ns: 1_000_000,
            request_end_ns: 5_000_000,
            // A different worker and no assigned ordinal: both are per-row values,
            // not the file-wide constants this column used to carry.
            worker_id: "rust-3".into(),
            global_dispatch_index: None,
            benchmark_phase: "profiling",
            was_cancelled: true,
            cancellation_time_ns: Some(5_000_000),
            error_code: Some(499),
            error_type: Some("RequestCancellationError"),
            error_message: Some("request was cancelled by benchmark policy".into()),
            // Deliberately missing request_latency to exercise null metric cells.
            metrics: BTreeMap::from([("error_isl".to_string(), 128.0)]),
            ..PerRecordRow::default()
        }
    }

    #[test]
    fn metric_columns_match_the_jsonl_record_filter() {
        let columns = record_metric_columns();
        let tags: Vec<&str> = columns.iter().map(|c| c.tag.as_str()).collect();
        // Common per-record metrics are present; hidden/aggregate ones are not.
        assert!(tags.contains(&"request_latency"));
        assert!(tags.contains(&"time_to_first_token"));
        // request_count is an aggregate (not a Record metric) — never a column.
        assert!(!tags.contains(&"request_count"));
        // Every metric carries a non-empty unit for the aiperf.units metadata.
        assert!(columns.iter().all(|c| !c.unit.is_empty()));
    }

    #[test]
    fn empty_rows_writes_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile_export.parquet");
        write_per_record_parquet(&path, &[], &record_metric_columns(), false).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn wide_schema_values_nulls_and_units_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile_export.parquet");
        let columns = record_metric_columns();
        write_per_record_parquet(&path, &[success_row(), cancelled_row()], &columns, false)
            .unwrap();

        let (batch, metadata) = read_back(&path);
        assert_eq!(batch.num_rows(), 2);

        // Fixed metadata columns.
        let session = column::<Int64Array>(&batch, "session_num");
        assert_eq!(session.value(0), 7);
        assert_eq!(session.value(1), 3);
        let phase = column::<StringArray>(&batch, "benchmark_phase");
        assert_eq!(phase.value(0), "profiling");
        let worker = column::<StringArray>(&batch, "worker_id");
        assert_eq!(worker.value(0), "rust-0");
        assert_eq!(
            worker.value(1),
            "rust-3",
            "worker_id must be the row's executing worker, not a file-wide constant"
        );
        let dispatch_index = column::<Int64Array>(&batch, "global_dispatch_index");
        assert_eq!(dispatch_index.value(0), 7);
        assert!(
            dispatch_index.is_null(1),
            "an unassigned dispatch ordinal must be null, never a fabricated position"
        );
        let conv = column::<StringArray>(&batch, "conversation_id");
        assert!(conv.is_valid(0));
        assert!(conv.is_null(1));
        let cancelled = column::<BooleanArray>(&batch, "was_cancelled");
        assert!(!cancelled.value(0));
        assert!(cancelled.value(1));

        // Metric columns: present ⇒ value, absent ⇒ null.
        let latency = column::<Float64Array>(&batch, "request_latency");
        assert_eq!(latency.value(0), 10.0);
        assert!(latency.is_null(1), "cancelled row has no request_latency");

        // Error tail.
        let code = column::<Int64Array>(&batch, "error_code");
        assert!(code.is_null(0));
        assert_eq!(code.value(1), 499);
        let etype = column::<StringArray>(&batch, "error_type");
        assert!(etype.is_null(0));
        assert_eq!(etype.value(1), "RequestCancellationError");

        // File metadata.
        assert_eq!(metadata.get("aiperf.schema_version").unwrap(), "1.0");
        let units: BTreeMap<String, String> =
            serde_json::from_str(metadata.get("aiperf.units").unwrap()).unwrap();
        assert_eq!(units.get("request_latency").map(String::as_str), Some("ms"));

        // No trace columns when include_trace is false.
        assert!(batch.schema().index_of("trace_duration_ns").is_err());
    }

    /// Read a Parquet file back as a single coalesced `RecordBatch` (batch size >=
    /// row count coalesces across row groups), returning the number of physical row
    /// groups, the one batch, and the file metadata. Used to compare the streaming
    /// writer against the one-shot writer independent of row-group chunking.
    fn read_coalesced(path: &Path) -> (usize, RecordBatch, HashMap<String, String>) {
        let file = File::open(path).unwrap();
        let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let num_row_groups = builder.metadata().num_row_groups();
        let total_rows: usize = builder
            .metadata()
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .sum();
        let metadata: HashMap<String, String> = builder
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .map(|kv| {
                kv.iter()
                    .filter_map(|entry| entry.value.clone().map(|value| (entry.key.clone(), value)))
                    .collect()
            })
            .unwrap_or_default();
        let mut reader = builder.with_batch_size(total_rows.max(1)).build().unwrap();
        let batch = reader.next().unwrap().unwrap();
        assert!(
            reader.next().is_none(),
            "batch_size >= total rows must coalesce to a single batch"
        );
        (num_row_groups, batch, metadata)
    }

    /// A larger deterministic slice: `n` alternating success/cancelled rows with
    /// distinct session numbers, so a small row-group bound forces several groups.
    fn sample_rows(n: usize) -> Vec<PerRecordRow> {
        (0..n)
            .map(|i| {
                let mut row = if i % 2 == 0 {
                    success_row()
                } else {
                    cancelled_row()
                };
                row.session_num = i as u64;
                row.x_request_id = format!("req-{i}");
                row
            })
            .collect()
    }

    /// The streaming writer and the one-shot writer produce the identical schema,
    /// file metadata, and logical row set (not byte identity — the streaming file
    /// has several row groups where the one-shot has one).
    fn assert_streaming_matches_batch(rows: &[PerRecordRow], bound: usize, include_trace: bool) {
        let columns = record_metric_columns();
        let dir = tempfile::tempdir().unwrap();
        let stream_path = dir.path().join("stream.parquet");
        let batch_path = dir.path().join("batch.parquet");

        let mut writer = StreamingPerRecordParquetWriter::new(
            stream_path.clone(),
            columns.clone(),
            include_trace,
            bound,
        );
        for row in rows {
            writer.push(row.clone()).unwrap();
        }
        writer.finish().unwrap();

        write_per_record_parquet(&batch_path, rows, &columns, include_trace).unwrap();

        let (stream_groups, stream_batch, stream_meta) = read_coalesced(&stream_path);
        let (batch_groups, batch_batch, batch_meta) = read_coalesced(&batch_path);

        // Streaming flushes produce ceil(rows / bound) row groups, while the
        // one-shot writer produces one.
        assert_eq!(batch_groups, 1, "one-shot writer emits one row group");
        assert_eq!(
            stream_groups,
            rows.len().div_ceil(bound),
            "streaming writer emits one row group per buffered window"
        );

        // Identical schema (fields + metadata), identical file KV metadata, identical
        // logical rows in identical order.
        assert_eq!(stream_batch.schema(), batch_batch.schema());
        assert_eq!(stream_meta, batch_meta);
        assert_eq!(stream_batch.num_rows(), batch_batch.num_rows());
        for (name, stream_col, batch_col) in stream_batch
            .schema()
            .fields()
            .iter()
            .zip(stream_batch.columns())
            .zip(batch_batch.columns())
            .map(|((field, s), b)| (field.name(), s, b))
        {
            assert_eq!(
                stream_col.to_data(),
                batch_col.to_data(),
                "column {name} differs between streaming and one-shot writers"
            );
        }
    }

    #[test]
    fn streaming_matches_batch_multiple_row_groups() {
        // bound 2 over 5 rows -> 3 row groups (2, 2, 1) exercising a trailing
        // partial flush.
        assert_streaming_matches_batch(&sample_rows(5), 2, false);
        assert_streaming_matches_batch(&sample_rows(5), 2, true);
    }

    #[test]
    fn streaming_matches_batch_single_partial_batch() {
        // Fewer rows than the bound -> one partial row group flushed on finish.
        assert_streaming_matches_batch(&sample_rows(3), 4096, false);
    }

    #[test]
    fn streaming_matches_batch_exact_multiple_of_bound() {
        // rows is an exact multiple of the bound -> every group full, nothing left
        // for the finish flush.
        assert_streaming_matches_batch(&sample_rows(4), 2, false);
    }

    /// Streaming disjoint shard slices through per-shard writers then
    /// concatenating them yields the identical schema, file KV metadata, and logical
    /// row SET as one batch writer over the union — proving the coordinator's
    /// per-shard parquet fusion is set-equivalent to the retain path. A shard with no
    /// rows leaves no file and contributes nothing; an all-empty set writes no file.
    #[test]
    fn concat_shards_matches_batch_over_union() {
        let columns = record_metric_columns();
        let dir = tempfile::tempdir().unwrap();
        let rows = sample_rows(7);
        // Three disjoint shard slices, one deliberately empty (no file).
        let shard_slices: [&[PerRecordRow]; 3] = [&rows[0..3], &[], &rows[3..7]];
        let mut shard_paths = Vec::new();
        for (id, slice) in shard_slices.iter().enumerate() {
            let path = dir.path().join(format!("shard-{id}.parquet"));
            let mut writer =
                StreamingPerRecordParquetWriter::new(path.clone(), columns.clone(), false, 2);
            for row in *slice {
                writer.push(row.clone()).unwrap();
            }
            writer.finish().unwrap();
            shard_paths.push(path);
        }
        // The empty shard left no file.
        assert!(
            !shard_paths[1].exists(),
            "an empty shard leaves no parquet file"
        );

        let combined = dir.path().join("combined.parquet");
        concat_per_record_parquet(&shard_paths, &combined).unwrap();

        let batch_path = dir.path().join("batch.parquet");
        write_per_record_parquet(&batch_path, &rows, &columns, false).unwrap();

        let (_, combined_batch, combined_meta) = read_coalesced(&combined);
        let (_, batch_batch, batch_meta) = read_coalesced(&batch_path);

        assert_eq!(combined_batch.schema(), batch_batch.schema());
        assert_eq!(combined_meta, batch_meta);
        assert_eq!(combined_batch.num_rows(), batch_batch.num_rows());

        // Key each row by session_num (unique per sample row) and
        // compare the request-latency cell, independent of cross-shard row order.
        let latency_by_session = |batch: &RecordBatch| -> BTreeMap<i64, Option<i64>> {
            let session = column::<Int64Array>(batch, "session_num");
            let latency = column::<Float64Array>(batch, "request_latency");
            (0..batch.num_rows())
                .map(|i| {
                    let value = latency.is_valid(i).then(|| latency.value(i) as i64);
                    (session.value(i), value)
                })
                .collect()
        };
        assert_eq!(
            latency_by_session(&combined_batch),
            latency_by_session(&batch_batch)
        );
    }

    /// An all-empty shard set (every shard saw zero rows) writes no combined file,
    /// matching the batch writer's empty-rows contract.
    #[test]
    fn concat_all_empty_shards_writes_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let missing = vec![
            dir.path().join("shard-0.parquet"),
            dir.path().join("shard-1.parquet"),
        ];
        let combined = dir.path().join("combined.parquet");
        concat_per_record_parquet(&missing, &combined).unwrap();
        assert!(!combined.exists());
    }

    #[test]
    fn streaming_empty_writes_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stream.parquet");
        let writer =
            StreamingPerRecordParquetWriter::new(path.clone(), record_metric_columns(), false, 2);
        writer.finish().unwrap();
        assert!(!path.exists(), "a streamed run with no rows leaves no file");
    }

    #[test]
    fn trace_columns_present_and_nullable_when_enabled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("profile_export.parquet");
        let mut with_trace = success_row();
        with_trace.trace = Some(PerRecordTrace {
            duration_ns: Some(9_000_000),
            connection_reused: Some(true),
            data_received_bytes: Some(4096),
            ..PerRecordTrace::default()
        });
        // Second row has no trace ⇒ all trace columns null.
        write_per_record_parquet(
            &path,
            &[with_trace, cancelled_row()],
            &record_metric_columns(),
            true,
        )
        .unwrap();

        let (batch, _) = read_back(&path);
        let duration = column::<Int64Array>(&batch, "trace_duration_ns");
        assert_eq!(duration.value(0), 9_000_000);
        assert!(duration.is_null(1));
        let reused = column::<BooleanArray>(&batch, "trace_connection_reused");
        assert!(reused.value(0));
        assert!(reused.is_null(1));
    }
}
