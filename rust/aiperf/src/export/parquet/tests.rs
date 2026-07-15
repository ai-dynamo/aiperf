// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trip tests for the native server-metrics Parquet sink.
//!
//! The primary test writes a synthetic wire JSONL, runs the sink, reads the
//! Parquet back with the `parquet`/`arrow` reader, and asserts the column schema,
//! the row values (including gauge/counter deltas and histogram normalization),
//! and the `aiperf.schema_version` metadata. A second, availability-gated test
//! cross-checks the Rust-produced file against the exact Python exporter logic via
//! `.venv/bin/python` when a suitable interpreter is present.

use std::fs::File;
use std::io::Write as _;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int64Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use super::*;
use crate::export::ParquetExportConfig;
use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::{
    AccumulatorSummary, NativeReport, ReportServerMetricsMetadata, ReportServerMetricsPhaseRange,
};

/// Synthetic wire JSONL exercising a labeled counter, an unlabeled gauge, and an
/// unlabeled histogram, each with a warmup reference sample before the profiling
/// boundary `[100, 300]`.
const WIRE_JSONL: &str = r#"
{"endpoint_url":"http://s1/metrics","timestamp_ns":50,"metrics":{"http_requests_total":{"type":"counter","description":"Total requests","samples":[{"labels":{"method":"GET"},"value":10.0}]},"latency_seconds":{"type":"histogram","description":"Request latency","samples":[{"buckets":{"0.5":1.0,"1.0":2.0,"+Inf":2.0},"sum":1.0,"count":2.0}]},"queue_size":{"type":"gauge","description":"Queue depth","samples":[{"value":1.0}]}}}
{"endpoint_url":"http://s1/metrics","timestamp_ns":150,"metrics":{"http_requests_total":{"type":"counter","description":"Total requests","samples":[{"labels":{"method":"GET"},"value":30.0}]},"queue_size":{"type":"gauge","description":"Queue depth","samples":[{"value":5.0}]}}}
{"endpoint_url":"http://s1/metrics","timestamp_ns":200,"metrics":{"latency_seconds":{"type":"histogram","description":"Request latency","samples":[{"buckets":{"0.5":3.0,"1.0":5.0,"+Inf":6.0},"sum":5.0,"count":6.0}]}}}
{"endpoint_url":"http://s1/metrics","timestamp_ns":250,"metrics":{"http_requests_total":{"type":"counter","description":"Total requests","samples":[{"labels":{"method":"GET"},"value":50.0}]},"queue_size":{"type":"gauge","description":"Queue depth","samples":[{"value":7.0}]}}}
"#;

fn report_with_boundary(start_ns: i64, end_ns: i64) -> NativeReport {
    let summary = AccumulatorSummary::new();
    let mut report = NativeReport::new(&summary, None);
    report.summary.server_metrics = Some(ReportServerMetricsMetadata {
        profiling: Some(ReportServerMetricsPhaseRange { start_ns, end_ns }),
        ..ReportServerMetricsMetadata::default()
    });
    report
}

fn enabled_cfg() -> ExportConfig {
    ExportConfig {
        parquet: ParquetExportConfig { enabled: true },
        ..ExportConfig::default()
    }
}

fn write_wire(dir: &Path, jsonl: &str) {
    let mut file = File::create(dir.join(WIRE_FILENAME)).unwrap();
    file.write_all(jsonl.as_bytes()).unwrap();
}

fn read_back(
    path: &Path,
) -> (
    arrow::datatypes::SchemaRef,
    arrow::record_batch::RecordBatch,
) {
    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let schema = builder.schema().clone();
    let mut reader = builder.build().unwrap();
    let batch = reader.next().expect("one batch").unwrap();
    (schema, batch)
}

fn strings(batch: &arrow::record_batch::RecordBatch, name: &str) -> Vec<Option<String>> {
    let col = batch.column_by_name(name).unwrap();
    let arr = col.as_any().downcast_ref::<StringArray>().unwrap();
    (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i).to_string())
            }
        })
        .collect()
}

fn floats(batch: &arrow::record_batch::RecordBatch, name: &str) -> Vec<Option<f64>> {
    let col = batch.column_by_name(name).unwrap();
    let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
    (0..arr.len())
        .map(|i| {
            if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            }
        })
        .collect()
}

fn ints(batch: &arrow::record_batch::RecordBatch, name: &str) -> Vec<i64> {
    let col = batch.column_by_name(name).unwrap();
    let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
    (0..arr.len()).map(|i| arr.value(i)).collect()
}

#[test]
fn writes_schema_rows_and_metadata() {
    let dir = tempfile::tempdir().unwrap();
    write_wire(dir.path(), WIRE_JSONL);

    let report = report_with_boundary(100, 300);
    ParquetExporter
        .export(&report, dir.path(), &enabled_cfg())
        .unwrap();

    let output = dir.path().join(OUTPUT_FILENAME);
    assert!(output.exists(), "parquet file should be created");
    let (schema, batch) = read_back(&output);

    // Column schema: fixed head, one discovered label column ("method"), fixed tail.
    let names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        names,
        vec![
            "endpoint_url",
            "metric_name",
            "metric_type",
            "unit",
            "description",
            "timestamp_ns",
            "method",
            "value",
            "sum",
            "count",
            "bucket_le",
            "bucket_count",
        ]
    );

    // 2 counter rows + 3 histogram-bucket rows + 2 gauge rows = 7. Row order is
    // first-seen metric key order (alphabetical within the first record).
    assert_eq!(batch.num_rows(), 7);

    assert_eq!(
        strings(&batch, "metric_name"),
        vec![
            Some("http_requests_total".to_string()),
            Some("http_requests_total".to_string()),
            Some("latency_seconds".to_string()),
            Some("latency_seconds".to_string()),
            Some("latency_seconds".to_string()),
            Some("queue_size".to_string()),
            Some("queue_size".to_string()),
        ]
    );
    assert_eq!(
        strings(&batch, "metric_type"),
        vec![
            Some("counter".to_string()),
            Some("counter".to_string()),
            Some("histogram".to_string()),
            Some("histogram".to_string()),
            Some("histogram".to_string()),
            Some("gauge".to_string()),
            Some("gauge".to_string()),
        ]
    );
    assert_eq!(
        strings(&batch, "unit"),
        vec![
            Some("requests".to_string()),
            Some("requests".to_string()),
            Some("seconds".to_string()),
            Some("seconds".to_string()),
            Some("seconds".to_string()),
            None,
            None,
        ]
    );
    assert_eq!(
        ints(&batch, "timestamp_ns"),
        vec![150, 250, 200, 200, 200, 150, 250]
    );
    assert_eq!(
        strings(&batch, "method"),
        vec![
            Some("GET".to_string()),
            Some("GET".to_string()),
            None,
            None,
            None,
            None,
            None,
        ]
    );

    // Counter cumulative deltas from the ts=50 reference (10): 20, 40. Gauges raw: 5, 7.
    assert_eq!(
        floats(&batch, "value"),
        vec![
            Some(20.0),
            Some(40.0),
            None,
            None,
            None,
            Some(5.0),
            Some(7.0),
        ]
    );
    // Histogram deltas from the ts=50 reference: sum 4, count 4, buckets 2/3/4.
    assert_eq!(
        floats(&batch, "sum"),
        vec![None, None, Some(4.0), Some(4.0), Some(4.0), None, None]
    );
    assert_eq!(
        floats(&batch, "count"),
        vec![None, None, Some(4.0), Some(4.0), Some(4.0), None, None]
    );
    assert_eq!(
        strings(&batch, "bucket_le"),
        vec![
            None,
            None,
            Some("0.5".to_string()),
            Some("1.0".to_string()),
            Some("+Inf".to_string()),
            None,
            None,
        ]
    );
    assert_eq!(
        floats(&batch, "bucket_count"),
        vec![None, None, Some(2.0), Some(3.0), Some(4.0), None, None]
    );

    // The parity anchor: schema-level key-value metadata carries schema_version 1.0.
    let schema_version = schema
        .metadata()
        .get("aiperf.schema_version")
        .map(String::as_str);
    assert_eq!(schema_version, Some("1.0"));
    assert_eq!(
        schema
            .metadata()
            .get("aiperf.label_columns")
            .map(String::as_str),
        Some(r#"["method"]"#)
    );
}

#[test]
fn missing_profiling_boundary_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    write_wire(dir.path(), WIRE_JSONL);

    let summary = AccumulatorSummary::new();
    let report = NativeReport::new(&summary, None); // no server_metrics metadata
    let err = ParquetExporter
        .export(&report, dir.path(), &enabled_cfg())
        .unwrap_err();
    assert!(
        err.to_string().contains("profiling boundary"),
        "unexpected error: {err}"
    );
}

#[test]
fn empty_in_range_data_skips_file_creation() {
    let dir = tempfile::tempdir().unwrap();
    write_wire(dir.path(), WIRE_JSONL);

    // A boundary that excludes every sample (all timestamps are < 10_000).
    let report = report_with_boundary(10_000, 20_000);
    ParquetExporter
        .export(&report, dir.path(), &enabled_cfg())
        .unwrap();
    assert!(
        !dir.path().join(OUTPUT_FILENAME).exists(),
        "no rows should mean no file, matching the Python exporter"
    );
}

/// Cross-check the Rust-produced Parquet against the exact Python exporter logic.
///
/// Availability-gated: requires a Python interpreter with `pyarrow` and the
/// `aiperf` package importable. Point `AIPERF_VENV_PYTHON` at such an interpreter
/// (or place one at `<workspace>/.venv/bin/python`); the test is skipped when none
/// is found. When present, it rebuilds the oracle rows with the real
/// `ServerMetricsParquetExporter` row-collection + schema and asserts the Rust
/// table matches column-for-column.
#[test]
fn cross_check_against_python_exporter() {
    let Some(python) = locate_python() else {
        eprintln!("SKIP cross_check_against_python_exporter: no pyarrow+aiperf python found");
        return;
    };

    let dir = tempfile::tempdir().unwrap();
    write_wire(dir.path(), WIRE_JSONL);
    let report = report_with_boundary(100, 300);
    ParquetExporter
        .export(&report, dir.path(), &enabled_cfg())
        .unwrap();
    // The sink consumes (deletes) the wire file once rendered; the Python oracle
    // below reads the same wire, so re-materialize it for the cross-check.
    write_wire(dir.path(), WIRE_JSONL);

    let script = dir.path().join("cross_check.py");
    std::fs::write(&script, PYTHON_CROSS_CHECK).unwrap();

    let output = std::process::Command::new(&python)
        .arg(&script)
        .arg(dir.path().join(WIRE_FILENAME))
        .arg(dir.path().join(OUTPUT_FILENAME))
        .arg("100")
        .arg("300")
        .output()
        .expect("spawn python cross-check");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if stdout.contains("SKIP") {
        eprintln!(
            "SKIP cross_check_against_python_exporter: {}",
            stdout.trim()
        );
        return;
    }
    assert!(
        output.status.success() && stdout.contains("OK"),
        "python cross-check failed:\nstdout: {stdout}\nstderr: {stderr}"
    );
}

fn locate_python() -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var("AIPERF_VENV_PYTHON") {
        let path = std::path::PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    let candidate = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    candidate.exists().then_some(candidate)
}

const PYTHON_CROSS_CHECK: &str = r#"
import json
import math
import sys

try:
    import pyarrow.parquet as pq
    from aiperf.common.models.server_metrics_models import (
        ServerMetricsRecord,
        TimeRangeFilter,
    )
    from aiperf.server_metrics.parquet_exporter import ServerMetricsParquetExporter
    from aiperf.server_metrics.storage import ServerMetricsHierarchy
except Exception as exc:  # noqa: BLE001
    print(f"SKIP import: {exc!r}")
    sys.exit(0)

wire_path, parquet_path, start_ns, end_ns = sys.argv[1:5]
start_ns, end_ns = int(start_ns), int(end_ns)

hierarchy = ServerMetricsHierarchy()
with open(wire_path, "rb") as source:
    for line in source:
        if not line.strip():
            continue
        hierarchy.add_record(ServerMetricsRecord.model_validate(json.loads(line)))


class Adapter:
    def get_hierarchy_for_export(self):
        return hierarchy


# Build the oracle exporter without running __init__ (which needs a full run cfg).
exporter = object.__new__(ServerMetricsParquetExporter)
exporter._accumulator = Adapter()
exporter._time_filter = TimeRangeFilter(start_ns=start_ns, end_ns=end_ns)

reserved = exporter._get_reserved_names()
label_keys = {k for k in exporter._discover_all_label_keys() if k not in reserved}
schema = exporter._build_pyarrow_schema(label_keys)
rows = exporter._collect_all_rows(label_keys)

expected = {col: [] for col in schema.names}
for row in rows:
    for col in schema.names:
        value = row.get(col)
        if col == "metric_type" and value is not None:
            value = value.value if hasattr(value, "value") else str(value)
        expected[col].append(value)

table = pq.read_table(parquet_path)

if list(table.column_names) != list(schema.names):
    print(f"FAIL columns: {table.column_names} != {schema.names}")
    sys.exit(1)


def eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    return a == b


for col in schema.names:
    got = table.column(col).to_pylist()
    want = expected[col]
    if len(got) != len(want) or not all(eq(x, y) for x, y in zip(got, want)):
        print(f"FAIL column {col}: rust={got} python={want}")
        sys.exit(1)

meta = table.schema.metadata or {}
if meta.get(b"aiperf.schema_version") != b"1.0":
    print(f"FAIL schema_version metadata: {meta.get(b'aiperf.schema_version')!r}")
    sys.exit(1)

print("OK")
"#;
