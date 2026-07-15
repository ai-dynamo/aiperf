// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust server-metrics Parquet sink: `server_metrics_export.parquet`.
//!
//! Ports the Python `server_metrics/parquet_exporter.py`
//! (`ServerMetricsParquetExporter`) plus the storage/unit machinery it depends on
//! (`server_metrics/storage.py`, `server_metrics/units.py`), which today is driven
//! by `orchestrator/native_report.py::_render_server_metrics_parquet`. That Python
//! path reads the Rust-emitted `.aiperf-server-metrics-parquet-wire.jsonl` wire
//! file (raw per-record [`crate::server_metrics::ServerMetricsRecord`] rows written
//! by `rust/runner/src/server_metrics.rs::write_parquet_wire_jsonl`), rebuilds the
//! `ServerMetricsHierarchy`, and renders Parquet with pyarrow. Doing it natively
//! here lets the integration owner eventually delete that round-trip.
//!
//! # Parity target (documented honestly)
//! Parquet byte-identity across writers (pyarrow vs arrow-rs) is **not** a goal:
//! row-group layout, encodings, and page/compression defaults differ. The target
//! is **schema + data + `aiperf.schema_version` equality**: identical column
//! names/types/order, identical row values (including the delta arithmetic and
//! histogram bucket normalization), and the `aiperf.schema_version = 1.0`
//! key-value metadata. This is verifiable by reading both files back with pyarrow
//! and asserting equal tables (see `parquet/tests.rs`).
//!
//! # Data path
//! The aggregated [`NativeReport`] does not carry the raw server-metric rows; they
//! come from the wire JSONL in the artifact dir. This sink reads that file, rebuilds
//! the endpoint/metric hierarchy in first-seen (insertion) order exactly like the
//! Python dicts, applies the profiling-boundary time filter taken from
//! `report.summary.server_metrics.profiling`, computes gauge/counter/histogram
//! deltas, and writes the normalized (one-row-per-bucket) Parquet table.
//!
//! Ported from (cite exact oracle):
//! - `src/aiperf/server_metrics/parquet_exporter.py` (schema, metadata, deltas)
//! - `src/aiperf/server_metrics/storage.py` (hierarchy, scalar/histogram series)
//! - `src/aiperf/server_metrics/units.py` + `common/enums/metric_enums.py`
//!   (`infer_unit` + `BaseMetricUnit.display_name`)

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::metadata::KeyValue;
use parquet::file::properties::WriterProperties;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::NativeReport;

mod units;

#[cfg(test)]
mod tests;

/// Wire-file basename written by the runner into the artifact dir. Mirrors
/// `orchestrator/rust_wire.py::SERVER_METRICS_PARQUET_WIRE_PATH`.
const WIRE_FILENAME: &str = ".aiperf-server-metrics-parquet-wire.jsonl";

/// Output basename. Mirrors `cfg.artifacts.server_metrics_export_parquet_file`.
const OUTPUT_FILENAME: &str = "server_metrics_export.parquet";

/// Schema version stamped into the file's key-value metadata.
const SCHEMA_VERSION: &str = "1.0";

/// Reserved column names that a Prometheus label may not collide with. Mirrors
/// `parquet_exporter.py::_get_reserved_names`.
const RESERVED_NAMES: &[&str] = &[
    "endpoint_url",
    "metric_name",
    "metric_type",
    "unit",
    "description",
    "timestamp_ns",
    "value",
    "sum",
    "count",
    "bucket_le",
    "bucket_count",
];

/// The server-metrics Parquet [`Exporter`].
pub struct ParquetExporter;

impl Exporter for ParquetExporter {
    fn name(&self) -> &'static str {
        "server_metrics_parquet"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.parquet.enabled
    }

    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let (start_ns, end_ns) = profiling_boundary(report)?;
        let wire_path = resolve_wire_path(artifact_dir);
        let hierarchy = Hierarchy::from_wire_file(&wire_path)?;
        // The wire JSONL is a private runner→sink intermediate, not a user
        // artifact; remove it once consumed so it never lingers in the run
        // directory. Mirrors the retired Python renderer's `wire_path.unlink()`
        // (`orchestrator/native_report.py::_render_server_metrics_parquet`).
        // Best-effort: a cleanup failure never aborts the export (the native-v2
        // report is the committed authority).
        if let Err(error) = std::fs::remove_file(&wire_path) {
            tracing::debug!(
                "server-metrics parquet: could not remove wire file {}: {error}",
                wire_path.display()
            );
        }
        let rows = hierarchy.collect_rows(start_ns, end_ns);

        // Mirror the Python exporter: no rows -> skip file creation entirely.
        if rows.is_empty() {
            tracing::debug!("server-metrics parquet: no rows to export; skipping file");
            return Ok(());
        }

        let label_keys = hierarchy.label_columns();
        let schema = build_schema(&label_keys, &hierarchy, start_ns, end_ns);
        let batch = build_record_batch(&schema, &label_keys, &rows)?;

        let output_path = artifact_dir.join(OUTPUT_FILENAME);
        write_parquet(&output_path, schema, &batch)
            .with_context(|| format!("writing {}", output_path.display()))?;
        Ok(())
    }
}

/// Resolve the runner-emitted parquet wire file for a given export directory.
///
/// The wire JSONL is an **input** the runner writes into the run's artifact root
/// (`rust/runner/src/execute.rs`), whereas the export directory handed to a sink
/// may be an OUTPUT redirect (the `AIPERF_EXPORT_SUBDIR` parity harness points
/// sinks at `<artifact_root>/<subdir>/` so Rust outputs coexist with the Python
/// files). The wire file is never redirected, so resolve it in `artifact_dir`
/// first and fall back to the parent directory when the subdir redirect is in
/// effect. Normal runs (no redirect) match on the first probe.
fn resolve_wire_path(artifact_dir: &Path) -> std::path::PathBuf {
    let direct = artifact_dir.join(WIRE_FILENAME);
    if direct.exists() {
        return direct;
    }
    if let Some(parent) = artifact_dir.parent() {
        let fallback = parent.join(WIRE_FILENAME);
        if fallback.exists() {
            return fallback;
        }
    }
    direct
}

/// Resolve the profiling `[start_ns, end_ns]` filter. Mirrors the Python
/// `_render_server_metrics_parquet` guard (`results.start_ns >= results.end_ns`
/// is a hard error) reading `metadata["profiling"]`.
fn profiling_boundary(report: &NativeReport) -> Result<(i64, i64)> {
    let range = report
        .summary
        .server_metrics
        .as_ref()
        .and_then(|meta| meta.profiling.as_ref())
        .context("native server-metrics report is missing a profiling boundary")?;
    if range.start_ns >= range.end_ns {
        bail!(
            "cannot render server-metrics Parquet without a positive profiling boundary \
             (start_ns={} end_ns={})",
            range.start_ns,
            range.end_ns
        );
    }
    Ok((range.start_ns, range.end_ns))
}

// =============================================================================
// Wire record decoding
// =============================================================================

/// One raw scrape record from the wire JSONL. Only the fields the Parquet render
/// consumes are decoded; unknown fields (`is_duplicate`, `benchmark_phase`, trace
/// timings, `endpoint_latency_ns`) are ignored. Mirrors the `full_record` shape in
/// `rust/runner/src/server_metrics.rs`.
#[derive(Debug, serde::Deserialize)]
struct WireRecord {
    endpoint_url: String,
    timestamp_ns: i64,
    #[serde(default)]
    metrics: BTreeMap<String, WireFamily>,
}

/// A metric family within a wire record. `BTreeMap` matches the sorted key order
/// the runner emits (it serializes a `BTreeMap`), so iteration order is stable and
/// identical to the Python dict insertion order.
#[derive(Debug, serde::Deserialize)]
struct WireFamily {
    #[serde(rename = "type")]
    metric_type: String,
    description: String,
    #[serde(default)]
    samples: Vec<WireSample>,
}

/// One sample: scalar (`value`) or histogram (`buckets` + `sum` + `count`).
///
/// All numeric fields decode through [`de_opt_f64`] / [`de_bucket_map`] rather
/// than serde_json's built-in number deserializer. serde_json's default float
/// parser is not always correctly rounded — for some decimals it lands 1 ULP off
/// the IEEE-754 nearest value that Rust's `f64::from_str` and Python's `float()`
/// both produce (e.g. `0.36366626900000004` decodes to `…da8a` via serde_json but
/// `…da8b` via `f64::from_str`). The runner writes this wire and Python's exporter
/// reads it with `float()`, so decoding the same bytes to a different f64 here
/// perturbs the cumulative-sum/count/bucket deltas by ~1 ULP away from the Python
/// output. Recovering the exact number text (the enabled `raw_value` feature) and
/// parsing it with `f64::from_str` restores byte-for-value parity.
#[derive(Debug, serde::Deserialize)]
struct WireSample {
    #[serde(default)]
    labels: Option<BTreeMap<String, String>>,
    #[serde(default, deserialize_with = "de_opt_f64")]
    value: Option<f64>,
    #[serde(default, deserialize_with = "de_bucket_map")]
    buckets: Option<BTreeMap<String, f64>>,
    #[serde(default, deserialize_with = "de_opt_f64")]
    sum: Option<f64>,
    #[serde(default, deserialize_with = "de_opt_f64")]
    count: Option<f64>,
}

/// Parse a JSON number token to the correctly-rounded nearest f64 via
/// `f64::from_str`, matching Python's `float()`. See [`WireSample`] for why the
/// default serde_json number path is unsuitable.
fn parse_exact_f64<E: serde::de::Error>(raw: &serde_json::value::RawValue) -> Result<f64, E> {
    let text = raw.get().trim();
    text.parse::<f64>()
        .map_err(|error| serde::de::Error::custom(format!("invalid wire number {text:?}: {error}")))
}

/// Correctly-rounded `Option<f64>` decoder. Absent (`#[serde(default)]`) and JSON
/// `null` both yield `None`.
fn de_opt_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw: Option<Box<serde_json::value::RawValue>> = Option::deserialize(deserializer)?;
    match raw {
        None => Ok(None),
        Some(raw) if raw.get().trim() == "null" => Ok(None),
        Some(raw) => parse_exact_f64(&raw).map(Some),
    }
}

/// Correctly-rounded histogram bucket map decoder (`le` string -> cumulative
/// count). Absent yields `None`.
fn de_bucket_map<'de, D>(deserializer: D) -> Result<Option<BTreeMap<String, f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    let raw: Option<BTreeMap<String, Box<serde_json::value::RawValue>>> =
        Option::deserialize(deserializer)?;
    match raw {
        None => Ok(None),
        Some(entries) => {
            let mut out = BTreeMap::new();
            for (le, value) in entries {
                out.insert(le, parse_exact_f64(&value)?);
            }
            Ok(Some(out))
        }
    }
}

// =============================================================================
// Hierarchy / time-series (port of storage.py)
// =============================================================================

/// Prometheus metric semantic type. Mirrors `enums.PrometheusMetricType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Unknown,
}

impl MetricType {
    /// Decode the wire `type` string. Unrecognized values map to `Unknown`,
    /// matching `PrometheusMetricType._missing_` fallback semantics.
    fn from_wire(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "counter" => MetricType::Counter,
            "gauge" => MetricType::Gauge,
            "histogram" => MetricType::Histogram,
            "summary" => MetricType::Summary,
            _ => MetricType::Unknown,
        }
    }

    /// The `metric_type` column value (the `PrometheusMetricType` enum's string).
    fn as_str(self) -> &'static str {
        match self {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
            MetricType::Summary => "summary",
            MetricType::Unknown => "unknown",
        }
    }

    /// Gauge-equivalent scalar storage: gauge and unknown export raw values.
    fn is_gauge(self) -> bool {
        matches!(self, MetricType::Gauge | MetricType::Unknown)
    }

    /// Whether `_collect_all_rows_generator` emits rows for this type. Summary is
    /// stored but never emitted (Python handles neither the scalar nor histogram
    /// branch for it).
    fn is_scalar_emitted(self) -> bool {
        matches!(
            self,
            MetricType::Gauge | MetricType::Counter | MetricType::Unknown
        )
    }
}

/// Sorted `(name, value)` label pairs uniquely identifying a series within a
/// metric family. Mirrors `ServerMetricKey`.
type MetricKey = (String, Vec<(String, String)>);

/// Build a `MetricKey` from a metric name and optional labels dict (sorted).
fn metric_key(name: &str, labels: &Option<BTreeMap<String, String>>) -> MetricKey {
    let sorted = labels
        .as_ref()
        .map(|labels| labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    (name.to_string(), sorted)
}

/// Scalar (gauge/counter/unknown) time series. Values are kept in insertion order
/// then stably sorted by timestamp on demand, reproducing `ScalarTimeSeries`'
/// stable-insertion behavior for equal timestamps.
#[derive(Debug, Default)]
struct ScalarSeries {
    points: Vec<(i64, f64)>,
}

/// Histogram time series with a fixed bucket schema locked on first append.
/// Mirrors `HistogramTimeSeries`.
#[derive(Debug, Default)]
struct HistogramSeries {
    bucket_les: Vec<String>,
    /// `(timestamp, sum, count, bucket_counts_in_bucket_les_order)`.
    points: Vec<(i64, f64, f64, Vec<f64>)>,
}

/// One stored metric series (type + description + typed data), created on the
/// first sample seen for its key. Mirrors `ServerMetricEntry`.
#[derive(Debug)]
struct MetricEntry {
    metric_type: MetricType,
    description: String,
    scalar: Option<ScalarSeries>,
    histogram: Option<HistogramSeries>,
}

impl MetricEntry {
    fn new(metric_type: MetricType, description: String) -> Self {
        let (scalar, histogram) = if metric_type == MetricType::Histogram {
            (None, Some(HistogramSeries::default()))
        } else {
            (Some(ScalarSeries::default()), None)
        };
        MetricEntry {
            metric_type,
            description,
            scalar,
            histogram,
        }
    }
}

/// Per-endpoint metric store preserving first-seen key order. Mirrors
/// `ServerMetricsTimeSeries.metrics` (a dict).
#[derive(Debug, Default)]
struct EndpointSeries {
    order: Vec<MetricKey>,
    entries: HashMap<MetricKey, MetricEntry>,
}

/// Multi-endpoint store preserving first-seen endpoint order. Mirrors
/// `ServerMetricsHierarchy.endpoints` (a dict).
#[derive(Debug, Default)]
struct Hierarchy {
    order: Vec<String>,
    endpoints: HashMap<String, EndpointSeries>,
}

impl Hierarchy {
    /// Rebuild the hierarchy from the wire JSONL, in file order. Mirrors the
    /// Python renderer's `for line in source: hierarchy.add_record(...)`.
    fn from_wire_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path).with_context(|| {
            format!(
                "reading server-metrics parquet wire file {}",
                path.display()
            )
        })?;
        let mut hierarchy = Hierarchy::default();
        for (line_number, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let record: WireRecord = serde_json::from_str(line).with_context(|| {
                format!(
                    "invalid server-metrics parquet wire record at {}:{}",
                    path.display(),
                    line_number + 1
                )
            })?;
            hierarchy.add_record(record);
        }
        Ok(hierarchy)
    }

    /// Ingest one record. Mirrors `ServerMetricsHierarchy.add_record` +
    /// `ServerMetricsTimeSeries.append_snapshot`: empty-metrics records are
    /// skipped; all samples (including duplicates) append to the series.
    fn add_record(&mut self, record: WireRecord) {
        if record.metrics.is_empty() {
            return;
        }
        if !self.endpoints.contains_key(&record.endpoint_url) {
            self.order.push(record.endpoint_url.clone());
            self.endpoints
                .insert(record.endpoint_url.clone(), EndpointSeries::default());
        }
        let endpoint = self
            .endpoints
            .get_mut(&record.endpoint_url)
            .expect("endpoint just inserted");
        for (name, family) in &record.metrics {
            let metric_type = MetricType::from_wire(&family.metric_type);
            for sample in &family.samples {
                let key = metric_key(name, &sample.labels);
                if !endpoint.entries.contains_key(&key) {
                    endpoint.order.push(key.clone());
                    endpoint.entries.insert(
                        key.clone(),
                        MetricEntry::new(metric_type, family.description.clone()),
                    );
                }
                let entry = endpoint.entries.get_mut(&key).expect("entry just inserted");
                entry.append(record.timestamp_ns, sample);
            }
        }
    }

    /// All non-reserved Prometheus label keys, sorted alphabetically. Mirrors
    /// `_discover_all_label_keys` filtered by `_get_reserved_names`.
    fn label_columns(&self) -> Vec<String> {
        let mut keys: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for endpoint in self.endpoints.values() {
            for key in &endpoint.order {
                for (label, _) in &key.1 {
                    if !RESERVED_NAMES.contains(&label.as_str()) {
                        keys.insert(label.clone());
                    }
                }
            }
        }
        keys.into_iter().collect()
    }

    /// Total stored series count. Mirrors the `metric_count` metadata.
    fn metric_count(&self) -> usize {
        self.endpoints.values().map(|e| e.order.len()).sum()
    }

    /// Per-type series counts. Mirrors the `metric_type_counts` metadata (summary
    /// is not one of the tracked keys, so it is skipped rather than panicking as
    /// the Python dict-index would).
    fn metric_type_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::from([
            ("gauge", 0),
            ("counter", 0),
            ("histogram", 0),
            ("unknown", 0),
        ]);
        for endpoint in self.endpoints.values() {
            for key in &endpoint.order {
                let entry = &endpoint.entries[key];
                if let Some(slot) = counts.get_mut(entry.metric_type.as_str()) {
                    *slot += 1;
                }
            }
        }
        counts
    }

    /// Configured/observed endpoint URLs, sorted. Mirrors `endpoint_urls`.
    fn endpoint_urls_sorted(&self) -> Vec<String> {
        let mut urls = self.order.clone();
        urls.sort();
        urls
    }

    /// Produce every export row in Python row order: endpoints in first-seen
    /// order, metrics in first-seen key order, samples in timestamp order. Mirrors
    /// `_collect_all_rows_generator`.
    fn collect_rows(&self, start_ns: i64, end_ns: i64) -> Vec<Row> {
        let mut rows = Vec::new();
        for endpoint_url in &self.order {
            let endpoint = &self.endpoints[endpoint_url];
            for key in &endpoint.order {
                let entry = &endpoint.entries[key];
                if entry.metric_type.is_scalar_emitted() {
                    if let Some(series) = &entry.scalar {
                        series.collect_rows(endpoint_url, key, entry, start_ns, end_ns, &mut rows);
                    }
                } else if entry.metric_type == MetricType::Histogram {
                    if let Some(series) = &entry.histogram {
                        series.collect_rows(endpoint_url, key, entry, start_ns, end_ns, &mut rows);
                    }
                }
            }
        }
        rows
    }
}

impl MetricEntry {
    /// Append one sample to the appropriate typed series.
    fn append(&mut self, timestamp_ns: i64, sample: &WireSample) {
        if let Some(scalar) = &mut self.scalar {
            if let Some(value) = sample.value {
                scalar.points.push((timestamp_ns, value));
            }
            // Summary/quantile samples without a scalar value are dropped, mirroring
            // that the exporter never emits rows for them.
        } else if let Some(histogram) = &mut self.histogram {
            histogram.append(timestamp_ns, sample);
        }
    }
}

impl ScalarSeries {
    /// Timestamps/values sorted stably by timestamp. Mirrors the sorted invariant
    /// `ScalarTimeSeries` maintains (stable for equal timestamps).
    fn sorted(&self) -> Vec<(i64, f64)> {
        let mut points = self.points.clone();
        points.sort_by_key(|(ts, _)| *ts);
        points
    }

    /// Emit scalar rows with gauge/counter delta semantics. Mirrors
    /// `_collect_scalar_rows`.
    fn collect_rows(
        &self,
        endpoint_url: &str,
        key: &MetricKey,
        entry: &MetricEntry,
        start_ns: i64,
        end_ns: i64,
        rows: &mut Vec<Row>,
    ) {
        let points = self.sorted();
        let timestamps: Vec<i64> = points.iter().map(|(ts, _)| *ts).collect();
        let values: Vec<f64> = points.iter().map(|(_, value)| *value).collect();

        let first = searchsorted_left(&timestamps, start_ns);
        let last = searchsorted_right(&timestamps, end_ns);
        if first >= last {
            return;
        }

        let unit = units::infer_unit(&key.0, &entry.description).map(|unit| unit.display_name());

        let exported: Vec<f64> = if entry.metric_type.is_gauge() {
            values[first..last].to_vec()
        } else {
            // Counter: cumulative delta from the last point before start, clamped >= 0.
            let reference = if first > 0 {
                values[first - 1]
            } else {
                values[first]
            };
            values[first..last]
                .iter()
                .map(|value| (value - reference).max(0.0))
                .collect()
        };

        for (offset, value) in exported.into_iter().enumerate() {
            rows.push(Row {
                endpoint_url: endpoint_url.to_string(),
                metric_name: key.0.clone(),
                metric_type: entry.metric_type.as_str(),
                unit: unit.clone(),
                description: entry.description.clone(),
                timestamp_ns: timestamps[first + offset],
                labels: key.1.clone(),
                value: Some(value),
                sum: None,
                count: None,
                bucket_le: None,
                bucket_count: None,
            });
        }
    }
}

impl HistogramSeries {
    /// Append a histogram sample, locking the bucket schema on the first one.
    /// Mirrors `HistogramTimeSeries.append` (missing buckets fill with 0.0).
    fn append(&mut self, timestamp_ns: i64, sample: &WireSample) {
        let Some(buckets) = &sample.buckets else {
            return;
        };
        if self.bucket_les.is_empty() {
            let mut les: Vec<String> = buckets.keys().cloned().collect();
            les.sort_by(|a, b| {
                bucket_sort_key(a)
                    .partial_cmp(&bucket_sort_key(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            self.bucket_les = les;
        }
        let row: Vec<f64> = self
            .bucket_les
            .iter()
            .map(|le| buckets.get(le).copied().unwrap_or(0.0))
            .collect();
        self.points.push((
            timestamp_ns,
            sample.sum.unwrap_or(0.0),
            sample.count.unwrap_or(0.0),
            row,
        ));
    }

    /// Emit one row per bucket per in-range timestamp with cumulative-delta
    /// semantics. Mirrors `_collect_histogram_rows`.
    fn collect_rows(
        &self,
        endpoint_url: &str,
        key: &MetricKey,
        entry: &MetricEntry,
        start_ns: i64,
        end_ns: i64,
        rows: &mut Vec<Row>,
    ) {
        if self.points.is_empty() {
            return;
        }
        let mut points = self.points.clone();
        points.sort_by_key(|(ts, ..)| *ts);
        let timestamps: Vec<i64> = points.iter().map(|(ts, ..)| *ts).collect();

        // get_indices_for_filter: reference = last < start, final = last <= end.
        let insert_start = searchsorted_left(&timestamps, start_ns);
        let reference_idx = if insert_start > 0 {
            Some(insert_start - 1)
        } else {
            None
        };
        let insert_end = searchsorted_right(&timestamps, end_ns);
        if insert_end == 0 {
            // final_idx is None -> no rows.
            return;
        }
        let final_idx = insert_end - 1;
        let first_idx = insert_start;
        if first_idx > final_idx {
            return;
        }

        let (ref_sum, ref_count, ref_buckets) = match reference_idx {
            Some(idx) => (points[idx].1, points[idx].2, points[idx].3.clone()),
            None => (
                points[first_idx].1,
                points[first_idx].2,
                points[first_idx].3.clone(),
            ),
        };

        let unit = units::infer_unit(&key.0, &entry.description).map(|unit| unit.display_name());

        for point in &points[first_idx..=final_idx] {
            let (timestamp, sum, count, buckets) = point;
            let sum_delta = (sum - ref_sum).max(0.0);
            let count_delta = (count - ref_count).max(0.0);
            for (j, le) in self.bucket_les.iter().enumerate() {
                let bucket_delta = (buckets[j] - ref_buckets[j]).max(0.0);
                rows.push(Row {
                    endpoint_url: endpoint_url.to_string(),
                    metric_name: key.0.clone(),
                    metric_type: entry.metric_type.as_str(),
                    unit: unit.clone(),
                    description: entry.description.clone(),
                    timestamp_ns: *timestamp,
                    labels: key.1.clone(),
                    value: None,
                    sum: Some(sum_delta),
                    count: Some(count_delta),
                    bucket_le: Some(le.clone()),
                    bucket_count: Some(bucket_delta),
                });
            }
        }
    }
}

/// Sort key for a histogram bucket boundary: numeric ascending with `+Inf` last.
/// Mirrors `storage.py::_bucket_sort_key`.
fn bucket_sort_key(le: &str) -> f64 {
    if le == "+Inf" {
        f64::INFINITY
    } else {
        le.parse::<f64>().unwrap_or(f64::INFINITY)
    }
}

/// First index `i` with `values[i] >= target` (numpy `searchsorted(..., "left")`).
fn searchsorted_left(values: &[i64], target: i64) -> usize {
    values.partition_point(|&v| v < target)
}

/// First index `i` with `values[i] > target` (numpy `searchsorted(..., "right")`).
fn searchsorted_right(values: &[i64], target: i64) -> usize {
    values.partition_point(|&v| v <= target)
}

// =============================================================================
// Row model + Arrow assembly
// =============================================================================

/// One materialized Parquet row prior to columnarization.
#[derive(Debug)]
struct Row {
    endpoint_url: String,
    metric_name: String,
    metric_type: &'static str,
    unit: Option<String>,
    description: String,
    timestamp_ns: i64,
    labels: Vec<(String, String)>,
    value: Option<f64>,
    sum: Option<f64>,
    count: Option<f64>,
    bucket_le: Option<String>,
    bucket_count: Option<f64>,
}

impl Row {
    /// Value for a dynamic label column (`None` when this series lacks the label).
    fn label(&self, key: &str) -> Option<String> {
        self.labels
            .iter()
            .find(|(name, _)| name == key)
            .map(|(_, value)| value.clone())
    }
}

/// Build the Arrow schema with `aiperf.*` metadata. Column order mirrors
/// `_build_pyarrow_schema`: fixed head, sorted label columns, fixed tail. All
/// fields are nullable to match pyarrow's `pa.field(..)` defaults.
fn build_schema(
    label_keys: &[String],
    hierarchy: &Hierarchy,
    start_ns: i64,
    end_ns: i64,
) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("endpoint_url", DataType::Utf8, true),
        Field::new("metric_name", DataType::Utf8, true),
        Field::new("metric_type", DataType::Utf8, true),
        Field::new("unit", DataType::Utf8, true),
        Field::new("description", DataType::Utf8, true),
        Field::new("timestamp_ns", DataType::Int64, true),
    ];
    for key in label_keys {
        fields.push(Field::new(key, DataType::Utf8, true));
    }
    fields.extend([
        Field::new("value", DataType::Float64, true),
        Field::new("sum", DataType::Float64, true),
        Field::new("count", DataType::Float64, true),
        Field::new("bucket_le", DataType::Utf8, true),
        Field::new("bucket_count", DataType::Float64, true),
    ]);

    let metadata = build_metadata(label_keys, hierarchy, start_ns, end_ns);
    Arc::new(Schema::new_with_metadata(fields, metadata))
}

/// Deterministic, data-derived file metadata. The `aiperf.schema_version` key is
/// the parity anchor; the config-derived keys the Python exporter also writes
/// (`input_config`, `benchmark_id`, `model_names`, `concurrency`, `request_rate`,
/// host/python/pyarrow versions, `export_timestamp_utc`) are intentionally omitted
/// here — they are not derivable from `NativeReport` + the wire file and are not
/// part of the schema/data parity target.
fn build_metadata(
    label_keys: &[String],
    hierarchy: &Hierarchy,
    start_ns: i64,
    end_ns: i64,
) -> HashMap<String, String> {
    let endpoint_urls = hierarchy.endpoint_urls_sorted();
    let type_counts = hierarchy.metric_type_counts();
    let duration_ns = end_ns - start_ns;

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
        "aiperf.exporter".to_string(),
        "ServerMetricsParquetExporter".to_string(),
    );
    metadata.insert(
        "aiperf.time_filter_start_ns".to_string(),
        start_ns.to_string(),
    );
    metadata.insert("aiperf.time_filter_end_ns".to_string(), end_ns.to_string());
    metadata.insert(
        "aiperf.profiling_duration_ns".to_string(),
        duration_ns.to_string(),
    );
    metadata.insert(
        "aiperf.profiling_duration_seconds".to_string(),
        (duration_ns as f64 / 1_000_000_000.0).to_string(),
    );
    metadata.insert(
        "aiperf.endpoint_urls".to_string(),
        serde_json::to_string(&endpoint_urls).unwrap_or_else(|_| "[]".to_string()),
    );
    metadata.insert(
        "aiperf.endpoint_count".to_string(),
        endpoint_urls.len().to_string(),
    );
    metadata.insert(
        "aiperf.label_columns".to_string(),
        serde_json::to_string(&label_keys).unwrap_or_else(|_| "[]".to_string()),
    );
    metadata.insert(
        "aiperf.label_count".to_string(),
        label_keys.len().to_string(),
    );
    metadata.insert(
        "aiperf.metric_count".to_string(),
        hierarchy.metric_count().to_string(),
    );
    metadata.insert(
        "aiperf.metric_type_counts".to_string(),
        serde_json::to_string(&type_counts).unwrap_or_else(|_| "{}".to_string()),
    );
    metadata.insert(
        "aiperf.schema_note".to_string(),
        "Label columns vary by endpoint/model. Use union_by_name=true for cross-file queries."
            .to_string(),
    );
    metadata
}

/// Columnarize the rows against the schema. Mirrors the pyarrow
/// `pa.table({col: [r.get(col) ...]})` construction.
fn build_record_batch(
    schema: &Arc<Schema>,
    label_keys: &[String],
    rows: &[Row],
) -> Result<RecordBatch> {
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());

    columns.push(string_column(
        rows.iter().map(|r| Some(r.endpoint_url.clone())),
    ));
    columns.push(string_column(
        rows.iter().map(|r| Some(r.metric_name.clone())),
    ));
    columns.push(string_column(
        rows.iter().map(|r| Some(r.metric_type.to_string())),
    ));
    columns.push(string_column(rows.iter().map(|r| r.unit.clone())));
    columns.push(string_column(
        rows.iter().map(|r| Some(r.description.clone())),
    ));
    columns.push(Arc::new(Int64Array::from(
        rows.iter().map(|r| r.timestamp_ns).collect::<Vec<_>>(),
    )) as ArrayRef);

    for key in label_keys {
        columns.push(string_column(rows.iter().map(|r| r.label(key))));
    }

    columns.push(float_column(rows.iter().map(|r| r.value)));
    columns.push(float_column(rows.iter().map(|r| r.sum)));
    columns.push(float_column(rows.iter().map(|r| r.count)));
    columns.push(string_column(rows.iter().map(|r| r.bucket_le.clone())));
    columns.push(float_column(rows.iter().map(|r| r.bucket_count)));

    RecordBatch::try_new(schema.clone(), columns).context("assembling parquet record batch")
}

/// Build a nullable UTF-8 column.
fn string_column<I: Iterator<Item = Option<String>>>(values: I) -> ArrayRef {
    Arc::new(StringArray::from_iter(values)) as ArrayRef
}

/// Build a nullable float64 column.
fn float_column<I: Iterator<Item = Option<f64>>>(values: I) -> ArrayRef {
    Arc::new(Float64Array::from_iter(values)) as ArrayRef
}

/// Write the record batch to Parquet with Snappy compression (matching the Python
/// `compression="snappy"`) and file-level key-value metadata mirroring the schema
/// metadata. Byte-identity with pyarrow is not attempted (see module docs).
fn write_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating parquet export directory {}", parent.display()))?;
    }
    let file = File::create(path)
        .with_context(|| format!("creating parquet export {}", path.display()))?;

    let kv: Vec<KeyValue> = schema
        .metadata()
        .iter()
        .map(|(key, value)| KeyValue::new(key.clone(), value.clone()))
        .collect();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_key_value_metadata(Some(kv))
        .build();

    let mut writer = ArrowWriter::try_new(file, schema, Some(props))
        .context("constructing parquet arrow writer")?;
    writer
        .write(batch)
        .context("writing parquet record batch")?;
    writer.close().context("finalizing parquet file")?;
    Ok(())
}
