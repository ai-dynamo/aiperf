// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Server-metrics Parquet sink for `server_metrics_export.parquet`.
//!
//! The artifact contract covers column names, types, order, row values, and
//! `aiperf.schema_version = 1.0` metadata; physical Parquet encoding is writer
//! dependent. Raw rows come from the private wire JSONL. Endpoints and metrics
//! retain first-seen order, profiling boundaries are inclusive, counters and
//! histograms are emitted as non-negative deltas, gauges retain raw values, and
//! each histogram bucket occupies one row.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use arrow::array::{ArrayRef, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::export::parquet_util::{float_column, string_column};

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::ReportView;

mod units;

#[cfg(test)]
mod tests;

/// Wire-file basename written by the runner into the artifact directory.
const WIRE_FILENAME: &str = ".aiperf-server-metrics-parquet-wire.jsonl";

/// Output basename.
const OUTPUT_FILENAME: &str = "server_metrics_export.parquet";

/// Schema version stamped into the file's key-value metadata.
const SCHEMA_VERSION: &str = "1.0";

/// Reserved column names that a Prometheus label may not collide with.
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
        report: &dyn ReportView,
        artifact_dir: &Path,
        _cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let (start_ns, end_ns) = profiling_boundary(report)?;
        let wire_path = resolve_wire_path(artifact_dir);
        let hierarchy = Hierarchy::from_wire_file(&wire_path)?;
        // Remove the private wire input after reading it. Cleanup failures do not
        // abort export.
        if let Err(error) = std::fs::remove_file(&wire_path) {
            tracing::debug!(
                "server-metrics parquet: could not remove wire file {}: {error}",
                wire_path.display()
            );
        }
        let rows = hierarchy.collect_rows(start_ns, end_ns);

        // Empty row sets do not create an artifact.
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

/// Resolve the Parquet wire file for an export directory.
///
/// The wire input remains in the run artifact root when sink output is redirected
/// to a child directory, so probe the configured directory and then its parent.
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

/// Resolve a positive profiling `[start_ns, end_ns]` filter.
fn profiling_boundary(report: &dyn ReportView) -> Result<(i64, i64)> {
    let range = report
        .run_summary()
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

/// One raw scrape record from the wire JSONL. Only the fields the Parquet render
/// consumes are decoded; unknown fields (`is_duplicate`, `benchmark_phase`, trace
/// timings, `endpoint_latency_ns`) are ignored.
#[derive(Debug, serde::Deserialize)]
struct WireRecord {
    endpoint_url: String,
    timestamp_ns: i64,
    #[serde(default)]
    metrics: BTreeMap<String, WireFamily>,
}

/// A metric family within a wire record, with keys in lexicographic order.
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
/// Numeric fields retain the raw JSON token and use `f64::from_str` to obtain the
/// correctly rounded IEEE-754 value. The default serde_json float path can differ
/// by one ULP for some decimal tokens, which would perturb cumulative deltas.
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

/// Parse a JSON number token to the correctly rounded nearest `f64`.
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

/// Prometheus metric semantic type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Unknown,
}

impl MetricType {
    /// Decode the wire type, mapping unrecognized values to
    /// [`MetricType::Unknown`].
    fn from_wire(value: &str) -> Self {
        match value.to_ascii_lowercase().as_str() {
            "counter" => MetricType::Counter,
            "gauge" => MetricType::Gauge,
            "histogram" => MetricType::Histogram,
            "summary" => MetricType::Summary,
            _ => MetricType::Unknown,
        }
    }

    /// The lowercase `metric_type` column value.
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

    /// Whether scalar rows are emitted for this type (histograms emit through
    /// their own path). Summaries remain stored but omitted.
    fn is_scalar_emitted(self) -> bool {
        matches!(
            self,
            MetricType::Gauge | MetricType::Counter | MetricType::Unknown
        )
    }
}

/// Sorted `(name, value)` label pairs uniquely identifying a metric series.
type MetricKey = (String, Vec<(String, String)>);

/// Build a `MetricKey` from a metric name and optional labels dict (sorted).
fn metric_key(name: &str, labels: &Option<BTreeMap<String, String>>) -> MetricKey {
    let sorted = labels
        .as_ref()
        .map(|labels| labels.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default();
    (name.to_string(), sorted)
}

/// Scalar time series, stably sorted by timestamp when collected.
#[derive(Debug, Default)]
struct ScalarSeries {
    points: Vec<(i64, f64)>,
}

/// Histogram time series retaining the union of every observed bucket boundary.
#[derive(Debug, Default)]
struct HistogramSeries {
    bucket_les: Vec<String>,
    /// `(timestamp, sum, count, bucket_counts_in_bucket_les_order)`.
    points: Vec<(i64, f64, f64, Vec<f64>)>,
}

/// Stored metric series created from the first sample seen for its key.
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

/// Per-endpoint metric store preserving first-seen key order.
#[derive(Debug, Default)]
struct EndpointSeries {
    order: Vec<MetricKey>,
    entries: HashMap<MetricKey, MetricEntry>,
}

/// Multi-endpoint store preserving first-seen endpoint order.
#[derive(Debug, Default)]
struct Hierarchy {
    order: Vec<String>,
    endpoints: HashMap<String, EndpointSeries>,
}

impl Hierarchy {
    /// Rebuild the hierarchy from wire records in file order.
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

    /// Skip empty records and append every sample, including duplicates.
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

    /// All non-reserved Prometheus label keys, sorted alphabetically.
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

    /// Total stored series count.
    fn metric_count(&self) -> usize {
        self.endpoints.values().map(|e| e.order.len()).sum()
    }

    /// Per-type series counts; summary series are not included.
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

    /// Endpoint URLs observed in the wire input, sorted.
    fn endpoint_urls_sorted(&self) -> Vec<String> {
        let mut urls = self.order.clone();
        urls.sort();
        urls
    }

    /// Produce rows with endpoints and metrics in first-seen order and samples in
    /// timestamp order.
    fn collect_rows(&self, start_ns: i64, end_ns: i64) -> Vec<Row> {
        let mut rows = Vec::new();
        for endpoint_url in &self.order {
            let endpoint = &self.endpoints[endpoint_url];
            for key in &endpoint.order {
                let entry = &endpoint.entries[key];
                if entry.metric_type.is_scalar_emitted()
                    && let Some(series) = &entry.scalar
                {
                    series.collect_rows(endpoint_url, key, entry, start_ns, end_ns, &mut rows);
                } else if entry.metric_type == MetricType::Histogram
                    && let Some(series) = &entry.histogram
                {
                    series.collect_rows(endpoint_url, key, entry, start_ns, end_ns, &mut rows);
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
            // Summary or quantile samples without scalar values are omitted.
        } else if let Some(histogram) = &mut self.histogram {
            histogram.append(timestamp_ns, sample);
        }
    }
}

impl ScalarSeries {
    /// Timestamps and values stably sorted by timestamp.
    fn sorted(&self) -> Vec<(i64, f64)> {
        let mut points = self.points.clone();
        points.sort_by_key(|(ts, _)| *ts);
        points
    }

    /// Emit raw gauges and non-negative counter deltas.
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
    /// Append a histogram sample, expanding the bucket schema and filling
    /// missing buckets with zero.
    fn append(&mut self, timestamp_ns: i64, sample: &WireSample) {
        let Some(buckets) = &sample.buckets else {
            return;
        };
        let mut bucket_les = self.bucket_les.clone();
        for le in buckets.keys() {
            if !bucket_les.contains(le) {
                bucket_les.push(le.clone());
            }
        }
        bucket_les.sort_by(|a, b| {
            bucket_sort_key(a)
                .total_cmp(&bucket_sort_key(b))
                .then_with(|| a.cmp(b))
        });
        if bucket_les != self.bucket_les {
            let previous_les = std::mem::replace(&mut self.bucket_les, bucket_les);
            let previous_positions: HashMap<&str, usize> = previous_les
                .iter()
                .enumerate()
                .map(|(index, le)| (le.as_str(), index))
                .collect();
            for point in &mut self.points {
                let previous_counts = std::mem::take(&mut point.3);
                point.3 = self
                    .bucket_les
                    .iter()
                    .map(|le| {
                        previous_positions
                            .get(le.as_str())
                            .map(|index| previous_counts[*index])
                            .unwrap_or(0.0)
                    })
                    .collect();
            }
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

    /// Emit one row per bucket per in-range timestamp using cumulative deltas.
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

        // Use the last point before `start_ns` as the delta reference and include
        // points through `end_ns`.
        let insert_start = searchsorted_left(&timestamps, start_ns);
        let reference_idx = if insert_start > 0 {
            Some(insert_start - 1)
        } else {
            None
        };
        let insert_end = searchsorted_right(&timestamps, end_ns);
        if insert_end == 0 {
            // No point occurs at or before `end_ns`.
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

/// Sort histogram boundaries numerically with `+Inf` last.
fn bucket_sort_key(le: &str) -> f64 {
    if le == "+Inf" {
        f64::INFINITY
    } else {
        le.parse::<f64>().unwrap_or(f64::INFINITY)
    }
}

/// First index `i` with `values[i] >= target`.
fn searchsorted_left(values: &[i64], target: i64) -> usize {
    values.partition_point(|&v| v < target)
}

/// First index `i` with `values[i] > target`.
fn searchsorted_right(values: &[i64], target: i64) -> usize {
    values.partition_point(|&v| v <= target)
}

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

/// Build a nullable Arrow schema with a fixed head, sorted labels, fixed tail,
/// and `aiperf.*` metadata.
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

/// Deterministic metadata derivable from the report and wire input. Run config,
/// host, writer version, and export timestamp metadata are intentionally omitted.
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

/// Columnarize rows against the schema.
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

/// Write Snappy-compressed Parquet with the schema metadata at file level.
fn write_parquet(path: &Path, schema: Arc<Schema>, batch: &RecordBatch) -> Result<()> {
    super::parquet_util::write_parquet_table(path, schema, batch, "parquet export")
}
