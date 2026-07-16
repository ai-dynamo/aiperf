// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native-Rust server-metrics summary sink: `server_metrics_export.json` + `.csv`.
//!
//! Ports the Python `aiperf/server_metrics/{json_exporter,csv_exporter}.py` to the
//! runner. The native-v2 report already carries the server-metrics metadata
//! (`summary.server_metrics`) and the labeled/typed series (gauge/counter/
//! histogram) under `report.server_metrics`; this sink serializes them to the
//! two legacy files byte-for-byte.
//!
//! Parity oracle (byte-exact source of truth), grounded at `path:line`:
//! - JSON: `aiperf/server_metrics/json_exporter.py::_generate_content` /
//!   `_build_hybrid_metrics` — the hybrid `ServerMetricsExportData` shape emitted
//!   via `orjson.dumps(..., OPT_INDENT_2)` with `exclude_none=True`. Pydantic
//!   field order is reproduced by an insertion-ordered map; two-space indent is
//!   matched by `serde_json::to_string_pretty` (`serde_json` runs with
//!   `preserve_order`).
//! - CSV: `aiperf/server_metrics/csv_exporter.py::_generate_content` /
//!   `_write_section` / `_write_info_section` — comment header lines terminated
//!   with `\n`, CSV rows with `\r\n` (Python `csv.writer` default), the
//!   `# schema_version: 1.0` line, per-type stat columns, union-find label
//!   column ordering, vertical clustering sort, and the transposed `_info`
//!   section.
//! - Unit display: `aiperf/server_metrics/units.py::infer_unit` (re-inferred here
//!   through `crate::server_metrics::infer_unit`) then
//!   `BaseMetricUnit.display_name` (member name lowercased, `_per_second`→`/s`),
//!   ported by [`display_unit`]. The native report's own `unit` field is
//!   deliberately not trusted — the Python compat path re-infers, so this sink
//!   does too, keeping byte parity.
//! - Data mapping: `aiperf/orchestrator/native_report.py::_project_server_metrics`
//!   describes how the native-v2 series/metadata map onto the Python models this
//!   sink reproduces without the Python round-trip.
//!
//! # Extension seam
//! Unit display policy is the one plausibly variable rule; it is isolated in
//! [`display_unit`] over `crate::server_metrics::infer_unit`. Everything else is
//! a fixed byte-for-byte contract against the two legacy artifacts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::{Map, Value};

use crate::export::{ExportConfig, Exporter, crlf_csv_writer, normalize_endpoint_display};
use crate::metrics_core::{
    MetricEntry, MetricSeries, NativeReport, ReportServerMetricsEndpointInfo,
    ReportServerMetricsMetadata, ReportStats, ReportTimeslice, ReportValue, Unit,
};
use crate::server_metrics::infer_unit;

#[cfg(test)]
mod tests;

/// Output filenames (joined onto the run's exclusive artifact directory).
const JSON_FILE: &str = "server_metrics_export.json";
const CSV_FILE: &str = "server_metrics_export.csv";

/// Hybrid JSON schema version, pinned to `ServerMetricsExportData.SCHEMA_VERSION`.
const JSON_SCHEMA_VERSION: &str = "1.1";
/// CSV comment-header schema version, pinned to the `# schema_version:` line.
const CSV_SCHEMA_VERSION: &str = "1.0";

/// Fixed percentile ladder shared by gauge stats and histogram estimates.
const PERCENTILES: [u32; 9] = [1, 5, 10, 25, 50, 75, 90, 95, 99];

/// Server-metrics summary export policy. The Python frontend projects these onto
/// the wire `cfg.export.server_metrics` block; an absent block decodes to
/// all-disabled defaults.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerMetricsExportConfig {
    /// Emit `server_metrics_export.json`.
    pub json: bool,
    /// Emit `server_metrics_export.csv`.
    pub csv: bool,
    /// AIPerf package version (`aiperf.__version__`) that generated the export,
    /// projected by the frontend. The native report carries only the Rust crate
    /// version (`0.0.0` in dev), so the frontend supplies the authoritative
    /// package version; when absent the sink falls back to the report's field.
    /// Rendered into the JSON `aiperf_version` field and the CSV `# aiperf_version:`
    /// comment header, matching `ServerMetricsExportData.aiperf_version`.
    #[serde(default)]
    pub aiperf_version: Option<String>,
    /// Benchmark run identity (UUID) shared across export formats. Rendered into
    /// the JSON `benchmark_id` field and the CSV `# benchmark_id:` header; absent
    /// when the frontend does not supply one.
    #[serde(default)]
    pub benchmark_id: Option<String>,
    /// User configuration used for this run (`cfg.model_dump(exclude_unset=True)`
    /// on the Python side). Emitted verbatim as the JSON `input_config` object.
    #[serde(default)]
    pub input_config: Value,
}

/// The server-metrics summary [`Exporter`] (JSON + CSV).
pub struct ServerMetricsExporter;

impl Exporter for ServerMetricsExporter {
    fn name(&self) -> &'static str {
        "server_metrics"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.server_metrics.json || cfg.server_metrics.csv
    }

    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let policy = &cfg.server_metrics;
        // Python raises `DataExporterDisabled` (skips both files) when no server
        // metrics were collected; mirror that as a clean no-op.
        let Some(meta) = report.summary.server_metrics.as_ref() else {
            return Ok(());
        };
        if report.server_metrics.is_empty() {
            return Ok(());
        }

        if policy.json {
            let content = build_json(report, meta, policy);
            std::fs::write(artifact_dir.join(JSON_FILE), content)?;
        }
        if policy.csv {
            let content = build_csv(report, policy);
            std::fs::write(artifact_dir.join(CSV_FILE), content)?;
        }
        Ok(())
    }
}

// ============================================================================
// Shared helpers
// ============================================================================

/// Returns the finite payload of a present report value, dropping non-finite
/// tails so `exclude_none`-style omission matches the Python compat path (a JSON
/// null becomes `None` and is dropped, never emitted).
fn finite(value: ReportValue) -> Option<f64> {
    match value {
        ReportValue::Finite(value) if value.is_finite() => Some(value),
        _ => None,
    }
}

/// Python `BaseMetricUnit.display_name`: enum member name lowercased with
/// `_per_second` rewritten to `/s`. The Python exporters serialize this string,
/// not the report's stable `Unit::as_str` spelling.
fn display_unit(name: &str, description: &str) -> Option<&'static str> {
    let description = (!description.is_empty()).then_some(description);
    infer_unit(name, description).map(|unit| match unit {
        Unit::Count => "count",
        Unit::Request => "requests",
        Unit::Token => "tokens",
        Unit::Byte => "bytes",
        Unit::Kilobyte => "kilobytes",
        Unit::Megabyte => "megabytes",
        Unit::Gigabyte => "gigabytes",
        Unit::Terabyte => "terabytes",
        Unit::Nanosecond => "nanoseconds",
        Unit::Microsecond => "microseconds",
        Unit::Millisecond => "milliseconds",
        Unit::Second => "seconds",
        Unit::Percent => "percent",
        Unit::Ratio => "ratio",
        Unit::RequestsPerSecond => "requests/s",
        Unit::TokensPerSecond => "tokens/s",
        Unit::TokensPerSecondPerUser => "tokens/s_per_user",
        Unit::ImagesPerSecond => "images/s",
        Unit::MillisecondsPerImage => "ms_per_image",
        Unit::VideosPerSecond => "videos/s",
        Unit::MillisecondsPerVideo => "ms_per_video",
        Unit::TokensPerJoule => "tokens_per_joule",
        Unit::JoulesPerUser => "joules_per_user",
        Unit::BytesPerSecond => "bytes/s",
        Unit::MegabytesPerSecond => "mb/s",
        Unit::GigabytesPerSecond => "gb/s",
        Unit::Watt => "watt",
        Unit::Milliwatt => "milliwatt",
        Unit::Joule => "joule",
        Unit::Millijoule => "millijoule",
        Unit::Megajoule => "megajoule",
        Unit::Hertz => "hertz",
        Unit::Megahertz => "megahertz",
        Unit::Gigahertz => "gigahertz",
        Unit::Celsius => "celsius",
        Unit::Kelvin => "kelvin",
        Unit::Fahrenheit => "fahrenheit",
        Unit::Image => "image",
        Unit::Frame => "frames",
        Unit::Video => "video",
    })
}

/// Original Prometheus semantic type for a family, preferring the metadata table
/// and falling back to the native stats shape (which cannot tell gauge from an
/// exporter-untyped scalar, defaulting to `gauge`).
fn prometheus_type(meta: &ReportServerMetricsMetadata, name: &str, entry: &MetricEntry) -> String {
    if let Some(kind) = meta.metric_types.get(name) {
        return kind.clone();
    }
    match entry.series.first().map(|series| &series.stats) {
        Some(ReportStats::Counter(_)) => "counter",
        Some(ReportStats::Histogram(_)) => "histogram",
        _ => "gauge",
    }
    .to_string()
}

/// Python `str(dict)` repr over a label map (BTreeMap iteration matches the
/// native report's sorted keys). Used only as a deterministic sort tie-break.
fn python_labels_repr(labels: Option<&BTreeMap<String, String>>) -> String {
    match labels {
        None => String::new(),
        Some(labels) if labels.is_empty() => "{}".to_string(),
        Some(labels) => {
            let mut out = String::from("{");
            for (index, (key, value)) in labels.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                out.push_str(&format!("'{key}': '{value}'"));
            }
            out.push('}');
            out
        }
    }
}

/// The two datetime rendering pieces are grouped so the fractional-second rule
/// (present only when microseconds are non-zero) can be unit-tested in isolation.
fn python_isoformat_from_ns(ns: i64) -> String {
    use chrono::{Local, TimeZone};
    // Mirror CPython `datetime.fromtimestamp(ns / 1e9)`: divide as f64, then
    // round the fractional part to whole microseconds.
    let seconds_f = ns as f64 / 1e9;
    let mut seconds = seconds_f.floor() as i64;
    let mut micros = ((seconds_f - seconds as f64) * 1e6).round() as i64;
    if micros >= 1_000_000 {
        seconds += 1;
        micros -= 1_000_000;
    }
    let naive = Local
        .timestamp_opt(seconds, (micros as u32) * 1_000)
        .single()
        .expect("valid local timestamp")
        .naive_local();
    isoformat_naive(naive, micros as u32)
}

/// Python `datetime.isoformat()` for a naive datetime: `YYYY-MM-DDTHH:MM:SS`,
/// plus a six-digit `.ffffff` fraction only when microseconds are non-zero.
fn isoformat_naive(naive: chrono::NaiveDateTime, micros: u32) -> String {
    let base = naive.format("%Y-%m-%dT%H:%M:%S").to_string();
    if micros == 0 {
        base
    } else {
        format!("{base}.{micros:06}")
    }
}

// ============================================================================
// JSON export
// ============================================================================

/// Builds the hybrid `server_metrics_export.json` content.
fn build_json(
    report: &NativeReport,
    meta: &ReportServerMetricsMetadata,
    policy: &ServerMetricsExportConfig,
) -> String {
    let mut root = Map::new();
    root.insert(
        "schema_version".into(),
        Value::String(JSON_SCHEMA_VERSION.into()),
    );
    // `aiperf_version` is the frontend-projected package version; the report's
    // own field (the Rust crate version) is a fallback only when the projection
    // is absent (e.g. unit tests). `None` would be dropped, but one of the two
    // always carries a version string.
    root.insert(
        "aiperf_version".into(),
        Value::String(
            policy
                .aiperf_version
                .clone()
                .unwrap_or_else(|| report.aiperf_version.clone()),
        ),
    );
    if let Some(benchmark_id) = &policy.benchmark_id {
        root.insert("benchmark_id".into(), Value::String(benchmark_id.clone()));
    }
    root.insert("summary".into(), build_json_summary(report, meta));
    // `metrics_phase` is kept at `profiling` for backward compatibility.
    root.insert("metrics_phase".into(), Value::String("profiling".into()));
    root.insert(
        "metrics".into(),
        build_json_metrics(&report.server_metrics, meta),
    );
    let warmup = build_json_metrics(&report.warmup_server_metrics, meta);
    if let Value::Object(map) = &warmup
        && !map.is_empty()
    {
        root.insert("warmup_metrics".into(), warmup);
    }
    root.insert(
        "input_config".into(),
        match &policy.input_config {
            Value::Null => Value::Object(Map::new()),
            other => other.clone(),
        },
    );

    let mut content = serde_json::to_string_pretty(&Value::Object(root))
        .expect("server-metrics JSON is always serializable");
    // orjson emits no trailing newline; `to_string_pretty` matches.
    content.truncate(content.trim_end().len());
    content
}

/// Builds the `summary` object (endpoints, phase datetimes, endpoint metadata).
fn build_json_summary(report: &NativeReport, meta: &ReportServerMetricsMetadata) -> Value {
    let mut summary = Map::new();
    summary.insert(
        "endpoints_configured".into(),
        Value::Array(
            meta.endpoints_configured
                .iter()
                .map(|url| Value::String(url.clone()))
                .collect(),
        ),
    );
    summary.insert(
        "endpoints_successful".into(),
        Value::Array(
            meta.endpoints_successful
                .iter()
                .map(|url| Value::String(url.clone()))
                .collect(),
        ),
    );
    let (start_ns, end_ns) = meta
        .profiling
        .as_ref()
        .map(|range| (range.start_ns, range.end_ns))
        .unwrap_or((0, 0));
    summary.insert(
        "start_time".into(),
        Value::String(python_isoformat_from_ns(start_ns)),
    );
    summary.insert(
        "end_time".into(),
        Value::String(python_isoformat_from_ns(end_ns)),
    );

    // Endpoint collection metadata is keyed by the endpoints that actually
    // contributed a profiling series (`_build_hybrid_metrics`), sorted by URL.
    let mut endpoints: BTreeSet<&str> = BTreeSet::new();
    for entry in report.server_metrics.values() {
        for series in &entry.series {
            if let Some(url) = &series.endpoint_url {
                endpoints.insert(url.as_str());
            }
        }
    }
    let mut endpoint_info = Map::new();
    for url in &endpoints {
        if let Some(info) = meta.endpoint_info.get(*url) {
            endpoint_info.insert((*url).to_string(), build_json_endpoint_info(info));
        }
    }
    if !endpoint_info.is_empty() {
        summary.insert("endpoint_info".into(), Value::Object(endpoint_info));
    }

    let mut phase_ranges = Map::new();
    if let Some(range) = &meta.profiling
        && range.start_ns < range.end_ns
    {
        phase_ranges.insert(
            "profiling".into(),
            phase_range_json(range.start_ns, range.end_ns),
        );
    }
    if let Some(range) = &meta.warmup
        && range.start_ns < range.end_ns
    {
        phase_ranges.insert(
            "warmup".into(),
            phase_range_json(range.start_ns, range.end_ns),
        );
    }
    if !phase_ranges.is_empty() {
        summary.insert("phase_time_ranges".into(), Value::Object(phase_ranges));
    }

    Value::Object(summary)
}

fn phase_range_json(start_ns: i64, end_ns: i64) -> Value {
    let mut range = Map::new();
    range.insert("start_ns".into(), Value::from(start_ns));
    range.insert("end_ns".into(), Value::from(end_ns));
    Value::Object(range)
}

fn build_json_endpoint_info(info: &ReportServerMetricsEndpointInfo) -> Value {
    let mut map = Map::new();
    map.insert("total_fetches".into(), Value::from(info.total_fetches));
    map.insert("first_fetch_ns".into(), Value::from(info.first_fetch_ns));
    map.insert("last_fetch_ns".into(), Value::from(info.last_fetch_ns));
    insert_f64(&mut map, "avg_fetch_latency_ms", info.avg_fetch_latency_ms);
    map.insert("unique_updates".into(), Value::from(info.unique_updates));
    map.insert("first_update_ns".into(), Value::from(info.first_update_ns));
    map.insert("last_update_ns".into(), Value::from(info.last_update_ns));
    insert_f64(&mut map, "duration_seconds", info.duration_seconds);
    insert_f64(
        &mut map,
        "avg_update_interval_ms",
        info.avg_update_interval_ms,
    );
    if let Some(median) = info.median_update_interval_ms {
        insert_f64(&mut map, "median_update_interval_ms", median);
    }
    Value::Object(map)
}

/// Builds the metrics-first `metrics` / `warmup_metrics` object.
fn build_json_metrics(
    metrics: &BTreeMap<String, MetricEntry>,
    meta: &ReportServerMetricsMetadata,
) -> Value {
    let mut out = Map::new();
    for (name, entry) in metrics {
        let kind = prometheus_type(meta, name, entry);
        let description = meta.descriptions.get(name).cloned().unwrap_or_default();
        let mut metric = Map::new();
        metric.insert("type".into(), Value::String(kind));
        metric.insert("description".into(), Value::String(description.clone()));
        if let Some(unit) = display_unit(name, &description) {
            metric.insert("unit".into(), Value::String(unit.to_string()));
        }

        let mut series: Vec<&MetricSeries> = entry.series.iter().collect();
        series.sort_by(|left, right| {
            let left_key = (
                left.endpoint_url.clone().unwrap_or_default(),
                python_labels_repr(left.labels.as_ref()),
            );
            let right_key = (
                right.endpoint_url.clone().unwrap_or_default(),
                python_labels_repr(right.labels.as_ref()),
            );
            left_key.cmp(&right_key)
        });
        metric.insert(
            "series".into(),
            Value::Array(
                series
                    .iter()
                    .map(|series| build_json_series(series))
                    .collect(),
            ),
        );
        out.insert(name.clone(), Value::Object(metric));
    }
    Value::Object(out)
}

fn build_json_series(series: &MetricSeries) -> Value {
    let mut out = Map::new();
    if let Some(url) = &series.endpoint_url {
        out.insert("endpoint_url".into(), Value::String(url.clone()));
    }
    if let Some(labels) = &series.labels {
        out.insert("labels".into(), labels_json(labels));
    }
    match &series.stats {
        ReportStats::Counter(counter) => {
            let mut stats = Map::new();
            if let Some(total) = finite(counter.total) {
                insert_f64(&mut stats, "total", total);
            }
            if let Some(rate) = counter.rate.and_then(finite) {
                insert_f64(&mut stats, "rate", rate);
            }
            out.insert("stats".into(), Value::Object(stats));
            if let Some(slices) = counter_timeslices(&series.timeslices) {
                out.insert("timeslices".into(), slices);
            }
        }
        ReportStats::Histogram(histogram) => {
            let mut stats = Map::new();
            stats.insert("count".into(), Value::from(histogram.count));
            if let Some(sum) = finite(histogram.sum) {
                insert_f64(&mut stats, "sum", sum);
            }
            if let Some(avg) = histogram.avg.and_then(finite) {
                insert_f64(&mut stats, "avg", avg);
            }
            if let Some(rate) = histogram.count_rate.and_then(finite) {
                insert_f64(&mut stats, "count_rate", rate);
            }
            if let Some(rate) = histogram.sum_rate.and_then(finite) {
                insert_f64(&mut stats, "sum_rate", rate);
            }
            for percentile in PERCENTILES {
                if let Some(value) = histogram
                    .percentiles
                    .get(&format!("p{percentile}"))
                    .copied()
                    .and_then(finite)
                {
                    insert_f64(&mut stats, &format!("p{percentile}_estimate"), value);
                }
            }
            out.insert("stats".into(), Value::Object(stats));
            if !histogram.buckets.is_empty() {
                out.insert("buckets".into(), buckets_json(&histogram.buckets));
            }
            if let Some(slices) = histogram_timeslices(&series.timeslices) {
                out.insert("timeslices".into(), slices);
            }
        }
        ReportStats::Distribution(dist) => {
            let mut stats = Map::new();
            for (key, value) in [
                ("avg", dist.avg),
                ("min", dist.min),
                ("max", dist.max),
                ("std", dist.std),
            ] {
                if let Some(value) = value.and_then(finite) {
                    insert_f64(&mut stats, key, value);
                }
            }
            for percentile in PERCENTILES {
                if let Some(value) = dist
                    .percentiles
                    .get(&format!("p{percentile}"))
                    .copied()
                    .and_then(finite)
                {
                    insert_f64(&mut stats, &format!("p{percentile}"), value);
                }
            }
            out.insert("stats".into(), Value::Object(stats));
            if let Some(slices) = gauge_timeslices(&series.timeslices) {
                out.insert("timeslices".into(), slices);
            }
        }
        ReportStats::Scalar(_) => {
            // Server-metric series are never scalar; emit an empty stats object
            // rather than panicking on an out-of-contract report.
            out.insert("stats".into(), Value::Object(Map::new()));
        }
    }
    Value::Object(out)
}

fn labels_json(labels: &BTreeMap<String, String>) -> Value {
    Value::Object(
        labels
            .iter()
            .map(|(key, value)| (key.clone(), Value::String(value.clone())))
            .collect(),
    )
}

fn buckets_json(buckets: &BTreeMap<String, u64>) -> Value {
    Value::Object(
        buckets
            .iter()
            .map(|(bound, count)| (bound.clone(), Value::from(*count)))
            .collect(),
    )
}

fn gauge_timeslices(slices: &[ReportTimeslice]) -> Option<Value> {
    let projected: Vec<Value> = slices
        .iter()
        .filter_map(|slice| {
            let ReportStats::Distribution(dist) = &slice.stats else {
                return None;
            };
            let avg = dist.avg.and_then(finite)?;
            let min = dist.min.and_then(finite)?;
            let max = dist.max.and_then(finite)?;
            let mut map = timeslice_head(slice);
            insert_f64(&mut map, "avg", avg);
            insert_f64(&mut map, "min", min);
            insert_f64(&mut map, "max", max);
            Some(Value::Object(map))
        })
        .collect();
    (!projected.is_empty()).then_some(Value::Array(projected))
}

fn counter_timeslices(slices: &[ReportTimeslice]) -> Option<Value> {
    let projected: Vec<Value> = slices
        .iter()
        .filter_map(|slice| {
            let ReportStats::Counter(counter) = &slice.stats else {
                return None;
            };
            let total = finite(counter.total)?;
            let rate = counter.rate.and_then(finite)?;
            let mut map = timeslice_head(slice);
            insert_f64(&mut map, "total", total);
            insert_f64(&mut map, "rate", rate);
            Some(Value::Object(map))
        })
        .collect();
    (!projected.is_empty()).then_some(Value::Array(projected))
}

fn histogram_timeslices(slices: &[ReportTimeslice]) -> Option<Value> {
    let projected: Vec<Value> = slices
        .iter()
        .filter_map(|slice| {
            let ReportStats::Histogram(histogram) = &slice.stats else {
                return None;
            };
            let sum = finite(histogram.sum)?;
            let avg = histogram.avg.and_then(finite)?;
            let mut map = timeslice_head(slice);
            map.insert("count".into(), Value::from(histogram.count));
            insert_f64(&mut map, "sum", sum);
            insert_f64(&mut map, "avg", avg);
            if !histogram.buckets.is_empty() {
                map.insert("buckets".into(), buckets_json(&histogram.buckets));
            }
            Some(Value::Object(map))
        })
        .collect();
    (!projected.is_empty()).then_some(Value::Array(projected))
}

/// Common timeslice head: `start_ns`, `end_ns`, and `is_complete` only when the
/// slice is partial (`None if complete else False`).
fn timeslice_head(slice: &ReportTimeslice) -> Map<String, Value> {
    let mut map = Map::new();
    map.insert("start_ns".into(), Value::from(slice.start_ns));
    map.insert("end_ns".into(), Value::from(slice.end_ns));
    if !slice.complete {
        map.insert("is_complete".into(), Value::Bool(false));
    }
    map
}

/// Inserts a finite f64. `serde_json::Number::from_f64` never fails for finite
/// input; a non-finite slip becomes JSON null, matching `scrub_non_finite`.
fn insert_f64(map: &mut Map<String, Value>, key: &str, value: f64) {
    let number = serde_json::Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null);
    map.insert(key.to_string(), number);
}

// ============================================================================
// CSV export
// ============================================================================

/// One CSV row's worth of metric facts (one native series).
struct CsvMetricInfo<'a> {
    endpoint: String,
    metric_name: &'a str,
    description: &'a str,
    unit: Option<&'static str>,
    stats: &'a ReportStats,
    labels: Option<&'a BTreeMap<String, String>>,
}

impl CsvMetricInfo<'_> {
    fn is_info_metric(&self) -> bool {
        self.metric_name.ends_with("_info")
    }
}

/// Gauge/unknown stat columns (model field order).
const GAUGE_STAT_KEYS: [&str; 13] = [
    "avg", "min", "max", "std", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99",
];
/// Counter stat columns.
const COUNTER_STAT_KEYS: [&str; 6] = [
    "total", "rate", "rate_avg", "rate_min", "rate_max", "rate_std",
];
/// Histogram stat columns (note the distinct order from the JSON stats shape).
const HISTOGRAM_STAT_KEYS: [&str; 14] = [
    "count",
    "count_rate",
    "sum",
    "sum_rate",
    "avg",
    "p1_estimate",
    "p5_estimate",
    "p10_estimate",
    "p25_estimate",
    "p50_estimate",
    "p75_estimate",
    "p90_estimate",
    "p95_estimate",
    "p99_estimate",
];

fn stat_keys(kind: &str) -> &'static [&'static str] {
    match kind {
        "counter" => &COUNTER_STAT_KEYS,
        "histogram" => &HISTOGRAM_STAT_KEYS,
        _ => &GAUGE_STAT_KEYS,
    }
}

/// A single CSV cell value awaiting `_format_number`.
enum StatCell {
    Absent,
    Int(u64),
    Float(f64),
}

impl StatCell {
    /// Python `_format_number`: `None`→"", int→exact, real→four decimals.
    fn format(&self) -> String {
        match self {
            StatCell::Absent => String::new(),
            StatCell::Int(value) => format!("{value}"),
            StatCell::Float(value) => format!("{value:.4}"),
        }
    }
}

fn csv_stat(stats: &ReportStats, key: &str) -> StatCell {
    match stats {
        ReportStats::Distribution(dist) => {
            let value = match key {
                "avg" => dist.avg,
                "min" => dist.min,
                "max" => dist.max,
                "std" => dist.std,
                other => dist.percentiles.get(other).copied(),
            };
            value
                .and_then(finite)
                .map_or(StatCell::Absent, StatCell::Float)
        }
        ReportStats::Counter(counter) => match key {
            "total" => finite(counter.total).map_or(StatCell::Absent, StatCell::Float),
            "rate" => counter
                .rate
                .and_then(finite)
                .map_or(StatCell::Absent, StatCell::Float),
            _ => StatCell::Absent,
        },
        ReportStats::Histogram(histogram) => match key {
            "count" => StatCell::Int(histogram.count),
            "count_rate" => histogram
                .count_rate
                .and_then(finite)
                .map_or(StatCell::Absent, StatCell::Float),
            "sum" => finite(histogram.sum).map_or(StatCell::Absent, StatCell::Float),
            "sum_rate" => histogram
                .sum_rate
                .and_then(finite)
                .map_or(StatCell::Absent, StatCell::Float),
            "avg" => histogram
                .avg
                .and_then(finite)
                .map_or(StatCell::Absent, StatCell::Float),
            estimate => {
                let percentile = estimate.strip_suffix("_estimate").unwrap_or(estimate);
                histogram
                    .percentiles
                    .get(percentile)
                    .copied()
                    .and_then(finite)
                    .map_or(StatCell::Absent, StatCell::Float)
            }
        },
        ReportStats::Scalar(_) => StatCell::Absent,
    }
}

/// Builds the full `server_metrics_export.csv` content.
fn build_csv(report: &NativeReport, policy: &ServerMetricsExportConfig) -> String {
    // Comment header lines use `\n`; the CSV body below uses `\r\n`.
    let mut content = String::new();
    content.push_str("# AIPerf Server Metrics Export (CSV)\n");
    content.push_str(&format!(
        "# aiperf_version: {}\n",
        policy
            .aiperf_version
            .as_deref()
            .unwrap_or(report.aiperf_version.as_str())
    ));
    content.push_str(&format!("# schema_version: {CSV_SCHEMA_VERSION}\n"));
    content.push_str(&format!(
        "# benchmark_id: {}\n",
        policy.benchmark_id.as_deref().unwrap_or("None")
    ));
    content.push_str("# Note: Same benchmark_id and version appear in JSON and Parquet exports\n");
    content.push_str("#\n");

    let meta = report
        .summary
        .server_metrics
        .as_ref()
        .expect("caller guarantees server-metrics metadata is present");

    // Group every series into its Prometheus type bucket.
    let mut by_type: BTreeMap<String, Vec<CsvMetricInfo<'_>>> = BTreeMap::new();
    for (name, entry) in &report.server_metrics {
        let kind = prometheus_type(meta, name, entry);
        let description = meta
            .descriptions
            .get(name)
            .map(String::as_str)
            .unwrap_or("");
        let unit = display_unit(name, description);
        for series in &entry.series {
            let endpoint = series
                .endpoint_url
                .as_deref()
                .map(normalize_endpoint_display)
                .unwrap_or_default();
            by_type
                .entry(kind.clone())
                .or_default()
                .push(CsvMetricInfo {
                    endpoint,
                    metric_name: name,
                    description,
                    unit,
                    stats: &series.stats,
                    labels: series.labels.as_ref(),
                });
        }
    }

    // Pull info metrics out of the gauge bucket into the transposed section.
    let mut info_metrics: Vec<CsvMetricInfo<'_>> = Vec::new();
    if let Some(gauges) = by_type.get_mut("gauge") {
        let (info, kept): (Vec<_>, Vec<_>) = std::mem::take(gauges)
            .into_iter()
            .partition(|m| m.is_info_metric());
        info_metrics = info;
        *gauges = kept;
        if gauges.is_empty() {
            by_type.remove("gauge");
        }
    }

    // Each section is serialized independently, then joined by a blank `\r\n`
    // line (Python's `writer.writerow([])` between sections); the csv crate
    // cannot emit a bare zero-field record without quoting it as `""`.
    let mut sections: Vec<Vec<u8>> = Vec::new();
    for kind in ["gauge", "counter", "histogram", "unknown"] {
        if let Some(metrics) = by_type.get(kind) {
            sections.push(render_section(kind, metrics));
        }
    }
    if !info_metrics.is_empty() {
        sections.push(render_info_section(&info_metrics));
    }

    let body = sections.join(&b"\r\n"[..]);
    content.push_str(&String::from_utf8(body).expect("csv is valid utf-8"));
    content
}

/// Serializes one type section to CSV bytes (CRLF-terminated rows).
fn render_section(kind: &str, metrics: &[CsvMetricInfo<'_>]) -> Vec<u8> {
    let mut writer = crlf_csv_writer(Vec::new());
    write_section(&mut writer, kind, metrics);
    writer.into_inner().expect("csv writer flush")
}

/// Serializes the transposed info section to CSV bytes.
fn render_info_section(metrics: &[CsvMetricInfo<'_>]) -> Vec<u8> {
    let mut writer = crlf_csv_writer(Vec::new());
    write_info_section(&mut writer, metrics);
    writer.into_inner().expect("csv writer flush")
}

fn write_section<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    kind: &str,
    metrics: &[CsvMetricInfo<'_>],
) {
    let keys = stat_keys(kind);
    let label_order = optimal_label_order(metrics);

    let mut header: Vec<String> = vec![
        "Endpoint".into(),
        "Type".into(),
        "Metric".into(),
        "Unit".into(),
    ];
    header.extend(keys.iter().map(|key| (*key).to_string()));
    header.extend(label_order.iter().cloned());
    header.push("Description".into());
    let is_histogram = kind == "histogram";
    if is_histogram {
        header.push("buckets".into());
    }
    writer.write_record(&header).expect("csv header row");

    let mut ordered: Vec<&CsvMetricInfo<'_>> = metrics.iter().collect();
    ordered.sort_by(|left, right| {
        vertical_sort_key(left, &label_order).cmp(&vertical_sort_key(right, &label_order))
    });

    for metric in ordered {
        let is_info = metric.is_info_metric();
        let mut row: Vec<String> = vec![
            metric.endpoint.clone(),
            kind.to_string(),
            metric.metric_name.to_string(),
            metric.unit.unwrap_or("").to_string(),
        ];
        for key in keys {
            row.push(csv_stat(metric.stats, key).format());
        }
        let empty = BTreeMap::new();
        let labels = metric.labels.unwrap_or(&empty);
        for label_key in &label_order {
            if is_info {
                row.push(String::new());
            } else {
                row.push(labels.get(label_key).cloned().unwrap_or_default());
            }
        }
        row.push(metric.description.to_string());
        if is_histogram {
            row.push(histogram_buckets_cell(metric.stats));
        }
        writer.write_record(&row).expect("csv data row");
    }
}

/// Histogram `buckets` column: `key=value` joined by `;` in bucket-map order.
fn histogram_buckets_cell(stats: &ReportStats) -> String {
    let ReportStats::Histogram(histogram) = stats else {
        return String::new();
    };
    histogram
        .buckets
        .iter()
        .map(|(bound, count)| format!("{bound}={}", StatCell::Int(*count).format()))
        .collect::<Vec<_>>()
        .join(";")
}

fn write_info_section<W: std::io::Write>(
    writer: &mut csv::Writer<W>,
    metrics: &[CsvMetricInfo<'_>],
) {
    writer
        .write_record(["Endpoint", "Metric", "Key", "Value", "Description"])
        .expect("csv info header");
    let mut ordered: Vec<&CsvMetricInfo<'_>> = metrics.iter().collect();
    ordered.sort_by(|left, right| {
        (left.metric_name, &left.endpoint).cmp(&(right.metric_name, &right.endpoint))
    });
    for metric in ordered {
        let empty = BTreeMap::new();
        let labels = metric.labels.unwrap_or(&empty);
        for (key, value) in labels {
            writer
                .write_record([
                    metric.endpoint.as_str(),
                    metric.metric_name,
                    key,
                    value,
                    metric.description,
                ])
                .expect("csv info row");
        }
    }
}

/// Union-find label-column ordering: exclusive labels before shared "bridge"
/// labels within each co-occurrence family, families ordered by their minimum
/// member. Ports `_get_optimal_label_order`.
fn optimal_label_order(metrics: &[CsvMetricInfo<'_>]) -> Vec<String> {
    let label_sets: Vec<BTreeSet<String>> = metrics
        .iter()
        .filter(|metric| !metric.is_info_metric())
        .filter_map(|metric| metric.labels)
        .filter(|labels| !labels.is_empty())
        .map(|labels| labels.keys().cloned().collect())
        .collect();

    let mut all_labels: BTreeSet<String> = BTreeSet::new();
    for set in &label_sets {
        all_labels.extend(set.iter().cloned());
    }
    if all_labels.is_empty() {
        return Vec::new();
    }

    let mut parent: BTreeMap<String, String> = BTreeMap::new();
    fn find(parent: &mut BTreeMap<String, String>, node: &str) -> String {
        let current = parent
            .entry(node.to_string())
            .or_insert_with(|| node.to_string())
            .clone();
        if current == node {
            node.to_string()
        } else {
            let root = find(parent, &current);
            parent.insert(node.to_string(), root.clone());
            root
        }
    }
    for set in &label_sets {
        let sorted: Vec<&String> = set.iter().collect();
        for pair in sorted.windows(2) {
            let root_b = find(&mut parent, pair[1]);
            let root_a = find(&mut parent, pair[0]);
            parent.insert(root_b, root_a);
        }
    }

    let mut families: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for label in &all_labels {
        let root = find(&mut parent, label);
        families.entry(root).or_default().insert(label.clone());
    }
    let bridges: BTreeSet<String> = all_labels
        .iter()
        .filter(|label| label_sets.iter().filter(|set| set.contains(*label)).count() > 1)
        .cloned()
        .collect();

    let mut family_vec: Vec<BTreeSet<String>> = families.into_values().collect();
    family_vec.sort_by(|left, right| left.iter().min().cmp(&right.iter().min()));

    let mut result = Vec::new();
    for family in family_vec {
        for label in &family {
            if !bridges.contains(label) {
                result.push(label.clone());
            }
        }
        for label in &family {
            if bridges.contains(label) {
                result.push(label.clone());
            }
        }
    }
    result
}

/// Vertical clustering key: fill-pattern bitmap over the label columns, then
/// name / endpoint / label-repr. Ports `_get_vertical_sort_key`.
fn vertical_sort_key(
    metric: &CsvMetricInfo<'_>,
    label_order: &[String],
) -> (String, String, String, String) {
    let empty = BTreeMap::new();
    let labels = metric.labels.unwrap_or(&empty);
    let pattern: String = label_order
        .iter()
        .map(|column| {
            if labels.contains_key(column) {
                '1'
            } else {
                '0'
            }
        })
        .collect();
    let labels_repr = if labels.is_empty() {
        String::new()
    } else {
        python_labels_repr(Some(labels))
    };
    (
        pattern,
        metric.metric_name.to_string(),
        metric.endpoint.clone(),
        labels_repr,
    )
}
