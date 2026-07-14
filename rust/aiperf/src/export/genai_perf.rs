// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AIPerf v1 summary-export compatibility sink.
//!
//! Emits the two canonical AIPerf summary artifacts — `<stem>_aiperf.json`
//! (`schema_version = "1.4"`) and `<stem>_aiperf.csv` — byte-for-byte identical
//! to the Python exporters this sink replaces, so downstream plotters, uploaders,
//! and the multi-run search layer keep working across the native transition. The
//! parity oracle is the AIPerf Python exporter suite, NOT the external NVIDIA
//! genai-perf tool; the [`Exporter`] type name is retained only because the
//! foundation wired the registry against it.
//!
//! # Byte-exact grounding (Python `path:line`, main checkout `src/aiperf/`)
//! - JSON serialization: `exporters/metrics_json_exporter.py:109-114` —
//!   `model_dump(mode="json", exclude_unset=True, exclude_none=True)` then
//!   `orjson.dumps(scrub_non_finite(payload), OPT_INDENT_2)`. serde_json's
//!   `to_string_pretty` is byte-identical to orjson `OPT_INDENT_2` (2-space
//!   indent, shortest-round-trip float repr, no trailing newline); the `aiperf`
//!   crate's `serde_json` carries the `preserve_order` feature so an insertion-
//!   ordered [`serde_json::Map`] reproduces Pydantic field order exactly.
//! - Top-level key order: `models/export_models.py:293-349` (`JsonExportData`
//!   field declaration order). Declared metric fields precede undeclared
//!   ("extra") metric tags, which Pydantic appends last in dict-insertion order.
//! - Per-metric object key order: `models/export_models.py:36-66`
//!   (`JsonMetricResult`): `unit, avg, p1, p5, p10, p25, p50, p75, p90, p95,
//!   p99, min, max, std, count, sum`. `count` is dropped for AGGREGATE/DERIVED
//!   scalars (`record_models.py:99-123` `to_json_result`).
//! - Value shapes per native metric type: `orchestrator/native_report.py:809-840`
//!   (`_legacy_stats`): distribution → count/avg/min/max/std/percentiles;
//!   scalar → avg=min=max=value; counter → avg=min=max=sum=total; histogram →
//!   count/sum/avg/percentiles.
//! - Non-finite discipline: `common/finite.py` + the null round-trip. A native
//!   `ReportValue::NonFinite` serializes to JSON `null` in the native-v2 report,
//!   which Python reads back as `None` (`native_report.py:862-865`
//!   `_optional_number`), so `exclude_none` drops it. This sink therefore treats
//!   `NonFinite` as ABSENT (omitted from JSON, empty string in CSV) to match the
//!   Python output, not as a present `null`.
//! - INTERNAL/EXPERIMENTAL filtering: `exporters/metrics_base_exporter.py:30-65`
//!   (`_prepare_metrics`) drops those flag classes from both artifacts.
//! - CSV layout: `exporters/metrics_csv_exporter.py` — two sections split on
//!   percentile presence (`_has_percentiles`), rows sorted by tag, request
//!   header `["Metric", *STAT_KEYS]`, a blank row between sections, system
//!   header `["Metric","Value"]`. `STAT_KEYS` order is `constants.py:23-38`.
//!   `_format_number` renders `None`→`""`, floats via `f"{v:.2f}"`.
//!   `_format_metric_name` appends ` (unit)` unless the unit is `count`/`requests`.
//!
//! # Extension seam
//! Every new summary artifact is a new [`Exporter`]; this module owns only the
//! v1 JSON/CSV projection and stays a pure function of the finalized
//! [`NativeReport`].

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use crate::export::{ExportConfig, Exporter};
use crate::metrics_core::catalog::{CATALOG, MetricSpec};
use crate::metrics_core::{
    MetricEntry, MetricFlags, MetricSeries, MetricType, NativeReport, ReportStats, ReportValue,
};
use serde_json::{Map, Value};

/// AIPerf v1 summary-export policy. Disabled unless the frontend requests the
/// `json` summary; `stem` is the profile-export filename stem
/// (`profile_export` → `profile_export_aiperf.{json,csv}`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenaiPerfExportConfig {
    /// Emit the v1 summary JSON/CSV artifacts.
    pub enabled: bool,
    /// Filename stem for the compat artifacts (before the `_aiperf` suffix).
    pub stem: String,
    /// Whether goodput was requested. Retained for wire compatibility with the
    /// frontend projection and surfaced into the projected `input_config`; the
    /// v1 summary emits whatever goodput metrics the native report carries.
    pub goodput: bool,
    /// Whether the run streamed. Retained for wire compatibility and surfaced
    /// into the projected `input_config`; the v1 summary emits whatever
    /// streaming metrics the native report carries.
    pub streaming: bool,
    /// Endpoint type string (e.g. `chat`, `embeddings`). Retained for wire
    /// compatibility and surfaced into the projected `input_config`.
    pub endpoint_type: String,
}

impl Default for GenaiPerfExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stem: "profile_export".to_owned(),
            goodput: false,
            streaming: false,
            endpoint_type: String::new(),
        }
    }
}

/// JSON export schema version. Pinned to the Python `JsonExportData.SCHEMA_VERSION`
/// (`export_models.py:291`), independent of the native-v2 report schema (`2.0`).
const JSON_SCHEMA_VERSION: &str = "1.4";

/// `JsonExportData` declared metric-field order (`export_models.py:305-329`).
/// Declared metrics serialize in this order; any report metric not listed here
/// is an "extra" appended after all declared fields in alphabetical (native
/// report `BTreeMap`) order, matching Pydantic `extra="allow"` semantics.
const JSON_METRIC_ORDER: &[&str] = &[
    "request_throughput",
    "request_latency",
    "request_count",
    "time_to_first_token",
    "time_to_second_token",
    "inter_token_latency",
    "output_token_throughput",
    "output_token_throughput_per_user",
    "output_sequence_length",
    "input_sequence_length",
    "goodput",
    "good_request_count",
    "output_token_count",
    "reasoning_token_count",
    "min_request_timestamp",
    "max_response_timestamp",
    "inter_chunk_latency",
    "total_output_tokens",
    "total_reasoning_tokens",
    "benchmark_duration",
    "total_isl",
    "total_osl",
    "error_request_count",
    "error_isl",
    "total_error_isl",
];

/// Percentile labels in `JsonMetricResult` declaration order
/// (`export_models.py:39-46`). Also the ascending order fetched from the native
/// distribution stats map.
const PERCENTILE_LABELS: [&str; 9] = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];

/// CSV request-section per-row stat order (`constants.py:23-38` `STAT_KEYS`).
const STAT_KEYS: [&str; 14] = [
    "avg", "min", "max", "sum", "p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "std",
];

/// The AIPerf v1 summary compat [`Exporter`].
pub struct GenaiPerfV1Exporter;

impl Exporter for GenaiPerfV1Exporter {
    fn name(&self) -> &'static str {
        "genai_perf_v1"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.genai_perf.enabled
    }

    fn export(
        &self,
        report: &NativeReport,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let stem = &cfg.genai_perf.stem;
        // Path-traversal guard: the stem is a filename stem, never a path.
        if stem.is_empty()
            || stem.contains('/')
            || stem.contains('\\')
            || stem.contains("..")
            || stem.contains('\0')
        {
            anyhow::bail!("invalid genai-perf export stem {stem:?}: must be a bare filename stem");
        }

        let json = render_json(report, cfg);
        let csv = render_csv(report)?;

        std::fs::write(artifact_dir.join(format!("{stem}_aiperf.json")), json)?;
        std::fs::write(artifact_dir.join(format!("{stem}_aiperf.csv")), csv)?;
        Ok(())
    }
}

/// Look up a metric's catalog spec by its report name (`tag`). The native report
/// keys are the catalog tag strings; tags outside the catalog (dynamically
/// injected metrics) return `None` and are kept unfiltered with the tag as
/// their header, mirroring `native_report.py:_metric_result`.
fn spec_for_name(name: &str) -> Option<&'static MetricSpec> {
    CATALOG.iter().find(|spec| spec.tag.as_str() == name)
}

/// One report metric projected into the flat v1 stat set. Field presence
/// mirrors `_legacy_stats` + `to_json_result`; `None` means "absent" (never a
/// present JSON `null`, since the Python pipeline collapses non-finite tails to
/// `None` before `exclude_none`).
struct Projected {
    header: String,
    unit: String,
    avg: Option<f64>,
    percentiles: [Option<f64>; 9],
    min: Option<f64>,
    max: Option<f64>,
    std: Option<f64>,
    count: Option<u64>,
    sum: Option<f64>,
}

impl Projected {
    /// True when any percentile is present — the CSV request/system split
    /// predicate (`metrics_csv_exporter.py:_has_percentiles`).
    fn has_percentiles(&self) -> bool {
        self.percentiles.iter().any(Option::is_some)
    }
}

/// A finite value is present; a non-finite or absent value is `None`. This is
/// the single choke point enforcing the null→None→exclude_none contract.
fn finite(value: Option<ReportValue>) -> Option<f64> {
    match value {
        Some(ReportValue::Finite(number)) if number.is_finite() => Some(number),
        _ => None,
    }
}

/// Select the summary series for a metric (`native_report.py:791-806`
/// `_summary_series`): the lone series, or the single unlabeled aggregate when
/// several labeled series exist. `None` skips the metric entirely.
fn summary_series(entry: &MetricEntry) -> Option<&MetricSeries> {
    match entry.series.as_slice() {
        [] => None,
        [only] => Some(only),
        many => {
            let mut unlabeled = many.iter().filter(|series| series.labels.is_none());
            let first = unlabeled.next();
            // Multiple unlabeled aggregates are ambiguous; Python raises. Be
            // graceful and skip rather than abort the best-effort export.
            if unlabeled.next().is_some() {
                return None;
            }
            first
        }
    }
}

/// Project one report metric into the flat stat set, applying the native
/// metric-type value shape and the `count`-drop rule for scalar-tier metrics.
/// Returns `None` when the metric has no usable series or a required scalar /
/// counter value is non-finite (Python would raise; we skip).
fn project(name: &str, entry: &MetricEntry, spec: Option<&MetricSpec>) -> Option<Projected> {
    let series = summary_series(entry)?;
    let header = spec.map_or_else(|| name.to_owned(), |spec| spec.header.to_owned());
    let unit = entry.unit.clone();

    let mut projected = Projected {
        header,
        unit,
        avg: None,
        percentiles: [None; 9],
        min: None,
        max: None,
        std: None,
        count: None,
        sum: None,
    };

    match &series.stats {
        ReportStats::Distribution(stats) => {
            projected.avg = finite(stats.avg);
            projected.min = finite(stats.min);
            projected.max = finite(stats.max);
            projected.std = finite(stats.std);
            projected.count = stats.count.map(|count| count as u64);
            for (index, label) in PERCENTILE_LABELS.iter().enumerate() {
                projected.percentiles[index] = finite(stats.percentiles.get(*label).copied());
            }
        }
        ReportStats::Scalar(stats) => {
            let value = finite(Some(stats.value))?;
            projected.avg = Some(value);
            projected.min = Some(value);
            projected.max = Some(value);
        }
        ReportStats::Counter(stats) => {
            let total = finite(Some(stats.total))?;
            projected.avg = Some(total);
            projected.min = Some(total);
            projected.max = Some(total);
            projected.sum = Some(total);
        }
        ReportStats::Histogram(stats) => {
            projected.count = Some(stats.count);
            projected.sum = finite(Some(stats.sum));
            projected.avg = stats.avg.and_then(|avg| finite(Some(avg)));
            for (index, label) in PERCENTILE_LABELS.iter().enumerate() {
                projected.percentiles[index] = finite(stats.percentiles.get(*label).copied());
            }
        }
    }

    // `to_json_result` (record_models.py:99-123) drops `count` for AGGREGATE /
    // DERIVED (scalar-tier) metrics, where it would trivially be 1.
    if let Some(spec) = spec
        && matches!(spec.kind, MetricType::Aggregate | MetricType::Derived)
    {
        projected.count = None;
    }

    Some(projected)
}

/// Whether a metric is excluded from file exports (`_prepare_metrics`): INTERNAL
/// and EXPERIMENTAL flag classes are dropped (dev show-flags are off on the
/// native path). Tags outside the catalog are always kept.
fn is_filtered(spec: Option<&MetricSpec>) -> bool {
    spec.is_some_and(|spec| {
        spec.flags
            .intersects(MetricFlags::INTERNAL | MetricFlags::EXPERIMENTAL)
    })
}

/// Collect the exportable metrics in report (alphabetical `BTreeMap`) order,
/// after filtering and projection.
fn collect_metrics(metrics: &BTreeMap<String, MetricEntry>) -> Vec<(String, Projected)> {
    metrics
        .iter()
        .filter_map(|(name, entry)| {
            let spec = spec_for_name(name);
            if is_filtered(spec) {
                return None;
            }
            project(name, entry, spec).map(|projected| (name.clone(), projected))
        })
        .collect()
}

/// Build one metric's JSON object in `JsonMetricResult` field order, omitting
/// absent fields (`exclude_none`).
fn metric_object(projected: &Projected) -> Value {
    let mut object = Map::new();
    object.insert("unit".to_owned(), Value::String(projected.unit.clone()));
    insert_number(&mut object, "avg", projected.avg);
    for (index, label) in PERCENTILE_LABELS.iter().enumerate() {
        insert_number(&mut object, label, projected.percentiles[index]);
    }
    insert_number(&mut object, "min", projected.min);
    insert_number(&mut object, "max", projected.max);
    insert_number(&mut object, "std", projected.std);
    if let Some(count) = projected.count {
        object.insert("count".to_owned(), Value::from(count));
    }
    insert_number(&mut object, "sum", projected.sum);
    Value::Object(object)
}

/// Insert a finite JSON number under `key`, or omit it when absent.
fn insert_number(object: &mut Map<String, Value>, key: &str, value: Option<f64>) {
    if let Some(number) = value
        && let Some(json) = serde_json::Number::from_f64(number)
    {
        object.insert(key.to_owned(), Value::Number(json));
    }
}

/// Render `<stem>_aiperf.json`. The metric objects are byte-exact against the
/// Python exporter; `schema_version`, `aiperf_version`, `was_cancelled`, and the
/// projected `input_config` are reproducible top-level scalars. Fields the sink
/// cannot reconstruct from the [`NativeReport`] alone (full `BenchmarkConfig`,
/// `run_info`, timestamps, telemetry) are reconciled by the Python frontend at
/// integration and are outside the byte-parity contract.
fn render_json(report: &NativeReport, cfg: &ExportConfig) -> String {
    let collected = collect_metrics(&report.metrics);
    let mut by_name: HashMap<&str, &Projected> = HashMap::new();
    for (name, projected) in &collected {
        by_name.insert(name.as_str(), projected);
    }

    let mut root = Map::new();
    root.insert(
        "schema_version".to_owned(),
        Value::String(JSON_SCHEMA_VERSION.to_owned()),
    );
    root.insert(
        "aiperf_version".to_owned(),
        Value::String(report.aiperf_version.clone()),
    );

    // Declared metric fields in JsonExportData order.
    let declared: HashSet<&str> = JSON_METRIC_ORDER.iter().copied().collect();
    for tag in JSON_METRIC_ORDER {
        if let Some(projected) = by_name.get(tag) {
            root.insert((*tag).to_owned(), metric_object(projected));
        }
    }

    // input_config: a reasonable deterministic projection (not byte-compared).
    root.insert("input_config".to_owned(), input_config(report, cfg));

    root.insert(
        "was_cancelled".to_owned(),
        Value::Bool(report.summary.was_cancelled),
    );

    // Warmup metrics (declared field), alphabetical by tag.
    if let Some(warmup) = &report.warmup_metrics {
        let warmup_metrics = collect_metrics(warmup);
        if !warmup_metrics.is_empty() {
            let mut object = Map::new();
            for (name, projected) in &warmup_metrics {
                object.insert(name.clone(), metric_object(projected));
            }
            root.insert("warmup_metrics".to_owned(), Value::Object(object));
        }
    }

    // Extra (undeclared) metric tags, appended last in alphabetical order.
    for (name, projected) in &collected {
        if !declared.contains(name.as_str()) {
            root.insert(name.clone(), metric_object(projected));
        }
    }

    serde_json::to_string_pretty(&Value::Object(root))
        .expect("v1 summary JSON value is always serializable")
}

/// Build the projected `input_config`. Deliberately minimal and deterministic;
/// the byte-parity contract covers metric objects only.
fn input_config(report: &NativeReport, cfg: &ExportConfig) -> Value {
    let mut object = Map::new();
    if !cfg.genai_perf.endpoint_type.is_empty() {
        object.insert(
            "endpoint_type".to_owned(),
            Value::String(cfg.genai_perf.endpoint_type.clone()),
        );
    }
    object.insert(
        "streaming".to_owned(),
        Value::Bool(cfg.genai_perf.streaming),
    );
    object.insert("goodput".to_owned(), Value::Bool(cfg.genai_perf.goodput));
    if let Some(mode) = &report.run.mode {
        object.insert("mode".to_owned(), Value::String(mode.clone()));
    }
    if let Some(model) = &report.run.model {
        object.insert("model".to_owned(), Value::String(model.clone()));
    }
    Value::Object(object)
}

/// Format a metric's display name (`metrics_csv_exporter.py:115-120`
/// `_format_metric_name`): append ` (unit)` unless the unit is empty or one of
/// `count`/`requests` (case-insensitive).
fn format_metric_name(header: &str, unit: &str) -> String {
    let lower = unit.to_ascii_lowercase();
    if unit.is_empty() || lower == "count" || lower == "requests" {
        header.to_owned()
    } else if header.is_empty() {
        format!("({unit})")
    } else {
        format!("{header} ({unit})")
    }
}

/// Format one CSV stat value (`metrics_csv_exporter.py:122-136` `_format_number`):
/// absent → empty string, finite float → `{:.2}` (matches Python `f"{v:.2f}"`).
fn format_number(value: Option<f64>) -> String {
    match value {
        Some(number) => format!("{number:.2}"),
        None => String::new(),
    }
}

/// Fetch a stat by its `STAT_KEYS` name for the CSV request row.
fn stat_value(projected: &Projected, stat: &str) -> Option<f64> {
    match stat {
        "avg" => projected.avg,
        "min" => projected.min,
        "max" => projected.max,
        "sum" => projected.sum,
        "std" => projected.std,
        percentile => PERCENTILE_LABELS
            .iter()
            .position(|label| *label == percentile)
            .and_then(|index| projected.percentiles[index]),
    }
}

/// Render `<stem>_aiperf.csv`: request section (metrics with percentiles), a
/// blank row, then the system section (scalar metrics), each sorted by tag. The
/// Python `csv.writer` excel dialect (CRLF terminator, minimal quoting) is
/// reproduced by the `csv` crate; the empty separator row is emitted manually
/// because the crate would otherwise write a quoted empty field.
fn render_csv(report: &NativeReport) -> anyhow::Result<String> {
    let collected = collect_metrics(&report.metrics);

    let mut request: Vec<&(String, Projected)> = collected
        .iter()
        .filter(|(_, projected)| projected.has_percentiles())
        .collect();
    let mut system: Vec<&(String, Projected)> = collected
        .iter()
        .filter(|(_, projected)| !projected.has_percentiles())
        .collect();
    request.sort_by(|left, right| left.0.cmp(&right.0));
    system.sort_by(|left, right| left.0.cmp(&right.0));

    let mut out: Vec<u8> = Vec::new();

    if !request.is_empty() {
        let mut writer = crlf_writer();
        let mut header = Vec::with_capacity(1 + STAT_KEYS.len());
        header.push("Metric".to_owned());
        header.extend(STAT_KEYS.iter().map(|key| (*key).to_owned()));
        writer.write_record(&header)?;
        for (_, projected) in &request {
            let mut row = Vec::with_capacity(1 + STAT_KEYS.len());
            row.push(format_metric_name(&projected.header, &projected.unit));
            for stat in STAT_KEYS {
                row.push(format_number(stat_value(projected, stat)));
            }
            writer.write_record(&row)?;
        }
        out.extend_from_slice(&writer.into_inner()?);
        // Blank separator row between sections: a bare CRLF (Python
        // `writer.writerow([])`), only when a system section follows.
        if !system.is_empty() {
            out.extend_from_slice(b"\r\n");
        }
    }

    if !system.is_empty() {
        let mut writer = crlf_writer();
        writer.write_record(["Metric", "Value"])?;
        for (_, projected) in &system {
            writer.write_record([
                format_metric_name(&projected.header, &projected.unit),
                format_number(projected.avg),
            ])?;
        }
        out.extend_from_slice(&writer.into_inner()?);
    }

    Ok(String::from_utf8(out)?)
}

/// A CRLF-terminated CSV writer over an in-memory buffer, matching Python's
/// default `csv.writer` excel dialect (`\r\n` terminator, minimal quoting).
fn crlf_writer() -> csv::Writer<Vec<u8>> {
    csv::WriterBuilder::new()
        .terminator(csv::Terminator::CRLF)
        .from_writer(Vec::new())
}

#[cfg(test)]
mod tests;
