// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AIPerf v1 summary-export compatibility sink.
//!
//! Emits the two canonical AIPerf summary artifacts — `<stem>_aiperf.json`
//! (`schema_version = "1.4"`) and `<stem>_aiperf.csv` — byte-for-byte identical
//! to the Python exporters so downstream plotters, uploaders, and the multi-run
//! search layer consume either output. The
//! parity oracle is the AIPerf Python exporter suite, NOT the external NVIDIA
//! genai-perf tool.
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

use crate::export::{ExportConfig, Exporter, crlf_csv_writer, normalize_endpoint_display};
use crate::metrics_core::{MetricEntry, MetricSeries, NativeReport, ReportStats, ReportValue};
use chrono::{Local, TimeZone};
use serde_json::{Map, Value};

/// AIPerf v1 summary-export policy. Disabled unless the frontend requests the
/// `genai_perf` summary; `stem` is the profile-export filename stem
/// (`profile_export` → `profile_export_aiperf.{json,csv}`).
///
/// The sink owns all assembly and serialization; the remaining fields are
/// frontend-owned data values the native report alone cannot reconstruct, so
/// the artifacts reproduce the Python exporters byte-for-byte:
/// - `header_map` — the display header for every registered metric tag, derived
///   exactly as `native_report._metric_result`
///   (`MetricRegistry.get_class_or_none(tag).header`); an absent key falls back
///   to the tag string, matching Python's `else tag` branch.
/// - `filtered_tags` — the registered tags the Python file exporters drop
///   (`metrics_base_exporter._prepare_metrics`: INTERNAL / EXPERIMENTAL classes,
///   honoring the dev show-flags). A tag outside this set is always kept,
///   including native-runtime tags Python never registered.
/// - `scalar_tags` — registered tags whose Python `MetricType` is `AGGREGATE` /
///   `DERIVED`, for which `record_models.to_json_result` drops `count`.
/// - `envelope` — `benchmark_id`, `aiperf_version`, `input_config`, and
///   `run_info` serialized exactly as `MetricsJsonExporter` emits them.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenaiPerfExportConfig {
    /// Emit the v1 summary JSON/CSV artifacts.
    pub enabled: bool,
    /// Filename stem for the compat artifacts (before the `_aiperf` suffix).
    pub stem: String,
    /// Frontend-projected `{tag: header}` for every registered metric class.
    pub header_map: HashMap<String, String>,
    /// Registered tags the Python file exporters drop from both artifacts.
    pub filtered_tags: HashSet<String>,
    /// Registered scalar-tier tags whose `count` field is dropped.
    pub scalar_tags: HashSet<String>,
    /// Frontend-owned top-level JSON envelope values.
    pub envelope: GenaiPerfEnvelope,
}

impl Default for GenaiPerfExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stem: "profile_export".to_owned(),
            header_map: HashMap::new(),
            filtered_tags: HashSet::new(),
            scalar_tags: HashSet::new(),
            envelope: GenaiPerfEnvelope::default(),
        }
    }
}

/// Frontend-owned top-level fields of Python's `JsonExportData` the native
/// report cannot reconstruct. Each is projected as the exact JSON value the
/// Python `MetricsJsonExporter` emits (`model_dump(mode="json",
/// exclude_unset=True, exclude_none=True)` then `scrub_non_finite`), so the sink
/// splices them verbatim in `JsonExportData` declaration order. Absent fields
/// (`benchmark_id` / `run_info` when the run omits them) stay `None` and are not
/// serialized, matching `exclude_none`.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GenaiPerfEnvelope {
    /// AIPerf package version (`aiperf.__version__`) that generated the export.
    pub aiperf_version: Option<String>,
    /// Unique benchmark-run identifier (`BenchmarkRun.benchmark_id`).
    pub benchmark_id: Option<Value>,
    /// The authored `BenchmarkConfig` dump (`input_config`).
    pub input_config: Option<Value>,
    /// The per-run reproducibility block (`RunInfo`).
    pub run_info: Option<Value>,
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

        let json = render_json(report, &cfg.genai_perf);
        let csv = render_csv(report, &cfg.genai_perf)?;

        std::fs::write(artifact_dir.join(format!("{stem}_aiperf.json")), json)?;
        std::fs::write(artifact_dir.join(format!("{stem}_aiperf.csv")), csv)?;
        Ok(())
    }
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
    value.and_then(crate::export::finite_guarded)
}

/// Select the summary series for a metric (`native_report.py:791-806`
/// `_summary_series`): the lone series, or the single unlabeled aggregate when
/// several labeled series exist. Every degenerate case (no series, no aggregate,
/// or multiple ambiguous aggregates — the last of which Python raises on) skips
/// the metric entirely rather than aborting the best-effort export.
fn summary_series(entry: &MetricEntry) -> Option<&MetricSeries> {
    match crate::export::summary_series(&entry.series) {
        crate::export::SummarySeries::Selected(series) => Some(series),
        _ => None,
    }
}

/// Project one report metric into the flat stat set, applying the native
/// metric-type value shape and the `count`-drop rule for scalar-tier metrics.
/// The display header and the scalar-tier `count`-drop are frontend-owned
/// (`cfg.header_map` / `cfg.scalar_tags`), reproducing `native_report._metric_result`
/// and `record_models.to_json_result` exactly. Returns `None` when the metric
/// has no usable series or a required scalar / counter value is non-finite
/// (Python would raise; we skip).
fn project(name: &str, entry: &MetricEntry, cfg: &GenaiPerfExportConfig) -> Option<Projected> {
    let series = summary_series(entry)?;
    let header = cfg
        .header_map
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_owned());
    let mut projected = project_stats(&series.stats, entry.unit.clone(), header)?;

    // `to_json_result` (record_models.py:99-123) drops `count` for AGGREGATE /
    // DERIVED (scalar-tier) metrics, where it would trivially be 1. The scalar
    // classification is the Python `MetricType`, projected as `cfg.scalar_tags`.
    if cfg.scalar_tags.contains(name) {
        projected.count = None;
    }

    Some(projected)
}

/// Map one series' [`ReportStats`] into the flat v1 stat set, applying the
/// native metric-type value shape (`native_report.py:_legacy_stats`,
/// `823-854`). Returns `None` when a required scalar / counter value is
/// non-finite (Python would raise; the best-effort export skips instead). The
/// `cfg.scalar_tags` `count`-drop is a request-metric concern applied by the
/// caller, not here, so this helper is reusable for GPU-telemetry series where
/// `count` is always retained.
fn project_stats(stats: &ReportStats, unit: String, header: String) -> Option<Projected> {
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

    match stats {
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

    Some(projected)
}

/// Collect the exportable metrics in report (alphabetical `BTreeMap`) order,
/// after filtering and projection. Filtering is frontend-owned
/// (`cfg.filtered_tags`, the Python `_prepare_metrics` INTERNAL / EXPERIMENTAL
/// drop set); a tag outside that set is always kept, including native-runtime
/// tags Python never registered.
fn collect_metrics(
    metrics: &BTreeMap<String, MetricEntry>,
    cfg: &GenaiPerfExportConfig,
) -> Vec<(String, Projected)> {
    metrics
        .iter()
        .filter_map(|(name, entry)| {
            if cfg.filtered_tags.contains(name) {
                return None;
            }
            project(name, entry, cfg).map(|projected| (name.clone(), projected))
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

/// One GPU accumulated across the report's metric series, in first-seen order.
struct TelemetryGpu {
    gpu_index: i64,
    gpu_name: String,
    gpu_uuid: String,
    hostname: Option<String>,
    namespace: Option<String>,
    pod_name: Option<String>,
    /// `metric_name -> JsonMetricResult` object, inserted in report (alphabetical) order.
    metrics: Map<String, Value>,
}

/// One DCGM endpoint accumulated across the report's metric series. The raw
/// scrape URL is the `endpoint_order` key; the `summary` endpoint lists reuse it.
struct TelemetryEndpoint {
    /// GPU UUIDs in first-seen order.
    gpu_order: Vec<String>,
    /// GPU accumulators keyed by UUID.
    gpus: HashMap<String, TelemetryGpu>,
}

/// Project the report's GPU-telemetry series into the Python `TelemetryExportData`
/// shape (`native_report.py:_project_gpu_telemetry`, `620-720`). A series is
/// GPU telemetry iff it carries an `endpoint_url` and string `gpu` / `gpu_uuid`
/// / `model_name` labels — this excludes request metrics and the unlabeled
/// `total_gpu_*` aggregates while naturally including custom (`sm_clock`, …)
/// signals. Grouping is by `endpoint_url` then `gpu_uuid`, both in first-seen
/// order with no sorting; per-metric objects reuse [`project_stats`] +
/// [`metric_object`] so gauge/counter rendering is byte-identical to the
/// top-level metrics. Returns `None` when the run collected no GPU telemetry, so
/// the caller omits the whole block (`exclude_none`).
fn render_telemetry_data(report: &NativeReport) -> Option<Value> {
    let mut endpoint_order: Vec<String> = Vec::new();
    let mut endpoints: HashMap<String, TelemetryEndpoint> = HashMap::new();

    for (name, entry) in &report.metrics {
        for series in &entry.series {
            let Some(labels) = &series.labels else {
                continue;
            };
            let Some(endpoint_url) = &series.endpoint_url else {
                continue;
            };
            // Required GPU-identity labels; a series missing any is not telemetry.
            let (Some(gpu_label), Some(gpu_uuid), Some(model_name)) = (
                labels.get("gpu"),
                labels.get("gpu_uuid"),
                labels.get("model_name"),
            ) else {
                continue;
            };
            // Best-effort export skips a malformed index rather than aborting.
            let Ok(gpu_index) = gpu_label.parse::<i64>() else {
                continue;
            };
            if gpu_index < 0 {
                continue;
            }
            let Some(projected) = project_stats(&series.stats, entry.unit.clone(), name.clone())
            else {
                continue;
            };

            if !endpoints.contains_key(endpoint_url) {
                endpoint_order.push(endpoint_url.clone());
                endpoints.insert(
                    endpoint_url.clone(),
                    TelemetryEndpoint {
                        gpu_order: Vec::new(),
                        gpus: HashMap::new(),
                    },
                );
            }
            let endpoint = endpoints
                .get_mut(endpoint_url)
                .expect("endpoint just inserted");
            if !endpoint.gpus.contains_key(gpu_uuid) {
                endpoint.gpu_order.push(gpu_uuid.clone());
                endpoint.gpus.insert(
                    gpu_uuid.clone(),
                    TelemetryGpu {
                        gpu_index,
                        gpu_name: model_name.clone(),
                        gpu_uuid: gpu_uuid.clone(),
                        hostname: labels.get("hostname").cloned(),
                        namespace: labels.get("namespace").cloned(),
                        pod_name: labels.get("pod").cloned(),
                        metrics: Map::new(),
                    },
                );
            }
            let gpu = endpoint.gpus.get_mut(gpu_uuid).expect("gpu just inserted");
            gpu.metrics.insert(name.clone(), metric_object(&projected));
        }
    }

    if endpoint_order.is_empty() {
        return None;
    }

    let mut endpoints_obj = Map::new();
    for raw_url in &endpoint_order {
        let endpoint = &endpoints[raw_url];
        let mut gpus_obj = Map::new();
        for gpu_uuid in &endpoint.gpu_order {
            let gpu = &endpoint.gpus[gpu_uuid];
            let mut gpu_obj = Map::new();
            gpu_obj.insert("gpu_index".to_owned(), Value::from(gpu.gpu_index));
            gpu_obj.insert("gpu_name".to_owned(), Value::String(gpu.gpu_name.clone()));
            gpu_obj.insert("gpu_uuid".to_owned(), Value::String(gpu.gpu_uuid.clone()));
            if let Some(hostname) = &gpu.hostname {
                gpu_obj.insert("hostname".to_owned(), Value::String(hostname.clone()));
            }
            if let Some(namespace) = &gpu.namespace {
                gpu_obj.insert("namespace".to_owned(), Value::String(namespace.clone()));
            }
            if let Some(pod_name) = &gpu.pod_name {
                gpu_obj.insert("pod_name".to_owned(), Value::String(pod_name.clone()));
            }
            gpu_obj.insert("metrics".to_owned(), Value::Object(gpu.metrics.clone()));
            gpus_obj.insert(format!("gpu_{}", gpu.gpu_index), Value::Object(gpu_obj));
        }
        let mut endpoint_obj = Map::new();
        endpoint_obj.insert("gpus".to_owned(), Value::Object(gpus_obj));
        endpoints_obj.insert(
            normalize_endpoint_display(raw_url),
            Value::Object(endpoint_obj),
        );
    }

    let raw_urls: Vec<Value> = endpoint_order
        .iter()
        .map(|url| Value::String(url.clone()))
        .collect();
    let mut summary = Map::new();
    summary.insert(
        "endpoints_configured".to_owned(),
        Value::Array(raw_urls.clone()),
    );
    summary.insert("endpoints_successful".to_owned(), Value::Array(raw_urls));
    summary.insert(
        "start_time".to_owned(),
        Value::String(format_native_time(report.summary.start_time)),
    );
    summary.insert(
        "end_time".to_owned(),
        Value::String(format_native_time(report.summary.end_time)),
    );

    let mut telemetry = Map::new();
    telemetry.insert("summary".to_owned(), Value::Object(summary));
    telemetry.insert("endpoints".to_owned(), Value::Object(endpoints_obj));
    Some(Value::Object(telemetry))
}

/// Format a run-timeline nanosecond timestamp as Python
/// `datetime.fromtimestamp(ns/1e9).isoformat()` does (`native_report.py:705-709`):
/// local-timezone ISO-8601 with microsecond precision, dropping the fractional
/// part when it is zero. `None`/negative clamps to the epoch, matching the
/// oracle's guard. These are wall-clock values and inherently non-reproducible
/// across machines/timezones (as in Python); byte-exact parity is on the
/// `endpoints` subtree, which the tests and product consume.
fn format_native_time(ns: Option<i64>) -> String {
    let ns = ns.filter(|value| *value >= 0).unwrap_or(0);
    let seconds = ns / 1_000_000_000;
    let sub_nanos = (ns % 1_000_000_000) as u32;
    let datetime = Local
        .timestamp_opt(seconds, sub_nanos)
        .single()
        .unwrap_or_else(|| {
            Local
                .timestamp_opt(0, 0)
                .single()
                .expect("unix epoch is a valid local timestamp")
        });
    let micros = datetime.timestamp_subsec_micros();
    if micros == 0 {
        datetime.format("%Y-%m-%dT%H:%M:%S").to_string()
    } else {
        format!("{}.{micros:06}", datetime.format("%Y-%m-%dT%H:%M:%S"))
    }
}

/// Render `<stem>_aiperf.json` byte-for-byte against the Python
/// `MetricsJsonExporter`. The top-level map is assembled in `JsonExportData`
/// declaration order (`export_models.py:293-349`): `schema_version`,
/// `aiperf_version`, `benchmark_id`, the declared metric slots, `input_config`,
/// `run_info`, `was_cancelled`, `error_summary`, `warmup_metrics`, then the
/// undeclared ("extra") metric tags Pydantic appends last in native-report
/// (alphabetical) order. Frontend-owned scalars (`aiperf_version`,
/// `benchmark_id`, `input_config`, `run_info`) are spliced from `cfg.envelope`;
/// the sink pins `schema_version` and derives `was_cancelled` / `error_summary`
/// from the [`NativeReport`]. `telemetry_data` is projected from the report's
/// GPU-telemetry series (see [`render_telemetry_data`]) and omitted when the run
/// carried none. `start_time` / `end_time` / `branch_stats` remain `None` on the
/// native compatibility path (the Python oracle passes `start_ns=end_ns=0`, no
/// DAG stats) and are therefore omitted by `exclude_none`.
fn render_json(report: &NativeReport, cfg: &GenaiPerfExportConfig) -> String {
    let collected = collect_metrics(&report.metrics, cfg);
    let mut by_name: HashMap<&str, &Projected> = HashMap::new();
    for (name, projected) in &collected {
        by_name.insert(name.as_str(), projected);
    }

    let mut root = Map::new();
    root.insert(
        "schema_version".to_owned(),
        Value::String(JSON_SCHEMA_VERSION.to_owned()),
    );
    // aiperf_version: the frontend package version; the report's own field is a
    // fallback only when the frontend projection is absent (e.g. unit tests).
    let aiperf_version = cfg
        .envelope
        .aiperf_version
        .clone()
        .unwrap_or_else(|| report.aiperf_version.clone());
    root.insert("aiperf_version".to_owned(), Value::String(aiperf_version));
    if let Some(benchmark_id) = &cfg.envelope.benchmark_id {
        root.insert("benchmark_id".to_owned(), benchmark_id.clone());
    }

    // Declared metric fields in JsonExportData order.
    let declared: HashSet<&str> = JSON_METRIC_ORDER.iter().copied().collect();
    for tag in JSON_METRIC_ORDER {
        if let Some(projected) = by_name.get(tag) {
            root.insert((*tag).to_owned(), metric_object(projected));
        }
    }

    // `telemetry_data` is the declared `JsonExportData` field immediately after
    // the last metric slot and before `input_config` (`export_models.py`), so it
    // is spliced here. Omitted when the run collected no GPU telemetry.
    if let Some(telemetry) = render_telemetry_data(report) {
        root.insert("telemetry_data".to_owned(), telemetry);
    }

    // Frontend-owned envelope values, spliced verbatim in declaration order.
    if let Some(input_config) = &cfg.envelope.input_config {
        root.insert("input_config".to_owned(), input_config.clone());
    }
    if let Some(run_info) = &cfg.envelope.run_info {
        root.insert("run_info".to_owned(), run_info.clone());
    }

    root.insert(
        "was_cancelled".to_owned(),
        Value::Bool(report.summary.was_cancelled),
    );
    // error_summary is always set (an empty array when the run had no errors),
    // matching the Python oracle's explicit `error_summary=[]`.
    root.insert("error_summary".to_owned(), error_summary(report));

    // Warmup metrics (declared field), alphabetical by tag.
    if let Some(warmup) = &report.warmup_metrics {
        let warmup_metrics = collect_metrics(warmup, cfg);
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

/// Build the `error_summary` array from the report's grouped errors, matching
/// `export_python_compatibility_reports`'s `ErrorDetailsCount` projection under
/// `exclude_unset` / `exclude_none`: each item is
/// `{"error_details": {code?, type, message}, "count": N}` with `code` present
/// only when the report carried one.
fn error_summary(report: &NativeReport) -> Value {
    let mut items = Vec::with_capacity(report.errors.len());
    for error in &report.errors {
        let mut details = Map::new();
        if let Some(code) = error.code {
            details.insert("code".to_owned(), Value::from(code));
        }
        details.insert("type".to_owned(), Value::String(error.error_type.clone()));
        details.insert("message".to_owned(), Value::String(error.message.clone()));
        let mut item = Map::new();
        item.insert("error_details".to_owned(), Value::Object(details));
        item.insert("count".to_owned(), Value::from(error.count as u64));
        items.push(Value::Object(item));
    }
    Value::Array(items)
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
fn render_csv(report: &NativeReport, cfg: &GenaiPerfExportConfig) -> anyhow::Result<String> {
    let collected = collect_metrics(&report.metrics, cfg);

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
        let mut writer = crlf_csv_writer(Vec::new());
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
        let mut writer = crlf_csv_writer(Vec::new());
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
#[cfg(test)]
mod tests;
