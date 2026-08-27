// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Timeslice sink for `profile_export_aiperf_timeslices.{json,csv}`.
//!
//! Per-series slices are grouped by `(start_ns, end_ns, complete)` and sorted by
//! that tuple. Each metric uses its sole series or unique unlabeled aggregate.
//! JSON omits `is_complete` for complete slices and orders metric fields as
//! `unit, avg, p1, p5, p10, p25, p50, p75, p90, p95, p99, min, max, std, count,
//! sum`; scalar counts and absent fields are omitted. CSV uses
//! `Timeslice,Start_NS,End_NS,Metric,Unit,Stat,Value`, tag-sorted metrics, fixed
//! stat order, CRLF records, and two-decimal values. Non-finite values are absent.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use anyhow::{Context, bail, ensure};
use serde_json::{Map, Value};

use crate::export::{ExportConfig, Exporter, crlf_csv_writer};
use crate::metrics_core::ReportView;
use crate::metrics_core::report::{MetricSeries, ReportStats, ReportValue};

/// Canonical percentile field order shared by the JSON `JsonMetricResult`
/// declaration and the CSV `STAT_KEYS` list. A `BTreeMap` key sort is *not*
/// usable here: lexical ordering places `p10` before `p5`, so the fixed numeric
/// order must be applied explicitly.
const PERCENTILE_ORDER: [&str; 9] = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"];

/// Timeslice export policy. Enabled when the run produced timeslices.
///
/// The caller supplies `input_config`, metric headers, filtered tags, and scalar tags.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TimesliceExportConfig {
    /// Emit `profile_export_aiperf_timeslices.json`.
    pub json: bool,
    /// Emit `profile_export_aiperf_timeslices.csv`.
    pub csv: bool,
    /// Filename stem (before the `_timeslices` suffix); default `profile_export_aiperf`.
    pub stem: Option<String>,
    /// Input configuration inserted after `timeslices`; `Null` omits the key.
    #[serde(default)]
    pub input_config: Value,
    /// Display header by metric tag; absent tags use the raw tag.
    #[serde(default)]
    pub header_map: std::collections::HashMap<String, String>,
    /// Metric tags omitted from both artifacts.
    #[serde(default)]
    pub filtered_tags: std::collections::HashSet<String>,
    /// Scalar metric tags whose JSON `count` field is omitted.
    #[serde(default)]
    pub scalar_tags: std::collections::HashSet<String>,
}

/// Per-slice, per-metric statistics lowered to the outer-orchestrator
/// contract. Every numeric field is either finite or structurally absent.
#[derive(Debug, Clone, Default)]
struct SliceStats {
    avg: Option<f64>,
    min: Option<f64>,
    max: Option<f64>,
    sum: Option<f64>,
    std: Option<f64>,
    /// Record-distribution observation count; suppressed for scalar metrics.
    count: Option<i64>,
    /// Percentiles retained in canonical numeric order.
    percentiles: Vec<(&'static str, f64)>,
}

#[derive(Debug, Clone)]
struct SliceMetric {
    /// Stable metric tag (report key); drives CSV metric-sort and JSON field key.
    tag: String,
    /// Display header (frontend `header_map`, falling back to the tag) for the CSV.
    header: String,
    /// Metric display unit copied from the report metric entry.
    unit: String,
    stats: SliceStats,
}

/// One regrouped compatibility slice keyed by `(start_ns, end_ns, complete)`.
#[derive(Debug, Clone)]
struct SliceGroup {
    start_ns: i64,
    end_ns: i64,
    complete: bool,
    /// Metrics in tag-sorted order (report `metrics` is a `BTreeMap`).
    metrics: Vec<SliceMetric>,
}

pub struct TimesliceExporter;

impl Exporter for TimesliceExporter {
    fn name(&self) -> &'static str {
        "timeslice"
    }

    fn enabled(&self, cfg: &ExportConfig) -> bool {
        cfg.timeslice.json || cfg.timeslice.csv
    }

    fn export(
        &self,
        report: &dyn ReportView,
        artifact_dir: &Path,
        cfg: &ExportConfig,
    ) -> anyhow::Result<()> {
        let slices = regroup_timeslices(report, &cfg.timeslice)?;
        // Empty timeslice sets produce no artifacts.
        if slices.is_empty() {
            return Ok(());
        }

        let base = cfg
            .timeslice
            .stem
            .as_deref()
            .unwrap_or("profile_export_aiperf");

        if cfg.timeslice.json {
            let content = render_json(&slices, &cfg.timeslice.input_config)?;
            write_artifact(artifact_dir, &format!("{base}_timeslices.json"), &content)?;
        }
        if cfg.timeslice.csv {
            let content = render_csv(&slices)?;
            write_artifact(artifact_dir, &format!("{base}_timeslices.csv"), &content)?;
        }
        Ok(())
    }
}

/// Filter configured tags, select summary series, and group slices by
/// `(start_ns, end_ns, complete)` in sorted order.
fn regroup_timeslices(
    report: &dyn ReportView,
    cfg: &TimesliceExportConfig,
) -> anyhow::Result<Vec<SliceGroup>> {
    let mut groups: BTreeMap<(i64, i64, bool), Vec<SliceMetric>> = BTreeMap::new();

    for (tag, entry) in report.metrics() {
        // Registered INTERNAL/EXPERIMENTAL metrics are dropped; a tag outside the
        // projected set is always kept (including unregistered native-runtime tags).
        if cfg.filtered_tags.contains(tag) {
            continue;
        }

        let series = match summary_series(tag, &entry.series)? {
            Some(series) => series,
            None => continue,
        };

        // Display header: the frontend `MetricRegistry` header, falling back to
        // the tag string for unregistered metrics.
        let header = cfg
            .header_map
            .get(tag)
            .cloned()
            .unwrap_or_else(|| tag.clone());
        // Scalar (AGGREGATE/DERIVED) metrics suppress `count` in JSON.
        let is_scalar = cfg.scalar_tags.contains(tag);

        for slice in &series.timeslices {
            let stats = lower_stats(&slice.stats, is_scalar);
            groups
                .entry((slice.start_ns, slice.end_ns, slice.complete))
                .or_default()
                .push(SliceMetric {
                    tag: tag.clone(),
                    header: header.clone(),
                    unit: entry.unit.clone(),
                    stats,
                });
        }
    }

    Ok(groups
        .into_iter()
        .map(|((start_ns, end_ns, complete), metrics)| SliceGroup {
            start_ns,
            end_ns,
            complete,
            metrics,
        })
        .collect())
}

/// Select a metric's summary series exactly as `_summary_series`: the single
/// series when there is one, otherwise the unique unlabeled aggregate series.
/// Zero unlabeled series among many yields `None` (metric contributes no
/// slices); more than one unlabeled aggregate is a hard report error.
fn summary_series<'a>(
    tag: &str,
    series: &'a [MetricSeries],
) -> anyhow::Result<Option<&'a MetricSeries>> {
    match crate::export::summary_series(series) {
        crate::export::SummarySeries::Empty => {
            bail!("metric {tag:?} must contain at least one series")
        }
        crate::export::SummarySeries::Selected(series) => Ok(Some(series)),
        crate::export::SummarySeries::NoAggregate => Ok(None),
        crate::export::SummarySeries::Ambiguous => {
            bail!("metric {tag:?} contains multiple unlabeled aggregate series")
        }
    }
}

/// Present, finite value of an optional [`ReportValue`]; non-finite/absent both
/// lower to `None`.
fn finite_opt(value: Option<ReportValue>) -> Option<f64> {
    value.and_then(crate::export::finite_passthrough)
}

/// Present, finite value of a required [`ReportValue`]; non-finite lowers to
/// `None`.
fn finite(value: ReportValue) -> Option<f64> {
    crate::export::finite_passthrough(value)
}

/// Percentiles retained in canonical numeric order, non-finite entries dropped.
fn ordered_percentiles(percentiles: &BTreeMap<String, ReportValue>) -> Vec<(&'static str, f64)> {
    PERCENTILE_ORDER
        .iter()
        .filter_map(|&key| match percentiles.get(key) {
            Some(ReportValue::Finite(value)) => Some((key, *value)),
            _ => None,
        })
        .collect()
}

/// Lower one type-specific report stats block to the compatibility stat set.
/// `is_scalar` controls `count` suppression.
fn lower_stats(stats: &ReportStats, is_scalar: bool) -> SliceStats {
    match stats {
        ReportStats::Distribution(dist) => SliceStats {
            avg: finite_opt(dist.avg),
            min: finite_opt(dist.min),
            max: finite_opt(dist.max),
            sum: None,
            std: finite_opt(dist.std),
            count: if is_scalar {
                None
            } else {
                dist.count.map(|count| count as i64)
            },
            percentiles: ordered_percentiles(&dist.percentiles),
        },
        ReportStats::Scalar(scalar) => {
            let value = finite(scalar.value);
            SliceStats {
                avg: value,
                min: value,
                max: value,
                ..SliceStats::default()
            }
        }
        ReportStats::Counter(counter) => {
            let total = finite(counter.total);
            SliceStats {
                avg: total,
                min: total,
                max: total,
                sum: total,
                ..SliceStats::default()
            }
        }
        ReportStats::Histogram(hist) => SliceStats {
            avg: finite_opt(hist.avg),
            sum: finite(hist.sum),
            count: if is_scalar {
                None
            } else {
                Some(hist.count as i64)
            },
            percentiles: ordered_percentiles(&hist.percentiles),
            ..SliceStats::default()
        },
    }
}

/// Serialize the regrouped slices to the compatibility JSON shape. Insertion order is
/// preserved by `serde_json`'s `preserve_order` feature; `to_string_pretty`
/// matches orjson's two-space indent byte-for-byte. The frontend-projected
/// `input_config` (if present) is wrapped after the `timeslices` array, in
/// `TimesliceCollectionExportData` field order; a `Null` projection omits the
/// key.
fn render_json(slices: &[SliceGroup], input_config: &Value) -> anyhow::Result<String> {
    let mut array = Vec::with_capacity(slices.len());
    for slice in slices {
        let mut object = Map::new();
        object.insert("start_ns".to_string(), Value::from(slice.start_ns));
        object.insert("end_ns".to_string(), Value::from(slice.end_ns));
        // `is_complete` is emitted only for partial slices; complete slices omit
        // it (`is_complete=None` under `exclude_none`).
        if !slice.complete {
            object.insert("is_complete".to_string(), Value::Bool(false));
        }
        for metric in &slice.metrics {
            object.insert(metric.tag.clone(), metric_json(metric));
        }
        array.push(Value::Object(object));
    }

    let mut root = Map::new();
    root.insert("timeslices".to_string(), Value::Array(array));
    if !input_config.is_null() {
        root.insert("input_config".to_string(), input_config.clone());
    }
    serde_json::to_string_pretty(&Value::Object(root)).context("serializing timeslice JSON export")
}

/// Preserves `JsonMetricResult` field order.
fn metric_json(metric: &SliceMetric) -> Value {
    let stats = &metric.stats;
    let mut object = Map::new();
    object.insert("unit".to_string(), Value::from(metric.unit.clone()));
    insert_number(&mut object, "avg", stats.avg);
    for (key, value) in &stats.percentiles {
        object.insert((*key).to_string(), number_value(*value));
    }
    insert_number(&mut object, "min", stats.min);
    insert_number(&mut object, "max", stats.max);
    insert_number(&mut object, "std", stats.std);
    if let Some(count) = stats.count {
        object.insert("count".to_string(), Value::from(count));
    }
    insert_number(&mut object, "sum", stats.sum);
    Value::Object(object)
}

/// Insert a present finite value; absent values leave the key out entirely.
fn insert_number(object: &mut Map<String, Value>, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        object.insert(key.to_string(), number_value(value));
    }
}

/// Wrap a finite f64 as a JSON number. Report values are always finite here, so
/// the fallback to null is defensive only.
fn number_value(value: f64) -> Value {
    serde_json::Number::from_f64(value).map_or(Value::Null, Value::Number)
}

/// Serialize regrouped slices with CRLF, minimal quoting, and two-decimal values.
fn render_csv(slices: &[SliceGroup]) -> anyhow::Result<String> {
    let mut writer = crlf_csv_writer(Vec::new());

    writer
        .write_record([
            "Timeslice",
            "Start_NS",
            "End_NS",
            "Metric",
            "Unit",
            "Stat",
            "Value",
        ])
        .context("writing timeslice CSV header")?;

    for (index, slice) in slices.iter().enumerate() {
        let timeslice_index = index.to_string();
        let start = slice.start_ns.to_string();
        let end = slice.end_ns.to_string();
        for metric in &slice.metrics {
            for (stat, value) in csv_stat_rows(&metric.stats) {
                writer
                    .write_record([
                        timeslice_index.as_str(),
                        start.as_str(),
                        end.as_str(),
                        metric.header.as_str(),
                        metric.unit.as_str(),
                        stat,
                        &format!("{value:.2}"),
                    ])
                    .context("writing timeslice CSV row")?;
            }
        }
    }

    let bytes = writer
        .into_inner()
        .context("flushing timeslice CSV writer")?;
    String::from_utf8(bytes).context("timeslice CSV is not valid UTF-8")
}

/// Present stats in `STAT_KEYS` order (`count` is intentionally excluded from
/// the CSV). Percentiles are interleaved between `sum` and `std`, in canonical
/// numeric order.
fn csv_stat_rows(stats: &SliceStats) -> Vec<(&'static str, f64)> {
    let mut rows = Vec::new();
    if let Some(value) = stats.avg {
        rows.push(("avg", value));
    }
    if let Some(value) = stats.min {
        rows.push(("min", value));
    }
    if let Some(value) = stats.max {
        rows.push(("max", value));
    }
    if let Some(value) = stats.sum {
        rows.push(("sum", value));
    }
    rows.extend(stats.percentiles.iter().copied());
    if let Some(value) = stats.std {
        rows.push(("std", value));
    }
    rows
}

/// Join `name` onto the run's artifact directory, rejecting any stem that would
/// escape it, then write `content` without a trailing newline.
fn write_artifact(artifact_dir: &Path, name: &str, content: &str) -> anyhow::Result<()> {
    let mut components = Path::new(name).components();
    ensure!(
        matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none(),
        "refusing to write timeslice artifact with unsafe name {name:?}"
    );
    std::fs::create_dir_all(artifact_dir)
        .with_context(|| format!("creating artifact directory {}", artifact_dir.display()))?;
    let path = artifact_dir.join(name);
    std::fs::write(&path, content)
        .with_context(|| format!("writing timeslice artifact {}", path.display()))
}

#[cfg(test)]
mod tests;
