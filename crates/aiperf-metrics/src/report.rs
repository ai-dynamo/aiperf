// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, deterministic native-v2 report construction.
//!
//! This module is IO-free. It translates an accumulator summary into the
//! metrics-first, type-specific-series representation; application-layer
//! exporters decide where to write it.

use crate::catalog::{
    AggregationKind, MetricConsoleGroup, MetricFlags, MetricTag, MetricType, spec_for,
};
use crate::{
    AccumulatorSummary, AccuracyAnalysis, AccuracyRecord, MetricResult, MetricResultData,
    MetricValue,
};
use serde::Serialize as DeriveSerialize;
use serde::ser::{Serialize, Serializer};
use std::collections::BTreeMap;

/// Native report schema identifier.
pub const NATIVE_REPORT_SCHEMA_VERSION: &str = "2.0";

/// A present report value: finite numbers serialize normally; non-finite tails
/// serialize as JSON null without colliding with structurally absent fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReportValue {
    /// Finite numeric value.
    Finite(f64),
    /// Present but non-finite value, reserved for error-adjusted tails.
    NonFinite,
}

impl Serialize for ReportValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Finite(value) => serializer.serialize_f64(*value),
            Self::NonFinite => serializer.serialize_none(),
        }
    }
}

/// Distribution statistics used by inference records and gauge series.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportDistributionStats {
    /// Number of observations.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count: Option<usize>,
    /// Arithmetic or duration-weighted average.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg: Option<ReportValue>,
    /// Minimum observation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min: Option<ReportValue>,
    /// Maximum observation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max: Option<ReportValue>,
    /// Population standard deviation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std: Option<ReportValue>,
    /// Percentiles keyed by `pN`.
    pub percentiles: BTreeMap<String, ReportValue>,
}

/// Scalar statistics used by derived and min/max aggregate metrics.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportScalarStats {
    /// The scalar value.
    pub value: ReportValue,
}

/// Counter statistics used by sum aggregates.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportCounterStats {
    /// Accumulated total.
    pub total: ReportValue,
    /// Optional rate paired with this counter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rate: Option<ReportValue>,
}

/// Type-specific statistics serialized without an additional wrapper tag.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
#[serde(untagged)]
pub enum ReportStats {
    /// Distribution-shaped statistics.
    Distribution(ReportDistributionStats),
    /// Scalar-shaped statistics.
    Scalar(ReportScalarStats),
    /// Counter-shaped statistics.
    Counter(ReportCounterStats),
}

/// One metric-series timeslice using the same stats shape as its parent.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportTimeslice {
    /// Inclusive window start in nanoseconds.
    pub start_ns: i64,
    /// Exclusive window end in nanoseconds.
    pub end_ns: i64,
    /// Whether the slice spans its full configured duration.
    pub complete: bool,
    /// Type-appropriate timeslice statistics.
    pub stats: ReportStats,
}

/// One labeled series for a metric.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct MetricSeries {
    /// Optional label set; inference metrics currently emit null.
    pub labels: Option<BTreeMap<String, String>>,
    /// Optional source endpoint for telemetry/server series.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub endpoint_url: Option<String>,
    /// Type-appropriate overall statistics.
    pub stats: ReportStats,
    /// Chronological non-empty timeslices.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub timeslices: Vec<ReportTimeslice>,
}

/// One metric keyed by stable name in the native report.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct MetricEntry {
    /// Consumer-facing stats shape.
    #[serde(rename = "type")]
    pub metric_type: &'static str,
    /// Display unit.
    pub unit: String,
    /// Console group.
    pub group: &'static str,
    /// Plot/SLO direction.
    pub higher_is_better: bool,
    /// Deterministically ordered labeled series.
    pub series: Vec<MetricSeries>,
}

/// Typed run identity shared by report consumers.
#[derive(Debug, Clone, Default, PartialEq, Eq, DeriveSerialize)]
pub struct ReportRunInfo {
    /// Execution mode, such as `online` or `graph`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    /// Requested model name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// Run-level summary metadata outside the metric namespace.
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportSummary {
    /// First request timestamp in nanoseconds on the run timeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_time: Option<i64>,
    /// Last response timestamp in nanoseconds on the run timeline.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_time: Option<i64>,
    /// Observation duration in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_s: Option<f64>,
    /// Whether the run was canceled.
    pub was_cancelled: bool,
    /// Configured endpoints in stable order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub endpoints_configured: Vec<String>,
    /// Endpoints that returned successful requests.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub endpoints_successful: Vec<String>,
}

/// One grouped API error in the unified report.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportError {
    /// HTTP or application error code.
    pub code: Option<u16>,
    /// Stable error type.
    #[serde(rename = "type")]
    pub error_type: String,
    /// Representative message.
    pub message: String,
    /// Number of matching records.
    pub count: usize,
}

/// Runtime facts supplied to a [`Reporter`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunOutcome {
    /// Run identity.
    pub run: ReportRunInfo,
    /// Summary metadata; missing timestamps/duration are filled from metrics.
    pub summary: ReportSummary,
    /// Optional warmup accumulator output.
    pub warmup: Option<AccumulatorSummary>,
    /// Optional accuracy/analyzer output.
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records in deterministic workload order.
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Grouped run errors.
    pub errors: Vec<ReportError>,
}

/// Summary-to-report extension seam.
pub trait Reporter {
    /// Typed report produced by this reporter.
    type Output;

    /// Builds a report without performing IO.
    fn report(&self, summary: &AccumulatorSummary, outcome: &RunOutcome) -> Self::Output;
}

/// Native-v2 metrics-first reporter.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeReporter;

impl Reporter for NativeReporter {
    type Output = NativeReport;

    fn report(&self, summary: &AccumulatorSummary, outcome: &RunOutcome) -> Self::Output {
        let mut run_summary = outcome.summary.clone();
        if run_summary.start_time.is_none() {
            run_summary.start_time = summary
                .finite_value(MetricTag::MinRequestTimestamp)
                .map(|value| value as i64);
        }
        if run_summary.end_time.is_none() {
            run_summary.end_time = summary
                .finite_value(MetricTag::MaxResponseTimestamp)
                .map(|value| value as i64);
        }
        if run_summary.duration_s.is_none() {
            run_summary.duration_s = summary.finite_value(MetricTag::BenchmarkDuration);
        }
        NativeReport {
            schema_version: NATIVE_REPORT_SCHEMA_VERSION,
            aiperf_version: env!("CARGO_PKG_VERSION").to_string(),
            run: outcome.run.clone(),
            summary: run_summary,
            metrics: build_metric_map(summary),
            warmup_metrics: outcome.warmup.as_ref().map(build_metric_map),
            accuracy: outcome.accuracy.clone(),
            accuracy_records: outcome.accuracy_records.clone(),
            errors: outcome.errors.clone(),
        }
    }
}

/// Native version-2 unified report shape.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct NativeReport {
    /// Native report schema version.
    pub schema_version: &'static str,
    /// AIPerf package version.
    pub aiperf_version: String,
    /// Run identity.
    pub run: ReportRunInfo,
    /// Run-level summary metadata.
    pub summary: ReportSummary,
    /// Profiling metrics keyed by stable name.
    pub metrics: BTreeMap<String, MetricEntry>,
    /// Warmup metrics using the same representation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup_metrics: Option<BTreeMap<String, MetricEntry>>,
    /// Optional accuracy analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records. Empty outside accuracy mode.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Grouped run errors.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<ReportError>,
}

impl NativeReport {
    /// Builds a native report from metrics and optional accuracy analysis.
    pub fn new(metrics: &AccumulatorSummary, accuracy: Option<AccuracyAnalysis>) -> Self {
        NativeReporter.report(
            metrics,
            &RunOutcome {
                accuracy,
                ..RunOutcome::default()
            },
        )
    }

    /// Builds a native report with explicit run metadata.
    pub fn from_outcome(metrics: &AccumulatorSummary, outcome: &RunOutcome) -> Self {
        NativeReporter.report(metrics, outcome)
    }
}

fn build_metric_map(summary: &AccumulatorSummary) -> BTreeMap<String, MetricEntry> {
    summary
        .results()
        .filter_map(|(name, result)| {
            let stats = report_stats(result, summary.result_map())?;
            let spec = result.source_tag.and_then(spec_for)?;
            let timeslices = summary
                .timeslices()
                .iter()
                .filter_map(|timeslice| {
                    let slice_result = timeslice.metrics.get(name)?;
                    Some(ReportTimeslice {
                        start_ns: timeslice.start_ns,
                        end_ns: timeslice.end_ns,
                        complete: timeslice.complete.unwrap_or(true),
                        stats: report_stats(slice_result, &timeslice.metrics)?,
                    })
                })
                .collect();
            Some((
                name.to_string(),
                MetricEntry {
                    metric_type: stats_type(&stats),
                    unit: result.unit.clone(),
                    group: console_group_name(spec.console_group),
                    higher_is_better: spec.flags.contains(MetricFlags::LARGER_IS_BETTER),
                    series: vec![MetricSeries {
                        labels: None,
                        endpoint_url: None,
                        stats,
                        timeslices,
                    }],
                },
            ))
        })
        .collect()
}

fn report_stats(
    result: &MetricResult,
    all_results: &BTreeMap<String, MetricResult>,
) -> Option<ReportStats> {
    match &result.data {
        MetricResultData::Distribution(stats) => {
            let adjusted = result.tag.starts_with("adj_");
            let percentiles = stats
                .percentiles
                .iter()
                .filter_map(|(percentile, value)| {
                    report_value(*value).map(|value| (format!("p{percentile}"), value))
                })
                .collect();
            Some(ReportStats::Distribution(ReportDistributionStats {
                count: (stats.count > 0).then_some(stats.count),
                avg: report_value(stats.avg),
                min: report_value(stats.min),
                max: report_value(stats.max),
                std: stats
                    .std
                    .map(ReportValue::Finite)
                    .or(adjusted.then_some(ReportValue::NonFinite)),
                percentiles,
            }))
        }
        MetricResultData::Scalar { value } => {
            let value = report_value(*value)?;
            let spec = result.source_tag.and_then(spec_for)?;
            if spec.kind == MetricType::Aggregate && spec.aggregation == Some(AggregationKind::Sum)
            {
                let rate = counter_rate(spec.tag)
                    .and_then(|tag| all_results.get(tag.as_str()))
                    .and_then(|result| report_value(result.representative_value()));
                Some(ReportStats::Counter(ReportCounterStats {
                    total: value,
                    rate,
                }))
            } else {
                Some(ReportStats::Scalar(ReportScalarStats { value }))
            }
        }
    }
}

fn report_value(value: MetricValue) -> Option<ReportValue> {
    match value {
        MetricValue::Finite(value) if value.is_finite() => Some(ReportValue::Finite(value)),
        MetricValue::PosInf => Some(ReportValue::NonFinite),
        MetricValue::Finite(_) | MetricValue::Absent => None,
    }
}

fn counter_rate(tag: MetricTag) -> Option<MetricTag> {
    match tag {
        MetricTag::RequestCount => Some(MetricTag::RequestThroughput),
        MetricTag::GoodRequestCount => Some(MetricTag::Goodput),
        _ => None,
    }
}

fn stats_type(stats: &ReportStats) -> &'static str {
    match stats {
        ReportStats::Distribution(_) => "distribution",
        ReportStats::Scalar(_) => "scalar",
        ReportStats::Counter(_) => "counter",
    }
}

fn console_group_name(group: MetricConsoleGroup) -> &'static str {
    match group {
        MetricConsoleGroup::None => "none",
        MetricConsoleGroup::Default => "default",
        MetricConsoleGroup::Usage => "usage",
        MetricConsoleGroup::Cache => "cache",
        MetricConsoleGroup::Prediction => "prediction",
        MetricConsoleGroup::Audio => "audio",
        MetricConsoleGroup::Reasoning => "reasoning",
        MetricConsoleGroup::Effective => "effective",
        MetricConsoleGroup::Active => "active",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MetricResult, MetricResultData};

    #[test]
    fn v2_uses_type_specific_series_and_null_for_non_finite_tail() {
        let mut summary = AccumulatorSummary::new();
        summary.insert_finite(MetricTag::RequestCount, 2.0);
        summary.insert_finite(MetricTag::RequestThroughput, 4.0);
        let mut percentiles = BTreeMap::new();
        percentiles.insert(50, MetricValue::Finite(10.0));
        percentiles.insert(99, MetricValue::PosInf);
        summary.insert_result(MetricResult {
            tag: "adj_request_latency".to_string(),
            source_tag: Some(MetricTag::RequestLatency),
            header: "Request Latency (error-adjusted)".to_string(),
            unit: "ms".to_string(),
            console_group: MetricConsoleGroup::Default,
            data: MetricResultData::Distribution(crate::DistributionStats {
                tag: "adj_request_latency".to_string(),
                avg: MetricValue::PosInf,
                min: MetricValue::Finite(10.0),
                max: MetricValue::PosInf,
                std: None,
                sum: MetricValue::PosInf,
                count: 2,
                percentiles,
            }),
        });

        let report = NativeReport::new(&summary, None);
        let serialized = serde_json::to_string_pretty(&report).unwrap();
        assert_eq!(
            serialized,
            include_str!("../tests/golden/native_v2.json").trim_end()
        );
        let value = serde_json::to_value(report).unwrap();
        assert_eq!(value["schema_version"], "2.0");
        assert_eq!(value["metrics"]["request_count"]["type"], "counter");
        assert_eq!(
            value["metrics"]["request_count"]["series"][0]["stats"]["total"],
            2.0
        );
        assert_eq!(
            value["metrics"]["request_count"]["series"][0]["stats"]["rate"],
            4.0
        );
        assert_eq!(
            value["metrics"]["adj_request_latency"]["type"],
            "distribution"
        );
        assert!(value["metrics"]["adj_request_latency"]["series"][0]["stats"]["avg"].is_null());
        assert!(
            value["metrics"]["adj_request_latency"]["series"][0]["stats"]["percentiles"]["p99"]
                .is_null()
        );
        assert!(value.get("warmup_metrics").is_none());
        assert!(value.get("accuracy").is_none());
        assert!(value.get("accuracy_records").is_none());
    }
}
