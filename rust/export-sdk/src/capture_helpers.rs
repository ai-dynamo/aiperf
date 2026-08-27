// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Navigation of the capture projections an exporter is handed.
//!
//! The finalized report crosses the boundary as
//! [`aiperf_core::capture::FinalReportV1`], carrying the exact JSON the host
//! commits. Several exporters need the same two decisions over it: which of a
//! metric's series stands for the metric as a whole, and how a metric's
//! type-specific statistics flatten into one shape. Both decisions live here
//! once, so a newly added stat field reaches every consumer instead of being
//! silently dropped by an out-of-date per-exporter match arm.
//!
//! The remaining helpers read the two non-report projections —
//! [`aiperf_core::capture::ExactRecordsV1`] and
//! [`aiperf_core::capture::GenAiClientHistogramsV1`] — which are already typed,
//! ordered, and merged by the host before an exporter sees them.

use std::sync::LazyLock;

use aiperf_core::capture::{
    ExactRecordV1, ExactRecordsV1, GenAiClientHistogramsV1, HistogramSeriesV1, MetricValueV1,
};
use serde_json::{Map, Value};

/// Classification of a metric's series for summary selection.
///
/// Exporters agree on the selection rule and disagree only on how they react to
/// the degenerate cases: the table renderers skip the metric, while the
/// timeslice exporter treats an empty or ambiguous metric as a hard error. This
/// classifier owns the rule; each caller maps the outcome to its own policy and
/// error text.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SummarySeries<'a> {
    /// The metric carried no series at all.
    Empty,
    /// The selected summary series: the sole series, or the unique unlabeled
    /// aggregate among several labeled series.
    Selected(&'a Value),
    /// Several series, none unlabeled — there is no aggregate to summarize.
    NoAggregate,
    /// Several series with more than one unlabeled aggregate — ambiguous.
    Ambiguous,
}

/// Select the sole series or the unique unlabeled aggregate.
///
/// A series is "unlabeled" when it carries no `labels` object, matching the
/// report projection in which a labeled series is a per-label breakdown and the
/// unlabeled one is the aggregate across them.
pub fn summary_series(series: &[Value]) -> SummarySeries<'_> {
    match series {
        [] => SummarySeries::Empty,
        [single] => SummarySeries::Selected(single),
        many => {
            let mut unlabeled = many.iter().filter(|series| is_unlabeled(series));
            let first = unlabeled.next();
            if unlabeled.next().is_some() {
                return SummarySeries::Ambiguous;
            }
            match first {
                Some(series) => SummarySeries::Selected(series),
                None => SummarySeries::NoAggregate,
            }
        }
    }
}

/// A series carries no label set when `labels` is absent or `null`.
fn is_unlabeled(series: &Value) -> bool {
    match series.get("labels") {
        None | Some(Value::Null) => true,
        Some(_) => false,
    }
}

/// Empty percentile table borrowed by the scalar-shaped [`CanonicalStats`]
/// variants, which carry no `pN` map of their own.
static EMPTY_PERCENTILES: LazyLock<Map<String, Value>> = LazyLock::new(Map::new);

/// A metric's type-specific statistics projected into one flat,
/// exporter-neutral shape.
///
/// Values are the raw boundary values — finiteness is each sink's own policy,
/// applied with [`crate::finite_passthrough`] or [`crate::finite_guarded`]. The
/// borrowed `percentiles` table is the variant's own `pN` map, empty for the
/// single-valued scalar and counter variants.
#[derive(Debug, Clone, Copy)]
pub struct CanonicalStats<'a> {
    /// Representative value: distribution/histogram average, or the scalar value
    /// or counter total. This is the bare-tag value, and the lone value
    /// `single_value` emitters broadcast across their columns.
    pub avg: Option<&'a Value>,
    /// Minimum observation (distribution only).
    pub min: Option<&'a Value>,
    /// Maximum observation (distribution only).
    pub max: Option<&'a Value>,
    /// Population standard deviation (distribution only).
    pub std: Option<&'a Value>,
    /// Observation count (distribution and histogram).
    pub count: Option<u64>,
    /// Sum of observations (histogram only).
    pub sum: Option<&'a Value>,
    /// Percentile table keyed by `pN`.
    pub percentiles: &'a Map<String, Value>,
    /// True for the single-valued variants: the representative stands in for the
    /// min/max/percentile columns.
    pub single_value: bool,
}

/// Project one metric's statistics into the flat [`CanonicalStats`] shape.
///
/// This is the single place the report's four stat shapes are discriminated.
/// They are serialized untagged, so the shape is recovered from its
/// discriminating key: `value` for a scalar, `total` for a counter, `buckets`
/// for a histogram, and `percentiles` alone for a distribution. A value that is
/// not an object, or an object matching none of the four, yields `None` rather
/// than a silently empty row.
pub fn flatten_stats(stats: &Value) -> Option<CanonicalStats<'_>> {
    let object = stats.as_object()?;
    if let Some(value) = object.get("value") {
        return Some(single(value));
    }
    if let Some(total) = object.get("total") {
        return Some(single(total));
    }
    if object.contains_key("buckets") {
        return Some(CanonicalStats {
            avg: object.get("avg"),
            min: None,
            max: None,
            std: None,
            count: object.get("count").and_then(Value::as_u64),
            sum: object.get("sum"),
            percentiles: percentiles(object),
            single_value: false,
        });
    }
    if object.contains_key("percentiles") {
        return Some(CanonicalStats {
            avg: object.get("avg"),
            min: object.get("min"),
            max: object.get("max"),
            std: object.get("std"),
            count: object.get("count").and_then(Value::as_u64),
            sum: None,
            percentiles: percentiles(object),
            single_value: false,
        });
    }
    None
}

fn single(value: &Value) -> CanonicalStats<'_> {
    CanonicalStats {
        avg: Some(value),
        min: None,
        max: None,
        std: None,
        count: None,
        sum: None,
        percentiles: &EMPTY_PERCENTILES,
        single_value: true,
    }
}

fn percentiles(object: &Map<String, Value>) -> &Map<String, Value> {
    match object.get("percentiles").and_then(Value::as_object) {
        Some(table) => table,
        None => &EMPTY_PERCENTILES,
    }
}

/// The finalized report's `metrics` table, when the projection carries one.
///
/// Returns `None` rather than an error for a report shape without metrics: a
/// report is committed even when a run produced no measurable record, and an
/// exporter reacting to that is its own policy.
pub fn report_metrics(report: &Value) -> Option<&Map<String, Value>> {
    report.get("metrics")?.as_object()
}

/// One named metric entry from the finalized report's `metrics` table.
pub fn report_metric<'a>(report: &'a Value, name: &str) -> Option<&'a Value> {
    report_metrics(report)?.get(name)
}

/// Every record that reached terminal without a classified failure, in the
/// projection's canonical order.
pub fn successful_records(records: &ExactRecordsV1) -> impl Iterator<Item = &ExactRecordV1> {
    records.records.iter().filter(|record| record.is_success())
}

/// One record's projected value for `metric`, in the report's display unit.
pub fn record_metric<'a>(record: &'a ExactRecordV1, metric: &str) -> Option<&'a MetricValueV1> {
    record.metrics.get(metric)
}

/// Every folded series emitted for one semantic-convention metric name, in the
/// projection's canonical order.
pub fn histogram_series<'a>(
    histograms: &'a GenAiClientHistogramsV1,
    metric: &'a str,
) -> impl Iterator<Item = &'a HistogramSeriesV1> {
    histograms
        .series
        .iter()
        .filter(move |series| series.metric == metric)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labeled(model: &str) -> Value {
        serde_json::json!({ "labels": { "model": model }, "stats": { "value": 1.0 } })
    }

    fn unlabeled() -> Value {
        serde_json::json!({ "stats": { "value": 2.0 } })
    }

    #[test]
    fn summary_selection_prefers_the_unique_unlabeled_aggregate() {
        assert!(matches!(summary_series(&[]), SummarySeries::Empty));
        assert!(matches!(
            summary_series(&[labeled("a")]),
            SummarySeries::Selected(_)
        ));
        assert!(matches!(
            summary_series(&[labeled("a"), unlabeled(), labeled("b")]),
            SummarySeries::Selected(_)
        ));
        assert!(matches!(
            summary_series(&[labeled("a"), labeled("b")]),
            SummarySeries::NoAggregate
        ));
        assert!(matches!(
            summary_series(&[unlabeled(), unlabeled()]),
            SummarySeries::Ambiguous
        ));
    }

    #[test]
    fn each_stat_shape_flattens_to_its_own_columns() {
        let scalar = flatten_stats(&serde_json::json!({ "value": 3.0 })).expect("scalar");
        assert!(scalar.single_value);
        assert_eq!(scalar.avg, Some(&serde_json::json!(3.0)));
        assert!(scalar.percentiles.is_empty());

        let counter = flatten_stats(&serde_json::json!({ "total": 7.0, "rate": 1.0 }))
            .expect("counter");
        assert!(counter.single_value);
        assert_eq!(counter.avg, Some(&serde_json::json!(7.0)));

        let distribution = flatten_stats(&serde_json::json!({
            "count": 4, "avg": 1.0, "min": 0.5, "max": 2.0, "std": 0.25,
            "percentiles": { "p99": 1.9 }
        }))
        .expect("distribution");
        assert!(!distribution.single_value);
        assert_eq!(distribution.count, Some(4));
        assert_eq!(distribution.max, Some(&serde_json::json!(2.0)));
        assert_eq!(distribution.sum, None);
        assert_eq!(distribution.percentiles.len(), 1);

        let histogram = flatten_stats(&serde_json::json!({
            "count": 2, "sum": 6.0, "avg": 3.0, "percentiles": {}, "buckets": { "1": 1 }
        }))
        .expect("histogram");
        assert_eq!(histogram.sum, Some(&serde_json::json!(6.0)));
        assert_eq!(histogram.min, None);
    }

    #[test]
    fn a_shape_that_is_no_stat_variant_is_refused() {
        assert!(flatten_stats(&serde_json::json!(1.0)).is_none());
        assert!(flatten_stats(&serde_json::json!({ "unrelated": 1 })).is_none());
    }

    #[test]
    fn report_navigation_returns_none_for_a_report_without_metrics() {
        let report = serde_json::json!({ "metrics": { "ttft": { "unit": "ms" } } });
        assert!(report_metric(&report, "ttft").is_some());
        assert!(report_metric(&report, "absent").is_none());
        assert!(report_metrics(&serde_json::json!({})).is_none());
    }
}
