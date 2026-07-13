// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict exposition parsing and explicit native server-metrics projection.
//!
//! The lossless grammar and semantic model live in `aiperf-prometheus`.
//! This module owns only the intentionally narrower compatibility projection
//! consumed by the existing server-metrics accumulator: finite scalars,
//! counters, and cumulative histograms. Its family filters and normalized
//! counter names port `src/aiperf/server_metrics/data_collector.py:246-361`
//! and `src/aiperf/server_metrics/data_collector.py:401-524`.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter, Result as FmtResult};

use aiperf_prometheus::{
    CompatibilityProjectionError, Exposition, ExpositionFormat, ExpositionParser, MetricPoint,
    MetricValue, NativeCompatibilityProjection, ParseError, ParseLimits, SemanticType,
    StrictExpositionParser, WireSampleRole,
};

use crate::model::{HistogramValue, MetricFamily, MetricSample, PrometheusMetricType};

/// Malformed exposition or failed native projection with a one-based line location.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsParseError {
    /// One-based line number, or zero for a body-level error.
    pub line: usize,
    /// Human-readable parser or projection detail.
    pub message: String,
}

impl Display for MetricsParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        if self.line == 0 {
            formatter.write_str(&self.message)
        } else {
            write!(formatter, "line {}: {}", self.line, self.message)
        }
    }
}

impl std::error::Error for MetricsParseError {}

impl From<ParseError> for MetricsParseError {
    fn from(error: ParseError) -> Self {
        let message = if error.column == 0 {
            error.message
        } else {
            format!("column {}: {}", error.column, error.message)
        };
        Self {
            line: error.line,
            message,
        }
    }
}

impl From<CompatibilityProjectionError> for MetricsParseError {
    fn from(error: CompatibilityProjectionError) -> Self {
        Self {
            line: 0,
            message: error.message,
        }
    }
}

/// Object-safe seam separating exact grammar selection from native projection.
pub trait MetricsTextParser {
    /// Parses one complete body under exactly the selected grammar.
    ///
    /// Implementations must not retry another grammar internally. A caller
    /// that needs a compatibility retry performs a second named call.
    fn parse_exposition(
        &self,
        format: ExpositionFormat,
        exact_body: &[u8],
    ) -> Result<Exposition, MetricsParseError>;

    /// Projects one successful lossless exposition into the existing native model.
    fn project_native(
        &self,
        exposition: &Exposition,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError>;

    /// Parses and projects classic Prometheus text exposition.
    fn parse_classic(
        &self,
        body: &str,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError> {
        let exposition =
            self.parse_exposition(ExpositionFormat::PrometheusText004, body.as_bytes())?;
        self.project_native(&exposition)
    }

    /// Parses and projects strict OpenMetrics text exposition without fallback.
    fn parse_openmetrics(
        &self,
        body: &str,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError> {
        let exposition =
            self.parse_exposition(ExpositionFormat::OpenMetricsText100, body.as_bytes())?;
        self.project_native(&exposition)
    }
}

/// Strict bounded parser paired with the native compatibility projection.
#[derive(Debug, Default, Clone, Copy)]
pub struct PrometheusTextParser;

impl MetricsTextParser for PrometheusTextParser {
    fn parse_exposition(
        &self,
        format: ExpositionFormat,
        exact_body: &[u8],
    ) -> Result<Exposition, MetricsParseError> {
        StrictExpositionParser
            .parse(format, exact_body, &ParseLimits::default())
            .map_err(Into::into)
    }

    fn project_native(
        &self,
        exposition: &Exposition,
    ) -> Result<BTreeMap<String, MetricFamily>, MetricsParseError> {
        NativeServerMetricsProjection
            .project(exposition)
            .map(|projection| projection.unwrap_or_default())
            .map_err(Into::into)
    }
}

/// Compatibility projection from lossless points to accumulator-shaped families.
///
/// Summary, Info, StateSet, creation-time, uptime, non-finite, and values
/// without a finite binary64 projection remain absent because the native
/// accumulator has no representation for them. This policy never changes the
/// success or failure of the strict parse that produced the exposition.
#[derive(Debug, Default, Clone, Copy)]
pub struct NativeServerMetricsProjection;

impl NativeCompatibilityProjection for NativeServerMetricsProjection {
    type Output = BTreeMap<String, MetricFamily>;

    fn project(
        &self,
        exposition: &Exposition,
    ) -> Result<Option<Self::Output>, CompatibilityProjectionError> {
        Ok(Some(project_exposition(exposition)))
    }
}

type LabelKey = Vec<(String, String)>;

fn project_exposition(exposition: &Exposition) -> BTreeMap<String, MetricFamily> {
    let mut projected = BTreeMap::<String, MetricFamily>::new();
    for family in &exposition.families {
        let Some(metric_type) = native_metric_type(family.semantic_type) else {
            continue;
        };
        let normalized_name = normalize_family_name(&family.name, metric_type);
        if should_skip_family(&normalized_name) {
            continue;
        }
        let mut samples = BTreeMap::<LabelKey, MetricSample>::new();
        for metric in &family.metrics {
            for point in &metric.points {
                let Some(sample) = project_point(point, metric_type) else {
                    continue;
                };
                let key = sample
                    .labels()
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
                    .collect();
                samples.insert(key, sample);
            }
        }
        if samples.is_empty() {
            continue;
        }
        let candidate = MetricFamily {
            metric_type,
            description: family
                .help
                .as_ref()
                .map(|help| help.value.clone())
                .unwrap_or_default(),
            samples: samples.into_values().collect(),
        };
        match projected.entry(normalized_name) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(candidate);
            }
            std::collections::btree_map::Entry::Occupied(mut entry)
                if family_priority(candidate.metric_type)
                    > family_priority(entry.get().metric_type) =>
            {
                entry.insert(candidate);
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
    }
    projected
}

fn native_metric_type(semantic_type: SemanticType) -> Option<PrometheusMetricType> {
    match semantic_type {
        SemanticType::Unknown => Some(PrometheusMetricType::Unknown),
        SemanticType::Gauge => Some(PrometheusMetricType::Gauge),
        SemanticType::Counter => Some(PrometheusMetricType::Counter),
        SemanticType::Histogram | SemanticType::GaugeHistogram => {
            Some(PrometheusMetricType::Histogram)
        }
        SemanticType::Summary | SemanticType::StateSet | SemanticType::Info => None,
    }
}

fn normalize_family_name(name: &str, metric_type: PrometheusMetricType) -> String {
    if metric_type == PrometheusMetricType::Counter {
        name.strip_suffix("_total").unwrap_or(name).to_string()
    } else {
        name.to_string()
    }
}

fn project_point(point: &MetricPoint, metric_type: PrometheusMetricType) -> Option<MetricSample> {
    match &point.value {
        MetricValue::Scalar { value, .. } => Some(MetricSample::Scalar {
            labels: point.labels.clone(),
            value: finite_value(value)?,
        }),
        MetricValue::Counter(value) => Some(MetricSample::Scalar {
            labels: point.labels.clone(),
            value: finite_value(&value.total)?,
        }),
        MetricValue::Histogram(value) if metric_type == PrometheusMetricType::Histogram => {
            let buckets = value
                .buckets
                .iter()
                .filter_map(|bucket| {
                    finite_value(&bucket.cumulative_count)
                        .map(|value| (bucket.upper_bound_lexeme.clone(), value))
                })
                .collect::<BTreeMap<_, _>>();
            let sum = point
                .wire_samples
                .iter()
                .any(|wire| {
                    matches!(
                        wire.role,
                        WireSampleRole::HistogramSum | WireSampleRole::GaugeHistogramSum
                    )
                })
                .then(|| finite_value(&value.sum))
                .flatten();
            let count = point
                .wire_samples
                .iter()
                .any(|wire| {
                    matches!(
                        wire.role,
                        WireSampleRole::HistogramCount | WireSampleRole::GaugeHistogramCount
                    )
                })
                .then(|| finite_value(&value.count))
                .flatten();
            (!buckets.is_empty() || sum.is_some() || count.is_some()).then(|| {
                MetricSample::Histogram {
                    labels: point.labels.clone(),
                    value: HistogramValue {
                        buckets,
                        sum,
                        count,
                    },
                }
            })
        }
        MetricValue::Histogram(_)
        | MetricValue::Summary(_)
        | MetricValue::StateSet(_)
        | MetricValue::Info(_) => None,
    }
}

fn finite_value(value: &aiperf_prometheus::ExactNumber) -> Option<f64> {
    value.finite_value.filter(|value| value.is_finite())
}

fn family_priority(metric_type: PrometheusMetricType) -> u8 {
    match metric_type {
        PrometheusMetricType::Counter => 4,
        PrometheusMetricType::Histogram => 3,
        PrometheusMetricType::Gauge => 2,
        PrometheusMetricType::Unknown => 1,
        PrometheusMetricType::Summary => 0,
    }
}

fn should_skip_family(name: &str) -> bool {
    name.ends_with("_created") || name.ends_with("_uptime") || name.contains("_uptime_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_parser_normalizes_deduplicates_and_structures_histograms() {
        let body = concat!(
            "# HELP vllm:prompt_tokens_total Prompt tokens.\n",
            "# TYPE vllm:prompt_tokens_total counter\n",
            "vllm:prompt_tokens_total{model=\"a\"} 10\n",
            "vllm:prompt_tokens_total{model=\"a\"} 12\n",
            "vllm:prompt_tokens_created{model=\"a\"} 123\n",
            "# HELP vllm:request_latency_seconds Request latency in seconds.\n",
            "# TYPE vllm:request_latency_seconds histogram\n",
            "vllm:request_latency_seconds_bucket{model=\"a\",le=\"0.1\"} 4\n",
            "vllm:request_latency_seconds_bucket{model=\"a\",le=\"+Inf\"} 5\n",
            "vllm:request_latency_seconds_sum{model=\"a\"} 0.9\n",
            "vllm:request_latency_seconds_count{model=\"a\"} 5\n",
            "# TYPE lifetime summary\n",
            "lifetime{quantile=\"0.5\"} 1\n",
            "# TYPE process_uptime_seconds gauge\n",
            "process_uptime_seconds 10\n",
        );
        let metrics = PrometheusTextParser.parse_classic(body).unwrap();
        assert_eq!(metrics.len(), 2);
        let counter = &metrics["vllm:prompt_tokens"];
        assert_eq!(counter.metric_type, PrometheusMetricType::Counter);
        assert_eq!(counter.description, "Prompt tokens.");
        assert!(matches!(
            &counter.samples[0],
            MetricSample::Scalar { value, .. } if *value == 12.0
        ));
        let histogram = &metrics["vllm:request_latency_seconds"];
        let MetricSample::Histogram { value, .. } = &histogram.samples[0] else {
            panic!("expected histogram")
        };
        assert_eq!(value.buckets.len(), 2);
        assert_eq!(value.buckets["+Inf"], 5.0);
        assert_eq!(value.sum, Some(0.9));
        assert_eq!(value.count, Some(5.0));
    }

    #[test]
    fn strict_openmetrics_never_falls_back_inside_the_parser() {
        let body = "# TYPE requests_total counter\nrequests_total 2\n";
        assert!(PrometheusTextParser.parse_openmetrics(body).is_err());
        assert_eq!(
            PrometheusTextParser.parse_classic(body).unwrap()["requests"].metric_type,
            PrometheusMetricType::Counter
        );

        let strict = "# TYPE requests counter\nrequests_total 2\n# EOF\n";
        assert_eq!(
            PrometheusTextParser.parse_openmetrics(strict).unwrap()["requests"].metric_type,
            PrometheusMetricType::Counter
        );
    }

    #[test]
    fn json_and_malformed_labels_are_rejected_atomically() {
        assert!(
            PrometheusTextParser
                .parse_classic("[{\"stats\": 1}]\n")
                .is_err()
        );
        assert!(
            PrometheusTextParser
                .parse_classic("metric{bad=unquoted} 1\n")
                .is_err()
        );
    }

    #[test]
    fn undeclared_untyped_samples_remain_unknown_gauges() {
        let metrics = PrometheusTextParser
            .parse_classic("node_netstat_Tcp_InSegs 3\n")
            .unwrap();
        assert_eq!(
            metrics["node_netstat_Tcp_InSegs"].metric_type,
            PrometheusMetricType::Unknown
        );
    }

    #[test]
    fn normalized_counter_wins_a_same_name_gauge_collision() {
        let body = concat!(
            "# TYPE sglang:num_retracted_reqs_total counter\n",
            "sglang:num_retracted_reqs_total 7\n",
            "# TYPE sglang:num_retracted_reqs gauge\n",
            "sglang:num_retracted_reqs 99\n",
        );

        let metrics = PrometheusTextParser.parse_classic(body).unwrap();
        let family = &metrics["sglang:num_retracted_reqs"];

        assert_eq!(family.metric_type, PrometheusMetricType::Counter);
        assert!(matches!(
            family.samples[0],
            MetricSample::Scalar { value: 7.0, .. }
        ));
    }

    #[test]
    fn tachometer_histogram_label_sets_never_cross_contaminate() {
        let body = concat!(
            "# TYPE vllm_request_queue_time_seconds histogram\n",
            "vllm_request_queue_time_seconds_bucket{model_name=\"model-a\",le=\"1\"} 2\n",
            "vllm_request_queue_time_seconds_bucket{model_name=\"model-b\",le=\"1\"} 7\n",
            "vllm_request_queue_time_seconds_bucket{model_name=\"model-a\",le=\"+Inf\"} 3\n",
            "vllm_request_queue_time_seconds_bucket{model_name=\"model-b\",le=\"+Inf\"} 11\n",
            "vllm_request_queue_time_seconds_count{model_name=\"model-a\"} 3\n",
            "vllm_request_queue_time_seconds_count{model_name=\"model-b\"} 11\n",
        );
        let metrics = PrometheusTextParser.parse_classic(body).unwrap();
        let samples = &metrics["vllm_request_queue_time_seconds"].samples;
        assert_eq!(samples.len(), 2);
        for sample in samples {
            let MetricSample::Histogram { labels, value } = sample else {
                panic!("expected histogram")
            };
            match labels["model_name"].as_str() {
                "model-a" => {
                    assert_eq!(value.buckets["1"], 2.0);
                    assert_eq!(value.count, Some(3.0));
                }
                "model-b" => {
                    assert_eq!(value.buckets["1"], 7.0);
                    assert_eq!(value.count, Some(11.0));
                }
                other => panic!("unexpected model {other:?}"),
            }
        }
    }

    #[test]
    fn tachometer_quoted_commas_and_float64_precision_survive_projection() {
        let body = "vllm:num_requests_running{model_name=\"meta-llama/Llama-3.1-8B, revision=\\\"prod\\\", path=C:\\\\models\"} 100000001\n";
        let metrics = PrometheusTextParser.parse_classic(body).unwrap();
        let MetricSample::Scalar { labels, value } =
            &metrics["vllm:num_requests_running"].samples[0]
        else {
            panic!("expected scalar")
        };
        assert_eq!(
            labels["model_name"],
            "meta-llama/Llama-3.1-8B, revision=\"prod\", path=C:\\models"
        );
        assert_eq!(*value, 100_000_001.0_f64);
    }

    #[test]
    fn non_finite_values_remain_outside_native_projection() {
        let body = "value NaN\nvalue 4\nvalue +Inf\n";
        let metrics = PrometheusTextParser.parse_classic(body).unwrap();
        assert!(matches!(
            metrics["value"].samples[0],
            MetricSample::Scalar { value: 4.0, .. }
        ));
    }
}
