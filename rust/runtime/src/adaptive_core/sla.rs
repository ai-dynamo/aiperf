// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Adaptive SLA filters and window evaluation.
//!
//! This module implements metric families, aliases, nanosecond-to-millisecond
//! conversion, empty TTFT/ITL behavior, goodput quality gates, and operators.

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::str::FromStr;

use crate::metrics_core::linear_distribution;
use serde::Serialize;

use crate::adaptive_core::error::AdaptiveError;
use crate::adaptive_core::window::{RequestSample, WindowStats};

const TTFT_METRICS: &[&str] = &["time_to_first_token", "ttft"];
const ITL_METRICS: &[&str] = &["inter_token_latency", "itl", "tpot"];
const THROUGHPUT_METRICS: &[&str] = &[
    "throughput",
    "request_throughput",
    "completed_request_throughput",
];
const SUCCESS_RATE_METRICS: &[&str] = &["success_rate", "request_success_rate"];
const ERROR_RATE_METRICS: &[&str] = &["error_rate", "request_error_rate"];
const CANCELLATION_RATE_METRICS: &[&str] = &["cancellation_rate", "request_cancellation_rate"];

/// Rate metrics that remain well-defined when a window produced zero successful
/// requests: they are computed from terminal counts (errors/cancellations over
/// the window total), not from latency/throughput samples. When every SLA
/// filter targets one of these, an all-error / all-cancellation window is
/// evaluable rather than discarded as inconclusive — otherwise an
/// error_rate/cancellation_rate-only SLA config could never converge because
/// the controller would early-return on every saturated window.
const ZERO_SUCCESS_WINDOW_METRICS: &[&str] = &[
    "error_rate",
    "request_error_rate",
    "cancellation_rate",
    "request_cancellation_rate",
];

/// Return `true` when a zero-success window is nonetheless evaluable because its
/// terminal states (errors and/or cancellations) are fully covered by the
/// configured SLA filters. Mirrors Python
/// `AdaptiveScaleSLAEvaluator.can_evaluate_without_successes`.
///
/// Requires that (a) every filter is a zero-success rate metric, (b) the window
/// is not a mix of BOTH errors and cancellations (ambiguous which drove it),
/// and (c) whichever terminal class is present has a matching rate filter.
pub fn can_evaluate_without_successes(filters: &[SlaFilter], stats: &WindowStats) -> bool {
    if filters.is_empty()
        || !filters
            .iter()
            .all(|f| ZERO_SUCCESS_WINDOW_METRICS.contains(&f.metric_tag.as_str()))
    {
        return false;
    }
    // A window that is simultaneously erroring and cancelling is ambiguous:
    // neither single rate filter can attribute the failure, so defer.
    if stats.errors > 0 && stats.cancelled > 0 {
        return false;
    }
    let has_error_rate_filter = filters
        .iter()
        .any(|f| ERROR_RATE_METRICS.contains(&f.metric_tag.as_str()));
    if stats.errors > 0 && !has_error_rate_filter {
        return false;
    }
    let has_cancellation_rate_filter = filters
        .iter()
        .any(|f| CANCELLATION_RATE_METRICS.contains(&f.metric_tag.as_str()));
    !(stats.cancelled > 0 && !has_cancellation_rate_filter)
}

/// Statistic selected from an adaptive SLA metric.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SlaStat {
    /// Arithmetic average.
    Avg,
    /// Minimum observation.
    Min,
    /// Maximum observation.
    Max,
    /// First percentile.
    P1,
    /// Fifth percentile.
    P5,
    /// Tenth percentile.
    P10,
    /// Twenty-fifth percentile.
    P25,
    /// Median.
    P50,
    /// Seventy-fifth percentile.
    P75,
    /// Ninetieth percentile.
    P90,
    /// Ninety-fifth percentile.
    P95,
    /// Ninety-ninth percentile.
    P99,
}

impl SlaStat {
    fn percentile(self) -> Option<u32> {
        match self {
            Self::P1 => Some(1),
            Self::P5 => Some(5),
            Self::P10 => Some(10),
            Self::P25 => Some(25),
            Self::P50 => Some(50),
            Self::P75 => Some(75),
            Self::P90 => Some(90),
            Self::P95 => Some(95),
            Self::P99 => Some(99),
            Self::Avg | Self::Min | Self::Max => None,
        }
    }

    fn is_rate_stat(self) -> bool {
        matches!(self, Self::Avg | Self::Min | Self::Max)
    }
}

impl Display for SlaStat {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Avg => "avg",
            Self::Min => "min",
            Self::Max => "max",
            Self::P1 => "p1",
            Self::P5 => "p5",
            Self::P10 => "p10",
            Self::P25 => "p25",
            Self::P50 => "p50",
            Self::P75 => "p75",
            Self::P90 => "p90",
            Self::P95 => "p95",
            Self::P99 => "p99",
        })
    }
}

impl FromStr for SlaStat {
    type Err = AdaptiveError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "avg" => Ok(Self::Avg),
            "min" => Ok(Self::Min),
            "max" => Ok(Self::Max),
            "p1" => Ok(Self::P1),
            "p5" => Ok(Self::P5),
            "p10" => Ok(Self::P10),
            "p25" => Ok(Self::P25),
            "p50" => Ok(Self::P50),
            "p75" => Ok(Self::P75),
            "p90" => Ok(Self::P90),
            "p95" => Ok(Self::P95),
            "p99" => Ok(Self::P99),
            other => Err(AdaptiveError::InvalidConfig(format!(
                "unsupported adaptive SLA statistic {other:?}"
            ))),
        }
    }
}

/// Comparison operator for an SLA filter.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SlaOp {
    /// Observed value must be strictly less than the threshold.
    Lt,
    /// Observed value must be less than or equal to the threshold.
    Le,
    /// Observed value must be strictly greater than the threshold.
    Gt,
    /// Observed value must be greater than or equal to the threshold.
    Ge,
}

impl Display for SlaOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        })
    }
}

impl FromStr for SlaOp {
    type Err = AdaptiveError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "lt" => Ok(Self::Lt),
            "le" => Ok(Self::Le),
            "gt" => Ok(Self::Gt),
            "ge" => Ok(Self::Ge),
            other => Err(AdaptiveError::InvalidConfig(format!(
                "unsupported adaptive SLA operator {other:?}"
            ))),
        }
    }
}

/// One conjunctive adaptive SLA constraint.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SlaFilter {
    /// Stable metric tag or supported alias.
    pub metric_tag: String,
    /// Statistic selected from the metric's window values.
    pub stat: SlaStat,
    /// Comparison operator.
    pub op: SlaOp,
    /// Finite comparison threshold in the metric's report unit.
    pub threshold: f64,
}

impl SlaFilter {
    /// Construct a filter, rejecting a blank tag or non-finite threshold.
    pub fn new(
        metric_tag: impl Into<String>,
        stat: SlaStat,
        op: SlaOp,
        threshold: f64,
    ) -> Result<Self, AdaptiveError> {
        let metric_tag = metric_tag.into();
        if metric_tag.trim().is_empty() {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive SLA metric tag must not be blank".to_string(),
            ));
        }
        if !threshold.is_finite() {
            return Err(AdaptiveError::InvalidConfig(format!(
                "adaptive SLA threshold must be finite, got {threshold}"
            )));
        }
        Ok(Self {
            metric_tag,
            stat,
            op,
            threshold,
        })
    }
}

/// Evaluated SLA values keyed by `metric:stat:op:threshold`.
pub type SlaValues = BTreeMap<String, f64>;

/// Object-safe evaluation seam from a completed window to SLA values.
pub trait SlaEvaluator {
    /// Validate the complete filter set before a run starts.
    fn validate_filters(&self, filters: &[SlaFilter]) -> Result<(), AdaptiveError>;
    /// Stable artifact key for a filter.
    fn key(&self, filter: &SlaFilter) -> String;
    /// Evaluate every filter against one window.
    fn values(
        &self,
        filters: &[SlaFilter],
        stats: &WindowStats,
    ) -> Result<SlaValues, AdaptiveError>;
    /// Return true only when every filter passes.
    fn passes(&self, filters: &[SlaFilter], observed: &SlaValues) -> Result<bool, AdaptiveError>;
    /// Normalized positive-is-headroom margin for a single filter.
    fn margin(&self, filter: &SlaFilter, observed: Option<f64>) -> Option<f64>;
    /// Key of the tightest filter margin in `observed`.
    fn binding_key(&self, filters: &[SlaFilter], observed: &SlaValues) -> Option<String>;
}

/// Built-in evaluator for latency, throughput, goodput, and return-rate metric
/// families.
#[derive(Default)]
pub struct DefaultSlaEvaluator;

impl DefaultSlaEvaluator {
    fn value(
        &self,
        filter: &SlaFilter,
        stats: &WindowStats,
        filters: &[SlaFilter],
    ) -> Result<f64, AdaptiveError> {
        let tag = filter.metric_tag.as_str();
        if tag == "request_latency" {
            return latency_value(&stats.latency_samples(), filter.stat, false);
        }
        if TTFT_METRICS.contains(&tag) {
            return latency_value(&stats.ttft_samples(), filter.stat, true);
        }
        if ITL_METRICS.contains(&tag) {
            return latency_value(&stats.itl_samples(), filter.stat, true);
        }
        if THROUGHPUT_METRICS.contains(&tag) {
            validate_rate_stat(filter.stat, "throughput")?;
            return Ok(stats.throughput());
        }
        if tag == "output_token_throughput" {
            validate_rate_stat(filter.stat, "output_token_throughput")?;
            return Ok(stats.output_token_throughput());
        }
        if tag == "goodput" {
            validate_rate_stat(filter.stat, "goodput")?;
            if stats.elapsed_sec <= 0.0 {
                return Ok(0.0);
            }
            return Ok(good_request_count(stats, filters)? as f64 / stats.elapsed_sec);
        }
        if tag == "goodput_ratio" {
            validate_rate_stat(filter.stat, "goodput_ratio")?;
            if stats.total() == 0 {
                return Ok(0.0);
            }
            return Ok(good_request_count(stats, filters)? as f64 / stats.total() as f64);
        }
        if SUCCESS_RATE_METRICS.contains(&tag) {
            validate_rate_stat(filter.stat, "success_rate")?;
            return Ok(if stats.total() == 0 {
                0.0
            } else {
                stats.completed() as f64 / stats.total() as f64
            });
        }
        if ERROR_RATE_METRICS.contains(&tag) {
            validate_rate_stat(filter.stat, "error_rate")?;
            return Ok(if stats.total() == 0 {
                0.0
            } else {
                stats.errors as f64 / stats.total() as f64
            });
        }
        if CANCELLATION_RATE_METRICS.contains(&tag) {
            validate_rate_stat(filter.stat, "cancellation_rate")?;
            return Ok(if stats.total() == 0 {
                0.0
            } else {
                stats.cancelled as f64 / stats.total() as f64
            });
        }
        Err(unsupported_metric(tag))
    }

    fn validate_single(&self, filter: &SlaFilter) -> Result<(), AdaptiveError> {
        let tag = filter.metric_tag.as_str();
        if tag == "request_latency" || TTFT_METRICS.contains(&tag) || ITL_METRICS.contains(&tag) {
            return Ok(());
        }
        if THROUGHPUT_METRICS.contains(&tag)
            || tag == "output_token_throughput"
            || tag == "goodput"
            || tag == "goodput_ratio"
            || SUCCESS_RATE_METRICS.contains(&tag)
            || ERROR_RATE_METRICS.contains(&tag)
            || CANCELLATION_RATE_METRICS.contains(&tag)
        {
            return validate_rate_stat(filter.stat, tag);
        }
        Err(unsupported_metric(tag))
    }

    /// Compare one already-evaluated observation to its filter.
    pub fn passes_single(filter: &SlaFilter, observed: f64) -> bool {
        match filter.op {
            SlaOp::Lt => observed < filter.threshold,
            SlaOp::Le => observed <= filter.threshold,
            SlaOp::Gt => observed > filter.threshold,
            SlaOp::Ge => observed >= filter.threshold,
        }
    }
}

impl SlaEvaluator for DefaultSlaEvaluator {
    fn validate_filters(&self, filters: &[SlaFilter]) -> Result<(), AdaptiveError> {
        if filters.is_empty() {
            return Err(AdaptiveError::InvalidConfig(
                "adaptive SLA filters are required".to_string(),
            ));
        }
        let has_goodput = filters
            .iter()
            .any(|filter| matches!(filter.metric_tag.as_str(), "goodput" | "goodput_ratio"));
        if has_goodput
            && !filters
                .iter()
                .any(|filter| is_quality_metric(&filter.metric_tag))
        {
            return Err(AdaptiveError::InvalidConfig(
                "quality goodput SLA requires at least one request_latency, time_to_first_token, or inter_token_latency quality filter".to_string(),
            ));
        }
        for filter in filters {
            self.validate_single(filter)?;
        }
        Ok(())
    }

    fn key(&self, filter: &SlaFilter) -> String {
        format!(
            "{}:{}:{}:{}",
            filter.metric_tag, filter.stat, filter.op, filter.threshold
        )
    }

    fn values(
        &self,
        filters: &[SlaFilter],
        stats: &WindowStats,
    ) -> Result<SlaValues, AdaptiveError> {
        filters
            .iter()
            .map(|filter| Ok((self.key(filter), self.value(filter, stats, filters)?)))
            .collect()
    }

    fn passes(&self, filters: &[SlaFilter], observed: &SlaValues) -> Result<bool, AdaptiveError> {
        for filter in filters {
            let key = self.key(filter);
            let value = observed.get(&key).copied().ok_or_else(|| {
                AdaptiveError::Evaluation(format!("missing observed SLA value for {key}"))
            })?;
            if !Self::passes_single(filter, value) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn margin(&self, filter: &SlaFilter, observed: Option<f64>) -> Option<f64> {
        let observed = observed?;
        if filter.threshold == 0.0 {
            return None;
        }
        let denominator = filter.threshold.abs();
        Some(match filter.op {
            SlaOp::Lt | SlaOp::Le => (filter.threshold - observed) / denominator,
            SlaOp::Gt | SlaOp::Ge => (observed - filter.threshold) / denominator,
        })
    }

    fn binding_key(&self, filters: &[SlaFilter], observed: &SlaValues) -> Option<String> {
        filters
            .iter()
            .filter_map(|filter| {
                let key = self.key(filter);
                self.margin(filter, observed.get(&key).copied())
                    .map(|margin| (key, margin))
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(key, _)| key)
    }
}

fn latency_value(
    samples_ns: &[f64],
    stat: SlaStat,
    empty_is_infinite: bool,
) -> Result<f64, AdaptiveError> {
    if samples_ns.is_empty() {
        if empty_is_infinite {
            return Ok(f64::INFINITY);
        }
        return Err(AdaptiveError::Evaluation(
            "request_latency SLA requires completed request samples".to_string(),
        ));
    }
    let value_ns = match stat {
        SlaStat::Avg => samples_ns.iter().sum::<f64>() / samples_ns.len() as f64,
        SlaStat::Min => samples_ns.iter().copied().fold(f64::INFINITY, f64::min),
        SlaStat::Max => samples_ns.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        percentile => percentile_value(samples_ns, percentile.percentile().expect("percentile"))?,
    };
    Ok(value_ns / 1_000_000.0)
}

fn percentile_value(samples: &[f64], percentile: u32) -> Result<f64, AdaptiveError> {
    let running_sum = samples.iter().sum();
    let distribution = linear_distribution("adaptive_window", samples.to_vec(), running_sum, 0)
        .ok_or_else(|| AdaptiveError::Evaluation("percentile requires samples".to_string()))?;
    distribution
        .percentiles
        .get(&percentile)
        .and_then(|value| value.as_f64())
        .ok_or_else(|| {
            AdaptiveError::Evaluation(format!(
                "adaptive percentile p{percentile} is not in the metrics kernel band"
            ))
        })
}

fn validate_rate_stat(stat: SlaStat, metric: &str) -> Result<(), AdaptiveError> {
    if stat.is_rate_stat() {
        Ok(())
    } else {
        Err(AdaptiveError::InvalidConfig(format!(
            "unsupported {metric} SLA statistic {stat}"
        )))
    }
}

fn unsupported_metric(metric: &str) -> AdaptiveError {
    AdaptiveError::InvalidConfig(format!(
        "adaptive scale supports request_latency, time_to_first_token, inter_token_latency, request throughput, output_token_throughput, goodput, goodput_ratio, success_rate, error_rate, and cancellation_rate; got {metric:?}"
    ))
}

fn is_quality_metric(metric: &str) -> bool {
    metric == "request_latency" || TTFT_METRICS.contains(&metric) || ITL_METRICS.contains(&metric)
}

fn good_request_count(stats: &WindowStats, filters: &[SlaFilter]) -> Result<usize, AdaptiveError> {
    let quality_filters: Vec<&SlaFilter> = filters
        .iter()
        .filter(|filter| is_quality_metric(&filter.metric_tag))
        .collect();
    if quality_filters.is_empty() {
        return Err(AdaptiveError::Evaluation(
            "quality goodput SLA requires at least one quality filter".to_string(),
        ));
    }
    Ok(stats
        .successful_requests
        .iter()
        .filter(|request| {
            quality_filters.iter().all(|filter| {
                per_request_quality_value(request, &filter.metric_tag)
                    .is_some_and(|value| DefaultSlaEvaluator::passes_single(filter, value))
            })
        })
        .count())
}

fn per_request_quality_value(request: &RequestSample, metric: &str) -> Option<f64> {
    if metric == "request_latency" {
        return Some(request.request_latency_ns as f64 / 1_000_000.0);
    }
    if TTFT_METRICS.contains(&metric) {
        return request.ttft_ns.map(|value| value as f64 / 1_000_000.0);
    }
    if ITL_METRICS.contains(&metric) {
        return request
            .inter_token_latency_ns
            .map(|value| value / 1_000_000.0);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filter(metric: &str, stat: SlaStat, op: SlaOp, threshold: f64) -> SlaFilter {
        SlaFilter::new(metric, stat, op, threshold).unwrap()
    }

    fn sample(latency_ms: i64, ttft_ms: Option<i64>, itl_ms: Option<f64>) -> RequestSample {
        RequestSample {
            request_latency_ns: latency_ms * 1_000_000,
            ttft_ns: ttft_ms.map(|value| value * 1_000_000),
            inter_token_latency_ns: itl_ms.map(|value| value * 1_000_000.0),
            output_sequence_length: Some(8),
        }
    }

    fn zero_success_stats(errors: usize, cancelled: usize) -> WindowStats {
        WindowStats {
            successful_requests: vec![],
            errors,
            cancelled,
            elapsed_sec: 1.0,
            start_ns: 0,
            end_ns: 1_000_000_000,
        }
    }

    #[test]
    fn zero_success_evaluable_for_error_rate_only_error_window() {
        let filters = vec![filter("error_rate", SlaStat::Avg, SlaOp::Le, 0.5)];
        assert!(can_evaluate_without_successes(
            &filters,
            &zero_success_stats(5, 0)
        ));
    }

    #[test]
    fn zero_success_evaluable_for_cancellation_rate_only_cancel_window() {
        let filters = vec![filter("cancellation_rate", SlaStat::Avg, SlaOp::Le, 0.5)];
        assert!(can_evaluate_without_successes(
            &filters,
            &zero_success_stats(0, 3)
        ));
    }

    #[test]
    fn zero_success_not_evaluable_when_latency_filter_present() {
        // A latency SLA cannot be evaluated with no successful samples.
        let filters = vec![
            filter("error_rate", SlaStat::Avg, SlaOp::Le, 0.5),
            filter("request_latency", SlaStat::Avg, SlaOp::Le, 100.0),
        ];
        assert!(!can_evaluate_without_successes(
            &filters,
            &zero_success_stats(5, 0)
        ));
    }

    #[test]
    fn zero_success_not_evaluable_for_mixed_error_and_cancel_window() {
        // Ambiguous: both terminal classes present, neither single rate filter
        // can attribute the failure.
        let filters = vec![
            filter("error_rate", SlaStat::Avg, SlaOp::Le, 0.5),
            filter("cancellation_rate", SlaStat::Avg, SlaOp::Le, 0.5),
        ];
        assert!(!can_evaluate_without_successes(
            &filters,
            &zero_success_stats(2, 2)
        ));
    }

    #[test]
    fn zero_success_not_evaluable_when_terminal_class_lacks_matching_filter() {
        // Errors present but only a cancellation_rate filter configured.
        let filters = vec![filter("cancellation_rate", SlaStat::Avg, SlaOp::Le, 0.5)];
        assert!(!can_evaluate_without_successes(
            &filters,
            &zero_success_stats(5, 0)
        ));
    }

    #[test]
    fn zero_success_not_evaluable_with_no_filters() {
        assert!(!can_evaluate_without_successes(
            &[],
            &zero_success_stats(5, 0)
        ));
    }

    #[test]
    fn latency_percentiles_reuse_the_metrics_linear_kernel() {
        let evaluator = DefaultSlaEvaluator;
        let sla = filter("request_latency", SlaStat::P95, SlaOp::Le, 100.0);
        let stats = WindowStats {
            successful_requests: vec![
                sample(10, None, None),
                sample(20, None, None),
                sample(30, None, None),
                sample(40, None, None),
                sample(50, None, None),
            ],
            errors: 0,
            cancelled: 0,
            elapsed_sec: 1.0,
            start_ns: 0,
            end_ns: 1_000_000_000,
        };
        let values = evaluator
            .values(std::slice::from_ref(&sla), &stats)
            .unwrap();
        assert_eq!(values[&evaluator.key(&sla)], 48.0);
    }

    #[test]
    fn missing_ttft_and_itl_are_infinite_and_fail_upper_bounds() {
        let evaluator = DefaultSlaEvaluator;
        let filters = vec![
            filter("ttft", SlaStat::P95, SlaOp::Le, 25.0),
            filter("itl", SlaStat::P95, SlaOp::Le, 25.0),
        ];
        let stats = WindowStats {
            successful_requests: vec![sample(100, None, None)],
            errors: 0,
            cancelled: 0,
            elapsed_sec: 1.0,
            start_ns: 0,
            end_ns: 1_000_000_000,
        };
        let values = evaluator.values(&filters, &stats).unwrap();
        assert!(values.values().all(|value| value.is_infinite()));
        assert!(!evaluator.passes(&filters, &values).unwrap());
    }

    #[test]
    fn goodput_is_per_request_quality_gated() {
        let evaluator = DefaultSlaEvaluator;
        let filters = vec![
            filter("request_latency", SlaStat::P95, SlaOp::Le, 100.0),
            filter("ttft", SlaStat::P95, SlaOp::Le, 30.0),
            filter("itl", SlaStat::P95, SlaOp::Le, 20.0),
            filter("goodput", SlaStat::Avg, SlaOp::Ge, 0.1),
        ];
        let stats = WindowStats {
            successful_requests: vec![
                sample(80, Some(20), Some(10.0)),
                sample(90, Some(35), Some(10.0)),
                sample(120, Some(20), Some(30.0)),
            ],
            errors: 0,
            cancelled: 0,
            elapsed_sec: 2.0,
            start_ns: 0,
            end_ns: 2_000_000_000,
        };
        let values = evaluator.values(&filters, &stats).unwrap();
        let goodput = &filters[3];
        assert_eq!(values[&evaluator.key(goodput)], 0.5);
    }

    #[test]
    fn rate_denominators_include_errors_and_cancellations() {
        let evaluator = DefaultSlaEvaluator;
        let filters = vec![
            filter("success_rate", SlaStat::Avg, SlaOp::Ge, 0.0),
            filter("error_rate", SlaStat::Avg, SlaOp::Le, 1.0),
            filter("cancellation_rate", SlaStat::Avg, SlaOp::Le, 1.0),
        ];
        let stats = WindowStats {
            successful_requests: vec![sample(10, None, None), sample(20, None, None)],
            errors: 1,
            cancelled: 1,
            elapsed_sec: 2.0,
            start_ns: 0,
            end_ns: 2_000_000_000,
        };
        let values = evaluator.values(&filters, &stats).unwrap();
        assert_eq!(values[&evaluator.key(&filters[0])], 0.5);
        assert_eq!(values[&evaluator.key(&filters[1])], 0.25);
        assert_eq!(values[&evaluator.key(&filters[2])], 0.25);
    }

    #[test]
    fn goodput_without_a_quality_filter_is_rejected() {
        let evaluator = DefaultSlaEvaluator;
        let filters = vec![filter("goodput", SlaStat::Avg, SlaOp::Ge, 1.0)];
        assert!(evaluator.validate_filters(&filters).is_err());
    }

    #[test]
    fn output_token_throughput_and_goodput_ratio_follow_python_denominators() {
        let evaluator = DefaultSlaEvaluator;
        let filters = vec![
            filter("request_latency", SlaStat::P95, SlaOp::Le, 100.0),
            filter("output_token_throughput", SlaStat::Avg, SlaOp::Ge, 1.0),
            filter("goodput_ratio", SlaStat::Avg, SlaOp::Ge, 0.0),
        ];
        let mut good = sample(80, Some(10), Some(5.0));
        good.output_sequence_length = Some(20);
        let mut slow = sample(120, Some(10), Some(5.0));
        slow.output_sequence_length = Some(40);
        let stats = WindowStats {
            successful_requests: vec![good, slow],
            errors: 1,
            cancelled: 1,
            elapsed_sec: 2.0,
            start_ns: 0,
            end_ns: 2_000_000_000,
        };
        let values = evaluator.values(&filters, &stats).unwrap();
        assert_eq!(values[&evaluator.key(&filters[1])], 30.0);
        assert_eq!(values[&evaluator.key(&filters[2])], 0.25);
    }

    #[test]
    fn aliases_operators_and_binding_margin_are_conjunctive() {
        let evaluator = DefaultSlaEvaluator;
        let filters = vec![
            filter("ttft", SlaStat::Avg, SlaOp::Lt, 30.0),
            filter("tpot", SlaStat::Avg, SlaOp::Le, 20.0),
            filter("request_throughput", SlaStat::Max, SlaOp::Ge, 1.0),
        ];
        let stats = WindowStats {
            successful_requests: vec![sample(80, Some(20), Some(10.0))],
            errors: 0,
            cancelled: 0,
            elapsed_sec: 1.0,
            start_ns: 0,
            end_ns: 1_000_000_000,
        };
        let values = evaluator.values(&filters, &stats).unwrap();
        assert!(evaluator.passes(&filters, &values).unwrap());
        assert_eq!(
            evaluator.binding_key(&filters, &values),
            Some(evaluator.key(&filters[2])),
            "throughput sits exactly on its ge threshold and therefore binds"
        );

        let at_bound = filter("request_latency", SlaStat::Avg, SlaOp::Lt, 80.0);
        let observed = evaluator
            .values(std::slice::from_ref(&at_bound), &stats)
            .unwrap();
        assert!(
            !evaluator
                .passes(std::slice::from_ref(&at_bound), &observed)
                .unwrap()
        );
    }

    #[test]
    fn unsupported_rate_percentile_and_metric_are_rejected() {
        let evaluator = DefaultSlaEvaluator;
        assert!(
            evaluator
                .validate_filters(&[filter("throughput", SlaStat::P95, SlaOp::Ge, 1.0)])
                .is_err()
        );
        assert!(
            evaluator
                .validate_filters(&[filter("tokens", SlaStat::Avg, SlaOp::Ge, 1.0)])
                .is_err()
        );
    }
}
