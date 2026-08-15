// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict timing normalization for recorded-agent replay.

use std::error::Error;
use std::fmt::{self, Display};

use serde::{Deserialize, Serialize};

use crate::graph::supplement::TraceTerminalSupplement;

/// Timing facts for one completed LLM call in a replay trace.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplayCallMeasurement {
    /// Owning replay trace.
    pub trace_id: String,
    /// Stable call ordinal within the trace.
    pub call_index: usize,
    /// Raw end-to-end request duration.
    pub raw_end_to_end_ms: f64,
    /// Raw inference duration.
    pub raw_inference_ms: f64,
    /// Raw generation duration.
    pub raw_generation_ms: f64,
    /// First meaningful output latency.
    pub ttft_ms: Option<f64>,
    /// Full response-stream duration.
    pub stream_total_ms: Option<f64>,
    /// Observed input-token count.
    pub observed_isl: u64,
    /// Observed output-token count.
    pub observed_osl: u64,
    /// Source recording output-token target.
    pub target_osl: u64,
    /// Source recording prompt-token count when it is available.
    pub recorded_prompt_isl: Option<u64>,
    /// Number of SSE events observed for the response.
    pub sse_event_count: u64,
    /// Whether the stream contained meaningful output.
    pub has_meaningful_output: bool,
    /// Whether the stream ended with its completion sentinel.
    pub has_done: bool,
    /// Whether required positive completion usage was observed.
    pub has_required_usage: bool,
}

impl ReplayCallMeasurement {
    /// Build a compact valid measurement for artifact and policy callers.
    pub fn completed(trace_id: impl Into<String>, call_index: usize) -> Self {
        Self {
            trace_id: trace_id.into(),
            call_index,
            raw_end_to_end_ms: 10.0,
            raw_inference_ms: 8.0,
            raw_generation_ms: 4.0,
            ttft_ms: Some(4.0),
            stream_total_ms: Some(8.0),
            observed_isl: 1,
            observed_osl: 2,
            target_osl: 2,
            recorded_prompt_isl: Some(1),
            sse_event_count: 2,
            has_meaningful_output: true,
            has_done: true,
            has_required_usage: true,
        }
    }
}

/// Normalized values and validity diagnostics for one replay call.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplayCallMetrics {
    /// Trace identity copied from the measurement.
    pub trace_id: String,
    /// Call ordinal copied from the measurement.
    pub call_index: usize,
    /// Safe raw end-to-end duration.
    pub raw_end_to_end_ms: f64,
    /// Safe raw inference duration.
    pub raw_inference_ms: f64,
    /// Safe raw generation duration.
    pub raw_generation_ms: f64,
    /// Observed input-token count.
    pub observed_isl: u64,
    /// Observed output-token count.
    pub observed_osl: u64,
    /// Source output-token target.
    pub target_osl: u64,
    /// Source prompt-token count when it is available.
    pub recorded_prompt_isl: Option<u64>,
    /// Ideal-prefix prompt growth attributed to this call.
    pub isl_delta: Option<u64>,
    /// Valid first-meaningful-output latency.
    pub ttft_ms: Option<f64>,
    /// Valid raw stream duration.
    pub stream_total_ms: Option<f64>,
    /// Normalized generation duration.
    pub normalized_generation_ms: Option<f64>,
    /// Normalized stream duration.
    pub normalized_stream_total_ms: Option<f64>,
    /// Normalized inference duration.
    pub normalized_inference_ms: Option<f64>,
    /// Normalized end-to-end duration.
    pub normalized_end_to_end_ms: Option<f64>,
    /// Structured invalidity and warning reasons in stable order.
    pub anomaly_reasons: Vec<String>,
}

impl ReplayCallMetrics {
    /// Whether this call can contribute decomposed and normalized timings.
    pub fn is_valid(&self) -> bool {
        self.anomaly_reasons
            .iter()
            .all(|reason| reason == "observed_osl_below_half_target")
    }
}

/// Folded normalized metrics for one successful replay trace.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ReplayTraceMetrics {
    /// Stable trace identity.
    pub trace_id: String,
    /// Per-call metrics in source-call order.
    pub calls: Vec<ReplayCallMetrics>,
    /// Aggregate normalized end-to-end time, absent after any anomaly.
    pub normalized_end_to_end_ms: Option<f64>,
    /// Aggregate normalized inference time, absent after any anomaly.
    pub normalized_inference_ms: Option<f64>,
    /// Aggregate normalized generation time, absent after any anomaly.
    pub normalized_generation_ms: Option<f64>,
    /// Aggregate TTFT, absent after any anomaly.
    pub ttft_ms: Option<f64>,
    /// Aggregate ideal-prefix ISL growth.
    pub isl_delta: u64,
}

/// Error returned by strict replay normalization or artifact production.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReplayMetricsError(String);

impl ReplayMetricsError {
    /// Build an explicit replay-metrics failure.
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl Display for ReplayMetricsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ReplayMetricsError {}

/// Injectable policy for comparable replay timing metrics.
pub trait ReplayMetricsPolicy: Send + Sync {
    /// Analyze one completed LLM call without mutating replay execution state.
    fn analyze_call(
        &self,
        measurement: &ReplayCallMeasurement,
    ) -> Result<ReplayCallMetrics, ReplayMetricsError>;

    /// Fold ordered call metrics into one successful trace result.
    fn fold_trace(
        &self,
        calls: &[ReplayCallMetrics],
        trace: &TraceTerminalSupplement,
    ) -> Result<ReplayTraceMetrics, ReplayMetricsError>;
}

/// Stock strict normalization policy used by recorded-agent replay.
#[derive(Clone, Copy, Debug, Default)]
pub struct StockReplayMetricsPolicy;

impl ReplayMetricsPolicy for StockReplayMetricsPolicy {
    fn analyze_call(
        &self,
        measurement: &ReplayCallMeasurement,
    ) -> Result<ReplayCallMetrics, ReplayMetricsError> {
        for (name, value) in [
            ("raw_end_to_end_ms", measurement.raw_end_to_end_ms),
            ("raw_inference_ms", measurement.raw_inference_ms),
            ("raw_generation_ms", measurement.raw_generation_ms),
        ] {
            if !value.is_finite() {
                return Err(ReplayMetricsError::new(format!(
                    "{name} must be finite for replay artifacts"
                )));
            }
        }
        let mut reasons = Vec::new();
        if measurement.raw_end_to_end_ms < 0.0
            || measurement.raw_inference_ms < 0.0
            || measurement.raw_generation_ms < 0.0
        {
            reasons.push("negative_duration".into());
        }
        let ttft = finite_positive(measurement.ttft_ms, "invalid_ttft", &mut reasons);
        let stream_total = finite_positive(
            measurement.stream_total_ms,
            "invalid_stream_total",
            &mut reasons,
        );
        if !measurement.has_meaningful_output {
            reasons.push("missing_meaningful_output".into());
        }
        if !measurement.has_done {
            reasons.push("missing_done".into());
        }
        if measurement.observed_osl > 0 && !measurement.has_required_usage {
            reasons.push("missing_required_usage".into());
        }
        if measurement.observed_osl > 1 && measurement.raw_generation_ms <= 0.0 {
            reasons.push("zero_generation_time".into());
        }
        if let (Some(ttft), Some(stream_total)) = (ttft, stream_total) {
            if ttft > stream_total || stream_total > measurement.raw_inference_ms {
                reasons.push("stream_outside_inference".into());
            }
            let generation = stream_total - ttft;
            if generation < 0.0 || (generation + ttft - stream_total).abs() > 1.0 {
                reasons.push("inconsistent_stream_timing".into());
            }
            if measurement.observed_osl > 1 && generation > 0.0 {
                let decode_tokens = measurement.observed_osl.saturating_sub(1).max(1) as f64;
                if decode_tokens / (generation / 1_000.0) > 10_000.0 {
                    reasons.push("impossible_generation_rate".into());
                }
            }
        }
        if measurement.observed_osl >= 16
            && measurement.sse_event_count > 0
            && measurement.observed_osl / measurement.sse_event_count > 32
        {
            reasons.push("sparse_sse_events".into());
        }
        if let Some(recorded_isl) = measurement.recorded_prompt_isl {
            let permitted = 128_u64.max(recorded_isl / 50);
            if measurement.observed_isl.saturating_add(permitted) < recorded_isl {
                reasons.push("server_isl_below_recorded_prompt".into());
            }
        }
        if measurement.target_osl > 0
            && measurement.observed_osl.saturating_mul(2) < measurement.target_osl
        {
            reasons.push("observed_osl_below_half_target".into());
        }

        let valid = reasons
            .iter()
            .all(|reason| reason == "observed_osl_below_half_target");
        let (
            normalized_generation_ms,
            normalized_stream_total_ms,
            normalized_inference_ms,
            normalized_end_to_end_ms,
        ) = if valid {
            let observed_decode = measurement.observed_osl.saturating_sub(1).max(1) as f64;
            let target_decode = measurement.target_osl.saturating_sub(1) as f64;
            let normalized_generation =
                measurement.raw_generation_ms / observed_decode * target_decode;
            let normalized_stream = ttft.unwrap_or_default() + normalized_generation;
            let normalized_inference = (measurement.raw_inference_ms
                - measurement.raw_generation_ms
                + normalized_generation)
                .max(0.0);
            let normalized_end_to_end = (measurement.raw_end_to_end_ms
                - measurement.raw_inference_ms
                + normalized_inference)
                .max(0.0);
            (
                Some(normalized_generation),
                Some(normalized_stream),
                Some(normalized_inference),
                Some(normalized_end_to_end),
            )
        } else {
            (None, None, None, None)
        };
        if [
            normalized_generation_ms,
            normalized_stream_total_ms,
            normalized_inference_ms,
            normalized_end_to_end_ms,
        ]
        .into_iter()
        .flatten()
        .any(|value| !value.is_finite())
        {
            reasons.push("nonfinite_normalization".into());
        }
        let normalized = reasons
            .iter()
            .all(|reason| reason == "observed_osl_below_half_target");
        Ok(ReplayCallMetrics {
            trace_id: measurement.trace_id.clone(),
            call_index: measurement.call_index,
            raw_end_to_end_ms: measurement.raw_end_to_end_ms,
            raw_inference_ms: measurement.raw_inference_ms,
            raw_generation_ms: measurement.raw_generation_ms,
            observed_isl: measurement.observed_isl,
            observed_osl: measurement.observed_osl,
            target_osl: measurement.target_osl,
            recorded_prompt_isl: measurement.recorded_prompt_isl,
            isl_delta: None,
            ttft_ms: normalized.then_some(ttft).flatten(),
            stream_total_ms: normalized.then_some(stream_total).flatten(),
            normalized_generation_ms: normalized.then_some(normalized_generation_ms).flatten(),
            normalized_stream_total_ms: normalized.then_some(normalized_stream_total_ms).flatten(),
            normalized_inference_ms: normalized.then_some(normalized_inference_ms).flatten(),
            normalized_end_to_end_ms: normalized.then_some(normalized_end_to_end_ms).flatten(),
            anomaly_reasons: reasons,
        })
    }

    fn fold_trace(
        &self,
        calls: &[ReplayCallMetrics],
        trace: &TraceTerminalSupplement,
    ) -> Result<ReplayTraceMetrics, ReplayMetricsError> {
        if calls.iter().any(|call| call.trace_id != trace.trace_id) {
            return Err(ReplayMetricsError::new(
                "replay call trace identity does not match terminal supplement",
            ));
        }
        let valid = calls.iter().all(ReplayCallMetrics::is_valid);
        let sum = |values: Vec<Option<f64>>| {
            values
                .into_iter()
                .collect::<Option<Vec<_>>>()
                .map(|values| values.into_iter().sum())
        };
        Ok(ReplayTraceMetrics {
            trace_id: trace.trace_id.clone(),
            calls: calls
                .iter()
                .enumerate()
                .map(|(index, call)| ReplayCallMetrics {
                    isl_delta: call.recorded_prompt_isl.map(|recorded| {
                        if index == 0 {
                            recorded
                        } else {
                            recorded
                                .saturating_sub(calls[index - 1].recorded_prompt_isl.unwrap_or(0))
                        }
                    }),
                    ..call.clone()
                })
                .collect(),
            normalized_end_to_end_ms: valid
                .then(|| {
                    sum(calls
                        .iter()
                        .map(|call| call.normalized_end_to_end_ms)
                        .collect())
                })
                .flatten(),
            normalized_inference_ms: valid
                .then(|| {
                    sum(calls
                        .iter()
                        .map(|call| call.normalized_inference_ms)
                        .collect())
                })
                .flatten(),
            normalized_generation_ms: valid
                .then(|| {
                    sum(calls
                        .iter()
                        .map(|call| call.normalized_generation_ms)
                        .collect())
                })
                .flatten(),
            ttft_ms: valid
                .then(|| sum(calls.iter().map(|call| call.ttft_ms).collect()))
                .flatten(),
            isl_delta: calls
                .iter()
                .enumerate()
                .filter_map(|(index, call)| {
                    call.recorded_prompt_isl.map(|recorded| {
                        if index == 0 {
                            recorded
                        } else {
                            recorded
                                .saturating_sub(calls[index - 1].recorded_prompt_isl.unwrap_or(0))
                        }
                    })
                })
                .sum(),
        })
    }
}

fn finite_positive(value: Option<f64>, reason: &str, reasons: &mut Vec<String>) -> Option<f64> {
    match value {
        Some(value) if value.is_finite() && value > 0.0 => Some(value),
        _ => {
            reasons.push(reason.into());
            None
        }
    }
}
