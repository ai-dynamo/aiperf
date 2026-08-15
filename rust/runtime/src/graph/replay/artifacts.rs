// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Controller-owned deterministic replay artifact writers.

use std::fs;
use std::path::PathBuf;

use serde::Serialize;

use crate::graph::supplement::TraceTerminalSupplement;
use crate::graph::tools::ToolBackendIdentity;

use super::{
    ReplayCallMeasurement, ReplayCallMetrics, ReplayMetricsError, ReplayMetricsPolicy,
    ReplayTraceMetrics, StockReplayMetricsPolicy,
};

/// Final output destinations for replay artifacts, resolved by the controller.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayArtifactPaths {
    /// Optional strict tool-time JSON destination.
    pub tool_time_path: Option<PathBuf>,
    /// Optional strict trace-summary JSON destination.
    pub trace_summary_path: Option<PathBuf>,
    /// Optional normalized metrics JSON destination.
    pub metrics_json_path: Option<PathBuf>,
    /// Optional normalized metrics CSV destination.
    pub metrics_csv_path: Option<PathBuf>,
}

/// One attempted tool command retained without command or output bytes.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolCallMeasurement {
    /// Stable attempted-command order within the trace.
    #[serde(default)]
    pub call_index: usize,
    /// Measured command duration in seconds.
    pub duration_s: f64,
    /// Resolved execution backend identity.
    pub backend: String,
}

impl ToolCallMeasurement {
    /// Construct one bounded command measurement.
    pub fn new(duration_s: f64, backend: impl Into<String>) -> Self {
        Self {
            call_index: 0,
            duration_s,
            backend: backend.into(),
        }
    }

    /// Attach the trace-local command order captured by the graph sink.
    pub fn with_call_index(mut self, call_index: usize) -> Self {
        self.call_index = call_index;
        self
    }
}

/// Controller-folded facts for one trace terminal.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayTraceSupplement {
    /// Stable trace identity.
    pub trace_id: String,
    /// Stable recorded trajectory identity for tie-free output ordering.
    pub trajectory_id: String,
    /// Stable owner identity for equal trace/trajectory inputs.
    pub worker_id: usize,
    /// True only when the full profiling graph executor completed successfully.
    pub completed: bool,
    /// Completed LLM measurements in source-call order.
    pub calls: Vec<ReplayCallMeasurement>,
    /// Attempted tool commands in execution order.
    pub tools: Vec<ToolCallMeasurement>,
    /// Full graph wall duration in milliseconds.
    pub trace_wall_ms: f64,
}

impl From<&TraceTerminalSupplement> for ReplayTraceSupplement {
    fn from(supplement: &TraceTerminalSupplement) -> Self {
        Self {
            trace_id: supplement.trace_id.clone(),
            trajectory_id: supplement.trajectory_id.clone(),
            worker_id: supplement.worker_id,
            completed: supplement.completed,
            calls: supplement.calls.clone(),
            tools: supplement.tools.clone(),
            trace_wall_ms: supplement.trace_wall_ms,
        }
    }
}

/// Write replay artifacts after the controller folds all successful profiling traces.
///
/// This is intentionally a single controller entry point: workers only return bounded
/// measurements and never create any shared replay output file.
pub fn write_replay_artifacts(
    paths: &ReplayArtifactPaths,
    traces: &[ReplayTraceSupplement],
) -> Result<(), ReplayMetricsError> {
    let policy = StockReplayMetricsPolicy;
    let mut successful = traces
        .iter()
        .filter(|trace| trace.completed)
        .collect::<Vec<_>>();
    successful.sort_by(replay_trace_order);
    for trace in &successful {
        ensure_finite_nonnegative(trace.trace_wall_ms, "trace_wall_ms")?;
        for tool in &trace.tools {
            ensure_finite_nonnegative(tool.duration_s, "tool duration")?;
            ToolBackendIdentity::parse(&tool.backend)
                .map_err(|error| ReplayMetricsError::new(error.to_string()))?;
        }
    }
    let mut folded = successful
        .iter()
        .map(|trace| fold_trace(&policy, trace))
        .collect::<Result<Vec<_>, _>>()?;
    folded.sort_by(|left, right| left.trace_id.cmp(&right.trace_id));

    if let Some(path) = &paths.tool_time_path {
        let mut tools = Vec::new();
        for trace in &successful {
            let mut trace_tools = trace.tools.iter().collect::<Vec<_>>();
            trace_tools.sort_by_key(|tool| tool.call_index);
            if trace_tools
                .windows(2)
                .any(|pair| pair[0].call_index == pair[1].call_index)
            {
                return Err(ReplayMetricsError::new(
                    "replay trace contains duplicate tool call indices",
                ));
            }
            tools.extend(trace_tools);
        }
        if !tools.is_empty() {
            let durations_s = tools.iter().map(|tool| tool.duration_s).collect::<Vec<_>>();
            let total_s = rounded(checked_sum(durations_s.iter().copied(), "tool total")?);
            ensure_finite_nonnegative(total_s, "tool total")?;
            let mut sorted = durations_s.clone();
            sorted.sort_by(f64::total_cmp);
            let command_count = durations_s.len();
            let mean_s = rounded(total_s / command_count as f64);
            let median_s = rounded(median(&sorted)?);
            ensure_finite_nonnegative(mean_s, "tool mean")?;
            ensure_finite_nonnegative(median_s, "tool median")?;
            write_json(
                path,
                &ToolTimeArtifact {
                    command_count,
                    trace_count: successful
                        .iter()
                        .filter(|trace| !trace.tools.is_empty())
                        .count(),
                    backend: backend_label(&tools),
                    total_s,
                    mean_s,
                    median_s,
                    max_s: sorted.last().copied().unwrap_or_default(),
                    durations_s,
                },
            )?;
        }
    }
    if let Some(path) = &paths.trace_summary_path {
        let traces = successful
            .iter()
            .map(|trace| {
                let total_s = rounded(trace.trace_wall_ms / 1_000.0);
                let model_s = rounded(
                    checked_sum(
                        trace
                            .calls
                            .iter()
                            .map(|call| call.raw_inference_ms.max(0.0)),
                        "trace model total",
                    )? / 1_000.0,
                );
                let tool_s = rounded(checked_sum(
                    trace.tools.iter().map(|tool| tool.duration_s),
                    "trace tool total",
                )?);
                for (name, value) in [
                    ("trace total", total_s),
                    ("trace model", model_s),
                    ("trace tool", tool_s),
                ] {
                    ensure_finite_nonnegative(value, name)?;
                }
                ensure_finite_fraction(model_s, total_s, "trace model fraction")?;
                ensure_finite_fraction(tool_s, total_s, "trace tool fraction")?;
                Ok(TraceSummary {
                    trace_id: trace.trace_id.clone(),
                    total_s,
                    model_s,
                    tool_s,
                    model_calls: trace.calls.len(),
                    tool_calls: trace.tools.len(),
                })
            })
            .collect::<Result<Vec<_>, ReplayMetricsError>>()?;
        write_json(path, &TraceSummaryArtifact::new(traces)?)?;
    }
    if let Some(path) = &paths.metrics_json_path {
        write_json(path, &ReplayMetricsArtifact::new(folded.clone())?)?;
    }
    if let Some(path) = &paths.metrics_csv_path {
        write_metrics_csv(path, &folded)?;
    }
    Ok(())
}

fn fold_trace(
    policy: &dyn ReplayMetricsPolicy,
    trace: &ReplayTraceSupplement,
) -> Result<ReplayTraceMetrics, ReplayMetricsError> {
    let mut calls = trace
        .calls
        .iter()
        .map(|call| policy.analyze_call(call))
        .collect::<Result<Vec<ReplayCallMetrics>, _>>()?;
    calls.sort_by_key(|call| call.call_index);
    if calls
        .windows(2)
        .any(|pair| pair[0].call_index == pair[1].call_index)
    {
        return Err(ReplayMetricsError::new(
            "replay trace contains duplicate LLM call indices",
        ));
    }
    policy.fold_trace(
        &calls,
        &TraceTerminalSupplement::new(
            "controller".into(),
            trace.trace_id.clone(),
            trace.trace_id.clone(),
            0,
            "recorded_replay",
        ),
    )
}

fn replay_trace_order(
    left: &&ReplayTraceSupplement,
    right: &&ReplayTraceSupplement,
) -> std::cmp::Ordering {
    left.trace_id
        .cmp(&right.trace_id)
        .then_with(|| left.trajectory_id.cmp(&right.trajectory_id))
        .then_with(|| left.worker_id.cmp(&right.worker_id))
}

fn ensure_finite_nonnegative(value: f64, name: &str) -> Result<(), ReplayMetricsError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ReplayMetricsError::new(format!(
            "{name} must be finite and non-negative"
        )))
    }
}

fn write_json<T: Serialize>(path: &PathBuf, value: &T) -> Result<(), ReplayMetricsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ReplayMetricsError::new(format!(
                "creating replay artifact directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        ReplayMetricsError::new(format!("serializing replay artifact: {error}"))
    })?;
    fs::write(path, bytes).map_err(|error| {
        ReplayMetricsError::new(format!(
            "writing replay artifact {}: {error}",
            path.display()
        ))
    })
}

fn write_metrics_csv(
    path: &PathBuf,
    traces: &[ReplayTraceMetrics],
) -> Result<(), ReplayMetricsError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            ReplayMetricsError::new(format!(
                "creating replay CSV directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let mut output = String::from(
        "trace_id,call_index,observed_isl,observed_osl,target_osl,raw_end_to_end_ms,normalized_end_to_end_ms,anomaly_reasons\n",
    );
    for trace in traces {
        for call in &trace.calls {
            output.push_str(&csv_cell(&call.trace_id));
            output.push(',');
            output.push_str(&call.call_index.to_string());
            output.push(',');
            output.push_str(&call.observed_isl.to_string());
            output.push(',');
            output.push_str(&call.observed_osl.to_string());
            output.push(',');
            output.push_str(&call.target_osl.to_string());
            output.push(',');
            output.push_str(&call.raw_end_to_end_ms.to_string());
            output.push(',');
            if let Some(value) = call.normalized_end_to_end_ms {
                output.push_str(&value.to_string());
            }
            output.push(',');
            output.push_str(&csv_cell(&call.anomaly_reasons.join(";")));
            output.push('\n');
        }
    }
    fs::write(path, output).map_err(|error| {
        ReplayMetricsError::new(format!("writing replay CSV {}: {error}", path.display()))
    })
}

fn csv_cell(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.into()
    }
}

fn backend_label(tools: &[&ToolCallMeasurement]) -> String {
    let first = tools
        .first()
        .map(|tool| tool.backend.as_str())
        .unwrap_or("local");
    if tools.iter().all(|tool| tool.backend == first) {
        first.to_owned()
    } else {
        "mixed".into()
    }
}

fn median(sorted: &[f64]) -> Result<f64, ReplayMetricsError> {
    match sorted.len() {
        0 => Ok(0.0),
        n if n % 2 == 1 => Ok(sorted[n / 2]),
        n => checked_add(sorted[n / 2 - 1] / 2.0, sorted[n / 2] / 2.0, "tool median"),
    }
}

fn checked_add(left: f64, right: f64, name: &str) -> Result<f64, ReplayMetricsError> {
    let value = left + right;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ReplayMetricsError::new(format!(
            "{name} overflowed to a non-finite value"
        )))
    }
}

fn checked_sum(
    values: impl IntoIterator<Item = f64>,
    name: &str,
) -> Result<f64, ReplayMetricsError> {
    values.into_iter().try_fold(0.0, |total, value| {
        ensure_finite_nonnegative(value, name)?;
        checked_add(total, value, name)
    })
}

fn ensure_finite_fraction(value: f64, total: f64, name: &str) -> Result<(), ReplayMetricsError> {
    if total > 0.0 && !(value / total).is_finite() {
        return Err(ReplayMetricsError::new(format!(
            "{name} overflowed to a non-finite value"
        )));
    }
    Ok(())
}

#[derive(Serialize)]
struct ToolTimeArtifact {
    command_count: usize,
    trace_count: usize,
    backend: String,
    total_s: f64,
    mean_s: f64,
    median_s: f64,
    max_s: f64,
    durations_s: Vec<f64>,
}

#[derive(Serialize)]
struct TraceSummaryArtifact {
    trace_count: usize,
    aggregate: TraceSummaryAggregate,
    traces: Vec<TraceSummary>,
}

impl TraceSummaryArtifact {
    fn new(traces: Vec<TraceSummary>) -> Result<Self, ReplayMetricsError> {
        let total_s = rounded(checked_sum(
            traces.iter().map(|trace| trace.total_s),
            "trace total",
        )?);
        let model_s = rounded(checked_sum(
            traces.iter().map(|trace| trace.model_s),
            "model total",
        )?);
        let tool_s = rounded(checked_sum(
            traces.iter().map(|trace| trace.tool_s),
            "tool total",
        )?);
        let model_calls = checked_usize_sum(
            traces.iter().map(|trace| trace.model_calls),
            "model call count",
        )?;
        let tool_calls = checked_usize_sum(
            traces.iter().map(|trace| trace.tool_calls),
            "tool call count",
        )?;
        for (name, value) in [
            ("trace total", total_s),
            ("model total", model_s),
            ("tool total", tool_s),
        ] {
            ensure_finite_nonnegative(value, name)?;
        }
        ensure_finite_fraction(model_s, total_s, "aggregate model fraction")?;
        ensure_finite_fraction(tool_s, total_s, "aggregate tool fraction")?;
        Ok(Self {
            trace_count: traces.len(),
            aggregate: TraceSummaryAggregate {
                total_s,
                model_s,
                tool_s,
                model_time_fraction: fraction(model_s, total_s),
                tool_time_fraction: fraction(tool_s, total_s),
                model_calls,
                tool_calls,
            },
            traces,
        })
    }
}

fn checked_usize_sum(
    values: impl IntoIterator<Item = usize>,
    name: &str,
) -> Result<usize, ReplayMetricsError> {
    values.into_iter().try_fold(0_usize, |total, value| {
        total
            .checked_add(value)
            .ok_or_else(|| ReplayMetricsError::new(format!("{name} overflowed usize")))
    })
}

#[derive(Serialize)]
struct TraceSummaryAggregate {
    total_s: f64,
    model_s: f64,
    tool_s: f64,
    model_time_fraction: f64,
    tool_time_fraction: f64,
    model_calls: usize,
    tool_calls: usize,
}

struct TraceSummary {
    trace_id: String,
    total_s: f64,
    model_s: f64,
    tool_s: f64,
    model_calls: usize,
    tool_calls: usize,
}

impl TraceSummary {
    fn model_time_fraction(&self) -> f64 {
        fraction(self.model_s, self.total_s)
    }

    fn tool_time_fraction(&self) -> f64 {
        fraction(self.tool_s, self.total_s)
    }
}

impl Serialize for TraceSummary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        #[derive(Serialize)]
        struct Wire<'a> {
            trace_id: &'a str,
            total_s: f64,
            model_s: f64,
            tool_s: f64,
            model_time_fraction: f64,
            tool_time_fraction: f64,
            model_calls: usize,
            tool_calls: usize,
        }
        Wire {
            trace_id: &self.trace_id,
            total_s: self.total_s,
            model_s: self.model_s,
            tool_s: self.tool_s,
            model_time_fraction: self.model_time_fraction(),
            tool_time_fraction: self.tool_time_fraction(),
            model_calls: self.model_calls,
            tool_calls: self.tool_calls,
        }
        .serialize(serializer)
    }
}

fn fraction(value: f64, total: f64) -> f64 {
    if total > 0.0 {
        rounded(value / total)
    } else {
        0.0
    }
}

/// Keep decimal artifact output stable across associative finite additions.
fn rounded(value: f64) -> f64 {
    if !value.is_finite() {
        return value;
    }
    let scaled = value * 1_000_000_000_000.0;
    if !scaled.is_finite() {
        return value;
    }
    scaled.round() / 1_000_000_000_000.0
}

#[derive(Serialize)]
struct ReplayMetricsArtifact {
    trace_count: usize,
    aggregate: ReplayMetricsAggregate,
    traces: Vec<ReplayTraceMetrics>,
}

impl ReplayMetricsArtifact {
    fn new(traces: Vec<ReplayTraceMetrics>) -> Result<Self, ReplayMetricsError> {
        for trace in &traces {
            validate_metrics_trace(trace)?;
        }
        let aggregate = ReplayMetricsAggregate {
            normalized_end_to_end_ms: checked_option_sum(
                traces.iter().map(|trace| trace.normalized_end_to_end_ms),
                "normalized end-to-end",
            )?,
            normalized_inference_ms: checked_option_sum(
                traces.iter().map(|trace| trace.normalized_inference_ms),
                "normalized inference",
            )?,
            normalized_generation_ms: checked_option_sum(
                traces.iter().map(|trace| trace.normalized_generation_ms),
                "normalized generation",
            )?,
            ttft_ms: checked_option_sum(traces.iter().map(|trace| trace.ttft_ms), "ttft")?,
            observed_osl: traces
                .iter()
                .flat_map(|trace| trace.calls.iter())
                .map(|call| call.observed_osl)
                .try_fold(0_u64, |total, value| {
                    total.checked_add(value).ok_or_else(|| {
                        ReplayMetricsError::new("observed OSL aggregate overflowed u64")
                    })
                })?,
        };
        for value in [
            aggregate.normalized_end_to_end_ms,
            aggregate.normalized_inference_ms,
            aggregate.normalized_generation_ms,
            aggregate.ttft_ms,
        ]
        .into_iter()
        .flatten()
        {
            if !value.is_finite() {
                return Err(ReplayMetricsError::new(
                    "replay aggregate contains a non-finite value",
                ));
            }
        }
        Ok(Self {
            trace_count: traces.len(),
            aggregate,
            traces,
        })
    }
}

fn checked_option_sum(
    values: impl IntoIterator<Item = Option<f64>>,
    name: &str,
) -> Result<Option<f64>, ReplayMetricsError> {
    let mut total = 0.0;
    for value in values {
        let Some(value) = value else { return Ok(None) };
        total = checked_add(total, value, name)?;
    }
    Ok(Some(total))
}

fn validate_metrics_trace(trace: &ReplayTraceMetrics) -> Result<(), ReplayMetricsError> {
    for call in &trace.calls {
        for (name, value) in [
            ("raw end-to-end", call.raw_end_to_end_ms),
            ("raw inference", call.raw_inference_ms),
            ("raw generation", call.raw_generation_ms),
        ] {
            if !value.is_finite() {
                return Err(ReplayMetricsError::new(format!("{name} is non-finite")));
            }
        }
        for (name, value) in [
            ("ttft", call.ttft_ms),
            ("stream total", call.stream_total_ms),
            ("normalized generation", call.normalized_generation_ms),
            ("normalized stream", call.normalized_stream_total_ms),
            ("normalized inference", call.normalized_inference_ms),
            ("normalized end-to-end", call.normalized_end_to_end_ms),
        ] {
            if value.is_some_and(|value| !value.is_finite()) {
                return Err(ReplayMetricsError::new(format!("{name} is non-finite")));
            }
        }
    }
    Ok(())
}

#[derive(Serialize)]
struct ReplayMetricsAggregate {
    normalized_end_to_end_ms: Option<f64>,
    normalized_inference_ms: Option<f64>,
    normalized_generation_ms: Option<f64>,
    ttft_ms: Option<f64>,
    observed_osl: u64,
}
