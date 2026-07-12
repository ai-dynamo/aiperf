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
    MetricValue, SidecarMetric, SidecarStats,
};
use serde::Serialize as DeriveSerialize;
use serde::ser::{Serialize, Serializer};
use serde_json::Value;
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

/// Histogram boundary-delta statistics supplied by server telemetry.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportHistogramStats {
    /// Number of phase observations.
    pub count: u64,
    /// Sum of phase observations.
    pub sum: ReportValue,
    /// Mean observation, when count is positive.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg: Option<ReportValue>,
    /// Observations per second over the authoritative phase window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub count_rate: Option<ReportValue>,
    /// Observation-value sum per second.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sum_rate: Option<ReportValue>,
    /// Polynomial percentile estimates keyed by `pN`.
    pub percentiles: BTreeMap<String, ReportValue>,
    /// Reset-clamped cumulative bucket deltas.
    pub buckets: BTreeMap<String, u64>,
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
    /// Prometheus histogram-shaped statistics.
    Histogram(ReportHistogramStats),
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
    /// Phase-bounded inference-server Prometheus metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ReportServerMetricsMetadata>,
}

/// Inclusive native phase window used to aggregate server telemetry.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct ReportServerMetricsPhaseRange {
    /// Start-boundary snapshot timestamp.
    pub start_ns: i64,
    /// End-boundary snapshot timestamp.
    pub end_ns: i64,
}

/// Fetch/update metadata for one server-metrics endpoint.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct ReportServerMetricsEndpointInfo {
    /// Successful fetch count including duplicate bodies.
    pub total_fetches: usize,
    /// First successful fetch timestamp.
    pub first_fetch_ns: i64,
    /// Last successful fetch timestamp.
    pub last_fetch_ns: i64,
    /// Mean successful HTTP latency in milliseconds.
    pub avg_fetch_latency_ms: f64,
    /// Changed-body count.
    pub unique_updates: usize,
    /// First changed-body timestamp, or zero when absent.
    pub first_update_ns: i64,
    /// Last changed-body timestamp, or zero when absent.
    pub last_update_ns: i64,
    /// Changed-body time span in seconds.
    pub duration_seconds: f64,
    /// Mean changed-body interval in milliseconds.
    pub avg_update_interval_ms: f64,
    /// Median changed-body interval in milliseconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub median_update_interval_ms: Option<f64>,
}

/// Metadata needed to render canonical Python server-metrics artifacts.
#[derive(Debug, Clone, Default, PartialEq, DeriveSerialize)]
pub struct ReportServerMetricsMetadata {
    /// Configured normalized endpoint URLs in stable order.
    pub endpoints_configured: Vec<String>,
    /// Endpoints contributing a complete profiling boundary pair.
    pub endpoints_successful: Vec<String>,
    /// Prometheus HELP text keyed by metric family name.
    pub descriptions: BTreeMap<String, String>,
    /// Original Prometheus semantic type keyed by metric family name.
    pub metric_types: BTreeMap<String, String>,
    /// Collection statistics keyed by credential-free endpoint URL.
    pub endpoint_info: BTreeMap<String, ReportServerMetricsEndpointInfo>,
    /// Profiling aggregation boundary.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ReportServerMetricsPhaseRange>,
    /// Warmup aggregation boundary when a warmup phase ran.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warmup: Option<ReportServerMetricsPhaseRange>,
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

/// Immutable dataset identity reported by the canonical accuracy evaluator.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluatorDatasetReportInfo {
    /// Dataset preparation implementation.
    pub provider: String,
    /// Canonical benchmark name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark: Option<String>,
    /// Dataset repository, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// Dataset subset/configuration, when applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subset: Option<String>,
    /// Immutable dataset revision.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revision: Option<String>,
    /// Evaluation splits selected by the canonical task.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub evaluation_splits: Vec<String>,
    /// Canonical task version, when exposed by the evaluator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_version: Option<u64>,
}

/// Exact evaluator runtime and benchmark identity retained in an accuracy report.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct EvaluatorReportInfo {
    /// Negotiated stdio protocol version.
    pub protocol: u32,
    /// Version of the AIPerf Python worker adapter.
    pub worker_version: String,
    /// Python runtime version.
    pub python_version: String,
    /// Python executable used for this run.
    pub python_executable: String,
    /// Evaluator package versions; absent optional packages remain null.
    pub packages: BTreeMap<String, Option<String>>,
    /// SHA-256 of the worker source.
    pub worker_source_sha256: String,
    /// SHA-256 of the fully pinned evaluator dependency lock, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependency_lock_sha256: Option<String>,
    /// Immutable worker container digest, when supplied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_digest: Option<String>,
    /// Worker capabilities negotiated during initialization.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Canonical benchmark name resolved by the worker.
    pub benchmark: String,
    /// Canonical grader or Lighteval metric implementation.
    pub grader: String,
    /// Dataset/task identity frozen by the load operation.
    pub dataset: EvaluatorDatasetReportInfo,
}

/// Exact stateful harness identity retained beside the generic worker identity.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct AgenticEvaluatorReportInfo {
    /// Canonical harness name.
    pub harness: String,
    /// Exact harness package version.
    pub harness_version: String,
    /// SHA-256 over the installed harness sources.
    pub harness_source_sha256: String,
    /// Agent scaffold name.
    pub agent: String,
    /// Exact adapter and inherited scaffold version.
    pub agent_version: String,
    /// Provider-owned canonical agent controls used for the evaluation.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub canonical_agent_config: BTreeMap<String, Value>,
    /// Environment provider used for task sandboxes.
    pub environment: String,
    /// Canonical verifier implementation description.
    pub verifier: String,
}

/// Reproducibility-relevant configuration for one agentic evaluation.
#[derive(Debug, Clone, PartialEq, Eq, DeriveSerialize)]
pub struct AgenticRunConfigReport {
    /// Requested immutable Harbor package or local dataset path.
    pub dataset: String,
    /// Optional exact task names selected from the package.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_names: Option<Vec<String>>,
    /// Optional deterministic episode cap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_episodes: Option<usize>,
    /// Maximum simultaneously active task environments.
    pub task_concurrency: usize,
    /// Maximum simultaneously active model calls.
    pub model_concurrency: usize,
    /// Harness artifact root.
    pub output_dir: String,
    /// Optional model-call limit per episode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_turns: Option<usize>,
    /// Maximum generated tokens per model call.
    pub max_tokens: usize,
    /// Explicit context-window limit used by the agent scaffold.
    pub context_window: usize,
    /// Canonical agent command parser.
    pub parser: String,
    /// Whether canonical context summarization was enabled.
    pub enable_summarize: bool,
    /// Optional explicitly selected primary verifier reward.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// Whether cached task packages could be replaced.
    pub overwrite: bool,
    /// Rust-owned callback ingress advertised to evaluator environments.
    ///
    /// The per-run bearer credential is intentionally never reported.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_gateway_base_url: Option<String>,
}

/// Generic aggregate statistics over one canonical verifier reward.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticRewardSummary {
    /// Number of completed episodes reporting this reward.
    pub n: usize,
    /// Arithmetic mean over canonical verifier values.
    pub avg: f64,
    /// Minimum canonical verifier value.
    pub min: f64,
    /// Maximum canonical verifier value.
    pub max: f64,
}

/// Run-level agentic result summary.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticEvaluationSummary {
    /// Every selected episode, regardless of terminal class.
    pub episode_count: usize,
    /// Episodes that reached canonical verification.
    pub completed_count: usize,
    /// Episodes that failed in inference, environment, harness, or verification infrastructure.
    pub infrastructure_error_count: usize,
    /// Episodes explicitly cancelled by Rust policy.
    pub cancelled_count: usize,
    /// All primary, environment, and verifier calls dispatched by Rust.
    pub model_calls: usize,
    /// Canonical agent calls emitted through the evaluator protocol.
    pub primary_model_calls: usize,
    /// Calls requested by task environments and canonical verifiers.
    pub auxiliary_model_calls: usize,
    /// Auxiliary calls requested by task environments.
    pub environment_model_calls: usize,
    /// Auxiliary calls requested by canonical verifiers.
    pub verifier_model_calls: usize,
    /// Prompt tokens across all calls when every call reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Completion tokens across all calls when every call reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Cached tokens across all calls when every call reported usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Prompt tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_prompt_tokens: Option<u64>,
    /// Completion tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_completion_tokens: Option<u64>,
    /// Cached tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_cached_tokens: Option<u64>,
    /// Prompt tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_prompt_tokens: Option<u64>,
    /// Completion tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_completion_tokens: Option<u64>,
    /// Cached tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_cached_tokens: Option<u64>,
    /// Uniform primary reward selected for the run, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// Mean primary reward over completed episodes only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_score: Option<f64>,
    /// Canonical reward aggregates keyed by verifier-owned name.
    pub rewards: BTreeMap<String, AgenticRewardSummary>,
}

/// Terminal class for one report-safe agentic episode record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, DeriveSerialize)]
#[serde(rename_all = "snake_case")]
pub enum AgenticEpisodeReportOutcome {
    /// The canonical verifier returned rewards.
    Completed,
    /// Inference, environment, harness, or verifier infrastructure failed.
    InfrastructureError,
    /// Rust policy cancelled the episode.
    Cancelled,
}

/// Full canonical result for one opaque agentic episode.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticEpisodeReport {
    /// Opaque evaluator-owned episode identifier.
    pub episode_id: String,
    /// Canonical task label.
    pub task: String,
    /// Explicit terminal classification.
    pub outcome: AgenticEpisodeReportOutcome,
    /// Finite verifier rewards, empty for non-completed episodes.
    pub rewards: BTreeMap<String, f64>,
    /// Per-episode selected primary reward.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_reward: Option<String>,
    /// End-to-end harness wall time.
    pub duration_seconds: f64,
    /// Number of Rust-owned inference calls.
    pub model_calls: usize,
    /// Canonical agent calls emitted through the evaluator protocol.
    pub primary_model_calls: usize,
    /// Calls requested by task environments and canonical verifiers.
    pub auxiliary_model_calls: usize,
    /// Auxiliary calls requested by the task environment.
    pub environment_model_calls: usize,
    /// Auxiliary calls requested by the canonical verifier.
    pub verifier_model_calls: usize,
    /// Aggregate prompt tokens reported by Rust.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    /// Aggregate completion tokens reported by Rust.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u64>,
    /// Aggregate cached prompt tokens reported by Rust.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u64>,
    /// Prompt tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_prompt_tokens: Option<u64>,
    /// Completion tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_completion_tokens: Option<u64>,
    /// Cached prompt tokens from canonical agent calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub primary_cached_tokens: Option<u64>,
    /// Prompt tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_prompt_tokens: Option<u64>,
    /// Completion tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_completion_tokens: Option<u64>,
    /// Cached prompt tokens from environment and verifier calls only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub auxiliary_cached_tokens: Option<u64>,
    /// Infrastructure or cancellation category.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    /// Infrastructure or cancellation detail.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    /// Canonical harness artifact path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_path: Option<String>,
}

/// Typed native-v2 agentic evaluation block.
#[derive(Debug, Clone, PartialEq, DeriveSerialize)]
pub struct AgenticEvaluationReport {
    /// Exact harness, agent, environment, and verifier identity.
    pub evaluator: AgenticEvaluatorReportInfo,
    /// Reproducibility-relevant authored configuration.
    pub config: AgenticRunConfigReport,
    /// Generic aggregates over canonical verifier outputs.
    pub summary: AgenticEvaluationSummary,
    /// Complete results in frozen evaluator order.
    pub records: Vec<AgenticEpisodeReport>,
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
    /// Profiling inference-server Prometheus series, kept outside request metrics.
    pub server_metrics: BTreeMap<String, SidecarMetric>,
    /// Warmup inference-server Prometheus series.
    pub warmup_server_metrics: BTreeMap<String, SidecarMetric>,
    /// Optional accuracy/analyzer output.
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records in deterministic workload order.
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Exact external evaluator identity for accuracy runs.
    pub evaluator: Option<EvaluatorReportInfo>,
    /// Optional stateful agentic evaluator result block.
    pub agentic: Option<AgenticEvaluationReport>,
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
            server_metrics: build_sidecar_map(&outcome.server_metrics),
            warmup_server_metrics: build_sidecar_map(&outcome.warmup_server_metrics),
            accuracy: outcome.accuracy.clone(),
            accuracy_records: outcome.accuracy_records.clone(),
            evaluator: outcome.evaluator.clone(),
            agentic: outcome.agentic.clone(),
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
    /// Profiling server telemetry keyed by original Prometheus family name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub server_metrics: BTreeMap<String, MetricEntry>,
    /// Warmup server telemetry keyed by original Prometheus family name.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub warmup_server_metrics: BTreeMap<String, MetricEntry>,
    /// Optional accuracy analysis.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub accuracy: Option<AccuracyAnalysis>,
    /// Full per-request grading records. Empty outside accuracy mode.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub accuracy_records: Vec<AccuracyRecord>,
    /// Exact canonical evaluator identity. Absent outside accuracy mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evaluator: Option<EvaluatorReportInfo>,
    /// Stateful harness identity, configuration, summary, and episode records.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agentic: Option<AgenticEvaluationReport>,
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
    let mut metrics = summary
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
        .collect::<BTreeMap<_, _>>();
    for (name, metric) in summary.sidecar_metrics() {
        metrics
            .entry(name.clone())
            .or_insert_with(|| report_sidecar_metric(metric));
    }
    metrics
}

fn build_sidecar_map(metrics: &BTreeMap<String, SidecarMetric>) -> BTreeMap<String, MetricEntry> {
    metrics
        .iter()
        .map(|(name, metric)| (name.clone(), report_sidecar_metric(metric)))
        .collect()
}

fn report_sidecar_metric(metric: &SidecarMetric) -> MetricEntry {
    let series = metric
        .series
        .iter()
        .map(|series| MetricSeries {
            labels: series.labels.clone(),
            endpoint_url: series.endpoint_url.clone(),
            stats: report_sidecar_stats(&series.stats),
            timeslices: series
                .timeslices
                .iter()
                .map(|slice| ReportTimeslice {
                    start_ns: slice.start_ns,
                    end_ns: slice.end_ns,
                    complete: slice.complete,
                    stats: report_sidecar_stats(&slice.stats),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let metric_type = series
        .first()
        .map(|series| stats_type(&series.stats))
        .unwrap_or("distribution");
    MetricEntry {
        metric_type,
        unit: metric
            .unit
            .map_or_else(String::new, |unit| unit.as_str().to_string()),
        group: console_group_name(metric.console_group),
        higher_is_better: metric.higher_is_better,
        series,
    }
}

fn report_sidecar_stats(stats: &SidecarStats) -> ReportStats {
    match stats {
        SidecarStats::Gauge(stats) => ReportStats::Distribution(report_distribution(stats, false)),
        SidecarStats::Counter { total, rate } => ReportStats::Counter(ReportCounterStats {
            total: report_value(*total).unwrap_or(ReportValue::NonFinite),
            rate: rate.and_then(report_value),
        }),
        SidecarStats::Histogram {
            count,
            sum,
            avg,
            count_rate,
            sum_rate,
            percentiles,
            buckets,
        } => ReportStats::Histogram(ReportHistogramStats {
            count: *count,
            sum: report_value(*sum).unwrap_or(ReportValue::NonFinite),
            avg: avg.and_then(report_value),
            count_rate: count_rate.and_then(report_value),
            sum_rate: sum_rate.and_then(report_value),
            percentiles: percentiles
                .iter()
                .filter_map(|(percentile, value)| {
                    report_value(*value).map(|value| (format!("p{percentile}"), value))
                })
                .collect(),
            buckets: buckets.clone(),
        }),
    }
}

fn report_distribution(
    stats: &crate::DistributionStats,
    adjusted: bool,
) -> ReportDistributionStats {
    ReportDistributionStats {
        count: (stats.count > 0).then_some(stats.count),
        avg: report_value(stats.avg),
        min: report_value(stats.min),
        max: report_value(stats.max),
        std: stats
            .std
            .map(ReportValue::Finite)
            .or(adjusted.then_some(ReportValue::NonFinite)),
        percentiles: stats
            .percentiles
            .iter()
            .filter_map(|(percentile, value)| {
                report_value(*value).map(|value| (format!("p{percentile}"), value))
            })
            .collect(),
    }
}

fn report_stats(
    result: &MetricResult,
    all_results: &BTreeMap<String, MetricResult>,
) -> Option<ReportStats> {
    match &result.data {
        MetricResultData::Distribution(stats) => {
            let adjusted = result.tag.starts_with("adj_");
            Some(ReportStats::Distribution(report_distribution(
                stats, adjusted,
            )))
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
        ReportStats::Histogram(_) => "histogram",
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
    use crate::{MetricResult, MetricResultData, SidecarMetric, SidecarSeries, SidecarStats, Unit};

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

    #[test]
    fn v2_retains_labeled_endpoint_histogram_sidecars() {
        let mut summary = AccumulatorSummary::new();
        summary.insert_sidecar_metric(
            "vllm:request_latency_seconds",
            SidecarMetric::new(
                Some(Unit::Second),
                vec![SidecarSeries {
                    labels: Some(BTreeMap::from([("model".to_string(), "m".to_string())])),
                    endpoint_url: Some("http://server/metrics".to_string()),
                    stats: SidecarStats::Histogram {
                        count: 2,
                        sum: MetricValue::Finite(0.3),
                        avg: Some(MetricValue::Finite(0.15)),
                        count_rate: Some(MetricValue::Finite(2.0)),
                        sum_rate: Some(MetricValue::Finite(0.3)),
                        percentiles: BTreeMap::from([(99, MetricValue::Finite(0.2))]),
                        buckets: BTreeMap::from([("0.1".to_string(), 1), ("+Inf".to_string(), 2)]),
                    },
                    timeslices: Vec::new(),
                }],
            ),
        );

        let value = serde_json::to_value(NativeReport::new(&summary, None)).unwrap();
        let metric = &value["metrics"]["vllm:request_latency_seconds"];
        assert_eq!(metric["type"], "histogram");
        assert_eq!(metric["unit"], "sec");
        assert_eq!(metric["series"][0]["endpoint_url"], "http://server/metrics");
        assert_eq!(metric["series"][0]["labels"]["model"], "m");
        assert_eq!(metric["series"][0]["stats"]["percentiles"]["p99"], 0.2);
        assert_eq!(metric["series"][0]["stats"]["buckets"]["+Inf"], 2);
    }
}
