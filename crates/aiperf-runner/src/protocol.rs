// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict JSON request/result contract for one native benchmark run.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// Current Python-orchestrator/Rust-runner protocol version.
pub const RUNNER_PROTOCOL_VERSION: u32 = 1;

/// Machine-readable runner capabilities returned by `--capabilities`.
#[derive(Debug, Serialize)]
pub struct RunnerCapabilities {
    /// Stable response discriminator.
    pub event: &'static str,
    /// Protocol versions accepted on stdin.
    pub protocol_versions: &'static [u32],
    /// Native report schema written after a successful run.
    pub report_schema_version: &'static str,
    /// Dataset variants accepted by the current protocol.
    pub dataset_types: &'static [&'static str],
    /// Phase variants accepted by the current protocol.
    pub phase_types: &'static [&'static str],
    /// Rust runner package version.
    pub runner_version: &'static str,
}

impl RunnerCapabilities {
    /// Describe the exact process contract implemented by this binary.
    pub const fn current() -> Self {
        Self {
            event: "runner_capabilities",
            protocol_versions: &[RUNNER_PROTOCOL_VERSION],
            report_schema_version: aiperf_metrics::NATIVE_REPORT_SCHEMA_VERSION,
            dataset_types: &["synthetic", "file"],
            phase_types: &[
                "concurrency",
                "poisson",
                "gamma",
                "constant",
                "user_centric",
                "fixed_schedule",
            ],
            runner_version: env!("CARGO_PKG_VERSION"),
        }
    }
}

/// One complete single-run request read from stdin.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunRequest {
    /// Wire protocol version, independent of Config v2 and report versions.
    pub protocol_version: u32,
    /// Fully resolved run identity and native benchmark configuration.
    pub run: RunSpec,
}

/// Fully resolved identity and execution inputs for one benchmark process.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunSpec {
    /// Stable Python-orchestrator benchmark identifier.
    pub benchmark_id: String,
    /// Outer sweep identifier.
    #[serde(default)]
    pub sweep_id: Option<String>,
    /// Human-readable run label.
    #[serde(default)]
    pub label: String,
    /// Zero-based trial number.
    #[serde(default)]
    pub trial: usize,
    /// Deterministic run seed; absent selects entropy-backed component streams.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Sweep variation metadata retained by the outer orchestrator.
    #[serde(default)]
    pub variation: Option<VariationSpec>,
    /// Exclusive per-run artifact directory selected by Python.
    pub artifact_dir: PathBuf,
    /// Model selection policy applied while composing requests.
    pub models: ModelsSpec,
    /// HTTP endpoint and dialect policy.
    pub endpoint: EndpointSpec,
    /// Dataset authored for this run.
    pub dataset: DatasetSpec,
    /// Tokenizer resolved and cache-localized by Python Config v2.
    #[serde(default)]
    pub tokenizer: TokenizerSpec,
    /// Ordered warmup/profiling phase list.
    pub phases: Vec<PhaseSpec>,
    /// Native metric-engine configuration.
    #[serde(default)]
    pub metrics: MetricsSpec,
    /// Per-run artifact outputs written by Rust.
    #[serde(default)]
    pub artifacts: ArtifactSpec,
}

/// Tokenizer source understood by the native dataset composer.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSpec {
    /// Built-in encoding name, local tokenizer.json, or local model directory.
    #[serde(default = "default_tokenizer_name")]
    pub name: String,
}

impl Default for TokenizerSpec {
    fn default() -> Self {
        Self {
            name: default_tokenizer_name(),
        }
    }
}

fn default_tokenizer_name() -> String {
    "builtin".into()
}

/// Native metric aggregation settings lowered from Config v2.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsSpec {
    /// Optional trend timeslice duration in seconds.
    #[serde(default)]
    pub slice_duration_seconds: Option<f64>,
    /// Per-request SLO thresholds in each metric's display unit.
    #[serde(default)]
    pub slos: BTreeMap<String, f64>,
}

/// Artifact paths relative to the exclusive run directory.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSpec {
    /// Per-request metrics JSONL path, or absent when records are disabled.
    #[serde(default)]
    pub records_path: Option<PathBuf>,
    /// Include transport timing details on JSONL records.
    #[serde(default)]
    pub trace: bool,
}

/// Outer-loop variation coordinates carried through process results.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VariationSpec {
    /// Zero-based variation index.
    pub index: usize,
    /// Stable display/search label.
    pub label: String,
    /// Canonical parameter path to authored value.
    #[serde(default)]
    pub values: BTreeMap<String, Value>,
}

/// Selection policy for one or more inference models.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelsSpec {
    /// Model selection algorithm.
    #[serde(default)]
    pub strategy: ModelSelectionStrategy,
    /// Non-empty model list.
    pub items: Vec<ModelItemSpec>,
}

/// Supported model selection algorithms.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelSelectionStrategy {
    /// Deterministic cycling in authored order.
    #[default]
    RoundRobin,
    /// Uniform random selection.
    Random,
    /// Authored weighted random selection.
    Weighted,
}

/// One selectable inference model.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelItemSpec {
    /// Server-facing model identifier.
    pub name: String,
    /// Required only for weighted selection.
    #[serde(default)]
    pub weight: Option<f64>,
}

/// HTTP and endpoint-dialect policy needed by the native transport.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EndpointSpec {
    /// Ordered non-empty endpoint URLs.
    pub urls: Vec<String>,
    /// Endpoint dialect registered in `aiperf-endpoints`.
    #[serde(rename = "type")]
    pub endpoint_type: aiperf_endpoints::EndpointType,
    /// Optional endpoint path override.
    #[serde(default)]
    pub path: Option<String>,
    /// Whether responses use SSE streaming.
    pub streaming: bool,
    /// Use legacy `max_tokens` instead of `max_completion_tokens`.
    #[serde(default)]
    pub use_legacy_max_tokens: bool,
    /// Request and trust server token usage.
    #[serde(default)]
    pub use_server_token_count: bool,
    /// Request-level timeout in seconds.
    #[serde(default = "default_timeout_seconds")]
    pub timeout_seconds: f64,
    /// Custom template body.
    #[serde(default)]
    pub template: Option<String>,
    /// Custom template response selector.
    #[serde(default)]
    pub response_field: Option<String>,
    /// Extra request-body fields.
    #[serde(default)]
    pub extra: Map<String, Value>,
    /// Headers merged into every materialized request.
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
    /// Optional bearer token, carried only over the stdin pipe.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Optional session-affinity header name.
    #[serde(default)]
    pub session_header: Option<String>,
    /// Use h2c prior knowledge for cleartext HTTP/2.
    #[serde(default)]
    pub http2: bool,
}

const fn default_timeout_seconds() -> f64 {
    6.0 * 60.0 * 60.0
}

/// Dataset variants accepted by protocol version 1.
#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum DatasetSpec {
    /// Generated text conversation dataset.
    Synthetic(SyntheticDatasetSpec),
    /// Local path or inline records parsed by the native loader registry.
    File(FileDatasetSpec),
}

/// Resolved file/inline dataset configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileDatasetSpec {
    /// Absolute resolved path, mutually exclusive with records.
    #[serde(default)]
    pub path: Option<PathBuf>,
    /// Inline records in the exact Config-v2 shape.
    #[serde(default)]
    pub records: Option<Value>,
    /// Native loader registration name.
    pub format: String,
    /// Conversation sampling strategy.
    #[serde(default = "default_sampling_strategy")]
    pub sampling: String,
    /// Optional row cap applied before composition.
    #[serde(default)]
    pub entries: Option<usize>,
    /// Dataset-local seed overriding the run seed.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Output-length fallback for rows without an authored limit.
    #[serde(default)]
    pub osl: Option<DistributionSpec>,
    /// Loader/composer-specific options after Config-v2 validation.
    #[serde(default)]
    pub options: Map<String, Value>,
}

fn default_sampling_strategy() -> String {
    "sequential".into()
}

/// Native synthetic dataset configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticDatasetSpec {
    /// Number of reusable conversations.
    pub entries: usize,
    /// Input/output token distributions.
    pub prompts: SyntheticPromptsSpec,
    /// Turns per conversation.
    #[serde(default = "one_distribution")]
    pub turns: DistributionSpec,
    /// Inter-turn delay in milliseconds.
    #[serde(default = "zero_distribution")]
    pub turn_delay_ms: DistributionSpec,
    /// Multiplicative delay scale.
    #[serde(default = "one_f64")]
    pub turn_delay_ratio: f64,
}

/// Synthetic prompt distributions.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticPromptsSpec {
    /// Input sequence length distribution.
    pub isl: DistributionSpec,
    /// Output sequence length distribution.
    pub osl: DistributionSpec,
    /// Independently generated prompt values per turn.
    #[serde(default = "one_usize")]
    pub batch_size: usize,
}

/// Config-v2 sampling distribution after Pydantic normalization.
#[derive(Clone, Deserialize)]
#[serde(untagged)]
pub enum DistributionSpec {
    /// Deterministic value.
    Fixed(FixedDistributionSpec),
    /// Positive normal distribution.
    Normal(NormalDistributionSpec),
    /// Real-space mean/median log-normal distribution.
    LogNormal(LogNormalDistributionSpec),
    /// Weighted mixture.
    Multimodal(MultimodalDistributionSpec),
    /// Discrete weighted values.
    Empirical(EmpiricalDistributionSpec),
}

/// Deterministic distribution configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixedDistributionSpec {
    /// Constant value.
    pub value: f64,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// Normal distribution configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalDistributionSpec {
    /// Mean.
    pub mean: f64,
    /// Standard deviation.
    pub stddev: f64,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// Log-normal distribution configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LogNormalDistributionSpec {
    /// Real-space mean.
    pub mean: f64,
    /// Real-space median.
    pub median: f64,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// Weighted mixture configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MultimodalDistributionSpec {
    /// Weighted component distributions.
    pub peaks: Vec<PeakSpec>,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// One weighted mixture component.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeakSpec {
    /// Nested distribution.
    pub distribution: DistributionSpec,
    /// Relative non-negative weight.
    pub weight: f64,
}

/// Discrete empirical configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmpiricalDistributionSpec {
    /// Weighted discrete values.
    pub points: Vec<EmpiricalPointSpec>,
    /// Optional lower bound.
    #[serde(default)]
    pub min: Option<f64>,
    /// Optional upper bound.
    #[serde(default)]
    pub max: Option<f64>,
}

/// One discrete value and weight.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmpiricalPointSpec {
    /// Sampled value.
    pub value: f64,
    /// Relative positive weight.
    pub weight: f64,
}

fn one_distribution() -> DistributionSpec {
    DistributionSpec::Fixed(FixedDistributionSpec {
        value: 1.0,
        min: None,
        max: None,
    })
}

fn zero_distribution() -> DistributionSpec {
    DistributionSpec::Fixed(FixedDistributionSpec {
        value: 0.0,
        min: None,
        max: None,
    })
}

const fn one_f64() -> f64 {
    1.0
}

const fn one_usize() -> usize {
    1
}

/// Ordered phase variants accepted by the native scheduler.
#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PhaseSpec {
    /// Closed-loop concurrency scheduling.
    Concurrency {
        /// Shared phase policy.
        #[serde(flatten)]
        common: PhaseCommonSpec,
        /// Active session limit.
        concurrency: usize,
    },
    /// Poisson request-rate scheduling.
    Poisson {
        /// Shared phase policy.
        #[serde(flatten)]
        common: PhaseCommonSpec,
        /// Mean turns per second.
        rate: f64,
        /// Optional active-session cap.
        #[serde(default)]
        concurrency: Option<usize>,
    },
    /// Gamma request-rate scheduling.
    Gamma {
        /// Shared phase policy.
        #[serde(flatten)]
        common: PhaseCommonSpec,
        /// Mean turns per second.
        rate: f64,
        /// Gamma shape parameter.
        #[serde(default)]
        smoothness: Option<f64>,
        /// Optional active-session cap.
        #[serde(default)]
        concurrency: Option<usize>,
    },
    /// Constant-interval request-rate scheduling.
    Constant {
        /// Shared phase policy.
        #[serde(flatten)]
        common: PhaseCommonSpec,
        /// Mean turns per second.
        rate: f64,
        /// Optional active-session cap.
        #[serde(default)]
        concurrency: Option<usize>,
    },
    /// Per-user open-loop pacing and churn.
    UserCentric {
        /// Shared phase policy.
        #[serde(flatten)]
        common: PhaseCommonSpec,
        /// Aggregate turns per second across users.
        rate: f64,
        /// Initial number of simulated users.
        users: usize,
        /// Optional concurrent-session cap.
        #[serde(default)]
        concurrency: Option<usize>,
    },
    /// Replay dataset-authored timestamps.
    FixedSchedule {
        /// Shared phase policy.
        #[serde(flatten)]
        common: PhaseCommonSpec,
        /// Normalize the first retained timestamp to phase start.
        #[serde(default = "true_value")]
        auto_offset: bool,
        /// Inclusive trace filter and manual schedule zero in milliseconds.
        #[serde(default)]
        start_offset: Option<f64>,
        /// Inclusive trace end filter in milliseconds.
        #[serde(default)]
        end_offset: Option<f64>,
    },
}

const fn true_value() -> bool {
    true
}

/// Policy shared by every phase scheduling variant.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseCommonSpec {
    /// Stable phase name (`warmup` or `profiling`).
    pub name: String,
    /// Exclude phase metrics from profiling output.
    pub exclude_from_results: bool,
    /// Stop after this many issued turns.
    #[serde(default)]
    pub requests: Option<u64>,
    /// Stop after this many started sessions.
    #[serde(default)]
    pub sessions: Option<u64>,
    /// Stop after this duration in seconds.
    #[serde(default)]
    pub duration: Option<f64>,
    /// Prefill concurrency cap.
    #[serde(default)]
    pub prefill_concurrency: Option<usize>,
    /// Additional return grace after duration expiry.
    #[serde(default)]
    pub grace_period: Option<f64>,
    /// Handoff after sending instead of waiting for returns.
    #[serde(default)]
    pub seamless: bool,
    /// Session-concurrency ramp.
    #[serde(default)]
    pub concurrency_ramp: Option<RampSpec>,
    /// Prefill-concurrency ramp.
    #[serde(default)]
    pub prefill_ramp: Option<RampSpec>,
    /// Request-rate ramp.
    #[serde(default)]
    pub rate_ramp: Option<RampSpec>,
    /// Post-send cancellation policy.
    #[serde(default)]
    pub cancellation: Option<CancellationSpec>,
}

/// One Clock-driven phase ramp.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RampSpec {
    /// Total duration in seconds.
    pub duration: f64,
    /// Curve type.
    #[serde(default)]
    pub strategy: RampStrategySpec,
}

/// Supported Clock-driven ramp curves.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RampStrategySpec {
    /// Linear curve.
    #[default]
    Linear,
    /// Exponential ease-in curve.
    Exponential,
    /// Seeded Poisson step trajectory.
    Poisson,
}

/// Post-send cancellation configuration.
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CancellationSpec {
    /// Percentage in `[0, 100]`.
    pub rate: f64,
    /// Delay after request send completion in seconds.
    #[serde(default)]
    pub delay: f64,
}

impl PhaseSpec {
    /// Shared policy fields.
    pub fn common(&self) -> &PhaseCommonSpec {
        match self {
            Self::Concurrency { common, .. }
            | Self::Poisson { common, .. }
            | Self::Gamma { common, .. }
            | Self::Constant { common, .. }
            | Self::UserCentric { common, .. }
            | Self::FixedSchedule { common, .. } => common,
        }
    }

    /// Effective session-concurrency target.
    pub fn concurrency(&self) -> Option<usize> {
        match self {
            Self::Concurrency { concurrency, .. } => Some(*concurrency),
            Self::Poisson { concurrency, .. }
            | Self::Gamma { concurrency, .. }
            | Self::Constant { concurrency, .. }
            | Self::UserCentric { concurrency, .. } => *concurrency,
            Self::FixedSchedule { .. } => None,
        }
    }

    /// Request-rate arrival policy, absent for schedule-authored workloads.
    pub fn request_arrival(
        &self,
    ) -> Option<(aiperf_timing::ArrivalPattern, Option<f64>, Option<f64>)> {
        match self {
            Self::Concurrency { .. } => {
                Some((aiperf_timing::ArrivalPattern::ConcurrencyBurst, None, None))
            }
            Self::Poisson { rate, .. } => {
                Some((aiperf_timing::ArrivalPattern::Poisson, Some(*rate), None))
            }
            Self::Gamma {
                rate, smoothness, ..
            } => Some((
                aiperf_timing::ArrivalPattern::Gamma,
                Some(*rate),
                *smoothness,
            )),
            Self::Constant { rate, .. } => {
                Some((aiperf_timing::ArrivalPattern::Constant, Some(*rate), None))
            }
            Self::UserCentric { .. } | Self::FixedSchedule { .. } => None,
        }
    }

    /// Target authored rate for request-rate and user-centric workloads.
    pub fn rate(&self) -> Option<f64> {
        match self {
            Self::Poisson { rate, .. }
            | Self::Gamma { rate, .. }
            | Self::Constant { rate, .. }
            | Self::UserCentric { rate, .. } => Some(*rate),
            Self::Concurrency { .. } | Self::FixedSchedule { .. } => None,
        }
    }
}

/// Terminal subprocess response written as exactly one JSON line.
#[derive(Debug, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RunTerminal {
    /// Protocol version used for this response.
    pub protocol_version: u32,
    /// Stable terminal event discriminator.
    pub event: &'static str,
    /// Run identifier when the request was decoded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub benchmark_id: Option<String>,
    /// Whether the native benchmark completed and committed its report.
    pub success: bool,
    /// Authoritative native-v2 report path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_path: Option<PathBuf>,
    /// Stable failure category.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_kind: Option<String>,
    /// Human-readable failure details.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Rust runner package version.
    pub runner_version: &'static str,
}

impl RunTerminal {
    /// Construct a successful terminal response.
    pub fn succeeded(benchmark_id: String, report_path: PathBuf) -> Self {
        Self {
            protocol_version: RUNNER_PROTOCOL_VERSION,
            event: "run_terminal",
            benchmark_id: Some(benchmark_id),
            success: true,
            report_path: Some(report_path),
            error_kind: None,
            error: None,
            runner_version: env!("CARGO_PKG_VERSION"),
        }
    }

    /// Construct a failed terminal response.
    pub fn failed(
        benchmark_id: Option<String>,
        error_kind: impl Into<String>,
        error: impl Into<String>,
    ) -> Self {
        Self {
            protocol_version: RUNNER_PROTOCOL_VERSION,
            event: "run_terminal",
            benchmark_id,
            success: false,
            report_path: None,
            error_kind: Some(error_kind.into()),
            error: Some(error.into()),
            runner_version: env!("CARGO_PKG_VERSION"),
        }
    }
}
