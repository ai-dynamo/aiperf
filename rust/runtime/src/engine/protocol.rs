// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict JSON request/result contract for one native benchmark run.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::extensions::AIPerfRegistry;

/// Admission strategy for `workers>1` scheduled execution.
///
/// - `Sharded` statically partitions request budget, concurrency, and rate
///   `1/workers`-ways up front, per worker thread as a throughput-oriented
///   opt-in.
/// - `Global` (default) admits from one shared [`crate::timing::slots::GlobalSlotPool`]
///   / [`crate::timing::rate_gate::GlobalRateGate`] per cell, so aggregate
///   concurrency and rate across all worker threads is byte-exact against a
///   single global limiter, matching the Python baseline.
/// - `GlobalHop` additionally routes every individual request through one
///   coordinator-owned dispatcher, for cases where `Global`'s shared-admission
///   fix alone does not reproduce exact request-to-thread assignment order.
///
/// For `workers>1` scheduled runs this selector changes execution behavior:
/// `Global`/`GlobalHop` build one per-cell [`crate::timing::GlobalSlotPool`] and
/// [`crate::timing::rate_gate::GlobalRateGate`] per phase that every worker
/// thread admits and paces against, so aggregate concurrency and request rate
/// match a single global limiter; `Sharded` retains the static `1/workers`
/// per-thread partition. `workers==1` and the single-thread coordinator path
/// have no cross-thread admission concern, so the mode is inert there.
///
/// The enum itself is defined in `crate::config::model::dispatch` (so the typed
/// config model and runtime share one serde-stable type) and re-exported here to
/// keep `crate::engine::protocol::DispatchMode` call sites unchanged.
pub use crate::config::model::DispatchMode;

/// Worker-assignment policy applied at the single [`DispatchMode::GlobalHop`]
/// pick site (`ThreadPerCoreExecutor::execute_command`) when `workers > 1`.
///
/// Defined in the leaf config model (`config::model::dispatch`, so the typed
/// config model and runtime share one serde-stable type) and re-exported here to
/// keep `crate::engine::protocol::HopRouting` call sites unchanged. See the
/// definition for the per-variant placement semantics.
pub use crate::config::model::HopRouting;

/// One plugins.yaml-shaped catalog entry.
#[derive(Debug, Serialize)]
pub struct CatalogEntry {
    /// Human-readable factory description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<&'static str>,
    /// Factory metadata consumed by Config preflight.
    #[serde(skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

/// Linked runner inventory emitted by `--capabilities`.
#[derive(Debug, Serialize)]
pub struct Catalog {
    /// Catalog document version.
    pub schema_version: &'static str,
    /// Endpoint dialect inventory.
    pub endpoint: BTreeMap<String, CatalogEntry>,
    /// Transport inventory.
    pub transport: BTreeMap<String, CatalogEntry>,
    /// Linked custom dataset loaders.
    pub custom_dataset_loader: BTreeMap<String, CatalogEntry>,
    /// Linked public dataset loaders.
    pub public_dataset_loader: BTreeMap<String, CatalogEntry>,
    /// Linked dataset samplers.
    pub dataset_sampler: BTreeMap<String, CatalogEntry>,
}

impl Catalog {
    /// Serialize the exact endpoint and transport catalog linked into this binary.
    pub fn from_registry(product_registry: &AIPerfRegistry) -> Self {
        let endpoint = product_registry
            .endpoints()
            .descriptors()
            .map(|descriptor| {
                (
                    descriptor.id.to_owned(),
                    CatalogEntry {
                        description: Some(descriptor.description),
                        metadata: serde_json::to_value(descriptor)
                            .expect("static endpoint descriptors are serializable"),
                    },
                )
            })
            .collect();
        let transport = product_registry
            .transport_descriptors()
            .into_iter()
            .map(|descriptor| {
                (
                    descriptor.id.to_owned(),
                    CatalogEntry {
                        description: Some(descriptor.description),
                        metadata: serde_json::json!({
                            "transport_type": descriptor.id,
                            "clock": descriptor.clock,
                            "features": descriptor.features,
                            "url_schemes": descriptor.url_schemes,
                        }),
                    },
                )
            })
            .collect();
        Self {
            schema_version: "1.0",
            endpoint,
            transport,
            custom_dataset_loader: BTreeMap::new(),
            public_dataset_loader: BTreeMap::new(),
            dataset_sampler: BTreeMap::new(),
        }
    }
}

pub use crate::engine::sidecar_input::{LiveStreamingSpec, MLflowStreamingSpec, OTelStreamingSpec};

/// Tokenizer source understood by the native dataset composer.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenizerSpec {
    /// Built-in encoding name, local tokenizer.json, or local model directory.
    #[serde(default = "default_tokenizer_name")]
    pub name: String,
    /// Count chat-shaped request bodies through the tokenizer's chat template.
    #[serde(default)]
    pub apply_chat_template: bool,
    /// Opt-in server-side tokenizer origin (e.g. `http://host:8000`).
    ///
    /// When set, tokenization is offloaded to the server's `/tokenize` and
    /// `/detokenize` endpoints instead of a local encoding, and [`name`] selects
    /// the model sent in tokenizer requests. Absent by default, which keeps the
    /// local built-in / Hugging Face path in force.
    ///
    /// [`name`]: Self::name
    #[serde(default)]
    pub server_url: Option<String>,
}

impl Default for TokenizerSpec {
    fn default() -> Self {
        Self {
            name: default_tokenizer_name(),
            apply_chat_template: false,
            server_url: None,
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
    /// Retain each Record-metric value in a bounded-memory t-digest sketch instead
    /// of the full value vector, trading exact percentiles for O(1) memory. Off by
    /// default; enabled by `AIPERF_METRICS_SKETCH` / `--sketch-metrics`.
    #[serde(default)]
    pub sketch: bool,
    /// Closed-loop steady-state windowing for concurrency-target runs. Disabled
    /// by default; enabled by `--steady-state`.
    #[serde(default)]
    pub steady_state: SteadyStateSpec,
}

/// Steady-state windowing settings lowered from Config v2.
#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SteadyStateSpec {
    /// Enables steady-state detection and summarization.
    #[serde(default)]
    pub enabled: bool,
    /// Occupancy fraction of the concurrency target; absent selects the native
    /// default (0.8).
    #[serde(default)]
    pub fraction: Option<f64>,
    /// Hybrid latency mode: latency/percentile metrics come from the whole
    /// profiling phase, only throughput comes from the steady window.
    #[serde(default)]
    pub hybrid_latency: bool,
}

/// Artifact paths relative to the exclusive run directory.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactSpec {
    /// Per-request metrics JSONL path, or absent when records are disabled.
    #[serde(default)]
    pub records_path: Option<PathBuf>,
    /// Wide per-request metrics Parquet sidecar path, or absent when the columnar
    /// export is disabled. Decodes on every build; a runner built without the
    /// `parquet` feature warns and skips it (the field stays present so the wire
    /// still decodes, as does `ParquetExportConfig`).
    #[serde(default)]
    pub records_parquet_path: Option<PathBuf>,
    /// Per-request metrics CSV sidecar path (`profile_export_records.csv`), or
    /// absent when the CSV export is disabled.
    #[serde(default)]
    pub records_csv_path: Option<PathBuf>,
    /// Raw request/response JSONL path, or absent when raw
    /// capture is disabled.
    #[serde(default)]
    pub raw_path: Option<PathBuf>,
    /// Aggregated profiling response text and selected metrics JSON path.
    #[serde(default)]
    pub outputs_path: Option<PathBuf>,
    /// Per-session formatted request payloads (`inputs.json`) path, or absent
    /// when the inputs export is disabled.
    #[serde(default)]
    pub inputs_path: Option<PathBuf>,
    /// Include transport timing details on JSONL records.
    #[serde(default)]
    pub trace: bool,
    /// Base path (relative to the run directory) for the `--dry-run` dataset
    /// analysis artifact family. When set, the graph path retains records and
    /// emits `dataset_analysis.{txt,json,csv,html}` beside this path. Absent when
    /// the analysis is not requested. Populated by the CLI dry-run gating.
    #[serde(default)]
    pub dataset_analysis_path: Option<PathBuf>,
    /// KV-cache block size (tokens) for the dry-run cache-reuse analysis. Absent →
    /// the analysis default (16). Ignored when `dataset_analysis_path` is absent.
    #[serde(default)]
    pub dataset_analysis_block_size: Option<u32>,
    /// Explicit realized-LRU cache capacity (blocks) added as a sweep point in the
    /// dry-run analysis. Ignored when `dataset_analysis_path` is absent.
    #[serde(default)]
    pub dataset_analysis_cache_blocks: Option<u64>,
    /// Request per-conversation breakdowns in the dry-run analysis. Ignored when
    /// `dataset_analysis_path` is absent.
    #[serde(default)]
    pub dataset_analysis_per_conversation: bool,
    /// Recorded-agent tool timing output path.
    #[serde(default)]
    pub graph_tool_time_path: Option<PathBuf>,
    /// Recorded-agent trace summary output path.
    #[serde(default)]
    pub graph_trace_summary_path: Option<PathBuf>,
    /// Recorded-agent normalized replay metrics JSON output path.
    #[serde(default)]
    pub graph_replay_metrics_path: Option<PathBuf>,
    /// Optional recorded-agent normalized replay metrics CSV output path.
    #[serde(default)]
    pub graph_replay_metrics_csv_path: Option<PathBuf>,
    /// Recorded-agent replay failure output path.
    #[serde(default)]
    pub graph_replay_failures_path: Option<PathBuf>,
    /// Recorded-agent replay provenance output path.
    #[serde(default)]
    pub graph_replay_provenance_path: Option<PathBuf>,
    /// Recorded-agent backend metadata output path.
    #[serde(default)]
    pub graph_replay_backend_metadata_path: Option<PathBuf>,
}

pub use crate::engine::sidecar_input::{
    GpuTelemetryMetricSpec, GpuTelemetrySourceSpec, GpuTelemetrySpec, GpuTelemetryUnitSpec,
    NetworkLatencyProbeSpec, NetworkLatencySpec, ServerMetricsFormatSpec, ServerMetricsSpec,
};

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

pub use crate::engine::dataset_input::*;

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
    /// AgentX agentic-replay: per-trajectory t\*-sampled warmup→profiling dispatch
    /// with byte-exact cache-bust. The mode owns its own timing (t\*, warmup leads,
    /// profiling offsets) and reuses the scheduled runtime for transport/metrics.
    AgenticReplay {
        /// Shared phase policy.
        #[serde(flatten)]
        common: PhaseCommonSpec,
        /// Trajectory-start window lower ratio (`--trajectory-start-min-ratio`).
        #[serde(default)]
        start_min_ratio: f64,
        /// Trajectory-start window upper ratio (`--trajectory-start-max-ratio`).
        #[serde(default = "one_value")]
        start_max_ratio: f64,
        /// Idle-gap cap in seconds for warmup-lead / leading-idle capping.
        #[serde(default)]
        idle_gap_cap_seconds: Option<f64>,
        /// Global system-idle cap in seconds; shifts pending replay work without rewriting trace timing.
        #[serde(default)]
        system_idle_gap_cap_seconds: Option<f64>,
        /// Anchor each phase-start burst at the earliest post-t\* request
        /// (`--burst-phase-starts`) instead of spreading by recorded offset.
        #[serde(default)]
        burst_phase_starts: bool,
    },
}

const fn true_value() -> bool {
    true
}

const fn one_value() -> f64 {
    1.0
}

/// Policy shared by every phase scheduling variant.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhaseCommonSpec {
    /// Stable workflow phase name.
    pub name: String,
    /// Semantic role (`warmup` or `profiling`); inferred for canonical names when omitted.
    #[serde(default)]
    pub kind: Option<PhaseRoleSpec>,
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
    /// Optional single-run adaptive load controller.
    #[serde(default)]
    pub adaptive_scale: Option<AdaptiveScaleSpec>,
    /// Optional agentic cache-warmup duration in seconds. Recorded-graph execution
    /// uses it as the cache-pressure window; absence selects the pair's default.
    #[serde(default)]
    pub agentic_cache_warmup_duration: Option<f64>,
    /// Agentic auto-warmup barrier grace in seconds (`--agentic-warmup-grace-period`).
    /// Applied only to the synthesized agentic warmup phase, not profiling grace.
    #[serde(default)]
    pub agentic_warmup_grace_period: Option<f64>,
    /// Abort profiling when `errors/total` exceeds this ratio after a grace floor
    /// of `max(concurrency, 10)` records (`--failed-request-threshold`).
    #[serde(default)]
    pub failed_request_threshold: Option<f64>,
    /// Piecewise-linear request-rate schedule (mutually exclusive with scalar `rate`).
    #[serde(default, alias = "rateSeries")]
    pub rate_series: Option<RateSeriesSpec>,
}

/// Semantic runtime role for one authored phase.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PhaseRoleSpec {
    /// Excluded from aggregate profiling results.
    Warmup,
    /// Contributes to benchmark results.
    Profiling,
}

/// Piecewise-linear request-rate schedule on the wire.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateSeriesSpec {
    /// Strictly increasing control points (≥2).
    pub points: Vec<RateSeriesPointSpec>,
}

/// One request-rate control point.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateSeriesPointSpec {
    #[serde(alias = "timeS")]
    pub time_s: f64,
    pub qps: f64,
}

impl PhaseCommonSpec {
    /// Resolve the semantic role, inferring canonical names when `kind` is omitted.
    pub fn semantic_role(&self) -> PhaseRoleSpec {
        if let Some(kind) = self.kind {
            return kind;
        }
        match self.name.as_str() {
            "warmup" => PhaseRoleSpec::Warmup,
            "profiling" => PhaseRoleSpec::Profiling,
            _ if self.exclude_from_results => PhaseRoleSpec::Warmup,
            _ => PhaseRoleSpec::Profiling,
        }
    }

    /// Whether this phase is excluded from aggregate profiling results.
    pub fn is_warmup(&self) -> bool {
        matches!(self.semantic_role(), PhaseRoleSpec::Warmup)
    }
}

/// Fully resolved adaptive-scale policy for one profiling phase.
///
/// The wire carries the effective maximum rather than asking the native runner
/// to rediscover an omitted-field default.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveScaleSpec {
    /// Controlled live load variable.
    pub control_variable: AdaptiveControlVariableSpec,
    /// Inclusive lower bound.
    pub minimum: f64,
    /// Inclusive upper bound after Config-v2 inference.
    pub maximum: f64,
    /// Tumbling assessment-window duration in seconds.
    pub assessment_period_seconds: f64,
    /// Required boundary hold duration in seconds.
    pub sustain_duration_seconds: f64,
    /// Minimum successful completions for a conclusive window.
    pub min_completed_requests: usize,
    /// Controller strategy; exactly one algorithm is currently supported.
    pub strategy_type: AdaptiveStrategyTypeSpec,
    /// Control increment policy.
    pub step_policy: AdaptiveStepPolicySpec,
    /// Minimum increment for SLA-margin scaling.
    pub base_step: usize,
    /// Largest SLA-margin multiplier.
    pub max_step_multiplier: usize,
    /// Current-value percentage for fixed-percent steps.
    pub step_percent: f64,
    /// Conjunctive SLA filters in authored order.
    pub sla_filters: Vec<AdaptiveSlaFilterSpec>,
}

/// Live control variable supported by the native actuator registry.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveControlVariableSpec {
    /// Session concurrency.
    Concurrency,
    /// Requests admitted but awaiting their first token.
    PrefillConcurrency,
    /// Mean issue rate.
    RequestRate,
    /// Active user-centric target.
    Users,
}

/// Supported adaptive controller strategy.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveStrategyTypeSpec {
    /// Monotone discover, boundary sustain, and one recovery.
    RampUntilFail,
}

/// Adaptive step-size policy.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdaptiveStepPolicySpec {
    /// Scale a base increment using the tightest normalized SLA margin.
    SlaMargin,
    /// Increment by a fixed percentage of the current control value.
    FixedPercentStep,
}

/// One adaptive SLA predicate.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveSlaFilterSpec {
    /// Supported metric tag or alias.
    pub metric_tag: String,
    /// Aggregate statistic.
    pub stat: String,
    /// Comparison operator.
    pub op: String,
    /// Finite threshold in the metric's public display unit.
    pub threshold: f64,
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
            | Self::FixedSchedule { common, .. }
            | Self::AgenticReplay { common, .. } => common,
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
            Self::FixedSchedule { .. } | Self::AgenticReplay { .. } => None,
        }
    }

    /// Request-rate arrival policy, absent for schedule-authored workloads.
    pub fn request_arrival(
        &self,
    ) -> Option<(crate::timing::ArrivalPattern, Option<f64>, Option<f64>)> {
        match self {
            Self::Concurrency { .. } => {
                Some((crate::timing::ArrivalPattern::ConcurrencyBurst, None, None))
            }
            Self::Poisson { rate, .. } => {
                Some((crate::timing::ArrivalPattern::Poisson, Some(*rate), None))
            }
            Self::Gamma {
                rate, smoothness, ..
            } => Some((
                crate::timing::ArrivalPattern::Gamma,
                Some(*rate),
                *smoothness,
            )),
            Self::Constant { rate, .. } => {
                Some((crate::timing::ArrivalPattern::Constant, Some(*rate), None))
            }
            Self::UserCentric { .. } | Self::FixedSchedule { .. } | Self::AgenticReplay { .. } => {
                None
            }
        }
    }

    /// Target authored rate for request-rate and user-centric workloads.
    pub fn rate(&self) -> Option<f64> {
        match self {
            Self::Poisson { rate, .. }
            | Self::Gamma { rate, .. }
            | Self::Constant { rate, .. }
            | Self::UserCentric { rate, .. } => Some(*rate),
            Self::Concurrency { .. } | Self::FixedSchedule { .. } | Self::AgenticReplay { .. } => {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_common_carries_agentic_cache_warmup_duration() {
        let phase: PhaseSpec = serde_json::from_str(
            r#"{
                "type": "concurrency",
                "name": "warmup",
                "exclude_from_results": true,
                "concurrency": 1,
                "agentic_cache_warmup_duration": 12.5
            }"#,
        )
        .unwrap();
        assert_eq!(phase.common().agentic_cache_warmup_duration, Some(12.5));
    }

    #[test]
    fn phase_common_cache_warmup_defaults_to_absent() {
        let phase: PhaseSpec = serde_json::from_str(
            r#"{
                "type": "concurrency",
                "name": "profiling",
                "exclude_from_results": false,
                "concurrency": 1
            }"#,
        )
        .unwrap();
        assert_eq!(phase.common().agentic_cache_warmup_duration, None);
    }

    #[test]
    fn phase_common_rejects_unknown_fields() {
        let result = serde_json::from_str::<PhaseSpec>(
            r#"{
                "type": "concurrency",
                "name": "warmup",
                "exclude_from_results": true,
                "concurrency": 1,
                "unknown_phase_field": true
            }"#,
        );
        let Err(error) = result else {
            panic!("phase accepted an unknown field")
        };
        assert!(error.to_string().contains("unknown field"));
    }
}
