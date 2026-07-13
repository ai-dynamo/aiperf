// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict JSON request/result contract for one native benchmark run.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::protocol_v2::RUNNER_PROTOCOL_V2;
use crate::registry::{RunnerBackendDescriptor, RunnerRegistry, RunnerWorkloadDescriptor};

/// Machine-readable runner capabilities returned by `--capabilities`.
#[derive(Debug, Serialize)]
pub struct RunnerCapabilities {
    /// Stable response discriminator.
    pub event: &'static str,
    /// Capability-document schema independent of stdin protocol versions.
    pub capabilities_schema_version: u32,
    /// Protocol versions accepted on stdin.
    pub protocol_versions: &'static [u32],
    /// Native report schema written after a successful run.
    pub report_schema_version: &'static str,
    /// BLAKE3 identity of the complete executable image serving this response.
    pub distribution_id: String,
    /// Endpoint dialects accepted by the native formatter/parser registry.
    pub endpoint_types: Vec<&'static str>,
    /// Canonical endpoint descriptors from the frozen endpoint registry.
    pub endpoints: Vec<&'static aiperf_endpoints::EndpointDescriptor>,
    /// Statically linked extension package names in deterministic order.
    pub extensions: Vec<String>,
    /// Backend descriptors recognized by protocol-v2 validation.
    pub backends: Vec<&'static RunnerBackendDescriptor>,
    /// Workload descriptors recognized by protocol-v2 validation.
    pub workloads: Vec<&'static RunnerWorkloadDescriptor>,
    /// Descriptor-compatible pairs, including pairs without an executable v2 adapter.
    pub statically_compatible_pairs: Vec<[String; 2]>,
    /// Pairs with a registered executable protocol-v2 adapter.
    pub supported_pairs: Vec<[String; 2]>,
    /// Evaluator providers whose registered launch distributions passed
    /// factory attestation and mandatory isolation availability checks.
    pub evaluation_providers: Vec<EvaluationProviderCapability>,
    /// Host operations with linked executable Rust adapters.
    pub evaluation_host_operations: Vec<EvaluationHostOperationCapability>,
    /// Fully executable backend/workload/provider/distribution combinations.
    pub supported_evaluation_combinations: Vec<SupportedEvaluationCombination>,
    /// Known stock provider/distribution selections that this exact image
    /// cannot execute, expressed only with closed path-free reason codes.
    pub evaluation_unavailable: Vec<EvaluationUnavailableCapability>,
    /// Dataset variants accepted by the current protocol.
    pub dataset_types: &'static [&'static str],
    /// Phase variants accepted by the current protocol.
    pub phase_types: &'static [&'static str],
    /// Optional policies accepted inside a phase.
    pub phase_features: &'static [&'static str],
    /// Optional single-run subsystems accepted by the runner.
    pub run_features: &'static [&'static str],
    /// GPU telemetry source implementations accepted by the runner.
    pub telemetry_source_types: &'static [&'static str],
    /// Server-metrics artifact formats accepted by the runner.
    pub server_metrics_formats: &'static [&'static str],
    /// Rust runner package version.
    pub runner_version: &'static str,
}

impl RunnerCapabilities {
    /// Build a deterministic capability document from already frozen registries.
    pub fn from_registries(
        distribution_id: String,
        runner_registry: &RunnerRegistry,
        product_registry: &aiperf_extensions::AiperfRegistry,
    ) -> Self {
        Self::from_registries_with_evaluation(
            distribution_id,
            runner_registry,
            product_registry,
            runner_registry.evaluation_capabilities().clone(),
        )
    }

    /// Build capabilities with an executable evaluator/provider inventory.
    pub fn from_registries_with_evaluation(
        distribution_id: String,
        runner_registry: &RunnerRegistry,
        product_registry: &aiperf_extensions::AiperfRegistry,
        evaluation: EvaluationCapabilityInventory,
    ) -> Self {
        let endpoints = product_registry
            .endpoints()
            .descriptors()
            .collect::<Vec<_>>();
        let endpoint_types = endpoints.iter().map(|descriptor| descriptor.id).collect();
        Self {
            event: "runner_capabilities",
            capabilities_schema_version: 2,
            protocol_versions: &[RUNNER_PROTOCOL_V2],
            report_schema_version: aiperf_metrics::NATIVE_REPORT_SCHEMA_VERSION,
            distribution_id,
            endpoint_types,
            endpoints,
            extensions: product_registry
                .extension_names()
                .map(str::to_owned)
                .collect(),
            backends: runner_registry.backend_descriptors(),
            workloads: runner_registry.workload_descriptors(),
            statically_compatible_pairs: runner_registry
                .statically_compatible_pairs()
                .into_iter()
                .map(|(backend, workload)| [backend.to_owned(), workload.to_owned()])
                .collect(),
            supported_pairs: runner_registry
                .supported_pairs()
                .into_iter()
                .map(|(backend, workload)| [backend.to_owned(), workload.to_owned()])
                .collect(),
            evaluation_providers: evaluation.providers,
            evaluation_host_operations: evaluation.host_operations,
            supported_evaluation_combinations: evaluation.supported_combinations,
            evaluation_unavailable: evaluation.unavailable,
            dataset_types: &["synthetic", "file", "public"],
            phase_types: &[
                "concurrency",
                "poisson",
                "gamma",
                "constant",
                "user_centric",
                "fixed_schedule",
            ],
            phase_features: &["adaptive_scale", "ramps", "request_cancellation"],
            run_features: &[
                "gpu_telemetry",
                "python_live_streaming",
                "outputs_json",
                "python_accuracy_evaluator",
                "raw_records",
                "http_transport_policy",
                "thread_per_core_execution",
                "network_latency",
                "server_metrics",
            ],
            telemetry_source_types: &["dcgm", "python"],
            server_metrics_formats: &["json", "csv", "jsonl", "parquet"],
            runner_version: env!("CARGO_PKG_VERSION"),
        }
    }
}

/// Safe exact identity for one factory-attested evaluator distribution.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct EvaluationDistributionCapability {
    /// Immutable selectable deployment ID.
    pub id: String,
    /// Exact provider package name.
    pub package: String,
    /// Exact package version.
    pub package_version: String,
    /// Provider source/commit SHA-256.
    pub provider_source_sha256: String,
    /// Worker bootstrap source SHA-256.
    pub worker_source_sha256: String,
    /// Complete dependency-lock SHA-256.
    pub dependency_lock_sha256: String,
    /// Factory-attested executable closure SHA-256.
    pub launch_closure_sha256: String,
    /// Optional immutable OCI identity.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub oci_digest: Option<String>,
}

/// Safe capability projection for one executable evaluator-provider factory.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct EvaluationProviderCapability {
    /// Open provider factory ID.
    pub id: String,
    /// Safe human-facing label.
    pub display_name: String,
    /// Supported evaluator-worker protocol versions.
    pub worker_protocol_versions: Vec<u32>,
    /// Supported unit granularities.
    pub execution_granularities: Vec<String>,
    /// Supported occurrence scheduling modes.
    pub scheduling_modes: Vec<String>,
    /// Factory-owned authored-schema version.
    pub config_schema_version: u32,
    /// Factory-owned authored-schema SHA-256.
    pub config_schema_sha256: String,
    /// Enforceable runner isolation implementation identity.
    pub isolation_profile_id: String,
    /// Provider-declared semantic operations; executable combinations publish
    /// only their intersection with linked host adapters.
    pub declared_operations: Vec<String>,
    /// Factory-attested immutable launch distributions.
    pub distributions: Vec<EvaluationDistributionCapability>,
}

/// One linked Rust host-operation adapter in the executing image.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct EvaluationHostOperationCapability {
    /// Open semantic operation ID.
    pub id: String,
    /// Executor family ID.
    pub family: String,
    /// Request-schema SHA-256.
    pub request_schema_sha256: String,
    /// Terminal-response schema SHA-256.
    pub response_schema_sha256: String,
    /// Incremental-event schema SHA-256 when true streaming is executable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_schema_sha256: Option<String>,
    /// Whether the adapter emits real incremental typed events.
    pub true_streaming: bool,
    /// Endpoint capabilities required for route compatibility.
    pub endpoint_capabilities: Vec<String>,
}

/// One provider selection that the exact image can execute end to end.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct SupportedEvaluationCombination {
    /// Registered backend pair half.
    pub backend: String,
    /// Registered workload pair half; currently `evaluation`.
    pub workload: String,
    /// Exact provider factory ID.
    pub provider: String,
    /// Exact immutable launch distribution ID.
    pub distribution: String,
    /// Linked operations executable for this combination.
    pub operations: Vec<String>,
    /// Linked resource adapter IDs available to authored bindings.
    pub resources: Vec<String>,
    /// Enforceable process-tree isolation implementation.
    pub isolation_profile_id: String,
}

/// Closed, secret-free reason why one known evaluator distribution is absent
/// from the executable capability combinations.
#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationUnavailableReasonCode {
    /// The deployment supplied no complete exact-content source-root closure.
    ProviderRootsUnavailable,
    /// The compiled host OS/architecture cannot execute the stock closure.
    UnsupportedPlatform,
    /// The mandatory process-tree isolation implementation did not attest.
    IsolationUnavailable,
}

/// One known provider/distribution selection unavailable in this exact image.
///
/// Deliberately no detail string exists: capabilities must never publish host
/// paths, environment contents, package errors, or other deployment secrets.
#[derive(Clone, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvaluationUnavailableCapability {
    /// Exact provider factory ID.
    pub provider: String,
    /// Exact immutable launch distribution ID.
    pub distribution: String,
    /// Closed machine-readable unavailability reason.
    pub reason_code: EvaluationUnavailableReasonCode,
}

/// Evaluator capability inputs composed from the same frozen registries used
/// for strict validation and execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvaluationCapabilityInventory {
    /// Executable provider descriptors.
    pub providers: Vec<EvaluationProviderCapability>,
    /// Linked host-operation adapters.
    pub host_operations: Vec<EvaluationHostOperationCapability>,
    /// Exact executable combinations.
    pub supported_combinations: Vec<SupportedEvaluationCombination>,
    /// Known selections omitted from executable combinations with stable,
    /// path-free reason codes.
    pub unavailable: Vec<EvaluationUnavailableCapability>,
}

pub use crate::sidecar_input::{LiveStreamingSpec, MLflowStreamingSpec, OTelStreamingSpec};

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
}

impl Default for TokenizerSpec {
    fn default() -> Self {
        Self {
            name: default_tokenizer_name(),
            apply_chat_template: false,
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
    /// Python-compatible raw request/response JSONL path, or absent when raw
    /// capture is disabled.
    #[serde(default)]
    pub raw_path: Option<PathBuf>,
    /// Aggregated profiling response text and selected metrics JSON path.
    #[serde(default)]
    pub outputs_path: Option<PathBuf>,
    /// Include transport timing details on JSONL records.
    #[serde(default)]
    pub trace: bool,
}

pub use crate::sidecar_input::{
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

pub use crate::dataset_input::*;

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
    /// Optional single-run adaptive load controller.
    #[serde(default)]
    pub adaptive_scale: Option<AdaptiveScaleSpec>,
}

/// Fully resolved adaptive-scale policy for one profiling phase.
///
/// Config v2 validation and defaulting are grounded in
/// `src/aiperf/config/adaptive_scale_phase.py:140-383`; the wire carries the
/// effective maximum rather than asking the native runner to rediscover an
/// omitted-field default.
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
    /// Controller strategy; protocol v1 intentionally has one exact algorithm.
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

/// Adaptive controller strategy accepted by protocol v1.
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

#[cfg(test)]
mod tests {
    use super::{EvaluationUnavailableCapability, EvaluationUnavailableReasonCode};

    #[test]
    fn evaluation_unavailability_serializes_only_closed_reason_codes() {
        for (reason_code, expected) in [
            (
                EvaluationUnavailableReasonCode::ProviderRootsUnavailable,
                "provider_roots_unavailable",
            ),
            (
                EvaluationUnavailableReasonCode::UnsupportedPlatform,
                "unsupported_platform",
            ),
            (
                EvaluationUnavailableReasonCode::IsolationUnavailable,
                "isolation_unavailable",
            ),
        ] {
            let value = serde_json::to_value(EvaluationUnavailableCapability {
                provider: "provider".to_owned(),
                distribution: "distribution".to_owned(),
                reason_code,
            })
            .unwrap();
            assert_eq!(
                value,
                serde_json::json!({
                    "provider": "provider",
                    "distribution": "distribution",
                    "reason_code": expected,
                })
            );
        }
    }
}
