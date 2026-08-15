// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Typed runner-consumed benchmark configuration.
//!
//! Unknown sections are ignored during deserialization so partial
//! configurations can be filtered through this type.

use serde::{Deserialize, Serialize};

use super::artifacts::Artifacts;
use super::dataset::Dataset;
use super::endpoint::Endpoint;
use super::export::Export;
use super::metrics::Metrics;
use super::models::Models;
use super::phase::Phase;
use super::runtime::Runtime;
use super::telemetry::{GpuTelemetryConfig, NetworkLatencyConfig, ServerMetricsConfig, Sidecars};
use super::tokenizer::Tokenizer;
use super::transport::Transport;

/// Accuracy policy present only when `--accuracy-benchmark` is set.
///
/// Optional fields serialize as explicit nulls.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Accuracy {
    /// Benchmark id (`--accuracy-benchmark`, e.g. `mmlu`).
    pub benchmark: String,
    /// Chain-of-thought toggle (`--accuracy-enable-cot`/`--accuracy-no-enable-cot`; null default).
    pub enable_cot: Option<bool>,
    /// Grader id (`--accuracy-grader`; null default).
    pub grader: Option<String>,
    /// Few-shot example count (`--accuracy-n-shots`; null default).
    pub n_shots: Option<i64>,
    /// System prompt override (`--accuracy-system-prompt`; null default).
    pub system_prompt: Option<String>,
    /// Selected task ids (`--accuracy-tasks`; null default).
    pub tasks: Option<Vec<String>>,
    /// Verbose grader output (`--accuracy-verbose`; false default).
    pub verbose: bool,
}

/// User-authored environment provenance for a benchmark run.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Metadata {
    /// Free-form endpoint hardware description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hardware: Option<String>,
    /// Endpoint placement relative to recorded tool execution.
    #[serde(default = "default_endpoint_placement")]
    pub endpoint_placement: String,
}

fn default_endpoint_placement() -> String {
    "unknown".to_string()
}

/// The runner-consumed benchmark configuration.
///
/// Unset sections are omitted from serialized requests.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Model-selection policy (`cfg.models`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Models>,
    /// Default endpoint profile (`cfg.endpoint`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<Endpoint>,
    /// Tokenizer acquisition policy (`cfg.tokenizer`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<Tokenizer>,
    /// Inline transport selection (`cfg.transport`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transport: Option<Transport>,
    /// Worker/cell runtime policy (`cfg.runtime`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Runtime>,
    /// Native metrics policy (`cfg.metrics`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metrics: Option<Metrics>,
    /// Goodput SLO thresholds (`cfg.slos`, metric→threshold; open bag).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slos: Option<serde_json::Map<String, serde_json::Value>>,
    /// Native output policy (`cfg.artifacts`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifacts: Option<Artifacts>,
    /// User-authored environment provenance (`cfg.metadata`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Metadata>,
    /// Canonical single-dataset list (`cfg.datasets`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub datasets: Option<Vec<Dataset>>,
    /// Ordered phase policy (`cfg.phases`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phases: Option<Vec<Phase>>,
    /// Post-report export policy (`cfg.export`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub export: Option<Export>,
    /// Raw GPU-telemetry policy (`cfg.gpu_telemetry`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_telemetry: Option<GpuTelemetryConfig>,
    /// Raw server-metrics policy (`cfg.server_metrics`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_metrics: Option<ServerMetricsConfig>,
    /// Raw network-latency policy (`cfg.network_latency`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub network_latency: Option<NetworkLatencyConfig>,
    /// Lowered side-channel sidecars (`cfg.sidecars`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sidecars: Option<Sidecars>,
    // These fields are always emitted, including their defaults.
    /// Accuracy-benchmark policy (`cfg.accuracy`; null unless an accuracy run).
    #[serde(default)]
    pub accuracy: Option<Accuracy>,
    /// Per-endpoint override profiles (`cfg.endpoint_profiles`; empty by default).
    #[serde(default)]
    pub endpoint_profiles: serde_json::Map<String, serde_json::Value>,
    /// Sweep/cluster failure policy (`cfg.failure_policy`; null on the CLI path).
    #[serde(default)]
    pub failure_policy: Option<serde_json::Value>,
    /// Named submission scenario (`cfg.scenario`; `--scenario`).
    #[serde(default)]
    pub scenario: Option<String>,
    /// Resolved WEKA reconstruction semantics (`legacy`|`graph-ir`); authored into
    /// the graph workload config so the engine selects the legacy agentic path or
    /// the graph-ir path. Unset defers to the graph-ir default.
    #[serde(default)]
    pub weka_semantics: Option<String>,
    /// Legacy Weka global idle cap (`--system-idle-gap-cap-seconds`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_idle_gap_cap_seconds: Option<f64>,
    /// Ignore recorded trace inter-message/inter-request delays for graph-ir runs
    /// (`--ignore-trace-delays`). Attached only to the graph workload DTO; the
    /// engine sets `ExecutorFlags::ignore_edge_delays` so nodes fire as soon as
    /// their inputs are ready.
    #[serde(default)]
    pub ignore_trace_delays: bool,
    /// Recorded-graph trajectory-start window upper ratio (`--trajectory-start-max-ratio`).
    #[serde(default)]
    pub trajectory_start_max_ratio: f64,
    /// Recorded-graph trajectory-start window lower ratio (`--trajectory-start-min-ratio`).
    #[serde(default)]
    pub trajectory_start_min_ratio: f64,
    /// Escape hatch that relaxes cross-field validation (`--unsafe-override`).
    #[serde(default)]
    pub unsafe_override: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_config_serde_round_trips() {
        // A minimal, mostly-default config exercises both the omitted-when-None
        // sections and the always-emitted defaulted fields through a full
        // serialize → value → deserialize → serialize cycle.
        let cfg = BenchmarkConfig {
            scenario: Some("smoke".to_string()),
            trajectory_start_min_ratio: 0.25,
            trajectory_start_max_ratio: 0.75,
            unsafe_override: true,
            ..BenchmarkConfig::default()
        };
        let value = serde_json::to_value(&cfg).expect("serialize BenchmarkConfig");
        let back: BenchmarkConfig =
            serde_json::from_value(value.clone()).expect("deserialize BenchmarkConfig");
        let value_again = serde_json::to_value(&back).expect("re-serialize BenchmarkConfig");
        assert_eq!(value, value_again);
        assert_eq!(back.scenario.as_deref(), Some("smoke"));
        assert!(back.unsafe_override);
        assert_eq!(back.trajectory_start_min_ratio, 0.25);
        assert_eq!(back.trajectory_start_max_ratio, 0.75);
    }
}
