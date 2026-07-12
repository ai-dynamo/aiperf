// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict JSON request/result contract for one native benchmark run.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::protocol_v2::RUNNER_PROTOCOL_V2;
use crate::registry::{
    BuiltinRunnerRegistryFactory, RunnerBackendDescriptor, RunnerRegistry, RunnerRegistryFactory,
    RunnerWorkloadDescriptor,
};

/// Current Python-orchestrator/Rust-runner protocol version.
pub const RUNNER_PROTOCOL_VERSION: u32 = 1;

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
    /// Describe the exact process contract implemented by this binary.
    pub fn current() -> anyhow::Result<Self> {
        let runner_registry = BuiltinRunnerRegistryFactory.build()?;
        let product_registry = aiperf_extensions::AiperfRegistryFactory::build(
            &aiperf_extensions::BuiltinAiperfRegistryFactory,
        )?;
        Ok(Self::from_registries(
            crate::distribution_identity::current_distribution_id()?,
            &runner_registry,
            &product_registry,
        ))
    }

    /// Build a deterministic capability document from already frozen registries.
    pub fn from_registries(
        distribution_id: String,
        runner_registry: &RunnerRegistry,
        product_registry: &aiperf_extensions::AiperfRegistry,
    ) -> Self {
        let endpoints = product_registry
            .endpoints()
            .descriptors()
            .collect::<Vec<_>>();
        let endpoint_types = endpoints.iter().map(|descriptor| descriptor.id).collect();
        Self {
            event: "runner_capabilities",
            capabilities_schema_version: 2,
            protocol_versions: &[RUNNER_PROTOCOL_VERSION, RUNNER_PROTOCOL_V2],
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
    /// Number of Rust HTTP execution workers behind the single dispatcher.
    #[serde(default = "default_worker_count")]
    pub workers: usize,
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
    /// Optional canonical Python-evaluated accuracy run.
    #[serde(default)]
    pub accuracy: Option<AccuracySpec>,
    /// Optional phase-bounded GPU telemetry collection.
    #[serde(default)]
    pub gpu_telemetry: Option<GpuTelemetrySpec>,
    /// Optional fixed or actively measured network RTT calibration.
    #[serde(default)]
    pub network_latency: Option<NetworkLatencySpec>,
    /// Optional phase-bounded inference-server Prometheus collection.
    #[serde(default)]
    pub server_metrics: Option<ServerMetricsSpec>,
    /// Optional live OTel/MLflow results extension supervised by Rust.
    #[serde(default)]
    pub live_streaming: Option<LiveStreamingSpec>,
}

fn default_worker_count() -> usize {
    1
}

/// Canonical Python live-results extension configuration.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LiveStreamingSpec {
    /// Absolute interpreter selected by the Python Config-v2 parent.
    pub python_executable: PathBuf,
    /// Importable strict-stdio worker module.
    #[serde(default = "default_live_streaming_worker_module")]
    pub worker_module: String,
    /// Bounded Rust-to-Python queue capacity with drop-oldest overflow.
    pub buffer_capacity: usize,
    /// Canonical OpenTelemetry streaming settings.
    pub otel: OTelStreamingSpec,
    /// Canonical live-MLflow settings.
    pub mlflow: MLflowStreamingSpec,
}

fn default_live_streaming_worker_module() -> String {
    "aiperf.post_processors.native_streaming_worker".to_string()
}

/// OpenTelemetry settings forwarded to the canonical Python processor.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OTelStreamingSpec {
    /// OTLP/HTTP metrics endpoint.
    #[serde(default)]
    pub metrics_url: Option<String>,
    /// Emit terminal request metric records.
    pub stream_metrics_enabled: bool,
    /// Emit phase lifecycle and progress records.
    pub stream_timing_enabled: bool,
    /// User-authored OTel resource attributes.
    #[serde(default)]
    pub custom_resource_attributes: BTreeMap<String, String>,
    /// Optional GenAI semantic-convention provider override.
    #[serde(default)]
    pub gen_ai_provider: Option<String>,
}

/// Live MLflow settings forwarded to the canonical Python fanout.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MLflowStreamingSpec {
    /// MLflow tracking server URI.
    #[serde(default)]
    pub tracking_uri: Option<String>,
    /// Experiment name.
    pub experiment: String,
    /// Optional run name.
    #[serde(default)]
    pub run_name: Option<String>,
    /// Optional run tags.
    #[serde(default)]
    pub tags: Option<BTreeMap<String, String>>,
    /// Optional parent run identity.
    #[serde(default)]
    pub parent_run_id: Option<String>,
    /// Optional post-run artifact selection retained in fanout metadata.
    #[serde(default)]
    pub artifact_globs: Option<Vec<String>>,
}

/// Canonical evaluator configuration for an accuracy-enabled native run.
///
/// Python Config v2 selects the benchmark and the exact Python interpreter;
/// Rust retains ownership of every inference request and sends only completed
/// response text back to this supervised worker for grading.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccuracySpec {
    /// Config-v2 benchmark name or stable alias.
    pub benchmark: String,
    /// Optional task/category subset.
    #[serde(default)]
    pub tasks: Option<Vec<String>>,
    /// Optional few-shot count; absent selects the benchmark default.
    #[serde(default)]
    pub n_shots: Option<usize>,
    /// Optional chain-of-thought selection; absent selects the benchmark default.
    #[serde(default)]
    pub enable_cot: Option<bool>,
    /// Optional legacy grader override. Canonical workers may reject overrides.
    #[serde(default)]
    pub grader: Option<String>,
    /// Optional benchmark system prompt override.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Absolute Python interpreter selected by the parent orchestrator.
    pub python_executable: PathBuf,
    /// Importable stdio worker module; defaults to the canonical AIPerf worker.
    #[serde(default = "default_accuracy_worker_module")]
    pub worker_module: String,
}

fn default_accuracy_worker_module() -> String {
    "aiperf.accuracy.worker".to_string()
}

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

/// Low-rate GPU telemetry synchronized to the profiling phase.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuTelemetrySpec {
    /// Clock cadence between continuous scrapes.
    pub collection_interval_ns: i64,
    /// Clock deadline applied independently to each telemetry HTTP request.
    pub request_timeout_ns: i64,
    /// Legacy-compatible per-GPU JSONL path relative to the run directory.
    pub records_path: PathBuf,
    /// Ordered source list after Config-v2 default expansion and deduplication.
    pub sources: Vec<GpuTelemetrySourceSpec>,
    /// Config-v2 custom DCGM fields registered for native sidecar reporting.
    #[serde(default)]
    pub custom_metrics: Vec<GpuTelemetryMetricSpec>,
}

/// Run-level network RTT calibration lowered from Config v2.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkLatencySpec {
    /// Fixed mean RTT in nanoseconds; mutually exclusive with active probing.
    #[serde(default)]
    pub mean_rtt_ns: Option<f64>,
    /// Active fresh-TCP probe policy; mutually exclusive with a fixed mean.
    #[serde(default)]
    pub probe: Option<NetworkLatencyProbeSpec>,
}

/// Profiling-bounded fresh-TCP probe policy.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetworkLatencyProbeSpec {
    /// Clock cadence between probe issuances.
    pub ping_interval_ns: i64,
    /// Per-connect Clock deadline.
    pub connect_timeout_ns: i64,
    /// Global Clock budget for phase-end sample top-up.
    pub complete_topup_timeout_ns: i64,
    /// Successful-sample floor applied independently to every unique target.
    pub min_successful_samples: usize,
    /// Legacy-compatible per-sample JSONL path relative to the run directory.
    pub records_path: PathBuf,
}

/// Inference-server Prometheus collection and artifact policy.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerMetricsSpec {
    /// Clock cadence between sequential continuous scrapes.
    pub collection_interval_ns: i64,
    /// Clock deadline used for source reachability/connect attempts.
    pub reachability_timeout_ns: i64,
    /// Ordered normalized endpoints after inference and explicit URL expansion.
    pub urls: Vec<String>,
    /// Canonical compatibility artifacts requested by Config v2.
    pub formats: Vec<ServerMetricsFormatSpec>,
    /// Slim JSONL output relative to the run directory when requested.
    #[serde(default)]
    pub jsonl_path: Option<PathBuf>,
    /// Full-record handoff relative to the run directory for Python Parquet rendering.
    #[serde(default)]
    pub parquet_wire_path: Option<PathBuf>,
}

/// Config-v2 server-metrics export formats.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ServerMetricsFormatSpec {
    /// Aggregate JSON rendered by Python's canonical exporter.
    Json,
    /// Aggregate CSV rendered by Python's canonical exporter.
    Csv,
    /// Slim per-scrape JSONL written by Rust.
    Jsonl,
    /// Raw time-series Parquet rendered by Python's canonical exporter.
    Parquet,
}

/// One injected GPU telemetry source.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum GpuTelemetrySourceSpec {
    /// NVIDIA DCGM Prometheus endpoint collected by Rust HTTP.
    Dcgm {
        /// Metrics endpoint; `/metrics` is appended when absent.
        url: String,
    },
    /// Canonical Python collector or user extension supervised by Rust.
    Python {
        /// Registered Config-v2 collector name.
        collector: String,
        /// Optional remote endpoint used by the DCGM collector.
        #[serde(default)]
        url: Option<String>,
        /// Optional custom DCGM metrics definition.
        #[serde(default)]
        metrics_file: Option<PathBuf>,
        /// Absolute interpreter selected by the Python orchestrator.
        python_executable: PathBuf,
        /// Importable strict-stdio worker module.
        worker_module: String,
    },
}

/// One Config-v2 custom GPU signal exposed in native-v2 output.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuTelemetryMetricSpec {
    /// Stable normalized field name emitted by the Python collector.
    pub name: String,
    /// Human-readable metric label.
    pub header: String,
    /// Native report unit.
    pub unit: GpuTelemetryUnitSpec,
}

/// Config-v2 GPU unit vocabulary accepted by the native report engine.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuTelemetryUnitSpec {
    /// Unitless count.
    Count,
    /// Kibibytes.
    Kilobyte,
    /// Mebibytes.
    Megabyte,
    /// Gibibytes.
    Gigabyte,
    /// Microseconds.
    Microsecond,
    /// Milliseconds.
    Millisecond,
    /// Seconds.
    Second,
    /// Percentage.
    Percent,
    /// Watts.
    Watt,
    /// Joules.
    Joule,
    /// Megajoules.
    Megajoule,
    /// Megahertz.
    Megahertz,
    /// Gigahertz.
    Gigahertz,
    /// Celsius.
    Celsius,
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
    /// HTTP connection reuse/lease strategy.
    #[serde(default)]
    pub connection_reuse: aiperf_transport_http::models::ConnectionReuseStrategy,
    /// Optional request-body content type after Config-v2 normalization.
    #[serde(default)]
    pub request_content_type: Option<aiperf_endpoints::RequestContentType>,
    /// Download completed video bytes after the polling lifecycle.
    #[serde(default)]
    pub download_video_content: bool,
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
    /// Optional endpoint API key, carried only over the stdin pipe. The
    /// selected dialect chooses bearer versus vendor-specific authentication.
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
    Synthetic(Box<SyntheticDatasetSpec>),
    /// Local path or inline records parsed by the native loader registry.
    File(Box<FileDatasetSpec>),
    /// Resolved built-in public dataset source and native loader selection.
    Public(Box<PublicDatasetSpec>),
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
