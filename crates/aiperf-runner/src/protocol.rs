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
    /// Endpoint dialects accepted by the native formatter/parser registry.
    pub endpoint_types: &'static [&'static str],
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
    pub const fn current() -> Self {
        Self {
            event: "runner_capabilities",
            protocol_versions: &[RUNNER_PROTOCOL_VERSION],
            report_schema_version: aiperf_metrics::NATIVE_REPORT_SCHEMA_VERSION,
            endpoint_types: &[
                "chat",
                "completions",
                "responses",
                "messages",
                "embeddings",
                "chat_embeddings",
                "nim_embeddings",
                "cohere_rankings",
                "hf_tei_rankings",
                "nim_rankings",
                "huggingface_generate",
                "image_generation",
                "image_edit",
                "video_generation",
                "image_retrieval",
                "solido_rag",
                "raw",
                "template",
            ],
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
    pub connection_reuse: aiperf_transport::models::ConnectionReuseStrategy,
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

/// Public dataset configuration resolved from the Python plugin registry.
///
/// Python keeps ownership of the named plugin catalog in
/// `src/aiperf/plugin/plugins.yaml:1733-1957`; Rust receives only the explicit
/// source coordinates and loader options needed for one run.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PublicDatasetSpec {
    /// Config-v2 public dataset name, retained for diagnostics.
    pub name: String,
    /// Native loader registration name.
    pub format: String,
    /// Fully resolved remote source.
    pub source: PublicDatasetSourceSpec,
    /// Conversation sampling strategy.
    #[serde(default = "default_sampling_strategy")]
    pub sampling: String,
    /// Optional row/conversation cap.
    #[serde(default)]
    pub entries: Option<usize>,
    /// Dataset-local seed overriding the run seed.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Validated loader/composer options from plugin metadata and Config v2.
    #[serde(default)]
    pub options: Map<String, Value>,
}

/// Network source for a resolved public dataset.
#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PublicDatasetSourceSpec {
    /// Generic pinned or authored URL.
    Url {
        /// JSON/JSONL/CSV/Parquet URL.
        url: String,
    },
    /// Hugging Face Dataset Viewer or revision-pinned repository source.
    HuggingFace {
        /// Namespace/repository identifier.
        dataset: String,
        /// Dataset configuration/subset.
        subset: String,
        /// Dataset split.
        split: String,
        /// Optional immutable or symbolic revision.
        #[serde(default)]
        revision: Option<String>,
    },
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
    /// Optional native trace transformation and caps.
    #[serde(default)]
    pub synthesis: Option<TraceSynthesisSpec>,
    /// Loader/composer-specific options after Config-v2 validation.
    #[serde(default)]
    pub options: Map<String, Value>,
}

/// Trace synthesis configuration from
/// `src/aiperf/config/dataset/trace.py:20-117`.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraceSynthesisSpec {
    /// Timestamp divisor.
    pub speedup_ratio: f64,
    /// Shared-prefix length multiplier.
    pub prefix_len_multiplier: f64,
    /// Independent prefix-root count.
    pub prefix_root_multiplier: u64,
    /// Unique-prompt length multiplier.
    pub prompt_len_multiplier: f64,
    /// Output-length multiplier.
    pub output_len_multiplier: f64,
    /// Original-row filter and transformed-length cap.
    #[serde(default)]
    pub max_isl: Option<u64>,
    /// Final output-length cap.
    #[serde(default)]
    pub max_osl: Option<u32>,
}

fn default_sampling_strategy() -> String {
    "sequential".into()
}

/// Native synthetic dataset configuration.
///
/// This is the process-boundary projection of
/// `src/aiperf/config/dataset/config.py:62-245`; content sub-shapes follow
/// `src/aiperf/config/dataset/content.py:50-459` and
/// `src/aiperf/config/dataset/video.py:41-205`.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticDatasetSpec {
    /// Number of reusable conversations.
    pub entries: usize,
    /// Dataset-local seed overriding the run seed for generation and sampling.
    #[serde(default)]
    pub random_seed: Option<u64>,
    /// Conversation sampling policy.
    #[serde(default = "default_sampling_strategy")]
    pub sampling: String,
    /// Optional text generation configuration.
    #[serde(default)]
    pub prompts: Option<SyntheticPromptsSpec>,
    /// Optional shared-prefix or per-session context configuration.
    #[serde(default)]
    pub prefix_prompts: Option<SyntheticPrefixPromptsSpec>,
    /// Turns per conversation.
    #[serde(default = "one_distribution")]
    pub turns: DistributionSpec,
    /// Inter-turn delay in milliseconds.
    #[serde(default = "zero_distribution")]
    pub turn_delay_ms: DistributionSpec,
    /// Multiplicative delay scale.
    #[serde(default = "one_f64")]
    pub turn_delay_ratio: f64,
    /// Optional synthetic images.
    #[serde(default)]
    pub images: Option<SyntheticImageSpec>,
    /// Optional synthetic audio.
    #[serde(default)]
    pub audio: Option<SyntheticAudioSpec>,
    /// Optional synthetic video.
    #[serde(default)]
    pub video: Option<SyntheticVideoSpec>,
    /// Optional query/passage shape for ranking endpoints.
    #[serde(default)]
    pub rankings: Option<SyntheticRankingsSpec>,
}

/// Synthetic prompt distributions.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticPromptsSpec {
    /// Input sequence length distribution; absent disables text generation.
    #[serde(default)]
    pub isl: Option<DistributionSpec>,
    /// Output sequence length distribution; absent leaves the server limit unset.
    #[serde(default)]
    pub osl: Option<DistributionSpec>,
    /// Hash block size retained for Config-v2 completeness. Synthetic rows have no hash IDs.
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Independently generated prompt values per turn.
    #[serde(default = "one_usize")]
    pub batch_size: usize,
    /// Paired ISL/OSL mixture, which takes precedence over independent lengths.
    #[serde(default)]
    pub sequence_distribution: Option<Vec<SequenceDistributionEntrySpec>>,
}

/// One paired input/output sequence-length bucket.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceDistributionEntrySpec {
    /// Input distribution; Config v2 reduces non-normal variants to their expected value.
    pub isl: DistributionSpec,
    /// Output distribution; Config v2 reduces non-normal variants to their expected value.
    pub osl: DistributionSpec,
    /// Percentage probability.
    pub probability: f64,
}

/// Synthetic shared-prefix and conversation-context shape.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticPrefixPromptsSpec {
    /// Number of reusable first-turn prefixes.
    #[serde(default)]
    pub pool_size: Option<usize>,
    /// Tokens in each reusable prefix.
    #[serde(default)]
    pub length: Option<usize>,
    /// Tokens in the one shared system prompt.
    #[serde(default)]
    pub shared_system_length: Option<usize>,
    /// Tokens in each per-session user context.
    #[serde(default)]
    pub user_context_length: Option<usize>,
}

/// Synthetic image configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticImageSpec {
    /// Images generated per turn.
    pub batch_size: usize,
    /// Width distribution in pixels.
    pub width: DistributionSpec,
    /// Height distribution in pixels.
    pub height: DistributionSpec,
    /// PNG, JPEG, or per-image random selection.
    pub format: SyntheticImageFormatSpec,
    /// `noise`, `assets`, or an absolute local source directory.
    pub source: String,
    /// Selection policy for finite source pools.
    pub source_sampling: SourceImageSamplingSpec,
}

/// Image encoding accepted on the run wire.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticImageFormatSpec {
    /// PNG.
    Png,
    /// JPEG.
    Jpeg,
    /// Randomly select PNG or JPEG per generated image.
    Random,
}

/// Source-image selection policy.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SourceImageSamplingSpec {
    /// Independent draws with replacement.
    RandomWithReplacement,
    /// Shuffled cycles without replacement.
    ShuffleCycle,
    /// Sorted cycles.
    SequentialCycle,
}

/// Synthetic audio configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticAudioSpec {
    /// Audio clips generated per turn.
    pub batch_size: usize,
    /// Duration distribution in seconds.
    pub length: DistributionSpec,
    /// WAV or MP3 output.
    pub format: SyntheticAudioFormatSpec,
    /// Candidate sample rates in kHz.
    pub sample_rates: Vec<f64>,
    /// Candidate PCM bit depths.
    pub depths: Vec<u16>,
    /// Mono or stereo.
    pub channels: u16,
}

/// Synthetic audio encoding.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticAudioFormatSpec {
    /// PCM WAV.
    Wav,
    /// MP3.
    Mp3,
}

/// Synthetic video configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticVideoSpec {
    /// Videos generated per turn.
    pub batch_size: usize,
    /// Duration in seconds.
    pub duration: f64,
    /// Frames per second.
    pub fps: u32,
    /// Optional frame width; native defaults apply when absent.
    #[serde(default)]
    pub width: Option<u32>,
    /// Optional frame height; native defaults apply when absent.
    #[serde(default)]
    pub height: Option<u32>,
    /// MP4 or WebM container.
    pub format: SyntheticVideoFormatSpec,
    /// FFmpeg video codec.
    pub codec: String,
    /// Deterministic frame-generation algorithm.
    pub synth_type: SyntheticVideoPatternSpec,
    /// Optional embedded generated audio track.
    pub audio: SyntheticVideoAudioSpec,
}

/// Synthetic video container.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticVideoFormatSpec {
    /// MP4.
    Mp4,
    /// WebM.
    Webm,
}

/// Synthetic video frame pattern.
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyntheticVideoPatternSpec {
    /// Animated geometric shapes.
    MovingShapes,
    /// Grid and frame clock.
    GridClock,
    /// Random noise frames.
    Noise,
}

/// Embedded video-audio configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticVideoAudioSpec {
    /// Sample rate in kHz.
    pub sample_rate: f64,
    /// Zero disables audio; one and two select mono/stereo.
    pub channels: u16,
    /// Optional FFmpeg audio codec.
    #[serde(default)]
    pub codec: Option<String>,
    /// PCM source bit depth.
    pub depth: u16,
}

/// Synthetic query/passage shape used by ranking endpoints.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SyntheticRankingsSpec {
    /// Passage count distribution.
    pub passages: DistributionSpec,
    /// Tokens per passage.
    pub passage_tokens: DistributionSpec,
    /// Query token distribution.
    pub query_tokens: DistributionSpec,
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
